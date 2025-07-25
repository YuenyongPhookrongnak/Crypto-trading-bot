import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import talib

from .base_strategy import BaseStrategy, TradingSignal

logger = logging.getLogger(__name__)

class GridType(Enum):
    """Grid trading types"""
    CLASSIC = "classic"
    FIBONACCI = "fibonacci"
    ADAPTIVE = "adaptive"

@dataclass
class GridLevel:
    """Represents a grid level"""
    price: float
    level: int
    is_filled: bool = False
    order_id: Optional[str] = None
    quantity: float = 0.0
    timestamp: Optional[datetime] = None

@dataclass
class GridPosition:
    """Represents a grid position"""
    symbol: str
    base_price: float
    grid_levels: List[GridLevel]
    total_investment: float
    unrealized_pnl: float
    realized_pnl: float
    active_orders: int
    grid_range: Tuple[float, float]  # (min_price, max_price)

class GridTradingStrategy(BaseStrategy):
    """
    Grid Trading Strategy
    
    Places buy and sell orders at predetermined intervals above and below current price
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Strategy parameters
        self.grid_size = config.get('grid_size', 0.02)  # 2% grid spacing
        self.num_grids = config.get('num_grids', 10)
        self.base_position_size = config.get('base_position_size', 0.1)
        self.grid_type = GridType(config.get('grid_type', 'classic'))
        self.take_profit_pct = config.get('take_profit_pct', 0.015)  # 1.5%
        self.stop_loss_pct = config.get('stop_loss_pct', 0.05)  # 5%
        
        # Advanced parameters
        self.price_range_multiplier = config.get('price_range_multiplier', 0.2)  # 20% range
        self.rebalance_threshold = config.get('rebalance_threshold', 0.1)  # 10%
        self.max_investment_ratio = config.get('max_investment_ratio', 0.8)  # 80% of available capital
        
        # Adaptive grid parameters
        self.volatility_lookback = config.get('volatility_lookback', 24)
        self.volatility_multiplier = config.get('volatility_multiplier', 2.0)
        
        # Internal state
        self.active_grids = {}  # symbol -> GridPosition
        self.market_conditions = {}
        self.price_history = {}
        
        logger.info(f"Grid Trading Strategy initialized with {self.num_grids} grids, {self.grid_size:.1%} spacing")
    
    async def initialize(self):
        """Initialize strategy"""
        try:
            await super().initialize()
            self.is_initialized = True
            logger.info("Grid Trading Strategy initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Grid Trading Strategy: {e}")
            raise
    
    async def generate_signal(self, symbol: str, market_data: pd.DataFrame, current_price: float) -> Optional[TradingSignal]:
        """
        Generate trading signal for grid trading
        """
        try:
            # Update price history
            self._update_price_history(symbol, market_data, current_price)
            
            # Check if suitable for grid trading
            if not await self._is_suitable_for_grid_trading(symbol, market_data):
                return None
            
            # Check if we already have an active grid
            if symbol in self.active_grids:
                return await self._manage_existing_grid(symbol, current_price)
            
            # Create new grid
            return await self._create_new_grid(symbol, market_data, current_price)
            
        except Exception as e:
            logger.error(f"Error generating grid trading signal for {symbol}: {e}")
            return None
    
    def _update_price_history(self, symbol: str, market_data: pd.DataFrame, current_price: float):
        """Update price history for the symbol"""
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        
        self.price_history[symbol].append({
            'timestamp': datetime.utcnow(),
            'price': current_price,
            'volume': market_data['volume'].iloc[-1] if len(market_data) > 0 else 0
        })
        
        # Keep only recent history
        max_history = self.volatility_lookback * 2
        if len(self.price_history[symbol]) > max_history:
            self.price_history[symbol] = self.price_history[symbol][-max_history:]
    
    async def _is_suitable_for_grid_trading(self, symbol: str, market_data: pd.DataFrame) -> bool:
        """Check if market conditions are suitable for grid trading"""
        try:
            if len(market_data) < self.volatility_lookback:
                return False
            
            # Calculate volatility
            returns = market_data['close'].pct_change().dropna()
            volatility = returns.std()
            
            # Calculate trend strength
            sma_short = market_data['close'].rolling(window=10).mean()
            sma_long = market_data['close'].rolling(window=20).mean()
            
            if len(sma_short) < 20 or len(sma_long) < 20:
                return False
            
            trend_strength = abs(sma_short.iloc[-1] - sma_long.iloc[-1]) / sma_long.iloc[-1]
            
            # Grid trading works best in ranging markets (low trend, moderate volatility)
            is_ranging = trend_strength < 0.03  # Less than 3% trend
            has_volatility = 0.01 < volatility < 0.05  # Between 1% and 5% volatility
            
            # Store market conditions
            self.market_conditions[symbol] = {
                'volatility': volatility,
                'trend_strength': trend_strength,
                'is_ranging': is_ranging,
                'has_volatility': has_volatility,
                'suitable': is_ranging and has_volatility
            }
            
            return is_ranging and has_volatility
            
        except Exception as e:
            logger.error(f"Error checking grid suitability for {symbol}: {e}")
            return False
    
    async def _create_new_grid(self, symbol: str, market_data: pd.DataFrame, current_price: float) -> Optional[TradingSignal]:
        """Create a new grid trading setup"""
        try:
            # Calculate grid parameters
            grid_params = await self._calculate_grid_parameters(symbol, market_data, current_price)
            
            if not grid_params:
                return None
            
            # Create grid levels
            grid_levels = self._create_grid_levels(
                current_price,
                grid_params['grid_spacing'],
                grid_params['num_levels'],
                grid_params['price_range']
            )
            
            # Create grid position
            grid_position = GridPosition(
                symbol=symbol,
                base_price=current_price,
                grid_levels=grid_levels,
                total_investment=0.0,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                active_orders=0,
                grid_range=grid_params['price_range']
            )
            
            # Store active grid
            self.active_grids[symbol] = grid_position
            
            # Generate initial buy signal (start accumulating)
            confidence = self._calculate_grid_confidence(symbol, grid_params)
            
            # Calculate position size
            position_size = self._calculate_position_size(current_price)
            
            reasoning = [
                f"Grid trading setup for ranging market",
                f"Grid spacing: {grid_params['grid_spacing']:.1%}",
                f"Number of levels: {grid_params['num_levels']}",
                f"Price range: ${grid_params['price_range'][0]:.2f} - ${grid_params['price_range'][1]:.2f}",
                f"Market volatility: {self.market_conditions[symbol]['volatility']:.2%}"
            ]
            
            return TradingSignal(
                symbol=symbol,
                signal_type='BUY',
                confidence=confidence,
                entry_price=current_price,
                stop_loss=current_price * (1 - self.stop_loss_pct),
                take_profit=current_price * (1 + self.take_profit_pct),
                reasoning=reasoning,
                indicators={
                    'grid_type': self.grid_type.value,
                    'grid_spacing': grid_params['grid_spacing'],
                    'num_levels': grid_params['num_levels'],
                    'price_range': grid_params['price_range'],
                    'volatility': self.market_conditions[symbol]['volatility'],
                    'trend_strength': self.market_conditions[symbol]['trend_strength']
                },
                strategy_id='grid_trading',
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error creating new grid for {symbol}: {e}")
            return None
    
    async def _calculate_grid_parameters(self, symbol: str, market_data: pd.DataFrame, current_price: float) -> Optional[Dict[str, Any]]:
        """Calculate optimal grid parameters"""
        try:
            market_conditions = self.market_conditions.get(symbol, {})
            volatility = market_conditions.get('volatility', 0.02)
            
            if self.grid_type == GridType.ADAPTIVE:
                # Adaptive grid based on volatility
                grid_spacing = volatility * self.volatility_multiplier
                grid_spacing = max(0.005, min(0.05, grid_spacing))  # Between 0.5% and 5%
                
                # Adjust number of levels based on volatility
                num_levels = max(5, min(20, int(self.num_grids * (0.02 / volatility))))
                
            elif self.grid_type == GridType.FIBONACCI:
                # Fibonacci-based grid spacing
                grid_spacing = self.grid_size
                num_levels = self.num_grids
                
            else:  # CLASSIC
                grid_spacing = self.grid_size
                num_levels = self.num_grids
            
            # Calculate price range
            price_range_pct = self.price_range_multiplier
            if self.grid_type == GridType.ADAPTIVE:
                price_range_pct = volatility * 10  # Adjust range based on volatility
                price_range_pct = max(0.1, min(0.4, price_range_pct))  # Between 10% and 40%
            
            min_price = current_price * (1 - price_range_pct / 2)
            max_price = current_price * (1 + price_range_pct / 2)
            
            return {
                'grid_spacing': grid_spacing,
                'num_levels': num_levels,
                'price_range': (min_price, max_price),
                'volatility_adjusted': self.grid_type == GridType.ADAPTIVE
            }
            
        except Exception as e:
            logger.error(f"Error calculating grid parameters: {e}")
            return None
    
    def _create_grid_levels(self, base_price: float, grid_spacing: float, 
                          num_levels: int, price_range: Tuple[float, float]) -> List[GridLevel]:
        """Create grid levels"""
        try:
            levels = []
            min_price, max_price = price_range
            
            if self.grid_type == GridType.FIBONACCI:
                # Fibonacci sequence for level spacing
                fib_sequence = self._generate_fibonacci_sequence(num_levels)
                
                for i, fib_ratio in enumerate(fib_sequence):
                    # Create levels above and below base price
                    if i < len(fib_sequence) // 2:
                        # Levels below base price (buy levels)
                        price = base_price * (1 - fib_ratio * grid_spacing)
                        if price >= min_price:
                            levels.append(GridLevel(price=price, level=-i-1))
                    else:
                        # Levels above base price (sell levels)
                        price = base_price * (1 + fib_ratio * grid_spacing)
                        if price <= max_price:
                            levels.append(GridLevel(price=price, level=i-len(fib_sequence)//2+1))
            
            else:  # CLASSIC or ADAPTIVE
                # Create levels below base price (buy levels)
                for i in range(1, num_levels // 2 + 1):
                    price = base_price * (1 - i * grid_spacing)
                    if price >= min_price:
                        levels.append(GridLevel(price=price, level=-i))
                
                # Create levels above base price (sell levels)
                for i in range(1, num_levels // 2 + 1):
                    price = base_price * (1 + i * grid_spacing)
                    if price <= max_price:
                        levels.append(GridLevel(price=price, level=i))
            
            # Sort levels by price
            levels.sort(key=lambda x: x.price)
            
            return levels
            
        except Exception as e:
            logger.error(f"Error creating grid levels: {e}")
            return []
    
    def _generate_fibonacci_sequence(self, length: int) -> List[float]:
        """Generate normalized Fibonacci sequence"""
        try:
            if length <= 0:
                return []
            
            # Generate Fibonacci numbers
            fib = [1, 1]
            for i in range(2, length):
                fib.append(fib[i-1] + fib[i-2])
            
            # Normalize to ratios
            max_fib = max(fib)
            return [f / max_fib for f in fib]
            
        except Exception as e:
            logger.error(f"Error generating Fibonacci sequence: {e}")
            return [i / length for i in range(1, length + 1)]  # Fallback to linear
    
    def _calculate_position_size(self, current_price: float) -> float:
        """Calculate position size for grid trading"""
        try:
            # Base position size adjusted for price
            if current_price > 1000:  # High-priced assets
                return self.base_position_size * 0.5
            elif current_price < 10:  # Low-priced assets
                return self.base_position_size * 2.0
            else:
                return self.base_position_size
                
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return self.base_position_size
    
    def _calculate_grid_confidence(self, symbol: str, grid_params: Dict[str, Any]) -> float:
        """Calculate confidence score for grid trading"""
        try:
            market_conditions = self.market_conditions.get(symbol, {})
            
            # Base confidence
            confidence = 60.0
            
            # Adjust for market suitability
            if market_conditions.get('is_ranging', False):
                confidence += 15
            
            if market_conditions.get('has_volatility', False):
                confidence += 10
            
            # Adjust for volatility level
            volatility = market_conditions.get('volatility', 0.02)
            if 0.015 <= volatility <= 0.035:  # Optimal volatility range
                confidence += 10
            
            # Adjust for trend strength (lower is better for grid)
            trend_strength = market_conditions.get('trend_strength', 0)
            if trend_strength < 0.02:
                confidence += 5
            
            return min(95, confidence)
            
        except Exception as e:
            logger.error(f"Error calculating grid confidence: {e}")
            return 60.0
    
    async def _manage_existing_grid(self, symbol: str, current_price: float) -> Optional[TradingSignal]:
        """Manage existing grid position"""
        try:
            grid_position = self.active_grids[symbol]
            
            # Check if price is within grid range
            min_price, max_price = grid_position.grid_range
            
            if current_price < min_price:
                return await self._handle_grid_breakdown(symbol, current_price, 'below')
            elif current_price > max_price:
                return await self._handle_grid_breakdown(symbol, current_price, 'above')
            
            # Find nearest grid levels
            nearest_buy_level = self._find_nearest_level(grid_position.grid_levels, current_price, 'buy')
            nearest_sell_level = self._find_nearest_level(grid_position.grid_levels, current_price, 'sell')
            
            # Generate signals for unfilled levels
            if nearest_buy_level and not nearest_buy_level.is_filled and current_price <= nearest_buy_level.price:
                return await self._generate_grid_buy_signal(symbol, nearest_buy_level, current_price)
            
            if nearest_sell_level and not nearest_sell_level.is_filled and current_price >= nearest_sell_level.price:
                return await self._generate_grid_sell_signal(symbol, nearest_sell_level, current_price)
            
            # No immediate signals
            return None
            
        except Exception as e:
            logger.error(f"Error managing existing grid for {symbol}: {e}")
            return None
    
    def _find_nearest_level(self, grid_levels: List[GridLevel], current_price: float, level_type: str) -> Optional[GridLevel]:
        """Find nearest unfilled grid level"""
        try:
            if level_type == 'buy':
                # Find highest unfilled level below current price
                buy_levels = [level for level in grid_levels if level.level < 0 and not level.is_filled and level.price < current_price]
                return max(buy_levels, key=lambda x: x.price) if buy_levels else None
            
            else:  # sell
                # Find lowest unfilled level above current price
                sell_levels = [level for level in grid_levels if level.level > 0 and not level.is_filled and level.price > current_price]
                return min(sell_levels, key=lambda x: x.price) if sell_levels else None
                
        except Exception as e:
            logger.error(f"Error finding nearest level: {e}")
            return None
    
    async def _generate_grid_buy_signal(self, symbol: str, grid_level: GridLevel, current_price: float) -> TradingSignal:
        """Generate buy signal for grid level"""
        try:
            position_size = self._calculate_position_size(current_price)
            
            reasoning = [
                f"Grid buy at level {grid_level.level}",
                f"Target price: ${grid_level.price:.4f}",
                f"Current price: ${current_price:.4f}",
                f"Grid accumulation strategy"
            ]
            
            return TradingSignal(
                symbol=symbol,
                signal_type='BUY',
                confidence=75.0,
                entry_price=grid_level.price,
                stop_loss=grid_level.price * (1 - self.stop_loss_pct),
                take_profit=grid_level.price * (1 + self.take_profit_pct),
                reasoning=reasoning,
                indicators={
                    'grid_level': grid_level.level,
                    'grid_price': grid_level.price,
                    'grid_type': 'buy_level'
                },
                strategy_id='grid_trading',
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error generating grid buy signal: {e}")
            return None
    
    async def _generate_grid_sell_signal(self, symbol: str, grid_level: GridLevel, current_price: float) -> TradingSignal:
        """Generate sell signal for grid level"""
        try:
            position_size = self._calculate_position_size(current_price)
            
            reasoning = [
                f"Grid sell at level {grid_level.level}",
                f"Target price: ${grid_level.price:.4f}",
                f"Current price: ${current_price:.4f}",
                f"Grid profit taking strategy"
            ]
            
            return TradingSignal(
                symbol=symbol,
                signal_type='SELL',
                confidence=75.0,
                entry_price=grid_level.price,
                stop_loss=grid_level.price * (1 + self.stop_loss_pct),
                take_profit=grid_level.price * (1 - self.take_profit_pct),
                reasoning=reasoning,
                indicators={
                    'grid_level': grid_level.level,
                    'grid_price': grid_level.price,
                    'grid_type': 'sell_level'
                },
                strategy_id='grid_trading',
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error generating grid sell signal: {e}")
            return None
    
    async def _handle_grid_breakdown(self, symbol: str, current_price: float, direction: str) -> Optional[TradingSignal]:
        """Handle price breaking out of grid range"""
        try:
            grid_position = self.active_grids[symbol]
            
            if direction == 'below':
                # Price broke below grid - potential stop loss
                reasoning = [
                    f"Price broke below grid range",
                    f"Current: ${current_price:.4f}, Grid min: ${grid_position.grid_range[0]:.4f}",
                    f"Consider closing grid positions"
                ]
                
                return TradingSignal(
                    symbol=symbol,
                    signal_type='SELL',
                    confidence=80.0,
                    entry_price=current_price,
                    stop_loss=None,  # Emergency exit
                    take_profit=None,
                    reasoning=reasoning,
                    indicators={
                        'grid_breakdown': direction,
                        'breakdown_price': current_price,
                        'grid_range': grid_position.grid_range
                    },
                    strategy_id='grid_trading',
                    timestamp=datetime.utcnow()
                )
            
            elif direction == 'above':
                # Price broke above grid - potential profit taking
                reasoning = [
                    f"Price broke above grid range",
                    f"Current: ${current_price:.4f}, Grid max: ${grid_position.grid_range[1]:.4f}",
                    f"Profit taking opportunity"
                ]
                
                return TradingSignal(
                    symbol=symbol,
                    signal_type='SELL',
                    confidence=85.0,
                    entry_price=current_price,
                    stop_loss=None,
                    take_profit=None,
                    reasoning=reasoning,
                    indicators={
                        'grid_breakdown': direction,
                        'breakdown_price': current_price,
                        'grid_range': grid_position.grid_range
                    },
                    strategy_id='grid_trading',
                    timestamp=datetime.utcnow()
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error handling grid breakdown: {e}")
            return None
    
    async def update_grid_level(self, symbol: str, level_price: float, filled: bool, order_id: str = None):
        """Update grid level status after order execution"""
        try:
            if symbol not in self.active_grids:
                return
            
            grid_position = self.active_grids[symbol]
            
            # Find and update the level
            for level in grid_position.grid_levels:
                if abs(level.price - level_price) < 0.0001:  # Price match with tolerance
                    level.is_filled = filled
                    level.order_id = order_id
                    level.timestamp = datetime.utcnow()
                    
                    if filled:
                        grid_position.active_orders += 1
                    
                    break
            
        except Exception as e:
            logger.error(f"Error updating grid level: {e}")
    
    async def get_grid_status(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current grid status"""
        try:
            if symbol not in self.active_grids:
                return None
            
            grid_position = self.active_grids[symbol]
            
            filled_levels = [level for level in grid_position.grid_levels if level.is_filled]
            unfilled_levels = [level for level in grid_position.grid_levels if not level.is_filled]
            
            return {
                'symbol': symbol,
                'base_price': grid_position.base_price,
                'grid_range': grid_position.grid_range,
                'total_levels': len(grid_position.grid_levels),
                'filled_levels': len(filled_levels),
                'unfilled_levels': len(unfilled_levels),
                'active_orders': grid_position.active_orders,
                'total_investment': grid_position.total_investment,
                'unrealized_pnl': grid_position.unrealized_pnl,
                'realized_pnl': grid_position.realized_pnl,
                'grid_efficiency': len(filled_levels) / len(grid_position.grid_levels) if grid_position.grid_levels else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting grid status: {e}")
            return None
    
    async def close_grid(self, symbol: str) -> bool:
        """Close and cleanup grid position"""
        try:
            if symbol in self.active_grids:
                del self.active_grids[symbol]
                logger.info(f"Grid for {symbol} closed successfully")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error closing grid for {symbol}: {e}")
            return False
    
    async def get_strategy_info(self) -> Dict[str, Any]:
        """Get current strategy information"""
        try:
            active_grids_info = {}
            
            for symbol, grid_position in self.active_grids.items():
                status = await self.get_grid_status(symbol)
                if status:
                    active_grids_info[symbol] = status
            
            return {
                'strategy_name': 'Grid Trading',
                'is_initialized': self.is_initialized,
                'active_grids': len(self.active_grids),
                'grid_details': active_grids_info,
                'parameters': {
                    'grid_size': self.grid_size,
                    'num_grids': self.num_grids,
                    'grid_type': self.grid_type.value,
                    'base_position_size': self.base_position_size,
                    'take_profit_pct': self.take_profit_pct,
                    'stop_loss_pct': self.stop_loss_pct
                },
                'market_conditions': self.market_conditions
            }
            
        except Exception as e:
            logger.error(f"Error getting grid trading strategy info: {e}")
            return {'error': str(e)}
    
    async def cleanup(self):
        """Cleanup strategy resources"""
        try:
            self.active_grids.clear()
            self.market_conditions.clear()
            self.price_history.clear()
            
            await super().cleanup()
            logger.info("Grid Trading Strategy cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during grid trading strategy cleanup: {e}")

# Factory function
def create_grid_trading_strategy(config: Dict[str, Any]) -> GridTradingStrategy:
    """Factory function to create Grid Trading Strategy instance"""
    return GridTradingStrategy(config)

# Example usage
async def main():
    """Example usage of Grid Trading Strategy"""
    try:
        # Configuration
        config = {
            'grid_size': 0.02,  # 2% spacing
            'num_grids': 10,
            'base_position_size': 0.1,
            'grid_type': 'adaptive',
            'take_profit_pct': 0.015,
            'stop_loss_pct': 0.05,
            'volatility_lookback': 24,
            'volatility_multiplier': 2.0
        }
        
        # Create strategy
        strategy = create_grid_trading_strategy(config)
        await strategy.initialize()
        
        print("⚡ Grid Trading Strategy Example")
        print("=" * 40)
        
        # Get strategy info
        info = await strategy.get_strategy_info()
        print(f"Strategy: {info['strategy_name']}")
        print(f"Grid Type: {info['parameters']['grid_type']}")
        print(f"Grid Size: {info['parameters']['grid_size']:.1%}")
        print(f"Number of Grids: {info['parameters']['num_grids']}")
        
        # Simulate market data for ranging market
        dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
        np.random.seed(42)
        
        # Create ranging market data
        base_price = 45000
        noise = np.random.randn(100) * 200  # Low volatility
        trend = np.sin(np.arange(100) * 0.1) * 500  # Sideways movement
        prices = base_price + trend + noise
        
        market_data = pd.DataFrame({
            'timestamp': dates,
            'open': prices * 0.999,
            'high': prices * 1.001,
            'low': prices * 0.999,
            'close': prices,
            'volume': np.random.randint(1000, 5000, 100)
        })
        
        # Generate signal
        current_price = market_data['close'].iloc[-1]
        signal = await strategy.generate_signal('BTC/USDT', market_data, current_price)
        
        if signal:
            print(f"\n📊 Generated Signal:")
            print(f"  Symbol: {signal.symbol}")
            print(f"  Type: {signal.signal_type}")
            print(f"  Confidence: {signal.confidence:.1f}%")
            print(f"  Entry Price: ${signal.entry_price:.2f}")
            print(f"  Stop Loss: ${signal.stop_loss:.2f}")
            print(f"  Take Profit: ${signal.take_profit:.2f}")
            
            # Grid specific info
            indicators = signal.indicators
            print(f"  Grid Type: {indicators.get('grid_type', 'N/A')}")
            print(f"  Grid Spacing: {indicators.get('grid_spacing', 0):.1%}")
            print(f"  Number of Levels: {indicators.get('num_levels', 0)}")
            
            if 'price_range' in indicators:
                price_range = indicators['price_range']
                print(f"  Price Range: ${price_range[0]:.2f} - ${price_range[1]:.2f}")
            
            print(f"  Reasoning: {signal.reasoning[0] if signal.reasoning else 'N/A'}")
            
            # Show grid status
            grid_status = await strategy.get_grid_status('BTC/USDT')
            if grid_status:
                print(f"\n📋 Grid Status:")
                print(f"  Total Levels: {grid_status['total_levels']}")
                print(f"  Grid Range: ${grid_status['grid_range'][0]:.2f} - ${grid_status['grid_range'][1]:.2f}")
                print(f"  Base Price: ${grid_status['base_price']:.2f}")
        else:
            print("\n📊 No signal generated - market not suitable for grid trading")
        
        # Cleanup
        await strategy.cleanup()
        print("\n✅ Example completed successfully")
        
    except Exception as e:
        print(f"❌ Error in example: {e}")

if __name__ == "__main__":
    asyncio.run(main())