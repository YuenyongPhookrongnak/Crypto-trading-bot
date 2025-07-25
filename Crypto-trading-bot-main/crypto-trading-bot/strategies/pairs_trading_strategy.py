import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
import talib

from .base_strategy import BaseStrategy, TradingSignal

logger = logging.getLogger(__name__)

@dataclass
class PairRelationship:
    """Represents a trading pair relationship"""
    symbol_a: str
    symbol_b: str
    correlation: float
    spread_mean: float
    spread_std: float
    last_spread: float
    z_score: float
    relationship_strength: float

class PairsTradingStrategy(BaseStrategy):
    """
    Pairs Trading Strategy
    
    Trades based on mean reversion of spread between correlated pairs
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Strategy parameters
        self.lookback_period = config.get('lookback_period', 20)
        self.entry_threshold = config.get('entry_threshold', 2.0)  # Z-score threshold
        self.exit_threshold = config.get('exit_threshold', 0.5)
        self.stop_loss_threshold = config.get('stop_loss_threshold', 3.0)
        self.correlation_threshold = config.get('correlation_threshold', 0.7)
        self.min_spread_volatility = config.get('min_spread_volatility', 0.001)
        
        # Pair management
        self.trading_pairs = config.get('trading_pairs', [
            ('BTC/USDT', 'ETH/USDT'),
            ('BTC/USDT', 'ADA/USDT'),
            ('ETH/USDT', 'ADA/USDT')
        ])
        
        # Internal state
        self.pair_relationships = {}
        self.active_pair_positions = {}
        self.spread_history = {}
        
        logger.info(f"Pairs Trading Strategy initialized with {len(self.trading_pairs)} pairs")
    
    async def initialize(self):
        """Initialize strategy"""
        try:
            await super().initialize()
            
            # Initialize pair relationships
            for pair in self.trading_pairs:
                pair_key = f"{pair[0]}_{pair[1]}"
                self.pair_relationships[pair_key] = None
                self.spread_history[pair_key] = []
            
            self.is_initialized = True
            logger.info("Pairs Trading Strategy initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Pairs Trading Strategy: {e}")
            raise
    
    async def generate_signal(self, symbol: str, market_data: pd.DataFrame, current_price: float) -> Optional[TradingSignal]:
        """
        Generate trading signal for pairs trading
        
        Note: This strategy works with pairs, so signals are generated
        when analyzing pair relationships rather than individual symbols
        """
        try:
            # Find pairs involving this symbol
            relevant_pairs = [
                pair for pair in self.trading_pairs 
                if symbol in pair
            ]
            
            if not relevant_pairs:
                return None
            
            # Analyze each relevant pair
            best_signal = None
            best_confidence = 0
            
            for pair in relevant_pairs:
                signal = await self._analyze_pair(pair, symbol, market_data, current_price)
                
                if signal and signal.confidence > best_confidence:
                    best_signal = signal
                    best_confidence = signal.confidence
            
            return best_signal
            
        except Exception as e:
            logger.error(f"Error generating pairs trading signal for {symbol}: {e}")
            return None
    
    async def _analyze_pair(self, pair: Tuple[str, str], target_symbol: str, 
                           market_data: pd.DataFrame, current_price: float) -> Optional[TradingSignal]:
        """Analyze a specific pair for trading opportunities"""
        try:
            symbol_a, symbol_b = pair
            pair_key = f"{symbol_a}_{symbol_b}"
            
            # Get market data for both symbols (simplified - in real implementation, 
            # you'd fetch data for the other symbol)
            if target_symbol == symbol_a:
                primary_symbol = symbol_a
                secondary_symbol = symbol_b
                primary_data = market_data
                primary_price = current_price
                # In real implementation, fetch secondary_data and secondary_price
                secondary_price = current_price * 0.85  # Simplified
            else:
                primary_symbol = symbol_b
                secondary_symbol = symbol_a
                primary_data = market_data
                primary_price = current_price
                secondary_price = current_price * 1.15  # Simplified
            
            # Calculate pair relationship
            relationship = await self._calculate_pair_relationship(
                primary_data, primary_price, secondary_price, pair_key
            )
            
            if not relationship:
                return None
            
            # Store relationship
            self.pair_relationships[pair_key] = relationship
            
            # Generate signal based on spread z-score
            return await self._generate_pair_signal(
                relationship, primary_symbol, primary_price
            )
            
        except Exception as e:
            logger.error(f"Error analyzing pair {pair}: {e}")
            return None
    
    async def _calculate_pair_relationship(self, primary_data: pd.DataFrame, 
                                         primary_price: float, secondary_price: float,
                                         pair_key: str) -> Optional[PairRelationship]:
        """Calculate the relationship between two symbols"""
        try:
            if len(primary_data) < self.lookback_period:
                return None
            
            # Calculate price ratio (spread)
            spread = primary_price / secondary_price
            
            # Update spread history
            self.spread_history[pair_key].append({
                'timestamp': datetime.utcnow(),
                'spread': spread,
                'primary_price': primary_price,
                'secondary_price': secondary_price
            })
            
            # Keep only recent history
            if len(self.spread_history[pair_key]) > self.lookback_period * 2:
                self.spread_history[pair_key] = self.spread_history[pair_key][-self.lookback_period * 2:]
            
            if len(self.spread_history[pair_key]) < self.lookback_period:
                return None
            
            # Calculate spread statistics
            recent_spreads = [
                item['spread'] for item in self.spread_history[pair_key][-self.lookback_period:]
            ]
            
            spread_mean = np.mean(recent_spreads)
            spread_std = np.std(recent_spreads)
            
            if spread_std < self.min_spread_volatility:
                return None  # Not enough volatility for pairs trading
            
            # Calculate z-score
            z_score = (spread - spread_mean) / spread_std if spread_std > 0 else 0
            
            # Calculate correlation (simplified)
            correlation = 0.8  # In real implementation, calculate from price history
            
            # Calculate relationship strength
            relationship_strength = abs(correlation) * min(1.0, spread_std / spread_mean)
            
            return PairRelationship(
                symbol_a=pair_key.split('_')[0],
                symbol_b=pair_key.split('_')[1],
                correlation=correlation,
                spread_mean=spread_mean,
                spread_std=spread_std,
                last_spread=spread,
                z_score=z_score,
                relationship_strength=relationship_strength
            )
            
        except Exception as e:
            logger.error(f"Error calculating pair relationship: {e}")
            return None
    
    async def _generate_pair_signal(self, relationship: PairRelationship, 
                                   symbol: str, current_price: float) -> Optional[TradingSignal]:
        """Generate trading signal based on pair relationship"""
        try:
            # Check if correlation is strong enough
            if abs(relationship.correlation) < self.correlation_threshold:
                return None
            
            z_score = relationship.z_score
            
            # Entry signals
            if abs(z_score) >= self.entry_threshold:
                
                # Determine signal direction
                if z_score > self.entry_threshold:
                    # Spread too high - short the ratio
                    # In pairs trading, this means short the expensive asset, long the cheap one
                    signal_type = 'SELL'  # Short the currently expensive symbol
                    reasoning = [
                        f"Pairs trading: spread z-score {z_score:.2f} above entry threshold {self.entry_threshold}",
                        f"Spread: {relationship.last_spread:.4f}, Mean: {relationship.spread_mean:.4f}",
                        f"Expected mean reversion of spread",
                        f"Correlation: {relationship.correlation:.2f}"
                    ]
                    
                elif z_score < -self.entry_threshold:
                    # Spread too low - long the ratio
                    signal_type = 'BUY'  # Long the currently cheap symbol
                    reasoning = [
                        f"Pairs trading: spread z-score {z_score:.2f} below entry threshold {-self.entry_threshold}",
                        f"Spread: {relationship.last_spread:.4f}, Mean: {relationship.spread_mean:.4f}",
                        f"Expected mean reversion of spread",
                        f"Correlation: {relationship.correlation:.2f}"
                    ]
                else:
                    return None
                
                # Calculate confidence
                confidence = min(95, 60 + abs(z_score) * 10)
                
                # Calculate stop loss and take profit
                spread_change_for_stop = self.stop_loss_threshold * relationship.spread_std
                spread_change_for_profit = self.exit_threshold * relationship.spread_std
                
                if signal_type == 'BUY':
                    # Long position
                    stop_loss = current_price * (1 - spread_change_for_stop / relationship.spread_mean)
                    take_profit = current_price * (1 + spread_change_for_profit / relationship.spread_mean)
                else:
                    # Short position
                    stop_loss = current_price * (1 + spread_change_for_stop / relationship.spread_mean)
                    take_profit = current_price * (1 - spread_change_for_profit / relationship.spread_mean)
                
                return TradingSignal(
                    symbol=symbol,
                    signal_type=signal_type,
                    confidence=confidence,
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    reasoning=reasoning,
                    indicators={
                        'z_score': z_score,
                        'spread': relationship.last_spread,
                        'spread_mean': relationship.spread_mean,
                        'spread_std': relationship.spread_std,
                        'correlation': relationship.correlation,
                        'relationship_strength': relationship.relationship_strength
                    },
                    strategy_id='pairs_trading',
                    timestamp=datetime.utcnow()
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating pair signal: {e}")
            return None
    
    async def should_exit_position(self, symbol: str, current_price: float, 
                                 entry_price: float, position_direction: str) -> Dict[str, Any]:
        """Check if should exit existing pairs trading position"""
        try:
            # Find the pair this symbol belongs to
            relevant_pair_key = None
            for pair_key in self.pair_relationships:
                if symbol in pair_key:
                    relevant_pair_key = pair_key
                    break
            
            if not relevant_pair_key or relevant_pair_key not in self.pair_relationships:
                return {'should_exit': False}
            
            relationship = self.pair_relationships[relevant_pair_key]
            if not relationship:
                return {'should_exit': False}
            
            z_score = relationship.z_score
            
            # Exit conditions
            if abs(z_score) <= self.exit_threshold:
                return {
                    'should_exit': True,
                    'reason': 'MEAN_REVERSION',
                    'z_score': z_score,
                    'exit_threshold': self.exit_threshold
                }
            
            # Stop loss condition
            if abs(z_score) >= self.stop_loss_threshold:
                return {
                    'should_exit': True,
                    'reason': 'STOP_LOSS',
                    'z_score': z_score,
                    'stop_loss_threshold': self.stop_loss_threshold
                }
            
            return {'should_exit': False, 'z_score': z_score}
            
        except Exception as e:
            logger.error(f"Error checking exit condition for {symbol}: {e}")
            return {'should_exit': False}
    
    async def get_strategy_info(self) -> Dict[str, Any]:
        """Get current strategy information"""
        try:
            pair_stats = {}
            
            for pair_key, relationship in self.pair_relationships.items():
                if relationship:
                    pair_stats[pair_key] = {
                        'correlation': relationship.correlation,
                        'z_score': relationship.z_score,
                        'spread': relationship.last_spread,
                        'spread_mean': relationship.spread_mean,
                        'spread_std': relationship.spread_std,
                        'relationship_strength': relationship.relationship_strength
                    }
            
            return {
                'strategy_name': 'Pairs Trading',
                'is_initialized': self.is_initialized,
                'trading_pairs': self.trading_pairs,
                'pair_relationships': pair_stats,
                'parameters': {
                    'lookback_period': self.lookback_period,
                    'entry_threshold': self.entry_threshold,
                    'exit_threshold': self.exit_threshold,
                    'stop_loss_threshold': self.stop_loss_threshold,
                    'correlation_threshold': self.correlation_threshold
                },
                'active_pairs': len([r for r in self.pair_relationships.values() if r is not None])
            }
            
        except Exception as e:
            logger.error(f"Error getting pairs trading strategy info: {e}")
            return {'error': str(e)}
    
    async def cleanup(self):
        """Cleanup strategy resources"""
        try:
            self.pair_relationships.clear()
            self.active_pair_positions.clear()
            self.spread_history.clear()
            
            await super().cleanup()
            logger.info("Pairs Trading Strategy cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during pairs trading strategy cleanup: {e}")

# Factory function
def create_pairs_trading_strategy(config: Dict[str, Any]) -> PairsTradingStrategy:
    """Factory function to create Pairs Trading Strategy instance"""
    return PairsTradingStrategy(config)

# Example usage
async def main():
    """Example usage of Pairs Trading Strategy"""
    try:
        # Configuration
        config = {
            'lookback_period': 20,
            'entry_threshold': 2.0,
            'exit_threshold': 0.5,
            'stop_loss_threshold': 3.0,
            'correlation_threshold': 0.7,
            'trading_pairs': [
                ('BTC/USDT', 'ETH/USDT'),
                ('BTC/USDT', 'ADA/USDT'),
                ('ETH/USDT', 'ADA/USDT')
            ]
        }
        
        # Create strategy
        strategy = create_pairs_trading_strategy(config)
        await strategy.initialize()
        
        print("🔄 Pairs Trading Strategy Example")
        print("=" * 40)
        
        # Get strategy info
        info = await strategy.get_strategy_info()
        print(f"Strategy: {info['strategy_name']}")
        print(f"Trading Pairs: {len(info['trading_pairs'])}")
        print(f"Entry Threshold: {info['parameters']['entry_threshold']}")
        print(f"Correlation Threshold: {info['parameters']['correlation_threshold']}")
        
        # Simulate market data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
        np.random.seed(42)
        
        prices = 45000 + np.cumsum(np.random.randn(100) * 100)
        market_data = pd.DataFrame({
            'timestamp': dates,
            'open': prices * 0.999,
            'high': prices * 1.002,
            'low': prices * 0.998,
            'close': prices,
            'volume': np.random.randint(1000, 10000, 100)
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
            print(f"  Z-Score: {signal.indicators.get('z_score', 'N/A'):.2f}")
            print(f"  Reasoning: {signal.reasoning[0] if signal.reasoning else 'N/A'}")
        else:
            print("\n📊 No signal generated")
        
        # Cleanup
        await strategy.cleanup()
        print("\n✅ Example completed successfully")
        
    except Exception as e:
        print(f"❌ Error in example: {e}")

if __name__ == "__main__":
    asyncio.run(main())