# Run the test
asyncio.run(test_portfolio_manager())
import asyncio
import ccxt
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
import json

from database.connection import get_db_session
from database.models import Trade, Position, PortfolioSnapshot, PerformanceMetric

logger = logging.getLogger(__name__)

class PositionStatus(Enum):
    """Position status types"""
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    PENDING = "PENDING"
    CANCELLED = "CANCELLED"

class PositionDirection(Enum):
    """Position direction types"""
    LONG = "LONG"
    SHORT = "SHORT"

class RebalanceAction(Enum):
    """Rebalancing action types"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    REDUCE = "REDUCE"
    INCREASE = "INCREASE"

@dataclass
class PortfolioPosition:
    """Portfolio position data structure"""
    symbol: str
    direction: PositionDirection
    quantity: float
    entry_price: float
    current_price: float
    
    # Position metrics
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    cost_basis: float
    
    # Risk metrics
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size_pct: float = 0.0
    risk_amount: float = 0.0
    
    # Metadata
    entry_time: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    status: PositionStatus = PositionStatus.OPEN
    
    # Performance tracking
    max_profit: float = 0.0
    max_loss: float = 0.0
    days_held: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'symbol': self.symbol,
            'direction': self.direction.value,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'market_value': self.market_value,
            'unrealized_pnl': self.unrealized_pnl,
            'unrealized_pnl_pct': self.unrealized_pnl_pct,
            'cost_basis': self.cost_basis,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'position_size_pct': self.position_size_pct,
            'risk_amount': self.risk_amount,
            'entry_time': self.entry_time.isoformat(),
            'last_updated': self.last_updated.isoformat(),
            'status': self.status.value,
            'max_profit': self.max_profit,
            'max_loss': self.max_loss,
            'days_held': self.days_held
        }

@dataclass
class PortfolioSummary:
    """Portfolio summary metrics"""
    total_value: float
    cash_balance: float
    invested_amount: float
    available_balance: float
    
    # Performance metrics
    total_pnl: float
    total_pnl_pct: float
    daily_pnl: float
    daily_pnl_pct: float
    
    # Risk metrics
    portfolio_beta: float
    sharpe_ratio: float
    max_drawdown: float
    current_drawdown: float
    var_95: float
    
    # Position metrics
    total_positions: int
    long_positions: int
    short_positions: int
    largest_position_pct: float
    
    # Allocation metrics
    allocation_by_symbol: Dict[str, float] = field(default_factory=dict)
    allocation_by_sector: Dict[str, float] = field(default_factory=dict)
    correlation_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Metadata
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'total_value': self.total_value,
            'cash_balance': self.cash_balance,
            'invested_amount': self.invested_amount,
            'available_balance': self.available_balance,
            'total_pnl': self.total_pnl,
            'total_pnl_pct': self.total_pnl_pct,
            'daily_pnl': self.daily_pnl,
            'daily_pnl_pct': self.daily_pnl_pct,
            'portfolio_beta': self.portfolio_beta,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'current_drawdown': self.current_drawdown,
            'var_95': self.var_95,
            'total_positions': self.total_positions,
            'long_positions': self.long_positions,
            'short_positions': self.short_positions,
            'largest_position_pct': self.largest_position_pct,
            'allocation_by_symbol': self.allocation_by_symbol,
            'allocation_by_sector': self.allocation_by_sector,
            'correlation_matrix': self.correlation_matrix,
            'last_updated': self.last_updated.isoformat()
        }

@dataclass
class RebalanceRecommendation:
    """Portfolio rebalancing recommendation"""
    symbol: str
    action: RebalanceAction
    current_allocation_pct: float
    target_allocation_pct: float
    recommended_quantity: float
    estimated_cost: float
    priority: int  # 1-5, 1 = highest priority
    reasoning: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'action': self.action.value,
            'current_allocation_pct': self.current_allocation_pct,
            'target_allocation_pct': self.target_allocation_pct,
            'recommended_quantity': self.recommended_quantity,
            'estimated_cost': self.estimated_cost,
            'priority': self.priority,
            'reasoning': self.reasoning
        }

class PortfolioManager:
    """Advanced Portfolio Management System"""
    
    def __init__(self, api_config, trading_config):
        self.api_config = api_config
        self.trading_config = trading_config
        self.exchange = None
        
        # Portfolio state
        self.positions: Dict[str, PortfolioPosition] = {}
        self.cash_balance = 0.0
        self.total_equity = 0.0
        self.initial_capital = trading_config.initial_capital
        
        # Performance tracking
        self.performance_history = []
        self.daily_snapshots = []
        self.trade_history = []
        
        # Risk metrics
        self.risk_metrics = {
            'var_95': 0.0,
            'max_drawdown': 0.0,
            'current_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'portfolio_beta': 1.0,
            'correlation_matrix': {},
            'sector_allocation': {},
            'last_calculated': datetime.utcnow()
        }
        
        # Portfolio targets and limits
        self.allocation_targets = {
            'BTC': 0.40,  # 40% Bitcoin
            'ETH': 0.25,  # 25% Ethereum
            'ALT': 0.30,  # 30% Altcoins
            'STABLE': 0.05  # 5% Stablecoins
        }
        
        self.position_limits = {
            'max_positions': trading_config.max_open_positions,
            'max_position_size': 0.30,  # 30% max single position
            'max_sector_allocation': 0.60,  # 60% max per sector
            'min_cash_reserve': 0.10,  # 10% minimum cash
            'rebalance_threshold': 0.05  # 5% threshold for rebalancing
        }
        
        # Performance benchmarks
        self.benchmarks = {
            'btc_benchmark': 0.0,
            'market_benchmark': 0.0,
            'risk_free_rate': 0.02  # 2% annual risk-free rate
        }
        
        logger.info("Portfolio Manager initialized")
    
    async def initialize(self):
        """Initialize portfolio manager"""
        try:
            # Initialize exchange connection
            self.exchange = ccxt.binance({
                'apiKey': self.api_config.binance_api_key,
                'secret': self.api_config.binance_secret,
                'sandbox': self.api_config.binance_testnet,
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            })
            
            await self.exchange.load_markets()
            
            # Load existing portfolio state
            await self._load_portfolio_state()
            
            # Update current prices
            await self._update_current_prices()
            
            # Calculate initial metrics
            await self._calculate_portfolio_metrics()
            
            logger.info("Portfolio Manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Portfolio Manager: {e}")
            raise
    
    async def _load_portfolio_state(self):
        """Load existing portfolio state from database"""
        try:
            with get_db_session() as session:
                # Load open positions
                open_positions = session.query(Position).filter(
                    Position.status == PositionStatus.OPEN.value
                ).all()
                
                for pos in open_positions:
                    portfolio_position = PortfolioPosition(
                        symbol=pos.symbol,
                        direction=PositionDirection(pos.direction),
                        quantity=pos.quantity,
                        entry_price=pos.entry_price,
                        current_price=pos.entry_price,  # Will be updated
                        market_value=pos.quantity * pos.entry_price,
                        unrealized_pnl=0.0,
                        unrealized_pnl_pct=0.0,
                        cost_basis=pos.quantity * pos.entry_price,
                        stop_loss=pos.stop_loss,
                        take_profit=pos.take_profit,
                        entry_time=pos.timestamp,
                        status=PositionStatus(pos.status)
                    )
                    
                    self.positions[pos.symbol] = portfolio_position
                
                # Load recent performance snapshots
                recent_snapshots = session.query(PortfolioSnapshot).order_by(
                    PortfolioSnapshot.timestamp.desc()
                ).limit(30).all()
                
                self.daily_snapshots = [
                    {
                        'timestamp': snap.timestamp,
                        'total_value': snap.total_value,
                        'cash_balance': snap.cash_balance,
                        'total_pnl': snap.total_pnl,
                        'daily_pnl': snap.daily_pnl
                    }
                    for snap in reversed(recent_snapshots)
                ]
                
                # Set cash balance from latest snapshot
                if self.daily_snapshots:
                    self.cash_balance = self.daily_snapshots[-1]['cash_balance']
                else:
                    self.cash_balance = self.initial_capital
                
                logger.info(f"Loaded {len(self.positions)} positions and {len(self.daily_snapshots)} snapshots")
                
        except Exception as e:
            logger.error(f"Error loading portfolio state: {e}")
            self.cash_balance = self.initial_capital
    
    async def _update_current_prices(self):
        """Update current prices for all positions"""
        try:
            if not self.positions:
                return
            
            symbols = list(self.positions.keys())
            
            # Fetch current prices
            tickers = await self.exchange.fetch_tickers(symbols)
            
            for symbol, position in self.positions.items():
                if symbol in tickers:
                    current_price = float(tickers[symbol]['last'])
                    position.current_price = current_price
                    
                    # Update market value
                    if position.direction == PositionDirection.LONG:
                        position.market_value = position.quantity * current_price
                        position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
                    else:  # SHORT
                        position.market_value = position.quantity * current_price
                        position.unrealized_pnl = (position.entry_price - current_price) * position.quantity
                    
                    # Update unrealized P&L percentage
                    if position.cost_basis > 0:
                        position.unrealized_pnl_pct = (position.unrealized_pnl / position.cost_basis) * 100
                    
                    # Update max profit/loss tracking
                    position.max_profit = max(position.max_profit, position.unrealized_pnl)
                    position.max_loss = min(position.max_loss, position.unrealized_pnl)
                    
                    # Update days held
                    position.days_held = (datetime.utcnow() - position.entry_time).days
                    
                    position.last_updated = datetime.utcnow()
            
            logger.debug(f"Updated prices for {len(symbols)} positions")
            
        except Exception as e:
            logger.error(f"Error updating current prices: {e}")
    
    async def add_position(self, 
                          symbol: str, 
                          direction: PositionDirection, 
                          quantity: float, 
                          entry_price: float,
                          stop_loss: Optional[float] = None,
                          take_profit: Optional[float] = None) -> bool:
        """Add new position to portfolio"""
        try:
            # Validate position
            cost = quantity * entry_price
            if cost > self.cash_balance:
                logger.error(f"Insufficient cash balance: need ${cost:.2f}, have ${self.cash_balance:.2f}")
                return False
            
            # Check position limits
            if len(self.positions) >= self.position_limits['max_positions']:
                logger.error(f"Maximum positions limit reached: {self.position_limits['max_positions']}")
                return False
            
            # Create portfolio position
            portfolio_position = PortfolioPosition(
                symbol=symbol,
                direction=direction,
                quantity=quantity,
                entry_price=entry_price,
                current_price=entry_price,
                market_value=cost,
                unrealized_pnl=0.0,
                unrealized_pnl_pct=0.0,
                cost_basis=cost,
                stop_loss=stop_loss,
                take_profit=take_profit,
                entry_time=datetime.utcnow(),
                status=PositionStatus.OPEN
            )
            
            # Add to portfolio
            self.positions[symbol] = portfolio_position
            
            # Update cash balance
            self.cash_balance -= cost
            
            # Calculate position size percentage
            await self._calculate_portfolio_metrics()
            
            # Save to database
            await self._save_position_to_db(portfolio_position)
            
            logger.info(f"Added position: {symbol} {direction.value} {quantity} @ ${entry_price:.4f}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding position {symbol}: {e}")
            return False
    
    async def close_position(self, 
                           symbol: str, 
                           exit_price: Optional[float] = None,
                           quantity: Optional[float] = None) -> bool:
        """Close position (fully or partially)"""
        try:
            if symbol not in self.positions:
                logger.error(f"Position {symbol} not found")
                return False
            
            position = self.positions[symbol]
            
            # Use current market price if not provided
            if exit_price is None:
                await self._update_current_prices()
                exit_price = position.current_price
            
            # Default to full position closure
            if quantity is None:
                quantity = position.quantity
            
            # Validate quantity
            if quantity > position.quantity:
                logger.error(f"Cannot close {quantity} units, position only has {position.quantity}")
                return False
            
            # Calculate realized P&L
            if position.direction == PositionDirection.LONG:
                realized_pnl = (exit_price - position.entry_price) * quantity
            else:  # SHORT
                realized_pnl = (position.entry_price - exit_price) * quantity
            
            # Calculate proceeds
            proceeds = quantity * exit_price
            
            # Update cash balance
            self.cash_balance += proceeds
            
            # Update or remove position
            if quantity == position.quantity:
                # Full closure
                position.status = PositionStatus.CLOSED
                position.last_updated = datetime.utcnow()
                
                # Save trade record
                await self._save_trade_to_db(position, exit_price, realized_pnl)
                
                # Remove from active positions
                del self.positions[symbol]
                
                logger.info(f"Closed position: {symbol} {quantity} @ ${exit_price:.4f}, P&L: ${realized_pnl:.2f}")
            else:
                # Partial closure
                position.quantity -= quantity
                position.cost_basis = position.quantity * position.entry_price
                position.last_updated = datetime.utcnow()
                
                # Save partial trade record
                partial_position = PortfolioPosition(
                    symbol=symbol,
                    direction=position.direction,
                    quantity=quantity,
                    entry_price=position.entry_price,
                    current_price=exit_price,
                    market_value=quantity * exit_price,
                    unrealized_pnl=realized_pnl,
                    cost_basis=quantity * position.entry_price,
                    entry_time=position.entry_time,
                    status=PositionStatus.CLOSED
                )
                
                await self._save_trade_to_db(partial_position, exit_price, realized_pnl)
                
                logger.info(f"Partially closed position: {symbol} {quantity} @ ${exit_price:.4f}, P&L: ${realized_pnl:.2f}")
            
            # Update portfolio metrics
            await self._calculate_portfolio_metrics()
            
            return True
            
        except Exception as e:
            logger.error(f"Error closing position {symbol}: {e}")
            return False
    
    async def update_position_stops(self, 
                                  symbol: str, 
                                  stop_loss: Optional[float] = None,
                                  take_profit: Optional[float] = None) -> bool:
        """Update stop loss and take profit for position"""
        try:
            if symbol not in self.positions:
                logger.error(f"Position {symbol} not found")
                return False
            
            position = self.positions[symbol]
            
            if stop_loss is not None:
                position.stop_loss = stop_loss
            
            if take_profit is not None:
                position.take_profit = take_profit
            
            position.last_updated = datetime.utcnow()
            
            # Update in database
            await self._update_position_in_db(position)
            
            logger.info(f"Updated stops for {symbol}: SL=${stop_loss}, TP=${take_profit}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating position stops for {symbol}: {e}")
            return False
    
    async def check_stop_loss_take_profit(self) -> List[Dict[str, Any]]:
        """Check all positions for stop loss and take profit triggers"""
        triggers = []
        
        try:
            await self._update_current_prices()
            
            for symbol, position in self.positions.items():
                current_price = position.current_price
                
                # Check stop loss
                if position.stop_loss:
                    if ((position.direction == PositionDirection.LONG and current_price <= position.stop_loss) or
                        (position.direction == PositionDirection.SHORT and current_price >= position.stop_loss)):
                        
                        triggers.append({
                            'symbol': symbol,
                            'type': 'STOP_LOSS',
                            'trigger_price': position.stop_loss,
                            'current_price': current_price,
                            'direction': position.direction.value,
                            'quantity': position.quantity,
                            'unrealized_pnl': position.unrealized_pnl
                        })
                
                # Check take profit
                if position.take_profit:
                    if ((position.direction == PositionDirection.LONG and current_price >= position.take_profit) or
                        (position.direction == PositionDirection.SHORT and current_price <= position.take_profit)):
                        
                        triggers.append({
                            'symbol': symbol,
                            'type': 'TAKE_PROFIT',
                            'trigger_price': position.take_profit,
                            'current_price': current_price,
                            'direction': position.direction.value,
                            'quantity': position.quantity,
                            'unrealized_pnl': position.unrealized_pnl
                        })
            
            if triggers:
                logger.info(f"Found {len(triggers)} stop/take profit triggers")
            
            return triggers
            
        except Exception as e:
            logger.error(f"Error checking stop loss/take profit: {e}")
            return []
    
    async def get_portfolio_summary(self) -> PortfolioSummary:
        """Get comprehensive portfolio summary"""
        try:
            await self._update_current_prices()
            await self._calculate_portfolio_metrics()
            
            # Calculate basic metrics
            invested_amount = sum(pos.cost_basis for pos in self.positions.values())
            total_market_value = sum(pos.market_value for pos in self.positions.values())
            total_value = self.cash_balance + total_market_value
            available_balance = self.cash_balance
            
            # P&L calculations
            total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
            total_pnl_pct = (total_unrealized_pnl / invested_amount * 100) if invested_amount > 0 else 0
            
            # Daily P&L (simplified - would need historical data for accuracy)
            daily_pnl = self._calculate_daily_pnl()
            daily_pnl_pct = (daily_pnl / total_value * 100) if total_value > 0 else 0
            
            # Position counts
            long_positions = len([pos for pos in self.positions.values() if pos.direction == PositionDirection.LONG])
            short_positions = len([pos for pos in self.positions.values() if pos.direction == PositionDirection.SHORT])
            
            # Largest position percentage
            largest_position_pct = 0
            if self.positions and total_value > 0:
                largest_position_value = max(pos.market_value for pos in self.positions.values())
                largest_position_pct = (largest_position_value / total_value) * 100
            
            # Allocation calculations
            allocation_by_symbol = {}
            allocation_by_sector = {}
            
            if total_value > 0:
                for symbol, position in self.positions.items():
                    allocation_pct = (position.market_value / total_value) * 100
                    allocation_by_symbol[symbol] = allocation_pct
                    
                    # Sector allocation (simplified)
                    sector = self._get_symbol_sector(symbol)
                    allocation_by_sector[sector] = allocation_by_sector.get(sector, 0) + allocation_pct
            
            # Create summary
            summary = PortfolioSummary(
                total_value=total_value,
                cash_balance=self.cash_balance,
                invested_amount=invested_amount,
                available_balance=available_balance,
                total_pnl=total_unrealized_pnl,
                total_pnl_pct=total_pnl_pct,
                daily_pnl=daily_pnl,
                daily_pnl_pct=daily_pnl_pct,
                portfolio_beta=self.risk_metrics.get('portfolio_beta', 1.0),
                sharpe_ratio=self.risk_metrics.get('sharpe_ratio', 0.0),
                max_drawdown=self.risk_metrics.get('max_drawdown', 0.0),
                current_drawdown=self.risk_metrics.get('current_drawdown', 0.0),
                var_95=self.risk_metrics.get('var_95', 0.0),
                total_positions=len(self.positions),
                long_positions=long_positions,
                short_positions=short_positions,
                largest_position_pct=largest_position_pct,
                allocation_by_symbol=allocation_by_symbol,
                allocation_by_sector=allocation_by_sector,
                correlation_matrix=self.risk_metrics.get('correlation_matrix', {})
            )
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting portfolio summary: {e}")
            return self._create_empty_summary()
    
    def _calculate_daily_pnl(self) -> float:
        """Calculate daily P&L"""
        try:
            if len(self.daily_snapshots) < 2:
                return 0.0
            
            today_value = self.daily_snapshots[-1]['total_value']
            yesterday_value = self.daily_snapshots[-2]['total_value']
            
            return today_value - yesterday_value
            
        except Exception as e:
            logger.error(f"Error calculating daily P&L: {e}")
            return 0.0
    
    def _get_symbol_sector(self, symbol: str) -> str:
        """Get sector classification for symbol"""
        # Simplified sector classification
        if symbol.startswith('BTC'):
            return 'Bitcoin'
        elif symbol.startswith('ETH'):
            return 'Ethereum'
        elif symbol in ['ADA/USDT', 'DOT/USDT', 'SOL/USDT', 'AVAX/USDT']:
            return 'Layer 1'
        elif symbol in ['LINK/USDT', 'UNI/USDT', 'AAVE/USDT', 'SUSHI/USDT']:
            return 'DeFi'
        elif symbol in ['USDT', 'USDC', 'BUSD']:
            return 'Stablecoin'
        else:
            return 'Altcoin'
    
    async def analyze_rebalancing_opportunities(self) -> List[RebalanceRecommendation]:
        """Analyze portfolio for rebalancing opportunities"""
        try:
            recommendations = []
            summary = await self.get_portfolio_summary()
            
            # Check allocation targets vs current allocation
            current_sectors = summary.allocation_by_sector
            
            for target_sector, target_pct in self.allocation_targets.items():
                current_pct = current_sectors.get(target_sector, 0.0)
                deviation = abs(current_pct - target_pct * 100)
                
                if deviation > self.position_limits['rebalance_threshold'] * 100:
                    if current_pct < target_pct * 100:
                        # Need to increase allocation
                        action = RebalanceAction.INCREASE
                        reasoning = [f"Sector {target_sector} underweight: {current_pct:.1f}% vs target {target_pct*100:.1f}%"]
                    else:
                        # Need to decrease allocation
                        action = RebalanceAction.REDUCE
                        reasoning = [f"Sector {target_sector} overweight: {current_pct:.1f}% vs target {target_pct*100:.1f}%"]
                    
                    # Find representative symbol for sector
                    sector_symbol = self._get_sector_representative_symbol(target_sector)
                    
                    if sector_symbol:
                        # Calculate recommended adjustment
                        target_value = summary.total_value * target_pct
                        current_value = summary.total_value * (current_pct / 100)
                        adjustment_value = target_value - current_value
                        
                        # Estimate quantity needed (simplified)
                        estimated_price = await self._get_symbol_price(sector_symbol)
                        recommended_quantity = abs(adjustment_value / estimated_price) if estimated_price > 0 else 0
                        
                        priority = min(5, max(1, int(deviation / 5)))  # Priority based on deviation
                        
                        recommendation = RebalanceRecommendation(
                            symbol=sector_symbol,
                            action=action,
                            current_allocation_pct=current_pct,
                            target_allocation_pct=target_pct * 100,
                            recommended_quantity=recommended_quantity,
                            estimated_cost=abs(adjustment_value),
                            priority=priority,
                            reasoning=reasoning
                        )
                        
                        recommendations.append(recommendation)
            
            # Check individual position concentration
            for symbol, position in self.positions.items():
                position_pct = (position.market_value / summary.total_value) * 100
                
                if position_pct > self.position_limits['max_position_size'] * 100:
                    reduction_needed = position_pct - (self.position_limits['max_position_size'] * 100)
                    reduction_value = summary.total_value * (reduction_needed / 100)
                    reduction_quantity = reduction_value / position.current_price
                    
                    recommendation = RebalanceRecommendation(
                        symbol=symbol,
                        action=RebalanceAction.REDUCE,
                        current_allocation_pct=position_pct,
                        target_allocation_pct=self.position_limits['max_position_size'] * 100,
                        recommended_quantity=reduction_quantity,
                        estimated_cost=reduction_value,
                        priority=4,  # High priority for concentration risk
                        reasoning=[f"Position oversized: {position_pct:.1f}% vs max {self.position_limits['max_position_size']*100:.1f}%"]
                    )
                    
                    recommendations.append(recommendation)
            
            # Sort by priority
            recommendations.sort(key=lambda x: x.priority, reverse=True)
            
            logger.info(f"Generated {len(recommendations)} rebalancing recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error analyzing rebalancing opportunities: {e}")
            return []

def _get_sector_representative_symbol(self, sector: str) -> Optional[str]:
        """Get representative symbol for sector"""
        sector_symbols = {
            'BTC': 'BTC/USDT',
            'ETH': 'ETH/USDT',
            'ALT': 'ADA/USDT',  # Default altcoin
            'STABLE': 'USDT',
            'Bitcoin': 'BTC/USDT',
            'Ethereum': 'ETH/USDT',
            'Layer 1': 'ADA/USDT',
            'DeFi': 'UNI/USDT',
            'Altcoin': 'DOT/USDT'
        }
        return sector_symbols.get(sector)
    
    async def _get_symbol_price(self, symbol: str) -> float:
        """Get current price for symbol"""
        try:
            if symbol in self.positions:
                return self.positions[symbol].current_price
            
            ticker = await self.exchange.fetch_ticker(symbol)
            return float(ticker['last'])
            
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return 0.0
    
    async def _calculate_portfolio_metrics(self):
        """Calculate comprehensive portfolio metrics"""
        try:
            if not self.positions:
                return
            
            # Update position size percentages
            total_value = self.cash_balance + sum(pos.market_value for pos in self.positions.values())
            
            for position in self.positions.values():
                if total_value > 0:
                    position.position_size_pct = (position.market_value / total_value) * 100
                    position.risk_amount = abs(position.unrealized_pnl) if position.unrealized_pnl < 0 else 0
            
            # Calculate correlation matrix (simplified)
            await self._calculate_correlation_matrix()
            
            # Calculate portfolio beta (vs BTC)
            await self._calculate_portfolio_beta()
            
            # Calculate Sharpe ratio
            await self._calculate_sharpe_ratio()
            
            # Calculate drawdown metrics
            await self._calculate_drawdown_metrics()
            
            # Calculate VaR
            await self._calculate_var()
            
            self.risk_metrics['last_calculated'] = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
    
    async def _calculate_correlation_matrix(self):
        """Calculate correlation matrix between positions"""
        try:
            # Simplified correlation calculation
            # In a real implementation, this would use historical price data
            
            symbols = list(self.positions.keys())
            correlation_matrix = {}
            
            # Pre-defined correlation estimates for crypto assets
            crypto_correlations = {
                ('BTC/USDT', 'ETH/USDT'): 0.8,
                ('BTC/USDT', 'ADA/USDT'): 0.7,
                ('BTC/USDT', 'DOT/USDT'): 0.6,
                ('ETH/USDT', 'ADA/USDT'): 0.6,
                ('ETH/USDT', 'DOT/USDT'): 0.7,
                ('ADA/USDT', 'DOT/USDT'): 0.5
            }
            
            for symbol1 in symbols:
                correlation_matrix[symbol1] = {}
                for symbol2 in symbols:
                    if symbol1 == symbol2:
                        correlation_matrix[symbol1][symbol2] = 1.0
                    else:
                        # Look up correlation or use default
                        pair = tuple(sorted([symbol1, symbol2]))
                        correlation = crypto_correlations.get(pair, 0.3)  # Default 0.3
                        correlation_matrix[symbol1][symbol2] = correlation
            
            self.risk_metrics['correlation_matrix'] = correlation_matrix
            
        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {e}")
    
    async def _calculate_portfolio_beta(self):
        """Calculate portfolio beta relative to BTC"""
        try:
            # Simplified beta calculation
            # In practice, this would use regression analysis on historical returns
            
            btc_weight = 0.0
            if 'BTC/USDT' in self.positions:
                total_value = self.cash_balance + sum(pos.market_value for pos in self.positions.values())
                btc_value = self.positions['BTC/USDT'].market_value
                btc_weight = btc_value / total_value if total_value > 0 else 0
            
            # Estimate portfolio beta based on composition
            estimated_beta = btc_weight * 1.0  # BTC has beta of 1.0 vs itself
            
            for symbol, position in self.positions.items():
                if symbol != 'BTC/USDT':
                    weight = position.position_size_pct / 100
                    
                    # Estimated betas for major cryptocurrencies vs BTC
                    symbol_betas = {
                        'ETH/USDT': 1.2,
                        'ADA/USDT': 1.5,
                        'DOT/USDT': 1.3,
                        'LINK/USDT': 1.1,
                        'SOL/USDT': 1.4
                    }
                    
                    beta = symbol_betas.get(symbol, 1.2)  # Default beta
                    estimated_beta += weight * beta
            
            self.risk_metrics['portfolio_beta'] = estimated_beta
            
        except Exception as e:
            logger.error(f"Error calculating portfolio beta: {e}")
            self.risk_metrics['portfolio_beta'] = 1.0
    
    async def _calculate_sharpe_ratio(self):
        """Calculate Sharpe ratio"""
        try:
            if len(self.daily_snapshots) < 30:
                self.risk_metrics['sharpe_ratio'] = 0.0
                return
            
            # Calculate daily returns
            returns = []
            for i in range(1, len(self.daily_snapshots)):
                prev_value = self.daily_snapshots[i-1]['total_value']
                curr_value = self.daily_snapshots[i]['total_value']
                
                if prev_value > 0:
                    daily_return = (curr_value - prev_value) / prev_value
                    returns.append(daily_return)
            
            if not returns:
                self.risk_metrics['sharpe_ratio'] = 0.0
                return
            
            # Calculate annualized metrics
            mean_return = np.mean(returns) * 365  # Annualized
            std_return = np.std(returns) * np.sqrt(365)  # Annualized
            risk_free_rate = self.benchmarks['risk_free_rate']
            
            if std_return > 0:
                sharpe_ratio = (mean_return - risk_free_rate) / std_return
                self.risk_metrics['sharpe_ratio'] = sharpe_ratio
            else:
                self.risk_metrics['sharpe_ratio'] = 0.0
            
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            self.risk_metrics['sharpe_ratio'] = 0.0
    
    async def _calculate_drawdown_metrics(self):
        """Calculate drawdown metrics"""
        try:
            if len(self.daily_snapshots) < 2:
                self.risk_metrics['max_drawdown'] = 0.0
                self.risk_metrics['current_drawdown'] = 0.0
                return
            
            # Get portfolio values
            portfolio_values = [snap['total_value'] for snap in self.daily_snapshots]
            
            # Calculate drawdowns
            peak = portfolio_values[0]
            max_drawdown = 0.0
            current_drawdown = 0.0
            
            for value in portfolio_values:
                if value > peak:
                    peak = value
                    current_drawdown = 0.0
                else:
                    current_drawdown = (peak - value) / peak
                    max_drawdown = max(max_drawdown, current_drawdown)
            
            self.risk_metrics['max_drawdown'] = max_drawdown
            self.risk_metrics['current_drawdown'] = current_drawdown
            
        except Exception as e:
            logger.error(f"Error calculating drawdown metrics: {e}")
            self.risk_metrics['max_drawdown'] = 0.0
            self.risk_metrics['current_drawdown'] = 0.0
    
    async def _calculate_var(self, confidence: float = 0.95):
        """Calculate Value at Risk (VaR)"""
        try:
            if len(self.daily_snapshots) < 30:
                self.risk_metrics['var_95'] = 0.0
                return
            
            # Calculate daily returns
            returns = []
            for i in range(1, len(self.daily_snapshots)):
                prev_value = self.daily_snapshots[i-1]['total_value']
                curr_value = self.daily_snapshots[i]['total_value']
                
                if prev_value > 0:
                    daily_return = (curr_value - prev_value) / prev_value
                    returns.append(daily_return)
            
            if returns:
                # Calculate VaR using historical simulation
                sorted_returns = sorted(returns)
                var_index = int((1 - confidence) * len(sorted_returns))
                var_95 = abs(sorted_returns[var_index]) if var_index < len(sorted_returns) else 0
                
                self.risk_metrics['var_95'] = var_95
            else:
                self.risk_metrics['var_95'] = 0.0
            
        except Exception as e:
            logger.error(f"Error calculating VaR: {e}")
            self.risk_metrics['var_95'] = 0.0
    
    async def _save_position_to_db(self, position: PortfolioPosition):
        """Save position to database"""
        try:
            with get_db_session() as session:
                db_position = Position(
                    symbol=position.symbol,
                    direction=position.direction.value,
                    quantity=position.quantity,
                    entry_price=position.entry_price,
                    stop_loss=position.stop_loss,
                    take_profit=position.take_profit,
                    status=position.status.value,
                    timestamp=position.entry_time
                )
                session.add(db_position)
                
        except Exception as e:
            logger.error(f"Error saving position to database: {e}")
    
    async def _update_position_in_db(self, position: PortfolioPosition):
        """Update position in database"""
        try:
            with get_db_session() as session:
                db_position = session.query(Position).filter(
                    Position.symbol == position.symbol,
                    Position.status == PositionStatus.OPEN.value
                ).first()
                
                if db_position:
                    db_position.stop_loss = position.stop_loss
                    db_position.take_profit = position.take_profit
                    db_position.quantity = position.quantity
                
        except Exception as e:
            logger.error(f"Error updating position in database: {e}")
    
    async def _save_trade_to_db(self, position: PortfolioPosition, exit_price: float, realized_pnl: float):
        """Save completed trade to database"""
        try:
            with get_db_session() as session:
                trade = Trade(
                    symbol=position.symbol,
                    direction=position.direction.value,
                    quantity=position.quantity,
                    entry_price=position.entry_price,
                    exit_price=exit_price,
                    pnl=realized_pnl,
                    entry_time=position.entry_time,
                    exit_time=datetime.utcnow(),
                    strategy_id='PORTFOLIO_MANAGER',
                    confidence=85.0  # Default confidence
                )
                session.add(trade)
                
                # Add to trade history
                self.trade_history.append({
                    'symbol': position.symbol,
                    'direction': position.direction.value,
                    'quantity': position.quantity,
                    'entry_price': position.entry_price,
                    'exit_price': exit_price,
                    'pnl': realized_pnl,
                    'pnl_pct': (realized_pnl / (position.quantity * position.entry_price)) * 100,
                    'days_held': position.days_held,
                    'timestamp': datetime.utcnow()
                })
                
                # Keep only recent trades
                if len(self.trade_history) > 1000:
                    self.trade_history = self.trade_history[-1000:]
                
        except Exception as e:
            logger.error(f"Error saving trade to database: {e}")
    
    async def save_daily_snapshot(self):
        """Save daily portfolio snapshot"""
        try:
            summary = await self.get_portfolio_summary()
            
            with get_db_session() as session:
                snapshot = PortfolioSnapshot(
                    total_value=summary.total_value,
                    cash_balance=summary.cash_balance,
                    invested_amount=summary.invested_amount,
                    total_pnl=summary.total_pnl,
                    daily_pnl=summary.daily_pnl,
                    num_positions=summary.total_positions,
                    largest_position_pct=summary.largest_position_pct,
                    portfolio_beta=summary.portfolio_beta,
                    sharpe_ratio=summary.sharpe_ratio,
                    max_drawdown=summary.max_drawdown,
                    var_95=summary.var_95
                )
                session.add(snapshot)
                
                # Add to daily snapshots
                self.daily_snapshots.append({
                    'timestamp': datetime.utcnow(),
                    'total_value': summary.total_value,
                    'cash_balance': summary.cash_balance,
                    'total_pnl': summary.total_pnl,
                    'daily_pnl': summary.daily_pnl
                })
                
                # Keep only last 365 days
                if len(self.daily_snapshots) > 365:
                    self.daily_snapshots = self.daily_snapshots[-365:]
                
                logger.info(f"Saved daily portfolio snapshot: ${summary.total_value:.2f}")
                
        except Exception as e:
            logger.error(f"Error saving daily snapshot: {e}")
    
    def _create_empty_summary(self) -> PortfolioSummary:
        """Create empty portfolio summary for error cases"""
        return PortfolioSummary(
            total_value=self.cash_balance,
            cash_balance=self.cash_balance,
            invested_amount=0.0,
            available_balance=self.cash_balance,
            total_pnl=0.0,
            total_pnl_pct=0.0,
            daily_pnl=0.0,
            daily_pnl_pct=0.0,
            portfolio_beta=1.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            current_drawdown=0.0,
            var_95=0.0,
            total_positions=0,
            long_positions=0,
            short_positions=0,
            largest_position_pct=0.0
        )
    
    async def get_position_performance_analytics(self) -> Dict[str, Any]:
        """Get detailed position performance analytics"""
        try:
            await self._update_current_prices()
            
            analytics = {
                'individual_positions': [],
                'sector_performance': {},
                'risk_metrics': {},
                'performance_summary': {},
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Individual position analytics
            for symbol, position in self.positions.items():
                position_analytics = {
                    'symbol': symbol,
                    'direction': position.direction.value,
                    'entry_price': position.entry_price,
                    'current_price': position.current_price,
                    'unrealized_pnl': position.unrealized_pnl,
                    'unrealized_pnl_pct': position.unrealized_pnl_pct,
                    'max_profit': position.max_profit,
                    'max_loss': position.max_loss,
                    'days_held': position.days_held,
                    'position_size_pct': position.position_size_pct,
                    'risk_amount': position.risk_amount,
                    'stop_loss_distance': None,
                    'take_profit_distance': None
                }
                
                # Calculate stop/take distances
                if position.stop_loss:
                    position_analytics['stop_loss_distance'] = abs(position.current_price - position.stop_loss) / position.current_price * 100
                
                if position.take_profit:
                    position_analytics['take_profit_distance'] = abs(position.take_profit - position.current_price) / position.current_price * 100
                
                analytics['individual_positions'].append(position_analytics)
            
            # Sector performance
            summary = await self.get_portfolio_summary()
            for sector, allocation in summary.allocation_by_sector.items():
                sector_pnl = 0.0
                sector_positions = 0
                
                for position in self.positions.values():
                    if self._get_symbol_sector(position.symbol) == sector:
                        sector_pnl += position.unrealized_pnl
                        sector_positions += 1
                
                analytics['sector_performance'][sector] = {
                    'allocation_pct': allocation,
                    'unrealized_pnl': sector_pnl,
                    'position_count': sector_positions
                }
            
            # Risk metrics summary
            analytics['risk_metrics'] = {
                'portfolio_beta': self.risk_metrics.get('portfolio_beta', 1.0),
                'sharpe_ratio': self.risk_metrics.get('sharpe_ratio', 0.0),
                'max_drawdown': self.risk_metrics.get('max_drawdown', 0.0),
                'current_drawdown': self.risk_metrics.get('current_drawdown', 0.0),
                'var_95': self.risk_metrics.get('var_95', 0.0),
                'correlation_matrix': self.risk_metrics.get('correlation_matrix', {})
            }
            
            # Performance summary
            total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
            winning_positions = len([pos for pos in self.positions.values() if pos.unrealized_pnl > 0])
            losing_positions = len([pos for pos in self.positions.values() if pos.unrealized_pnl < 0])
            
            analytics['performance_summary'] = {
                'total_positions': len(self.positions),
                'winning_positions': winning_positions,
                'losing_positions': losing_positions,
                'win_rate': (winning_positions / len(self.positions) * 100) if self.positions else 0,
                'total_unrealized_pnl': total_unrealized_pnl,
                'average_position_pnl': total_unrealized_pnl / len(self.positions) if self.positions else 0,
                'largest_winner': max([pos.unrealized_pnl for pos in self.positions.values()]) if self.positions else 0,
                'largest_loser': min([pos.unrealized_pnl for pos in self.positions.values()]) if self.positions else 0
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting position performance analytics: {e}")
            return {'error': str(e)}
    
    async def get_portfolio_health_check(self) -> Dict[str, Any]:
        """Perform portfolio health check"""
        try:
            health = {
                'status': 'healthy',
                'checks': {},
                'warnings': [],
                'recommendations': []
            }
            
            summary = await self.get_portfolio_summary()
            
            # Check 1: Cash reserve
            cash_pct = (summary.cash_balance / summary.total_value) * 100 if summary.total_value > 0 else 100
            min_cash_pct = self.position_limits['min_cash_reserve'] * 100
            
            if cash_pct < min_cash_pct:
                health['warnings'].append(f"Low cash reserve: {cash_pct:.1f}% (minimum: {min_cash_pct:.1f}%)")
                health['recommendations'].append("Consider reducing position sizes to maintain cash reserve")
            
            health['checks']['cash_reserve'] = {
                'status': 'ok' if cash_pct >= min_cash_pct else 'warning',
                'current_pct': cash_pct,
                'minimum_pct': min_cash_pct
            }
            
            # Check 2: Position concentration
            max_position_pct = self.position_limits['max_position_size'] * 100
            
            if summary.largest_position_pct > max_position_pct:
                health['warnings'].append(f"Position concentration risk: {summary.largest_position_pct:.1f}% (max: {max_position_pct:.1f}%)")
                health['recommendations'].append("Consider reducing largest position size")
            
            health['checks']['position_concentration'] = {
                'status': 'ok' if summary.largest_position_pct <= max_position_pct else 'warning',
                'largest_position_pct': summary.largest_position_pct,
                'max_allowed_pct': max_position_pct
            }
            
            # Check 3: Drawdown
            if summary.current_drawdown > 0.15:  # 15% drawdown threshold
                health['warnings'].append(f"High current drawdown: {summary.current_drawdown:.1%}")
                health['recommendations'].append("Review risk management and consider reducing exposure")
                health['status'] = 'warning'
            
            health['checks']['drawdown'] = {
                'status': 'ok' if summary.current_drawdown <= 0.15 else 'warning',
                'current_drawdown': summary.current_drawdown,
                'max_drawdown': summary.max_drawdown
            }
            
            # Check 4: Position count
            if summary.total_positions >= self.position_limits['max_positions']:
                health['warnings'].append(f"Maximum positions reached: {summary.total_positions}")
                health['recommendations'].append("Close some positions before opening new ones")
            
            health['checks']['position_count'] = {
                'status': 'ok' if summary.total_positions < self.position_limits['max_positions'] else 'warning',
                'current_positions': summary.total_positions,
                'max_positions': self.position_limits['max_positions']
            }
            
            # Check 5: Stop loss coverage
            positions_without_stops = len([pos for pos in self.positions.values() if not pos.stop_loss])
            
            if positions_without_stops > 0:
                health['warnings'].append(f"{positions_without_stops} positions without stop loss")
                health['recommendations'].append("Set stop losses for all positions")
            
            health['checks']['stop_loss_coverage'] = {
                'status': 'ok' if positions_without_stops == 0 else 'warning',
                'positions_without_stops': positions_without_stops,
                'total_positions': summary.total_positions
            }
            
            # Overall health status
            if len(health['warnings']) > 3:
                health['status'] = 'unhealthy'
            elif len(health['warnings']) > 0:
                health['status'] = 'warning'
            
            health['timestamp'] = datetime.utcnow().isoformat()
            
            return health
            
        except Exception as e:
            logger.error(f"Error in portfolio health check: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def cleanup(self):
        """Cleanup portfolio manager resources"""
        try:
            # Save final snapshot
            await self.save_daily_snapshot()
            
            # Close exchange connection
            if self.exchange:
                await self.exchange.close()
            
            logger.info("Portfolio Manager cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during portfolio manager cleanup: {e}")

# Factory function
def create_portfolio_manager(api_config, trading_config) -> PortfolioManager:
    """Factory function to create Portfolio Manager instance"""
    return PortfolioManager(api_config, trading_config)

# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    # Mock configurations for testing
    class MockApiConfig:
        def __init__(self):
            self.binance_api_key = "test_api_key"
            self.binance_secret = "test_secret"
            self.binance_testnet = True
    
    class MockTradingConfig:
        def __init__(self):
            self.initial_capital = 10000
            self.max_open_positions = 5
            self.max_risk_per_trade = 0.02
    
    async def test_portfolio_manager():
        print("💼 Portfolio Manager Test")
        print("=" * 50)
        
        try:
            # Create portfolio manager (using mock for testing)
            api_config = MockApiConfig()
            trading_config = MockTradingConfig()
            portfolio_manager = create_portfolio_manager(api_config, trading_config)
            
            # Initialize with mock data (skip exchange initialization for testing)
            portfolio_manager.cash_balance = 10000
            
            print(f"✅ Portfolio Manager created")
            print(f"💰 Initial cash balance: ${portfolio_manager.cash_balance:,.2f}")
            print(f"🎯 Max positions: {portfolio_manager.position_limits['max_positions']}")
            print(f"📊 Allocation targets: {portfolio_manager.allocation_targets}")
            
            # Test 1: Add positions
            print(f"\n🧪 Test 1: Adding Positions")
            
            # Add BTC position
            btc_success = await portfolio_manager.add_position(
                symbol='BTC/USDT',
                direction=PositionDirection.LONG,
                quantity=0.2,
                entry_price=50000,
                stop_loss=47500,
                take_profit=55000
            )
            print(f"  BTC position added: {'✅' if btc_success else '❌'}")
            
            # Add ETH position
            eth_success = await portfolio_manager.add_position(
                symbol='ETH/USDT',
                direction=PositionDirection.LONG,
                quantity=3.0,
                entry_price=3000,
                stop_loss=2850,
                take_profit=3300
            )
            print(f"  ETH position added: {'✅' if eth_success else '❌'}")
            
            # Add ADA position
            ada_success = await portfolio_manager.add_position(
                symbol='ADA/USDT',
                direction=PositionDirection.LONG,
                quantity=10000,
                entry_price=0.45,
                stop_loss=0.42,
                take_profit=0.50
            )
            print(f"  ADA position added: {'✅' if ada_success else '❌'}")
            
            print(f"  Total positions: {len(portfolio_manager.positions)}")
            print(f"  Remaining cash: ${portfolio_manager.cash_balance:,.2f}")
            
            # Test 2: Update prices and calculate metrics
            print(f"\n🧪 Test 2: Price Updates and Metrics")
            
            # Simulate price changes
            portfolio_manager.positions['BTC/USDT'].current_price = 52000  # +4%
            portfolio_manager.positions['ETH/USDT'].current_price = 2950   # -1.67%
            portfolio_manager.positions['ADA/USDT'].current_price = 0.48   # +6.67%
            
            # Update P&L for each position
            for symbol, position in portfolio_manager.positions.items():
                if position.direction == PositionDirection.LONG:
                    position.unrealized_pnl = (position.current_price - position.entry_price) * position.quantity
                    position.market_value = position.quantity * position.current_price
                    position.unrealized_pnl_pct = (position.unrealized_pnl / position.cost_basis) * 100
            
            # Calculate portfolio metrics
            await portfolio_manager._calculate_portfolio_metrics()
            
            for symbol, position in portfolio_manager.positions.items():
                print(f"  {symbol}: ${position.current_price:,.2f} "
                      f"(P&L: ${position.unrealized_pnl:+.2f}, "
                      f"{position.unrealized_pnl_pct:+.1f}%)")
            
            # Test 3: Portfolio summary
            print(f"\n🧪 Test 3: Portfolio Summary")
            
            summary = await portfolio_manager.get_portfolio_summary()
            
            print(f"  Total Value: ${summary.total_value:,.2f}")
            print(f"  Cash Balance: ${summary.cash_balance:,.2f}")
            print(f"  Invested Amount: ${summary.invested_amount:,.2f}")
            print(f"  Total P&L: ${summary.total_pnl:+.2f} ({summary.total_pnl_pct:+.1f}%)")
            print(f"  Positions: {summary.total_positions} (Long: {summary.long_positions})")
            print(f"  Largest Position: {summary.largest_position_pct:.1f}%")
            
            print(f"  Allocation by symbol:")
            for symbol, allocation in summary.allocation_by_symbol.items():
                print(f"    {symbol}: {allocation:.1f}%")
            
            print(f"  Allocation by sector:")
            for sector, allocation in summary.allocation_by_sector.items():
                print(f"    {sector}: {allocation:.1f}%")
            
            # Test 4: Stop loss/take profit checks
            print(f"\n🧪 Test 4: Stop Loss/Take Profit Checks")
            
            # Simulate BTC price hitting take profit
            portfolio_manager.positions['BTC/USDT'].current_price = 55500
            
            triggers = await portfolio_manager.check_stop_loss_take_profit()
            
            print(f"  Triggers found: {len(triggers)}")
            for trigger in triggers:
                print(f"    {trigger['symbol']}: {trigger['type']} at ${trigger['trigger_price']}")
                print(f"      Current: ${trigger['current_price']}, P&L: ${trigger['unrealized_pnl']:+.2f}")
            
            # Test 5: Position closure
            print(f"\n🧪 Test 5: Position Closure")
            
            # Close BTC position at take profit
            btc_close_success = await portfolio_manager.close_position(
                symbol='BTC/USDT',
                exit_price=55000
            )
            print(f"  BTC position closed: {'✅' if btc_close_success else '❌'}")
            
            if btc_close_success:
                print(f"  Positions remaining: {len(portfolio_manager.positions)}")
                print(f"  Updated cash balance: ${portfolio_manager.cash_balance:,.2f}")
                print(f"  Trade history entries: {len(portfolio_manager.trade_history)}")
            
            # Test 6: Partial position closure
            print(f"\n🧪 Test 6: Partial Position Closure")
            
            # Partially close ETH position (50%)
            eth_partial_success = await portfolio_manager.close_position(
                symbol='ETH/USDT',
                exit_price=2980,
                quantity=1.5  # Close half
            )
            print(f"  ETH partial closure: {'✅' if eth_partial_success else '❌'}")
            
            if eth_partial_success and 'ETH/USDT' in portfolio_manager.positions:
                remaining_eth = portfolio_manager.positions['ETH/USDT']
                print(f"  Remaining ETH quantity: {remaining_eth.quantity}")
                print(f"  Remaining ETH value: ${remaining_eth.market_value:,.2f}")
            
            # Test 7: Rebalancing analysis
            print(f"\n🧪 Test 7: Rebalancing Analysis")
            
            rebalance_recommendations = await portfolio_manager.analyze_rebalancing_opportunities()
            
            print(f"  Rebalancing recommendations: {len(rebalance_recommendations)}")
            for rec in rebalance_recommendations:
                print(f"    {rec.symbol}: {rec.action.value}")
                print(f"      Current: {rec.current_allocation_pct:.1f}%, "
                      f"Target: {rec.target_allocation_pct:.1f}%")
                print(f"      Priority: {rec.priority}, Cost: ${rec.estimated_cost:.2f}")
                print(f"      Reasoning: {rec.reasoning[0] if rec.reasoning else 'No reason'}")
            
            # Test 8: Performance analytics
            print(f"\n🧪 Test 8: Performance Analytics")
            
            analytics = await portfolio_manager.get_position_performance_analytics()
            
            print(f"  Individual positions: {len(analytics['individual_positions'])}")
            for pos_analytics in analytics['individual_positions']:
                print(f"    {pos_analytics['symbol']}: "
                      f"{pos_analytics['unrealized_pnl_pct']:+.1f}%, "
                      f"Days held: {pos_analytics['days_held']}")
            
            print(f"  Sector performance:")
            for sector, perf in analytics['sector_performance'].items():
                print(f"    {sector}: {perf['allocation_pct']:.1f}%, "
                      f"P&L: ${perf['unrealized_pnl']:+.2f}")
            
            perf_summary = analytics['performance_summary']
            print(f"  Performance summary:")
            print(f"    Win rate: {perf_summary['win_rate']:.1f}%")
            print(f"    Total unrealized P&L: ${perf_summary['total_unrealized_pnl']:+.2f}")
            print(f"    Largest winner: ${perf_summary['largest_winner']:+.2f}")
            print(f"    Largest loser: ${perf_summary['largest_loser']:+.2f}")
            
            # Test 9: Health check
            print(f"\n🧪 Test 9: Portfolio Health Check")
            
            health = await portfolio_manager.get_portfolio_health_check()
            
            print(f"  Overall health: {health['status']}")
            print(f"  Checks performed: {len(health['checks'])}")
            print(f"  Warnings: {len(health['warnings'])}")
            
            for check_name, check_result in health['checks'].items():
                status_icon = "✅" if check_result['status'] == 'ok' else "⚠️"
                print(f"    {check_name}: {status_icon} {check_result['status']}")
            
            if health['warnings']:
                print(f"  Warning messages:")
                for warning in health['warnings'][:3]:  # Show first 3 warnings
                    print(f"    - {warning}")
            
            if health['recommendations']:
                print(f"  Recommendations:")
                for rec in health['recommendations'][:3]:  # Show first 3 recommendations
                    print(f"    - {rec}")
            
            # Test 10: Risk metrics
            print(f"\n🧪 Test 10: Risk Metrics")
            
            risk_metrics = portfolio_manager.risk_metrics
            print(f"  Portfolio Beta: {risk_metrics.get('portfolio_beta', 'N/A'):.2f}")
            print(f"  Sharpe Ratio: {risk_metrics.get('sharpe_ratio', 'N/A'):.2f}")
            print(f"  Max Drawdown: {risk_metrics.get('max_drawdown', 'N/A'):.1%}")
            print(f"  Current Drawdown: {risk_metrics.get('current_drawdown', 'N/A'):.1%}")
            print(f"  VaR 95%: {risk_metrics.get('var_95', 'N/A'):.1%}")
            
            # Correlation matrix
            correlation_matrix = risk_metrics.get('correlation_matrix', {})
            if correlation_matrix:
                print(f"  Correlation matrix:")
                symbols = list(correlation_matrix.keys())
                for symbol1 in symbols:
                    correlations = []
                    for symbol2 in symbols:
                        if symbol1 != symbol2:
                            corr = correlation_matrix[symbol1].get(symbol2, 0)
                            correlations.append(f"{symbol2}: {corr:.1f}")
                    print(f"    {symbol1}: {', '.join(correlations[:2])}")  # Show first 2 correlations
            
            # Test 11: Position management
            print(f"\n🧪 Test 11: Position Management")
            
            # Update stop losses
            for symbol in list(portfolio_manager.positions.keys()):
                new_stop = portfolio_manager.positions[symbol].current_price * 0.95  # 5% trailing stop
                update_success = await portfolio_manager.update_position_stops(
                    symbol=symbol,
                    stop_loss=new_stop
                )
                print(f"  Updated {symbol} stop loss: {'✅' if update_success else '❌'} (${new_stop:.2f})")
            
            # Test 12: Data serialization
            print(f"\n🧪 Test 12: Data Serialization")
            
            # Test portfolio position serialization
            for symbol, position in portfolio_manager.positions.items():
                position_dict = position.to_dict()
                print(f"  {symbol} serialization: {len(position_dict)} fields")
                
                # Key metrics
                key_metrics = {
                    'unrealized_pnl': position_dict['unrealized_pnl'],
                    'unrealized_pnl_pct': position_dict['unrealized_pnl_pct'],
                    'position_size_pct': position_dict['position_size_pct'],
                    'days_held': position_dict['days_held']
                }
                print(f"    Key metrics: {key_metrics}")
                break  # Show only first position
            
            # Test portfolio summary serialization
            summary_dict = summary.to_dict()
            print(f"  Summary serialization: {len(summary_dict)} fields")
            
            print(f"\n🎉 Portfolio Manager test completed successfully!")
            
            # Final portfolio state
            print(f"\n📈 Final Portfolio State:")
            final_summary = await portfolio_manager.get_portfolio_summary()
            print(f"  Total Value: ${final_summary.total_value:,.2f}")
            print(f"  Total Return: {final_summary.total_pnl_pct:+.1f}%")
            print(f"  Active Positions: {final_summary.total_positions}")
            print(f"  Cash Balance: ${final_summary.cash_balance:,.2f}")
            print(f"  Largest Position: {final_summary.largest_position_pct:.1f}%")
            
            # Portfolio composition
            print(f"  Portfolio Composition:")
            for symbol, allocation in final_summary.allocation_by_symbol.items():
                position = portfolio_manager.positions[symbol]
                print(f"    {symbol}: {allocation:.1f}% "
                      f"(${position.market_value:,.2f}, "
                      f"P&L: {position.unrealized_pnl_pct:+.1f}%)")
            
        except Exception as e:
            print(f"❌ Error in Portfolio Manager test: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Cleanup
            try:
                await portfolio_manager.cleanup()
                print(f"🧹 Portfolio Manager cleanup completed")
            except:
                pass
    