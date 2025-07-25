import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
import json
import sqlite3
from collections import defaultdict

from database.connection import get_db_session
from database.models import Trade, PerformanceMetric, PortfolioSnapshot

logger = logging.getLogger(__name__)

class PerformancePeriod(Enum):
    """Performance calculation periods"""
    DAILY = "DAILY"
    WEEKLY = "WEEKLY"
    MONTHLY = "MONTHLY"
    QUARTERLY = "QUARTERLY"
    YEARLY = "YEARLY"
    ALL_TIME = "ALL_TIME"

class PerformanceMetricType(Enum):
    """Types of performance metrics"""
    RETURN = "RETURN"
    VOLATILITY = "VOLATILITY"
    SHARPE_RATIO = "SHARPE_RATIO"
    SORTINO_RATIO = "SORTINO_RATIO"
    MAX_DRAWDOWN = "MAX_DRAWDOWN"
    WIN_RATE = "WIN_RATE"
    PROFIT_FACTOR = "PROFIT_FACTOR"
    CALMAR_RATIO = "CALMAR_RATIO"
    VAR = "VAR"
    CVAR = "CVAR"

@dataclass
class PerformanceSnapshot:
    """Performance snapshot at a point in time"""
    timestamp: datetime
    total_value: float
    daily_pnl: float
    total_pnl: float
    daily_return: float
    cumulative_return: float
    
    # Risk metrics
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    current_drawdown: float
    
    # Trading metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    
    # Position metrics
    active_positions: int
    cash_balance: float
    invested_amount: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'total_value': self.total_value,
            'daily_pnl': self.daily_pnl,
            'total_pnl': self.total_pnl,
            'daily_return': self.daily_return,
            'cumulative_return': self.cumulative_return,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'current_drawdown': self.current_drawdown,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'active_positions': self.active_positions,
            'cash_balance': self.cash_balance,
            'invested_amount': self.invested_amount
        }

@dataclass
class TradeAnalysis:
    """Comprehensive trade analysis"""
    symbol: str
    strategy_id: str
    direction: str
    
    # Trade metrics
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_percentage: float
    
    # Timing metrics
    entry_time: datetime
    exit_time: datetime
    hold_duration: timedelta
    hold_duration_hours: float
    
    # Performance metrics
    max_profit: float
    max_loss: float
    mae: float  # Maximum Adverse Excursion
    mfe: float  # Maximum Favorable Excursion
    
    # Risk metrics
    risk_amount: float
    reward_amount: float
    risk_reward_ratio: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'symbol': self.symbol,
            'strategy_id': self.strategy_id,
            'direction': self.direction,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'quantity': self.quantity,
            'pnl': self.pnl,
            'pnl_percentage': self.pnl_percentage,
            'entry_time': self.entry_time.isoformat(),
            'exit_time': self.exit_time.isoformat(),
            'hold_duration_hours': self.hold_duration_hours,
            'max_profit': self.max_profit,
            'max_loss': self.max_loss,
            'mae': self.mae,
            'mfe': self.mfe,
            'risk_amount': self.risk_amount,
            'reward_amount': self.reward_amount,
            'risk_reward_ratio': self.risk_reward_ratio
        }

@dataclass
class StrategyPerformance:
    """Strategy-specific performance metrics"""
    strategy_id: str
    
    # Basic metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    
    # P&L metrics
    total_pnl: float
    gross_profit: float
    gross_loss: float
    profit_factor: float
    
    # Risk metrics
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Trade metrics
    avg_winning_trade: float
    avg_losing_trade: float
    largest_winner: float
    largest_loser: float
    avg_hold_time: float
    
    # Efficiency metrics
    expectancy: float
    recovery_factor: float
    payoff_ratio: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'strategy_id': self.strategy_id,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'total_pnl': self.total_pnl,
            'gross_profit': self.gross_profit,
            'gross_loss': self.gross_loss,
            'profit_factor': self.profit_factor,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'avg_winning_trade': self.avg_winning_trade,
            'avg_losing_trade': self.avg_losing_trade,
            'largest_winner': self.largest_winner,
            'largest_loser': self.largest_loser,
            'avg_hold_time': self.avg_hold_time,
            'expectancy': self.expectancy,
            'recovery_factor': self.recovery_factor,
            'payoff_ratio': self.payoff_ratio
        }

class PerformanceTracker:
    """Advanced Performance Tracking and Analysis System"""
    
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        
        # Performance data storage
        self.portfolio_snapshots: List[PerformanceSnapshot] = []
        self.trade_analyses: List[TradeAnalysis] = []
        self.daily_returns: List[float] = []
        self.portfolio_values: List[float] = []
        
        # Current state
        self.current_portfolio_value = initial_capital
        self.current_cash_balance = initial_capital
        self.current_invested_amount = 0.0
        
        # Benchmark data
        self.benchmark_values: List[float] = []
        self.benchmark_returns: List[float] = []
        
        # Strategy tracking
        self.strategy_performances: Dict[str, StrategyPerformance] = {}
        self.strategy_trades: Dict[str, List[TradeAnalysis]] = defaultdict(list)
        
        # Risk-free rate (annual)
        self.risk_free_rate = 0.02  # 2%
        
        # Performance calculation cache
        self._performance_cache = {}
        self._cache_timestamp = None
        self._cache_duration = timedelta(minutes=5)
        
        # Statistics
        self.stats = {
            'total_snapshots': 0,
            'total_trades_analyzed': 0,
            'last_update': None,
            'calculation_count': 0
        }
        
        logger.info("Performance Tracker initialized")
    
    async def initialize(self):
        """Initialize performance tracker"""
        try:
            # Load historical data from database
            await self._load_historical_data()
            
            # Calculate initial metrics
            await self._calculate_all_metrics()
            
            logger.info("Performance Tracker initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Performance Tracker: {e}")
            raise
    
    async def _load_historical_data(self):
        """Load historical performance data from database"""
        try:
            with get_db_session() as session:
                # Load portfolio snapshots
                snapshots = session.query(PortfolioSnapshot).order_by(
                    PortfolioSnapshot.timestamp
                ).all()
                
                for snapshot in snapshots:
                    perf_snapshot = PerformanceSnapshot(
                        timestamp=snapshot.timestamp,
                        total_value=snapshot.total_value,
                        daily_pnl=snapshot.daily_pnl,
                        total_pnl=snapshot.total_pnl,
                        daily_return=snapshot.daily_pnl / snapshot.total_value * 100 if snapshot.total_value > 0 else 0,
                        cumulative_return=(snapshot.total_value - self.initial_capital) / self.initial_capital * 100,
                        volatility=0.0,  # Will be calculated
                        sharpe_ratio=snapshot.sharpe_ratio,
                        max_drawdown=snapshot.max_drawdown,
                        current_drawdown=0.0,  # Will be calculated
                        total_trades=0,  # Will be calculated
                        winning_trades=0,
                        losing_trades=0,
                        win_rate=0.0,
                        profit_factor=0.0,
                        active_positions=snapshot.num_positions,
                        cash_balance=snapshot.cash_balance,
                        invested_amount=snapshot.invested_amount
                    )
                    
                    self.portfolio_snapshots.append(perf_snapshot)
                    self.portfolio_values.append(snapshot.total_value)
                    
                    if len(self.portfolio_values) > 1:
                        daily_return = (snapshot.total_value - self.portfolio_values[-2]) / self.portfolio_values[-2]
                        self.daily_returns.append(daily_return)
                
                # Load completed trades
                trades = session.query(Trade).filter(
                    Trade.exit_time.isnot(None)
                ).order_by(Trade.exit_time).all()
                
                for trade in trades:
                    trade_analysis = await self._analyze_trade(trade)
                    if trade_analysis:
                        self.trade_analyses.append(trade_analysis)
                        self.strategy_trades[trade.strategy_id].append(trade_analysis)
                
                # Update current state
                if self.portfolio_snapshots:
                    latest = self.portfolio_snapshots[-1]
                    self.current_portfolio_value = latest.total_value
                    self.current_cash_balance = latest.cash_balance
                    self.current_invested_amount = latest.invested_amount
                
                logger.info(f"Loaded {len(self.portfolio_snapshots)} snapshots and {len(self.trade_analyses)} trades")
                
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
    
    async def _analyze_trade(self, trade) -> Optional[TradeAnalysis]:
        """Analyze individual trade performance"""
        try:
            if not trade.exit_time:
                return None
            
            # Calculate basic metrics
            pnl = trade.pnl
            cost_basis = trade.quantity * trade.entry_price
            pnl_percentage = (pnl / cost_basis) * 100 if cost_basis > 0 else 0
            
            # Calculate hold duration
            hold_duration = trade.exit_time - trade.entry_time
            hold_duration_hours = hold_duration.total_seconds() / 3600
            
            # Calculate risk/reward (simplified)
            if trade.direction == 'LONG':
                risk_amount = abs(trade.entry_price - (trade.stop_loss or trade.entry_price * 0.95)) * trade.quantity
                reward_amount = abs((trade.take_profit or trade.entry_price * 1.05) - trade.entry_price) * trade.quantity
            else:  # SHORT
                risk_amount = abs((trade.stop_loss or trade.entry_price * 1.05) - trade.entry_price) * trade.quantity
                reward_amount = abs(trade.entry_price - (trade.take_profit or trade.entry_price * 0.95)) * trade.quantity
            
            risk_reward_ratio = reward_amount / risk_amount if risk_amount > 0 else 0
            
            # For simplicity, set MAE/MFE to PnL (would need tick data for accurate calculation)
            mae = min(0, pnl)  # Maximum Adverse Excursion
            mfe = max(0, pnl)  # Maximum Favorable Excursion
            
            trade_analysis = TradeAnalysis(
                symbol=trade.symbol,
                strategy_id=trade.strategy_id,
                direction=trade.direction,
                entry_price=trade.entry_price,
                exit_price=trade.exit_price,
                quantity=trade.quantity,
                pnl=pnl,
                pnl_percentage=pnl_percentage,
                entry_time=trade.entry_time,
                exit_time=trade.exit_time,
                hold_duration=hold_duration,
                hold_duration_hours=hold_duration_hours,
                max_profit=mfe,
                max_loss=mae,
                mae=mae,
                mfe=mfe,
                risk_amount=risk_amount,
                reward_amount=reward_amount,
                risk_reward_ratio=risk_reward_ratio
            )
            
            return trade_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing trade {trade.id}: {e}")
            return None
    
    async def update_portfolio_snapshot(self, 
                                      portfolio_value: float,
                                      cash_balance: float,
                                      invested_amount: float,
                                      active_positions: int,
                                      daily_pnl: float = None):
        """Update portfolio snapshot"""
        try:
            # Calculate daily P&L if not provided
            if daily_pnl is None:
                if self.portfolio_values:
                    daily_pnl = portfolio_value - self.portfolio_values[-1]
                else:
                    daily_pnl = portfolio_value - self.initial_capital
            
            # Calculate returns
            daily_return = (daily_pnl / (portfolio_value - daily_pnl)) * 100 if (portfolio_value - daily_pnl) > 0 else 0
            cumulative_return = ((portfolio_value - self.initial_capital) / self.initial_capital) * 100
            
            # Calculate metrics
            metrics = await self._calculate_current_metrics(portfolio_value)
            
            # Create snapshot
            snapshot = PerformanceSnapshot(
                timestamp=datetime.utcnow(),
                total_value=portfolio_value,
                daily_pnl=daily_pnl,
                total_pnl=portfolio_value - self.initial_capital,
                daily_return=daily_return,
                cumulative_return=cumulative_return,
                volatility=metrics.get('volatility', 0),
                sharpe_ratio=metrics.get('sharpe_ratio', 0),
                max_drawdown=metrics.get('max_drawdown', 0),
                current_drawdown=metrics.get('current_drawdown', 0),
                total_trades=len(self.trade_analyses),
                winning_trades=len([t for t in self.trade_analyses if t.pnl > 0]),
                losing_trades=len([t for t in self.trade_analyses if t.pnl < 0]),
                win_rate=metrics.get('win_rate', 0),
                profit_factor=metrics.get('profit_factor', 0),
                active_positions=active_positions,
                cash_balance=cash_balance,
                invested_amount=invested_amount
            )
            
            # Add to history
            self.portfolio_snapshots.append(snapshot)
            self.portfolio_values.append(portfolio_value)
            
            if len(self.portfolio_values) > 1:
                daily_return_decimal = daily_pnl / (portfolio_value - daily_pnl) if (portfolio_value - daily_pnl) > 0 else 0
                self.daily_returns.append(daily_return_decimal)
            
            # Update current state
            self.current_portfolio_value = portfolio_value
            self.current_cash_balance = cash_balance
            self.current_invested_amount = invested_amount
            
            # Clear cache
            self._performance_cache.clear()
            
            # Update statistics
            self.stats['total_snapshots'] += 1
            self.stats['last_update'] = datetime.utcnow()
            
            # Keep only recent data (configurable)
            max_snapshots = 10000
            if len(self.portfolio_snapshots) > max_snapshots:
                self.portfolio_snapshots = self.portfolio_snapshots[-max_snapshots:]
                self.portfolio_values = self.portfolio_values[-max_snapshots:]
                self.daily_returns = self.daily_returns[-max_snapshots:]
            
            logger.debug(f"Portfolio snapshot updated: ${portfolio_value:,.2f}")
            
        except Exception as e:
            logger.error(f"Error updating portfolio snapshot: {e}")
    
    async def add_completed_trade(self, trade_data: Dict[str, Any]):
        """Add completed trade for analysis"""
        try:
            # Create trade analysis
            trade_analysis = TradeAnalysis(
                symbol=trade_data['symbol'],
                strategy_id=trade_data.get('strategy_id', 'unknown'),
                direction=trade_data['direction'],
                entry_price=trade_data['entry_price'],
                exit_price=trade_data['exit_price'],
                quantity=trade_data['quantity'],
                pnl=trade_data['pnl'],
                pnl_percentage=(trade_data['pnl'] / (trade_data['quantity'] * trade_data['entry_price'])) * 100,
                entry_time=trade_data['entry_time'],
                exit_time=trade_data['exit_time'],
                hold_duration=trade_data['exit_time'] - trade_data['entry_time'],
                hold_duration_hours=(trade_data['exit_time'] - trade_data['entry_time']).total_seconds() / 3600,
                max_profit=trade_data.get('max_profit', max(0, trade_data['pnl'])),
                max_loss=trade_data.get('max_loss', min(0, trade_data['pnl'])),
                mae=trade_data.get('mae', min(0, trade_data['pnl'])),
                mfe=trade_data.get('mfe', max(0, trade_data['pnl'])),
                risk_amount=trade_data.get('risk_amount', 0),
                reward_amount=trade_data.get('reward_amount', 0),
                risk_reward_ratio=trade_data.get('risk_reward_ratio', 0)
            )
            
            # Add to collections
            self.trade_analyses.append(trade_analysis)
            self.strategy_trades[trade_analysis.strategy_id].append(trade_analysis)
            
            # Clear cache
            self._performance_cache.clear()
            
            # Update statistics
            self.stats['total_trades_analyzed'] += 1
            
            logger.debug(f"Added trade analysis: {trade_analysis.symbol} {trade_analysis.direction} P&L: ${trade_analysis.pnl:.2f}")
            
        except Exception as e:
            logger.error(f"Error adding completed trade: {e}")
    
    async def _calculate_current_metrics(self, current_value: float) -> Dict[str, float]:
        """Calculate current performance metrics"""
        try:
            metrics = {}
            
            if len(self.daily_returns) == 0:
                return metrics
            
            # Volatility (annualized)
            if len(self.daily_returns) > 1:
                volatility = np.std(self.daily_returns) * np.sqrt(365)
                metrics['volatility'] = volatility
            
            # Sharpe ratio
            mean_return = np.mean(self.daily_returns)
            excess_return = mean_return - (self.risk_free_rate / 365)
            if 'volatility' in metrics and metrics['volatility'] > 0:
                metrics['sharpe_ratio'] = excess_return / (metrics['volatility'] / np.sqrt(365))
            
            # Drawdown metrics
            if self.portfolio_values:
                peak = self.portfolio_values[0]
                max_drawdown = 0
                current_drawdown = 0
                
                for value in self.portfolio_values:
                    if value > peak:
                        peak = value
                        current_drawdown = 0
                    else:
                        current_drawdown = (peak - value) / peak
                        max_drawdown = max(max_drawdown, current_drawdown)
                
                metrics['max_drawdown'] = max_drawdown
                metrics['current_drawdown'] = current_drawdown
            
            # Trading metrics
            if self.trade_analyses:
                winning_trades = [t for t in self.trade_analyses if t.pnl > 0]
                losing_trades = [t for t in self.trade_analyses if t.pnl < 0]
                
                metrics['win_rate'] = (len(winning_trades) / len(self.trade_analyses)) * 100
                
                if winning_trades and losing_trades:
                    gross_profit = sum(t.pnl for t in winning_trades)
                    gross_loss = abs(sum(t.pnl for t in losing_trades))
                    metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else 0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating current metrics: {e}")
            return {}
    
    async def _calculate_all_metrics(self):
        """Calculate all performance metrics and update cache"""
        try:
            if not self.portfolio_values:
                return
            
            # Clear existing cache
            self._performance_cache.clear()
            
            # Calculate for different periods
            for period in PerformancePeriod:
                period_metrics = await self._calculate_period_metrics(period)
                self._performance_cache[period.value] = period_metrics
            
            # Update cache timestamp
            self._cache_timestamp = datetime.utcnow()
            self.stats['calculation_count'] += 1
            
        except Exception as e:
            logger.error(f"Error calculating all metrics: {e}")
    
    async def _calculate_period_metrics(self, period: PerformancePeriod) -> Dict[str, float]:
        """Calculate metrics for specific period"""
        try:
            # Get data for period
            period_data = self._get_period_data(period)
            
            if not period_data['values'] or len(period_data['values']) < 2:
                return {}
            
            values = period_data['values']
            returns = period_data['returns']
            trades = period_data['trades']
            
            metrics = {}
            
            # Basic return metrics
            total_return = ((values[-1] - values[0]) / values[0]) * 100 if values[0] > 0 else 0
            metrics['total_return'] = total_return
            
            # Annualized return
            if period != PerformancePeriod.ALL_TIME:
                days = len(values)
                if days > 0:
                    annualized_return = ((values[-1] / values[0]) ** (365 / days) - 1) * 100
                    metrics['annualized_return'] = annualized_return
            
            # Volatility (annualized)
            if len(returns) > 1:
                volatility = np.std(returns) * np.sqrt(365) * 100
                metrics['volatility'] = volatility
                
                # Sharpe ratio
                mean_return = np.mean(returns)
                excess_return = mean_return - (self.risk_free_rate / 365)
                sharpe_ratio = excess_return / (volatility / 100 / np.sqrt(365)) if volatility > 0 else 0
                metrics['sharpe_ratio'] = sharpe_ratio
                
                # Sortino ratio (downside deviation)
                downside_returns = [r for r in returns if r < 0]
                if downside_returns:
                    downside_deviation = np.std(downside_returns) * np.sqrt(365) * 100
                    sortino_ratio = excess_return / (downside_deviation / 100 / np.sqrt(365)) if downside_deviation > 0 else 0
                    metrics['sortino_ratio'] = sortino_ratio
            
            # Drawdown metrics
            peak = values[0]
            drawdowns = []
            max_drawdown = 0
            
            for value in values:
                if value > peak:
                    peak = value
                
                drawdown = (peak - value) / peak if peak > 0 else 0
                drawdowns.append(drawdown)
                max_drawdown = max(max_drawdown, drawdown)
            
            metrics['max_drawdown'] = max_drawdown * 100
            metrics['current_drawdown'] = drawdowns[-1] * 100 if drawdowns else 0
            
            # Calmar ratio
            if max_drawdown > 0 and 'annualized_return' in metrics:
                calmar_ratio = metrics['annualized_return'] / (max_drawdown * 100)
                metrics['calmar_ratio'] = calmar_ratio
            
            # Trading metrics
            if trades:
                winning_trades = [t for t in trades if t.pnl > 0]
                losing_trades = [t for t in trades if t.pnl < 0]
                
                metrics['total_trades'] = len(trades)
                metrics['winning_trades'] = len(winning_trades)
                metrics['losing_trades'] = len(losing_trades)
                metrics['win_rate'] = (len(winning_trades) / len(trades)) * 100
                
                # P&L metrics
                total_pnl = sum(t.pnl for t in trades)
                gross_profit = sum(t.pnl for t in winning_trades) if winning_trades else 0
                gross_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 0
                
                metrics['total_pnl'] = total_pnl
                metrics['gross_profit'] = gross_profit
                metrics['gross_loss'] = gross_loss
                metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else 0
                
                # Average trade metrics
                metrics['avg_trade'] = total_pnl / len(trades)
                if winning_trades:
                    metrics['avg_winning_trade'] = gross_profit / len(winning_trades)
                    metrics['largest_winner'] = max(t.pnl for t in winning_trades)
                if losing_trades:
                    metrics['avg_losing_trade'] = -gross_loss / len(losing_trades)
                    metrics['largest_loser'] = min(t.pnl for t in losing_trades)
                
                # Expectancy
                if len(trades) > 0:
                    win_rate_decimal = len(winning_trades) / len(trades)
                    avg_win = gross_profit / len(winning_trades) if winning_trades else 0
                    avg_loss = gross_loss / len(losing_trades) if losing_trades else 0
                    expectancy = (win_rate_decimal * avg_win) - ((1 - win_rate_decimal) * avg_loss)
                    metrics['expectancy'] = expectancy
                
                # Hold time analysis
                hold_times = [t.hold_duration_hours for t in trades]
                metrics['avg_hold_time'] = np.mean(hold_times)
                metrics['median_hold_time'] = np.median(hold_times)
            
            # VaR calculations
            if len(returns) >= 30:
                var_95 = np.percentile(returns, 5) * 100  # 5th percentile
                var_99 = np.percentile(returns, 1) * 100  # 1st percentile
                metrics['var_95'] = var_95
                metrics['var_99'] = var_99
                
                # Conditional VaR (Expected Shortfall)
                cvar_95 = np.mean([r for r in returns if r <= np.percentile(returns, 5)]) * 100
                metrics['cvar_95'] = cvar_95
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating period metrics for {period}: {e}")
            return {}
    
    def _get_period_data(self, period: PerformancePeriod) -> Dict[str, List]:
        """Get data for specific period"""
        
        if period == PerformancePeriod.ALL_TIME:
            return {
                'values': self.portfolio_values.copy(),
                'returns': self.daily_returns.copy(),
                'trades': self.trade_analyses.copy()
            }
        
        # Calculate cutoff date
        now = datetime.utcnow()
        
        if period == PerformancePeriod.DAILY:
            cutoff = now - timedelta(days=1)
        elif period == PerformancePeriod.WEEKLY:
            cutoff = now - timedelta(days=7)
        elif period == PerformancePeriod.MONTHLY:
            cutoff = now - timedelta(days=30)
        elif period == PerformancePeriod.QUARTERLY:
            cutoff = now - timedelta(days=90)
        elif period == PerformancePeriod.YEARLY:
            cutoff = now - timedelta(days=365)
        else:
            cutoff = now - timedelta(days=30)  # Default to monthly
        
        # Filter data
        period_snapshots = [s for s in self.portfolio_snapshots if s.timestamp >= cutoff]
        period_values = [s.total_value for s in period_snapshots]
        
        # Calculate returns for period
        period_returns = []
        for i in range(1, len(period_values)):
            if period_values[i-1] > 0:
                daily_return = (period_values[i] - period_values[i-1]) / period_values[i-1]
                period_returns.append(daily_return)
        
        # Filter trades
        period_trades = [t for t in self.trade_analyses if t.exit_time >= cutoff]
        
        return {
            'values': period_values,
            'returns': period_returns,
            'trades': period_trades
        }
    
    async def get_performance_summary(self, period: PerformancePeriod = PerformancePeriod.ALL_TIME) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        try:
            # Check cache
            if (self._cache_timestamp and 
                datetime.utcnow() - self._cache_timestamp < self._cache_duration and
                period.value in self._performance_cache):
                
                cached_metrics = self._performance_cache[period.value]
            else:
                # Recalculate metrics
                await self._calculate_all_metrics()
                cached_metrics = self._performance_cache.get(period.value, {})
            
            # Add current state information
            summary = {
                'period': period.value,
                'current_portfolio_value': self.current_portfolio_value,
                'initial_capital': self.initial_capital,
                'cash_balance': self.current_cash_balance,
                'invested_amount': self.current_invested_amount,
                'total_return': cached_metrics.get('total_return', 0),
                'annualized_return': cached_metrics.get('annualized_return', 0),
                'volatility': cached_metrics.get('volatility', 0),
                'sharpe_ratio': cached_metrics.get('sharpe_ratio', 0),
                'sortino_ratio': cached_metrics.get('sortino_ratio', 0),
                'calmar_ratio': cached_metrics.get('calmar_ratio', 0),
                'max_drawdown': cached_metrics.get('max_drawdown', 0),
                'current_drawdown': cached_metrics.get('current_drawdown', 0),
                'var_95': cached_metrics.get('var_95', 0),
                'cvar_95': cached_metrics.get('cvar_95', 0),
                'total_trades': cached_metrics.get('total_trades', 0),
                'winning_trades': cached_metrics.get('winning_trades', 0),
                'losing_trades': cached_metrics.get('losing_trades', 0),
                'win_rate': cached_metrics.get('win_rate', 0),
                'profit_factor': cached_metrics.get('profit_factor', 0),
                'expectancy': cached_metrics.get('expectancy', 0),
                'avg_winning_trade': cached_metrics.get('avg_winning_trade', 0),
                'avg_losing_trade': cached_metrics.get('avg_losing_trade', 0),
                'largest_winner': cached_metrics.get('largest_winner', 0),
                'largest_loser': cached_metrics.get('largest_loser', 0),
                'avg_hold_time': cached_metrics.get('avg_hold_time', 0),
                'last_updated': self.stats['last_update'].isoformat() if self.stats['last_update'] else None
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {'error': str(e)}
    
    async def get_strategy_performance(self, strategy_id: str = None) -> Dict[str, StrategyPerformance]:
        """Get strategy-specific performance analysis"""
        try:
            strategy_performances = {}
            
            # Get strategies to analyze
            strategies = [strategy_id] if strategy_id else list(self.strategy_trades.keys())
            
            for strat_id in strategies:
                trades = self.strategy_trades.get(strat_id, [])
                
                if not trades:
                    continue
                
                # Calculate strategy metrics
                winning_trades = [t for t in trades if t.pnl > 0]
                losing_trades = [t for t in trades if t.pnl < 0]
                
                total_pnl = sum(t.pnl for t in trades)
                gross_profit = sum(t.pnl for t in winning_trades) if winning_trades else 0
                gross_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 0
                
                win_rate = (len(winning_trades) / len(trades)) * 100
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
                
                # Risk metrics (simplified)
                returns = [t.pnl_percentage / 100 for t in trades]
                
                if len(returns) > 1:
                    volatility = np.std(returns) * np.sqrt(252) * 100  # Annualized
                    mean_return = np.mean(returns)
                    sharpe_ratio = mean_return / (volatility / 100) if volatility > 0 else 0
                    
                    # Sortino ratio
                    downside_returns = [r for r in returns if r < 0]
                    if downside_returns:
                        downside_deviation = np.std(downside_returns) * np.sqrt(252) * 100
                        sortino_ratio = mean_return / (downside_deviation / 100) if downside_deviation > 0 else 0
                    else:
                        sortino_ratio = sharpe_ratio
                else:
                    sharpe_ratio = 0
                    sortino_ratio = 0
                
                # Drawdown calculation for strategy
                cumulative_pnl = []
                running_total = 0
                for trade in sorted(trades, key=lambda x: x.exit_time):
                    running_total += trade.pnl
                    cumulative_pnl.append(running_total)
                
                peak = cumulative_pnl[0] if cumulative_pnl else 0
                max_drawdown = 0
                
                for pnl in cumulative_pnl:
                    if pnl > peak:
                        peak = pnl
                    drawdown = (peak - pnl) / abs(peak) if peak != 0 else 0
                    max_drawdown = max(max_drawdown, drawdown)
                
                # Additional metrics
                avg_winning_trade = gross_profit / len(winning_trades) if winning_trades else 0
                avg_losing_trade = -gross_loss / len(losing_trades) if losing_trades else 0
                largest_winner = max(t.pnl for t in winning_trades) if winning_trades else 0
                largest_loser = min(t.pnl for t in losing_trades) if losing_trades else 0
                avg_hold_time = np.mean([t.hold_duration_hours for t in trades])
                
                # Expectancy
                win_rate_decimal = len(winning_trades) / len(trades)
                expectancy = (win_rate_decimal * avg_winning_trade) - ((1 - win_rate_decimal) * abs(avg_losing_trade))
                
                # Recovery factor
                recovery_factor = total_pnl / (max_drawdown * abs(peak)) if max_drawdown > 0 and peak != 0 else 0
                
                # Payoff ratio
                payoff_ratio = avg_winning_trade / abs(avg_losing_trade) if avg_losing_trade != 0 else 0
                
                # Calmar ratio
                calmar_ratio = (mean_return * 252 * 100) / (max_drawdown * 100) if max_drawdown > 0 else 0
                
                strategy_performance = StrategyPerformance(
                    strategy_id=strat_id,
                    total_trades=len(trades),
                    winning_trades=len(winning_trades),
                    losing_trades=len(losing_trades),
                    win_rate=win_rate,
                    total_pnl=total_pnl,
                    gross_profit=gross_profit,
                    gross_loss=gross_loss,
                    profit_factor=profit_factor,
                    max_drawdown=max_drawdown * 100,
                    sharpe_ratio=sharpe_ratio,
                    sortino_ratio=sortino_ratio,
                    calmar_ratio=calmar_ratio,
                    avg_winning_trade=avg_winning_trade,
                    avg_losing_trade=avg_losing_trade,
                    largest_winner=largest_winner,
                    largest_loser=largest_loser,
                    avg_hold_time=avg_hold_time,
                    expectancy=expectancy,
                    recovery_factor=recovery_factor,
                    payoff_ratio=payoff_ratio
                )
                
                strategy_performances[strat_id] = strategy_performance
            
            return strategy_performances
            
        except Exception as e:
            logger.error(f"Error getting strategy performance: {e}")
            return {}
    
    async def get_trade_analytics(self, 
                                symbol: str = None,
                                strategy_id: str = None,
                                limit: int = 100) -> Dict[str, Any]:
        """Get detailed trade analytics"""
        try:
            # Filter trades
            filtered_trades = self.trade_analyses.copy()
            
            if symbol:
                filtered_trades = [t for t in filtered_trades if t.symbol == symbol]
            
            if strategy_id:
                filtered_trades = [t for t in filtered_trades if t.strategy_id == strategy_id]
            
            # Sort by exit time (most recent first)
            filtered_trades = sorted(filtered_trades, key=lambda x: x.exit_time, reverse=True)
            
            # Limit results
            if limit:
                filtered_trades = filtered_trades[:limit]
            
            if not filtered_trades:
                return {'trades': [], 'analytics': {}}
            
            # Calculate analytics
            winning_trades = [t for t in filtered_trades if t.pnl > 0]
            losing_trades = [t for t in filtered_trades if t.pnl < 0]
            
            analytics = {
                'total_trades': len(filtered_trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': (len(winning_trades) / len(filtered_trades)) * 100,
                'total_pnl': sum(t.pnl for t in filtered_trades),
                'avg_pnl': np.mean([t.pnl for t in filtered_trades]),
                'median_pnl': np.median([t.pnl for t in filtered_trades]),
                'std_pnl': np.std([t.pnl for t in filtered_trades]),
                'avg_hold_time': np.mean([t.hold_duration_hours for t in filtered_trades]),
                'median_hold_time': np.median([t.hold_duration_hours for t in filtered_trades]),
                'avg_risk_reward': np.mean([t.risk_reward_ratio for t in filtered_trades if t.risk_reward_ratio > 0]),
                'best_trade': max(filtered_trades, key=lambda x: x.pnl).to_dict(),
                'worst_trade': min(filtered_trades, key=lambda x: x.pnl).to_dict(),
                'longest_hold': max(filtered_trades, key=lambda x: x.hold_duration_hours).to_dict(),
                'shortest_hold': min(filtered_trades, key=lambda x: x.hold_duration_hours).to_dict()
            }
            
            # Symbol breakdown
            symbol_stats = defaultdict(lambda: {'trades': 0, 'pnl': 0, 'win_rate': 0})
            for trade in filtered_trades:
                symbol_stats[trade.symbol]['trades'] += 1
                symbol_stats[trade.symbol]['pnl'] += trade.pnl
            
            for symbol_key, stats in symbol_stats.items():
                symbol_trades = [t for t in filtered_trades if t.symbol == symbol_key]
                winning_symbol_trades = [t for t in symbol_trades if t.pnl > 0]
                stats['win_rate'] = (len(winning_symbol_trades) / len(symbol_trades)) * 100
            
            analytics['symbol_breakdown'] = dict(symbol_stats)
            
            # Strategy breakdown
            strategy_stats = defaultdict(lambda: {'trades': 0, 'pnl': 0, 'win_rate': 0})
            for trade in filtered_trades:
                strategy_stats[trade.strategy_id]['trades'] += 1
                strategy_stats[trade.strategy_id]['pnl'] += trade.pnl
            
            for strat_key, stats in strategy_stats.items():
                strategy_trades = [t for t in filtered_trades if t.strategy_id == strat_key]
                winning_strategy_trades = [t for t in strategy_trades if t.pnl > 0]
                stats['win_rate'] = (len(winning_strategy_trades) / len(strategy_trades)) * 100
            
            analytics['strategy_breakdown'] = dict(strategy_stats)
            
            # Return data
            return {
                'trades': [t.to_dict() for t in filtered_trades],
                'analytics': analytics,
                'filters_applied': {
                    'symbol': symbol,
                    'strategy_id': strategy_id,
                    'limit': limit
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting trade analytics: {e}")
            return {'error': str(e)}
    
    async def get_performance_chart_data(self, 
                                       period: PerformancePeriod = PerformancePeriod.MONTHLY,
                                       chart_type: str = 'equity_curve') -> Dict[str, Any]:
        """Get data for performance charts"""
        try:
            period_data = self._get_period_data(period)
            
            if chart_type == 'equity_curve':
                # Portfolio value over time
                timestamps = [s.timestamp for s in self.portfolio_snapshots if s.timestamp >= self._get_period_cutoff(period)]
                values = period_data['values']
                
                return {
                    'chart_type': 'equity_curve',
                    'data': [
                        {'timestamp': ts.isoformat(), 'value': val}
                        for ts, val in zip(timestamps, values)
                    ],
                    'initial_value': values[0] if values else self.initial_capital,
                    'final_value': values[-1] if values else self.initial_capital
                }
            
            elif chart_type == 'drawdown':
                # Drawdown chart
                values = period_data['values']
                timestamps = [s.timestamp for s in self.portfolio_snapshots if s.timestamp >= self._get_period_cutoff(period)]
                
                peak = values[0] if values else self.initial_capital
                drawdowns = []
                
                for val in values:
                    if val > peak:
                        peak = val
                    drawdown = ((peak - val) / peak) * 100 if peak > 0 else 0
                    drawdowns.append(drawdown)
                
                return {
                    'chart_type': 'drawdown',
                    'data': [
                        {'timestamp': ts.isoformat(), 'drawdown': dd}
                        for ts, dd in zip(timestamps, drawdowns)
                    ],
                    'max_drawdown': max(drawdowns) if drawdowns else 0
                }
            
            elif chart_type == 'monthly_returns':
                # Monthly returns heatmap data
                monthly_returns = self._calculate_monthly_returns()
                
                return {
                    'chart_type': 'monthly_returns',
                    'data': monthly_returns
                }
            
            elif chart_type == 'trade_distribution':
                # P&L distribution
                trades = period_data['trades']
                pnl_buckets = self._create_pnl_distribution(trades)
                
                return {
                    'chart_type': 'trade_distribution',
                    'data': pnl_buckets,
                    'total_trades': len(trades)
                }
            
            else:
                return {'error': f'Unknown chart type: {chart_type}'}
                
        except Exception as e:
            logger.error(f"Error getting chart data: {e}")
            return {'error': str(e)}
    
    def _get_period_cutoff(self, period: PerformancePeriod) -> datetime:
        """Get cutoff date for period"""
        now = datetime.utcnow()
        
        if period == PerformancePeriod.DAILY:
            return now - timedelta(days=1)
        elif period == PerformancePeriod.WEEKLY:
            return now - timedelta(days=7)
        elif period == PerformancePeriod.MONTHLY:
            return now - timedelta(days=30)
        elif period == PerformancePeriod.QUARTERLY:
            return now - timedelta(days=90)
        elif period == PerformancePeriod.YEARLY:
            return now - timedelta(days=365)
        else:
            return datetime.min  # All time
    
    def _calculate_monthly_returns(self) -> List[Dict[str, Any]]:
        """Calculate monthly returns for heatmap"""
        try:
            monthly_data = defaultdict(lambda: {'start_value': None, 'end_value': None})
            
            for snapshot in self.portfolio_snapshots:
                month_key = snapshot.timestamp.strftime('%Y-%m')
                
                if monthly_data[month_key]['start_value'] is None:
                    monthly_data[month_key]['start_value'] = snapshot.total_value
                monthly_data[month_key]['end_value'] = snapshot.total_value
            
            monthly_returns = []
            for month, data in monthly_data.items():
                if data['start_value'] and data['end_value']:
                    monthly_return = ((data['end_value'] - data['start_value']) / data['start_value']) * 100
                    year, month_num = month.split('-')
                    
                    monthly_returns.append({
                        'year': int(year),
                        'month': int(month_num),
                        'return': monthly_return
                    })
            
            return sorted(monthly_returns, key=lambda x: (x['year'], x['month']))
            
        except Exception as e:
            logger.error(f"Error calculating monthly returns: {e}")
            return []
    
    def _create_pnl_distribution(self, trades: List[TradeAnalysis]) -> List[Dict[str, Any]]:
        """Create P&L distribution buckets"""
        try:
            if not trades:
                return []
            
            pnl_values = [t.pnl for t in trades]
            min_pnl = min(pnl_values)
            max_pnl = max(pnl_values)
            
            # Create 20 buckets
            num_buckets = 20
            bucket_size = (max_pnl - min_pnl) / num_buckets
            
            buckets = []
            for i in range(num_buckets):
                bucket_min = min_pnl + (i * bucket_size)
                bucket_max = bucket_min + bucket_size
                
                # Count trades in bucket
                trades_in_bucket = len([
                    t for t in trades 
                    if bucket_min <= t.pnl < bucket_max or (i == num_buckets - 1 and t.pnl <= bucket_max)
                ])
                
                buckets.append({
                    'range_min': bucket_min,
                    'range_max': bucket_max,
                    'count': trades_in_bucket,
                    'percentage': (trades_in_bucket / len(trades)) * 100
                })
            
            return buckets
            
        except Exception as e:
            logger.error(f"Error creating P&L distribution: {e}")
            return []
    
    async def generate_performance_report(self, 
                                        period: PerformancePeriod = PerformancePeriod.MONTHLY) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        try:
            # Get basic performance summary
            summary = await self.get_performance_summary(period)
            
            # Get strategy performance
            strategy_performance = await self.get_strategy_performance()
            
            # Get trade analytics
            trade_analytics = await self.get_trade_analytics(limit=50)
            
            # Generate report
            report = {
                'report_period': period.value,
                'generated_at': datetime.utcnow().isoformat(),
                'portfolio_summary': summary,
                'strategy_performance': {k: v.to_dict() for k, v in strategy_performance.items()},
                'trade_analytics': trade_analytics,
                'key_metrics': {
                    'total_return': summary.get('total_return', 0),
                    'sharpe_ratio': summary.get('sharpe_ratio', 0),
                    'max_drawdown': summary.get('max_drawdown', 0),
                    'win_rate': summary.get('win_rate', 0),
                    'profit_factor': summary.get('profit_factor', 0)
                },
                'risk_assessment': {
                    'volatility': summary.get('volatility', 0),
                    'var_95': summary.get('var_95', 0),
                    'current_drawdown': summary.get('current_drawdown', 0),
                    'risk_rating': self._calculate_risk_rating(summary)
                },
                'recommendations': self._generate_recommendations(summary, strategy_performance)
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {'error': str(e)}
    
    def _calculate_risk_rating(self, summary: Dict[str, Any]) -> str:
        """Calculate overall risk rating"""
        try:
            risk_score = 0
            
            # Volatility component
            volatility = summary.get('volatility', 0)
            if volatility > 50:
                risk_score += 3
            elif volatility > 30:
                risk_score += 2
            elif volatility > 20:
                risk_score += 1
            
            # Drawdown component
            max_drawdown = summary.get('max_drawdown', 0)
            if max_drawdown > 30:
                risk_score += 3
            elif max_drawdown > 20:
                risk_score += 2
            elif max_drawdown > 10:
                risk_score += 1
            
            # Sharpe ratio component (inverse)
            sharpe_ratio = summary.get('sharpe_ratio', 0)
            if sharpe_ratio < 0.5:
                risk_score += 2
            elif sharpe_ratio < 1.0:
                risk_score += 1
            
            # Determine rating
            if risk_score >= 6:
                return "HIGH"
            elif risk_score >= 4:
                return "MEDIUM"
            elif risk_score >= 2:
                return "LOW"
            else:
                return "VERY_LOW"
                
        except Exception as e:
            logger.error(f"Error calculating risk rating: {e}")
            return "UNKNOWN"
    
    def _generate_recommendations(self, 
                                summary: Dict[str, Any], 
                                strategy_performance: Dict[str, StrategyPerformance]) -> List[str]:
        """Generate performance improvement recommendations"""
        try:
            recommendations = []
            
            # Win rate recommendations
            win_rate = summary.get('win_rate', 0)
            if win_rate < 40:
                recommendations.append("Win rate is below 40% - consider reviewing entry criteria and signal quality")
            elif win_rate > 80:
                recommendations.append("Very high win rate - ensure adequate position sizing and risk management")
            
            # Profit factor recommendations
            profit_factor = summary.get('profit_factor', 0)
            if profit_factor < 1.2:
                recommendations.append("Profit factor below 1.2 - focus on improving risk/reward ratios")
            elif profit_factor > 3.0:
                recommendations.append("Excellent profit factor - consider scaling up proven strategies")
            
            # Sharpe ratio recommendations
            sharpe_ratio = summary.get('sharpe_ratio', 0)
            if sharpe_ratio < 1.0:
                recommendations.append("Sharpe ratio below 1.0 - work on reducing volatility while maintaining returns")
            elif sharpe_ratio > 2.0:
                recommendations.append("Excellent risk-adjusted returns - maintain current approach")
            
            # Drawdown recommendations
            max_drawdown = summary.get('max_drawdown', 0)
            if max_drawdown > 20:
                recommendations.append("High maximum drawdown - implement stricter position sizing and stop losses")
            
            # Strategy-specific recommendations
            if strategy_performance:
                best_strategy = max(strategy_performance.values(), key=lambda s: s.profit_factor)
                worst_strategy = min(strategy_performance.values(), key=lambda s: s.profit_factor)
                
                if best_strategy.profit_factor > 2.0:
                    recommendations.append(f"Strategy '{best_strategy.strategy_id}' performing exceptionally well - consider increased allocation")
                
                if worst_strategy.profit_factor < 1.0:
                    recommendations.append(f"Strategy '{worst_strategy.strategy_id}' underperforming - review or disable")
            
            # General recommendations
            total_trades = summary.get('total_trades', 0)
            if total_trades < 10:
                recommendations.append("Limited trade history - continue building track record before making major adjustments")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return ["Unable to generate recommendations due to error"]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get performance tracker statistics"""
        return {
            'total_snapshots': self.stats['total_snapshots'],
            'total_trades_analyzed': self.stats['total_trades_analyzed'],
            'calculation_count': self.stats['calculation_count'],
            'cache_hits': len(self._performance_cache),
            'strategies_tracked': len(self.strategy_trades),
            'data_points': len(self.portfolio_values),
            'last_update': self.stats['last_update'].isoformat() if self.stats['last_update'] else None,
            'memory_usage': {
                'snapshots': len(self.portfolio_snapshots),
                'trades': len(self.trade_analyses),
                'returns': len(self.daily_returns),
                'values': len(self.portfolio_values)
            }
        }
    
    async def cleanup(self):
        """Cleanup performance tracker resources"""
        try:
            # Clear caches
            self._performance_cache.clear()
            
            # Keep only essential data in memory
            max_snapshots = 1000
            if len(self.portfolio_snapshots) > max_snapshots:
                self.portfolio_snapshots = self.portfolio_snapshots[-max_snapshots:]
                self.portfolio_values = self.portfolio_values[-max_snapshots:]
                self.daily_returns = self.daily_returns[-max_snapshots:]
            
            logger.info("Performance Tracker cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during performance tracker cleanup: {e}")

# Factory function
def create_performance_tracker(initial_capital: float = 10000) -> PerformanceTracker:
    """Factory function to create Performance Tracker instance"""
    return PerformanceTracker(initial_capital)

# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_performance_tracker():
        print("📊 Performance Tracker Test")
        print("=" * 50)
        
        try:
            # Create performance tracker
            performance_tracker = create_performance_tracker(initial_capital=10000)
            
            print(f"✅ Performance Tracker created")
            print(f"💰 Initial capital: ${performance_tracker.initial_capital:,.2f}")
            
            # Test 1: Add portfolio snapshots
            print(f"\n🧪 Test 1: Adding Portfolio Snapshots")
            
            # Simulate portfolio performance over 30 days
            base_value = 10000
            dates = [datetime.utcnow() - timedelta(days=30-i) for i in range(30)]
            
            np.random.seed(42)  # For reproducible results
            
            for i, date in enumerate(dates):
                # Simulate portfolio growth with volatility
                if i == 0:
                    value = base_value
                else:
                    # Random daily return between -3% and +3%
                    daily_return = np.random.normal(0.001, 0.02)  # Slight positive bias
                    value = performance_tracker.portfolio_values[-1] * (1 + daily_return)
                    value = max(value, base_value * 0.8)  # Prevent extreme losses
                
                cash_balance = max(1000, value * 0.1)  # 10% cash reserve
                invested_amount = value - cash_balance
                active_positions = max(1, int(invested_amount / 2000))  # Rough position count
                
                # Manually set timestamp for testing
                performance_tracker.portfolio_snapshots.clear()
                performance_tracker.portfolio_values.clear()
                performance_tracker.daily_returns.clear()
                
                # Add snapshots sequentially
                for j in range(i + 1):
                    test_value = base_value * (1 + (j * 0.002) + np.random.normal(0, 0.01))
                    test_value = max(test_value, base_value * 0.9)
                    
                    await performance_tracker.update_portfolio_snapshot(
                        portfolio_value=test_value,
                        cash_balance=max(1000, test_value * 0.1),
                        invested_amount=test_value * 0.9,
                        active_positions=3
                    )
                
                if i % 10 == 0:
                    print(f"  Day {i+1}: ${value:,.2f}")
            
            final_value = performance_tracker.current_portfolio_value
            total_return = ((final_value - base_value) / base_value) * 100
            
            print(f"  Final portfolio value: ${final_value:,.2f}")
            print(f"  Total return: {total_return:+.2f}%")
            print(f"  Total snapshots: {len(performance_tracker.portfolio_snapshots)}")
            
            # Test 2: Add completed trades
            print(f"\n🧪 Test 2: Adding Completed Trades")
            
            # Create sample trades
            sample_trades = [
                {
                    'symbol': 'BTC/USDT',
                    'strategy_id': 'momentum_strategy',
                    'direction': 'LONG',
                    'entry_price': 45000,
                    'exit_price': 47000,
                    'quantity': 0.1,
                    'pnl': 200,
                    'entry_time': datetime.utcnow() - timedelta(days=5),
                    'exit_time': datetime.utcnow() - timedelta(days=4),
                    'risk_amount': 100,
                    'reward_amount': 200
                },
                {
                    'symbol': 'ETH/USDT',
                    'strategy_id': 'rsi_strategy',
                    'direction': 'LONG',
                    'entry_price': 3000,
                    'exit_price': 2950,
                    'quantity': 1.0,
                    'pnl': -50,
                    'entry_time': datetime.utcnow() - timedelta(days=3),
                    'exit_time': datetime.utcnow() - timedelta(days=2),
                    'risk_amount': 150,
                    'reward_amount': 150
                },
                {
                    'symbol': 'ADA/USDT',
                    'strategy_id': 'momentum_strategy',
                    'direction': 'LONG',
                    'entry_price': 0.45,
                    'exit_price': 0.48,
                    'quantity': 2000,
                    'pnl': 60,
                    'entry_time': datetime.utcnow() - timedelta(days=2),
                    'exit_time': datetime.utcnow() - timedelta(days=1),
                    'risk_amount': 40,
                    'reward_amount': 60
                },
                {
                    'symbol': 'DOT/USDT',
                    'strategy_id': 'rsi_strategy',
                    'direction': 'SHORT',
                    'entry_price': 25.0,
                    'exit_price': 24.0,
                    'quantity': 10,
                    'pnl': 100,
                    'entry_time': datetime.utcnow() - timedelta(days=1),
                    'exit_time': datetime.utcnow() - timedelta(hours=6),
                    'risk_amount': 50,
                    'reward_amount': 100
                }
            ]
            
            for trade in sample_trades:
                await performance_tracker.add_completed_trade(trade)
            
            print(f"  Added {len(sample_trades)} completed trades")
            print(f"  Total trades analyzed: {len(performance_tracker.trade_analyses)}")
            
            # Show trade breakdown
            winning_trades = [t for t in performance_tracker.trade_analyses if t.pnl > 0]
            losing_trades = [t for t in performance_tracker.trade_analyses if t.pnl < 0]
            
            print(f"  Winning trades: {len(winning_trades)}")
            print(f"  Losing trades: {len(losing_trades)}")
            print(f"  Win rate: {(len(winning_trades) / len(sample_trades)) * 100:.1f}%")
            
            # Test 3: Performance summary
            print(f"\n🧪 Test 3: Performance Summary")
            
            summary = await performance_tracker.get_performance_summary(PerformancePeriod.ALL_TIME)
            
            print(f"  Period: {summary['period']}")
            print(f"  Total Return: {summary['total_return']:+.2f}%")
            print(f"  Sharpe Ratio: {summary['sharpe_ratio']:.2f}")
            print(f"  Max Drawdown: {summary['max_drawdown']:.2f}%")
            print(f"  Volatility: {summary['volatility']:.2f}%")
            print(f"  Win Rate: {summary['win_rate']:.1f}%")
            print(f"  Profit Factor: {summary['profit_factor']:.2f}")
            print(f"  Total Trades: {summary['total_trades']}")
            
            # Test 4: Strategy performance
            print(f"\n🧪 Test 4: Strategy Performance Analysis")
            
            strategy_performance = await performance_tracker.get_strategy_performance()
            
            for strategy_id, performance in strategy_performance.items():
                print(f"  Strategy: {strategy_id}")
                print(f"    Total Trades: {performance.total_trades}")
                print(f"    Win Rate: {performance.win_rate:.1f}%")
                print(f"    Total P&L: ${performance.total_pnl:+.2f}")
                print(f"    Profit Factor: {performance.profit_factor:.2f}")
                print(f"    Sharpe Ratio: {performance.sharpe_ratio:.2f}")
                print(f"    Max Drawdown: {performance.max_drawdown:.2f}%")
                print(f"    Avg Hold Time: {performance.avg_hold_time:.1f} hours")
            
            # Test 5: Trade analytics
            print(f"\n🧪 Test 5: Trade Analytics")
            
            trade_analytics = await performance_tracker.get_trade_analytics(limit=10)
            
            analytics = trade_analytics['analytics']
            print(f"  Total Trades: {analytics['total_trades']}")
            print(f"  Win Rate: {analytics['win_rate']:.1f}%")
            print(f"  Average P&L: ${analytics['avg_pnl']:+.2f}")
            print(f"  Best Trade: ${analytics['best_trade']['pnl']:+.2f} ({analytics['best_trade']['symbol']})")
            print(f"  Worst Trade: ${analytics['worst_trade']['pnl']:+.2f} ({analytics['worst_trade']['symbol']})")
            print(f"  Avg Hold Time: {analytics['avg_hold_time']:.1f} hours")
            
            # Symbol breakdown
            print(f"  Symbol Breakdown:")
            for symbol, stats in analytics['symbol_breakdown'].items():
                print(f"    {symbol}: {stats['trades']} trades, ${stats['pnl']:+.2f} P&L, {stats['win_rate']:.1f}% win rate")
            
            # Test 6: Chart data
            print(f"\n🧪 Test 6: Chart Data Generation")
            
            # Equity curve data
            equity_data = await performance_tracker.get_performance_chart_data(
                PerformancePeriod.ALL_TIME, 'equity_curve'
            )
            
            print(f"  Equity Curve:")
            print(f"    Data points: {len(equity_data['data'])}")
            print(f"    Initial value: ${equity_data['initial_value']:,.2f}")
            print(f"    Final value: ${equity_data['final_value']:,.2f}")
            
            # Drawdown data
            drawdown_data = await performance_tracker.get_performance_chart_data(
                PerformancePeriod.ALL_TIME, 'drawdown'
            )
            
            print(f"  Drawdown Chart:")
            print(f"    Data points: {len(drawdown_data['data'])}")
            print(f"    Max drawdown: {drawdown_data['max_drawdown']:.2f}%")
            
            # Trade distribution
            distribution_data = await performance_tracker.get_performance_chart_data(
                PerformancePeriod.ALL_TIME, 'trade_distribution'
            )
            
            print(f"  Trade Distribution:")
            print(f"    Total trades: {distribution_data['total_trades']}")
            print(f"    Distribution buckets: {len(distribution_data['data'])}")
            
            # Test 7: Performance report
            print(f"\n🧪 Test 7: Comprehensive Performance Report")
            
            report = await performance_tracker.generate_performance_report(PerformancePeriod.ALL_TIME)
            
            print(f"  Report Period: {report['report_period']}")
            print(f"  Generated At: {report['generated_at']}")
            
            key_metrics = report['key_metrics']
            print(f"  Key Metrics:")
            print(f"    Total Return: {key_metrics['total_return']:+.2f}%")
            print(f"    Sharpe Ratio: {key_metrics['sharpe_ratio']:.2f}")
            print(f"    Max Drawdown: {key_metrics['max_drawdown']:.2f}%")
            print(f"    Win Rate: {key_metrics['win_rate']:.1f}%")
            print(f"    Profit Factor: {key_metrics['profit_factor']:.2f}")
            
            risk_assessment = report['risk_assessment']
            print(f"  Risk Assessment:")
            print(f"    Volatility: {risk_assessment['volatility']:.2f}%")
            print(f"    VaR 95%: {risk_assessment['var_95']:.2f}%")
            print(f"    Risk Rating: {risk_assessment['risk_rating']}")
            
            recommendations = report['recommendations']
            print(f"  Recommendations ({len(recommendations)}):")
            for i, rec in enumerate(recommendations[:3], 1):  # Show first 3
                print(f"    {i}. {rec}")
            
            # Test 8: Different time periods
            print(f"\n🧪 Test 8: Different Time Periods")
            
            periods = [PerformancePeriod.DAILY, PerformancePeriod.WEEKLY, PerformancePeriod.MONTHLY]
            
            for period in periods:
                period_summary = await performance_tracker.get_performance_summary(period)
                print(f"  {period.value}:")
                print(f"    Total Return: {period_summary['total_return']:+.2f}%")
                print(f"    Volatility: {period_summary['volatility']:.2f}%")
                print(f"    Trades: {period_summary['total_trades']}")
            
            # Test 9: Statistics
            print(f"\n🧪 Test 9: Performance Tracker Statistics")
            
            stats = performance_tracker.get_statistics()
            
            print(f"  Total snapshots: {stats['total_snapshots']}")
            print(f"  Total trades analyzed: {stats['total_trades_analyzed']}")
            print(f"  Calculation count: {stats['calculation_count']}")
            print(f"  Strategies tracked: {stats['strategies_tracked']}")
            print(f"  Cache hits: {stats['cache_hits']}")
            print(f"  Last update: {stats['last_update']}")
            
            memory_usage = stats['memory_usage']
            print(f"  Memory usage:")
            print(f"    Snapshots: {memory_usage['snapshots']}")
            print(f"    Trades: {memory_usage['trades']}")
            print(f"    Returns: {memory_usage['returns']}")
            print(f"    Values: {memory_usage['values']}")
            
            # Test 10: Edge cases and validation
            print(f"\n🧪 Test 10: Edge Cases and Validation")
            
            # Test with no data
            empty_tracker = create_performance_tracker(5000)
            empty_summary = await empty_tracker.get_performance_summary()
            print(f"  Empty tracker summary: {len(empty_summary)} fields")
            
            # Test invalid period
            try:
                invalid_chart = await performance_tracker.get_performance_chart_data(
                    PerformancePeriod.MONTHLY, 'invalid_chart_type'
                )
                print(f"  Invalid chart type handled: {'error' in invalid_chart}")
            except Exception as e:
                print(f"  Invalid chart type error: {str(e)[:50]}...")
            
            # Test strategy filtering
            momentum_analytics = await performance_tracker.get_trade_analytics(
                strategy_id='momentum_strategy'
            )
            momentum_trades = len(momentum_analytics['trades'])
            print(f"  Momentum strategy trades: {momentum_trades}")
            
            # Test symbol filtering
            btc_analytics = await performance_tracker.get_trade_analytics(symbol='BTC/USDT')
            btc_trades = len(btc_analytics['trades'])
            print(f"  BTC trades: {btc_trades}")
            
            print(f"\n🎉 Performance Tracker test completed successfully!")
            
            # Final summary
            print(f"\n📈 Final Performance Summary:")
            final_summary = await performance_tracker.get_performance_summary()
            print(f"  Portfolio Value: ${final_summary['current_portfolio_value']:,.2f}")
            print(f"  Total Return: {final_summary['total_return']:+.2f}%")
            print(f"  Win Rate: {final_summary['win_rate']:.1f}%")
            print(f"  Sharpe Ratio: {final_summary['sharpe_ratio']:.2f}")
            print(f"  Max Drawdown: {final_summary['max_drawdown']:.2f}%")
            print(f"  Total Trades: {final_summary['total_trades']}")
            
        except Exception as e:
            print(f"❌ Error in Performance Tracker test: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Cleanup
            try:
                await performance_tracker.cleanup()
                print(f"🧹 Performance Tracker cleanup completed")
            except:
                pass
    
    # Run the test
    asyncio.run(test_performance_tracker())
