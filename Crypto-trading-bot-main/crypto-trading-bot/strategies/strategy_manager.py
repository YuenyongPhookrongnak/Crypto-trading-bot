"""
Strategy Manager - Centralized Management of All Trading Strategies

This module manages multiple trading strategies, handles signal generation,
portfolio allocation, and strategy performance monitoring.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import json

import pandas as pd
import numpy as np

from strategies.base_strategy import BaseStrategy, TradingSignal, SignalType, MarketCondition
from strategies.rsi_strategy import RSIStrategy, create_rsi_strategy
from strategies.volume_profile_strategy import VolumeProfileStrategy, create_volume_profile_strategy
# from strategies.multi_timeframe_strategy import MultiTimeframeStrategy, create_multi_timeframe_strategy

from database.connection import get_db_session
from database.models import Strategy as StrategyModel, Trade, SystemLog, LogLevel

logger = logging.getLogger(__name__)

@dataclass
class StrategyAllocation:
    """Strategy allocation configuration"""
    strategy_id: str
    allocation_percentage: float
    max_concurrent_positions: int
    risk_multiplier: float = 1.0
    enabled: bool = True
    
    def __post_init__(self):
        if not 0 <= self.allocation_percentage <= 1:
            raise ValueError("Allocation percentage must be between 0 and 1")
        if self.max_concurrent_positions < 0:
            raise ValueError("Max concurrent positions cannot be negative")

@dataclass
class SignalEvaluation:
    """Signal evaluation result"""
    signal: TradingSignal
    strategy_id: str
    raw_confidence: float
    adjusted_confidence: float
    portfolio_allocation: float
    risk_score: float
    recommendation: str  # EXECUTE, REDUCE_SIZE, REJECT
    reasoning: List[str] = field(default_factory=list)

@dataclass
class StrategyPerformance:
    """Strategy performance metrics"""
    strategy_id: str
    total_signals: int = 0
    executed_signals: int = 0
    successful_trades: int = 0
    failed_trades: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    avg_confidence: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    last_signal_time: Optional[datetime] = None
    performance_score: float = 0.0
    
    @property
    def execution_rate(self) -> float:
        """Calculate signal execution rate"""
        return (self.executed_signals / self.total_signals * 100) if self.total_signals > 0 else 0.0

class StrategyManager:
    """Centralized Strategy Management System"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Strategy registry
        self.strategies: Dict[str, BaseStrategy] = {}
        self.strategy_allocations: Dict[str, StrategyAllocation] = {}
        self.strategy_performance: Dict[str, StrategyPerformance] = {}
        
        # Signal management
        self.active_signals: Dict[str, TradingSignal] = {}  # signal_id -> signal
        self.signal_history: List[TradingSignal] = []
        self.signal_queue: asyncio.Queue = asyncio.Queue()
        
        # Performance tracking
        self.portfolio_allocation_used: float = 0.0
        self.total_active_positions: int = 0
        self.last_rebalance_time: Optional[datetime] = None
        
        # Threading for concurrent strategy execution
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # State management
        self.is_running = False
        self.last_market_data: Dict[str, Any] = {}
        
        logger.info("Strategy Manager initialized")
    
    async def initialize(self):
        """Initialize strategy manager and all strategies"""
        try:
            # Load strategy configurations from database/config
            await self._load_strategy_configurations()
            
            # Register and initialize default strategies
            await self._register_default_strategies()
            
            # Load strategy performance history
            await self._load_strategy_performance()
            
            # Initialize all registered strategies
            for strategy in self.strategies.values():
                await strategy.initialize()
            
            self.is_running = True
            logger.info(f"Strategy Manager initialized with {len(self.strategies)} strategies")
            
        except Exception as e:
            logger.error(f"Failed to initialize Strategy Manager: {e}")
            raise
    
    async def _load_strategy_configurations(self):
        """Load strategy configurations from database"""
        try:
            with get_db_session() as session:
                db_strategies = session.query(StrategyModel).filter(
                    StrategyModel.enabled == True
                ).all()
                
                for db_strategy in db_strategies:
                    allocation = StrategyAllocation(
                        strategy_id=db_strategy.id,
                        allocation_percentage=db_strategy.allocation_percentage or 0.33,
                        max_concurrent_positions=int(db_strategy.max_position_size * 10) or 3,
                        risk_multiplier=1.0,
                        enabled=db_strategy.enabled
                    )
                    self.strategy_allocations[db_strategy.id] = allocation
                    
                    # Initialize performance tracking
                    perf = StrategyPerformance(
                        strategy_id=db_strategy.id,
                        total_signals=db_strategy.signals_generated or 0,
                        successful_trades=db_strategy.winning_trades or 0,
                        failed_trades=db_strategy.losing_trades or 0,
                        total_pnl=float(db_strategy.total_pnl or 0),
                        win_rate=db_strategy.win_rate or 0,
                        sharpe_ratio=float(db_strategy.sharpe_ratio or 0),
                        max_drawdown=float(db_strategy.max_drawdown or 0),
                        last_signal_time=db_strategy.last_signal_time
                    )
                    self.strategy_performance[db_strategy.id] = perf
                
                logger.info(f"Loaded {len(self.strategy_allocations)} strategy configurations from database")
                
        except Exception as e:
            logger.warning(f"Could not load strategy configurations from database: {e}")
            # Use default configurations
            await self._setup_default_allocations()
    
    async def _setup_default_allocations(self):
        """Setup default strategy allocations"""
        default_allocations = [
            StrategyAllocation("rsi_strategy", 0.35, 3, 1.0, True),
            StrategyAllocation("volume_profile_strategy", 0.35, 3, 1.0, True),
            # StrategyAllocation("multi_timeframe_strategy", 0.30, 2, 1.0, True),
        ]
        
        for allocation in default_allocations:
            self.strategy_allocations[allocation.strategy_id] = allocation
            self.strategy_performance[allocation.strategy_id] = StrategyPerformance(allocation.strategy_id)
        
        logger.info("Setup default strategy allocations")
    
    async def _register_default_strategies(self):
        """Register default trading strategies"""
        try:
            # Get strategy configurations
            strategy_configs = {
                'rsi_strategy': self.config.get('rsi_strategy', {}),
                'volume_profile_strategy': self.config.get('volume_profile_strategy', {}),
                # 'multi_timeframe_strategy': self.config.get('multi_timeframe_strategy', {}),
            }
            
            # Register RSI Strategy
            if 'rsi_strategy' in self.strategy_allocations:
                rsi_strategy = create_rsi_strategy(strategy_configs['rsi_strategy'])
                await self.register_strategy(rsi_strategy)
            
            # Register Volume Profile Strategy
            if 'volume_profile_strategy' in self.strategy_allocations:
                vp_strategy = create_volume_profile_strategy(strategy_configs['volume_profile_strategy'])
                await self.register_strategy(vp_strategy)
            
            # Register Multi-Timeframe Strategy
            # if 'multi_timeframe_strategy' in self.strategy_allocations:
            #     mtf_strategy = create_multi_timeframe_strategy(strategy_configs['multi_timeframe_strategy'])
            #     await self.register_strategy(mtf_strategy)
            
            logger.info(f"Registered {len(self.strategies)} default strategies")
            
        except Exception as e:
            logger.error(f"Failed to register default strategies: {e}")
            raise
    
    async def _load_strategy_performance(self):
        """Load historical strategy performance data"""
        try:
            # Load recent trade data for each strategy
            with get_db_session() as session:
                for strategy_id in self.strategy_allocations.keys():
                    recent_trades = session.query(Trade).filter(
                        Trade.strategy_id == strategy_id,
                        Trade.status == 'CLOSED',
                        Trade.exit_time >= datetime.utcnow() - timedelta(days=30)
                    ).all()
                    
                    if recent_trades:
                        perf = self.strategy_performance[strategy_id]
                        perf.total_pnl = sum(float(t.pnl or 0) for t in recent_trades)
                        perf.successful_trades = len([t for t in recent_trades if (t.pnl or 0) > 0])
                        perf.failed_trades = len([t for t in recent_trades if (t.pnl or 0) <= 0])
                        perf.win_rate = (perf.successful_trades / len(recent_trades) * 100) if recent_trades else 0
                        
                        # Calculate Sharpe ratio (simplified)
                        returns = [float(t.pnl_percentage or 0) for t in recent_trades]
                        if len(returns) > 1:
                            perf.sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
                        
                        # Performance score (weighted metric)
                        perf.performance_score = self._calculate_performance_score(perf)
            
            logger.info("Loaded strategy performance history")
            
        except Exception as e:
            logger.warning(f"Could not load strategy performance history: {e}")
    
    def _calculate_performance_score(self, performance: StrategyPerformance) -> float:
        """Calculate weighted performance score for strategy ranking"""
        try:
            # Weighted scoring system
            weights = {
                'win_rate': 0.3,
                'total_pnl': 0.25,
                'sharpe_ratio': 0.25,
                'execution_rate': 0.1,
                'consistency': 0.1
            }
            
            score = 0.0
            
            # Win rate component (0-100)
            score += (performance.win_rate / 100) * weights['win_rate'] * 100
            
            # PnL component (normalized)
            pnl_score = min(max(performance.total_pnl * 10, -100), 100)  # Cap between -100 and 100
            score += pnl_score * weights['total_pnl']
            
            # Sharpe ratio component
            sharpe_score = min(max(performance.sharpe_ratio * 25, -100), 100)
            score += sharpe_score * weights['sharpe_ratio']
            
            # Execution rate component
            score += performance.execution_rate * weights['execution_rate']
            
            # Consistency component (inverse of max drawdown)
            consistency_score = max(0, 100 - abs(performance.max_drawdown * 100))
            score += consistency_score * weights['consistency']
            
            return max(0, score)  # Ensure non-negative
            
        except Exception as e:
            logger.error(f"Error calculating performance score: {e}")
            return 50.0  # Default neutral score
    
    async def register_strategy(self, strategy: BaseStrategy):
        """Register a trading strategy"""
        try:
            if strategy.strategy_id in self.strategies:
                logger.warning(f"Strategy {strategy.strategy_id} already registered, updating...")
            
            self.strategies[strategy.strategy_id] = strategy
            
            # Ensure allocation exists
            if strategy.strategy_id not in self.strategy_allocations:
                self.strategy_allocations[strategy.strategy_id] = StrategyAllocation(
                    strategy_id=strategy.strategy_id,
                    allocation_percentage=0.33,
                    max_concurrent_positions=3,
                    enabled=True
                )
            
            # Ensure performance tracking exists
            if strategy.strategy_id not in self.strategy_performance:
                self.strategy_performance[strategy.strategy_id] = StrategyPerformance(strategy.strategy_id)
            
            logger.info(f"Registered strategy: {strategy.strategy_id}")
            
        except Exception as e:
            logger.error(f"Failed to register strategy {strategy.strategy_id}: {e}")
            raise
    
    async def unregister_strategy(self, strategy_id: str):
        """Unregister a trading strategy"""
        try:
            if strategy_id in self.strategies:
                # Cleanup strategy
                await self.strategies[strategy_id].cleanup()
                del self.strategies[strategy_id]
                
                # Update allocation to disabled
                if strategy_id in self.strategy_allocations:
                    self.strategy_allocations[strategy_id].enabled = False
                
                logger.info(f"Unregistered strategy: {strategy_id}")
            
        except Exception as e:
            logger.error(f"Failed to unregister strategy {strategy_id}: {e}")
    
    async def generate_signals(self, 
                             symbols: List[str], 
                             market_data: Dict[str, pd.DataFrame],
                             current_prices: Dict[str, float],
                             portfolio_data: Dict[str, Any]) -> List[SignalEvaluation]:
        """Generate signals from all enabled strategies"""
        
        if not self.is_running:
            logger.warning("Strategy Manager not running, skipping signal generation")
            return []
        
        try:
            start_time = datetime.utcnow()
            all_signals = []
            
            # Get enabled strategies
            enabled_strategies = {
                strategy_id: strategy for strategy_id, strategy in self.strategies.items()
                if self.strategy_allocations.get(strategy_id, StrategyAllocation('', 0, 0)).enabled
            }
            
            if not enabled_strategies:
                logger.warning("No enabled strategies found")
                return []
            
            # Generate signals concurrently
            tasks = []
            for symbol in symbols:
                if symbol not in market_data or symbol not in current_prices:
                    continue
                
                symbol_data = market_data[symbol]
                current_price = current_prices[symbol]
                
                # Create tasks for each strategy
                for strategy_id, strategy in enabled_strategies.items():
                    task = self._generate_strategy_signal(
                        strategy, symbol, symbol_data, current_price, portfolio_data
                    )
                    tasks.append((strategy_id, symbol, task))
            
            # Execute tasks concurrently
            if tasks:
                results = await asyncio.gather(
                    *[task for _, _, task in tasks], 
                    return_exceptions=True
                )
                
                # Process results
                for i, (strategy_id, symbol, _) in enumerate(tasks):
                    result = results[i]
                    
                    if isinstance(result, Exception):
                        logger.error(f"Error generating signal for {strategy_id} on {symbol}: {result}")
                        continue
                    
                    if result:  # Signal generated
                        # Evaluate signal quality and portfolio fit
                        evaluation = await self._evaluate_signal(
                            result, strategy_id, portfolio_data
                        )
                        
                        if evaluation:
                            all_signals.append(evaluation)
                            
                            # Update strategy performance tracking
                            self.strategy_performance[strategy_id].total_signals += 1
                            self.strategy_performance[strategy_id].last_signal_time = datetime.utcnow()
            
            # Rank and filter signals
            ranked_signals = await self._rank_and_filter_signals(all_signals, portfolio_data)
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            logger.info(f"Generated {len(all_signals)} signals ({len(ranked_signals)} recommended) "
                       f"from {len(enabled_strategies)} strategies in {execution_time:.2f}s")
            
            # Store signals in history
            for evaluation in ranked_signals:
                self.signal_history.append(evaluation.signal)
                if len(self.signal_history) > 1000:  # Keep last 1000 signals
                    self.signal_history = self.signal_history[-1000:]
            
            return ranked_signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return []
    
    async def _generate_strategy_signal(self,
                                      strategy: BaseStrategy,
                                      symbol: str,
                                      market_data: pd.DataFrame,
                                      current_price: float,
                                      portfolio_data: Dict[str, Any]) -> Optional[TradingSignal]:
        """Generate signal from a specific strategy"""
        try:
            # Check if strategy has sufficient data
            if len(market_data) < strategy.get_lookback_period():
                return None
            
            # Generate signal
            signal = await strategy.analyze_symbol(
                symbol=symbol,
                ohlcv_data=market_data,
                current_price=current_price,
                additional_data={
                    'portfolio_value': portfolio_data.get('total_value', 10000),
                    'available_balance': portfolio_data.get('available_balance', 5000),
                    'risk_level': portfolio_data.get('risk_level', 'MEDIUM')
                }
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal from {strategy.strategy_id} for {symbol}: {e}")
            return None
    
    async def _evaluate_signal(self,
                             signal: TradingSignal,
                             strategy_id: str,
                             portfolio_data: Dict[str, Any]) -> Optional[SignalEvaluation]:
        """Evaluate signal quality and portfolio alignment"""
        
        try:
            allocation = self.strategy_allocations.get(strategy_id)
            performance = self.strategy_performance.get(strategy_id)
            
            if not allocation or not allocation.enabled:
                return None
            
            # Basic signal validation
            if signal.confidence < 50.0:  # Minimum confidence threshold
                return None
            
            # Calculate risk score
            risk_score = await self._calculate_signal_risk_score(signal, strategy_id, portfolio_data)
            
            # Adjust confidence based on strategy performance
            performance_multiplier = self._get_performance_multiplier(performance)
            adjusted_confidence = signal.confidence * performance_multiplier
            
            # Calculate portfolio allocation impact
            position_value = signal.position_size_recommendation * signal.entry_price if signal.position_size_recommendation else 0
            portfolio_allocation = position_value / portfolio_data.get('total_value', 10000)
            
            # Determine recommendation
            recommendation, reasoning = self._determine_recommendation(
                signal, adjusted_confidence, risk_score, portfolio_allocation, allocation
            )
            
            evaluation = SignalEvaluation(
                signal=signal,
                strategy_id=strategy_id,
                raw_confidence=signal.confidence,
                adjusted_confidence=adjusted_confidence,
                portfolio_allocation=portfolio_allocation,
                risk_score=risk_score,
                recommendation=recommendation,
                reasoning=reasoning
            )
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Error evaluating signal: {e}")
            return None
    
    async def _calculate_signal_risk_score(self,
                                         signal: TradingSignal,
                                         strategy_id: str,
                                         portfolio_data: Dict[str, Any]) -> float:
        """Calculate risk score for a signal (0-100, lower is better)"""
        
        try:
            risk_factors = []
            
            # Position size risk
            if signal.position_size_recommendation:
                position_value = signal.position_size_recommendation * signal.entry_price
                portfolio_percentage = position_value / portfolio_data.get('total_value', 10000)
                if portfolio_percentage > 0.1:  # More than 10%
                    risk_factors.append(30)
                elif portfolio_percentage > 0.05:  # More than 5%
                    risk_factors.append(15)
                else:
                    risk_factors.append(5)
            
            # Market condition risk
            if signal.market_condition == MarketCondition.VOLATILE:
                risk_factors.append(20)
            elif signal.market_condition == MarketCondition.LOW_VOLUME:
                risk_factors.append(25)
            else:
                risk_factors.append(5)
            
            # Confidence risk (inverse relationship)
            confidence_risk = max(0, 100 - signal.confidence) * 0.3
            risk_factors.append(confidence_risk)
            
            # Risk/Reward ratio risk
            if signal.risk_reward_ratio:
                if signal.risk_reward_ratio < 1.5:
                    risk_factors.append(25)
                elif signal.risk_reward_ratio < 2.0:
                    risk_factors.append(10)
                else:
                    risk_factors.append(0)
            else:
                risk_factors.append(15)  # No R/R defined
            
            # Strategy performance risk
            performance = self.strategy_performance.get(strategy_id)
            if performance:
                if performance.win_rate < 40:
                    risk_factors.append(20)
                elif performance.win_rate < 50:
                    risk_factors.append(10)
                else:
                    risk_factors.append(0)
            
            # Calculate weighted average
            total_risk = sum(risk_factors) / len(risk_factors) if risk_factors else 50
            return min(100, max(0, total_risk))
            
        except Exception as e:
            logger.error(f"Error calculating signal risk score: {e}")
            return 50.0  # Default medium risk
    
    def _get_performance_multiplier(self, performance: StrategyPerformance) -> float:
        """Get performance-based confidence multiplier"""
        if not performance:
            return 1.0
        
        # Base multiplier on performance score
        if performance.performance_score > 80:
            return 1.2
        elif performance.performance_score > 60:
            return 1.1
        elif performance.performance_score > 40:
            return 1.0
        elif performance.performance_score > 20:
            return 0.9
        else:
            return 0.8
    
    def _determine_recommendation(self,
                                signal: TradingSignal,
                                adjusted_confidence: float,
                                risk_score: float,
                                portfolio_allocation: float,
                                allocation: StrategyAllocation) -> Tuple[str, List[str]]:
        """Determine trading recommendation"""
        
        reasoning = []
        
        # High confidence, low risk
        if adjusted_confidence >= 80 and risk_score <= 20:
            return "EXECUTE", [f"High confidence ({adjusted_confidence:.1f}%), low risk ({risk_score:.1f})"]
        
        # Good confidence, acceptable risk
        elif adjusted_confidence >= 65 and risk_score <= 40:
            if portfolio_allocation <= allocation.allocation_percentage:
                return "EXECUTE", [f"Good confidence ({adjusted_confidence:.1f}%), acceptable risk"]
            else:
                return "REDUCE_SIZE", [f"Good signal but exceeds allocation ({portfolio_allocation:.1%} > {allocation.allocation_percentage:.1%})"]
        
        # Medium confidence
        elif adjusted_confidence >= 50 and risk_score <= 60:
            if portfolio_allocation <= allocation.allocation_percentage * 0.5:
                return "REDUCE_SIZE", [f"Medium confidence, reduce position size"]
            else:
                return "REJECT", [f"Medium confidence with high allocation risk"]
        
        # Low confidence or high risk
        else:
            return "REJECT", [f"Low confidence ({adjusted_confidence:.1f}%) or high risk ({risk_score:.1f})"]
    
    async def _rank_and_filter_signals(self,
                                     signals: List[SignalEvaluation],
                                     portfolio_data: Dict[str, Any]) -> List[SignalEvaluation]:
        """Rank signals by quality and filter based on portfolio constraints"""
        
        try:
            if not signals:
                return []
            
            # Filter out rejected signals
            viable_signals = [s for s in signals if s.recommendation in ['EXECUTE', 'REDUCE_SIZE']]
            
            if not viable_signals:
                return []
            
            # Calculate composite score for ranking
            for signal_eval in viable_signals:
                score = self._calculate_composite_score(signal_eval)
                signal_eval.composite_score = score
            
            # Sort by composite score (descending)
            viable_signals.sort(key=lambda x: getattr(x, 'composite_score', 0), reverse=True)
            
            # Apply portfolio constraints
            filtered_signals = await self._apply_portfolio_constraints(viable_signals, portfolio_data)
            
            return filtered_signals
            
        except Exception as e:
            logger.error(f"Error ranking and filtering signals: {e}")
            return signals
    
    def _calculate_composite_score(self, signal_eval: SignalEvaluation) -> float:
        """Calculate composite score for signal ranking"""
        try:
            # Weighted scoring components
            confidence_weight = 0.4
            performance_weight = 0.3
            risk_weight = 0.2  # Lower risk = higher score
            allocation_weight = 0.1
            
            # Confidence component
            confidence_score = signal_eval.adjusted_confidence
            
            # Strategy performance component
            performance = self.strategy_performance.get(signal_eval.strategy_id)
            performance_score = performance.performance_score if performance else 50
            
            # Risk component (inverted - lower risk = higher score)
            risk_score = 100 - signal_eval.risk_score
            
            # Allocation component
            allocation = self.strategy_allocations.get(signal_eval.strategy_id)
            allocation_score = 100 if allocation and signal_eval.portfolio_allocation <= allocation.allocation_percentage else 50
            
            # Calculate weighted composite score
            composite = (
                confidence_score * confidence_weight +
                performance_score * performance_weight +
                risk_score * risk_weight +
                allocation_score * allocation_weight
            )
            
            return composite
            
        except Exception as e:
            logger.error(f"Error calculating composite score: {e}")
            return 50.0
    
    async def _apply_portfolio_constraints(self,
                                         signals: List[SignalEvaluation],
                                         portfolio_data: Dict[str, Any]) -> List[SignalEvaluation]:
        """Apply portfolio-level constraints to signals"""
        
        try:
            filtered_signals = []
            total_allocation = 0.0
            strategy_positions = {}
            
            available_balance = portfolio_data.get('available_balance', 5000)
            max_portfolio_risk = portfolio_data.get('max_risk', 0.1)  # 10%
            
            for signal_eval in signals:
                strategy_id = signal_eval.strategy_id
                allocation = self.strategy_allocations.get(strategy_id)
                
                if not allocation:
                    continue
                
                # Check strategy position limits
                current_positions = strategy_positions.get(strategy_id, 0)
                if current_positions >= allocation.max_concurrent_positions:
                    signal_eval.reasoning.append(f"Strategy position limit reached ({current_positions}/{allocation.max_concurrent_positions})")
                    continue
                
                # Check portfolio allocation limits
                if total_allocation + signal_eval.portfolio_allocation > 0.8:  # Max 80% allocated
                    signal_eval.reasoning.append(f"Portfolio allocation limit would be exceeded")
                    continue
                
                # Check available balance
                position_value = signal_eval.signal.position_size_recommendation * signal_eval.signal.entry_price if signal_eval.signal.position_size_recommendation else 0
                if position_value > available_balance * 0.8:  # Max 80% of available balance
                    signal_eval.reasoning.append(f"Insufficient available balance")
                    continue
                
                # Apply risk scaling for REDUCE_SIZE recommendations
                if signal_eval.recommendation == 'REDUCE_SIZE':
                    # Reduce position size by 50%
                    if signal_eval.signal.position_size_recommendation:
                        signal_eval.signal.position_size_recommendation *= 0.5
                    signal_eval.portfolio_allocation *= 0.5
                    signal_eval.reasoning.append("Position size reduced by 50%")
                
                # Add to filtered list
                filtered_signals.append(signal_eval)
                total_allocation += signal_eval.portfolio_allocation
                strategy_positions[strategy_id] = current_positions + 1
                
                # Limit total number of concurrent signals
                if len(filtered_signals) >= 5:  # Max 5 concurrent positions
                    break
            
            logger.info(f"Applied portfolio constraints: {len(signals)} -> {len(filtered_signals)} signals")
            return filtered_signals
            
        except Exception as e:
            logger.error(f"Error applying portfolio constraints: {e}")
            return signals
    
    async def update_strategy_performance(self, 
                                        strategy_id: str, 
                                        trade_result: Dict[str, Any]):
        """Update strategy performance metrics"""
        try:
            if strategy_id not in self.strategy_performance:
                return
            
            performance = self.strategy_performance[strategy_id]
            
            # Update trade counts
            if trade_result.get('executed', False):
                performance.executed_signals += 1
                
                # Update trade outcome
                pnl = trade_result.get('pnl', 0)
                if pnl > 0:
                    performance.successful_trades += 1
                else:
                    performance.failed_trades += 1
                
                performance.total_pnl += pnl
                
                # Recalculate win rate
                total_trades = performance.successful_trades + performance.failed_trades
                performance.win_rate = (performance.successful_trades / total_trades * 100) if total_trades > 0 else 0
                
                # Update performance score
                performance.performance_score = self._calculate_performance_score(performance)
                
                # Update database
                await self._save_strategy_performance(strategy_id, performance)
                
                logger.info(f"Updated performance for {strategy_id}: Win Rate {performance.win_rate:.1f}%, "
                           f"Total PnL ${performance.total_pnl:.2f}")
            
        except Exception as e:
            logger.error(f"Error updating strategy performance: {e}")
    
    async def _save_strategy_performance(self, strategy_id: str, performance: StrategyPerformance):
        """Save strategy performance to database"""
        try:
            with get_db_session() as session:
                db_strategy = session.query(StrategyModel).filter(
                    StrategyModel.id == strategy_id
                ).first()
                
                if db_strategy:
                    db_strategy.total_trades = performance.successful_trades + performance.failed_trades
                    db_strategy.winning_trades = performance.successful_trades
                    db_strategy.losing_trades = performance.failed_trades
                    db_strategy.total_pnl = performance.total_pnl
                    db_strategy.sharpe_ratio = performance.sharpe_ratio
                    db_strategy.max_drawdown = performance.max_drawdown
                    db_strategy.last_signal_time = performance.last_signal_time
                    db_strategy.signals_generated = performance.total_signals
                    db_strategy.signals_executed = performance.executed_signals
                    
                    logger.debug(f"Saved performance data for strategy {strategy_id}")
                
        except Exception as e:
            logger.error(f"Error saving strategy performance: {e}")
    
    async def rebalance_allocations(self, market_conditions: Dict[str, Any] = None):
        """Rebalance strategy allocations based on performance"""
        try:
            if not self.strategy_performance:
                return
            
            # Calculate performance-based allocation adjustments
            total_performance = sum(perf.performance_score for perf in self.strategy_performance.values())
            
            if total_performance <= 0:
                return
            
            # Rebalance allocations based on performance
            base_allocation = 1.0 / len(self.strategy_allocations)
            
            for strategy_id, allocation in self.strategy_allocations.items():
                if not allocation.enabled:
                    continue
                
                performance = self.strategy_performance.get(strategy_id)
                if not performance:
                    continue
                
                # Calculate new allocation based on performance
                performance_ratio = performance.performance_score / total_performance
                new_allocation = (base_allocation * 0.5) + (performance_ratio * 0.5)  # 50% base + 50% performance
                
                # Apply constraints
                new_allocation = max(0.1, min(0.5, new_allocation))  # Between 10% and 50%
                
                # Update allocation
                old_allocation = allocation.allocation_percentage
                allocation.allocation_percentage = new_allocation
                
                if abs(new_allocation - old_allocation) > 0.05:  # 5% change threshold
                    logger.info(f"Rebalanced {strategy_id}: {old_allocation:.1%} -> {new_allocation:.1%}")
            
            self.last_rebalance_time = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Error rebalancing allocations: {e}")
    
    async def get_strategy_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive strategy diagnostics"""
        try:
            diagnostics = {
                'timestamp': datetime.utcnow().isoformat(),
                'total_strategies': len(self.strategies),
                'enabled_strategies': len([a for a in self.strategy_allocations.values() if a.enabled]),
                'total_signals_generated': sum(p.total_signals for p in self.strategy_performance.values()),
                'total_signals_executed': sum(p.executed_signals for p in self.strategy_performance.values()),
                'overall_win_rate': 0.0,
                'total_pnl': sum(p.total_pnl for p in self.strategy_performance.values()),
                'strategies': {},
                'allocations': {},
                'recent_signals': len(self.signal_history),
                'portfolio_allocation_used': self.portfolio_allocation_used,
                'last_rebalance': self.last_rebalance_time.isoformat() if self.last_rebalance_time else None
            }
            
            # Calculate overall win rate
            total_trades = sum(p.successful_trades + p.failed_trades for p in self.strategy_performance.values())
            total_wins = sum(p.successful_trades for p in self.strategy_performance.values())
            diagnostics['overall_win_rate'] = (total_wins / total_trades * 100) if total_trades > 0 else 0.0
            
            # Strategy-specific diagnostics
            for strategy_id, performance in self.strategy_performance.items():
                strategy = self.strategies.get(strategy_id)
                allocation = self.strategy_allocations.get(strategy_id)
                
                diagnostics['strategies'][strategy_id] = {
                    'enabled': allocation.enabled if allocation else False,
                    'allocation_percentage': allocation.allocation_percentage if allocation else 0.0,
                    'max_positions': allocation.max_concurrent_positions if allocation else 0,
                    'total_signals': performance.total_signals,
                    'executed_signals': performance.executed_signals,
                    'execution_rate': performance.execution_rate,
                    'successful_trades': performance.successful_trades,
                    'failed_trades': performance.failed_trades,
                    'win_rate': performance.win_rate,
                    'total_pnl': performance.total_pnl,
                    'sharpe_ratio': performance.sharpe_ratio,
                    'performance_score': performance.performance_score,
                    'last_signal_time': performance.last_signal_time.isoformat() if performance.last_signal_time else None,
                    'strategy_type': strategy.__class__.__name__ if strategy else 'Unknown'
                }
                
                diagnostics['allocations'][strategy_id] = {
                    'current_allocation': allocation.allocation_percentage if allocation else 0.0,
                    'risk_multiplier': allocation.risk_multiplier if allocation else 1.0,
                    'max_positions': allocation.max_concurrent_positions if allocation else 0
                }
            
            return diagnostics
            
        except Exception as e:
            logger.error(f"Error getting strategy diagnostics: {e}")
            return {'error': str(e)}
    
    async def get_signal_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of recent signals"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            recent_signals = [s for s in self.signal_history if s.timestamp >= cutoff_time]
            
            if not recent_signals:
                return {
                    'period_hours': hours,
                    'total_signals': 0,
                    'signals_by_strategy': {},
                    'signals_by_type': {},
                    'avg_confidence': 0.0,
                    'symbols_analyzed': []
                }
            
            # Analyze signals
            signals_by_strategy = {}
            signals_by_type = {}
            confidences = []
            symbols = set()
            
            for signal in recent_signals:
                # By strategy
                if signal.strategy_id not in signals_by_strategy:
                    signals_by_strategy[signal.strategy_id] = 0
                signals_by_strategy[signal.strategy_id] += 1
                
                # By type
                signal_type = signal.signal_type.value
                if signal_type not in signals_by_type:
                    signals_by_type[signal_type] = 0
                signals_by_type[signal_type] += 1
                
                confidences.append(signal.confidence)
                symbols.add(signal.symbol)
            
            return {
                'period_hours': hours,
                'total_signals': len(recent_signals),
                'signals_by_strategy': signals_by_strategy,
                'signals_by_type': signals_by_type,
                'avg_confidence': np.mean(confidences) if confidences else 0.0,
                'symbols_analyzed': list(symbols),
                'top_symbols': list(symbols)[:10]  # Top 10 most active
            }
            
        except Exception as e:
            logger.error(f"Error getting signal summary: {e}")
            return {'error': str(e)}
    
    async def optimize_strategies(self, 
                                optimization_period_days: int = 30,
                                symbols: List[str] = None) -> Dict[str, Any]:
        """Optimize strategy parameters based on historical performance"""
        try:
            optimization_results = {}
            
            for strategy_id, strategy in self.strategies.items():
                if not hasattr(strategy, 'optimize_parameters'):
                    continue
                
                try:
                    # Get historical data for optimization
                    # This would typically come from your data source
                    historical_data = await self._get_historical_data_for_optimization(
                        symbols or ['BTC/USDT'], optimization_period_days
                    )
                    
                    if not historical_data:
                        continue
                    
                    # Run optimization
                    optimization_result = await strategy.optimize_parameters(
                        historical_data, 
                        'BTC/USDT',  # Primary symbol for optimization
                        optimization_period_days
                    )
                    
                    if optimization_result.get('success'):
                        optimization_results[strategy_id] = {
                            'status': 'completed',
                            'best_score': optimization_result['best_score'],
                            'best_parameters': optimization_result['best_parameters'],
                            'combinations_tested': optimization_result['optimization_summary']['total_combinations_tested']
                        }
                        
                        # Optionally update strategy configuration
                        # strategy.config.update_from_dict(optimization_result['best_parameters'])
                        
                        logger.info(f"Optimized {strategy_id}: Score {optimization_result['best_score']:.2f}")
                    else:
                        optimization_results[strategy_id] = {
                            'status': 'failed',
                            'error': optimization_result.get('error', 'Unknown error')
                        }
                
                except Exception as e:
                    logger.error(f"Error optimizing {strategy_id}: {e}")
                    optimization_results[strategy_id] = {
                        'status': 'error',
                        'error': str(e)
                    }
            
            return {
                'optimization_period_days': optimization_period_days,
                'strategies_optimized': len(optimization_results),
                'results': optimization_results,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error optimizing strategies: {e}")
            return {'error': str(e)}
    
    async def _get_historical_data_for_optimization(self, 
                                                  symbols: List[str], 
                                                  days: int) -> Optional[pd.DataFrame]:
        """Get historical data for strategy optimization"""
        # This is a placeholder - in real implementation, you would fetch from your data source
        try:
            # Generate sample data for demonstration
            dates = pd.date_range(
                start=datetime.utcnow() - timedelta(days=days),
                end=datetime.utcnow(),
                freq='1H'
            )
            
            # Create realistic OHLCV data
            np.random.seed(42)
            base_price = 50000
            prices = [base_price]
            
            for _ in range(len(dates) - 1):
                change = np.random.normal(0, 200)  # $200 volatility
                new_price = max(prices[-1] + change, base_price * 0.8)
                prices.append(new_price)
            
            data = pd.DataFrame({
                'timestamp': dates,
                'open': [p + np.random.normal(0, 50) for p in prices],
                'high': [p + abs(np.random.normal(100, 50)) for p in prices],
                'low': [p - abs(np.random.normal(100, 50)) for p in prices],
                'close': prices,
                'volume': np.random.lognormal(15, 0.5, len(dates))
            })
            
            # Ensure OHLC relationships
            for i in range(len(data)):
                row = data.iloc[i]
                high = max(row['open'], row['close']) + abs(np.random.normal(0, 20))
                low = min(row['open'], row['close']) - abs(np.random.normal(0, 20))
                data.loc[i, 'high'] = high
                data.loc[i, 'low'] = low
            
            return data
            
        except Exception as e:
            logger.error(f"Error generating historical data: {e}")
            return None
    
    async def log_system_event(self, level: LogLevel, component: str, message: str, details: Dict[str, Any] = None):
        """Log system event to database"""
        try:
            with get_db_session() as session:
                log_entry = SystemLog(
                    level=level,
                    component=component,
                    message=message,
                    details=details or {}
                )
                session.add(log_entry)
                
        except Exception as e:
            logger.error(f"Error logging system event: {e}")
    
    async def shutdown(self):
        """Shutdown strategy manager and cleanup resources"""
        try:
            logger.info("Shutting down Strategy Manager...")
            
            self.is_running = False
            
            # Cleanup all strategies
            for strategy in self.strategies.values():
                try:
                    await strategy.cleanup()
                except Exception as e:
                    logger.error(f"Error cleaning up strategy {strategy.strategy_id}: {e}")
            
            # Shutdown thread pool
            self.thread_pool.shutdown(wait=True)
            
            # Save final performance data
            for strategy_id, performance in self.strategy_performance.items():
                await self._save_strategy_performance(strategy_id, performance)
            
            logger.info("Strategy Manager shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during Strategy Manager shutdown: {e}")
    
    def __str__(self) -> str:
        enabled_strategies = len([a for a in self.strategy_allocations.values() if a.enabled])
        return f"StrategyManager(strategies={len(self.strategies)}, enabled={enabled_strategies}, running={self.is_running})"
    
    def __repr__(self) -> str:
        return self.__str__()

# Factory function
def create_strategy_manager(config: Dict[str, Any] = None) -> StrategyManager:
    """Factory function to create Strategy Manager instance"""
    return StrategyManager(config)

# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_strategy_manager():
        print("🎯 Strategy Manager Test")
        print("=" * 50)
        
        # Create strategy manager
        config = {
            'rsi_strategy': {
                'rsi_period': 14,
                'oversold_threshold': 30,
                'overbought_threshold': 70,
                'min_confidence': 65.0
            },
            'volume_profile_strategy': {
                'lookback_period': 24,
                'volume_multiplier': 1.5,
                'min_confidence': 65.0
            }
        }
        
        manager = create_strategy_manager(config)
        
        try:
            # Initialize
            await manager.initialize()
            print(f"✅ Initialized: {manager}")
            
            # Create sample market data
            symbols = ['BTC/USDT', 'ETH/USDT']
            market_data = {}
            current_prices = {}
            
            for symbol in symbols:
                # Generate sample OHLCV data
                np.random.seed(42 if symbol == 'BTC/USDT' else 123)
                dates = pd.date_range('2024-01-01', periods=200, freq='1H')
                
                base_price = 50000 if symbol == 'BTC/USDT' else 3000
                prices = [base_price]
                
                for _ in range(199):
                    change = np.random.normal(0, base_price * 0.02)
                    new_price = max(prices[-1] + change, base_price * 0.8)
                    prices.append(new_price)
                
                market_data[symbol] = pd.DataFrame({
                    'timestamp': dates,
                    'open': [p + np.random.normal(0, base_price * 0.01) for p in prices],
                    'high': [p + abs(np.random.normal(base_price * 0.01, base_price * 0.005)) for p in prices],
                    'low': [p - abs(np.random.normal(base_price * 0.01, base_price * 0.005)) for p in prices],
                    'close': prices,
                    'volume': np.random.lognormal(15, 0.5, 200)
                })
                
                # Ensure OHLC relationships
                for i in range(len(market_data[symbol])):
                    row = market_data[symbol].iloc[i]
                    high = max(row['open'], row['close']) + abs(np.random.normal(0, base_price * 0.005))
                    low = min(row['open'], row['close']) - abs(np.random.normal(0, base_price * 0.005))
                    market_data[symbol].loc[i, 'high'] = high
                    market_data[symbol].loc[i, 'low'] = low
                
                current_prices[symbol] = prices[-1]
            
            # Portfolio data
            portfolio_data = {
                'total_value': 10000,
                'available_balance': 5000,
                'max_risk': 0.1,
                'risk_level': 'MEDIUM'
            }
            
            print(f"📊 Market data prepared for {len(symbols)} symbols")
            print(f"Current prices: {', '.join(f'{s}: ${p:.2f}' for s, p in current_prices.items())}")
            
            # Generate signals
            signal_evaluations = await manager.generate_signals(
                symbols=symbols,
                market_data=market_data,
                current_prices=current_prices,
                portfolio_data=portfolio_data
            )
            
            print(f"\n🎯 Generated {len(signal_evaluations)} signal evaluations")
            
            for i, evaluation in enumerate(signal_evaluations, 1):
                signal = evaluation.signal
                print(f"\n  Signal {i}:")
                print(f"    Strategy: {evaluation.strategy_id}")
                print(f"    Symbol: {signal.symbol}")
                print(f"    Type: {signal.signal_type.value}")
                print(f"    Confidence: {evaluation.raw_confidence:.1f}% -> {evaluation.adjusted_confidence:.1f}%")
                print(f"    Entry: ${signal.entry_price:.2f}")
                print(f"    Portfolio Allocation: {evaluation.portfolio_allocation:.1%}")
                print(f"    Risk Score: {evaluation.risk_score:.1f}")
                print(f"    Recommendation: {evaluation.recommendation}")
                print(f"    Reasoning: {'; '.join(evaluation.reasoning)}")
            
            # Test diagnostics
            diagnostics = await manager.get_strategy_diagnostics()
            print(f"\n📈 Strategy Diagnostics:")
            print(f"  Total Strategies: {diagnostics['total_strategies']}")
            print(f"  Enabled Strategies: {diagnostics['enabled_strategies']}")
            print(f"  Overall Win Rate: {diagnostics['overall_win_rate']:.1f}%")
            print(f"  Total P&L: ${diagnostics['total_pnl']:.2f}")
            
            # Test signal summary
            signal_summary = await manager.get_signal_summary(hours=1)
            print(f"\n📊 Signal Summary (1 hour):")
            print(f"  Total Signals: {signal_summary['total_signals']}")
            print(f"  Average Confidence: {signal_summary['avg_confidence']:.1f}%")
            print(f"  Symbols Analyzed: {signal_summary['symbols_analyzed']}")
            
            # Test performance update
            if signal_evaluations:
                test_trade_result = {
                    'executed': True,
                    'pnl': 150.0,
                    'success': True
                }
                
                await manager.update_strategy_performance(
                    signal_evaluations[0].strategy_id,
                    test_trade_result
                )
                print(f"✅ Updated performance for {signal_evaluations[0].strategy_id}")
            
            # Test rebalancing
            await manager.rebalance_allocations()
            print("✅ Strategy allocations rebalanced")
            
            print(f"\n🎉 Strategy Manager test completed successfully!")
            
        except Exception as e:
            print(f"❌ Error in Strategy Manager test: {e}")
            
        finally:
            # Cleanup
            await manager.shutdown()
            print("🛑 Strategy Manager shutdown completed")
    
    # Run the test
    asyncio.run(test_strategy_manager())