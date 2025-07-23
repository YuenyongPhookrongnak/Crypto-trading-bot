"""
Base Strategy Framework for Crypto Trading Bot

This module provides the abstract base class and core functionality
for all trading strategies in the system.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

import pandas as pd
import numpy as np
import talib

logger = logging.getLogger(__name__)

class SignalType(Enum):
    """Signal types for trading decisions"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    STRONG_BUY = "STRONG_BUY"
    STRONG_SELL = "STRONG_SELL"

class MarketCondition(Enum):
    """Market condition classifications"""
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    SIDEWAYS = "SIDEWAYS"
    VOLATILE = "VOLATILE"
    LOW_VOLUME = "LOW_VOLUME"

@dataclass
class TradingSignal:
    """Trading signal data structure"""
    strategy_id: str
    symbol: str
    signal_type: SignalType
    confidence: float  # 0-100
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Risk management
    risk_reward_ratio: Optional[float] = None
    position_size_recommendation: Optional[float] = None
    
    # Supporting data
    market_condition: Optional[MarketCondition] = None
    technical_indicators: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    
    # Validation flags
    volume_confirmed: bool = False
    trend_confirmed: bool = False
    multiple_timeframe_confirmed: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary"""
        return {
            'strategy_id': self.strategy_id,
            'symbol': self.symbol,
            'signal_type': self.signal_type.value,
            'confidence': self.confidence,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'timestamp': self.timestamp.isoformat(),
            'risk_reward_ratio': self.risk_reward_ratio,
            'position_size_recommendation': self.position_size_recommendation,
            'market_condition': self.market_condition.value if self.market_condition else None,
            'technical_indicators': self.technical_indicators,
            'reasoning': self.reasoning,
            'volume_confirmed': self.volume_confirmed,
            'trend_confirmed': self.trend_confirmed,
            'multiple_timeframe_confirmed': self.multiple_timeframe_confirmed
        }

@dataclass 
class StrategyConfig:
    """Strategy configuration base class"""
    enabled: bool = True
    risk_per_trade: float = 0.02  # 2%
    max_positions: int = 3
    min_confidence: float = 60.0
    timeframe: str = '1h'
    
    # Risk management
    stop_loss_percentage: float = 0.02  # 2%
    take_profit_percentage: float = 0.04  # 4%
    trailing_stop: bool = False
    
    # Filters
    volume_filter: bool = True
    trend_filter: bool = False
    min_volume_24h: float = 1000000  # $1M
    
    def update_from_dict(self, config_dict: Dict[str, Any]):
        """Update configuration from dictionary"""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

class BaseStrategy(ABC):
    """Abstract base class for all trading strategies"""
    
    def __init__(self, strategy_id: str, config: Dict[str, Any] = None):
        self.strategy_id = strategy_id
        self.config = StrategyConfig()
        
        if config:
            self.config.update_from_dict(config)
        
        # State tracking
        self.is_initialized = False
        self.last_signal_time = None
        self.active_signals = {}
        self.performance_metrics = {
            'signals_generated': 0,
            'successful_signals': 0,
            'failed_signals': 0,
            'avg_confidence': 0.0,
            'last_updated': datetime.utcnow()
        }
        
        # Technical analysis setup
        self.required_indicators = self.get_required_indicators()
        self.lookback_period = self.get_lookback_period()
        
        logger.info(f"Strategy {self.strategy_id} initialized with config: {self.config.__dict__}")
    
    @abstractmethod
    def get_required_indicators(self) -> List[str]:
        """Return list of required technical indicators"""
        pass
    
    @abstractmethod
    def get_lookback_period(self) -> int:
        """Return minimum lookback period required for analysis"""
        pass
    
    @abstractmethod
    async def analyze_symbol(self, 
                           symbol: str, 
                           ohlcv_data: pd.DataFrame,
                           current_price: float,
                           additional_data: Dict[str, Any] = None) -> Optional[TradingSignal]:
        """Analyze symbol and generate trading signal"""
        pass
    
    async def initialize(self):
        """Initialize strategy (override if needed)"""
        self.is_initialized = True
        logger.info(f"Strategy {self.strategy_id} initialized successfully")
    
    async def validate_signal(self, signal: TradingSignal, market_data: Dict[str, Any]) -> bool:
        """Validate generated signal before execution"""
        try:
            # Basic validation
            if signal.confidence < self.config.min_confidence:
                logger.debug(f"Signal rejected: confidence {signal.confidence} < {self.config.min_confidence}")
                return False
            
            # Volume validation
            if self.config.volume_filter and market_data:
                volume_24h = market_data.get('volume_24h', 0)
                if volume_24h < self.config.min_volume_24h:
                    logger.debug(f"Signal rejected: volume {volume_24h} < {self.config.min_volume_24h}")
                    return False
            
            # Risk/Reward validation
            if signal.risk_reward_ratio and signal.risk_reward_ratio < 1.5:
                logger.debug(f"Signal rejected: R/R ratio {signal.risk_reward_ratio} < 1.5")
                return False
            
            # Timing validation (avoid too frequent signals)
            if self.last_signal_time:
                time_since_last = datetime.utcnow() - self.last_signal_time
                if time_since_last < timedelta(minutes=15):
                    logger.debug("Signal rejected: too frequent signals")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating signal: {e}")
            return False
    
    def calculate_position_size(self, 
                              signal: TradingSignal, 
                              portfolio_value: float,
                              current_price: float) -> float:
        """Calculate recommended position size"""
        try:
            # Risk-based position sizing
            risk_amount = portfolio_value * self.config.risk_per_trade
            
            if signal.stop_loss:
                risk_per_unit = abs(current_price - signal.stop_loss)
                if risk_per_unit > 0:
                    position_size = risk_amount / risk_per_unit
                    
                    # Apply confidence adjustment
                    confidence_multiplier = signal.confidence / 100.0
                    adjusted_position_size = position_size * confidence_multiplier
                    
                    return adjusted_position_size
            
            # Fallback: percentage of portfolio
            fallback_percentage = 0.05  # 5%
            return (portfolio_value * fallback_percentage) / current_price
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    def calculate_stop_loss(self, 
                          entry_price: float, 
                          signal_type: SignalType,
                          atr: float = None) -> float:
        """Calculate stop loss price"""
        try:
            if signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                # Long position - stop loss below entry
                if atr:
                    # ATR-based stop loss (more dynamic)
                    return entry_price - (2.0 * atr)
                else:
                    # Percentage-based stop loss
                    return entry_price * (1 - self.config.stop_loss_percentage)
            
            elif signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
                # Short position - stop loss above entry
                if atr:
                    return entry_price + (2.0 * atr)
                else:
                    return entry_price * (1 + self.config.stop_loss_percentage)
            
            return entry_price
            
        except Exception as e:
            logger.error(f"Error calculating stop loss: {e}")
            return entry_price
    
    def calculate_take_profit(self, 
                            entry_price: float, 
                            signal_type: SignalType,
                            risk_reward_ratio: float = 2.0) -> float:
        """Calculate take profit price"""
        try:
            if signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                # Long position - take profit above entry
                profit_percentage = self.config.stop_loss_percentage * risk_reward_ratio
                return entry_price * (1 + profit_percentage)
            
            elif signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
                # Short position - take profit below entry
                profit_percentage = self.config.stop_loss_percentage * risk_reward_ratio
                return entry_price * (1 - profit_percentage)
            
            return entry_price
            
        except Exception as e:
            logger.error(f"Error calculating take profit: {e}")
            return entry_price
    
    def detect_market_condition(self, ohlcv_data: pd.DataFrame) -> MarketCondition:
        """Detect current market condition"""
        try:
            if len(ohlcv_data) < 20:
                return MarketCondition.SIDEWAYS
            
            # Calculate trend indicators
            sma_20 = talib.SMA(ohlcv_data['close'].values, timeperiod=20)
            sma_50 = talib.SMA(ohlcv_data['close'].values, timeperiod=50)
            
            current_price = ohlcv_data['close'].iloc[-1]
            current_sma_20 = sma_20[-1]
            current_sma_50 = sma_50[-1]
            
            # Volume analysis
            avg_volume = ohlcv_data['volume'].rolling(20).mean().iloc[-1]
            recent_volume = ohlcv_data['volume'].iloc[-1]
            
            # Volatility analysis
            returns = ohlcv_data['close'].pct_change()
            volatility = returns.rolling(20).std().iloc[-1]
            
            # Determine conditions
            if recent_volume < avg_volume * 0.5:
                return MarketCondition.LOW_VOLUME
            
            if volatility > returns.std() * 1.5:
                return MarketCondition.VOLATILE
            
            # Trend detection
            if current_price > current_sma_20 > current_sma_50:
                return MarketCondition.TRENDING_UP
            elif current_price < current_sma_20 < current_sma_50:
                return MarketCondition.TRENDING_DOWN
            else:
                return MarketCondition.SIDEWAYS
                
        except Exception as e:
            logger.error(f"Error detecting market condition: {e}")
            return MarketCondition.SIDEWAYS
    
    def calculate_technical_indicators(self, ohlcv_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate common technical indicators"""
        try:
            indicators = {}
            
            high = ohlcv_data['high'].values
            low = ohlcv_data['low'].values
            close = ohlcv_data['close'].values
            volume = ohlcv_data['volume'].values
            
            # Trend indicators
            indicators['sma_20'] = talib.SMA(close, timeperiod=20)
            indicators['sma_50'] = talib.SMA(close, timeperiod=50)
            indicators['ema_12'] = talib.EMA(close, timeperiod=12)
            indicators['ema_26'] = talib.EMA(close, timeperiod=26)
            
            # Momentum indicators
            indicators['rsi'] = talib.RSI(close, timeperiod=14)
            indicators['macd'], indicators['macd_signal'], indicators['macd_histogram'] = talib.MACD(close)
            indicators['stoch_k'], indicators['stoch_d'] = talib.STOCH(high, low, close)
            
            # Volatility indicators
            indicators['bb_upper'], indicators['bb_middle'], indicators['bb_lower'] = talib.BBANDS(close)
            indicators['atr'] = talib.ATR(high, low, close, timeperiod=14)
            
            # Volume indicators
            indicators['obv'] = talib.OBV(close, volume)
            indicators['ad'] = talib.AD(high, low, close, volume)
            
            # Support/Resistance levels
            indicators['pivot_point'] = (high[-1] + low[-1] + close[-1]) / 3
            indicators['resistance_1'] = 2 * indicators['pivot_point'] - low[-1]
            indicators['support_1'] = 2 * indicators['pivot_point'] - high[-1]
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return {}
    
    def confirm_with_volume(self, 
                          signal: TradingSignal, 
                          ohlcv_data: pd.DataFrame) -> bool:
        """Confirm signal with volume analysis"""
        try:
            if len(ohlcv_data) < 10:
                return False
            
            current_volume = ohlcv_data['volume'].iloc[-1]
            avg_volume = ohlcv_data['volume'].rolling(20).mean().iloc[-1]
            
            # Volume should be above average for strong signals
            volume_ratio = current_volume / avg_volume
            
            if signal.signal_type in [SignalType.STRONG_BUY, SignalType.STRONG_SELL]:
                return volume_ratio > 1.5  # 50% above average
            else:
                return volume_ratio > 1.2  # 20% above average
                
        except Exception as e:
            logger.error(f"Error confirming with volume: {e}")
            return False
    
    def confirm_with_trend(self, 
                         signal: TradingSignal, 
                         market_condition: MarketCondition) -> bool:
        """Confirm signal with trend analysis"""
        try:
            if not self.config.trend_filter:
                return True
            
            # Buy signals should align with uptrend
            if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                return market_condition == MarketCondition.TRENDING_UP
            
            # Sell signals should align with downtrend
            elif signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
                return market_condition == MarketCondition.TRENDING_DOWN
            
            return True
            
        except Exception as e:
            logger.error(f"Error confirming with trend: {e}")
            return False
    
    def update_performance_metrics(self, signal_success: bool, confidence: float):
        """Update strategy performance metrics"""
        try:
            self.performance_metrics['signals_generated'] += 1
            
            if signal_success:
                self.performance_metrics['successful_signals'] += 1
            else:
                self.performance_metrics['failed_signals'] += 1
            
            # Update average confidence
            total_signals = self.performance_metrics['signals_generated']
            current_avg = self.performance_metrics['avg_confidence']
            self.performance_metrics['avg_confidence'] = (
                (current_avg * (total_signals - 1) + confidence) / total_signals
            )
            
            self.performance_metrics['last_updated'] = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get strategy performance summary"""
        metrics = self.performance_metrics.copy()
        
        if metrics['signals_generated'] > 0:
            metrics['success_rate'] = (
                metrics['successful_signals'] / metrics['signals_generated'] * 100
            )
        else:
            metrics['success_rate'] = 0.0
        
        return metrics
    
    def should_exit_position(self, 
                           entry_signal: TradingSignal,
                           current_price: float,
                           current_data: pd.DataFrame) -> Optional[str]:
        """Check if position should be exited (override in derived classes)"""
        try:
            # Basic stop loss/take profit logic
            if entry_signal.stop_loss and current_price <= entry_signal.stop_loss:
                return "STOP_LOSS"
            
            if entry_signal.take_profit and current_price >= entry_signal.take_profit:
                return "TAKE_PROFIT"
            
            # Time-based exit (if holding too long)
            if entry_signal.timestamp:
                holding_time = datetime.utcnow() - entry_signal.timestamp
                if holding_time > timedelta(hours=24):  # 24 hours max
                    return "TIME_LIMIT"
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking exit conditions: {e}")
            return None
    
    async def cleanup(self):
        """Cleanup strategy resources"""
        self.active_signals.clear()
        logger.info(f"Strategy {self.strategy_id} cleaned up successfully")
    
    def __str__(self) -> str:
        return f"BaseStrategy(id={self.strategy_id}, enabled={self.config.enabled})"
    
    def __repr__(self) -> str:
        return self.__str__()

class StrategyValidationMixin:
    """Mixin class for additional strategy validation"""
    
    def validate_data_quality(self, ohlcv_data: pd.DataFrame) -> bool:
        """Validate OHLCV data quality"""
        try:
            if ohlcv_data.empty:
                return False
            
            # Check for required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in ohlcv_data.columns for col in required_columns):
                return False
            
            # Check for NaN values
            if ohlcv_data[required_columns].isnull().any().any():
                return False
            
            # Check for reasonable OHLC relationships
            if not (ohlcv_data['high'] >= ohlcv_data['low']).all():
                return False
            
            if not (ohlcv_data['high'] >= ohlcv_data['open']).all():
                return False
            
            if not (ohlcv_data['high'] >= ohlcv_data['close']).all():
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating data quality: {e}")
            return False

class StrategyBacktestMixin:
    """Mixin class for strategy backtesting capabilities"""
    
    def backtest_signal(self, 
                       signal: TradingSignal,
                       future_data: pd.DataFrame,
                       days_ahead: int = 5) -> Dict[str, Any]:
        """Backtest a signal against future data"""
        try:
            if future_data.empty or len(future_data) == 0:
                return {'success': False, 'error': 'No future data'}
            
            entry_price = signal.entry_price
            max_profit = 0
            max_loss = 0
            exit_price = None
            exit_reason = None
            
            for i, (timestamp, row) in enumerate(future_data.iterrows()):
                current_price = row['close']
                
                # Calculate current P&L
                if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                    pnl = (current_price - entry_price) / entry_price
                else:
                    pnl = (entry_price - current_price) / entry_price
                
                max_profit = max(max_profit, pnl)
                max_loss = min(max_loss, pnl)
                
                # Check exit conditions
                if signal.stop_loss:
                    if (signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY] and 
                        current_price <= signal.stop_loss):
                        exit_price = signal.stop_loss
                        exit_reason = 'STOP_LOSS'
                        break
                
                if signal.take_profit:
                    if (signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY] and 
                        current_price >= signal.take_profit):
                        exit_price = signal.take_profit
                        exit_reason = 'TAKE_PROFIT'
                        break
                
                # Time-based exit
                if i >= days_ahead * 24:  # Assuming hourly data
                    exit_price = current_price
                    exit_reason = 'TIME_LIMIT'
                    break
            
            # Calculate final results
            if exit_price:
                if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                    final_pnl = (exit_price - entry_price) / entry_price
                else:
                    final_pnl = (entry_price - exit_price) / entry_price
            else:
                final_pnl = max_loss  # Position still open, use current loss
                exit_reason = 'OPEN'
            
            return {
                'success': True,
                'final_pnl': final_pnl,
                'max_profit': max_profit,
                'max_loss': max_loss,
                'exit_price': exit_price,
                'exit_reason': exit_reason,
                'signal_successful': final_pnl > 0
            }
            
        except Exception as e:
            logger.error(f"Error backtesting signal: {e}")
            return {'success': False, 'error': str(e)}

if __name__ == "__main__":
    # Example usage and testing
    print("🎯 Base Strategy Framework Test")
    print("=" * 50)
    
    # Create a mock strategy class for testing
    class MockStrategy(BaseStrategy):
        def get_required_indicators(self) -> List[str]:
            return ['sma_20', 'rsi', 'macd']
        
        def get_lookback_period(self) -> int:
            return 50
        
        async def analyze_symbol(self, symbol: str, ohlcv_data: pd.DataFrame, 
                               current_price: float, additional_data: Dict[str, Any] = None) -> Optional[TradingSignal]:
            # Mock analysis
            return TradingSignal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                signal_type=SignalType.BUY,
                confidence=75.0,
                entry_price=current_price,
                reasoning="Mock signal for testing"
            )
    
    # Test the base strategy
    strategy = MockStrategy("test_strategy", {"enabled": True, "min_confidence": 60.0})
    
    print(f"Strategy ID: {strategy.strategy_id}")
    print(f"Required indicators: {strategy.required_indicators}")
    print(f"Lookback period: {strategy.lookback_period}")
    print(f"Configuration: {strategy.config.__dict__}")
    
    # Test signal creation
    test_signal = TradingSignal(
        strategy_id="test",
        symbol="BTC/USDT",
        signal_type=SignalType.BUY,
        confidence=80.0,
        entry_price=50000.0
    )
    
    print(f"\nTest signal: {test_signal}")
    print(f"Signal dict: {test_signal.to_dict()}")
    
    print("\n✅ Base strategy framework test completed!")