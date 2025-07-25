"""
RSI Mean Reversion Trading Strategy

This strategy identifies potential reversal points using RSI (Relative Strength Index)
combined with additional confirmation indicators for enhanced accuracy.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import talib

from strategies.base_strategy import (
    BaseStrategy, TradingSignal, SignalType, MarketCondition,
    StrategyConfig, StrategyValidationMixin, StrategyBacktestMixin
)

logger = logging.getLogger(__name__)

@dataclass
class RSIStrategyConfig(StrategyConfig):
    """RSI Strategy specific configuration"""
    
    # RSI Parameters
    rsi_period: int = 14
    oversold_threshold: float = 30.0
    overbought_threshold: float = 70.0
    extreme_oversold: float = 20.0
    extreme_overbought: float = 80.0
    
    # Confirmation parameters
    confirmation_period: int = 3
    divergence_lookback: int = 10
    volume_confirmation: bool = True
    trend_alignment: bool = False
    
    # Signal strength levels
    weak_signal_threshold: float = 50.0
    medium_signal_threshold: float = 70.0
    strong_signal_threshold: float = 85.0
    
    # Exit conditions
    rsi_exit_level: float = 55.0  # Exit when RSI returns to neutral
    profit_target_multiplier: float = 2.0
    trailing_stop_enabled: bool = True
    trailing_stop_percentage: float = 0.01  # 1%
    
    # Risk management
    max_holding_period_hours: int = 48
    scale_in_enabled: bool = False
    scale_out_enabled: bool = True

class RSIStrategy(BaseStrategy, StrategyValidationMixin, StrategyBacktestMixin):
    """RSI Mean Reversion Trading Strategy Implementation"""
    
    def __init__(self, strategy_id: str = "rsi_strategy", config: Dict[str, Any] = None):
        # Initialize with RSI-specific config
        super().__init__(strategy_id, config)
        self.config = RSIStrategyConfig()
        
        if config:
            self.config.update_from_dict(config)
        
        # Strategy state
        self.active_positions = {}
        self.recent_signals = []
        self.rsi_history = {}
        
        logger.info(f"RSI Strategy initialized with config: {self.config.__dict__}")
    
    def get_required_indicators(self) -> List[str]:
        """Return list of required technical indicators"""
        return [
            'rsi', 'sma_20', 'sma_50', 'ema_9', 'macd', 'macd_signal', 
            'bb_upper', 'bb_lower', 'atr', 'obv', 'stoch_k', 'stoch_d'
        ]
    
    def get_lookback_period(self) -> int:
        """Return minimum lookback period required for analysis"""
        return max(50, self.config.rsi_period * 3, self.config.divergence_lookback * 2)
    
    async def analyze_symbol(self, 
                           symbol: str, 
                           ohlcv_data: pd.DataFrame,
                           current_price: float,
                           additional_data: Dict[str, Any] = None) -> Optional[TradingSignal]:
        """Main analysis method for RSI strategy"""
        
        try:
            # Validate input data
            if not self.validate_data_quality(ohlcv_data):
                logger.warning(f"Invalid data quality for {symbol}")
                return None
            
            if len(ohlcv_data) < self.get_lookback_period():
                logger.debug(f"Insufficient data for {symbol}: {len(ohlcv_data)} < {self.get_lookback_period()}")
                return None
            
            # Calculate technical indicators
            indicators = self.calculate_technical_indicators(ohlcv_data)
            if not indicators:
                logger.warning(f"Failed to calculate indicators for {symbol}")
                return None
            
            # Get current values
            current_rsi = indicators['rsi'][-1]
            current_stoch_k = indicators['stoch_k'][-1]
            current_macd = indicators['macd'][-1]
            current_macd_signal = indicators['macd_signal'][-1]
            
            # Store RSI history for divergence analysis
            self.rsi_history[symbol] = indicators['rsi'][-self.config.divergence_lookback:]
            
            # Detect market condition
            market_condition = self.detect_market_condition(ohlcv_data)
            
            # Generate trading signal
            signal = await self._generate_rsi_signal(
                symbol, ohlcv_data, current_price, indicators, market_condition
            )
            
            if signal:
                # Enhance signal with additional analysis
                signal = await self._enhance_signal_with_confirmations(
                    signal, ohlcv_data, indicators, market_condition
                )
                
                # Calculate position sizing
                signal.position_size_recommendation = self.calculate_position_size(
                    signal, 
                    portfolio_value=additional_data.get('portfolio_value', 10000),
                    current_price=current_price
                )
                
                # Store signal for tracking
                self.recent_signals.append(signal)
                if len(self.recent_signals) > 20:
                    self.recent_signals = self.recent_signals[-20:]
                
                logger.info(f"RSI signal generated for {symbol}: {signal.signal_type.value} "
                           f"(Confidence: {signal.confidence:.1f}%, RSI: {current_rsi:.1f})")
                
                return signal
            
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol} with RSI strategy: {e}")
            return None
    
    async def _generate_rsi_signal(self, 
                                 symbol: str,
                                 ohlcv_data: pd.DataFrame,
                                 current_price: float,
                                 indicators: Dict[str, Any],
                                 market_condition: MarketCondition) -> Optional[TradingSignal]:
        """Generate RSI-based trading signal"""
        
        try:
            current_rsi = indicators['rsi'][-1]
            previous_rsi = indicators['rsi'][-2] if len(indicators['rsi']) > 1 else current_rsi
            
            signal_type = None
            base_confidence = 0.0
            reasoning = []
            
            # Check for oversold conditions (Buy signals)
            if current_rsi <= self.config.oversold_threshold:
                if current_rsi <= self.config.extreme_oversold:
                    signal_type = SignalType.STRONG_BUY
                    base_confidence = 85.0
                    reasoning.append(f"Extreme oversold RSI: {current_rsi:.1f}")
                else:
                    signal_type = SignalType.BUY
                    base_confidence = 70.0
                    reasoning.append(f"Oversold RSI: {current_rsi:.1f}")
                
                # Additional confidence for RSI turning up
                if current_rsi > previous_rsi:
                    base_confidence += 10.0
                    reasoning.append("RSI showing upward momentum")
            
            # Check for overbought conditions (Sell signals)
            elif current_rsi >= self.config.overbought_threshold:
                if current_rsi >= self.config.extreme_overbought:
                    signal_type = SignalType.STRONG_SELL
                    base_confidence = 85.0
                    reasoning.append(f"Extreme overbought RSI: {current_rsi:.1f}")
                else:
                    signal_type = SignalType.SELL
                    base_confidence = 70.0
                    reasoning.append(f"Overbought RSI: {current_rsi:.1f}")
                
                # Additional confidence for RSI turning down
                if current_rsi < previous_rsi:
                    base_confidence += 10.0
                    reasoning.append("RSI showing downward momentum")
            
            if not signal_type:
                return None
            
            # Check for RSI divergence
            divergence_strength = self._detect_rsi_divergence(symbol, ohlcv_data, indicators)
            if divergence_strength > 0:
                base_confidence += divergence_strength
                reasoning.append(f"RSI divergence detected (strength: {divergence_strength:.1f})")
            
            # Create signal
            signal = TradingSignal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                signal_type=signal_type,
                confidence=min(base_confidence, 95.0),  # Cap at 95%
                entry_price=current_price,
                market_condition=market_condition,
                technical_indicators={
                    'rsi': current_rsi,
                    'previous_rsi': previous_rsi,
                    'rsi_trend': 'up' if current_rsi > previous_rsi else 'down'
                },
                reasoning='; '.join(reasoning)
            )
            
            # Calculate stop loss and take profit
            signal.stop_loss = self._calculate_rsi_stop_loss(signal, indicators, ohlcv_data)
            signal.take_profit = self._calculate_rsi_take_profit(signal, indicators, ohlcv_data)
            
            # Calculate risk/reward ratio
            if signal.stop_loss and signal.take_profit:
                risk = abs(signal.entry_price - signal.stop_loss)
                reward = abs(signal.take_profit - signal.entry_price)
                signal.risk_reward_ratio = reward / risk if risk > 0 else None
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating RSI signal for {symbol}: {e}")
            return None
    
    async def _enhance_signal_with_confirmations(self,
                                               signal: TradingSignal,
                                               ohlcv_data: pd.DataFrame,
                                               indicators: Dict[str, Any],
                                               market_condition: MarketCondition) -> TradingSignal:
        """Enhance signal with additional confirmations"""
        
        try:
            confirmation_score = 0.0
            confirmations = []
            
            # 1. Stochastic confirmation
            stoch_confirmation = self._check_stochastic_confirmation(signal, indicators)
            if stoch_confirmation['confirmed']:
                confirmation_score += 15.0
                confirmations.append(stoch_confirmation['reason'])
            
            # 2. MACD confirmation
            macd_confirmation = self._check_macd_confirmation(signal, indicators)
            if macd_confirmation['confirmed']:
                confirmation_score += 10.0
                confirmations.append(macd_confirmation['reason'])
            
            # 3. Volume confirmation
            if self.config.volume_confirmation:
                volume_confirmed = self.confirm_with_volume(signal, ohlcv_data)
                if volume_confirmed:
                    confirmation_score += 15.0
                    confirmations.append("High volume confirmation")
                    signal.volume_confirmed = True
            
            # 4. Bollinger Bands confirmation
            bb_confirmation = self._check_bollinger_bands_confirmation(signal, indicators)
            if bb_confirmation['confirmed']:
                confirmation_score += 10.0
                confirmations.append(bb_confirmation['reason'])
            
            # 5. Trend alignment (if enabled)
            if self.config.trend_alignment:
                trend_confirmed = self.confirm_with_trend(signal, market_condition)
                if trend_confirmed:
                    confirmation_score += 20.0
                    confirmations.append("Trend alignment confirmed")
                    signal.trend_confirmed = True
                else:
                    confirmation_score -= 10.0  # Penalty for going against trend
                    confirmations.append("Against main trend")
            
            # 6. Support/Resistance levels
            sr_confirmation = self._check_support_resistance_confirmation(signal, indicators, ohlcv_data)
            if sr_confirmation['confirmed']:
                confirmation_score += 12.0
                confirmations.append(sr_confirmation['reason'])
            
            # Apply confirmation adjustments
            signal.confidence += confirmation_score
            signal.confidence = max(0.0, min(signal.confidence, 98.0))  # Cap between 0-98%
            
            # Update reasoning
            if confirmations:
                signal.reasoning += f" | Confirmations: {'; '.join(confirmations)}"
            
            # Set multiple timeframe confirmed flag
            if confirmation_score >= 30.0:
                signal.multiple_timeframe_confirmed = True
            
            return signal
            
        except Exception as e:
            logger.error(f"Error enhancing signal with confirmations: {e}")
            return signal
    
    def _detect_rsi_divergence(self, symbol: str, ohlcv_data: pd.DataFrame, 
                             indicators: Dict[str, Any]) -> float:
        """Detect RSI divergence patterns"""
        
        try:
            if symbol not in self.rsi_history or len(self.rsi_history[symbol]) < 5:
                return 0.0
            
            recent_prices = ohlcv_data['close'].tail(self.config.divergence_lookback).values
            recent_rsi = self.rsi_history[symbol]
            
            if len(recent_prices) != len(recent_rsi):
                return 0.0
            
            # Look for bullish divergence (price makes lower lows, RSI makes higher lows)
            price_trend = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
            rsi_trend = np.polyfit(range(len(recent_rsi)), recent_rsi, 1)[0]
            
            divergence_strength = 0.0
            
            # Bullish divergence
            if price_trend < 0 and rsi_trend > 0:
                divergence_strength = min(abs(price_trend) + abs(rsi_trend), 15.0)
                logger.debug(f"Bullish divergence detected for {symbol}: {divergence_strength:.1f}")
            
            # Bearish divergence  
            elif price_trend > 0 and rsi_trend < 0:
                divergence_strength = min(abs(price_trend) + abs(rsi_trend), 15.0)
                logger.debug(f"Bearish divergence detected for {symbol}: {divergence_strength:.1f}")
            
            return divergence_strength
            
        except Exception as e:
            logger.error(f"Error detecting RSI divergence for {symbol}: {e}")
            return 0.0
    
    def _check_stochastic_confirmation(self, signal: TradingSignal, 
                                     indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Check Stochastic oscillator confirmation"""
        
        try:
            stoch_k = indicators['stoch_k'][-1]
            stoch_d = indicators['stoch_d'][-1]
            
            # Buy signal confirmations
            if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                if stoch_k < 20 and stoch_d < 20:  # Both oversold
                    return {
                        'confirmed': True,
                        'reason': f"Stochastic oversold (K:{stoch_k:.1f}, D:{stoch_d:.1f})"
                    }
                elif stoch_k > stoch_d and stoch_k < 30:  # Bullish crossover in oversold
                    return {
                        'confirmed': True,
                        'reason': "Stochastic bullish crossover in oversold"
                    }
            
            # Sell signal confirmations
            elif signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
                if stoch_k > 80 and stoch_d > 80:  # Both overbought
                    return {
                        'confirmed': True,
                        'reason': f"Stochastic overbought (K:{stoch_k:.1f}, D:{stoch_d:.1f})"
                    }
                elif stoch_k < stoch_d and stoch_k > 70:  # Bearish crossover in overbought
                    return {
                        'confirmed': True,
                        'reason': "Stochastic bearish crossover in overbought"
                    }
            
            return {'confirmed': False, 'reason': 'No stochastic confirmation'}
            
        except Exception as e:
            logger.error(f"Error checking stochastic confirmation: {e}")
            return {'confirmed': False, 'reason': 'Stochastic error'}
    
    def _check_macd_confirmation(self, signal: TradingSignal, 
                               indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Check MACD confirmation"""
        
        try:
            macd = indicators['macd'][-1]
            macd_signal = indicators['macd_signal'][-1]
            macd_prev = indicators['macd'][-2] if len(indicators['macd']) > 1 else macd
            signal_prev = indicators['macd_signal'][-2] if len(indicators['macd_signal']) > 1 else macd_signal
            
            # Buy signal confirmations
            if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                # MACD bullish crossover
                if macd > macd_signal and macd_prev <= signal_prev:
                    return {
                        'confirmed': True,
                        'reason': "MACD bullish crossover"
                    }
                # MACD above zero and rising
                elif macd > 0 and macd > macd_prev:
                    return {
                        'confirmed': True,
                        'reason': "MACD positive and rising"
                    }
            
            # Sell signal confirmations
            elif signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
                # MACD bearish crossover
                if macd < macd_signal and macd_prev >= signal_prev:
                    return {
                        'confirmed': True,
                        'reason': "MACD bearish crossover"
                    }
                # MACD below zero and falling
                elif macd < 0 and macd < macd_prev:
                    return {
                        'confirmed': True,
                        'reason': "MACD negative and falling"
                    }
            
            return {'confirmed': False, 'reason': 'No MACD confirmation'}
            
        except Exception as e:
            logger.error(f"Error checking MACD confirmation: {e}")
            return {'confirmed': False, 'reason': 'MACD error'}
    
    def _check_bollinger_bands_confirmation(self, signal: TradingSignal,
                                          indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Check Bollinger Bands confirmation"""
        
        try:
            current_price = signal.entry_price
            bb_upper = indicators['bb_upper'][-1]
            bb_lower = indicators['bb_lower'][-1]
            bb_middle = indicators['bb_middle'][-1]
            
            # Buy signal confirmations
            if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                if current_price <= bb_lower:
                    return {
                        'confirmed': True,
                        'reason': "Price at lower Bollinger Band"
                    }
                elif current_price < bb_middle and current_price > bb_lower:
                    return {
                        'confirmed': True,
                        'reason': "Price below BB middle"
                    }
            
            # Sell signal confirmations
            elif signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
                if current_price >= bb_upper:
                    return {
                        'confirmed': True,
                        'reason': "Price at upper Bollinger Band"
                    }
                elif current_price > bb_middle and current_price < bb_upper:
                    return {
                        'confirmed': True,
                        'reason': "Price above BB middle"
                    }
            
            return {'confirmed': False, 'reason': 'No BB confirmation'}
            
        except Exception as e:
            logger.error(f"Error checking Bollinger Bands confirmation: {e}")
            return {'confirmed': False, 'reason': 'BB error'}
    
    def _check_support_resistance_confirmation(self, signal: TradingSignal,
                                             indicators: Dict[str, Any],
                                             ohlcv_data: pd.DataFrame) -> Dict[str, Any]:
        """Check Support/Resistance level confirmation"""
        
        try:
            current_price = signal.entry_price
            pivot_point = indicators['pivot_point']
            support_1 = indicators['support_1']
            resistance_1 = indicators['resistance_1']
            
            # Calculate additional S/R levels from recent highs/lows
            recent_highs = ohlcv_data['high'].rolling(20).max().iloc[-1]
            recent_lows = ohlcv_data['low'].rolling(20).min().iloc[-1]
            
            tolerance = current_price * 0.005  # 0.5% tolerance
            
            # Buy signal confirmations
            if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                # Near support levels
                if abs(current_price - support_1) <= tolerance:
                    return {
                        'confirmed': True,
                        'reason': f"Price near support level: {support_1:.2f}"
                    }
                elif abs(current_price - recent_lows) <= tolerance:
                    return {
                        'confirmed': True,
                        'reason': f"Price near recent support: {recent_lows:.2f}"
                    }
            
            # Sell signal confirmations
            elif signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
                # Near resistance levels
                if abs(current_price - resistance_1) <= tolerance:
                    return {
                        'confirmed': True,
                        'reason': f"Price near resistance level: {resistance_1:.2f}"
                    }
                elif abs(current_price - recent_highs) <= tolerance:
                    return {
                        'confirmed': True,
                        'reason': f"Price near recent resistance: {recent_highs:.2f}"
                    }
            
            return {'confirmed': False, 'reason': 'No S/R confirmation'}
            
        except Exception as e:
            logger.error(f"Error checking Support/Resistance confirmation: {e}")
            return {'confirmed': False, 'reason': 'S/R error'}
    
    def _calculate_rsi_stop_loss(self, signal: TradingSignal, 
                               indicators: Dict[str, Any],
                               ohlcv_data: pd.DataFrame) -> float:
        """Calculate RSI-specific stop loss"""
        
        try:
            atr = indicators['atr'][-1]
            entry_price = signal.entry_price
            
            # Use ATR-based stop loss for better volatility adjustment
            if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                # For buy signals, stop loss below entry
                atr_stop = entry_price - (2.0 * atr)
                percentage_stop = entry_price * (1 - self.config.stop_loss_percentage)
                
                # Use the more conservative (closer to entry) stop loss
                stop_loss = max(atr_stop, percentage_stop)
                
                # Additional protection: don't go below recent low
                recent_low = ohlcv_data['low'].tail(10).min()
                stop_loss = max(stop_loss, recent_low * 0.995)  # 0.5% below recent low
                
            else:
                # For sell signals, stop loss above entry
                atr_stop = entry_price + (2.0 * atr)
                percentage_stop = entry_price * (1 + self.config.stop_loss_percentage)
                
                # Use the more conservative (closer to entry) stop loss
                stop_loss = min(atr_stop, percentage_stop)
                
                # Additional protection: don't go above recent high
                recent_high = ohlcv_data['high'].tail(10).max()
                stop_loss = min(stop_loss, recent_high * 1.005)  # 0.5% above recent high
            
            return stop_loss
            
        except Exception as e:
            logger.error(f"Error calculating RSI stop loss: {e}")
            return self.calculate_stop_loss(signal.entry_price, signal.signal_type, indicators.get('atr', [0])[-1])
    
    def _calculate_rsi_take_profit(self, signal: TradingSignal,
                                 indicators: Dict[str, Any],
                                 ohlcv_data: pd.DataFrame) -> float:
        """Calculate RSI-specific take profit"""
        
        try:
            entry_price = signal.entry_price
            
            # Base take profit calculation
            base_take_profit = self.calculate_take_profit(
                entry_price, signal.signal_type, self.config.profit_target_multiplier
            )
            
            # Adjust based on RSI levels and market volatility
            atr = indicators['atr'][-1]
            current_rsi = indicators['rsi'][-1]
            
            # More aggressive take profit for extreme RSI levels
            if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                if current_rsi <= self.config.extreme_oversold:
                    # More aggressive target for extreme oversold
                    aggressive_target = entry_price + (3.0 * atr)
                    base_take_profit = max(base_take_profit, aggressive_target)
                
                # Consider resistance levels
                resistance_1 = indicators.get('resistance_1', entry_price * 1.02)
                if resistance_1 > entry_price:
                    # Take profit slightly below resistance
                    resistance_target = resistance_1 * 0.995
                    base_take_profit = min(base_take_profit, resistance_target)
            
            elif signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
                if current_rsi >= self.config.extreme_overbought:
                    # More aggressive target for extreme overbought
                    aggressive_target = entry_price - (3.0 * atr)
                    base_take_profit = min(base_take_profit, aggressive_target)
                
                # Consider support levels
                support_1 = indicators.get('support_1', entry_price * 0.98)
                if support_1 < entry_price:
                    # Take profit slightly above support
                    support_target = support_1 * 1.005
                    base_take_profit = max(base_take_profit, support_target)
            
            return base_take_profit
            
        except Exception as e:
            logger.error(f"Error calculating RSI take profit: {e}")
            return self.calculate_take_profit(signal.entry_price, signal.signal_type)
    
    async def should_exit_position(self, entry_signal: TradingSignal,
                                 current_price: float,
                                 current_data: pd.DataFrame) -> Optional[str]:
        """Check RSI-specific exit conditions"""
        
        try:
            # Call parent exit conditions first
            parent_exit = super().should_exit_position(entry_signal, current_price, current_data)
            if parent_exit:
                return parent_exit
            
            # Calculate current RSI
            if len(current_data) >= self.config.rsi_period:
                current_rsi = talib.RSI(current_data['close'].values, 
                                       timeperiod=self.config.rsi_period)[-1]
                
                # RSI-based exit conditions
                if entry_signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                    # Exit long position when RSI returns to neutral or overbought
                    if current_rsi >= self.config.rsi_exit_level:
                        return "RSI_NEUTRAL_EXIT"
                    elif current_rsi >= self.config.overbought_threshold:
                        return "RSI_OVERBOUGHT_EXIT"
                
                elif entry_signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
                    # Exit short position when RSI returns to neutral or oversold
                    if current_rsi <= (100 - self.config.rsi_exit_level):
                        return "RSI_NEUTRAL_EXIT"
                    elif current_rsi <= self.config.oversold_threshold:
                        return "RSI_OVERSOLD_EXIT"
            
            # Trailing stop logic
            if self.config.trailing_stop_enabled and entry_signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                # Implement trailing stop for long positions
                if current_price > entry_signal.entry_price * 1.02:  # Only after 2% profit
                    trailing_stop = current_price * (1 - self.config.trailing_stop_percentage)
                    if current_price <= trailing_stop:
                        return "TRAILING_STOP"
            
            # Maximum holding period
            if entry_signal.timestamp:
                holding_time = datetime.utcnow() - entry_signal.timestamp
                if holding_time > timedelta(hours=self.config.max_holding_period_hours):
                    return "MAX_HOLDING_PERIOD"
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking RSI exit conditions: {e}")
            return None
    
    def get_strategy_summary(self) -> Dict[str, Any]:
        """Get RSI strategy summary and statistics"""
        
        base_summary = self.get_performance_summary()
        
        rsi_specific = {
            'strategy_type': 'RSI Mean Reversion',
            'rsi_period': self.config.rsi_period,
            'oversold_threshold': self.config.oversold_threshold,
            'overbought_threshold': self.config.overbought_threshold,
            'confirmations_enabled': {
                'volume': self.config.volume_confirmation,
                'trend_alignment': self.config.trend_alignment,
                'stochastic': True,
                'macd': True,
                'bollinger_bands': True,
                'support_resistance': True
            },
            'recent_signals_count': len(self.recent_signals),
            'active_positions_count': len(self.active_positions),
            'config': self.config.__dict__
        }
        
        return {**base_summary, **rsi_specific}

    async def optimize_parameters(self, historical_data: pd.DataFrame,
                                symbol: str,
                                optimization_period_days: int = 90) -> Dict[str, Any]:
        """Optimize RSI strategy parameters using historical data"""
        
        try:
            logger.info(f"Starting RSI parameter optimization for {symbol}")
            
            # Parameter ranges to test
            rsi_periods = [10, 14, 18, 21]
            oversold_thresholds = [20, 25, 30, 35]
            overbought_thresholds = [65, 70, 75, 80]
            
            best_params = None
            best_score = -float('inf')
            optimization_results = []
            
            for rsi_period in rsi_periods:
                for oversold in oversold_thresholds:
                    for overbought in overbought_thresholds:
                        if oversold >= overbought - 20:  # Ensure reasonable gap
                            continue
                        
                        # Test parameters
                        test_config = self.config.__dict__.copy()
                        test_config.update({
                            'rsi_period': rsi_period,
                            'oversold_threshold': oversold,
                            'overbought_threshold': overbought
                        })
                        
                        # Create temporary strategy instance
                        temp_strategy = RSIStrategy(f"{self.strategy_id}_temp", test_config)
                        
                        # Backtest with these parameters
                        backtest_result = await self._backtest_parameters(
                            temp_strategy, historical_data, symbol
                        )
                        
                        if backtest_result['success']:
                            score = self._calculate_optimization_score(backtest_result)
                            
                            optimization_results.append({
                                'rsi_period': rsi_period,
                                'oversold_threshold': oversold,
                                'overbought_threshold': overbought,
                                'score': score,
                                'total_return': backtest_result['total_return'],
                                'win_rate': backtest_result['win_rate'],
                                'sharpe_ratio': backtest_result['sharpe_ratio'],
                                'max_drawdown': backtest_result['max_drawdown']
                            })
                            
                            if score > best_score:
                                best_score = score
                                best_params = test_config
            
            if best_params:
                logger.info(f"RSI optimization completed. Best score: {best_score:.2f}")
                return {
                    'success': True,
                    'best_parameters': best_params,
                    'best_score': best_score,
                    'all_results': optimization_results,
                    'optimization_summary': {
                        'total_combinations_tested': len(optimization_results),
                        'best_rsi_period': best_params['rsi_period'],
                        'best_oversold': best_params['oversold_threshold'],
                        'best_overbought': best_params['overbought_threshold']
                    }
                }
            else:
                return {'success': False, 'error': 'No valid parameter combinations found'}
                
        except Exception as e:
            logger.error(f"Error optimizing RSI parameters: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _backtest_parameters(self, strategy: 'RSIStrategy',
                                 historical_data: pd.DataFrame,
                                 symbol: str) -> Dict[str, Any]:
        """Backtest strategy with specific parameters"""
        
        try:
            signals = []
            trades = []
            
            # Slide through historical data
            lookback = strategy.get_lookback_period()
            
            for i in range(lookback, len(historical_data) - 1):
                window_data = historical_data.iloc[:i+1]
                current_price = window_data['close'].iloc[-1]
                
                # Generate signal
                signal = await strategy.analyze_symbol(symbol, window_data, current_price)
                
                if signal and signal.confidence >= strategy.config.min_confidence:
                    signals.append(signal)
                    
                    # Simulate trade execution
                    future_data = historical_data.iloc[i+1:i+25]  # Next 24 hours
                    if not future_data.empty:
                        trade_result = strategy.backtest_signal(signal, future_data, days_ahead=1)
                        if trade_result['success']:
                            trades.append({
                                'entry_time': window_data.index[-1],
                                'exit_time': future_data.index[0] if len(future_data) > 0 else window_data.index[-1],
                                'pnl': trade_result['final_pnl'],
                                'signal_confidence': signal.confidence
                            })
            
            if not trades:
                return {'success': False, 'error': 'No trades generated'}
            
            # Calculate performance metrics
            total_trades = len(trades)
            winning_trades = len([t for t in trades if t['pnl'] > 0])
            total_return = sum(t['pnl'] for t in trades)
            
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            
            # Calculate Sharpe ratio (simplified)
            returns = [t['pnl'] for t in trades]
            sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
            
            # Calculate max drawdown
            cumulative_returns = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = cumulative_returns - running_max
            max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0
            
            return {
                'success': True,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'total_return': total_return,
                'win_rate': win_rate,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'avg_trade_return': np.mean(returns),
                'trades': trades
            }
            
        except Exception as e:
            logger.error(f"Error backtesting parameters: {e}")
            return {'success': False, 'error': str(e)}
    
    def _calculate_optimization_score(self, backtest_result: Dict[str, Any]) -> float:
        """Calculate optimization score based on multiple factors"""
        
        try:
            # Weighted scoring system
            weights = {
                'total_return': 0.3,
                'win_rate': 0.25,
                'sharpe_ratio': 0.25,
                'max_drawdown': 0.2  # Negative impact
            }
            
            score = 0.0
            
            # Total return score (normalized)
            total_return = backtest_result.get('total_return', 0)
            return_score = min(total_return * 100, 100)  # Cap at 100%
            score += return_score * weights['total_return']
            
            # Win rate score
            win_rate = backtest_result.get('win_rate', 0)
            score += win_rate * weights['win_rate']
            
            # Sharpe ratio score (normalized to 0-100)
            sharpe_ratio = backtest_result.get('sharpe_ratio', 0)
            sharpe_score = min(max(sharpe_ratio * 25, 0), 100)  # Convert to 0-100 scale
            score += sharpe_score * weights['sharpe_ratio']
            
            # Max drawdown penalty
            max_drawdown = abs(backtest_result.get('max_drawdown', 0))
            drawdown_penalty = min(max_drawdown * 200, 100)  # Convert to penalty
            score -= drawdown_penalty * weights['max_drawdown']
            
            return max(score, 0)  # Ensure non-negative score
            
        except Exception as e:
            logger.error(f"Error calculating optimization score: {e}")
            return 0.0

# Factory function for creating RSI strategy instances
def create_rsi_strategy(config: Dict[str, Any] = None) -> RSIStrategy:
    """Factory function to create RSI strategy instance"""
    return RSIStrategy("rsi_strategy", config)

# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_rsi_strategy():
        print("📈 RSI Strategy Test")
        print("=" * 50)
        
        # Create RSI strategy with custom config
        config = {
            'enabled': True,
            'rsi_period': 14,
            'oversold_threshold': 30,
            'overbought_threshold': 70,
            'volume_confirmation': True,
            'trend_alignment': False,
            'min_confidence': 65.0
        }
        
        strategy = create_rsi_strategy(config)
        await strategy.initialize()
        
        print(f"Strategy ID: {strategy.strategy_id}")
        print(f"Required indicators: {strategy.get_required_indicators()}")
        print(f"Lookback period: {strategy.get_lookback_period()}")
        
        # Generate sample OHLCV data
        np.random.seed(42)  # For reproducible results
        dates = pd.date_range('2024-01-01', periods=100, freq='1H')
        
        # Create realistic price movement with some oversold/overbought conditions
        base_price = 50000
        price_changes = np.random.normal(0, 200, 100)
        prices = [base_price]
        
        for change in price_changes[1:]:
            new_price = prices[-1] + change
            prices.append(max(new_price, base_price * 0.8))  # Floor price
        
        ohlcv_data = pd.DataFrame({
            'timestamp': dates,
            'open': [p + np.random.normal(0, 50) for p in prices],
            'high': [p + abs(np.random.normal(100, 50)) for p in prices],
            'low': [p - abs(np.random.normal(100, 50)) for p in prices],
            'close': prices,
            'volume': np.random.lognormal(15, 0.5, 100)
        })
        
        # Ensure OHLC relationship
        for i in range(len(ohlcv_data)):
            row = ohlcv_data.iloc[i]
            high = max(row['open'], row['close']) + abs(np.random.normal(0, 20))
            low = min(row['open'], row['close']) - abs(np.random.normal(0, 20))
            ohlcv_data.loc[i, 'high'] = high
            ohlcv_data.loc[i, 'low'] = low
        
        current_price = ohlcv_data['close'].iloc[-1]
        
        print(f"\nCurrent price: ${current_price:.2f}")
        
        # Test signal generation
        try:
            signal = await strategy.analyze_symbol(
                'BTC/USDT', 
                ohlcv_data, 
                current_price,
                {'portfolio_value': 10000}
            )
            
            if signal:
                print(f"\n🎯 Signal Generated:")
                print(f"  Type: {signal.signal_type.value}")
                print(f"  Confidence: {signal.confidence:.1f}%")
                print(f"  Entry Price: ${signal.entry_price:.2f}")
                print(f"  Stop Loss: ${signal.stop_loss:.2f}")
                print(f"  Take Profit: ${signal.take_profit:.2f}")
                print(f"  R/R Ratio: {signal.risk_reward_ratio:.2f}")
                print(f"  Position Size: {signal.position_size_recommendation:.6f}")
                print(f"  Market Condition: {signal.market_condition}")
                print(f"  Volume Confirmed: {signal.volume_confirmed}")
                print(f"  Reasoning: {signal.reasoning}")
                
                # Test signal validation
                market_data = {'volume_24h': 100_000_000}
                is_valid = await strategy.validate_signal(signal, market_data)
                print(f"  Signal Valid: {is_valid}")
                
                # Test backtest
                future_data = ohlcv_data.tail(10)  # Use last 10 periods as "future"
                backtest_result = strategy.backtest_signal(signal, future_data)
                
                if backtest_result['success']:
                    print(f"\n📊 Backtest Result:")
                    print(f"  Final P&L: {backtest_result['final_pnl']:.2%}")
                    print(f"  Max Profit: {backtest_result['max_profit']:.2%}")
                    print(f"  Max Loss: {backtest_result['max_loss']:.2%}")
                    print(f"  Exit Reason: {backtest_result['exit_reason']}")
                    
                    # Update performance metrics
                    strategy.update_performance_metrics(
                        backtest_result['signal_successful'],
                        signal.confidence
                    )
            else:
                print("\n❌ No signal generated")
                
                # Calculate and display current RSI for debugging
                rsi_values = talib.RSI(ohlcv_data['close'].values, timeperiod=14)
                current_rsi = rsi_values[-1]
                print(f"Current RSI: {current_rsi:.1f}")
                print(f"RSI Threshold: Oversold < {strategy.config.oversold_threshold}, Overbought > {strategy.config.overbought_threshold}")
        
        except Exception as e:
            print(f"❌ Error testing strategy: {e}")
        
        # Display strategy summary
        summary = strategy.get_strategy_summary()
        print(f"\n📊 Strategy Summary:")
        print(f"  Signals Generated: {summary['signals_generated']}")
        print(f"  Success Rate: {summary.get('success_rate', 0):.1f}%")
        print(f"  Average Confidence: {summary['avg_confidence']:.1f}")
        
        print("\n✅ RSI Strategy test completed!")
    
    # Run the test
    asyncio.run(test_rsi_strategy())