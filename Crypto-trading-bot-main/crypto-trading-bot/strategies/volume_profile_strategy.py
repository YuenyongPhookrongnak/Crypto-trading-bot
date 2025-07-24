"""
Volume Profile Breakout Trading Strategy

This strategy identifies high-probability breakout opportunities using Volume Profile analysis,
Point of Control (POC), Value Area, and volume-based confirmation signals.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict

import pandas as pd
import numpy as np
import talib

from strategies.base_strategy import (
    BaseStrategy, TradingSignal, SignalType, MarketCondition,
    StrategyConfig, StrategyValidationMixin, StrategyBacktestMixin
)

logger = logging.getLogger(__name__)

@dataclass
class VolumeProfileConfig(StrategyConfig):
    """Volume Profile Strategy specific configuration"""
    
    # Volume Profile Parameters
    lookback_period: int = 24  # Hours of data for volume profile
    price_bins: int = 50  # Number of price bins for profile
    value_area_percentage: float = 0.7  # 70% of volume defines value area
    
    # Point of Control (POC) Parameters
    poc_deviation_threshold: float = 0.002  # 0.2% deviation from POC
    poc_strength_minimum: float = 0.05  # Min 5% of total volume at POC
    poc_breakout_confirmation: int = 2  # Bars to confirm POC breakout
    
    # Volume Analysis
    volume_multiplier: float = 1.5  # Volume must be 1.5x average
    volume_spike_threshold: float = 2.0  # 2x volume spike for strong signals
    min_volume_for_signal: float = 1000000  # $1M minimum volume
    
    # Breakout Parameters
    breakout_confirmation_bars: int = 2  # Bars to confirm breakout
    false_breakout_filter: bool = True
    breakout_pullback_allowed: float = 0.005  # 0.5% pullback allowed
    
    # High Volume Node (HVN) and Low Volume Node (LVN) Detection
    hvn_threshold_percentile: float = 0.8  # Top 20% volume levels
    lvn_threshold_percentile: float = 0.2  # Bottom 20% volume levels
    node_proximity_threshold: float = 0.01  # 1% proximity for node clustering
    
    # Signal Strength Parameters
    strong_breakout_volume_multiplier: float = 3.0  # 3x volume for strong signals
    institutional_volume_threshold: float = 10000000  # $10M for institutional activity
    
    # Risk Management
    atr_stop_loss_multiplier: float = 2.5
    value_area_stop_loss: bool = True  # Use VA boundaries as stop loss
    breakout_failure_stop: bool = True  # Quick stop if breakout fails

class VolumeNode:
    """Represents a volume node in the profile"""
    
    def __init__(self, price_level: float, volume: float, volume_percentage: float):
        self.price_level = price_level
        self.volume = volume
        self.volume_percentage = volume_percentage
        self.node_type = None  # Will be set as HVN, LVN, or POC
        
    def __repr__(self):
        return f"VolumeNode(price={self.price_level:.2f}, volume={self.volume:.0f}, " \
               f"percentage={self.volume_percentage:.2%}, type={self.node_type})"

class VolumeProfile:
    """Volume Profile calculation and analysis"""
    
    def __init__(self, ohlcv_data: pd.DataFrame, price_bins: int = 50):
        self.ohlcv_data = ohlcv_data
        self.price_bins = price_bins
        self.volume_nodes = []
        self.poc = None  # Point of Control
        self.value_area_high = None
        self.value_area_low = None
        self.total_volume = 0
        
        self._calculate_profile()
    
    def _calculate_profile(self):
        """Calculate the volume profile"""
        try:
            if self.ohlcv_data.empty:
                return
            
            # Get price range
            min_price = self.ohlcv_data['low'].min()
            max_price = self.ohlcv_data['high'].max()
            
            # Create price bins
            price_levels = np.linspace(min_price, max_price, self.price_bins + 1)
            volume_at_price = np.zeros(self.price_bins)
            
            # Distribute volume across price levels for each bar
            for _, row in self.ohlcv_data.iterrows():
                bar_volume = row['volume']
                bar_high = row['high']
                bar_low = row['low']
                
                # Simple assumption: volume distributed evenly across price range
                # In reality, you might use tick data or more sophisticated methods
                for i in range(self.price_bins):
                    if price_levels[i] >= bar_low and price_levels[i] <= bar_high:
                        volume_at_price[i] += bar_volume / self.price_bins
            
            self.total_volume = volume_at_price.sum()
            
            # Create volume nodes
            for i in range(self.price_bins):
                if volume_at_price[i] > 0:
                    volume_percentage = volume_at_price[i] / self.total_volume
                    node = VolumeNode(
                        price_level=(price_levels[i] + price_levels[i+1]) / 2,
                        volume=volume_at_price[i],
                        volume_percentage=volume_percentage
                    )
                    self.volume_nodes.append(node)
            
            # Sort by volume descending
            self.volume_nodes.sort(key=lambda x: x.volume, reverse=True)
            
            # Identify POC (highest volume node)
            if self.volume_nodes:
                self.poc = self.volume_nodes[0]
                self.poc.node_type = 'POC'
            
            # Calculate Value Area
            self._calculate_value_area()
            self._classify_nodes()
            
        except Exception as e:
            logger.error(f"Error calculating volume profile: {e}")
    
    def _calculate_value_area(self, va_percentage: float = 0.7):
        """Calculate Value Area (70% of volume by default)"""
        try:
            if not self.volume_nodes:
                return
            
            target_volume = self.total_volume * va_percentage
            accumulated_volume = 0
            va_nodes = []
            
            # Start from POC and expand up/down alternately
            poc_index = 0
            sorted_by_price = sorted(self.volume_nodes, key=lambda x: x.price_level)
            
            # Find POC index in price-sorted list
            for i, node in enumerate(sorted_by_price):
                if node == self.poc:
                    poc_index = i
                    break
            
            va_nodes.append(sorted_by_price[poc_index])
            accumulated_volume += sorted_by_price[poc_index].volume
            
            # Expand value area
            up_index = poc_index + 1
            down_index = poc_index - 1
            
            while accumulated_volume < target_volume:
                up_volume = sorted_by_price[up_index].volume if up_index < len(sorted_by_price) else 0
                down_volume = sorted_by_price[down_index].volume if down_index >= 0 else 0
                
                if up_volume >= down_volume and up_index < len(sorted_by_price):
                    va_nodes.append(sorted_by_price[up_index])
                    accumulated_volume += up_volume
                    up_index += 1
                elif down_index >= 0:
                    va_nodes.append(sorted_by_price[down_index])
                    accumulated_volume += down_volume
                    down_index -= 1
                else:
                    break
            
            # Set value area boundaries
            if va_nodes:
                va_prices = [node.price_level for node in va_nodes]
                self.value_area_high = max(va_prices)
                self.value_area_low = min(va_prices)
            
        except Exception as e:
            logger.error(f"Error calculating value area: {e}")
    
    def _classify_nodes(self):
        """Classify nodes as HVN or LVN"""
        try:
            if not self.volume_nodes:
                return
            
            volumes = [node.volume for node in self.volume_nodes]
            hvn_threshold = np.percentile(volumes, 80)
            lvn_threshold = np.percentile(volumes, 20)
            
            for node in self.volume_nodes:
                if node.node_type == 'POC':
                    continue
                elif node.volume >= hvn_threshold:
                    node.node_type = 'HVN'
                elif node.volume <= lvn_threshold:
                    node.node_type = 'LVN'
                else:
                    node.node_type = 'NORMAL'
                    
        except Exception as e:
            logger.error(f"Error classifying nodes: {e}")
    
    def get_nearby_nodes(self, price: float, proximity_threshold: float = 0.01) -> List[VolumeNode]:
        """Get volume nodes near a specific price"""
        nearby_nodes = []
        for node in self.volume_nodes:
            price_diff = abs(node.price_level - price) / price
            if price_diff <= proximity_threshold:
                nearby_nodes.append(node)
        return nearby_nodes
    
    def is_at_significant_level(self, price: float, threshold: float = 0.005) -> Dict[str, Any]:
        """Check if price is at a significant volume level"""
        result = {
            'at_poc': False,
            'at_value_area_boundary': False,
            'at_hvn': False,
            'at_lvn': False,
            'nearby_nodes': []
        }
        
        try:
            # Check POC proximity
            if self.poc:
                poc_diff = abs(price - self.poc.price_level) / price
                if poc_diff <= threshold:
                    result['at_poc'] = True
            
            # Check Value Area boundaries
            if self.value_area_high and self.value_area_low:
                va_high_diff = abs(price - self.value_area_high) / price
                va_low_diff = abs(price - self.value_area_low) / price
                if va_high_diff <= threshold or va_low_diff <= threshold:
                    result['at_value_area_boundary'] = True
            
            # Check HVN/LVN proximity
            nearby_nodes = self.get_nearby_nodes(price, threshold)
            result['nearby_nodes'] = nearby_nodes
            
            for node in nearby_nodes:
                if node.node_type == 'HVN':
                    result['at_hvn'] = True
                elif node.node_type == 'LVN':
                    result['at_lvn'] = True
            
            return result
            
        except Exception as e:
            logger.error(f"Error checking significant levels: {e}")
            return result

class VolumeProfileStrategy(BaseStrategy, StrategyValidationMixin, StrategyBacktestMixin):
    """Volume Profile Breakout Strategy Implementation"""
    
    def __init__(self, strategy_id: str = "volume_profile_strategy", config: Dict[str, Any] = None):
        super().__init__(strategy_id, config)
        self.config = VolumeProfileConfig()
        
        if config:
            self.config.update_from_dict(config)
        
        # Strategy state
        self.volume_profiles = {}  # Store profiles per symbol
        self.recent_breakouts = {}  # Track recent breakout attempts
        self.institutional_activity = {}  # Track high-volume activity
        
        logger.info(f"Volume Profile Strategy initialized with config: {self.config.__dict__}")
    
    def get_required_indicators(self) -> List[str]:
        """Return list of required technical indicators"""
        return [
            'sma_20', 'sma_50', 'ema_9', 'atr', 'obv', 'ad', 'rsi',
            'bb_upper', 'bb_lower', 'bb_middle', 'macd', 'macd_signal'
        ]
    
    def get_lookback_period(self) -> int:
        """Return minimum lookback period required for analysis"""
        return max(self.config.lookback_period * 2, 100)
    
    async def analyze_symbol(self, 
                           symbol: str, 
                           ohlcv_data: pd.DataFrame,
                           current_price: float,
                           additional_data: Dict[str, Any] = None) -> Optional[TradingSignal]:
        """Main analysis method for Volume Profile strategy"""
        
        try:
            # Validate input data
            if not self.validate_data_quality(ohlcv_data):
                logger.warning(f"Invalid data quality for {symbol}")
                return None
            
            if len(ohlcv_data) < self.get_lookback_period():
                logger.debug(f"Insufficient data for {symbol}: {len(ohlcv_data)} < {self.get_lookback_period()}")
                return None
            
            # Calculate volume profile
            lookback_data = ohlcv_data.tail(self.config.lookback_period)
            volume_profile = VolumeProfile(lookback_data, self.config.price_bins)
            self.volume_profiles[symbol] = volume_profile
            
            # Calculate technical indicators
            indicators = self.calculate_technical_indicators(ohlcv_data)
            if not indicators:
                logger.warning(f"Failed to calculate indicators for {symbol}")
                return None
            
            # Detect market condition
            market_condition = self.detect_market_condition(ohlcv_data)
            
            # Analyze volume characteristics
            volume_analysis = self._analyze_volume_characteristics(ohlcv_data, symbol)
            
            # Generate trading signal
            signal = await self._generate_volume_profile_signal(
                symbol, ohlcv_data, current_price, volume_profile, 
                indicators, market_condition, volume_analysis
            )
            
            if signal:
                # Enhance signal with confirmations
                signal = await self._enhance_signal_with_volume_analysis(
                    signal, ohlcv_data, volume_profile, indicators, volume_analysis
                )
                
                # Calculate position sizing
                signal.position_size_recommendation = self.calculate_position_size(
                    signal,
                    portfolio_value=additional_data.get('portfolio_value', 10000),
                    current_price=current_price
                )
                
                logger.info(f"Volume Profile signal generated for {symbol}: {signal.signal_type.value} "
                           f"(Confidence: {signal.confidence:.1f}%, Volume: {volume_analysis['current_volume_ratio']:.1f}x)")
                
                return signal
            
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol} with Volume Profile strategy: {e}")
            return None
    
    def _analyze_volume_characteristics(self, ohlcv_data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Analyze volume characteristics and patterns"""
        try:
            # Current volume metrics
            current_volume = ohlcv_data['volume'].iloc[-1]
            avg_volume_20 = ohlcv_data['volume'].rolling(20).mean().iloc[-1]
            avg_volume_5 = ohlcv_data['volume'].rolling(5).mean().iloc[-1]
            
            volume_ratio_20 = current_volume / avg_volume_20 if avg_volume_20 > 0 else 0
            volume_ratio_5 = current_volume / avg_volume_5 if avg_volume_5 > 0 else 0
            
            # Volume trend analysis
            volume_trend = self._calculate_volume_trend(ohlcv_data)
            
            # Price-volume relationship
            price_volume_correlation = self._calculate_price_volume_correlation(ohlcv_data)
            
            # Accumulation/Distribution analysis
            recent_ad = ohlcv_data.apply(
                lambda row: ((row['close'] - row['low']) - (row['high'] - row['close'])) / 
                           (row['high'] - row['low']) * row['volume'] 
                           if row['high'] != row['low'] else 0, axis=1
            ).tail(10).sum()
            
            # Institutional activity detection
            institutional_threshold = self.config.institutional_volume_threshold
            institutional_activity = current_volume >= institutional_threshold
            
            # Volume spike detection
            volume_spike = volume_ratio_20 >= self.config.volume_spike_threshold
            
            # On-Balance Volume momentum
            obv_values = self._calculate_obv(ohlcv_data)
            obv_trend = 'up' if obv_values[-1] > obv_values[-5] else 'down' if len(obv_values) > 5 else 'neutral'
            
            return {
                'current_volume': current_volume,
                'avg_volume_20': avg_volume_20,
                'current_volume_ratio': volume_ratio_20,
                'volume_ratio_5': volume_ratio_5,
                'volume_trend': volume_trend,
                'price_volume_correlation': price_volume_correlation,
                'recent_ad': recent_ad,
                'institutional_activity': institutional_activity,
                'volume_spike': volume_spike,
                'obv_trend': obv_trend,
                'volume_strength': self._classify_volume_strength(volume_ratio_20)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing volume characteristics: {e}")
            return {}
    
    def _calculate_volume_trend(self, ohlcv_data: pd.DataFrame, period: int = 10) -> str:
        """Calculate volume trend direction"""
        try:
            recent_volumes = ohlcv_data['volume'].tail(period)
            if len(recent_volumes) < period:
                return 'neutral'
            
            # Linear regression to determine trend
            x = np.arange(len(recent_volumes))
            slope = np.polyfit(x, recent_volumes.values, 1)[0]
            
            if slope > recent_volumes.mean() * 0.1:
                return 'increasing'
            elif slope < -recent_volumes.mean() * 0.1:
                return 'decreasing'
            else:
                return 'stable'
                
        except Exception as e:
            logger.error(f"Error calculating volume trend: {e}")
            return 'neutral'
    
    def _calculate_price_volume_correlation(self, ohlcv_data: pd.DataFrame, period: int = 20) -> float:
        """Calculate price-volume correlation"""
        try:
            if len(ohlcv_data) < period:
                return 0.0
            
            price_changes = ohlcv_data['close'].pct_change().tail(period)
            volume_changes = ohlcv_data['volume'].pct_change().tail(period)
            
            correlation = price_changes.corr(volume_changes)
            return correlation if not np.isnan(correlation) else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating price-volume correlation: {e}")
            return 0.0
    
    def _calculate_obv(self, ohlcv_data: pd.DataFrame) -> np.ndarray:
        """Calculate On-Balance Volume"""
        try:
            return talib.OBV(ohlcv_data['close'].values, ohlcv_data['volume'].values)
        except Exception as e:
            logger.error(f"Error calculating OBV: {e}")
            return np.zeros(len(ohlcv_data))
    
    def _classify_volume_strength(self, volume_ratio: float) -> str:
        """Classify volume strength based on ratio"""
        if volume_ratio >= 3.0:
            return 'exceptional'
        elif volume_ratio >= 2.0:
            return 'strong'
        elif volume_ratio >= 1.5:
            return 'above_average'
        elif volume_ratio >= 1.0:
            return 'average'
        else:
            return 'weak'
    
    async def _generate_volume_profile_signal(self,
                                            symbol: str,
                                            ohlcv_data: pd.DataFrame,
                                            current_price: float,
                                            volume_profile: VolumeProfile,
                                            indicators: Dict[str, Any],
                                            market_condition: MarketCondition,
                                            volume_analysis: Dict[str, Any]) -> Optional[TradingSignal]:
        """Generate volume profile-based trading signal"""
        
        try:
            # Check minimum volume requirement
            if volume_analysis['current_volume'] < self.config.min_volume_for_signal:
                return None
            
            # Check if volume is sufficient for signal generation
            if volume_analysis['current_volume_ratio'] < self.config.volume_multiplier:
                return None
            
            signal_type = None
            base_confidence = 0.0
            reasoning = []
            
            # Analyze current price position relative to volume profile
            level_analysis = volume_profile.is_at_significant_level(
                current_price, self.config.poc_deviation_threshold
            )
            
            # POC Breakout Analysis
            poc_signal = self._analyze_poc_breakout(
                current_price, volume_profile, ohlcv_data, volume_analysis
            )
            
            if poc_signal['signal']:
                signal_type = poc_signal['signal_type']
                base_confidence = poc_signal['confidence']
                reasoning.extend(poc_signal['reasoning'])
            
            # Value Area Breakout Analysis  
            if not signal_type:
                va_signal = self._analyze_value_area_breakout(
                    current_price, volume_profile, ohlcv_data, volume_analysis
                )
                
                if va_signal['signal']:
                    signal_type = va_signal['signal_type']
                    base_confidence = va_signal['confidence']
                    reasoning.extend(va_signal['reasoning'])
            
            # HVN/LVN Analysis
            if not signal_type:
                node_signal = self._analyze_hvn_lvn_interaction(
                    current_price, volume_profile, ohlcv_data, volume_analysis
                )
                
                if node_signal['signal']:
                    signal_type = node_signal['signal_type']
                    base_confidence = node_signal['confidence']
                    reasoning.extend(node_signal['reasoning'])
            
            if not signal_type:
                return None
            
            # Volume strength bonus
            volume_strength_bonus = self._calculate_volume_strength_bonus(volume_analysis)
            base_confidence += volume_strength_bonus
            
            # Create signal
            signal = TradingSignal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                signal_type=signal_type,
                confidence=min(base_confidence, 95.0),
                entry_price=current_price,
                market_condition=market_condition,
                technical_indicators={
                    'poc_price': volume_profile.poc.price_level if volume_profile.poc else None,
                    'value_area_high': volume_profile.value_area_high,
                    'value_area_low': volume_profile.value_area_low,
                    'current_volume_ratio': volume_analysis['current_volume_ratio'],
                    'volume_strength': volume_analysis['volume_strength'],
                    'obv_trend': volume_analysis['obv_trend']
                },
                reasoning='; '.join(reasoning)
            )
            
            # Calculate stop loss and take profit
            signal.stop_loss = self._calculate_volume_stop_loss(signal, volume_profile, indicators, ohlcv_data)
            signal.take_profit = self._calculate_volume_take_profit(signal, volume_profile, indicators, ohlcv_data)
            
            # Calculate risk/reward ratio
            if signal.stop_loss and signal.take_profit:
                risk = abs(signal.entry_price - signal.stop_loss)
                reward = abs(signal.take_profit - signal.entry_price)
                signal.risk_reward_ratio = reward / risk if risk > 0 else None
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating volume profile signal: {e}")
            return None
    
    def _analyze_poc_breakout(self,
                            current_price: float,
                            volume_profile: VolumeProfile,
                            ohlcv_data: pd.DataFrame,
                            volume_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze Point of Control breakout scenarios"""
        
        result = {'signal': False, 'signal_type': None, 'confidence': 0.0, 'reasoning': []}
        
        try:
            if not volume_profile.poc:
                return result
            
            poc_price = volume_profile.poc.price_level
            price_diff = (current_price - poc_price) / poc_price
            
            # Check for POC breakout
            if abs(price_diff) > self.config.poc_deviation_threshold:
                
                # Bullish POC breakout
                if price_diff > 0:
                    # Price above POC with strong volume
                    if volume_analysis['current_volume_ratio'] >= self.config.volume_multiplier:
                        result['signal'] = True
                        result['signal_type'] = SignalType.STRONG_BUY if volume_analysis['current_volume_ratio'] >= self.config.strong_breakout_volume_multiplier else SignalType.BUY
                        result['confidence'] = 75.0
                        result['reasoning'].append(f"Bullish POC breakout: Price ${current_price:.2f} above POC ${poc_price:.2f}")
                        result['reasoning'].append(f"Volume confirmation: {volume_analysis['current_volume_ratio']:.1f}x average")
                
                # Bearish POC breakout
                elif price_diff < 0:
                    # Price below POC with strong volume
                    if volume_analysis['current_volume_ratio'] >= self.config.volume_multiplier:
                        result['signal'] = True
                        result['signal_type'] = SignalType.STRONG_SELL if volume_analysis['current_volume_ratio'] >= self.config.strong_breakout_volume_multiplier else SignalType.SELL
                        result['confidence'] = 75.0
                        result['reasoning'].append(f"Bearish POC breakout: Price ${current_price:.2f} below POC ${poc_price:.2f}")
                        result['reasoning'].append(f"Volume confirmation: {volume_analysis['current_volume_ratio']:.1f}x average")
            
            # POC rejection (reversal signal)
            elif abs(price_diff) <= self.config.poc_deviation_threshold / 2:
                # Price near POC with high volume might indicate rejection
                if volume_analysis['current_volume_ratio'] >= self.config.volume_spike_threshold:
                    
                    # Determine rejection direction based on recent price action
                    recent_trend = self._get_recent_price_trend(ohlcv_data, periods=3)
                    
                    if recent_trend == 'down' and current_price <= poc_price:
                        # Potential bullish rejection at POC
                        result['signal'] = True
                        result['signal_type'] = SignalType.BUY
                        result['confidence'] = 65.0
                        result['reasoning'].append(f"Bullish POC rejection: High volume at POC support ${poc_price:.2f}")
                        
                    elif recent_trend == 'up' and current_price >= poc_price:
                        # Potential bearish rejection at POC
                        result['signal'] = True
                        result['signal_type'] = SignalType.SELL
                        result['confidence'] = 65.0
                        result['reasoning'].append(f"Bearish POC rejection: High volume at POC resistance ${poc_price:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing POC breakout: {e}")
            return result
    
    def _analyze_value_area_breakout(self,
                                   current_price: float,
                                   volume_profile: VolumeProfile,
                                   ohlcv_data: pd.DataFrame,
                                   volume_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze Value Area breakout scenarios"""
        
        result = {'signal': False, 'signal_type': None, 'confidence': 0.0, 'reasoning': []}
        
        try:
            if not volume_profile.value_area_high or not volume_profile.value_area_low:
                return result
            
            va_high = volume_profile.value_area_high
            va_low = volume_profile.value_area_low
            
            # Check for Value Area breakouts
            if current_price > va_high:
                # Bullish Value Area breakout
                if volume_analysis['current_volume_ratio'] >= self.config.volume_multiplier:
                    distance_from_va = (current_price - va_high) / va_high
                    
                    if distance_from_va > 0.002:  # 0.2% above VA high
                        result['signal'] = True
                        result['signal_type'] = SignalType.BUY
                        result['confidence'] = 70.0
                        result['reasoning'].append(f"Bullish VA breakout: Price ${current_price:.2f} above VA high ${va_high:.2f}")
                        result['reasoning'].append(f"Volume strength: {volume_analysis['volume_strength']}")
                        
                        # Higher confidence for strong volume
                        if volume_analysis['current_volume_ratio'] >= self.config.strong_breakout_volume_multiplier:
                            result['signal_type'] = SignalType.STRONG_BUY
                            result['confidence'] = 80.0
            
            elif current_price < va_low:
                # Bearish Value Area breakout
                if volume_analysis['current_volume_ratio'] >= self.config.volume_multiplier:
                    distance_from_va = (va_low - current_price) / va_low
                    
                    if distance_from_va > 0.002:  # 0.2% below VA low
                        result['signal'] = True
                        result['signal_type'] = SignalType.SELL
                        result['confidence'] = 70.0
                        result['reasoning'].append(f"Bearish VA breakout: Price ${current_price:.2f} below VA low ${va_low:.2f}")
                        result['reasoning'].append(f"Volume strength: {volume_analysis['volume_strength']}")
                        
                        # Higher confidence for strong volume
                        if volume_analysis['current_volume_ratio'] >= self.config.strong_breakout_volume_multiplier:
                            result['signal_type'] = SignalType.STRONG_SELL
                            result['confidence'] = 80.0
            
            # Value Area rejection signals
            else:
                # Price within Value Area - look for rejection at boundaries
                va_high_distance = abs(current_price - va_high) / current_price
                va_low_distance = abs(current_price - va_low) / current_price
                
                # Near VA high with high volume (potential rejection)
                if va_high_distance <= 0.005 and volume_analysis['current_volume_ratio'] >= self.config.volume_spike_threshold:
                    recent_trend = self._get_recent_price_trend(ohlcv_data, periods=3)
                    if recent_trend == 'up':
                        result['signal'] = True
                        result['signal_type'] = SignalType.SELL
                        result['confidence'] = 60.0
                        result['reasoning'].append(f"VA high rejection: High volume near resistance ${va_high:.2f}")
                
                # Near VA low with high volume (potential rejection)
                elif va_low_distance <= 0.005 and volume_analysis['current_volume_ratio'] >= self.config.volume_spike_threshold:
                    recent_trend = self._get_recent_price_trend(ohlcv_data, periods=3)
                    if recent_trend == 'down':
                        result['signal'] = True
                        result['signal_type'] = SignalType.BUY
                        result['confidence'] = 60.0
                        result['reasoning'].append(f"VA low rejection: High volume near support ${va_low:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing Value Area breakout: {e}")
            return result
    
    def _analyze_hvn_lvn_interaction(self,
                                   current_price: float,
                                   volume_profile: VolumeProfile,
                                   ohlcv_data: pd.DataFrame,
                                   volume_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze High Volume Node (HVN) and Low Volume Node (LVN) interactions"""
        
        result = {'signal': False, 'signal_type': None, 'confidence': 0.0, 'reasoning': []}
        
        try:
            nearby_nodes = volume_profile.get_nearby_nodes(current_price, self.config.node_proximity_threshold)
            
            if not nearby_nodes:
                return result
            
            # Analyze node types and volume characteristics
            hvn_nodes = [node for node in nearby_nodes if node.node_type == 'HVN']
            lvn_nodes = [node for node in nearby_nodes if node.node_type == 'LVN']
            
            # HVN Analysis (High Volume Nodes act as support/resistance)
            if hvn_nodes and volume_analysis['current_volume_ratio'] >= self.config.volume_multiplier:
                strongest_hvn = max(hvn_nodes, key=lambda x: x.volume)
                price_vs_hvn = current_price - strongest_hvn.price_level
                
                # Price bouncing off HVN support
                if price_vs_hvn >= 0 and abs(price_vs_hvn) / current_price <= 0.005:
                    recent_trend = self._get_recent_price_trend(ohlcv_data, periods=5)
                    if recent_trend == 'down':
                        result['signal'] = True
                        result['signal_type'] = SignalType.BUY
                        result['confidence'] = 65.0
                        result['reasoning'].append(f"HVN support bounce: Price holding above HVN ${strongest_hvn.price_level:.2f}")
                        result['reasoning'].append(f"HVN volume strength: {strongest_hvn.volume_percentage:.1%} of total volume")
                
                # Price rejected at HVN resistance
                elif price_vs_hvn <= 0 and abs(price_vs_hvn) / current_price <= 0.005:
                    recent_trend = self._get_recent_price_trend(ohlcv_data, periods=5)
                    if recent_trend == 'up':
                        result['signal'] = True
                        result['signal_type'] = SignalType.SELL
                        result['confidence'] = 65.0
                        result['reasoning'].append(f"HVN resistance rejection: Price failing at HVN ${strongest_hvn.price_level:.2f}")
                        result['reasoning'].append(f"HVN volume strength: {strongest_hvn.volume_percentage:.1%} of total volume")
            
            # LVN Analysis (Low Volume Nodes are areas of low acceptance - breakout potential)
            elif lvn_nodes and volume_analysis['current_volume_ratio'] >= self.config.volume_spike_threshold:
                # LVNs with high current volume suggest breakout potential
                strongest_lvn = min(lvn_nodes, key=lambda x: x.volume)  # Lowest volume = strongest LVN
                
                result['signal'] = True
                result['confidence'] = 55.0
                result['reasoning'].append(f"LVN breakout potential: High volume at low acceptance area ${strongest_lvn.price_level:.2f}")
                result['reasoning'].append(f"LVN weakness: Only {strongest_lvn.volume_percentage:.1%} of total volume")
                
                # Determine breakout direction based on price momentum
                price_momentum = self._calculate_price_momentum(ohlcv_data, periods=3)
                if price_momentum > 0.001:  # 0.1% positive momentum
                    result['signal_type'] = SignalType.BUY
                    result['reasoning'].append("Bullish momentum through LVN")
                elif price_momentum < -0.001:  # 0.1% negative momentum
                    result['signal_type'] = SignalType.SELL
                    result['reasoning'].append("Bearish momentum through LVN")
                else:
                    result['signal'] = False  # No clear direction
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing HVN/LVN interaction: {e}")
            return result
    
    def _get_recent_price_trend(self, ohlcv_data: pd.DataFrame, periods: int = 3) -> str:
        """Determine recent price trend direction"""
        try:
            if len(ohlcv_data) < periods + 1:
                return 'neutral'
            
            recent_closes = ohlcv_data['close'].tail(periods + 1)
            if recent_closes.iloc[-1] > recent_closes.iloc[0]:
                return 'up'
            elif recent_closes.iloc[-1] < recent_closes.iloc[0]:
                return 'down'
            else:
                return 'neutral'
                
        except Exception as e:
            logger.error(f"Error getting recent price trend: {e}")
            return 'neutral'
    
    def _calculate_price_momentum(self, ohlcv_data: pd.DataFrame, periods: int = 3) -> float:
        """Calculate price momentum over specified periods"""
        try:
            if len(ohlcv_data) < periods + 1:
                return 0.0
            
            recent_closes = ohlcv_data['close'].tail(periods + 1)
            return (recent_closes.iloc[-1] - recent_closes.iloc[0]) / recent_closes.iloc[0]
            
        except Exception as e:
            logger.error(f"Error calculating price momentum: {e}")
            return 0.0
    
    def _calculate_volume_strength_bonus(self, volume_analysis: Dict[str, Any]) -> float:
        """Calculate confidence bonus based on volume strength"""
        try:
            volume_ratio = volume_analysis['current_volume_ratio']
            volume_strength = volume_analysis['volume_strength']
            
            # Base volume bonus
            if volume_ratio >= 3.0:
                volume_bonus = 15.0
            elif volume_ratio >= 2.0:
                volume_bonus = 10.0
            elif volume_ratio >= 1.5:
                volume_bonus = 5.0
            else:
                volume_bonus = 0.0
            
            # Institutional activity bonus
            if volume_analysis['institutional_activity']:
                volume_bonus += 10.0
            
            # OBV trend alignment bonus
            if volume_analysis['obv_trend'] != 'neutral':
                volume_bonus += 5.0
            
            # Price-volume correlation bonus
            pv_correlation = abs(volume_analysis.get('price_volume_correlation', 0))
            if pv_correlation > 0.5:
                volume_bonus += 5.0
            
            return min(volume_bonus, 25.0)  # Cap at 25%
            
        except Exception as e:
            logger.error(f"Error calculating volume strength bonus: {e}")
            return 0.0
    
    async def _enhance_signal_with_volume_analysis(self,
                                                 signal: TradingSignal,
                                                 ohlcv_data: pd.DataFrame,
                                                 volume_profile: VolumeProfile,
                                                 indicators: Dict[str, Any],
                                                 volume_analysis: Dict[str, Any]) -> TradingSignal:
        """Enhance signal with additional volume-based confirmations"""
        
        try:
            confirmation_score = 0.0
            confirmations = []
            
            # 1. Volume trend confirmation
            volume_trend = volume_analysis['volume_trend']
            if ((signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY] and volume_trend == 'increasing') or
                (signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL] and volume_trend == 'increasing')):
                confirmation_score += 8.0
                confirmations.append(f"Volume trend {volume_trend}")
            
            # 2. OBV trend confirmation
            obv_trend = volume_analysis['obv_trend']
            if ((signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY] and obv_trend == 'up') or
                (signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL] and obv_trend == 'down')):
                confirmation_score += 10.0
                confirmations.append(f"OBV trend alignment ({obv_trend})")
                signal.volume_confirmed = True
            
            # 3. Accumulation/Distribution confirmation
            recent_ad = volume_analysis.get('recent_ad', 0)
            if ((signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY] and recent_ad > 0) or
                (signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL] and recent_ad < 0)):
                confirmation_score += 8.0
                confirmations.append("A/D line confirmation")
            
            # 4. Price-volume relationship confirmation
            pv_correlation = volume_analysis.get('price_volume_correlation', 0)
            if abs(pv_correlation) > 0.3:
                confirmation_score += 6.0
                confirmations.append(f"Price-volume correlation: {pv_correlation:.2f}")
            
            # 5. Institutional volume confirmation
            if volume_analysis['institutional_activity']:
                confirmation_score += 12.0
                confirmations.append("Institutional volume activity")
            
            # 6. Technical indicator alignment
            tech_confirmation = self._check_technical_alignment(signal, indicators)
            if tech_confirmation['confirmed']:
                confirmation_score += tech_confirmation['score']
                confirmations.append(tech_confirmation['reason'])
            
            # 7. Volume profile level significance
            level_significance = self._assess_level_significance(signal.entry_price, volume_profile)
            if level_significance['significant']:
                confirmation_score += level_significance['score']
                confirmations.append(level_significance['reason'])
            
            # Apply confirmation adjustments
            signal.confidence += confirmation_score
            signal.confidence = max(0.0, min(signal.confidence, 97.0))
            
            # Update reasoning
            if confirmations:
                signal.reasoning += f" | Confirmations: {'; '.join(confirmations)}"
            
            # Set multiple timeframe confirmed flag
            if confirmation_score >= 25.0:
                signal.multiple_timeframe_confirmed = True
            
            return signal
            
        except Exception as e:
            logger.error(f"Error enhancing signal with volume analysis: {e}")
            return signal
    
    def _check_technical_alignment(self, signal: TradingSignal, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Check technical indicator alignment with volume signal"""
        
        result = {'confirmed': False, 'score': 0.0, 'reason': ''}
        
        try:
            alignments = []
            score = 0.0
            
            # RSI alignment
            current_rsi = indicators['rsi'][-1]
            if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                if current_rsi < 50:  # RSI supports bullish signal
                    score += 4.0
                    alignments.append("RSI < 50")
            elif signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
                if current_rsi > 50:  # RSI supports bearish signal
                    score += 4.0
                    alignments.append("RSI > 50")
            
            # MACD alignment
            macd = indicators['macd'][-1]
            macd_signal = indicators['macd_signal'][-1]
            if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                if macd > macd_signal:  # MACD bullish
                    score += 5.0
                    alignments.append("MACD bullish")
            elif signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
                if macd < macd_signal:  # MACD bearish
                    score += 5.0
                    alignments.append("MACD bearish")
            
            # Moving average alignment
            current_price = signal.entry_price
            sma_20 = indicators['sma_20'][-1]
            if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                if current_price > sma_20:  # Price above SMA20
                    score += 3.0
                    alignments.append("Price > SMA20")
            elif signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
                if current_price < sma_20:  # Price below SMA20
                    score += 3.0
                    alignments.append("Price < SMA20")
            
            if alignments:
                result['confirmed'] = True
                result['score'] = score
                result['reason'] = f"Technical alignment: {', '.join(alignments)}"
            
            return result
            
        except Exception as e:
            logger.error(f"Error checking technical alignment: {e}")
            return result
    
    def _assess_level_significance(self, price: float, volume_profile: VolumeProfile) -> Dict[str, Any]:
        """Assess the significance of the current price level"""
        
        result = {'significant': False, 'score': 0.0, 'reason': ''}
        
        try:
            level_analysis = volume_profile.is_at_significant_level(price, 0.005)
            significance_factors = []
            score = 0.0
            
            if level_analysis['at_poc']:
                score += 8.0
                significance_factors.append("At POC")
            
            if level_analysis['at_value_area_boundary']:
                score += 6.0
                significance_factors.append("At VA boundary")
            
            if level_analysis['at_hvn']:
                score += 5.0
                significance_factors.append("At HVN")
            
            if level_analysis['at_lvn']:
                score += 4.0
                significance_factors.append("At LVN")
            
            if significance_factors:
                result['significant'] = True
                result['score'] = score
                result['reason'] = f"Significant level: {', '.join(significance_factors)}"
            
            return result
            
        except Exception as e:
            logger.error(f"Error assessing level significance: {e}")
            return result
    
    def _calculate_volume_stop_loss(self, 
                                  signal: TradingSignal,
                                  volume_profile: VolumeProfile,
                                  indicators: Dict[str, Any],
                                  ohlcv_data: pd.DataFrame) -> float:
        """Calculate volume profile-specific stop loss"""
        
        try:
            entry_price = signal.entry_price
            atr = indicators['atr'][-1]
            
            # Base ATR stop loss
            atr_stop = self.calculate_stop_loss(entry_price, signal.signal_type, atr)
            
            # Volume profile based stop loss
            if self.config.value_area_stop_loss and volume_profile.value_area_high and volume_profile.value_area_low:
                if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                    # For buy signals, use value area low as support
                    va_stop = volume_profile.value_area_low * 0.995  # 0.5% below VA low
                    # Use the closer (more conservative) stop
                    stop_loss = max(atr_stop, va_stop)
                else:
                    # For sell signals, use value area high as resistance
                    va_stop = volume_profile.value_area_high * 1.005  # 0.5% above VA high
                    # Use the closer (more conservative) stop
                    stop_loss = min(atr_stop, va_stop)
            else:
                stop_loss = atr_stop
            
            # Additional protection using recent high/low
            if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                recent_low = ohlcv_data['low'].tail(10).min()
                stop_loss = max(stop_loss, recent_low * 0.998)
            else:
                recent_high = ohlcv_data['high'].tail(10).max()
                stop_loss = min(stop_loss, recent_high * 1.002)
            
            return stop_loss
            
        except Exception as e:
            logger.error(f"Error calculating volume stop loss: {e}")
            return self.calculate_stop_loss(signal.entry_price, signal.signal_type, indicators.get('atr', [0])[-1])
    
    def _calculate_volume_take_profit(self,
                                    signal: TradingSignal,
                                    volume_profile: VolumeProfile,
                                    indicators: Dict[str, Any],
                                    ohlcv_data: pd.DataFrame) -> float:
        """Calculate volume profile-specific take profit"""
        
        try:
            entry_price = signal.entry_price
            
            # Base take profit
            base_tp = self.calculate_take_profit(entry_price, signal.signal_type, 2.5)
            
            # Adjust based on volume profile levels
            if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                # Look for resistance levels above entry
                if volume_profile.value_area_high and volume_profile.value_area_high > entry_price:
                    va_target = volume_profile.value_area_high * 0.998  # Slightly below VA high
                    base_tp = min(base_tp, va_target)
                
                # Look for HVN resistance levels
                hvn_resistance = self._find_next_hvn_level(entry_price, volume_profile, direction='up')
                if hvn_resistance:
                    hvn_target = hvn_resistance * 0.998
                    base_tp = min(base_tp, hvn_target)
            
            else:  # SELL signals
                # Look for support levels below entry
                if volume_profile.value_area_low and volume_profile.value_area_low < entry_price:
                    va_target = volume_profile.value_area_low * 1.002  # Slightly above VA low
                    base_tp = max(base_tp, va_target)
                
                # Look for HVN support levels
                hvn_support = self._find_next_hvn_level(entry_price, volume_profile, direction='down')
                if hvn_support:
                    hvn_target = hvn_support * 1.002
                    base_tp = max(base_tp, hvn_target)
            
            return base_tp
            
        except Exception as e:
            logger.error(f"Error calculating volume take profit: {e}")
            return self.calculate_take_profit(signal.entry_price, signal.signal_type)
    
    def _find_next_hvn_level(self, price: float, volume_profile: VolumeProfile, direction: str) -> Optional[float]:
        """Find the next High Volume Node level in the specified direction"""
        
        try:
            hvn_nodes = [node for node in volume_profile.volume_nodes if node.node_type == 'HVN']
            
            if not hvn_nodes:
                return None
            
            if direction == 'up':
                # Find HVN levels above current price
                above_nodes = [node for node in hvn_nodes if node.price_level > price]
                if above_nodes:
                    return min(above_nodes, key=lambda x: x.price_level).price_level
            else:  # direction == 'down'
                # Find HVN levels below current price
                below_nodes = [node for node in hvn_nodes if node.price_level < price]
                if below_nodes:
                    return max(below_nodes, key=lambda x: x.price_level).price_level
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding next HVN level: {e}")
            return None
    
    async def should_exit_position(self,
                                 entry_signal: TradingSignal,
                                 current_price: float,
                                 current_data: pd.DataFrame) -> Optional[str]:
        """Check volume profile-specific exit conditions"""
        
        try:
            # Call parent exit conditions first
            parent_exit = super().should_exit_position(entry_signal, current_price, current_data)
            if parent_exit:
                return parent_exit
            
            # Volume profile specific exits
            symbol = entry_signal.symbol
            if symbol in self.volume_profiles:
                volume_profile = self.volume_profiles[symbol]
                
                # Check if price returned to POC (potential exit signal)
                if volume_profile.poc:
                    poc_distance = abs(current_price - volume_profile.poc.price_level) / current_price
                    
                    if poc_distance <= 0.002:  # Within 0.2% of POC
                        # Check if this represents a significant move back to POC
                        entry_distance = abs(entry_signal.entry_price - volume_profile.poc.price_level) / entry_signal.entry_price
                        
                        if entry_distance > 0.01:  # Entry was more than 1% from POC
                            return "POC_RETURN"
                
                # Check for volume profile structure changes
                # (In practice, you'd recalculate volume profile and compare)
                
            # Breakout failure detection
            if self.config.breakout_failure_stop:
                if entry_signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                    # Check if price fell back below breakout level
                    breakout_level = entry_signal.technical_indicators.get('breakout_level')
                    if breakout_level and current_price < breakout_level * (1 - self.config.breakout_pullback_allowed):
                        return "BREAKOUT_FAILURE"
                
                elif entry_signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
                    # Check if price moved back above breakout level
                    breakout_level = entry_signal.technical_indicators.get('breakout_level')
                    if breakout_level and current_price > breakout_level * (1 + self.config.breakout_pullback_allowed):
                        return "BREAKOUT_FAILURE"
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking volume exit conditions: {e}")
            return None
    
    def get_strategy_summary(self) -> Dict[str, Any]:
        """Get Volume Profile strategy summary and statistics"""
        
        base_summary = self.get_performance_summary()
        
        volume_profile_specific = {
            'strategy_type': 'Volume Profile Breakout',
            'lookback_period': self.config.lookback_period,
            'price_bins': self.config.price_bins,
            'value_area_percentage': self.config.value_area_percentage,
            'volume_multiplier': self.config.volume_multiplier,
            'breakout_confirmation_bars': self.config.breakout_confirmation_bars,
            'current_profiles_tracked': len(self.volume_profiles),
            'recent_breakouts_tracked': len(self.recent_breakouts),
            'config': self.config.__dict__
        }
        
        return {**base_summary, **volume_profile_specific}

# Factory function for creating Volume Profile strategy instances
def create_volume_profile_strategy(config: Dict[str, Any] = None) -> VolumeProfileStrategy:
    """Factory function to create Volume Profile strategy instance"""
    return VolumeProfileStrategy("volume_profile_strategy", config)

# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_volume_profile_strategy():
        print("📊 Volume Profile Strategy Test")
        print("=" * 50)
        
        # Create strategy with custom config
        config = {
            'enabled': True,
            'lookback_period': 24,
            'volume_multiplier': 1.5,
            'volume_spike_threshold': 2.0,
            'poc_deviation_threshold': 0.002,
            'value_area_percentage': 0.7,
            'min_confidence': 65.0
        }
        
        strategy = create_volume_profile_strategy(config)
        await strategy.initialize()
        
        print(f"Strategy ID: {strategy.strategy_id}")
        print(f"Required indicators: {strategy.get_required_indicators()}")
        print(f"Lookback period: {strategy.get_lookback_period()}")
        
        # Generate realistic volume profile data
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='1H')
        
        # Create price data with volume clustering
        base_price = 50000
        prices = []
        volumes = []
        
        # Simulate volume profile with high activity around certain price levels
        poc_price = base_price + np.random.normal(0, 500)  # Point of Control
        
        for i in range(100):
            # Price movement with tendency to cluster around POC
            if i == 0:
                price = base_price
            else:
                # Bias toward POC
                bias_to_poc = (poc_price - prices[-1]) * 0.02
                random_move = np.random.normal(bias_to_poc, 200)
                price = prices[-1] + random_move
                price = max(price, base_price * 0.85)  # Floor
            
            prices.append(price)
            
            # Higher volume when closer to POC
            distance_from_poc = abs(price - poc_price) / poc_price
            volume_multiplier = max(0.5, 2.0 - (distance_from_poc * 10))
            base_volume = np.random.lognormal(15, 0.3)
            volume = base_volume * volume_multiplier
            volumes.append(volume)
        
        ohlcv_data = pd.DataFrame({
            'timestamp': dates,
            'open': [p + np.random.normal(0, 30) for p in prices],
            'high': [p + abs(np.random.normal(50, 30)) for p in prices],
            'low': [p - abs(np.random.normal(50, 30)) for p in prices],
            'close': prices,
            'volume': volumes
        })
        
        # Ensure OHLC relationships
        for i in range(len(ohlcv_data)):
            row = ohlcv_data.iloc[i]
            high = max(row['open'], row['close']) + abs(np.random.normal(0, 20))
            low = min(row['open'], row['close']) - abs(np.random.normal(0, 20))
            ohlcv_data.loc[i, 'high'] = high
            ohlcv_data.loc[i, 'low'] = low
        
        current_price = ohlcv_data['close'].iloc[-1]
        
        print(f"\nCurrent price: ${current_price:.2f}")
        print(f"POC price (simulated): ${poc_price:.2f}")
        
        # Test Volume Profile calculation
        lookback_data = ohlcv_data.tail(24)
        volume_profile = VolumeProfile(lookback_data, 30)
        
        print(f"\n📊 Volume Profile Analysis:")
        print(f"  Total volume: {volume_profile.total_volume:,.0f}")
        if volume_profile.poc:
            print(f"  POC price: ${volume_profile.poc.price_level:.2f}")
            print(f"  POC volume: {volume_profile.poc.volume_percentage:.1%}")
        if volume_profile.value_area_high and volume_profile.value_area_low:
            print(f"  Value Area: ${volume_profile.value_area_low:.2f} - ${volume_profile.value_area_high:.2f}")
        
        # Test signal generation
        try:
            signal = await strategy.analyze_symbol(
                'BTC/USDT',
                ohlcv_data,
                current_price,
                {'portfolio_value': 10000}
            )
            
            if signal:
                print(f"\n🎯 Volume Profile Signal Generated:")
                print(f"  Type: {signal.signal_type.value}")
                print(f"  Confidence: {signal.confidence:.1f}%")
                print(f"  Entry Price: ${signal.entry_price:.2f}")
                print(f"  Stop Loss: ${signal.stop_loss:.2f}")
                print(f"  Take Profit: ${signal.take_profit:.2f}")
                print(f"  R/R Ratio: {signal.risk_reward_ratio:.2f}")
                print(f"  Position Size: {signal.position_size_recommendation:.6f}")
                print(f"  Volume Confirmed: {signal.volume_confirmed}")
                print(f"  Reasoning: {signal.reasoning}")
                
                # Display technical indicators
                tech_indicators = signal.technical_indicators
                if tech_indicators.get('poc_price'):
                    print(f"  POC Price: ${tech_indicators['poc_price']:.2f}")
                if tech_indicators.get('value_area_high'):
                    print(f"  VA High: ${tech_indicators['value_area_high']:.2f}")
                if tech_indicators.get('value_area_low'):
                    print(f"  VA Low: ${tech_indicators['value_area_low']:.2f}")
                print(f"  Volume Ratio: {tech_indicators.get('current_volume_ratio', 0):.1f}x")
                print(f"  Volume Strength: {tech_indicators.get('volume_strength', 'unknown')}")
                
                # Test signal validation
                market_data = {'volume_24h': 150_000_000}
                is_valid = await strategy.validate_signal(signal, market_data)
                print(f"  Signal Valid: {is_valid}")
                
                # Test backtest
                future_data = ohlcv_data.tail(15)
                backtest_result = strategy.backtest_signal(signal, future_data)
                
                if backtest_result['success']:
                    print(f"\n📈 Backtest Result:")
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
                print("\n❌ No Volume Profile signal generated")
                
                # Debug volume analysis
                volume_analysis = strategy._analyze_volume_characteristics(ohlcv_data, 'BTC/USDT')
                print(f"Current volume ratio: {volume_analysis.get('current_volume_ratio', 0):.1f}x")
                print(f"Volume strength: {volume_analysis.get('volume_strength', 'unknown')}")
                print(f"Min volume threshold: {strategy.config.min_volume_for_signal:,}")
                print(f"Volume multiplier threshold: {strategy.config.volume_multiplier}x")
        
        except Exception as e:
            print(f"❌ Error testing strategy: {e}")
        
        # Display strategy summary
        summary = strategy.get_strategy_summary()
        print(f"\n📊 Strategy Summary:")
        print(f"  Strategy Type: {summary['strategy_type']}")
        print(f"  Signals Generated: {summary['signals_generated']}")
        print(f"  Success Rate: {summary.get('success_rate', 0):.1f}%")
        print(f"  Average Confidence: {summary['avg_confidence']:.1f}")
        print(f"  Profiles Tracked: {summary['current_profiles_tracked']}")
        
        # Test volume profile analysis
        print(f"\n📊 Volume Profile Details:")
        if volume_profile.volume_nodes:
            print(f"  Total Nodes: {len(volume_profile.volume_nodes)}")
            
            # Show top 5 volume nodes
            top_nodes = sorted(volume_profile.volume_nodes, key=lambda x: x.volume, reverse=True)[:5]
            print(f"  Top Volume Nodes:")
            for i, node in enumerate(top_nodes, 1):
                print(f"    {i}. ${node.price_level:.2f} - {node.volume_percentage:.1%} ({node.node_type})")
            
            # Show significant levels relative to current price
            level_analysis = volume_profile.is_at_significant_level(current_price, 0.01)
            print(f"  Current Price Analysis:")
            print(f"    At POC: {level_analysis['at_poc']}")
            print(f"    At VA Boundary: {level_analysis['at_value_area_boundary']}")
            print(f"    At HVN: {level_analysis['at_hvn']}")
            print(f"    At LVN: {level_analysis['at_lvn']}")
            print(f"    Nearby Nodes: {len(level_analysis['nearby_nodes'])}")
        
        print("\n✅ Volume Profile Strategy test completed!")
        
        # Test volume profile visualization data (for external plotting)
        if volume_profile.volume_nodes:
            print(f"\n📊 Volume Profile Data (for visualization):")
            profile_data = []
            for node in volume_profile.volume_nodes:
                profile_data.append({
                    'price': node.price_level,
                    'volume': node.volume,
                    'percentage': node.volume_percentage,
                    'type': node.node_type
                })
            
            # Sort by price for plotting
            profile_data.sort(key=lambda x: x['price'])
            
            print("  Price Levels | Volume % | Type")
            print("  " + "-" * 35)
            for data in profile_data[:10]:  # Show first 10
                print(f"  ${data['price']:>8.2f} | {data['percentage']:>6.1%} | {data['type']}")
    
    # Run the test
    asyncio.run(test_volume_profile_strategy())