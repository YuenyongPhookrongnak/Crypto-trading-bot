
    # Run the test
asyncio.run(test_risk_manager())
import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
import json

from database.connection import get_db_session
from database.models import Trade, PerformanceMetric, SystemLog, LogLevel

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Risk level classifications"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
    EXTREME = "EXTREME"

class CircuitBreakerStatus(Enum):
    """Circuit breaker status"""
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    TRIGGERED = "TRIGGERED"
    LOCKED = "LOCKED"

@dataclass
class RiskAssessment:
    """Risk assessment result"""
    risk_level: RiskLevel
    risk_score: float  # 0-100
    confidence: float  # 0-100
    
    # Risk factors
    position_risk: float
    portfolio_risk: float
    market_risk: float
    correlation_risk: float
    liquidity_risk: float
    
    # Risk limits
    max_position_size: float
    recommended_position_size: float
    max_portfolio_exposure: float
    
    # Risk mitigation
    stop_loss_recommendation: float
    position_size_adjustment: float
    hedge_recommendation: Optional[str] = None
    
    # Metadata
    assessment_timestamp: datetime = field(default_factory=datetime.utcnow)
    reasoning: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'risk_level': self.risk_level.value,
            'risk_score': self.risk_score,
            'confidence': self.confidence,
            'position_risk': self.position_risk,
            'portfolio_risk': self.portfolio_risk,
            'market_risk': self.market_risk,
            'correlation_risk': self.correlation_risk,
            'liquidity_risk': self.liquidity_risk,
            'max_position_size': self.max_position_size,
            'recommended_position_size': self.recommended_position_size,
            'max_portfolio_exposure': self.max_portfolio_exposure,
            'stop_loss_recommendation': self.stop_loss_recommendation,
            'position_size_adjustment': self.position_size_adjustment,
            'hedge_recommendation': self.hedge_recommendation,
            'assessment_timestamp': self.assessment_timestamp.isoformat(),
            'reasoning': self.reasoning,
            'warnings': self.warnings
        }

@dataclass
class CircuitBreakerState:
    """Circuit breaker state"""
    status: CircuitBreakerStatus
    trigger_count: int
    last_trigger_time: Optional[datetime]
    cooldown_until: Optional[datetime]
    
    # Trigger conditions
    daily_loss_triggered: bool = False
    drawdown_triggered: bool = False
    consecutive_losses_triggered: bool = False
    volatility_triggered: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'status': self.status.value,
            'trigger_count': self.trigger_count,
            'last_trigger_time': self.last_trigger_time.isoformat() if self.last_trigger_time else None,
            'cooldown_until': self.cooldown_until.isoformat() if self.cooldown_until else None,
            'daily_loss_triggered': self.daily_loss_triggered,
            'drawdown_triggered': self.drawdown_triggered,
            'consecutive_losses_triggered': self.consecutive_losses_triggered,
            'volatility_triggered': self.volatility_triggered
        }

class RiskManager:
    """Advanced Risk Management System"""
    
    def __init__(self, trading_config):
        self.config = trading_config
        
        # Risk metrics tracking
        self.risk_metrics = {
            'daily_pnl': 0.0,
            'max_drawdown': 0.0,
            'current_drawdown': 0.0,
            'var_95': 0.0,
            'portfolio_correlation': {},
            'position_sizes': {},
            'risk_exposure': 0.0,
            'leverage_ratio': 1.0,
            'last_updated': datetime.utcnow()
        }
        
        # Circuit breakers
        self.circuit_breakers = {
            'daily_loss_limit': trading_config.daily_loss_limit,
            'max_drawdown_limit': 0.20,  # 20%
            'consecutive_losses': trading_config.max_consecutive_losses,
            'position_concentration': 0.30,  # 30% max in single position
            'portfolio_var_limit': 0.15,  # 15% VaR limit
            'correlation_limit': 0.8,  # 80% max correlation
            'volatility_spike_limit': 0.5  # 50% volatility spike
        }
        
        # Circuit breaker state
        self.circuit_breaker_state = CircuitBreakerState(
            status=CircuitBreakerStatus.NORMAL,
            trigger_count=0,
            last_trigger_time=None,
            cooldown_until=None
        )
        
        # Risk history
        self.risk_history = []
        self.assessment_history = []
        
        # Performance tracking
        self.recent_trades = []
        self.consecutive_losses = 0
        self.daily_trades = []
        
        # Market volatility tracking
        self.volatility_metrics = {
            'current_volatility': 0.0,
            'average_volatility': 0.0,
            'volatility_spike_threshold': 0.5,
            'last_volatility_update': datetime.utcnow()
        }
        
        logger.info("Risk Manager initialized")
    
    async def evaluate_trade_proposal(self, 
                                    symbol: str, 
                                    signal_data: Dict[str, Any], 
                                    portfolio_context: Dict[str, Any]) -> RiskAssessment:
        """Comprehensive trade proposal risk evaluation"""
        
        try:
            # Get current portfolio state
            portfolio_value = portfolio_context.get('total_value', 10000)
            available_balance = portfolio_context.get('available_balance', 5000)
            current_positions = portfolio_context.get('current_positions', [])
            
            # Calculate position risk
            position_risk = await self._assess_position_risk(signal_data, portfolio_value)
            
            # Calculate portfolio risk
            portfolio_risk = await self._assess_portfolio_risk(
                symbol, signal_data, current_positions, portfolio_value
            )
            
            # Calculate market risk
            market_risk = await self._assess_market_risk(symbol, signal_data)
            
            # Calculate correlation risk
            correlation_risk = await self._assess_correlation_risk(symbol, current_positions)
            
            # Calculate liquidity risk
            liquidity_risk = await self._assess_liquidity_risk(symbol, signal_data)
            
            # Calculate position sizing
            position_sizing = await self._calculate_optimal_position_size(
                signal_data, portfolio_value, position_risk, correlation_risk
            )
            
            # Aggregate risk score
            risk_score = self._calculate_aggregate_risk_score(
                position_risk, portfolio_risk, market_risk, correlation_risk, liquidity_risk
            )
            
            # Determine risk level
            risk_level = self._determine_risk_level(risk_score)
            
            # Generate recommendations
            recommendations = self._generate_risk_recommendations(
                risk_score, risk_level, position_sizing, signal_data
            )
            
            # Create assessment
            assessment = RiskAssessment(
                risk_level=risk_level,
                risk_score=risk_score,
                confidence=recommendations['confidence'],
                position_risk=position_risk,
                portfolio_risk=portfolio_risk,
                market_risk=market_risk,
                correlation_risk=correlation_risk,
                liquidity_risk=liquidity_risk,
                max_position_size=position_sizing['max_size'],
                recommended_position_size=position_sizing['recommended_size'],
                max_portfolio_exposure=position_sizing['max_exposure'],
                stop_loss_recommendation=recommendations['stop_loss'],
                position_size_adjustment=recommendations['size_adjustment'],
                hedge_recommendation=recommendations.get('hedge'),
                reasoning=recommendations['reasoning'],
                warnings=recommendations['warnings']
            )
            
            # Store assessment
            self.assessment_history.append(assessment)
            if len(self.assessment_history) > 1000:
                self.assessment_history = self.assessment_history[-1000:]
            
            # Update risk metrics
            await self._update_risk_metrics(assessment, signal_data, portfolio_context)
            
            logger.info(f"Risk assessment completed for {symbol}: {risk_level.value} "
                       f"(Score: {risk_score:.1f}, Confidence: {recommendations['confidence']:.1f}%)")
            
            return assessment
            
        except Exception as e:
            logger.error(f"Error evaluating trade proposal for {symbol}: {e}")
            return self._create_error_assessment(symbol, str(e))
    
    async def _assess_position_risk(self, signal_data: Dict[str, Any], portfolio_value: float) -> float:
        """Assess risk specific to the proposed position"""
        
        try:
            risk_factors = []
            
            # Position size risk
            proposed_size = signal_data.get('position_size', 0)
            if proposed_size > 0:
                size_percentage = (proposed_size * signal_data.get('entry_price', 0)) / portfolio_value
                if size_percentage > 0.2:  # >20% of portfolio
                    risk_factors.append(40)
                elif size_percentage > 0.1:  # >10% of portfolio
                    risk_factors.append(25)
                elif size_percentage > 0.05:  # >5% of portfolio
                    risk_factors.append(15)
                else:
                    risk_factors.append(5)
            
            # Stop loss distance risk
            entry_price = signal_data.get('entry_price', 0)
            stop_loss = signal_data.get('stop_loss', 0)
            
            if entry_price > 0 and stop_loss > 0:
                stop_distance = abs(entry_price - stop_loss) / entry_price
                if stop_distance > 0.1:  # >10% stop distance
                    risk_factors.append(30)
                elif stop_distance > 0.05:  # >5% stop distance
                    risk_factors.append(20)
                elif stop_distance > 0.02:  # >2% stop distance
                    risk_factors.append(10)
                else:
                    risk_factors.append(5)
            else:
                risk_factors.append(35)  # No stop loss defined
            
            # Confidence risk (inverse)
            confidence = signal_data.get('confidence', 50)
            confidence_risk = (100 - confidence) * 0.3
            risk_factors.append(confidence_risk)
            
            # Risk/Reward ratio risk
            risk_reward = signal_data.get('risk_reward_ratio', 0)
            if risk_reward > 0:
                if risk_reward < 1.5:
                    risk_factors.append(25)
                elif risk_reward < 2.0:
                    risk_factors.append(15)
                elif risk_reward < 3.0:
                    risk_factors.append(10)
                else:
                    risk_factors.append(0)
            else:
                risk_factors.append(20)
            
            return min(100, np.mean(risk_factors))
            
        except Exception as e:
            logger.error(f"Error assessing position risk: {e}")
            return 75.0  # Default high risk on error
    
    async def _assess_portfolio_risk(self, 
                                   symbol: str, 
                                   signal_data: Dict[str, Any], 
                                   current_positions: List[Dict[str, Any]], 
                                   portfolio_value: float) -> float:
        """Assess portfolio-level risk"""
        
        try:
            risk_factors = []
            
            # Portfolio concentration risk
            current_exposure = sum(
                abs(pos.get('market_value', 0)) for pos in current_positions
            ) / portfolio_value if portfolio_value > 0 else 0
            
            proposed_value = signal_data.get('position_size', 0) * signal_data.get('entry_price', 0)
            new_exposure = (current_exposure * portfolio_value + proposed_value) / portfolio_value
            
            if new_exposure > 0.9:  # >90% exposure
                risk_factors.append(50)
            elif new_exposure > 0.8:  # >80% exposure
                risk_factors.append(35)
            elif new_exposure > 0.6:  # >60% exposure
                risk_factors.append(20)
            else:
                risk_factors.append(5)
            
            # Position count risk
            active_positions = len([pos for pos in current_positions if pos.get('status') == 'OPEN'])
            max_positions = self.config.max_open_positions
            
            if active_positions >= max_positions:
                risk_factors.append(40)
            elif active_positions >= max_positions * 0.8:
                risk_factors.append(25)
            elif active_positions >= max_positions * 0.6:
                risk_factors.append(15)
            else:
                risk_factors.append(5)
            
            # Sector concentration (crypto pairs)
            base_currency = symbol.split('/')[0] if '/' in symbol else symbol
            same_base_positions = len([
                pos for pos in current_positions 
                if pos.get('symbol', '').startswith(base_currency)
            ])
            
            if same_base_positions >= 3:
                risk_factors.append(30)
            elif same_base_positions >= 2:
                risk_factors.append(15)
            else:
                risk_factors.append(0)
            
            # Leverage risk
            total_margin_used = sum(
                pos.get('margin_used', 0) for pos in current_positions
            )
            leverage_ratio = total_margin_used / portfolio_value if portfolio_value > 0 else 0
            
            if leverage_ratio > 0.5:  # >50% margin usage
                risk_factors.append(35)
            elif leverage_ratio > 0.3:  # >30% margin usage
                risk_factors.append(20)
            elif leverage_ratio > 0.1:  # >10% margin usage
                risk_factors.append(10)
            else:
                risk_factors.append(0)
            
            return min(100, np.mean(risk_factors))
            
        except Exception as e:
            logger.error(f"Error assessing portfolio risk: {e}")
            return 50.0
    
    async def _assess_market_risk(self, symbol: str, signal_data: Dict[str, Any]) -> float:
        """Assess market-wide risk factors"""
        
        try:
            risk_factors = []
            
            # Volatility risk
            current_volatility = self.volatility_metrics.get('current_volatility', 0)
            average_volatility = self.volatility_metrics.get('average_volatility', 0)
            
            if average_volatility > 0:
                volatility_ratio = current_volatility / average_volatility
                if volatility_ratio > 2.0:  # 2x normal volatility
                    risk_factors.append(40)
                elif volatility_ratio > 1.5:  # 1.5x normal volatility
                    risk_factors.append(25)
                elif volatility_ratio > 1.2:  # 1.2x normal volatility
                    risk_factors.append(15)
                else:
                    risk_factors.append(5)
            else:
                risk_factors.append(20)  # Unknown volatility
            
            # Market condition risk
            market_condition = signal_data.get('market_condition', 'UNKNOWN')
            if market_condition == 'VOLATILE':
                risk_factors.append(30)
            elif market_condition == 'TRENDING':
                risk_factors.append(10)
            elif market_condition == 'SIDEWAYS':
                risk_factors.append(15)
            else:
                risk_factors.append(20)
            
            # Liquidity risk based on symbol
            if symbol.startswith('BTC') or symbol.startswith('ETH'):
                risk_factors.append(5)  # High liquidity
            elif symbol.endswith('/USDT'):
                risk_factors.append(10)  # Medium liquidity
            else:
                risk_factors.append(25)  # Lower liquidity
            
            # Time-based risk (market hours, weekends)
            current_hour = datetime.utcnow().hour
            current_weekday = datetime.utcnow().weekday()
            
            # Weekend risk
            if current_weekday >= 5:  # Saturday = 5, Sunday = 6
                risk_factors.append(15)
            
            # Low activity hours (typically 2-6 UTC)
            if 2 <= current_hour <= 6:
                risk_factors.append(10)
            else:
                risk_factors.append(0)
            
            return min(100, np.mean(risk_factors))
            
        except Exception as e:
            logger.error(f"Error assessing market risk: {e}")
            return 30.0
    
    async def _assess_correlation_risk(self, symbol: str, current_positions: List[Dict[str, Any]]) -> float:
        """Assess correlation risk with existing positions"""
        
        try:
            if not current_positions:
                return 0.0
            
            # Simplified correlation matrix for crypto
            correlation_matrix = {
                'BTC': {'ETH': 0.8, 'ADA': 0.7, 'DOT': 0.6, 'LINK': 0.5, 'SOL': 0.7},
                'ETH': {'BTC': 0.8, 'ADA': 0.6, 'DOT': 0.7, 'LINK': 0.6, 'SOL': 0.8},
                'ADA': {'BTC': 0.7, 'ETH': 0.6, 'DOT': 0.5, 'LINK': 0.4, 'SOL': 0.6},
                'DOT': {'BTC': 0.6, 'ETH': 0.7, 'ADA': 0.5, 'LINK': 0.5, 'SOL': 0.6},
                'LINK': {'BTC': 0.5, 'ETH': 0.6, 'ADA': 0.4, 'DOT': 0.5, 'SOL': 0.5},
                'SOL': {'BTC': 0.7, 'ETH': 0.8, 'ADA': 0.6, 'DOT': 0.6, 'LINK': 0.5}
            }
            
            base_currency = symbol.split('/')[0] if '/' in symbol else symbol
            
            correlations = []
            weighted_correlations = []
            
            for position in current_positions:
                pos_symbol = position.get('symbol', '')
                pos_base = pos_symbol.split('/')[0] if '/' in pos_symbol else pos_symbol
                
                # Get correlation
                correlation = 0.3  # Default moderate correlation
                if base_currency in correlation_matrix and pos_base in correlation_matrix[base_currency]:
                    correlation = correlation_matrix[base_currency][pos_base]
                elif base_currency == pos_base:
                    correlation = 1.0  # Perfect correlation (same asset)
                
                correlations.append(correlation)
                
                # Weight by position size
                position_weight = abs(position.get('market_value', 0))
                weighted_correlations.append(correlation * position_weight)
            
            if not correlations:
                return 0.0
            
            # Calculate risk score based on correlations
            avg_correlation = np.mean(correlations)
            max_correlation = max(correlations)
            
            # Weight by position sizes if available
            total_weight = sum(abs(pos.get('market_value', 1)) for pos in current_positions)
            if total_weight > 0:
                weighted_avg_correlation = sum(weighted_correlations) / total_weight
            else:
                weighted_avg_correlation = avg_correlation
            
            # Convert correlation to risk score
            if max_correlation >= 0.9:
                correlation_risk = 50
            elif max_correlation >= 0.8:
                correlation_risk = 40
            elif weighted_avg_correlation >= 0.7:
                correlation_risk = 30
            elif weighted_avg_correlation >= 0.5:
                correlation_risk = 20
            else:
                correlation_risk = 10
            
            return min(100, correlation_risk)
            
        except Exception as e:
            logger.error(f"Error assessing correlation risk: {e}")
            return 25.0
    
    async def _assess_liquidity_risk(self, symbol: str, signal_data: Dict[str, Any]) -> float:
        """Assess liquidity risk for the symbol"""
        
        try:
            risk_factors = []
            
            # Volume-based liquidity assessment
            volume_24h = signal_data.get('volume_24h', 0)
            
            if volume_24h >= 100_000_000:  # $100M+
                risk_factors.append(0)
            elif volume_24h >= 50_000_000:   # $50M+
                risk_factors.append(5)
            elif volume_24h >= 10_000_000:   # $10M+
                risk_factors.append(15)
            elif volume_24h >= 1_000_000:    # $1M+
                risk_factors.append(30)
            else:
                risk_factors.append(50)
            
            # Spread risk (if available)
            spread = signal_data.get('bid_ask_spread', 0)
            if spread > 0:
                if spread > 0.01:  # >1% spread
                    risk_factors.append(30)
                elif spread > 0.005:  # >0.5% spread
                    risk_factors.append(20)
                elif spread > 0.002:  # >0.2% spread
                    risk_factors.append(10)
                else:
                    risk_factors.append(0)
            
            # Symbol-specific liquidity
            if symbol.startswith('BTC') or symbol.startswith('ETH'):
                risk_factors.append(0)  # Highest liquidity
            elif any(symbol.startswith(major) for major in ['BNB', 'ADA', 'SOL', 'DOT']):
                risk_factors.append(5)  # High liquidity
            elif symbol.endswith('/USDT'):
                risk_factors.append(10)  # Medium liquidity
            else:
                risk_factors.append(25)  # Lower liquidity
            
            return min(100, np.mean(risk_factors))
            
        except Exception as e:
            logger.error(f"Error assessing liquidity risk: {e}")
            return 20.0
    
    async def _calculate_optimal_position_size(self, 
                                             signal_data: Dict[str, Any], 
                                             portfolio_value: float, 
                                             position_risk: float, 
                                             correlation_risk: float) -> Dict[str, float]:
        """Calculate optimal position sizing"""
        
        try:
            # Base position size from risk per trade
            base_risk_per_trade = self.config.max_risk_per_trade
            confidence = signal_data.get('confidence', 50)
            
            # Adjust risk based on confidence
            confidence_multiplier = min(1.2, confidence / 100 * 1.5)  # Max 1.2x for high confidence
            adjusted_risk = base_risk_per_trade * confidence_multiplier
            
            # Risk reduction based on position and correlation risk
            risk_reduction = 1.0
            
            if position_risk > 70:
                risk_reduction *= 0.5  # Halve size for high position risk
            elif position_risk > 50:
                risk_reduction *= 0.75
            
            if correlation_risk > 60:
                risk_reduction *= 0.6  # Reduce for high correlation
            elif correlation_risk > 40:
                risk_reduction *= 0.8
            
            final_risk = adjusted_risk * risk_reduction
            
            # Calculate position size
            entry_price = signal_data.get('entry_price', 1)
            stop_loss = signal_data.get('stop_loss', entry_price * 0.95)  # Default 5% stop
            
            risk_per_unit = abs(entry_price - stop_loss)
            risk_amount = portfolio_value * final_risk
            
            if risk_per_unit > 0:
                recommended_size = risk_amount / risk_per_unit
            else:
                recommended_size = 0
            
            # Maximum position size (portfolio percentage limit)
            max_position_value = portfolio_value * self.circuit_breakers['position_concentration']
            max_size = max_position_value / entry_price
            
            # Maximum portfolio exposure
            max_exposure = min(0.8, 1.0 - correlation_risk / 100)  # Reduce max exposure based on correlation
            
            return {
                'recommended_size': min(recommended_size, max_size),
                'max_size': max_size,
                'max_exposure': max_exposure,
                'risk_reduction_factor': risk_reduction,
                'confidence_multiplier': confidence_multiplier
            }
            
        except Exception as e:
            logger.error(f"Error calculating optimal position size: {e}")
            return {
                'recommended_size': 0,
                'max_size': 0,
                'max_exposure': 0.1,
                'risk_reduction_factor': 0.5,
                'confidence_multiplier': 1.0
            }
    
    def _calculate_aggregate_risk_score(self, 
                                      position_risk: float, 
                                      portfolio_risk: float, 
                                      market_risk: float, 
                                      correlation_risk: float, 
                                      liquidity_risk: float) -> float:
        """Calculate aggregate risk score with weighted factors"""
        
        # Risk factor weights
        weights = {
            'position': 0.3,
            'portfolio': 0.25,
            'market': 0.2,
            'correlation': 0.15,
            'liquidity': 0.1
        }
        
        aggregate_score = (
            position_risk * weights['position'] +
            portfolio_risk * weights['portfolio'] +
            market_risk * weights['market'] +
            correlation_risk * weights['correlation'] +
            liquidity_risk * weights['liquidity']
        )
        
        return min(100, max(0, aggregate_score))
    
    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine risk level from aggregate score"""
        
        if risk_score >= 80:
            return RiskLevel.EXTREME
        elif risk_score >= 65:
            return RiskLevel.CRITICAL
        elif risk_score >= 45:
            return RiskLevel.HIGH
        elif risk_score >= 25:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _generate_risk_recommendations(self, 
                                     risk_score: float, 
                                     risk_level: RiskLevel, 
                                     position_sizing: Dict[str, float], 
                                     signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate risk management recommendations"""
        
        recommendations = {
            'confidence': 100 - risk_score,
            'reasoning': [],
            'warnings': [],
            'stop_loss': signal_data.get('stop_loss', 0),
            'size_adjustment': 1.0,
            'hedge': None
        }
        
        # Risk level specific recommendations
        if risk_level == RiskLevel.EXTREME:
            recommendations['size_adjustment'] = 0.0  # No position
            recommendations['reasoning'].append("Risk level EXTREME - position rejected")
            recommendations['warnings'].append("Market conditions extremely unfavorable")
            
        elif risk_level == RiskLevel.CRITICAL:
            recommendations['size_adjustment'] = 0.2  # 20% of calculated size
            recommendations['reasoning'].append("Risk level CRITICAL - severely reduced position size")
            recommendations['warnings'].append("High risk trade - consider avoiding")
            
        elif risk_level == RiskLevel.HIGH:
            recommendations['size_adjustment'] = 0.5  # 50% of calculated size
            recommendations['reasoning'].append("Risk level HIGH - reduced position size")
            recommendations['warnings'].append("Elevated risk - tight risk management required")
            
        elif risk_level == RiskLevel.MEDIUM:
            recommendations['size_adjustment'] = 0.8  # 80% of calculated size
            recommendations['reasoning'].append("Risk level MEDIUM - slightly reduced position")
            
        else:  # LOW risk
            recommendations['size_adjustment'] = 1.0  # Full calculated size
            recommendations['reasoning'].append("Risk level LOW - proceed with calculated position")
        
        # Stop loss recommendations
        entry_price = signal_data.get('entry_price', 0)
        current_stop = signal_data.get('stop_loss', 0)
        
        if entry_price > 0 and current_stop > 0:
            stop_distance = abs(entry_price - current_stop) / entry_price
            
            if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL, RiskLevel.EXTREME]:
                # Tighten stop loss for high risk trades
                tighter_stop = entry_price * 0.98 if signal_data.get('direction') == 'LONG' else entry_price * 1.02
                recommendations['stop_loss'] = tighter_stop
                recommendations['reasoning'].append("Tightened stop loss due to high risk")
            
            if stop_distance > 0.05:  # >5% stop distance
                recommendations['warnings'].append("Wide stop loss increases risk")
        
        # Position size warnings
        recommended_size = position_sizing.get('recommended_size', 0)
        max_size = position_sizing.get('max_size', 0)
        
        if recommended_size >= max_size * 0.9:
            recommendations['warnings'].append("Position size near maximum limit")
        
        # Correlation warnings
        if 'correlation_risk' in signal_data and signal_data['correlation_risk'] > 60:
            recommendations['warnings'].append("High correlation with existing positions")
            recommendations['hedge'] = "Consider hedging with uncorrelated asset"
        
        # Market condition warnings
        market_condition = signal_data.get('market_condition', 'UNKNOWN')
        if market_condition == 'VOLATILE':
            recommendations['warnings'].append("High market volatility detected")
        elif market_condition == 'LOW_VOLUME':
            recommendations['warnings'].append("Low volume market conditions")
        
        return recommendations
    
    async def check_circuit_breakers(self, portfolio_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Check all circuit breaker conditions"""
        
        try:
            current_time = datetime.utcnow()
            triggers = []
            
            # Check if in cooldown period
            if (self.circuit_breaker_state.cooldown_until and 
                current_time < self.circuit_breaker_state.cooldown_until):
                return {
                    'triggered': True,
                    'status': 'COOLDOWN',
                    'reason': 'Circuit breaker in cooldown period',
                    'cooldown_until': self.circuit_breaker_state.cooldown_until.isoformat(),
                    'triggers': []
                }
            
            # Daily loss check
            daily_pnl_pct = await self._get_daily_pnl_percentage()
            if daily_pnl_pct < -self.circuit_breakers['daily_loss_limit']:
                triggers.append({
                    'type': 'DAILY_LOSS',
                    'value': daily_pnl_pct,
                    'threshold': -self.circuit_breakers['daily_loss_limit'],
                    'severity': 'HIGH'
                })
                self.circuit_breaker_state.daily_loss_triggered = True
            
            # Drawdown check
            current_drawdown = await self._calculate_current_drawdown()
            if current_drawdown > self.circuit_breakers['max_drawdown_limit']:
                triggers.append({
                    'type': 'MAX_DRAWDOWN',
                    'value': current_drawdown,
                    'threshold': self.circuit_breakers['max_drawdown_limit'],
                    'severity': 'CRITICAL'
                })
                self.circuit_breaker_state.drawdown_triggered = True
            
            # Consecutive losses check
            consecutive_losses = await self._count_consecutive_losses()
            if consecutive_losses >= self.circuit_breakers['consecutive_losses']:
                triggers.append({
                    'type': 'CONSECUTIVE_LOSSES',
                    'value': consecutive_losses,
                    'threshold': self.circuit_breakers['consecutive_losses'],
                    'severity': 'MEDIUM'
                })
                self.circuit_breaker_state.consecutive_losses_triggered = True
            
            # Volatility spike check
            volatility_spike = await self._detect_volatility_spike()
            if volatility_spike > self.circuit_breakers['volatility_spike_limit']:
                triggers.append({
                    'type': 'VOLATILITY_SPIKE',
                    'value': volatility_spike,
                    'threshold': self.circuit_breakers['volatility_spike_limit'],
                    'severity': 'MEDIUM'
                })
                self.circuit_breaker_state.volatility_triggered = True
            
            # Portfolio concentration check
            if portfolio_context:
                concentration_risk = await self._check_portfolio_concentration(portfolio_context)
                if concentration_risk > self.circuit_breakers['position_concentration']:
                    triggers.append({
                        'type': 'PORTFOLIO_CONCENTRATION',
                        'value': concentration_risk,
                        'threshold': self.circuit_breakers['position_concentration'],
                        'severity': 'MEDIUM'
                    })
            
            # VaR limit check
            current_var = self.risk_metrics.get('var_95', 0)
            if current_var > self.circuit_breakers['portfolio_var_limit']:
                triggers.append({
                    'type': 'VAR_LIMIT',
                    'value': current_var,
                    'threshold': self.circuit_breakers['portfolio_var_limit'],
                    'severity': 'HIGH'
                })
            
            # Determine circuit breaker action
            if triggers:
                # Find highest severity
                severity_order = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'CRITICAL': 4}
                max_severity = max(trigger['severity'] for trigger in triggers)
                
                # Update circuit breaker state
                self.circuit_breaker_state.trigger_count += 1
                self.circuit_breaker_state.last_trigger_time = current_time
                
                if max_severity == 'CRITICAL':
                    self.circuit_breaker_state.status = CircuitBreakerStatus.LOCKED
                    self.circuit_breaker_state.cooldown_until = current_time + timedelta(hours=24)
                elif max_severity == 'HIGH':
                    self.circuit_breaker_state.status = CircuitBreakerStatus.TRIGGERED
                    self.circuit_breaker_state.cooldown_until = current_time + timedelta(hours=4)
                else:
                    self.circuit_breaker_state.status = CircuitBreakerStatus.WARNING
                    self.circuit_breaker_state.cooldown_until = current_time + timedelta(hours=1)
                
                # Log circuit breaker activation
                await self._log_circuit_breaker_activation(triggers, max_severity)
                
                return {
                    'triggered': True,
                    'status': self.circuit_breaker_state.status.value,
                    'triggers': triggers,
                    'severity': max_severity,
                    'reason': f"Circuit breaker triggered: {triggers[0]['type']}",
                    'cooldown_until': self.circuit_breaker_state.cooldown_until.isoformat(),
                    'trigger_count': self.circuit_breaker_state.trigger_count
                }
            else:
                # Reset circuit breaker state if no triggers
                if self.circuit_breaker_state.status != CircuitBreakerStatus.NORMAL:
                    self.circuit_breaker_state.status = CircuitBreakerStatus.NORMAL
                    self.circuit_breaker_state.daily_loss_triggered = False
                    self.circuit_breaker_state.drawdown_triggered = False
                    self.circuit_breaker_state.consecutive_losses_triggered = False
                    self.circuit_breaker_state.volatility_triggered = False
                
                return {
                    'triggered': False,
                    'status': 'NORMAL',
                    'triggers': []
                }
            
        except Exception as e:
            logger.error(f"Error checking circuit breakers: {e}")
            return {
                'triggered': False,
                'status': 'ERROR',
                'error': str(e),
                'triggers': []
            }
    
    async def _get_daily_pnl_percentage(self) -> float:
        """Get daily P&L as percentage of portfolio"""
        try:
            # In a real implementation, this would query the database
            # For now, return the cached value
            return self.risk_metrics.get('daily_pnl', 0.0)
        except Exception as e:
            logger.error(f"Error getting daily P&L: {e}")
            return 0.0
    
    async def _calculate_current_drawdown(self) -> float:
        """Calculate current drawdown from peak"""
        try:
            # In a real implementation, this would calculate from historical data
            return self.risk_metrics.get('current_drawdown', 0.0)
        except Exception as e:
            logger.error(f"Error calculating drawdown: {e}")
            return 0.0
    
    async def _count_consecutive_losses(self) -> int:
        """Count consecutive losing trades"""
        try:
            if not self.recent_trades:
                return 0
            
            consecutive = 0
            for trade in reversed(self.recent_trades):
                if trade.get('pnl', 0) < 0:
                    consecutive += 1
                else:
                    break
            
            return consecutive
            
        except Exception as e:
            logger.error(f"Error counting consecutive losses: {e}")
            return 0
    
    async def _detect_volatility_spike(self) -> float:
        """Detect volatility spike relative to average"""
        try:
            current_vol = self.volatility_metrics.get('current_volatility', 0)
            average_vol = self.volatility_metrics.get('average_volatility', 0)
            
            if average_vol > 0:
                spike_ratio = (current_vol - average_vol) / average_vol
                return max(0, spike_ratio)
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error detecting volatility spike: {e}")
            return 0.0
    
    async def _check_portfolio_concentration(self, portfolio_context: Dict[str, Any]) -> float:
        """Check portfolio concentration risk"""
        try:
            positions = portfolio_context.get('current_positions', [])
            if not positions:
                return 0.0
            
            total_value = portfolio_context.get('total_value', 1)
            
            # Find largest position as percentage of portfolio
            max_position_value = 0
            for position in positions:
                position_value = abs(position.get('market_value', 0))
                max_position_value = max(max_position_value, position_value)
            
            concentration = max_position_value / total_value if total_value > 0 else 0
            return concentration
            
        except Exception as e:
            logger.error(f"Error checking portfolio concentration: {e}")
            return 0.0
    
    async def _log_circuit_breaker_activation(self, triggers: List[Dict[str, Any]], severity: str):
        """Log circuit breaker activation to database"""
        try:
            with get_db_session() as session:
                log_entry = SystemLog(
                    level=LogLevel.CRITICAL if severity == 'CRITICAL' else LogLevel.ERROR,
                    component='RISK_MANAGER',
                    message=f"Circuit breaker activated: {len(triggers)} triggers",
                    details={
                        'triggers': triggers,
                        'severity': severity,
                        'circuit_breaker_state': self.circuit_breaker_state.to_dict()
                    }
                )
                session.add(log_entry)
                
        except Exception as e:
            logger.error(f"Error logging circuit breaker activation: {e}")
    
    async def update_risk_metrics(self, trade_result: Dict[str, Any]):
        """Update risk metrics with trade result"""
        try:
            # Add to recent trades
            self.recent_trades.append(trade_result)
            if len(self.recent_trades) > 100:  # Keep last 100 trades
                self.recent_trades = self.recent_trades[-100:]
            
            # Update daily P&L
            trade_pnl = trade_result.get('pnl', 0)
            self.risk_metrics['daily_pnl'] += trade_pnl
            
            # Update consecutive losses counter
            if trade_pnl < 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0
            
            # Calculate new drawdown
            await self._update_drawdown_metrics()
            
            # Calculate VaR
            await self._calculate_value_at_risk()
            
            # Update timestamp
            self.risk_metrics['last_updated'] = datetime.utcnow()
            
            logger.debug(f"Risk metrics updated with trade result: P&L ${trade_pnl:.2f}")
            
        except Exception as e:
            logger.error(f"Error updating risk metrics: {e}")
    
    async def _update_risk_metrics(self, 
                                 assessment: RiskAssessment, 
                                 signal_data: Dict[str, Any], 
                                 portfolio_context: Dict[str, Any]):
        """Update internal risk metrics"""
        try:
            # Update risk exposure
            position_value = (signal_data.get('position_size', 0) * 
                            signal_data.get('entry_price', 0))
            portfolio_value = portfolio_context.get('total_value', 1)
            
            self.risk_metrics['risk_exposure'] = position_value / portfolio_value
            
            # Update position sizes tracking
            symbol = signal_data.get('symbol', 'UNKNOWN')
            self.risk_metrics['position_sizes'][symbol] = {
                'size': signal_data.get('position_size', 0),
                'value': position_value,
                'risk_score': assessment.risk_score,
                'timestamp': datetime.utcnow()
            }
            
            # Update correlation matrix (simplified)
            # In a real implementation, this would be more sophisticated
            
            self.risk_metrics['last_updated'] = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Error updating risk metrics: {e}")
    
    async def _update_drawdown_metrics(self):
        """Update drawdown calculations"""
        try:
            if len(self.recent_trades) < 2:
                return
            
            # Calculate cumulative P&L
            cumulative_pnl = []
            running_total = 0
            
            for trade in self.recent_trades:
                running_total += trade.get('pnl', 0)
                cumulative_pnl.append(running_total)
            
            # Calculate drawdown
            peak = cumulative_pnl[0]
            max_drawdown = 0
            current_drawdown = 0
            
            for pnl in cumulative_pnl:
                if pnl > peak:
                    peak = pnl
                    current_drawdown = 0
                else:
                    current_drawdown = (peak - pnl) / abs(peak) if peak != 0 else 0
                    max_drawdown = max(max_drawdown, current_drawdown)
            
            self.risk_metrics['max_drawdown'] = max_drawdown
            self.risk_metrics['current_drawdown'] = current_drawdown
            
        except Exception as e:
            logger.error(f"Error updating drawdown metrics: {e}")
    
    async def _calculate_value_at_risk(self, confidence: float = 0.95):
        """Calculate Value at Risk (VaR)"""
        try:
            if len(self.recent_trades) < 10:
                self.risk_metrics['var_95'] = 0.0
                return
            
            # Get returns
            returns = [trade.get('pnl_percentage', 0) for trade in self.recent_trades[-30:]]
            
            if returns:
                # Calculate VaR using historical simulation
                sorted_returns = sorted(returns)
                var_index = int((1 - confidence) * len(sorted_returns))
                var_95 = abs(sorted_returns[var_index]) if var_index < len(sorted_returns) else 0
                
                self.risk_metrics['var_95'] = var_95
            
        except Exception as e:
            logger.error(f"Error calculating VaR: {e}")
            self.risk_metrics['var_95'] = 0.0
    
    def _create_error_assessment(self, symbol: str, error_message: str) -> RiskAssessment:
        """Create error assessment when evaluation fails"""
        return RiskAssessment(
            risk_level=RiskLevel.HIGH,
            risk_score=75.0,
            confidence=0.0,
            position_risk=75.0,
            portfolio_risk=50.0,
            market_risk=50.0,
            correlation_risk=25.0,
            liquidity_risk=25.0,
            max_position_size=0.0,
            recommended_position_size=0.0,
            max_portfolio_exposure=0.0,
            stop_loss_recommendation=0.0,
            position_size_adjustment=0.0,
            reasoning=[f"Risk assessment failed: {error_message}"],
            warnings=["Risk evaluation error - manual review required"]
        )
    
    async def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary"""
        try:
            circuit_status = await self.check_circuit_breakers()
            
            summary = {
                'risk_metrics': self.risk_metrics.copy(),
                'circuit_breaker_status': circuit_status,
                'circuit_breaker_state': self.circuit_breaker_state.to_dict(),
                'volatility_metrics': self.volatility_metrics.copy(),
                'recent_assessments_count': len(self.assessment_history),
                'consecutive_losses': self.consecutive_losses,
                'recent_trades_count': len(self.recent_trades),
                'risk_limits': self.circuit_breakers.copy(),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Add performance statistics
            if self.assessment_history:
                risk_scores = [a.risk_score for a in self.assessment_history[-20:]]
                summary['avg_risk_score'] = np.mean(risk_scores)
                summary['max_risk_score'] = max(risk_scores)
                summary['min_risk_score'] = min(risk_scores)
            
            # Add recent trade statistics
            if self.recent_trades:
                recent_pnl = [t.get('pnl', 0) for t in self.recent_trades[-10:]]
                summary['recent_avg_pnl'] = np.mean(recent_pnl)
                summary['recent_total_pnl'] = sum(recent_pnl)
                summary['recent_win_rate'] = len([p for p in recent_pnl if p > 0]) / len(recent_pnl) * 100
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting risk summary: {e}")
            return {'error': str(e)}
    
    async def update_volatility_metrics(self, market_data: Dict[str, Any]):
        """Update volatility metrics with market data"""
        try:
            # This would typically calculate from OHLCV data
            # For now, use mock calculation
            
            current_vol = market_data.get('volatility', 0.02)  # 2% default
            
            # Update moving average of volatility
            alpha = 0.1  # Smoothing factor
            old_avg = self.volatility_metrics['average_volatility']
            new_avg = alpha * current_vol + (1 - alpha) * old_avg
            
            self.volatility_metrics.update({
                'current_volatility': current_vol,
                'average_volatility': new_avg,
                'last_volatility_update': datetime.utcnow()
            })
            
        except Exception as e:
            logger.error(f"Error updating volatility metrics: {e}")
    
    async def reset_circuit_breakers(self, admin_override: bool = False) -> bool:
        """Reset circuit breakers (admin function)"""
        try:
            if not admin_override:
                # Check if enough time has passed
                if (self.circuit_breaker_state.cooldown_until and 
                    datetime.utcnow() < self.circuit_breaker_state.cooldown_until):
                    logger.warning("Cannot reset circuit breakers - still in cooldown period")
                    return False
            
            # Reset circuit breaker state
            self.circuit_breaker_state = CircuitBreakerState(
                status=CircuitBreakerStatus.NORMAL,
                trigger_count=0,
                last_trigger_time=None,
                cooldown_until=None
            )
            
            # Log reset
            with get_db_session() as session:
                log_entry = SystemLog(
                    level=LogLevel.INFO,
                    component='RISK_MANAGER',
                    message="Circuit breakers reset",
                    details={'admin_override': admin_override}
                )
                session.add(log_entry)
            
            logger.info(f"Circuit breakers reset (admin_override: {admin_override})")
            return True
            
        except Exception as e:
            logger.error(f"Error resetting circuit breakers: {e}")
            return False
    
    async def get_risk_health_check(self) -> Dict[str, Any]:
        """Perform risk management system health check"""
        try:
            health = {
                'status': 'healthy',
                'checks': {},
                'warnings': [],
                'errors': []
            }
            
            # Check circuit breaker functionality
            circuit_status = await self.check_circuit_breakers()
            health['checks']['circuit_breakers'] = {
                'status': 'ok',
                'current_status': circuit_status.get('status', 'UNKNOWN')
            }
            
            # Check risk metrics freshness
            last_update = self.risk_metrics.get('last_updated')
            if last_update:
                time_since_update = datetime.utcnow() - last_update
                if time_since_update > timedelta(hours=1):
                    health['warnings'].append("Risk metrics not updated in over 1 hour")
                health['checks']['metrics_freshness'] = {
                    'status': 'ok' if time_since_update < timedelta(hours=1) else 'warning',
                    'last_update': last_update.isoformat(),
                    'time_since_update_minutes': time_since_update.total_seconds() / 60
                }
            else:
                health['warnings'].append("No risk metrics timestamp found")
            
            # Check assessment history
            if len(self.assessment_history) == 0:
                health['warnings'].append("No risk assessments in history")
            
            health['checks']['assessment_history'] = {
                'status': 'ok' if len(self.assessment_history) > 0 else 'warning',
                'count': len(self.assessment_history)
            }
            
            # Overall health status
            if health['errors']:
                health['status'] = 'unhealthy'
            elif health['warnings']:
                health['status'] = 'warning'
            
            health['timestamp'] = datetime.utcnow().isoformat()
            
            return health
            
        except Exception as e:
            logger.error(f"Error in risk health check: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def cleanup(self):
        """Cleanup risk manager resources"""
        try:
            # Save final metrics to database if needed
            logger.info("Risk manager cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during risk manager cleanup: {e}")

# Factory function
def create_risk_manager(trading_config) -> RiskManager:
    """Factory function to create Risk Manager instance"""
    return RiskManager(trading_config)

# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    # Mock trading config for testing
    class MockTradingConfig:
        def __init__(self):
            self.max_risk_per_trade = 0.02  # 2%
            self.max_open_positions = 5
            self.daily_loss_limit = 0.05    # 5%
            self.max_consecutive_losses = 3
    
    async def test_risk_manager():
        print("⚠️ Risk Manager Test")
        print("=" * 50)
        
        try:
            # Create risk manager
            trading_config = MockTradingConfig()
            risk_manager = create_risk_manager(trading_config)
            
            print(f"✅ Risk Manager created")
            print(f"📊 Circuit breakers: {risk_manager.circuit_breakers}")
            print(f"🎯 Risk limits: Daily loss {trading_config.daily_loss_limit:.1%}, "
                  f"Max positions {trading_config.max_open_positions}")
            
            # Test 1: Low risk trade proposal
            print(f"\n🧪 Test 1: Low Risk Trade Proposal")
            
            low_risk_signal = {
                'symbol': 'BTC/USDT',
                'entry_price': 50000,
                'stop_loss': 49000,  # 2% stop loss
                'take_profit': 52000,  # 4% take profit
                'position_size': 0.1,
                'confidence': 85,
                'direction': 'LONG',
                'risk_reward_ratio': 2.0,
                'volume_24h': 200_000_000,
                'market_condition': 'TRENDING'
            }
            
            portfolio_context = {
                'total_value': 10000,
                'available_balance': 5000,
                'current_positions': []
            }
            
            assessment = await risk_manager.evaluate_trade_proposal(
                'BTC/USDT', low_risk_signal, portfolio_context
            )
            
            print(f"  Risk Level: {assessment.risk_level.value}")
            print(f"  Risk Score: {assessment.risk_score:.1f}/100")
            print(f"  Confidence: {assessment.confidence:.1f}%")
            print(f"  Recommended Size: {assessment.recommended_position_size:.4f}")
            print(f"  Position Risk: {assessment.position_risk:.1f}")
            print(f"  Portfolio Risk: {assessment.portfolio_risk:.1f}")
            print(f"  Reasoning: {assessment.reasoning[:2]}")
            
            # Test 2: High risk trade proposal
            print(f"\n🧪 Test 2: High Risk Trade Proposal")
            
            high_risk_signal = {
                'symbol': 'NEWCOIN/USDT',
                'entry_price': 0.05,
                'stop_loss': 0.03,   # 40% stop loss!
                'take_profit': 0.06,
                'position_size': 50000,  # Large position
                'confidence': 45,    # Low confidence
                'direction': 'LONG',
                'risk_reward_ratio': 0.5,  # Poor R/R
                'volume_24h': 500_000,  # Low volume
                'market_condition': 'VOLATILE'
            }
            
            high_risk_assessment = await risk_manager.evaluate_trade_proposal(
                'NEWCOIN/USDT', high_risk_signal, portfolio_context
            )
            
            print(f"  Risk Level: {high_risk_assessment.risk_level.value}")
            print(f"  Risk Score: {high_risk_assessment.risk_score:.1f}/100")
            print(f"  Confidence: {high_risk_assessment.confidence:.1f}%")
            print(f"  Recommended Size: {high_risk_assessment.recommended_position_size:.4f}")
            print(f"  Size Adjustment: {high_risk_assessment.position_size_adjustment:.1%}")
            print(f"  Warnings: {len(high_risk_assessment.warnings)}")
            
            # Test 3: Portfolio concentration risk
            print(f"\n🧪 Test 3: Portfolio Concentration Risk")
            
            crowded_portfolio = {
                'total_value': 10000,
                'available_balance': 1000,  # Low available balance
                'current_positions': [
                    {'symbol': 'BTC/USDT', 'market_value': 3000, 'status': 'OPEN'},
                    {'symbol': 'ETH/USDT', 'market_value': 2500, 'status': 'OPEN'},
                    {'symbol': 'ADA/USDT', 'market_value': 2000, 'status': 'OPEN'},
                    {'symbol': 'DOT/USDT', 'market_value': 1500, 'status': 'OPEN'}
                ]
            }
            
            concentration_assessment = await risk_manager.evaluate_trade_proposal(
                'SOL/USDT', low_risk_signal, crowded_portfolio
            )
            
            print(f"  Risk Level: {concentration_assessment.risk_level.value}")
            print(f"  Portfolio Risk: {concentration_assessment.portfolio_risk:.1f}")
            print(f"  Correlation Risk: {concentration_assessment.correlation_risk:.1f}")
            print(f"  Max Exposure: {concentration_assessment.max_portfolio_exposure:.1%}")
            print(f"  Warnings: {concentration_assessment.warnings}")
            
            # Test 4: Circuit breaker functionality
            print(f"\n🧪 Test 4: Circuit Breaker Testing")
            
            # Simulate some losses to trigger circuit breakers
            for i in range(4):
                loss_trade = {'pnl': -200, 'pnl_percentage': -2.0}
                await risk_manager.update_risk_metrics(loss_trade)
            
            # Update daily P&L to trigger daily loss limit
            risk_manager.risk_metrics['daily_pnl'] = -600  # 6% loss on $10k portfolio
            
            circuit_status = await risk_manager.check_circuit_breakers(portfolio_context)
            print(f"  Circuit Breaker Status: {circuit_status['status']}")
            print(f"  Triggered: {circuit_status['triggered']}")
            
            if circuit_status['triggered']:
                print(f"  Triggers: {len(circuit_status['triggers'])}")
                for trigger in circuit_status['triggers']:
                    print(f"    - {trigger['type']}: {trigger['value']:.1%} "
                          f"(limit: {abs(trigger['threshold']):.1%})")
            
            # Test 5: Risk summary
            print(f"\n🧪 Test 5: Risk Summary")
            
            risk_summary = await risk_manager.get_risk_summary()
            print(f"  Total Assessments: {risk_summary['recent_assessments_count']}")
            print(f"  Recent Trades: {risk_summary['recent_trades_count']}")
            print(f"  Consecutive Losses: {risk_summary['consecutive_losses']}")
            print(f"  Current Drawdown: {risk_summary['risk_metrics']['current_drawdown']:.1%}")
            print(f"  VaR 95%: {risk_summary['risk_metrics']['var_95']:.1%}")
            
            if 'avg_risk_score' in risk_summary:
                print(f"  Average Risk Score: {risk_summary['avg_risk_score']:.1f}")
                print(f"  Recent Win Rate: {risk_summary.get('recent_win_rate', 0):.1f}%")
            
            # Test 6: Health check
            print(f"\n🧪 Test 6: Health Check")
            
            health = await risk_manager.get_risk_health_check()
            print(f"  Health Status: {health['status']}")
            print(f"  Checks: {len(health['checks'])}")
            print(f"  Warnings: {len(health['warnings'])}")
            
            for check_name, check_result in health['checks'].items():
                print(f"    {check_name}: {check_result['status']}")
            
            if health['warnings']:
                print(f"  Warning messages:")
                for warning in health['warnings']:
                    print(f"    - {warning}")
            
            # Test 7: Circuit breaker reset
            print(f"\n🧪 Test 7: Circuit Breaker Reset")
            
            reset_success = await risk_manager.reset_circuit_breakers(admin_override=True)
            print(f"  Reset successful: {reset_success}")
            
            if reset_success:
                new_status = await risk_manager.check_circuit_breakers()
                print(f"  New status: {new_status['status']}")
                print(f"  Triggers after reset: {len(new_status['triggers'])}")
            
            # Test 8: Position sizing optimization
            print(f"\n🧪 Test 8: Position Sizing Optimization")
            
            # Test different confidence levels
            confidence_levels = [30, 50, 70, 90]
            
            for confidence in confidence_levels:
                test_signal = low_risk_signal.copy()
                test_signal['confidence'] = confidence
                
                sizing_assessment = await risk_manager.evaluate_trade_proposal(
                    'BTC/USDT', test_signal, portfolio_context
                )
                
                print(f"  Confidence {confidence}%: "
                      f"Size {sizing_assessment.recommended_position_size:.4f}, "
                      f"Risk {sizing_assessment.risk_score:.1f}")
            
            # Test 9: Volatility impact
            print(f"\n🧪 Test 9: Volatility Impact Testing")
            
            # Update volatility metrics
            low_vol_market = {'volatility': 0.01}  # 1% volatility
            await risk_manager.update_volatility_metrics(low_vol_market)
            
            low_vol_assessment = await risk_manager.evaluate_trade_proposal(
                'BTC/USDT', low_risk_signal, portfolio_context
            )
            
            high_vol_market = {'volatility': 0.08}  # 8% volatility
            await risk_manager.update_volatility_metrics(high_vol_market)
            
            high_vol_assessment = await risk_manager.evaluate_trade_proposal(
                'BTC/USDT', low_risk_signal, portfolio_context
            )
            
            print(f"  Low Volatility (1%): Risk Score {low_vol_assessment.risk_score:.1f}")
            print(f"  High Volatility (8%): Risk Score {high_vol_assessment.risk_score:.1f}")
            print(f"  Volatility Impact: {high_vol_assessment.risk_score - low_vol_assessment.risk_score:.1f} points")
            
            # Test 10: Risk assessment serialization
            print(f"\n🧪 Test 10: Risk Assessment Serialization")
            
            assessment_dict = assessment.to_dict()
            print(f"  Assessment keys: {len(assessment_dict.keys())}")
            print(f"  Serializable: {isinstance(assessment_dict, dict)}")
            
            # Key metrics from serialized data
            key_metrics = {
                'risk_level': assessment_dict['risk_level'],
                'risk_score': assessment_dict['risk_score'],
                'confidence': assessment_dict['confidence'],
                'recommended_size': assessment_dict['recommended_position_size']
            }
            print(f"  Key metrics: {key_metrics}")
            
            print(f"\n🎉 Risk Manager test completed successfully!")
            
            # Performance summary
            print(f"\n📈 Performance Summary:")
            print(f"  Assessments completed: {len(risk_manager.assessment_history)}")
            print(f"  Circuit breaker triggers: {risk_manager.circuit_breaker_state.trigger_count}")
            print(f"  Risk metrics tracked: {len(risk_manager.risk_metrics)}")
            
            # Risk level distribution
            risk_levels = [a.risk_level.value for a in risk_manager.assessment_history]
            level_counts = {level: risk_levels.count(level) for level in set(risk_levels)}
            print(f"  Risk level distribution: {level_counts}")
            
        except Exception as e:
            print(f"❌ Error in Risk Manager test: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Cleanup
            try:
                await risk_manager.cleanup()
                print(f"🧹 Risk Manager cleanup completed")
            except:
                pass
    
