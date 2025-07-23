"""
Claude AI Market Analyzer

This module integrates Claude AI for advanced market analysis, sentiment analysis,
and trading decision support using natural language processing.
"""

import asyncio
import aiohttp
import logging
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

import pandas as pd
import numpy as np

from database.connection import get_db_session
from database.models import AIAnalysisLog, Trade, MarketCondition
from config.settings import config

logger = logging.getLogger(__name__)

class AnalysisType(Enum):
    """Types of AI analysis"""
    MARKET_SENTIMENT = "MARKET_SENTIMENT"
    TRADING_OPPORTUNITY = "TRADING_OPPORTUNITY"
    RISK_ASSESSMENT = "RISK_ASSESSMENT"
    NEWS_ANALYSIS = "NEWS_ANALYSIS"
    TECHNICAL_ANALYSIS = "TECHNICAL_ANALYSIS"
    PORTFOLIO_REVIEW = "PORTFOLIO_REVIEW"

class MarketSentiment(Enum):
    """Market sentiment classifications"""
    EXTREMELY_BULLISH = "EXTREMELY_BULLISH"
    BULLISH = "BULLISH"
    NEUTRAL = "NEUTRAL"
    BEARISH = "BEARISH"
    EXTREMELY_BEARISH = "EXTREMELY_BEARISH"

@dataclass
class AIAnalysisResult:
    """AI Analysis result structure"""
    analysis_type: AnalysisType
    symbol: str
    confidence: float  # 0-100
    sentiment: Optional[MarketSentiment] = None
    key_insights: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    opportunities: List[str] = field(default_factory=list)
    market_condition: Optional[str] = None
    price_targets: Dict[str, float] = field(default_factory=dict)  # support, resistance, target
    timeframe: str = "SHORT_TERM"  # SHORT_TERM, MEDIUM_TERM, LONG_TERM
    raw_response: str = ""
    processing_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'analysis_type': self.analysis_type.value,
            'symbol': self.symbol,
            'confidence': self.confidence,
            'sentiment': self.sentiment.value if self.sentiment else None,
            'key_insights': self.key_insights,
            'recommendations': self.recommendations,
            'risk_factors': self.risk_factors,
            'opportunities': self.opportunities,
            'market_condition': self.market_condition,
            'price_targets': self.price_targets,
            'timeframe': self.timeframe,
            'raw_response': self.raw_response,
            'processing_time': self.processing_time,
            'timestamp': self.timestamp.isoformat()
        }

class ClaudeAnalyzer:
    """Claude AI Market Analyzer"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or config.api.claude_api_key
        self.base_url = "https://api.anthropic.com/v1/messages"
        self.model = "claude-sonnet-4-20250514"
        self.max_tokens = 2000
        self.session = None
        
        # Analysis history for learning
        self.analysis_history: List[AIAnalysisResult] = []
        self.performance_feedback: Dict[str, Dict[str, Any]] = {}
        
        # Rate limiting
        self.request_count = 0
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Minimum 1 second between requests
        
        # Analysis templates
        self.analysis_templates = {
            AnalysisType.MARKET_SENTIMENT: self._get_sentiment_analysis_template(),
            AnalysisType.TRADING_OPPORTUNITY: self._get_trading_opportunity_template(),
            AnalysisType.RISK_ASSESSMENT: self._get_risk_assessment_template(),
            AnalysisType.TECHNICAL_ANALYSIS: self._get_technical_analysis_template(),
            AnalysisType.PORTFOLIO_REVIEW: self._get_portfolio_review_template()
        }
        
        logger.info("Claude Analyzer initialized")
    
    async def initialize(self):
        """Initialize the analyzer"""
        try:
            if not self.api_key:
                raise ValueError("Claude API key not configured")
            
            # Create aiohttp session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
            
            # Test API connectivity
            test_result = await self._test_api_connection()
            if not test_result:
                raise ConnectionError("Could not connect to Claude API")
            
            # Load analysis history
            await self._load_analysis_history()
            
            logger.info("Claude Analyzer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Claude Analyzer: {e}")
            raise
    
    async def _test_api_connection(self) -> bool:
        """Test Claude API connection"""
        try:
            test_prompt = "Hello! Please respond with 'API connection successful' if you can see this message."
            
            payload = {
                "model": self.model,
                "max_tokens": 50,
                "messages": [
                    {"role": "user", "content": test_prompt}
                ]
            }
            
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01"
            }
            
            async with self.session.post(self.base_url, json=payload, headers=headers) as response:
                if response.status == 200:
                    result = await response.json()
                    content = result.get('content', [])
                    if content and 'API connection successful' in content[0].get('text', ''):
                        return True
                
                logger.error(f"API test failed with status {response.status}")
                return False
                
        except Exception as e:
            logger.error(f"API connection test error: {e}")
            return False
    
    async def _load_analysis_history(self):
        """Load recent analysis history from database"""
        try:
            with get_db_session() as session:
                recent_analyses = session.query(AIAnalysisLog).filter(
                    AIAnalysisLog.timestamp >= datetime.utcnow() - timedelta(days=7)
                ).order_by(AIAnalysisLog.timestamp.desc()).limit(50).all()
                
                for analysis in recent_analyses:
                    try:
                        if analysis.processed_response:
                            # Convert stored analysis back to AIAnalysisResult
                            data = analysis.processed_response
                            result = AIAnalysisResult(
                                analysis_type=AnalysisType(data.get('analysis_type', 'MARKET_SENTIMENT')),
                                symbol=analysis.symbol,
                                confidence=data.get('confidence', 0),
                                sentiment=MarketSentiment(data['sentiment']) if data.get('sentiment') else None,
                                key_insights=data.get('key_insights', []),
                                recommendations=data.get('recommendations', []),
                                risk_factors=data.get('risk_factors', []),
                                opportunities=data.get('opportunities', []),
                                timestamp=analysis.timestamp
                            )
                            self.analysis_history.append(result)
                    except Exception as e:
                        logger.warning(f"Could not load analysis history entry: {e}")
                
                logger.info(f"Loaded {len(self.analysis_history)} analysis history entries")
                
        except Exception as e:
            logger.warning(f"Could not load analysis history: {e}")
    
    async def analyze_market_sentiment(self, 
                                     symbol: str,
                                     market_data: pd.DataFrame,
                                     current_price: float,
                                     additional_context: Dict[str, Any] = None) -> AIAnalysisResult:
        """Analyze overall market sentiment for a symbol"""
        
        try:
            # Prepare market data context
            context = self._prepare_market_context(symbol, market_data, current_price, additional_context)
            
            # Create analysis prompt
            prompt = self.analysis_templates[AnalysisType.MARKET_SENTIMENT].format(
                symbol=symbol,
                current_price=current_price,
                context=context,
                recent_performance=self._get_recent_performance_summary(),
                timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            )
            
            # Get AI analysis
            ai_response = await self._call_claude_api(prompt)
            
            if not ai_response:
                raise Exception("Failed to get AI response")
            
            # Parse AI response
            result = await self._parse_sentiment_analysis(ai_response, symbol)
            
            # Store analysis
            await self._store_analysis_result(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing market sentiment for {symbol}: {e}")
            return self._create_error_result(AnalysisType.MARKET_SENTIMENT, symbol, str(e))
    
    async def analyze_trading_opportunity(self,
                                        symbol: str,
                                        market_data: pd.DataFrame,
                                        current_price: float,
                                        strategy_signals: List[Dict[str, Any]] = None,
                                        portfolio_context: Dict[str, Any] = None) -> AIAnalysisResult:
        """Analyze specific trading opportunities"""
        
        try:
            # Prepare comprehensive context
            context = self._prepare_market_context(symbol, market_data, current_price)
            
            # Add strategy signals context
            strategy_context = ""
            if strategy_signals:
                strategy_context = self._format_strategy_signals(strategy_signals)
            
            # Portfolio context
            portfolio_info = ""
            if portfolio_context:
                portfolio_info = self._format_portfolio_context(portfolio_context)
            
            # Create analysis prompt
            prompt = self.analysis_templates[AnalysisType.TRADING_OPPORTUNITY].format(
                symbol=symbol,
                current_price=current_price,
                context=context,
                strategy_signals=strategy_context,
                portfolio_info=portfolio_info,
                risk_tolerance=portfolio_context.get('risk_tolerance', 'MEDIUM') if portfolio_context else 'MEDIUM',
                timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            )
            
            # Get AI analysis
            ai_response = await self._call_claude_api(prompt)
            
            if not ai_response:
                raise Exception("Failed to get AI response")
            
            # Parse AI response
            result = await self._parse_trading_opportunity(ai_response, symbol)
            
            # Store analysis
            await self._store_analysis_result(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing trading opportunity for {symbol}: {e}")
            return self._create_error_result(AnalysisType.TRADING_OPPORTUNITY, symbol, str(e))
    
    async def analyze_risk_assessment(self,
                                    symbol: str,
                                    position_data: Dict[str, Any],
                                    market_data: pd.DataFrame,
                                    portfolio_context: Dict[str, Any] = None) -> AIAnalysisResult:
        """Analyze risk factors for a position or potential trade"""
        
        try:
            # Prepare context
            context = self._prepare_market_context(symbol, market_data, position_data.get('current_price', 0))
            position_context = self._format_position_context(position_data)
            portfolio_info = self._format_portfolio_context(portfolio_context) if portfolio_context else ""
            
            # Create risk assessment prompt
            prompt = self.analysis_templates[AnalysisType.RISK_ASSESSMENT].format(
                symbol=symbol,
                position_context=position_context,
                market_context=context,
                portfolio_info=portfolio_info,
                timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            )
            
            # Get AI analysis
            ai_response = await self._call_claude_api(prompt)
            
            if not ai_response:
                raise Exception("Failed to get AI response")
            
            # Parse AI response
            result = await self._parse_risk_assessment(ai_response, symbol)
            
            # Store analysis
            await self._store_analysis_result(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing risk assessment for {symbol}: {e}")
            return self._create_error_result(AnalysisType.RISK_ASSESSMENT, symbol, str(e))
    
    async def analyze_portfolio_review(self,
                                     portfolio_data: Dict[str, Any],
                                     market_overview: Dict[str, Any] = None) -> AIAnalysisResult:
        """Analyze overall portfolio and provide recommendations"""
        
        try:
            # Format portfolio data
            portfolio_context = self._format_comprehensive_portfolio_context(portfolio_data)
            market_context = self._format_market_overview_context(market_overview) if market_overview else ""
            
            # Create portfolio review prompt
            prompt = self.analysis_templates[AnalysisType.PORTFOLIO_REVIEW].format(
                portfolio_context=portfolio_context,
                market_overview=market_context,
                timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            )
            
            # Get AI analysis
            ai_response = await self._call_claude_api(prompt)
            
            if not ai_response:
                raise Exception("Failed to get AI response")
            
            # Parse AI response
            result = await self._parse_portfolio_review(ai_response, "PORTFOLIO")
            
            # Store analysis
            await self._store_analysis_result(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing portfolio review: {e}")
            return self._create_error_result(AnalysisType.PORTFOLIO_REVIEW, "PORTFOLIO", str(e))
    
    async def _call_claude_api(self, prompt: str) -> Optional[str]:
        """Make API call to Claude"""
        
        try:
            # Rate limiting
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            
            if time_since_last < self.min_request_interval:
                await asyncio.sleep(self.min_request_interval - time_since_last)
            
            start_time = time.time()
            
            # Prepare API request
            payload = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
            
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01"
            }
            
            # Make API call
            async with self.session.post(self.base_url, json=payload, headers=headers) as response:
                processing_time = time.time() - start_time
                self.last_request_time = time.time()
                self.request_count += 1
                
                if response.status == 200:
                    result = await response.json()
                    content = result.get('content', [])
                    
                    if content and len(content) > 0:
                        ai_response = content[0].get('text', '')
                        
                        # Log API usage
                        logger.debug(f"Claude API call successful - Processing time: {processing_time:.2f}s, "
                                   f"Response length: {len(ai_response)}")
                        
                        return ai_response
                    else:
                        logger.error("Empty response from Claude API")
                        return None
                
                elif response.status == 429:
                    logger.warning("Claude API rate limit exceeded")
                    await asyncio.sleep(5)  # Wait 5 seconds before retry
                    return None
                
                else:
                    error_text = await response.text()
                    logger.error(f"Claude API error {response.status}: {error_text}")
                    return None
                    
        except asyncio.TimeoutError:
            logger.error("Claude API timeout")
            return None
        except Exception as e:
            logger.error(f"Error calling Claude API: {e}")
            return None
    
    def _prepare_market_context(self, 
                              symbol: str, 
                              market_data: pd.DataFrame, 
                              current_price: float,
                              additional_context: Dict[str, Any] = None) -> str:
        """Prepare market context for AI analysis"""
        
        try:
            if market_data.empty:
                return f"Symbol: {symbol}, Current Price: ${current_price:,.2f}"
            
            # Calculate basic metrics
            recent_data = market_data.tail(24)  # Last 24 periods
            
            price_change_24h = ((current_price - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0]) * 100
            high_24h = recent_data['high'].max()
            low_24h = recent_data['low'].min()
            volume_24h = recent_data['volume'].sum()
            avg_volume = recent_data['volume'].mean()
            
            # Volatility
            returns = recent_data['close'].pct_change().dropna()
            volatility = returns.std() * 100
            
            # Support/Resistance levels
            resistance = recent_data['high'].quantile(0.9)
            support = recent_data['low'].quantile(0.1)
            
            context = f"""
Market Data for {symbol}:
- Current Price: ${current_price:,.2f}
- 24h Change: {price_change_24h:+.2f}%
- 24h High: ${high_24h:,.2f}
- 24h Low: ${low_24h:,.2f}
- 24h Volume: {volume_24h:,.0f}
- Average Volume: {avg_volume:,.0f}
- Volatility: {volatility:.2f}%
- Support Level: ${support:,.2f}
- Resistance Level: ${resistance:,.2f}
"""
            
            # Add technical indicators if available
            if len(market_data) >= 20:
                sma_20 = recent_data['close'].rolling(20).mean().iloc[-1]
                context += f"- 20-period SMA: ${sma_20:,.2f}\n"
                
                # Price relative to SMA
                sma_distance = ((current_price - sma_20) / sma_20) * 100
                context += f"- Distance from SMA20: {sma_distance:+.2f}%\n"
            
            # Add additional context if provided
            if additional_context:
                context += "\nAdditional Context:\n"
                for key, value in additional_context.items():
                    context += f"- {key}: {value}\n"
            
            return context
            
        except Exception as e:
            logger.error(f"Error preparing market context: {e}")
            return f"Symbol: {symbol}, Current Price: ${current_price:,.2f}"
    
    def _format_strategy_signals(self, strategy_signals: List[Dict[str, Any]]) -> str:
        """Format strategy signals for AI context"""
        
        if not strategy_signals:
            return "No active strategy signals."
        
        context = "Active Strategy Signals:\n"
        for signal in strategy_signals:
            context += f"- {signal.get('strategy_id', 'Unknown')}: {signal.get('signal_type', 'N/A')} "
            context += f"(Confidence: {signal.get('confidence', 0):.1f}%)\n"
            if signal.get('reasoning'):
                context += f"  Reasoning: {signal.get('reasoning')}\n"
        
        return context
    
    def _format_portfolio_context(self, portfolio_context: Dict[str, Any]) -> str:
        """Format portfolio context for AI analysis"""
        
        if not portfolio_context:
            return ""
        
        context = "Portfolio Context:\n"
        if 'total_value' in portfolio_context:
            context += f"- Total Portfolio Value: ${portfolio_context['total_value']:,.2f}\n"
        if 'available_balance' in portfolio_context:
            context += f"- Available Balance: ${portfolio_context['available_balance']:,.2f}\n"
        if 'current_positions' in portfolio_context:
            context += f"- Current Positions: {len(portfolio_context['current_positions'])}\n"
        if 'daily_pnl' in portfolio_context:
            context += f"- Daily P&L: ${portfolio_context['daily_pnl']:,.2f}\n"
        if 'risk_level' in portfolio_context:
            context += f"- Risk Tolerance: {portfolio_context['risk_level']}\n"
        
        return context
    
    def _format_position_context(self, position_data: Dict[str, Any]) -> str:
        """Format position context for risk analysis"""
        
        context = "Position Details:\n"
        if 'direction' in position_data:
            context += f"- Direction: {position_data['direction']}\n"
        if 'entry_price' in position_data:
            context += f"- Entry Price: ${position_data['entry_price']:,.2f}\n"
        if 'current_price' in position_data:
            context += f"- Current Price: ${position_data['current_price']:,.2f}\n"
        if 'quantity' in position_data:
            context += f"- Quantity: {position_data['quantity']}\n"
        if 'unrealized_pnl' in position_data:
            context += f"- Unrealized P&L: ${position_data['unrealized_pnl']:,.2f}\n"
        if 'stop_loss' in position_data:
            context += f"- Stop Loss: ${position_data['stop_loss']:,.2f}\n"
        if 'take_profit' in position_data:
            context += f"- Take Profit: ${position_data['take_profit']:,.2f}\n"
        
        return context
    
    def _format_comprehensive_portfolio_context(self, portfolio_data: Dict[str, Any]) -> str:
        """Format comprehensive portfolio context"""
        
        context = "Portfolio Overview:\n"
        
        # Basic metrics
        if 'total_value' in portfolio_data:
            context += f"- Total Value: ${portfolio_data['total_value']:,.2f}\n"
        if 'cash_balance' in portfolio_data:
            context += f"- Cash Balance: ${portfolio_data['cash_balance']:,.2f}\n"
        if 'invested_amount' in portfolio_data:
            context += f"- Invested Amount: ${portfolio_data['invested_amount']:,.2f}\n"
        
        # Performance metrics
        if 'total_return' in portfolio_data:
            context += f"- Total Return: {portfolio_data['total_return']:+.2f}%\n"
        if 'daily_pnl' in portfolio_data:
            context += f"- Daily P&L: ${portfolio_data['daily_pnl']:+,.2f}\n"
        if 'win_rate' in portfolio_data:
            context += f"- Win Rate: {portfolio_data['win_rate']:.1f}%\n"
        
        # Positions
        if 'positions' in portfolio_data:
            context += f"\nCurrent Positions ({len(portfolio_data['positions'])}):\n"
            for position in portfolio_data['positions']:
                symbol = position.get('symbol', 'Unknown')
                direction = position.get('direction', 'Unknown')
                pnl = position.get('unrealized_pnl', 0)
                context += f"- {symbol} {direction}: ${pnl:+,.2f} P&L\n"
        
        return context
    
    def _format_market_overview_context(self, market_overview: Dict[str, Any]) -> str:
        """Format market overview context"""
        
        context = "Market Overview:\n"
        
        if 'btc_dominance' in market_overview:
            context += f"- BTC Dominance: {market_overview['btc_dominance']:.1f}%\n"
        if 'total_market_cap' in market_overview:
            context += f"- Total Market Cap: ${market_overview['total_market_cap']/1e9:.1f}B\n"
        if 'market_sentiment' in market_overview:
            context += f"- Market Sentiment: {market_overview['market_sentiment']}\n"
        if 'fear_greed_index' in market_overview:
            context += f"- Fear & Greed Index: {market_overview['fear_greed_index']}\n"
        if 'volatility_index' in market_overview:
            context += f"- Volatility Index: {market_overview['volatility_index']}\n"
        
        return context
    
    def _get_recent_performance_summary(self) -> str:
        """Get recent analysis performance summary"""
        
        if not self.analysis_history:
            return "No recent analysis history available."
        
        recent_analyses = [a for a in self.analysis_history if a.timestamp >= datetime.utcnow() - timedelta(hours=24)]
        
        if not recent_analyses:
            return "No analysis performed in the last 24 hours."
        
        avg_confidence = np.mean([a.confidence for a in recent_analyses])
        analysis_types = [a.analysis_type.value for a in recent_analyses]
        type_counts = {t: analysis_types.count(t) for t in set(analysis_types)}
        
        summary = f"Recent Analysis Performance (24h): {len(recent_analyses)} analyses, "
        summary += f"Average Confidence: {avg_confidence:.1f}%. "
        summary += f"Analysis Types: {type_counts}"
        
        return summary
    
    # Analysis Templates
    def _get_sentiment_analysis_template(self) -> str:
        return """
As an expert cryptocurrency market analyst, analyze the market sentiment for {symbol}.

Market Data:
{context}

Recent Analysis Performance:
{recent_performance}

Current Time: {timestamp}

Please provide a comprehensive market sentiment analysis including:

1. Overall Market Sentiment (EXTREMELY_BULLISH, BULLISH, NEUTRAL, BEARISH, EXTREMELY_BEARISH)
2. Confidence Level (0-100)
3. Key Market Insights (3-5 points)
4. Risk Factors (2-4 points)
5. Opportunities (2-4 points)
6. Short-term Price Targets (support, resistance, target)
7. Timeframe for analysis (SHORT_TERM, MEDIUM_TERM, LONG_TERM)

Format your response as JSON:
{{
    "sentiment": "BULLISH",
    "confidence": 75,
    "key_insights": ["insight1", "insight2", "insight3"],
    "recommendations": ["rec1", "rec2"],
    "risk_factors": ["risk1", "risk2"],
    "opportunities": ["opp1", "opp2"],
    "price_targets": {{"support": 45000, "resistance": 52000, "target": 50000}},
    "timeframe": "SHORT_TERM",
    "market_condition": "CONSOLIDATING"
}}

Provide actionable insights based on the technical data and market conditions.
"""
    
    def _get_trading_opportunity_template(self) -> str:
        return """
As an expert cryptocurrency trading analyst, evaluate the trading opportunity for {symbol}.

Market Data:
{context}

Strategy Signals:
{strategy_signals}

Portfolio Information:
{portfolio_info}

Risk Tolerance: {risk_tolerance}
Current Time: {timestamp}

Please analyze this trading opportunity and provide:

1. Trade Recommendation (STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL)
2. Confidence Level (0-100)
3. Entry Strategy and Price Levels
4. Risk Management (Stop Loss, Take Profit)
5. Position Sizing Recommendation
6. Key Supporting Factors
7. Risk Factors to Consider
8. Alternative Scenarios

Format your response as JSON:
{{
    "trade_recommendation": "BUY",
    "confidence": 80,
    "key_insights": ["Strong technical setup", "Volume confirmation"],
    "recommendations": ["Enter at $48,500-$49,000", "Set stop loss at $47,000"],
    "risk_factors": ["High volatility", "Resistance at $52,000"],
    "opportunities": ["Breakout potential", "Support holding"],
    "price_targets": {{"entry": 48750, "stop_loss": 47000, "take_profit": 52000}},
    "timeframe": "SHORT_TERM",
    "market_condition": "BULLISH_SETUP"
}}

Focus on actionable trading insights with clear risk/reward analysis.
"""
    
    def _get_risk_assessment_template(self) -> str:
        return """
As a professional risk management analyst, assess the risk profile for this cryptocurrency position.

Position Details:
{position_context}

Market Context:
{market_context}

Portfolio Context:
{portfolio_info}

Current Time: {timestamp}

Please provide a comprehensive risk assessment including:

1. Overall Risk Level (LOW, MEDIUM, HIGH, CRITICAL)
2. Risk Confidence (0-100)
3. Primary Risk Factors
4. Risk Mitigation Strategies
5. Portfolio Impact Assessment
6. Recommended Actions
7. Monitoring Points
8. Scenario Analysis

Format your response as JSON:
{{
    "risk_level": "MEDIUM",
    "confidence": 85,
    "key_insights": ["Position size appropriate", "Market volatility elevated"],
    "recommendations": ["Reduce position by 20%", "Tighten stop loss"],
    "risk_factors": ["High correlation with BTC", "Low volume environment"],
    "opportunities": ["Defensive positioning", "Risk reduction"],
    "price_targets": {{"stop_loss": 47000, "risk_level": 48000}},
    "timeframe": "IMMEDIATE",
    "market_condition": "HIGH_RISK"
}}

Focus on practical risk management and position protection strategies.
"""
    
    def _get_technical_analysis_template(self) -> str:
        return """
As a professional technical analyst specializing in cryptocurrency markets, provide detailed technical analysis for {symbol}.

Market Data:
{context}

Current Time: {timestamp}

Please provide comprehensive technical analysis including:

1. Technical Outlook (BULLISH, BEARISH, NEUTRAL)
2. Analysis Confidence (0-100)
3. Key Technical Levels
4. Chart Pattern Recognition
5. Indicator Analysis
6. Support/Resistance Levels
7. Volume Analysis
8. Price Projections

Format your response as JSON:
{{
    "technical_outlook": "BULLISH",
    "confidence": 78,
    "key_insights": ["Bullish divergence on RSI", "Volume breakout confirmed"],
    "recommendations": ["Long above $49,000", "Target $52,500"],
    "risk_factors": ["Resistance at $51,000", "Overbought conditions"],
    "opportunities": ["Momentum continuation", "Volume expansion"],
    "price_targets": {{"support": 48000, "resistance": 51000, "target": 52500}},
    "timeframe": "SHORT_TERM",
    "market_condition": "BULLISH_MOMENTUM"
}}

Focus on technical patterns, key levels, and actionable trading insights.
"""
    
    def _get_portfolio_review_template(self) -> str:
        return """
As a professional portfolio manager specializing in cryptocurrency investments, review this portfolio performance and strategy.

Portfolio Data:
{portfolio_context}

Market Overview:
{market_overview}

Current Time: {timestamp}

Please provide comprehensive portfolio analysis including:

1. Portfolio Health Assessment
2. Performance Analysis
3. Risk Assessment
4. Diversification Review
5. Optimization Recommendations
6. Risk Management Suggestions
7. Strategic Adjustments
8. Market Positioning

Format your response as JSON:
{{
    "portfolio_health": "GOOD",
    "confidence": 82,
    "key_insights": ["Well diversified", "Strong risk management", "Good performance"],
    "recommendations": ["Rebalance towards DeFi", "Reduce BTC allocation", "Add stablecoin buffer"],
    "risk_factors": ["High correlation", "Concentrated positions", "Market exposure"],
    "opportunities": ["Sector rotation", "New allocations", "Risk reduction"],
    "price_targets": {{"portfolio_target": 15000, "risk_limit": 8000}},
    "timeframe": "MEDIUM_TERM",
    "market_condition": "OPTIMIZATION_NEEDED"
}}

Focus on portfolio optimization, risk management, and strategic positioning.
"""

    # Response Parsers
    async def _parse_sentiment_analysis(self, ai_response: str, symbol: str) -> AIAnalysisResult:
        """Parse sentiment analysis response"""
        try:
            # Try to extract JSON from response
            json_data = self._extract_json_from_response(ai_response)
            
            if json_data:
                sentiment_str = json_data.get('sentiment', 'NEUTRAL')
                sentiment = MarketSentiment(sentiment_str) if sentiment_str in [s.value for s in MarketSentiment] else MarketSentiment.NEUTRAL
                
                result = AIAnalysisResult(
                    analysis_type=AnalysisType.MARKET_SENTIMENT,
                    symbol=symbol,
                    confidence=float(json_data.get('confidence', 50)),
                    sentiment=sentiment,
                    key_insights=json_data.get('key_insights', []),
                    recommendations=json_data.get('recommendations', []),
                    risk_factors=json_data.get('risk_factors', []),
                    opportunities=json_data.get('opportunities', []),
                    market_condition=json_data.get('market_condition', 'UNKNOWN'),
                    price_targets=json_data.get('price_targets', {}),
                    timeframe=json_data.get('timeframe', 'SHORT_TERM'),
                    raw_response=ai_response
                )
                
                return result
            
            else:
                # Fallback: parse key information from text
                return self._parse_text_response(ai_response, AnalysisType.MARKET_SENTIMENT, symbol)
                
        except Exception as e:
            logger.error(f"Error parsing sentiment analysis: {e}")
            return self._create_error_result(AnalysisType.MARKET_SENTIMENT, symbol, str(e))
    
    async def _parse_trading_opportunity(self, ai_response: str, symbol: str) -> AIAnalysisResult:
        """Parse trading opportunity response"""
        try:
            json_data = self._extract_json_from_response(ai_response)
            
            if json_data:
                result = AIAnalysisResult(
                    analysis_type=AnalysisType.TRADING_OPPORTUNITY,
                    symbol=symbol,
                    confidence=float(json_data.get('confidence', 50)),
                    key_insights=json_data.get('key_insights', []),
                    recommendations=json_data.get('recommendations', []),
                    risk_factors=json_data.get('risk_factors', []),
                    opportunities=json_data.get('opportunities', []),
                    market_condition=json_data.get('market_condition', 'UNKNOWN'),
                    price_targets=json_data.get('price_targets', {}),
                    timeframe=json_data.get('timeframe', 'SHORT_TERM'),
                    raw_response=ai_response
                )
                
                return result
            else:
                return self._parse_text_response(ai_response, AnalysisType.TRADING_OPPORTUNITY, symbol)
                
        except Exception as e:
            logger.error(f"Error parsing trading opportunity: {e}")
            return self._create_error_result(AnalysisType.TRADING_OPPORTUNITY, symbol, str(e))
    
    async def _parse_risk_assessment(self, ai_response: str, symbol: str) -> AIAnalysisResult:
        """Parse risk assessment response"""
        try:
            json_data = self._extract_json_from_response(ai_response)
            
            if json_data:
                result = AIAnalysisResult(
                    analysis_type=AnalysisType.RISK_ASSESSMENT,
                    symbol=symbol,
                    confidence=float(json_data.get('confidence', 50)),
                    key_insights=json_data.get('key_insights', []),
                    recommendations=json_data.get('recommendations', []),
                    risk_factors=json_data.get('risk_factors', []),
                    opportunities=json_data.get('opportunities', []),
                    market_condition=json_data.get('market_condition', 'UNKNOWN'),
                    price_targets=json_data.get('price_targets', {}),
                    timeframe=json_data.get('timeframe', 'IMMEDIATE'),
                    raw_response=ai_response
                )
                
                return result
            else:
                return self._parse_text_response(ai_response, AnalysisType.RISK_ASSESSMENT, symbol)
                
        except Exception as e:
            logger.error(f"Error parsing risk assessment: {e}")
            return self._create_error_result(AnalysisType.RISK_ASSESSMENT, symbol, str(e))
    
    async def _parse_portfolio_review(self, ai_response: str, symbol: str) -> AIAnalysisResult:
        """Parse portfolio review response"""
        try:
            json_data = self._extract_json_from_response(ai_response)
            
            if json_data:
                result = AIAnalysisResult(
                    analysis_type=AnalysisType.PORTFOLIO_REVIEW,
                    symbol=symbol,
                    confidence=float(json_data.get('confidence', 50)),
                    key_insights=json_data.get('key_insights', []),
                    recommendations=json_data.get('recommendations', []),
                    risk_factors=json_data.get('risk_factors', []),
                    opportunities=json_data.get('opportunities', []),
                    market_condition=json_data.get('market_condition', 'UNKNOWN'),
                    price_targets=json_data.get('price_targets', {}),
                    timeframe=json_data.get('timeframe', 'MEDIUM_TERM'),
                    raw_response=ai_response
                )
                
                return result
            else:
                return self._parse_text_response(ai_response, AnalysisType.PORTFOLIO_REVIEW, symbol)
                
        except Exception as e:
            logger.error(f"Error parsing portfolio review: {e}")
            return self._create_error_result(AnalysisType.PORTFOLIO_REVIEW, symbol, str(e))
    
    def _extract_json_from_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Extract JSON data from AI response"""
        try:
            # Try to find JSON block in response
            import re
            
            # Look for JSON between curly braces
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            
            # Try to parse entire response as JSON
            return json.loads(response)
            
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract key-value pairs
            try:
                return self._extract_key_values_from_text(response)
            except Exception:
                return None
        except Exception as e:
            logger.warning(f"Error extracting JSON: {e}")
            return None
    
    def _extract_key_values_from_text(self, text: str) -> Dict[str, Any]:
        """Extract key-value pairs from text response"""
        import re
        
        result = {
            'confidence': 50,
            'key_insights': [],
            'recommendations': [],
            'risk_factors': [],
            'opportunities': [],
            'price_targets': {},
            'timeframe': 'SHORT_TERM',
            'market_condition': 'UNKNOWN'
        }
        
        # Extract confidence
        confidence_match = re.search(r'confidence[:\s]+(\d+)', text, re.IGNORECASE)
        if confidence_match:
            result['confidence'] = int(confidence_match.group(1))
        
        # Extract sentiment/recommendation
        sentiment_patterns = ['BULLISH', 'BEARISH', 'NEUTRAL', 'BUY', 'SELL', 'HOLD']
        for pattern in sentiment_patterns:
            if pattern.lower() in text.lower():
                result['market_condition'] = pattern
                break
        
        # Extract insights (look for bullet points or numbered lists)
        insights = re.findall(r'[•\-\*]\s*([^\n\r]+)', text)
        if insights:
            result['key_insights'] = insights[:5]  # Limit to 5
        
        return result
    
    def _parse_text_response(self, response: str, analysis_type: AnalysisType, symbol: str) -> AIAnalysisResult:
        """Parse text response when JSON parsing fails"""
        
        # Extract key information from text
        extracted_data = self._extract_key_values_from_text(response)
        
        result = AIAnalysisResult(
            analysis_type=analysis_type,
            symbol=symbol,
            confidence=extracted_data.get('confidence', 50),
            key_insights=extracted_data.get('key_insights', ['Analysis available in raw response']),
            recommendations=extracted_data.get('recommendations', ['See detailed analysis below']),
            risk_factors=extracted_data.get('risk_factors', ['General market risks apply']),
            opportunities=extracted_data.get('opportunities', ['Opportunities identified in analysis']),
            market_condition=extracted_data.get('market_condition', 'UNKNOWN'),
            timeframe='SHORT_TERM',
            raw_response=response
        )
        
        return result
    
    def _create_error_result(self, analysis_type: AnalysisType, symbol: str, error_message: str) -> AIAnalysisResult:
        """Create error result when analysis fails"""
        
        result = AIAnalysisResult(
            analysis_type=analysis_type,
            symbol=symbol,
            confidence=0,
            key_insights=[f"Analysis failed: {error_message}"],
            recommendations=["Manual analysis recommended"],
            risk_factors=["Analysis unavailable - exercise caution"],
            opportunities=["Unable to identify opportunities"],
            market_condition="ERROR",
            timeframe="UNKNOWN",
            raw_response=f"Error: {error_message}"
        )
        
        return result
    
    async def _store_analysis_result(self, result: AIAnalysisResult):
        """Store analysis result in database"""
        try:
            with get_db_session() as session:
                # Calculate tokens (rough estimate)
                prompt_tokens = len(result.raw_response.split()) * 0.75  # Rough estimate
                completion_tokens = len(result.raw_response.split())
                total_tokens = prompt_tokens + completion_tokens
                
                analysis_log = AIAnalysisLog(
                    symbol=result.symbol,
                    analysis_type=result.analysis_type.value,
                    input_data={
                        'symbol': result.symbol,
                        'analysis_type': result.analysis_type.value,
                        'timestamp': result.timestamp.isoformat()
                    },
                    ai_response={'raw_response': result.raw_response},
                    processed_response=result.to_dict(),
                    confidence_score=int(result.confidence),
                    market_condition=result.market_condition,
                    processing_time=result.processing_time,
                    prompt_tokens=int(prompt_tokens),
                    completion_tokens=int(completion_tokens),
                    total_tokens=int(total_tokens),
                    model_name=self.model,
                    model_version="2024-01"
                )
                
                session.add(analysis_log)
                
                # Add to history
                self.analysis_history.append(result)
                if len(self.analysis_history) > 100:
                    self.analysis_history = self.analysis_history[-100:]
                
                logger.debug(f"Stored AI analysis result for {result.symbol}")
                
        except Exception as e:
            logger.error(f"Error storing analysis result: {e}")
    
    async def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get comprehensive analysis statistics"""
        try:
            with get_db_session() as session:
                # Get recent analyses
                recent_analyses = session.query(AIAnalysisLog).filter(
                    AIAnalysisLog.timestamp >= datetime.utcnow() - timedelta(days=7)
                ).all()
                
                if not recent_analyses:
                    return {
                        'total_analyses': 0,
                        'avg_confidence': 0,
                        'analysis_types': {},
                        'symbols_analyzed': [],
                        'avg_processing_time': 0,
                        'total_tokens_used': 0
                    }
                
                # Calculate statistics
                total_analyses = len(recent_analyses)
                avg_confidence = np.mean([a.confidence_score or 0 for a in recent_analyses])
                avg_processing_time = np.mean([a.processing_time or 0 for a in recent_analyses])
                total_tokens = sum(a.total_tokens or 0 for a in recent_analyses)
                
                # Analysis types distribution
                analysis_types = {}
                for analysis in recent_analyses:
                    analysis_type = analysis.analysis_type
                    analysis_types[analysis_type] = analysis_types.get(analysis_type, 0) + 1
                
                # Symbols analyzed
                symbols = list(set(a.symbol for a in recent_analyses))
                
                # Success rate (confidence > 60)
                high_confidence_analyses = len([a for a in recent_analyses if (a.confidence_score or 0) > 60])
                success_rate = (high_confidence_analyses / total_analyses * 100) if total_analyses > 0 else 0
                
                return {
                    'period': '7 days',
                    'total_analyses': total_analyses,
                    'avg_confidence': round(avg_confidence, 1),
                    'success_rate': round(success_rate, 1),
                    'analysis_types': analysis_types,
                    'symbols_analyzed': symbols[:10],  # Top 10
                    'avg_processing_time': round(avg_processing_time, 2),
                    'total_tokens_used': int(total_tokens),
                    'api_requests': self.request_count,
                    'model_used': self.model
                }
                
        except Exception as e:
            logger.error(f"Error getting analysis statistics: {e}")
            return {'error': str(e)}
    
    async def provide_feedback(self, 
                             analysis_id: str, 
                             actual_outcome: str, 
                             accuracy_score: float,
                             notes: str = None):
        """Provide feedback on analysis accuracy for learning"""
        try:
            with get_db_session() as session:
                analysis = session.query(AIAnalysisLog).filter(
                    AIAnalysisLog.id == analysis_id
                ).first()
                
                if analysis:
                    analysis.actual_outcome = actual_outcome
                    analysis.accuracy_score = accuracy_score
                    analysis.feedback_notes = notes
                    
                    # Store in performance feedback for learning
                    self.performance_feedback[analysis_id] = {
                        'analysis_type': analysis.analysis_type,
                        'symbol': analysis.symbol,
                        'original_confidence': analysis.confidence_score,
                        'actual_outcome': actual_outcome,
                        'accuracy_score': accuracy_score,
                        'timestamp': datetime.utcnow()
                    }
                    
                    logger.info(f"Feedback provided for analysis {analysis_id}: {actual_outcome} (accuracy: {accuracy_score:.1f})")
                else:
                    logger.warning(f"Analysis {analysis_id} not found for feedback")
                    
        except Exception as e:
            logger.error(f"Error providing feedback: {e}")
    
    async def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights from feedback for model improvement"""
        try:
            if not self.performance_feedback:
                return {'message': 'No feedback data available for learning insights'}
            
            feedback_data = list(self.performance_feedback.values())
            
            # Calculate average accuracy by analysis type
            accuracy_by_type = {}
            confidence_vs_accuracy = []
            
            for feedback in feedback_data:
                analysis_type = feedback['analysis_type']
                accuracy = feedback['accuracy_score']
                confidence = feedback['original_confidence']
                
                if analysis_type not in accuracy_by_type:
                    accuracy_by_type[analysis_type] = []
                accuracy_by_type[analysis_type].append(accuracy)
                
                confidence_vs_accuracy.append((confidence, accuracy))
            
            # Average accuracy by type
            avg_accuracy_by_type = {
                analysis_type: np.mean(accuracies)
                for analysis_type, accuracies in accuracy_by_type.items()
            }
            
            # Confidence vs accuracy correlation
            if len(confidence_vs_accuracy) > 1:
                confidences, accuracies = zip(*confidence_vs_accuracy)
                correlation = np.corrcoef(confidences, accuracies)[0, 1]
            else:
                correlation = 0
            
            return {
                'total_feedback_entries': len(feedback_data),
                'avg_accuracy_by_type': avg_accuracy_by_type,
                'overall_avg_accuracy': np.mean([f['accuracy_score'] for f in feedback_data]),
                'confidence_accuracy_correlation': round(correlation, 3),
                'learning_recommendations': self._generate_learning_recommendations(avg_accuracy_by_type, correlation)
            }
            
        except Exception as e:
            logger.error(f"Error getting learning insights: {e}")
            return {'error': str(e)}
    
    def _generate_learning_recommendations(self, 
                                         accuracy_by_type: Dict[str, float], 
                                         correlation: float) -> List[str]:
        """Generate learning recommendations based on feedback"""
        recommendations = []
        
        # Low accuracy analysis types
        for analysis_type, accuracy in accuracy_by_type.items():
            if accuracy < 0.6:  # Less than 60% accuracy
                recommendations.append(f"Improve {analysis_type} analysis - current accuracy: {accuracy:.1%}")
        
        # Confidence calibration
        if correlation < 0.3:
            recommendations.append("Confidence scores poorly calibrated with actual outcomes - review confidence calculation")
        elif correlation > 0.7:
            recommendations.append("Good confidence calibration - maintain current approach")
        
        # High performing areas
        best_type = max(accuracy_by_type.items(), key=lambda x: x[1]) if accuracy_by_type else None
        if best_type and best_type[1] > 0.8:
            recommendations.append(f"Excellent performance in {best_type[0]} - consider applying similar approach to other analysis types")
        
        return recommendations
    
    async def cleanup(self):
        """Cleanup resources and close connections"""
        try:
            if self.session:
                await self.session.close()
            
            # Save performance feedback to database
            if self.performance_feedback:
                logger.info(f"Saving {len(self.performance_feedback)} performance feedback entries")
            
            logger.info("Claude Analyzer cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def __str__(self) -> str:
        return f"ClaudeAnalyzer(model={self.model}, requests={self.request_count}, analyses={len(self.analysis_history)})"
    
    def __repr__(self) -> str:
        return self.__str__()

# Factory function
def create_claude_analyzer(api_key: str = None) -> ClaudeAnalyzer:
    """Factory function to create Claude Analyzer instance"""
    return ClaudeAnalyzer(api_key)

# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_claude_analyzer():
        print("🤖 Claude AI Analyzer Test")
        print("=" * 50)
        
        try:
            # Create analyzer
            analyzer = create_claude_analyzer()
            await analyzer.initialize()
            print(f"✅ Initialized: {analyzer}")
            
            # Create sample market data
            np.random.seed(42)
            dates = pd.date_range('2024-01-01', periods=100, freq='1H')
            
            base_price = 50000
            prices = []
            volumes = []
            
            for i in range(100):
                if i == 0:
                    price = base_price
                else:
                    change = np.random.normal(0, 500)
                    price = max(prices[-1] + change, base_price * 0.9)
                
                prices.append(price)
                volumes.append(np.random.lognormal(15, 0.3))
            
            market_data = pd.DataFrame({
                'timestamp': dates,
                'open': [p + np.random.normal(0, 100) for p in prices],
                'high': [p + abs(np.random.normal(200, 100)) for p in prices],
                'low': [p - abs(np.random.normal(200, 100)) for p in prices],
                'close': prices,
                'volume': volumes
            })
            
            current_price = prices[-1]
            print(f"📊 Sample market data created: {len(market_data)} periods, current price: ${current_price:,.2f}")
            
            # Test Market Sentiment Analysis
            print("\n🔍 Testing Market Sentiment Analysis...")
            sentiment_result = await analyzer.analyze_market_sentiment(
                symbol='BTC/USDT',
                market_data=market_data,
                current_price=current_price,
                additional_context={'volume_spike': True, 'news_sentiment': 'positive'}
            )
            
            print(f"  Sentiment: {sentiment_result.sentiment}")
            print(f"  Confidence: {sentiment_result.confidence:.1f}%")
            print(f"  Key Insights: {len(sentiment_result.key_insights)} items")
            print(f"  Recommendations: {len(sentiment_result.recommendations)} items")
            
            # Test Trading Opportunity Analysis
            print("\n📈 Testing Trading Opportunity Analysis...")
            strategy_signals = [
                {
                    'strategy_id': 'rsi_strategy',
                    'signal_type': 'BUY',
                    'confidence': 75,
                    'reasoning': 'RSI oversold condition'
                },
                {
                    'strategy_id': 'volume_profile_strategy', 
                    'signal_type': 'STRONG_BUY',
                    'confidence': 85,
                    'reasoning': 'Volume breakout confirmed'
                }
            ]
            
            portfolio_context = {
                'total_value': 10000,
                'available_balance': 3000,
                'risk_tolerance': 'MEDIUM',
                'current_positions': 2
            }
            
            opportunity_result = await analyzer.analyze_trading_opportunity(
                symbol='BTC/USDT',
                market_data=market_data,
                current_price=current_price,
                strategy_signals=strategy_signals,
                portfolio_context=portfolio_context
            )
            
            print(f"  Trade Analysis: {opportunity_result.market_condition}")
            print(f"  Confidence: {opportunity_result.confidence:.1f}%")
            print(f"  Price Targets: {opportunity_result.price_targets}")
            print(f"  Risk Factors: {len(opportunity_result.risk_factors)} identified")
            
            # Test Risk Assessment
            print("\n⚠️ Testing Risk Assessment...")
            position_data = {
                'direction': 'LONG',
                'entry_price': 48500,
                'current_price': current_price,
                'quantity': 0.2,
                'unrealized_pnl': (current_price - 48500) * 0.2,
                'stop_loss': 47000,
                'take_profit': 52000
            }
            
            risk_result = await analyzer.analyze_risk_assessment(
                symbol='BTC/USDT',
                position_data=position_data,
                market_data=market_data,
                portfolio_context=portfolio_context
            )
            
            print(f"  Risk Assessment: {risk_result.market_condition}")
            print(f"  Confidence: {risk_result.confidence:.1f}%")
            print(f"  Risk Factors: {risk_result.risk_factors[:2] if risk_result.risk_factors else []}")
            
            # Test Portfolio Review
            print("\n📊 Testing Portfolio Review...")
            portfolio_data = {
                'total_value': 12500,
                'cash_balance': 2500,
                'invested_amount': 10000,
                'total_return': 25.0,
                'daily_pnl': 150,
                'win_rate': 68.5,
                'positions': [
                    {'symbol': 'BTC/USDT', 'direction': 'LONG', 'unrealized_pnl': 300},
                    {'symbol': 'ETH/USDT', 'direction': 'LONG', 'unrealized_pnl': -50}
                ]
            }
            
            market_overview = {
                'btc_dominance': 52.3,
                'total_market_cap': 1200000000000,
                'market_sentiment': 'BULLISH',
                'fear_greed_index': 75,
                'volatility_index': 65
            }
            
            portfolio_result = await analyzer.analyze_portfolio_review(
                portfolio_data=portfolio_data,
                market_overview=market_overview
            )
            
            print(f"  Portfolio Health: {portfolio_result.market_condition}")
            print(f"  Confidence: {portfolio_result.confidence:.1f}%")
            print(f"  Key Insights: {portfolio_result.key_insights[:2] if portfolio_result.key_insights else []}")
            
            # Get Analysis Statistics
            print("\n📈 Analysis Statistics:")
            stats = await analyzer.get_analysis_statistics()
            print(f"  Total Analyses: {stats.get('total_analyses', 0)}")
            print(f"  Average Confidence: {stats.get('avg_confidence', 0):.1f}%")
            print(f"  Success Rate: {stats.get('success_rate', 0):.1f}%")
            print(f"  API Requests: {stats.get('api_requests', 0)}")
            
            print(f"\n🎉 Claude Analyzer test completed successfully!")
            
        except Exception as e:
            print(f"❌ Error in Claude Analyzer test: {e}")
            logger.error(f"Test error: {e}")
        
        finally:
            # Cleanup
            try:
                await analyzer.cleanup()
                print("🛑 Claude Analyzer cleanup completed")
            except:
                pass
    
    # Run the test
    asyncio.run(test_claude_analyzer())