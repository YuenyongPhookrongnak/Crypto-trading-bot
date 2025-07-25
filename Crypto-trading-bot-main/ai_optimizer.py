#!/usr/bin/env python3
"""
Enhanced AI Optimization System

ระบบ AI ที่ปรับ R/R และ Leverage แบบไดนามิกตามสถานการณ์:
- ข่าวเชิงลบแรง + Break structure → ลด Leverage และขยาย SL
- Market volatility → ปรับ position sizing
- News sentiment analysis → ปรับ risk parameters
"""

import asyncio
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import os

class MarketCondition(Enum):
    VERY_BULLISH = "VERY_BULLISH"
    BULLISH = "BULLISH"
    NEUTRAL = "NEUTRAL"
    BEARISH = "BEARISH"
    VERY_BEARISH = "VERY_BEARISH"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"

class NewsImpact(Enum):
    VERY_POSITIVE = 5
    POSITIVE = 3
    NEUTRAL = 0
    NEGATIVE = -3
    VERY_NEGATIVE = -5

@dataclass
class DynamicTradingParameters:
    """Dynamic trading parameters based on AI analysis"""
    # Core parameters
    leverage: int
    risk_per_trade: float
    risk_reward_ratio: float
    stop_loss_pct: float
    
    # Dynamic adjustments
    base_leverage: int
    volatility_multiplier: float
    news_adjustment: float
    structure_confidence: float
    
    # Position management
    position_sizing_method: str  # "FIXED", "VOLATILITY_ADJUSTED", "KELLY"
    partial_close_levels: List[float]
    trailing_stop_trigger: float
    
    # AI reasoning
    adjustment_reasons: List[str]
    market_condition: MarketCondition
    confidence_score: float

class EnhancedNewsAnalyzer:
    """Enhanced news analysis with sentiment scoring"""
    
    def __init__(self):
        self.news_api_key = os.getenv('NEWS_API_KEY', '')
        
        # Enhanced sentiment keywords
        self.sentiment_keywords = {
            'very_positive': ['moon', 'rally', 'surge', 'explosive', 'breakthrough', 'adoption', 'institutional'],
            'positive': ['bullish', 'rise', 'gains', 'positive', 'growth', 'increase', 'buy'],
            'negative': ['bearish', 'fall', 'decline', 'negative', 'drop', 'sell', 'concern'],
            'very_negative': ['crash', 'dump', 'collapse', 'banned', 'regulation', 'hack', 'scam']
        }
        
        # Market-moving events
        self.high_impact_events = [
            'fed', 'federal reserve', 'interest rate', 'inflation',
            'regulation', 'sec', 'etf approval', 'institutional adoption',
            'hack', 'exchange', 'whale movement', 'government'
        ]
    
    async def analyze_market_news(self, symbol: str, hours_lookback: int = 24) -> Dict[str, Any]:
        """Analyze news sentiment and impact"""
        try:
            # Get base asset from symbol
            base_asset = symbol.split('/')[0].lower()
            
            # Fetch news
            news_data = await self._fetch_crypto_news(base_asset, hours_lookback)
            
            # Analyze sentiment
            sentiment_analysis = self._analyze_sentiment(news_data)
            
            # Assess market impact
            impact_assessment = self._assess_market_impact(news_data, sentiment_analysis)
            
            return {
                'sentiment_score': sentiment_analysis['score'],
                'sentiment_level': sentiment_analysis['level'],
                'impact_level': impact_assessment['level'],
                'key_events': impact_assessment['key_events'],
                'articles_count': len(news_data),
                'high_impact_news': impact_assessment['high_impact_count'],
                'adjustment_factor': self._calculate_adjustment_factor(sentiment_analysis, impact_assessment)
            }
            
        except Exception as e:
            print(f"❌ Error analyzing news: {e}")
            return self._get_neutral_news_analysis()
    
    async def _fetch_crypto_news(self, asset: str, hours: int) -> List[Dict]:
        """Fetch cryptocurrency news"""
        if not self.news_api_key:
            return self._get_mock_news_data(asset)
        
        try:
            # Calculate time range
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=hours)
            
            # NewsAPI query
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': f'{asset} OR bitcoin OR cryptocurrency OR crypto',
                'sortBy': 'publishedAt',
                'pageSize': 50,
                'from': start_time.isoformat(),
                'to': end_time.isoformat(),
                'apiKey': self.news_api_key,
                'language': 'en'
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json().get('articles', [])
            else:
                print(f"⚠️ News API error: {response.status_code}")
                return self._get_mock_news_data(asset)
                
        except Exception as e:
            print(f"⚠️ Error fetching news: {e}")
            return self._get_mock_news_data(asset)
    
    def _get_mock_news_data(self, asset: str) -> List[Dict]:
        """Generate mock news data for testing"""
        import random
        
        mock_headlines = [
            f"{asset.upper()} shows strong momentum as institutional adoption grows",
            f"Major exchange lists {asset.upper()} futures contracts",
            f"Technical analysis suggests {asset.upper()} consolidation phase",
            f"Market volatility affects {asset.upper()} trading volumes",
            f"Regulatory clarity boosts {asset.upper()} investor confidence"
        ]
        
        return [
            {
                'title': random.choice(mock_headlines),
                'description': f"Analysis of {asset.upper()} market conditions and trends",
                'publishedAt': (datetime.utcnow() - timedelta(hours=random.randint(1, 24))).isoformat(),
                'source': {'name': 'Crypto News'}
            }
            for _ in range(random.randint(5, 15))
        ]
    
    def _analyze_sentiment(self, articles: List[Dict]) -> Dict[str, Any]:
        """Analyze sentiment from news articles"""
        total_score = 0
        article_count = len(articles)
        
        if article_count == 0:
            return {'score': 0, 'level': NewsImpact.NEUTRAL}
        
        for article in articles:
            title = article.get('title', '').lower()
            description = article.get('description', '').lower()
            text = f"{title} {description}"
            
            article_score = 0
            
            # Score based on keywords
            for word in self.sentiment_keywords['very_positive']:
                article_score += text.count(word) * 2
            
            for word in self.sentiment_keywords['positive']:
                article_score += text.count(word) * 1
            
            for word in self.sentiment_keywords['negative']:
                article_score -= text.count(word) * 1
            
            for word in self.sentiment_keywords['very_negative']:
                article_score -= text.count(word) * 2
            
            total_score += article_score
        
        # Average score
        avg_score = total_score / article_count
        
        # Determine sentiment level
        if avg_score >= 2:
            level = NewsImpact.VERY_POSITIVE
        elif avg_score >= 0.5:
            level = NewsImpact.POSITIVE
        elif avg_score <= -2:
            level = NewsImpact.VERY_NEGATIVE
        elif avg_score <= -0.5:
            level = NewsImpact.NEGATIVE
        else:
            level = NewsImpact.NEUTRAL
        
        return {'score': avg_score, 'level': level}
    
    def _assess_market_impact(self, articles: List[Dict], sentiment: Dict[str, Any]) -> Dict[str, Any]:
        """Assess potential market impact of news"""
        high_impact_count = 0
        key_events = []
        
        for article in articles:
            title = article.get('title', '').lower()
            description = article.get('description', '').lower()
            text = f"{title} {description}"
            
            # Check for high-impact events
            for event in self.high_impact_events:
                if event in text:
                    high_impact_count += 1
                    if len(key_events) < 3:  # Keep top 3 events
                        key_events.append(article.get('title', '')[:100])
                    break
        
        # Determine impact level
        if high_impact_count >= 3:
            impact_level = "HIGH"
        elif high_impact_count >= 1:
            impact_level = "MEDIUM"
        else:
            impact_level = "LOW"
        
        return {
            'level': impact_level,
            'high_impact_count': high_impact_count,
            'key_events': key_events
        }
    
    def _calculate_adjustment_factor(self, sentiment: Dict[str, Any], impact: Dict[str, Any]) -> float:
        """Calculate news-based adjustment factor for trading parameters"""
        base_factor = 1.0
        
        # Sentiment adjustment
        sentiment_score = sentiment['score']
        if sentiment_score >= 2:
            base_factor *= 1.2  # Increase confidence for very positive news
        elif sentiment_score >= 0.5:
            base_factor *= 1.1  # Slight increase for positive news
        elif sentiment_score <= -2:
            base_factor *= 0.7  # Reduce confidence for very negative news
        elif sentiment_score <= -0.5:
            base_factor *= 0.85  # Slight reduction for negative news
        
        # Impact adjustment
        if impact['level'] == "HIGH":
            base_factor *= 0.8  # Reduce confidence for high-impact events
        elif impact['level'] == "MEDIUM":
            base_factor *= 0.9  # Slight reduction for medium impact
        
        return base_factor
    
    def _get_neutral_news_analysis(self) -> Dict[str, Any]:
        """Return neutral news analysis as fallback"""
        return {
            'sentiment_score': 0,
            'sentiment_level': NewsImpact.NEUTRAL,
            'impact_level': "LOW",
            'key_events': [],
            'articles_count': 0,
            'high_impact_news': 0,
            'adjustment_factor': 1.0
        }

class DynamicAIOptimizer:
    """Dynamic AI optimizer with enhanced parameter adjustment"""
    
    def __init__(self):
        self.news_analyzer = EnhancedNewsAnalyzer()
        self.openai_api_key = os.getenv('OPENAI_API_KEY', '')
        self.claude_api_key = os.getenv('CLAUDE_API_KEY', '')
        
        # Base parameter ranges
        self.base_ranges = {
            'leverage': (1, 25),
            'risk_per_trade': (0.005, 0.10),
            'risk_reward': (2.0, 5.0),
            'stop_loss': (0.01, 0.05)
        }
        
    async def optimize_dynamic_parameters(self, market_data: Dict[str, Any], 
                                        news_analysis: Dict[str, Any] = None) -> DynamicTradingParameters:
        """Optimize parameters dynamically based on market conditions and news"""
        try:
            # Get news analysis if not provided
            if not news_analysis:
                news_analysis = await self.news_analyzer.analyze_market_news(
                    market_data.get('symbol', 'BTC/USDT')
                )
            
            # Determine market condition
            market_condition = self._assess_market_condition(market_data, news_analysis)
            
            # Get AI optimization
            ai_params = await self._get_ai_parameter_optimization(market_data, news_analysis, market_condition)
            
            # Apply dynamic adjustments
            dynamic_params = self._apply_dynamic_adjustments(ai_params, market_data, news_analysis, market_condition)
            
            return dynamic_params
            
        except Exception as e:
            print(f"❌ Error in dynamic optimization: {e}")
            return self._get_safe_fallback_parameters(market_data)
    
    def _assess_market_condition(self, market_data: Dict[str, Any], news_analysis: Dict[str, Any]) -> MarketCondition:
        """Assess overall market condition"""
        # Technical condition
        signal_strength = market_data.get('signal_strength', 60)
        volatility = market_data.get('atr_pct', 2.0)
        price_change = market_data.get('price_change_24h', 0)
        
        # News condition
        sentiment_score = news_analysis.get('sentiment_score', 0)
        impact_level = news_analysis.get('impact_level', 'LOW')
        
        # High volatility override
        if volatility > 5 or impact_level == "HIGH":
            return MarketCondition.HIGH_VOLATILITY
        
        # Combined sentiment and technical analysis
        combined_score = signal_strength + (sentiment_score * 10) + (price_change * 2)
        
        if combined_score >= 85:
            return MarketCondition.VERY_BULLISH
        elif combined_score >= 70:
            return MarketCondition.BULLISH
        elif combined_score <= 35:
            return MarketCondition.VERY_BEARISH
        elif combined_score <= 50:
            return MarketCondition.BEARISH
        else:
            return MarketCondition.NEUTRAL
    
    async def _get_ai_parameter_optimization(self, market_data: Dict[str, Any], 
                                           news_analysis: Dict[str, Any], 
                                           market_condition: MarketCondition) -> Dict[str, Any]:
        """Get AI-optimized parameters"""
        
        context = f"""
        DYNAMIC PARAMETER OPTIMIZATION REQUEST
        
        Current Market Analysis:
        - Symbol: {market_data.get('symbol', 'BTC/USDT')}
        - Signal Strength: {market_data.get('signal_strength', 60)}%
        - Price Change 24h: {market_data.get('price_change_24h', 0):+.2f}%
        - Volatility (ATR): {market_data.get('atr_pct', 2.0):.2f}%
        - Market Condition: {market_condition.value}
        
        News Analysis:
        - Sentiment Score: {news_analysis.get('sentiment_score', 0):.2f}
        - Impact Level: {news_analysis.get('impact_level', 'LOW')}
        - Key Events: {news_analysis.get('key_events', [])[:2]}
        - High Impact News: {news_analysis.get('high_impact_news', 0)}
        
        Technical Confluences:
        - BOS Confirmed: {market_data.get('bos_bullish') or market_data.get('bos_bearish')}
        - FVG Available: {len(market_data.get('fvgs', []))}
        - EMA Alignment: {market_data.get('bullish_alignment') or market_data.get('bearish_alignment')}
        - Volume Confirmation: {market_data.get('volume_ratio', 1) > 1.2}
        
        DYNAMIC ADJUSTMENT RULES:
        
        1. NEGATIVE NEWS + BREAK STRUCTURE:
           - Reduce leverage by 30-50%
           - Widen stop loss by 20-40%
           - Lower risk per trade
           
        2. HIGH VOLATILITY CONDITIONS:
           - Reduce position size
           - Use tighter stops or wider stops based on volatility type
           - Adjust R/R ratio
           
        3. STRONG CONFLUENCE + POSITIVE NEWS:
           - Can increase leverage moderately
           - Tighten risk management
           - Optimize R/R for maximum efficiency
           
        4. NEUTRAL/UNCERTAIN CONDITIONS:
           - Conservative parameters
           - Lower leverage
           - Standard risk management
        
        Optimize parameters considering ALL factors above.
        """
        
        prompt = f"""
        {context}
        
        Based on this comprehensive analysis, determine optimal trading parameters with dynamic adjustments.
        
        CRITICAL CONSIDERATIONS:
        1. News sentiment impact on market psychology
        2. Volatility-adjusted position sizing
        3. Structure break reliability in current conditions
        4. Risk-adjusted returns optimization
        
        Respond ONLY with valid JSON:
        {{
            "base_leverage": integer (5-25),
            "adjusted_leverage": integer (1-25),
            "base_risk_pct": float (0.01-0.08),
            "adjusted_risk_pct": float (0.005-0.10),
            "risk_reward_ratio": float (2.0-5.0),
            "stop_loss_pct": float (0.01-0.05),
            "volatility_adjustment": float (0.5-1.5),
            "news_adjustment": float (0.6-1.4),
            "confidence_multiplier": float (0.7-1.3),
            "position_sizing_method": "FIXED" | "VOLATILITY_ADJUSTED" | "KELLY",
            "trailing_stop_trigger": float (1.0-3.0),
            "partial_close_levels": [float, float, float],
            "market_regime_factor": float (0.6-1.2),
            "adjustment_reasons": [
                "Primary reason for adjustments",
                "Secondary consideration",
                "Risk factor addressed"
            ]
        }}
        """
        
        # Try AI providers
        if self.openai_api_key:
            try:
                import openai
                openai.api_key = self.openai_api_key
                
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are an expert in dynamic cryptocurrency trading optimization. Focus on adaptive risk management and market condition analysis. Respond only with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=800,
                    temperature=0.2
                )
                
                response_text = response.choices[0].message.content.strip()
                response_text = response_text.replace('```json', '').replace('```', '').strip()
                
                return json.loads(response_text)
                
            except Exception as e:
                print(f"⚠️ OpenAI optimization failed: {e}")
        
        # Fallback to rule-based optimization
        return self._rule_based_dynamic_optimization(market_data, news_analysis, market_condition)
    
    def _rule_based_dynamic_optimization(self, market_data: Dict[str, Any], 
                                       news_analysis: Dict[str, Any], 
                                       market_condition: MarketCondition) -> Dict[str, Any]:
        """Rule-based dynamic parameter optimization"""
        
        # Base parameters
        base_leverage = 10
        base_risk = 0.02
        base_rr = 3.0
        base_sl = 0.02
        
        # Market condition adjustments
        if market_condition == MarketCondition.VERY_BULLISH:
            leverage_mult = 1.3
            risk_mult = 1.2
            rr_mult = 1.1
        elif market_condition == MarketCondition.BULLISH:
            leverage_mult = 1.1
            risk_mult = 1.05
            rr_mult = 1.0
        elif market_condition == MarketCondition.VERY_BEARISH:
            leverage_mult = 0.6
            risk_mult = 0.7
            rr_mult = 1.2
        elif market_condition == MarketCondition.BEARISH:
            leverage_mult = 0.8
            risk_mult = 0.85
            rr_mult = 1.1
        elif market_condition == MarketCondition.HIGH_VOLATILITY:
            leverage_mult = 0.5
            risk_mult = 0.6
            rr_mult = 1.3
        else:  # NEUTRAL
            leverage_mult = 1.0
            risk_mult = 1.0
            rr_mult = 1.0
        
        # News adjustments
        news_factor = news_analysis.get('adjustment_factor', 1.0)
        
        # Volatility adjustments
        volatility = market_data.get('atr_pct', 2.0)
        vol_adjustment = max(0.5, min(1.5, 2.0 / volatility))
        
        # Calculate final parameters
        adjusted_leverage = int(base_leverage * leverage_mult * news_factor)
        adjusted_leverage = max(1, min(25, adjusted_leverage))
        
        adjusted_risk = base_risk * risk_mult * news_factor * 0.9  # Slightly conservative
        adjusted_risk = max(0.005, min(0.08, adjusted_risk))
        
        adjusted_rr = base_rr * rr_mult
        adjusted_sl = base_sl * vol_adjustment
        
        # Position sizing method based on conditions
        if volatility > 4:
            sizing_method = "VOLATILITY_ADJUSTED"
        elif market_condition in [MarketCondition.VERY_BULLISH, MarketCondition.VERY_BEARISH]:
            sizing_method = "KELLY"
        else:
            sizing_method = "FIXED"
        
        return {
            "base_leverage": base_leverage,
            "adjusted_leverage": adjusted_leverage,
            "base_risk_pct": base_risk,
            "adjusted_risk_pct": adjusted_risk,
            "risk_reward_ratio": adjusted_rr,
            "stop_loss_pct": adjusted_sl,
            "volatility_adjustment": vol_adjustment,
            "news_adjustment": news_factor,
            "confidence_multiplier": min(leverage_mult * news_factor, 1.2),
            "position_sizing_method": sizing_method,
            "trailing_stop_trigger": 1.5 if volatility > 3 else 2.0,
            "partial_close_levels": [0.4, 0.35, 0.25],
            "market_regime_factor": leverage_mult,
            "adjustment_reasons": [
                f"Market condition: {market_condition.value}",
                f"News adjustment: {news_factor:.2f}",
                f"Volatility factor: {vol_adjustment:.2f}"
            ]
        }
    
    def _apply_dynamic_adjustments(self, ai_params: Dict[str, Any], 
                                 market_data: Dict[str, Any], 
                                 news_analysis: Dict[str, Any], 
                                 market_condition: MarketCondition) -> DynamicTradingParameters:
        """Apply final dynamic adjustments and create parameters object"""
        
        return DynamicTradingParameters(
            leverage=ai_params.get('adjusted_leverage', 10),
            risk_per_trade=ai_params.get('adjusted_risk_pct', 0.02),
            risk_reward_ratio=ai_params.get('risk_reward_ratio', 3.0),
            stop_loss_pct=ai_params.get('stop_loss_pct', 0.02),
            
            base_leverage=ai_params.get('base_leverage', 10),
            volatility_multiplier=ai_params.get('volatility_adjustment', 1.0),
            news_adjustment=ai_params.get('news_adjustment', 1.0),
            structure_confidence=market_data.get('signal_strength', 70) / 100,
            
            position_sizing_method=ai_params.get('position_sizing_method', 'FIXED'),
            partial_close_levels=ai_params.get('partial_close_levels', [0.4, 0.35, 0.25]),
            trailing_stop_trigger=ai_params.get('trailing_stop_trigger', 2.0),
            
            adjustment_reasons=ai_params.get('adjustment_reasons', []),
            market_condition=market_condition,
            confidence_score=ai_params.get('confidence_multiplier', 1.0) * market_data.get('signal_strength', 70)
        )
    
    def _get_safe_fallback_parameters(self, market_data: Dict[str, Any]) -> DynamicTradingParameters:
        """Safe fallback parameters"""
        return DynamicTradingParameters(
            leverage=3,
            risk_per_trade=0.01,
            risk_reward_ratio=3.0,
            stop_loss_pct=0.02,
            
            base_leverage=5,
            volatility_multiplier=1.0,
            news_adjustment=1.0,
            structure_confidence=0.7,
            
            position_sizing_method='FIXED',
            partial_close_levels=[0.4, 0.35, 0.25],
            trailing_stop_trigger=2.0,
            
            adjustment_reasons=["Safe fallback due to optimization error"],
            market_condition=MarketCondition.NEUTRAL,
            confidence_score=60.0
        )

# Demo function
async def demo_dynamic_optimization():
    """Demo dynamic AI optimization"""
    print("🧠 Dynamic AI Optimization Demo")
    print("=" * 40)
    
    optimizer = DynamicAIOptimizer()
    
    # Sample market scenarios
    scenarios = [
        {
            'name': 'Bullish Break + Positive News',
            'market_data': {
                'symbol': 'BTC/USDT',
                'signal_strength': 85,
                'price_change_24h': 3.5,
                'atr_pct': 2.1,
                'bos_bullish': True,
                'fvgs': [{'type': 'bullish'}],
                'volume_ratio': 1.8
            },
            'news_analysis': {
                'sentiment_score': 2.5,
                'impact_level': 'MEDIUM',
                'adjustment_factor': 1.2
            }
        },
        {
            'name': 'Bearish Break + Negative News',
            'market_data': {
                'symbol': 'ETH/USDT',
                'signal_strength': 75,
                'price_change_24h': -4.2,
                'atr_pct': 3.8,
                'bos_bearish': True,
                'fvgs': [],
                'volume_ratio': 2.1
            },
            'news_analysis': {
                'sentiment_score': -2.8,
                'impact_level': 'HIGH',
                'adjustment_factor': 0.7
            }
        },
        {
            'name': 'High Volatility Sideways',
            'market_data': {
                'symbol': 'BTC/USDT',
                'signal_strength': 65,
                'price_change_24h': 0.8,
                'atr_pct': 5.2,
                'bos_bullish': False,
                'fvgs': [{'type': 'bearish'}],
                'volume_ratio': 0.9
            },
            'news_analysis': {
                'sentiment_score': -0.5,
                'impact_level': 'HIGH',
                'adjustment_factor': 0.8
            }
        }
    ]
    
    for scenario in scenarios:
        print(f"\n📊 Scenario: {scenario['name']}")
        print("-" * 40)
        
        params = await optimizer.optimize_dynamic_parameters(
            scenario['market_data'], 
            scenario['news_analysis']
        )
        
        print(f"🎯 Market Condition: {params.market_condition.value}")
        print(f"📈 Leverage: {params.base_leverage}x → {params.leverage}x")
        print(f"⚠️ Risk: {params.risk_per_trade:.1%}")
        print(f"📊 R/R Ratio: 1:{params.risk_reward_ratio:.1f}")
        print(f"🛑 Stop Loss: {params.stop_loss_pct:.1%}")
        print(f"🎲 Confidence: {params.confidence_score:.1f}%")
        print(f"📏 Position Method: {params.position_sizing_method}")
        print(f"⚡ Volatility Adj: {params.volatility_multiplier:.2f}")
        print(f"📰 News Adj: {params.news_adjustment:.2f}")
        
        print(f"🧠 AI Reasoning:")
        for reason in params.adjustment_reasons:
            print(f"   • {reason}")

if __name__ == "__main__":
    asyncio.run(demo_dynamic_optimization())