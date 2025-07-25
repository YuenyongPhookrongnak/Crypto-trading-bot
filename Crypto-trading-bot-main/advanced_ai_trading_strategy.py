#!/usr/bin/env python3
"""
Advanced AI-Enhanced Trading Strategy

This strategy combines technical analysis with AI insights from GPT/Claude
based on professional trading confluences and risk management.
"""

import asyncio
import ccxt
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import os
import ta
from dataclasses import dataclass

# Try to import AI API libraries
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

@dataclass
class TradingSetup:
    """Trading setup with confluences"""
    symbol: str
    direction: str  # BUY/SELL
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward: float
    confluences: List[str]
    confidence: float
    timeframe: str
    analysis: str
    ai_insight: str = ""

class AdvancedMarketAnalyzer:
    """Advanced market analysis with AI integration"""
    
    def __init__(self):
        # AI API Keys
        self.openai_api_key = os.getenv('OPENAI_API_KEY', '')
        self.claude_api_key = os.getenv('CLAUDE_API_KEY', '')
        
        # News API
        self.news_api_key = os.getenv('NEWS_API_KEY', '')
        
        # Technical Analysis Settings
        self.timeframes = ['4h', '1h', '15m']
        self.major_pairs = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
        
        print("🧠 Advanced AI Trading Strategy")
        print("=" * 40)
        
        if OPENAI_AVAILABLE and self.openai_api_key:
            print("✅ OpenAI GPT available")
            openai.api_key = self.openai_api_key
        else:
            print("⚠️  OpenAI not available")
        
        if self.claude_api_key:
            print("✅ Claude API available")
        else:
            print("⚠️  Claude API not available")
    
    def calculate_ema(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return data.ewm(span=period).mean()
    
    def identify_fvg(self, df: pd.DataFrame) -> List[Dict]:
        """Identify Fair Value Gaps (FVG)"""
        fvgs = []
        
        for i in range(2, len(df)):
            current = df.iloc[i]
            previous = df.iloc[i-1]
            before = df.iloc[i-2]
            
            # Bullish FVG: previous high < current low
            if previous['high'] < current['low']:
                fvg = {
                    'type': 'bullish',
                    'start_idx': i-1,
                    'end_idx': i,
                    'top': current['low'],
                    'bottom': previous['high'],
                    'timestamp': current.name,
                    'filled': False
                }
                fvgs.append(fvg)
            
            # Bearish FVG: previous low > current high
            elif previous['low'] > current['high']:
                fvg = {
                    'type': 'bearish',
                    'start_idx': i-1,
                    'end_idx': i,
                    'top': previous['low'],
                    'bottom': current['high'],
                    'timestamp': current.name,
                    'filled': False
                }
                fvgs.append(fvg)
        
        return fvgs[-10:]  # Keep last 10 FVGs
    
    def detect_bos_coc(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect Break of Structure (BOS) and Change of Character (CoC)"""
        highs = df['high'].rolling(10).max()
        lows = df['low'].rolling(10).min()
        
        current_price = df['close'].iloc[-1]
        recent_high = highs.iloc[-5:].max()
        recent_low = lows.iloc[-5:].min()
        
        # Simple BOS detection
        bos_bullish = current_price > recent_high
        bos_bearish = current_price < recent_low
        
        # Change of Character (more complex pattern)
        price_changes = df['close'].pct_change().rolling(5).mean()
        momentum_change = abs(price_changes.iloc[-1]) > abs(price_changes.iloc[-5])
        
        return {
            'bos_bullish': bos_bullish,
            'bos_bearish': bos_bearish,
            'momentum_change': momentum_change,
            'recent_high': recent_high,
            'recent_low': recent_low
        }
    
    def analyze_support_resistance(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        """Identify support and resistance levels"""
        # Find pivot points
        high_pivots = []
        low_pivots = []
        
        for i in range(5, len(df) - 5):
            # High pivot: higher than 5 bars before and after
            if all(df['high'].iloc[i] >= df['high'].iloc[i-j] for j in range(1, 6)) and \\
               all(df['high'].iloc[i] >= df['high'].iloc[i+j] for j in range(1, 6)):
                high_pivots.append(df['high'].iloc[i])
            
            # Low pivot: lower than 5 bars before and after
            if all(df['low'].iloc[i] <= df['low'].iloc[i-j] for j in range(1, 6)) and \\
               all(df['low'].iloc[i] <= df['low'].iloc[i+j] for j in range(1, 6)):
                low_pivots.append(df['low'].iloc[i])
        
        # Get recent levels
        resistance_levels = sorted(high_pivots[-5:], reverse=True)
        support_levels = sorted(low_pivots[-5:])
        
        return {
            'resistance': resistance_levels,
            'support': support_levels
        }
    
    def check_ema_alignment(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check EMA alignment for trend confirmation"""
        df['ema20'] = self.calculate_ema(df['close'], 20)
        df['ema50'] = self.calculate_ema(df['close'], 50)
        df['ema200'] = self.calculate_ema(df['close'], 200)
        
        current_price = df['close'].iloc[-1]
        ema20 = df['ema20'].iloc[-1]
        ema50 = df['ema50'].iloc[-1]
        ema200 = df['ema200'].iloc[-1]
        
        # Bullish alignment: price > ema20 > ema50 > ema200
        bullish_alignment = current_price > ema20 > ema50 > ema200
        
        # Bearish alignment: price < ema20 < ema50 < ema200
        bearish_alignment = current_price < ema20 < ema50 < ema200
        
        return {
            'bullish_alignment': bullish_alignment,
            'bearish_alignment': bearish_alignment,
            'ema20': ema20,
            'ema50': ema50,
            'ema200': ema200,
            'price_above_ema20': current_price > ema20
        }
    
    async def get_market_sentiment_news(self, symbol: str) -> Dict[str, Any]:
        """Get market sentiment from news"""
        try:
            # Extract base asset (BTC from BTC/USDT)
            base_asset = symbol.split('/')[0].lower()
            
            # NewsAPI integration (if available)
            if self.news_api_key:
                url = f"https://newsapi.org/v2/everything"
                params = {
                    'q': f'{base_asset} OR bitcoin OR crypto' if base_asset == 'btc' else f'{base_asset} OR cryptocurrency',
                    'sortBy': 'publishedAt',
                    'pageSize': 10,
                    'apiKey': self.news_api_key,
                    'language': 'en'
                }
                
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    news_data = response.json()
                    articles = news_data.get('articles', [])
                    
                    # Simple sentiment analysis
                    positive_words = ['bullish', 'rally', 'surge', 'gains', 'positive', 'up', 'rise']
                    negative_words = ['bearish', 'crash', 'dump', 'negative', 'down', 'fall', 'decline']
                    
                    sentiment_score = 0
                    for article in articles:
                        title = article.get('title', '').lower()
                        description = article.get('description', '').lower()
                        text = f"{title} {description}"
                        
                        for word in positive_words:
                            sentiment_score += text.count(word) * 1
                        for word in negative_words:
                            sentiment_score -= text.count(word) * 1
                    
                    return {
                        'sentiment_score': sentiment_score,
                        'news_count': len(articles),
                        'latest_headline': articles[0].get('title', '') if articles else ''
                    }
            
            return {'sentiment_score': 0, 'news_count': 0, 'latest_headline': ''}
            
        except Exception as e:
            print(f"⚠️  Error fetching news: {e}")
            return {'sentiment_score': 0, 'news_count': 0, 'latest_headline': ''}
    
    async def get_ai_analysis(self, market_data: Dict[str, Any], symbol: str) -> str:
        """Get AI analysis from GPT or Claude"""
        try:
            # Prepare market context
            context = f"""
            Market Analysis for {symbol}:
            
            Current Price: ${market_data['current_price']:,.2f}
            24h Change: {market_data.get('price_change_24h', 0):.2f}%
            
            Technical Indicators:
            - RSI: {market_data.get('rsi', 50):.1f}
            - EMA20: ${market_data.get('ema20', 0):,.2f}
            - EMA50: ${market_data.get('ema50', 0):,.2f}
            - EMA200: ${market_data.get('ema200', 0):,.2f}
            
            Structure Analysis:
            - BOS Bullish: {market_data.get('bos_bullish', False)}
            - BOS Bearish: {market_data.get('bos_bearish', False)}
            - EMA Alignment: {market_data.get('ema_alignment', 'Neutral')}
            
            Fair Value Gaps: {len(market_data.get('fvgs', []))} identified
            Support/Resistance: {len(market_data.get('support_levels', []))} support, {len(market_data.get('resistance_levels', []))} resistance
            
            News Sentiment: {market_data.get('sentiment_score', 0)} (from {market_data.get('news_count', 0)} articles)
            """
            
            # Try OpenAI GPT first
            if OPENAI_AVAILABLE and self.openai_api_key:
                try:
                    prompt = f"""
                    As a professional cryptocurrency trader, analyze the following market data and provide:
                    1. Current market structure assessment
                    2. Key confluence factors for potential trades
                    3. Risk assessment and market bias
                    4. Specific entry/exit recommendations if any
                    
                    {context}
                    
                    Provide a concise but comprehensive analysis focusing on actionable insights.
                    """
                    
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are an expert cryptocurrency trader with deep knowledge of technical analysis, market structure, and risk management."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=500,
                        temperature=0.7
                    )
                    
                    return response.choices[0].message.content.strip()
                
                except Exception as e:
                    print(f"⚠️  OpenAI API error: {e}")
            
            # Try Claude API as fallback
            if self.claude_api_key:
                try:
                    claude_url = "https://api.anthropic.com/v1/messages"
                    headers = {
                        "Content-Type": "application/json",
                        "x-api-key": self.claude_api_key,
                        "anthropic-version": "2023-06-01"
                    }
                    
                    prompt = f"""
                    As a professional cryptocurrency trader, analyze this market data for {symbol}:
                    
                    {context}
                    
                    Provide:
                    1. Market structure analysis
                    2. Key confluences for potential trades
                    3. Risk assessment and bias
                    4. Specific recommendations
                    
                    Keep it concise and actionable.
                    """
                    
                    data = {
                        "model": "claude-3-sonnet-20240229",
                        "max_tokens": 500,
                        "messages": [
                            {"role": "user", "content": prompt}
                        ]
                    }
                    
                    response = requests.post(claude_url, headers=headers, json=data, timeout=30)
                    if response.status_code == 200:
                        return response.json()['content'][0]['text']
                
                except Exception as e:
                    print(f"⚠️  Claude API error: {e}")
            
            # Fallback: Rule-based analysis
            return self.generate_rule_based_analysis(market_data, symbol)
            
        except Exception as e:
            print(f"❌ Error getting AI analysis: {e}")
            return f"Technical analysis shows mixed signals for {symbol}. Monitor key levels."
    
    def generate_rule_based_analysis(self, market_data: Dict[str, Any], symbol: str) -> str:
        """Generate analysis using rule-based logic as fallback"""
        analysis = []
        
        # Price action analysis
        current_price = market_data.get('current_price', 0)
        price_change = market_data.get('price_change_24h', 0)
        
        if price_change > 2:
            analysis.append("📈 Strong bullish momentum (+2%)")
        elif price_change < -2:
            analysis.append("📉 Strong bearish pressure (-2%)")
        else:
            analysis.append("⚡ Consolidating price action")
        
        # RSI analysis
        rsi = market_data.get('rsi', 50)
        if rsi > 70:
            analysis.append("🔴 RSI overbought - potential reversal")
        elif rsi < 30:
            analysis.append("🟢 RSI oversold - potential bounce")
        else:
            analysis.append(f"⚪ RSI neutral ({rsi:.1f})")
        
        # Structure analysis
        if market_data.get('bos_bullish'):
            analysis.append("🔥 Bullish break of structure confirmed")
        elif market_data.get('bos_bearish'):
            analysis.append("❄️ Bearish break of structure confirmed")
        
        # EMA analysis
        if market_data.get('bullish_alignment'):
            analysis.append("📊 Bullish EMA alignment - trend up")
        elif market_data.get('bearish_alignment'):
            analysis.append("📊 Bearish EMA alignment - trend down")
        
        # FVG analysis
        fvg_count = len(market_data.get('fvgs', []))
        if fvg_count > 0:
            analysis.append(f"🎯 {fvg_count} FVG(s) identified for potential reversal")
        
        return f"{symbol} Analysis: " + " | ".join(analysis)
    
    async def comprehensive_analysis(self, exchange: ccxt.Exchange, symbol: str, timeframe: str = '1h') -> Dict[str, Any]:
        """Perform comprehensive market analysis"""
        try:
            # Fetch market data
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=200)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Calculate technical indicators
            df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
            df['macd'] = ta.trend.MACD(df['close']).macd()
            df['bb_upper'] = ta.volatility.BollingerBands(df['close']).bollinger_hband()
            df['bb_lower'] = ta.volatility.BollingerBands(df['close']).bollinger_lband()
            
            current_price = df['close'].iloc[-1]
            price_change_24h = ((current_price - df['close'].iloc[-24]) / df['close'].iloc[-24]) * 100
            
            # Advanced analysis
            fvgs = self.identify_fvg(df)
            bos_coc = self.detect_bos_coc(df)
            sr_levels = self.analyze_support_resistance(df)
            ema_analysis = self.check_ema_alignment(df)
            
            # Get news sentiment
            news_sentiment = await self.get_market_sentiment_news(symbol)
            
            # Compile all data
            market_data = {
                'symbol': symbol,
                'timeframe': timeframe,
                'current_price': current_price,
                'price_change_24h': price_change_24h,
                'rsi': df['rsi'].iloc[-1],
                'macd': df['macd'].iloc[-1],
                'volume': df['volume'].iloc[-1],
                'fvgs': fvgs,
                'bos_bullish': bos_coc['bos_bullish'],
                'bos_bearish': bos_coc['bos_bearish'],
                'momentum_change': bos_coc['momentum_change'],
                'support_levels': sr_levels['support'],
                'resistance_levels': sr_levels['resistance'],
                'ema20': ema_analysis['ema20'],
                'ema50': ema_analysis['ema50'],
                'ema200': ema_analysis['ema200'],
                'bullish_alignment': ema_analysis['bullish_alignment'],
                'bearish_alignment': ema_analysis['bearish_alignment'],
                'sentiment_score': news_sentiment['sentiment_score'],
                'news_count': news_sentiment['news_count'],
                'latest_headline': news_sentiment['latest_headline']
            }
            
            # Get AI analysis
            ai_insight = await self.get_ai_analysis(market_data, symbol)
            market_data['ai_insight'] = ai_insight
            
            return market_data
            
        except Exception as e:
            print(f"❌ Error in comprehensive analysis: {e}")
            return {}
    
    def evaluate_trading_confluences(self, market_data: Dict[str, Any]) -> Tuple[bool, List[str], float]:
        """Evaluate trading confluences based on professional strategy"""
        confluences = []
        signal_strength = 0
        
        # 1. BOS or CoC confluence
        if market_data.get('bos_bullish') or market_data.get('bos_bearish'):
            confluences.append("✅ BOS/CoC confirmed")
            signal_strength += 25
        
        # 2. FVG confluence
        fvgs = market_data.get('fvgs', [])
        if fvgs:
            current_price = market_data['current_price']
            for fvg in fvgs[-3:]:  # Check last 3 FVGs
                if fvg['bottom'] <= current_price <= fvg['top']:
                    confluences.append("✅ Price in FVG zone")
                    signal_strength += 20
                    break
        
        # 3. RSI confluence
        rsi = market_data.get('rsi', 50)
        if rsi > 70:
            confluences.append("✅ RSI overbought (reversal zone)")
            signal_strength += 15
        elif rsi < 30:
            confluences.append("✅ RSI oversold (reversal zone)")
            signal_strength += 15
        
        # 4. EMA alignment confluence
        if market_data.get('bullish_alignment'):
            confluences.append("✅ Bullish EMA alignment")
            signal_strength += 20
        elif market_data.get('bearish_alignment'):
            confluences.append("✅ Bearish EMA alignment")
            signal_strength += 20
        
        # 5. Support/Resistance retest
        current_price = market_data['current_price']
        support_levels = market_data.get('support_levels', [])
        resistance_levels = market_data.get('resistance_levels', [])
        
        for level in support_levels + resistance_levels:
            if abs(current_price - level) / current_price <= 0.005:  # Within 0.5%
                confluences.append("✅ S/R level retest")
                signal_strength += 15
                break
        
        # 6. News sentiment confluence
        sentiment = market_data.get('sentiment_score', 0)
        if abs(sentiment) >= 3:
            direction = "bullish" if sentiment > 0 else "bearish"
            confluences.append(f"✅ {direction.title()} news sentiment")
            signal_strength += 10
        
        # Minimum 3 confluences required
        has_enough_confluences = len(confluences) >= 3 and signal_strength >= 60
        
        return has_enough_confluences, confluences, signal_strength
    
    def generate_trading_setup(self, market_data: Dict[str, Any]) -> Optional[TradingSetup]:
        """Generate trading setup based on confluences"""
        try:
            has_signal, confluences, strength = self.evaluate_trading_confluences(market_data)
            
            if not has_signal:
                return None
            
            current_price = market_data['current_price']
            symbol = market_data['symbol']
            
            # Determine direction
            bullish_signals = market_data.get('bos_bullish', False) or market_data.get('bullish_alignment', False)
            bearish_signals = market_data.get('bos_bearish', False) or market_data.get('bearish_alignment', False)
            
            if bullish_signals and not bearish_signals:
                direction = 'BUY'
                # Calculate levels for long position
                stop_loss = current_price * 0.98  # 2% stop loss
                take_profit = current_price * 1.06  # 6% take profit (1:3 RR)
                
            elif bearish_signals and not bullish_signals:
                direction = 'SELL'
                # Calculate levels for short position
                stop_loss = current_price * 1.02  # 2% stop loss
                take_profit = current_price * 0.94  # 6% take profit (1:3 RR)
                
            else:
                return None  # Conflicting signals
            
            # Calculate risk-reward ratio
            if direction == 'BUY':
                risk = abs(current_price - stop_loss)
                reward = abs(take_profit - current_price)
            else:
                risk = abs(stop_loss - current_price)
                reward = abs(current_price - take_profit)
            
            risk_reward = reward / risk if risk > 0 else 0
            
            # Minimum 1:2.5 RR required
            if risk_reward < 2.5:
                return None
            
            # Generate setup
            setup = TradingSetup(
                symbol=symbol,
                direction=direction,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward=risk_reward,
                confluences=confluences,
                confidence=min(strength, 95),  # Cap at 95%
                timeframe=market_data.get('timeframe', '1h'),
                analysis=market_data.get('ai_insight', ''),
                ai_insight=market_data.get('ai_insight', '')
            )
            
            return setup
            
        except Exception as e:
            print(f"❌ Error generating trading setup: {e}")
            return None

class AdvancedAITradingBot:
    """Advanced AI Trading Bot with professional confluences"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.analyzer = AdvancedMarketAnalyzer()
        
        # Initialize exchange
        self.exchange = None
        self.api_key = os.getenv('BINANCE_API_KEY', '')
        self.secret = os.getenv('BINANCE_SECRET', '')
        self.testnet = os.getenv('BINANCE_TESTNET', 'true').lower() == 'true'
        
        # Trading settings
        self.symbols = self.config.get('symbols', ['BTC/USDT', 'ETH/USDT'])
        self.timeframes = self.config.get('timeframes', ['4h', '1h'])
        self.max_positions = self.config.get('max_positions', 2)
        self.risk_per_trade = self.config.get('risk_per_trade', 0.01)
        
        # Risk management
        self.daily_loss_limit = 3  # Stop after 3 losses
        self.consecutive_losses = 0
        self.daily_trades = 0
        self.max_daily_trades = 5
        
        print("🤖 Advanced AI Trading Bot initialized")
        print(f"🎯 Symbols: {', '.join(self.symbols)}")
        print(f"📊 Timeframes: {', '.join(self.timeframes)}")
        print(f"🧪 Mode: {'TESTNET' if self.testnet else '🔴 LIVE'}")
    
    def initialize_exchange(self):
        """Initialize exchange connection"""
        try:
            self.exchange = ccxt.binance({
                'apiKey': self.api_key,
                'secret': self.secret,
                'sandbox': self.testnet,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',
                    'adjustForTimeDifference': True,
                }
            })
            
            markets = self.exchange.load_markets()
            balance = self.exchange.fetch_balance()
            
            print("✅ Exchange connected successfully")
            print(f"💰 USDT Balance: {balance['USDT']['total']:,.2f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Exchange initialization failed: {e}")
            return False
    
    async def scan_for_setups(self) -> List[TradingSetup]:
        """Scan all symbols and timeframes for trading setups"""
        setups = []
        
        print("🔍 Scanning for AI-enhanced trading setups...")
        
        for symbol in self.symbols:
            for timeframe in self.timeframes:
                try:
                    print(f"📊 Analyzing {symbol} {timeframe}...")
                    
                    # Get comprehensive analysis
                    analysis = await self.analyzer.comprehensive_analysis(
                        self.exchange, symbol, timeframe
                    )
                    
                    if analysis:
                        # Generate trading setup
                        setup = self.analyzer.generate_trading_setup(analysis)
                        
                        if setup:
                            setups.append(setup)
                            print(f"✅ Setup found: {setup.direction} {symbol}")
                            print(f"   Confluences: {len(setup.confluences)}")
                            print(f"   Confidence: {setup.confidence:.1f}%")
                            print(f"   RR: 1:{setup.risk_reward:.1f}")
                        else:
                            print(f"⏳ No setup for {symbol} {timeframe}")
                
                except Exception as e:
                    print(f"❌ Error analyzing {symbol} {timeframe}: {e}")
                
                # Brief pause between analyses
                await asyncio.sleep(2)
        
        return setups
    
    def display_setup_details(self, setup: TradingSetup):
        """Display detailed setup information"""
        print(f"\\n🎯 TRADING SETUP DETECTED")
        print("=" * 40)
        print(f"📊 Symbol: {setup.symbol}")
        print(f"📈 Direction: {setup.direction}")
        print(f"💰 Entry: ${setup.entry_price:,.2f}")
        print(f"🛑 Stop Loss: ${setup.stop_loss:,.2f}")
        print(f"🎯 Take Profit: ${setup.take_profit:,.2f}")
        print(f"📊 Risk/Reward: 1:{setup.risk_reward:.1f}")
        print(f"🎲 Confidence: {setup.confidence:.1f}%")
        print(f"⏰ Timeframe: {setup.timeframe}")
        
        print(f"\\n✅ Confluences ({len(setup.confluences)}):")
        for confluence in setup.confluences:
            print(f"   {confluence}")
        
        if setup.ai_insight:
            print(f"\\n🧠 AI Analysis:")
            print(f"   {setup.ai_insight}")
    
    async def run_ai_trading_session(self, duration_hours: int = 2):
        """Run AI-enhanced trading session"""
        print(f"\\n🚀 Starting Advanced AI Trading Session")
        print("=" * 50)
        print(f"⏰ Duration: {duration_hours} hours")
        print(f"🧠 AI Analysis: {'✅ Active' if self.analyzer.openai_api_key or self.analyzer.claude_api_key else '⚠️ Fallback mode'}")
        
        end_time = datetime.utcnow() + timedelta(hours=duration_hours)
        scan_count = 0
        
        try:
            while datetime.utcnow() < end_time:
                scan_count += 1
                print(f"\n🔍 AI Scan #{scan_count} - {datetime.utcnow().strftime('%H:%M:%S')}")
                
                # Check daily limits
                if self.consecutive_losses >= self.daily_loss_limit:
                    print(f"🛑 Daily loss limit reached ({self.daily_loss_limit} losses)")
                    break
                
                if self.daily_trades >= self.max_daily_trades:
                    print(f"🛑 Daily trade limit reached ({self.max_daily_trades} trades)")
                    break
                
                # Scan for setups
                setups = await self.scan_for_setups()
                
                if setups:
                    print(f"\n🎯 Found {len(setups)} potential setups:")
                    
                    # Sort by confidence
                    setups.sort(key=lambda x: x.confidence, reverse=True)
                    
                    for i, setup in enumerate(setups[:2]):  # Top 2 setups
                        self.display_setup_details(setup)
                        
                        # Ask user for confirmation (in real trading, this would be automated)
                        if not self.testnet:
                            confirm = input(f"\n💰 Execute this trade? (y/N): ")
                            if confirm.lower() != 'y':
                                print("⏭️ Skipping trade")
                                continue
                        
                        # Execute trade (placeholder)
                        print(f"\n📝 Trade logged for execution")
                        self.daily_trades += 1
                
                else:
                    print("⏳ No AI setups found in this scan")
                
                # Wait 10 minutes between scans
                print(f"⏱️ Next scan in 10 minutes...")
                for i in range(600):  # 10 minutes
                    await asyncio.sleep(1)
                    if (i + 1) % 60 == 0:
                        print(f"   {10 - (i + 1)//60} minutes remaining...")
                
        except KeyboardInterrupt:
            print(f"\n⏹️ AI Trading session stopped by user")
        
        # Session summary
        print(f"\n📋 AI Trading Session Summary:")
        print("=" * 40)
        print(f"Duration: {scan_count} scans")
        print(f"Trades executed: {self.daily_trades}")
        print(f"Consecutive losses: {self.consecutive_losses}")

async def demo_ai_analysis():
    """Demo the AI analysis system"""
    print("🧪 AI Trading Strategy Demo")
    print("=" * 35)
    
    # Initialize analyzer
    analyzer = AdvancedMarketAnalyzer()
    
    # Initialize exchange
    try:
        exchange = ccxt.binance({
            'apiKey': os.getenv('BINANCE_API_KEY', ''),
            'secret': os.getenv('BINANCE_SECRET', ''),
            'sandbox': True,
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        exchange.load_markets()
        print("✅ Exchange connected for demo")
    except Exception as e:
        print(f"❌ Exchange connection failed: {e}")
        return
    
    # Demo analysis
    test_symbols = ['BTC/USDT', 'ETH/USDT']
    
    for symbol in test_symbols:
        print(f"\n🔍 Demo Analysis: {symbol}")
        print("-" * 30)
        
        try:
            # Get comprehensive analysis
            analysis = await analyzer.comprehensive_analysis(exchange, symbol, '1h')
            
            if analysis:
                print(f"💰 Current Price: ${analysis['current_price']:,.2f}")
                print(f"📊 24h Change: {analysis['price_change_24h']:+.2f}%")
                print(f"📈 RSI: {analysis['rsi']:.1f}")
                print(f"🎯 FVGs: {len(analysis['fvgs'])}")
                print(f"🔄 BOS Bullish: {analysis['bos_bullish']}")
                print(f"🔄 BOS Bearish: {analysis['bos_bearish']}")
                print(f"📰 News Sentiment: {analysis['sentiment_score']}")
                
                # Check for trading setup
                setup = analyzer.generate_trading_setup(analysis)
                
                if setup:
                    print(f"\n✅ TRADING SETUP FOUND!")
                    print(f"   Direction: {setup.direction}")
                    print(f"   Confidence: {setup.confidence:.1f}%")
                    print(f"   Risk/Reward: 1:{setup.risk_reward:.1f}")
                    print(f"   Confluences: {len(setup.confluences)}")
                    
                    for confluence in setup.confluences:
                        print(f"     • {confluence}")
                    
                    if setup.ai_insight:
                        print(f"\n🧠 AI Insight:")
                        print(f"   {setup.ai_insight[:200]}...")
                else:
                    print("⏳ No trading setup - insufficient confluences")
                    
                    # Show what confluences were found
                    has_signal, confluences, strength = analyzer.evaluate_trading_confluences(analysis)
                    print(f"📊 Signal strength: {strength}% (need 60%+)")
                    print(f"✅ Confluences found: {len(confluences)}/3 required")
                    for confluence in confluences:
                        print(f"     • {confluence}")
        
        except Exception as e:
            print(f"❌ Analysis failed: {e}")

async def main():
    """Main function"""
    print("🤖 Advanced AI Trading Strategy")
    print("=" * 40)
    
    print("\n🎯 Options:")
    print("1. Demo AI Analysis")
    print("2. Run AI Trading Bot")
    print("3. Exit")
    
    try:
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            await demo_ai_analysis()
            
        elif choice == "2":
            # Configure bot
            config = {
                'symbols': ['BTC/USDT', 'ETH/USDT'],
                'timeframes': ['4h', '1h'],
                'max_positions': 2,
                'risk_per_trade': 0.01
            }
            
            bot = AdvancedAITradingBot(config)
            
            if bot.initialize_exchange():
                duration = int(input("Trading duration (hours): ") or "2")
                await bot.run_ai_trading_session(duration)
            else:
                print("❌ Failed to initialize trading bot")
        
        print("\n✅ AI Trading Strategy session completed!")
        
    except KeyboardInterrupt:
        print("\n👋 Session interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())