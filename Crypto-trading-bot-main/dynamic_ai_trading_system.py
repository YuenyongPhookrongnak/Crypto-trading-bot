#!/usr/bin/env python3
"""
Dynamic AI-Controlled Trading System

AI analyzes market conditions and dynamically adjusts:
- Leverage (1x-25x based on confidence)
- Risk per trade (1%-10% based on setup quality)
- Risk/Reward ratio (1:3 to 1:5 based on market structure)
- Stop loss placement (FVG, S/R levels)
- Take profit targets (multiple levels)
"""

import asyncio
import ccxt
import pandas as pd
import numpy as np
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import os

@dataclass
class DynamicTradingParameters:
    """AI-determined trading parameters"""
    leverage: int
    risk_pct: float
    risk_reward_ratio: float
    stop_loss_level: float
    take_profit_levels: List[float]
    position_size: float
    confidence_score: float
    reasoning: List[str]
    trade_style: str  # "SCALP", "SWING", "BREAKOUT"
    urgency: str  # "LOW", "MEDIUM", "HIGH"

class AIParameterOptimizer:
    """AI system that dynamically optimizes trading parameters"""
    
    def __init__(self):
        # AI API configuration
        self.openai_api_key = os.getenv('OPENAI_API_KEY', '')
        self.claude_api_key = os.getenv('CLAUDE_API_KEY', '')
        
        # Dynamic ranges
        self.leverage_range = (1, 25)  # 1x to 25x
        self.risk_range = (0.01, 0.10)  # 1% to 10%
        self.rr_range = (3.0, 5.0)  # 1:3 to 1:5
        
        # Market condition thresholds
        self.high_confidence_threshold = 85
        self.medium_confidence_threshold = 70
        self.volatility_high_threshold = 0.05  # 5%
        
        print("🧠 AI Parameter Optimizer initialized")
        print(f"   Leverage Range: {self.leverage_range[0]}x - {self.leverage_range[1]}x")
        print(f"   Risk Range: {self.risk_range[0]:.0%} - {self.risk_range[1]:.0%}")
        print(f"   R/R Range: 1:{self.rr_range[0]} - 1:{self.rr_range[1]}")
    
    async def get_ai_parameter_analysis(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get AI analysis for optimal trading parameters"""
        try:
            # Prepare comprehensive market context
            context = self._prepare_market_context(market_data)
            
            # Try different AI providers
            ai_response = await self._get_ai_response(context, market_data['symbol'])
            
            if ai_response:
                return ai_response
            else:
                # Fallback to rule-based optimization
                return self._rule_based_parameter_optimization(market_data)
                
        except Exception as e:
            print(f"❌ Error in AI parameter analysis: {e}")
            return self._rule_based_parameter_optimization(market_data)
    
    def _prepare_market_context(self, market_data: Dict[str, Any]) -> str:
        """Prepare detailed market context for AI analysis"""
        
        # Calculate additional metrics
        volatility = abs(market_data.get('price_change_24h', 0))
        volume_ratio = market_data.get('volume_ratio', 1.0)
        rsi = market_data.get('rsi', 50)
        
        # Determine market regime
        if volatility > 5:
            market_regime = "HIGH_VOLATILITY"
        elif volatility > 2:
            market_regime = "MEDIUM_VOLATILITY"
        else:
            market_regime = "LOW_VOLATILITY"
        
        # Count confluences
        confluences = len(market_data.get('confluences', []))
        fvg_count = len(market_data.get('fvgs', []))
        
        context = f"""
        MARKET ANALYSIS FOR DYNAMIC PARAMETER OPTIMIZATION:
        
        Symbol: {market_data['symbol']}
        Current Price: ${market_data['current_price']:,.2f}
        24h Change: {market_data.get('price_change_24h', 0):+.2f}%
        Market Regime: {market_regime}
        
        TECHNICAL INDICATORS:
        - RSI: {rsi:.1f} ({'Overbought' if rsi > 70 else 'Oversold' if rsi < 30 else 'Neutral'})
        - Volume Ratio: {volume_ratio:.1f}x average
        - Volatility: {volatility:.2f}%
        
        STRUCTURE ANALYSIS:
        - BOS Bullish: {market_data.get('bos_bullish', False)}
        - BOS Bearish: {market_data.get('bos_bearish', False)}
        - EMA Alignment: {market_data.get('ema_alignment', 'None')}
        - Confluence Count: {confluences}/6 possible
        - FVG Zones: {fvg_count} identified
        
        SUPPORT/RESISTANCE:
        - Support Levels: {len(market_data.get('support_levels', []))}
        - Resistance Levels: {len(market_data.get('resistance_levels', []))}
        
        NEWS SENTIMENT:
        - Sentiment Score: {market_data.get('sentiment_score', 0)}
        - News Count: {market_data.get('news_count', 0)} articles
        
        CURRENT SIGNAL STRENGTH: {market_data.get('signal_strength', 0)}%
        """
        
        return context
    
    async def _get_ai_response(self, context: str, symbol: str) -> Optional[Dict[str, Any]]:
        """Get AI response for parameter optimization"""
        
        prompt = f"""
        As an expert cryptocurrency trader, analyze the market data and determine OPTIMAL TRADING PARAMETERS.
        
        {context}
        
        Based on this analysis, determine:
        
        1. LEVERAGE (1-25x):
        - High confidence + low volatility = Higher leverage (15-25x)
        - Medium confidence + medium volatility = Medium leverage (5-15x)
        - Low confidence + high volatility = Lower leverage (1-5x)
        
        2. RISK PER TRADE (1%-10%):
        - Excellent setup (85%+ confidence, 4+ confluences) = 5-10%
        - Good setup (70-85% confidence, 3+ confluences) = 3-5%
        - Marginal setup (60-70% confidence, 2+ confluences) = 1-3%
        
        3. RISK/REWARD RATIO (1:3 to 1:5):
        - Scalping (high confidence, quick moves) = 1:3
        - Swing trading (strong structure) = 1:4 to 1:5
        - Based on nearby resistance levels
        
        4. STOP LOSS PLACEMENT:
        - Below/above nearest FVG zone
        - Below/above last significant support/resistance
        - Account for volatility (wider stops in volatile markets)
        
        5. TAKE PROFIT STRATEGY:
        - Partial profits at multiple levels
        - Based on resistance/support zones
        - Trail remaining position
        
        Respond ONLY with a JSON object in this exact format:
        {{
            "leverage": integer (1-25),
            "risk_pct": float (0.01-0.10),
            "risk_reward_ratio": float (3.0-5.0),
            "stop_loss_pct": float (0.01-0.05),
            "take_profit_levels": [float, float, float],
            "confidence_score": float (0-100),
            "trade_style": "SCALP" or "SWING" or "BREAKOUT",
            "urgency": "LOW" or "MEDIUM" or "HIGH",
            "reasoning": [
                "reason 1",
                "reason 2",
                "reason 3"
            ]
        }}
        
        DO NOT include any text outside the JSON object.
        """
        
        # Try OpenAI first
        if self.openai_api_key:
            try:
                import openai
                openai.api_key = self.openai_api_key
                
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are an expert cryptocurrency trader specializing in dynamic parameter optimization. Respond only with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=800,
                    temperature=0.3
                )
                
                response_text = response.choices[0].message.content.strip()
                # Clean response
                response_text = response_text.replace('```json', '').replace('```', '').strip()
                
                ai_params = json.loads(response_text)
                print(f"✅ OpenAI parameter optimization successful")
                return ai_params
                
            except Exception as e:
                print(f"⚠️ OpenAI parameter optimization failed: {e}")
        
        # Try Claude as fallback
        if self.claude_api_key:
            try:
                claude_url = "https://api.anthropic.com/v1/messages"
                headers = {
                    "Content-Type": "application/json",
                    "x-api-key": self.claude_api_key,
                    "anthropic-version": "2023-06-01"
                }
                
                data = {
                    "model": "claude-3-sonnet-20240229",
                    "max_tokens": 800,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ]
                }
                
                response = requests.post(claude_url, headers=headers, json=data, timeout=30)
                if response.status_code == 200:
                    response_text = response.json()['content'][0]['text']
                    response_text = response_text.replace('```json', '').replace('```', '').strip()
                    
                    ai_params = json.loads(response_text)
                    print(f"✅ Claude parameter optimization successful")
                    return ai_params
                    
            except Exception as e:
                print(f"⚠️ Claude parameter optimization failed: {e}")
        
        return None
    
    def _rule_based_parameter_optimization(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback rule-based parameter optimization"""
        print("🔄 Using rule-based parameter optimization")
        
        # Calculate base parameters
        confluences = len(market_data.get('confluences', []))
        signal_strength = market_data.get('signal_strength', 60)
        volatility = abs(market_data.get('price_change_24h', 0))
        rsi = market_data.get('rsi', 50)
        
        # Determine leverage based on confidence and volatility
        if signal_strength >= 85 and volatility < 2:
            leverage = 20  # High confidence, low volatility
        elif signal_strength >= 75 and volatility < 3:
            leverage = 15  # Good confidence, medium volatility
        elif signal_strength >= 65:
            leverage = 10  # Decent confidence
        else:
            leverage = 5   # Low confidence
        
        # Determine risk based on setup quality
        if confluences >= 4 and signal_strength >= 80:
            risk_pct = 0.08  # Excellent setup - 8%
        elif confluences >= 3 and signal_strength >= 70:
            risk_pct = 0.05  # Good setup - 5%
        elif confluences >= 2 and signal_strength >= 60:
            risk_pct = 0.03  # Marginal setup - 3%
        else:
            risk_pct = 0.01  # Poor setup - 1%
        
        # Determine R/R based on market structure
        if confluences >= 4:
            risk_reward = 4.0  # Strong structure
        elif confluences >= 3:
            risk_reward = 3.5  # Good structure
        else:
            risk_reward = 3.0  # Basic structure
        
        # Determine trade style
        if volatility > 4:
            trade_style = "SCALP"
            risk_reward = 3.0  # Shorter targets for scalping
        elif signal_strength >= 80:
            trade_style = "BREAKOUT"
        else:
            trade_style = "SWING"
        
        # Calculate stop loss (adaptive to volatility)
        base_sl = 0.02  # 2% base
        volatility_multiplier = min(volatility / 2, 2.0)  # Max 2x adjustment
        stop_loss_pct = base_sl * (1 + volatility_multiplier)
        stop_loss_pct = min(stop_loss_pct, 0.05)  # Cap at 5%
        
        # Calculate take profit levels
        tp1 = stop_loss_pct * (risk_reward * 0.5)  # 50% at first target
        tp2 = stop_loss_pct * (risk_reward * 0.8)  # 30% at second target
        tp3 = stop_loss_pct * risk_reward            # 20% at final target
        
        return {
            "leverage": leverage,
            "risk_pct": risk_pct,
            "risk_reward_ratio": risk_reward,
            "stop_loss_pct": stop_loss_pct,
            "take_profit_levels": [tp1, tp2, tp3],
            "confidence_score": signal_strength,
            "trade_style": trade_style,
            "urgency": "HIGH" if confluences >= 4 else "MEDIUM" if confluences >= 3 else "LOW",
            "reasoning": [
                f"Confluence count: {confluences}/6",
                f"Signal strength: {signal_strength}%",
                f"Volatility: {volatility:.1f}%",
                f"Adaptive parameters based on market conditions"
            ]
        }
    
    def optimize_parameters(self, market_data: Dict[str, Any], ai_params: Dict[str, Any]) -> DynamicTradingParameters:
        """Create optimized trading parameters"""
        try:
            current_price = market_data['current_price']
            signal_direction = market_data.get('signal_direction', 'BUY')
            
            # Extract AI parameters with safety bounds
            leverage = max(1, min(25, ai_params.get('leverage', 5)))
            risk_pct = max(0.01, min(0.10, ai_params.get('risk_pct', 0.02)))
            rr_ratio = max(3.0, min(5.0, ai_params.get('risk_reward_ratio', 3.0)))
            sl_pct = max(0.01, min(0.05, ai_params.get('stop_loss_pct', 0.02)))
            
            # Calculate precise levels based on market structure
            if signal_direction == 'BUY':
                # For long positions
                stop_loss_level = current_price * (1 - sl_pct)
                
                # Adjust stop loss to FVG or support if nearby
                support_levels = market_data.get('support_levels', [])
                if support_levels:
                    nearest_support = max([s for s in support_levels if s < current_price], default=stop_loss_level)
                    if abs(nearest_support - stop_loss_level) / current_price < 0.01:  # Within 1%
                        stop_loss_level = nearest_support * 0.998  # Slightly below support
                
                # Calculate take profit levels
                tp_base = current_price + (current_price - stop_loss_level) * rr_ratio
                tp_levels = ai_params.get('take_profit_levels', [sl_pct * rr_ratio * 0.5, sl_pct * rr_ratio * 0.8, sl_pct * rr_ratio])
                take_profit_levels = [current_price * (1 + tp) for tp in tp_levels]
                
            else:  # SELL
                # For short positions
                stop_loss_level = current_price * (1 + sl_pct)
                
                # Adjust stop loss to resistance if nearby
                resistance_levels = market_data.get('resistance_levels', [])
                if resistance_levels:
                    nearest_resistance = min([r for r in resistance_levels if r > current_price], default=stop_loss_level)
                    if abs(nearest_resistance - stop_loss_level) / current_price < 0.01:  # Within 1%
                        stop_loss_level = nearest_resistance * 1.002  # Slightly above resistance
                
                # Calculate take profit levels
                tp_levels = ai_params.get('take_profit_levels', [sl_pct * rr_ratio * 0.5, sl_pct * rr_ratio * 0.8, sl_pct * rr_ratio])
                take_profit_levels = [current_price * (1 - tp) for tp in tp_levels]
            
            # Calculate position size based on risk
            available_capital = 10000  # This should come from exchange balance
            risk_amount = available_capital * risk_pct
            
            if signal_direction == 'BUY':
                price_diff = current_price - stop_loss_level
            else:
                price_diff = stop_loss_level - current_price
            
            position_value = (risk_amount * leverage) / (price_diff / current_price)
            position_size = position_value / current_price
            
            # Create optimized parameters
            optimized_params = DynamicTradingParameters(
                leverage=leverage,
                risk_pct=risk_pct,
                risk_reward_ratio=rr_ratio,
                stop_loss_level=stop_loss_level,
                take_profit_levels=take_profit_levels,
                position_size=position_size,
                confidence_score=ai_params.get('confidence_score', 70),
                reasoning=ai_params.get('reasoning', []),
                trade_style=ai_params.get('trade_style', 'SWING'),
                urgency=ai_params.get('urgency', 'MEDIUM')
            )
            
            return optimized_params
            
        except Exception as e:
            print(f"❌ Error optimizing parameters: {e}")
            # Return safe defaults
            return DynamicTradingParameters(
                leverage=3,
                risk_pct=0.02,
                risk_reward_ratio=3.0,
                stop_loss_level=current_price * 0.98,
                take_profit_levels=[current_price * 1.03, current_price * 1.05, current_price * 1.06],
                position_size=0.001,
                confidence_score=60,
                reasoning=["Fallback parameters due to error"],
                trade_style="SWING",
                urgency="LOW"
            )

class DynamicAITradingBot:
    """Dynamic AI Trading Bot with adaptive parameters"""
    
    def __init__(self):
        # Initialize AI optimizer
        self.ai_optimizer = AIParameterOptimizer()
        
        # Exchange setup
        self.exchange = None
        self.api_key = os.getenv('BINANCE_API_KEY', '')
        self.secret = os.getenv('BINANCE_SECRET', '')
        self.testnet = os.getenv('BINANCE_TESTNET', 'true').lower() == 'true'
        
        # Risk management
        self.consecutive_losses = 0
        self.max_consecutive_losses = 3
        self.daily_trades = 0
        self.max_daily_trades = 10
        
        # Performance tracking
        self.trades_today = []
        self.total_pnl = 0.0
        
        print("🤖 Dynamic AI Trading Bot initialized")
        print("🧠 AI will dynamically adjust all parameters based on market conditions")
    
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
            
            print("✅ Exchange connected")
            print(f"💰 USDT Balance: {balance['USDT']['total']:,.2f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Exchange initialization failed: {e}")
            return False
    
    async def execute_dynamic_trade(self, market_data: Dict[str, Any]) -> bool:
        """Execute trade with AI-optimized parameters"""
        try:
            symbol = market_data['symbol']
            signal_direction = market_data.get('signal_direction', 'BUY')
            
            print(f"\n🧠 AI analyzing optimal parameters for {symbol}...")
            
            # Get AI parameter optimization
            ai_params = await self.ai_optimizer.get_ai_parameter_analysis(market_data)
            
            # Optimize parameters
            optimized = self.ai_optimizer.optimize_parameters(market_data, ai_params)
            
            # Display AI decisions
            self.display_ai_parameters(optimized, symbol, signal_direction)
            
            # Check if we should proceed
            if optimized.confidence_score < 60:
                print("⚠️ AI confidence too low - skipping trade")
                return False
            
            if self.consecutive_losses >= self.max_consecutive_losses:
                print("🛑 Max consecutive losses reached - stopping")
                return False
            
            # Execute the trade
            success = await self._place_dynamic_order(symbol, signal_direction, optimized)
            
            if success:
                self.daily_trades += 1
                print(f"✅ Dynamic trade executed successfully!")
                return True
            else:
                print(f"❌ Trade execution failed")
                return False
                
        except Exception as e:
            print(f"❌ Error executing dynamic trade: {e}")
            return False
    
    def display_ai_parameters(self, params: DynamicTradingParameters, symbol: str, direction: str):
        """Display AI-optimized parameters"""
        print(f"\n🎯 AI-OPTIMIZED TRADING PARAMETERS")
        print("=" * 45)
        print(f"📊 Symbol: {symbol} {direction}")
        print(f"🤖 AI Confidence: {params.confidence_score:.1f}%")
        print(f"⚡ Trade Style: {params.trade_style}")
        print(f"🚨 Urgency: {params.urgency}")
        print(f"")
        print(f"📈 Leverage: {params.leverage}x")
        print(f"⚠️ Risk per Trade: {params.risk_pct:.1%}")
        print(f"📊 Risk/Reward: 1:{params.risk_reward_ratio:.1f}")
        print(f"🛑 Stop Loss: ${params.stop_loss_level:,.2f}")
        print(f"💰 Position Size: {params.position_size:.6f}")
        
        print(f"\n🎯 Take Profit Levels:")
        for i, tp in enumerate(params.take_profit_levels, 1):
            print(f"   TP{i}: ${tp:,.2f}")
        
        print(f"\n🧠 AI Reasoning:")
        for reason in params.reasoning:
            print(f"   • {reason}")
    
    async def _place_dynamic_order(self, symbol: str, direction: str, params: DynamicTradingParameters) -> bool:
        """Place order with dynamic parameters"""
        try:
            # Set leverage
            self.exchange.set_leverage(params.leverage, symbol)
            print(f"📈 Leverage set to {params.leverage}x")
            
            # Set margin mode
            try:
                self.exchange.set_margin_mode('ISOLATED', symbol)
                print(f"🛡️ Margin mode: ISOLATED")
            except Exception as e:
                if "No need to change" not in str(e):
                    print(f"⚠️ Margin mode warning: {e}")
            
            # Place main order
            side = direction.lower()
            order = self.exchange.create_market_order(
                symbol=symbol,
                side=side,
                amount=params.position_size
            )
            
            print(f"✅ Market order placed: {order['id']}")
            
            # Place stop loss
            try:
                sl_side = 'sell' if direction == 'BUY' else 'buy'
                sl_order = self.exchange.create_order(
                    symbol=symbol,
                    type='STOP_MARKET',
                    side=sl_side,
                    amount=params.position_size,
                    params={'stopPrice': params.stop_loss_level}
                )
                print(f"🛑 Stop loss placed: ${params.stop_loss_level:,.2f}")
            except Exception as e:
                print(f"⚠️ Stop loss failed: {e}")
            
            # Place take profit orders
            tp_quantities = [
                params.position_size * 0.5,  # 50% at TP1
                params.position_size * 0.3,  # 30% at TP2  
                params.position_size * 0.2   # 20% at TP3
            ]
            
            for i, (tp_price, tp_qty) in enumerate(zip(params.take_profit_levels, tp_quantities), 1):
                try:
                    tp_side = 'sell' if direction == 'BUY' else 'buy'
                    tp_order = self.exchange.create_order(
                        symbol=symbol,
                        type='TAKE_PROFIT_MARKET',
                        side=tp_side,
                        amount=tp_qty,
                        params={'stopPrice': tp_price}
                    )
                    print(f"💰 TP{i} placed: ${tp_price:,.2f} ({tp_qty:.6f})")
                except Exception as e:
                    print(f"⚠️ TP{i} failed: {e}")
            
            # Track trade
            trade_record = {
                'symbol': symbol,
                'direction': direction,
                'leverage': params.leverage,
                'risk_pct': params.risk_pct,
                'confidence': params.confidence_score,
                'trade_style': params.trade_style,
                'timestamp': datetime.utcnow()
            }
            
            self.trades_today.append(trade_record)
            
            return True
            
        except Exception as e:
            print(f"❌ Order placement failed: {e}")
            return False
    
    def update_trade_result(self, success: bool, pnl: float = 0):
        """Update trade results for learning"""
        if success and pnl > 0:
            self.consecutive_losses = 0
            print(f"✅ Trade won: ${pnl:,.2f}")
        else:
            self.consecutive_losses += 1
            print(f"❌ Trade lost: ${pnl:,.2f} (Streak: {self.consecutive_losses})")
        
        self.total_pnl += pnl
        
        # Check if we hit max losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            print(f"🛑 STOP TRADING: {self.max_consecutive_losses} consecutive losses")
            print("   AI recommends taking a break and reviewing strategy")

async def demo_dynamic_ai_trading():
    """Demo the dynamic AI trading system"""
    print("🧪 Dynamic AI Trading Demo")
    print("=" * 30)
    
    # Initialize bot
    bot = DynamicAITradingBot()
    
    if not bot.initialize_exchange():
        print("❌ Cannot demo without exchange connection")
        return
    
    # Create sample market data
    sample_market_data = {
        'symbol': 'BTC/USDT',
        'current_price': 67500.0,
        'price_change_24h': 3.2,
        'rsi': 45.0,
        'signal_direction': 'BUY',
        'signal_strength': 82.0,
        'confluences': [
            '✅ Bullish BOS confirmed',
            '✅ FVG zone entry',
            '✅ RSI showing bullish divergence',
            '✅ EMA alignment bullish',
            '✅ Volume confirmation'
        ],
        'fvgs': [{'type': 'bullish', 'bottom': 67200, 'top': 67400}],
        'support_levels': [67000, 66500, 66000],
        'resistance_levels': [68000, 68500, 69000],
        'volume_ratio': 1.8,
        'sentiment_score': 5,
        'news_count': 12,
        'bos_bullish': True,
        'bos_bearish': False,
        'ema_alignment': 'bullish'
    }
    
    print("📊 Sample market conditions prepared")
    print("🧠 AI will analyze and optimize parameters...")
    
    # Execute dynamic trade
    await bot.execute_dynamic_trade(sample_market_data)
    
    print("\n📋 Demo completed!")
    print("In real trading, the bot will:")
    print("• Continuously analyze market conditions")
    print("• Adjust leverage based on confidence (1x-25x)")
    print("• Scale risk based on setup quality (1%-10%)")
    print("• Optimize R/R ratios dynamically (1:3-1:5)")
    print("• Place intelligent stop losses and take profits")
    print("• Learn from results and adapt")

if __name__ == "__main__":
    asyncio.run(demo_dynamic_ai_trading())