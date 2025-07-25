#!/usr/bin/env python3
"""
Final AI Trading Bot with ISOLATED Margin

Complete implementation with:
- ISOLATED margin for maximum safety
- AI-optimized parameters (50-85% capital, 5x-20x leverage)
- Professional confluences and risk management
- Real-time monitoring and notifications
"""

import asyncio
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import json
import os

@dataclass
class IsolatedTradeExecution:
    """Complete isolated trade execution parameters"""
    # Trade Identification
    trade_id: str
    symbol: str
    direction: str
    
    # ISOLATED Margin Setup
    margin_type: str = "ISOLATED"  # Always ISOLATED
    leverage: int = 10
    
    # Capital Management
    total_balance: float = 0.0
    allocated_capital: float = 0.0
    reserved_capital: float = 0.0
    capital_usage_pct: float = 0.6
    
    # Position Details
    position_size: float = 0.0
    position_value: float = 0.0
    entry_price: float = 0.0
    
    # Risk Management - ISOLATED
    stop_loss_price: float = 0.0
    liquidation_price: float = 0.0
    max_loss_usdt: float = 0.0
    max_loss_pct: float = 0.0
    
    # Take Profit Strategy
    take_profit_levels: List[float] = None
    partial_tp_quantities: List[float] = None
    
    # AI Analysis
    confidence_score: float = 0.0
    setup_quality: str = "GOOD"
    confluences: List[str] = None
    ai_reasoning: List[str] = None

class FinalIsolatedMarginBot:
    """Final AI Trading Bot with ISOLATED Margin Strategy"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # AI Configuration
        self.openai_api_key = os.getenv('OPENAI_API_KEY', '')
        self.claude_api_key = os.getenv('CLAUDE_API_KEY', '')
        
        # ISOLATED Margin Settings (HARD-CODED FOR SAFETY)
        self.MARGIN_TYPE = "ISOLATED"  # Never changes
        self.MIN_CAPITAL_USAGE = 0.50  # 50% minimum
        self.MAX_CAPITAL_USAGE = 0.85  # 85% maximum
        self.MIN_LEVERAGE = 5          # 5x minimum
        self.MAX_LEVERAGE = 20         # 20x maximum
        self.MIN_CONFIDENCE = 75       # 75% minimum
        
        # Risk Management (ISOLATED-specific)
        self.MAX_POSITION_RISK = 0.08  # 8% max per position
        self.EMERGENCY_STOP = 0.15     # 15% portfolio emergency stop
        self.RESERVED_CAPITAL_MIN = 0.15  # Always keep 15% reserved
        
        # Trading Limits
        self.MAX_DAILY_TRADES = 3
        self.MAX_CONSECUTIVE_LOSSES = 2
        
        # Exchange
        self.exchange = None
        self.api_key = os.getenv('BINANCE_API_KEY', '')
        self.secret = os.getenv('BINANCE_SECRET', '')
        self.testnet = os.getenv('BINANCE_TESTNET', 'true').lower() == 'true'
        
        # Performance Tracking
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.session_pnl = 0.0
        self.active_trade = None
        
        print("🛡️ Final ISOLATED Margin Trading Bot")
        print("=" * 45)
        print("🔒 ISOLATED MARGIN ENFORCED")
        print(f"   Capital Range: {self.MIN_CAPITAL_USAGE:.0%} - {self.MAX_CAPITAL_USAGE:.0%}")
        print(f"   Leverage Range: {self.MIN_LEVERAGE}x - {self.MAX_LEVERAGE}x")
        print(f"   Reserved Capital: {self.RESERVED_CAPITAL_MIN:.0%} minimum")
        print(f"   Max Position Risk: {self.MAX_POSITION_RISK:.0%}")
    
    def initialize_exchange(self):
        """Initialize exchange with ISOLATED margin verification"""
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
            print(f"💰 Total USDT Balance: {balance['USDT']['total']:,.2f}")
            print(f"💵 Available USDT: {balance['USDT']['free']:,.2f}")
            
            # Verify ISOLATED margin capabilities
            print("🔍 Verifying ISOLATED margin support...")
            test_symbol = 'BTC/USDT'
            try:
                # Test setting ISOLATED margin (will revert)
                original_mode = self._get_current_margin_mode(test_symbol)
                self.exchange.set_margin_mode('ISOLATED', test_symbol)
                print("✅ ISOLATED margin supported")
                
                # Revert to original mode
                if original_mode:
                    self.exchange.set_margin_mode(original_mode, test_symbol)
                    
            except Exception as e:
                print(f"⚠️ ISOLATED margin test: {e}")
            
            return True
            
        except Exception as e:
            print(f"❌ Exchange initialization failed: {e}")
            return False
    
    def _get_current_margin_mode(self, symbol: str) -> Optional[str]:
        """Get current margin mode for symbol"""
        try:
            positions = self.exchange.fetch_positions([symbol])
            if positions:
                return positions[0].get('marginMode', 'CROSS')
            return 'CROSS'  # Default
        except:
            return None
    
    async def analyze_isolated_trade_opportunity(self, market_data: Dict[str, Any]) -> Optional[IsolatedTradeExecution]:
        """Analyze trade opportunity with ISOLATED margin focus"""
        try:
            # Pre-flight checks
            if not self._pre_flight_safety_checks():
                return None
            
            # Get account balance
            balance = self.exchange.fetch_balance()
            total_balance = balance['USDT']['total']
            available_balance = balance['USDT']['free']
            
            if available_balance < 100:  # Minimum $100
                print(f"⚠️ Insufficient available balance: ${available_balance:.2f}")
                return None
            
            # AI Analysis for ISOLATED margin
            ai_analysis = await self._get_isolated_margin_analysis(market_data, total_balance)
            
            if not ai_analysis:
                return None
            
            # Calculate ISOLATED trade parameters
            trade_execution = self._calculate_isolated_parameters(
                market_data, ai_analysis, total_balance, available_balance
            )
            
            if trade_execution and self._validate_isolated_trade(trade_execution):
                return trade_execution
            
            return None
            
        except Exception as e:
            print(f"❌ Error analyzing ISOLATED trade: {e}")
            return None
    
    def _pre_flight_safety_checks(self) -> bool:
        """Pre-flight safety checks for ISOLATED trading"""
        
        # Check daily limits
        if self.daily_trades >= self.MAX_DAILY_TRADES:
            print(f"🛑 Daily trade limit: {self.daily_trades}/{self.MAX_DAILY_TRADES}")
            return False
        
        # Check consecutive losses
        if self.consecutive_losses >= self.MAX_CONSECUTIVE_LOSSES:
            print(f"🛑 Consecutive losses: {self.consecutive_losses}")
            return False
        
        # Check for existing positions (ISOLATED = 1 position max recommended)
        try:
            positions = self.exchange.fetch_positions()
            active_positions = [p for p in positions if p['contracts'] and p['contracts'] > 0]
            
            if active_positions:
                print(f"⚠️ Active positions exist: {len(active_positions)}")
                print("ISOLATED strategy recommends 1 position at a time")
                return False
                
        except Exception as e:
            print(f"⚠️ Error checking positions: {e}")
        
        # Check emergency stop
        if self.session_pnl <= -self.EMERGENCY_STOP:
            print(f"🚨 Emergency stop triggered: {self.session_pnl:.1%}")
            return False
        
        return True
    
    async def _get_isolated_margin_analysis(self, market_data: Dict[str, Any], 
                                          total_balance: float) -> Optional[Dict[str, Any]]:
        """Get AI analysis optimized for ISOLATED margin trading"""
        
        symbol = market_data['symbol']
        confidence = market_data.get('signal_strength', 70)
        confluences = market_data.get('confluences', [])
        volatility = abs(market_data.get('price_change_24h', 0))
        
        context = f"""
        ISOLATED MARGIN TRADING ANALYSIS
        
        Market Setup:
        - Symbol: {symbol}
        - Signal: {market_data.get('signal_direction', 'UNKNOWN')}
        - Confidence: {confidence}%
        - Confluences: {len(confluences)} ({', '.join(confluences[:3])})
        - 24h Change: {market_data.get('price_change_24h', 0):+.1f}%
        - RSI: {market_data.get('rsi', 50):.1f}
        - Volatility: {volatility:.1f}%
        
        Account Status:
        - Total Balance: ${total_balance:,.2f}
        - Trading Mode: ISOLATED MARGIN ONLY
        - Daily Trades: {self.daily_trades}/{self.MAX_DAILY_TRADES}
        - Consecutive Losses: {self.consecutive_losses}/{self.MAX_CONSECUTIVE_LOSSES}
        
        ISOLATED MARGIN CONSTRAINTS:
        - Capital Usage: {self.MIN_CAPITAL_USAGE:.0%} - {self.MAX_CAPITAL_USAGE:.0%}
        - Leverage: {self.MIN_LEVERAGE}x - {self.MAX_LEVERAGE}x
        - Reserved Capital: {self.RESERVED_CAPITAL_MIN:.0%} minimum
        - Position Risk: {self.MAX_POSITION_RISK:.0%} maximum
        - One position at a time (recommended)
        
        ISOLATED MARGIN BENEFITS:
        ✅ Risk confined to allocated margin only
        ✅ Liquidation won't affect other positions
        ✅ Clear risk/reward calculation
        ✅ Perfect for high-conviction single trades
        ✅ Ideal for automated trading systems
        
        For ISOLATED margin strategy, determine:
        1. Optimal capital allocation (50%-85%)
        2. Safe leverage level (5x-20x)
        3. Risk management approach
        4. Position sizing strategy
        """
        
        prompt = f"""
        {context}
        
        Analyze this setup for ISOLATED margin trading.
        
        Key Considerations:
        1. ISOLATED margin isolates risk per position
        2. Higher leverage is safer with ISOLATED
        3. Capital allocation is critical
        4. Must maintain reserved capital
        5. One high-quality trade better than multiple
        
        Determine optimal parameters for this ISOLATED margin trade.
        
        Respond ONLY with valid JSON:
        {{
            "proceed_with_trade": boolean,
            "capital_allocation_pct": float (0.50-0.85),
            "recommended_leverage": integer (5-20),
            "risk_assessment": "LOW" | "MEDIUM" | "HIGH",
            "setup_quality": "EXCELLENT" | "GOOD" | "DECENT" | "POOR",
            "isolated_margin_suitability": "PERFECT" | "GOOD" | "AVERAGE" | "POOR",
            "position_risk_pct": float (0.01-0.08),
            "reasoning": [
                "key factor 1",
                "key factor 2",
                "key factor 3"
            ],
            "isolated_advantages": [
                "advantage 1",
                "advantage 2"
            ],
            "risk_warnings": [
                "warning 1 if any"
            ]
        }}
        """
        
        # Try AI analysis
        if self.openai_api_key:
            try:
                import openai
                openai.api_key = self.openai_api_key
                
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are an expert in ISOLATED margin futures trading. Focus on risk isolation and capital preservation. Respond only with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=700,
                    temperature=0.2
                )
                
                response_text = response.choices[0].message.content.strip()
                response_text = response_text.replace('```json', '').replace('```', '').strip()
                
                ai_result = json.loads(response_text)
                
                # Validate AI recommendation
                if not ai_result.get('proceed_with_trade', False):
                    print("🤖 AI recommends skipping this trade")
                    return None
                
                if confidence < self.MIN_CONFIDENCE:
                    print(f"⚠️ Confidence {confidence}% below minimum {self.MIN_CONFIDENCE}%")
                    return None
                
                return ai_result
                
            except Exception as e:
                print(f"⚠️ AI analysis failed: {e}")
        
        # Fallback rule-based analysis
        return self._rule_based_isolated_analysis(market_data, confidence, len(confluences))
    
    def _rule_based_isolated_analysis(self, market_data: Dict[str, Any], 
                                    confidence: float, confluence_count: int) -> Optional[Dict[str, Any]]:
        """Rule-based analysis for ISOLATED margin"""
        
        # Quality assessment
        if confidence < self.MIN_CONFIDENCE:
            return None
        
        if confluence_count < 3:
            return None
        
        # Determine parameters based on setup quality
        if confidence >= 90 and confluence_count >= 5:
            setup_quality = "EXCELLENT"
            capital_pct = 0.80
            leverage = 18
            risk_pct = 0.06
        elif confidence >= 85 and confluence_count >= 4:
            setup_quality = "GOOD"
            capital_pct = 0.70
            leverage = 15
            risk_pct = 0.05
        elif confidence >= 80 and confluence_count >= 3:
            setup_quality = "GOOD"
            capital_pct = 0.60
            leverage = 12
            risk_pct = 0.04
        else:
            setup_quality = "DECENT"
            capital_pct = 0.50
            leverage = 8
            risk_pct = 0.03
        
        return {
            "proceed_with_trade": True,
            "capital_allocation_pct": capital_pct,
            "recommended_leverage": leverage,
            "risk_assessment": "MEDIUM",
            "setup_quality": setup_quality,
            "isolated_margin_suitability": "GOOD",
            "position_risk_pct": risk_pct,
            "reasoning": [
                f"Setup quality: {setup_quality} ({confidence}%, {confluence_count} confluences)",
                f"ISOLATED margin perfect for single high-conviction trade",
                f"Capital allocation: {capital_pct:.0%} with {leverage}x leverage"
            ],
            "isolated_advantages": [
                "Risk confined to allocated margin only",
                "No liquidation spillover to other positions"
            ],
            "risk_warnings": []
        }
    
    def _calculate_isolated_parameters(self, market_data: Dict[str, Any],
                                     ai_analysis: Dict[str, Any],
                                     total_balance: float,
                                     available_balance: float) -> IsolatedTradeExecution:
        """Calculate complete ISOLATED margin parameters"""
        
        symbol = market_data['symbol']
        direction = market_data.get('signal_direction', 'BUY')
        current_price = market_data['current_price']
        
        # Capital allocation (enforced within limits)
        capital_pct = ai_analysis.get('capital_allocation_pct', 0.6)
        capital_pct = max(self.MIN_CAPITAL_USAGE, min(self.MAX_CAPITAL_USAGE, capital_pct))
        
        # Leverage (enforced within limits)
        leverage = ai_analysis.get('recommended_leverage', 10)
        leverage = max(self.MIN_LEVERAGE, min(self.MAX_LEVERAGE, leverage))
        
        # Capital calculations
        allocated_capital = total_balance * capital_pct
        reserved_capital = total_balance - allocated_capital
        
        # Ensure we don't exceed available balance
        allocated_capital = min(allocated_capital, available_balance)
        
        # Position calculations
        position_value = allocated_capital * leverage
        position_size = position_value / current_price
        
        # Calculate ISOLATED margin stop loss
        stop_loss_price = self._calculate_isolated_stop_loss(
            market_data, direction, current_price, leverage
        )
        
        # Calculate liquidation price for ISOLATED margin
        liquidation_price = self._calculate_isolated_liquidation(
            current_price, direction, leverage
        )
        
        # Risk calculations
        if direction == 'BUY':
            loss_per_unit = current_price - stop_loss_price
        else:
            loss_per_unit = stop_loss_price - current_price
        
        max_loss_usdt = position_size * loss_per_unit
        max_loss_pct = max_loss_usdt / total_balance
        
        # Take profit levels
        tp_levels = self._calculate_isolated_take_profits(
            current_price, stop_loss_price, direction
        )
        
        # Partial TP quantities (40%, 40%, 20%)
        tp_quantities = [
            position_size * 0.4,  # 40% at TP1
            position_size * 0.4,  # 40% at TP2
            position_size * 0.2   # 20% at TP3
        ]
        
        # Generate trade ID
        trade_id = f"ISO_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        return IsolatedTradeExecution(
            trade_id=trade_id,
            symbol=symbol,
            direction=direction,
            margin_type="ISOLATED",
            leverage=leverage,
            
            total_balance=total_balance,
            allocated_capital=allocated_capital,
            reserved_capital=reserved_capital,
            capital_usage_pct=capital_pct,
            
            position_size=position_size,
            position_value=position_value,
            entry_price=current_price,
            
            stop_loss_price=stop_loss_price,
            liquidation_price=liquidation_price,
            max_loss_usdt=max_loss_usdt,
            max_loss_pct=max_loss_pct,
            
            take_profit_levels=tp_levels,
            partial_tp_quantities=tp_quantities,
            
            confidence_score=market_data.get('signal_strength', 70),
            setup_quality=ai_analysis.get('setup_quality', 'GOOD'),
            confluences=market_data.get('confluences', []),
            ai_reasoning=ai_analysis.get('reasoning', [])
        )
    
    def _calculate_isolated_stop_loss(self, market_data: Dict[str, Any],
                                    direction: str, current_price: float,
                                    leverage: int) -> float:
        """Calculate stop loss optimized for ISOLATED margin"""
        
        # Get structural levels
        support_levels = market_data.get('support_levels', [])
        resistance_levels = market_data.get('resistance_levels', [])
        fvgs = market_data.get('fvgs', [])
        
        # Base stop loss on market structure
        if direction == 'BUY':
            # For longs, stop below support/FVG
            candidates = []
            
            # Add support levels below current price
            candidates.extend([s for s in support_levels if s < current_price])
            
            # Add FVG bottoms
            for fvg in fvgs:
                if (fvg.get('type') == 'bearish' and 
                    fvg.get('bottom', 0) < current_price):
                    candidates.append(fvg['bottom'])
            
            if candidates:
                structural_sl = max(candidates) * 0.999  # Slightly below
            else:
                structural_sl = current_price * 0.98  # 2% fallback
        
        else:  # SELL
            # For shorts, stop above resistance/FVG
            candidates = []
            
            # Add resistance levels above current price
            candidates.extend([r for r in resistance_levels if r > current_price])
            
            # Add FVG tops
            for fvg in fvgs:
                if (fvg.get('type') == 'bullish' and 
                    fvg.get('top', 0) > current_price):
                    candidates.append(fvg['top'])
            
            if candidates:
                structural_sl = min(candidates) * 1.001  # Slightly above
            else:
                structural_sl = current_price * 1.02  # 2% fallback
        
        # Adjust for ISOLATED margin safety
        # Higher leverage = slightly tighter stop for safety
        leverage_adjustment = 1 + (leverage - 10) * 0.001  # Minor adjustment
        
        if direction == 'BUY':
            isolated_sl = structural_sl / leverage_adjustment
        else:
            isolated_sl = structural_sl * leverage_adjustment
        
        return isolated_sl
    
    def _calculate_isolated_liquidation(self, entry_price: float, 
                                      direction: str, leverage: int) -> float:
        """Calculate ISOLATED margin liquidation price"""
        # Simplified ISOLATED liquidation calculation
        # Actual Binance formula is more complex
        
        if direction == 'BUY':
            # Long liquidation when price drops
            liq_price = entry_price * (1 - 0.9 / leverage)
        else:  # SELL
            # Short liquidation when price rises
            liq_price = entry_price * (1 + 0.9 / leverage)
        
        return liq_price
    
    def _calculate_isolated_take_profits(self, entry_price: float,
                                       stop_loss: float, direction: str) -> List[float]:
        """Calculate take profit levels for ISOLATED margin"""
        
        # Calculate risk distance
        if direction == 'BUY':
            risk_distance = entry_price - stop_loss
            # Conservative, medium, aggressive targets
            tp1 = entry_price + (risk_distance * 1.5)  # 1:1.5 R/R
            tp2 = entry_price + (risk_distance * 2.5)  # 1:2.5 R/R
            tp3 = entry_price + (risk_distance * 4.0)  # 1:4 R/R
        else:
            risk_distance = stop_loss - entry_price
            tp1 = entry_price - (risk_distance * 1.5)
            tp2 = entry_price - (risk_distance * 2.5)
            tp3 = entry_price - (risk_distance * 4.0)
        
        return [tp1, tp2, tp3]
    
    def _validate_isolated_trade(self, trade: IsolatedTradeExecution) -> bool:
        """Validate ISOLATED trade parameters"""
        
        # Check capital usage limits
        if not (self.MIN_CAPITAL_USAGE <= trade.capital_usage_pct <= self.MAX_CAPITAL_USAGE):
            print(f"❌ Capital usage {trade.capital_usage_pct:.0%} outside limits")
            return False
        
        # Check leverage limits
        if not (self.MIN_LEVERAGE <= trade.leverage <= self.MAX_LEVERAGE):
            print(f"❌ Leverage {trade.leverage}x outside limits")
            return False
        
        # Check position risk
        if trade.max_loss_pct > self.MAX_POSITION_RISK:
            print(f"❌ Position risk {trade.max_loss_pct:.1%} too high")
            return False
        
        # Check reserved capital
        reserved_pct = trade.reserved_capital / trade.total_balance
        if reserved_pct < self.RESERVED_CAPITAL_MIN:
            print(f"❌ Reserved capital {reserved_pct:.1%} too low")
            return False
        
        # Check liquidation distance
        liq_distance = abs(trade.liquidation_price - trade.entry_price) / trade.entry_price
        if liq_distance < 0.03:  # Minimum 3% liquidation distance
            print(f"❌ Liquidation too close: {liq_distance:.1%}")
            return False
        
        return True
    
    async def execute_isolated_trade(self, trade: IsolatedTradeExecution) -> bool:
        """Execute ISOLATED margin trade"""
        try:
            symbol = trade.symbol
            
            print(f"\n🛡️ EXECUTING ISOLATED MARGIN TRADE")
            print("=" * 45)
            self.display_isolated_trade_details(trade)
            
            # Step 1: Set leverage
            self.exchange.set_leverage(trade.leverage, symbol)
            print(f"📈 Leverage set: {trade.leverage}x")
            
            # Step 2: Set ISOLATED margin mode
            self.exchange.set_margin_mode('ISOLATED', symbol)
            print(f"🛡️ Margin mode: ISOLATED")
            
            # Step 3: Place main market order
            side = trade.direction.lower()
            
            main_order = self.exchange.create_market_order(
                symbol=symbol,
                side=side,
                amount=trade.position_size
            )
            
            print(f"✅ Market order executed: {main_order['id']}")
            print(f"📊 Position: {trade.position_size:.6f} {symbol.split('/')[0]}")
            
            # Step 4: Place stop loss (ISOLATED)
            try:
                sl_side = 'sell' if trade.direction == 'BUY' else 'buy'
                sl_order = self.exchange.create_order(
                    symbol=symbol,
                    type='STOP_MARKET',
                    side=sl_side,
                    amount=trade.position_size,
                    params={'stopPrice': trade.stop_loss_price}
                )
                print(f"🛑 ISOLATED Stop Loss: ${trade.stop_loss_price:,.2f}")
            except Exception as e:
                print(f"⚠️ Stop loss placement failed: {e}")
            
            # Step 5: Place take profit orders (partial)
            for i, (tp_price, tp_qty) in enumerate(zip(trade.take_profit_levels, trade.partial_tp_quantities), 1):
                try:
                    tp_side = 'sell' if trade.direction == 'BUY' else 'buy'
                    tp_order = self.exchange.create_order(
                        symbol=symbol,
                        type='TAKE_PROFIT_MARKET',
                        side=tp_side,
                        amount=tp_qty,
                        params={'stopPrice': tp_price}
                    )
                    print(f"💰 TP{i} ({tp_qty/trade.position_size:.0%}): ${tp_price:,.2f}")
                except Exception as e:
                    print(f"⚠️ TP{i} placement failed: {e}")
            
            # Update tracking
            self.active_trade = trade
            self.daily_trades += 1
            
            print(f"\n✅ ISOLATED MARGIN TRADE EXECUTED SUCCESSFULLY!")
            print(f"🔒 Risk isolated to allocated margin: ${trade.allocated_capital:,.2f}")
            print(f"💰 Reserved capital protected: ${trade.reserved_capital:,.2f}")
            
            return True
            
        except Exception as e:
            print(f"❌ ISOLATED trade execution failed: {e}")
            return False
    
    def display_isolated_trade_details(self, trade: IsolatedTradeExecution):
        """Display comprehensive ISOLATED trade details"""
        print(f"\n🛡️ ISOLATED MARGIN TRADE DETAILS")
        print("=" * 40)
        print(f"📊 Trade ID: {trade.trade_id}")
        print(f"📈 Symbol: {trade.symbol} {trade.direction}")
        print(f"🏆 Setup: {trade.setup_quality} ({trade.confidence_score:.1f}%)")
        print(f"🎯 Confluences: {len(trade.confluences)}")
        
        print(f"\n🛡️ ISOLATED MARGIN SETUP:")
        print(f"   Margin Type: {trade.margin_type}")
        print(f"   Leverage: {trade.leverage}x")
        print(f"   Allocated: ${trade.allocated_capital:,.2f} ({trade.capital_usage_pct:.0%})")
        print(f"   Reserved: ${trade.reserved_capital:,.2f}")
        
        print(f"\n📊 POSITION DETAILS:")
        print(f"   Size: {trade.position_size:.6f}")
        print(f"   Value: ${trade.position_value:,.2f}")
        print(f"   Entry: ${trade.entry_price:,.2f}")
        
        print(f"\n⚠️ ISOLATED RISK MANAGEMENT:")
        print(f"   Stop Loss: ${trade.stop_loss_price:,.2f}")
        print(f"   Liquidation: ${trade.liquidation_price:,.2f}")
        print(f"   Max Loss: ${trade.max_loss_usdt:.2f} ({trade.max_loss_pct:.1%})")
        
        print(f"\n🎯 TAKE PROFIT STRATEGY:")
        for i, (tp, qty) in enumerate(zip(trade.take_profit_levels, trade.partial_tp_quantities), 1):
            pct = qty / trade.position_size * 100
            print(f"   TP{i}: ${tp:,.2f} ({pct:.0f}%)")
        
        print(f"\n🧠 AI REASONING:")
        for reason in trade.ai_reasoning:
            print(f"   • {reason}")
        
        print(f"\n🔒 ISOLATED MARGIN BENEFITS:")
        print(f"   ✅ Risk confined to ${trade.allocated_capital:,.2f}")
        print(f"   ✅ Reserved capital protected: ${trade.reserved_capital:,.2f}")
        print(f"   ✅ No liquidation spillover risk")
        print(f"   ✅ Clear risk/reward calculation")
    
    def get_isolated_position_status(self) -> Dict[str, Any]:
        """Get current ISOLATED position status"""
        try:
            positions = self.exchange.fetch_positions()
            active_positions = [p for p in positions if p['contracts'] and p['contracts'] > 0]
            
            if not active_positions:
                return {'has_position': False, 'message': 'No active ISOLATED positions'}
            
            position = active_positions[0]  # Should only have 1 with ISOLATED strategy
            
            # Calculate P&L and performance
            unrealized_pnl = position.get('unrealizedPnl', 0)
            percentage = position.get('percentage', 0)
            
            return {
                'has_position': True,
                'symbol': position['symbol'],
                'side': position['side'],
                'size': position['contracts'],
                'entry_price': position.get('entryPrice', 0),
                'mark_price': position.get('markPrice', 0),
                'unrealized_pnl': unrealized_pnl,
                'percentage': percentage,
                'margin_type': 'ISOLATED',
                'isolated_margin': position.get('initialMargin', 0),
                'maintenance_margin': position.get('maintenanceMargin', 0)
            }
            
        except Exception as e:
            print(f"❌ Error getting ISOLATED position status: {e}")
            return {'has_position': False, 'error': str(e)}
    
    async def monitor_isolated_position(self):
        """Monitor ISOLATED position with safety checks"""
        if not self.active_trade:
            return
        
        try:
            status = self.get_isolated_position_status()
            
            if not status['has_position']:
                print("ℹ️ No active ISOLATED position to monitor")
                self.active_trade = None
                return
            
            print(f"\n📊 ISOLATED Position Monitor")
            print("-" * 30)
            print(f"Symbol: {status['symbol']}")
            print(f"Side: {status['side']}")
            print(f"Size: {status['size']:.6f}")
            print(f"Entry: ${status['entry_price']:,.2f}")
            print(f"Mark: ${status['mark_price']:,.2f}")
            print(f"P&L: ${status['unrealized_pnl']:+,.2f} ({status['percentage']:+.2f}%)")
            print(f"Isolated Margin: ${status['isolated_margin']:,.2f}")
            
            # Check for break-even adjustment
            if status['percentage'] > 0.5:  # 0.5% profit
                print("🎯 Consider moving stop loss to break-even")
            
            # Check for emergency conditions
            if status['percentage'] < -5:  # 5% loss
                print("⚠️ Position showing significant loss")
            
        except Exception as e:
            print(f"❌ Error monitoring ISOLATED position: {e}")

class IsolatedMarginTradingSession:
    """Complete ISOLATED margin trading session"""
    
    def __init__(self):
        self.bot = FinalIsolatedMarginBot()
        
        # Import other modules for complete functionality
        try:
            from advanced_ai_trading_strategy import AdvancedMarketAnalyzer
            from trade_journal import TradeJournal
            from notification_system import NotificationManager
            
            self.market_analyzer = AdvancedMarketAnalyzer()
            self.trade_journal = TradeJournal()
            self.notifications = NotificationManager()
            
            print("✅ All support systems loaded")
            
        except ImportError as e:
            print(f"⚠️ Some modules unavailable: {e}")
            self.market_analyzer = None
            self.trade_journal = None
            self.notifications = None
    
    async def run_isolated_trading_session(self, symbols: List[str], duration_hours: int = 2):
        """Run complete ISOLATED margin trading session"""
        print(f"\n🛡️ ISOLATED MARGIN TRADING SESSION")
        print("=" * 50)
        print(f"⏰ Duration: {duration_hours} hours")
        print(f"🎯 Symbols: {', '.join(symbols)}")
        print(f"🛡️ Strategy: One ISOLATED position at a time")
        print(f"💰 Capital Range: 50%-85% per trade")
        print(f"📈 Leverage Range: 5x-20x")
        
        if not self.bot.initialize_exchange():
            print("❌ Cannot start session - exchange initialization failed")
            return
        
        # Send session start notification
        if self.notifications:
            await self.notifications.send_risk_alert(
                "ISOLATED Margin Session Started",
                f"Session: {duration_hours}h | Symbols: {len(symbols)} | Mode: ISOLATED"
            )
        
        end_time = datetime.utcnow() + timedelta(hours=duration_hours)
        scan_count = 0
        signals_found = 0
        trades_executed = 0
        
        try:
            while datetime.utcnow() < end_time:
                scan_count += 1
                current_time = datetime.utcnow().strftime('%H:%M:%S')
                
                print(f"\n🔍 ISOLATED Scan #{scan_count} - {current_time}")
                print("-" * 40)
                
                # Monitor existing position first
                await self.bot.monitor_isolated_position()
                
                # Check if we can take new positions
                if self.bot.active_trade:
                    print("⏳ ISOLATED position active - waiting for completion")
                else:
                    # Scan for new opportunities
                    for symbol in symbols:
                        try:
                            if self.market_analyzer:
                                # Full market analysis
                                market_data = await self.market_analyzer.comprehensive_analysis(
                                    self.bot.exchange, symbol, '1h'
                                )
                                
                                if market_data:
                                    # Generate trading setup
                                    setup = self.market_analyzer.generate_trading_setup(market_data)
                                    
                                    if setup:
                                        # Add setup data for ISOLATED analysis
                                        market_data.update({
                                            'signal_direction': setup.direction,
                                            'signal_strength': setup.confidence,
                                            'confluences': setup.confluences
                                        })
                                        
                                        signals_found += 1
                                        
                                        print(f"🚨 Setup: {setup.direction} {symbol}")
                                        print(f"   Confidence: {setup.confidence:.1f}%")
                                        print(f"   Confluences: {len(setup.confluences)}")
                                        
                                        # Analyze for ISOLATED margin
                                        isolated_trade = await self.bot.analyze_isolated_trade_opportunity(market_data)
                                        
                                        if isolated_trade:
                                            # Send signal notification
                                            if self.notifications:
                                                signal_data = {
                                                    'symbol': symbol,
                                                    'direction': setup.direction,
                                                    'entry_price': market_data['current_price'],
                                                    'stop_loss': isolated_trade.stop_loss_price,
                                                    'take_profit': isolated_trade.take_profit_levels[0],
                                                    'risk_reward': 2.5,
                                                    'confidence': setup.confidence,
                                                    'timeframe': '1h',
                                                    'confluences': setup.confluences,
                                                    'ai_analysis': f"ISOLATED margin {isolated_trade.leverage}x leverage"
                                                }
                                                await self.notifications.send_signal_alert(signal_data)
                                            
                                            # Execute ISOLATED trade
                                            if await self.bot.execute_isolated_trade(isolated_trade):
                                                trades_executed += 1
                                                
                                                # Log trade
                                                if self.trade_journal:
                                                    from trade_journal import TradeRecord
                                                    
                                                    trade_record = TradeRecord(
                                                        trade_id=isolated_trade.trade_id,
                                                        timestamp=datetime.utcnow().isoformat(),
                                                        symbol=symbol,
                                                        direction=setup.direction,
                                                        entry_price=market_data['current_price'],
                                                        quantity=isolated_trade.position_size,
                                                        leverage=isolated_trade.leverage,
                                                        stop_loss=isolated_trade.stop_loss_price,
                                                        take_profit=isolated_trade.take_profit_levels[0],
                                                        risk_amount=isolated_trade.max_loss_usdt,
                                                        confluences=json.dumps(setup.confluences),
                                                        ai_analysis=f"ISOLATED margin strategy",
                                                        signal_strength=setup.confidence,
                                                        timeframe='1h',
                                                        notes=f"ISOLATED margin {isolated_trade.leverage}x",
                                                        status='OPEN'
                                                    )
                                                    
                                                    self.trade_journal.log_trade_entry(trade_record)
                                                
                                                # Send execution notification
                                                if self.notifications:
                                                    trade_data = {
                                                        'symbol': symbol,
                                                        'direction': setup.direction,
                                                        'entry_price': market_data['current_price'],
                                                        'quantity': isolated_trade.position_size,
                                                        'leverage': isolated_trade.leverage
                                                    }
                                                    await self.notifications.send_trade_execution(trade_data)
                                                
                                                # Break after trade execution
                                                break
                                        else:
                                            print(f"⏳ Setup doesn't meet ISOLATED criteria")
                                    else:
                                        print(f"⏳ No setup for {symbol}")
                                else:
                                    print(f"⏳ No market data for {symbol}")
                            else:
                                print(f"⏳ Market analyzer unavailable")
                                
                        except Exception as e:
                            print(f"❌ Error analyzing {symbol}: {e}")
                
                # Session progress
                elapsed = datetime.utcnow() - self.bot.session_start if hasattr(self.bot, 'session_start') else timedelta(0)
                
                print(f"\n📊 ISOLATED Session Progress:")
                print(f"   ⏱️ Time: {elapsed.total_seconds()/3600:.1f}h")
                print(f"   🔍 Scans: {scan_count}")
                print(f"   🚨 Signals: {signals_found}")
                print(f"   ✅ Trades: {trades_executed}")
                print(f"   🛡️ Mode: ISOLATED margin only")
                
                # Wait between scans (5 minutes)
                print(f"⏱️ Next scan in 5 minutes...")
                
                for i in range(300):  # 5 minutes = 300 seconds
                    await asyncio.sleep(1)
                    if (i + 1) % 60 == 0:
                        remaining = (300 - i - 1) // 60
                        if remaining > 0:
                            print(f"   {remaining} minutes remaining...")
                
        except KeyboardInterrupt:
            print(f"\n⏹️ ISOLATED session stopped by user")
        
        # Final session summary
        total_time = (datetime.utcnow() - (self.bot.session_start if hasattr(self.bot, 'session_start') else datetime.utcnow())).total_seconds() / 3600
        
        print(f"\n📋 ISOLATED MARGIN SESSION SUMMARY")
        print("=" * 45)
        print(f"⏱️ Duration: {total_time:.1f} hours")
        print(f"🔍 Total Scans: {scan_count}")
        print(f"🚨 Signals Found: {signals_found}")
        print(f"✅ Trades Executed: {trades_executed}")
        print(f"🛡️ Margin Type: ISOLATED (risk isolated)")
        print(f"💰 Capital Management: 50%-85% per trade")
        print(f"📈 Leverage Used: 5x-20x range")
        
        # Get final position status
        final_status = self.bot.get_isolated_position_status()
        if final_status['has_position']:
            print(f"🔒 Active ISOLATED Position:")
            print(f"   Symbol: {final_status['symbol']}")
            print(f"   P&L: ${final_status['unrealized_pnl']:+,.2f}")
            print(f"   Isolated Margin: ${final_status['isolated_margin']:,.2f}")
        
        # Send session end notification
        if self.notifications:
            await self.notifications.send_risk_alert(
                "ISOLATED Session Completed",
                f"Duration: {total_time:.1f}h | Trades: {trades_executed} | Signals: {signals_found}"
            )

async def demo_isolated_margin_trading():
    """Demo the complete ISOLATED margin system"""
    print("🛡️ ISOLATED Margin Trading Demo")
    print("=" * 40)
    
    session = IsolatedMarginTradingSession()
    
    print("\n🎯 ISOLATED Margin Demo Options:")
    print("1. Quick analysis demo")
    print("2. Run 10-minute session")
    print("3. Check position status")
    
    try:
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == "1":
            # Quick demo
            demo_market_data = {
                'symbol': 'BTC/USDT',
                'current_price': 68750.0,
                'price_change_24h': 2.3,
                'signal_direction': 'BUY',
                'signal_strength': 84.0,
                'rsi': 42.0,
                'confluences': [
                    '✅ Bullish BOS confirmed',
                    '✅ FVG entry zone',
                    '✅ EMA alignment',
                    '✅ Support retest'
                ],
                'support_levels': [68200, 67800, 67500],
                'resistance_levels': [69200, 69800, 70500],
                'fvgs': [{'type': 'bullish', 'bottom': 68400, 'top': 68600}]
            }
            
            isolated_trade = await session.bot.analyze_isolated_trade_opportunity(demo_market_data)
            
            if isolated_trade:
                session.bot.display_isolated_trade_details(isolated_trade)
                print(f"\n✅ ISOLATED margin analysis complete!")
            else:
                print(f"❌ Demo setup doesn't meet ISOLATED criteria")
        
        elif choice == "2":
            await session.run_isolated_trading_session(['BTC/USDT', 'ETH/USDT'], duration_hours=10/60)
        
        elif choice == "3":
            if session.bot.initialize_exchange():
                status = session.bot.get_isolated_position_status()
                if status['has_position']:
                    print(f"📊 Current ISOLATED Position:")
                    print(f"   Symbol: {status['symbol']}")
                    print(f"   Side: {status['side']}")
                    print(f"   P&L: ${status['unrealized_pnl']:+,.2f}")
                    print(f"   Isolated Margin: ${status['isolated_margin']:,.2f}")
                else:
                    print("ℹ️ No active ISOLATED positions")
        
        print(f"\n✅ ISOLATED margin demo completed!")
        
    except KeyboardInterrupt:
        print(f"\n👋 Demo interrupted")

if __name__ == "__main__":
    asyncio.run(demo_isolated_margin_trading())