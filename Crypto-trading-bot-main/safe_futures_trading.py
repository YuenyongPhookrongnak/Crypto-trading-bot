#!/usr/bin/env python3
"""
Safe Futures Trading

This version has built-in safety limits to prevent dangerous configurations.
"""

import asyncio
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import os
import json

class SafeFuturesTrader:
    """Safe Futures Trading Bot with Built-in Limits"""
    
    # SAFETY LIMITS - CANNOT BE OVERRIDDEN
    MAX_LEVERAGE = 3  # Maximum 3x leverage
    MAX_RISK_PER_TRADE = 0.02  # Maximum 2% risk per trade
    MIN_CAPITAL = 100  # Minimum $100 capital
    MAX_DAILY_LOSS = 0.05  # Maximum 5% daily loss
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # API Configuration
        self.api_key = os.getenv('BINANCE_API_KEY', '')
        self.secret = os.getenv('BINANCE_SECRET', '')
        self.testnet = os.getenv('BINANCE_TESTNET', 'true').lower() == 'true'
        
        # Apply SAFETY LIMITS
        requested_capital = self.config.get('initial_capital', 100.0)
        requested_leverage = self.config.get('leverage', 1)
        requested_risk = self.config.get('risk_per_trade', 0.01)
        
        # ENFORCE SAFETY LIMITS
        self.initial_capital = max(requested_capital, self.MIN_CAPITAL)
        self.leverage = min(max(requested_leverage, 1), self.MAX_LEVERAGE)
        self.risk_per_trade = min(max(requested_risk, 0.005), self.MAX_RISK_PER_TRADE)
        
        # Show warnings if limits were applied
        if requested_leverage > self.MAX_LEVERAGE:
            print(f"🚨 SAFETY LIMIT: Leverage reduced from {requested_leverage}x to {self.leverage}x")
        if requested_risk > self.MAX_RISK_PER_TRADE:
            print(f"🚨 SAFETY LIMIT: Risk reduced from {requested_risk:.1%} to {self.risk_per_trade:.1%}")
        
        # Other settings
        self.max_positions = self.config.get('max_positions', 2)
        self.symbols = self.config.get('symbols', ['BTC/USDT'])  # Start with just BTC
        
        # Risk Management
        self.stop_loss_pct = 0.02  # 2% stop loss
        self.take_profit_pct = 0.04  # 4% take profit
        
        # Strategy Settings
        self.sma_short = 10
        self.sma_long = 21
        self.rsi_period = 14
        
        # Initialize
        self.exchange = None
        self.positions = {}
        self.daily_pnl = 0.0
        self.session_start = datetime.utcnow()
        self.trades_today = 0
        
        self.print_safe_config()
    
    def print_safe_config(self):
        """Print safe trading configuration"""
        print("🛡️ Safe Futures Trading Bot")
        print("=" * 40)
        print(f"💰 Capital: ${self.initial_capital:,.2f}")
        print(f"📊 Max Positions: {self.max_positions}")
        print(f"⚠️ Risk per Trade: {self.risk_per_trade:.1%} (MAX: {self.MAX_RISK_PER_TRADE:.1%})")
        print(f"📈 Leverage: {self.leverage}x (MAX: {self.MAX_LEVERAGE}x)")
        print(f"🎯 Symbols: {', '.join(self.symbols)}")
        print(f"🛡️ Stop Loss: {self.stop_loss_pct:.1%}")
        print(f"💰 Take Profit: {self.take_profit_pct:.1%}")
        print(f"🚨 Max Daily Loss: {self.MAX_DAILY_LOSS:.1%}")
        print(f"🧪 Mode: {'TESTNET' if self.testnet else '🔴 LIVE TRADING'}")
        
        if not self.testnet:
            print("\n🚨 LIVE TRADING ACTIVE - REAL MONEY AT RISK!")
        print("")
    
    def initialize(self):
        """Initialize exchange and setup (synchronous)"""
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
            
            # Test connection
            self.exchange.load_markets()
            balance = self.exchange.fetch_balance()
            
            print("✅ Exchange connected successfully")
            
            # Setup symbols
            for symbol in self.symbols:
                self.setup_symbol(symbol)
            
            # Show initial balance
            usdt_balance = balance['USDT']['total']
            print(f"💰 USDT Balance: {usdt_balance:,.2f}")
            
            # Safety check: Ensure sufficient balance
            if usdt_balance < self.initial_capital * 0.1:
                print(f"⚠️ WARNING: Low balance ({usdt_balance:,.2f}) for capital ({self.initial_capital:,.2f})")
            
            return True
            
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            return False
    
    def setup_symbol(self, symbol: str):
        """Setup symbol for trading (synchronous)"""
        try:
            # Set leverage
            self.exchange.set_leverage(self.leverage, symbol)
            print(f"📈 {symbol}: Leverage set to {self.leverage}x")
            
            # Set margin type to ISOLATED
            try:
                self.exchange.set_margin_mode('ISOLATED', symbol)
                print(f"🛡️ {symbol}: Margin type set to ISOLATED")
            except Exception as e:
                if "No need to change margin type" in str(e):
                    print(f"🛡️ {symbol}: Margin type already ISOLATED")
                else:
                    print(f"⚠️ {symbol}: Could not set margin type: {e}")
            
        except Exception as e:
            print(f"❌ Error setting up {symbol}: {e}")
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50)
        except:
            return pd.Series([50] * len(prices))
    
    def get_market_data(self, symbol: str, timeframe='1h', limit=50):
        """Get market data for analysis (synchronous)"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Calculate indicators
            df['sma_short'] = df['close'].rolling(self.sma_short).mean()
            df['sma_long'] = df['close'].rolling(self.sma_long).mean()
            df['rsi'] = self.calculate_rsi(df['close'], self.rsi_period)
            
            return df
            
        except Exception as e:
            print(f"❌ Error fetching data for {symbol}: {e}")
            return None
    
    def analyze_signal(self, df: pd.DataFrame, symbol: str) -> Optional[Dict[str, Any]]:
        """Analyze market data for trading signals"""
        try:
            if len(df) < self.sma_long + 2:
                return None
            
            current = df.iloc[-1]
            previous = df.iloc[-2]
            
            # Check for moving average crossover
            signal = None
            confidence = 60.0
            
            # Bullish signal
            if (current['sma_short'] > current['sma_long'] and 
                previous['sma_short'] <= previous['sma_long']):
                
                # Additional filters
                if current['rsi'] < 70:  # Not overbought
                    signal = 'BUY'
                    confidence = 70.0
                    
                    # Boost confidence if RSI is oversold
                    if current['rsi'] < 30:
                        confidence = 80.0
            
            # Bearish signal
            elif (current['sma_short'] < current['sma_long'] and 
                  previous['sma_short'] >= previous['sma_long']):
                
                # Additional filters  
                if current['rsi'] > 30:  # Not oversold
                    signal = 'SELL'
                    confidence = 65.0
                    
                    # Boost confidence if RSI is overbought
                    if current['rsi'] > 70:
                        confidence = 75.0
            
            if signal:
                return {
                    'symbol': symbol,
                    'signal': signal,
                    'confidence': confidence,
                    'price': current['close'],
                    'rsi': current['rsi'],
                    'sma_short': current['sma_short'],
                    'sma_long': current['sma_long'],
                    'timestamp': current['timestamp']
                }
            
            return None
            
        except Exception as e:
            print(f"❌ Error analyzing signal: {e}")
            return None
    
    def calculate_safe_position_size(self, symbol: str, price: float) -> float:
        """Calculate safe position size with multiple safety checks"""
        try:
            # Get available balance
            balance = self.exchange.fetch_balance()
            available_usdt = balance['USDT']['free']
            
            # Safety check 1: Minimum balance
            if available_usdt < 50:
                print(f"⚠️ Balance too low: ${available_usdt:.2f}")
                return 0
            
            # Safety check 2: Risk amount
            risk_amount = available_usdt * self.risk_per_trade
            max_risk = available_usdt * 0.02  # Never risk more than 2%
            risk_amount = min(risk_amount, max_risk)
            
            # Safety check 3: Position value with leverage
            position_value = risk_amount * self.leverage / self.stop_loss_pct
            
            # Safety check 4: Maximum position size (10% of balance)
            max_position_value = available_usdt * 0.1
            position_value = min(position_value, max_position_value)
            
            # Convert to contracts
            position_size = position_value / price
            
            # Safety check 5: Market minimum
            market = self.exchange.market(symbol)
            min_size = market.get('limits', {}).get('amount', {}).get('min', 0.001)
            position_size = max(position_size, min_size)
            
            # Safety check 6: Maximum position size
            max_size = min_size * 1000  # Max 1000x minimum
            position_size = min(position_size, max_size)
            
            print(f"💰 Position sizing: Risk=${risk_amount:.2f}, Size={position_size:.6f}")
            
            return position_size
            
        except Exception as e:
            print(f"❌ Error calculating position size: {e}")
            return 0
    
    def open_position(self, signal: Dict[str, Any]) -> bool:
        """Open a futures position with safety checks"""
        try:
            symbol = signal['symbol']
            side = signal['signal'].lower()
            price = signal['price']
            
            # Safety check: Daily trade limit
            if self.trades_today >= 10:
                print(f"⚠️ Daily trade limit reached (10)")
                return False
            
            # Calculate safe position size
            size = self.calculate_safe_position_size(symbol, price)
            
            if size <= 0:
                print(f"⚠️ Position size too small or unsafe")
                return False
            
            # Place market order
            order = self.exchange.create_market_order(symbol, side, size)
            
            print(f"🎯 SAFE POSITION OPENED:")
            print(f"   📊 Symbol: {symbol}")
            print(f"   📈 Side: {side.upper()}")
            print(f"   💰 Size: {size:.6f}")
            print(f"   💵 Price: ${price:,.2f}")
            print(f"   🎲 Confidence: {signal['confidence']:.1f}%")
            print(f"   📈 Leverage: {self.leverage}x")
            
            # Set stop orders
            self.set_stop_orders(symbol, side, price, size)
            
            # Track position
            self.positions[symbol] = {
                'side': side,
                'size': size,
                'entry_price': price,
                'timestamp': datetime.utcnow(),
                'order_id': order['id']
            }
            
            self.trades_today += 1
            
            return True
            
        except Exception as e:
            print(f"❌ Error opening position: {e}")
            return False
    
    def set_stop_orders(self, symbol: str, side: str, entry_price: float, size: float):
        """Set stop loss and take profit orders"""
        try:
            if side == 'buy':
                stop_price = entry_price * (1 - self.stop_loss_pct)
                profit_price = entry_price * (1 + self.take_profit_pct)
                close_side = 'sell'
            else:
                stop_price = entry_price * (1 + self.stop_loss_pct)
                profit_price = entry_price * (1 - self.take_profit_pct)
                close_side = 'buy'
            
            # Stop loss order
            try:
                self.exchange.create_order(
                    symbol=symbol,
                    type='STOP_MARKET',
                    side=close_side,
                    amount=size,
                    params={'stopPrice': stop_price}
                )
                print(f"   🛡️ Stop Loss: ${stop_price:.2f}")
            except Exception as e:
                print(f"   ⚠️ Stop loss failed: {e}")
            
            # Take profit order
            try:
                self.exchange.create_order(
                    symbol=symbol,
                    type='TAKE_PROFIT_MARKET',
                    side=close_side,
                    amount=size,
                    params={'stopPrice': profit_price}
                )
                print(f"   💰 Take Profit: ${profit_price:.2f}")
            except Exception as e:
                print(f"   ⚠️ Take profit failed: {e}")
                
        except Exception as e:
            print(f"❌ Error setting stop orders: {e}")
    
    def check_positions(self):
        """Check current positions"""
        try:
            positions = self.exchange.fetch_positions()
            active_positions = [p for p in positions if p['contracts'] and p['contracts'] > 0]
            
            total_pnl = 0
            for position in active_positions:
                symbol = position['symbol']
                pnl = position.get('unrealizedPnl', 0)
                total_pnl += pnl
                
                print(f"📊 {symbol}: {position['side']} {position['contracts']:.6f} "
                      f"(PnL: ${pnl:,.2f})")
            
            if active_positions:
                print(f"💰 Total Unrealized PnL: ${total_pnl:,.2f}")
            
            return active_positions
            
        except Exception as e:
            print(f"❌ Error checking positions: {e}")
            return []
    
    def check_safety_limits(self):
        """Check all safety limits"""
        try:
            positions = self.check_positions()
            total_pnl = sum(p.get('unrealizedPnl', 0) for p in positions)
            
            # Check daily loss limit (CRITICAL)
            loss_pct = abs(min(total_pnl, 0)) / self.initial_capital
            
            if loss_pct > self.MAX_DAILY_LOSS:
                print(f"🚨 CRITICAL: Daily loss limit exceeded: {loss_pct:.1%}")
                print("🛑 EMERGENCY CLOSE ALL POSITIONS")
                
                # Emergency close all positions
                for position in positions:
                    if position['contracts'] > 0:
                        self.emergency_close_position(position['symbol'])
                
                return False
            
            # Check trade count limit
            if self.trades_today >= 10:
                print(f"⚠️ Daily trade limit reached: {self.trades_today}")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Error checking safety limits: {e}")
            return True
    
    def emergency_close_position(self, symbol: str):
        """Emergency close position"""
        try:
            positions = self.exchange.fetch_positions([symbol])
            
            for position in positions:
                if position['contracts'] > 0:
                    side = 'sell' if position['side'] == 'long' else 'buy'
                    
                    order = self.exchange.create_market_order(
                        symbol, side, position['contracts']
                    )
                    
                    pnl = position.get('unrealizedPnl', 0)
                    print(f"🔴 EMERGENCY CLOSE: {symbol} (PnL: ${pnl:,.2f})")
                    
                    # Remove from tracking
                    if symbol in self.positions:
                        del self.positions[symbol]
                    
                    return True
            
            return False
            
        except Exception as e:
            print(f"❌ Error emergency closing position: {e}")
            return False
    
    def trading_loop(self, duration_minutes: int):
        """Safe trading loop with frequent safety checks"""
        print(f"\n🔄 Starting Safe Trading Loop ({duration_minutes} minutes)")
        print("=" * 50)
        
        end_time = datetime.utcnow() + timedelta(minutes=duration_minutes)
        loop_count = 0
        
        try:
            while datetime.utcnow() < end_time:
                loop_count += 1
                print(f"\n🔍 Loop #{loop_count} - {datetime.utcnow().strftime('%H:%M:%S')}")
                
                # CRITICAL: Check safety limits first
                if not self.check_safety_limits():
                    print("🛑 Safety limits exceeded - STOPPING TRADING")
                    break
                
                # Check current positions
                current_positions = self.check_positions()
                active_symbols = [p['symbol'] for p in current_positions]
                
                # Look for new signals (only if under position limit)
                if len(current_positions) < self.max_positions:
                    for symbol in self.symbols:
                        # Skip if already have position
                        if symbol in active_symbols:
                            continue
                        
                        # Get market data and analyze
                        df = self.get_market_data(symbol)
                        if df is None:
                            continue
                        
                        signal = self.analyze_signal(df, symbol)
                        if signal:
                            print(f"🚨 Signal: {signal['signal']} {symbol} @ ${signal['price']:,.2f}")
                            print(f"   RSI: {signal['rsi']:.1f}, Confidence: {signal['confidence']:.1f}%")
                            
                            # Open position (with safety checks)
                            if self.open_position(signal):
                                current_positions = self.check_positions()
                                break  # Only one position per loop
                
                # Show session summary
                session_duration = datetime.utcnow() - self.session_start
                total_pnl = sum(p.get('unrealizedPnl', 0) for p in current_positions)
                
                print(f"📈 Session: {session_duration.total_seconds()/60:.1f}m, "
                      f"PnL: ${total_pnl:,.2f}, Positions: {len(current_positions)}, "
                      f"Trades: {self.trades_today}")
                
                # Wait 3 minutes between loops for safety
                print("⏱️ Waiting 3 minutes...")
                for i in range(180):  # 3 minutes = 180 seconds
                    time.sleep(1)
                    if (i + 1) % 60 == 0:  # Every minute
                        # Quick safety check
                        positions = self.exchange.fetch_positions()
                        active_positions = [p for p in positions if p['contracts'] and p['contracts'] > 0]
                        total_pnl = sum(p.get('unrealizedPnl', 0) for p in active_positions)
                        loss_pct = abs(min(total_pnl, 0)) / self.initial_capital
                        
                        if loss_pct > self.MAX_DAILY_LOSS:
                            print(f"\n🚨 EMERGENCY: Loss limit exceeded during wait!")
                            return
                
        except KeyboardInterrupt:
            print(f"\n⏹️ Trading stopped by user")
        
        # Final summary
        final_positions = self.check_positions()
        final_pnl = sum(p.get('unrealizedPnl', 0) for p in final_positions)
        
        print(f"\n📋 Safe Trading Session Complete:")
        print(f"   Duration: {loop_count} loops")
        print(f"   Final PnL: ${final_pnl:,.2f}")
        print(f"   Open Positions: {len(final_positions)}")
        print(f"   Trades Executed: {self.trades_today}")

def main():
    """Main function with built-in safety"""
    print("🛡️ Safe Futures Trading Launcher")
    print("=" * 35)
    
    # Force testnet for safety
    testnet = os.getenv('BINANCE_TESTNET', 'true').lower() == 'true'
    
    if not testnet:
        print("\n🚨 CRITICAL WARNING: LIVE TRADING DETECTED!")
        print("This bot has safety limits but REAL MONEY is still at risk!")
        print("Recommended: Set BINANCE_TESTNET=true in .env file")
        print("\nSAFETY LIMITS ACTIVE:")
        print(f"- Maximum leverage: {SafeFuturesTrader.MAX_LEVERAGE}x")
        print(f"- Maximum risk per trade: {SafeFuturesTrader.MAX_RISK_PER_TRADE:.1%}")
        print(f"- Maximum daily loss: {SafeFuturesTrader.MAX_DAILY_LOSS:.1%}")
        print(f"- Maximum trades per day: 10")
        
        confirm = input(f"\nType 'I UNDERSTAND THE RISKS' to proceed: ")
        if confirm != 'I UNDERSTAND THE RISKS':
            print("👋 Cancelled for safety")
            return
    
    # Get configuration with safety limits
    try:
        print(f"\n🎯 Safe Trading Configuration:")
        capital = float(input(f"💰 Capital ($100-$5000): ") or "100")
        capital = min(max(capital, 100), 5000)  # Limit $100-$5000
        
        leverage = int(input(f"📈 Leverage (1-{SafeFuturesTrader.MAX_LEVERAGE}x): ") or "1")
        leverage = min(max(leverage, 1), SafeFuturesTrader.MAX_LEVERAGE)
        
        risk = float(input(f"⚠️ Risk per trade (0.005-0.02): ") or "0.01")
        risk = min(max(risk, 0.005), SafeFuturesTrader.MAX_RISK_PER_TRADE)
        
        duration = int(input(f"⏱️ Duration (10-120 minutes): ") or "30")
        duration = min(max(duration, 10), 120)  # Limit 10-120 minutes
        
        config = {
            'initial_capital': capital,
            'leverage': leverage,
            'risk_per_trade': risk,
            'max_positions': 1,  # Start with 1 position only
            'symbols': ['BTC/USDT']  # Start with BTC only
        }
        
        # Initialize and run trader
        trader = SafeFuturesTrader(config)
        
        if trader.initialize():
            import time
            trader.trading_loop(duration)
        else:
            print("❌ Failed to initialize safe trader")
            
    except KeyboardInterrupt:
        print("\n👋 Safe trading cancelled by user")
    except ValueError:
        print("❌ Invalid input - please enter numbers only")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()