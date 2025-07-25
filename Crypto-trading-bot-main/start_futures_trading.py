#!/usr/bin/env python3
"""
Start Futures Trading

This script helps you start futures trading with proper configuration
and safety checks.
"""

import asyncio
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import os
import json

class FuturesTrader:
    """Production-ready Futures Trading Bot"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # API Configuration
        self.api_key = os.getenv('BINANCE_API_KEY', '')
        self.secret = os.getenv('BINANCE_SECRET', '')
        self.testnet = os.getenv('BINANCE_TESTNET', 'true').lower() == 'true'
        
        # Trading Configuration
        self.initial_capital = self.config.get('initial_capital', 1000.0)
        self.max_positions = self.config.get('max_positions', 2)
        self.risk_per_trade = self.config.get('risk_per_trade', 0.01)  # 1%
        self.leverage = self.config.get('leverage', 2)  # 2x leverage
        self.symbols = self.config.get('symbols', ['BTC/USDT', 'ETH/USDT'])
        
        # Risk Management
        self.max_daily_loss = self.config.get('max_daily_loss', 0.05)  # 5%
        self.stop_loss_pct = self.config.get('stop_loss_pct', 0.02)  # 2%
        self.take_profit_pct = self.config.get('take_profit_pct', 0.04)  # 4%
        
        # Strategy Settings
        self.sma_short = self.config.get('sma_short', 10)
        self.sma_long = self.config.get('sma_long', 21)
        self.rsi_period = self.config.get('rsi_period', 14)
        self.rsi_oversold = self.config.get('rsi_oversold', 30)
        self.rsi_overbought = self.config.get('rsi_overbought', 70)
        
        # Initialize
        self.exchange = None
        self.positions = {}
        self.daily_pnl = 0.0
        self.session_start = datetime.utcnow()
        
        self.print_config()
    
    def print_config(self):
        """Print trading configuration"""
        print("🚀 Futures Trading Bot - Production Ready")
        print("=" * 50)
        print(f"💰 Initial Capital: ${self.initial_capital:,.2f}")
        print(f"📊 Max Positions: {self.max_positions}")
        print(f"⚠️ Risk per Trade: {self.risk_per_trade:.1%}")
        print(f"📈 Leverage: {self.leverage}x")
        print(f"🎯 Symbols: {', '.join(self.symbols)}")
        print(f"🛡️ Stop Loss: {self.stop_loss_pct:.1%}")
        print(f"💰 Take Profit: {self.take_profit_pct:.1%}")
        print(f"🚨 Max Daily Loss: {self.max_daily_loss:.1%}")
        print(f"🧪 Mode: {'TESTNET' if self.testnet else '🔴 LIVE TRADING'}")
        print("")
    
    async def initialize(self):
        """Initialize exchange and setup"""
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
            
            # Test connection (synchronous call)
            self.exchange.load_markets()
            balance = self.exchange.fetch_balance()
            
            print("✅ Exchange connected successfully")
            
            # Setup symbols
            for symbol in self.symbols:
                await self.setup_symbol(symbol)
            
            # Show initial balance
            usdt_balance = balance['USDT']['total']
            print(f"💰 USDT Balance: {usdt_balance:,.2f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            return False
    
    async def setup_symbol(self, symbol: str):
        """Setup symbol for trading"""
        try:
            # Set leverage (synchronous call)
            self.exchange.set_leverage(self.leverage, symbol)
            print(f"📈 {symbol}: Leverage set to {self.leverage}x")
            
            # Set margin type to ISOLATED (synchronous call)
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
    
    async def get_market_data(self, symbol: str, timeframe='1h', limit=100):
        """Get market data for analysis"""
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
    
    def analyze_signal(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Analyze market data for trading signals"""
        try:
            if len(df) < 2:
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
                if current['rsi'] < self.rsi_overbought:  # Not overbought
                    signal = 'BUY'
                    confidence = 75.0
                    
                    # Boost confidence if RSI is oversold
                    if current['rsi'] < self.rsi_oversold:
                        confidence = 85.0
            
            # Bearish signal
            elif (current['sma_short'] < current['sma_long'] and 
                  previous['sma_short'] >= previous['sma_long']):
                
                # Additional filters
                if current['rsi'] > self.rsi_oversold:  # Not oversold
                    signal = 'SELL'
                    confidence = 70.0
                    
                    # Boost confidence if RSI is overbought
                    if current['rsi'] > self.rsi_overbought:
                        confidence = 80.0
            
            if signal:
                return {
                    'symbol': df.iloc[-1].name if hasattr(df.iloc[-1], 'name') else 'UNKNOWN',
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
    
    def calculate_position_size(self, symbol: str, price: float) -> float:
        """Calculate position size based on risk management"""
        try:
            # Get available balance
            balance = self.exchange.fetch_balance()
            available_usdt = balance['USDT']['free']
            
            # Calculate risk amount
            risk_amount = available_usdt * self.risk_per_trade
            
            # Account for leverage
            position_value = risk_amount * self.leverage / self.stop_loss_pct
            position_size = position_value / price
            
            # Get minimum size from market info
            market = self.exchange.market(symbol)
            min_size = market.get('limits', {}).get('amount', {}).get('min', 0.001)
            
            return max(position_size, min_size)
            
        except Exception as e:
            print(f"❌ Error calculating position size: {e}")
            return 0.001
    
    async def open_position(self, signal: Dict[str, Any]) -> bool:
        """Open a futures position"""
        try:
            symbol = signal['symbol']
            side = signal['signal'].lower()
            price = signal['price']
            
            # Calculate position size
            size = self.calculate_position_size(symbol, price)
            
            if size <= 0:
                return False
            
            # Place market order
            order = self.exchange.create_market_order(symbol, side, size)
            
            print(f"🎯 POSITION OPENED:")
            print(f"   📊 Symbol: {symbol}")
            print(f"   📈 Side: {side.upper()}")
            print(f"   💰 Size: {size:.6f}")
            print(f"   💵 Price: ${price:,.2f}")
            print(f"   🎲 Confidence: {signal['confidence']:.1f}%")
            print(f"   📈 Leverage: {self.leverage}x")
            
            # Set stop loss and take profit
            await self.set_stop_orders(symbol, side, price, size)
            
            # Track position
            self.positions[symbol] = {
                'side': side,
                'size': size,
                'entry_price': price,
                'timestamp': datetime.utcnow(),
                'order_id': order['id']
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Error opening position: {e}")
            return False
    
    async def set_stop_orders(self, symbol: str, side: str, entry_price: float, size: float):
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
    
    async def check_positions(self):
        """Check and update positions"""
        try:
            positions = self.exchange.fetch_positions()
            active_positions = [p for p in positions if p['contracts'] and p['contracts'] > 0]
            
            for position in active_positions:
                symbol = position['symbol']
                pnl = position.get('unrealizedPnl', 0)
                
                print(f"📊 {symbol}: {position['side']} {position['contracts']:.6f} "
                      f"(PnL: ${pnl:,.2f})")
            
            return active_positions
            
        except Exception as e:
            print(f"❌ Error checking positions: {e}")
            return []
    
    async def check_risk_limits(self):
        """Check risk management limits"""
        try:
            positions = await self.check_positions()
            total_pnl = sum(p.get('unrealizedPnl', 0) for p in positions)
            
            # Check daily loss limit
            loss_pct = abs(total_pnl) / self.initial_capital
            
            if loss_pct > self.max_daily_loss:
                print(f"🚨 DAILY LOSS LIMIT EXCEEDED: {loss_pct:.1%}")
                print("🛑 Closing all positions...")
                
                # Close all positions
                for position in positions:
                    if position['contracts'] > 0:
                        await self.close_position(position['symbol'], "Risk Limit")
                
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Error checking risk limits: {e}")
            return True
    
    async def close_position(self, symbol: str, reason: str = "Manual"):
        """Close a position"""
        try:
            positions = self.exchange.fetch_positions([symbol])
            
            for position in positions:
                if position['contracts'] > 0:
                    side = 'sell' if position['side'] == 'long' else 'buy'
                    
                    order = self.exchange.create_market_order(
                        symbol, side, position['contracts']
                    )
                    
                    pnl = position.get('unrealizedPnl', 0)
                    print(f"🔴 POSITION CLOSED: {symbol} (PnL: ${pnl:,.2f}) - {reason}")
                    
                    # Remove from tracking
                    if symbol in self.positions:
                        del self.positions[symbol]
                    
                    return True
            
            return False
            
        except Exception as e:
            print(f"❌ Error closing position: {e}")
            return False
    
    async def trading_loop(self, duration_hours: int):
        """Main trading loop"""
        print(f"\n🔄 Starting Trading Loop ({duration_hours} hours)")
        print("=" * 40)
        
        end_time = datetime.utcnow() + timedelta(hours=duration_hours)
        loop_count = 0
        
        try:
            while datetime.utcnow() < end_time:
                loop_count += 1
                print(f"\n🔍 Loop #{loop_count} - {datetime.utcnow().strftime('%H:%M:%S')}")
                
                # Check risk limits first
                if not await self.check_risk_limits():
                    print("🛑 Risk limits exceeded - stopping trading")
                    break
                
                # Check current positions
                current_positions = await self.check_positions()
                active_symbols = [p['symbol'] for p in current_positions]
                
                # Look for new signals
                for symbol in self.symbols:
                    # Skip if already have position
                    if symbol in active_symbols:
                        continue
                    
                    # Skip if max positions reached
                    if len(current_positions) >= self.max_positions:
                        break
                    
                    # Get market data and analyze
                    df = await self.get_market_data(symbol)
                    if df is None:
                        continue
                    
                    signal = self.analyze_signal(df)
                    if signal:
                        signal['symbol'] = symbol  # Ensure symbol is set
                        print(f"🚨 Signal: {signal['signal']} {symbol} @ ${signal['price']:,.2f}")
                        print(f"   RSI: {signal['rsi']:.1f}, Confidence: {signal['confidence']:.1f}%")
                        
                        # Open position
                        if await self.open_position(signal):
                            current_positions = await self.check_positions()
                
                # Show session summary
                session_duration = datetime.utcnow() - self.session_start
                total_pnl = sum(p.get('unrealizedPnl', 0) for p in current_positions)
                
                print(f"📈 Session: {session_duration.total_seconds()/3600:.1f}h, "
                      f"PnL: ${total_pnl:,.2f}, Positions: {len(current_positions)}")
                
                # Wait 2 minutes between loops
                await asyncio.sleep(120)
                
        except KeyboardInterrupt:
            print(f"\n⏹️ Trading stopped by user")
        
        # Final summary
        final_positions = await self.check_positions()
        final_pnl = sum(p.get('unrealizedPnl', 0) for p in final_positions)
        
        print(f"\n📋 Trading Session Complete:")
        print(f"   Duration: {loop_count} loops")
        print(f"   Final PnL: ${final_pnl:,.2f}")
        print(f"   Open Positions: {len(final_positions)}")

async def main():
    """Main function"""
    print("🚀 Futures Trading Launcher")
    print("=" * 30)
    
    # Get user preferences
    print("\n🎯 Trading Configuration:")
    
    try:
        capital = float(input("💰 Initial Capital ($): ") or "1000")
        leverage = int(input("📈 Leverage (1-5x): ") or "2")
        risk = float(input("⚠️ Risk per Trade (0.01 = 1%): ") or "0.01")
        duration = int(input("⏱️ Trading Duration (hours): ") or "1")
        
        # Confirm live trading
        testnet = os.getenv('BINANCE_TESTNET', 'true').lower() == 'true'
        if not testnet:
            print("\n🚨 WARNING: LIVE TRADING MODE!")
            confirm = input("This will use REAL MONEY! Type 'CONFIRM' to proceed: ")
            if confirm != 'CONFIRM':
                print("👋 Cancelled")
                return
        
        config = {
            'initial_capital': capital,
            'leverage': min(max(leverage, 1), 5),  # Limit 1-5x
            'risk_per_trade': min(max(risk, 0.005), 0.02),  # Limit 0.5%-2%
            'max_positions': 2,
            'symbols': ['BTC/USDT', 'ETH/USDT']
        }
        
        # Initialize and run trader
        trader = FuturesTrader(config)
        
        if await trader.initialize():
            await trader.trading_loop(duration)
        else:
            print("❌ Failed to initialize trader")
            
    except KeyboardInterrupt:
        print("\n👋 Trading cancelled by user")
    except ValueError:
        print("❌ Invalid input - please enter numbers only")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())