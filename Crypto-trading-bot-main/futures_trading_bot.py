#!/usr/bin/env python3
"""
Binance Futures Trading Bot

This bot is specifically designed for Binance Futures trading with
proper risk management, leverage control, and position management.
"""

import asyncio
import ccxt
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import os
import sys
from pathlib import Path

# Try to import python-dotenv, use fallback if not available
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    def load_dotenv():
        """Fallback function when python-dotenv is not available"""
        pass

# Load environment variables
load_dotenv()

class BinanceFuturesBot:
    """
    Binance Futures Trading Bot with advanced features:
    - Leverage management
    - Position sizing
    - Risk management
    - Multiple timeframes
    - Advanced order types
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # API Configuration
        self.api_key = os.getenv('BINANCE_API_KEY', '')
        self.secret = os.getenv('BINANCE_SECRET', '')
        self.testnet = os.getenv('BINANCE_TESTNET', 'true').lower() == 'true'
        
        # Trading Configuration
        self.initial_capital = self.config.get('initial_capital', 10000.0)
        self.max_positions = self.config.get('max_positions', 3)
        self.risk_per_trade = self.config.get('risk_per_trade', 0.02)  # 2%
        self.max_leverage = self.config.get('max_leverage', 5)  # Max 5x leverage
        self.default_leverage = self.config.get('default_leverage', 1)  # Start with 1x
        
        # Futures Specific Settings
        self.margin_type = self.config.get('margin_type', 'ISOLATED')  # ISOLATED or CROSSED
        self.position_side = self.config.get('position_side', 'BOTH')  # BOTH, LONG, SHORT
        
        # Risk Management
        self.max_daily_loss = self.config.get('max_daily_loss', 0.05)  # 5%
        self.max_drawdown = self.config.get('max_drawdown', 0.10)  # 10%
        self.force_close_loss = self.config.get('force_close_loss', 0.15)  # 15%
        
        # Initialize exchange
        self.exchange = None
        self.positions = {}
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        
        print("🚀 Binance Futures Trading Bot")
        print("=" * 40)
        print(f"💰 Initial Capital: ${self.initial_capital:,.2f}")
        print(f"📊 Max Positions: {self.max_positions}")
        print(f"⚠️ Risk per Trade: {self.risk_per_trade:.1%}")
        print(f"📈 Max Leverage: {self.max_leverage}x")
        print(f"🛡️ Margin Type: {self.margin_type}")
        print(f"🧪 Testnet: {'YES' if self.testnet else 'NO (LIVE TRADING!)'}")
    
    async def initialize_exchange(self):
        """Initialize Binance Futures exchange"""
        try:
            self.exchange = ccxt.binance({
                'apiKey': self.api_key,
                'secret': self.secret,
                'sandbox': self.testnet,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',  # Use futures
                    'adjustForTimeDifference': True,
                }
            })
            
            # Test connection
            markets = self.exchange.load_markets()
            futures_markets = [m for m in markets.values() if m['type'] == 'future']
            
            print(f"✅ Connected to Binance Futures")
            print(f"📊 Available contracts: {len(futures_markets)}")
            
            # Get account info
            balance = await self.get_futures_balance()
            print(f"💰 USDT Balance: {balance.get('USDT', 0):,.2f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize exchange: {e}")
            return False
    
    async def get_futures_balance(self):
        """Get futures account balance"""
        try:
            balance = self.exchange.fetch_balance()
            return balance['total']
        except Exception as e:
            print(f"❌ Error fetching balance: {e}")
            return {}
    
    async def set_leverage(self, symbol: str, leverage: int):
        """Set leverage for a symbol"""
        try:
            result = self.exchange.set_leverage(leverage, symbol)
            print(f"📈 Set leverage for {symbol}: {leverage}x")
            return True
        except Exception as e:
            print(f"⚠️ Error setting leverage for {symbol}: {e}")
            return False
    
    async def set_margin_type(self, symbol: str, margin_type: str = None):
        """Set margin type for a symbol"""
        try:
            margin_type = margin_type or self.margin_type
            self.exchange.set_margin_mode(margin_type, symbol)
            print(f"🛡️ Set margin type for {symbol}: {margin_type}")
            return True
        except Exception as e:
            # Margin type might already be set
            if "No need to change margin type" in str(e):
                return True
            print(f"⚠️ Error setting margin type for {symbol}: {e}")
            return False
    
    async def get_position_info(self, symbol: str = None):
        """Get position information"""
        try:
            positions = self.exchange.fetch_positions(symbol)
            
            # Filter only positions with contracts > 0
            active_positions = [
                pos for pos in positions 
                if pos['contracts'] and pos['contracts'] > 0
            ]
            
            return active_positions
            
        except Exception as e:
            print(f"❌ Error fetching positions: {e}")
            return []
    
    def calculate_position_size(self, symbol: str, entry_price: float, 
                              leverage: int = None) -> float:
        """Calculate position size based on risk management"""
        try:
            leverage = leverage or self.default_leverage
            
            # Get available balance
            balance = self.exchange.fetch_balance()
            available_usdt = balance['USDT']['free']
            
            # Calculate risk amount
            risk_amount = available_usdt * self.risk_per_trade
            
            # Calculate position size with leverage
            # Position value = risk_amount * leverage / (stop_loss_percentage)
            stop_loss_pct = 0.02  # 2% stop loss
            position_value = (risk_amount * leverage) / stop_loss_pct
            
            # Convert to contracts
            position_size = position_value / entry_price
            
            # Ensure minimum size
            market = self.exchange.market(symbol)
            min_size = market.get('limits', {}).get('amount', {}).get('min', 0.001)
            position_size = max(position_size, min_size)
            
            return position_size
            
        except Exception as e:
            print(f"❌ Error calculating position size: {e}")
            return 0.001  # Minimum fallback
    
    async def open_position(self, symbol: str, side: str, entry_price: float, 
                          leverage: int = None) -> bool:
        """Open a futures position"""
        try:
            leverage = leverage or self.default_leverage
            
            # Set up symbol for trading
            await self.set_leverage(symbol, leverage)
            await self.set_margin_type(symbol)
            
            # Calculate position size
            position_size = self.calculate_position_size(symbol, entry_price, leverage)
            
            if position_size <= 0:
                print(f"⚠️ Position size too small for {symbol}")
                return False
            
            # Place market order
            order = self.exchange.create_market_order(
                symbol=symbol,
                side=side.lower(),
                amount=position_size
            )
            
            print(f"🎯 FUTURES POSITION OPENED:")
            print(f"   Symbol: {symbol}")
            print(f"   Side: {side}")
            print(f"   Size: {position_size:.6f}")
            print(f"   Leverage: {leverage}x")
            print(f"   Entry Price: ${entry_price:,.2f}")
            print(f"   Order ID: {order['id']}")
            
            # Set stop loss and take profit
            await self.set_stop_loss_take_profit(symbol, side, entry_price, position_size)
            
            return True
            
        except Exception as e:
            print(f"❌ Error opening position: {e}")
            return False
    
    async def set_stop_loss_take_profit(self, symbol: str, side: str, 
                                      entry_price: float, position_size: float):
        """Set stop loss and take profit orders"""
        try:
            # Calculate stop loss and take profit levels
            if side.upper() == 'BUY':
                stop_loss_price = entry_price * 0.98  # 2% stop loss
                take_profit_price = entry_price * 1.04  # 4% take profit
                sl_side = 'sell'
                tp_side = 'sell'
            else:  # SELL
                stop_loss_price = entry_price * 1.02  # 2% stop loss
                take_profit_price = entry_price * 0.96  # 4% take profit
                sl_side = 'buy'
                tp_side = 'buy'
            
            # Place stop loss order
            try:
                sl_order = self.exchange.create_order(
                    symbol=symbol,
                    type='STOP_MARKET',
                    side=sl_side,
                    amount=position_size,
                    params={'stopPrice': stop_loss_price}
                )
                print(f"🛡️ Stop Loss set: ${stop_loss_price:.2f}")
            except Exception as e:
                print(f"⚠️ Failed to set stop loss: {e}")
            
            # Place take profit order
            try:
                tp_order = self.exchange.create_order(
                    symbol=symbol,
                    type='TAKE_PROFIT_MARKET',
                    side=tp_side,
                    amount=position_size,
                    params={'stopPrice': take_profit_price}
                )
                print(f"🎯 Take Profit set: ${take_profit_price:.2f}")
            except Exception as e:
                print(f"⚠️ Failed to set take profit: {e}")
                
        except Exception as e:
            print(f"❌ Error setting SL/TP: {e}")
    
    async def close_position(self, symbol: str, reason: str = "Manual"):
        """Close a futures position"""
        try:
            positions = await self.get_position_info(symbol)
            
            for position in positions:
                if position['contracts'] > 0:
                    # Determine close side (opposite of position)
                    close_side = 'sell' if position['side'] == 'long' else 'buy'
                    
                    # Close position with market order
                    order = self.exchange.create_market_order(
                        symbol=symbol,
                        side=close_side,
                        amount=position['contracts']
                    )
                    
                    pnl = position.get('unrealizedPnl', 0)
                    
                    print(f"🔴 POSITION CLOSED:")
                    print(f"   Symbol: {symbol}")
                    print(f"   Side: {position['side']}")
                    print(f"   Size: {position['contracts']:.6f}")
                    print(f"   PnL: ${pnl:,.2f}")
                    print(f"   Reason: {reason}")
                    
                    return True
            
            return False
            
        except Exception as e:
            print(f"❌ Error closing position: {e}")
            return False
    
    async def check_risk_management(self):
        """Check risk management rules"""
        try:
            # Get all positions
            positions = await self.get_position_info()
            
            total_unrealized_pnl = sum(pos.get('unrealizedPnl', 0) for pos in positions)
            
            # Check daily loss limit
            if abs(total_unrealized_pnl) > self.initial_capital * self.max_daily_loss:
                print(f"🚨 DAILY LOSS LIMIT EXCEEDED: ${total_unrealized_pnl:,.2f}")
                print("🛑 Closing all positions...")
                
                for position in positions:
                    await self.close_position(position['symbol'], "Daily Loss Limit")
                
                return False
            
            # Check drawdown limit
            if abs(total_unrealized_pnl) > self.initial_capital * self.max_drawdown:
                print(f"🚨 DRAWDOWN LIMIT EXCEEDED: ${total_unrealized_pnl:,.2f}")
                print("🛑 Closing all positions...")
                
                for position in positions:
                    await self.close_position(position['symbol'], "Drawdown Limit")
                
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Error checking risk management: {e}")
            return True
    
    async def get_futures_signals(self, symbols: List[str]):
        """Get trading signals for futures"""
        try:
            signals = []
            
            for symbol in symbols:
                # Fetch market data
                ohlcv = self.exchange.fetch_ohlcv(symbol, '1h', limit=100)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
                # Simple moving average strategy
                df['sma_short'] = df['close'].rolling(10).mean()
                df['sma_long'] = df['close'].rolling(21).mean()
                
                current = df.iloc[-1]
                previous = df.iloc[-2]
                
                # Check for crossover
                if (current['sma_short'] > current['sma_long'] and 
                    previous['sma_short'] <= previous['sma_long']):
                    
                    signals.append({
                        'symbol': symbol,
                        'side': 'BUY',
                        'entry_price': current['close'],
                        'confidence': 75.0,
                        'leverage': self.default_leverage
                    })
                    
                elif (current['sma_short'] < current['sma_long'] and 
                      previous['sma_short'] >= previous['sma_long']):
                    
                    signals.append({
                        'symbol': symbol,
                        'side': 'SELL',
                        'entry_price': current['close'],
                        'confidence': 70.0,
                        'leverage': self.default_leverage
                    })
            
            return signals
            
        except Exception as e:
            print(f"❌ Error getting signals: {e}")
            return []
    
    async def run_trading_session(self, symbols: List[str], duration_minutes: int = 60):
        """Run futures trading session"""
        print(f"\n🚀 Starting Futures Trading Session")
        print("=" * 45)
        print(f"📊 Symbols: {', '.join(symbols)}")
        print(f"⏱️ Duration: {duration_minutes} minutes")
        print(f"🎯 Max Positions: {self.max_positions}")
        print("")
        
        start_time = datetime.utcnow()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        scan_count = 0
        signals_generated = 0
        positions_opened = 0
        
        try:
            while datetime.utcnow() < end_time:
                scan_count += 1
                print(f"🔍 Scan #{scan_count} - {datetime.utcnow().strftime('%H:%M:%S')}")
                
                # Check risk management first
                if not await self.check_risk_management():
                    print("🛑 Risk management triggered - stopping trading")
                    break
                
                # Get current positions
                current_positions = await self.get_position_info()
                active_symbols = [pos['symbol'] for pos in current_positions]
                
                print(f"📊 Active positions: {len(current_positions)}")
                
                # Get trading signals
                signals = await self.get_futures_signals(symbols)
                
                for signal in signals:
                    symbol = signal['symbol']
                    
                    # Skip if already have position in this symbol
                    if symbol in active_symbols:
                        continue
                    
                    # Skip if max positions reached
                    if len(current_positions) >= self.max_positions:
                        print(f"⚠️ Max positions reached ({self.max_positions})")
                        break
                    
                    signals_generated += 1
                    print(f"🚨 Signal: {signal['side']} {symbol} @ ${signal['entry_price']:,.2f}")
                    
                    # Open position
                    if await self.open_position(
                        symbol=symbol,
                        side=signal['side'],
                        entry_price=signal['entry_price'],
                        leverage=signal['leverage']
                    ):
                        positions_opened += 1
                        current_positions = await self.get_position_info()  # Refresh
                
                # Show position summary
                if current_positions:
                    total_pnl = sum(pos.get('unrealizedPnl', 0) for pos in current_positions)
                    print(f"💰 Total Unrealized PnL: ${total_pnl:,.2f}")
                
                # Wait before next scan
                await asyncio.sleep(60)  # 1 minute between scans
                
        except KeyboardInterrupt:
            print(f"\n⏹️ Trading session stopped by user")
        
        # Session summary
        print(f"\n📋 Futures Trading Session Summary:")
        print("=" * 40)
        print(f"Duration: {(datetime.utcnow() - start_time).total_seconds()/60:.1f} minutes")
        print(f"Total Scans: {scan_count}")
        print(f"Signals Generated: {signals_generated}")
        print(f"Positions Opened: {positions_opened}")
        
        # Final positions
        final_positions = await self.get_position_info()
        if final_positions:
            total_pnl = sum(pos.get('unrealizedPnl', 0) for pos in final_positions)
            print(f"Final Unrealized PnL: ${total_pnl:,.2f}")
            print(f"Open Positions: {len(final_positions)}")

async def main():
    """Main function"""
    print("🚀 Binance Futures Trading Bot")
    print("=" * 40)
    
    # Configuration
    config = {
        'initial_capital': 10000.0,
        'max_positions': 2,
        'risk_per_trade': 0.02,  # 2%
        'default_leverage': 2,   # 2x leverage
        'max_leverage': 5,       # Max 5x
        'margin_type': 'ISOLATED'
    }
    
    bot = BinanceFuturesBot(config)
    
    if not await bot.initialize_exchange():
        print("❌ Failed to initialize exchange")
        return
    
    print("\n🎯 Futures Trading Options:")
    print("1. Run 10-minute trading session")
    print("2. Check current positions")
    print("3. Check account balance")
    print("4. Exit")
    
    try:
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            symbols = ['BTC/USDT', 'ETH/USDT']
            await bot.run_trading_session(symbols, duration_minutes=10)
            
        elif choice == "2":
            positions = await bot.get_position_info()
            if positions:
                print(f"\n📊 Current Positions:")
                for pos in positions:
                    print(f"   {pos['symbol']}: {pos['side']} {pos['contracts']:.6f} "
                          f"(PnL: ${pos.get('unrealizedPnl', 0):,.2f})")
            else:
                print("📊 No open positions")
                
        elif choice == "3":
            balance = await bot.get_futures_balance()
            print(f"\n💰 Account Balance:")
            for asset, amount in balance.items():
                if amount > 0:
                    print(f"   {asset}: {amount:,.6f}")
        
        print("\n✅ Futures bot session completed!")
        
    except KeyboardInterrupt:
        print("\n👋 Futures bot interrupted by user")

if __name__ == "__main__":
    asyncio.run(main())