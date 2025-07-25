#!/usr/bin/env python3
"""
Test Strategy with Live Market Data

This script tests your trading strategy with real market data
from Binance exchange to see how it performs with current conditions.
"""

import asyncio
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# Add strategies directory to path for imports
sys.path.append(str(Path(__file__).parent / 'strategies'))

try:
    from simple_strategy import SimpleMovingAverageStrategy, TradingSignal
    STRATEGY_AVAILABLE = True
except ImportError:
    STRATEGY_AVAILABLE = False

class LiveStrategyTester:
    """Test trading strategies with live market data"""
    
    def __init__(self):
        self.exchange = None
        self.strategy = None
        
    async def initialize_exchange(self):
        """Initialize exchange connection"""
        try:
            self.exchange = ccxt.binance({
                'sandbox': True,  # Use testnet
                'enableRateLimit': True,
            })
            
            # Test connection
            markets = self.exchange.load_markets()
            print(f"✓ Connected to Binance Testnet")
            print(f"  Available markets: {len(markets)}")
            
            return True
            
        except Exception as e:
            print(f"✗ Exchange connection failed: {e}")
            return False
    
    async def fetch_live_data(self, symbol='BTC/USDT', timeframe='1h', limit=100):
        """Fetch live market data"""
        try:
            print(f"\nFetching live data for {symbol}...")
            
            # Get OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            print(f"✓ Fetched {len(df)} candles")
            print(f"  Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"  Price range: ${df['low'].min():,.2f} - ${df['high'].max():,.2f}")
            
            return df
            
        except Exception as e:
            print(f"✗ Error fetching data: {e}")
            return None
    
    def analyze_market_condition(self, df):
        """Analyze current market conditions"""
        print(f"\n📊 Market Condition Analysis:")
        print("-" * 40)
        
        current_price = df['close'].iloc[-1]
        prev_price = df['close'].iloc[-2]
        daily_change = ((current_price - df['close'].iloc[-24]) / df['close'].iloc[-24]) * 100 if len(df) >= 24 else 0
        
        # Calculate volatility
        returns = df['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(24)  # 24h volatility
        
        # Volume analysis
        avg_volume = df['volume'].rolling(20).mean().iloc[-1]
        current_volume = df['volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        print(f"💰 Current Price: ${current_price:,.2f}")
        print(f"📈 Price Change (1h): {((current_price - prev_price) / prev_price) * 100:+.2f}%")
        print(f"📊 Daily Change: {daily_change:+.2f}%")
        print(f"⚡ Volatility (24h): {volatility * 100:.2f}%")
        print(f"📊 Volume Ratio: {volume_ratio:.2f}x")
        
        # Market condition
        if volatility > 0.05:
            condition = "HIGH VOLATILITY"
        elif abs(daily_change) > 5:
            condition = "TRENDING" if daily_change > 0 else "DECLINING"
        elif volume_ratio > 1.5:
            condition = "HIGH ACTIVITY"
        else:
            condition = "CONSOLIDATING"
        
        print(f"🔍 Market Condition: {condition}")
        
        return {
            'current_price': current_price,
            'daily_change': daily_change,
            'volatility': volatility,
            'volume_ratio': volume_ratio,
            'condition': condition
        }
    
    async def test_strategy_with_live_data(self, symbol='BTC/USDT'):
        """Test strategy with live market data"""
        print(f"\n🧪 Testing Strategy with Live Data: {symbol}")
        print("=" * 50)
        
        # Fetch live data
        df = await self.fetch_live_data(symbol, '1h', 50)
        if df is None:
            return False
        
        # Analyze market conditions
        market_info = self.analyze_market_condition(df)
        
        # Initialize strategy
        if STRATEGY_AVAILABLE:
            self.strategy = SimpleMovingAverageStrategy({
                'short_period': 10,
                'long_period': 21
            })
            
            print(f"\n📈 Strategy Analysis:")
            print("-" * 25)
            
            # Test strategy
            current_price = market_info['current_price']
            signal = await self.strategy.generate_signal(symbol, df, current_price)
            
            # Calculate moving averages for display
            sma_10 = df['close'].rolling(10).mean().iloc[-1]
            sma_21 = df['close'].rolling(21).mean().iloc[-1]
            trend = "BULLISH" if sma_10 > sma_21 else "BEARISH"
            
            print(f"📊 SMA 10: ${sma_10:,.2f}")
            print(f"📊 SMA 21: ${sma_21:,.2f}")
            print(f"📈 Trend: {trend}")
            print(f"🎯 Distance to crossover: {abs(sma_10 - sma_21):,.2f} ({abs(sma_10 - sma_21)/sma_21*100:.2f}%)")
            
            if signal:
                print(f"\n🚨 TRADING SIGNAL DETECTED!")
                print("-" * 30)
                print(f"📊 Symbol: {signal.symbol}")
                print(f"📈 Signal: {signal.signal_type}")
                print(f"🎯 Confidence: {signal.confidence}%")
                print(f"💰 Entry Price: ${signal.entry_price:,.2f}")
                print(f"⏰ Timestamp: {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                
                print(f"\n💡 Reasoning:")
                for reason in signal.reasoning:
                    print(f"   • {reason}")
                
                # Risk analysis
                if signal.signal_type == 'BUY':
                    potential_sl = current_price * 0.97  # 3% stop loss
                    potential_tp = current_price * 1.06  # 6% take profit
                    print(f"\n⚠️  Suggested Risk Management:")
                    print(f"   Stop Loss: ${potential_sl:,.2f} (-3%)")
                    print(f"   Take Profit: ${potential_tp:,.2f} (+6%)")
                    print(f"   Risk/Reward: 1:2")
                
                return True
            else:
                print(f"\n⏳ No trading signal at this time")
                
                # Show what would trigger a signal
                if trend == "BEARISH":
                    price_for_crossover = sma_21 * 1.001  # Need price to push SMA 10 above SMA 21
                    print(f"   Need price above ~${price_for_crossover:,.2f} for bullish crossover")
                else:
                    price_for_crossover = sma_21 * 0.999
                    print(f"   Need price below ~${price_for_crossover:,.2f} for bearish crossover")
                
                return False
        
        else:
            print("⚠️  Strategy module not available")
            return False
    
    async def compare_multiple_symbols(self, symbols=['BTC/USDT', 'ETH/USDT', 'ADA/USDT']):
        """Compare strategy performance across multiple symbols"""
        print(f"\n🔍 Multi-Symbol Strategy Analysis")
        print("=" * 50)
        
        results = []
        
        for symbol in symbols:
            try:
                print(f"\n📊 Analyzing {symbol}...")
                
                # Fetch data
                df = await self.fetch_live_data(symbol, '1h', 50)
                if df is None:
                    continue
                
                current_price = df['close'].iloc[-1]
                
                # Quick analysis
                sma_10 = df['close'].rolling(10).mean().iloc[-1]
                sma_21 = df['close'].rolling(21).mean().iloc[-1]
                trend = "BULLISH" if sma_10 > sma_21 else "BEARISH"
                
                # Calculate distance to crossover
                distance_pct = abs(sma_10 - sma_21) / sma_21 * 100
                
                # Test for signal
                signal = None
                if STRATEGY_AVAILABLE:
                    signal = await self.strategy.generate_signal(symbol, df, current_price)
                
                result = {
                    'symbol': symbol,
                    'price': current_price,
                    'trend': trend,
                    'distance_to_crossover': distance_pct,
                    'has_signal': signal is not None,
                    'signal_type': signal.signal_type if signal else None
                }
                
                results.append(result)
                
                print(f"   Price: ${current_price:,.2f} | Trend: {trend} | Signal: {'✓' if signal else '✗'}")
                
            except Exception as e:
                print(f"   ✗ Error analyzing {symbol}: {e}")
        
        # Summary
        print(f"\n📋 Summary:")
        print("-" * 20)
        
        total_symbols = len(results)
        signals = [r for r in results if r['has_signal']]
        bullish_trends = [r for r in results if r['trend'] == 'BULLISH']
        
        print(f"Total Symbols: {total_symbols}")
        print(f"Active Signals: {len(signals)}")
        print(f"Bullish Trends: {len(bullish_trends)}")
        print(f"Signal Rate: {len(signals)/total_symbols*100:.1f}%")
        
        if signals:
            print(f"\n🚨 Active Signals:")
            for signal in signals:
                print(f"   {signal['symbol']}: {signal['signal_type']} (${signal['price']:,.2f})")
        
        return results

async def main():
    """Main testing function"""
    print("🧪 Live Strategy Testing Tool")
    print("=" * 40)
    
    tester = LiveStrategyTester()
    
    # Initialize exchange
    if not await tester.initialize_exchange():
        print("❌ Cannot connect to exchange. Check internet connection.")
        return
    
    print("\nWhat would you like to test?")
    print("1. Single symbol analysis (BTC/USDT)")
    print("2. Multi-symbol comparison")
    print("3. Custom symbol")
    
    try:
        choice = input("\nEnter choice (1-3) or press Enter for option 1: ").strip()
        
        if choice == "2":
            await tester.compare_multiple_symbols()
        elif choice == "3":
            symbol = input("Enter symbol (e.g., ETH/USDT): ").strip().upper()
            if symbol:
                await tester.test_strategy_with_live_data(symbol)
            else:
                await tester.test_strategy_with_live_data()
        else:
            # Default: BTC/USDT analysis
            await tester.test_strategy_with_live_data('BTC/USDT')
        
        print(f"\n✅ Live strategy testing completed!")
        print(f"\n💡 Tips:")
        print("   • Signals are more reliable in trending markets")
        print("   • High volatility can cause false signals")
        print("   • Always use proper risk management")
        print("   • Consider multiple timeframes for confirmation")
        
    except KeyboardInterrupt:
        print("\n👋 Testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")

if __name__ == "__main__":
    asyncio.run(main())