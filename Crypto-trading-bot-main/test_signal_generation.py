#!/usr/bin/env python3
"""
Test Signal Generation

This script creates different market scenarios to test
how your strategy responds to various market conditions.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add strategies directory to path
sys.path.append(str(Path(__file__).parent / 'strategies'))

try:
    from simple_strategy import SimpleMovingAverageStrategy, TradingSignal
    STRATEGY_AVAILABLE = True
except ImportError:
    STRATEGY_AVAILABLE = False

def create_bullish_crossover_data():
    """Create data that will trigger a bullish crossover signal"""
    periods = 50
    np.random.seed(123)  # Different seed for different pattern
    
    # Start with bearish trend, then create bullish crossover
    base_price = 45000
    prices = []
    
    # First 30 periods: bearish trend
    for i in range(30):
        trend = -0.002  # Downward trend
        noise = np.random.normal(0, 0.01)
        price = base_price * (1 + (trend + noise) * i/10)
        prices.append(max(price, base_price * 0.8))
    
    # Last 20 periods: recovery and bullish crossover
    for i in range(20):
        trend = 0.003  # Upward trend
        noise = np.random.normal(0, 0.008)
        price = prices[-1] * (1 + trend + noise)
        prices.append(price)
    
    # Create DataFrame
    start_time = datetime.utcnow() - timedelta(hours=periods)
    timestamps = [start_time + timedelta(hours=i) for i in range(periods)]
    
    data = []
    for i, (ts, close) in enumerate(zip(timestamps, prices)):
        open_price = prices[i-1] if i > 0 else close
        high = close * (1 + abs(np.random.normal(0, 0.005)))
        low = close * (1 - abs(np.random.normal(0, 0.005)))
        volume = np.random.randint(5000, 15000)
        
        data.append({
            'timestamp': ts,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    return pd.DataFrame(data)

def create_bearish_crossover_data():
    """Create data that will trigger a bearish crossover signal"""
    periods = 50
    np.random.seed(456)  # Different pattern
    
    base_price = 50000
    prices = []
    
    # First 30 periods: bullish trend
    for i in range(30):
        trend = 0.002  # Upward trend
        noise = np.random.normal(0, 0.01)
        price = base_price * (1 + (trend + noise) * i/10)
        prices.append(price)
    
    # Last 20 periods: decline and bearish crossover
    for i in range(20):
        trend = -0.004  # Strong downward trend
        noise = np.random.normal(0, 0.01)
        price = prices[-1] * (1 + trend + noise)
        prices.append(max(price, base_price * 0.7))
    
    # Create DataFrame
    start_time = datetime.utcnow() - timedelta(hours=periods)
    timestamps = [start_time + timedelta(hours=i) for i in range(periods)]
    
    data = []
    for i, (ts, close) in enumerate(zip(timestamps, prices)):
        open_price = prices[i-1] if i > 0 else close
        high = close * (1 + abs(np.random.normal(0, 0.005)))
        low = close * (1 - abs(np.random.normal(0, 0.005)))
        volume = np.random.randint(5000, 15000)
        
        data.append({
            'timestamp': ts,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    return pd.DataFrame(data)

def create_sideways_market_data():
    """Create sideways/consolidating market data"""
    periods = 50
    np.random.seed(789)
    
    base_price = 47000
    prices = []
    
    # Sideways movement with noise
    for i in range(periods):
        # Small cyclical movement
        cycle = 0.001 * np.sin(i / 5)
        noise = np.random.normal(0, 0.015)
        price = base_price * (1 + cycle + noise)
        prices.append(price)
    
    # Create DataFrame
    start_time = datetime.utcnow() - timedelta(hours=periods)
    timestamps = [start_time + timedelta(hours=i) for i in range(periods)]
    
    data = []
    for i, (ts, close) in enumerate(zip(timestamps, prices)):
        open_price = prices[i-1] if i > 0 else close
        high = close * (1 + abs(np.random.normal(0, 0.008)))
        low = close * (1 - abs(np.random.normal(0, 0.008)))
        volume = np.random.randint(3000, 8000)
        
        data.append({
            'timestamp': ts,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    return pd.DataFrame(data)

async def test_scenario(scenario_name, market_data, strategy):
    """Test strategy with a specific market scenario"""
    print(f"\n📊 Testing Scenario: {scenario_name}")
    print("-" * 40)
    
    current_price = market_data['close'].iloc[-1]
    
    # Calculate moving averages for display
    sma_10 = market_data['close'].rolling(10).mean().iloc[-1]
    sma_21 = market_data['close'].rolling(21).mean().iloc[-1]
    trend = "BULLISH" if sma_10 > sma_21 else "BEARISH"
    
    print(f"💰 Current Price: ${current_price:.2f}")
    print(f"📊 SMA 10: ${sma_10:.2f}")
    print(f"📊 SMA 21: ${sma_21:.2f}")
    print(f"📈 Trend: {trend}")
    
    # Test strategy
    signal = await strategy.generate_signal('BTC/USDT', market_data, current_price)
    
    if signal:
        print(f"\n🚨 SIGNAL GENERATED!")
        print(f"   Type: {signal.signal_type}")
        print(f"   Confidence: {signal.confidence}%")
        print(f"   Entry: ${signal.entry_price:.2f}")
        print(f"   Reasoning:")
        for reason in signal.reasoning:
            print(f"     • {reason}")
        return True
    else:
        print(f"\n⏳ No signal generated")
        distance = abs(sma_10 - sma_21)
        print(f"   Distance to crossover: ${distance:.2f}")
        return False

async def main():
    """Main testing function"""
    print("🧪 Signal Generation Testing")
    print("=" * 40)
    print("Testing strategy with different market scenarios...")
    
    if not STRATEGY_AVAILABLE:
        print("❌ Strategy module not available")
        return
    
    # Initialize strategy
    strategy = SimpleMovingAverageStrategy({
        'short_period': 10,
        'long_period': 21
    })
    
    print(f"📈 Strategy: {strategy.name}")
    print(f"🔧 Parameters: SMA {strategy.short_period} / SMA {strategy.long_period}")
    
    # Test different scenarios
    scenarios = [
        ("Bullish Crossover", create_bullish_crossover_data()),
        ("Bearish Crossover", create_bearish_crossover_data()),
        ("Sideways Market", create_sideways_market_data())
    ]
    
    results = []
    
    for scenario_name, data in scenarios:
        signal_generated = await test_scenario(scenario_name, data, strategy)
        results.append((scenario_name, signal_generated))
    
    # Summary
    print(f"\n📋 Test Results Summary:")
    print("=" * 30)
    
    for scenario, has_signal in results:
        status = "✅ SIGNAL" if has_signal else "⏳ NO SIGNAL"
        print(f"{scenario:20} {status}")
    
    signals_generated = sum(1 for _, has_signal in results if has_signal)
    print(f"\nSignal Generation Rate: {signals_generated}/{len(results)} ({signals_generated/len(results)*100:.1f}%)")
    
    print(f"\n💡 Key Insights:")
    print("   • Crossover patterns generate clear signals")
    print("   • Sideways markets rarely produce signals (good!)")
    print("   • Strategy waits for clear directional moves")
    print("   • This prevents overtrading in choppy markets")
    
    print(f"\n🎯 Next Steps:")
    print("   • Test with live market data: python test_live_strategy.py")
    print("   • Adjust strategy parameters if needed")
    print("   • Add additional filters (volume, RSI, etc.)")
    print("   • Test with different timeframes")

if __name__ == "__main__":
    asyncio.run(main())