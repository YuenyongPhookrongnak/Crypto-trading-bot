#!/usr/bin/env python3
"""
Simple Moving Average Trading Strategy

A basic strategy that demonstrates the core concepts of the trading bot.
This strategy uses two moving averages to generate buy/sell signals.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

class TradingSignal:
    """Simple trading signal class"""
    def __init__(self, symbol: str, signal_type: str, confidence: float, 
                 entry_price: float, reasoning: list = None):
        self.symbol = symbol
        self.signal_type = signal_type
        self.confidence = confidence
        self.entry_price = entry_price
        self.reasoning = reasoning or []
        self.timestamp = datetime.utcnow()
        
    def to_dict(self):
        return {
            'symbol': self.symbol,
            'signal_type': self.signal_type,
            'confidence': self.confidence,
            'entry_price': self.entry_price,
            'reasoning': self.reasoning,
            'timestamp': self.timestamp.isoformat()
        }

class SimpleMovingAverageStrategy:
    """
    Simple Moving Average Crossover Strategy
    
    This strategy generates:
    - BUY signal when short MA crosses above long MA
    - SELL signal when short MA crosses below long MA
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        self.name = "Simple Moving Average Strategy"
        self.short_period = config.get('short_period', 10)
        self.long_period = config.get('long_period', 21)
        self.min_confidence = config.get('min_confidence', 60.0)
        
    async def generate_signal(self, symbol: str, market_data: pd.DataFrame, 
                            current_price: float) -> Optional[TradingSignal]:
        """Generate trading signal based on MA crossover"""
        try:
            # Need enough data for long MA
            if len(market_data) < self.long_period + 1:
                return None
            
            # Calculate moving averages
            market_data['sma_short'] = market_data['close'].rolling(
                window=self.short_period
            ).mean()
            market_data['sma_long'] = market_data['close'].rolling(
                window=self.long_period
            ).mean()
            
            # Get current and previous values
            current = market_data.iloc[-1]
            previous = market_data.iloc[-2]
            
            # Check for bullish crossover (buy signal)
            if (current['sma_short'] > current['sma_long'] and 
                previous['sma_short'] <= previous['sma_long']):
                
                reasoning = [
                    f"Bullish crossover detected",
                    f"SMA {self.short_period}: ${current['sma_short']:.2f}",
                    f"SMA {self.long_period}: ${current['sma_long']:.2f}",
                    f"Current price: ${current_price:.2f}"
                ]
                
                return TradingSignal(
                    symbol=symbol,
                    signal_type='BUY',
                    confidence=75.0,
                    entry_price=current_price,
                    reasoning=reasoning
                )
            
            # Check for bearish crossover (sell signal)
            elif (current['sma_short'] < current['sma_long'] and 
                  previous['sma_short'] >= previous['sma_long']):
                
                reasoning = [
                    f"Bearish crossover detected",
                    f"SMA {self.short_period}: ${current['sma_short']:.2f}",
                    f"SMA {self.long_period}: ${current['sma_long']:.2f}",
                    f"Current price: ${current_price:.2f}"
                ]
                
                return TradingSignal(
                    symbol=symbol,
                    signal_type='SELL',
                    confidence=70.0,
                    entry_price=current_price,
                    reasoning=reasoning
                )
            
            return None
            
        except Exception as e:
            print(f"Error in strategy: {e}")
            return None
    
    def get_info(self) -> Dict[str, Any]:
        """Get strategy information"""
        return {
            'name': self.name,
            'short_period': self.short_period,
            'long_period': self.long_period,
            'description': f'MA crossover strategy ({self.short_period}/{self.long_period})'
        }

def generate_sample_market_data(symbol: str = 'BTC/USDT', periods: int = 50) -> pd.DataFrame:
    """Generate sample market data for testing"""
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # Base price
    base_price = 65000 if 'BTC' in symbol else 3200
    
    # Generate realistic price movement
    returns = np.random.normal(0, 0.02, periods)  # 2% daily volatility
    prices = [base_price]
    
    for i in range(1, periods):
        # Add some trend and mean reversion
        trend = 0.001 * np.sin(i / 10)  # Slight cyclical trend
        new_price = prices[-1] * (1 + trend + returns[i])
        prices.append(max(new_price, base_price * 0.8))  # Price floor
    
    # Create timestamps
    start_time = datetime.utcnow() - timedelta(hours=periods)
    timestamps = [start_time + timedelta(hours=i) for i in range(periods)]
    
    # Create OHLCV data
    data = []
    for i, (ts, close) in enumerate(zip(timestamps, prices)):
        open_price = prices[i-1] if i > 0 else close
        high = close * (1 + abs(np.random.normal(0, 0.01)))
        low = close * (1 - abs(np.random.normal(0, 0.01)))
        volume = np.random.randint(1000, 10000)
        
        data.append({
            'timestamp': ts,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    return pd.DataFrame(data)

async def test_strategy():
    """Test the simple strategy with sample data"""
    print("\n[Testing Simple Moving Average Strategy...]")
    
    # Create strategy
    strategy = SimpleMovingAverageStrategy({
        'short_period': 10,
        'long_period': 21
    })
    
    print(f"Strategy: {strategy.get_info()['name']}")
    print(f"Parameters: {strategy.short_period}/{strategy.long_period} MA")
    
    # Generate sample data
    print("\nGenerating sample market data...")
    market_data = generate_sample_market_data('BTC/USDT', 50)
    current_price = market_data['close'].iloc[-1]
    
    print(f"Generated {len(market_data)} data points")
    print(f"Current price: ${current_price:.2f}")
    print(f"Price range: ${market_data['low'].min():.2f} - ${market_data['high'].max():.2f}")
    
    # Test strategy
    print("\nAnalyzing market data...")
    signal = await strategy.generate_signal('BTC/USDT', market_data, current_price)
    
    if signal:
        print("\n[TRADING SIGNAL GENERATED!]")
        print("-" * 30)
        print(f"Symbol: {signal.symbol}")
        print(f"Signal: {signal.signal_type}")
        print(f"Confidence: {signal.confidence}%")
        print(f"Entry Price: ${signal.entry_price:.2f}")
        print(f"Timestamp: {signal.timestamp.strftime('%H:%M:%S')}")
        print("\nReasoning:")
        for reason in signal.reasoning:
            print(f"   - {reason}")
    else:
        print("\n[No trading signal at this time]")
        print("   Waiting for moving average crossover...")
        
        # Show current MA values
        sma_short = market_data['close'].rolling(window=strategy.short_period).mean().iloc[-1]
        sma_long = market_data['close'].rolling(window=strategy.long_period).mean().iloc[-1]
        trend = "BULLISH" if sma_short > sma_long else "BEARISH"
        
        print(f"   Current trend: {trend}")
        print(f"   SMA {strategy.short_period}: ${sma_short:.2f}")
        print(f"   SMA {strategy.long_period}: ${sma_long:.2f}")
    
    return True

if __name__ == "__main__":
    try:
        asyncio.run(test_strategy())
    except Exception as e:
        print(f"Error running strategy test: {e}")
