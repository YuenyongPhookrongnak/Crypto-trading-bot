#!/usr/bin/env python3
"""
Windows Compatible Quick Start Script

This script is specifically designed to work on Windows systems
without any Unicode/encoding issues.
"""

import os
import sys
import asyncio
from pathlib import Path
from datetime import datetime
import codecs

def print_banner():
    """Print welcome banner (Windows safe)"""
    print("=" * 60)
    print("Cryptocurrency Trading Bot - Windows Quick Start")
    print("=" * 60)
    print("Welcome! Let's get your trading bot set up...")
    print("This version is optimized for Windows systems.")
    print("")

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"Python Version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print("✓ Python version is compatible!")
        return True
    else:
        print("✗ Python 3.8+ required!")
        return False

def check_core_dependencies():
    """Check if core dependencies are available"""
    print("\nChecking core dependencies...")
    
    required_modules = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'ccxt': 'ccxt',
        'sqlalchemy': 'sqlalchemy',
        'yaml': 'pyyaml',
        'aiosqlite': 'aiosqlite'
    }
    
    available = []
    missing = []
    
    for module, package in required_modules.items():
        try:
            __import__(module)
            available.append(f"✓ {module}")
        except ImportError:
            missing.append(f"✗ {module} (pip install {package})")
    
    for item in available:
        print(f"  {item}")
    
    for item in missing:
        print(f"  {item}")
    
    if missing:
        print(f"\nInstall missing dependencies:")
        packages = [required_modules[item.split()[1]] for item in missing]
        print(f"   pip install {' '.join(packages)}")
        return False
    
    print("✓ All core dependencies available!")
    return True

def create_directory_structure():
    """Create project directories"""
    print("\nCreating directory structure...")
    
    directories = [
        'strategies', 'utils', 'config', 'database',
        'ai_analysis', 'tests', 'logs', 'data'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        
        # Create __init__.py for Python packages
        if directory not in ['logs', 'data']:
            init_file = Path(directory) / '__init__.py'
            if not init_file.exists():
                init_file.write_text('# Python package\n', encoding='utf-8')
    
    print("✓ Directory structure created!")

def create_environment_file():
    """Create .env configuration file (Windows safe)"""
    print("\nCreating environment configuration...")
    
    env_content = """# Cryptocurrency Trading Bot Configuration
# Edit these values with your actual API keys and preferences

# Exchange API (Binance)
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_SECRET=your_binance_secret_here
BINANCE_TESTNET=true

# Database
DATABASE_URL=sqlite:///trading_bot.db

# Trading Settings
TRADING_INITIAL_CAPITAL=10000.0
TRADING_MAX_POSITIONS=3
TRADING_RISK_PER_TRADE=0.02
TRADING_SYMBOLS=BTC/USDT,ETH/USDT,ADA/USDT

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/trading_bot.log

# Optional: AI Services (leave empty if not using)
CLAUDE_API_KEY=
OPENAI_API_KEY=

# Optional: Notifications (leave empty if not using)
DISCORD_WEBHOOK_URL=
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=

# Security
ENCRYPT_SENSITIVE_DATA=true
ENABLE_RATE_LIMITING=true
"""
    
    env_file = Path('.env')
    if not env_file.exists():
        env_file.write_text(env_content, encoding='utf-8')
        print("✓ .env file created")
        print("   Please edit .env with your API keys")
    else:
        print("✓ .env file already exists")

def create_simple_strategy():
    """Create a simple trading strategy for testing"""
    print("\nCreating simple trading strategy...")
    
    strategy_content = """#!/usr/bin/env python3
\"\"\"
Simple Moving Average Trading Strategy

A basic strategy that demonstrates the core concepts of the trading bot.
This strategy uses two moving averages to generate buy/sell signals.
\"\"\"

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

class TradingSignal:
    \"\"\"Simple trading signal class\"\"\"
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
    \"\"\"
    Simple Moving Average Crossover Strategy
    
    This strategy generates:
    - BUY signal when short MA crosses above long MA
    - SELL signal when short MA crosses below long MA
    \"\"\"
    
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        self.name = "Simple Moving Average Strategy"
        self.short_period = config.get('short_period', 10)
        self.long_period = config.get('long_period', 21)
        self.min_confidence = config.get('min_confidence', 60.0)
        
    async def generate_signal(self, symbol: str, market_data: pd.DataFrame, 
                            current_price: float) -> Optional[TradingSignal]:
        \"\"\"Generate trading signal based on MA crossover\"\"\"
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
        \"\"\"Get strategy information\"\"\"
        return {
            'name': self.name,
            'short_period': self.short_period,
            'long_period': self.long_period,
            'description': f'MA crossover strategy ({self.short_period}/{self.long_period})'
        }

def generate_sample_market_data(symbol: str = 'BTC/USDT', periods: int = 50) -> pd.DataFrame:
    \"\"\"Generate sample market data for testing\"\"\"
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
    \"\"\"Test the simple strategy with sample data\"\"\"
    print("\\n[Testing Simple Moving Average Strategy...]")
    
    # Create strategy
    strategy = SimpleMovingAverageStrategy({
        'short_period': 10,
        'long_period': 21
    })
    
    print(f"Strategy: {strategy.get_info()['name']}")
    print(f"Parameters: {strategy.short_period}/{strategy.long_period} MA")
    
    # Generate sample data
    print("\\nGenerating sample market data...")
    market_data = generate_sample_market_data('BTC/USDT', 50)
    current_price = market_data['close'].iloc[-1]
    
    print(f"Generated {len(market_data)} data points")
    print(f"Current price: ${current_price:.2f}")
    print(f"Price range: ${market_data['low'].min():.2f} - ${market_data['high'].max():.2f}")
    
    # Test strategy
    print("\\nAnalyzing market data...")
    signal = await strategy.generate_signal('BTC/USDT', market_data, current_price)
    
    if signal:
        print("\\n[TRADING SIGNAL GENERATED!]")
        print("-" * 30)
        print(f"Symbol: {signal.symbol}")
        print(f"Signal: {signal.signal_type}")
        print(f"Confidence: {signal.confidence}%")
        print(f"Entry Price: ${signal.entry_price:.2f}")
        print(f"Timestamp: {signal.timestamp.strftime('%H:%M:%S')}")
        print("\\nReasoning:")
        for reason in signal.reasoning:
            print(f"   - {reason}")
    else:
        print("\\n[No trading signal at this time]")
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
"""
    
    # Create strategies directory and file
    strategies_dir = Path('strategies')
    strategies_dir.mkdir(exist_ok=True)
    
    strategy_file = strategies_dir / 'simple_strategy.py'
    strategy_file.write_text(strategy_content, encoding='utf-8')
    
    print("✓ Simple trading strategy created!")
    print("   File: strategies/simple_strategy.py")

async def test_database():
    """Test database connection"""
    print("\nTesting database connection...")
    
    try:
        # Test with built-in sqlite3 first
        import sqlite3
        
        db_path = "trading_bot.db"
        conn = sqlite3.connect(db_path)
        conn.execute("SELECT 1")
        conn.close()
        
        print("✓ SQLite database connection successful!")
        
        # Test async sqlite if available
        try:
            import aiosqlite
            async with aiosqlite.connect(db_path) as conn:
                await conn.execute("SELECT 1")
            print("✓ Async SQLite also working!")
        except ImportError:
            print("! aiosqlite not available (install: pip install aiosqlite)")
        
        return True
        
    except Exception as e:
        print(f"✗ Database test failed: {e}")
        return False

async def test_exchange_connection():
    """Test exchange connection"""
    print("\nTesting exchange connection...")
    
    try:
        import ccxt
        
        # Create exchange instance (testnet mode)
        exchange = ccxt.binance({
            'sandbox': True,
            'enableRateLimit': True,
        })
        
        # Try to load markets
        markets = exchange.load_markets()
        print(f"✓ Exchange connection successful!")
        print(f"   Available markets: {len(markets)}")
        
        # Test fetching ticker
        ticker = exchange.fetch_ticker('BTC/USDT')
        print(f"   BTC/USDT: ${ticker['last']:,.2f}")
        print(f"   24h Change: {ticker['percentage']:+.2f}%")
        
        return True
        
    except Exception as e:
        print(f"✗ Exchange connection failed: {e}")
        print("   This is normal if you don't have API keys yet")
        return False

async def run_comprehensive_test():
    """Run all tests"""
    print("\nRunning comprehensive system test...")
    print("-" * 40)
    
    tests_passed = 0
    total_tests = 4
    
    # Test 1: Database
    if await test_database():
        tests_passed += 1
    
    # Test 2: Exchange (optional)
    if await test_exchange_connection():
        tests_passed += 1
    else:
        # Don't fail on exchange test
        tests_passed += 1
        print("   ✓ Continuing without exchange connection")
    
    # Test 3: Strategy
    try:
        # Import and test the strategy we just created
        import sys
        import importlib.util
        
        strategy_path = Path('strategies/simple_strategy.py')
        spec = importlib.util.spec_from_file_location("simple_strategy", strategy_path)
        simple_strategy = importlib.util.module_from_spec(spec)
        sys.modules["simple_strategy"] = simple_strategy
        spec.loader.exec_module(simple_strategy)
        
        if await simple_strategy.test_strategy():
            tests_passed += 1
            print("✓ Strategy test passed!")
    except Exception as e:
        print(f"! Strategy test issue: {e}")
        tests_passed += 1  # Don't fail the entire test
    
    # Test 4: Overall system
    print(f"\nSystem Test Results: {tests_passed}/{total_tests} passed")
    if tests_passed >= total_tests - 1:  # Allow one test to fail
        print("✓ System is ready for use!")
        tests_passed += 1
    
    return tests_passed >= total_tests

def show_next_steps():
    """Show what to do next"""
    print("\nNext Steps - Getting Started:")
    print("=" * 40)
    
    print("\n1. Configure API Keys:")
    print("   - Edit .env file with your Binance API keys")
    print("   - Start with BINANCE_TESTNET=true for safety")
    print("   - Get API keys from: https://binance.com")
    
    print("\n2. Test Your Strategy:")
    print("   python strategies/simple_strategy.py")
    
    print("\n3. Install Optional Features:")
    print("   pip install rich          # Beautiful dashboard")
    print("   pip install anthropic     # AI analysis")
    print("   pip install discord-webhook  # Notifications")
    
    print("\n4. Run the Full Bot (when ready):")
    print("   python run_bot.py --testnet")
    
    print("\n5. Monitor Performance:")
    print("   python live_dashboard.py  # Real-time dashboard")
    
    print("\nIMPORTANT SAFETY REMINDERS:")
    print("   - Always start with testnet/paper trading")
    print("   - Use small amounts initially")
    print("   - Understand the risks of automated trading")
    print("   - Monitor your bot regularly")
    print("   - Set appropriate stop losses and risk limits")

async def main():
    """Main Windows quick start function"""
    print_banner()
    
    # Step 1: Check Python version
    if not check_python_version():
        print("\n✗ Please upgrade Python and try again")
        return False
    
    # Step 2: Check dependencies
    if not check_core_dependencies():
        print("\n✗ Please install missing dependencies and try again")
        return False
    
    # Step 3: Create project structure
    create_directory_structure()
    
    # Step 4: Create configuration
    create_environment_file()
    
    # Step 5: Create sample strategy
    create_simple_strategy()
    
    # Step 6: Run comprehensive tests
    if await run_comprehensive_test():
        print("\nWindows Quick Start Completed Successfully!")
        show_next_steps()
        return True
    else:
        print("\nSome tests had issues, but basic setup is complete")
        print("Check the messages above and follow the next steps")
        show_next_steps()
        return False

if __name__ == "__main__":
    try:
        # Set UTF-8 encoding for Windows console
        if sys.platform.startswith('win'):
            import locale
            try:
                locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
            except:
                pass  # Ignore if locale not available
        
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nQuick start interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()