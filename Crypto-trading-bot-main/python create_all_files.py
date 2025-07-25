#!/usr/bin/env python3
"""
Create All Missing Files Script

This script creates all the missing files for the trading bot system.
Run this once to generate all required files.
"""

import os
from pathlib import Path

def create_enhanced_strategy():
    """Create enhanced_strategy.py"""
    content = '''#!/usr/bin/env python3
"""
Enhanced Trading Strategy

This strategy improves upon the simple MA strategy by adding:
- Volume confirmation
- RSI filter
- Risk management
- Market condition analysis
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent))

try:
    from strategies.simple_strategy import TradingSignal
    SIMPLE_STRATEGY_AVAILABLE = True
except ImportError:
    SIMPLE_STRATEGY_AVAILABLE = False
    # Define TradingSignal if not available
    class TradingSignal:
        def __init__(self, symbol, signal_type, confidence, entry_price, reasoning=None):
            self.symbol = symbol
            self.signal_type = signal_type
            self.confidence = confidence
            self.entry_price = entry_price
            self.reasoning = reasoning or []
            self.timestamp = datetime.utcnow()

class EnhancedMovingAverageStrategy:
    """Enhanced Moving Average Strategy with multiple filters"""
    
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        
        # Strategy identification
        self.name = "Enhanced Moving Average Strategy"
        self.version = "1.1"
        
        # Moving Average parameters
        self.short_period = config.get('short_period', 10)
        self.long_period = config.get('long_period', 21)
        
        # Volume filter
        self.volume_filter = config.get('volume_filter', True)
        self.volume_period = config.get('volume_period', 20)
        self.min_volume_ratio = config.get('min_volume_ratio', 1.2)
        
        # RSI filter
        self.rsi_filter = config.get('rsi_filter', True)
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_oversold = config.get('rsi_oversold', 30)
        self.rsi_overbought = config.get('rsi_overbought', 70)
        
        # Risk management
        self.base_confidence = config.get('base_confidence', 70.0)
        self.min_confidence = config.get('min_confidence', 55.0)
        self.max_confidence = config.get('max_confidence', 90.0)
    
    def calculate_rsi(self, prices: pd.Series, period: int = None) -> pd.Series:
        """Calculate RSI indicator"""
        if period is None:
            period = self.rsi_period
            
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50)
            
        except Exception as e:
            print(f"Error calculating RSI: {e}")
            return pd.Series([50] * len(prices), index=prices.index)
    
    def analyze_volume_profile(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume profile"""
        try:
            if 'volume' not in data.columns:
                return {'ratio': 1.0, 'trend': 'neutral', 'confirmation': False}
            
            volume_ma = data['volume'].rolling(window=self.volume_period).mean()
            current_volume = data['volume'].iloc[-1]
            avg_volume = volume_ma.iloc[-1]
            
            if pd.isna(avg_volume) or avg_volume == 0:
                return {'ratio': 1.0, 'trend': 'neutral', 'confirmation': False}
            
            volume_ratio = current_volume / avg_volume
            volume_confirmation = volume_ratio >= self.min_volume_ratio
            
            return {
                'ratio': volume_ratio,
                'confirmation': volume_confirmation,
                'current': current_volume,
                'average': avg_volume
            }
            
        except Exception as e:
            print(f"Error analyzing volume: {e}")
            return {'ratio': 1.0, 'trend': 'neutral', 'confirmation': False}
    
    async def generate_signal(self, symbol: str, market_data: pd.DataFrame, 
                            current_price: float) -> Optional[TradingSignal]:
        """Generate enhanced trading signal"""
        try:
            if len(market_data) < max(self.long_period, self.volume_period, self.rsi_period) + 2:
                return None
            
            # Calculate moving averages
            market_data = market_data.copy()
            market_data['sma_short'] = market_data['close'].rolling(window=self.short_period).mean()
            market_data['sma_long'] = market_data['close'].rolling(window=self.long_period).mean()
            
            # Calculate RSI
            market_data['rsi'] = self.calculate_rsi(market_data['close'])
            
            # Get current and previous values
            current = market_data.iloc[-1]
            previous = market_data.iloc[-2]
            
            # Check for moving average crossover
            crossover_signal = None
            base_confidence = self.base_confidence
            
            if (current['sma_short'] > current['sma_long'] and 
                previous['sma_short'] <= previous['sma_long']):
                crossover_signal = 'BUY'
                
            elif (current['sma_short'] < current['sma_long'] and 
                  previous['sma_short'] >= previous['sma_long']):
                crossover_signal = 'SELL'
                base_confidence = self.base_confidence - 5
            
            if not crossover_signal:
                return None
            
            # Apply filters
            volume_analysis = self.analyze_volume_profile(market_data)
            current_rsi = current['rsi']
            
            # Volume filter
            if self.volume_filter and not volume_analysis['confirmation']:
                return None
            
            # RSI filter
            if self.rsi_filter:
                if crossover_signal == 'BUY' and current_rsi > self.rsi_overbought:
                    return None
                elif crossover_signal == 'SELL' and current_rsi < self.rsi_oversold:
                    return None
            
            # Build reasoning
            reasoning = [
                f"{'Bullish' if crossover_signal == 'BUY' else 'Bearish'} MA crossover detected",
                f"SMA {self.short_period}: ${current['sma_short']:.2f}",
                f"SMA {self.long_period}: ${current['sma_long']:.2f}",
                f"RSI: {current_rsi:.1f}",
                f"Volume: {volume_analysis['ratio']:.1f}x average"
            ]
            
            return TradingSignal(
                symbol=symbol,
                signal_type=crossover_signal,
                confidence=base_confidence,
                entry_price=current_price,
                reasoning=reasoning
            )
            
        except Exception as e:
            print(f"Error generating enhanced signal: {e}")
            return None
    
    def get_info(self) -> Dict[str, Any]:
        """Get strategy information"""
        return {
            'name': self.name,
            'version': self.version,
            'parameters': {
                'short_period': self.short_period,
                'long_period': self.long_period,
                'volume_filter': self.volume_filter,
                'rsi_filter': self.rsi_filter,
            },
            'description': 'Enhanced MA strategy with volume and RSI filters'
        }

async def test_enhanced_strategy():
    """Test the enhanced strategy"""
    print("🧪 Testing Enhanced Moving Average Strategy")
    print("=" * 50)
    
    # Create test data
    np.random.seed(42)
    periods = 60
    base_price = 50000
    
    prices = [base_price]
    volumes = []
    
    for i in range(1, periods):
        if i < 30:
            trend = -0.001
        elif i < 40:
            trend = 0.0005
        else:
            trend = 0.002
        
        noise = np.random.normal(0, 0.015)
        new_price = prices[-1] * (1 + trend + noise)
        prices.append(new_price)
        
        if 35 <= i <= 45:
            volume = np.random.randint(8000, 15000)
        else:
            volume = np.random.randint(3000, 8000)
        volumes.append(volume)
    
    volumes.insert(0, 5000)
    
    timestamps = [datetime.utcnow() - timedelta(hours=periods-i) for i in range(periods)]
    
    test_data = pd.DataFrame({
        'timestamp': timestamps,
        'open': [p * 0.999 for p in prices],
        'high': [p * 1.005 for p in prices],
        'low': [p * 0.995 for p in prices],
        'close': prices,
        'volume': volumes
    })
    
    strategy = EnhancedMovingAverageStrategy({
        'short_period': 10,
        'long_period': 21,
        'volume_filter': True,
        'rsi_filter': True
    })
    
    print(f"📊 Strategy: {strategy.get_info()['name']}")
    
    current_price = test_data['close'].iloc[-1]
    signal = await strategy.generate_signal('BTC/USDT', test_data, current_price)
    
    print(f"\\nCurrent Price: ${current_price:.2f}")
    
    if signal:
        print(f"\\n🚨 ENHANCED SIGNAL GENERATED!")
        print(f"   Type: {signal.signal_type}")
        print(f"   Confidence: {signal.confidence:.1f}%")
        print(f"   Entry Price: ${signal.entry_price:.2f}")
        print(f"\\n💡 Analysis:")
        for reason in signal.reasoning:
            print(f"     • {reason}")
    else:
        print(f"\\n⏳ No signal generated")
        print("   Enhanced filters prevented signal generation")
    
    print(f"\\n✅ Enhanced strategy test completed!")

if __name__ == "__main__":
    asyncio.run(test_enhanced_strategy())
'''
    
    with open('enhanced_strategy.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Created enhanced_strategy.py")

def create_multi_strategy_trader():
    """Create multi_strategy_trader.py"""
    content = '''#!/usr/bin/env python3
"""
Multi-Strategy Trading System

This system combines multiple trading strategies and makes
trading decisions based on consensus and weighted signals.
"""

import asyncio
import ccxt
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import json
from pathlib import Path
import sys

# Add strategies to path
sys.path.append(str(Path(__file__).parent / 'strategies'))
sys.path.append(str(Path(__file__).parent))

try:
    from strategies.simple_strategy import SimpleMovingAverageStrategy, TradingSignal
    SIMPLE_STRATEGY_AVAILABLE = True
except ImportError:
    SIMPLE_STRATEGY_AVAILABLE = False

try:
    from enhanced_strategy import EnhancedMovingAverageStrategy
    ENHANCED_STRATEGY_AVAILABLE = True
except ImportError:
    ENHANCED_STRATEGY_AVAILABLE = False

class MultiStrategyTrader:
    """Multi-Strategy Trading System"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Trading settings
        self.initial_capital = self.config.get('initial_capital', 10000.0)
        self.max_positions = self.config.get('max_positions', 3)
        self.risk_per_trade = self.config.get('risk_per_trade', 0.02)
        
        # Strategy settings
        self.min_consensus = self.config.get('min_consensus', 0.6)
        self.min_total_confidence = self.config.get('min_total_confidence', 70.0)
        
        # Initialize strategies
        self.strategies = {}
        self.strategy_weights = {}
        self.exchange = None
        
        # Portfolio tracking
        self.current_capital = self.initial_capital
        self.positions = {}
        self.trade_history = []
        
        print(f"🤖 Multi-Strategy Trader initialized")
        print(f"   Initial Capital: ${self.initial_capital:,.2f}")
        print(f"   Max Positions: {self.max_positions}")
        print(f"   Risk per Trade: {self.risk_per_trade:.1%}")
    
    async def initialize_strategies(self):
        """Initialize all available strategies"""
        try:
            strategy_count = 0
            
            # Simple MA Strategy
            if SIMPLE_STRATEGY_AVAILABLE:
                self.strategies['simple_ma'] = SimpleMovingAverageStrategy({
                    'short_period': 10,
                    'long_period': 21
                })
                self.strategy_weights['simple_ma'] = 0.4
                strategy_count += 1
            
            # Enhanced MA Strategy
            if ENHANCED_STRATEGY_AVAILABLE:
                self.strategies['enhanced_ma'] = EnhancedMovingAverageStrategy({
                    'short_period': 10,
                    'long_period': 21,
                    'volume_filter': True,
                    'rsi_filter': True
                })
                self.strategy_weights['enhanced_ma'] = 0.6
                strategy_count += 1
            
            if strategy_count == 0:
                print("❌ No strategies available")
                return False
            
            print(f"✅ Initialized {strategy_count} strategies:")
            for name, weight in self.strategy_weights.items():
                print(f"   • {name}: {weight:.1%} weight")
            
            return True
            
        except Exception as e:
            print(f"❌ Error initializing strategies: {e}")
            return False
    
    async def initialize_exchange(self):
        """Initialize exchange connection"""
        try:
            self.exchange = ccxt.binance({
                'sandbox': True,
                'enableRateLimit': True,
            })
            
            markets = self.exchange.load_markets()
            print(f"✅ Connected to exchange ({len(markets)} markets)")
            return True
            
        except Exception as e:
            print(f"❌ Exchange connection failed: {e}")
            return False
    
    async def fetch_market_data(self, symbol: str, timeframe: str = '1h', limit: int = 100):
        """Fetch market data for analysis"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
            
        except Exception as e:
            print(f"❌ Error fetching data for {symbol}: {e}")
            return None
    
    async def get_consensus_signal(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get consensus signal from all strategies"""
        try:
            market_data = await self.fetch_market_data(symbol)
            if market_data is None:
                return None
            
            current_price = market_data['close'].iloc[-1]
            
            # Collect signals from all strategies
            signals = {}
            for strategy_name, strategy in self.strategies.items():
                try:
                    signal = await strategy.generate_signal(symbol, market_data, current_price)
                    signals[strategy_name] = signal
                except Exception as e:
                    print(f"⚠️  Error getting signal from {strategy_name}: {e}")
                    signals[strategy_name] = None
            
            # Analyze consensus
            consensus_analysis = self.analyze_signal_consensus(signals)
            
            if consensus_analysis['has_consensus']:
                return {
                    'symbol': symbol,
                    'consensus_signal': consensus_analysis['consensus_signal'],
                    'total_confidence': consensus_analysis['total_confidence'],
                    'consensus_ratio': consensus_analysis['consensus_ratio'],
                    'individual_signals': signals,
                    'current_price': current_price,
                    'timestamp': datetime.utcnow()
                }
            
            return None
            
        except Exception as e:
            print(f"❌ Error getting consensus signal: {e}")
            return None
    
    def analyze_signal_consensus(self, signals: Dict[str, Optional[TradingSignal]]) -> Dict[str, Any]:
        """Analyze consensus among strategy signals"""
        try:
            valid_signals = {k: v for k, v in signals.items() if v is not None}
            
            if not valid_signals:
                return {'has_consensus': False, 'reason': 'no_signals'}
            
            signal_votes = {'BUY': 0, 'SELL': 0}
            total_confidence = 0
            total_weight = 0
            
            for strategy_name, signal in valid_signals.items():
                weight = self.strategy_weights.get(strategy_name, 1.0)
                signal_votes[signal.signal_type] += weight
                total_confidence += signal.confidence * weight
                total_weight += weight
            
            max_votes = max(signal_votes.values())
            consensus_signal = [k for k, v in signal_votes.items() if v == max_votes][0]
            
            consensus_ratio = max_votes / total_weight if total_weight > 0 else 0
            weighted_confidence = total_confidence / total_weight if total_weight > 0 else 0
            
            has_consensus = (
                consensus_ratio >= self.min_consensus and
                weighted_confidence >= self.min_total_confidence
            )
            
            return {
                'has_consensus': has_consensus,
                'consensus_signal': consensus_signal,
                'consensus_ratio': consensus_ratio,
                'total_confidence': weighted_confidence,
                'valid_signals_count': len(valid_signals)
            }
            
        except Exception as e:
            print(f"❌ Error analyzing consensus: {e}")
            return {'has_consensus': False, 'reason': 'analysis_error'}
    
    async def execute_paper_trade(self, signal_data: Dict[str, Any]) -> bool:
        """Execute paper trade (simulation)"""
        try:
            symbol = signal_data['symbol']
            signal_type = signal_data['consensus_signal']
            current_price = signal_data['current_price']
            
            # Calculate position size
            available_capital = self.current_capital
            risk_amount = available_capital * self.risk_per_trade
            confidence_multiplier = min(signal_data['total_confidence'] / 100.0, 1.0)
            position_size = risk_amount * confidence_multiplier
            
            if position_size <= 0:
                return False
            
            quantity = position_size / current_price
            
            # Create trade record
            trade = {
                'id': len(self.trade_history) + 1,
                'symbol': symbol,
                'type': signal_type,
                'quantity': quantity,
                'entry_price': current_price,
                'position_size': position_size,
                'timestamp': datetime.utcnow(),
                'confidence': signal_data['total_confidence'],
                'consensus_ratio': signal_data['consensus_ratio'],
                'status': 'OPEN'
            }
            
            if symbol not in self.positions:
                self.positions[symbol] = []
            
            self.positions[symbol].append(trade)
            self.trade_history.append(trade)
            self.current_capital -= position_size
            
            print(f"📊 PAPER TRADE EXECUTED:")
            print(f"   Symbol: {symbol}")
            print(f"   Type: {signal_type}")
            print(f"   Quantity: {quantity:.6f}")
            print(f"   Entry Price: ${current_price:,.2f}")
            print(f"   Position Size: ${position_size:,.2f}")
            print(f"   Confidence: {signal_data['total_confidence']:.1f}%")
            print(f"   Consensus: {signal_data['consensus_ratio']:.1%}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error executing paper trade: {e}")
            return False
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get current portfolio summary"""
        try:
            total_positions = sum(len(positions) for positions in self.positions.values())
            
            return {
                'current_capital': self.current_capital,
                'initial_capital': self.initial_capital,
                'total_positions': total_positions,
                'total_trades': len(self.trade_history),
                'risk_utilization': ((self.initial_capital - self.current_capital) / self.initial_capital) * 100
            }
            
        except Exception as e:
            print(f"❌ Error getting portfolio summary: {e}")
            return {}
    
    async def run_trading_session(self, symbols: List[str], duration_minutes: int = 60):
        """Run a trading session"""
        print(f"\\n🚀 Starting Multi-Strategy Trading Session")
        print("=" * 50)
        print(f"Symbols: {', '.join(symbols)}")
        print(f"Duration: {duration_minutes} minutes\\n")
        
        start_time = datetime.utcnow()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        scan_count = 0
        signals_generated = 0
        trades_executed = 0
        
        try:
            while datetime.utcnow() < end_time:
                scan_count += 1
                print(f"🔍 Scan #{scan_count} - {datetime.utcnow().strftime('%H:%M:%S')}")
                
                for symbol in symbols:
                    try:
                        consensus_data = await self.get_consensus_signal(symbol)
                        
                        if consensus_data:
                            signals_generated += 1
                            print(f"   🚨 Consensus signal for {symbol}:")
                            print(f"      Signal: {consensus_data['consensus_signal']}")
                            print(f"      Confidence: {consensus_data['total_confidence']:.1f}%")
                            print(f"      Consensus: {consensus_data['consensus_ratio']:.1%}")
                            
                            total_positions = sum(len(pos) for pos in self.positions.values())
                            
                            if total_positions < self.max_positions:
                                if await self.execute_paper_trade(consensus_data):
                                    trades_executed += 1
                            else:
                                print(f"   ⚠️  Max positions reached")
                        else:
                            print(f"   ⏳ No consensus for {symbol}")
                            
                    except Exception as e:
                        print(f"   ❌ Error scanning {symbol}: {e}")
                
                portfolio = self.get_portfolio_summary()
                print(f"   💼 Portfolio: ${portfolio.get('current_capital', 0):,.2f} available")
                print(f"   📊 Positions: {portfolio.get('total_positions', 0)} open")
                
                await asyncio.sleep(30)
                
        except KeyboardInterrupt:
            print(f"\\n⏹️  Trading session stopped by user")
        
        print(f"\\n📋 Trading Session Summary:")
        print("=" * 30)
        print(f"Duration: {(datetime.utcnow() - start_time).total_seconds()/60:.1f} minutes")
        print(f"Total Scans: {scan_count}")
        print(f"Signals Generated: {signals_generated}")
        print(f"Trades Executed: {trades_executed}")
        
        portfolio = self.get_portfolio_summary()
        print(f"Capital Utilization: {portfolio.get('risk_utilization', 0):.1f}%")
        print(f"Open Positions: {portfolio.get('total_positions', 0)}")

async def test_multi_strategy_system():
    """Test the multi-strategy trading system"""
    print("🧪 Testing Multi-Strategy Trading System")
    print("=" * 50)
    
    config = {
        'initial_capital': 10000.0,
        'max_positions': 2,
        'risk_per_trade': 0.02,
        'min_consensus': 0.6,
        'min_total_confidence': 65.0
    }
    
    trader = MultiStrategyTrader(config)
    
    if not await trader.initialize_strategies():
        print("❌ Failed to initialize strategies")
        return
    
    if not await trader.initialize_exchange():
        print("❌ Failed to initialize exchange")
        return
    
    print(f"\\n✅ Multi-Strategy System Ready!")
    
    # Test consensus signal generation
    print(f"\\n🔍 Testing Consensus Signal Generation:")
    symbols_to_test = ['BTC/USDT', 'ETH/USDT']
    
    for symbol in symbols_to_test:
        print(f"\\n📊 Analyzing {symbol}...")
        consensus_data = await trader.get_consensus_signal(symbol)
        
        if consensus_data:
            print(f"   ✅ Consensus Signal: {consensus_data['consensus_signal']}")
            print(f"   🎯 Total Confidence: {consensus_data['total_confidence']:.1f}%")
            print(f"   🤝 Consensus Ratio: {consensus_data['consensus_ratio']:.1%}")
            print(f"   💰 Current Price: ${consensus_data['current_price']:,.2f}")
            
            print(f"   📋 Individual Signals:")
            for strategy_name, signal in consensus_data['individual_signals'].items():
                if signal:
                    weight = trader.strategy_weights.get(strategy_name, 1.0)
                    print(f"     • {strategy_name}: {signal.signal_type} ({signal.confidence:.1f}%) [Weight: {weight:.1f}]")
                else:
                    print(f"     • {strategy_name}: No signal")
        else:
            print(f"   ⏳ No consensus signal for {symbol}")
    
    # Ask user for options
    print(f"\\n🎯 Options:")
    print("1. Run 5-minute paper trading session")
    print("2. Exit")
    
    try:
        choice = input("\\nEnter choice (1-2): ").strip()
        
        if choice == "1":
            await trader.run_trading_session(['BTC/USDT', 'ETH/USDT'], duration_minutes=5)
        
        print(f"\\n✅ Multi-strategy system test completed!")
        
    except KeyboardInterrupt:
        print(f"\\n👋 Test interrupted by user")

if __name__ == "__main__":
    asyncio.run(test_multi_strategy_system())
'''
    
    with open('multi_strategy_trader.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Created multi_strategy_trader.py")

def create_run_bot():
    """Create run_bot.py"""
    content = '''#!/usr/bin/env python3
"""
Trading Bot Launcher

This script launches the trading bot with various options and configurations.
"""

import asyncio
import argparse
import sys
from pathlib import Path
from datetime import datetime
import json

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

try:
    from multi_strategy_trader import MultiStrategyTrader
    MULTI_STRATEGY_AVAILABLE = True
except ImportError:
    MULTI_STRATEGY_AVAILABLE = False

def load_config(config_file: str = None) -> dict:
    """Load configuration from file or use defaults"""
    default_config = {
        'initial_capital': 10000.0,
        'max_positions': 3,
        'risk_per_trade': 0.02,
        'min_consensus': 0.7,
        'min_total_confidence': 75.0,
        'symbols': ['BTC/USDT', 'ETH/USDT'],
        'scan_interval': 30,
        'session_duration': 1440  # 24 hours
    }
    
    if config_file and Path(config_file).exists():
        try:
            with open(config_file) as f:
                file_config = json.load(f)
            default_config.update(file_config)
            print(f"✅ Loaded configuration from {config_file}")
        except Exception as e:
            print(f"⚠️  Error loading config file: {e}")
            print("   Using default configuration")
    
    return default_config

def print_startup_banner():
    """Print startup banner"""
    print("🤖 Cryptocurrency Trading Bot")
    print("=" * 50)
    print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("⚠️  IMPORTANT: This bot trades with real money!")
    print("   Always start with small amounts and testnet")
    print("")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Cryptocurrency Trading Bot')
    
    parser.add_argument('--testnet', action='store_true', 
                       help='Use testnet/sandbox mode (recommended)')
    parser.add_argument('--live', action='store_true',
                       help='Use live trading (REAL MONEY!)')
    parser.add_argument('--capital', type=float, default=10000.0,
                       help='Initial capital amount')
    parser.add_argument('--risk', type=float, default=0.02,
                       help='Risk per trade (0.02 = 2%)')
    parser.add_argument('--symbols', nargs='+', default=['BTC/USDT', 'ETH/USDT'],
                       help='Trading symbols')
    parser.add_argument('--duration', type=int, default=1440,
                       help='Session duration in minutes (1440 = 24 hours)')
    parser.add_argument('--config', type=str,
                       help='Configuration file path')
    parser.add_argument('--check-only', action='store_true',
                       help='Check system and exit')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')
    
    return parser.parse_args()

def check_system_requirements():
    """Check if system requirements are met"""
    print("🔍 Checking system requirements...")
    
    requirements_met = True
    
    # Check Python version
    import sys
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    else:
        print(f"❌ Python version too old: {version.major}.{version.minor}.{version.micro}")
        requirements_met = False
    
    # Check required modules
    required_modules = ['pandas', 'numpy', 'ccxt', 'asyncio']
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module} not installed")
            requirements_met = False
    
    # Check multi-strategy availability
    if MULTI_STRATEGY_AVAILABLE:
        print("✅ Multi-strategy system available")
    else:
        print("❌ Multi-strategy system not available")
        requirements_met = False
    
    return requirements_met

async def main():
    """Main function"""
    args = parse_arguments()
    
    print_startup_banner()
    
    # Check system requirements
    if not check_system_requirements():
        print("\\n❌ System requirements not met!")
        print("Please install missing dependencies and try again.")
        return False
    
    if args.check_only:
        print("\\n✅ System check completed successfully!")
        return True
    
    # Load configuration
    config = load_config(args.config)
    
    # Override with command line arguments
    if args.capital:
        config['initial_capital'] = args.capital
    if args.risk:
        config['risk_per_trade'] = args.risk
    if args.symbols:
        config['symbols'] = args.symbols
    if args.duration:
        config['session_duration'] = args.duration
    
    # Validate trading mode
    if args.live and args.testnet:
        print("❌ Cannot use both --live and --testnet flags")
        return False
    
    if not args.live and not args.testnet:
        print("⚠️  No trading mode specified, using testnet by default")
        args.testnet = True
    
    if args.live:
        print("🚨 LIVE TRADING MODE ENABLED")
        print("   This will use REAL MONEY!")
        confirmation = input("   Type 'CONFIRM' to proceed: ")
        if confirmation != 'CONFIRM':
            print("👋 Cancelled by user")
            return False
    
    print(f"\\n🚀 Starting trading bot...")
    print(f"   Mode: {'TESTNET' if args.testnet else 'LIVE'}")
    print(f"   Capital: ${config['initial_capital']:,.2f}")
    print(f"   Risk per trade: {config['risk_per_trade']:.1%}")
    print(f"   Symbols: {', '.join(config['symbols'])}")
    print(f"   Duration: {config['session_duration']} minutes")
    
    if not MULTI_STRATEGY_AVAILABLE:
        print("❌ Multi-strategy system not available")
        return False
    
    # Initialize and run trader
    trader = MultiStrategyTrader(config)
    
    if await trader.initialize_strategies() and await trader.initialize_exchange():
        await trader.run_trading_session(
            symbols=config['symbols'],
            duration_minutes=config['session_duration']
        )
        print("✅ Trading session completed")
        return True
    else:
        print("❌ Failed to initialize trading system")
        return False

if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\\n👋 Bot stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
'''
    
    with open('run_bot.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Created run_bot.py")

def main():
    """Main function to create all files"""
    print("🔧 Creating all missing trading bot files...")
    print("=" * 50)
    
    # Create all the files
    create_enhanced_strategy()
    create_multi_strategy_trader()
    create_run_bot()
    
    print("\n🎉 All files created successfully!")
    print("\n📋 Next steps:")
    print("1. Test enhanced strategy:")
    print("   python enhanced_strategy.py")
    print("\n2. Test multi-strategy system:")
    print("   python multi_strategy_trader.py")
    print("\n3. Run the trading bot:")
    print("   python run_bot.py --testnet --capital 1000")
    print("\n4. Check system requirements:")
    print("   python run_bot.py --check-only")
    print("\n⚠️  Remember: Always start with testnet mode!")

if __name__ == "__main__":
    main()