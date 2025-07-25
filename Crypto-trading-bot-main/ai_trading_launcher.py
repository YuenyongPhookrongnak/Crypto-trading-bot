#!/usr/bin/env python3
"""
AI-Enhanced Futures Trading Bot - Main Launcher

Complete trading system with AI analysis, professional confluences,
risk management, and comprehensive logging.
"""

import asyncio
import argparse
import sys
import os
import json
from pathlib import Path
from datetime import datetime

# Import our custom modules
try:
    from advanced_ai_trading_strategy import AdvancedAITradingBot, AdvancedMarketAnalyzer
    from trade_journal import TradeJournal, TradeRecord
    from notification_system import NotificationManager
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"❌ Module import error: {e}")
    MODULES_AVAILABLE = False

def print_banner():
    """Print startup banner"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                  🤖 AI-Enhanced Futures Trading Bot          ║
║                     Professional Trading System              ║
║                                                              ║
║  📊 Features:                                                ║
║  • AI Analysis (GPT + Claude)                               ║
║  • Professional Confluences                                 ║
║  • Advanced Risk Management                                  ║
║  • Real-time Notifications                                  ║
║  • Comprehensive Trade Journal                              ║
║  • Multiple Timeframe Analysis                              ║
║                                                              ║
║  ⚠️  IMPORTANT: Start with testnet mode!                     ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_environment():
    """Check if environment is properly configured"""
    print("🔍 Checking Environment Configuration...")
    
    required_vars = [
        'BINANCE_API_KEY',
        'BINANCE_SECRET', 
        'BINANCE_TESTNET'
    ]
    
    optional_vars = [
        'OPENAI_API_KEY',
        'CLAUDE_API_KEY',
        'DISCORD_WEBHOOK_URL',
        'TELEGRAM_BOT_TOKEN',
        'NEWS_API_KEY'
    ]
    
    missing_required = []
    available_optional = []
    
    # Check required variables
    for var in required_vars:
        value = os.getenv(var)
        if not value or value.startswith('your_'):
            missing_required.append(var)
        else:
            # Mask sensitive data
            if 'KEY' in var or 'SECRET' in var:
                masked = f"{value[:8]}...{value[-8:]}" if len(value) > 16 else "***"
                print(f"   ✅ {var}: {masked}")
            else:
                print(f"   ✅ {var}: {value}")
    
    # Check optional variables
    for var in optional_vars:
        value = os.getenv(var)
        if value and not value.startswith('your_'):
            available_optional.append(var)
            print(f"   🔧 {var}: Configured")
    
    # Report status
    if missing_required:
        print(f"\n❌ Missing required variables:")
        for var in missing_required:
            print(f"   - {var}")
        return False
    
    print(f"\n✅ Required configuration complete!")
    
    if available_optional:
        print(f"🔧 Optional features enabled:")
        for var in available_optional:
            feature = {
                'OPENAI_API_KEY': 'GPT Analysis',
                'CLAUDE_API_KEY': 'Claude Analysis', 
                'DISCORD_WEBHOOK_URL': 'Discord Notifications',
                'TELEGRAM_BOT_TOKEN': 'Telegram Alerts',
                'NEWS_API_KEY': 'News Sentiment Analysis'
            }.get(var, var)
            print(f"   • {feature}")
    
    return True

def load_trading_config():
    """Load trading configuration"""
    config_file = Path('trading_config.json')
    
    if config_file.exists():
        try:
            with open(config_file) as f:
                config = json.load(f)
            print(f"✅ Loaded configuration from {config_file}")
            return config
        except Exception as e:
            print(f"⚠️  Error loading config: {e}")
    
    # Default configuration
    default_config = {
        'trading_strategy': {
            'name': 'AI Enhanced Confluence Strategy',
            'version': '2.0'
        },
        'symbols': ['BTC/USDT', 'ETH/USDT'],
        'timeframes': ['4h', '1h'],
        'max_positions': 2,
        'risk_per_trade': 0.01,
        'max_leverage': 3,
        'min_confluences': 3,
        'min_signal_strength': 60
    }
    
    print("📋 Using default configuration")
    return default_config

async def run_ai_analysis_demo():
    """Run AI analysis demonstration"""
    print("\n🧪 AI Analysis Demo Mode")
    print("=" * 35)
    
    if not MODULES_AVAILABLE:
        print("❌ Required modules not available")
        return
    
    # Initialize analyzer
    analyzer = AdvancedMarketAnalyzer()
    
    # Demo analysis
    from advanced_ai_trading_strategy import demo_ai_analysis
    await demo_ai_analysis()

async def run_trading_bot(args):
    """Run the main trading bot"""
    print(f"\n🚀 Starting AI Trading Bot")
    print("=" * 35)
    
    if not MODULES_AVAILABLE:
        print("❌ Required modules not available")
        return
    
    # Load configuration
    config = load_trading_config()
    
    # Override with command line arguments
    if args.capital:
        config['initial_capital'] = args.capital
    if args.risk:
        config['risk_per_trade'] = args.risk
    if args.leverage:
        config['max_leverage'] = args.leverage
    if args.symbols:
        config['symbols'] = args.symbols.split(',')
    
    # Safety checks
    testnet = os.getenv('BINANCE_TESTNET', 'true').lower() == 'true'
    
    if not testnet and not args.live:
        print("🚨 LIVE TRADING MODE DETECTED!")
        print("Use --live flag to confirm live trading")
        return
    
    if not testnet:
        print("\n🚨 LIVE TRADING CONFIRMATION REQUIRED")
        print("This will trade with REAL MONEY!")
        print(f"Capital: ${config.get('initial_capital', 1000):,.2f}")
        print(f"Risk per trade: {config.get('risk_per_trade', 0.01):.1%}")
        print(f"Max leverage: {config.get('max_leverage', 3)}x")
        
        confirm = input("\nType 'I ACCEPT THE RISKS' to proceed: ")
        if confirm != 'I ACCEPT THE RISKS':
            print("👋 Trading cancelled for safety")
            return
    
    # Initialize systems
    print("\n🔧 Initializing Trading Systems...")
    
    # Initialize trade journal
    journal = TradeJournal()
    print("✅ Trade journal ready")
    
    # Initialize notifications
    notifications = NotificationManager()
    print("✅ Notification system ready")
    
    # Initialize trading bot
    bot = AdvancedAITradingBot(config)
    
    if not bot.initialize_exchange():
        print("❌ Failed to initialize exchange connection")
        return
    
    print("✅ AI Trading Bot initialized")
    
    # Show configuration summary
    print(f"\n📋 Trading Configuration:")
    print(f"   Mode: {'🧪 TESTNET' if testnet else '🔴 LIVE TRADING'}")
    print(f"   Symbols: {', '.join(config.get('symbols', []))}")
    print(f"   Capital: ${config.get('initial_capital', 1000):,.2f}")
    print(f"   Risk/Trade: {config.get('risk_per_trade', 0.01):.1%}")
    print(f"   Max Leverage: {config.get('max_leverage', 3)}x")
    print(f"   AI Analysis: {'✅ Active' if (os.getenv('OPENAI_API_KEY') or os.getenv('CLAUDE_API_KEY')) else '⚠️ Fallback mode'}")
    
    # Run trading session
    duration = args.duration or 2
    print(f"\n🎯 Starting {duration}-hour trading session...")
    
    try:
        # Send startup notification
        if notifications.telegram.enabled or notifications.discord.enabled:
            await notifications.send_risk_alert(
                "Trading Session Started",
                f"AI Trading Bot started in {'TESTNET' if testnet else 'LIVE'} mode\n"
                f"Duration: {duration} hours\n"
                f"Symbols: {', '.join(config.get('symbols', []))}"
            )
        
        # Run bot
        await bot.run_ai_trading_session(duration)
        
        # Get daily stats and send summary
        today = datetime.utcnow().strftime('%Y-%m-%d')
        daily_stats = journal.get_daily_stats(today)
        
        if daily_stats and daily_stats['total_trades'] > 0:
            journal.print_daily_report(today)
            
            if notifications.telegram.enabled:
                await notifications.send_daily_summary(daily_stats)
        
    except KeyboardInterrupt:
        print("\n⏹️ Trading session stopped by user")
        
        if notifications.telegram.enabled or notifications.discord.enabled:
            await notifications.send_risk_alert(
                "Trading Session Stopped",
                "AI Trading Bot session manually stopped by user"
            )
    
    print("\n✅ Trading session completed!")

def run_system_check():
    """Run comprehensive system check"""
    print("\n🔍 System Check Mode")
    print("=" * 25)
    
    # Check environment
    env_ok = check_environment()
    
    # Check modules
    modules_ok = MODULES_AVAILABLE
    print(f"\n📦 Module Check:")
    print(f"   Core modules: {'✅ Available' if modules_ok else '❌ Missing'}")
    
    # Check directories
    directories = ['data', 'logs', 'exports']
    print(f"\n📁 Directory Check:")
    for directory in directories:
        path = Path(directory)
        if not path.exists():
            path.mkdir(exist_ok=True)
            print(f"   {directory}/: ✅ Created")
        else:
            print(f"   {directory}/: ✅ Exists")
    
    # Check configuration
    config_file = Path('trading_config.json')
    print(f"\n⚙️  Configuration Check:")
    print(f"   trading_config.json: {'✅ Found' if config_file.exists() else '⚠️ Using defaults'}")
    
    # Overall status
    overall_ok = env_ok and modules_ok
    
    print(f"\n📋 System Status:")
    print(f"   Environment: {'✅ Ready' if env_ok else '❌ Needs setup'}")
    print(f"   Modules: {'✅ Ready' if modules_ok else '❌ Import errors'}")
    print(f"   Overall: {'✅ READY FOR TRADING' if overall_ok else '❌ SETUP REQUIRED'}")
    
    if not overall_ok:
        print(f"\n🔧 Setup Steps:")
        if not env_ok:
            print("   1. Copy .env.template to .env")
            print("   2. Fill in your API keys")
            print("   3. Set BINANCE_TESTNET=true")
        if not modules_ok:
            print("   4. Install required dependencies")
            print("   5. Check file locations")
    
    return overall_ok

def main():
    """Main entry point"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='AI-Enhanced Futures Trading Bot')
    
    # Mode selection
    parser.add_argument('--mode', choices=['trade', 'demo', 'check'], default='trade',
                       help='Operating mode')
    
    # Trading parameters
    parser.add_argument('--capital', type=float,
                       help='Initial capital amount')
    parser.add_argument('--risk', type=float,
                       help='Risk per trade (0.01 = 1%%)')
    parser.add_argument('--leverage', type=int,
                       help='Maximum leverage')
    parser.add_argument('--symbols', type=str,
                       help='Comma-separated symbols (BTC/USDT,ETH/USDT)')
    parser.add_argument('--duration', type=int,
                       help='Trading duration in hours')
    
    # Safety flags
    parser.add_argument('--live', action='store_true',
                       help='Enable live trading (REAL MONEY!)')
    parser.add_argument('--testnet-only', action='store_true',
                       help='Force testnet mode only')
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Force testnet if requested
    if args.testnet_only:
        os.environ['BINANCE_TESTNET'] = 'true'
        print("🧪 Forced testnet mode - safe for testing")
    
    try:
        if args.mode == 'check':
            # System check mode
            run_system_check()
            
        elif args.mode == 'demo':
            # Demo mode
            asyncio.run(run_ai_analysis_demo())
            
        elif args.mode == 'trade':
            # Trading mode
            if not check_environment():
                print("\n❌ Environment check failed!")
                print("Run with --mode check for details")
                sys.exit(1)
            
            asyncio.run(run_trading_bot(args))
        
    except KeyboardInterrupt:
        print("\n👋 Session interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()