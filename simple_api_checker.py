#!/usr/bin/env python3
"""
Simple Binance API Checker

This checker works without external dependencies like python-dotenv.
It will guide you through setting up and testing your Binance API.
"""

import asyncio
import ccxt
import os
from datetime import datetime

class SimpleBinanceChecker:
    """Simple Binance API checker"""
    
    def __init__(self):
        self.api_key = ''
        self.secret = ''
        self.testnet = True
        
        print("🔍 Simple Binance API Checker")
        print("=" * 35)
    
    def get_credentials_from_user(self):
        """Get API credentials from user input"""
        print("\n🔑 API Credentials Setup:")
        print("-" * 25)
        
        # Check environment variables first
        env_api_key = os.getenv('BINANCE_API_KEY', '')
        env_secret = os.getenv('BINANCE_SECRET', '')
        env_testnet = os.getenv('BINANCE_TESTNET', 'true').lower() == 'true'
        
        if env_api_key and env_secret and env_api_key != 'your_binance_api_key_here':
            print("✅ Found API credentials in environment variables")
            self.api_key = env_api_key
            self.secret = env_secret
            self.testnet = env_testnet
            
            print(f"   API Key: {self.api_key[:8]}...{self.api_key[-8:]}")
            print(f"   Testnet: {'YES' if self.testnet else 'NO'}")
            return True
        
        print("⚠️  No valid API credentials found in environment")
        print("\nPlease enter your Binance API credentials:")
        print("(You can get these from https://binance.com → API Management)")
        
        try:
            self.api_key = input("\n📋 Enter API Key: ").strip()
            self.secret = input("🔐 Enter Secret Key: ").strip()
            
            testnet_choice = input("🧪 Use Testnet? (Y/n): ").strip().lower()
            self.testnet = testnet_choice != 'n'
            
            if not self.api_key or not self.secret:
                print("❌ API credentials cannot be empty")
                return False
            
            if len(self.api_key) != 64 or len(self.secret) != 64:
                print("⚠️  API key/secret length unusual (expected 64 characters)")
                confirm = input("Continue anyway? (y/N): ").strip().lower()
                if confirm != 'y':
                    return False
            
            return True
            
        except KeyboardInterrupt:
            print("\n👋 Setup cancelled by user")
            return False
    
    async def test_connection(self):
        """Test API connection"""
        print(f"\n📡 Testing API Connection:")
        print("-" * 25)
        
        try:
            # Configure exchange
            exchange = ccxt.binance({
                'apiKey': self.api_key,
                'secret': self.secret,
                'sandbox': self.testnet,
                'enableRateLimit': True,
            })
            
            # Test 1: Load markets (public)
            print("📊 Loading markets...")
            markets = exchange.load_markets()
            print(f"✅ Markets loaded: {len(markets)} trading pairs")
            
            # Test 2: Get ticker (public)
            print("📈 Testing market data...")
            ticker = exchange.fetch_ticker('BTC/USDT')
            print(f"✅ BTC/USDT: ${ticker['last']:,.2f} ({ticker['percentage']:+.2f}%)")
            
            # Test 3: Account access (private)
            print("🔐 Testing account access...")
            balance = exchange.fetch_balance()
            print("✅ Account access successful")
            
            # Show balance info (safe for testnet)
            if self.testnet:
                usdt_balance = balance.get('USDT', {}).get('total', 0)
                btc_balance = balance.get('BTC', {}).get('total', 0)
                print(f"   USDT Balance: {usdt_balance:,.2f}")
                print(f"   BTC Balance: {btc_balance:.6f}")
            else:
                print("   (Balance details hidden for live account)")
            
            return True
            
        except ccxt.AuthenticationError as e:
            print(f"❌ Authentication failed: Invalid API credentials")
            print("   Please check your API key and secret")
            return False
            
        except ccxt.PermissionDenied as e:
            print(f"❌ Permission denied: {e}")
            print("   Please enable required permissions in Binance API settings")
            return False
            
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    async def test_futures_access(self):
        """Test futures trading access"""
        print(f"\n🚀 Testing Futures Access:")
        print("-" * 25)
        
        try:
            # Configure for futures
            exchange = ccxt.binance({
                'apiKey': self.api_key,
                'secret': self.secret,
                'sandbox': self.testnet,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future'
                }
            })
            
            # Test futures markets
            print("📊 Loading futures markets...")
            markets = exchange.load_markets()
            futures_markets = [m for m in markets.values() if m['type'] == 'future']
            print(f"✅ Futures markets: {len(futures_markets)} contracts")
            
            # Test futures ticker
            print("📈 Testing futures data...")
            ticker = exchange.fetch_ticker('BTC/USDT')
            print(f"✅ BTC/USDT Futures: ${ticker['last']:,.2f}")
            
            # Test futures account
            print("🔐 Testing futures account...")
            positions = exchange.fetch_positions()
            print("✅ Futures account access successful")
            
            if self.testnet:
                open_positions = [p for p in positions if p['contracts'] and p['contracts'] > 0]
                print(f"   Open positions: {len(open_positions)}")
            
            return True
            
        except Exception as e:
            if "futures" in str(e).lower() or "margin" in str(e).lower():
                print("⚠️  Futures trading not enabled")
                print("   Enable 'Futures' permission in Binance API settings")
            else:
                print(f"❌ Futures test failed: {e}")
            return False
    
    def show_recommendations(self, spot_ok: bool, futures_ok: bool):
        """Show recommendations based on test results"""
        print(f"\n💡 Recommendations:")
        print("-" * 18)
        
        if spot_ok and futures_ok:
            print("✅ Your API is fully configured for trading!")
            print("\n🎯 Next Steps:")
            print("1. 🧪 Start with testnet trading:")
            print("   python futures_trading_bot.py")
            print("2. 📊 Monitor with dashboard:")
            print("   python live_dashboard.py")
            print("3. 📈 When ready, switch to live trading")
            
        elif spot_ok and not futures_ok:
            print("✅ Spot trading ready, but futures needs setup")
            print("\n🔧 To enable futures:")
            print("1. Go to Binance.com → API Management")
            print("2. Enable 'Futures' permission")
            print("3. Re-run this checker")
            
        else:
            print("❌ API needs attention before trading")
            print("\n🔧 Setup steps:")
            print("1. Go to https://binance.com → API Management")
            print("2. Create new API key with these permissions:")
            print("   ✅ Enable Reading")
            print("   ✅ Enable Spot & Margin Trading")
            print("   ✅ Enable Futures")
            print("   ✅ Permits Universal Transfer")
            print("3. Set IP restrictions (recommended)")
            print("4. Re-run this checker")
    
    def save_to_env_file(self):
        """Save credentials to .env file"""
        try:
            env_content = f"""# Binance API Configuration
BINANCE_API_KEY={self.api_key}
BINANCE_SECRET={self.secret}
BINANCE_TESTNET={'true' if self.testnet else 'false'}

# Trading Settings
TRADING_INITIAL_CAPITAL=10000.0
TRADING_MAX_POSITIONS=3
TRADING_RISK_PER_TRADE=0.02
TRADING_LEVERAGE=1
TRADING_MARGIN_TYPE=ISOLATED

# Symbols
TRADING_SYMBOLS=BTC/USDT,ETH/USDT
"""
            
            with open('.env', 'w') as f:
                f.write(env_content)
            
            print(f"\n💾 Credentials saved to .env file")
            print("   Other scripts will now automatically use these settings")
            return True
            
        except Exception as e:
            print(f"⚠️  Could not save to .env file: {e}")
            return False
    
    async def run_complete_check(self):
        """Run complete API check"""
        # Get credentials
        if not self.get_credentials_from_user():
            return False
        
        # Test connections
        spot_ok = await self.test_connection()
        futures_ok = await self.test_futures_access() if spot_ok else False
        
        # Show results
        print(f"\n📋 API Test Summary:")
        print("=" * 25)
        print(f"API Connection:      {'✅' if spot_ok else '❌'}")
        print(f"Spot Trading:        {'✅' if spot_ok else '❌'}")
        print(f"Futures Trading:     {'✅' if futures_ok else '❌'}")
        print(f"Mode:                {'🧪 TESTNET' if self.testnet else '🔴 LIVE'}")
        
        # Save configuration
        if spot_ok:
            save_choice = input(f"\n💾 Save configuration to .env file? (Y/n): ").strip().lower()
            if save_choice != 'n':
                self.save_to_env_file()
        
        # Show recommendations
        self.show_recommendations(spot_ok, futures_ok)
        
        return spot_ok

async def main():
    """Main function"""
    checker = SimpleBinanceChecker()
    
    try:
        result = await checker.run_complete_check()
        
        if result:
            print(f"\n🎉 API setup completed successfully!")
        else:
            print(f"\n⚠️  API setup needs attention")
            
    except KeyboardInterrupt:
        print(f"\n👋 API checker cancelled by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    asyncio.run(main())