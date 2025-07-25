#!/usr/bin/env python3
"""
Binance API Configuration Checker

This script checks your Binance API configuration and ensures it follows
the official Binance API guidelines and best practices.
"""

import asyncio
import ccxt
import hmac
import hashlib
import time
import requests
from datetime import datetime
from pathlib import Path
import os

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

class BinanceAPIChecker:
    """Comprehensive Binance API configuration checker"""
    
    def __init__(self):
        self.api_key = os.getenv('BINANCE_API_KEY', '')
        self.secret = os.getenv('BINANCE_SECRET', '')
        self.testnet = os.getenv('BINANCE_TESTNET', 'true').lower() == 'true'
        
        # Binance API endpoints
        self.base_urls = {
            'spot_live': 'https://api.binance.com',
            'spot_testnet': 'https://testnet.binance.vision',
            'futures_live': 'https://fapi.binance.com',
            'futures_testnet': 'https://testnet.binancefuture.com'
        }
        
        print("🔍 Binance API Configuration Checker")
        print("=" * 50)
        
    def check_api_key_format(self):
        """Check API key and secret format"""
        print("\n📋 API Key Format Check:")
        print("-" * 25)
        
        issues = []
        
        # Check API key
        if not self.api_key or self.api_key == 'your_binance_api_key_here':
            issues.append("❌ API key not set or using placeholder")
        elif len(self.api_key) != 64:
            issues.append(f"⚠️  API key length unusual: {len(self.api_key)} chars (expected: 64)")
        else:
            print(f"✅ API Key: {self.api_key[:8]}...{self.api_key[-8:]} (64 chars)")
        
        # Check secret
        if not self.secret or self.secret == 'your_binance_secret_here':
            issues.append("❌ Secret key not set or using placeholder")
        elif len(self.secret) != 64:
            issues.append(f"⚠️  Secret key length unusual: {len(self.secret)} chars (expected: 64)")
        else:
            print(f"✅ Secret Key: {self.secret[:8]}...{self.secret[-8:]} (64 chars)")
        
        # Check testnet setting
        print(f"🧪 Testnet Mode: {'✅ ENABLED' if self.testnet else '🚨 DISABLED (LIVE TRADING)'}")
        
        if issues:
            print("\n🚨 Configuration Issues:")
            for issue in issues:
                print(f"   {issue}")
            return False
        
        return True
    
    def create_signature(self, query_string, secret):
        """Create HMAC SHA256 signature for Binance API"""
        return hmac.new(
            secret.encode('utf-8'), 
            query_string.encode('utf-8'), 
            hashlib.sha256
        ).hexdigest()
    
    async def test_spot_connection(self):
        """Test spot trading API connection"""
        print("\n💱 Spot Trading API Test:")
        print("-" * 25)
        
        try:
            # Configure exchange
            exchange_config = {
                'apiKey': self.api_key,
                'secret': self.secret,
                'sandbox': self.testnet,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot'  # Explicitly set to spot
                }
            }
            
            exchange = ccxt.binance(exchange_config)
            
            # Test 1: Load markets
            print("📊 Loading markets...")
            markets = exchange.load_markets()
            print(f"✅ Markets loaded: {len(markets)} trading pairs")
            
            # Test 2: Get server time
            print("🕒 Checking server time...")
            server_time = exchange.fetch_time()
            local_time = int(time.time() * 1000)
            time_diff = abs(server_time - local_time)
            
            print(f"   Server time: {datetime.fromtimestamp(server_time/1000)}")
            print(f"   Local time:  {datetime.fromtimestamp(local_time/1000)}")
            print(f"   Time diff:   {time_diff}ms")
            
            if time_diff > 1000:
                print("⚠️  Time difference > 1000ms - may cause API issues")
            else:
                print("✅ Time synchronization OK")
            
            # Test 3: Get ticker (public endpoint)
            print("📈 Testing public endpoint...")
            ticker = exchange.fetch_ticker('BTC/USDT')
            print(f"✅ BTC/USDT: ${ticker['last']:,.2f} ({ticker['percentage']:+.2f}%)")
            
            # Test 4: Check account access (private endpoint)
            if self.api_key and self.secret:
                print("🔐 Testing private endpoint...")
                try:
                    balance = exchange.fetch_balance()
                    print("✅ Account access successful")
                    
                    # Show some balance info (safe for testnet)
                    if self.testnet:
                        total_usdt = balance.get('USDT', {}).get('total', 0)
                        total_btc = balance.get('BTC', {}).get('total', 0)
                        print(f"   USDT Balance: {total_usdt:,.2f}")
                        print(f"   BTC Balance: {total_btc:.6f}")
                    else:
                        print("   (Balance details hidden for live account)")
                        
                except Exception as e:
                    print(f"❌ Account access failed: {e}")
                    return False
            
            return True
            
        except Exception as e:
            print(f"❌ Spot API connection failed: {e}")
            return False
    
    async def test_futures_connection(self):
        """Test futures trading API connection"""
        print("\n🚀 Futures Trading API Test:")
        print("-" * 27)
        
        try:
            # Configure for futures
            exchange_config = {
                'apiKey': self.api_key,
                'secret': self.secret,
                'sandbox': self.testnet,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future'  # Set to futures
                }
            }
            
            exchange = ccxt.binance(exchange_config)
            
            # Test futures markets
            print("📊 Loading futures markets...")
            markets = exchange.load_markets()
            futures_markets = [m for m in markets.values() if m['type'] == 'future']
            print(f"✅ Futures markets: {len(futures_markets)} contracts")
            
            # Test futures ticker
            print("📈 Testing futures ticker...")
            ticker = exchange.fetch_ticker('BTC/USDT')
            print(f"✅ BTC/USDT Futures: ${ticker['last']:,.2f}")
            
            # Test futures account (if API keys provided)
            if self.api_key and self.secret:
                print("🔐 Testing futures account access...")
                try:
                    # For futures, we need to check positions instead of balance
                    positions = exchange.fetch_positions()
                    print("✅ Futures account access successful")
                    
                    if self.testnet:
                        open_positions = [p for p in positions if p['contracts'] > 0]
                        print(f"   Open positions: {len(open_positions)}")
                    
                except Exception as e:
                    if "futures" in str(e).lower() or "margin" in str(e).lower():
                        print("⚠️  Futures not enabled on this account")
                        print("   Enable futures trading in Binance account settings")
                        return False
                    else:
                        print(f"❌ Futures account access failed: {e}")
                        return False
            
            return True
            
        except Exception as e:
            print(f"❌ Futures API connection failed: {e}")
            return False
    
    def check_api_permissions(self):
        """Check API key permissions"""
        print("\n🔑 API Permissions Check:")
        print("-" * 24)
        
        if not self.api_key or not self.secret:
            print("❌ Cannot check permissions - API keys not configured")
            return False
        
        try:
            # Use requests to check account info
            base_url = self.base_urls['spot_testnet'] if self.testnet else self.base_urls['spot_live']
            endpoint = '/api/v3/account'
            
            timestamp = int(time.time() * 1000)
            query_string = f'timestamp={timestamp}'
            signature = self.create_signature(query_string, self.secret)
            
            headers = {
                'X-MBX-APIKEY': self.api_key
            }
            
            params = {
                'timestamp': timestamp,
                'signature': signature
            }
            
            response = requests.get(f"{base_url}{endpoint}", headers=headers, params=params)
            
            if response.status_code == 200:
                account_info = response.json()
                print("✅ Account access successful")
                
                # Check account status
                account_type = account_info.get('accountType', 'UNKNOWN')
                can_trade = account_info.get('canTrade', False)
                can_withdraw = account_info.get('canWithdraw', False)
                can_deposit = account_info.get('canDeposit', False)
                
                print(f"   Account Type: {account_type}")
                print(f"   Can Trade: {'✅' if can_trade else '❌'}")
                print(f"   Can Withdraw: {'✅' if can_withdraw else '❌'}")
                print(f"   Can Deposit: {'✅' if can_deposit else '❌'}")
                
                if not can_trade:
                    print("⚠️  Trading is not enabled on this API key")
                    print("   Enable 'Spot & Margin Trading' in API settings")
                
                return can_trade
                
            else:
                print(f"❌ Permission check failed: {response.status_code}")
                print(f"   Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Permission check error: {e}")
            return False
    
    def check_rate_limits(self):
        """Check and explain rate limits"""
        print("\n⚡ Rate Limits Information:")
        print("-" * 26)
        
        print("📊 Binance Rate Limits:")
        print("   Spot Trading:")
        print("   • Weight limit: 1,200 per minute")
        print("   • Order limit: 10 per second, 100,000 per 24h")
        print("   • Raw requests: 6,000 per 5 minutes")
        print("")
        print("   Futures Trading:")
        print("   • Weight limit: 2,400 per minute")
        print("   • Order limit: 300 per 10 seconds")
        print("   • Position limit: 200 per minute")
        print("")
        print("✅ CCXT automatically handles rate limiting")
        print("💡 Use 'enableRateLimit': True in exchange config")
    
    def generate_optimal_config(self):
        """Generate optimal configuration"""
        print("\n⚙️  Optimal Configuration:")
        print("-" * 25)
        
        config_spot = {
            'apiKey': 'your_api_key',
            'secret': 'your_secret',
            'sandbox': True,  # Start with testnet
            'enableRateLimit': True,
            'timeout': 30000,
            'options': {
                'defaultType': 'spot',
                'adjustForTimeDifference': True,
            }
        }
        
        config_futures = {
            'apiKey': 'your_api_key',
            'secret': 'your_secret',
            'sandbox': True,
            'enableRateLimit': True,
            'timeout': 30000,
            'options': {
                'defaultType': 'future',
                'adjustForTimeDifference': True,
            }
        }
        
        print("📋 Spot Trading Config:")
        print("```python")
        print("exchange = ccxt.binance({")
        for key, value in config_spot.items():
            if key in ['apiKey', 'secret']:
                print(f"    '{key}': os.getenv('BINANCE_{key.upper()}'),")
            else:
                print(f"    '{key}': {repr(value)},")
        print("})")
        print("```")
        
        print("\n📋 Futures Trading Config:")
        print("```python")
        print("exchange = ccxt.binance({")
        for key, value in config_futures.items():
            if key in ['apiKey', 'secret']:
                print(f"    '{key}': os.getenv('BINANCE_{key.upper()}'),")
            else:
                print(f"    '{key}': {repr(value)},")
        print("})")
        print("```")
    
    def check_security_recommendations(self):
        """Check security recommendations"""
        print("\n🔒 Security Recommendations:")
        print("-" * 28)
        
        recommendations = [
            "✅ Use API keys with minimal required permissions",
            "✅ Enable IP whitelist restriction in Binance",
            "✅ Never share API keys or commit them to version control",
            "✅ Use environment variables for API credentials",
            "✅ Regularly rotate API keys",
            "✅ Enable 2FA on your Binance account",
            "✅ Start with testnet before live trading",
            "✅ Use separate API keys for different applications",
            "✅ Monitor API key usage in Binance dashboard",
            "✅ Set withdrawal restrictions on API keys"
        ]
        
        for rec in recommendations:
            print(f"   {rec}")
        
        print(f"\n⚠️  Current Setup Security Level:")
        if self.testnet:
            print("   🟢 SAFE - Using testnet (no real money risk)")
        else:
            print("   🔴 LIVE - Real money trading enabled!")
            print("   Double check all settings before trading")
    
    async def run_complete_check(self):
        """Run complete API configuration check"""
        print("🚀 Starting Complete Binance API Check...")
        
        # Step 1: Format check
        format_ok = self.check_api_key_format()
        
        # Step 2: Permissions check
        permissions_ok = self.check_permissions_check() if format_ok else False
        
        # Step 3: Connection tests
        spot_ok = await self.test_spot_connection() if format_ok else False
        futures_ok = await self.test_futures_connection() if format_ok else False
        
        # Step 4: Additional info
        self.check_rate_limits()
        self.generate_optimal_config()
        self.check_security_recommendations()
        
        # Summary
        print(f"\n📋 API Configuration Summary:")
        print("=" * 35)
        print(f"API Key Format:      {'✅' if format_ok else '❌'}")
        print(f"Permissions:         {'✅' if permissions_ok else '❌'}")
        print(f"Spot Trading:        {'✅' if spot_ok else '❌'}")
        print(f"Futures Trading:     {'✅' if futures_ok else '❌'}")
        print(f"Security Level:      {'🟢 SAFE' if self.testnet else '🔴 LIVE'}")
        
        overall_status = format_ok and permissions_ok and spot_ok
        
        if overall_status:
            print(f"\n🎉 API Configuration: ✅ READY FOR TRADING")
            if self.testnet:
                print("   Start with paper trading to test your strategies")
            else:
                print("   ⚠️  LIVE TRADING - Use small amounts initially")
        else:
            print(f"\n⚠️  API Configuration: ❌ NEEDS ATTENTION")
            print("   Fix the issues above before trading")
        
        return overall_status

async def main():
    """Main function"""
    checker = BinanceAPIChecker()
    
    try:
        result = await checker.run_complete_check()
        
        print(f"\n🎯 Next Steps:")
        if result:
            print("1. 🧪 Test strategies in paper trading mode")
            print("2. 📊 Run live dashboard: python live_dashboard.py")
            print("3. 🤖 Start bot: python run_bot.py --testnet")
            print("4. 📈 Monitor performance and adjust parameters")
        else:
            print("1. 🔧 Fix API configuration issues above")
            print("2. ✅ Re-run this checker: python binance_api_checker.py")
            print("3. 📖 Review Binance API documentation")
            print("4. 🆘 Check .env file and API key permissions")
        
    except KeyboardInterrupt:
        print("\n👋 API check interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error during API check: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())