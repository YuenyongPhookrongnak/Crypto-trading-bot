#!/usr/bin/env python3
"""
Manual Environment Variable Loader

This script loads environment variables from .env file manually
without requiring python-dotenv package.
"""

import os
from pathlib import Path

def load_env_manual():
    """Load environment variables from .env file manually"""
    env_file = Path('.env')
    
    if not env_file.exists():
        print("⚠️  .env file not found - using system environment variables")
        return False
    
    try:
        with open(env_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        loaded_count = 0
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Parse KEY=VALUE
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Remove quotes if present
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]
                
                # Set environment variable
                os.environ[key] = value
                loaded_count += 1
        
        print(f"✅ Loaded {loaded_count} environment variables from .env")
        return True
        
    except Exception as e:
        print(f"❌ Error loading .env file: {e}")
        return False

def create_sample_env():
    """Create sample .env file"""
    env_content = """# Cryptocurrency Trading Bot Environment Variables

# Exchange API Keys
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_SECRET=your_binance_secret_here
BINANCE_TESTNET=true

# Database
DATABASE_URL=sqlite:///trading_bot.db

# Trading Settings
TRADING_INITIAL_CAPITAL=10000.0
TRADING_MAX_POSITIONS=3
TRADING_RISK_PER_TRADE=0.02
TRADING_SYMBOLS=BTC/USDT,ETH/USDT

# Futures Settings
TRADING_LEVERAGE=1
TRADING_MARGIN_TYPE=ISOLATED

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/trading_bot.log

# Optional: AI Services
CLAUDE_API_KEY=
OPENAI_API_KEY=

# Optional: Notifications
DISCORD_WEBHOOK_URL=
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
"""
    
    env_file = Path('.env')
    if not env_file.exists():
        with open(env_file, 'w', encoding='utf-8') as f:
            f.write(env_content)
        print("✅ Created .env file template")
        print("📝 Please edit .env with your actual API keys")
        return True
    else:
        print("✅ .env file already exists")
        return True

def check_env_variables():
    """Check if required environment variables are set"""
    required_vars = [
        'BINANCE_API_KEY',
        'BINANCE_SECRET',
        'BINANCE_TESTNET'
    ]
    
    missing_vars = []
    
    for var in required_vars:
        value = os.getenv(var)
        if not value or value == f'your_{var.lower()}_here':
            missing_vars.append(var)
        else:
            # Mask sensitive data
            if 'KEY' in var or 'SECRET' in var:
                masked_value = f"{value[:8]}...{value[-8:]}" if len(value) > 16 else "***"
                print(f"✅ {var}: {masked_value}")
            else:
                print(f"✅ {var}: {value}")
    
    if missing_vars:
        print(f"\n❌ Missing environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        return False
    
    return True

def main():
    """Main function"""
    print("🔧 Manual Environment Variable Loader")
    print("=" * 40)
    
    # Try to load .env file
    if not load_env_manual():
        print("Creating sample .env file...")
        create_sample_env()
    
    # Check required variables
    print(f"\n🔍 Checking environment variables:")
    if check_env_variables():
        print(f"\n✅ All required environment variables are set!")
    else:
        print(f"\n⚠️  Please edit .env file with your actual values")
    
    print(f"\n💡 Usage in your scripts:")
    print("```python")
    print("import os")
    print("from manual_env_loader import load_env_manual")
    print("")
    print("# Load environment variables")
    print("load_env_manual()")
    print("")
    print("# Use environment variables")
    print("api_key = os.getenv('BINANCE_API_KEY')")
    print("```")

if __name__ == "__main__":
    main()