#!/usr/bin/env python3
"""
Dependency Installation Helper

This script helps install the right dependencies based on user needs
and system requirements.
"""

import subprocess
import sys
import os
import platform
from pathlib import Path

def run_command(command):
    """Run a command and return success status"""
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not compatible. Requires Python 3.8+")
        return False

def install_core_dependencies():
    """Install core dependencies"""
    print("📦 Installing core dependencies...")
    
    success, output = run_command("pip install -r requirements.txt")
    if success:
        print("✅ Core dependencies installed successfully")
        return True
    else:
        print(f"❌ Failed to install core dependencies: {output}")
        return False

def install_optional_dependencies(categories=None):
    """Install optional dependencies by category"""
    if not categories:
        categories = []
    
    print(f"📦 Installing optional dependencies: {', '.join(categories)}")
    
    # Read optional requirements
    try:
        with open("requirements-optional.txt", "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print("❌ requirements-optional.txt not found")
        return False
    
    # Category mappings
    category_packages = {
        "database": ["aiosqlite", "asyncpg", "aiomysql", "pymongo"],
        "ai": ["anthropic", "openai"],
        "notifications": ["discord-webhook", "python-telegram-bot", "slack-sdk", "aiosmtplib"],
        "visualization": ["matplotlib", "plotly", "streamlit"],
        "ml": ["scikit-learn", "tensorflow", "torch", "xgboost"],
        "cloud": ["boto3", "google-cloud-storage", "azure-storage-blob"],
        "monitoring": ["prometheus-client", "jaeger-client", "opentelemetry-api"],
        "web": ["fastapi", "uvicorn"],
        "queues": ["celery", "redis"],
        "development": ["jupyter", "jupyterlab"]
    }
    
    packages_to_install = []
    
    for category in categories:
        if category in category_packages:
            for package in category_packages[category]:
                # Find the package line in requirements-optional.txt
                for line in lines:
                    if line.strip().startswith(package) and not line.strip().startswith("#"):
                        packages_to_install.append(line.strip())
                        break
    
    if packages_to_install:
        for package in packages_to_install:
            print(f"  Installing {package}...")
            success, output = run_command(f"pip install '{package}'")
            if success:
                print(f"  ✅ {package} installed")
            else:
                print(f"  ⚠️ Failed to install {package}: {output}")
    
    return True

def detect_database_needs():
    """Detect what database drivers might be needed"""
    recommendations = []
    
    # Check for existing database files
    if Path("trading_bot.db").exists() or Path("*.db").exists():
        recommendations.append("aiosqlite>=0.19.0")
        print("📁 SQLite database detected - recommending aiosqlite")
    
    # Check environment variables
    db_url = os.getenv("DATABASE_URL", "")
    if "postgresql" in db_url.lower():
        recommendations.append("asyncpg>=0.28.0")
        print("🐘 PostgreSQL detected in DATABASE_URL - recommending asyncpg")
    elif "mysql" in db_url.lower():
        recommendations.append("aiomysql>=0.2.0")
        print("🐬 MySQL detected in DATABASE_URL - recommending aiomysql")
    
    return recommendations

def interactive_installation():
    """Interactive installation process"""
    print("🤖 Cryptocurrency Trading Bot - Dependency Installation")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install core dependencies
    print("\n1. Installing core dependencies...")
    if not install_core_dependencies():
        return False
    
    # Detect database needs
    print("\n2. Detecting database requirements...")
    db_recommendations = detect_database_needs()
    
    if db_recommendations:
        print("📋 Recommended database drivers:")
        for rec in db_recommendations:
            print(f"  - {rec}")
        
        install_db = input("\nInstall recommended database drivers? (y/n): ").lower().startswith('y')
        if install_db:
            for package in db_recommendations:
                success, output = run_command(f"pip install '{package}'")
                if success:
                    print(f"✅ {package} installed")
                else:
                    print(f"⚠️ Failed to install {package}")
    
    # Ask about optional features
    print("\n3. Optional features:")
    print("Available categories:")
    print("  - ai: AI analysis (Claude, OpenAI)")
    print("  - notifications: Discord, Telegram, Slack")
    print("  - visualization: Charts and dashboards")
    print("  - ml: Machine learning models")
    print("  - cloud: AWS, GCP, Azure integration")
    print("  - monitoring: Prometheus, Jaeger")
    print("  - development: Jupyter notebooks")
    
    categories_input = input("\nWhich optional categories would you like? (comma-separated, or 'none'): ")
    
    if categories_input.lower() != 'none':
        categories = [cat.strip() for cat in categories_input.split(',') if cat.strip()]
        if categories:
            install_optional_dependencies(categories)
    
    print("\n🎉 Installation completed!")
    print("\nNext steps:")
    print("1. Configure your bot: python config/settings.py")
    print("2. Test installation: python run_bot.py --check-only")
    print("3. Run the bot: python run_bot.py")
    
    return True

def install_minimal():
    """Install minimal dependencies for basic functionality"""
    print("📦 Installing minimal dependencies...")
    
    minimal_packages = [
        "pandas>=1.5.0",
        "numpy>=1.24.0", 
        "ccxt>=4.0.0",
        "aiohttp>=3.8.0",
        "sqlalchemy>=2.0.0",
        "pyyaml>=6.0",
        "aiosqlite>=0.19.0"  # Include SQLite for basic database support
    ]
    
    for package in minimal_packages:
        print(f"Installing {package}...")
        success, output = run_command(f"pip install '{package}'")
        if success:
            print(f"✅ {package} installed")
        else:
            print(f"❌ Failed to install {package}: {output}")
            return False
    
    print("✅ Minimal installation completed")
    return True

def main():
    """Main installation function"""
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "minimal":
            install_minimal()
        elif command == "core":
            install_core_dependencies()
        elif command == "optional":
            categories = sys.argv[2:] if len(sys.argv) > 2 else []
            install_optional_dependencies(categories)
        elif command == "help":
            print("Usage:")
            print("  python install_dependencies.py           # Interactive installation")
            print("  python install_dependencies.py minimal   # Install minimal dependencies")
            print("  python install_dependencies.py core      # Install core dependencies only")
            print("  python install_dependencies.py optional [categories]  # Install optional deps")
            print("  python install_dependencies.py help      # Show this help")
        else:
            print(f"Unknown command: {command}")
            print("Use 'python install_dependencies.py help' for usage information")
    else:
        interactive_installation()

if __name__ == "__main__":
    main()