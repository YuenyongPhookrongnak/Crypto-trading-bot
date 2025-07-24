#!/usr/bin/env python3
"""
Dependency Checker

This script checks which dependencies are available and provides
recommendations for missing ones.
"""

import sys
import importlib
from typing import Dict, List, Tuple

def check_import(module_name: str, package_name: str = None) -> Tuple[bool, str]:
    """Check if a module can be imported"""
    try:
        importlib.import_module(module_name)
        return True, "✅ Available"
    except ImportError as e:
        package = package_name or module_name
        return False, f"❌ Missing (install with: pip install {package})"

def check_dependencies() -> Dict[str, Dict[str, Tuple[bool, str]]]:
    """Check all dependencies"""
    
    dependencies = {
        "Core Dependencies": {
            "pandas": check_import("pandas"),
            "numpy": check_import("numpy"),
            "ccxt": check_import("ccxt"),
            "aiohttp": check_import("aiohttp"),
            "sqlalchemy": check_import("sqlalchemy"),
            "yaml": check_import("yaml", "pyyaml"),
            "talib": check_import("talib", "talib"),
            "scipy": check_import("scipy"),
        },
        
        "Database Drivers": {
            "aiosqlite": check_import("aiosqlite"),
            "asyncpg": check_import("asyncpg"),
            "aiomysql": check_import("aiomysql"),
            "pymongo": check_import("pymongo"),
        },
        
        "AI Services": {
            "anthropic": check_import("anthropic"),
            "openai": check_import("openai"),
        },
        
        "Notification Services": {
            "discord-webhook": check_import("discord_webhook", "discord-webhook"),
            "telegram": check_import("telegram", "python-telegram-bot"),
            "slack": check_import("slack_sdk", "slack-sdk"),
            "aiosmtplib": check_import("aiosmtplib"),
        },
        
        "Visualization": {
            "matplotlib": check_import("matplotlib"),
            "plotly": check_import("plotly"),
            "streamlit": check_import("streamlit"),
        },
        
        "Machine Learning": {
            "sklearn": check_import("sklearn", "scikit-learn"),
            "tensorflow": check_import("tensorflow"),
            "torch": check_import("torch"),
            "xgboost": check_import("xgboost"),
        },
        
        "Development Tools": {
            "pytest": check_import("pytest"),
            "black": check_import("black"),
            "isort": check_import("isort"),
            "flake8": check_import("flake8"),
        }
    }
    
    return dependencies

def print_dependency_report():
    """Print comprehensive dependency report"""
    print("🔍 Cryptocurrency Trading Bot - Dependency Check")
    print("=" * 60)
    
    dependencies = check_dependencies()
    
    total_available = 0
    total_missing = 0
    critical_missing = []
    
    for category, deps in dependencies.items():
        print(f"\n📦 {category}:")
        print("-" * (len(category) + 4))
        
        for dep_name, (available, status) in deps.items():
            print(f"  {dep_name:20} {status}")
            
            if available:
                total_available += 1
            else:
                total_missing += 1
                
                # Mark critical dependencies
                if category == "Core Dependencies":
                    critical_missing.append(dep_name)
    
    # Summary
    print(f"\n📊 Summary:")
    print("-" * 12)
    print(f"Available: {total_available}")
    print(f"Missing: {total_missing}")
    total_deps = total_available + total_missing
    print(f"Coverage: {(total_available/total_deps)*100:.1f}%" if total_deps > 0 else "Coverage: 0%")
    
    # Critical missing dependencies
    if critical_missing:
        print(f"\n🚨 Critical Missing Dependencies:")
        print("-" * 35)
        for dep in critical_missing:
            print(f"  - {dep}")
        print(f"\nInstall critical dependencies with:")
        print(f"  pip install {' '.join(critical_missing)}")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    print("-" * 20)
    
    if not any(deps[0] for deps in dependencies["Database Drivers"].values()):
        print("  - Install at least one database driver (recommended: aiosqlite)")
        print("    pip install aiosqlite")
    
    if not dependencies["AI Services"]["anthropic"][0] and not dependencies["AI Services"]["openai"][0]:
        print("  - Install AI service for enhanced analysis (optional)")
        print("    pip install anthropic  # or openai")
    
    if not any(deps[0] for deps in dependencies["Notification Services"].values()):
        print("  - Install notification service for alerts (optional)")
        print("    pip install discord-webhook  # or python-telegram-bot")
    
    # Installation commands
    print(f"\n🚀 Quick Installation Commands:")
    print("-" * 32)
    print("  Minimal setup:")
    print("    python install_dependencies.py minimal")
    print("  Full interactive setup:")
    print("    python install_dependencies.py")
    print("  Install specific category:")
    print("    python install_dependencies.py optional database ai")

def check_python_compatibility():
    """Check Python version compatibility"""
    version = sys.version_info
    
    print("🐍 Python Version Check:")
    print("-" * 25)
    print(f"Current version: Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print("✅ Python version is compatible")
        return True
    else:
        print("❌ Python version is NOT compatible")
        print("   Requires: Python 3.8 or higher")
        print("   Please upgrade your Python version")
        return False

def check_system_requirements():
    """Check system requirements"""
    import platform
    import os
    
    print("\n💻 System Information:")
    print("-" * 24)
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Python path: {sys.executable}")
    
    # Check disk space (simplified)
    try:
        import shutil
        total, used, free = shutil.disk_usage(".")
        free_gb = free // (1024**3)
        print(f"Free disk space: {free_gb} GB")
        
        if free_gb < 1:
            print("⚠️  Warning: Low disk space (< 1GB)")
    except:
        print("Disk space: Unable to check")

def main():
    """Main function"""
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        # Quick check for critical dependencies only
        core_deps = check_dependencies()["Core Dependencies"]
        missing_core = [name for name, (available, _) in core_deps.items() if not available]
        
        if missing_core:
            print(f"❌ Missing critical dependencies: {', '.join(missing_core)}")
            print(f"Install with: pip install {' '.join(missing_core)}")
            sys.exit(1)
        else:
            print("✅ All critical dependencies available")
            sys.exit(0)
    else:
        # Full dependency check
        check_python_compatibility()
        check_system_requirements()
        print_dependency_report()

if __name__ == "__main__":
    main()