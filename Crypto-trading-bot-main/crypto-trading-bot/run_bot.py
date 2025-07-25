#!/usr/bin/env python3
"""
Cryptocurrency Trading Bot Launcher

This script provides a comprehensive launcher for the trading bot with:
- Configuration management
- Environment setup
- Error handling and recovery
- Logging configuration
- Bot lifecycle management
- Interactive controls
"""

import asyncio
import logging
import os
import sys
import json
import argparse
import signal
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import psutil
import yaml

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import bot components
from main_bot import TradingBot, BotConfiguration, BotState, create_trading_bot
from config.settings import config
from database.connection import init_database, close_database

# Configure logging
def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Setup comprehensive logging configuration"""
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Default log file with timestamp
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"trading_bot_{timestamp}.log"
    
    # Logging configuration
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Reduce noise from external libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("ccxt").setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized - Level: {log_level}, File: {log_file}")
    
    return logger

class ConfigurationManager:
    """Manages bot configuration from multiple sources"""
    
    def __init__(self, config_file: str = None):
        self.config_file = config_file or "config/bot_config.yaml"
        self.config_data = {}
        self.logger = logging.getLogger(__name__)
    
    def load_configuration(self) -> BotConfiguration:
        """Load configuration from file and environment variables"""
        try:
            # Load from YAML file
            config_path = Path(self.config_file)
            if config_path.exists():
                with open(config_path, 'r') as f:
                    self.config_data = yaml.safe_load(f) or {}
                self.logger.info(f"Configuration loaded from {config_path}")
            else:
                self.logger.warning(f"Configuration file not found: {config_path}")
                self.config_data = {}
            
            # Override with environment variables
            self._load_env_overrides()
            
            # Create bot configuration
            bot_config = self._create_bot_configuration()
            
            self.logger.info("Configuration loaded successfully")
            return bot_config
            
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            raise
    
    def _load_env_overrides(self):
        """Load configuration overrides from environment variables"""
        env_mappings = {
            'TRADING_INITIAL_CAPITAL': ('trading', 'initial_capital', float),
            'TRADING_MAX_POSITIONS': ('trading', 'max_open_positions', int),
            'TRADING_RISK_PER_TRADE': ('trading', 'max_risk_per_trade', float),
            'TRADING_DAILY_LOSS_LIMIT': ('trading', 'daily_loss_limit', float),
            'TRADING_SYMBOLS': ('trading', 'symbols', lambda x: x.split(',')),
            'TRADING_SCAN_INTERVAL': ('trading', 'scan_interval', int),
            'TRADING_USE_AI': ('trading', 'use_ai_analysis', lambda x: x.lower() == 'true'),
            'TRADING_TESTNET': ('trading', 'testnet', lambda x: x.lower() == 'true'),
            
            'BINANCE_API_KEY': ('api', 'binance_api_key', str),
            'BINANCE_SECRET': ('api', 'binance_secret', str),
            'CLAUDE_API_KEY': ('api', 'claude_api_key', str),
            
            'DISCORD_WEBHOOK': ('notifications', 'discord_webhook_url', str),
            'TELEGRAM_BOT_TOKEN': ('notifications', 'telegram_bot_token', str),
            'TELEGRAM_CHAT_ID': ('notifications', 'telegram_chat_id', str),
            'EMAIL_USER': ('notifications', 'email_user', str),
            'EMAIL_PASSWORD': ('notifications', 'email_password', str),
        }
        
        for env_var, (section, key, converter) in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value:
                try:
                    converted_value = converter(env_value)
                    
                    # Ensure section exists
                    if section not in self.config_data:
                        self.config_data[section] = {}
                    
                    self.config_data[section][key] = converted_value
                    self.logger.debug(f"Environment override: {env_var} -> {section}.{key}")
                    
                except Exception as e:
                    self.logger.warning(f"Error converting environment variable {env_var}: {e}")
    
    def _create_bot_configuration(self) -> BotConfiguration:
        """Create BotConfiguration from loaded data"""
        trading_config = self.config_data.get('trading', {})
        
        return BotConfiguration(
            initial_capital=trading_config.get('initial_capital', 10000.0),
            max_open_positions=trading_config.get('max_open_positions', 5),
            max_risk_per_trade=trading_config.get('max_risk_per_trade', 0.02),
            daily_loss_limit=trading_config.get('daily_loss_limit', 0.05),
            max_consecutive_losses=trading_config.get('max_consecutive_losses', 3),
            symbols=trading_config.get('symbols', ['BTC/USDT', 'ETH/USDT', 'ADA/USDT']),
            primary_timeframe=trading_config.get('primary_timeframe', '1h'),
            scan_interval=trading_config.get('scan_interval', 300),
            use_ai_analysis=trading_config.get('use_ai_analysis', True),
            ai_confidence_threshold=trading_config.get('ai_confidence_threshold', 70.0),
            enable_notifications=trading_config.get('enable_notifications', True),
            notification_channels=trading_config.get('notification_channels', ['discord', 'email']),
            exchange=trading_config.get('exchange', 'binance'),
            testnet=trading_config.get('testnet', True)
        )
    
    def get_api_config(self):
        """Get API configuration"""
        api_config = self.config_data.get('api', {})
        
        class ApiConfig:
            def __init__(self, config_data):
                # Binance API
                self.binance_api_key = config_data.get('binance_api_key', '')
                self.binance_secret = config_data.get('binance_secret', '')
                self.binance_testnet = config_data.get('binance_testnet', True)
                
                # Claude API
                self.claude_api_key = config_data.get('claude_api_key', '')
                
                # Notification APIs
                notifications = config_data.get('notifications', {})
                self.discord_webhook_url = notifications.get('discord_webhook_url', '')
                self.telegram_bot_token = notifications.get('telegram_bot_token', '')
                self.telegram_chat_id = notifications.get('telegram_chat_id', '')
                self.slack_webhook_url = notifications.get('slack_webhook_url', '')
                self.email_user = notifications.get('email_user', '')
                self.email_password = notifications.get('email_password', '')
                self.from_email = notifications.get('from_email', '')
                self.to_emails = notifications.get('to_emails', [])
                self.smtp_server = notifications.get('smtp_server', 'smtp.gmail.com')
                self.smtp_port = notifications.get('smtp_port', 587)
        
        # Merge notifications into api config
        merged_config = api_config.copy()
        merged_config.update(self.config_data.get('notifications', {}))
        
        return ApiConfig(merged_config)
    
    def get_claude_config(self):
        """Get Claude configuration"""
        api_config = self.config_data.get('api', {})
        
        if not api_config.get('claude_api_key'):
            return None
        
        class ClaudeConfig:
            def __init__(self, api_key):
                self.api_key = api_key
        
        return ClaudeConfig(api_config['claude_api_key'])
    
    def save_configuration(self, bot_config: BotConfiguration):
        """Save current configuration to file"""
        try:
            config_dict = {
                'trading': {
                    'initial_capital': bot_config.initial_capital,
                    'max_open_positions': bot_config.max_open_positions,
                    'max_risk_per_trade': bot_config.max_risk_per_trade,
                    'daily_loss_limit': bot_config.daily_loss_limit,
                    'symbols': bot_config.symbols,
                    'scan_interval': bot_config.scan_interval,
                    'use_ai_analysis': bot_config.use_ai_analysis,
                    'testnet': bot_config.testnet
                }
            }
            
            # Merge with existing config data (to preserve API keys)
            for section in ['api', 'notifications']:
                if section in self.config_data:
                    config_dict[section] = self.config_data[section]
            
            # Ensure config directory exists
            config_path = Path(self.config_file)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save to file
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            self.logger.info(f"Configuration saved to {config_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")

class BotLauncher:
    """Main bot launcher with lifecycle management"""
    
    def __init__(self, config_file: str = None, log_level: str = "INFO"):
        self.config_manager = ConfigurationManager(config_file)
        self.logger = setup_logging(log_level)
        self.bot: Optional[TradingBot] = None
        self.shutdown_requested = False
        self.restart_requested = False
        
        # Process management
        self.pid_file = Path("trading_bot.pid")
        self.status_file = Path("bot_status.json")
        
        # Performance monitoring
        self.start_time = None
        self.process = psutil.Process()
        
        self.logger.info("Bot Launcher initialized")
    
    def check_prerequisites(self) -> bool:
        """Check system prerequisites before starting"""
        try:
            self.logger.info("Checking system prerequisites...")
            
            checks = []
            
            # Check Python version
            python_version = sys.version_info
            if python_version >= (3, 8):
                checks.append(("Python version", True, f"{python_version.major}.{python_version.minor}"))
            else:
                checks.append(("Python version", False, f"Requires 3.8+, found {python_version.major}.{python_version.minor}"))
            
            # Check required directories
            required_dirs = ['logs', 'config', 'database']
            for dir_name in required_dirs:
                dir_path = Path(dir_name)
                dir_path.mkdir(exist_ok=True)
                checks.append((f"Directory {dir_name}", True, "Created/Exists"))
            
            # Check database connection
            try:
                init_database()
                close_database()
                checks.append(("Database connection", True, "Connected"))
            except Exception as e:
                checks.append(("Database connection", False, str(e)))
            
            # Check configuration
            try:
                bot_config = self.config_manager.load_configuration()
                api_config = self.config_manager.get_api_config()
                
                # Validate essential API keys
                if api_config.binance_api_key and api_config.binance_secret:
                    checks.append(("Binance API", True, "Configured"))
                else:
                    checks.append(("Binance API", False, "Missing API keys"))
                
                checks.append(("Configuration", True, "Loaded"))
                
            except Exception as e:
                checks.append(("Configuration", False, str(e)))
            
            # Check memory
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            if available_gb >= 1.0:
                checks.append(("Available memory", True, f"{available_gb:.1f} GB"))
            else:
                checks.append(("Available memory", False, f"Only {available_gb:.1f} GB available"))
            
            # Check disk space
            disk = psutil.disk_usage('.')
            free_gb = disk.free / (1024**3)
            if free_gb >= 1.0:
                checks.append(("Disk space", True, f"{free_gb:.1f} GB free"))
            else:
                checks.append(("Disk space", False, f"Only {free_gb:.1f} GB free"))
            
            # Display results
            self.logger.info("Prerequisite check results:")
            all_passed = True
            
            for check_name, passed, details in checks:
                status = "✅ PASS" if passed else "❌ FAIL"
                self.logger.info(f"  {check_name}: {status} - {details}")
                if not passed:
                    all_passed = False
            
            if all_passed:
                self.logger.info("All prerequisite checks passed")
            else:
                self.logger.error("Some prerequisite checks failed")
            
            return all_passed
            
        except Exception as e:
            self.logger.error(f"Error checking prerequisites: {e}")
            return False
    
    def is_bot_running(self) -> bool:
        """Check if another bot instance is running"""
        try:
            if not self.pid_file.exists():
                return False
            
            with open(self.pid_file, 'r') as f:
                pid = int(f.read().strip())
            
            # Check if process exists
            if psutil.pid_exists(pid):
                process = psutil.Process(pid)
                # Check if it's actually our bot
                if 'python' in process.name().lower() and 'trading' in ' '.join(process.cmdline()).lower():
                    self.logger.warning(f"Bot already running with PID {pid}")
                    return True
            
            # Stale PID file
            self.pid_file.unlink()
            return False
            
        except Exception as e:
            self.logger.warning(f"Error checking running bot: {e}")
            return False
    
    def write_pid_file(self):
        """Write current process PID to file"""
        try:
            with open(self.pid_file, 'w') as f:
                f.write(str(os.getpid()))
            self.logger.debug(f"PID file written: {self.pid_file}")
        except Exception as e:
            self.logger.warning(f"Error writing PID file: {e}")
    
    def cleanup_pid_file(self):
        """Clean up PID file"""
        try:
            if self.pid_file.exists():
                self.pid_file.unlink()
                self.logger.debug("PID file cleaned up")
        except Exception as e:
            self.logger.warning(f"Error cleaning up PID file: {e}")
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            signal_names = {2: 'SIGINT', 15: 'SIGTERM'}
            signal_name = signal_names.get(signum, f'Signal {signum}')
            self.logger.info(f"Received {signal_name}, initiating graceful shutdown...")
            self.shutdown_requested = True
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        if hasattr(signal, 'SIGUSR1'):  # Unix-specific
            def restart_handler(signum, frame):
                self.logger.info("Received SIGUSR1, initiating restart...")
                self.restart_requested = True
            signal.signal(signal.SIGUSR1, restart_handler)
    
    async def start_bot(self) -> bool:
        """Start the trading bot"""
        try:
            self.logger.info("Starting trading bot...")
            
            # Load configuration
            bot_config = self.config_manager.load_configuration()
            api_config = self.config_manager.get_api_config()
            claude_config = self.config_manager.get_claude_config()
            
            # Create bot
            self.bot = create_trading_bot(bot_config, api_config, claude_config)
            
            # Initialize bot
            await self.bot.initialize()
            
            self.start_time = datetime.utcnow()
            self.logger.info("Trading bot started successfully")
            
            # Update status
            await self.update_status_file()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting bot: {e}")
            return False
    
    async def run_bot(self):
        """Run the bot with monitoring and recovery"""
        try:
            # Start main bot execution
            bot_task = asyncio.create_task(self.bot.run())
            
            # Start monitoring task
            monitor_task = asyncio.create_task(self.monitor_bot())
            
            # Wait for completion or signals
            done, pending = await asyncio.wait(
                [bot_task, monitor_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Cancel pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
            # Check results
            for task in done:
                try:
                    await task
                except Exception as e:
                    self.logger.error(f"Task error: {e}")
            
        except Exception as e:
            self.logger.error(f"Error running bot: {e}")
    
    async def monitor_bot(self):
        """Monitor bot health and performance"""
        self.logger.info("Bot monitoring started")
        
        while not self.shutdown_requested and not self.restart_requested:
            try:
                # Update status file
                await self.update_status_file()
                
                # Monitor memory usage
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)
                
                if memory_mb > 500:  # 500MB threshold
                    self.logger.warning(f"High memory usage: {memory_mb:.1f} MB")
                
                # Monitor bot state
                if self.bot and self.bot.state == BotState.ERROR:
                    self.logger.error("Bot entered error state")
                    break
                
                # Check if bot is responsive
                if self.bot:
                    try:
                        status = await asyncio.wait_for(self.bot.get_bot_status(), timeout=30)
                        if not status:
                            self.logger.warning("Bot status check failed")
                    except asyncio.TimeoutError:
                        self.logger.error("Bot status check timed out - bot may be unresponsive")
                        break
                
                # Sleep for monitoring interval
                await asyncio.sleep(60)  # Monitor every minute
                
            except Exception as e:
                self.logger.error(f"Error in bot monitoring: {e}")
                await asyncio.sleep(30)
        
        self.logger.info("Bot monitoring stopped")
    
    async def update_status_file(self):
        """Update bot status file"""
        try:
            status_data = {
                'pid': os.getpid(),
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'last_update': datetime.utcnow().isoformat(),
                'memory_mb': self.process.memory_info().rss / (1024 * 1024),
                'cpu_percent': self.process.cpu_percent(),
                'bot_state': self.bot.state.value if self.bot else 'NOT_STARTED'
            }
            
            # Add bot status if available
            if self.bot:
                try:
                    bot_status = await self.bot.get_bot_status()
                    status_data.update({
                        'portfolio_value': bot_status.get('portfolio', {}).get('total_value', 0),
                        'active_positions': bot_status.get('portfolio', {}).get('active_positions', 0),
                        'total_trades': bot_status.get('bot_metrics', {}).get('total_trades_executed', 0),
                        'uptime_hours': bot_status.get('uptime_hours', 0)
                    })
                except:
                    pass  # Don't fail on status update error
            
            with open(self.status_file, 'w') as f:
                json.dump(status_data, f, indent=2)
                
        except Exception as e:
            self.logger.warning(f"Error updating status file: {e}")
    
    async def stop_bot(self):
        """Stop the trading bot gracefully"""
        try:
            if self.bot:
                self.logger.info("Stopping trading bot...")
                
                # Request bot shutdown
                self.bot.shutdown_requested = True
                
                # Wait for graceful shutdown with timeout
                try:
                    await asyncio.wait_for(self.bot._cleanup(), timeout=60)
                    self.logger.info("Bot stopped gracefully")
                except asyncio.TimeoutError:
                    self.logger.warning("Bot shutdown timed out")
                
                self.bot = None
            
        except Exception as e:
            self.logger.error(f"Error stopping bot: {e}")
    
    async def restart_bot(self):
        """Restart the trading bot"""
        try:
            self.logger.info("Restarting trading bot...")
            
            # Stop current bot
            await self.stop_bot()
            
            # Wait a moment
            await asyncio.sleep(5)
            
            # Start new bot
            if await self.start_bot():
                self.restart_requested = False
                self.logger.info("Bot restarted successfully")
                return True
            else:
                self.logger.error("Failed to restart bot")
                return False
                
        except Exception as e:
            self.logger.error(f"Error restarting bot: {e}")
            return False
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            self.logger.info("Cleaning up launcher resources...")
            
            # Stop bot
            await self.stop_bot()
            
            # Cleanup files
            self.cleanup_pid_file()
            
            # Clean up status file
            try:
                if self.status_file.exists():
                    self.status_file.unlink()
            except:
                pass
            
            self.logger.info("Launcher cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    async def run(self):
        """Main execution loop with restart capability"""
        try:
            while True:
                # Start bot
                if not await self.start_bot():
                    self.logger.error("Failed to start bot")
                    break
                
                # Run bot
                await self.run_bot()
                
                # Check for restart or shutdown
                if self.shutdown_requested:
                    self.logger.info("Shutdown requested")
                    break
                elif self.restart_requested:
                    self.logger.info("Restart requested")
                    await asyncio.sleep(5)  # Brief pause before restart
                    continue
                else:
                    self.logger.info("Bot execution completed")
                    break
            
        except Exception as e:
            self.logger.error(f"Error in main execution loop: {e}")
        finally:
            await self.cleanup()

def create_sample_config():
    """Create a sample configuration file"""
    sample_config = {
        'trading': {
            'initial_capital': 10000.0,
            'max_open_positions': 5,
            'max_risk_per_trade': 0.02,
            'daily_loss_limit': 0.05,
            'max_consecutive_losses': 3,
            'symbols': ['BTC/USDT', 'ETH/USDT', 'ADA/USDT'],
            'primary_timeframe': '1h',
            'scan_interval': 300,
            'use_ai_analysis': True,
            'ai_confidence_threshold': 70.0,
            'enable_notifications': True,
            'notification_channels': ['discord', 'email'],
            'exchange': 'binance',
            'testnet': True
        },
        'api': {
            'binance_api_key': 'your_binance_api_key_here',
            'binance_secret': 'your_binance_secret_here',
            'claude_api_key': 'your_claude_api_key_here'
        },
        'notifications': {
            'discord_webhook_url': 'your_discord_webhook_url_here',
            'telegram_bot_token': 'your_telegram_bot_token_here',
            'telegram_chat_id': 'your_telegram_chat_id_here',
            'email_user': 'your_email@gmail.com',
            'email_password': 'your_app_password_here',
            'to_emails': ['recipient@gmail.com']
        }
    }
    
    config_path = Path("config/bot_config.yaml")
    config_path.parent.mkdir(exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(sample_config, f, default_flow_style=False, indent=2)
    
    print(f"Sample configuration created at: {config_path}")
    print("Please edit the configuration file with your actual API keys and settings.")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Cryptocurrency Trading Bot Launcher")
    parser.add_argument("--config", "-c", default="config/bot_config.yaml", 
                       help="Configuration file path")
    parser.add_argument("--log-level", "-l", default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--create-config", action="store_true",
                       help="Create sample configuration file")
    parser.add_argument("--check-only", action="store_true",
                       help="Only run prerequisite checks")
    parser.add_argument("--force", "-f", action="store_true",
                       help="Force start even if another instance is running")
    
    args = parser.parse_args()
    
    # Create sample config if requested
    if args.create_config:
        create_sample_config()
        return
    
    print("🤖 Cryptocurrency Trading Bot Launcher")
    print("=" * 50)
    
    try:
        # Create launcher
        launcher = BotLauncher(args.config, args.log_level)
        
        # Check prerequisites
        if not launcher.check_prerequisites():
            print("❌ Prerequisite checks failed. Please fix the issues and try again.")
            return 1
        
        if args.check_only:
            print("✅ All prerequisite checks passed.")
            return 0
        
        # Check for existing instance
        if not args.force and launcher.is_bot_running():
            print("❌ Another bot instance is already running. Use --force to override.")
            return 1
        
        # Setup signal handlers
        launcher.setup_signal_handlers()
        
        # Write PID file
        launcher.write_pid_file()
        
        print("🚀 Starting trading bot...")
        
        # Run bot
        asyncio.run(launcher.run())
        
        print("✅ Trading bot shutdown completed.")
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️ Shutdown requested by user")
        return 0
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        logging.error(f"Fatal error: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())