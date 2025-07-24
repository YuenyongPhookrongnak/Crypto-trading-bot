"""
Configuration Management System

This module handles all configuration loading, validation, and management
for the trading bot system.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class TradingConfig:
    """Trading configuration settings"""
    initial_capital: float = 10000.0
    max_open_positions: int = 5
    max_risk_per_trade: float = 0.02  # 2%
    daily_loss_limit: float = 0.05    # 5%
    max_consecutive_losses: int = 3
    
    # Market settings
    symbols: List[str] = field(default_factory=lambda: ['BTC/USDT', 'ETH/USDT'])
    primary_timeframe: str = '1h'
    scan_interval: int = 300  # 5 minutes
    
    # Strategy settings
    enabled_strategies: List[str] = field(default_factory=lambda: ['momentum', 'rsi'])
    strategy_weights: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.strategy_weights:
            # Default equal weights
            weight = 1.0 / len(self.enabled_strategies) if self.enabled_strategies else 1.0
            self.strategy_weights = {strategy: weight for strategy in self.enabled_strategies}

@dataclass
class APIConfig:
    """API configuration settings"""
    # Exchange APIs
    binance_api_key: str = ""
    binance_secret: str = ""
    binance_testnet: bool = True
    
    # AI APIs
    claude_api_key: str = ""
    openai_api_key: str = ""
    
    # Notification APIs
    discord_webhook_url: str = ""
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    slack_webhook_url: str = ""
    
    # Email settings
    email_user: str = ""
    email_password: str = ""
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    to_emails: List[str] = field(default_factory=list)

@dataclass 
class DatabaseConfig:
    """Database configuration settings"""
    database_url: str = "sqlite:///trading_bot.db"
    echo_sql: bool = False
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: int = 30

@dataclass
class LoggingConfig:
    """Logging configuration settings"""
    log_level: str = "INFO"
    log_file: str = "trading_bot.log"
    max_file_size: int = 10  # MB
    backup_count: int = 5
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

@dataclass
class RiskConfig:
    """Risk management configuration"""
    max_portfolio_risk: float = 0.1    # 10%
    var_confidence: float = 0.95        # 95% VaR
    stress_test_scenarios: int = 1000
    correlation_threshold: float = 0.8   # High correlation warning
    
    # Circuit breaker settings
    daily_loss_circuit: float = 0.05    # 5%
    drawdown_circuit: float = 0.15      # 15%
    consecutive_loss_circuit: int = 5
    volatility_circuit: float = 0.3     # 30%

class ConfigurationManager:
    """Centralized configuration management"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # Configuration instances
        self.trading_config = TradingConfig()
        self.api_config = APIConfig()
        self.database_config = DatabaseConfig()
        self.logging_config = LoggingConfig()
        self.risk_config = RiskConfig()
        
        # Configuration files
        self.config_files = {
            'main': self.config_dir / 'bot_config.yaml',
            'secrets': self.config_dir / 'secrets.yaml',
            'strategies': self.config_dir / 'strategies.yaml'
        }
        
        logger.info(f"Configuration manager initialized with config dir: {self.config_dir}")
    
    def load_configuration(self) -> bool:
        """Load all configuration from files and environment"""
        try:
            # Load main configuration
            self._load_main_config()
            
            # Load secrets (API keys, etc.)
            self._load_secrets_config()
            
            # Load strategy configurations
            self._load_strategy_config()
            
            # Override with environment variables
            self._load_environment_overrides()
            
            # Validate configuration
            self._validate_configuration()
            
            logger.info("Configuration loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return False
    
    def _load_main_config(self):
        """Load main configuration file"""
        config_file = self.config_files['main']
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f) or {}
            
            # Update trading config
            if 'trading' in config_data:
                trading_data = config_data['trading']
                for key, value in trading_data.items():
                    if hasattr(self.trading_config, key):
                        setattr(self.trading_config, key, value)
            
            # Update database config
            if 'database' in config_data:
                db_data = config_data['database']
                for key, value in db_data.items():
                    if hasattr(self.database_config, key):
                        setattr(self.database_config, key, value)
            
            # Update logging config
            if 'logging' in config_data:
                log_data = config_data['logging']
                for key, value in log_data.items():
                    if hasattr(self.logging_config, key):
                        setattr(self.logging_config, key, value)
            
            # Update risk config
            if 'risk' in config_data:
                risk_data = config_data['risk']
                for key, value in risk_data.items():
                    if hasattr(self.risk_config, key):
                        setattr(self.risk_config, key, value)
            
            logger.info(f"Main configuration loaded from {config_file}")
        else:
            logger.info("Main configuration file not found, using defaults")
            self._create_default_config()
    
    def _load_secrets_config(self):
        """Load secrets configuration"""
        secrets_file = self.config_files['secrets']
        
        if secrets_file.exists():
            with open(secrets_file, 'r') as f:
                secrets_data = yaml.safe_load(f) or {}
            
            # Update API config with secrets
            for key, value in secrets_data.items():
                if hasattr(self.api_config, key):
                    setattr(self.api_config, key, value)
            
            logger.info("Secrets configuration loaded")
        else:
            logger.warning("Secrets file not found - API keys may not be configured")
    
    def _load_strategy_config(self):
        """Load strategy-specific configurations"""
        strategies_file = self.config_files['strategies']
        
        if strategies_file.exists():
            with open(strategies_file, 'r') as f:
                self.strategy_configs = yaml.safe_load(f) or {}
            
            logger.info(f"Strategy configurations loaded: {list(self.strategy_configs.keys())}")
        else:
            logger.info("Strategy configuration file not found, using defaults")
            self.strategy_configs = self._get_default_strategy_configs()
    
    def _load_environment_overrides(self):
        """Load configuration overrides from environment variables"""
        env_mappings = {
            # Trading config
            'TRADING_INITIAL_CAPITAL': ('trading_config', 'initial_capital', float),
            'TRADING_MAX_POSITIONS': ('trading_config', 'max_open_positions', int),
            'TRADING_RISK_PER_TRADE': ('trading_config', 'max_risk_per_trade', float),
            'TRADING_DAILY_LOSS_LIMIT': ('trading_config', 'daily_loss_limit', float),
            'TRADING_SYMBOLS': ('trading_config', 'symbols', lambda x: x.split(',')),
            'TRADING_TESTNET': ('trading_config', 'testnet', lambda x: x.lower() == 'true'),
            
            # API config
            'BINANCE_API_KEY': ('api_config', 'binance_api_key', str),
            'BINANCE_SECRET': ('api_config', 'binance_secret', str),
            'BINANCE_TESTNET': ('api_config', 'binance_testnet', lambda x: x.lower() == 'true'),
            'CLAUDE_API_KEY': ('api_config', 'claude_api_key', str),
            'DISCORD_WEBHOOK': ('api_config', 'discord_webhook_url', str),
            'TELEGRAM_BOT_TOKEN': ('api_config', 'telegram_bot_token', str),
            'TELEGRAM_CHAT_ID': ('api_config', 'telegram_chat_id', str),
            
            # Database config
            'DATABASE_URL': ('database_config', 'database_url', str),
            'DATABASE_ECHO_SQL': ('database_config', 'echo_sql', lambda x: x.lower() == 'true'),
            
            # Logging config
            'LOG_LEVEL': ('logging_config', 'log_level', str),
            'LOG_FILE': ('logging_config', 'log_file', str),
        }
        
        overrides_applied = 0
        
        for env_var, (config_obj, config_key, converter) in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value:
                try:
                    converted_value = converter(env_value)
                    config_instance = getattr(self, config_obj)
                    setattr(config_instance, config_key, converted_value)
                    overrides_applied += 1
                    logger.debug(f"Environment override: {env_var} -> {config_obj}.{config_key}")
                except Exception as e:
                    logger.warning(f"Error applying environment override {env_var}: {e}")
        
        if overrides_applied > 0:
            logger.info(f"Applied {overrides_applied} environment overrides")
    
    def _validate_configuration(self):
        """Validate configuration settings"""
        errors = []
        
        # Validate trading config
        if self.trading_config.initial_capital <= 0:
            errors.append("Initial capital must be positive")
        
        if self.trading_config.max_open_positions <= 0:
            errors.append("Max open positions must be positive")
        
        if not 0 < self.trading_config.max_risk_per_trade <= 1:
            errors.append("Max risk per trade must be between 0 and 1")
        
        if not self.trading_config.symbols:
            errors.append("At least one trading symbol must be specified")
        
        # Validate API config for critical keys
        if not self.api_config.binance_api_key and not self.api_config.binance_testnet:
            errors.append("Binance API key required for live trading")
        
        # Validate risk config
        if not 0 < self.risk_config.max_portfolio_risk <= 1:
            errors.append("Max portfolio risk must be between 0 and 1")
        
        if errors:
            error_msg = "Configuration validation errors:\n" + "\n".join(f"- {error}" for error in errors)
            raise ValueError(error_msg)
        
        logger.info("Configuration validation passed")
    
    def _create_default_config(self):
        """Create default configuration files"""
        try:
            # Create main config
            default_config = {
                'trading': {
                    'initial_capital': 10000.0,
                    'max_open_positions': 5,
                    'max_risk_per_trade': 0.02,
                    'daily_loss_limit': 0.05,
                    'symbols': ['BTC/USDT', 'ETH/USDT', 'ADA/USDT'],
                    'primary_timeframe': '1h',
                    'scan_interval': 300,
                    'enabled_strategies': ['momentum', 'rsi']
                },
                'database': {
                    'database_url': 'sqlite:///trading_bot.db',
                    'echo_sql': False
                },
                'logging': {
                    'log_level': 'INFO',
                    'log_file': 'trading_bot.log'
                },
                'risk': {
                    'max_portfolio_risk': 0.1,
                    'var_confidence': 0.95,
                    'daily_loss_circuit': 0.05,
                    'drawdown_circuit': 0.15
                }
            }
            
            with open(self.config_files['main'], 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False, indent=2)
            
            # Create secrets template
            secrets_template = {
                'binance_api_key': 'your_binance_api_key_here',
                'binance_secret': 'your_binance_secret_here',
                'claude_api_key': 'your_claude_api_key_here',
                'discord_webhook_url': 'your_discord_webhook_here',
                'telegram_bot_token': 'your_telegram_bot_token_here',
                'telegram_chat_id': 'your_telegram_chat_id_here'
            }
            
            with open(self.config_files['secrets'], 'w') as f:
                yaml.dump(secrets_template, f, default_flow_style=False, indent=2)
            
            # Create strategies config
            strategies_config = self._get_default_strategy_configs()
            
            with open(self.config_files['strategies'], 'w') as f:
                yaml.dump(strategies_config, f, default_flow_style=False, indent=2)
            
            logger.info("Default configuration files created")
            
        except Exception as e:
            logger.error(f"Error creating default configuration: {e}")
    
    def _get_default_strategy_configs(self) -> Dict[str, Any]:
        """Get default strategy configurations"""
        return {
            'momentum': {
                'lookback_period': 14,
                'momentum_threshold': 0.02,
                'stop_loss_pct': 0.03,
                'take_profit_pct': 0.06,
                'volume_filter': True,
                'min_volume_ratio': 1.5
            },
            'rsi': {
                'rsi_period': 14,
                'oversold_threshold': 30,
                'overbought_threshold': 70,
                'stop_loss_pct': 0.025,
                'take_profit_pct': 0.05,
                'volume_confirmation': True
            },
            'volume_profile': {
                'profile_period': 24,
                'value_area_percentage': 0.7,
                'poc_threshold': 0.1,
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.04
            },
            'pairs_trading': {
                'lookback_period': 20,
                'entry_threshold': 2.0,
                'exit_threshold': 0.5,
                'stop_loss_threshold': 3.0,
                'correlation_threshold': 0.7
            },
            'grid_trading': {
                'grid_size': 0.015,
                'num_grids': 8,
                'base_position_size': 0.1,
                'take_profit_pct': 0.012,
                'stop_loss_pct': 0.04
            }
        }
    
    def get_strategy_config(self, strategy_name: str) -> Dict[str, Any]:
        """Get configuration for specific strategy"""
        return self.strategy_configs.get(strategy_name, {})
    
    def update_strategy_config(self, strategy_name: str, config: Dict[str, Any]) -> bool:
        """Update strategy configuration"""
        try:
            self.strategy_configs[strategy_name] = config
            
            # Save to file
            with open(self.config_files['strategies'], 'w') as f:
                yaml.dump(self.strategy_configs, f, default_flow_style=False, indent=2)
            
            logger.info(f"Strategy configuration updated: {strategy_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating strategy config: {e}")
            return False
    
    def save_configuration(self):
        """Save current configuration to files"""
        try:
            # Save main config
            main_config = {
                'trading': {
                    'initial_capital': self.trading_config.initial_capital,
                    'max_open_positions': self.trading_config.max_open_positions,
                    'max_risk_per_trade': self.trading_config.max_risk_per_trade,
                    'daily_loss_limit': self.trading_config.daily_loss_limit,
                    'symbols': self.trading_config.symbols,
                    'primary_timeframe': self.trading_config.primary_timeframe,
                    'scan_interval': self.trading_config.scan_interval,
                    'enabled_strategies': self.trading_config.enabled_strategies
                },
                'database': {
                    'database_url': self.database_config.database_url,
                    'echo_sql': self.database_config.echo_sql
                },
                'logging': {
                    'log_level': self.logging_config.log_level,
                    'log_file': self.logging_config.log_file
                },
                'risk': {
                    'max_portfolio_risk': self.risk_config.max_portfolio_risk,
                    'var_confidence': self.risk_config.var_confidence,
                    'daily_loss_circuit': self.risk_config.daily_loss_circuit,
                    'drawdown_circuit': self.risk_config.drawdown_circuit
                }
            }
            
            with open(self.config_files['main'], 'w') as f:
                yaml.dump(main_config, f, default_flow_style=False, indent=2)
            
            logger.info("Configuration saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False
    
    def get_full_config(self) -> Dict[str, Any]:
        """Get complete configuration as dictionary"""
        return {
            'trading': self.trading_config.__dict__,
            'api': self.api_config.__dict__,
            'database': self.database_config.__dict__,
            'logging': self.logging_config.__dict__,
            'risk': self.risk_config.__dict__,
            'strategies': self.strategy_configs
        }
    
    def print_config_summary(self):
        """Print configuration summary"""
        print("📋 Configuration Summary")
        print("=" * 50)
        print(f"💰 Initial Capital: ${self.trading_config.initial_capital:,.2f}")
        print(f"📊 Max Positions: {self.trading_config.max_open_positions}")
        print(f"⚠️ Risk per Trade: {self.trading_config.max_risk_per_trade:.1%}")
        print(f"💱 Symbols: {', '.join(self.trading_config.symbols)}")
        print(f"🔧 Strategies: {', '.join(self.trading_config.enabled_strategies)}")
        print(f"🧪 Testnet: {self.api_config.binance_testnet}")
        print(f"📁 Database: {self.database_config.database_url}")
        print(f"📝 Log Level: {self.logging_config.log_level}")

# Global configuration instance
config = ConfigurationManager()

# Convenience functions
def load_config() -> bool:
    """Load configuration"""
    return config.load_configuration()

def get_trading_config() -> TradingConfig:
    """Get trading configuration"""
    return config.trading_config

def get_api_config() -> APIConfig:
    """Get API configuration"""
    return config.api_config

def get_database_config() -> DatabaseConfig:
    """Get database configuration"""
    return config.database_config

def get_risk_config() -> RiskConfig:
    """Get risk configuration"""
    return config.risk_config

def get_strategy_config(strategy_name: str) -> Dict[str, Any]:
    """Get strategy configuration"""
    return config.get_strategy_config(strategy_name)

# Example usage and testing
if __name__ == "__main__":
    print("🔧 Configuration Management System Test")
    print("=" * 50)
    
    # Initialize configuration manager
    config_manager = ConfigurationManager()
    
    # Load configuration
    if config_manager.load_configuration():
        print("✅ Configuration loaded successfully")
        
        # Print summary
        config_manager.print_config_summary()
        
        # Test strategy config
        momentum_config = config_manager.get_strategy_config('momentum')
        print(f"\n📈 Momentum Strategy Config:")
        for key, value in momentum_config.items():
            print(f"  {key}: {value}")
        
    else:
        print("❌ Configuration loading failed")