import os
from typing import Dict, Any
from dotenv import load_dotenv
from dataclasses import dataclass

# โหลด environment variables
load_dotenv()

@dataclass
class APIConfig:
    """API Configuration Class"""
    binance_api_key: str = os.getenv('BINANCE_API_KEY', '')
    binance_secret: str = os.getenv('BINANCE_SECRET_KEY', '')
    binance_testnet: bool = os.getenv('BINANCE_TESTNET', 'true').lower() == 'true'
    claude_api_key: str = os.getenv('CLAUDE_API_KEY', '')
    
    def validate(self) -> bool:
        """ตรวจสอบความครบถ้วนของ API keys"""
        required_keys = [self.binance_api_key, self.binance_secret, self.claude_api_key]
        return all(key.strip() for key in required_keys)

@dataclass
class TradingConfig:
    """Trading Parameters Configuration"""
    initial_capital: float = float(os.getenv('INITIAL_CAPITAL', '10000'))
    max_risk_per_trade: float = float(os.getenv('MAX_RISK_PER_TRADE', '0.02'))
    max_portfolio_risk: float = float(os.getenv('MAX_PORTFOLIO_RISK', '0.10'))
    max_open_positions: int = int(os.getenv('MAX_OPEN_POSITIONS', '5'))
    commission_rate: float = float(os.getenv('COMMISSION_RATE', '0.001'))
    
    # Risk Management
    daily_loss_limit: float = float(os.getenv('DAILY_LOSS_LIMIT', '0.05'))
    max_consecutive_losses: int = int(os.getenv('MAX_CONSECUTIVE_LOSSES', '3'))
    circuit_breaker_threshold: float = float(os.getenv('CIRCUIT_BREAKER_THRESHOLD', '0.08'))

@dataclass
class SystemConfig:
    """System Configuration"""
    database_url: str = os.getenv('DATABASE_URL', 'sqlite:///data/trading_bot.db')
    log_level: str = os.getenv('LOG_LEVEL', 'INFO')
    
    # Intervals (in seconds)
    market_scan_interval: int = int(os.getenv('MARKET_SCAN_INTERVAL', '300'))
    position_check_interval: int = int(os.getenv('POSITION_CHECK_INTERVAL', '60'))
    performance_report_interval: int = int(os.getenv('PERFORMANCE_REPORT_INTERVAL', '3600'))

@dataclass
class NotificationConfig:
    """Notification Configuration"""
    discord_webhook: str = os.getenv('DISCORD_WEBHOOK_URL', '')
    telegram_bot_token: str = os.getenv('TELEGRAM_BOT_TOKEN', '')
    telegram_chat_id: str = os.getenv('TELEGRAM_CHAT_ID', '')
    enable_notifications: bool = os.getenv('ENABLE_NOTIFICATIONS', 'true').lower() == 'true'

class Config:
    """Main Configuration Class"""
    
    def __init__(self):
        self.api = APIConfig()
        self.trading = TradingConfig()
        self.system = SystemConfig()
        self.notifications = NotificationConfig()
        
        # Strategy-specific configurations
        self.strategies = {
            'rsi_strategy': {
                'rsi_period': 14,
                'oversold_threshold': 30,
                'overbought_threshold': 70,
                'confirmation_period': 3,
                'stop_loss_percentage': 0.02,
                'take_profit_percentage': 0.04,
                'volume_filter': True,
                'trend_filter': False,
                'enabled': True
            },
            'volume_profile_strategy': {
                'lookback_period': 24,
                'volume_multiplier': 1.5,
                'poc_deviation_threshold': 0.002,
                'breakout_confirmation_bars': 2,
                'min_volume_spike': 1.3,
                'stop_loss_percentage': 0.015,
                'take_profit_multiplier': 2.5,
                'value_area_percentage': 0.7,
                'enabled': True
            },
            'multi_timeframe_strategy': {
                'higher_timeframe': '4h',
                'entry_timeframe': '1h',
                'execution_timeframe': '15m',
                'trend_strength_threshold': 0.6,
                'pullback_percentage': 0.03,
                'momentum_period': 14,
                'volume_confirmation': True,
                'stop_loss_atr_multiplier': 2.0,
                'take_profit_rr_ratio': 3.0,
                'max_holding_period': 48,
                'enabled': True
            }
        }
        
        # Market scanning criteria
        self.scanning = {
            'min_volume_24h': 50_000_000,
            'min_price_change': 3.0,
            'volume_spike_threshold': 1.5,
            'max_symbols_to_scan': 50,
            'excluded_symbols': ['USDCUSDT', 'BUSDUSDT', 'TUSDUSDT', 'USTUSDT'],
            'min_market_cap': 100_000_000,
            'max_spread_percentage': 0.5,
            'min_liquidity_score': 60
        }
        
        # Backtesting configuration
        self.backtesting = {
            'default_start_date': '2023-01-01',
            'default_end_date': '2024-12-31',
            'default_timeframe': '1h',
            'commission_rate': 0.001,
            'slippage': 0.0005,
            'initial_capital': 10000
        }
        
        # Performance thresholds
        self.performance_thresholds = {
            'min_sharpe_ratio': 1.0,
            'max_drawdown': 0.15,
            'min_win_rate': 0.45,
            'min_profit_factor': 1.2,
            'max_correlation': 0.8
        }
    
    def validate_config(self) -> Dict[str, Any]:
        """ตรวจสอบการตั้งค่าทั้งหมด"""
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # ตรวจสอบ API keys
        if not self.api.validate():
            validation_results['valid'] = False
            validation_results['errors'].append("Missing required API keys")
        
        # ตรวจสอบ trading parameters
        if self.trading.initial_capital <= 0:
            validation_results['valid'] = False
            validation_results['errors'].append("Initial capital must be positive")
        
        if not (0 < self.trading.max_risk_per_trade <= 1):
            validation_results['valid'] = False
            validation_results['errors'].append("Max risk per trade must be between 0 and 1")
        
        if not (0 < self.trading.max_portfolio_risk <= 1):
            validation_results['valid'] = False
            validation_results['errors'].append("Max portfolio risk must be between 0 and 1")
        
        if self.trading.max_open_positions <= 0:
            validation_results['valid'] = False
            validation_results['errors'].append("Max open positions must be positive")
        
        # ตรวจสอบ system configuration
        if self.system.market_scan_interval < 60:
            validation_results['warnings'].append("Market scan interval less than 1 minute may cause rate limiting")
        
        if self.system.position_check_interval < 30:
            validation_results['warnings'].append("Position check interval less than 30 seconds may cause high CPU usage")
        
        # ตรวจสอบ strategies
        enabled_strategies = [name for name, config in self.strategies.items() if config.get('enabled', False)]
        if not enabled_strategies:
            validation_results['valid'] = False
            validation_results['errors'].append("At least one strategy must be enabled")
        
        # ตรวจสอบ database connection
        try:
            if self.system.database_url.startswith('sqlite'):
                # ตรวจสอบว่า directory สำหรับ SQLite มีอยู่หรือไม่
                import os
                db_path = self.system.database_url.replace('sqlite:///', '')
                db_dir = os.path.dirname(db_path)
                if db_dir and not os.path.exists(db_dir):
                    validation_results['warnings'].append(f"Database directory does not exist: {db_dir}")
            elif self.system.database_url.startswith('postgresql'):
                # Basic PostgreSQL URL validation
                if '@' not in self.system.database_url or '/' not in self.system.database_url:
                    validation_results['errors'].append("Invalid PostgreSQL database URL format")
                    validation_results['valid'] = False
        except Exception as e:
            validation_results['warnings'].append(f"Could not validate database URL: {e}")
        
        # ตรวจสอบ notification configuration
        if self.notifications.enable_notifications:
            if not self.notifications.discord_webhook and not self.notifications.telegram_bot_token:
                validation_results['warnings'].append("Notifications enabled but no webhook/bot token configured")
        
        return validation_results
    
    def get_strategy_config(self, strategy_name: str) -> Dict[str, Any]:
        """ดึงการตั้งค่าของ strategy เฉพาะ"""
        return self.strategies.get(strategy_name, {})
    
    def update_strategy_config(self, strategy_name: str, new_config: Dict[str, Any]):
        """อัพเดทการตั้งค่าของ strategy"""
        if strategy_name in self.strategies:
            self.strategies[strategy_name].update(new_config)
        else:
            self.strategies[strategy_name] = new_config
    
    def enable_strategy(self, strategy_name: str):
        """เปิดใช้งาน strategy"""
        if strategy_name in self.strategies:
            self.strategies[strategy_name]['enabled'] = True
    
    def disable_strategy(self, strategy_name: str):
        """ปิดใช้งาน strategy"""
        if strategy_name in self.strategies:
            self.strategies[strategy_name]['enabled'] = False
    
    def get_enabled_strategies(self) -> list:
        """ดึงรายชื่อ strategies ที่เปิดใช้งาน"""
        return [name for name, config in self.strategies.items() if config.get('enabled', False)]
    
    def to_dict(self) -> Dict[str, Any]:
        """แปลง configuration เป็น dictionary"""
        return {
            'api': {
                'binance_testnet': self.api.binance_testnet,
                # ไม่ include sensitive data
            },
            'trading': {
                'initial_capital': self.trading.initial_capital,
                'max_risk_per_trade': self.trading.max_risk_per_trade,
                'max_portfolio_risk': self.trading.max_portfolio_risk,
                'max_open_positions': self.trading.max_open_positions,
                'commission_rate': self.trading.commission_rate,
                'daily_loss_limit': self.trading.daily_loss_limit,
                'max_consecutive_losses': self.trading.max_consecutive_losses,
                'circuit_breaker_threshold': self.trading.circuit_breaker_threshold
            },
            'system': {
                'log_level': self.system.log_level,
                'market_scan_interval': self.system.market_scan_interval,
                'position_check_interval': self.system.position_check_interval,
                'performance_report_interval': self.system.performance_report_interval
            },
            'strategies': self.strategies,
            'scanning': self.scanning,
            'backtesting': self.backtesting,
            'performance_thresholds': self.performance_thresholds
        }

# Global config instance
config = Config()

# Export สำหรับ backward compatibility
def get_config() -> Config:
    """ดึง global config instance"""
    return config

def validate_environment() -> bool:
    """ตรวจสอบ environment variables"""
    validation = config.validate_config()
    return validation['valid']

if __name__ == "__main__":
    # ทดสอบ configuration
    print("🔧 Testing Configuration...")
    
    validation = config.validate_config()
    
    if validation['valid']:
        print("✅ Configuration is valid!")
        
        if validation['warnings']:
            print("\n⚠️  Warnings:")
            for warning in validation['warnings']:
                print(f"  • {warning}")
        
        print(f"\n📊 Enabled strategies: {config.get_enabled_strategies()}")
        print(f"💰 Initial capital: ${config.trading.initial_capital:,.2f}")
        print(f"⚠️  Max risk per trade: {config.trading.max_risk_per_trade:.1%}")
        print(f"🔄 Market scan interval: {config.system.market_scan_interval} seconds")
        
    else:
        print("❌ Configuration is invalid!")
        print("\nErrors:")
        for error in validation['errors']:
            print(f"  • {error}")
        
        if validation['warnings']:
            print("\nWarnings:")
            for warning in validation['warnings']:
                print(f"  • {warning}")