
# main_bot.py - Fixed Import Section
import asyncio
import logging
import ccxt
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import json
import signal
import sys
from dataclasses import dataclass
from enum import Enum

# Import all components with proper error handling
try:
    # Strategy imports
    from strategies.momentum_strategy import MomentumStrategy
    from strategies.rsi_strategy import RSIStrategy
    from strategies.volume_profile_strategy import VolumeProfileStrategy
    from strategies.pairs_trading_strategy import PairsTradingStrategy
    from strategies.grid_trading_strategy import GridTradingStrategy
    
    # Alternative import method using registry
    from strategies import STRATEGY_REGISTRY, create_strategy
    
    print("✅ All strategy imports successful")
    
except ImportError as e:
    print(f"❌ Strategy import error: {e}")
    print("📝 Creating missing strategy files...")
    
    # If imports fail, we can create basic strategy classes
    from strategies.base_strategy import BaseStrategy, TradingSignal
    
    class PairsTradingStrategy(BaseStrategy):
        """Fallback Pairs Trading Strategy"""
        def __init__(self, config):
            super().__init__(config)
            
        async def generate_signal(self, symbol, market_data, current_price):
            return None  # Placeholder
    
    class GridTradingStrategy(BaseStrategy):
        """Fallback Grid Trading Strategy"""
        def __init__(self, config):
            super().__init__(config)
            
        async def generate_signal(self, symbol, market_data, current_price):
            return None  # Placeholder

try:
    # Utility imports
    from utils.risk_manager import create_risk_manager, RiskLevel
    from utils.portfolio_manager import create_portfolio_manager, PositionDirection
    from utils.performance_tracker import create_performance_tracker, PerformancePeriod
    from utils.market_scanner import create_market_scanner
    from utils.notification_manager import create_notification_manager, NotificationType, NotificationPriority
    
    print("✅ All utility imports successful")
    
except ImportError as e:
    print(f"⚠️ Utility import warning: {e}")
    print("📝 Some utilities may not be available")

try:
    # AI analysis import (optional)
    from ai_analysis.claude_analyzer import create_claude_analyzer
    AI_ANALYSIS_AVAILABLE = True
    print("✅ AI analysis imports successful")
    
except ImportError as e:
    print(f"ℹ️ AI analysis not available: {e}")
    AI_ANALYSIS_AVAILABLE = False
    
    # Create dummy AI analyzer
    class DummyClaudeAnalyzer:
        async def initialize(self):
            pass
        async def analyze_trading_opportunity(self, **kwargs):
            return None
        async def cleanup(self):
            pass
    
    def create_claude_analyzer(api_key):
        return DummyClaudeAnalyzer()

try:
    # Database imports (optional)
    from database.connection import get_db_session
    from database.models import Trade, SystemLog, LogLevel
    DATABASE_AVAILABLE = True
    print("✅ Database imports successful")
    
except ImportError as e:
    print(f"ℹ️ Database not available: {e}")
    DATABASE_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Bot state enum
class BotState(Enum):
    """Trading bot states"""
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    ERROR = "ERROR"

@dataclass
class BotConfiguration:
    """Bot configuration settings"""
    # Trading settings
    initial_capital: float = 10000.0
    max_open_positions: int = 5
    max_risk_per_trade: float = 0.02  # 2%
    daily_loss_limit: float = 0.05    # 5%
    max_consecutive_losses: int = 3
    
    # Market settings
    symbols: List[str] = None
    primary_timeframe: str = '1h'
    scan_interval: int = 300  # 5 minutes
    
    # AI settings
    use_ai_analysis: bool = True
    ai_confidence_threshold: float = 70.0
    
    # Notification settings
    enable_notifications: bool = True
    notification_channels: List[str] = None
    
    # API settings
    exchange: str = 'binance'
    testnet: bool = True
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'DOT/USDT']
        if self.notification_channels is None:
            self.notification_channels = ['discord', 'email']

class TradingBot:
    """Advanced Cryptocurrency Trading Bot with Import Error Handling"""
    
    def __init__(self, config: BotConfiguration, api_config, claude_config=None):
        self.config = config
        self.api_config = api_config
        self.claude_config = claude_config
        
        # Bot state
        self.state = BotState.STARTING
        self.start_time = None
        self.last_scan_time = None
        self.shutdown_requested = False
        
        # Core components
        self.exchange = None
        self.strategies = {}
        self.risk_manager = None
        self.portfolio_manager = None
        self.performance_tracker = None
        self.market_scanner = None
        self.notification_manager = None
        self.claude_analyzer = None
        
        # Component availability flags
        self.components_available = {
            'strategies': True,
            'risk_manager': True,
            'portfolio_manager': True,
            'performance_tracker': True,
            'market_scanner': True,
            'notification_manager': True,
            'ai_analyzer': AI_ANALYSIS_AVAILABLE,
            'database': DATABASE_AVAILABLE
        }
        
        # Trading state
        self.active_signals = {}
        self.pending_trades = []
        self.market_data_cache = {}
        
        # Performance metrics
        self.bot_metrics = {
            'total_signals_generated': 0,
            'total_trades_executed': 0,
            'total_ai_analyses': 0,
            'circuit_breaker_activations': 0,
            'last_performance_update': None,
            'uptime_hours': 0,
            'import_errors': 0,
            'component_status': self.components_available
        }
        
        # Event loop
        self.monitoring_task = None
        
        logger.info("Trading Bot initialized with import error handling")
    
    async def initialize(self):
        """Initialize all bot components with graceful fallbacks"""
        try:
            logger.info("Initializing Trading Bot components...")
            
            # Initialize exchange
            await self._initialize_exchange()
            
            # Initialize strategies with error handling
            await self._initialize_strategies_safe()
            
            # Initialize other components with fallbacks
            await self._initialize_components_safe()
            
            # Setup signal handlers
            self._setup_signal_handlers()
            
            self.state = BotState.RUNNING
            self.start_time = datetime.utcnow()
            
            logger.info("Trading Bot initialization completed successfully")
            
            # Send startup notification if available
            if self.notification_manager:
                try:
                    await self.notification_manager.send_custom_notification(
                        title="🚀 Trading Bot Started",
                        message=f"Trading bot initialized successfully\nComponents available: {sum(self.components_available.values())}/{len(self.components_available)}",
                        channels=[],
                        priority=NotificationPriority.MEDIUM
                    )
                except Exception as e:
                    logger.warning(f"Could not send startup notification: {e}")
            
        except Exception as e:
            self.state = BotState.ERROR
            logger.error(f"Failed to initialize Trading Bot: {e}")
            raise
    
    async def _initialize_exchange(self):
        """Initialize exchange connection"""
        try:
            if self.config.exchange.lower() == 'binance':
                self.exchange = ccxt.binance({
                    'apiKey': self.api_config.binance_api_key,
                    'secret': self.api_config.binance_secret,
                    'sandbox': self.config.testnet,
                    'enableRateLimit': True,
                    'options': {'defaultType': 'spot'}
                })
            else:
                raise ValueError(f"Unsupported exchange: {self.config.exchange}")
            
            await self.exchange.load_markets()
            
            # Test connection
            balance = await self.exchange.fetch_balance()
            logger.info(f"Exchange connected: {self.config.exchange} ({'testnet' if self.config.testnet else 'live'})")
            
        except Exception as e:
            logger.error(f"Failed to initialize exchange: {e}")
            raise
    
    async def _initialize_strategies_safe(self):
        """Initialize trading strategies with error handling"""
        try:
            strategy_configs = {
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
            
            strategies_initialized = 0
            
            # Try to initialize each strategy individually
            for strategy_name, config in strategy_configs.items():
                try:
                    if strategy_name == 'momentum':
                        strategy = MomentumStrategy(config)
                    elif strategy_name == 'rsi':
                        strategy = RSIStrategy(config)
                    elif strategy_name == 'volume_profile':
                        strategy = VolumeProfileStrategy(config)
                    elif strategy_name == 'pairs_trading':
                        strategy = PairsTradingStrategy(config)
                    elif strategy_name == 'grid_trading':
                        strategy = GridTradingStrategy(config)
                    else:
                        continue
                    
                    await strategy.initialize()
                    self.strategies[strategy_name] = strategy
                    strategies_initialized += 1
                    logger.info(f"✅ {strategy_name} strategy initialized")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to initialize {strategy_name} strategy: {e}")
                    self.bot_metrics['import_errors'] += 1
                    continue
            
            if strategies_initialized == 0:
                logger.error("No strategies could be initialized")
                raise RuntimeError("No trading strategies available")
            
            logger.info(f"Initialized {strategies_initialized}/{len(strategy_configs)} trading strategies")
            
        except Exception as e:
            logger.error(f"Failed to initialize strategies: {e}")
            raise
    
    async def _initialize_components_safe(self):
        """Initialize other components with graceful fallbacks"""
        
        # Risk Manager
        try:
            self.risk_manager = create_risk_manager(self.config)
            logger.info("✅ Risk Manager initialized")
        except Exception as e:
            logger.warning(f"⚠️ Risk Manager initialization failed: {e}")
            self.components_available['risk_manager'] = False
            self.risk_manager = self._create_dummy_risk_manager()
        
        # Portfolio Manager
        try:
            self.portfolio_manager = create_portfolio_manager(self.api_config, self.config)
            await self.portfolio_manager.initialize()
            logger.info("✅ Portfolio Manager initialized")
        except Exception as e:
            logger.warning(f"⚠️ Portfolio Manager initialization failed: {e}")
            self.components_available['portfolio_manager'] = False
            self.portfolio_manager = self._create_dummy_portfolio_manager()
        
        # Performance Tracker
        try:
            self.performance_tracker = create_performance_tracker(self.config.initial_capital)
            await self.performance_tracker.initialize()
            logger.info("✅ Performance Tracker initialized")
        except Exception as e:
            logger.warning(f"⚠️ Performance Tracker initialization failed: {e}")
            self.components_available['performance_tracker'] = False
            self.performance_tracker = self._create_dummy_performance_tracker()
        
        # Market Scanner
        try:
            self.market_scanner = create_market_scanner(self.api_config)
            await self.market_scanner.initialize()
            logger.info("✅ Market Scanner initialized")
        except Exception as e:
            logger.warning(f"⚠️ Market Scanner initialization failed: {e}")
            self.components_available['market_scanner'] = False
            self.market_scanner = self._create_dummy_market_scanner()
        
        # Notification Manager
        try:
            if self.config.enable_notifications:
                self.notification_manager = create_notification_manager(self.api_config)
                await self.notification_manager.initialize()
                logger.info("✅ Notification Manager initialized")
        except Exception as e:
            logger.warning(f"⚠️ Notification Manager initialization failed: {e}")
            self.components_available['notification_manager'] = False
            
        # AI Analyzer (optional)
        try:
            if self.config.use_ai_analysis and self.claude_config and AI_ANALYSIS_AVAILABLE:
                self.claude_analyzer = create_claude_analyzer(self.claude_config.api_key)
                await self.claude_analyzer.initialize()
                logger.info("✅ Claude AI Analyzer initialized")
        except Exception as e:
            logger.warning(f"⚠️ AI Analyzer initialization failed: {e}")
            self.components_available['ai_analyzer'] = False
    
    def _create_dummy_risk_manager(self):
        """Create dummy risk manager for fallback"""
        class DummyRiskManager:
            async def check_circuit_breakers(self):
                return {'triggered': False}
            
            async def evaluate_trade_proposal(self, symbol, proposal, context):
                class DummyRiskAssessment:
                    def __init__(self):
                        self.risk_level = 'LOW'
                        self.recommended_position_size = 0.01
                        self.risk_score = 0.3
                return DummyRiskAssessment()
            
            async def get_risk_summary(self):
                return {'risk_metrics': {}, 'circuit_breaker_status': {'status': 'INACTIVE'}}
                
            async def get_risk_health_check(self):
                return {'status': 'healthy', 'warnings': ['Using dummy risk manager']}
        
        return DummyRiskManager()
    
    def _create_dummy_portfolio_manager(self):
        """Create dummy portfolio manager for fallback"""
        class DummyPortfolioManager:
            def __init__(self):
                self.positions = {}
            
            async def initialize(self):
                pass
            
            async def get_portfolio_summary(self):
                class DummySummary:
                    def __init__(self):
                        self.total_value = 10000
                        self.cash_balance = 10000
                        self.invested_amount = 0
                        self.total_positions = 0
                        self.daily_pnl = 0
                        self.total_pnl_pct = 0
                        self.largest_position_pct = 0
                return DummySummary()
            
            async def add_position(self, **kwargs):
                return False  # Cannot add positions in dummy mode
            
            async def close_position(self, symbol, exit_price=None):
                return False
                
            async def check_stop_loss_take_profit(self):
                return []
                
            async def get_portfolio_health_check(self):
                return {'status': 'healthy', 'warnings': ['Using dummy portfolio manager']}
                
            async def save_daily_snapshot(self):
                pass
                
            async def cleanup(self):
                pass
        
        return DummyPortfolioManager()
    
    def _create_dummy_performance_tracker(self):
        """Create dummy performance tracker for fallback"""
        class DummyPerformanceTracker:
            async def initialize(self):
                pass
            
            async def update_portfolio_snapshot(self, **kwargs):
                pass
            
            async def add_completed_trade(self, trade_data):
                pass
            
            async def get_performance_summary(self, period=None):
                return {
                    'current_portfolio_value': 10000,
                    'total_return': 0,
                    'sharpe_ratio': 0,
                    'max_drawdown': 0,
                    'win_rate': 0,
                    'total_trades': 0,
                    'daily_pnl': 0
                }
            
            def get_statistics(self):
                return {
                    'total_snapshots': 0,
                    'total_trades_analyzed': 0,
                    'calculation_count': 0
                }
            
            async def cleanup(self):
                pass
        
        return DummyPerformanceTracker()
    
    def _create_dummy_market_scanner(self):
        """Create dummy market scanner for fallback"""
        class DummyMarketScanner:
            async def initialize(self):
                pass
            
            async def scan_for_opportunities(self, custom_criteria=None):
                # Return empty opportunities
                return []
            
            async def health_check(self):
                return {'exchange_connected': False}
            
            async def cleanup(self):
                pass
        
        return DummyMarketScanner()
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self.shutdown_requested = True
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def get_bot_status(self) -> Dict[str, Any]:
        """Get comprehensive bot status with component availability"""
        try:
            # Portfolio summary
            portfolio_summary = await self.portfolio_manager.get_portfolio_summary()
            
            # Performance summary
            performance_summary = await self.performance_tracker.get_performance_summary()
            
            # Risk summary (with fallback)
            try:
                risk_summary = await self.risk_manager.get_risk_summary()
            except:
                risk_summary = {'risk_metrics': {}, 'circuit_breaker_status': {'status': 'UNKNOWN'}}
            
            status = {
                'bot_state': self.state.value,
                'uptime_hours': self.bot_metrics['uptime_hours'],
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'last_scan_time': self.last_scan_time.isoformat() if self.last_scan_time else None,
                
                # Configuration
                'configuration': {
                    'symbols': self.config.symbols,
                    'max_positions': self.config.max_open_positions,
                    'scan_interval': self.config.scan_interval,
                    'use_ai_analysis': self.config.use_ai_analysis,
                    'testnet': self.config.testnet
                },
                
                # Portfolio status
                'portfolio': {
                    'total_value': portfolio_summary.total_value,
                    'cash_balance': portfolio_summary.cash_balance,
                    'daily_pnl': portfolio_summary.daily_pnl,
                    'total_return': portfolio_summary.total_pnl_pct,
                    'active_positions': portfolio_summary.total_positions,
                    'largest_position_pct': portfolio_summary.largest_position_pct
                },
                
                # Performance metrics
                'performance': {
                    'total_return': performance_summary.get('total_return', 0),
                    'sharpe_ratio': performance_summary.get('sharpe_ratio', 0),
                    'max_drawdown': performance_summary.get('max_drawdown', 0),
                    'win_rate': performance_summary.get('win_rate', 0),
                    'total_trades': performance_summary.get('total_trades', 0)
                },
                
                # Risk metrics
                'risk': {
                    'current_drawdown': risk_summary.get('risk_metrics', {}).get('current_drawdown', 0),
                    'circuit_breaker_status': risk_summary.get('circuit_breaker_status', {}).get('status', 'UNKNOWN'),
                    'consecutive_losses': risk_summary.get('consecutive_losses', 0)
                },
                
                # Bot metrics
                'bot_metrics': self.bot_metrics.copy(),
                
                # Component status (NEW)
                'components': {
                    'strategies': len(self.strategies),
                    'risk_manager': self.components_available['risk_manager'],
                    'portfolio_manager': self.components_available['portfolio_manager'],
                    'performance_tracker': self.components_available['performance_tracker'],
                    'market_scanner': self.components_available['market_scanner'],
                    'notification_manager': self.components_available['notification_manager'],
                    'ai_analyzer': self.components_available['ai_analyzer'],
                    'database': self.components_available['database']
                },
                
                'timestamp': datetime.utcnow().isoformat()
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting bot status: {e}")
            return {'error': str(e), 'components_available': self.components_available}
    
    # Rest of the TradingBot methods would continue here...
    # (The main trading loop, cleanup, etc. would be the same as in the original)
    
    async def run(self):
        """Main bot execution loop with error handling"""
        try:
            logger.info("Starting Trading Bot main execution loop...")
            
            # Check component availability before starting
            critical_components = ['portfolio_manager', 'strategies']
            for component in critical_components:
                if not self.components_available.get(component, False):
                    logger.warning(f"Critical component {component} not available - bot may have limited functionality")
            
            # Start main trading loop
            self.main_loop_task = asyncio.create_task(self._main_trading_loop())
            
            # Start monitoring task
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            # Wait for tasks to complete or shutdown signal
            await asyncio.gather(
                self.main_loop_task,
                self.monitoring_task,
                return_exceptions=True
            )
            
        except Exception as e:
            logger.error(f"Error in main bot execution: {e}")
            self.state = BotState.ERROR
        finally:
            await self._cleanup()
    
    async def _main_trading_loop(self):
        """Main trading logic loop with component availability checks"""
        logger.info("Main trading loop started")
        
        while not self.shutdown_requested and self.state == BotState.RUNNING:
            try:
                loop_start_time = datetime.utcnow()
                
                # Check circuit breakers (if available)
                if self.components_available.get('risk_manager', False):
                    circuit_status = await self.risk_manager.check_circuit_breakers()
                    if circuit_status['triggered']:
                        logger.warning(f"Circuit breaker triggered: {circuit_status.get('reason', 'Unknown')}")
                        await asyncio.sleep(60)
                        continue
                
                # Update portfolio snapshot (if available)
                if self.components_available.get('performance_tracker', False):
                    await self._update_portfolio_snapshot()
                
                # Scan markets and generate signals (if components available)
                if self.components_available.get('market_scanner', False) and len(self.strategies) > 0:
                    await self._scan_and_generate_signals()
                
                # Process pending trades (if portfolio manager available)
                if self.components_available.get('portfolio_manager', False):
                    await self._process_pending_trades()
                    await self._check_position_exits()
                
                # Update bot metrics
                self._update_bot_metrics()
                
                # Calculate loop duration and sleep
                loop_duration = (datetime.utcnow() - loop_start_time).total_seconds()
                sleep_time = max(0, self.config.scan_interval - loop_duration)
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                
                self.last_scan_time = datetime.utcnow()
                
            except Exception as e:
                logger.error(f"Error in main trading loop: {e}")
                await asyncio.sleep(30)
        
        logger.info("Main trading loop stopped")
    
    async def _monitoring_loop(self):
        """System monitoring and maintenance loop"""
        logger.info("Monitoring loop started")
        
        while not self.shutdown_requested and self.state == BotState.RUNNING:
            try:
                # Component health checks
                await self._perform_health_checks()
                
                # Update performance metrics every hour
                if (datetime.utcnow().minute == 0 and 
                    (not self.bot_metrics['last_performance_update'] or
                     datetime.utcnow() - self.bot_metrics['last_performance_update'] > timedelta(hours=1))):
                    
                    await self._update_performance_metrics()
                    await self._send_performance_update()
                
                # Sleep for 5 minutes
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)
        
        logger.info("Monitoring loop stopped")
    
    # Add placeholder methods for the remaining functionality
    async def _update_portfolio_snapshot(self):
        """Update portfolio snapshot for performance tracking"""
        if not self.components_available.get('performance_tracker', False):
            return
            
        try:
            summary = await self.portfolio_manager.get_portfolio_summary()
            await self.performance_tracker.update_portfolio_snapshot(
                portfolio_value=summary.total_value,
                cash_balance=summary.cash_balance,
                invested_amount=summary.invested_amount,
                active_positions=summary.total_positions,
                daily_pnl=summary.daily_pnl
            )
        except Exception as e:
            logger.error(f"Error updating portfolio snapshot: {e}")
    
    async def _scan_and_generate_signals(self):
        """Placeholder for signal generation"""
        # Implementation would be similar to original but with component checks
        pass
    
    async def _process_pending_trades(self):
        """Placeholder for trade processing"""
        # Implementation would be similar to original but with component checks
        pass
    
    async def _check_position_exits(self):
        """Placeholder for position exit checks"""
        # Implementation would be similar to original but with component checks
        pass
    
    def _update_bot_metrics(self):
        """Update bot performance metrics"""
        try:
            if self.start_time:
                self.bot_metrics['uptime_hours'] = (datetime.utcnow() - self.start_time).total_seconds() / 3600
            self.bot_metrics['component_status'] = self.components_available.copy()
        except Exception as e:
            logger.error(f"Error updating bot metrics: {e}")
    
    async def _perform_health_checks(self):
        """Perform system health checks with component availability"""
        try:
            health_issues = []
            
            # Check each component if available
            for component_name, is_available in self.components_available.items():
                if not is_available:
                    health_issues.append(f"{component_name} not available")
            
            # Report issues if notification manager is available
            if health_issues and self.components_available.get('notification_manager', False):
                try:
                    await self.notification_manager.send_custom_notification(
                        title="⚠️ Component Health Issues",
                        message=f"Some components unavailable:\n" + "\n".join(f"• {issue}" for issue in health_issues[:5]),
                        channels=[],
                        priority=NotificationPriority.MEDIUM
                    )
                except Exception as e:
                    logger.warning(f"Could not send health notification: {e}")
                    
        except Exception as e:
            logger.error(f"Error performing health checks: {e}")
    
    async def _update_performance_metrics(self):
        """Update performance metrics"""
        if not self.components_available.get('portfolio_manager', False):
            return
            
        try:
            await self.portfolio_manager.save_daily_snapshot()
            self.bot_metrics['last_performance_update'] = datetime.utcnow()
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    async def _send_performance_update(self):
        """Send performance update notification"""
        if not self.components_available.get('notification_manager', False):
            return
            
        try:
            summary = await self.performance_tracker.get_performance_summary()
            
            await self.notification_manager.send_custom_notification(
                title="📊 Performance Update (Limited Mode)" if sum(self.components_available.values()) < len(self.components_available) else "📊 Performance Update",
                message=f"Portfolio: ${summary['current_portfolio_value']:,.2f}\n"
                       f"Daily P&L: ${summary.get('daily_pnl', 0):+,.2f}\n"
                       f"Total Return: {summary['total_return']:+.2f}%\n"
                       f"Components: {sum(self.components_available.values())}/{len(self.components_available)} available",
                channels=[],
                priority=NotificationPriority.LOW
            )
        except Exception as e:
            logger.error(f"Error sending performance update: {e}")
    
    async def _cleanup(self):
        """Cleanup resources and shut down gracefully"""
        try:
            logger.info("Starting bot cleanup...")
            self.state = BotState.STOPPING
            
            # Cancel running tasks
            if self.main_loop_task and not self.main_loop_task.done():
                self.main_loop_task.cancel()
                try:
                    await self.main_loop_task
                except asyncio.CancelledError:
                    pass
            
            if self.monitoring_task and not self.monitoring_task.done():
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass
            
            # Cleanup components (only if available)
            cleanup_tasks = []
            
            if self.portfolio_manager and self.components_available.get('portfolio_manager', False):
                cleanup_tasks.append(self.portfolio_manager.cleanup())
            
            if self.performance_tracker and self.components_available.get('performance_tracker', False):
                cleanup_tasks.append(self.performance_tracker.cleanup())
            
            if self.market_scanner and self.components_available.get('market_scanner', False):
                cleanup_tasks.append(self.market_scanner.cleanup())
            
            if self.notification_manager and self.components_available.get('notification_manager', False):
                cleanup_tasks.append(self.notification_manager.cleanup())
            
            if self.claude_analyzer and self.components_available.get('ai_analyzer', False):
                cleanup_tasks.append(self.claude_analyzer.cleanup())
            
            if self.exchange:
                cleanup_tasks.append(self.exchange.close())
            
            # Wait for all cleanup tasks
            if cleanup_tasks:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            
            # Send shutdown notification
            if self.notification_manager and self.components_available.get('notification_manager', False) and self.state != BotState.ERROR:
                try:
                    uptime = self.bot_metrics['uptime_hours']
                    await self.notification_manager.send_custom_notification(
                        title="🛑 Trading Bot Shutdown",
                        message=f"Trading bot shut down gracefully.\n"
                               f"Uptime: {uptime:.1f} hours\n"
                               f"Import errors: {self.bot_metrics['import_errors']}\n"
                               f"Components available: {sum(self.components_available.values())}/{len(self.components_available)}",
                        channels=[],
                        priority=NotificationPriority.MEDIUM
                    )
                except:
                    pass
            
            self.state = BotState.STOPPED
            logger.info("Bot cleanup completed successfully")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            self.state = BotState.ERROR

# Factory function with error handling
def create_trading_bot(config: BotConfiguration, api_config, claude_config=None) -> TradingBot:
    """Factory function to create Trading Bot instance with error handling"""
    try:
        return TradingBot(config, api_config, claude_config)
    except Exception as e:
        logger.error(f"Error creating trading bot: {e}")
        raise

# Example configuration classes
class ExampleApiConfig:
    def __init__(self):
        # Exchange API
        self.binance_api_key = "your_binance_api_key"
        self.binance_secret = "your_binance_secret"
        self.binance_testnet = True
        
        # Notification APIs
        self.discord_webhook_url = "your_discord_webhook"
        self.telegram_bot_token = "your_telegram_bot_token"
        self.telegram_chat_id = "your_telegram_chat_id"
        self.email_user = "your_email@gmail.com"
        self.email_password = "your_app_password"
        self.to_emails = ["recipient@gmail.com"]

class ExampleClaudeConfig:
    def __init__(self):
        self.api_key = "your_claude_api_key"

# Example usage with error handling
async def main():
    """Example main function with comprehensive error handling"""
    try:
        print("🤖 Trading Bot with Import Error Handling")
        print("=" * 50)
        
        # Configuration
        bot_config = BotConfiguration(
            initial_capital=10000.0,
            max_open_positions=5,
            symbols=['BTC/USDT', 'ETH/USDT', 'ADA/USDT'],
            testnet=True,
            use_ai_analysis=True
        )
        
        api_config = ExampleApiConfig()
        claude_config = ExampleClaudeConfig()
        
        # Create and initialize bot
        bot = create_trading_bot(bot_config, api_config, claude_config)
        await bot.initialize()
        
        print("🤖 Trading Bot initialized successfully!")
        print(f"📊 Strategies: {list(bot.strategies.keys())}")
        print(f"💱 Symbols: {bot.config.symbols}")
        print(f"💰 Initial capital: ${bot.config.initial_capital:,.2f}")
        print(f"🔧 Component availability: {sum(bot.components_available.values())}/{len(bot.components_available)}")
        
        # Show component status
        print(f"\n🔧 Component Status:")
        for component, available in bot.components_available.items():
            status = "✅" if available else "❌"
            print(f"  {status} {component}")
        
        # Get initial status
        status = await bot.get_bot_status()
        print(f"\n📈 Initial Status:")
        print(f"  Portfolio Value: ${status['portfolio']['total_value']:,.2f}")
        print(f"  Cash Balance: ${status['portfolio']['cash_balance']:,.2f}")
        print(f"  Active Positions: {status['portfolio']['active_positions']}")
        print(f"  Import Errors: {status['bot_metrics']['import_errors']}")
        
        # Show available strategies
        if bot.strategies:
            print(f"\n🎯 Available Strategies:")
            for strategy_name in bot.strategies.keys():
                print(f"  • {strategy_name}")
        else:
            print(f"\n⚠️ No strategies available")
        
        print(f"\n🚀 Bot ready to start trading (with available components)...")
        
        # Note: In a real implementation, you would call await bot.run() here
        # For this example, we'll just show the setup
        
    except KeyboardInterrupt:
        print(f"\n⏹️ Shutdown requested by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Error in main: {e}")
    finally:
        try:
            if 'bot' in locals():
                await bot._cleanup()
        except:
            pass

if __name__ == "__main__":
    # Run the trading bot with error handling
    asyncio.run(main())
