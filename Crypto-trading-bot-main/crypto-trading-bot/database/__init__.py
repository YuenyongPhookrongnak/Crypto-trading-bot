"""
Database Package

This package contains database connection management, models, and utilities
for the trading bot system.
"""

# Import main components
try:
    from .connection import (
        DatabaseManager, 
        db_manager, 
        init_database, 
        get_db_session, 
        get_sync_db_session,
        execute_query, 
        execute_command, 
        close_database,
        auto_init_database,
        Base
    )
    CONNECTION_AVAILABLE = True
except ImportError:
    CONNECTION_AVAILABLE = False

try:
    from .models import (
        Trade, 
        Signal, 
        PortfolioSnapshot, 
        SystemLog, 
        StrategyPerformance,
        MarketData,
        RiskEvent,
        LogLevel,
        TradeDirection,
        TradeStatus,
        SignalType,
        create_trade_record,
        update_trade_record,
        create_signal_record,
        create_portfolio_snapshot,
        log_system_message,
        get_trades_by_strategy,
        get_recent_signals,
        get_portfolio_history,
        MODEL_REGISTRY,
        get_model_class,
        DATABASE_MODELS_AVAILABLE
    )
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False

# Package info
DATABASE_PACKAGE_AVAILABLE = CONNECTION_AVAILABLE and MODELS_AVAILABLE

# Version info
__version__ = "1.0.0"
__author__ = "Trading Bot Team"

# Export availability flags and main components
__all__ = [
    'CONNECTION_AVAILABLE',
    'MODELS_AVAILABLE', 
    'DATABASE_PACKAGE_AVAILABLE'
]

# Add available components to __all__
if CONNECTION_AVAILABLE:
    __all__.extend([
        'DatabaseManager',
        'db_manager',
        'init_database',
        'get_db_session',
        'get_sync_db_session', 
        'execute_query',
        'execute_command',
        'close_database',
        'auto_init_database',
        'Base'
    ])

if MODELS_AVAILABLE:
    __all__.extend([
        'Trade',
        'Signal', 
        'PortfolioSnapshot',
        'SystemLog',
        'StrategyPerformance',
        'MarketData',
        'RiskEvent',
        'LogLevel',
        'TradeDirection', 
        'TradeStatus',
        'SignalType',
        'create_trade_record',
        'update_trade_record',
        'create_signal_record', 
        'create_portfolio_snapshot',
        'log_system_message',
        'get_trades_by_strategy',
        'get_recent_signals',
        'get_portfolio_history',
        'MODEL_REGISTRY',
        'get_model_class',
        'DATABASE_MODELS_AVAILABLE'
    ])