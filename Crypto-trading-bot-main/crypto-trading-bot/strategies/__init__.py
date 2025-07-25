"""
Trading Strategies Package

This package contains all trading strategies for the cryptocurrency trading bot.
"""

from .base_strategy import BaseStrategy, TradingSignal
from .momentum_strategy import MomentumStrategy, create_momentum_strategy
from .rsi_strategy import RSIStrategy, create_rsi_strategy
from .volume_profile_strategy import VolumeProfileStrategy, create_volume_profile_strategy
from .pairs_trading_strategy import PairsTradingStrategy, create_pairs_trading_strategy
from .grid_trading_strategy import GridTradingStrategy, create_grid_trading_strategy

# Strategy registry for dynamic loading
STRATEGY_REGISTRY = {
    'momentum': MomentumStrategy,
    'rsi': RSIStrategy,
    'volume_profile': VolumeProfileStrategy,
    'pairs_trading': PairsTradingStrategy,
    'grid_trading': GridTradingStrategy
}

# Factory functions registry
STRATEGY_FACTORIES = {
    'momentum': create_momentum_strategy,
    'rsi': create_rsi_strategy,
    'volume_profile': create_volume_profile_strategy,
    'pairs_trading': create_pairs_trading_strategy,
    'grid_trading': create_grid_trading_strategy
}

def get_strategy_class(strategy_name: str):
    """Get strategy class by name"""
    return STRATEGY_REGISTRY.get(strategy_name.lower())

def create_strategy(strategy_name: str, config: dict):
    """Create strategy instance by name"""
    factory = STRATEGY_FACTORIES.get(strategy_name.lower())
    if factory:
        return factory(config)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

def list_available_strategies():
    """List all available strategies"""
    return list(STRATEGY_REGISTRY.keys())

# Version info
__version__ = "1.0.0"
__author__ = "Trading Bot Team"

# Export all
__all__ = [
    'BaseStrategy',
    'TradingSignal',
    'MomentumStrategy',
    'RSIStrategy',
    'VolumeProfileStrategy',
    'PairsTradingStrategy',
    'GridTradingStrategy',
    'STRATEGY_REGISTRY',
    'STRATEGY_FACTORIES',
    'get_strategy_class',
    'create_strategy',
    'list_available_strategies'
]