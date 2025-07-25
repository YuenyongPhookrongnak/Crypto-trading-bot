"""
Utils Package

This package contains utility modules for the trading bot including:
- Risk management
- Portfolio management  
- Performance tracking
- Market scanning
- Notification management
"""

# Import main utility classes and functions
try:
    from .risk_manager import RiskManager, create_risk_manager, RiskLevel, RiskAssessment
    RISK_MANAGER_AVAILABLE = True
except ImportError:
    RISK_MANAGER_AVAILABLE = False

try:
    from .portfolio_manager import PortfolioManager, create_portfolio_manager, PositionDirection
    PORTFOLIO_MANAGER_AVAILABLE = True
except ImportError:
    PORTFOLIO_MANAGER_AVAILABLE = False

try:
    from .performance_tracker import PerformanceTracker, create_performance_tracker, PerformancePeriod
    PERFORMANCE_TRACKER_AVAILABLE = True
except ImportError:
    PERFORMANCE_TRACKER_AVAILABLE = False

try:
    from .market_scanner import MarketScanner, create_market_scanner
    MARKET_SCANNER_AVAILABLE = True
except ImportError:
    MARKET_SCANNER_AVAILABLE = False

try:
    from .notification_manager import NotificationManager, create_notification_manager, NotificationType, NotificationPriority
    NOTIFICATION_MANAGER_AVAILABLE = True
except ImportError:
    NOTIFICATION_MANAGER_AVAILABLE = False

# Utility registry
UTILITY_REGISTRY = {}

if RISK_MANAGER_AVAILABLE:
    UTILITY_REGISTRY['risk_manager'] = create_risk_manager

if PORTFOLIO_MANAGER_AVAILABLE:
    UTILITY_REGISTRY['portfolio_manager'] = create_portfolio_manager

if PERFORMANCE_TRACKER_AVAILABLE:
    UTILITY_REGISTRY['performance_tracker'] = create_performance_tracker

if MARKET_SCANNER_AVAILABLE:
    UTILITY_REGISTRY['market_scanner'] = create_market_scanner

if NOTIFICATION_MANAGER_AVAILABLE:
    UTILITY_REGISTRY['notification_manager'] = create_notification_manager

def get_available_utilities():
    """Get list of available utilities"""
    return list(UTILITY_REGISTRY.keys())

def is_utility_available(utility_name: str) -> bool:
    """Check if a utility is available"""
    availability_map = {
        'risk_manager': RISK_MANAGER_AVAILABLE,
        'portfolio_manager': PORTFOLIO_MANAGER_AVAILABLE,
        'performance_tracker': PERFORMANCE_TRACKER_AVAILABLE,
        'market_scanner': MARKET_SCANNER_AVAILABLE,
        'notification_manager': NOTIFICATION_MANAGER_AVAILABLE
    }
    return availability_map.get(utility_name, False)

# Export availability flags
__all__ = [
    'RISK_MANAGER_AVAILABLE',
    'PORTFOLIO_MANAGER_AVAILABLE', 
    'PERFORMANCE_TRACKER_AVAILABLE',
    'MARKET_SCANNER_AVAILABLE',
    'NOTIFICATION_MANAGER_AVAILABLE',
    'UTILITY_REGISTRY',
    'get_available_utilities',
    'is_utility_available'
]

# Version info
__version__ = "1.0.0"
__author__ = "Trading Bot Team"