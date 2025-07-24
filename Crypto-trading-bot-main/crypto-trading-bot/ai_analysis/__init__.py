"""
AI Analysis Package

This package contains AI-powered analysis tools for the trading bot,
including Claude integration and other AI services.
"""

# Import AI analysis components
try:
    from .claude_analyzer import (
        ClaudeAnalyzer,
        create_claude_analyzer,
        MarketAnalysis,
        TradingRecommendation
    )
    CLAUDE_ANALYZER_AVAILABLE = True
except ImportError:
    CLAUDE_ANALYZER_AVAILABLE = False

# AI Analysis availability
AI_ANALYSIS_AVAILABLE = CLAUDE_ANALYZER_AVAILABLE

# Version info
__version__ = "1.0.0"
__author__ = "Trading Bot Team"

# Export components
__all__ = [
    'CLAUDE_ANALYZER_AVAILABLE',
    'AI_ANALYSIS_AVAILABLE'
]

if CLAUDE_ANALYZER_AVAILABLE:
    __all__.extend([
        'ClaudeAnalyzer',
        'create_claude_analyzer', 
        'MarketAnalysis',
        'TradingRecommendation'
    ])

def get_available_ai_tools():
    """Get list of available AI analysis tools"""
    tools = []
    if CLAUDE_ANALYZER_AVAILABLE:
        tools.append('claude_analyzer')
    return tools

def is_ai_tool_available(tool_name: str) -> bool:
    """Check if specific AI tool is available"""
    availability_map = {
        'claude_analyzer': CLAUDE_ANALYZER_AVAILABLE
    }
    return availability_map.get(tool_name, False)