"""
ICT (Inner Circle Trader) Technical Analysis Engine

This module provides comprehensive ICT trading analysis with the following components:

Main Classes:
- TechnicalAnalyzer: Main analysis engine integrating all components
- OrderBlockDetector: Institutional order block detection
- FairValueGapAnalyzer: Fair value gap identification and tracking
- MarketStructureAnalyzer: Market structure analysis (BOS/CHoCH)
- TimeFrameManager: Multi-timeframe analysis coordination
- PatternValidationEngine: Pattern validation and reliability assessment
- TechnicalIndicators: Traditional technical indicators
- ICTIndicatorIntegration: ICT and indicator integration
- ICTPatternIndicatorFusion: Advanced pattern validation with indicators

Usage:
    from trading_bot.analysis import TechnicalAnalyzer

    # Initialize analyzer
    analyzer = TechnicalAnalyzer(
        enable_validation=True,
        enable_parallel_processing=True,
        memory_optimization=True
    )

    # Perform comprehensive analysis
    result = analyzer.analyze_comprehensive(ohlc_data, timeframe="1H")

    # Get real-time signals
    signals = analyzer.get_real_time_signals(latest_data)

    # Access individual components
    order_blocks = result.order_blocks
    fair_value_gaps = result.fair_value_gaps
    market_structures = result.market_structures
    indicators = result.indicators
    pattern_validations = result.pattern_validations
"""

# ICT Pattern Analysis
from .ict_analyzer import (  # Core analyzers; Data classes; Enums
    BacktestResult,
    FairValueGap,
    FairValueGapAnalyzer,
    MarketStructure,
    MarketStructureAnalyzer,
    OrderBlock,
    OrderBlockDetector,
    PatternType,
    PatternValidationEngine,
    PatternValidationResult,
    SwingPoint,
    TimeFrameManager,
    TrendDirection,
)

# Main analyzer
from .main_analyzer import AnalysisResult, PerformanceMetrics, TechnicalAnalyzer

# Technical Indicators
from .technical_indicators import (
    ICTIndicatorIntegration,
    ICTPatternIndicatorFusion,
    TechnicalIndicators,
)

__all__ = [
    # Main analyzer
    "TechnicalAnalyzer",
    "AnalysisResult",
    "PerformanceMetrics",
    # ICT analyzers
    "OrderBlockDetector",
    "FairValueGapAnalyzer",
    "MarketStructureAnalyzer",
    "TimeFrameManager",
    "PatternValidationEngine",
    # Data classes
    "OrderBlock",
    "FairValueGap",
    "MarketStructure",
    "SwingPoint",
    "PatternValidationResult",
    "BacktestResult",
    # Enums
    "PatternType",
    "TrendDirection",
    # Technical indicators
    "TechnicalIndicators",
    "ICTIndicatorIntegration",
    "ICTPatternIndicatorFusion",
]

__version__ = "1.0.0"
__author__ = "ICT Analysis Team"
__description__ = "Comprehensive ICT (Inner Circle Trader) Technical Analysis Engine"
