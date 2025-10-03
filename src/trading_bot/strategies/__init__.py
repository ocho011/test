"""Trading strategy framework with plugin architecture."""

from .base_strategy import AbstractStrategy
from .ict_strategy import ICTStrategy
from .traditional_strategy import TraditionalIndicatorStrategy
from .strategy_registry import StrategyRegistry, get_registry
from .strategy_selector import StrategySelector
from .strategy_coordinator import StrategyCoordinator
from .strategy_system_integration import IntegratedStrategySystem
from .performance_tracker import StrategyPerformanceTracker, StrategyMetrics, TradeRecord
from .comparator import StrategyComparator, ComparisonResult
from .strategy_optimizer import (
    StrategyOptimizer,
    OptimizationConfig,
    OptimizedStrategySystem,
    CachedOptimization,
)

__all__ = [
    # Base
    "AbstractStrategy",
    # Concrete Strategies
    "ICTStrategy",
    "TraditionalIndicatorStrategy",
    # Registry
    "StrategyRegistry",
    "get_registry",
    # Selector
    "StrategySelector",
    # Coordination
    "StrategyCoordinator",
    "IntegratedStrategySystem",
    # Performance Tracking
    "StrategyPerformanceTracker",
    "StrategyMetrics",
    "TradeRecord",
    # Comparison
    "StrategyComparator",
    "ComparisonResult",
    # Optimization
    "StrategyOptimizer",
    "OptimizationConfig",
    "OptimizedStrategySystem",
    "CachedOptimization",
]
