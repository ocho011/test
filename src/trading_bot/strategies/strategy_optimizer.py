"""
Strategy parameter optimization with walk-forward analysis integration.

This module integrates the WalkForwardOptimizer with the strategy framework,
providing parameter tuning, overfitting prevention, and optimization result caching.
"""

import asyncio
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from decimal import Decimal

import pandas as pd
import numpy as np

from ..config import get_logger
from ..backtesting.walk_forward_optimizer import WalkForwardOptimizer, OptimizationResult
from ..backtesting.backtest_engine import BacktestEngine
from ..backtesting.performance_analyzer import PerformanceAnalyzer
from .base_strategy import AbstractStrategy
from .performance_tracker import StrategyPerformanceTracker, StrategyMetrics

if TYPE_CHECKING:
    from .strategy_registry import StrategyRegistry

logger = get_logger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for strategy optimization."""
    
    in_sample_ratio: float = 0.7
    window_size_days: int = 180
    step_size_days: int = 30
    optimization_metric: str = "sharpe_ratio"
    max_iterations: int = 100
    convergence_threshold: float = 0.001
    enable_caching: bool = True
    cache_dir: str = ".cache/optimizations"


@dataclass
class CachedOptimization:
    """Cached optimization result."""
    
    strategy_name: str
    parameters: Dict[str, Any]
    optimization_result: Dict[str, Any]
    timestamp: str
    data_hash: str
    config: Dict[str, Any]


class StrategyOptimizer:
    """
    Parameter optimizer for trading strategies with walk-forward analysis.
    
    Features:
    - Integration with WalkForwardOptimizer
    - Automatic parameter grid generation from strategy schemas
    - Optimization result caching
    - Overfitting detection and prevention
    - Parameter versioning
    - Multi-strategy comparison
    """
    
    def __init__(
        self,
        backtest_engine: BacktestEngine,
        performance_analyzer: PerformanceAnalyzer,
        config: Optional[OptimizationConfig] = None,
    ):
        """
        Initialize strategy optimizer.
        
        Args:
            backtest_engine: BacktestEngine instance for running backtests
            performance_analyzer: PerformanceAnalyzer for metric calculation
            config: Optimization configuration (uses defaults if None)
        """
        self.backtest_engine = backtest_engine
        self.performance_analyzer = performance_analyzer
        self.config = config or OptimizationConfig()
        
        # Initialize walk-forward optimizer
        self.walk_forward_optimizer = WalkForwardOptimizer(
            backtest_engine=backtest_engine,
            performance_analyzer=performance_analyzer,
            in_sample_ratio=self.config.in_sample_ratio,
            window_size_days=self.config.window_size_days,
            step_size_days=self.config.step_size_days,
        )
        
        # Optimization cache
        self._cache: Dict[str, CachedOptimization] = {}
        self._cache_dir = Path(self.config.cache_dir)
        if self.config.enable_caching:
            self._load_cache()
        
        logger.info(f"StrategyOptimizer initialized with config: {self.config}")
    
    async def optimize_strategy(
        self,
        strategy: AbstractStrategy,
        parameter_grid: Optional[Dict[str, List]] = None,
        symbol: str = "BTCUSDT",
        force_reoptimize: bool = False,
    ) -> OptimizationResult:
        """
        Optimize strategy parameters using walk-forward analysis.
        
        Args:
            strategy: Strategy instance to optimize
            parameter_grid: Parameter grid (auto-generated if None)
            symbol: Trading symbol
            force_reoptimize: Skip cache check and force re-optimization
            
        Returns:
            OptimizationResult with best parameters
        """
        strategy_name = strategy.__class__.__name__
        
        # Generate parameter grid from strategy schema if not provided
        if parameter_grid is None:
            parameter_grid = self._generate_parameter_grid(strategy)
        
        # Check cache
        if not force_reoptimize and self.config.enable_caching:
            cached = self._check_cache(strategy_name, parameter_grid, symbol)
            if cached:
                logger.info(f"Using cached optimization for {strategy_name}")
                return self._result_from_cache(cached)
        
        logger.info(f"Optimizing {strategy_name} with {len(parameter_grid)} parameters")
        
        # Run walk-forward optimization
        result = await self.walk_forward_optimizer.optimize_grid_search(
            parameter_grid=parameter_grid,
            optimization_metric=self.config.optimization_metric,
            symbol=symbol,
        )
        
        # Check for overfitting
        overfitting_analysis = self.walk_forward_optimizer.detect_overfitting(
            in_sample_metrics=result.in_sample_metrics,
            out_sample_metrics=result.out_sample_metrics,
        )
        
        if overfitting_analysis["is_overfitted"]:
            logger.warning(
                f"Overfitting detected for {strategy_name}: "
                f"{overfitting_analysis['metric_analysis']}"
            )
        
        # Cache result
        if self.config.enable_caching:
            self._cache_result(strategy_name, parameter_grid, symbol, result)
        
        logger.info(
            f"Optimization complete for {strategy_name}. "
            f"Best {self.config.optimization_metric}: "
            f"IS={result.in_sample_metrics.get(self.config.optimization_metric, 0):.4f}, "
            f"OOS={result.out_sample_metrics.get(self.config.optimization_metric, 0):.4f}, "
            f"Stability={result.stability_score:.4f}"
        )
        
        return result
    
    def _generate_parameter_grid(self, strategy: AbstractStrategy) -> Dict[str, List]:
        """
        Generate parameter grid from strategy parameter schema.
        
        Args:
            strategy: Strategy instance
            
        Returns:
            Parameter grid for optimization
        """
        schema = strategy.get_parameter_schema()
        grid = {}
        
        for param_name, param_info in schema.items():
            param_type = param_info.get("type")
            
            if param_type == "int":
                min_val = param_info.get("min", 5)
                max_val = param_info.get("max", 50)
                step = param_info.get("step", 5)
                grid[param_name] = list(range(min_val, max_val + 1, step))
                
            elif param_type == "float":
                min_val = param_info.get("min", 0.1)
                max_val = param_info.get("max", 1.0)
                step = param_info.get("step", 0.1)
                grid[param_name] = list(np.arange(min_val, max_val + step, step))
                
            elif param_type == "bool":
                grid[param_name] = [True, False]
                
            elif param_type == "str":
                # Use allowed values if specified
                allowed = param_info.get("allowed", [])
                if allowed:
                    grid[param_name] = allowed
        
        logger.info(f"Generated parameter grid with {len(grid)} parameters")
        return grid
    
    def _check_cache(
        self,
        strategy_name: str,
        parameter_grid: Dict[str, List],
        symbol: str,
    ) -> Optional[CachedOptimization]:
        """Check if optimization result is cached."""
        cache_key = self._generate_cache_key(strategy_name, parameter_grid, symbol)
        return self._cache.get(cache_key)
    
    def _cache_result(
        self,
        strategy_name: str,
        parameter_grid: Dict[str, List],
        symbol: str,
        result: OptimizationResult,
    ):
        """Cache optimization result."""
        cache_key = self._generate_cache_key(strategy_name, parameter_grid, symbol)
        
        cached = CachedOptimization(
            strategy_name=strategy_name,
            parameters=result.parameters,
            optimization_result={
                "in_sample_metrics": result.in_sample_metrics,
                "out_sample_metrics": result.out_sample_metrics,
                "stability_score": result.stability_score,
            },
            timestamp=datetime.now().isoformat(),
            data_hash=self._compute_data_hash(),
            config=asdict(self.config),
        )
        
        self._cache[cache_key] = cached
        self._save_cache()
    
    def _generate_cache_key(
        self,
        strategy_name: str,
        parameter_grid: Dict[str, List],
        symbol: str,
    ) -> str:
        """Generate cache key from inputs."""
        grid_str = json.dumps(parameter_grid, sort_keys=True)
        return f"{strategy_name}_{symbol}_{hash(grid_str)}"
    
    def _compute_data_hash(self) -> str:
        """Compute hash of current historical data."""
        if self.backtest_engine.historical_data is None:
            return "no_data"
        
        df = self.backtest_engine.historical_data
        # Use first/last timestamp and length as simple hash
        return f"{df.index[0]}_{df.index[-1]}_{len(df)}"
    
    def _load_cache(self):
        """Load optimization cache from disk."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self._cache_dir / "optimization_cache.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    for key, value in data.items():
                        self._cache[key] = CachedOptimization(**value)
                logger.info(f"Loaded {len(self._cache)} cached optimizations")
            except Exception as e:
                logger.error(f"Error loading cache: {e}")
    
    def _save_cache(self):
        """Save optimization cache to disk."""
        cache_file = self._cache_dir / "optimization_cache.json"
        
        try:
            data = {
                key: asdict(value)
                for key, value in self._cache.items()
            }
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
    
    def _result_from_cache(self, cached: CachedOptimization) -> OptimizationResult:
        """Convert cached optimization to OptimizationResult."""
        return OptimizationResult(
            parameters=cached.parameters,
            in_sample_metrics=cached.optimization_result["in_sample_metrics"],
            out_sample_metrics=cached.optimization_result["out_sample_metrics"],
            stability_score=cached.optimization_result["stability_score"],
        )
    
    def clear_cache(self):
        """Clear optimization cache."""
        self._cache.clear()
        cache_file = self._cache_dir / "optimization_cache.json"
        if cache_file.exists():
            cache_file.unlink()
        logger.info("Optimization cache cleared")
    
    async def compare_parameter_sets(
        self,
        strategy: AbstractStrategy,
        parameter_sets: List[Dict[str, Any]],
        symbol: str = "BTCUSDT",
    ) -> List[OptimizationResult]:
        """
        Compare multiple parameter sets for a strategy.
        
        Args:
            strategy: Strategy instance
            parameter_sets: List of parameter dictionaries to compare
            symbol: Trading symbol
            
        Returns:
            List of OptimizationResults, sorted by performance
        """
        results = []
        
        for params in parameter_sets:
            # Create parameter grid with single values
            grid = {key: [value] for key, value in params.items()}
            
            result = await self.walk_forward_optimizer.optimize_grid_search(
                parameter_grid=grid,
                optimization_metric=self.config.optimization_metric,
                symbol=symbol,
            )
            results.append(result)
        
        # Sort by combined score (IS + OOS weighted by stability)
        results.sort(
            key=lambda r: (
                r.in_sample_metrics.get(self.config.optimization_metric, 0) * 0.3 +
                r.out_sample_metrics.get(self.config.optimization_metric, 0) * 0.7
            ) * r.stability_score,
            reverse=True,
        )
        
        return results
    
    def get_optimization_summary(self, result: OptimizationResult) -> Dict[str, Any]:
        """
        Get human-readable optimization summary.
        
        Args:
            result: OptimizationResult to summarize
            
        Returns:
            Summary dictionary
        """
        return {
            "best_parameters": result.parameters,
            "in_sample_performance": result.in_sample_metrics,
            "out_of_sample_performance": result.out_sample_metrics,
            "stability_score": result.stability_score,
            "overfitting_risk": "High" if result.stability_score < 0.7 else "Low",
            "recommendation": (
                "Parameters look good for production" if result.stability_score >= 0.7
                else "Consider re-optimization or simpler parameters"
            ),
        }


# Enhanced IntegratedStrategySystem with optimization
class OptimizedStrategySystem:
    """
    Complete strategy system with optimization capabilities.
    
    Combines strategy registry, selector, performance tracking, comparison,
    and parameter optimization in a unified interface.
    """
    
    def __init__(
        self,
        backtest_engine: BacktestEngine,
        performance_analyzer: PerformanceAnalyzer,
    ):
        """
        Initialize optimized strategy system.
        
        Args:
            backtest_engine: BacktestEngine for optimization
            performance_analyzer: PerformanceAnalyzer for metrics
        """
        from .strategy_registry import get_registry
        from .strategy_selector import StrategySelector
        from .performance_tracker import StrategyPerformanceTracker
        from .comparator import StrategyComparator
        
        self.registry = get_registry()
        self.registry.auto_register_builtin_strategies()
        self.selector = StrategySelector()
        self.performance_tracker = StrategyPerformanceTracker()
        self.comparator = StrategyComparator(self.performance_tracker)
        self.optimizer = StrategyOptimizer(
            backtest_engine=backtest_engine,
            performance_analyzer=performance_analyzer,
        )
        
        logger.info("OptimizedStrategySystem initialized")
    
    async def optimize_and_deploy(
        self,
        strategy_name: str,
        symbol: str = "BTCUSDT",
        parameter_grid: Optional[Dict[str, List]] = None,
    ) -> AbstractStrategy:
        """
        Optimize strategy parameters and deploy with best parameters.
        
        Args:
            strategy_name: Name of strategy to optimize
            symbol: Trading symbol
            parameter_grid: Optional parameter grid
            
        Returns:
            Configured strategy with optimized parameters
        """
        # Create strategy with default parameters
        strategy = self.registry.create_strategy(strategy_name)
        
        # Optimize parameters
        result = await self.optimizer.optimize_strategy(
            strategy=strategy,
            parameter_grid=parameter_grid,
            symbol=symbol,
        )
        
        # Log optimization summary
        summary = self.optimizer.get_optimization_summary(result)
        logger.info(f"Optimization summary for {strategy_name}: {summary}")
        
        # Set as active strategy with optimized parameters
        await self.selector.set_strategy(
            strategy_name=strategy_name,
            parameters=result.parameters,
        )

        # Get the configured strategy instance
        optimized_strategy = self.selector.current_strategy
        
        logger.info(f"Deployed optimized {strategy_name} with parameters: {result.parameters}")
        
        return optimized_strategy
    
    async def run_optimization_comparison(
        self,
        strategy_names: List[str],
        symbol: str = "BTCUSDT",
        parameter_grids: Optional[Dict[str, Dict[str, List]]] = None,
    ) -> Dict[str, OptimizationResult]:
        """
        Optimize and compare multiple strategies.

        Args:
            strategy_names: List of strategy names to optimize
            symbol: Trading symbol
            parameter_grids: Optional dict mapping strategy names to parameter grids

        Returns:
            Dictionary mapping strategy names to OptimizationResults
        """
        results = {}

        for strategy_name in strategy_names:
            strategy = self.registry.create_strategy(strategy_name)
            grid = parameter_grids.get(strategy_name) if parameter_grids else None
            result = await self.optimizer.optimize_strategy(
                strategy=strategy,
                parameter_grid=grid,
                symbol=symbol,
            )
            results[strategy_name] = result
        
        # Log comparison
        logger.info("Optimization Comparison:")
        for name, result in results.items():
            metric = self.optimizer.config.optimization_metric
            logger.info(
                f"  {name}: "
                f"IS={result.in_sample_metrics.get(metric, 0):.4f}, "
                f"OOS={result.out_sample_metrics.get(metric, 0):.4f}, "
                f"Stability={result.stability_score:.4f}"
            )
        
        return results
