"""
Complete strategy framework integration example.

Demonstrates the full capabilities of the strategy abstraction framework
including plugin management, runtime switching, performance tracking,
and A/B testing.
"""

import asyncio
import logging
from typing import Dict, List, Optional
import pandas as pd

from .base_strategy import AbstractStrategy
from .ict_strategy import ICTStrategy
from .traditional_strategy import TraditionalIndicatorStrategy
from .strategy_registry import get_registry
from .strategy_selector import StrategySelector
from .performance_tracker import StrategyPerformanceTracker
from .comparator import StrategyComparator


class IntegratedStrategySystem:
    """
    Complete integrated strategy management system.

    Provides end-to-end strategy lifecycle management:
    - Strategy registration and discovery
    - Runtime strategy switching
    - Performance monitoring and tracking
    - A/B testing and comparison
    - Parameter optimization integration
    """

    def __init__(self):
        """Initialize the integrated strategy system."""
        self.logger = logging.getLogger("trading_bot.strategy_system")
        
        # Core components
        self.registry = get_registry()
        self.selector = StrategySelector()
        self.performance_tracker = StrategyPerformanceTracker()
        self.comparator = StrategyComparator(self.performance_tracker)
        
        # Initialize registry with built-in strategies
        self._initialize_registry()
        
        self.logger.info("IntegratedStrategySystem initialized")

    def _initialize_registry(self) -> None:
        """Initialize the registry with built-in strategies."""
        self.registry.auto_register_builtin_strategies()
        
        self.logger.info(
            f"Registered {len(self.registry.list_strategies())} strategies: "
            f"{', '.join(self.registry.list_strategies())}"
        )

    async def start(self, strategy_name: str, parameters: Optional[Dict] = None) -> None:
        """
        Start the system with a specific strategy.

        Args:
            strategy_name: Name of strategy to start with
            parameters: Optional strategy parameters
        """
        await self.selector.set_strategy(strategy_name, parameters)
        await self.selector.start()
        
        self.logger.info(f"System started with strategy: {strategy_name}")

    async def stop(self) -> None:
        """Stop the system and current strategy."""
        await self.selector.stop()
        self.logger.info("System stopped")

    async def switch_strategy(
        self,
        new_strategy: str,
        parameters: Optional[Dict] = None
    ) -> None:
        """
        Switch to a different strategy.

        Args:
            new_strategy: Strategy name to switch to
            parameters: Optional strategy parameters
        """
        await self.selector.switch_strategy(new_strategy, parameters)
        
        self.logger.info(f"Switched to strategy: {new_strategy}")

    async def generate_signals(self, df: pd.DataFrame) -> List:
        """
        Generate trading signals using the current strategy.

        Args:
            df: Market data DataFrame

        Returns:
            List of generated signals
        """
        signals = await self.selector.generate_signals(df)
        
        self.logger.info(
            f"Generated {len(signals)} signals from "
            f"{self.selector.current_strategy_name}"
        )
        
        return signals

    async def run_ab_test(
        self,
        champion_name: str,
        challenger_name: str,
        test_data: pd.DataFrame,
        champion_params: Optional[Dict] = None,
        challenger_params: Optional[Dict] = None
    ) -> Dict:
        """
        Run A/B test between two strategies.

        Args:
            champion_name: Current production strategy
            challenger_name: New strategy to test
            test_data: Historical data for testing
            champion_params: Optional champion parameters
            challenger_params: Optional challenger parameters

        Returns:
            Comparison results dictionary
        """
        # Create strategy instances
        champion = self.registry.create_strategy(
            champion_name,
            parameters=champion_params
        )
        challenger = self.registry.create_strategy(
            challenger_name,
            parameters=challenger_params
        )
        
        # Start both strategies
        await champion.start()
        await challenger.start()
        
        try:
            # Run comparison
            result = await self.comparator.run_ab_test(
                champion,
                challenger,
                test_data
            )
            
            self.logger.info(
                f"A/B test complete: Recommended={result.recommended_strategy}"
            )
            
            return self.comparator.export_comparison(result, format="dict")
            
        finally:
            # Clean up
            await champion.stop()
            await challenger.stop()

    def get_performance_summary(self, strategy_name: Optional[str] = None) -> Dict:
        """
        Get performance summary for strategies.

        Args:
            strategy_name: Optional specific strategy (all if None)

        Returns:
            Performance metrics dictionary
        """
        if strategy_name:
            metrics = self.performance_tracker.get_strategy_metrics(strategy_name)
            return {
                "strategy": strategy_name,
                "total_trades": metrics.total_trades,
                "win_rate": metrics.win_rate,
                "total_pnl": float(metrics.total_pnl),
                "sharpe_ratio": metrics.sharpe_ratio,
                "max_drawdown": float(metrics.max_drawdown)
            }
        else:
            all_metrics = self.performance_tracker.get_all_metrics()
            return {
                name: {
                    "total_trades": m.total_trades,
                    "win_rate": m.win_rate,
                    "total_pnl": float(m.total_pnl)
                }
                for name, m in all_metrics.items()
            }

    def list_available_strategies(self) -> List[Dict]:
        """
        List all available strategies with their information.

        Returns:
            List of strategy information dictionaries
        """
        strategies = []
        
        for name in self.registry.list_strategies():
            info = self.registry.get_strategy_info(name)
            strategies.append({
                "name": name,
                "class": info["class_name"],
                "version": info.get("version"),
                "parameters": list(info["parameter_schema"].keys())
            })
        
        return strategies

    def get_strategy_config(self, strategy_name: str) -> Dict:
        """
        Get detailed configuration for a strategy.

        Args:
            strategy_name: Strategy name

        Returns:
            Strategy configuration including parameter schema
        """
        return self.registry.get_strategy_info(strategy_name)

    async def optimize_strategy_parameters(
        self,
        strategy_name: str,
        test_data: pd.DataFrame,
        parameter_grid: Dict[str, List]
    ) -> Dict:
        """
        Optimize strategy parameters using grid search.

        Args:
            strategy_name: Strategy to optimize
            test_data: Data for optimization
            parameter_grid: Dictionary of parameter names to list of values to test

        Returns:
            Best parameters and their performance
        """
        self.logger.info(
            f"Starting parameter optimization for {strategy_name} "
            f"with {len(parameter_grid)} parameters"
        )
        
        best_params = None
        best_performance = float('-inf')
        results = []
        
        # Generate parameter combinations (simplified grid search)
        import itertools
        
        param_names = list(parameter_grid.keys())
        param_values = list(parameter_grid.values())
        
        for combination in itertools.product(*param_values):
            params = dict(zip(param_names, combination))
            
            # Create strategy with these parameters
            strategy = self.registry.create_strategy(
                strategy_name,
                parameters=params
            )
            
            await strategy.start()
            
            try:
                # Generate signals
                signals = await strategy.generate_signals(test_data)
                
                # Evaluate performance (simplified)
                performance_score = len(signals) * 1.0  # Placeholder metric
                
                results.append({
                    "parameters": params,
                    "performance": performance_score,
                    "signals_generated": len(signals)
                })
                
                if performance_score > best_performance:
                    best_performance = performance_score
                    best_params = params
                
            finally:
                await strategy.stop()
        
        self.logger.info(
            f"Optimization complete: Best performance={best_performance}, "
            f"Parameters={best_params}"
        )
        
        return {
            "best_parameters": best_params,
            "best_performance": best_performance,
            "all_results": results
        }

    def get_system_status(self) -> Dict:
        """
        Get current system status.

        Returns:
            System status dictionary
        """
        current_strategy = self.selector.current_strategy
        
        return {
            "system_running": self.selector.is_running(),
            "current_strategy": current_strategy.name if current_strategy else None,
            "current_strategy_state": current_strategy.state.value if current_strategy else None,
            "available_strategies": self.registry.list_strategies(),
            "strategy_switches": len(self.selector.get_switch_history()),
            "active_comparisons": len(self.comparator.get_comparison_history())
        }


# Example usage function
async def example_usage():
    """
    Example demonstrating the integrated strategy system usage.
    """
    # Initialize system
    system = IntegratedStrategySystem()
    
    # List available strategies
    print("Available strategies:")
    for strategy in system.list_available_strategies():
        print(f"  - {strategy['name']} (v{strategy['version']})")
    
    # Start with ICT strategy
    await system.start("ICT")
    
    # Generate some dummy data
    import numpy as np
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')
    df = pd.DataFrame({
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 102,
        'low': np.random.randn(100).cumsum() + 98,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100),
        'timestamp': dates
    })
    
    # Generate signals
    signals = await system.generate_signals(df)
    print(f"\nGenerated {len(signals)} signals from ICT strategy")
    
    # Switch to traditional strategy
    await system.switch_strategy("Traditional")
    signals = await system.generate_signals(df)
    print(f"Generated {len(signals)} signals from Traditional strategy")
    
    # Run A/B test
    print("\nRunning A/B test...")
    comparison = await system.run_ab_test("ICT", "Traditional", df)
    print(f"Recommended strategy: {comparison['recommended']}")
    print(f"Reason: {comparison['reason']}")
    
    # Get performance summary
    print("\nPerformance summary:")
    perf = system.get_performance_summary()
    for strategy_name, metrics in perf.items():
        print(f"  {strategy_name}: {metrics}")
    
    # System status
    status = system.get_system_status()
    print(f"\nSystem status: {status}")
    
    # Stop system
    await system.stop()
    print("\nSystem stopped")


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())
