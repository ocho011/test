# Trading Strategy Framework Guide

Complete guide to the extensible strategy framework with plugin architecture, performance tracking, A/B testing, and parameter optimization.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Quick Start](#quick-start)
4. [Creating Custom Strategies](#creating-custom-strategies)
5. [Strategy Registry](#strategy-registry)
6. [Runtime Strategy Switching](#runtime-strategy-switching)
7. [Performance Tracking](#performance-tracking)
8. [A/B Testing](#ab-testing)
9. [Parameter Optimization](#parameter-optimization)
10. [Complete Integration Example](#complete-integration-example)
11. [Best Practices](#best-practices)

---

## Overview

The strategy framework provides a comprehensive system for managing trading strategies with the following features:

- **Abstract Base Class**: `AbstractStrategy` with standardized interface
- **Plugin Architecture**: Dynamic strategy registration and discovery
- **Runtime Switching**: Safe strategy transitions with lifecycle management
- **Performance Tracking**: Comprehensive metrics calculation and monitoring
- **A/B Testing**: Statistical comparison with champion/challenger pattern
- **Parameter Optimization**: Walk-forward optimization with overfitting prevention
- **BaseComponent Integration**: Proper lifecycle management (INITIALIZED → STARTING → RUNNING → STOPPING → STOPPED)

---

## Architecture

### Component Hierarchy

```
AbstractStrategy (BaseComponent)
├── ICTStrategy
├── TraditionalIndicatorStrategy
└── [Your Custom Strategy]

StrategyRegistry (Singleton)
└── Manages all registered strategies

StrategySelector
└── Handles runtime strategy switching

StrategyPerformanceTracker
└── Tracks trades and calculates metrics

StrategyComparator
└── Performs A/B testing and comparison

StrategyOptimizer
└── Parameter optimization with walk-forward analysis

OptimizedStrategySystem
└── Unified interface combining all components
```

### Data Flow

```
Market Data → Strategy.generate_signals() → GeneratedSignal[]
                        ↓
            PerformanceTracker.record_trade()
                        ↓
                 StrategyMetrics
                        ↓
              Comparator.compare()
                        ↓
              ComparisonResult
                        ↓
        Optimizer.optimize_strategy()
                        ↓
            OptimizationResult
```

---

## Quick Start

### Basic Usage

```python
from trading_bot.strategies import (
    get_registry,
    StrategySelector,
    StrategyPerformanceTracker,
)

# 1. Get registry and list available strategies
registry = get_registry()
strategies = registry.list_strategies()
print(f"Available strategies: {strategies}")

# 2. Create a strategy
strategy = registry.create_strategy('ICTStrategy')

# 3. Generate signals
import pandas as pd
df = pd.read_csv('market_data.csv')  # Your OHLCV data
signals = await strategy.generate_signals(df)

# 4. Track performance
tracker = StrategyPerformanceTracker()
tracker.record_trade(
    strategy_name='ICTStrategy',
    entry_price=Decimal('50000'),
    exit_price=Decimal('51000'),
    quantity=Decimal('1.0'),
    is_winning=True,
)

# 5. Get metrics
metrics = tracker.get_metrics('ICTStrategy')
print(f"Win rate: {metrics.win_rate}%")
print(f"Total PnL: {metrics.total_pnl}")
```

---

## Creating Custom Strategies

### Step 1: Extend AbstractStrategy

```python
from trading_bot.strategies import AbstractStrategy
from trading_bot.core.event_types import MarketDataEvent
from trading_bot.signal_generation.types import GeneratedSignal
from typing import Dict, List, Any, Optional
import pandas as pd

class MyCustomStrategy(AbstractStrategy):
    """Your custom trading strategy."""
    
    def __init__(self, **kwargs):
        """Initialize strategy with parameters."""
        default_params = {
            'param1': 14,
            'param2': 2.0,
            'param3': True,
        }
        super().__init__(
            name='MyCustomStrategy',
            description='Description of my strategy',
            parameters=default_params,
            **kwargs
        )
    
    async def generate_signals(
        self,
        df: pd.DataFrame,
        current_event: Optional[MarketDataEvent] = None
    ) -> List[GeneratedSignal]:
        """
        Generate trading signals from market data.
        
        Args:
            df: OHLCV DataFrame
            current_event: Optional current market event
            
        Returns:
            List of GeneratedSignal objects
        """
        signals = []
        
        # Your signal generation logic here
        # Example: Simple moving average crossover
        short_ma = df['close'].rolling(window=self.parameters['param1']).mean()
        long_ma = df['close'].rolling(window=self.parameters['param1'] * 2).mean()
        
        if short_ma.iloc[-1] > long_ma.iloc[-1]:
            signal = GeneratedSignal(
                direction=SignalDirection.LONG,
                strength=0.8,
                symbol='BTCUSDT',
                price=float(df['close'].iloc[-1]),
                timestamp=df.index[-1],
                source='MyCustomStrategy',
                metadata={'condition': 'MA crossover'}
            )
            signals.append(signal)
        
        return signals
    
    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """
        Validate parameter values.
        
        Args:
            params: Parameter dictionary to validate
            
        Returns:
            True if valid, False otherwise
        """
        if 'param1' in params:
            if not isinstance(params['param1'], int) or params['param1'] <= 0:
                return False
        
        if 'param2' in params:
            if not isinstance(params['param2'], (int, float)) or params['param2'] <= 0:
                return False
        
        return True
    
    def get_parameter_schema(self) -> Dict[str, Any]:
        """
        Get parameter schema for optimization.
        
        Returns:
            Dictionary describing parameters
        """
        return {
            'param1': {
                'type': 'int',
                'min': 5,
                'max': 50,
                'step': 5,
                'description': 'Period for calculation'
            },
            'param2': {
                'type': 'float',
                'min': 1.0,
                'max': 3.0,
                'step': 0.5,
                'description': 'Multiplier factor'
            },
            'param3': {
                'type': 'bool',
                'description': 'Enable feature'
            },
        }
```

### Step 2: Register Your Strategy

```python
from trading_bot.strategies import get_registry

# Get registry instance
registry = get_registry()

# Register your strategy
registry.register(
    strategy_class=MyCustomStrategy,
    name='MyCustomStrategy',
    description='My custom trading strategy',
    category='custom',
)

# Now it's available
strategy = registry.create_strategy('MyCustomStrategy')
```

---

## Strategy Registry

The registry provides centralized management of all strategies.

### Registry Operations

```python
from trading_bot.strategies import get_registry

registry = get_registry()

# List all strategies
strategies = registry.list_strategies()
print(strategies)  # ['ICTStrategy', 'TraditionalIndicatorStrategy', 'MyCustomStrategy']

# Get strategy information
info = registry.get_strategy_info('ICTStrategy')
print(info['description'])
print(info['category'])

# Create strategy with parameters
strategy = registry.create_strategy(
    'TraditionalIndicatorStrategy',
    parameters={
        'rsi_period': 21,
        'rsi_oversold': 25,
        'rsi_overbought': 75,
    }
)

# List strategies by category
trend_strategies = registry.list_strategies(category='trend')
```

---

## Runtime Strategy Switching

The `StrategySelector` enables safe runtime strategy transitions.

### Basic Switching

```python
from trading_bot.strategies import StrategySelector, get_registry

selector = StrategySelector(
    registry=get_registry(),
    allow_mid_trade_switch=False  # Prevent switching during active trading
)

# Set initial strategy
await selector.set_strategy('ICTStrategy')

# Generate signals
signals = await selector.generate_signals(df)

# Switch strategy (only when not running)
await selector.set_strategy('TraditionalIndicatorStrategy')
```

### Forced Switching

```python
# Force switch even during active trading (use with caution)
await selector.set_strategy(
    'TraditionalIndicatorStrategy',
    force=True
)
```

### Automatic Rollback

```python
# If new strategy fails to start, automatically rollback
try:
    await selector.set_strategy('NewStrategy')
except Exception as e:
    # Selector automatically reverts to previous strategy
    print(f"Switch failed, reverted to {selector.current_strategy}")
```

---

## Performance Tracking

Track and analyze strategy performance with comprehensive metrics.

### Recording Trades

```python
from trading_bot.strategies import StrategyPerformanceTracker
from decimal import Decimal

tracker = StrategyPerformanceTracker()

# Record a trade
tracker.record_trade(
    strategy_name='ICTStrategy',
    entry_price=Decimal('50000.00'),
    exit_price=Decimal('51000.00'),
    quantity=Decimal('1.0'),
    is_winning=True,
    timestamp=datetime.now(),
)
```

### Getting Metrics

```python
# Get current metrics
metrics = tracker.get_metrics('ICTStrategy')

print(f"Total Trades: {metrics.total_trades}")
print(f"Win Rate: {metrics.win_rate}%")
print(f"Total PnL: ${metrics.total_pnl}")
print(f"Average Win: ${metrics.average_win}")
print(f"Average Loss: ${metrics.average_loss}")
print(f"Profit Factor: {metrics.profit_factor}")
print(f"Sharpe Ratio: {metrics.sharpe_ratio}")
print(f"Max Drawdown: ${metrics.max_drawdown}")
print(f"Max Consecutive Wins: {metrics.max_consecutive_wins}")

# Get trade history
history = tracker.get_trade_history('ICTStrategy', limit=10)
for trade in history:
    print(f"Entry: ${trade.entry_price}, Exit: ${trade.exit_price}, PnL: ${trade.pnl}")
```

### Resetting Metrics

```python
# Reset for new period
tracker.reset_metrics('ICTStrategy')
```

---

## A/B Testing

Compare strategies statistically with the `StrategyComparator`.

### Basic Comparison

```python
from trading_bot.strategies import StrategyComparator

comparator = StrategyComparator(tracker)

# Compare two strategies
result = await comparator.compare_strategies(
    strategy_a=strategy1,
    strategy_b=strategy2,
    test_data=df,
    confidence_level=0.95,
)

print(f"Strategy A: {result.strategy_a_name}")
print(f"Win Rate: {result.metrics_a.win_rate}%")
print(f"Sharpe: {result.metrics_a.sharpe_ratio}")

print(f"\nStrategy B: {result.strategy_b_name}")
print(f"Win Rate: {result.metrics_b.win_rate}%")
print(f"Sharpe: {result.metrics_b.sharpe_ratio}")

print(f"\nStatistically Significant: {result.is_statistically_significant}")
print(f"P-Value: {result.p_value}")
print(f"Recommendation: {result.recommendation}")
print(f"Confidence: {result.confidence}")
```

### Champion/Challenger Pattern

```python
# Test challenger against current champion
result = await comparator.champion_challenger_test(
    champion=current_strategy,
    challenger=new_strategy,
    test_data=df,
    traffic_split=0.8,  # 80% champion, 20% challenger
    min_sample_size=100,
)

if result.recommendation == 'B':  # Challenger won
    print("Promoting challenger to champion!")
    await selector.set_strategy_instance(new_strategy)
```

---

## Parameter Optimization

Optimize strategy parameters with walk-forward analysis.

### Basic Optimization

```python
from trading_bot.strategies import StrategyOptimizer, OptimizationConfig
from trading_bot.backtesting import BacktestEngine, PerformanceAnalyzer

# Setup
backtest_engine = BacktestEngine()
performance_analyzer = PerformanceAnalyzer()

# Load historical data
df = pd.read_csv('historical_data.csv')
backtest_engine.historical_data = df

# Create optimizer
config = OptimizationConfig(
    in_sample_ratio=0.7,
    window_size_days=180,
    step_size_days=30,
    optimization_metric='sharpe_ratio',
)

optimizer = StrategyOptimizer(
    backtest_engine=backtest_engine,
    performance_analyzer=performance_analyzer,
    config=config,
)

# Optimize strategy
strategy = registry.create_strategy('TraditionalIndicatorStrategy')
result = await optimizer.optimize_strategy(
    strategy=strategy,
    parameter_grid={
        'rsi_period': [14, 21, 28],
        'rsi_oversold': [20, 30],
        'rsi_overbought': [70, 80],
    },
)

print(f"Best Parameters: {result.parameters}")
print(f"In-Sample Sharpe: {result.in_sample_metrics['sharpe_ratio']}")
print(f"Out-of-Sample Sharpe: {result.out_sample_metrics['sharpe_ratio']}")
print(f"Stability Score: {result.stability_score}")
```

### Auto Grid Generation

```python
# Optimizer can generate parameter grid automatically
result = await optimizer.optimize_strategy(
    strategy=strategy,
    # parameter_grid=None,  # Auto-generated from schema
)
```

### Optimization Caching

```python
# Results are automatically cached
result1 = await optimizer.optimize_strategy(strategy, grid1)  # Computes
result2 = await optimizer.optimize_strategy(strategy, grid1)  # Uses cache

# Force re-optimization
result3 = await optimizer.optimize_strategy(
    strategy,
    grid1,
    force_reoptimize=True
)

# Clear cache
optimizer.clear_cache()
```

---

## Complete Integration Example

```python
from trading_bot.strategies import OptimizedStrategySystem
from trading_bot.backtesting import BacktestEngine, PerformanceAnalyzer
import pandas as pd

async def main():
    # 1. Setup
    backtest_engine = BacktestEngine()
    performance_analyzer = PerformanceAnalyzer()
    
    # Load data
    df = pd.read_csv('market_data.csv', index_col=0, parse_dates=True)
    backtest_engine.historical_data = df
    
    # Create integrated system
    system = OptimizedStrategySystem(
        backtest_engine=backtest_engine,
        performance_analyzer=performance_analyzer,
    )
    
    # 2. Discover strategies
    strategies = system.registry.list_strategies()
    print(f"Available: {strategies}")
    
    # 3. Optimize and deploy
    optimized = await system.optimize_and_deploy(
        strategy_name='TraditionalIndicatorStrategy',
        parameter_grid={
            'rsi_period': [14, 21, 28],
            'macd_fast': [12, 16],
            'macd_slow': [26, 30],
        }
    )
    
    # 4. Run comparison
    results = await system.run_optimization_comparison(
        strategy_names=['ICTStrategy', 'TraditionalIndicatorStrategy']
    )
    
    # 5. Select best strategy
    best_strategy_name = max(
        results.items(),
        key=lambda x: x[1].out_sample_metrics.get('sharpe_ratio', 0)
    )[0]
    
    print(f"Best strategy: {best_strategy_name}")
    
    # 6. Deploy best strategy
    best_strategy = system.registry.create_strategy(
        best_strategy_name,
        parameters=results[best_strategy_name].parameters
    )
    await system.selector.set_strategy_instance(best_strategy)
    
    # 7. Start trading
    signals = await best_strategy.generate_signals(df.tail(100))
    print(f"Generated {len(signals)} signals")
    
    # 8. Monitor performance
    metrics = system.performance_tracker.get_metrics(best_strategy_name)
    print(f"Current win rate: {metrics.win_rate}%")

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
```

---

## Best Practices

### 1. Strategy Development

- Always extend `AbstractStrategy` for proper interface
- Implement all required abstract methods
- Provide comprehensive parameter schemas
- Use meaningful parameter names and descriptions
- Validate parameters thoroughly

### 2. Parameter Optimization

- Use walk-forward optimization to prevent overfitting
- Monitor stability scores (>0.7 recommended)
- Test on out-of-sample data before production
- Cache optimization results for efficiency
- Re-optimize periodically as market conditions change

### 3. Performance Tracking

- Record all trades consistently
- Reset metrics at appropriate intervals
- Monitor both winning and losing streaks
- Track drawdowns carefully
- Calculate Sharpe ratio for risk-adjusted returns

### 4. Strategy Switching

- Avoid switching during active trades unless necessary
- Test new strategies thoroughly before deployment
- Use A/B testing for validation
- Monitor metrics after switches
- Keep rollback capability ready

### 5. Testing

- Write unit tests for each strategy
- Test parameter validation
- Test signal generation edge cases
- Test lifecycle transitions
- Use integration tests for complete workflows

### 6. Production Deployment

- Start with paper trading
- Use champion/challenger pattern for new strategies
- Monitor performance metrics continuously
- Set up alerts for unusual behavior
- Keep audit logs of all trades and switches

### 7. Maintenance

- Review and update strategies regularly
- Monitor for concept drift
- Adjust parameters as needed
- Archive old strategies properly
- Document all changes

---

## API Reference

### AbstractStrategy

```python
class AbstractStrategy(BaseComponent):
    async def generate_signals(df: pd.DataFrame, current_event: Optional[MarketDataEvent]) -> List[GeneratedSignal]
    def validate_parameters(params: Dict[str, Any]) -> bool
    def get_parameter_schema() -> Dict[str, Any]
    def get_config() -> Dict[str, Any]
    def update_parameters(params: Dict[str, Any]) -> None
```

### StrategyRegistry

```python
class StrategyRegistry:
    def register(strategy_class: Type[AbstractStrategy], name: str, ...) -> None
    def create_strategy(name: str, parameters: Optional[Dict]) -> AbstractStrategy
    def list_strategies(category: Optional[str]) -> List[str]
    def get_strategy_info(name: str) -> Dict[str, Any]
```

### StrategySelector

```python
class StrategySelector:
    async def set_strategy(strategy_name: str, parameters: Optional[Dict], force: bool) -> None
    async def set_strategy_instance(strategy: AbstractStrategy) -> None
    async def generate_signals(df: pd.DataFrame) -> List[GeneratedSignal]
```

### StrategyPerformanceTracker

```python
class StrategyPerformanceTracker:
    def record_trade(strategy_name: str, entry_price: Decimal, exit_price: Decimal, ...) -> None
    def get_metrics(strategy_name: str) -> StrategyMetrics
    def get_trade_history(strategy_name: str, limit: int) -> List[TradeRecord]
    def reset_metrics(strategy_name: str) -> None
```

### StrategyComparator

```python
class StrategyComparator:
    async def compare_strategies(strategy_a: AbstractStrategy, strategy_b: AbstractStrategy, ...) -> ComparisonResult
    async def champion_challenger_test(champion: AbstractStrategy, challenger: AbstractStrategy, ...) -> ComparisonResult
```

### StrategyOptimizer

```python
class StrategyOptimizer:
    async def optimize_strategy(strategy: AbstractStrategy, parameter_grid: Optional[Dict], ...) -> OptimizationResult
    async def compare_parameter_sets(strategy: AbstractStrategy, parameter_sets: List[Dict], ...) -> List[OptimizationResult]
    def get_optimization_summary(result: OptimizationResult) -> Dict[str, Any]
    def clear_cache() -> None
```

---

## Troubleshooting

### Common Issues

**Q: Strategy switching fails with "Cannot switch strategy while running"**
A: Either stop the current strategy first, or use `force=True` flag (with caution).

**Q: Optimization takes too long**
A: Reduce parameter grid size, use caching, or increase step_size_days.

**Q: Low stability scores in optimization**
A: Indicates overfitting. Try simpler parameters or increase in_sample_ratio.

**Q: Signals not generating**
A: Check DataFrame format (must have OHLCV columns), verify parameter validation.

**Q: Performance metrics seem incorrect**
A: Ensure trade recording includes correct timestamps and decimal precision.

---

## Additional Resources

- [ICT Strategy Documentation](./ict_strategy.md)
- [Traditional Indicators Guide](./traditional_indicators.md)
- [Backtesting Framework](./backtesting_guide.md)
- [Signal Generation System](./signal_generation.md)

