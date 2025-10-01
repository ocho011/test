"""
Integration tests for the complete strategy framework.

Tests the interaction between all strategy components:
- AbstractStrategy base class
- Concrete strategy implementations (ICT, Traditional)
- StrategyRegistry plugin system
- StrategySelector runtime switching
- StrategyPerformanceTracker monitoring
- StrategyComparator A/B testing
- StrategyOptimizer parameter optimization
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch

from trading_bot.strategies import (
    AbstractStrategy,
    ICTStrategy,
    TraditionalIndicatorStrategy,
    StrategyRegistry,
    get_registry,
    StrategySelector,
    StrategyPerformanceTracker,
    StrategyComparator,
    StrategyOptimizer,
    OptimizationConfig,
    OptimizedStrategySystem,
)
from trading_bot.core.events import MarketDataEvent
from trading_bot.backtesting.backtest_engine import BacktestEngine
from trading_bot.backtesting.performance_analyzer import PerformanceAnalyzer


# Test fixtures
@pytest.fixture
def sample_dataframe():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='1H')
    np.random.seed(42)
    
    df = pd.DataFrame({
        'open': 50000 + np.cumsum(np.random.randn(1000) * 100),
        'high': 50000 + np.cumsum(np.random.randn(1000) * 100) + 50,
        'low': 50000 + np.cumsum(np.random.randn(1000) * 100) - 50,
        'close': 50000 + np.cumsum(np.random.randn(1000) * 100),
        'volume': np.random.uniform(100, 1000, 1000),
    }, index=dates)
    
    # Ensure high is highest and low is lowest
    df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
    df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)
    
    return df


@pytest.fixture
def backtest_engine(sample_dataframe):
    """Create BacktestEngine with sample data."""
    from unittest.mock import Mock
    mock_client = Mock()
    engine = BacktestEngine(binance_client=mock_client)
    engine.historical_data = sample_dataframe
    return engine


@pytest.fixture
def performance_analyzer():
    """Create PerformanceAnalyzer instance."""
    return PerformanceAnalyzer()


@pytest.fixture
def strategy_registry():
    """Create fresh StrategyRegistry."""
    registry = StrategyRegistry()
    registry.auto_register_builtin_strategies()
    return registry


# Test AbstractStrategy contract
class TestAbstractStrategyContract:
    """Test that strategies properly implement the abstract base class."""
    
    @pytest.mark.asyncio
    async def test_ict_strategy_implements_interface(self, sample_dataframe):
        """Test ICTStrategy implements all required methods."""
        strategy = ICTStrategy()
        
        # Test required methods exist
        assert hasattr(strategy, 'generate_signals')
        assert hasattr(strategy, 'validate_parameters')
        assert hasattr(strategy, 'get_parameter_schema')
        assert hasattr(strategy, 'get_config')
        assert hasattr(strategy, 'update_parameters')
        
        # Test methods work
        signals = await strategy.generate_signals(sample_dataframe)
        assert isinstance(signals, list)
        
        schema = strategy.get_parameter_schema()
        assert isinstance(schema, dict)
        
        config = strategy.get_config()
        assert isinstance(config, dict)
        assert 'parameters' in config
    
    @pytest.mark.asyncio
    async def test_traditional_strategy_implements_interface(self, sample_dataframe):
        """Test TraditionalIndicatorStrategy implements all required methods."""
        strategy = TraditionalIndicatorStrategy()
        
        # Test required methods
        signals = await strategy.generate_signals(sample_dataframe)
        assert isinstance(signals, list)
        
        schema = strategy.get_parameter_schema()
        assert isinstance(schema, dict)
        assert 'rsi_period' in schema
        
        config = strategy.get_config()
        assert isinstance(config, dict)
    
    def test_parameter_validation(self):
        """Test parameter validation works correctly."""
        strategy = TraditionalIndicatorStrategy()
        
        # Valid parameters
        valid_params = {
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
        }
        assert strategy.validate_parameters(valid_params) is True
        
        # Invalid parameters
        invalid_params = {
            'rsi_period': -5,  # Negative period
            'rsi_oversold': 30,
        }
        assert strategy.validate_parameters(invalid_params) is False


# Test StrategyRegistry
class TestStrategyRegistry:
    """Test strategy plugin registry."""
    
    def test_registry_singleton(self):
        """Test registry is a singleton."""
        registry1 = get_registry()
        registry2 = get_registry()
        assert registry1 is registry2
    
    def test_auto_register_builtin_strategies(self, strategy_registry):
        """Test built-in strategies are auto-registered."""
        strategies = strategy_registry.list_strategies()
        assert 'ICT' in strategies
        assert 'Traditional' in strategies
    
    def test_create_strategy(self, strategy_registry):
        """Test strategy creation from registry."""
        strategy = strategy_registry.create_strategy('ICT')
        assert isinstance(strategy, ICTStrategy)
        
        strategy = strategy_registry.create_strategy(
            'Traditional')
        assert isinstance(strategy, TraditionalIndicatorStrategy)
    
    def test_create_strategy_with_parameters(self, strategy_registry):
        """Test creating strategy with custom parameters."""
        params = {
            'rsi_period': 21,
            'rsi_oversold': 25,
        }
        strategy = strategy_registry.create_strategy(
            'Traditional',
            parameters=params
        )
        assert strategy.parameters['rsi_period'] == 21
        assert strategy.parameters['rsi_oversold'] == 25
    
    def test_get_strategy_info(self, strategy_registry):
        """Test retrieving strategy metadata."""
        info = strategy_registry.get_strategy_info('ICT')
        assert info is not None
        assert 'name' in info
        assert 'description' in info


# Test StrategySelector
class TestStrategySelector:
    """Test runtime strategy switching."""
    
    @pytest.mark.asyncio
    async def test_set_strategy(self, strategy_registry):
        """Test setting active strategy."""
        selector = StrategySelector(registry=strategy_registry)
        
        await selector.set_strategy('ICTStrategy')
        assert selector.current_strategy is not None
        assert isinstance(selector.current_strategy, ICTStrategy)
    
    @pytest.mark.asyncio
    async def test_strategy_switching_lifecycle(self, strategy_registry):
        """Test strategy lifecycle during switching."""
        selector = StrategySelector(registry=strategy_registry)
        
        # Set first strategy
        await selector.set_strategy('ICTStrategy')
        first_strategy = selector.current_strategy
        
        # Switch to second strategy
        await selector.set_strategy('TraditionalIndicatorStrategy')
        second_strategy = selector.current_strategy
        
        assert first_strategy is not second_strategy
        assert isinstance(second_strategy, TraditionalIndicatorStrategy)
    
    @pytest.mark.asyncio
    async def test_prevent_mid_trade_switching(self, strategy_registry):
        """Test that mid-trade switching is prevented by default."""
        selector = StrategySelector(
            registry=strategy_registry,
            allow_mid_trade_switch=False
        )
        
        await selector.set_strategy('ICTStrategy')
        
        # Mock strategy as running
        selector._current_strategy._state = selector._current_strategy.ComponentState.RUNNING
        
        # Should raise error without force flag
        with pytest.raises(RuntimeError, match="Cannot switch strategy while running"):
            await selector.set_strategy('TraditionalIndicatorStrategy')
        
        # Should work with force flag
        await selector.set_strategy('TraditionalIndicatorStrategy', force=True)
        assert isinstance(selector.current_strategy, TraditionalIndicatorStrategy)


# Test StrategyPerformanceTracker
class TestStrategyPerformanceTracker:
    """Test strategy performance monitoring."""
    
    def test_record_trade(self):
        """Test recording trade results."""
        tracker = StrategyPerformanceTracker()
        
        tracker.record_trade(
            strategy_name='ICTStrategy',
            entry_price=Decimal('50000'),
            exit_price=Decimal('51000'),
            quantity=Decimal('1.0'),
            is_winning=True,
        )
        
        metrics = tracker.get_metrics('ICTStrategy')
        assert metrics.total_trades == 1
        assert metrics.winning_trades == 1
        assert metrics.win_rate == 100.0
        assert metrics.total_pnl > 0
    
    def test_multiple_trades(self):
        """Test tracking multiple trades."""
        tracker = StrategyPerformanceTracker()
        
        # Record winning trade
        tracker.record_trade(
            strategy_name='ICTStrategy',
            entry_price=Decimal('50000'),
            exit_price=Decimal('51000'),
            quantity=Decimal('1.0'),
            is_winning=True,
        )
        
        # Record losing trade
        tracker.record_trade(
            strategy_name='ICTStrategy',
            entry_price=Decimal('51000'),
            exit_price=Decimal('50000'),
            quantity=Decimal('1.0'),
            is_winning=False,
        )
        
        metrics = tracker.get_metrics('ICTStrategy')
        assert metrics.total_trades == 2
        assert metrics.winning_trades == 1
        assert metrics.win_rate == 50.0
    
    def test_consecutive_tracking(self):
        """Test consecutive wins/losses tracking."""
        tracker = StrategyPerformanceTracker()
        
        # Three consecutive wins
        for _ in range(3):
            tracker.record_trade(
                strategy_name='ICTStrategy',
                entry_price=Decimal('50000'),
                exit_price=Decimal('51000'),
                quantity=Decimal('1.0'),
                is_winning=True,
            )
        
        metrics = tracker.get_metrics('ICTStrategy')
        assert metrics.consecutive_wins == 3
        assert metrics.max_consecutive_wins == 3


# Test StrategyComparator
class TestStrategyComparator:
    """Test A/B testing and strategy comparison."""
    
    @pytest.mark.asyncio
    async def test_compare_strategies(self, sample_dataframe):
        """Test comparing two strategies."""
        tracker = StrategyPerformanceTracker()
        comparator = StrategyComparator(tracker)
        
        strategy_a = ICTStrategy()
        strategy_b = TraditionalIndicatorStrategy()
        
        result = await comparator.compare_strategies(
            strategy_a=strategy_a,
            strategy_b=strategy_b,
            test_data=sample_dataframe,
        )
        
        assert result.strategy_a_name == 'ICTStrategy'
        assert result.strategy_b_name == 'TraditionalIndicatorStrategy'
        assert result.metrics_a is not None
        assert result.metrics_b is not None
        assert result.recommendation in ['A', 'B', 'Neutral']
    
    @pytest.mark.asyncio
    async def test_champion_challenger(self, sample_dataframe):
        """Test champion/challenger pattern."""
        tracker = StrategyPerformanceTracker()
        comparator = StrategyComparator(tracker)
        
        champion = ICTStrategy()
        challenger = TraditionalIndicatorStrategy()
        
        result = await comparator.champion_challenger_test(
            champion=champion,
            challenger=challenger,
            test_data=sample_dataframe,
            traffic_split=0.8,
        )
        
        assert result is not None
        assert result.recommendation is not None


# Test StrategyOptimizer integration
class TestStrategyOptimizer:
    """Test parameter optimization integration."""
    
    @pytest.mark.asyncio
    async def test_parameter_grid_generation(
        self,
        backtest_engine,
        performance_analyzer,
    ):
        """Test automatic parameter grid generation."""
        optimizer = StrategyOptimizer(
            backtest_engine=backtest_engine,
            performance_analyzer=performance_analyzer,
        )
        
        strategy = TraditionalIndicatorStrategy()
        grid = optimizer._generate_parameter_grid(strategy)
        
        assert isinstance(grid, dict)
        assert 'rsi_period' in grid
        assert len(grid['rsi_period']) > 1  # Multiple values to test
    
    @pytest.mark.asyncio
    async def test_optimization_caching(
        self,
        backtest_engine,
        performance_analyzer,
    ):
        """Test optimization result caching."""
        config = OptimizationConfig(enable_caching=True)
        optimizer = StrategyOptimizer(
            backtest_engine=backtest_engine,
            performance_analyzer=performance_analyzer,
            config=config,
        )
        
        # Clear cache first
        optimizer.clear_cache()
        
        strategy = TraditionalIndicatorStrategy()
        parameter_grid = {'rsi_period': [14, 21]}
        
        # First optimization - should hit optimizer
        result1 = await optimizer.optimize_strategy(
            strategy=strategy,
            parameter_grid=parameter_grid,
        )
        
        # Second optimization - should hit cache
        result2 = await optimizer.optimize_strategy(
            strategy=strategy,
            parameter_grid=parameter_grid,
        )
        
        assert result1.parameters == result2.parameters


# Test OptimizedStrategySystem
class TestOptimizedStrategySystem:
    """Test complete integrated system."""
    
    @pytest.mark.asyncio
    async def test_optimize_and_deploy(
        self,
        backtest_engine,
        performance_analyzer,
    ):
        """Test end-to-end optimization and deployment."""
        system = OptimizedStrategySystem(
            backtest_engine=backtest_engine,
            performance_analyzer=performance_analyzer,
        )
        
        strategy = await system.optimize_and_deploy(
            strategy_name='TraditionalIndicatorStrategy',
            parameter_grid={'rsi_period': [14, 21]},
        )
        
        assert strategy is not None
        assert isinstance(strategy, TraditionalIndicatorStrategy)
        assert system.selector.current_strategy is strategy
    
    @pytest.mark.asyncio
    async def test_multi_strategy_optimization(
        self,
        backtest_engine,
        performance_analyzer,
    ):
        """Test optimizing multiple strategies."""
        system = OptimizedStrategySystem(
            backtest_engine=backtest_engine,
            performance_analyzer=performance_analyzer,
        )

        # Use minimal parameter grids for fast testing
        parameter_grids = {
            'ICTStrategy': {
                'lookback_period': [50, 100],
                'min_confidence_threshold': [0.6, 0.7],
            },
            'TraditionalIndicatorStrategy': {
                'rsi_period': [14, 21],
            },
        }

        results = await system.run_optimization_comparison(
            strategy_names=['ICTStrategy', 'TraditionalIndicatorStrategy'],
            parameter_grids=parameter_grids,
        )

        assert len(results) == 2
        assert 'ICTStrategy' in results
        assert 'TraditionalIndicatorStrategy' in results


# Integration test with real workflow
class TestRealWorldWorkflow:
    """Test realistic usage scenarios."""
    
    @pytest.mark.asyncio
    async def test_complete_strategy_workflow(
        self,
        sample_dataframe,
        backtest_engine,
        performance_analyzer,
    ):
        """Test complete workflow from registration to optimization."""
        # 1. Setup system
        system = OptimizedStrategySystem(
            backtest_engine=backtest_engine,
            performance_analyzer=performance_analyzer,
        )
        
        # 2. List available strategies
        strategies = system.registry.list_strategies()
        assert len(strategies) > 0
        
        # 3. Optimize a strategy
        optimized = await system.optimize_and_deploy(
            strategy_name='TraditionalIndicatorStrategy',
            parameter_grid={'rsi_period': [14, 21]},
        )
        
        # 4. Generate signals
        signals = await optimized.generate_signals(sample_dataframe)
        assert isinstance(signals, list)
        
        # 5. Track performance (simulated)
        for _ in range(5):
            system.performance_tracker.record_trade(
                strategy_name='TraditionalIndicatorStrategy',
                entry_price=Decimal('50000'),
                exit_price=Decimal('51000'),
                quantity=Decimal('1.0'),
                is_winning=True,
            )
        
        # 6. Get metrics
        metrics = system.performance_tracker.get_metrics('TraditionalIndicatorStrategy')
        assert metrics.total_trades == 5
        assert metrics.win_rate == 100.0
        
        # 7. Compare with another strategy
        ict_strategy = system.registry.create_strategy('ICTStrategy')
        comparison = await system.comparator.compare_strategies(
            strategy_a=optimized,
            strategy_b=ict_strategy,
            test_data=sample_dataframe,
        )
        
        assert comparison.recommendation in ['A', 'B', 'Neutral']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
