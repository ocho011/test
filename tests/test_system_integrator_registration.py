"""
Tests for SystemIntegrator component registration.

Verifies that MarketDataAggregator and StrategyCoordinator are properly
imported and can be instantiated.
"""

import pytest
from unittest.mock import Mock, AsyncMock


def test_imports_are_valid():
    """Test that all new components can be imported."""
    from trading_bot.data import MarketDataAggregator
    from trading_bot.strategies import StrategyCoordinator, IntegratedStrategySystem
    
    # Verify classes are importable
    assert MarketDataAggregator is not None
    assert StrategyCoordinator is not None
    assert IntegratedStrategySystem is not None


def test_market_data_aggregator_instantiation():
    """Test that MarketDataAggregator can be instantiated with mocked dependencies."""
    from trading_bot.data import MarketDataAggregator
    
    # Create mocks
    mock_binance_client = Mock()
    mock_event_bus = Mock()
    
    # Instantiate
    aggregator = MarketDataAggregator(
        binance_client=mock_binance_client,
        event_bus=mock_event_bus,
        symbols=["BTCUSDT"],
        intervals=["5m", "15m"],
        lookback_bars=100
    )
    
    # Verify basic attributes
    assert aggregator.symbols == ["BTCUSDT"]
    assert aggregator.intervals == ["5m", "15m"]
    assert aggregator.lookback_bars == 100
    assert aggregator.binance_client is mock_binance_client
    assert aggregator.event_bus is mock_event_bus


def test_integrated_strategy_system_instantiation():
    """Test that IntegratedStrategySystem can be instantiated."""
    from trading_bot.strategies import IntegratedStrategySystem
    
    # Instantiate (no dependencies)
    strategy_system = IntegratedStrategySystem()
    
    # Verify basic attributes
    assert strategy_system.registry is not None
    assert strategy_system.selector is not None
    assert strategy_system.performance_tracker is not None


def test_strategy_coordinator_instantiation():
    """Test that StrategyCoordinator can be instantiated with mocked dependencies."""
    from trading_bot.strategies import StrategyCoordinator, IntegratedStrategySystem
    
    # Create mocks
    mock_event_bus = Mock()
    mock_ict_analyzer = Mock()
    strategy_system = IntegratedStrategySystem()
    
    # Instantiate
    coordinator = StrategyCoordinator(
        event_bus=mock_event_bus,
        ict_analyzer=mock_ict_analyzer,
        strategy_system=strategy_system,
        subscribed_intervals=["5m", "15m", "4h", "1d"],
        min_confluence_timeframes=2
    )
    
    # Verify basic attributes
    assert coordinator.event_bus is mock_event_bus
    assert coordinator.ict_analyzer is mock_ict_analyzer
    assert coordinator.strategy_system is strategy_system
    assert coordinator.subscribed_intervals == ["5m", "15m", "4h", "1d"]
    assert coordinator.min_confluence_timeframes == 2


def test_system_integrator_imports():
    """Test that SystemIntegrator correctly imports new components."""
    from trading_bot.system_integrator import (
        MarketDataAggregator,
        StrategyCoordinator,
        IntegratedStrategySystem,
    )
    
    # Verify imports work from SystemIntegrator module
    assert MarketDataAggregator is not None
    assert StrategyCoordinator is not None
    assert IntegratedStrategySystem is not None


def test_component_exports_in_packages():
    """Test that components are exported in their respective packages."""
    # Test data package
    from trading_bot.data import MarketDataAggregator
    from trading_bot import data as data_module
    assert "MarketDataAggregator" in data_module.__all__
    
    # Test strategies package
    from trading_bot.strategies import StrategyCoordinator, IntegratedStrategySystem
    from trading_bot import strategies as strategies_module
    assert "StrategyCoordinator" in strategies_module.__all__
    assert "IntegratedStrategySystem" in strategies_module.__all__
