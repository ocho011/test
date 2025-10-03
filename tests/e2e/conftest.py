"""
Shared fixtures for E2E integration tests.

Provides common test infrastructure including SystemIntegrator,
mock clients, event capture, and performance monitoring.
"""

import asyncio
import pytest
import pytest_asyncio
from pathlib import Path
from typing import AsyncGenerator, Generator

from trading_bot.system_integrator import SystemIntegrator
from trading_bot.core.event_bus import EventBus
from trading_bot.config.config_manager import ConfigManager

from .fixtures.mock_binance_client import MockBinanceClient
from .utils.event_capture import EventCapture
from .utils.performance_collector import PerformanceCollector


@pytest.fixture(scope="session")
def event_loop_policy():
    """Set event loop policy for the test session."""
    return asyncio.get_event_loop_policy()


@pytest.fixture
def test_config_path(tmp_path: Path) -> Path:
    """
    Create a temporary test configuration file.
    
    Args:
        tmp_path: pytest temporary directory
        
    Returns:
        Path to test configuration file
    """
    config_content = """
environment: testing

binance:
  api_key: test_api_key
  api_secret: test_api_secret
  testnet: true

system:
  event_queue_size: 10000
  max_concurrent_operations: 10
  startup_timeout_seconds: 30

risk:
  initial_capital: 10000.0
  max_position_size_pct: 0.02
  max_drawdown_pct: 0.10
  max_consecutive_losses: 3
  risk_per_trade_pct: 0.01

data:
  cache_size: 1000
  cache_ttl: 300
  default_interval: 1m
  klines_limit: 500

signals:
  min_confidence: 0.6
  signal_timeout_minutes: 60
  validation_required: true

strategies:
  default_strategy: ict
  enable_portfolio: false

execution:
  default_order_type: MARKET
  slippage_tolerance_pct: 0.001
  max_retries: 3
  retry_delay_ms: 1000
  dry_run: true

discord:
  enabled: false
  bot_token: "test_token"
  channel_id: "123456789"
  webhook_url: ""

trading:
  paper_trading: true
  max_positions: 5
  default_timeframe: "1h"

logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  log_to_file: false
"""
    
    config_file = tmp_path / "test_config.yml"
    config_file.write_text(config_content)
    return config_file


@pytest_asyncio.fixture
async def mock_binance_client(request) -> AsyncGenerator[MockBinanceClient, None]:
    """
    Provide a mock Binance client for testing.
    
    Can be parametrized with different scenarios:
    - trending: Clear uptrend with FVG and order block patterns
    - ranging: Sideways market movement
    - volatile: High volatility swings
    - low_liquidity: Wide spreads and low volume
    
    Usage:
        @pytest.mark.parametrize("scenario", ["trending"], indirect=["mock_binance_client"])
    """
    # Get scenario from parametrize or use default
    scenario = getattr(request, 'param', 'trending')
    
    client = MockBinanceClient(
        scenario=scenario,
        slippage_pct=0.001,
        order_delay_ms=50
    )
    
    yield client
    
    # Cleanup
    client.reset()


@pytest_asyncio.fixture
async def event_bus() -> AsyncGenerator[EventBus, None]:
    """Provide an EventBus instance for testing."""
    bus = EventBus(max_queue_size=10000)
    await bus.start()
    
    yield bus
    
    await bus.stop()


@pytest_asyncio.fixture
async def event_capture(request) -> AsyncGenerator[EventCapture, None]:
    """
    Provide an EventCapture instance for recording events.

    Uses system_integrator's event_bus if available, otherwise creates standalone.

    Yields:
        EventCapture instance (already started)
    """
    # Try to get system_integrator's event_bus first
    bus = None
    if 'system_integrator' in request.fixturenames:
        try:
            system = request.getfixturevalue('system_integrator')
            bus = system.event_bus
        except:
            pass

    # Fall back to standalone event_bus fixture
    if bus is None:
        bus = request.getfixturevalue('event_bus')

    capture = EventCapture(bus)
    await capture.start()

    yield capture

    await capture.stop()


@pytest.fixture
def performance_collector() -> Generator[PerformanceCollector, None, None]:
    """
    Provide a PerformanceCollector for metrics collection.
    
    Yields:
        PerformanceCollector instance
    """
    collector = PerformanceCollector()
    
    yield collector
    
    # Cleanup is synchronous
    collector.clear()


@pytest_asyncio.fixture
async def system_integrator(
    test_config_path: Path,
    mock_binance_client: MockBinanceClient
) -> AsyncGenerator[SystemIntegrator, None]:
    """
    Provide a fully initialized SystemIntegrator for E2E testing.
    
    The system is initialized with test configuration and mock Binance client.
    All components are started and ready for testing.
    
    Args:
        test_config_path: Path to test configuration
        mock_binance_client: Mock Binance client
        
    Yields:
        Started SystemIntegrator instance
    """
    # Create system with test config
    system = SystemIntegrator(
        config_path=str(test_config_path),
        environment="testing"
    )
    
    # Start system
    await system.start()
    
    # Replace real BinanceClient with mock
    if "binance_client" in system.components:
        # Stop real client
        await system.components["binance_client"].stop()
        # Replace with mock
        system.components["binance_client"] = mock_binance_client
        system.di_container.register_instance(type(mock_binance_client), mock_binance_client)
    
    yield system
    
    # Cleanup
    await system.stop()


@pytest_asyncio.fixture
async def lightweight_system(
    test_config_path: Path
) -> AsyncGenerator[SystemIntegrator, None]:
    """
    Provide a lightweight SystemIntegrator without full component initialization.
    
    Useful for testing specific components in isolation without full system overhead.
    
    Args:
        test_config_path: Path to test configuration
        
    Yields:
        SystemIntegrator with configuration loaded but components not started
    """
    system = SystemIntegrator(
        config_path=str(test_config_path),
        environment="testing"
    )
    
    # Only load configuration, don't start
    await system._load_configuration()
    await system._initialize_infrastructure()
    
    yield system
    
    # Cleanup infrastructure
    await system._stop()


@pytest.fixture
def sample_market_data():
    """
    Provide sample market data for testing.
    
    Returns:
        Dict with OHLCV data
    """
    return {
        "timestamp": 1704067200000,  # 2024-01-01 00:00:00
        "open": 50000.0,
        "high": 50500.0,
        "low": 49800.0,
        "close": 50200.0,
        "volume": 1000.0
    }


@pytest.fixture
def sample_signal_data():
    """
    Provide sample trading signal for testing.
    
    Returns:
        Dict with signal data
    """
    return {
        "signal_id": "test_signal_001",
        "symbol": "BTCUSDT",
        "direction": "LONG",
        "confidence": 0.85,
        "entry_price": 50000.0,
        "stop_loss": 49500.0,
        "take_profit": 51000.0,
        "pattern_type": "FVG",
        "timeframe": "1h",
        "timestamp": 1704067200000
    }


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "e2e: mark test as end-to-end integration test"
    )
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers",
        "performance: mark test as performance test"
    )


# Auto-use fixtures
@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton instances between tests."""
    from trading_bot.strategies.strategy_registry import StrategyRegistry
    
    # Reset strategy registry
    StrategyRegistry._instance = None
    
    yield
    
    # Cleanup after test
    StrategyRegistry._instance = None
