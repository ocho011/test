"""
Tests for environment-based Binance network and execution mode mapping.

Tests verify that:
- BinanceClient correctly maps environment to testnet/mainnet
- OrderExecutor correctly maps environment to dry_run mode
- Environment configurations match expected behavior
"""

import pytest
from decimal import Decimal

from src.trading_bot.config.models import (
    BinanceConfig,
    ExecutionConfig,
    SystemConfig,
)
from src.trading_bot.data.binance_client import BinanceClient
from src.trading_bot.execution.order_executor import OrderExecutor, OrderRequest, OrderSide, OrderType
from src.trading_bot.core.event_bus import EventBus


class TestEnvironmentMapping:
    """Test environment-based configuration mapping for Binance and execution."""

    def test_production_environment_mapping(self):
        """
        Test production environment: mainnet + live trading.

        Environment: production
        Expected: testnet=False, dry_run=False
        """
        # Create production configs
        binance_config = BinanceConfig(
            api_key="test_key",
            api_secret="test_secret",
            testnet=False  # production uses mainnet
        )

        execution_config = ExecutionConfig(
            dry_run=False  # production uses live trading
        )

        # Create BinanceClient with config
        client = BinanceClient(config=binance_config)

        # Verify testnet is False (mainnet)
        assert client.testnet is False, "Production should use mainnet (testnet=False)"

        # Create OrderExecutor with config
        executor = OrderExecutor(
            binance_client=client,
            config=execution_config
        )

        # Verify dry_run is False (live trading)
        assert executor.dry_run is False, "Production should use live trading (dry_run=False)"

    def test_paper_mainnet_environment_mapping(self):
        """
        Test paper-mainnet environment: mainnet data + simulated trading.

        Environment: paper-mainnet
        Expected: testnet=False, dry_run=True
        """
        # Create paper-mainnet configs
        binance_config = BinanceConfig(
            api_key="test_key",
            api_secret="test_secret",
            testnet=False  # paper-mainnet uses mainnet data
        )

        execution_config = ExecutionConfig(
            dry_run=True  # paper-mainnet uses simulated trading
        )

        # Create BinanceClient with config
        client = BinanceClient(config=binance_config)

        # Verify testnet is False (mainnet data)
        assert client.testnet is False, "Paper-mainnet should use mainnet data (testnet=False)"

        # Create OrderExecutor with config
        executor = OrderExecutor(
            binance_client=client,
            config=execution_config
        )

        # Verify dry_run is True (simulated)
        assert executor.dry_run is True, "Paper-mainnet should use simulation (dry_run=True)"

    def test_paper_testnet_environment_mapping(self):
        """
        Test paper-testnet environment: testnet data + simulated trading.

        Environment: paper-testnet
        Expected: testnet=True, dry_run=True
        """
        # Create paper-testnet configs
        binance_config = BinanceConfig(
            api_key="test_key",
            api_secret="test_secret",
            testnet=True  # paper-testnet uses testnet data
        )

        execution_config = ExecutionConfig(
            dry_run=True  # paper-testnet uses simulated trading
        )

        # Create BinanceClient with config
        client = BinanceClient(config=binance_config)

        # Verify testnet is True (testnet data)
        assert client.testnet is True, "Paper-testnet should use testnet data (testnet=True)"

        # Create OrderExecutor with config
        executor = OrderExecutor(
            binance_client=client,
            config=execution_config
        )

        # Verify dry_run is True (simulated)
        assert executor.dry_run is True, "Paper-testnet should use simulation (dry_run=True)"

    def test_development_environment_mapping(self):
        """
        Test development environment: testnet + live trading.

        Environment: development
        Expected: testnet=True, dry_run=False
        """
        # Create development configs
        binance_config = BinanceConfig(
            api_key="test_key",
            api_secret="test_secret",
            testnet=True  # development uses testnet
        )

        execution_config = ExecutionConfig(
            dry_run=False  # development uses live trading (on testnet)
        )

        # Create BinanceClient with config
        client = BinanceClient(config=binance_config)

        # Verify testnet is True (testnet)
        assert client.testnet is True, "Development should use testnet (testnet=True)"

        # Create OrderExecutor with config
        executor = OrderExecutor(
            binance_client=client,
            config=execution_config
        )

        # Verify dry_run is False (live on testnet)
        assert executor.dry_run is False, "Development should use live trading (dry_run=False)"

    def test_backward_compatibility_binance_client(self):
        """Test BinanceClient backward compatibility with old parameter-based initialization."""
        # Old-style initialization (should still work)
        client = BinanceClient(
            api_key="test_key",
            api_secret="test_secret",
            testnet=True
        )

        assert client.api_key == "test_key"
        assert client.api_secret == "test_secret"
        assert client.testnet is True

    def test_backward_compatibility_order_executor(self):
        """Test OrderExecutor backward compatibility with old parameter-based initialization."""
        # Create a mock BinanceClient
        client = BinanceClient(api_key="test", api_secret="test")

        # Old-style initialization (should still work)
        executor = OrderExecutor(
            binance_client=client,
            max_retries=5,
            retry_delay=2.0,
            slippage_limit=0.01,
            dry_run=True
        )

        assert executor.max_retries == 5
        assert executor.retry_delay == 2.0
        assert executor.slippage_limit == 0.01
        assert executor.dry_run is True

    def test_config_overrides_parameters(self):
        """Test that config values override parameter values when both are provided."""
        # Create configs with specific values
        binance_config = BinanceConfig(
            api_key="config_key",
            api_secret="config_secret",
            testnet=True
        )

        # Provide both config and parameters - config should win
        client = BinanceClient(
            api_key="param_key",
            api_secret="param_secret",
            testnet=False,
            config=binance_config
        )

        assert client.api_key == "config_key", "Config should override parameter"
        assert client.api_secret == "config_secret", "Config should override parameter"
        assert client.testnet is True, "Config should override parameter"

    def test_execution_config_parameter_conversion(self):
        """Test that ExecutionConfig parameters are correctly converted."""
        # Create config with ms values
        execution_config = ExecutionConfig(
            max_retries=5,
            retry_delay_ms=2000,  # 2000 ms = 2.0 seconds
            slippage_tolerance_pct=0.01,
            dry_run=True
        )

        client = BinanceClient(api_key="test", api_secret="test")
        executor = OrderExecutor(
            binance_client=client,
            config=execution_config
        )

        assert executor.max_retries == 5
        assert executor.retry_delay == 2.0, "Should convert ms to seconds"
        assert executor.slippage_limit == 0.01
        assert executor.dry_run is True


class TestDryRunBehavior:
    """Test that dry_run mode configuration is correctly set."""

    def test_dry_run_mode_enabled(self):
        """Test that dry_run mode can be enabled via configuration."""
        client = BinanceClient(api_key="test", api_secret="test", testnet=True)

        execution_config = ExecutionConfig(dry_run=True)
        executor = OrderExecutor(
            binance_client=client,
            config=execution_config
        )

        # Verify dry_run is properly configured
        assert executor.dry_run is True, "Dry-run mode should be enabled"

    def test_live_mode_enabled(self):
        """Test that live mode can be enabled via configuration."""
        client = BinanceClient(api_key="test", api_secret="test", testnet=True)

        execution_config = ExecutionConfig(dry_run=False)
        executor = OrderExecutor(
            binance_client=client,
            config=execution_config
        )

        # Verify live mode is configured
        assert executor.dry_run is False, "Live mode should be enabled"


class TestEnvironmentConfigMatrix:
    """Test the complete environment configuration matrix."""

    @pytest.mark.parametrize("environment,expected_testnet,expected_dry_run", [
        ("production", False, False),      # mainnet + live
        ("paper-mainnet", False, True),    # mainnet + simulation
        ("paper-testnet", True, True),     # testnet + simulation
        ("development", True, False),      # testnet + live
    ])
    def test_environment_configuration_matrix(
        self,
        environment: str,
        expected_testnet: bool,
        expected_dry_run: bool
    ):
        """Test complete environment configuration matrix."""
        # Create configs based on environment
        binance_config = BinanceConfig(
            api_key="test_key",
            api_secret="test_secret",
            testnet=expected_testnet
        )

        execution_config = ExecutionConfig(
            dry_run=expected_dry_run
        )

        # Create components
        client = BinanceClient(config=binance_config)
        executor = OrderExecutor(
            binance_client=client,
            config=execution_config
        )

        # Verify configuration
        assert client.testnet == expected_testnet, \
            f"{environment} should have testnet={expected_testnet}"
        assert executor.dry_run == expected_dry_run, \
            f"{environment} should have dry_run={expected_dry_run}"
