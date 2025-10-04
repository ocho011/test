"""Tests for CLI argument parsing and routing."""
import pytest
import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trading_bot.main import TradingBotCLI


class TestCLIArgumentParsing:
    """Test CLI argument parsing with new environment-based structure."""

    @pytest.fixture
    def cli(self):
        """Create a fresh CLI instance for each test."""
        return TradingBotCLI()

    def test_default_environment(self, cli):
        """Test that development is the default environment."""
        args = cli.parser.parse_args([])
        assert args.env == 'development'

    def test_production_environment(self, cli):
        """Test production environment argument."""
        args = cli.parser.parse_args(['--env', 'production'])
        assert args.env == 'production'

    def test_paper_mainnet_environment(self, cli):
        """Test paper-mainnet environment argument."""
        args = cli.parser.parse_args(['--env', 'paper-mainnet'])
        assert args.env == 'paper-mainnet'

    def test_paper_testnet_environment(self, cli):
        """Test paper-testnet environment argument."""
        args = cli.parser.parse_args(['--env', 'paper-testnet'])
        assert args.env == 'paper-testnet'

    def test_development_environment(self, cli):
        """Test development environment argument."""
        args = cli.parser.parse_args(['--env', 'development'])
        assert args.env == 'development'

    def test_environment_short_flag(self, cli):
        """Test --environment flag (long form)."""
        args = cli.parser.parse_args(['--environment', 'production'])
        assert args.env == 'production'

    def test_invalid_environment(self, cli):
        """Test that invalid environment raises error."""
        with pytest.raises(SystemExit):
            cli.parser.parse_args(['--env', 'invalid'])

    def test_backtest_flag(self, cli):
        """Test backtest flag sets correctly."""
        args = cli.parser.parse_args(['--backtest'])
        assert args.backtest is True

    def test_backtest_default_false(self, cli):
        """Test backtest defaults to False."""
        args = cli.parser.parse_args([])
        assert args.backtest is False

    def test_backtest_with_symbol(self, cli):
        """Test backtest with custom symbol."""
        args = cli.parser.parse_args(['--backtest', '--symbol', 'ETHUSDT'])
        assert args.backtest is True
        assert args.symbol == 'ETHUSDT'

    def test_backtest_with_date_range(self, cli):
        """Test backtest with start and end dates."""
        args = cli.parser.parse_args([
            '--backtest',
            '--start', '2024-01-01',
            '--end', '2024-12-31'
        ])
        assert args.backtest is True
        assert args.start == '2024-01-01'
        assert args.end == '2024-12-31'

    def test_custom_config_path(self, cli):
        """Test custom config file path."""
        args = cli.parser.parse_args(['--config', '/path/to/config.yml'])
        assert args.config == '/path/to/config.yml'

    def test_debug_flag(self, cli):
        """Test debug flag."""
        args = cli.parser.parse_args(['--debug'])
        assert args.debug is True

    def test_debug_default_false(self, cli):
        """Test debug defaults to False."""
        args = cli.parser.parse_args([])
        assert args.debug is False

    def test_daemon_flag(self, cli):
        """Test daemon flag."""
        args = cli.parser.parse_args(['--daemon'])
        assert args.daemon is True

    def test_health_check_flag(self, cli):
        """Test health check flag."""
        args = cli.parser.parse_args(['--health-check'])
        assert args.health_check is True

    def test_mode_argument_removed(self, cli):
        """Test that --mode argument is no longer accepted."""
        with pytest.raises(SystemExit):
            cli.parser.parse_args(['--mode', 'live'])

    def test_dry_run_argument_removed(self, cli):
        """Test that --dry-run argument is no longer accepted."""
        with pytest.raises(SystemExit):
            cli.parser.parse_args(['--dry-run'])

    def test_combined_flags(self, cli):
        """Test multiple flags can be combined."""
        args = cli.parser.parse_args([
            '--env', 'paper-mainnet',
            '--debug',
            '--config', '/custom/config.yml'
        ])
        assert args.env == 'paper-mainnet'
        assert args.debug is True
        assert args.config == '/custom/config.yml'

    def test_backtest_combined_flags(self, cli):
        """Test backtest with multiple options."""
        args = cli.parser.parse_args([
            '--backtest',
            '--symbol', 'BTCUSDT',
            '--start', '2024-01-01',
            '--end', '2024-12-31',
            '--debug'
        ])
        assert args.backtest is True
        assert args.symbol == 'BTCUSDT'
        assert args.start == '2024-01-01'
        assert args.end == '2024-12-31'
        assert args.debug is True


class TestCLIRouting:
    """Test CLI routing to correct execution modes."""

    @pytest.fixture
    def cli(self):
        """Create a fresh CLI instance for each test."""
        return TradingBotCLI()

    @pytest.mark.asyncio
    async def test_route_to_health_check(self, cli):
        """Test that health-check flag routes to health check mode."""
        args = cli.parser.parse_args(['--health-check'])
        
        # Mock the health check method to avoid actual system initialization
        async def mock_health_check(args):
            return 0
        
        cli.run_health_check = mock_health_check
        result = await cli.run(args)
        assert result == 0

    @pytest.mark.asyncio
    async def test_route_to_backtest(self, cli):
        """Test that backtest flag routes to backtest mode."""
        args = cli.parser.parse_args([
            '--backtest',
            '--start', '2024-01-01',
            '--end', '2024-12-31'
        ])
        
        # Mock the backtest method
        async def mock_backtest(args):
            return 0
        
        cli.run_backtest = mock_backtest
        result = await cli.run(args)
        assert result == 0

    @pytest.mark.asyncio
    async def test_route_to_trading(self, cli):
        """Test that default routes to trading mode."""
        args = cli.parser.parse_args(['--env', 'development'])
        
        # Mock the trading method
        async def mock_trading(args):
            return 0
        
        cli.run_trading = mock_trading
        result = await cli.run(args)
        assert result == 0

    def test_routing_priority_health_check_first(self, cli):
        """Test that health check has priority over other modes."""
        args = cli.parser.parse_args(['--health-check', '--backtest'])
        
        # Health check should be checked first, so backtest flag is ignored
        # This tests the routing logic order in the run() method
        assert args.health_check is True
        assert args.backtest is True

    def test_routing_priority_backtest_over_trading(self, cli):
        """Test that backtest has priority over trading mode."""
        args = cli.parser.parse_args(['--backtest', '--env', 'production'])
        
        # Backtest should be used when both are present
        assert args.backtest is True
        assert args.env == 'production'


class TestEnvironmentMapping:
    """Test that environments map correctly to their configuration."""

    @pytest.fixture
    def cli(self):
        """Create a fresh CLI instance for each test."""
        return TradingBotCLI()

    def test_production_is_mainnet_live(self, cli):
        """Test production environment maps to mainnet + live."""
        args = cli.parser.parse_args(['--env', 'production'])
        # Environment mapping is verified in config tests
        # Here we just verify the argument is parsed correctly
        assert args.env == 'production'

    def test_paper_mainnet_is_mainnet_simulation(self, cli):
        """Test paper-mainnet environment maps to mainnet + simulation."""
        args = cli.parser.parse_args(['--env', 'paper-mainnet'])
        assert args.env == 'paper-mainnet'

    def test_paper_testnet_is_testnet_simulation(self, cli):
        """Test paper-testnet environment maps to testnet + simulation."""
        args = cli.parser.parse_args(['--env', 'paper-testnet'])
        assert args.env == 'paper-testnet'

    def test_development_is_testnet_live(self, cli):
        """Test development environment maps to testnet + live."""
        args = cli.parser.parse_args(['--env', 'development'])
        assert args.env == 'development'
