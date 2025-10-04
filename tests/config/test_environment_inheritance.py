"""Tests for environment configuration inheritance and standardization."""
import os
import pytest
from pathlib import Path
from trading_bot.config.config_manager import ConfigManager


@pytest.fixture
def config_manager():
    """Create a fresh config manager instance for each test."""
    # Clear singleton
    ConfigManager._instance = None
    ConfigManager._config = None
    return ConfigManager()


@pytest.fixture
def environments_dir():
    """Get path to environments directory."""
    return Path(__file__).parent.parent.parent / "src" / "trading_bot" / "config" / "environments"


class TestEnvironmentFiles:
    """Test that all required environment files exist."""

    def test_base_config_exists(self, environments_dir):
        """Test that base.yml exists."""
        base_file = environments_dir / "base.yml"
        assert base_file.exists(), "base.yml should exist"

    def test_production_config_exists(self, environments_dir):
        """Test that production.yml exists."""
        prod_file = environments_dir / "production.yml"
        assert prod_file.exists(), "production.yml should exist"

    def test_paper_mainnet_config_exists(self, environments_dir):
        """Test that paper-mainnet.yml exists."""
        paper_mainnet_file = environments_dir / "paper-mainnet.yml"
        assert paper_mainnet_file.exists(), "paper-mainnet.yml should exist"

    def test_paper_testnet_config_exists(self, environments_dir):
        """Test that paper-testnet.yml exists."""
        paper_testnet_file = environments_dir / "paper-testnet.yml"
        assert paper_testnet_file.exists(), "paper-testnet.yml should exist"

    def test_development_config_exists(self, environments_dir):
        """Test that development.yml exists."""
        dev_file = environments_dir / "development.yml"
        assert dev_file.exists(), "development.yml should exist"


class TestConfigInheritance:
    """Test configuration inheritance mechanism."""

    def test_deep_merge_basic(self, config_manager):
        """Test basic deep merge functionality."""
        base = {
            'system': {'debug': False, 'workers': 2},
            'binance': {'testnet': True}
        }
        override = {
            'system': {'debug': True},
            'logging': {'level': 'DEBUG'}
        }

        result = config_manager._deep_merge(base, override)

        assert result['system']['debug'] is True  # Overridden
        assert result['system']['workers'] == 2  # Preserved from base
        assert result['binance']['testnet'] is True  # Preserved from base
        assert result['logging']['level'] == 'DEBUG'  # New from override

    def test_deep_merge_nested(self, config_manager):
        """Test deep merge with nested dictionaries."""
        base = {
            'logging': {
                'level': 'INFO',
                'format': 'json',
                'rotation': {
                    'max_bytes': 10485760,
                    'backup_count': 5
                }
            }
        }
        override = {
            'logging': {
                'level': 'DEBUG',
                'rotation': {
                    'backup_count': 3
                }
            }
        }

        result = config_manager._deep_merge(base, override)

        assert result['logging']['level'] == 'DEBUG'  # Overridden
        assert result['logging']['format'] == 'json'  # Preserved
        assert result['logging']['rotation']['max_bytes'] == 10485760  # Preserved
        assert result['logging']['rotation']['backup_count'] == 3  # Overridden


class TestEnvironmentLoading:
    """Test loading different environment configurations."""

    @pytest.fixture(autouse=True)
    def setup_env_vars(self):
        """Set required environment variables for tests."""
        os.environ['TRADING_BINANCE_API_KEY'] = 'test_api_key'
        os.environ['TRADING_BINANCE_API_SECRET'] = 'test_api_secret'
        os.environ['TRADING_DISCORD_BOT_TOKEN'] = 'test_bot_token'
        os.environ['TRADING_DISCORD_CHANNEL_ID'] = '12345'
        yield
        # Cleanup
        for key in ['TRADING_BINANCE_API_KEY', 'TRADING_BINANCE_API_SECRET',
                    'TRADING_DISCORD_BOT_TOKEN', 'TRADING_DISCORD_CHANNEL_ID']:
            os.environ.pop(key, None)

    def test_load_production_config(self, config_manager):
        """Test loading production environment config."""
        config = config_manager.load_config(environment='production')

        assert config.system.environment == 'production'
        assert config.binance.testnet is False
        assert config.custom.get('dry_run') is False
        assert config.system.debug is False

    def test_load_paper_mainnet_config(self, config_manager):
        """Test loading paper-mainnet environment config."""
        config = config_manager.load_config(environment='paper-mainnet')

        assert config.system.environment == 'paper-mainnet'
        assert config.binance.testnet is False
        assert config.custom.get('dry_run') is True
        assert config.custom.get('paper_mode') is True

    def test_load_paper_testnet_config(self, config_manager):
        """Test loading paper-testnet environment config."""
        config = config_manager.load_config(environment='paper-testnet')

        assert config.system.environment == 'paper-testnet'
        assert config.binance.testnet is True
        assert config.custom.get('dry_run') is True
        assert config.custom.get('paper_mode') is True

    def test_load_development_config(self, config_manager):
        """Test loading development environment config."""
        config = config_manager.load_config(environment='development')

        assert config.system.environment == 'development'
        assert config.binance.testnet is True
        assert config.custom.get('dry_run') is False
        assert config.system.debug is True

    def test_inheritance_from_base(self, config_manager):
        """Test that environments inherit from base.yml."""
        config = config_manager.load_config(environment='development')

        # These should come from base.yml
        assert hasattr(config, 'data')
        assert hasattr(config, 'risk')
        assert hasattr(config, 'strategies')
        assert hasattr(config, 'execution')

        # Check some base values
        assert config.strategies.default_strategy == 'ict'
        assert 'ict' in config.strategies.enabled_strategies


class TestEnvironmentMapping:
    """Test environment to binance/execution mapping."""

    @pytest.fixture(autouse=True)
    def setup_env_vars(self):
        """Set required environment variables for tests."""
        os.environ['TRADING_BINANCE_API_KEY'] = 'test_api_key'
        os.environ['TRADING_BINANCE_API_SECRET'] = 'test_api_secret'
        os.environ['TRADING_DISCORD_BOT_TOKEN'] = 'test_bot_token'
        os.environ['TRADING_DISCORD_CHANNEL_ID'] = '12345'
        yield
        # Cleanup
        for key in ['TRADING_BINANCE_API_KEY', 'TRADING_BINANCE_API_SECRET',
                    'TRADING_DISCORD_BOT_TOKEN', 'TRADING_DISCORD_CHANNEL_ID']:
            os.environ.pop(key, None)

    def test_production_mapping(self, config_manager):
        """Test production environment mapping: mainnet + live."""
        config_manager.load_config(environment='production')
        info = config_manager.get_environment_info()

        assert info['environment'] == 'production'
        assert info['binance_network'] == 'mainnet'
        assert info['execution_mode'] == 'live'
        assert info['dry_run'] is False
        assert info['testnet'] is False

    def test_paper_mainnet_mapping(self, config_manager):
        """Test paper-mainnet environment mapping: mainnet + simulation."""
        config_manager.load_config(environment='paper-mainnet')
        info = config_manager.get_environment_info()

        assert info['environment'] == 'paper-mainnet'
        assert info['binance_network'] == 'mainnet'
        assert info['execution_mode'] == 'simulation'
        assert info['dry_run'] is True
        assert info['testnet'] is False

    def test_paper_testnet_mapping(self, config_manager):
        """Test paper-testnet environment mapping: testnet + simulation."""
        config_manager.load_config(environment='paper-testnet')
        info = config_manager.get_environment_info()

        assert info['environment'] == 'paper-testnet'
        assert info['binance_network'] == 'testnet'
        assert info['execution_mode'] == 'simulation'
        assert info['dry_run'] is True
        assert info['testnet'] is True

    def test_development_mapping(self, config_manager):
        """Test development environment mapping: testnet + live."""
        config_manager.load_config(environment='development')
        info = config_manager.get_environment_info()

        assert info['environment'] == 'development'
        assert info['binance_network'] == 'testnet'
        assert info['execution_mode'] == 'live'
        assert info['dry_run'] is False
        assert info['testnet'] is True

    def test_list_available_environments(self):
        """Test listing all available environments."""
        envs = ConfigManager.list_available_environments()

        assert 'production' in envs
        assert 'paper-mainnet' in envs
        assert 'paper-testnet' in envs
        assert 'development' in envs

        # Verify production mapping
        assert envs['production']['binance_network'] == 'mainnet'
        assert envs['production']['execution_mode'] == 'live'
        assert envs['production']['dry_run'] is False

        # Verify paper-mainnet mapping
        assert envs['paper-mainnet']['binance_network'] == 'mainnet'
        assert envs['paper-mainnet']['execution_mode'] == 'simulation'
        assert envs['paper-mainnet']['dry_run'] is True

        # Verify paper-testnet mapping
        assert envs['paper-testnet']['binance_network'] == 'testnet'
        assert envs['paper-testnet']['execution_mode'] == 'simulation'
        assert envs['paper-testnet']['dry_run'] is True

        # Verify development mapping
        assert envs['development']['binance_network'] == 'testnet'
        assert envs['development']['execution_mode'] == 'live'
        assert envs['development']['dry_run'] is False


class TestEnvironmentValidation:
    """Test environment configuration validation."""

    @pytest.fixture(autouse=True)
    def setup_env_vars(self):
        """Set required environment variables for tests."""
        os.environ['TRADING_BINANCE_API_KEY'] = 'test_api_key'
        os.environ['TRADING_BINANCE_API_SECRET'] = 'test_api_secret'
        os.environ['TRADING_DISCORD_BOT_TOKEN'] = 'test_bot_token'
        os.environ['TRADING_DISCORD_CHANNEL_ID'] = '12345'
        yield
        # Cleanup
        for key in ['TRADING_BINANCE_API_KEY', 'TRADING_BINANCE_API_SECRET',
                    'TRADING_DISCORD_BOT_TOKEN', 'TRADING_DISCORD_CHANNEL_ID']:
            os.environ.pop(key, None)

    def test_all_environments_load_successfully(self, config_manager):
        """Test that all standard environments can be loaded without errors."""
        environments = ['production', 'paper-mainnet', 'paper-testnet', 'development']

        for env in environments:
            # Clear config between loads
            ConfigManager._instance = None
            ConfigManager._config = None
            config_manager = ConfigManager()

            # Should not raise any exception
            config = config_manager.load_config(environment=env)
            assert config is not None
            assert config.system.environment == env

    def test_env_var_override_works_with_inheritance(self, config_manager):
        """Test that environment variable overrides work with inherited configs."""
        # Set a custom log level via env var
        os.environ['TRADING_LOG_LEVEL'] = 'WARNING'

        config = config_manager.load_config(environment='development')

        # Should be overridden by env var
        assert config.logging.level.value == 'WARNING'

        # Cleanup
        os.environ.pop('TRADING_LOG_LEVEL', None)
