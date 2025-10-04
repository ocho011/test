"""Tests for configuration migration tools.

Tests the migration, validation, and comparison scripts.
"""

import json
import tempfile
from pathlib import Path

import pytest
import yaml

# Import migration tools
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "scripts"))

from config_migration import ConfigMigrator
from validate_config import ConfigValidator
from compare_environments import EnvironmentComparator


class TestConfigMigrator:
    """Tests for config_migration.py script."""

    @pytest.fixture
    def legacy_config(self):
        """Sample legacy configuration."""
        return {
            "capital": 50000.0,
            "leverage": 3,
            "symbols": ["BTCUSDT", "ETHUSDT"],
            "data": {
                "cache_size": 1000,
                "cache_ttl": 300,
                "historical_lookback_days": 90,
            },
            "risk": {
                "max_position_size": 10000.0,
                "max_portfolio_risk": 0.03,
                "risk_per_trade": 0.01,
                "max_drawdown": 0.15,
            },
            "strategy": {
                "default_timeframes": ["15m", "1h", "4h"],
                "confirmation_required": True,
                "min_signal_strength": 0.6,
            },
            "execution": {
                "order_type": "limit",
                "slippage_tolerance": 0.001,
                "retry_attempts": 3,
                "timeout": 30,
            },
            "notifications": {
                "enabled": True,
                "channels": ["discord"],
                "discord": {
                    "bot_token": "${TRADING_DISCORD_BOT_TOKEN}",
                    "channel_id": "${TRADING_DISCORD_CHANNEL_ID}",
                },
            },
            "logging": {
                "level": "INFO",
                "console": True,
                "file": True,
                "max_file_size": 10485760,
                "backup_count": 5,
            },
        }

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def migrator(self, legacy_config, temp_dir):
        """Create ConfigMigrator instance with test data."""
        # Write legacy config to file
        input_file = temp_dir / "legacy_config.yaml"
        with open(input_file, "w") as f:
            yaml.dump(legacy_config, f)

        output_dir = temp_dir / "environments"
        return ConfigMigrator(input_file, output_dir)

    def test_load_legacy_config(self, migrator):
        """Test loading legacy configuration file."""
        migrator.load_legacy_config()

        assert migrator.legacy_config is not None
        assert "capital" in migrator.legacy_config
        assert "leverage" in migrator.legacy_config
        assert migrator.legacy_config["capital"] == 50000.0

    def test_migrate_data_section(self, migrator):
        """Test migration of data section."""
        migrator.load_legacy_config()
        data = migrator._migrate_data_section(migrator.legacy_config["data"])

        assert "cache_size" in data
        assert "cache_ttl" in data
        assert "historical_lookback_days" in data
        assert data["cache_size"] == 1000
        assert data["cache_ttl"] == 300
        assert data["historical_lookback_days"] == 90

    def test_migrate_risk_section(self, migrator):
        """Test migration of risk section."""
        migrator.load_legacy_config()
        risk = migrator._migrate_risk_section(migrator.legacy_config["risk"])

        assert "initial_capital" in risk
        assert "max_position_size_pct" in risk
        assert "max_drawdown_pct" in risk
        assert "max_consecutive_losses" in risk
        assert "volatility_threshold" in risk

    def test_migrate_execution_section(self, migrator):
        """Test migration of execution section."""
        migrator.load_legacy_config()
        execution = migrator._migrate_execution_section(
            migrator.legacy_config["execution"]
        )

        assert "slippage_tolerance_pct" in execution
        assert "max_retry_attempts" in execution
        assert "order_timeout_seconds" in execution
        assert execution["slippage_tolerance_pct"] == 0.001
        assert execution["max_retry_attempts"] == 3
        assert execution["order_timeout_seconds"] == 30

    def test_migrate_trading_section(self, migrator):
        """Test migration of trading section from top-level fields."""
        migrator.load_legacy_config()
        trading = migrator._migrate_trading_section(migrator.legacy_config)

        assert "symbol" in trading
        assert "leverage" in trading
        assert "timeframe" in trading
        assert trading["symbol"] == "BTCUSDT"  # First symbol
        assert trading["leverage"] == 3

    def test_migrate_logging_section(self, migrator):
        """Test migration of logging section."""
        migrator.load_legacy_config()
        logging = migrator._migrate_logging_section(migrator.legacy_config["logging"])

        assert "level" in logging
        assert "format" in logging
        assert "max_bytes" in logging
        assert "backup_count" in logging
        assert "mask_sensitive_data" in logging
        assert logging["level"] == "INFO"
        assert logging["format"] == "json"

    def test_migrate_discord_section(self, migrator):
        """Test migration of notifications to discord section."""
        migrator.load_legacy_config()
        discord = migrator._migrate_discord_section(
            migrator.legacy_config["notifications"]
        )

        assert "bot_token" in discord
        assert "channel_id" in discord
        assert "enable_notifications" in discord
        assert discord["enable_notifications"] is True
        assert "${TRADING_DISCORD_BOT_TOKEN}" in discord["bot_token"]

    def test_migrate_to_new_structure(self, migrator):
        """Test full migration to new structure."""
        migrator.load_legacy_config()
        new_config = migrator.migrate_to_new_structure()

        # Check all major sections exist
        assert "data" in new_config
        assert "risk" in new_config
        assert "strategy" in new_config
        assert "execution" in new_config
        assert "trading" in new_config
        assert "logging" in new_config
        assert "discord" in new_config
        assert "binance" in new_config

    def test_generate_environment_overrides(self, migrator):
        """Test generation of environment-specific overrides."""
        migrator.load_legacy_config()
        overrides = migrator.generate_environment_overrides()

        # Check all environments exist
        assert "development" in overrides
        assert "paper" in overrides
        assert "paper-testnet" in overrides
        assert "paper-mainnet" in overrides
        assert "production" in overrides

        # Check environment-specific settings
        assert overrides["development"]["binance"]["network"] == "testnet"
        assert overrides["production"]["binance"]["network"] == "mainnet"
        assert overrides["production"]["binance"]["execution_mode"] == "live"

    def test_write_environment_config(self, migrator, temp_dir):
        """Test writing environment configuration file."""
        migrator.load_legacy_config()
        new_config = migrator.migrate_to_new_structure()

        # Write development config
        migrator.write_environment_config("development", new_config)

        # Verify file was created
        output_file = temp_dir / "environments" / "development.yml"
        assert output_file.exists()

        # Verify content
        with open(output_file) as f:
            loaded_config = yaml.safe_load(f)

        assert "data" in loaded_config
        assert "trading" in loaded_config

    def test_deep_merge(self, migrator):
        """Test deep merge of dictionaries."""
        base = {"a": {"b": 1, "c": 2}, "d": 3}
        override = {"a": {"b": 10, "e": 4}, "f": 5}

        result = migrator._deep_merge(base, override)

        assert result["a"]["b"] == 10  # Override
        assert result["a"]["c"] == 2  # Keep from base
        assert result["a"]["e"] == 4  # New from override
        assert result["d"] == 3  # Keep from base
        assert result["f"] == 5  # New from override


class TestConfigValidator:
    """Tests for validate_config.py script."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def valid_config(self):
        """Sample valid configuration."""
        return {
            "data": {
                "cache_size": 1000,
                "cache_ttl": 300,
                "historical_lookback_days": 60,
            },
            "risk": {
                "initial_capital": 10000.0,
                "max_position_size_pct": 0.1,
                "max_drawdown_pct": 0.2,
                "max_consecutive_losses": 3,
                "volatility_threshold": 0.05,
            },
            "strategy": {
                "enabled_strategies": ["ict"],
                "default_strategy": "ict",
                "strategy_selection_mode": "auto",
            },
            "execution": {
                "slippage_tolerance_pct": 0.001,
                "max_retry_attempts": 3,
                "order_timeout_seconds": 30,
            },
            "trading": {
                "symbol": "BTCUSDT",
                "timeframe": "1h",
                "max_position_size": 0.1,
                "leverage": 1,
                "max_risk_per_trade": 0.02,
                "daily_loss_limit": 0.05,
            },
            "system": {
                "async_workers": 2,
                "event_queue_size": 1000,
                "health_check_interval": 300,
                "memory_threshold_mb": 1024,
            },
            "logging": {
                "level": "INFO",
                "format": "json",
                "output_dir": "logs",
                "max_bytes": 10485760,
                "backup_count": 5,
                "rotation_time": "midnight",
                "trade_log_enabled": True,
                "trade_log_file": "trades.log",
                "performance_log_enabled": True,
                "performance_log_file": "performance.log",
                "mask_sensitive_data": True,
                "debug_unmask": False,
            },
            "discord": {
                "bot_token": "test_" + ("x" * 64),  # Mock token (70 chars)
                "channel_id": 123456789012345678,
                "enable_notifications": True,
                "spam_protection_seconds": 60,
            },
            "binance": {
                "api_key": "test_api_key_for_validation_purposes_only",
                "api_secret": "test_api_secret_for_validation_purposes",
                "testnet": True,
            },
        }

    @pytest.fixture
    def validator(self, temp_dir):
        """Create ConfigValidator instance."""
        return ConfigValidator(temp_dir / "environments")

    def test_load_yaml_file(self, validator, temp_dir, valid_config):
        """Test loading YAML configuration file."""
        # Create test file
        config_dir = temp_dir / "environments"
        config_dir.mkdir(parents=True)
        test_file = config_dir / "test.yml"

        with open(test_file, "w") as f:
            yaml.dump(valid_config, f)

        # Load file
        loaded = validator.load_yaml_file(test_file)

        assert loaded == valid_config

    def test_find_environment_files(self, validator, temp_dir, valid_config):
        """Test finding environment configuration files."""
        # Create test files
        config_dir = temp_dir / "environments"
        config_dir.mkdir(parents=True)

        for env in ["development", "production", "base"]:
            test_file = config_dir / f"{env}.yml"
            with open(test_file, "w") as f:
                yaml.dump(valid_config, f)

        # Find files
        files = validator.find_environment_files()

        # Should exclude base.yml
        assert len(files) == 2
        assert all(f.stem != "base" for f in files)

    def test_validate_valid_config(self, validator, temp_dir, valid_config):
        """Test validation of a valid configuration."""
        # Create test file
        config_dir = temp_dir / "environments"
        config_dir.mkdir(parents=True)
        test_file = config_dir / "development.yml"

        with open(test_file, "w") as f:
            yaml.dump(valid_config, f)

        # Validate
        result = validator.validate_config_file(test_file)

        # Print errors if validation failed (for debugging)
        if not result["valid"]:
            print("\nValidation errors:")
            for error in result["errors"]:
                print(f"  - {error}")

        assert result["valid"] is True
        assert len(result["errors"]) == 0

    def test_validate_invalid_config(self, validator, temp_dir):
        """Test validation of an invalid configuration."""
        # Create invalid config (missing required fields)
        invalid_config = {"data": {"cache_size": 1000}}

        config_dir = temp_dir / "environments"
        config_dir.mkdir(parents=True)
        test_file = config_dir / "invalid.yml"

        with open(test_file, "w") as f:
            yaml.dump(invalid_config, f)

        # Validate
        result = validator.validate_config_file(test_file)

        assert result["valid"] is False
        assert len(result["errors"]) > 0

    def test_find_env_var_references(self, validator):
        """Test finding environment variable references."""
        # Create config with env var references
        config_with_env_vars = {
            "binance": {
                "api_key": "${TRADING_BINANCE_API_KEY}",
                "api_secret": "${TRADING_BINANCE_API_SECRET}",
            },
            "discord": {
                "bot_token": "${TRADING_DISCORD_BOT_TOKEN}",
                "channel_id": "${TRADING_DISCORD_CHANNEL_ID}",
            },
        }
        
        env_vars = validator._find_env_var_references(config_with_env_vars)

        assert "TRADING_BINANCE_API_KEY" in env_vars
        assert "TRADING_BINANCE_API_SECRET" in env_vars
        assert "TRADING_DISCORD_BOT_TOKEN" in env_vars
        assert "TRADING_DISCORD_CHANNEL_ID" in env_vars


class TestEnvironmentComparator:
    """Tests for compare_environments.py script."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def dev_config(self):
        """Development environment configuration."""
        return {
            "trading": {"symbol": "BTCUSDT", "leverage": 1},
            "binance": {"network": "testnet", "execution_mode": "paper"},
            "logging": {"level": "DEBUG"},
        }

    @pytest.fixture
    def prod_config(self):
        """Production environment configuration."""
        return {
            "trading": {"symbol": "BTCUSDT", "leverage": 3},
            "binance": {"network": "mainnet", "execution_mode": "live"},
            "logging": {"level": "INFO"},
        }

    @pytest.fixture
    def comparator(self, temp_dir):
        """Create EnvironmentComparator instance."""
        return EnvironmentComparator(temp_dir / "environments")

    def test_load_environment_configs(
        self, comparator, temp_dir, dev_config, prod_config
    ):
        """Test loading environment configuration files."""
        # Create test files
        config_dir = temp_dir / "environments"
        config_dir.mkdir(parents=True)

        with open(config_dir / "development.yml", "w") as f:
            yaml.dump(dev_config, f)

        with open(config_dir / "production.yml", "w") as f:
            yaml.dump(prod_config, f)

        # Load configs
        comparator.load_environment_configs()

        assert len(comparator.environments) == 2
        assert "development" in comparator.environments
        assert "production" in comparator.environments

    def test_get_all_config_paths(self, comparator, temp_dir, dev_config, prod_config):
        """Test extracting all configuration paths."""
        # Create test files
        config_dir = temp_dir / "environments"
        config_dir.mkdir(parents=True)

        with open(config_dir / "development.yml", "w") as f:
            yaml.dump(dev_config, f)

        with open(config_dir / "production.yml", "w") as f:
            yaml.dump(prod_config, f)

        # Load and extract paths
        comparator.load_environment_configs()
        paths = comparator.get_all_config_paths()

        # Check paths exist
        assert ("trading",) in paths
        assert ("trading", "symbol") in paths
        assert ("trading", "leverage") in paths
        assert ("binance", "network") in paths
        assert ("logging", "level") in paths

    def test_compare_setting(self, comparator, temp_dir, dev_config, prod_config):
        """Test comparing a specific setting across environments."""
        # Create test files
        config_dir = temp_dir / "environments"
        config_dir.mkdir(parents=True)

        with open(config_dir / "development.yml", "w") as f:
            yaml.dump(dev_config, f)

        with open(config_dir / "production.yml", "w") as f:
            yaml.dump(prod_config, f)

        # Load and compare
        comparator.load_environment_configs()
        result = comparator.compare_setting(("trading", "leverage"))

        assert result["path"] == "trading.leverage"
        assert result["values"]["development"] == 1
        assert result["values"]["production"] == 3
        assert result["different"] is True

    def test_compare_all(self, comparator, temp_dir, dev_config, prod_config):
        """Test comparing all settings across environments."""
        # Create test files
        config_dir = temp_dir / "environments"
        config_dir.mkdir(parents=True)

        with open(config_dir / "development.yml", "w") as f:
            yaml.dump(dev_config, f)

        with open(config_dir / "production.yml", "w") as f:
            yaml.dump(prod_config, f)

        # Load and compare
        comparator.load_environment_configs()
        comparisons = comparator.compare_all()

        # Should have comparisons for all settings
        assert len(comparisons) > 0

        # Check for specific differences
        leverage_comparison = next(
            c for c in comparisons if c["path"] == "trading.leverage"
        )
        assert leverage_comparison["different"] is True

    def test_deep_merge(self, comparator):
        """Test deep merging of configurations."""
        base = {"a": {"b": 1, "c": 2}, "d": 3}
        override = {"a": {"b": 10}, "e": 4}

        result = comparator._deep_merge(base, override)

        assert result["a"]["b"] == 10  # Overridden
        assert result["a"]["c"] == 2  # Preserved from base
        assert result["d"] == 3  # Preserved from base
        assert result["e"] == 4  # Added from override


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
