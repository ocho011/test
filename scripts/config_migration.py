#!/usr/bin/env python3
"""Configuration Migration Tool

Migrates legacy trading_config.yaml to new environment-based configuration structure.

Usage:
    python scripts/config_migration.py <input_yaml> [--output-dir <dir>] [--environments <env1,env2,...>]

Example:
    python scripts/config_migration.py config/old_trading_config.yaml --output-dir src/trading_bot/config/environments
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml


class ConfigMigrator:
    """Migrates legacy configuration to new environment-based structure."""

    def __init__(self, input_path: Path, output_dir: Path):
        """Initialize migrator.

        Args:
            input_path: Path to legacy trading_config.yaml
            output_dir: Directory to write new environment config files
        """
        self.input_path = input_path
        self.output_dir = output_dir
        self.legacy_config: Dict[str, Any] = {}

    def load_legacy_config(self) -> None:
        """Load legacy configuration file."""
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")

        with open(self.input_path, "r", encoding="utf-8") as f:
            self.legacy_config = yaml.safe_load(f) or {}

        print(f"✓ Loaded legacy config from {self.input_path}")

    def migrate_to_new_structure(self) -> Dict[str, Any]:
        """Convert legacy config structure to new environment-based structure.

        Returns:
            New configuration dictionary
        """
        new_config: Dict[str, Any] = {}

        # Migrate data section
        if "data" in self.legacy_config:
            new_config["data"] = self._migrate_data_section(self.legacy_config["data"])

        # Migrate risk section
        if "risk" in self.legacy_config:
            new_config["risk"] = self._migrate_risk_section(self.legacy_config["risk"])

        # Migrate strategy section
        if "strategy" in self.legacy_config:
            new_config["strategy"] = self._migrate_strategy_section(
                self.legacy_config["strategy"]
            )

        # Migrate execution section
        if "execution" in self.legacy_config:
            new_config["execution"] = self._migrate_execution_section(
                self.legacy_config["execution"]
            )

        # Migrate trading section (from top-level fields)
        new_config["trading"] = self._migrate_trading_section(self.legacy_config)

        # Migrate logging section
        if "logging" in self.legacy_config:
            new_config["logging"] = self._migrate_logging_section(
                self.legacy_config["logging"]
            )

        # Migrate notifications to discord section
        if "notifications" in self.legacy_config:
            new_config["discord"] = self._migrate_discord_section(
                self.legacy_config["notifications"]
            )

        # Add binance section with environment variable placeholders
        new_config["binance"] = {
            "api_key": "${TRADING_BINANCE_API_KEY}",
            "api_secret": "${TRADING_BINANCE_API_SECRET}",
        }

        return new_config

    def _migrate_data_section(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate data configuration section."""
        return {
            "cache_size": data.get("cache_size", 1000),
            "cache_ttl": data.get("cache_ttl", 300),
            "historical_lookback_days": data.get("historical_lookback_days", 60),
        }

    def _migrate_risk_section(self, risk: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate risk management configuration section."""
        # Map old field names to new field names
        return {
            "initial_capital": risk.get("max_position_size", 10000.0),
            "max_position_size_pct": risk.get("risk_per_trade", 0.1),
            "max_drawdown_pct": risk.get("max_drawdown", 0.2),
            "max_consecutive_losses": 3,  # Default value
            "volatility_threshold": 0.05,  # Default value
        }

    def _migrate_strategy_section(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate strategy configuration section."""
        return {
            "enabled_strategies": ["ict", "traditional"],  # Default
            "default_strategy": "ict",
            "strategy_selection_mode": "auto",
        }

    def _migrate_execution_section(self, execution: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate execution configuration section."""
        return {
            "slippage_tolerance_pct": execution.get("slippage_tolerance", 0.001),
            "max_retry_attempts": execution.get("retry_attempts", 3),
            "order_timeout_seconds": execution.get("timeout", 30),
        }

    def _migrate_trading_section(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate trading configuration from top-level fields."""
        # Handle symbols - take first if multiple, otherwise default
        symbols = config.get("symbols", ["BTCUSDT"])
        symbol = symbols[0] if isinstance(symbols, list) and symbols else "BTCUSDT"

        return {
            "symbol": symbol,
            "timeframe": "1h",  # Default
            "max_position_size": 0.1,
            "leverage": config.get("leverage", 1),
            "max_risk_per_trade": 0.02,
            "daily_loss_limit": 0.05,
        }

    def _migrate_logging_section(self, logging: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate logging configuration section."""
        return {
            "level": logging.get("level", "INFO"),
            "format": "json",  # New default
            "output_dir": "logs",
            "max_bytes": logging.get("max_file_size", 10485760),
            "backup_count": logging.get("backup_count", 5),
            "rotation_time": "midnight",
            "trade_log_enabled": True,
            "trade_log_file": "trades.log",
            "performance_log_enabled": True,
            "performance_log_file": "performance.log",
            "mask_sensitive_data": True,
            "debug_unmask": False,
        }

    def _migrate_discord_section(
        self, notifications: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Migrate notifications to discord configuration section."""
        discord_config = notifications.get("discord", {})

        return {
            "bot_token": discord_config.get(
                "bot_token", "${TRADING_DISCORD_BOT_TOKEN}"
            ),
            "channel_id": discord_config.get(
                "channel_id", "${TRADING_DISCORD_CHANNEL_ID}"
            ),
            "enable_notifications": notifications.get("enabled", True),
            "spam_protection_seconds": 60,  # Default
        }

    def write_environment_config(
        self, env_name: str, config: Dict[str, Any], overrides: Dict[str, Any] = None
    ) -> None:
        """Write environment-specific configuration file.

        Args:
            env_name: Environment name (e.g., 'development', 'production')
            config: Base configuration dictionary
            overrides: Environment-specific overrides
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"{env_name}.yml"

        # Merge overrides if provided
        final_config = config.copy()
        if overrides:
            final_config = self._deep_merge(final_config, overrides)

        # Add environment comment header
        output_lines = [f"# {env_name.capitalize()} Environment Configuration\n"]
        output_lines.append(
            f"# Migrated from legacy trading_config.yaml\n"
        )
        output_lines.append("\n")

        # Write YAML content
        yaml_content = yaml.dump(
            final_config, default_flow_style=False, sort_keys=False, allow_unicode=True
        )
        output_lines.append(yaml_content)

        with open(output_path, "w", encoding="utf-8") as f:
            f.writelines(output_lines)

        print(f"✓ Created {env_name} config: {output_path}")

    def _deep_merge(
        self, base: Dict[str, Any], override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Deep merge two dictionaries.

        Args:
            base: Base dictionary
            override: Dictionary with override values

        Returns:
            Merged dictionary
        """
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def generate_environment_overrides(self) -> Dict[str, Dict[str, Any]]:
        """Generate environment-specific configuration overrides.

        Returns:
            Dictionary mapping environment names to their override configs
        """
        return {
            "development": {
                "binance": {"network": "testnet"},
                "logging": {"level": "DEBUG"},
                "trading": {"leverage": 1},
            },
            "paper": {
                "binance": {"network": "testnet", "execution_mode": "paper"},
                "logging": {"level": "INFO"},
                "trading": {"leverage": 1},
            },
            "paper-testnet": {
                "binance": {"network": "testnet", "execution_mode": "paper"},
                "logging": {"level": "INFO"},
                "trading": {"leverage": 1},
            },
            "paper-mainnet": {
                "binance": {"network": "mainnet", "execution_mode": "paper"},
                "logging": {"level": "INFO"},
                "trading": {"leverage": 2},
            },
            "production": {
                "binance": {"network": "mainnet", "execution_mode": "live"},
                "logging": {"level": "INFO", "mask_sensitive_data": True},
                "trading": {"leverage": self.legacy_config.get("leverage", 3)},
            },
        }

    def migrate(self, environments: List[str] = None) -> None:
        """Execute full migration process.

        Args:
            environments: List of environment names to generate. If None, generates all.
        """
        print("\n=== Configuration Migration ===\n")

        # Load legacy config
        self.load_legacy_config()

        # Migrate to new structure
        print("\n→ Converting configuration structure...")
        new_config = self.migrate_to_new_structure()

        # Generate environment-specific configs
        print("\n→ Generating environment configurations...\n")
        overrides = self.generate_environment_overrides()

        # Filter environments if specified
        if environments:
            overrides = {k: v for k, v in overrides.items() if k in environments}

        for env_name, env_overrides in overrides.items():
            self.write_environment_config(env_name, new_config, env_overrides)

        # Print summary
        print("\n=== Migration Complete ===\n")
        print(f"Migrated: {self.input_path}")
        print(f"Output directory: {self.output_dir}")
        print(f"Environments created: {', '.join(overrides.keys())}")
        print("\nNext steps:")
        print("1. Review generated configuration files")
        print("2. Run validation: python scripts/validate_config.py")
        print("3. Update .env file with required environment variables")
        print("4. Test with: TRADING_ENV=development python -m trading_bot.main")


def main():
    """Main entry point for migration script."""
    parser = argparse.ArgumentParser(
        description="Migrate legacy trading_config.yaml to new environment-based structure"
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to legacy trading_config.yaml file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/trading_bot/config/environments"),
        help="Output directory for environment configs (default: src/trading_bot/config/environments)",
    )
    parser.add_argument(
        "--environments",
        type=str,
        help="Comma-separated list of environments to generate (default: all)",
    )

    args = parser.parse_args()

    # Parse environments list
    environments = None
    if args.environments:
        environments = [e.strip() for e in args.environments.split(",")]

    try:
        migrator = ConfigMigrator(args.input_file, args.output_dir)
        migrator.migrate(environments)
    except Exception as e:
        print(f"\n✗ Migration failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
