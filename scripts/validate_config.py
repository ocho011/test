#!/usr/bin/env python3
"""Configuration Validation Tool

Validates environment configuration files against Pydantic models.

Usage:
    python scripts/validate_config.py [--environment <env>] [--config-dir <dir>]

Example:
    python scripts/validate_config.py --environment development
    python scripts/validate_config.py --config-dir src/trading_bot/config/environments
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any

import yaml
from pydantic import ValidationError

# Add project root to path to allow imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from trading_bot.config.models import TradingBotConfig


class ConfigValidator:
    """Validates configuration files against Pydantic models."""

    def __init__(self, config_dir: Path):
        """Initialize validator.

        Args:
            config_dir: Directory containing environment configuration files
        """
        self.config_dir = config_dir
        self.validation_results: Dict[str, Dict[str, Any]] = {}

    def load_yaml_file(self, file_path: Path) -> Dict[str, Any]:
        """Load YAML configuration file.

        Args:
            file_path: Path to YAML file

        Returns:
            Loaded configuration dictionary
        """
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def find_environment_files(self) -> List[Path]:
        """Find all environment configuration files.

        Returns:
            List of paths to environment .yml files
        """
        if not self.config_dir.exists():
            raise FileNotFoundError(f"Config directory not found: {self.config_dir}")

        yaml_files = list(self.config_dir.glob("*.yml")) + list(
            self.config_dir.glob("*.yaml")
        )

        # Exclude base.yml as it's not a complete environment config
        return [f for f in yaml_files if f.stem != "base"]

    def validate_config_file(self, file_path: Path) -> Dict[str, Any]:
        """Validate a single configuration file.

        Args:
            file_path: Path to configuration file

        Returns:
            Validation result dictionary with status, errors, and warnings
        """
        result = {
            "file": str(file_path),
            "environment": file_path.stem,
            "valid": False,
            "errors": [],
            "warnings": [],
        }

        try:
            # Load configuration
            config_data = self.load_yaml_file(file_path)

            # Check for base.yml and merge if exists
            base_file = self.config_dir / "base.yml"
            if base_file.exists():
                base_data = self.load_yaml_file(base_file)
                config_data = self._deep_merge(base_data, config_data)

            # Validate against Pydantic model
            validated_config = TradingBotConfig(**config_data)

            # If we get here, validation passed
            result["valid"] = True
            result["config"] = validated_config.model_dump()

            # Check for warnings
            result["warnings"] = self._check_warnings(config_data, validated_config)

        except ValidationError as e:
            result["errors"] = self._format_validation_errors(e)
        except yaml.YAMLError as e:
            result["errors"] = [f"YAML parsing error: {str(e)}"]
        except Exception as e:
            result["errors"] = [f"Unexpected error: {str(e)}"]

        return result

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

    def _format_validation_errors(self, error: ValidationError) -> List[str]:
        """Format Pydantic validation errors into readable messages.

        Args:
            error: Pydantic ValidationError

        Returns:
            List of formatted error messages
        """
        formatted_errors = []

        for err in error.errors():
            location = " → ".join(str(loc) for loc in err["loc"])
            msg = err["msg"]
            error_type = err["type"]

            formatted_errors.append(f"{location}: {msg} (type: {error_type})")

        return formatted_errors

    def _check_warnings(
        self, config_data: Dict[str, Any], validated_config: TradingBotConfig
    ) -> List[str]:
        """Check for configuration warnings (non-critical issues).

        Args:
            config_data: Raw configuration data
            validated_config: Validated configuration model

        Returns:
            List of warning messages
        """
        warnings = []

        # Check for environment variables that need to be set
        env_vars = self._find_env_var_references(config_data)
        if env_vars:
            warnings.append(
                f"Environment variables required: {', '.join(sorted(env_vars))}"
            )

        # Check for production-specific warnings (mainnet = not testnet)
        if not validated_config.binance.testnet:
            warnings.append(
                "⚠️  MAINNET MODE: Trading on mainnet - ensure proper risk management"
            )

        # Check leverage settings
        if validated_config.trading.leverage > 3:
            warnings.append(
                f"High leverage detected ({validated_config.trading.leverage}x) - increased risk"
            )

        # Check logging level for production (mainnet = not testnet)
        if (
            not validated_config.binance.testnet
            and validated_config.logging.level == "DEBUG"
        ):
            warnings.append(
                "DEBUG logging in production may impact performance and expose sensitive data"
            )

        return warnings

    def _find_env_var_references(
        self, config: Dict[str, Any], prefix: str = ""
    ) -> set:
        """Find all environment variable references in configuration.

        Args:
            config: Configuration dictionary
            prefix: Current key prefix for nested structures

        Returns:
            Set of environment variable names
        """
        env_vars = set()

        for key, value in config.items():
            if isinstance(value, dict):
                env_vars.update(
                    self._find_env_var_references(value, f"{prefix}{key}.")
                )
            elif isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_var_name = value[2:-1]  # Remove ${ and }
                env_vars.add(env_var_name)

        return env_vars

    def validate_all(self, environment: str = None) -> bool:
        """Validate all configuration files or a specific environment.

        Args:
            environment: Optional specific environment to validate

        Returns:
            True if all validations passed, False otherwise
        """
        print("\n=== Configuration Validation ===\n")

        # Find files to validate
        all_files = self.find_environment_files()

        if environment:
            all_files = [f for f in all_files if f.stem == environment]
            if not all_files:
                print(f"✗ Environment '{environment}' not found")
                return False

        if not all_files:
            print("✗ No configuration files found")
            return False

        # Validate each file
        all_valid = True

        for file_path in sorted(all_files):
            result = self.validate_config_file(file_path)
            self.validation_results[result["environment"]] = result

            # Print result
            status = "✓" if result["valid"] else "✗"
            env_name = result["environment"]

            print(f"{status} {env_name.ljust(20)}", end="")

            if result["valid"]:
                print("VALID")
                if result["warnings"]:
                    for warning in result["warnings"]:
                        print(f"  ⚠️  {warning}")
            else:
                print("INVALID")
                all_valid = False
                for error in result["errors"]:
                    print(f"  ✗ {error}")

            print()

        # Print summary
        print("=== Validation Summary ===\n")

        total = len(self.validation_results)
        valid = sum(1 for r in self.validation_results.values() if r["valid"])
        invalid = total - valid

        print(f"Total environments: {total}")
        print(f"Valid: {valid}")
        print(f"Invalid: {invalid}")

        if all_valid:
            print("\n✓ All configurations are valid!")
        else:
            print("\n✗ Some configurations have errors - please fix before deploying")

        return all_valid

    def export_results(self, output_path: Path) -> None:
        """Export validation results to JSON file.

        Args:
            output_path: Path to write JSON results
        """
        import json

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.validation_results, f, indent=2, default=str)

        print(f"\n→ Validation results exported to: {output_path}")


def main():
    """Main entry point for validation script."""
    parser = argparse.ArgumentParser(
        description="Validate environment configuration files"
    )
    parser.add_argument(
        "--environment",
        "-e",
        type=str,
        help="Specific environment to validate (default: all)",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("src/trading_bot/config/environments"),
        help="Configuration directory (default: src/trading_bot/config/environments)",
    )
    parser.add_argument(
        "--export",
        type=Path,
        help="Export validation results to JSON file",
    )

    args = parser.parse_args()

    try:
        validator = ConfigValidator(args.config_dir)
        success = validator.validate_all(args.environment)

        if args.export:
            validator.export_results(args.export)

        sys.exit(0 if success else 1)

    except Exception as e:
        print(f"\n✗ Validation failed: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
