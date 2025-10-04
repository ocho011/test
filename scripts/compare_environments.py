#!/usr/bin/env python3
"""Environment Configuration Comparison Tool

Compares configuration settings across different environments.

Usage:
    python scripts/compare_environments.py [--environments <env1,env2,...>] [--config-dir <dir>]

Example:
    python scripts/compare_environments.py --environments development,production
    python scripts/compare_environments.py  # Compare all environments
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import yaml


class EnvironmentComparator:
    """Compares configuration across multiple environments."""

    def __init__(self, config_dir: Path):
        """Initialize comparator.

        Args:
            config_dir: Directory containing environment configuration files
        """
        self.config_dir = config_dir
        self.environments: Dict[str, Dict[str, Any]] = {}
        self.base_config: Dict[str, Any] = {}

    def load_base_config(self) -> None:
        """Load base configuration if it exists."""
        base_file = self.config_dir / "base.yml"

        if base_file.exists():
            with open(base_file, "r", encoding="utf-8") as f:
                self.base_config = yaml.safe_load(f) or {}
            print(f"✓ Loaded base configuration from {base_file}")

    def load_environment_configs(self, environment_names: List[str] = None) -> None:
        """Load environment configuration files.

        Args:
            environment_names: Specific environments to load. If None, loads all.
        """
        # Find environment files
        yaml_files = list(self.config_dir.glob("*.yml")) + list(
            self.config_dir.glob("*.yaml")
        )

        # Filter out base.yml
        env_files = [f for f in yaml_files if f.stem != "base"]

        # Filter by specific environments if requested
        if environment_names:
            env_files = [f for f in env_files if f.stem in environment_names]

        if not env_files:
            raise FileNotFoundError("No environment configuration files found")

        # Load each environment
        for file_path in sorted(env_files):
            env_name = file_path.stem
            with open(file_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}

            # Merge with base config
            merged_config = self._deep_merge(self.base_config.copy(), config)
            self.environments[env_name] = merged_config

        print(f"✓ Loaded {len(self.environments)} environment(s): {', '.join(sorted(self.environments.keys()))}")

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

    def get_all_config_paths(self) -> Set[Tuple[str, ...]]:
        """Get all configuration paths across all environments.

        Returns:
            Set of configuration path tuples
        """
        all_paths = set()

        for config in self.environments.values():
            all_paths.update(self._extract_paths(config))

        return all_paths

    def _extract_paths(
        self, config: Dict[str, Any], prefix: Tuple[str, ...] = ()
    ) -> Set[Tuple[str, ...]]:
        """Extract all paths from a configuration dictionary.

        Args:
            config: Configuration dictionary
            prefix: Current path prefix

        Returns:
            Set of path tuples
        """
        paths = set()

        for key, value in config.items():
            current_path = prefix + (key,)
            paths.add(current_path)

            if isinstance(value, dict):
                paths.update(self._extract_paths(value, current_path))

        return paths

    def get_value_at_path(
        self, config: Dict[str, Any], path: Tuple[str, ...]
    ) -> Any:
        """Get value at a specific configuration path.

        Args:
            config: Configuration dictionary
            path: Path tuple

        Returns:
            Value at path, or None if path doesn't exist
        """
        current = config

        for key in path:
            if not isinstance(current, dict) or key not in current:
                return None
            current = current[key]

        return current

    def compare_setting(self, path: Tuple[str, ...]) -> Dict[str, Any]:
        """Compare a specific setting across all environments.

        Args:
            path: Configuration path

        Returns:
            Comparison result dictionary
        """
        values = {}
        for env_name, config in self.environments.items():
            value = self.get_value_at_path(config, path)
            values[env_name] = value

        # Determine if values are different
        unique_values = set(
            str(v) if not isinstance(v, dict) else str(sorted(v.items()))
            for v in values.values()
            if v is not None
        )

        return {
            "path": ".".join(path),
            "values": values,
            "different": len(unique_values) > 1,
            "missing_in": [env for env, val in values.items() if val is None],
        }

    def compare_all(self) -> List[Dict[str, Any]]:
        """Compare all settings across environments.

        Returns:
            List of comparison results
        """
        all_paths = self.get_all_config_paths()
        comparisons = []

        for path in sorted(all_paths):
            # Skip nested dict comparisons (we only care about leaf values)
            is_leaf = True
            for env_config in self.environments.values():
                value = self.get_value_at_path(env_config, path)
                if isinstance(value, dict):
                    is_leaf = False
                    break

            if is_leaf:
                comparison = self.compare_setting(path)
                comparisons.append(comparison)

        return comparisons

    def print_differences(self, comparisons: List[Dict[str, Any]]) -> None:
        """Print configuration differences in readable format.

        Args:
            comparisons: List of comparison results
        """
        print("\n=== Configuration Differences ===\n")

        # Filter to only different settings
        differences = [c for c in comparisons if c["different"]]

        if not differences:
            print("✓ All environments have identical configurations")
            return

        # Group by top-level section
        sections: Dict[str, List[Dict[str, Any]]] = {}
        for diff in differences:
            section = diff["path"].split(".")[0]
            if section not in sections:
                sections[section] = []
            sections[section].append(diff)

        # Print differences by section
        for section_name in sorted(sections.keys()):
            print(f"\n[{section_name.upper()}]")
            print("-" * 80)

            for diff in sections[section_name]:
                print(f"\n  {diff['path']}:")
                for env_name in sorted(self.environments.keys()):
                    value = diff["values"].get(env_name)
                    if value is None:
                        print(f"    {env_name:15} = <not set>")
                    else:
                        print(f"    {env_name:15} = {value}")

        # Print summary
        print("\n" + "=" * 80)
        print(f"\nTotal differences: {len(differences)}")
        print(f"Sections affected: {len(sections)}")

    def print_summary(self, comparisons: List[Dict[str, Any]]) -> None:
        """Print comparison summary.

        Args:
            comparisons: List of comparison results
        """
        print("\n=== Comparison Summary ===\n")

        total_settings = len(comparisons)
        different_settings = sum(1 for c in comparisons if c["different"])
        identical_settings = total_settings - different_settings

        print(f"Environments compared: {', '.join(sorted(self.environments.keys()))}")
        print(f"Total settings: {total_settings}")
        print(f"Identical across all: {identical_settings}")
        print(f"Different: {different_settings}")

        # Settings missing in some environments
        missing_settings = [
            c for c in comparisons if c["missing_in"] and len(c["missing_in"]) > 0
        ]
        if missing_settings:
            print(f"\nSettings missing in some environments: {len(missing_settings)}")
            for setting in missing_settings[:5]:  # Show first 5
                print(f"  • {setting['path']}: missing in {', '.join(setting['missing_in'])}")
            if len(missing_settings) > 5:
                print(f"  ... and {len(missing_settings) - 5} more")

    def export_comparison(
        self, comparisons: List[Dict[str, Any]], output_path: Path
    ) -> None:
        """Export comparison results to file.

        Args:
            comparisons: List of comparison results
            output_path: Path to write comparison results
        """
        import json

        # Prepare export data
        export_data = {
            "environments": list(self.environments.keys()),
            "total_settings": len(comparisons),
            "differences": [c for c in comparisons if c["different"]],
            "all_comparisons": comparisons,
        }

        # Write to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, default=str)

        print(f"\n→ Comparison results exported to: {output_path}")

    def compare(self, show_all: bool = False) -> None:
        """Execute comparison and display results.

        Args:
            show_all: If True, show all settings. If False, only show differences.
        """
        print("\n=== Environment Configuration Comparison ===\n")

        # Load configurations
        self.load_base_config()

        # Compare all settings
        comparisons = self.compare_all()

        # Print results
        if show_all:
            print("\n=== All Settings ===\n")
            for comp in comparisons:
                print(f"{comp['path']}:")
                for env_name in sorted(self.environments.keys()):
                    value = comp["values"].get(env_name)
                    print(f"  {env_name:15} = {value}")
                print()
        else:
            self.print_differences(comparisons)

        self.print_summary(comparisons)


def main():
    """Main entry point for comparison script."""
    parser = argparse.ArgumentParser(
        description="Compare configuration settings across environments"
    )
    parser.add_argument(
        "--environments",
        "-e",
        type=str,
        help="Comma-separated list of environments to compare (default: all)",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("src/trading_bot/config/environments"),
        help="Configuration directory (default: src/trading_bot/config/environments)",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Show all settings, not just differences",
    )
    parser.add_argument(
        "--export",
        type=Path,
        help="Export comparison results to JSON file",
    )

    args = parser.parse_args()

    # Parse environments list
    environment_names = None
    if args.environments:
        environment_names = [e.strip() for e in args.environments.split(",")]

    try:
        comparator = EnvironmentComparator(args.config_dir)
        comparator.load_environment_configs(environment_names)
        comparator.compare(show_all=args.show_all)

        if args.export:
            comparisons = comparator.compare_all()
            comparator.export_comparison(comparisons, args.export)

    except Exception as e:
        print(f"\n✗ Comparison failed: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
