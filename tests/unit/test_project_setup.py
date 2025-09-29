"""
Basic setup verification tests for ICT Trading Bot
"""

from pathlib import Path

import pytest


class TestProjectStructure:
    """Test project structure and basic imports"""

    def test_src_directory_exists(self):
        """Test that src directory exists"""
        src_path = Path("src")
        assert src_path.exists(), "src directory should exist"
        assert src_path.is_dir(), "src should be a directory"

    def test_trading_bot_package_exists(self):
        """Test that trading_bot package exists"""
        package_path = Path("src/trading_bot")
        assert package_path.exists(), "trading_bot package should exist"
        assert package_path.is_dir(), "trading_bot should be a directory"

        init_file = package_path / "__init__.py"
        assert init_file.exists(), "trading_bot should have __init__.py"

    def test_all_modules_exist(self):
        """Test that all required modules exist"""
        base_path = Path("src/trading_bot")
        required_modules = [
            "core",
            "strategies",
            "analysis",
            "risk",
            "execution",
            "notifications",
            "config",
            "utils",
        ]

        for module in required_modules:
            module_path = base_path / module
            assert module_path.exists(), f"{module} module should exist"
            assert module_path.is_dir(), f"{module} should be a directory"

            init_file = module_path / "__init__.py"
            assert init_file.exists(), f"{module} should have __init__.py"


class TestDependencyImports:
    """Test that all dependencies can be imported"""

    def test_core_dependencies(self):
        """Test core dependencies import correctly"""
        try:
            import asyncio  # noqa: F401

            import aiohttp  # noqa: F401
            import numpy  # noqa: F401
            import pandas  # noqa: F401
            import pydantic  # noqa: F401
            import yaml  # noqa: F401
            from dotenv import load_dotenv  # noqa: F401
        except ImportError as e:
            pytest.fail(f"Failed to import core dependency: {e}")

    def test_trading_dependencies(self):
        """Test trading-specific dependencies"""
        try:
            import binance  # noqa: F401
            import plotly  # noqa: F401
            import ta  # noqa: F401
        except ImportError as e:
            pytest.fail(f"Failed to import trading dependency: {e}")

    def test_notification_dependencies(self):
        """Test notification dependencies"""
        try:
            import discord  # noqa: F401
        except ImportError as e:
            pytest.fail(f"Failed to import notification dependency: {e}")

    def test_web_dependencies(self):
        """Test web interface dependencies"""
        try:
            import fastapi  # noqa: F401
            import uvicorn  # noqa: F401
        except ImportError as e:
            pytest.fail(f"Failed to import web dependency: {e}")


class TestEnvironmentConfiguration:
    """Test environment configuration"""

    def test_env_example_exists(self):
        """Test that .env.example exists"""
        env_example = Path(".env.example")
        assert env_example.exists(), ".env.example should exist"
        assert env_example.is_file(), ".env.example should be a file"

    def test_env_example_has_required_vars(self):
        """Test that .env.example contains required variables"""
        env_example = Path(".env.example")
        content = env_example.read_text()

        required_vars = [
            "BINANCE_API_KEY",
            "BINANCE_SECRET_KEY",
            "BINANCE_TESTNET",
            "DEFAULT_SYMBOL",
            "MAX_RISK_PER_TRADE",
            "LOG_LEVEL",
            "DATABASE_URL",
        ]

        for var in required_vars:
            assert var in content, f"Required variable {var} should be in .env.example"

    def test_dotenv_loading(self):
        """Test that dotenv can load environment variables"""
        from dotenv import load_dotenv

        # This should not raise an exception
        load_dotenv(".env.example")


class TestProjectConfiguration:
    """Test project configuration files"""

    def test_pyproject_toml_exists(self):
        """Test that pyproject.toml exists and is valid"""
        pyproject_path = Path("pyproject.toml")
        assert pyproject_path.exists(), "pyproject.toml should exist"
        assert pyproject_path.is_file(), "pyproject.toml should be a file"

    def test_pyproject_toml_has_project_info(self):
        """Test that pyproject.toml contains project information"""
        try:
            import tomllib

            pyproject_path = Path("pyproject.toml")
            with open(pyproject_path, "rb") as f:
                config = tomllib.load(f)
        except ImportError:
            # Fallback for Python < 3.11
            import toml

            pyproject_path = Path("pyproject.toml")
            with open(pyproject_path, "r") as f:
                config = toml.load(f)

        assert "project" in config, "pyproject.toml should have [project] section"
        assert "name" in config["project"], "Project should have name"
        assert "version" in config["project"], "Project should have version"
        assert "dependencies" in config["project"], "Project should have dependencies"

    def test_readme_exists(self):
        """Test that README.md exists"""
        readme_path = Path("README.md")
        assert readme_path.exists(), "README.md should exist"
        assert readme_path.is_file(), "README.md should be a file"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
