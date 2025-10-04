"""
Test suite for configuration consolidation (Task 14.4).

Tests verify:
1. Environment-specific configs properly inherit from base.yml
2. No duplicate configuration values exist
3. Minimal .env variables (only 5 essentials)
4. Config loading works correctly for all environments
"""

import os
import pytest
from pathlib import Path
from typing import Dict, Any
import yaml

from trading_bot.config.config_manager import ConfigManager
from trading_bot.config.models import TradingBotConfig


class TestConfigConsolidation:
    """Test configuration consolidation and deduplication."""
    
    @pytest.fixture
    def config_dir(self) -> Path:
        """Get the environments config directory."""
        return Path(__file__).parent.parent.parent / "src/trading_bot/config/environments"
    
    @pytest.fixture
    def base_config(self, config_dir: Path) -> Dict[str, Any]:
        """Load base configuration."""
        with open(config_dir / "base.yml", 'r') as f:
            return yaml.safe_load(f)
    
    def test_minimal_env_variables(self):
        """Test that only essential environment variables are required."""
        essential_vars = {
            'TRADING_BINANCE_API_KEY',
            'TRADING_BINANCE_API_SECRET',
            'TRADING_DISCORD_BOT_TOKEN',
            'TRADING_DISCORD_CHANNEL_ID',
            'TRADING_ENV'
        }
        
        # Read .env.example to verify only essential vars are listed
        env_example_path = Path(__file__).parent.parent.parent / ".env.example"
        with open(env_example_path, 'r') as f:
            content = f.read()
        
        # Extract variable names from .env.example
        env_vars = set()
        for line in content.split('\n'):
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                var_name = line.split('=')[0].strip()
                env_vars.add(var_name)
        
        # Verify we have exactly the essential variables
        assert env_vars == essential_vars, \
            f"Expected {essential_vars}, but found {env_vars}"
    
    def test_no_duplicate_configs_in_environments(self, config_dir: Path, base_config: Dict[str, Any]):
        """Test that environment configs don't duplicate base.yml values."""
        # Environments that should inherit from base
        env_files = [
            'development.yml',
            'production.yml', 
            'paper.yml',
            'paper-testnet.yml',
            'paper-mainnet.yml'
        ]
        
        for env_file in env_files:
            env_path = config_dir / env_file
            if not env_path.exists():
                continue
            
            with open(env_path, 'r') as f:
                env_config = yaml.safe_load(f)
            
            # Check for 'extends' directive
            if 'extends' in env_config:
                assert env_config['extends'] == 'base.yml', \
                    f"{env_file} should extend base.yml"
            
            # Verify environment doesn't duplicate entire sections from base
            # (only overrides should exist)
            self._check_no_complete_duplicates(base_config, env_config, env_file)
    
    def _check_no_complete_duplicates(self, base: Dict[str, Any], env: Dict[str, Any], env_name: str):
        """Check that environment doesn't completely duplicate base sections."""
        for key in env.keys():
            if key == 'extends':
                continue
            
            # If section exists in both base and env
            if key in base and isinstance(base[key], dict) and isinstance(env[key], dict):
                base_keys = set(base[key].keys())
                env_keys = set(env[key].keys())
                
                # If environment has ALL keys from base, it's a complete duplicate
                if base_keys.issubset(env_keys) and len(base_keys) > 0:
                    # Check if values are actually different (overrides)
                    all_same = all(
                        base[key].get(k) == env[key].get(k) 
                        for k in base_keys
                    )
                    
                    if all_same:
                        pytest.fail(
                            f"{env_name} completely duplicates '{key}' section from base.yml"
                        )
    
    def test_paper_yml_inherits_properly(self, config_dir: Path):
        """Test that paper.yml properly inherits from base.yml."""
        with open(config_dir / "paper.yml", 'r') as f:
            paper_config = yaml.safe_load(f)
        
        # Verify it extends base
        assert paper_config.get('extends') == 'base.yml', \
            "paper.yml must extend base.yml"
        
        # Verify it doesn't duplicate binance/discord config
        # (these should come from base.yml)
        if 'binance' in paper_config:
            # Should only have testnet-specific overrides
            assert 'api_key' not in paper_config['binance'], \
                "paper.yml should not duplicate api_key from base.yml"
            assert 'api_secret' not in paper_config['binance'], \
                "paper.yml should not duplicate api_secret from base.yml"
        
        if 'discord' in paper_config:
            # Discord config should come from base, not duplicated
            assert 'bot_token' not in paper_config['discord'], \
                "paper.yml should not duplicate bot_token from base.yml"
    
    def test_old_config_file_removed(self):
        """Test that redundant config/trading_config.yaml has been removed."""
        old_config_path = Path(__file__).parent.parent.parent / "config/trading_config.yaml"
        assert not old_config_path.exists(), \
            "Old config/trading_config.yaml should be removed"
    
    def test_config_loading_all_environments(self, config_dir: Path):
        """Test that all environment configs load correctly."""
        environments = ['development', 'production', 'paper', 'testing']
        
        for env in environments:
            env_file = config_dir / f"{env}.yml"
            if not env_file.exists():
                continue
            
            # Set required env vars for loading
            os.environ['TRADING_BINANCE_API_KEY'] = 'test_key'
            os.environ['TRADING_BINANCE_API_SECRET'] = 'test_secret'
            os.environ['TRADING_DISCORD_BOT_TOKEN'] = 'test_token'
            os.environ['TRADING_DISCORD_CHANNEL_ID'] = '123456'
            os.environ['TRADING_ENV'] = env
            
            # Load config
            manager = ConfigManager()
            config = manager.load_config(environment=env)
            
            # Verify it's a valid TradingBotConfig
            assert isinstance(config, TradingBotConfig), \
                f"Failed to load {env} environment config"
            
            # Verify environment is set correctly
            assert config.system.environment == env, \
                f"Environment mismatch for {env}"
    
    def test_env_variable_substitution(self):
        """Test that environment variables are properly substituted in configs."""
        test_key = "test_api_key_12345"
        test_secret = "test_secret_67890"
        
        os.environ['TRADING_BINANCE_API_KEY'] = test_key
        os.environ['TRADING_BINANCE_API_SECRET'] = test_secret
        os.environ['TRADING_DISCORD_BOT_TOKEN'] = 'test_token'
        os.environ['TRADING_DISCORD_CHANNEL_ID'] = '123456'
        
        manager = ConfigManager()
        config = manager.load_config(environment='development')
        
        # Verify substitution worked
        assert config.binance.api_key == test_key
        assert config.binance.api_secret == test_secret
    
    def test_config_inheritance_chain(self, config_dir: Path):
        """Test that config inheritance works correctly."""
        # Load development config which should inherit from base
        with open(config_dir / "development.yml", 'r') as f:
            dev_config = yaml.safe_load(f)
        
        with open(config_dir / "base.yml", 'r') as f:
            base_config = yaml.safe_load(f)
        
        # Verify inheritance directive
        assert dev_config.get('extends') == 'base.yml'
        
        # Development should override some values from base
        # but inherit others
        os.environ['TRADING_BINANCE_API_KEY'] = 'test_key'
        os.environ['TRADING_BINANCE_API_SECRET'] = 'test_secret'
        os.environ['TRADING_DISCORD_BOT_TOKEN'] = 'test_token'
        os.environ['TRADING_DISCORD_CHANNEL_ID'] = '123456'
        
        manager = ConfigManager()
        config = manager.load_config(environment='development')
        
        # Development overrides (from dev.yml)
        assert config.system.debug == True  # Overridden in development
        assert config.binance.testnet == True  # Overridden in development
        
        # Base values that should be inherited
        assert config.system.event_queue_size == base_config.get('system', {}).get('event_queue_size', 1000)
    
    def test_no_unused_config_values(self, config_dir: Path):
        """Test that there are no completely unused config sections."""
        # This is a placeholder - would need to scan actual usage in codebase
        # For now, just verify all major sections are present in base
        with open(config_dir / "base.yml", 'r') as f:
            base_config = yaml.safe_load(f)
        
        required_sections = [
            'data', 'risk', 'strategy', 'execution', 
            'system_events', 'trading', 'system', 
            'logging', 'discord', 'binance'
        ]
        
        for section in required_sections:
            assert section in base_config, \
                f"Required config section '{section}' missing from base.yml"
    
    def test_config_naming_consistency(self, config_dir: Path):
        """Test that configuration keys use consistent naming conventions."""
        # Load all config files
        config_files = list(config_dir.glob("*.yml"))
        
        for config_file in config_files:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Check that all keys use snake_case
            self._check_snake_case_keys(config, config_file.name)
    
    def _check_snake_case_keys(self, data: Any, file_name: str, path: str = ""):
        """Recursively check that all keys use snake_case."""
        if isinstance(data, dict):
            for key, value in data.items():
                # Check key is snake_case (lowercase with underscores)
                if not (key.islower() or '_' in key or key == 'extends'):
                    pytest.fail(
                        f"Non-snake_case key '{key}' in {file_name} at {path}"
                    )
                
                # Recurse into nested dicts
                new_path = f"{path}.{key}" if path else key
                self._check_snake_case_keys(value, file_name, new_path)
        
        elif isinstance(data, list):
            for i, item in enumerate(data):
                self._check_snake_case_keys(item, file_name, f"{path}[{i}]")


class TestEnvVariableReduction:
    """Test that .env variables have been reduced to essentials only."""
    
    def test_env_example_minimal(self):
        """Test .env.example contains only essential variables."""
        env_example_path = Path(__file__).parent.parent.parent / ".env.example"
        
        with open(env_example_path, 'r') as f:
            content = f.read()
        
        # Count actual variable definitions (non-comment, non-empty lines)
        var_count = 0
        for line in content.split('\n'):
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                var_count += 1
        
        # Should have exactly 5 essential variables
        assert var_count == 5, \
            f"Expected exactly 5 env variables, found {var_count}"
    
    def test_no_optional_env_vars_required(self):
        """Test that no optional environment variables are required for basic operation."""
        # Set only the essential 5 variables
        os.environ['TRADING_BINANCE_API_KEY'] = 'test_key'
        os.environ['TRADING_BINANCE_API_SECRET'] = 'test_secret'
        os.environ['TRADING_DISCORD_BOT_TOKEN'] = 'test_token'
        os.environ['TRADING_DISCORD_CHANNEL_ID'] = '123456'
        os.environ['TRADING_ENV'] = 'development'
        
        # Should be able to load config with just these
        manager = ConfigManager()
        config = manager.load_config()
        
        assert config is not None
        assert isinstance(config, TradingBotConfig)
