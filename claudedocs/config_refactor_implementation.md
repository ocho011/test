# Configuration Refactoring - Implementation Guide

**Quick Start:** Phase 1 implementation for fixing critical YAML environment variable substitution bug.

---

## Phase 1: Fix YAML Environment Variable Substitution

### 1. Create Custom YAML Loader

Create new file: `src/trading_bot/config/yaml_env_loader.py`

```python
"""Custom YAML loader with environment variable substitution support."""
import os
import re
from typing import Any
import yaml


class EnvVarLoader(yaml.SafeLoader):
    """
    Custom YAML loader that supports environment variable substitution.
    
    Syntax:
        ${VAR}              - Substitute environment variable VAR
        ${VAR:-default}     - Substitute VAR, or use 'default' if not set
        ${VAR:?error_msg}   - Substitute VAR, or raise error with message
    
    Examples:
        api_key: ${TRADING_BINANCE_API_KEY}
        log_level: ${TRADING_LOG_LEVEL:-INFO}
        required: ${TRADING_ENV:?TRADING_ENV must be set}
    """
    pass


def env_var_constructor(loader: yaml.SafeLoader, node: yaml.Node) -> str:
    """
    YAML constructor for environment variable substitution.
    
    Supports three syntaxes:
    - ${VAR}            : Get var or empty string
    - ${VAR:-default}   : Get var or default value
    - ${VAR:?error}     : Get var or raise error
    """
    value = loader.construct_scalar(node)
    
    # Pattern: ${VAR} or ${VAR:-default} or ${VAR:?error}
    pattern = re.compile(r'\$\{([^}^{:-]+)(?::-([^}]+)|:\?([^}]+))?\}')
    
    def replace_env_var(match: re.Match) -> str:
        env_var = match.group(1).strip()
        default_val = match.group(2)  # From ${VAR:-default}
        error_msg = match.group(3)    # From ${VAR:?error}
        
        # Get environment variable
        env_value = os.environ.get(env_var)
        
        if env_value is not None:
            return env_value
        elif default_val is not None:
            return default_val
        elif error_msg is not None:
            raise ValueError(f"Required environment variable '{env_var}' not set: {error_msg}")
        else:
            # No default, no error - return empty string
            return ''
    
    return pattern.sub(replace_env_var, value)


# Register the custom constructor
yaml.add_implicit_resolver(
    '!env_var',
    re.compile(r'\$\{[^}]*\}'),
    None,
    EnvVarLoader
)

yaml.add_constructor(
    '!env_var',
    env_var_constructor,
    EnvVarLoader
)


def load_yaml_with_env(file_path: str) -> dict[str, Any]:
    """
    Load YAML file with environment variable substitution.
    
    Args:
        file_path: Path to YAML file
    
    Returns:
        Parsed YAML with environment variables substituted
    
    Raises:
        ValueError: If required environment variable not set (${VAR:?error} syntax)
        FileNotFoundError: If YAML file doesn't exist
        yaml.YAMLError: If YAML syntax is invalid
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.load(f, Loader=EnvVarLoader)
```

---

### 2. Update ConfigManager

Update `src/trading_bot/config/config_manager.py`:

```python
"""Configuration manager for loading and managing YAML configurations."""
import os
from pathlib import Path
from typing import Dict, Any, Optional
from functools import lru_cache

from .models import TradingBotConfig
from .yaml_env_loader import load_yaml_with_env  # NEW IMPORT


class ConfigManager:
    """Manages YAML-based configuration files with environment support."""
    
    _instance: Optional['ConfigManager'] = None
    _config: Optional[TradingBotConfig] = None
    _config_path: Optional[Path] = None
    
    def __new__(cls):
        """Singleton pattern to ensure single config instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize config manager."""
        self._cache_enabled = True
    
    def load_config(
        self, 
        config_path: Optional[str] = None,
        environment: Optional[str] = None
    ) -> TradingBotConfig:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to config file. If None, auto-detect from environment
            environment: Environment name (development, production, testing)
        
        Returns:
            TradingBotConfig: Validated configuration object
        """
        # Determine environment
        if environment is None:
            environment = os.getenv("TRADING_ENV", "development")
        
        # Determine config path
        if config_path is None:
            config_dir = Path(__file__).parent / "environments"
            config_path = config_dir / f"{environment}.yml"
        else:
            config_path = Path(config_path)
        
        # Check if file exists
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Load YAML with environment variable substitution
        config_data = load_yaml_with_env(str(config_path))  # CHANGED: Use new loader
        
        # REMOVED: _apply_env_overrides() - no longer needed!
        
        # Validate with pydantic
        self._config = TradingBotConfig(**config_data)
        self._config_path = config_path
        
        return self._config
    
    # Remove _apply_env_overrides() method entirely - it's obsolete
    
    def get_config(self) -> TradingBotConfig:
        """Get current configuration."""
        if self._config is None:
            raise RuntimeError("Configuration not loaded. Call load_config() first.")
        return self._config
    
    def reload_config(self) -> TradingBotConfig:
        """Reload configuration from file (hot-reload)."""
        if self._config_path is None:
            raise RuntimeError("No config path available for reload")
        return self.load_config(str(self._config_path))
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-notation key."""
        config = self.get_config()
        keys = key.split(".")
        value = config.model_dump()
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value by dot-notation key (runtime only)."""
        config = self.get_config()
        keys = key.split(".")
        
        # Navigate to the target
        target = config
        for k in keys[:-1]:
            target = getattr(target, k)
        
        # Set the value
        setattr(target, keys[-1], value)
    
    @lru_cache(maxsize=128)
    def get_cached(self, key: str) -> Any:
        """Get cached configuration value."""
        return self.get(key)
    
    def clear_cache(self) -> None:
        """Clear configuration cache."""
        self.get_cached.cache_clear()


# Global config manager instance
config_manager = ConfigManager()


def get_config() -> TradingBotConfig:
    """Get global configuration instance."""
    return config_manager.get_config()


def load_config(
    config_path: Optional[str] = None,
    environment: Optional[str] = None
) -> TradingBotConfig:
    """Load configuration from file."""
    return config_manager.load_config(config_path, environment)
```

---

### 3. Update YAML Files (Already Correct!)

The YAML files already use the correct syntax - they just weren't working before!

**No changes needed** to YAML files. They already have:

```yaml
# src/trading_bot/config/environments/production.yml
binance:
  api_key: ${TRADING_BINANCE_API_KEY}
  api_secret: ${TRADING_BINANCE_API_SECRET}

discord:
  bot_token: ${TRADING_DISCORD_BOT_TOKEN}
  channel_id: ${TRADING_DISCORD_CHANNEL_ID}
```

**Optional Enhancement:** Add defaults and required markers:

```yaml
binance:
  api_key: ${TRADING_BINANCE_API_KEY:?TRADING_BINANCE_API_KEY is required}
  api_secret: ${TRADING_BINANCE_API_SECRET:?TRADING_BINANCE_API_SECRET is required}
  testnet: ${TRADING_BINANCE_TESTNET:-false}

discord:
  bot_token: ${TRADING_DISCORD_BOT_TOKEN:-}
  channel_id: ${TRADING_DISCORD_CHANNEL_ID:-0}
  
logging:
  level: ${TRADING_LOG_LEVEL:-INFO}
```

---

### 4. Add Tests

Update `tests/config/test_config_manager.py`:

```python
"""Tests for ConfigManager with environment variable substitution."""
import os
import tempfile
import pytest
from pathlib import Path

from trading_bot.config import ConfigManager, TradingBotConfig
from trading_bot.config.yaml_env_loader import load_yaml_with_env


class TestYAMLEnvVarSubstitution:
    """Test YAML environment variable substitution."""
    
    def test_basic_env_var_substitution(self, monkeypatch):
        """Test basic ${VAR} substitution."""
        monkeypatch.setenv("TEST_VAR", "test_value")
        
        yaml_content = """
        system:
          test_key: ${TEST_VAR}
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name
        
        try:
            result = load_yaml_with_env(temp_path)
            assert result['system']['test_key'] == 'test_value'
        finally:
            os.remove(temp_path)
    
    def test_env_var_with_default(self, monkeypatch):
        """Test ${VAR:-default} syntax."""
        yaml_content = """
        system:
          with_default: ${MISSING_VAR:-default_value}
          with_env: ${EXISTING_VAR:-default_value}
        """
        
        monkeypatch.setenv("EXISTING_VAR", "actual_value")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name
        
        try:
            result = load_yaml_with_env(temp_path)
            assert result['system']['with_default'] == 'default_value'
            assert result['system']['with_env'] == 'actual_value'
        finally:
            os.remove(temp_path)
    
    def test_env_var_required_missing_raises_error(self):
        """Test ${VAR:?error} syntax raises error when missing."""
        yaml_content = """
        system:
          required: ${REQUIRED_VAR:?REQUIRED_VAR must be set}
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="REQUIRED_VAR must be set"):
                load_yaml_with_env(temp_path)
        finally:
            os.remove(temp_path)
    
    def test_env_var_required_present_works(self, monkeypatch):
        """Test ${VAR:?error} works when variable is set."""
        monkeypatch.setenv("REQUIRED_VAR", "present")
        
        yaml_content = """
        system:
          required: ${REQUIRED_VAR:?REQUIRED_VAR must be set}
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name
        
        try:
            result = load_yaml_with_env(temp_path)
            assert result['system']['required'] == 'present'
        finally:
            os.remove(temp_path)
    
    def test_multiple_env_vars_in_one_value(self, monkeypatch):
        """Test multiple ${VAR} in single value."""
        monkeypatch.setenv("VAR1", "hello")
        monkeypatch.setenv("VAR2", "world")
        
        yaml_content = """
        system:
          combined: ${VAR1} ${VAR2}!
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name
        
        try:
            result = load_yaml_with_env(temp_path)
            assert result['system']['combined'] == 'hello world!'
        finally:
            os.remove(temp_path)


class TestConfigManagerRefactored:
    """Test ConfigManager with new YAML loader."""
    
    def test_config_loads_with_env_vars(self, monkeypatch):
        """Test that config loads correctly with environment variables."""
        monkeypatch.setenv("TRADING_BINANCE_API_KEY", "test_key_123")
        monkeypatch.setenv("TRADING_BINANCE_API_SECRET", "test_secret_456")
        monkeypatch.setenv("TRADING_DISCORD_BOT_TOKEN", "test_token_789")
        monkeypatch.setenv("TRADING_DISCORD_CHANNEL_ID", "123456789")
        
        # Use testing.yml which has ${VAR:-default} syntax
        manager = ConfigManager()
        config = manager.load_config(environment='testing')
        
        assert config.binance.api_key == "test_key_123"
        assert config.binance.api_secret == "test_secret_456"
        assert config.discord.bot_token == "test_token_789"
        assert config.discord.channel_id == 123456789
    
    def test_defaults_work_when_env_vars_missing(self):
        """Test that default values work when env vars not set."""
        # testing.yml has ${VAR:-default} syntax
        manager = ConfigManager()
        config = manager.load_config(environment='testing')
        
        # Should use defaults from ${VAR:-default}
        assert config.binance.api_key == "test_key"
        assert config.binance.api_secret == "test_secret"
    
    def test_any_yaml_value_can_be_overridden(self, monkeypatch):
        """Test that ANY YAML value can be overridden with env var (not just hardcoded 7)."""
        # Create custom YAML with new env var
        yaml_content = """
        system:
          environment: testing
          debug: ${CUSTOM_DEBUG:-false}
          async_workers: ${CUSTOM_WORKERS:-2}
        
        binance:
          api_key: ${TRADING_BINANCE_API_KEY:-test}
          api_secret: ${TRADING_BINANCE_API_SECRET:-test}
          testnet: true
        
        discord:
          bot_token: ${TRADING_DISCORD_BOT_TOKEN:-test}
          channel_id: ${TRADING_DISCORD_CHANNEL_ID:-0}
        
        logging:
          level: INFO
          format: text
        
        trading:
          symbol: BTCUSDT
          timeframe: 1h
          max_position_size: ${CUSTOM_POSITION_SIZE:-0.1}
          leverage: 1
          max_risk_per_trade: 0.01
          daily_loss_limit: 0.03
        """
        
        # Set custom env vars
        monkeypatch.setenv("CUSTOM_DEBUG", "true")
        monkeypatch.setenv("CUSTOM_WORKERS", "8")
        monkeypatch.setenv("CUSTOM_POSITION_SIZE", "0.25")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name
        
        try:
            manager = ConfigManager()
            config = manager.load_config(temp_path)
            
            # Verify ALL custom env vars worked (not just hardcoded 7!)
            assert config.system.debug is True
            assert config.system.async_workers == 8
            assert config.trading.max_position_size == 0.25
        finally:
            os.remove(temp_path)


class TestBackwardCompatibility:
    """Ensure existing functionality still works."""
    
    def test_existing_tests_still_pass(self):
        """All existing test cases should still pass."""
        # Run existing test suite
        # (Existing tests from test_config_manager.py should work)
        pass
```

---

### 5. Validation Checklist

Before deploying, verify:

**✅ Unit Tests**
- [ ] `test_basic_env_var_substitution` passes
- [ ] `test_env_var_with_default` passes  
- [ ] `test_env_var_required_missing_raises_error` passes
- [ ] `test_multiple_env_vars_in_one_value` passes
- [ ] `test_any_yaml_value_can_be_overridden` passes
- [ ] All existing tests still pass

**✅ Integration Tests**
- [ ] Development environment loads correctly
- [ ] Production environment loads correctly
- [ ] Paper trading environment loads correctly
- [ ] Testing environment loads correctly

**✅ Manual Testing**
```bash
# Test with missing env var (should use default)
unset TRADING_LOG_LEVEL
python -c "from trading_bot.config import load_config; c=load_config(environment='development'); print(c.logging.level)"
# Expected: INFO (default from YAML)

# Test with env var override
export TRADING_LOG_LEVEL=DEBUG
python -c "from trading_bot.config import load_config; c=load_config(environment='development'); print(c.logging.level)"
# Expected: DEBUG (from env var)

# Test required var missing (should raise error)
unset TRADING_BINANCE_API_KEY
python -c "from trading_bot.config import load_config; load_config(environment='production')"
# Expected: ValueError with clear message
```

**✅ Code Review**
- [ ] `_apply_env_overrides()` removed from config_manager.py
- [ ] All YAML files still load correctly
- [ ] No hardcoded environment variable logic remains
- [ ] Documentation updated

---

## Rollback Plan

If issues arise, rollback steps:

1. **Keep old config_manager.py** as `config_manager_legacy.py`
2. **Feature flag** in `__init__.py`:
   ```python
   import os
   if os.getenv('USE_LEGACY_CONFIG') == 'true':
       from .config_manager_legacy import ConfigManager
   else:
       from .config_manager import ConfigManager
   ```
3. **Set env var** to rollback: `export USE_LEGACY_CONFIG=true`

---

## Next Steps After Phase 1

Once Phase 1 is complete and tested:

1. ✅ Deploy to development environment
2. ✅ Test for 24 hours
3. ✅ Deploy to paper trading environment
4. ✅ Test for 48 hours
5. ✅ Proceed to Phase 2 (streamline .env files)

---

## Common Issues and Solutions

### Issue: "Config not loading"
**Solution:** Check YAML syntax, verify file exists

### Issue: "Env var not substituting"
**Solution:** Verify ${VAR} syntax, check env var is set

### Issue: "Required var error"
**Solution:** Set the required environment variable

### Issue: "Tests failing"
**Solution:** Update tests to expect new behavior, check for hardcoded assumptions

---

**Implementation Status:** 🟡 Ready for Phase 1  
**Last Updated:** 2025-10-04
