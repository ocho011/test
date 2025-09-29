"""Tests for ConfigManager."""
import os
import tempfile
import pytest
from pathlib import Path

from trading_bot.config import (
    ConfigManager,
    TradingBotConfig,
    load_config,
    get_config,
)


@pytest.fixture
def temp_config_file():
    """Create temporary config file for testing."""
    config_content = """
system:
  environment: testing
  debug: true
  async_workers: 2
  health_check_interval: 300
  memory_threshold_mb: 512

binance:
  api_key: test_key_123
  api_secret: test_secret_456
  testnet: true
  base_url: https://testnet.binance.vision

discord:
  bot_token: test_token_789
  channel_id: 123456789
  enable_notifications: false
  spam_protection_seconds: 30

logging:
  level: DEBUG
  format: json
  output_dir: test_logs
  max_bytes: 1048576
  backup_count: 3
  rotation_time: midnight
  trade_log_enabled: true
  trade_log_file: trades.log
  performance_log_enabled: true
  performance_log_file: performance.log
  mask_sensitive_data: true
  debug_unmask: true

trading:
  symbol: BTCUSDT
  timeframe: 1h
  max_position_size: 0.01
  leverage: 1
  max_risk_per_trade: 0.01
  daily_loss_limit: 0.03

custom:
  test_mode: true
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
        f.write(config_content)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)


class TestConfigManager:
    """Test cases for ConfigManager."""
    
    def test_singleton_pattern(self):
        """Test that ConfigManager follows singleton pattern."""
        manager1 = ConfigManager()
        manager2 = ConfigManager()
        assert manager1 is manager2
    
    def test_load_config_from_file(self, temp_config_file):
        """Test loading configuration from file."""
        manager = ConfigManager()
        config = manager.load_config(temp_config_file)
        
        assert isinstance(config, TradingBotConfig)
        assert config.system.environment == "testing"
        assert config.binance.api_key == "test_key_123"
        assert config.trading.symbol == "BTCUSDT"
    
    def test_load_config_with_env_override(self, temp_config_file, monkeypatch):
        """Test environment variable override."""
        # Set environment variables
        monkeypatch.setenv("TRADING_BINANCE_API_KEY", "env_key_override")
        monkeypatch.setenv("TRADING_DISCORD_CHANNEL_ID", "999999999")
        
        manager = ConfigManager()
        config = manager.load_config(temp_config_file)
        
        assert config.binance.api_key == "env_key_override"
        assert config.discord.channel_id == 999999999
    
    def test_get_config_before_load_raises_error(self):
        """Test that get_config raises error if not loaded."""
        manager = ConfigManager()
        manager._config = None  # Reset
        
        with pytest.raises(RuntimeError):
            manager.get_config()
    
    def test_get_config_value(self, temp_config_file):
        """Test getting configuration value by dot notation."""
        manager = ConfigManager()
        manager.load_config(temp_config_file)
        
        assert manager.get("binance.api_key") == "test_key_123"
        assert manager.get("trading.symbol") == "BTCUSDT"
        assert manager.get("nonexistent.key", "default") == "default"
    
    def test_set_config_value(self, temp_config_file):
        """Test setting configuration value at runtime."""
        manager = ConfigManager()
        manager.load_config(temp_config_file)
        
        manager.set("system.debug", False)
        assert manager.get("system.debug") is False
    
    def test_config_validation(self, temp_config_file):
        """Test pydantic validation."""
        manager = ConfigManager()
        config = manager.load_config(temp_config_file)
        
        # Valid risk percentage
        assert 0 < config.trading.max_risk_per_trade <= 1
        
        # Test validation error
        with pytest.raises(Exception):
            config.trading.max_risk_per_trade = 1.5  # Invalid > 1
    
    def test_global_functions(self, temp_config_file):
        """Test global convenience functions."""
        # Load config
        config = load_config(temp_config_file)
        assert isinstance(config, TradingBotConfig)
        
        # Get config
        retrieved_config = get_config()
        assert retrieved_config is config
    
    def test_reload_config(self, temp_config_file):
        """Test hot-reload configuration."""
        manager = ConfigManager()
        config1 = manager.load_config(temp_config_file)
        
        # Modify config file
        with open(temp_config_file, 'r') as f:
            content = f.read()
        
        modified_content = content.replace('test_key_123', 'reloaded_key_456')
        with open(temp_config_file, 'w') as f:
            f.write(modified_content)
        
        # Reload
        config2 = manager.reload_config()
        assert config2.binance.api_key == "reloaded_key_456"
    
    def test_cache_functionality(self, temp_config_file):
        """Test configuration value caching."""
        manager = ConfigManager()
        manager.load_config(temp_config_file)
        
        # Get cached value
        value1 = manager.get_cached("binance.api_key")
        value2 = manager.get_cached("binance.api_key")
        
        assert value1 == value2
        
        # Clear cache
        manager.clear_cache()
        value3 = manager.get_cached("binance.api_key")
        assert value3 == value1


class TestConfigModels:
    """Test configuration model validation."""
    
    def test_trading_config_risk_validation(self):
        """Test risk percentage validation."""
        from trading_bot.config import TradingConfig
        
        # Valid values
        config = TradingConfig(
            symbol="BTCUSDT",
            timeframe="1h",
            max_position_size=0.1,
            leverage=1,
            max_risk_per_trade=0.02,
            daily_loss_limit=0.05
        )
        assert config.max_risk_per_trade == 0.02
        
        # Invalid values
        with pytest.raises(Exception):
            TradingConfig(
                symbol="BTCUSDT",
                timeframe="1h",
                max_position_size=0.1,
                leverage=1,
                max_risk_per_trade=1.5,  # > 1
                daily_loss_limit=0.05
            )
    
    def test_log_level_enum(self):
        """Test LogLevel enum."""
        from trading_bot.config import LogLevel
        
        assert LogLevel.DEBUG == "DEBUG"
        assert LogLevel.INFO == "INFO"
        assert LogLevel.WARNING == "WARNING"
        assert LogLevel.ERROR == "ERROR"
        assert LogLevel.CRITICAL == "CRITICAL"