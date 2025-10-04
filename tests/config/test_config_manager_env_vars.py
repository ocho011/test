"""Integration tests for ConfigManager with environment variable substitution."""
import os
import pytest
from pathlib import Path
import tempfile

from trading_bot.config.config_manager import ConfigManager


class TestConfigManagerEnvVarIntegration:
    """Test ConfigManager with YAML env var substitution."""
    
    @pytest.fixture
    def config_manager(self):
        """Create fresh ConfigManager instance."""
        # Reset singleton
        ConfigManager._instance = None
        ConfigManager._config = None
        ConfigManager._config_path = None
        return ConfigManager()
    
    def test_load_config_with_env_vars(self, config_manager, monkeypatch, tmp_path):
        """Test loading config with environment variable substitution."""
        # Set environment variables
        monkeypatch.setenv("TEST_API_KEY", "test_key_123")
        monkeypatch.setenv("TEST_API_SECRET", "test_secret_456")
        monkeypatch.setenv("TEST_DEBUG", "true")
        monkeypatch.setenv("TEST_PORT", "8080")
        
        # Create test config file
        config_file = tmp_path / "test.yml"
        config_file.write_text("""
system:
  environment: testing
  debug: ${TEST_DEBUG}
  async_workers: 2
  event_queue_size: 1000
  health_check_interval: 600
  memory_threshold_mb: 512

binance:
  api_key: ${TEST_API_KEY}
  api_secret: ${TEST_API_SECRET}
  testnet: true
  base_url: https://testnet.binance.vision

discord:
  bot_token: test_token
  channel_id: ${TEST_PORT}
  enable_notifications: false
  spam_protection_seconds: 30

logging:
  level: DEBUG
  format: text
  output_dir: logs
  max_bytes: 5242880
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

data:
  cache_size: 500
  cache_ttl: 300
  historical_lookback_days: 30

risk:
  initial_capital: 10000.0
  max_position_size_pct: 0.05
  max_drawdown_pct: 0.15
  max_consecutive_losses: 3
  volatility_threshold: 0.05

strategy:
  enabled_strategies:
    - ict
    - traditional
  default_strategy: ict
  strategy_selection_mode: auto

execution:
  slippage_tolerance_pct: 0.002
  max_retry_attempts: 3
  order_timeout_seconds: 30

system_events:
  event_queue_size: 1000
  message_hub_max_subscribers: 100

custom:
  mock_api: false
  enable_backtesting: true
  dry_run: false
        """)
        
        # Load config
        config = config_manager.load_config(str(config_file))
        
        # Verify environment variable substitution
        assert config.binance.api_key == "test_key_123"
        assert config.binance.api_secret == "test_secret_456"
        assert config.system.debug is True
        assert config.discord.channel_id == 8080
    
    def test_env_var_with_defaults(self, config_manager, monkeypatch, tmp_path):
        """Test environment variables with default values."""
        # Don't set TEST_OPTIONAL, it should use default
        monkeypatch.delenv("TEST_OPTIONAL", raising=False)
        monkeypatch.setenv("TEST_REQUIRED", "required_value")
        
        config_file = tmp_path / "test.yml"
        config_file.write_text("""
system:
  environment: testing
  debug: ${TEST_DEBUG:false}
  async_workers: 2
  event_queue_size: 1000
  health_check_interval: 600
  memory_threshold_mb: 512

binance:
  api_key: ${TEST_REQUIRED}
  api_secret: ${TEST_OPTIONAL:default_secret}
  testnet: true
  base_url: https://testnet.binance.vision

discord:
  bot_token: test_token
  channel_id: 0
  enable_notifications: false
  spam_protection_seconds: 30

logging:
  level: DEBUG
  format: text
  output_dir: logs
  max_bytes: 5242880
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

data:
  cache_size: 500
  cache_ttl: 300
  historical_lookback_days: 30

risk:
  initial_capital: 10000.0
  max_position_size_pct: 0.05
  max_drawdown_pct: 0.15
  max_consecutive_losses: 3
  volatility_threshold: 0.05

strategy:
  enabled_strategies:
    - ict
    - traditional
  default_strategy: ict
  strategy_selection_mode: auto

execution:
  slippage_tolerance_pct: 0.002
  max_retry_attempts: 3
  order_timeout_seconds: 30

system_events:
  event_queue_size: 1000
  message_hub_max_subscribers: 100

custom:
  mock_api: false
  enable_backtesting: true
  dry_run: false
        """)
        
        config = config_manager.load_config(str(config_file))
        
        assert config.binance.api_key == "required_value"
        assert config.binance.api_secret == "default_secret"  # Used default
        assert config.system.debug is False  # Used default
    
    def test_type_preservation_in_config(self, config_manager, monkeypatch, tmp_path):
        """Test that types are preserved correctly in config."""
        monkeypatch.setenv("WORKERS", "4")
        monkeypatch.setenv("QUEUE_SIZE", "2000")
        monkeypatch.setenv("ENABLE_DEBUG", "true")
        monkeypatch.setenv("THRESHOLD", "0.85")
        
        config_file = tmp_path / "test.yml"
        config_file.write_text("""
system:
  environment: testing
  debug: ${ENABLE_DEBUG}
  async_workers: ${WORKERS}
  event_queue_size: ${QUEUE_SIZE}
  health_check_interval: 600
  memory_threshold_mb: 512

binance:
  api_key: test_key
  api_secret: test_secret
  testnet: true
  base_url: https://testnet.binance.vision

discord:
  bot_token: test_token
  channel_id: 0
  enable_notifications: false
  spam_protection_seconds: 30

logging:
  level: DEBUG
  format: text
  output_dir: logs
  max_bytes: 5242880
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

data:
  cache_size: 500
  cache_ttl: 300
  historical_lookback_days: 30

risk:
  initial_capital: 10000.0
  max_position_size_pct: ${THRESHOLD}
  max_drawdown_pct: 0.15
  max_consecutive_losses: 3
  volatility_threshold: 0.05

strategy:
  enabled_strategies:
    - ict
    - traditional
  default_strategy: ict
  strategy_selection_mode: auto

execution:
  slippage_tolerance_pct: 0.002
  max_retry_attempts: 3
  order_timeout_seconds: 30

system_events:
  event_queue_size: 1000
  message_hub_max_subscribers: 100

custom:
  mock_api: false
  enable_backtesting: true
  dry_run: false
        """)
        
        config = config_manager.load_config(str(config_file))
        
        # Check integer preservation
        assert config.system.async_workers == 4
        assert isinstance(config.system.async_workers, int)
        
        assert config.system.event_queue_size == 2000
        assert isinstance(config.system.event_queue_size, int)
        
        # Check boolean preservation
        assert config.system.debug is True
        assert isinstance(config.system.debug, bool)
        
        # Check float preservation
        assert config.risk.max_position_size_pct == 0.85
        assert isinstance(config.risk.max_position_size_pct, float)
    
    def test_real_development_config(self, config_manager, monkeypatch):
        """Test with actual development.yml config file."""
        # Set required environment variables
        monkeypatch.setenv("TRADING_BINANCE_API_KEY", "dev_key_123")
        monkeypatch.setenv("TRADING_BINANCE_API_SECRET", "dev_secret_456")
        monkeypatch.setenv("TRADING_DISCORD_BOT_TOKEN", "dev_bot_token")
        
        # Load the actual development config
        config_path = Path("src/trading_bot/config/environments/development.yml")
        
        if config_path.exists():
            config = config_manager.load_config(str(config_path))
            
            # Verify the environment variables were substituted
            assert config.binance.api_key == "dev_key_123"
            assert config.binance.api_secret == "dev_secret_456"
            assert config.discord.bot_token == "dev_bot_token"
            
            # Verify other config values are intact
            assert config.system.environment == "development"
            assert config.binance.testnet is True
