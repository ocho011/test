"""Tests for Pydantic v2 configuration validation and type safety."""
import pytest
from pydantic import ValidationError
from trading_bot.config.models import (
    BinanceConfig,
    DiscordConfig,
    LoggingConfig,
    SystemConfig,
    TradingBotConfig,
    LogLevel,
    Environment,
)


class TestBinanceConfigValidation:
    """Test Binance configuration validation."""

    def test_valid_binance_config(self):
        """Test valid Binance configuration."""
        config = BinanceConfig(
            api_key="valid_api_key_12345678901234567890",
            api_secret="valid_api_secret_12345678901234567890",
            testnet=False,
        )
        assert config.api_key == "valid_api_key_12345678901234567890"
        assert config.api_secret == "valid_api_secret_12345678901234567890"
        assert config.testnet is False

    def test_api_key_empty_raises_error(self):
        """Test that empty API key raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            BinanceConfig(
                api_key="",
                api_secret="valid_secret",
            )
        assert "api_key cannot be empty" in str(exc_info.value)

    def test_api_key_placeholder_raises_error(self):
        """Test that placeholder API key raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            BinanceConfig(
                api_key="your-api-key",
                api_secret="valid_secret_12345678901234567890",
            )
        assert "placeholder value" in str(exc_info.value)

    def test_api_key_too_short_raises_error(self):
        """Test that too short API key raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            BinanceConfig(
                api_key="short",
                api_secret="valid_secret_12345678901234567890",
            )
        assert "too short" in str(exc_info.value)

    def test_api_secret_validation(self):
        """Test API secret validation."""
        with pytest.raises(ValidationError) as exc_info:
            BinanceConfig(
                api_key="valid_key_12345678901234567890",
                api_secret="placeholder",
            )
        assert "placeholder value" in str(exc_info.value)

    def test_base_url_valid_format(self):
        """Test valid base URL format."""
        config = BinanceConfig(
            api_key="valid_key_12345678901234567890",
            api_secret="valid_secret_12345678901234567890",
            base_url="https://api.binance.com",
        )
        assert config.base_url == "https://api.binance.com"

    def test_base_url_invalid_format_raises_error(self):
        """Test invalid base URL format raises error."""
        with pytest.raises(ValidationError) as exc_info:
            BinanceConfig(
                api_key="valid_key_12345678901234567890",
                api_secret="valid_secret_12345678901234567890",
                base_url="not-a-valid-url",
            )
        assert "Invalid URL format" in str(exc_info.value)

    def test_testnet_consistency_with_testnet_url(self):
        """Test testnet flag consistency with testnet URL."""
        config = BinanceConfig(
            api_key="valid_key_12345678901234567890",
            api_secret="valid_secret_12345678901234567890",
            testnet=True,
            base_url="https://testnet.binance.vision",
        )
        assert config.testnet is True
        assert "testnet" in config.base_url.lower()

    def test_testnet_true_but_production_url_raises_error(self):
        """Test testnet=True with production URL raises error."""
        with pytest.raises(ValidationError) as exc_info:
            BinanceConfig(
                api_key="valid_key_12345678901234567890",
                api_secret="valid_secret_12345678901234567890",
                testnet=True,
                base_url="https://api.binance.com",
            )
        assert "testnet mode is enabled" in str(exc_info.value).lower()

    def test_testnet_false_but_testnet_url_raises_error(self):
        """Test testnet=False with testnet URL raises error."""
        with pytest.raises(ValidationError) as exc_info:
            BinanceConfig(
                api_key="valid_key_12345678901234567890",
                api_secret="valid_secret_12345678901234567890",
                testnet=False,
                base_url="https://testnet.binance.vision",
            )
        assert "testnet mode is disabled" in str(exc_info.value).lower()


class TestDiscordConfigValidation:
    """Test Discord configuration validation."""

    def test_valid_discord_config(self):
        """Test valid Discord configuration."""
        config = DiscordConfig(
            bot_token="valid_discord_token_12345678901234567890123456789012345678901234567890",
            channel_id=123456789012345678,
        )
        assert len(config.bot_token) >= 50
        assert config.channel_id > 0

    def test_bot_token_empty_raises_error(self):
        """Test empty bot token raises error."""
        with pytest.raises(ValidationError) as exc_info:
            DiscordConfig(
                bot_token="",
                channel_id=123456789012345678,
            )
        assert "bot_token cannot be empty" in str(exc_info.value)

    def test_bot_token_placeholder_raises_error(self):
        """Test placeholder bot token raises error."""
        with pytest.raises(ValidationError) as exc_info:
            DiscordConfig(
                bot_token="your-bot-token",
                channel_id=123456789012345678,
            )
        assert "placeholder value" in str(exc_info.value)

    def test_bot_token_too_short_raises_error(self):
        """Test too short bot token raises error."""
        with pytest.raises(ValidationError) as exc_info:
            DiscordConfig(
                bot_token="short_token",
                channel_id=123456789012345678,
            )
        assert "too short" in str(exc_info.value)

    def test_channel_id_negative_raises_error(self):
        """Test negative channel ID raises error."""
        with pytest.raises(ValidationError) as exc_info:
            DiscordConfig(
                bot_token="valid_token_12345678901234567890123456789012345678901234567890",
                channel_id=-1,
            )
        assert "positive integer" in str(exc_info.value)

    def test_channel_id_too_small_raises_error(self):
        """Test too small channel ID raises error."""
        with pytest.raises(ValidationError) as exc_info:
            DiscordConfig(
                bot_token="valid_token_12345678901234567890123456789012345678901234567890",
                channel_id=12345,
            )
        assert "Snowflake" in str(exc_info.value)

    def test_spam_protection_negative_raises_error(self):
        """Test negative spam protection raises error."""
        with pytest.raises(ValidationError) as exc_info:
            DiscordConfig(
                bot_token="valid_token_12345678901234567890123456789012345678901234567890",
                channel_id=123456789012345678,
                spam_protection_seconds=-1,
            )
        assert "cannot be negative" in str(exc_info.value)

    def test_spam_protection_too_large_raises_error(self):
        """Test too large spam protection raises error."""
        with pytest.raises(ValidationError) as exc_info:
            DiscordConfig(
                bot_token="valid_token_12345678901234567890123456789012345678901234567890",
                channel_id=123456789012345678,
                spam_protection_seconds=5000,
            )
        assert "too large" in str(exc_info.value)


class TestLoggingConfigValidation:
    """Test Logging configuration validation."""

    def test_valid_logging_config(self):
        """Test valid logging configuration."""
        config = LoggingConfig(
            level=LogLevel.INFO,
            format="json",
            output_dir="logs",
        )
        assert config.level == LogLevel.INFO
        assert config.format == "json"
        assert config.output_dir == "logs"

    def test_format_invalid_raises_error(self):
        """Test invalid log format raises error."""
        with pytest.raises(ValidationError) as exc_info:
            LoggingConfig(format="xml")
        assert "Invalid log format" in str(exc_info.value)

    def test_format_case_insensitive(self):
        """Test format validation is case insensitive."""
        config = LoggingConfig(format="JSON")
        assert config.format == "json"

    def test_output_dir_empty_raises_error(self):
        """Test empty output directory raises error."""
        with pytest.raises(ValidationError) as exc_info:
            LoggingConfig(output_dir="")
        assert "output_dir cannot be empty" in str(exc_info.value)

    def test_output_dir_invalid_characters_raises_error(self):
        """Test output directory with invalid characters raises error."""
        with pytest.raises(ValidationError) as exc_info:
            LoggingConfig(output_dir="logs<invalid>")
        assert "invalid characters" in str(exc_info.value)

    def test_max_bytes_negative_raises_error(self):
        """Test negative max bytes raises error."""
        with pytest.raises(ValidationError) as exc_info:
            LoggingConfig(max_bytes=-1)
        assert "must be positive" in str(exc_info.value)

    def test_max_bytes_too_small_raises_error(self):
        """Test too small max bytes raises error."""
        with pytest.raises(ValidationError) as exc_info:
            LoggingConfig(max_bytes=100)
        assert "too small" in str(exc_info.value)

    def test_max_bytes_too_large_raises_error(self):
        """Test too large max bytes raises error."""
        with pytest.raises(ValidationError) as exc_info:
            LoggingConfig(max_bytes=2000000000)  # 2GB
        assert "too large" in str(exc_info.value)

    def test_backup_count_negative_raises_error(self):
        """Test negative backup count raises error."""
        with pytest.raises(ValidationError) as exc_info:
            LoggingConfig(backup_count=-1)
        assert "cannot be negative" in str(exc_info.value)

    def test_backup_count_too_large_raises_error(self):
        """Test too large backup count raises error."""
        with pytest.raises(ValidationError) as exc_info:
            LoggingConfig(backup_count=200)
        assert "too large" in str(exc_info.value)


class TestSystemConfigValidation:
    """Test System configuration validation."""

    def test_valid_system_config(self):
        """Test valid system configuration."""
        config = SystemConfig(
            environment="development",
            debug=True,
            async_workers=4,
        )
        assert config.environment == "development"
        assert config.debug is True
        assert config.async_workers == 4

    def test_environment_invalid_raises_error(self):
        """Test invalid environment raises error."""
        with pytest.raises(ValidationError) as exc_info:
            SystemConfig(environment="invalid")
        assert "Invalid environment" in str(exc_info.value)

    def test_environment_valid_values(self):
        """Test all valid environment values."""
        for env in ["development", "staging", "production"]:
            config = SystemConfig(environment=env)
            assert config.environment == env

    def test_environment_case_insensitive(self):
        """Test environment validation is case insensitive."""
        config = SystemConfig(environment="DEVELOPMENT")
        assert config.environment == "development"

    def test_async_workers_negative_raises_error(self):
        """Test negative async workers raises error."""
        with pytest.raises(ValidationError) as exc_info:
            SystemConfig(async_workers=-1)
        assert "must be positive" in str(exc_info.value)

    def test_async_workers_too_large_raises_error(self):
        """Test too many async workers raises error."""
        with pytest.raises(ValidationError) as exc_info:
            SystemConfig(async_workers=50)
        assert "too large" in str(exc_info.value)

    def test_event_queue_size_negative_raises_error(self):
        """Test negative event queue size raises error."""
        with pytest.raises(ValidationError) as exc_info:
            SystemConfig(event_queue_size=-1)
        assert "must be positive" in str(exc_info.value)

    def test_event_queue_size_too_small_raises_error(self):
        """Test too small event queue size raises error."""
        with pytest.raises(ValidationError) as exc_info:
            SystemConfig(event_queue_size=5)
        assert "too small" in str(exc_info.value)

    def test_event_queue_size_too_large_raises_error(self):
        """Test too large event queue size raises error."""
        with pytest.raises(ValidationError) as exc_info:
            SystemConfig(event_queue_size=200000)
        assert "too large" in str(exc_info.value)

    def test_health_check_interval_negative_raises_error(self):
        """Test negative health check interval raises error."""
        with pytest.raises(ValidationError) as exc_info:
            SystemConfig(health_check_interval=-1)
        assert "must be positive" in str(exc_info.value)

    def test_health_check_interval_too_frequent_raises_error(self):
        """Test too frequent health check interval raises error."""
        with pytest.raises(ValidationError) as exc_info:
            SystemConfig(health_check_interval=5)
        assert "too frequent" in str(exc_info.value)

    def test_health_check_interval_too_infrequent_raises_error(self):
        """Test too infrequent health check interval raises error."""
        with pytest.raises(ValidationError) as exc_info:
            SystemConfig(health_check_interval=5000)
        assert "too infrequent" in str(exc_info.value)

    def test_memory_threshold_negative_raises_error(self):
        """Test negative memory threshold raises error."""
        with pytest.raises(ValidationError) as exc_info:
            SystemConfig(memory_threshold_mb=-1)
        assert "must be positive" in str(exc_info.value)

    def test_memory_threshold_too_low_raises_error(self):
        """Test too low memory threshold raises error."""
        with pytest.raises(ValidationError) as exc_info:
            SystemConfig(memory_threshold_mb=50)
        assert "too low" in str(exc_info.value)

    def test_memory_threshold_too_high_raises_error(self):
        """Test too high memory threshold raises error."""
        with pytest.raises(ValidationError) as exc_info:
            SystemConfig(memory_threshold_mb=50000)
        assert "too high" in str(exc_info.value)


class TestTradingBotConfigIntegration:
    """Test TradingBotConfig BaseSettings integration."""

    def test_trading_bot_config_with_valid_data(self):
        """Test TradingBotConfig with all valid data."""
        config = TradingBotConfig(
            system=SystemConfig(environment="development"),
            binance=BinanceConfig(
                api_key="valid_key_12345678901234567890",
                api_secret="valid_secret_12345678901234567890",
            ),
            discord=DiscordConfig(
                bot_token="valid_token_12345678901234567890123456789012345678901234567890",
                channel_id=123456789012345678,
            ),
            logging=LoggingConfig(),
            trading={
                "symbol": "BTCUSDT",
                "timeframe": "1h",
                "max_position_size": 0.1,
                "leverage": 1,
                "paper_trading": True,
                "max_positions": 5,
                "default_timeframe": "1h",
                "max_risk_per_trade": 0.02,
                "daily_loss_limit": 0.05,
            },
        )
        assert config.system.environment == "development"
        assert config.binance.testnet is False
        assert config.discord.enable_notifications is True

    def test_trading_bot_config_is_base_settings(self):
        """Test TradingBotConfig inherits from BaseSettings."""
        from pydantic_settings import BaseSettings

        assert issubclass(TradingBotConfig, BaseSettings)

    def test_trading_bot_config_has_settings_config(self):
        """Test TradingBotConfig has proper SettingsConfigDict."""
        config_dict = TradingBotConfig.model_config
        assert config_dict.get("env_prefix") == "TRADING_BOT_"
        assert config_dict.get("env_nested_delimiter") == "__"
        assert config_dict.get("case_sensitive") is False
        assert config_dict.get("env_file") == ".env"

    def test_nested_config_validation_cascades(self):
        """Test validation errors cascade from nested configs."""
        with pytest.raises(ValidationError) as exc_info:
            TradingBotConfig(
                system=SystemConfig(environment="invalid"),
                binance=BinanceConfig(
                    api_key="valid_key_12345678901234567890",
                    api_secret="valid_secret_12345678901234567890",
                ),
                discord=DiscordConfig(
                    bot_token="valid_token_12345678901234567890123456789012345678901234567890",
                    channel_id=123456789012345678,
                ),
                logging=LoggingConfig(),
                trading={
                    "symbol": "BTCUSDT",
                    "timeframe": "1h",
                    "max_position_size": 0.1,
                    "leverage": 1,
                    "paper_trading": True,
                    "max_positions": 5,
                    "default_timeframe": "1h",
                    "max_risk_per_trade": 0.02,
                    "daily_loss_limit": 0.05,
                },
            )
        assert "Invalid environment" in str(exc_info.value)
