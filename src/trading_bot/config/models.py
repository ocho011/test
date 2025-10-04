"""Pydantic models for configuration validation."""
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from enum import Enum
import re
from pathlib import Path


class LogLevel(str, Enum):
    """Log level enumeration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Environment(str, Enum):
    """Environment enumeration."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class BinanceConfig(BaseModel):
    """Binance API configuration."""
    api_key: str = Field(..., description="Binance API key")
    api_secret: str = Field(..., description="Binance API secret")
    testnet: bool = Field(default=False, description="Use testnet")
    base_url: Optional[str] = Field(None, description="Custom base URL")

    @field_validator("api_key", "api_secret")
    @classmethod
    def validate_api_credentials(cls, v: str, info) -> str:
        """Validate API credentials are not empty or placeholder."""
        if not v or not v.strip():
            raise ValueError(f"{info.field_name} cannot be empty")

        # Check for common placeholder values
        placeholders = ["your-api-key", "your-key-here", "placeholder", "xxx", "test"]
        if v.lower() in placeholders:
            raise ValueError(
                f"{info.field_name} appears to be a placeholder value. "
                "Please provide a valid Binance API credential."
            )

        # Check minimum length (Binance keys are typically 64 characters)
        if len(v) < 10:
            raise ValueError(
                f"{info.field_name} is too short. "
                "Valid Binance API credentials should be at least 10 characters."
            )

        return v.strip()

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, v: Optional[str]) -> Optional[str]:
        """Validate base URL format if provided."""
        if v is None:
            return v

        v = v.strip()
        if not v:
            return None

        # Validate URL format
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain
            r'localhost|'  # localhost
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # or IP
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE
        )

        if not url_pattern.match(v):
            raise ValueError(
                f"Invalid URL format: {v}. "
                "URL must start with http:// or https:// and be a valid domain or IP."
            )

        return v

    @model_validator(mode='after')
    def validate_testnet_consistency(self) -> 'BinanceConfig':
        """Validate testnet flag is consistent with base_url."""
        if self.base_url and self.testnet:
            # If testnet is True, base_url should contain 'testnet'
            if 'testnet' not in self.base_url.lower():
                raise ValueError(
                    "Testnet mode is enabled but base_url does not appear to be a testnet URL. "
                    "When testnet=True, base_url should contain 'testnet' (e.g., 'https://testnet.binance.vision')."
                )
        elif self.base_url and not self.testnet:
            # If testnet is False, base_url should NOT contain 'testnet'
            if 'testnet' in self.base_url.lower():
                raise ValueError(
                    "Testnet mode is disabled but base_url appears to be a testnet URL. "
                    "When testnet=False, base_url should be a production URL (e.g., 'https://api.binance.com')."
                )

        return self


class DiscordConfig(BaseModel):
    """Discord bot configuration."""
    bot_token: str = Field(..., description="Discord bot token")
    channel_id: int = Field(..., description="Target channel ID")
    enable_notifications: bool = Field(default=True, description="Enable Discord notifications")
    spam_protection_seconds: int = Field(default=60, description="Spam protection interval")

    @field_validator("bot_token")
    @classmethod
    def validate_bot_token(cls, v: str) -> str:
        """Validate Discord bot token is not empty or placeholder."""
        if not v or not v.strip():
            raise ValueError("Discord bot_token cannot be empty")

        # Check for common placeholder values
        placeholders = ["your-bot-token", "your-token-here", "placeholder", "xxx"]
        if v.lower() in placeholders:
            raise ValueError(
                "bot_token appears to be a placeholder value. "
                "Please provide a valid Discord bot token."
            )

        # Discord tokens typically have a specific format
        if len(v) < 50:
            raise ValueError(
                "bot_token is too short. "
                "Valid Discord bot tokens are typically 50+ characters."
            )

        return v.strip()

    @field_validator("channel_id")
    @classmethod
    def validate_channel_id(cls, v: int) -> int:
        """Validate Discord channel ID is a positive integer."""
        if v <= 0:
            raise ValueError("channel_id must be a positive integer (Discord Snowflake ID)")

        # Discord Snowflake IDs are typically 17-19 digits
        if v < 100000000000000000:  # 17 digits minimum
            raise ValueError(
                "channel_id appears to be invalid. "
                "Discord channel IDs are typically 17-19 digit Snowflake IDs."
            )

        return v

    @field_validator("spam_protection_seconds")
    @classmethod
    def validate_spam_protection(cls, v: int) -> int:
        """Validate spam protection interval is reasonable."""
        if v < 0:
            raise ValueError("spam_protection_seconds cannot be negative")

        if v > 3600:  # 1 hour
            raise ValueError(
                "spam_protection_seconds is too large (max: 3600 seconds / 1 hour). "
                "Consider using a smaller interval to avoid excessive delays."
            )

        return v


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: LogLevel = Field(default=LogLevel.INFO, description="Default log level")
    format: str = Field(default="json", description="Log format: json or text")
    output_dir: str = Field(default="logs", description="Log output directory")

    # Rotation settings
    max_bytes: int = Field(default=10485760, description="Max log file size (10MB)")
    backup_count: int = Field(default=5, description="Number of backup files")
    rotation_time: str = Field(default="midnight", description="Time-based rotation")

    # Trade logs
    trade_log_enabled: bool = Field(default=True, description="Enable trade-specific logs")
    trade_log_file: str = Field(default="trades.log", description="Trade log filename")

    # Performance logs
    performance_log_enabled: bool = Field(default=True, description="Enable performance logs")
    performance_log_file: str = Field(default="performance.log", description="Performance log filename")

    # Sensitive data masking
    mask_sensitive_data: bool = Field(default=True, description="Enable data masking")
    debug_unmask: bool = Field(default=False, description="Unmask in debug mode")

    @field_validator("format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        """Validate log format is either 'json' or 'text'."""
        v = v.lower().strip()
        if v not in ["json", "text"]:
            raise ValueError(
                f"Invalid log format: {v}. Must be either 'json' or 'text'."
            )
        return v

    @field_validator("output_dir")
    @classmethod
    def validate_output_dir(cls, v: str) -> str:
        """Validate output directory path."""
        if not v or not v.strip():
            raise ValueError("output_dir cannot be empty")

        v = v.strip()

        # Check for invalid characters in path
        invalid_chars = ['<', '>', ':', '"', '|', '?', '*']
        if any(char in v for char in invalid_chars):
            raise ValueError(
                f"output_dir contains invalid characters. "
                f"Avoid using: {', '.join(invalid_chars)}"
            )

        return v

    @field_validator("max_bytes")
    @classmethod
    def validate_max_bytes(cls, v: int) -> int:
        """Validate max log file size is reasonable."""
        if v <= 0:
            raise ValueError("max_bytes must be positive")

        # Warn if less than 1KB or more than 1GB
        if v < 1024:
            raise ValueError(
                "max_bytes is too small (min: 1024 bytes / 1KB). "
                "Consider using at least 1KB for log files."
            )

        if v > 1073741824:  # 1GB
            raise ValueError(
                "max_bytes is too large (max: 1GB). "
                "Consider using smaller log files with rotation."
            )

        return v

    @field_validator("backup_count")
    @classmethod
    def validate_backup_count(cls, v: int) -> int:
        """Validate backup count is reasonable."""
        if v < 0:
            raise ValueError("backup_count cannot be negative")

        if v > 100:
            raise ValueError(
                "backup_count is too large (max: 100). "
                "Consider using fewer backup files to save disk space."
            )

        return v


class TradingConfig(BaseModel):
    """Trading strategy configuration."""
    symbol: str = Field(default="BTCUSDT", description="Trading symbol")
    timeframe: str = Field(default="1h", description="Chart timeframe")
    max_position_size: float = Field(default=0.1, description="Max position size")
    leverage: int = Field(default=1, description="Trading leverage")
    paper_trading: bool = Field(default=True, description="Enable paper trading mode")
    max_positions: int = Field(default=5, description="Maximum concurrent positions")
    default_timeframe: str = Field(default="1h", description="Default chart timeframe")

    # Risk management
    max_risk_per_trade: float = Field(default=0.02, description="Max 2% risk per trade")
    daily_loss_limit: float = Field(default=0.05, description="Daily loss limit 5%")

    @field_validator("max_risk_per_trade", "daily_loss_limit")
    @classmethod
    def validate_risk_percentage(cls, v: float) -> float:
        """Validate risk percentages are between 0 and 1."""
        if not 0 < v <= 1:
            raise ValueError("Risk percentage must be between 0 and 1")
        return v


class DataConfig(BaseModel):
    """Data layer configuration."""
    cache_size: int = Field(default=1000, description="Cache size")
    cache_ttl: int = Field(default=300, description="Cache TTL in seconds")
    default_interval: str = Field(default="1m", description="Default interval")
    klines_limit: int = Field(default=500, description="Klines limit")
    historical_lookback_days: int = Field(default=90, description="Historical lookback days")


class RiskConfig(BaseModel):
    """Risk management configuration."""
    initial_capital: float = Field(default=10000.0, description="Initial capital")
    max_position_size_pct: float = Field(default=0.1, description="Max position size percentage")
    max_drawdown_pct: float = Field(default=0.2, description="Max drawdown percentage")
    max_consecutive_losses: int = Field(default=3, description="Max consecutive losses")
    risk_per_trade_pct: float = Field(default=0.01, description="Risk per trade percentage")
    volatility_threshold: float = Field(default=0.05, description="Volatility threshold")


class SignalsConfig(BaseModel):
    """Signal generation configuration."""
    min_confidence: float = Field(default=0.6, description="Minimum signal confidence")
    signal_timeout_minutes: int = Field(default=60, description="Signal timeout in minutes")
    validation_required: bool = Field(default=True, description="Require signal validation")


class StrategiesConfig(BaseModel):
    """Strategy configuration."""
    default_strategy: str = Field(default="ict", description="Default strategy")
    enable_portfolio: bool = Field(default=False, description="Enable portfolio strategies")
    enabled_strategies: List[str] = Field(default_factory=lambda: ["ict", "traditional"], description="Enabled strategies")
    strategy_selection_mode: str = Field(default="auto", description="Strategy selection mode")


class ExecutionConfig(BaseModel):
    """Execution configuration."""
    default_order_type: str = Field(default="MARKET", description="Default order type")
    slippage_tolerance_pct: float = Field(default=0.001, description="Slippage tolerance percentage")
    max_retries: int = Field(default=3, description="Max retry attempts")
    retry_delay_ms: int = Field(default=1000, description="Retry delay in milliseconds")
    order_timeout_seconds: int = Field(default=30, description="Order timeout in seconds")
    dry_run: bool = Field(default=False, description="Dry-run mode: log orders without execution")


class SystemConfig(BaseModel):
    """System configuration."""
    environment: str = Field(default="development", description="Environment name")
    debug: bool = Field(default=False, description="Debug mode")
    async_workers: int = Field(default=4, description="Number of async workers")

    # Event system
    event_queue_size: int = Field(default=1000, description="Event queue size")

    # Health check
    health_check_interval: int = Field(default=300, description="Health check interval (seconds)")
    memory_threshold_mb: int = Field(default=1024, description="Memory warning threshold")

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment is one of the allowed values."""
        v = v.lower().strip()
        valid_environments = [e.value for e in Environment]

        if v not in valid_environments:
            raise ValueError(
                f"Invalid environment: {v}. "
                f"Must be one of: {', '.join(valid_environments)}"
            )

        return v

    @field_validator("async_workers")
    @classmethod
    def validate_async_workers(cls, v: int) -> int:
        """Validate async workers count is reasonable."""
        if v <= 0:
            raise ValueError("async_workers must be positive")

        if v > 32:
            raise ValueError(
                "async_workers is too large (max: 32). "
                "Using too many workers can degrade performance due to context switching."
            )

        return v

    @field_validator("event_queue_size")
    @classmethod
    def validate_event_queue_size(cls, v: int) -> int:
        """Validate event queue size is reasonable."""
        if v <= 0:
            raise ValueError("event_queue_size must be positive")

        if v < 10:
            raise ValueError(
                "event_queue_size is too small (min: 10). "
                "Consider using at least 10 for reliable event handling."
            )

        if v > 100000:
            raise ValueError(
                "event_queue_size is too large (max: 100000). "
                "Large queues can consume excessive memory."
            )

        return v

    @field_validator("health_check_interval")
    @classmethod
    def validate_health_check_interval(cls, v: int) -> int:
        """Validate health check interval is reasonable."""
        if v <= 0:
            raise ValueError("health_check_interval must be positive")

        if v < 10:
            raise ValueError(
                "health_check_interval is too frequent (min: 10 seconds). "
                "Very frequent health checks can impact performance."
            )

        if v > 3600:
            raise ValueError(
                "health_check_interval is too infrequent (max: 3600 seconds / 1 hour). "
                "Health checks should run at least once per hour."
            )

        return v

    @field_validator("memory_threshold_mb")
    @classmethod
    def validate_memory_threshold(cls, v: int) -> int:
        """Validate memory threshold is reasonable."""
        if v <= 0:
            raise ValueError("memory_threshold_mb must be positive")

        if v < 128:
            raise ValueError(
                "memory_threshold_mb is too low (min: 128 MB). "
                "Consider using at least 128MB threshold."
            )

        if v > 32768:  # 32GB
            raise ValueError(
                "memory_threshold_mb is too high (max: 32768 MB / 32 GB). "
                "Consider using a lower threshold."
            )

        return v


class TradingBotConfig(BaseSettings):
    """Main trading bot configuration with environment variable support."""
    system: SystemConfig
    binance: BinanceConfig
    discord: DiscordConfig
    logging: LoggingConfig
    trading: TradingConfig

    # Configuration models
    data: DataConfig = Field(default_factory=DataConfig, description="Data layer configuration")
    risk: RiskConfig = Field(default_factory=RiskConfig, description="Risk management configuration")
    signals: SignalsConfig = Field(default_factory=SignalsConfig, description="Signal generation configuration")
    strategies: StrategiesConfig = Field(default_factory=StrategiesConfig, description="Strategy configuration")
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig, description="Execution configuration")

    # System events configuration (kept as dict - not accessed with attributes)
    system_events: Dict[str, Any] = Field(
        default_factory=lambda: {
            "event_queue_size": 1000,
            "message_hub_max_subscribers": 100,
        },
        description="Event system configuration"
    )

    # Additional custom settings
    custom: Dict[str, Any] = Field(default_factory=dict, description="Custom settings")

    model_config = SettingsConfigDict(
        env_prefix="TRADING_BOT_",
        env_nested_delimiter="__",
        case_sensitive=False,
        validate_assignment=True,
        extra="allow",
        env_file=".env",
        env_file_encoding="utf-8",
    )