"""Pydantic models for configuration validation."""
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class LogLevel(str, Enum):
    """Log level enumeration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class BinanceConfig(BaseModel):
    """Binance API configuration."""
    api_key: str = Field(..., description="Binance API key")
    api_secret: str = Field(..., description="Binance API secret")
    testnet: bool = Field(default=False, description="Use testnet")
    base_url: Optional[str] = Field(None, description="Custom base URL")


class DiscordConfig(BaseModel):
    """Discord bot configuration."""
    bot_token: str = Field(..., description="Discord bot token")
    channel_id: int = Field(..., description="Target channel ID")
    enable_notifications: bool = Field(default=True, description="Enable Discord notifications")
    spam_protection_seconds: int = Field(default=60, description="Spam protection interval")


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


class TradingConfig(BaseModel):
    """Trading strategy configuration."""
    symbol: str = Field(default="BTCUSDT", description="Trading symbol")
    timeframe: str = Field(default="1h", description="Chart timeframe")
    max_position_size: float = Field(default=0.1, description="Max position size")
    leverage: int = Field(default=1, description="Trading leverage")
    
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


class SystemConfig(BaseModel):
    """System configuration."""
    environment: str = Field(default="development", description="Environment name")
    debug: bool = Field(default=False, description="Debug mode")
    async_workers: int = Field(default=4, description="Number of async workers")
    
    # Health check
    health_check_interval: int = Field(default=300, description="Health check interval (seconds)")
    memory_threshold_mb: int = Field(default=1024, description="Memory warning threshold")


class TradingBotConfig(BaseModel):
    """Main trading bot configuration."""
    system: SystemConfig
    binance: BinanceConfig
    discord: DiscordConfig
    logging: LoggingConfig
    trading: TradingConfig
    
    # Additional custom settings
    custom: Dict[str, Any] = Field(default_factory=dict, description="Custom settings")
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True
        extra = "allow"