"""Centralized logging manager for the trading bot."""
import logging
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor

from ..models import LoggingConfig, LogLevel
from .formatters import (
    JSONFormatter,
    TradeLogFormatter,
    PerformanceLogFormatter
)
from .handlers import (
    RotatingFileHandlerWithCompression,
    TimedRotatingFileHandlerWithCompression,
    DiskSpaceHandler
)
from .filters import SensitiveDataFilter, DebugUnmaskFilter


class LogManager:
    """
    Centralized logging manager with structured logging support.
    
    Features:
    - JSON structured logging
    - Async logging for performance
    - Automatic log rotation and compression
    - Separate trade and performance logs
    - Sensitive data masking
    - Configurable log levels
    """
    
    _instance: Optional['LogManager'] = None
    _initialized: bool = False
    
    def __new__(cls):
        """Singleton pattern to ensure single log manager instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize log manager."""
        if not LogManager._initialized:
            self._loggers: Dict[str, logging.Logger] = {}
            self._executor: Optional[ThreadPoolExecutor] = None
            self._config: Optional[LoggingConfig] = None
            LogManager._initialized = True
    
    def setup(self, config: LoggingConfig) -> None:
        """
        Setup logging system with configuration.
        
        Args:
            config: Logging configuration
        """
        self._config = config
        
        # Create log directory
        log_dir = Path(config.output_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup root logger
        self._setup_root_logger()
        
        # Setup trade logger
        if config.trade_log_enabled:
            self._setup_trade_logger()
        
        # Setup performance logger
        if config.performance_log_enabled:
            self._setup_performance_logger()
        
        # Setup async executor
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="log_")
    
    def _setup_root_logger(self) -> None:
        """Setup root logger with standard configuration."""
        if self._config is None:
            raise RuntimeError("Config not set. Call setup() first.")
        
        # Get root logger
        logger = logging.getLogger()
        logger.setLevel(self._get_log_level(self._config.level))
        
        # Remove existing handlers
        logger.handlers.clear()
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self._get_log_level(self._config.level))
        
        # Create file handler with rotation
        log_file = Path(self._config.output_dir) / "bot.log"
        file_handler = RotatingFileHandlerWithCompression(
            filename=str(log_file),
            maxBytes=self._config.max_bytes,
            backupCount=self._config.backup_count,
            compress_old_logs=True
        )
        file_handler.setLevel(self._get_log_level(self._config.level))
        
        # Setup formatters
        if self._config.format == "json":
            formatter = JSONFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # Add sensitive data filter
        if self._config.mask_sensitive_data:
            sensitive_filter = SensitiveDataFilter(
                mask_enabled=True,
                preserve_length=True
            )
            console_handler.addFilter(sensitive_filter)
            file_handler.addFilter(sensitive_filter)
            
            # Debug unmask filter
            if self._config.debug_unmask:
                debug_filter = DebugUnmaskFilter(debug_mode=True)
                console_handler.addFilter(debug_filter)
        
        # Add handlers
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        self._loggers["root"] = logger
    
    def _setup_trade_logger(self) -> None:
        """Setup specialized logger for trade events."""
        if self._config is None:
            raise RuntimeError("Config not set. Call setup() first.")
        
        # Create trade logger
        logger = logging.getLogger("trading_bot.trades")
        logger.setLevel(logging.INFO)
        logger.propagate = False  # Don't propagate to root logger
        
        # Create file handler
        log_file = Path(self._config.output_dir) / self._config.trade_log_file
        file_handler = TimedRotatingFileHandlerWithCompression(
            filename=str(log_file),
            when=self._config.rotation_time,
            backupCount=self._config.backup_count,
            compress_old_logs=True
        )
        
        # Use trade-specific formatter
        formatter = TradeLogFormatter()
        file_handler.setFormatter(formatter)
        
        # Add sensitive data filter
        if self._config.mask_sensitive_data:
            sensitive_filter = SensitiveDataFilter(mask_enabled=True)
            file_handler.addFilter(sensitive_filter)
        
        logger.addHandler(file_handler)
        self._loggers["trades"] = logger
    
    def _setup_performance_logger(self) -> None:
        """Setup specialized logger for performance metrics."""
        if self._config is None:
            raise RuntimeError("Config not set. Call setup() first.")
        
        # Create performance logger
        logger = logging.getLogger("trading_bot.performance")
        logger.setLevel(logging.INFO)
        logger.propagate = False  # Don't propagate to root logger
        
        # Create file handler
        log_file = Path(self._config.output_dir) / self._config.performance_log_file
        file_handler = TimedRotatingFileHandlerWithCompression(
            filename=str(log_file),
            when=self._config.rotation_time,
            backupCount=self._config.backup_count,
            compress_old_logs=True
        )
        
        # Use performance-specific formatter
        formatter = PerformanceLogFormatter()
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        self._loggers["performance"] = logger
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        Get logger by name.
        
        Args:
            name: Logger name
        
        Returns:
            Logger instance
        """
        if name in self._loggers:
            return self._loggers[name]
        
        # Create new logger
        logger = logging.getLogger(name)
        self._loggers[name] = logger
        return logger
    
    def log_trade(
        self,
        message: str,
        trade_id: str,
        symbol: str,
        side: str,
        price: float,
        quantity: float,
        pnl: Optional[float] = None,
        **kwargs: Any
    ) -> None:
        """
        Log trade event with structured data.
        
        Args:
            message: Log message
            trade_id: Trade identifier
            symbol: Trading symbol
            side: Trade side (buy/sell)
            price: Trade price
            quantity: Trade quantity
            pnl: Profit/loss
            **kwargs: Additional trade data
        """
        if "trades" not in self._loggers:
            return
        
        logger = self._loggers["trades"]
        
        # Create log record with extra fields
        extra = {
            "trade_id": trade_id,
            "symbol": symbol,
            "side": side,
            "price": price,
            "quantity": quantity,
            "pnl": pnl,
            **kwargs
        }
        
        logger.info(message, extra=extra)
    
    def log_performance(
        self,
        message: str,
        metric_name: str,
        value: float,
        unit: str = "",
        **kwargs: Any
    ) -> None:
        """
        Log performance metric.
        
        Args:
            message: Log message
            metric_name: Name of metric
            value: Metric value
            unit: Unit of measurement
            **kwargs: Additional metric data
        """
        if "performance" not in self._loggers:
            return
        
        logger = self._loggers["performance"]
        
        # Create log record with extra fields
        extra = {
            "metric_name": metric_name,
            "value": value,
            "unit": unit,
            **kwargs
        }
        
        logger.info(message, extra=extra)
    
    async def log_async(
        self,
        logger_name: str,
        level: int,
        message: str,
        *args: Any,
        **kwargs: Any
    ) -> None:
        """
        Log message asynchronously.
        
        Args:
            logger_name: Name of logger
            level: Log level
            message: Log message
            *args: Message args
            **kwargs: Extra fields
        """
        if self._executor is None:
            # Fallback to sync logging
            logger = self.get_logger(logger_name)
            logger.log(level, message, *args, **kwargs)
            return
        
        # Execute logging in thread pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self._executor,
            self._log_sync,
            logger_name,
            level,
            message,
            args,
            kwargs
        )
    
    def _log_sync(
        self,
        logger_name: str,
        level: int,
        message: str,
        args: tuple,
        kwargs: dict
    ) -> None:
        """
        Synchronous logging helper for async logging.
        
        Args:
            logger_name: Logger name
            level: Log level
            message: Log message
            args: Message args
            kwargs: Extra fields
        """
        logger = self.get_logger(logger_name)
        logger.log(level, message, *args, **kwargs)
    
    def _get_log_level(self, level: LogLevel) -> int:
        """
        Convert LogLevel enum to logging level.
        
        Args:
            level: LogLevel enum
        
        Returns:
            Logging level constant
        """
        level_map = {
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.INFO: logging.INFO,
            LogLevel.WARNING: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
            LogLevel.CRITICAL: logging.CRITICAL,
        }
        return level_map.get(level, logging.INFO)
    
    def shutdown(self) -> None:
        """Shutdown logging system and cleanup resources."""
        # Shutdown executor
        if self._executor:
            self._executor.shutdown(wait=True)
        
        # Close all handlers
        for logger in self._loggers.values():
            for handler in logger.handlers:
                handler.close()
        
        logging.shutdown()


# Global log manager instance
log_manager = LogManager()


def get_logger(name: str = "trading_bot") -> logging.Logger:
    """
    Get logger instance.
    
    Args:
        name: Logger name
    
    Returns:
        Logger instance
    """
    return log_manager.get_logger(name)


def setup_logging(config: LoggingConfig) -> None:
    """
    Setup logging system.
    
    Args:
        config: Logging configuration
    """
    log_manager.setup(config)