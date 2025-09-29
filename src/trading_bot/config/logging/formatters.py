"""Custom log formatters for structured logging."""
import json
import logging
import traceback
from datetime import datetime
from typing import Dict, Any, Optional


class JSONFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.
    
    Outputs logs in JSON format with:
    - Timestamp
    - Log level
    - Logger name
    - Message
    - Context information
    - Exception details (if present)
    """
    
    def __init__(
        self,
        include_timestamp: bool = True,
        include_logger: bool = True,
        include_context: bool = True,
        timestamp_format: str = "%Y-%m-%d %H:%M:%S.%f",
    ):
        """
        Initialize JSON formatter.
        
        Args:
            include_timestamp: Include timestamp in output
            include_logger: Include logger name in output
            include_context: Include context information
            timestamp_format: Timestamp format string
        """
        super().__init__()
        self.include_timestamp = include_timestamp
        self.include_logger = include_logger
        self.include_context = include_context
        self.timestamp_format = timestamp_format
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.
        
        Args:
            record: Log record to format
        
        Returns:
            JSON formatted string
        """
        log_data: Dict[str, Any] = {
            "level": record.levelname,
            "message": record.getMessage(),
        }
        
        # Add timestamp
        if self.include_timestamp:
            log_data["timestamp"] = datetime.fromtimestamp(
                record.created
            ).strftime(self.timestamp_format)
        
        # Add logger name
        if self.include_logger:
            log_data["logger"] = record.name
        
        # Add context information
        if self.include_context:
            log_data.update({
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno,
                "thread": record.thread,
                "thread_name": record.threadName,
                "process": record.process,
            })
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }
        
        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in [
                "name", "msg", "args", "created", "filename", "funcName",
                "levelname", "levelno", "lineno", "module", "msecs",
                "message", "pathname", "process", "processName",
                "relativeCreated", "thread", "threadName", "exc_info",
                "exc_text", "stack_info"
            ]:
                log_data[key] = value
        
        return json.dumps(log_data, default=str)


class MaskingFormatter(logging.Formatter):
    """
    Formatter that wraps another formatter and applies masking.
    
    This allows combining masking with any other formatter.
    """
    
    def __init__(
        self,
        base_formatter: logging.Formatter,
        mask_patterns: Optional[Dict[str, str]] = None
    ):
        """
        Initialize masking formatter.
        
        Args:
            base_formatter: Base formatter to wrap
            mask_patterns: Pattern replacements for masking
        """
        super().__init__()
        self.base_formatter = base_formatter
        self.mask_patterns = mask_patterns or {}
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format record with base formatter, then apply masking.
        
        Args:
            record: Log record to format
        
        Returns:
            Formatted and masked string
        """
        # Format with base formatter
        formatted = self.base_formatter.format(record)
        
        # Apply masking patterns
        for pattern, replacement in self.mask_patterns.items():
            formatted = formatted.replace(pattern, replacement)
        
        return formatted


class TradeLogFormatter(JSONFormatter):
    """
    Specialized formatter for trade logs.
    
    Includes trade-specific fields:
    - Trade ID
    - Symbol
    - Side (buy/sell)
    - Price
    - Quantity
    - PnL
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format trade log record.
        
        Args:
            record: Log record to format
        
        Returns:
            JSON formatted trade log
        """
        # Get base JSON format
        formatted = super().format(record)
        log_data = json.loads(formatted)
        
        # Add trade-specific marker
        log_data["log_type"] = "trade"
        
        # Extract trade fields if present
        trade_fields = [
            "trade_id", "symbol", "side", "price", "quantity",
            "pnl", "fee", "order_type", "strategy"
        ]
        
        for field in trade_fields:
            if hasattr(record, field):
                log_data[field] = getattr(record, field)
        
        return json.dumps(log_data, default=str)


class PerformanceLogFormatter(JSONFormatter):
    """
    Specialized formatter for performance metrics logs.
    
    Includes performance-specific fields:
    - Metric name
    - Value
    - Unit
    - Timestamp
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format performance log record.
        
        Args:
            record: Log record to format
        
        Returns:
            JSON formatted performance log
        """
        # Get base JSON format
        formatted = super().format(record)
        log_data = json.loads(formatted)
        
        # Add performance-specific marker
        log_data["log_type"] = "performance"
        
        # Extract performance fields if present
        perf_fields = [
            "metric_name", "value", "unit", "target", "threshold",
            "cpu_percent", "memory_mb", "latency_ms"
        ]
        
        for field in perf_fields:
            if hasattr(record, field):
                log_data[field] = getattr(record, field)
        
        return json.dumps(log_data, default=str)