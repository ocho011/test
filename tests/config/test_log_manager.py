"""Tests for LogManager and logging functionality."""
import os
import json
import tempfile
import pytest
import logging
from pathlib import Path

from trading_bot.config import (
    LogManager,
    LoggingConfig,
    LogLevel,
    setup_logging,
    get_logger,
)


@pytest.fixture
def temp_log_dir():
    """Create temporary log directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    
    # Cleanup
    import shutil
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def logging_config(temp_log_dir):
    """Create logging configuration for testing."""
    return LoggingConfig(
        level=LogLevel.DEBUG,
        format="json",
        output_dir=temp_log_dir,
        max_bytes=1048576,
        backup_count=3,
        rotation_time="midnight",
        trade_log_enabled=True,
        trade_log_file="trades.log",
        performance_log_enabled=True,
        performance_log_file="performance.log",
        mask_sensitive_data=True,
        debug_unmask=False
    )


class TestLogManager:
    """Test cases for LogManager."""
    
    def test_singleton_pattern(self):
        """Test that LogManager follows singleton pattern."""
        manager1 = LogManager()
        manager2 = LogManager()
        assert manager1 is manager2
    
    def test_setup_logging(self, logging_config):
        """Test logging setup."""
        manager = LogManager()
        manager.setup(logging_config)
        
        # Check log directory created
        assert os.path.exists(logging_config.output_dir)
        
        # Check loggers created
        root_logger = manager.get_logger("root")
        assert root_logger is not None
    
    def test_get_logger(self, logging_config):
        """Test getting logger."""
        manager = LogManager()
        manager.setup(logging_config)
        
        logger = manager.get_logger("test_logger")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_logger"
    
    def test_log_trade(self, logging_config, temp_log_dir):
        """Test trade logging."""
        manager = LogManager()
        manager.setup(logging_config)
        
        manager.log_trade(
            message="Trade executed",
            trade_id="TRADE_001",
            symbol="BTCUSDT",
            side="buy",
            price=50000.0,
            quantity=0.1,
            pnl=100.0,
            fee=5.0
        )
        
        # Check trade log file created
        trade_log_path = Path(temp_log_dir) / "trades.log"
        assert trade_log_path.exists()
        
        # Read and verify log content
        with open(trade_log_path, 'r') as f:
            log_line = f.readline()
            log_data = json.loads(log_line)
            
            assert log_data["message"] == "Trade executed"
            assert log_data["trade_id"] == "TRADE_001"
            assert log_data["symbol"] == "BTCUSDT"
            assert log_data["price"] == 50000.0
    
    def test_log_performance(self, logging_config, temp_log_dir):
        """Test performance logging."""
        manager = LogManager()
        manager.setup(logging_config)
        
        manager.log_performance(
            message="CPU usage high",
            metric_name="cpu_usage",
            value=85.5,
            unit="%",
            threshold=80.0
        )
        
        # Check performance log file created
        perf_log_path = Path(temp_log_dir) / "performance.log"
        assert perf_log_path.exists()
        
        # Read and verify log content
        with open(perf_log_path, 'r') as f:
            log_line = f.readline()
            log_data = json.loads(log_line)
            
            assert log_data["metric_name"] == "cpu_usage"
            assert log_data["value"] == 85.5
            assert log_data["unit"] == "%"
    
    @pytest.mark.asyncio
    async def test_async_logging(self, logging_config):
        """Test asynchronous logging."""
        manager = LogManager()
        manager.setup(logging_config)
        
        await manager.log_async(
            "test_logger",
            logging.INFO,
            "Async test message"
        )
        
        # Should complete without errors
        assert True
    
    def test_global_functions(self, logging_config):
        """Test global convenience functions."""
        setup_logging(logging_config)
        
        logger = get_logger("test")
        assert isinstance(logger, logging.Logger)


class TestSensitiveDataFilter:
    """Test sensitive data masking."""
    
    def test_mask_api_key(self, logging_config):
        """Test API key masking."""
        from trading_bot.config.logging import SensitiveDataFilter
        
        filter_obj = SensitiveDataFilter(mask_enabled=True, preserve_length=True)
        
        # Create log record with API key
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="API key: test_api_key_1234567890",
            args=(),
            exc_info=None
        )
        
        filter_obj.filter(record)
        
        # Check that key is masked
        assert "1234567890" not in str(record.msg)
        assert "****" in str(record.msg)
    
    def test_mask_email(self, logging_config):
        """Test email masking."""
        from trading_bot.config.logging import SensitiveDataFilter
        
        filter_obj = SensitiveDataFilter(mask_enabled=True)
        
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="User email: test@example.com",
            args=(),
            exc_info=None
        )
        
        filter_obj.filter(record)
        
        # Check that email is masked
        assert "test@example.com" not in str(record.msg)
    
    def test_mask_disabled(self):
        """Test masking can be disabled."""
        from trading_bot.config.logging import SensitiveDataFilter
        
        filter_obj = SensitiveDataFilter(mask_enabled=False)
        
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="API key: test_api_key_1234567890",
            args=(),
            exc_info=None
        )
        
        filter_obj.filter(record)
        
        # Key should NOT be masked
        assert "test_api_key_1234567890" in str(record.msg)


class TestJSONFormatter:
    """Test JSON formatter."""
    
    def test_json_format(self):
        """Test JSON log formatting."""
        from trading_bot.config.logging import JSONFormatter
        
        formatter = JSONFormatter()
        
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="/path/to/file.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        formatted = formatter.format(record)
        log_data = json.loads(formatted)
        
        assert log_data["level"] == "INFO"
        assert log_data["message"] == "Test message"
        assert log_data["logger"] == "test_logger"
        assert "timestamp" in log_data
    
    def test_json_format_with_exception(self):
        """Test JSON formatting with exception."""
        from trading_bot.config.logging import JSONFormatter
        
        formatter = JSONFormatter()
        
        try:
            raise ValueError("Test error")
        except ValueError:
            import sys
            exc_info = sys.exc_info()
        
        record = logging.LogRecord(
            name="test_logger",
            level=logging.ERROR,
            pathname="/path/to/file.py",
            lineno=42,
            msg="Error occurred",
            args=(),
            exc_info=exc_info
        )
        
        formatted = formatter.format(record)
        log_data = json.loads(formatted)
        
        assert "exception" in log_data
        assert log_data["exception"]["type"] == "ValueError"
        assert "Test error" in log_data["exception"]["message"]


class TestLogRotation:
    """Test log rotation and compression."""
    
    def test_rotation_handler_creation(self, temp_log_dir):
        """Test creating rotation handler."""
        from trading_bot.config.logging import RotatingFileHandlerWithCompression
        
        log_file = os.path.join(temp_log_dir, "test.log")
        handler = RotatingFileHandlerWithCompression(
            filename=log_file,
            maxBytes=1024,
            backupCount=3,
            compress_old_logs=True
        )
        
        assert handler.maxBytes == 1024
        assert handler.backupCount == 3
        assert handler.compress_old_logs is True
        
        handler.close()
    
    def test_timed_rotation_handler_creation(self, temp_log_dir):
        """Test creating timed rotation handler."""
        from trading_bot.config.logging import TimedRotatingFileHandlerWithCompression
        
        log_file = os.path.join(temp_log_dir, "test.log")
        handler = TimedRotatingFileHandlerWithCompression(
            filename=log_file,
            when='midnight',
            backupCount=5,
            compress_old_logs=True
        )
        
        assert handler.backupCount == 5
        assert handler.compress_old_logs is True
        
        handler.close()