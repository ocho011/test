"""Custom log handlers with rotation and compression."""
import gzip
import os
import shutil
from pathlib import Path
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from typing import Optional, Any


class RotatingFileHandlerWithCompression(RotatingFileHandler):
    """
    Rotating file handler that compresses rotated files.
    
    Features:
    - Size-based rotation
    - Automatic compression of old log files
    - Configurable backup count
    - Disk space monitoring
    """
    
    def __init__(
        self,
        filename: str,
        mode: str = 'a',
        maxBytes: int = 0,
        backupCount: int = 0,
        encoding: Optional[str] = None,
        delay: bool = False,
        compress_old_logs: bool = True
    ):
        """
        Initialize rotating handler with compression.
        
        Args:
            filename: Log file path
            mode: File open mode
            maxBytes: Max file size before rotation
            backupCount: Number of backup files to keep
            encoding: File encoding
            delay: Delay file opening until first emit
            compress_old_logs: Enable compression of rotated logs
        """
        super().__init__(filename, mode, maxBytes, backupCount, encoding, delay)
        self.compress_old_logs = compress_old_logs
    
    def doRollover(self) -> None:
        """
        Perform rollover and compress old log file.
        """
        if self.stream:
            self.stream.close()
            self.stream = None
        
        # Rotate files
        if self.backupCount > 0:
            for i in range(self.backupCount - 1, 0, -1):
                sfn = self.rotation_filename(f"{self.baseFilename}.{i}")
                dfn = self.rotation_filename(f"{self.baseFilename}.{i + 1}")
                
                if os.path.exists(sfn):
                    if os.path.exists(dfn):
                        os.remove(dfn)
                    os.rename(sfn, dfn)
            
            dfn = self.rotation_filename(f"{self.baseFilename}.1")
            if os.path.exists(dfn):
                os.remove(dfn)
            
            # Rename current log to .1
            self.rotate(self.baseFilename, dfn)
            
            # Compress the rotated file
            if self.compress_old_logs:
                self._compress_file(dfn)
        
        # Open new log file
        if not self.delay:
            self.stream = self._open()
    
    def _compress_file(self, filename: str) -> None:
        """
        Compress a log file using gzip.
        
        Args:
            filename: File to compress
        """
        try:
            compressed_filename = f"{filename}.gz"
            
            with open(filename, 'rb') as f_in:
                with gzip.open(compressed_filename, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Remove original file after successful compression
            os.remove(filename)
            
        except Exception as e:
            # If compression fails, keep the original file
            print(f"Failed to compress {filename}: {e}")


class TimedRotatingFileHandlerWithCompression(TimedRotatingFileHandler):
    """
    Time-based rotating file handler with compression.
    
    Features:
    - Time-based rotation (hourly, daily, weekly, etc.)
    - Automatic compression of old log files
    - Configurable backup count
    """
    
    def __init__(
        self,
        filename: str,
        when: str = 'midnight',
        interval: int = 1,
        backupCount: int = 0,
        encoding: Optional[str] = None,
        delay: bool = False,
        utc: bool = False,
        atTime: Optional[Any] = None,
        compress_old_logs: bool = True
    ):
        """
        Initialize timed rotating handler with compression.
        
        Args:
            filename: Log file path
            when: Rotation time unit ('S', 'M', 'H', 'D', 'W0'-'W6', 'midnight')
            interval: Rotation interval
            backupCount: Number of backup files to keep
            encoding: File encoding
            delay: Delay file opening until first emit
            utc: Use UTC time
            atTime: Time of day for rotation
            compress_old_logs: Enable compression of rotated logs
        """
        super().__init__(
            filename, when, interval, backupCount,
            encoding, delay, utc, atTime
        )
        self.compress_old_logs = compress_old_logs
    
    def doRollover(self) -> None:
        """
        Perform time-based rollover and compress old log file.
        """
        # Perform standard rollover
        super().doRollover()
        
        # Find and compress the rotated file
        if self.compress_old_logs and self.backupCount > 0:
            # Get list of log files
            dir_name, base_name = os.path.split(self.baseFilename)
            file_names = os.listdir(dir_name) if dir_name else os.listdir('.')
            
            # Find uncompressed backup files
            prefix = base_name + "."
            for filename in file_names:
                if filename.startswith(prefix) and not filename.endswith('.gz'):
                    full_path = os.path.join(dir_name, filename) if dir_name else filename
                    if full_path != self.baseFilename:
                        self._compress_file(full_path)
    
    def _compress_file(self, filename: str) -> None:
        """
        Compress a log file using gzip.
        
        Args:
            filename: File to compress
        """
        try:
            compressed_filename = f"{filename}.gz"
            
            with open(filename, 'rb') as f_in:
                with gzip.open(compressed_filename, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Remove original file after successful compression
            os.remove(filename)
            
        except Exception as e:
            # If compression fails, keep the original file
            print(f"Failed to compress {filename}: {e}")


class DiskSpaceHandler(RotatingFileHandler):
    """
    Handler that monitors disk space and manages log files accordingly.
    
    Features:
    - Monitors available disk space
    - Automatically removes oldest logs when space is low
    - Configurable space threshold
    """
    
    def __init__(
        self,
        filename: str,
        mode: str = 'a',
        maxBytes: int = 0,
        backupCount: int = 0,
        encoding: Optional[str] = None,
        delay: bool = False,
        min_free_space_mb: int = 100
    ):
        """
        Initialize disk space aware handler.
        
        Args:
            filename: Log file path
            mode: File open mode
            maxBytes: Max file size before rotation
            backupCount: Number of backup files to keep
            encoding: File encoding
            delay: Delay file opening
            min_free_space_mb: Minimum free space in MB
        """
        super().__init__(filename, mode, maxBytes, backupCount, encoding, delay)
        self.min_free_space_mb = min_free_space_mb
    
    def shouldRollover(self, record: Any) -> bool:
        """
        Check if rollover should occur based on size or disk space.
        
        Args:
            record: Log record
        
        Returns:
            True if should rollover
        """
        # Check standard size-based rollover
        should_rollover = super().shouldRollover(record)
        
        # Check disk space
        if not should_rollover:
            free_space_mb = self._get_free_space_mb()
            if free_space_mb < self.min_free_space_mb:
                self._cleanup_old_logs()
        
        return should_rollover
    
    def _get_free_space_mb(self) -> int:
        """
        Get available disk space in MB.
        
        Returns:
            Free space in MB
        """
        try:
            stat = os.statvfs(os.path.dirname(self.baseFilename) or '.')
            free_bytes = stat.f_bavail * stat.f_frsize
            return free_bytes // (1024 * 1024)
        except Exception:
            return float('inf')  # If we can't check, assume enough space
    
    def _cleanup_old_logs(self) -> None:
        """Remove oldest log files to free up space."""
        dir_name = os.path.dirname(self.baseFilename) or '.'
        base_name = os.path.basename(self.baseFilename)
        
        # Get all related log files
        log_files = []
        for filename in os.listdir(dir_name):
            if filename.startswith(base_name):
                full_path = os.path.join(dir_name, filename)
                if os.path.isfile(full_path):
                    log_files.append((full_path, os.path.getmtime(full_path)))
        
        # Sort by modification time (oldest first)
        log_files.sort(key=lambda x: x[1])
        
        # Remove oldest files (keep current file)
        for file_path, _ in log_files[:-1]:
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Failed to remove old log {file_path}: {e}")