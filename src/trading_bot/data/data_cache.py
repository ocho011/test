"""
Historical data caching system with file-based storage.

This module provides efficient caching of historical market data with
compression, data integrity validation, and automatic cache management.
"""

import asyncio
import gzip
import json
import logging
import os
import time
import hashlib
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor

import pandas as pd


@dataclass
class CacheEntry:
    """Cache entry metadata."""
    symbol: str
    interval: str
    start_time: int
    end_time: int
    data_hash: str
    file_path: str
    created_at: float
    last_accessed: float
    size_bytes: int
    compressed: bool = True


@dataclass
class CacheStats:
    """Cache statistics."""
    total_entries: int
    total_size_bytes: int
    hit_count: int
    miss_count: int
    eviction_count: int
    oldest_entry: Optional[float]
    newest_entry: Optional[float]


class DataCache:
    """
    File-based historical data cache with compression and integrity validation.

    Provides efficient storage and retrieval of historical kline data
    with automatic cleanup and data validation.
    """

    def __init__(
        self,
        cache_dir: str = "data/cache",
        max_cache_size_gb: float = 5.0,
        max_age_days: int = 30,
        compression_enabled: bool = True
    ):
        """
        Initialize data cache.

        Args:
            cache_dir: Directory for cache storage
            max_cache_size_gb: Maximum cache size in GB
            max_age_days: Maximum age for cache entries
            compression_enabled: Enable gzip compression
        """
        self.cache_dir = Path(cache_dir)
        self.max_cache_size_bytes = int(max_cache_size_gb * 1024 * 1024 * 1024)
        self.max_age_seconds = max_age_days * 24 * 3600
        self.compression_enabled = compression_enabled

        self.logger = logging.getLogger(__name__)

        # Cache management
        self._metadata_file = self.cache_dir / "cache_metadata.json"
        self._entries: Dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()
        self._executor = ThreadPoolExecutor(max_workers=4)

        # Statistics
        self._stats = CacheStats(
            total_entries=0,
            total_size_bytes=0,
            hit_count=0,
            miss_count=0,
            eviction_count=0,
            oldest_entry=None,
            newest_entry=None
        )

        # Initialize cache directory
        self._initialize_cache()

    def _initialize_cache(self) -> None:
        """Initialize cache directory and load metadata."""
        try:
            # Create cache directory if it doesn't exist
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            # Create subdirectories for different data types
            (self.cache_dir / "klines").mkdir(exist_ok=True)
            (self.cache_dir / "tickers").mkdir(exist_ok=True)

            # Load existing metadata
            self._load_metadata()

            self.logger.info(f"Cache initialized: {len(self._entries)} entries, {self._stats.total_size_bytes / 1024 / 1024:.1f} MB")

        except Exception as e:
            self.logger.error(f"Failed to initialize cache: {e}")
            raise

    def _load_metadata(self) -> None:
        """Load cache metadata from file."""
        if not self._metadata_file.exists():
            return

        try:
            with open(self._metadata_file, 'r') as f:
                metadata = json.load(f)

            # Load cache entries
            for key, entry_data in metadata.get('entries', {}).items():
                entry = CacheEntry(**entry_data)

                # Verify file still exists
                if Path(entry.file_path).exists():
                    self._entries[key] = entry
                else:
                    self.logger.warning(f"Cache file missing: {entry.file_path}")

            # Update statistics
            self._update_stats()

        except Exception as e:
            self.logger.error(f"Failed to load cache metadata: {e}")

    def _save_metadata(self) -> None:
        """Save cache metadata to file."""
        try:
            metadata = {
                'entries': {key: asdict(entry) for key, entry in self._entries.items()},
                'last_updated': time.time()
            }

            with open(self._metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

        except Exception as e:
            self.logger.error(f"Failed to save cache metadata: {e}")

    def _update_stats(self) -> None:
        """Update cache statistics."""
        self._stats.total_entries = len(self._entries)
        self._stats.total_size_bytes = sum(entry.size_bytes for entry in self._entries.values())

        if self._entries:
            created_times = [entry.created_at for entry in self._entries.values()]
            self._stats.oldest_entry = min(created_times)
            self._stats.newest_entry = max(created_times)
        else:
            self._stats.oldest_entry = None
            self._stats.newest_entry = None

    def _generate_cache_key(self, symbol: str, interval: str, start_time: int, end_time: int) -> str:
        """Generate unique cache key for data."""
        key_data = f"{symbol}_{interval}_{start_time}_{end_time}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _calculate_data_hash(self, data: List[List[Any]]) -> str:
        """Calculate hash of data for integrity verification."""
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()

    async def get_kline_data(
        self,
        symbol: str,
        interval: str,
        start_time: int,
        end_time: int
    ) -> Optional[List[List[Any]]]:
        """
        Get cached kline data.

        Args:
            symbol: Trading symbol
            interval: Time interval
            start_time: Start timestamp
            end_time: End timestamp

        Returns:
            Cached kline data or None if not found
        """
        cache_key = self._generate_cache_key(symbol, interval, start_time, end_time)

        async with self._lock:
            entry = self._entries.get(cache_key)

            if not entry:
                self._stats.miss_count += 1
                return None

            # Check if entry is expired
            if time.time() - entry.created_at > self.max_age_seconds:
                self.logger.debug(f"Cache entry expired: {cache_key}")
                await self._remove_entry(cache_key)
                self._stats.miss_count += 1
                return None

            # Update access time
            entry.last_accessed = time.time()

        try:
            # Load data from file
            data = await self._load_data_file(entry.file_path, entry.compressed)

            # Verify data integrity
            if entry.data_hash != self._calculate_data_hash(data):
                self.logger.error(f"Data integrity check failed for {cache_key}")
                await self._remove_entry(cache_key)
                self._stats.miss_count += 1
                return None

            self._stats.hit_count += 1
            return data

        except Exception as e:
            self.logger.error(f"Failed to load cached data: {e}")
            await self._remove_entry(cache_key)
            self._stats.miss_count += 1
            return None

    async def store_kline_data(
        self,
        symbol: str,
        interval: str,
        start_time: int,
        end_time: int,
        data: List[List[Any]]
    ) -> bool:
        """
        Store kline data in cache.

        Args:
            symbol: Trading symbol
            interval: Time interval
            start_time: Start timestamp
            end_time: End timestamp
            data: Kline data to cache

        Returns:
            True if stored successfully
        """
        if not data:
            return False

        cache_key = self._generate_cache_key(symbol, interval, start_time, end_time)
        data_hash = self._calculate_data_hash(data)

        # Generate file path
        filename = f"{symbol}_{interval}_{start_time}_{end_time}.json"
        if self.compression_enabled:
            filename += ".gz"

        file_path = self.cache_dir / "klines" / filename

        try:
            # Save data to file
            await self._save_data_file(str(file_path), data, self.compression_enabled)

            # Get file size
            file_size = os.path.getsize(file_path)

            # Create cache entry
            entry = CacheEntry(
                symbol=symbol,
                interval=interval,
                start_time=start_time,
                end_time=end_time,
                data_hash=data_hash,
                file_path=str(file_path),
                created_at=time.time(),
                last_accessed=time.time(),
                size_bytes=file_size,
                compressed=self.compression_enabled
            )

            async with self._lock:
                self._entries[cache_key] = entry
                self._update_stats()

                # Check cache size and evict if necessary
                await self._enforce_cache_limits()

            # Save metadata
            self._save_metadata()

            self.logger.debug(f"Cached data: {cache_key} ({file_size} bytes)")
            return True

        except Exception as e:
            self.logger.error(f"Failed to store cache data: {e}")
            return False

    async def _save_data_file(self, file_path: str, data: List[List[Any]], compress: bool) -> None:
        """Save data to file with optional compression."""
        def _save_sync():
            data_json = json.dumps(data)

            if compress:
                with gzip.open(file_path, 'wt', encoding='utf-8') as f:
                    f.write(data_json)
            else:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(data_json)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self._executor, _save_sync)

    async def _load_data_file(self, file_path: str, compressed: bool) -> List[List[Any]]:
        """Load data from file with optional decompression."""
        def _load_sync():
            if compressed:
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    return json.load(f)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, _load_sync)

    async def _remove_entry(self, cache_key: str) -> None:
        """Remove cache entry and associated file."""
        entry = self._entries.get(cache_key)
        if not entry:
            return

        try:
            # Remove file
            file_path = Path(entry.file_path)
            if file_path.exists():
                file_path.unlink()

            # Remove from metadata
            del self._entries[cache_key]
            self._update_stats()

        except Exception as e:
            self.logger.error(f"Failed to remove cache entry {cache_key}: {e}")

    async def _enforce_cache_limits(self) -> None:
        """Enforce cache size and age limits."""
        current_time = time.time()

        # Remove expired entries
        expired_keys = []
        for key, entry in self._entries.items():
            if current_time - entry.created_at > self.max_age_seconds:
                expired_keys.append(key)

        for key in expired_keys:
            await self._remove_entry(key)
            self._stats.eviction_count += 1

        # Remove oldest entries if cache is too large
        while self._stats.total_size_bytes > self.max_cache_size_bytes and self._entries:
            # Find oldest entry by last access time
            oldest_key = min(
                self._entries.keys(),
                key=lambda k: self._entries[k].last_accessed
            )

            await self._remove_entry(oldest_key)
            self._stats.eviction_count += 1

    async def cleanup_expired(self) -> int:
        """
        Clean up expired cache entries.

        Returns:
            Number of entries removed
        """
        current_time = time.time()
        expired_keys = []

        async with self._lock:
            for key, entry in self._entries.items():
                if current_time - entry.created_at > self.max_age_seconds:
                    expired_keys.append(key)

        removed_count = 0
        for key in expired_keys:
            await self._remove_entry(key)
            removed_count += 1

        if removed_count > 0:
            self._save_metadata()
            self.logger.info(f"Cleaned up {removed_count} expired cache entries")

        return removed_count

    async def clear_symbol(self, symbol: str) -> int:
        """
        Clear all cache entries for a specific symbol.

        Args:
            symbol: Trading symbol to clear

        Returns:
            Number of entries removed
        """
        to_remove = []

        async with self._lock:
            for key, entry in self._entries.items():
                if entry.symbol.upper() == symbol.upper():
                    to_remove.append(key)

        removed_count = 0
        for key in to_remove:
            await self._remove_entry(key)
            removed_count += 1

        if removed_count > 0:
            self._save_metadata()
            self.logger.info(f"Cleared {removed_count} cache entries for {symbol}")

        return removed_count

    async def clear_all(self) -> int:
        """
        Clear all cache entries.

        Returns:
            Number of entries removed
        """
        removed_count = len(self._entries)

        # Remove all files
        for entry in self._entries.values():
            try:
                file_path = Path(entry.file_path)
                if file_path.exists():
                    file_path.unlink()
            except Exception as e:
                self.logger.error(f"Failed to remove cache file {entry.file_path}: {e}")

        # Clear metadata
        self._entries.clear()
        self._update_stats()
        self._save_metadata()

        self.logger.info(f"Cleared all cache entries ({removed_count} entries)")
        return removed_count

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats

    def get_cache_info(self) -> Dict[str, Any]:
        """Get detailed cache information."""
        hit_rate = (
            self._stats.hit_count / (self._stats.hit_count + self._stats.miss_count)
            if (self._stats.hit_count + self._stats.miss_count) > 0
            else 0.0
        )

        return {
            'cache_dir': str(self.cache_dir),
            'max_size_gb': self.max_cache_size_bytes / (1024 ** 3),
            'current_size_gb': self._stats.total_size_bytes / (1024 ** 3),
            'max_age_days': self.max_age_seconds / (24 * 3600),
            'compression_enabled': self.compression_enabled,
            'total_entries': self._stats.total_entries,
            'hit_count': self._stats.hit_count,
            'miss_count': self._stats.miss_count,
            'hit_rate': hit_rate,
            'eviction_count': self._stats.eviction_count,
            'oldest_entry_age_hours': (
                (time.time() - self._stats.oldest_entry) / 3600
                if self._stats.oldest_entry
                else None
            ),
            'newest_entry_age_hours': (
                (time.time() - self._stats.newest_entry) / 3600
                if self._stats.newest_entry
                else None
            )
        }

    def list_cached_symbols(self) -> Dict[str, Dict[str, List[str]]]:
        """
        List all cached symbols and their intervals.

        Returns:
            Dictionary mapping symbols to their cached intervals
        """
        symbols = {}

        for entry in self._entries.values():
            symbol = entry.symbol
            interval = entry.interval

            if symbol not in symbols:
                symbols[symbol] = {}
            if interval not in symbols[symbol]:
                symbols[symbol][interval] = []

            # Add time range
            start_dt = datetime.fromtimestamp(entry.start_time / 1000)
            end_dt = datetime.fromtimestamp(entry.end_time / 1000)
            time_range = f"{start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}"
            symbols[symbol][interval].append(time_range)

        return symbols

    async def optimize_cache(self) -> Dict[str, int]:
        """
        Optimize cache by removing duplicates and consolidating entries.

        Returns:
            Dictionary with optimization results
        """
        # This could be implemented to merge overlapping time ranges,
        # remove duplicate data, and optimize file storage
        # For now, just cleanup expired entries

        expired_removed = await self.cleanup_expired()

        return {
            'expired_removed': expired_removed,
            'total_entries': self._stats.total_entries,
            'total_size_mb': self._stats.total_size_bytes / (1024 * 1024)
        }