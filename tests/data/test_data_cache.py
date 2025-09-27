"""
Tests for DataCache file-based caching system.

Test cases cover cache storage, retrieval, compression,
data integrity, cache management, and performance optimization.
"""

import pytest
import asyncio
import json
import gzip
import time
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

from trading_bot.data.data_cache import DataCache, CacheEntry, CacheStats


class TestDataCache:
    """Test suite for DataCache file-based caching."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def cache(self, temp_cache_dir):
        """Create DataCache instance for testing."""
        return DataCache(
            cache_dir=temp_cache_dir,
            max_cache_size_gb=0.1,  # 100MB for testing
            max_age_days=1,
            compression_enabled=True
        )

    @pytest.fixture
    def sample_kline_data(self):
        """Sample kline data for testing."""
        return [
            ['1640995200000', '50000.00', '50100.00', '49900.00', '50050.00', '100.0'],
            ['1640995500000', '50050.00', '50150.00', '49950.00', '50100.00', '95.5'],
            ['1640995800000', '50100.00', '50200.00', '50000.00', '50150.00', '110.2']
        ]

    @pytest.mark.asyncio
    async def test_initialization(self, temp_cache_dir):
        """Test cache initialization and directory structure."""
        cache = DataCache(cache_dir=temp_cache_dir)

        # Check directory structure
        cache_path = Path(temp_cache_dir)
        assert cache_path.exists()
        assert (cache_path / "klines").exists()
        assert (cache_path / "tickers").exists()

        # Check configuration
        assert cache.cache_dir == cache_path
        assert cache.compression_enabled is True
        assert cache.max_cache_size_bytes > 0

    @pytest.mark.asyncio
    async def test_store_kline_data(self, cache, sample_kline_data):
        """Test storing kline data in cache."""
        symbol = "BTCUSDT"
        interval = "5m"
        start_time = 1640995200000
        end_time = 1640995800000

        # Store data
        success = await cache.store_kline_data(
            symbol, interval, start_time, end_time, sample_kline_data
        )

        assert success is True

        # Verify cache entry was created
        cache_key = cache._generate_cache_key(symbol, interval, start_time, end_time)
        assert cache_key in cache._entries

        entry = cache._entries[cache_key]
        assert entry.symbol == symbol
        assert entry.interval == interval
        assert entry.start_time == start_time
        assert entry.end_time == end_time
        assert entry.compressed is True

        # Verify file was created
        file_path = Path(entry.file_path)
        assert file_path.exists()
        assert file_path.suffix == ".gz"

    @pytest.mark.asyncio
    async def test_get_kline_data(self, cache, sample_kline_data):
        """Test retrieving kline data from cache."""
        symbol = "BTCUSDT"
        interval = "5m"
        start_time = 1640995200000
        end_time = 1640995800000

        # Store data first
        await cache.store_kline_data(
            symbol, interval, start_time, end_time, sample_kline_data
        )

        # Retrieve data
        retrieved_data = await cache.get_kline_data(
            symbol, interval, start_time, end_time
        )

        assert retrieved_data is not None
        assert retrieved_data == sample_kline_data

        # Check stats
        stats = cache.get_stats()
        assert stats.hit_count == 1
        assert stats.miss_count == 0

    @pytest.mark.asyncio
    async def test_cache_miss(self, cache):
        """Test cache miss behavior."""
        # Try to get non-existent data
        data = await cache.get_kline_data("ETHUSDT", "15m", 1640995200000, 1640995800000)

        assert data is None

        # Check stats
        stats = cache.get_stats()
        assert stats.miss_count == 1
        assert stats.hit_count == 0

    @pytest.mark.asyncio
    async def test_data_integrity_validation(self, cache, sample_kline_data):
        """Test data integrity validation."""
        symbol = "BTCUSDT"
        interval = "5m"
        start_time = 1640995200000
        end_time = 1640995800000

        # Store data
        await cache.store_kline_data(
            symbol, interval, start_time, end_time, sample_kline_data
        )

        # Corrupt the cached file
        cache_key = cache._generate_cache_key(symbol, interval, start_time, end_time)
        entry = cache._entries[cache_key]

        # Write corrupted data
        with gzip.open(entry.file_path, 'wt') as f:
            f.write('{"corrupted": "data"}')

        # Try to retrieve - should detect corruption and return None
        data = await cache.get_kline_data(symbol, interval, start_time, end_time)

        assert data is None
        assert cache_key not in cache._entries  # Entry should be removed

    @pytest.mark.asyncio
    async def test_expired_cache_cleanup(self, cache, sample_kline_data):
        """Test cleanup of expired cache entries."""
        symbol = "BTCUSDT"
        interval = "5m"
        start_time = 1640995200000
        end_time = 1640995800000

        # Store data
        await cache.store_kline_data(
            symbol, interval, start_time, end_time, sample_kline_data
        )

        cache_key = cache._generate_cache_key(symbol, interval, start_time, end_time)
        entry = cache._entries[cache_key]

        # Manually expire the entry
        entry.created_at = time.time() - (cache.max_age_seconds + 1)

        # Try to retrieve - should be treated as expired
        data = await cache.get_kline_data(symbol, interval, start_time, end_time)

        assert data is None
        assert cache_key not in cache._entries

    @pytest.mark.asyncio
    async def test_compression_toggle(self, temp_cache_dir, sample_kline_data):
        """Test cache with compression disabled."""
        cache = DataCache(
            cache_dir=temp_cache_dir,
            compression_enabled=False
        )

        symbol = "BTCUSDT"
        interval = "5m"
        start_time = 1640995200000
        end_time = 1640995800000

        # Store data without compression
        await cache.store_kline_data(
            symbol, interval, start_time, end_time, sample_kline_data
        )

        cache_key = cache._generate_cache_key(symbol, interval, start_time, end_time)
        entry = cache._entries[cache_key]

        # File should not be compressed
        file_path = Path(entry.file_path)
        assert file_path.suffix == ".json"
        assert entry.compressed is False

        # Data should still be retrievable
        data = await cache.get_kline_data(symbol, interval, start_time, end_time)
        assert data == sample_kline_data

    @pytest.mark.asyncio
    async def test_cache_size_enforcement(self, cache, sample_kline_data):
        """Test cache size limit enforcement."""
        # Set very small cache size
        cache.max_cache_size_bytes = 1024  # 1KB

        # Store multiple entries to exceed limit
        for i in range(10):
            await cache.store_kline_data(
                f"SYMBOL{i}", "5m", 1640995200000 + i, 1640995800000 + i, sample_kline_data
            )

        # Cache should have evicted older entries
        stats = cache.get_stats()
        assert stats.total_size_bytes <= cache.max_cache_size_bytes
        assert stats.eviction_count > 0

    @pytest.mark.asyncio
    async def test_clear_symbol(self, cache, sample_kline_data):
        """Test clearing cache entries for specific symbol."""
        # Store data for multiple symbols
        await cache.store_kline_data("BTCUSDT", "5m", 1640995200000, 1640995800000, sample_kline_data)
        await cache.store_kline_data("ETHUSDT", "5m", 1640995200000, 1640995800000, sample_kline_data)
        await cache.store_kline_data("BTCUSDT", "15m", 1640995200000, 1640995800000, sample_kline_data)

        initial_count = len(cache._entries)
        assert initial_count == 3

        # Clear BTCUSDT entries
        removed_count = await cache.clear_symbol("BTCUSDT")

        assert removed_count == 2
        assert len(cache._entries) == 1

        # Only ETHUSDT entry should remain
        remaining_entry = list(cache._entries.values())[0]
        assert remaining_entry.symbol == "ETHUSDT"

    @pytest.mark.asyncio
    async def test_clear_all(self, cache, sample_kline_data):
        """Test clearing all cache entries."""
        # Store multiple entries
        await cache.store_kline_data("BTCUSDT", "5m", 1640995200000, 1640995800000, sample_kline_data)
        await cache.store_kline_data("ETHUSDT", "15m", 1640995200000, 1640995800000, sample_kline_data)

        assert len(cache._entries) == 2

        # Clear all
        removed_count = await cache.clear_all()

        assert removed_count == 2
        assert len(cache._entries) == 0

        # Verify files were deleted
        cache_dir = Path(cache.cache_dir)
        klines_dir = cache_dir / "klines"
        assert len(list(klines_dir.glob("*.json*"))) == 0

    @pytest.mark.asyncio
    async def test_cleanup_expired(self, cache, sample_kline_data):
        """Test cleanup of expired entries."""
        # Store data
        await cache.store_kline_data("BTCUSDT", "5m", 1640995200000, 1640995800000, sample_kline_data)

        # Manually expire entry
        entry = list(cache._entries.values())[0]
        entry.created_at = time.time() - (cache.max_age_seconds + 1)

        # Run cleanup
        removed_count = await cache.cleanup_expired()

        assert removed_count == 1
        assert len(cache._entries) == 0

    @pytest.mark.asyncio
    async def test_cache_info(self, cache, sample_kline_data):
        """Test cache information reporting."""
        # Store some data
        await cache.store_kline_data("BTCUSDT", "5m", 1640995200000, 1640995800000, sample_kline_data)

        cache_info = cache.get_cache_info()

        assert 'cache_dir' in cache_info
        assert 'max_size_gb' in cache_info
        assert 'current_size_gb' in cache_info
        assert 'compression_enabled' in cache_info
        assert 'total_entries' in cache_info
        assert 'hit_count' in cache_info
        assert 'miss_count' in cache_info
        assert 'hit_rate' in cache_info

        assert cache_info['total_entries'] == 1
        assert cache_info['compression_enabled'] is True

    @pytest.mark.asyncio
    async def test_list_cached_symbols(self, cache, sample_kline_data):
        """Test listing cached symbols and intervals."""
        # Store data for different symbols and intervals
        await cache.store_kline_data("BTCUSDT", "5m", 1640995200000, 1640995800000, sample_kline_data)
        await cache.store_kline_data("BTCUSDT", "15m", 1640995200000, 1640995800000, sample_kline_data)
        await cache.store_kline_data("ETHUSDT", "5m", 1640995200000, 1640995800000, sample_kline_data)

        symbols = cache.list_cached_symbols()

        assert "BTCUSDT" in symbols
        assert "ETHUSDT" in symbols
        assert "5m" in symbols["BTCUSDT"]
        assert "15m" in symbols["BTCUSDT"]
        assert "5m" in symbols["ETHUSDT"]

    @pytest.mark.asyncio
    async def test_optimize_cache(self, cache, sample_kline_data):
        """Test cache optimization."""
        # Store some data with an expired entry
        await cache.store_kline_data("BTCUSDT", "5m", 1640995200000, 1640995800000, sample_kline_data)

        # Expire one entry
        entry = list(cache._entries.values())[0]
        entry.created_at = time.time() - (cache.max_age_seconds + 1)

        # Run optimization
        results = await cache.optimize_cache()

        assert 'expired_removed' in results
        assert 'total_entries' in results
        assert 'total_size_mb' in results
        assert results['expired_removed'] == 1

    @pytest.mark.asyncio
    async def test_concurrent_access(self, cache, sample_kline_data):
        """Test concurrent cache operations."""
        symbol = "BTCUSDT"
        interval = "5m"
        start_time = 1640995200000
        end_time = 1640995800000

        # Store data first
        await cache.store_kline_data(symbol, interval, start_time, end_time, sample_kline_data)

        async def read_cache():
            return await cache.get_kline_data(symbol, interval, start_time, end_time)

        async def write_cache(data):
            return await cache.store_kline_data(
                symbol, interval, start_time + 1000, end_time + 1000, data
            )

        # Run concurrent operations
        tasks = [
            read_cache(),
            read_cache(),
            write_cache(sample_kline_data),
            read_cache()
        ]

        results = await asyncio.gather(*tasks)

        # All read operations should succeed
        assert results[0] == sample_kline_data
        assert results[1] == sample_kline_data
        assert results[2] is True  # Write operation
        assert results[3] == sample_kline_data

    @pytest.mark.asyncio
    async def test_empty_data_handling(self, cache):
        """Test handling of empty data."""
        # Try to store empty data
        success = await cache.store_kline_data("BTCUSDT", "5m", 1640995200000, 1640995800000, [])

        assert success is False
        assert len(cache._entries) == 0

    @pytest.mark.asyncio
    async def test_file_corruption_recovery(self, cache, sample_kline_data):
        """Test recovery from file corruption."""
        symbol = "BTCUSDT"
        interval = "5m"
        start_time = 1640995200000
        end_time = 1640995800000

        # Store data
        await cache.store_kline_data(symbol, interval, start_time, end_time, sample_kline_data)

        cache_key = cache._generate_cache_key(symbol, interval, start_time, end_time)
        entry = cache._entries[cache_key]

        # Corrupt file by making it unreadable
        with open(entry.file_path, 'wb') as f:
            f.write(b'corrupted binary data')

        # Try to read - should handle corruption gracefully
        data = await cache.get_kline_data(symbol, interval, start_time, end_time)

        assert data is None
        assert cache_key not in cache._entries  # Entry should be removed

    @pytest.mark.asyncio
    async def test_metadata_persistence(self, temp_cache_dir, sample_kline_data):
        """Test cache metadata persistence across restarts."""
        # Create cache and store data
        cache1 = DataCache(cache_dir=temp_cache_dir)
        await cache1.store_kline_data("BTCUSDT", "5m", 1640995200000, 1640995800000, sample_kline_data)

        assert len(cache1._entries) == 1

        # Create new cache instance (simulating restart)
        cache2 = DataCache(cache_dir=temp_cache_dir)

        # Should load existing metadata
        assert len(cache2._entries) == 1

        # Should be able to retrieve data
        data = await cache2.get_kline_data("BTCUSDT", "5m", 1640995200000, 1640995800000)
        assert data == sample_kline_data

    @pytest.mark.asyncio
    async def test_missing_file_handling(self, cache, sample_kline_data):
        """Test handling when cache file is missing."""
        # Store data
        await cache.store_kline_data("BTCUSDT", "5m", 1640995200000, 1640995800000, sample_kline_data)

        cache_key = cache._generate_cache_key("BTCUSDT", "5m", 1640995200000, 1640995800000)
        entry = cache._entries[cache_key]

        # Delete the cache file
        Path(entry.file_path).unlink()

        # Try to retrieve - should handle missing file gracefully
        data = await cache.get_kline_data("BTCUSDT", "5m", 1640995200000, 1640995800000)

        assert data is None
        assert cache_key not in cache._entries  # Entry should be removed

    @pytest.mark.asyncio
    async def test_cache_stats_accuracy(self, cache, sample_kline_data):
        """Test accuracy of cache statistics."""
        # Initial stats
        stats = cache.get_stats()
        assert stats.hit_count == 0
        assert stats.miss_count == 0
        assert stats.total_entries == 0

        # Store data
        await cache.store_kline_data("BTCUSDT", "5m", 1640995200000, 1640995800000, sample_kline_data)

        stats = cache.get_stats()
        assert stats.total_entries == 1
        assert stats.total_size_bytes > 0

        # Cache hit
        await cache.get_kline_data("BTCUSDT", "5m", 1640995200000, 1640995800000)

        stats = cache.get_stats()
        assert stats.hit_count == 1
        assert stats.miss_count == 0

        # Cache miss
        await cache.get_kline_data("ETHUSDT", "15m", 1640995200000, 1640995800000)

        stats = cache.get_stats()
        assert stats.hit_count == 1
        assert stats.miss_count == 1