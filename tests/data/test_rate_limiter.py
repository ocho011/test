"""
Tests for RateLimiter token bucket algorithm.

Test cases cover rate limiting, token consumption, request queuing,
priority handling, and performance monitoring.
"""

import asyncio
import time
from unittest.mock import AsyncMock

import pytest

from trading_bot.data.rate_limiter import (
    Priority,
    RateLimit,
    RateLimiter,
    RequestQueue,
    TokenBucket,
)


class TestTokenBucket:
    """Test suite for TokenBucket algorithm."""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test token bucket initialization."""
        bucket = TokenBucket(capacity=100, refill_rate=10)

        assert bucket.capacity == 100
        assert bucket.refill_rate == 10
        assert bucket.tokens == 100  # Starts full
        assert bucket.last_refill <= time.time()

    @pytest.mark.asyncio
    async def test_token_consumption(self):
        """Test token consumption from bucket."""
        bucket = TokenBucket(capacity=100, refill_rate=10)

        # Consume tokens successfully
        assert await bucket.consume(50) is True
        assert bucket.tokens == 50

        # Try to consume more than available
        assert await bucket.consume(60) is False
        assert bucket.tokens == 50  # Unchanged

        # Consume exact amount available
        assert await bucket.consume(50) is True
        assert bucket.tokens == 0

    @pytest.mark.asyncio
    async def test_token_refill(self):
        """Test token refill mechanism."""
        bucket = TokenBucket(capacity=100, refill_rate=50)  # 50 tokens per second

        # Consume all tokens
        await bucket.consume(100)
        assert bucket.tokens == 0

        # Wait for refill
        await asyncio.sleep(0.1)  # 100ms = 5 tokens should be added
        bucket._refill_tokens()

        # Should have some tokens back (approximately 5)
        assert bucket.tokens > 0
        assert bucket.tokens <= 10  # Account for timing variance

    @pytest.mark.asyncio
    async def test_max_capacity_limit(self):
        """Test that tokens don't exceed capacity."""
        bucket = TokenBucket(capacity=100, refill_rate=200)

        # Start with full bucket
        assert bucket.tokens == 100

        # Wait and refill - should not exceed capacity
        await asyncio.sleep(0.1)
        bucket._refill_tokens()

        assert bucket.tokens <= 100

    @pytest.mark.asyncio
    async def test_concurrent_consumption(self):
        """Test concurrent token consumption."""
        bucket = TokenBucket(capacity=100, refill_rate=10)

        async def consume_tokens(amount):
            return await bucket.consume(amount)

        # Concurrent consumption
        results = await asyncio.gather(
            consume_tokens(30),
            consume_tokens(30),
            consume_tokens(30),
            consume_tokens(30),
        )

        # Only 3 out of 4 should succeed (90 tokens consumed)
        successful = sum(1 for result in results if result)
        assert successful == 3
        assert bucket.tokens == 10


class TestRequestQueue:
    """Test suite for priority-based request queue."""

    @pytest.mark.asyncio
    async def test_queue_initialization(self):
        """Test request queue initialization."""
        queue = RequestQueue(max_size=100)

        assert queue.max_size == 100
        assert queue.qsize() == 0
        assert queue.empty()

    @pytest.mark.asyncio
    async def test_priority_ordering(self):
        """Test priority-based request ordering."""
        queue = RequestQueue(max_size=100)

        # Add requests with different priorities
        await queue.put("low_priority", Priority.LOW)
        await queue.put("high_priority", Priority.HIGH)
        await queue.put("medium_priority", Priority.MEDIUM)
        await queue.put("critical_priority", Priority.CRITICAL)

        # Should dequeue in priority order
        assert (await queue.get())[0] == "critical_priority"
        assert (await queue.get())[0] == "high_priority"
        assert (await queue.get())[0] == "medium_priority"
        assert (await queue.get())[0] == "low_priority"

    @pytest.mark.asyncio
    async def test_fifo_within_priority(self):
        """Test FIFO ordering within same priority level."""
        queue = RequestQueue(max_size=100)

        # Add multiple requests with same priority
        await queue.put("first", Priority.MEDIUM)
        await queue.put("second", Priority.MEDIUM)
        await queue.put("third", Priority.MEDIUM)

        # Should dequeue in FIFO order
        assert (await queue.get())[0] == "first"
        assert (await queue.get())[0] == "second"
        assert (await queue.get())[0] == "third"

    @pytest.mark.asyncio
    async def test_queue_size_limit(self):
        """Test queue size limit enforcement."""
        queue = RequestQueue(max_size=2)

        # Fill queue to capacity
        await queue.put("request1", Priority.LOW)
        await queue.put("request2", Priority.LOW)

        # Third request should be rejected
        with pytest.raises(asyncio.QueueFull):
            await queue.put("request3", Priority.LOW)

    @pytest.mark.asyncio
    async def test_get_with_timeout(self):
        """Test getting requests with timeout."""
        queue = RequestQueue(max_size=100)

        # Should timeout on empty queue
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(queue.get(), timeout=0.1)


class TestRateLimit:
    """Test suite for RateLimit configuration."""

    def test_rate_limit_creation(self):
        """Test rate limit configuration creation."""
        rate_limit = RateLimit(requests_per_minute=1200, burst_capacity=50)

        assert rate_limit.requests_per_minute == 1200
        assert rate_limit.burst_capacity == 50
        assert rate_limit.requests_per_second == 20.0  # 1200 / 60

    def test_rate_limit_validation(self):
        """Test rate limit validation."""
        # Valid configuration
        rate_limit = RateLimit(requests_per_minute=600, burst_capacity=25)
        assert rate_limit.requests_per_minute == 600

        # Invalid configuration should still work but may not be optimal
        rate_limit = RateLimit(requests_per_minute=0, burst_capacity=0)
        assert rate_limit.requests_per_minute == 0


class TestRateLimiter:
    """Test suite for RateLimiter main class."""

    @pytest.fixture
    def rate_limiter(self):
        """Create RateLimiter instance for testing."""
        return RateLimiter(
            requests_per_minute=600,  # 10 requests per second
            burst_capacity=20,
            queue_size=100,
        )

    @pytest.mark.asyncio
    async def test_initialization(self, rate_limiter):
        """Test rate limiter initialization."""
        assert rate_limiter.rate_limit.requests_per_minute == 600
        assert rate_limiter.rate_limit.burst_capacity == 20
        assert not rate_limiter._shutdown

    @pytest.mark.asyncio
    async def test_start_and_stop(self, rate_limiter):
        """Test rate limiter lifecycle."""
        # Start processing
        await rate_limiter.start()
        assert not rate_limiter._shutdown

        # Stop processing
        await rate_limiter.stop()
        assert rate_limiter._shutdown

    @pytest.mark.asyncio
    async def test_successful_request(self, rate_limiter):
        """Test successful request processing."""
        await rate_limiter.start()

        mock_func = AsyncMock(return_value="success")

        # Should process request successfully
        result = await rate_limiter.request(
            mock_func, "test_arg", priority=Priority.MEDIUM
        )

        assert result == "success"
        mock_func.assert_called_once_with("test_arg")

        await rate_limiter.stop()

    @pytest.mark.asyncio
    async def test_request_queuing(self, rate_limiter):
        """Test request queuing when rate limited."""
        # Create bucket with very limited capacity
        rate_limiter._bucket = TokenBucket(capacity=1, refill_rate=1)

        await rate_limiter.start()

        mock_func = AsyncMock(return_value="success")

        # First request should succeed immediately
        result1 = await rate_limiter.request(mock_func, priority=Priority.HIGH)
        assert result1 == "success"

        # Second request should be queued and processed after refill
        start_time = time.time()
        result2 = await rate_limiter.request(mock_func, priority=Priority.HIGH)
        end_time = time.time()

        assert result2 == "success"
        assert end_time - start_time >= 0.5  # Should have waited for refill

        await rate_limiter.stop()

    @pytest.mark.asyncio
    async def test_priority_processing(self, rate_limiter):
        """Test priority-based request processing."""
        # Limit bucket to force queuing
        rate_limiter._bucket = TokenBucket(capacity=1, refill_rate=10)

        await rate_limiter.start()

        results = []

        async def mock_func(priority_name):
            results.append(priority_name)
            return priority_name

        # Consume initial token
        await rate_limiter._bucket.consume(1)

        # Queue requests with different priorities
        tasks = [
            rate_limiter.request(mock_func, "low", priority=Priority.LOW),
            rate_limiter.request(mock_func, "high", priority=Priority.HIGH),
            rate_limiter.request(mock_func, "critical", priority=Priority.CRITICAL),
            rate_limiter.request(mock_func, "medium", priority=Priority.MEDIUM),
        ]

        await asyncio.gather(*tasks)

        # Should process in priority order (after initial token refill)
        assert "critical" in results
        assert "high" in results
        assert results.index("critical") < results.index("high")
        assert results.index("high") < results.index("medium")

        await rate_limiter.stop()

    @pytest.mark.asyncio
    async def test_request_timeout(self, rate_limiter):
        """Test request timeout handling."""
        # Set very slow refill rate to force timeout
        rate_limiter._bucket = TokenBucket(capacity=0, refill_rate=0.1)

        await rate_limiter.start()

        mock_func = AsyncMock(return_value="success")

        # Should timeout waiting for tokens
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                rate_limiter.request(mock_func, priority=Priority.LOW), timeout=0.1
            )

        await rate_limiter.stop()

    @pytest.mark.asyncio
    async def test_function_exception_handling(self, rate_limiter):
        """Test handling of exceptions in rate-limited functions."""
        await rate_limiter.start()

        async def failing_func():
            raise ValueError("Test error")

        # Should propagate the exception
        with pytest.raises(ValueError, match="Test error"):
            await rate_limiter.request(failing_func, priority=Priority.MEDIUM)

        await rate_limiter.stop()

    @pytest.mark.asyncio
    async def test_statistics_tracking(self, rate_limiter):
        """Test request statistics tracking."""
        await rate_limiter.start()

        mock_func = AsyncMock(return_value="success")

        # Make several requests
        for _ in range(5):
            await rate_limiter.request(mock_func, priority=Priority.MEDIUM)

        stats = rate_limiter.get_stats()

        assert stats["total_requests"] == 5
        assert stats["successful_requests"] == 5
        assert stats["failed_requests"] == 0
        assert stats["current_queue_size"] == 0
        assert "average_wait_time" in stats

        await rate_limiter.stop()

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, rate_limiter):
        """Test handling of many concurrent requests."""
        await rate_limiter.start()

        mock_func = AsyncMock(return_value="success")

        # Submit many concurrent requests
        tasks = [
            rate_limiter.request(mock_func, f"request_{i}", priority=Priority.MEDIUM)
            for i in range(20)
        ]

        results = await asyncio.gather(*tasks)

        # All should succeed
        assert len(results) == 20
        assert all(result == "success" for result in results)

        await rate_limiter.stop()

    @pytest.mark.asyncio
    async def test_queue_overflow_handling(self, rate_limiter):
        """Test handling when request queue overflows."""
        # Create small queue
        small_limiter = RateLimiter(
            requests_per_minute=60, burst_capacity=1, queue_size=2
        )

        # Block all tokens
        small_limiter._bucket = TokenBucket(capacity=0, refill_rate=0.1)

        await small_limiter.start()

        slow_func = AsyncMock(side_effect=lambda: asyncio.sleep(1))

        # Fill queue beyond capacity
        tasks = []
        for i in range(5):
            task = asyncio.create_task(
                small_limiter.request(slow_func, priority=Priority.LOW)
            )
            tasks.append(task)
            await asyncio.sleep(0.01)  # Small delay to ensure ordering

        # Some requests should fail due to queue overflow
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Should have some queue full exceptions
        queue_full_count = sum(1 for r in results if isinstance(r, Exception))
        assert queue_full_count > 0

        await small_limiter.stop()

    @pytest.mark.asyncio
    async def test_shutdown_behavior(self, rate_limiter):
        """Test behavior during shutdown."""
        await rate_limiter.start()

        mock_func = AsyncMock(return_value="success")

        # Start a request
        task = asyncio.create_task(
            rate_limiter.request(mock_func, priority=Priority.LOW)
        )

        # Shutdown while request is processing
        await rate_limiter.stop()

        # Request should still complete or be cancelled gracefully
        try:
            result = await task
            assert result == "success"
        except asyncio.CancelledError:
            # Acceptable if cancelled during shutdown
            pass

    @pytest.mark.asyncio
    async def test_adaptive_rate_limiting(self, rate_limiter):
        """Test rate limiter adaptation to load."""
        await rate_limiter.start()

        mock_func = AsyncMock(return_value="success")

        # Monitor performance under different loads
        start_time = time.time()

        # Light load
        for _ in range(5):
            await rate_limiter.request(mock_func, priority=Priority.MEDIUM)

        light_load_time = time.time() - start_time

        # Heavy load
        start_time = time.time()
        tasks = [
            rate_limiter.request(mock_func, priority=Priority.MEDIUM) for _ in range(20)
        ]
        await asyncio.gather(*tasks)

        heavy_load_time = time.time() - start_time

        # Heavy load should take longer (rate limiting in effect)
        assert heavy_load_time > light_load_time

        await rate_limiter.stop()

    @pytest.mark.asyncio
    async def test_rate_limiter_reset(self, rate_limiter):
        """Test rate limiter state reset."""
        await rate_limiter.start()

        mock_func = AsyncMock(return_value="success")

        # Make some requests to build up stats
        for _ in range(3):
            await rate_limiter.request(mock_func, priority=Priority.MEDIUM)

        # Reset stats
        rate_limiter.reset_stats()

        stats = rate_limiter.get_stats()
        assert stats["total_requests"] == 0
        assert stats["successful_requests"] == 0

        await rate_limiter.stop()
