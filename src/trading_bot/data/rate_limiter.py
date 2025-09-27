"""
Rate limiter for API request management.

This module provides rate limiting functionality to ensure compliance
with exchange API limits using asyncio semaphores and token bucket algorithm.
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class RateLimit:
    """Rate limit configuration."""

    requests_per_minute: int
    max_burst: int = None

    def __post_init__(self):
        if self.max_burst is None:
            self.max_burst = min(self.requests_per_minute, 50)


class TokenBucket:
    """Token bucket implementation for rate limiting."""

    def __init__(self, rate_limit: RateLimit):
        """
        Initialize token bucket.

        Args:
            rate_limit: Rate limit configuration
        """
        self.rate_limit = rate_limit
        self.tokens = float(rate_limit.max_burst)
        self.max_tokens = float(rate_limit.max_burst)
        self.refill_rate = rate_limit.requests_per_minute / 60.0  # tokens per second
        self.last_refill = time.time()
        self._lock = asyncio.Lock()

    async def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens from the bucket.

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if tokens were consumed, False if not enough tokens
        """
        async with self._lock:
            now = time.time()
            time_passed = now - self.last_refill

            # Refill bucket
            new_tokens = time_passed * self.refill_rate
            self.tokens = min(self.max_tokens, self.tokens + new_tokens)
            self.last_refill = now

            # Try to consume tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    def get_wait_time(self, tokens: int = 1) -> float:
        """
        Get estimated wait time for tokens to be available.

        Args:
            tokens: Number of tokens needed

        Returns:
            Wait time in seconds
        """
        if self.tokens >= tokens:
            return 0.0

        tokens_needed = tokens - self.tokens
        return tokens_needed / self.refill_rate


class RequestQueue:
    """Priority queue for API requests."""

    def __init__(self, max_size: int = 1000):
        """
        Initialize request queue.

        Args:
            max_size: Maximum queue size
        """
        self.max_size = max_size
        self._high_priority = deque()
        self._normal_priority = deque()
        self._low_priority = deque()
        self._lock = asyncio.Lock()
        self._not_empty = asyncio.Condition()

    async def put(
        self, request_future: asyncio.Future, priority: str = "normal"
    ) -> bool:
        """
        Add request to queue.

        Args:
            request_future: Future representing the request
            priority: Request priority (high, normal, low)

        Returns:
            True if request was queued, False if queue is full
        """
        async with self._lock:
            total_size = (
                len(self._high_priority)
                + len(self._normal_priority)
                + len(self._low_priority)
            )

            if total_size >= self.max_size:
                return False

            if priority == "high":
                self._high_priority.append(request_future)
            elif priority == "low":
                self._low_priority.append(request_future)
            else:
                self._normal_priority.append(request_future)

        async with self._not_empty:
            self._not_empty.notify()

        return True

    async def get(self) -> asyncio.Future:
        """
        Get next request from queue (prioritized).

        Returns:
            Next request future to process
        """
        async with self._not_empty:
            while True:
                async with self._lock:
                    # Check high priority first
                    if self._high_priority:
                        return self._high_priority.popleft()
                    # Then normal priority
                    if self._normal_priority:
                        return self._normal_priority.popleft()
                    # Finally low priority
                    if self._low_priority:
                        return self._low_priority.popleft()

                # Wait for requests to be added
                await self._not_empty.wait()

    def qsize(self) -> Dict[str, int]:
        """Get queue sizes by priority."""
        return {
            "high": len(self._high_priority),
            "normal": len(self._normal_priority),
            "low": len(self._low_priority),
            "total": len(self._high_priority)
            + len(self._normal_priority)
            + len(self._low_priority),
        }


class RateLimiter:
    """
    Advanced rate limiter for API requests.

    Implements token bucket algorithm with priority queuing and
    automatic request retry with exponential backoff.
    """

    def __init__(self, default_rate_limit: RateLimit):
        """
        Initialize rate limiter.

        Args:
            default_rate_limit: Default rate limit configuration
        """
        self.default_rate_limit = default_rate_limit
        self.logger = logging.getLogger(__name__)

        # Rate limiting per endpoint type
        self._buckets: Dict[str, TokenBucket] = {}
        self._endpoint_limits: Dict[str, RateLimit] = {}

        # Request management
        self._request_queue = RequestQueue()
        self._concurrent_limit = asyncio.Semaphore(10)  # Max concurrent requests
        self._running = False
        self._processor_task: Optional[asyncio.Task] = None

        # Statistics
        self._stats = {
            "requests_made": 0,
            "requests_queued": 0,
            "requests_rejected": 0,
            "rate_limit_hits": 0,
            "average_wait_time": 0.0,
        }

        # Create default bucket
        self._buckets["default"] = TokenBucket(default_rate_limit)

    async def start(self) -> None:
        """Start the rate limiter processor."""
        if self._running:
            self.logger.warning("RateLimiter is already running")
            return

        self._running = True
        self._processor_task = asyncio.create_task(self._process_requests())
        self.logger.info("RateLimiter started")

    async def stop(self) -> None:
        """Stop the rate limiter processor."""
        if not self._running:
            return

        self._running = False

        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass

        self.logger.info("RateLimiter stopped")

    def configure_endpoint(self, endpoint: str, rate_limit: RateLimit) -> None:
        """
        Configure rate limit for specific endpoint.

        Args:
            endpoint: Endpoint identifier
            rate_limit: Rate limit configuration
        """
        self._endpoint_limits[endpoint] = rate_limit
        self._buckets[endpoint] = TokenBucket(rate_limit)
        self.logger.info(
            f"Configured rate limit for {endpoint}: {rate_limit.requests_per_minute}/min"
        )

    async def acquire(
        self,
        endpoint: str = "default",
        priority: str = "normal",
        timeout: Optional[float] = None,
    ) -> bool:
        """
        Acquire permission to make a request.

        Args:
            endpoint: Endpoint identifier
            priority: Request priority (high, normal, low)
            timeout: Maximum time to wait in seconds

        Returns:
            True if permission granted, False if timeout or rejected
        """
        if not self._running:
            self.logger.warning("RateLimiter is not running")
            return False

        # Get appropriate bucket
        bucket = self._buckets.get(endpoint, self._buckets["default"])

        # Try immediate acquisition
        if await bucket.consume():
            self._stats["requests_made"] += 1
            return True

        # Queue the request if immediate acquisition failed
        request_future = asyncio.Future()
        request_future.endpoint = endpoint
        request_future.priority = priority
        request_future.timestamp = time.time()

        if not await self._request_queue.put(request_future, priority):
            self.logger.warning("Request queue is full, rejecting request")
            self._stats["requests_rejected"] += 1
            return False

        self._stats["requests_queued"] += 1

        try:
            if timeout:
                await asyncio.wait_for(request_future, timeout=timeout)
            else:
                await request_future
            return True

        except asyncio.TimeoutError:
            self.logger.warning(f"Request timed out after {timeout}s")
            return False
        except Exception as e:
            self.logger.error(f"Error waiting for rate limit: {e}")
            return False

    async def _process_requests(self) -> None:
        """Main request processing loop."""
        self.logger.info("Rate limiter processor started")

        while self._running:
            try:
                # Get next request from queue
                request_future = await self._request_queue.get()

                # Respect concurrent request limit
                async with self._concurrent_limit:
                    await self._process_single_request(request_future)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in rate limiter processor: {e}")
                await asyncio.sleep(0.1)

        self.logger.info("Rate limiter processor stopped")

    async def _process_single_request(self, request_future: asyncio.Future) -> None:
        """
        Process a single queued request.

        Args:
            request_future: Request future to process
        """
        try:
            endpoint = getattr(request_future, "endpoint", "default")
            timestamp = getattr(request_future, "timestamp", time.time())

            bucket = self._buckets.get(endpoint, self._buckets["default"])

            # Wait for tokens to be available
            while self._running:
                if await bucket.consume():
                    # Calculate wait time for statistics
                    wait_time = time.time() - timestamp
                    self._update_wait_time_stats(wait_time)

                    # Grant permission
                    if not request_future.done():
                        request_future.set_result(True)
                    self._stats["requests_made"] += 1
                    break

                # Wait before retrying
                wait_time = bucket.get_wait_time()
                await asyncio.sleep(min(wait_time, 1.0))  # Max 1 second wait

        except Exception as e:
            self.logger.error(f"Error processing request: {e}")
            if not request_future.done():
                request_future.set_exception(e)

    def _update_wait_time_stats(self, wait_time: float) -> None:
        """Update average wait time statistics."""
        current_avg = self._stats["average_wait_time"]
        requests_made = self._stats["requests_made"]

        if requests_made == 0:
            self._stats["average_wait_time"] = wait_time
        else:
            # Exponential moving average
            alpha = 0.1
            self._stats["average_wait_time"] = (
                alpha * wait_time + (1 - alpha) * current_avg
            )

    def get_stats(self) -> Dict[str, Any]:
        """
        Get rate limiter statistics.

        Returns:
            Statistics dictionary
        """
        stats = self._stats.copy()
        stats["queue_sizes"] = self._request_queue.qsize()
        stats["buckets_status"] = {}

        for endpoint, bucket in self._buckets.items():
            stats["buckets_status"][endpoint] = {
                "tokens": bucket.tokens,
                "max_tokens": bucket.max_tokens,
                "refill_rate": bucket.refill_rate,
            }

        return stats

    def get_current_limits(self) -> Dict[str, Dict[str, Any]]:
        """
        Get current rate limit configurations.

        Returns:
            Dictionary of endpoint limits
        """
        limits = {
            "default": {
                "requests_per_minute": self.default_rate_limit.requests_per_minute,
                "max_burst": self.default_rate_limit.max_burst,
            }
        }

        for endpoint, rate_limit in self._endpoint_limits.items():
            limits[endpoint] = {
                "requests_per_minute": rate_limit.requests_per_minute,
                "max_burst": rate_limit.max_burst,
            }

        return limits

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()
