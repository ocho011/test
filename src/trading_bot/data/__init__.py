"""
Data module for market data providers and exchange connectivity.

This module contains components for connecting to cryptocurrency exchanges,
receiving real-time market data, and managing historical data caching.
"""

from .binance_client import BinanceClient, BinanceClientError
from .data_cache import CacheEntry, CacheStats, DataCache
from .market_data_provider import MarketDataProvider, StreamSubscription, StreamType
from .market_data_aggregator import MarketDataAggregator
from .rate_limiter import RateLimit, RateLimiter, TokenBucket

__all__ = [
    "BinanceClient",
    "BinanceClientError",
    "MarketDataProvider",
    "MarketDataAggregator",
    "StreamType",
    "StreamSubscription",
    "RateLimiter",
    "RateLimit",
    "TokenBucket",
    "DataCache",
    "CacheEntry",
    "CacheStats",
]
