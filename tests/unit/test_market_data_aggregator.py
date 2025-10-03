"""
Unit tests for MarketDataAggregator component.
"""

import asyncio
from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from trading_bot.core.event_bus import EventBus
from trading_bot.core.events import (
    CandleClosedEvent,
    EventPriority,
    EventType,
    MarketDataEvent,
)
from trading_bot.data.binance_client import BinanceClient
from trading_bot.data.market_data_aggregator import MarketDataAggregator


@pytest.fixture
def mock_binance_client():
    """Create mock BinanceClient."""
    client = AsyncMock(spec=BinanceClient)
    client.get_historical_klines = AsyncMock(return_value=[])
    return client


@pytest.fixture
def event_bus():
    """Create EventBus instance."""
    return EventBus()


@pytest.fixture
def aggregator(mock_binance_client, event_bus):
    """Create MarketDataAggregator instance."""
    return MarketDataAggregator(
        binance_client=mock_binance_client,
        event_bus=event_bus,
        symbols=["BTCUSDT"],
        intervals=["5m"],
        lookback_bars=10,
    )


@pytest.mark.asyncio
class TestMarketDataAggregator:
    """Test suite for MarketDataAggregator."""

    async def test_initialization(self, aggregator, mock_binance_client, event_bus):
        """Test proper initialization of aggregator."""
        assert aggregator.binance_client == mock_binance_client
        assert aggregator.event_bus == event_bus
        assert aggregator.symbols == ["BTCUSDT"]
        assert aggregator.intervals == ["5m"]
        assert aggregator.lookback_bars == 10
        assert aggregator._events_processed == 0
        assert aggregator._candles_closed == 0

    async def test_start_loads_historical_data(
        self, aggregator, mock_binance_client
    ):
        """Test that start() loads historical data."""
        # Setup mock historical data
        mock_klines = [
            [
                1609459200000,  # timestamp
                "40000.0",  # open
                "41000.0",  # high
                "39000.0",  # low
                "40500.0",  # close
                "100.5",  # volume
                1609459499999,  # close_time
                "4050000.0",  # quote_volume
                1000,  # trades
                "50.0",  # taker_buy_base
                "2025000.0",  # taker_buy_quote
                "0",  # ignore
            ]
        ]
        mock_binance_client.get_historical_klines.return_value = mock_klines

        await aggregator.start()

        # Verify historical data was loaded
        mock_binance_client.get_historical_klines.assert_called_once_with(
            symbol="BTCUSDT", interval="5m", limit=10
        )

        # Verify DataFrame was created
        df = aggregator.get_candles("BTCUSDT", "5m")
        assert df is not None
        assert len(df) == 1
        assert df.iloc[0]["open"] == 40000.0
        assert df.iloc[0]["close"] == 40500.0

        await aggregator.stop()

    async def test_klines_to_dataframe_conversion(self, aggregator):
        """Test conversion of Binance klines to DataFrame."""
        klines = [
            [
                1609459200000,
                "40000.0",
                "41000.0",
                "39000.0",
                "40500.0",
                "100.5",
                1609459499999,
                "4050000.0",
                1000,
                "50.0",
                "2025000.0",
                "0",
            ],
            [
                1609459500000,
                "40500.0",
                "42000.0",
                "40000.0",
                "41500.0",
                "120.3",
                1609459799999,
                "4980000.0",
                1200,
                "60.0",
                "2490000.0",
                "0",
            ],
        ]

        df = aggregator._klines_to_dataframe(klines)

        assert len(df) == 2
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert df.iloc[0]["open"] == 40000.0
        assert df.iloc[0]["high"] == 41000.0
        assert df.iloc[1]["close"] == 41500.0

    async def test_klines_to_dataframe_empty(self, aggregator):
        """Test conversion with empty klines."""
        df = aggregator._klines_to_dataframe([])
        assert len(df) == 0
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]

    async def test_update_current_candle_new(self, aggregator):
        """Test updating current candle with new candle."""
        await aggregator.start()

        event = MarketDataEvent(
            source="test",
            symbol="BTCUSDT",
            price=Decimal("40000.0"),
            volume=Decimal("100.0"),
            open_price=Decimal("39500.0"),
            high_price=Decimal("40500.0"),
            low_price=Decimal("39000.0"),
            close_price=Decimal("40000.0"),
            metadata={"interval": "5m", "open_time": 1609459200000},
        )

        aggregator._update_current_candle(event)

        current = aggregator._current_candles["BTCUSDT"]["5m"]
        assert current["open"] == 39500.0
        assert current["high"] == 40500.0
        assert current["low"] == 39000.0
        assert current["close"] == 40000.0
        assert current["volume"] == 100.0

        await aggregator.stop()

    async def test_update_current_candle_existing(self, aggregator):
        """Test updating existing candle with new high/low."""
        await aggregator.start()

        # Initial candle
        event1 = MarketDataEvent(
            source="test",
            symbol="BTCUSDT",
            price=Decimal("40000.0"),
            volume=Decimal("100.0"),
            open_price=Decimal("39500.0"),
            high_price=Decimal("40500.0"),
            low_price=Decimal("39000.0"),
            close_price=Decimal("40000.0"),
            metadata={"interval": "5m", "open_time": 1609459200000},
        )
        aggregator._update_current_candle(event1)

        # Update with new high
        event2 = MarketDataEvent(
            source="test",
            symbol="BTCUSDT",
            price=Decimal("41000.0"),
            volume=Decimal("120.0"),
            high_price=Decimal("41000.0"),
            low_price=Decimal("40000.0"),
            close_price=Decimal("41000.0"),
            metadata={"interval": "5m", "open_time": 1609459200000},
        )
        aggregator._update_current_candle(event2)

        current = aggregator._current_candles["BTCUSDT"]["5m"]
        assert current["open"] == 39500.0  # Unchanged
        assert current["high"] == 41000.0  # Updated
        assert current["low"] == 39000.0  # Unchanged (lower)
        assert current["close"] == 41000.0  # Updated
        assert current["volume"] == 120.0  # Updated

        await aggregator.stop()

    async def test_handle_candle_closed(self, aggregator, event_bus):
        """Test handling candle closure and event emission."""
        await event_bus.start()
        await aggregator.start()

        # Setup current candle
        aggregator._current_candles["BTCUSDT"]["5m"] = {
            "open": 40000.0,
            "high": 41000.0,
            "low": 39000.0,
            "close": 40500.0,
            "volume": 100.0,
            "timestamp": 1609459200000,
        }

        # Create closed event
        event = MarketDataEvent(
            source="test",
            symbol="BTCUSDT",
            price=Decimal("40500.0"),
            volume=Decimal("100.0"),
            metadata={"interval": "5m", "is_closed": True, "open_time": 1609459200000},
        )

        # Subscribe to CandleClosedEvent
        candle_events = []

        async def capture_event(evt):
            candle_events.append(evt)

        await event_bus.subscribe(capture_event, EventType.CANDLE_CLOSED)

        # Handle closure
        await aggregator._handle_candle_closed(event)

        # Wait for event processing
        await asyncio.sleep(0.5)

        # Verify DataFrame was updated
        df = aggregator.get_candles("BTCUSDT", "5m")
        assert len(df) == 1
        assert df.iloc[0]["open"] == 40000.0
        assert df.iloc[0]["close"] == 40500.0

        # Verify current candle was cleared
        assert aggregator._current_candles["BTCUSDT"]["5m"] == {}

        # Verify event was published
        assert len(candle_events) == 1
        assert candle_events[0].symbol == "BTCUSDT"
        assert candle_events[0].interval == "5m"
        assert isinstance(candle_events[0].df, pd.DataFrame)

        await aggregator.stop()
        await event_bus.stop()

    async def test_on_market_data_event_closed(self, aggregator, event_bus):
        """Test full flow of market data event with closure."""
        await event_bus.start()
        await aggregator.start()

        candle_events = []

        async def capture_event(evt):
            candle_events.append(evt)

        await event_bus.subscribe(capture_event, EventType.CANDLE_CLOSED)

        # Send opening event
        event1 = MarketDataEvent(
            source="test",
            symbol="BTCUSDT",
            price=Decimal("40000.0"),
            volume=Decimal("100.0"),
            open_price=Decimal("39500.0"),
            high_price=Decimal("40500.0"),
            low_price=Decimal("39000.0"),
            close_price=Decimal("40000.0"),
            metadata={
                "interval": "5m",
                "is_closed": False,
                "open_time": 1609459200000,
            },
        )
        await aggregator._on_market_data_event(event1)

        # Send closing event
        event2 = MarketDataEvent(
            source="test",
            symbol="BTCUSDT",
            price=Decimal("40500.0"),
            volume=Decimal("150.0"),
            close_price=Decimal("40500.0"),
            metadata={
                "interval": "5m",
                "is_closed": True,
                "open_time": 1609459200000,
            },
        )
        await aggregator._on_market_data_event(event2)

        await asyncio.sleep(0.1)

        # Verify stats
        assert aggregator._events_processed == 2
        assert aggregator._candles_closed == 1

        # Verify candle event was emitted
        assert len(candle_events) == 1

        await aggregator.stop()
        await event_bus.stop()

    async def test_ignore_unsubscribed_symbols(self, aggregator):
        """Test that events for unsubscribed symbols are ignored."""
        await aggregator.start()

        event = MarketDataEvent(
            source="test",
            symbol="ETHUSDT",  # Not subscribed
            price=Decimal("3000.0"),
            volume=Decimal("50.0"),
            metadata={"interval": "5m", "is_closed": False},
        )

        await aggregator._on_market_data_event(event)

        # Verify event was processed but not stored
        assert aggregator._events_processed == 1
        assert "ETHUSDT" not in aggregator._current_candles

        await aggregator.stop()

    async def test_lookback_bars_limit(self, aggregator, mock_binance_client):
        """Test that only lookback_bars candles are kept."""
        await aggregator.start()

        # Add more candles than lookback_bars (10)
        for i in range(15):
            aggregator._current_candles["BTCUSDT"]["5m"] = {
                "open": 40000.0 + i,
                "high": 41000.0 + i,
                "low": 39000.0 + i,
                "close": 40500.0 + i,
                "volume": 100.0,
                "timestamp": 1609459200000 + (i * 300000),  # 5min intervals
            }

            event = MarketDataEvent(
                source="test",
                symbol="BTCUSDT",
                price=Decimal(str(40500.0 + i)),
                volume=Decimal("100.0"),
                metadata={
                    "interval": "5m",
                    "is_closed": True,
                    "open_time": 1609459200000 + (i * 300000),
                },
            )
            await aggregator._handle_candle_closed(event)

        # Verify only last 10 candles are kept
        df = aggregator.get_candles("BTCUSDT", "5m")
        assert len(df) == 10

        await aggregator.stop()

    async def test_get_candles(self, aggregator):
        """Test retrieving candles for symbol/interval."""
        await aggregator.start()

        # Add test data
        test_df = pd.DataFrame(
            {"open": [40000], "high": [41000], "low": [39000], "close": [40500], "volume": [100]},
            index=[pd.Timestamp("2021-01-01")]
        )
        aggregator._candles["BTCUSDT"]["5m"] = test_df

        # Get candles
        df = aggregator.get_candles("BTCUSDT", "5m")
        assert df is not None
        assert len(df) == 1
        assert df.iloc[0]["close"] == 40500

        # Verify it's a copy
        df.iloc[0, df.columns.get_loc("close")] = 99999
        assert aggregator._candles["BTCUSDT"]["5m"].iloc[0]["close"] == 40500

        await aggregator.stop()

    async def test_get_candles_invalid(self, aggregator):
        """Test retrieving candles for invalid symbol/interval."""
        await aggregator.start()

        df = aggregator.get_candles("INVALID", "5m")
        assert df is None

        df = aggregator.get_candles("BTCUSDT", "invalid")
        assert df is None

        await aggregator.stop()

    async def test_get_stats(self, aggregator):
        """Test statistics retrieval."""
        await aggregator.start()

        stats = aggregator.get_stats()

        assert stats["events_processed"] == 0
        assert stats["candles_closed"] == 0
        assert stats["symbols"] == ["BTCUSDT"]
        assert stats["intervals"] == ["5m"]
        assert "candle_counts" in stats

        await aggregator.stop()

    async def test_multiple_symbols_intervals(self, mock_binance_client, event_bus):
        """Test aggregator with multiple symbols and intervals."""
        aggregator = MarketDataAggregator(
            binance_client=mock_binance_client,
            event_bus=event_bus,
            symbols=["BTCUSDT", "ETHUSDT"],
            intervals=["5m", "15m"],
            lookback_bars=10,
        )

        await aggregator.start()

        # Verify all combinations initialized
        assert "BTCUSDT" in aggregator._candles
        assert "ETHUSDT" in aggregator._candles
        assert "5m" in aggregator._candles["BTCUSDT"]
        assert "15m" in aggregator._candles["BTCUSDT"]
        assert "5m" in aggregator._candles["ETHUSDT"]
        assert "15m" in aggregator._candles["ETHUSDT"]

        await aggregator.stop()

    async def test_error_handling_in_event_handler(self, aggregator, caplog):
        """Test error handling in event processing."""
        await aggregator.start()

        # Create malformed event
        event = MarketDataEvent(
            source="test",
            symbol="BTCUSDT",
            price=Decimal("40000.0"),
            volume=Decimal("100.0"),
            metadata={},  # Missing interval
        )

        # Should not crash
        await aggregator._on_market_data_event(event)

        assert aggregator._events_processed == 1

        await aggregator.stop()
