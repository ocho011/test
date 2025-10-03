"""
Market Data Aggregator for converting real-time MarketDataEvents to OHLCV DataFrames.

This component subscribes to MarketDataEvent streams, accumulates candle data,
and publishes CandleClosedEvent when candles complete.
"""

import logging
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional

import pandas as pd

from trading_bot.core.events import (
    CandleClosedEvent,
    EventType,
    MarketDataEvent,
)
from trading_bot.core.event_bus import EventBus
from trading_bot.core.base_component import BaseComponent
from trading_bot.data.binance_client import BinanceClient


class MarketDataAggregator(BaseComponent):
    """
    Aggregates real-time MarketDataEvents into OHLCV DataFrames.

    Features:
    - Subscribes to MarketDataEvent for each symbol/interval combination
    - Accumulates candle data in memory using pandas DataFrame
    - Detects candle closure based on interval
    - Publishes CandleClosedEvent with complete DataFrame
    - Initializes with historical data from BinanceClient
    """

    def __init__(
        self,
        binance_client: BinanceClient,
        event_bus: EventBus,
        symbols: List[str],
        intervals: List[str],
        lookback_bars: int = 100,
        name: str = "MarketDataAggregator",
    ):
        """
        Initialize MarketDataAggregator.

        Args:
            binance_client: BinanceClient for fetching historical data
            event_bus: EventBus for subscribing and publishing events
            symbols: List of trading symbols (e.g., ["BTCUSDT", "ETHUSDT"])
            intervals: List of timeframe intervals (e.g., ["5m", "15m", "4h"])
            lookback_bars: Number of historical candles to load (default: 100)
            name: Component name for lifecycle management
        """
        super().__init__(name=name)
        self.binance_client = binance_client
        self.event_bus = event_bus
        self.symbols = symbols
        self.intervals = intervals
        self.lookback_bars = lookback_bars

        # Storage: {symbol: {interval: DataFrame}}
        self._candles: Dict[str, Dict[str, pd.DataFrame]] = {}

        # Current accumulating candles: {symbol: {interval: dict}}
        self._current_candles: Dict[str, Dict[str, Dict]] = {}

        # Statistics
        self._events_processed = 0
        self._candles_closed = 0
        
        # Store subscription for cleanup
        self._subscription = None

        self.logger = logging.getLogger(self.__class__.__name__)

    async def _start(self) -> None:
        """Start the aggregator and load historical data."""
        self.logger.info(
            f"Starting {self.name} for symbols={self.symbols}, intervals={self.intervals}"
        )

        # Initialize storage structures
        for symbol in self.symbols:
            self._candles[symbol] = {}
            self._current_candles[symbol] = {}
            for interval in self.intervals:
                self._candles[symbol][interval] = pd.DataFrame()
                self._current_candles[symbol][interval] = {}

        # Load historical data for each symbol/interval
        for symbol in self.symbols:
            for interval in self.intervals:
                await self._initialize_historical_data(symbol, interval)

        # Subscribe to MarketDataEvent
        self._subscription = await self.event_bus.subscribe(
            self._on_market_data_event, EventType.MARKET_DATA
        )

        self.logger.info(f"{self.name} started successfully")

    async def _stop(self) -> None:
        """Stop the aggregator and cleanup."""
        self.logger.info(f"Stopping {self.name}")

        # Unsubscribe from events
        if self._subscription:
            await self.event_bus.unsubscribe(self._subscription)

        self.logger.info(
            f"{self.name} stopped. Processed {self._events_processed} events, "
            f"closed {self._candles_closed} candles"
        )

    async def _initialize_historical_data(self, symbol: str, interval: str) -> None:
        """
        Load historical kline data from BinanceClient.

        Args:
            symbol: Trading symbol
            interval: Timeframe interval
        """
        try:
            self.logger.info(
                f"Loading {self.lookback_bars} historical candles for {symbol} {interval}"
            )

            # Fetch historical klines
            klines = await self.binance_client.get_historical_klines(
                symbol=symbol, interval=interval, limit=self.lookback_bars
            )

            if not klines:
                self.logger.warning(f"No historical data for {symbol} {interval}")
                return

            # Convert to DataFrame
            df = self._klines_to_dataframe(klines)
            self._candles[symbol][interval] = df

            self.logger.info(
                f"Loaded {len(df)} candles for {symbol} {interval}, "
                f"range: {df.index[0]} to {df.index[-1]}"
            )

        except Exception as e:
            self.logger.error(
                f"Error loading historical data for {symbol} {interval}: {e}"
            )

    def _klines_to_dataframe(self, klines: List[List[str]]) -> pd.DataFrame:
        """
        Convert Binance kline data to OHLCV DataFrame.

        Binance kline format:
        [timestamp, open, high, low, close, volume, close_time,
         quote_volume, trades, taker_buy_base, taker_buy_quote, ignore]

        Args:
            klines: List of kline data from Binance API

        Returns:
            DataFrame with columns: open, high, low, close, volume
            Index: datetime from timestamp
        """
        if not klines:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(
            klines,
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_volume",
                "trades",
                "taker_buy_base",
                "taker_buy_quote",
                "ignore",
            ],
        )

        # Convert types
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["close"] = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)

        # Set index and select OHLCV columns
        df.set_index("timestamp", inplace=True)
        return df[["open", "high", "low", "close", "volume"]]

    async def _on_market_data_event(self, event: MarketDataEvent) -> None:
        """
        Handle incoming MarketDataEvent.

        Updates current candle and emits CandleClosedEvent when candle closes.

        Args:
            event: MarketDataEvent containing price/volume data
        """
        try:
            self._events_processed += 1

            symbol = event.symbol
            interval = event.metadata.get("interval")
            is_closed = event.metadata.get("is_closed", False)

            # Skip if symbol/interval not configured
            if symbol not in self.symbols or interval not in self.intervals:
                return

            # Update current candle with new data
            self._update_current_candle(event)

            # If candle closed, append to DataFrame and emit event
            if is_closed:
                await self._handle_candle_closed(event)

        except Exception as e:
            self.logger.error(f"Error handling market data event: {e}", exc_info=True)

    def _update_current_candle(self, event: MarketDataEvent) -> None:
        """
        Update the current accumulating candle with new OHLCV data.

        Args:
            event: MarketDataEvent with price/volume information
        """
        symbol = event.symbol
        interval = event.metadata.get("interval")

        current = self._current_candles[symbol][interval]

        # Initialize candle if empty
        if not current:
            current["open"] = float(event.open_price or event.price)
            current["high"] = float(event.high_price or event.price)
            current["low"] = float(event.low_price or event.price)
            current["close"] = float(event.close_price or event.price)
            current["volume"] = float(event.volume)
            current["timestamp"] = event.metadata.get("open_time")
        else:
            # Update high/low/close/volume
            current["high"] = max(current["high"], float(event.high_price or event.price))
            current["low"] = min(current["low"], float(event.low_price or event.price))
            current["close"] = float(event.close_price or event.price)
            current["volume"] = float(event.volume)

    async def _handle_candle_closed(self, event: MarketDataEvent) -> None:
        """
        Handle candle closure: append to DataFrame and publish CandleClosedEvent.

        Args:
            event: MarketDataEvent marking candle closure
        """
        symbol = event.symbol
        interval = event.metadata.get("interval")
        current = self._current_candles[symbol][interval]

        if not current:
            self.logger.warning(f"No current candle for {symbol} {interval}")
            return

        # Create new row for DataFrame
        timestamp = pd.to_datetime(current["timestamp"], unit="ms")
        new_row = pd.DataFrame(
            {
                "open": [current["open"]],
                "high": [current["high"]],
                "low": [current["low"]],
                "close": [current["close"]],
                "volume": [current["volume"]],
            },
            index=[timestamp],
        )

        # Append to stored DataFrame
        df = self._candles[symbol][interval]
        self._candles[symbol][interval] = pd.concat([df, new_row])

        # Keep only last lookback_bars candles
        if len(self._candles[symbol][interval]) > self.lookback_bars:
            self._candles[symbol][interval] = self._candles[symbol][interval].iloc[
                -self.lookback_bars :
            ]

        # Clear current candle
        self._current_candles[symbol][interval] = {}

        # Publish CandleClosedEvent
        candle_event = CandleClosedEvent(
            source=self.name,
            symbol=symbol,
            interval=interval,
            df=self._candles[symbol][interval].copy(),
            timestamp=datetime.utcnow(),
        )

        await self.event_bus.publish(candle_event)

        self._candles_closed += 1
        self.logger.debug(
            f"Candle closed for {symbol} {interval}, total candles: "
            f"{len(self._candles[symbol][interval])}"
        )

    def get_candles(
        self, symbol: str, interval: str
    ) -> Optional[pd.DataFrame]:
        """
        Get current candle DataFrame for symbol/interval.

        Args:
            symbol: Trading symbol
            interval: Timeframe interval

        Returns:
            DataFrame with OHLCV data or None if not available
        """
        if symbol in self._candles and interval in self._candles[symbol]:
            return self._candles[symbol][interval].copy()
        return None

    def get_stats(self) -> Dict:
        """
        Get aggregator statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "events_processed": self._events_processed,
            "candles_closed": self._candles_closed,
            "symbols": self.symbols,
            "intervals": self.intervals,
            "candle_counts": {
                symbol: {
                    interval: len(self._candles[symbol][interval])
                    for interval in self.intervals
                }
                for symbol in self.symbols
            },
        }
