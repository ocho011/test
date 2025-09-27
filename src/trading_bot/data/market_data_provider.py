"""
Market data provider with real-time WebSocket streams and auto-reconnection.

This module provides real-time market data streaming from Binance with
automatic reconnection, data normalization, and event publishing.
"""

import asyncio
import json
import logging
import time
from decimal import Decimal
from typing import Dict, List, Optional, Set, Any, Callable
from dataclasses import dataclass
from enum import Enum

from binance import BinanceSocketManager
from binance.exceptions import BinanceWebsocketUnableToConnect

from ..core.base_component import BaseComponent
from ..core.events import MarketDataEvent, EventPriority
from .binance_client import BinanceClient


class StreamType(Enum):
    """WebSocket stream types."""
    KLINE = "kline"
    TICKER = "ticker"
    DEPTH = "depth"
    TRADE = "trade"


@dataclass
class StreamSubscription:
    """WebSocket stream subscription configuration."""
    symbol: str
    stream_type: StreamType
    interval: Optional[str] = None  # For kline streams
    callback: Optional[Callable] = None
    active: bool = True


class ConnectionState(Enum):
    """WebSocket connection states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"


class MarketDataProvider(BaseComponent):
    """
    Real-time market data provider with WebSocket streams.

    Manages multiple WebSocket connections for different data types
    with automatic reconnection using exponential backoff.
    """

    def __init__(
        self,
        binance_client: BinanceClient,
        event_bus=None,
        max_reconnect_attempts: int = 10
    ):
        """
        Initialize market data provider.

        Args:
            binance_client: Binance client instance
            event_bus: Event bus for publishing market data
            max_reconnect_attempts: Maximum reconnection attempts
        """
        super().__init__("MarketDataProvider")

        self.binance_client = binance_client
        self.event_bus = event_bus
        self.max_reconnect_attempts = max_reconnect_attempts

        # Stream management
        self._subscriptions: Dict[str, StreamSubscription] = {}
        self._active_streams: Dict[str, Any] = {}
        self._socket_manager: Optional[BinanceSocketManager] = None

        # Connection state management
        self._connection_state = ConnectionState.DISCONNECTED
        self._reconnect_attempts = 0
        self._last_reconnect_time = 0
        self._backoff_factor = 2.0
        self._base_delay = 1.0
        self._max_delay = 60.0

        # Health monitoring
        self._last_message_time = 0
        self._heartbeat_interval = 30.0
        self._message_count = 0
        self._error_count = 0

        # Tasks
        self._connection_monitor_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None

        # Supported timeframes
        self.supported_intervals = ['5m', '15m', '4h', '1d']

    async def _start(self) -> None:
        """Start the market data provider."""
        if not self.binance_client.is_connected():
            raise RuntimeError("Binance client must be connected before starting MarketDataProvider")

        self._socket_manager = BinanceSocketManager(self.binance_client._client)
        self._connection_state = ConnectionState.CONNECTING

        # Start monitoring tasks
        self._connection_monitor_task = asyncio.create_task(self._monitor_connection())
        self._heartbeat_task = asyncio.create_task(self._heartbeat_monitor())

        self.logger.info("MarketDataProvider started")

    async def _stop(self) -> None:
        """Stop the market data provider."""
        self._connection_state = ConnectionState.DISCONNECTED

        # Cancel monitoring tasks
        if self._connection_monitor_task:
            self._connection_monitor_task.cancel()
            try:
                await self._connection_monitor_task
            except asyncio.CancelledError:
                pass

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        # Close all active streams
        await self._close_all_streams()

        self.logger.info("MarketDataProvider stopped")

    async def subscribe_klines(
        self,
        symbol: str,
        intervals: List[str],
        callback: Optional[Callable] = None
    ) -> bool:
        """
        Subscribe to kline/candlestick data streams.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            intervals: List of intervals to subscribe to
            callback: Optional callback for processed data

        Returns:
            True if subscription successful
        """
        symbol = symbol.upper()

        for interval in intervals:
            if interval not in self.supported_intervals:
                self.logger.error(f"Unsupported interval: {interval}")
                continue

            subscription_key = f"{symbol}_{interval}_kline"

            subscription = StreamSubscription(
                symbol=symbol,
                stream_type=StreamType.KLINE,
                interval=interval,
                callback=callback
            )

            self._subscriptions[subscription_key] = subscription

            try:
                await self._create_kline_stream(symbol, interval, subscription_key)
                self.logger.info(f"Subscribed to {symbol} {interval} klines")

            except Exception as e:
                self.logger.error(f"Failed to subscribe to {symbol} {interval} klines: {e}")
                del self._subscriptions[subscription_key]
                return False

        return True

    async def subscribe_ticker(self, symbol: str, callback: Optional[Callable] = None) -> bool:
        """
        Subscribe to ticker price stream.

        Args:
            symbol: Trading symbol
            callback: Optional callback for ticker data

        Returns:
            True if subscription successful
        """
        symbol = symbol.upper()
        subscription_key = f"{symbol}_ticker"

        subscription = StreamSubscription(
            symbol=symbol,
            stream_type=StreamType.TICKER,
            callback=callback
        )

        self._subscriptions[subscription_key] = subscription

        try:
            await self._create_ticker_stream(symbol, subscription_key)
            self.logger.info(f"Subscribed to {symbol} ticker")
            return True

        except Exception as e:
            self.logger.error(f"Failed to subscribe to {symbol} ticker: {e}")
            del self._subscriptions[subscription_key]
            return False

    async def unsubscribe(self, subscription_key: str) -> bool:
        """
        Unsubscribe from a stream.

        Args:
            subscription_key: Key identifying the subscription

        Returns:
            True if unsubscription successful
        """
        if subscription_key not in self._subscriptions:
            self.logger.warning(f"Subscription {subscription_key} not found")
            return False

        try:
            # Close the stream
            if subscription_key in self._active_streams:
                stream = self._active_streams[subscription_key]
                await stream.close()
                del self._active_streams[subscription_key]

            # Remove subscription
            del self._subscriptions[subscription_key]

            self.logger.info(f"Unsubscribed from {subscription_key}")
            return True

        except Exception as e:
            self.logger.error(f"Error unsubscribing from {subscription_key}: {e}")
            return False

    async def _create_kline_stream(self, symbol: str, interval: str, subscription_key: str) -> None:
        """Create kline WebSocket stream."""
        if not self._socket_manager:
            raise RuntimeError("Socket manager not initialized")

        # Map intervals to Binance constants
        interval_map = {
            '5m': '5m',
            '15m': '15m',
            '4h': '4h',
            '1d': '1d'
        }

        stream = self._socket_manager.kline_futures_socket(
            symbol=symbol.lower(),
            interval=interval_map[interval]
        )

        self._active_streams[subscription_key] = stream

        # Start processing the stream
        asyncio.create_task(self._process_kline_stream(stream, subscription_key))

    async def _create_ticker_stream(self, symbol: str, subscription_key: str) -> None:
        """Create ticker WebSocket stream."""
        if not self._socket_manager:
            raise RuntimeError("Socket manager not initialized")

        stream = self._socket_manager.symbol_ticker_futures_socket(symbol=symbol.lower())
        self._active_streams[subscription_key] = stream

        # Start processing the stream
        asyncio.create_task(self._process_ticker_stream(stream, subscription_key))

    async def _process_kline_stream(self, stream, subscription_key: str) -> None:
        """Process kline stream messages."""
        try:
            async with stream as stream_context:
                self._connection_state = ConnectionState.CONNECTED
                self._reconnect_attempts = 0

                async for msg in stream_context:
                    if not self._subscriptions.get(subscription_key, {}).get('active', False):
                        break

                    await self._handle_kline_message(msg, subscription_key)
                    self._last_message_time = time.time()
                    self._message_count += 1

        except Exception as e:
            self.logger.error(f"Error in kline stream {subscription_key}: {e}")
            self._error_count += 1
            await self._handle_stream_error(subscription_key, e)

    async def _process_ticker_stream(self, stream, subscription_key: str) -> None:
        """Process ticker stream messages."""
        try:
            async with stream as stream_context:
                self._connection_state = ConnectionState.CONNECTED
                self._reconnect_attempts = 0

                async for msg in stream_context:
                    if not self._subscriptions.get(subscription_key, {}).get('active', False):
                        break

                    await self._handle_ticker_message(msg, subscription_key)
                    self._last_message_time = time.time()
                    self._message_count += 1

        except Exception as e:
            self.logger.error(f"Error in ticker stream {subscription_key}: {e}")
            self._error_count += 1
            await self._handle_stream_error(subscription_key, e)

    async def _handle_kline_message(self, msg: Dict, subscription_key: str) -> None:
        """Handle kline message and publish market data event."""
        try:
            if msg.get('e') != 'kline':
                return

            kline_data = msg.get('k', {})
            subscription = self._subscriptions.get(subscription_key)

            if not subscription:
                return

            # Create market data event
            event = MarketDataEvent(
                source=self.name,
                symbol=kline_data.get('s'),
                price=Decimal(kline_data.get('c')),  # Close price
                volume=Decimal(kline_data.get('v')),  # Volume
                open_price=Decimal(kline_data.get('o')),
                high_price=Decimal(kline_data.get('h')),
                low_price=Decimal(kline_data.get('l')),
                close_price=Decimal(kline_data.get('c')),
                priority=EventPriority.HIGH,
                metadata={
                    'interval': subscription.interval,
                    'is_closed': kline_data.get('x', False),  # Is kline closed
                    'open_time': kline_data.get('t'),
                    'close_time': kline_data.get('T'),
                    'trades': kline_data.get('n'),  # Number of trades
                    'stream_type': 'kline'
                }
            )

            # Publish event
            if self.event_bus:
                await self.event_bus.publish(event)

            # Call custom callback if provided
            if subscription.callback:
                try:
                    await subscription.callback(event)
                except Exception as e:
                    self.logger.error(f"Error in callback for {subscription_key}: {e}")

        except Exception as e:
            self.logger.error(f"Error handling kline message: {e}")

    async def _handle_ticker_message(self, msg: Dict, subscription_key: str) -> None:
        """Handle ticker message and publish market data event."""
        try:
            if msg.get('e') != '24hrTicker':
                return

            subscription = self._subscriptions.get(subscription_key)
            if not subscription:
                return

            # Create market data event
            event = MarketDataEvent(
                source=self.name,
                symbol=msg.get('s'),
                price=Decimal(msg.get('c')),  # Current price
                volume=Decimal(msg.get('v')),  # 24h volume
                bid=Decimal(msg.get('b')) if msg.get('b') else None,
                ask=Decimal(msg.get('a')) if msg.get('a') else None,
                open_price=Decimal(msg.get('o')),
                high_price=Decimal(msg.get('h')),
                low_price=Decimal(msg.get('l')),
                close_price=Decimal(msg.get('c')),
                priority=EventPriority.NORMAL,
                metadata={
                    'price_change': msg.get('p'),
                    'price_change_percent': msg.get('P'),
                    'weighted_avg_price': msg.get('w'),
                    'prev_close_price': msg.get('x'),
                    'last_qty': msg.get('q'),
                    'count': msg.get('n'),  # Trade count
                    'stream_type': 'ticker'
                }
            )

            # Publish event
            if self.event_bus:
                await self.event_bus.publish(event)

            # Call custom callback if provided
            if subscription.callback:
                try:
                    await subscription.callback(event)
                except Exception as e:
                    self.logger.error(f"Error in callback for {subscription_key}: {e}")

        except Exception as e:
            self.logger.error(f"Error handling ticker message: {e}")

    async def _handle_stream_error(self, subscription_key: str, error: Exception) -> None:
        """Handle stream errors and trigger reconnection."""
        self.logger.error(f"Stream error for {subscription_key}: {error}")

        if self._connection_state != ConnectionState.FAILED:
            self._connection_state = ConnectionState.RECONNECTING
            await self._schedule_reconnect(subscription_key)

    async def _schedule_reconnect(self, subscription_key: str) -> None:
        """Schedule reconnection with exponential backoff."""
        if self._reconnect_attempts >= self.max_reconnect_attempts:
            self.logger.error(f"Max reconnect attempts reached for {subscription_key}")
            self._connection_state = ConnectionState.FAILED
            return

        self._reconnect_attempts += 1

        # Calculate delay with exponential backoff
        delay = min(
            self._base_delay * (self._backoff_factor ** (self._reconnect_attempts - 1)),
            self._max_delay
        )

        self.logger.info(f"Scheduling reconnect for {subscription_key} in {delay:.1f}s (attempt {self._reconnect_attempts})")

        await asyncio.sleep(delay)

        # Attempt reconnection
        await self._reconnect_stream(subscription_key)

    async def _reconnect_stream(self, subscription_key: str) -> None:
        """Reconnect a specific stream."""
        subscription = self._subscriptions.get(subscription_key)
        if not subscription or not subscription.active:
            return

        try:
            self.logger.info(f"Reconnecting stream: {subscription_key}")

            # Close existing stream if any
            if subscription_key in self._active_streams:
                try:
                    await self._active_streams[subscription_key].close()
                except:
                    pass
                del self._active_streams[subscription_key]

            # Recreate stream based on type
            if subscription.stream_type == StreamType.KLINE:
                await self._create_kline_stream(
                    subscription.symbol,
                    subscription.interval,
                    subscription_key
                )
            elif subscription.stream_type == StreamType.TICKER:
                await self._create_ticker_stream(
                    subscription.symbol,
                    subscription_key
                )

            self.logger.info(f"Successfully reconnected {subscription_key}")

        except Exception as e:
            self.logger.error(f"Failed to reconnect {subscription_key}: {e}")
            # Schedule another reconnect attempt
            await self._schedule_reconnect(subscription_key)

    async def _monitor_connection(self) -> None:
        """Monitor connection health and trigger reconnections if needed."""
        while self._connection_state != ConnectionState.DISCONNECTED:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds

                # Check if we've received messages recently
                if self._last_message_time > 0:
                    time_since_last_message = time.time() - self._last_message_time

                    # If no messages for too long, consider connection stale
                    if time_since_last_message > 60:
                        self.logger.warning("No messages received recently, checking connection")
                        # Trigger reconnection for all streams
                        for subscription_key in list(self._subscriptions.keys()):
                            if self._subscriptions[subscription_key].active:
                                await self._handle_stream_error(subscription_key, Exception("Connection timeout"))

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in connection monitor: {e}")

    async def _heartbeat_monitor(self) -> None:
        """Monitor and log heartbeat statistics."""
        while self._connection_state != ConnectionState.DISCONNECTED:
            try:
                await asyncio.sleep(self._heartbeat_interval)

                stats = self.get_stats()
                self.logger.debug(f"MarketDataProvider stats: {stats}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in heartbeat monitor: {e}")

    async def _close_all_streams(self) -> None:
        """Close all active streams."""
        for subscription_key, stream in list(self._active_streams.items()):
            try:
                await stream.close()
                self.logger.debug(f"Closed stream: {subscription_key}")
            except Exception as e:
                self.logger.error(f"Error closing stream {subscription_key}: {e}")

        self._active_streams.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get market data provider statistics."""
        return {
            'connection_state': self._connection_state.value,
            'active_subscriptions': len([s for s in self._subscriptions.values() if s.active]),
            'total_subscriptions': len(self._subscriptions),
            'active_streams': len(self._active_streams),
            'reconnect_attempts': self._reconnect_attempts,
            'max_reconnect_attempts': self.max_reconnect_attempts,
            'message_count': self._message_count,
            'error_count': self._error_count,
            'last_message_time': self._last_message_time,
            'time_since_last_message': time.time() - self._last_message_time if self._last_message_time > 0 else 0
        }

    def get_subscriptions(self) -> Dict[str, Dict[str, Any]]:
        """Get current subscriptions status."""
        return {
            key: {
                'symbol': sub.symbol,
                'stream_type': sub.stream_type.value,
                'interval': sub.interval,
                'active': sub.active
            }
            for key, sub in self._subscriptions.items()
        }