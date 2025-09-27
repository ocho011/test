"""
Tests for MarketDataProvider WebSocket data streaming.

Test cases cover WebSocket connections, subscription management,
data streaming, auto-reconnection, and event publishing.
"""

import pytest
import asyncio
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from trading_bot.data.market_data_provider import (
    MarketDataProvider, StreamType, StreamSubscription
)
from trading_bot.core.base_component import ComponentState
from trading_bot.core.events import MarketDataEvent


class TestMarketDataProvider:
    """Test suite for MarketDataProvider WebSocket streaming."""

    @pytest.fixture
    async def mock_websocket(self):
        """Mock BinanceSocketManager for testing."""
        socket_manager = AsyncMock()
        socket_manager.start = AsyncMock()
        socket_manager.stop = AsyncMock()

        # Mock kline socket
        mock_socket = AsyncMock()
        mock_socket.__aenter__ = AsyncMock(return_value=mock_socket)
        mock_socket.__aexit__ = AsyncMock(return_value=None)
        socket_manager.kline_socket.return_value = mock_socket

        return socket_manager

    @pytest.fixture
    async def market_provider(self, mock_websocket):
        """Create MarketDataProvider instance for testing."""
        with patch('trading_bot.data.market_data_provider.BinanceSocketManager', return_value=mock_websocket):
            provider = MarketDataProvider()
            await provider.start()
            yield provider
            await provider.stop()

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test provider initialization."""
        provider = MarketDataProvider()

        assert provider.name == "MarketDataProvider"
        assert provider.state == ComponentState.INITIALIZED
        assert len(provider._subscriptions) == 0
        assert not provider.is_connected()

    @pytest.mark.asyncio
    async def test_connection_lifecycle(self, mock_websocket):
        """Test connection establishment and cleanup."""
        with patch('trading_bot.data.market_data_provider.BinanceSocketManager', return_value=mock_websocket):
            provider = MarketDataProvider()

            # Test start
            await provider.start()
            assert provider.is_running()
            assert provider.is_connected()
            mock_websocket.start.assert_called_once()

            # Test stop
            await provider.stop()
            assert not provider.is_running()
            assert not provider.is_connected()
            mock_websocket.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_subscribe_kline_data(self, market_provider, mock_websocket):
        """Test kline data subscription."""
        subscription = await market_provider.subscribe_kline_data("BTCUSDT", "5m")

        assert isinstance(subscription, StreamSubscription)
        assert subscription.symbol == "BTCUSDT"
        assert subscription.interval == "5m"
        assert subscription.stream_type == StreamType.KLINE
        assert subscription in market_provider._subscriptions

    @pytest.mark.asyncio
    async def test_subscribe_ticker_data(self, market_provider, mock_websocket):
        """Test ticker data subscription."""
        subscription = await market_provider.subscribe_ticker_data("ETHUSDT")

        assert isinstance(subscription, StreamSubscription)
        assert subscription.symbol == "ETHUSDT"
        assert subscription.stream_type == StreamType.TICKER
        assert subscription in market_provider._subscriptions

    @pytest.mark.asyncio
    async def test_unsupported_interval_error(self, market_provider):
        """Test error for unsupported interval."""
        with pytest.raises(ValueError, match="Unsupported interval: 1m"):
            await market_provider.subscribe_kline_data("BTCUSDT", "1m")

    @pytest.mark.asyncio
    async def test_duplicate_subscription_prevention(self, market_provider):
        """Test prevention of duplicate subscriptions."""
        # First subscription should work
        sub1 = await market_provider.subscribe_kline_data("BTCUSDT", "5m")

        # Second identical subscription should return existing
        sub2 = await market_provider.subscribe_kline_data("BTCUSDT", "5m")

        assert sub1 is sub2
        assert len(market_provider._subscriptions) == 1

    @pytest.mark.asyncio
    async def test_unsubscribe(self, market_provider):
        """Test unsubscribing from streams."""
        subscription = await market_provider.subscribe_kline_data("BTCUSDT", "5m")

        await market_provider.unsubscribe(subscription)

        assert subscription not in market_provider._subscriptions

    @pytest.mark.asyncio
    async def test_unsubscribe_all(self, market_provider):
        """Test unsubscribing from all streams."""
        await market_provider.subscribe_kline_data("BTCUSDT", "5m")
        await market_provider.subscribe_ticker_data("ETHUSDT")

        assert len(market_provider._subscriptions) == 2

        await market_provider.unsubscribe_all()

        assert len(market_provider._subscriptions) == 0

    @pytest.mark.asyncio
    async def test_kline_message_handling(self, market_provider):
        """Test kline message processing and event publishing."""
        mock_event_bus = Mock()
        market_provider._event_bus = mock_event_bus

        # Mock kline message
        kline_msg = {
            'stream': 'btcusdt@kline_5m',
            'data': {
                'k': {
                    's': 'BTCUSDT',
                    'i': '5m',
                    'o': '50000.00',
                    'h': '50100.00',
                    'l': '49900.00',
                    'c': '50050.00',
                    'v': '100.50',
                    'q': '5005000.00',
                    't': 1640995200000,
                    'T': 1640995499999,
                    'x': True
                }
            }
        }

        # Add subscription to handle the message
        await market_provider.subscribe_kline_data("BTCUSDT", "5m")

        # Process message
        await market_provider._handle_kline_message(kline_msg, "BTCUSDT_5m")

        # Verify event was published
        mock_event_bus.publish.assert_called_once()
        event = mock_event_bus.publish.call_args[0][0]

        assert isinstance(event, MarketDataEvent)
        assert event.symbol == "BTCUSDT"
        assert event.price == Decimal('50050.00')
        assert event.volume == Decimal('100.50')

    @pytest.mark.asyncio
    async def test_ticker_message_handling(self, market_provider):
        """Test ticker message processing and event publishing."""
        mock_event_bus = Mock()
        market_provider._event_bus = mock_event_bus

        # Mock ticker message
        ticker_msg = {
            'stream': 'btcusdt@ticker',
            'data': {
                's': 'BTCUSDT',
                'c': '50000.00',
                'v': '1000.50',
                'q': '50005000.00',
                'P': '1.25',
                'p': '625.00'
            }
        }

        # Add subscription to handle the message
        await market_provider.subscribe_ticker_data("BTCUSDT")

        # Process message
        await market_provider._handle_ticker_message(ticker_msg, "BTCUSDT_ticker")

        # Verify event was published
        mock_event_bus.publish.assert_called_once()
        event = mock_event_bus.publish.call_args[0][0]

        assert isinstance(event, MarketDataEvent)
        assert event.symbol == "BTCUSDT"
        assert event.price == Decimal('50000.00')
        assert event.volume == Decimal('1000.50')

    @pytest.mark.asyncio
    async def test_connection_error_handling(self):
        """Test handling of WebSocket connection errors."""
        with patch('trading_bot.data.market_data_provider.BinanceSocketManager') as mock_manager:
            mock_manager.side_effect = Exception("Connection failed")

            provider = MarketDataProvider()

            # Should handle connection error gracefully
            await provider.start()
            assert provider.state == ComponentState.ERROR

    @pytest.mark.asyncio
    async def test_reconnection_logic(self, market_provider):
        """Test automatic reconnection logic."""
        original_connect = market_provider._connect_websocket
        connect_calls = []

        async def mock_connect():
            connect_calls.append(1)
            if len(connect_calls) < 3:
                raise Exception("Connection failed")
            await original_connect()

        market_provider._connect_websocket = mock_connect

        # Trigger reconnection
        await market_provider._handle_reconnection()

        # Should have attempted multiple connections
        assert len(connect_calls) >= 2

    @pytest.mark.asyncio
    async def test_subscription_management_during_reconnection(self, market_provider):
        """Test subscription restoration after reconnection."""
        # Add subscriptions
        await market_provider.subscribe_kline_data("BTCUSDT", "5m")
        await market_provider.subscribe_ticker_data("ETHUSDT")

        initial_count = len(market_provider._subscriptions)

        # Simulate disconnection and reconnection
        market_provider._connected = False
        await market_provider._handle_reconnection()

        # Subscriptions should be restored
        assert len(market_provider._subscriptions) == initial_count

    @pytest.mark.asyncio
    async def test_malformed_message_handling(self, market_provider):
        """Test handling of malformed WebSocket messages."""
        # Invalid kline message
        invalid_msg = {
            'stream': 'btcusdt@kline_5m',
            'data': {
                'invalid': 'structure'
            }
        }

        # Should handle gracefully without crashing
        await market_provider._handle_kline_message(invalid_msg, "BTCUSDT_5m")

        # Provider should still be running
        assert market_provider.is_running()

    @pytest.mark.asyncio
    async def test_subscription_status_tracking(self, market_provider):
        """Test subscription status tracking."""
        subscription = await market_provider.subscribe_kline_data("BTCUSDT", "5m")

        # Initially active
        assert subscription.is_active()

        # Pause subscription
        subscription.pause()
        assert not subscription.is_active()

        # Resume subscription
        subscription.resume()
        assert subscription.is_active()

    @pytest.mark.asyncio
    async def test_get_active_subscriptions(self, market_provider):
        """Test getting active subscriptions."""
        sub1 = await market_provider.subscribe_kline_data("BTCUSDT", "5m")
        sub2 = await market_provider.subscribe_ticker_data("ETHUSDT")

        active_subs = market_provider.get_active_subscriptions()
        assert len(active_subs) == 2
        assert sub1 in active_subs
        assert sub2 in active_subs

        # Pause one subscription
        sub1.pause()
        active_subs = market_provider.get_active_subscriptions()
        assert len(active_subs) == 1
        assert sub2 in active_subs

    @pytest.mark.asyncio
    async def test_connection_status_reporting(self, market_provider):
        """Test connection status reporting."""
        status = market_provider.get_connection_status()

        assert status['connected'] is True
        assert status['subscriptions_count'] == 0
        assert status['connection_retries'] >= 0
        assert status['last_heartbeat'] is not None

    @pytest.mark.asyncio
    async def test_not_connected_operations(self):
        """Test operations when provider is not connected."""
        provider = MarketDataProvider()

        with pytest.raises(RuntimeError, match="not connected"):
            await provider.subscribe_kline_data("BTCUSDT", "5m")

        with pytest.raises(RuntimeError, match="not connected"):
            await provider.subscribe_ticker_data("ETHUSDT")

    @pytest.mark.asyncio
    async def test_multiple_symbols_same_interval(self, market_provider):
        """Test subscribing to multiple symbols with same interval."""
        sub1 = await market_provider.subscribe_kline_data("BTCUSDT", "5m")
        sub2 = await market_provider.subscribe_kline_data("ETHUSDT", "5m")

        assert sub1 != sub2
        assert len(market_provider._subscriptions) == 2
        assert sub1.symbol == "BTCUSDT"
        assert sub2.symbol == "ETHUSDT"
        assert sub1.interval == sub2.interval == "5m"

    @pytest.mark.asyncio
    async def test_heartbeat_mechanism(self, market_provider):
        """Test WebSocket heartbeat mechanism."""
        initial_heartbeat = market_provider._last_heartbeat

        # Simulate heartbeat update
        await asyncio.sleep(0.1)
        market_provider._update_heartbeat()

        assert market_provider._last_heartbeat > initial_heartbeat

    @pytest.mark.asyncio
    async def test_resource_cleanup(self, market_provider):
        """Test proper resource cleanup on shutdown."""
        # Add subscriptions and connections
        await market_provider.subscribe_kline_data("BTCUSDT", "5m")
        await market_provider.subscribe_ticker_data("ETHUSDT")

        initial_subs = len(market_provider._subscriptions)
        assert initial_subs > 0

        # Stop provider
        await market_provider.stop()

        # Resources should be cleaned up
        assert not market_provider.is_connected()
        assert not market_provider.is_running()