"""
Tests for BinanceClient wrapper class.

Test cases cover connection management, API operations,
error handling, and rate limiting compliance.
"""

from decimal import Decimal
from unittest.mock import AsyncMock, patch

import pytest

from trading_bot.core.base_component import ComponentState
from trading_bot.data.binance_client import BinanceClient, BinanceClientError


class TestBinanceClient:
    """Test suite for BinanceClient wrapper."""

    @pytest.fixture
    async def mock_async_client(self):
        """Mock AsyncClient for testing."""
        client = AsyncMock()
        client.ping = AsyncMock()
        client.get_server_time = AsyncMock(return_value={"serverTime": 1640995200000})
        client.futures_exchange_info = AsyncMock(
            return_value={
                "symbols": [
                    {"symbol": "BTCUSDT", "status": "TRADING"},
                    {"symbol": "ETHUSDT", "status": "TRADING"},
                ]
            }
        )
        client.close_connection = AsyncMock()
        return client

    @pytest.fixture
    async def binance_client(self, mock_async_client):
        """Create BinanceClient instance for testing."""
        with patch(
            "trading_bot.data.binance_client.AsyncClient.create",
            return_value=mock_async_client,
        ):
            client = BinanceClient(
                api_key="test_key", api_secret="test_secret", testnet=True
            )
            await client.start()
            yield client
            await client.stop()

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test client initialization."""
        client = BinanceClient(
            api_key="test_key", api_secret="test_secret", testnet=True
        )

        assert client.name == "BinanceClient"
        assert client.api_key == "test_key"
        assert client.api_secret == "test_secret"
        assert client.testnet is True
        assert client.state == ComponentState.INITIALIZED

    @pytest.mark.asyncio
    async def test_connection_lifecycle(self, mock_async_client):
        """Test connection establishment and cleanup."""
        with patch(
            "trading_bot.data.binance_client.AsyncClient.create",
            return_value=mock_async_client,
        ):
            client = BinanceClient(api_key="test_key", api_secret="test_secret")

            # Test start
            await client.start()
            assert client.is_running()
            assert client.is_connected()
            mock_async_client.ping.assert_called_once()

            # Test stop
            await client.stop()
            assert not client.is_running()
            assert not client.is_connected()
            mock_async_client.close_connection.assert_called_once()

    @pytest.mark.asyncio
    async def test_connection_failure(self):
        """Test connection failure handling."""
        with patch(
            "trading_bot.data.binance_client.AsyncClient.create",
            side_effect=Exception("Connection failed"),
        ):
            client = BinanceClient(api_key="test_key", api_secret="test_secret")

            with pytest.raises(BinanceClientError):
                await client.start()

            assert client.state == ComponentState.ERROR

    @pytest.mark.asyncio
    async def test_get_symbol_info(self, binance_client, mock_async_client):
        """Test symbol information retrieval."""
        symbol_info = await binance_client.get_symbol_info("BTCUSDT")

        assert symbol_info == {"symbol": "BTCUSDT", "status": "TRADING"}
        mock_async_client.futures_exchange_info.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_symbol_info_not_found(self, binance_client):
        """Test symbol not found error."""
        with pytest.raises(BinanceClientError, match="Symbol INVALID not found"):
            await binance_client.get_symbol_info("INVALID")

    @pytest.mark.asyncio
    async def test_get_ticker_price(self, binance_client, mock_async_client):
        """Test ticker price retrieval."""
        mock_async_client.futures_symbol_ticker = AsyncMock(
            return_value={"price": "50000.00"}
        )

        price = await binance_client.get_ticker_price("BTCUSDT")

        assert price == Decimal("50000.00")
        mock_async_client.futures_symbol_ticker.assert_called_once_with(
            symbol="BTCUSDT"
        )

    @pytest.mark.asyncio
    async def test_get_orderbook(self, binance_client, mock_async_client):
        """Test order book retrieval."""
        mock_orderbook = {"bids": [["50000.00", "1.0"]], "asks": [["50001.00", "1.0"]]}
        mock_async_client.futures_order_book = AsyncMock(return_value=mock_orderbook)

        orderbook = await binance_client.get_orderbook("BTCUSDT", limit=100)

        assert orderbook == mock_orderbook
        mock_async_client.futures_order_book.assert_called_once_with(
            symbol="BTCUSDT", limit=100
        )

    @pytest.mark.asyncio
    async def test_get_historical_klines(self, binance_client, mock_async_client):
        """Test historical klines retrieval."""
        mock_klines = [
            ["1640995200000", "50000.00", "50100.00", "49900.00", "50050.00", "100.0"]
        ]
        mock_async_client.futures_klines = AsyncMock(return_value=mock_klines)

        klines = await binance_client.get_historical_klines("BTCUSDT", "5m", limit=100)

        assert klines == mock_klines
        mock_async_client.futures_klines.assert_called_once()

    @pytest.mark.asyncio
    async def test_unsupported_interval(self, binance_client):
        """Test unsupported interval error."""
        with pytest.raises(BinanceClientError, match="Unsupported interval: 1m"):
            await binance_client.get_historical_klines("BTCUSDT", "1m")

    @pytest.mark.asyncio
    async def test_account_info_without_credentials(self):
        """Test account info without API credentials."""
        client = BinanceClient()

        with patch("trading_bot.data.binance_client.AsyncClient.create") as mock_create:
            mock_create.return_value = AsyncMock()
            await client.start()

            with pytest.raises(BinanceClientError, match="API credentials required"):
                await client.get_account_info()

            await client.stop()

    @pytest.mark.asyncio
    async def test_account_info_with_credentials(
        self, binance_client, mock_async_client
    ):
        """Test account info with valid credentials."""
        mock_account = {
            "totalWalletBalance": "1000.00",
            "assets": [
                {
                    "asset": "USDT",
                    "availableBalance": "1000.00",
                    "initialMargin": "0.00",
                }
            ],
        }
        mock_async_client.futures_account = AsyncMock(return_value=mock_account)

        account = await binance_client.get_account_info()

        assert account == mock_account
        mock_async_client.futures_account.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_balance(self, binance_client, mock_async_client):
        """Test balance retrieval."""
        mock_account = {
            "assets": [
                {
                    "asset": "USDT",
                    "availableBalance": "1000.00",
                    "initialMargin": "100.00",
                },
                {"asset": "BTC", "availableBalance": "0.1", "initialMargin": "0.01"},
            ]
        }
        mock_async_client.futures_account = AsyncMock(return_value=mock_account)

        balances = await binance_client.get_balance()

        expected_balances = {
            "USDT": {"free": "1000.00", "locked": "100.00", "total": "1100.00"},
            "BTC": {"free": "0.1", "locked": "0.01", "total": "0.11"},
        }
        assert balances == expected_balances

    @pytest.mark.asyncio
    async def test_place_order(self, binance_client, mock_async_client):
        """Test order placement."""
        mock_order = {"orderId": "12345", "symbol": "BTCUSDT", "status": "NEW"}
        mock_async_client.futures_create_order = AsyncMock(return_value=mock_order)

        order = await binance_client.place_order(
            symbol="BTCUSDT", side="BUY", order_type="MARKET", quantity="0.001"
        )

        assert order == mock_order
        mock_async_client.futures_create_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_order(self, binance_client, mock_async_client):
        """Test order cancellation."""
        mock_result = {"orderId": "12345", "status": "CANCELED"}
        mock_async_client.futures_cancel_order = AsyncMock(return_value=mock_result)

        result = await binance_client.cancel_order("BTCUSDT", "12345")

        assert result == mock_result
        mock_async_client.futures_cancel_order.assert_called_once_with(
            symbol="BTCUSDT", orderId="12345"
        )

    @pytest.mark.asyncio
    async def test_get_open_orders(self, binance_client, mock_async_client):
        """Test open orders retrieval."""
        mock_orders = [{"orderId": "12345", "symbol": "BTCUSDT", "status": "NEW"}]
        mock_async_client.futures_get_open_orders = AsyncMock(return_value=mock_orders)

        orders = await binance_client.get_open_orders("BTCUSDT")

        assert orders == mock_orders
        mock_async_client.futures_get_open_orders.assert_called_once_with(
            symbol="BTCUSDT"
        )

    @pytest.mark.asyncio
    async def test_api_error_handling(self, binance_client, mock_async_client):
        """Test API error handling."""
        from binance.exceptions import BinanceAPIException

        mock_async_client.futures_symbol_ticker.side_effect = BinanceAPIException(
            None, "API error", None, None
        )

        with pytest.raises(BinanceClientError, match="API error"):
            await binance_client.get_ticker_price("BTCUSDT")

    @pytest.mark.asyncio
    async def test_connection_status(self, binance_client):
        """Test connection status reporting."""
        status = binance_client.get_connection_status()

        assert status["connected"] is True
        assert status["testnet"] is True
        assert status["connection_retries"] == 0
        assert status["max_retries"] == 5
        assert status["last_heartbeat"] is not None

    @pytest.mark.asyncio
    async def test_not_connected_operations(self):
        """Test operations when client is not connected."""
        client = BinanceClient()

        with pytest.raises(BinanceClientError, match="Client not connected"):
            await client.get_ticker_price("BTCUSDT")

        with pytest.raises(BinanceClientError, match="Client not connected"):
            await client.get_orderbook("BTCUSDT")

        with pytest.raises(BinanceClientError, match="Client not connected"):
            await client.get_historical_klines("BTCUSDT", "5m")

    @pytest.mark.asyncio
    async def test_client_without_credentials(self):
        """Test client operations without API credentials."""
        with patch("trading_bot.data.binance_client.AsyncClient.create") as mock_create:
            mock_client = AsyncMock()
            mock_client.ping = AsyncMock()
            mock_client.get_server_time = AsyncMock(
                return_value={"serverTime": 1640995200000}
            )
            mock_client.close_connection = AsyncMock()
            mock_create.return_value = mock_client

            client = BinanceClient()  # No API credentials
            await client.start()

            # Public operations should work
            assert client.is_connected()

            # Private operations should fail
            with pytest.raises(BinanceClientError, match="API credentials required"):
                await client.get_account_info()

            with pytest.raises(BinanceClientError, match="API credentials required"):
                await client.place_order("BTCUSDT", "BUY", "MARKET", "0.001")

            await client.stop()
