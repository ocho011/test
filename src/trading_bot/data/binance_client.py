"""
Binance API client wrapper for futures trading.

This module provides a wrapper around the python-binance library with
async support, error handling, and logging for futures trading operations.
"""

import asyncio
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union

from binance import AsyncClient, BinanceSocketManager
from binance.enums import (
    KLINE_INTERVAL_1DAY,
    KLINE_INTERVAL_4HOUR,
    KLINE_INTERVAL_5MINUTE,
    KLINE_INTERVAL_15MINUTE,
)
from binance.exceptions import BinanceAPIException

from ..core.base_component import BaseComponent


# Type checking imports
if False:  # TYPE_CHECKING
    from ..config.models import BinanceConfig


class BinanceClientError(Exception):
    """Custom exception for Binance client errors."""

    pass


class BinanceClient(BaseComponent):
    """
    Asynchronous Binance futures client wrapper.

    Provides methods for market data retrieval, order management,
    and account information with proper error handling and logging.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = False,
        event_bus=None,
        config: Optional["BinanceConfig"] = None,
    ):
        """
        Initialize the Binance client.

        Args:
            api_key: Binance API key (deprecated - use config)
            api_secret: Binance API secret (deprecated - use config)
            testnet: Whether to use testnet (deprecated - use config)
            event_bus: Event bus for publishing market data events
            config: BinanceConfig instance (preferred method)
        """
        super().__init__("BinanceClient")

        # Use config if provided, otherwise fall back to parameters
        if config is not None:
            self.api_key = config.api_key
            self.api_secret = config.api_secret
            self.testnet = config.testnet
        else:
            self.api_key = api_key
            self.api_secret = api_secret
            self.testnet = testnet

        self.event_bus = event_bus

        self._client: Optional[AsyncClient] = None
        self._socket_manager: Optional[BinanceSocketManager] = None
        self._is_connected = False

        # Rate limiting and connection management
        self._connection_lock = asyncio.Lock()
        self._last_heartbeat = None
        self._connection_retries = 0
        self._max_retries = 5

    async def _start(self) -> None:
        """Initialize the Binance client connection."""
        try:
            async with self._connection_lock:
                self._client = await AsyncClient.create(
                    api_key=self.api_key,
                    api_secret=self.api_secret,
                    testnet=self.testnet,
                )

                # Test connection
                await self._test_connection()

                self._socket_manager = BinanceSocketManager(self._client)
                self._is_connected = True
                self._connection_retries = 0

                self.logger.info(f"Binance client connected (testnet={self.testnet})")

        except Exception as e:
            self.logger.error(f"Failed to initialize Binance client: {e}")
            raise BinanceClientError(f"Connection failed: {e}") from e

    async def _stop(self) -> None:
        """Clean up the Binance client resources."""
        try:
            if self._socket_manager:
                self._socket_manager = None

            if self._client:
                await self._client.close_connection()
                self._client = None

            self._is_connected = False
            self.logger.info("Binance client disconnected")

        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    async def _test_connection(self) -> None:
        """Test the connection to Binance API."""
        if not self._client:
            raise BinanceClientError("Client not initialized")

        try:
            # Test with a simple ping
            await self._client.ping()

            # Get server time to verify connection
            server_time = await self._client.get_server_time()
            self._last_heartbeat = datetime.fromtimestamp(
                server_time["serverTime"] / 1000
            )

            self.logger.debug("Connection test successful")

        except Exception as e:
            raise BinanceClientError(f"Connection test failed: {e}") from e

    async def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get detailed symbol information.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')

        Returns:
            Symbol information dictionary
        """
        if not self._client:
            raise BinanceClientError("Client not connected")

        try:
            exchange_info = await self._client.futures_exchange_info()

            for symbol_info in exchange_info["symbols"]:
                if symbol_info["symbol"] == symbol:
                    return symbol_info

            raise BinanceClientError(f"Symbol {symbol} not found")

        except BinanceAPIException as e:
            self.logger.error(f"API error getting symbol info: {e}")
            raise BinanceClientError(f"API error: {e}") from e
        except Exception as e:
            self.logger.error(f"Error getting symbol info: {e}")
            raise BinanceClientError(f"Failed to get symbol info: {e}") from e

    async def get_ticker_price(self, symbol: str) -> Decimal:
        """
        Get current ticker price for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Current price as Decimal
        """
        if not self._client:
            raise BinanceClientError("Client not connected")

        try:
            ticker = await self._client.futures_symbol_ticker(symbol=symbol)
            return Decimal(ticker["price"])

        except BinanceAPIException as e:
            self.logger.error(f"API error getting ticker price: {e}")
            raise BinanceClientError(f"API error: {e}") from e
        except Exception as e:
            self.logger.error(f"Error getting ticker price: {e}")
            raise BinanceClientError(f"Failed to get ticker price: {e}") from e

    async def get_orderbook(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """
        Get order book for a symbol.

        Args:
            symbol: Trading symbol
            limit: Number of entries to return (default: 100)

        Returns:
            Order book data
        """
        if not self._client:
            raise BinanceClientError("Client not connected")

        try:
            orderbook = await self._client.futures_order_book(
                symbol=symbol, limit=limit
            )
            return orderbook

        except BinanceAPIException as e:
            self.logger.error(f"API error getting orderbook: {e}")
            raise BinanceClientError(f"API error: {e}") from e
        except Exception as e:
            self.logger.error(f"Error getting orderbook: {e}")
            raise BinanceClientError(f"Failed to get orderbook: {e}") from e

    async def get_historical_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[Union[str, int, datetime]] = None,
        end_time: Optional[Union[str, int, datetime]] = None,
        limit: int = 500,
    ) -> List[List[str]]:
        """
        Get historical kline/candlestick data.

        Args:
            symbol: Trading symbol
            interval: Kline interval (5m, 15m, 4h, 1d)
            start_time: Start time for data
            end_time: End time for data
            limit: Number of klines to return (max 1500)

        Returns:
            List of kline data
        """
        if not self._client:
            raise BinanceClientError("Client not connected")

        # Map interval strings to Binance constants
        interval_map = {
            "5m": KLINE_INTERVAL_5MINUTE,
            "15m": KLINE_INTERVAL_15MINUTE,
            "4h": KLINE_INTERVAL_4HOUR,
            "1d": KLINE_INTERVAL_1DAY,
        }

        if interval not in interval_map:
            raise BinanceClientError(f"Unsupported interval: {interval}")

        try:
            klines = await self._client.futures_klines(
                symbol=symbol,
                interval=interval_map[interval],
                startTime=start_time,
                endTime=end_time,
                limit=limit,
            )
            return klines

        except BinanceAPIException as e:
            self.logger.error(f"API error getting historical klines: {e}")
            raise BinanceClientError(f"API error: {e}") from e
        except Exception as e:
            self.logger.error(f"Error getting historical klines: {e}")
            raise BinanceClientError(f"Failed to get historical klines: {e}") from e

    async def get_account_info(self) -> Dict[str, Any]:
        """
        Get futures account information.

        Returns:
            Account information dictionary
        """
        if not self._client:
            raise BinanceClientError("Client not connected")

        if not (self.api_key and self.api_secret):
            raise BinanceClientError("API credentials required for account info")

        try:
            account = await self._client.futures_account()
            return account

        except BinanceAPIException as e:
            self.logger.error(f"API error getting account info: {e}")
            raise BinanceClientError(f"API error: {e}") from e
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            raise BinanceClientError(f"Failed to get account info: {e}") from e

    async def get_balance(self) -> Dict[str, Dict[str, str]]:
        """
        Get account balance information.

        Returns:
            Balance information by asset
        """
        if not self._client:
            raise BinanceClientError("Client not connected")

        if not (self.api_key and self.api_secret):
            raise BinanceClientError("API credentials required for balance info")

        try:
            account = await self.get_account_info()
            balances = {}

            for asset_info in account["assets"]:
                asset = asset_info["asset"]
                balances[asset] = {
                    "free": asset_info["availableBalance"],
                    "locked": asset_info["initialMargin"],
                    "total": str(
                        Decimal(asset_info["availableBalance"])
                        + Decimal(asset_info["initialMargin"])
                    ),
                }

            return balances

        except Exception as e:
            self.logger.error(f"Error getting balance: {e}")
            raise BinanceClientError(f"Failed to get balance: {e}") from e

    async def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: Union[str, Decimal],
        price: Optional[Union[str, Decimal]] = None,
        stop_price: Optional[Union[str, Decimal]] = None,
        time_in_force: str = "GTC",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Place a futures order.

        Args:
            symbol: Trading symbol
            side: Order side (BUY/SELL)
            order_type: Order type (MARKET/LIMIT/STOP/STOP_MARKET)
            quantity: Order quantity
            price: Order price (for LIMIT orders)
            stop_price: Stop price (for STOP orders)
            time_in_force: Time in force (GTC/IOC/FOK)
            **kwargs: Additional order parameters

        Returns:
            Order response data
        """
        if not self._client:
            raise BinanceClientError("Client not connected")

        if not (self.api_key and self.api_secret):
            raise BinanceClientError("API credentials required for placing orders")

        try:
            order_params = {
                "symbol": symbol,
                "side": side.upper(),
                "type": order_type.upper(),
                "quantity": str(quantity),
                "timeInForce": time_in_force,
                **kwargs,
            }

            if price is not None:
                order_params["price"] = str(price)
            if stop_price is not None:
                order_params["stopPrice"] = str(stop_price)

            order = await self._client.futures_create_order(**order_params)

            self.logger.info(f"Order placed: {order['orderId']} for {symbol}")
            return order

        except BinanceAPIException as e:
            self.logger.error(f"API error placing order: {e}")
            raise BinanceClientError(f"Order failed: {e}") from e
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            raise BinanceClientError(f"Failed to place order: {e}") from e

    async def cancel_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """
        Cancel an existing order.

        Args:
            symbol: Trading symbol
            order_id: Order ID to cancel

        Returns:
            Cancellation response data
        """
        if not self._client:
            raise BinanceClientError("Client not connected")

        if not (self.api_key and self.api_secret):
            raise BinanceClientError("API credentials required for canceling orders")

        try:
            result = await self._client.futures_cancel_order(
                symbol=symbol, orderId=order_id
            )

            self.logger.info(f"Order cancelled: {order_id} for {symbol}")
            return result

        except BinanceAPIException as e:
            self.logger.error(f"API error canceling order: {e}")
            raise BinanceClientError(f"Cancel failed: {e}") from e
        except Exception as e:
            self.logger.error(f"Error canceling order: {e}")
            raise BinanceClientError(f"Failed to cancel order: {e}") from e

    async def get_open_orders(
        self, symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get open orders.

        Args:
            symbol: Trading symbol (optional, if None returns all open orders)

        Returns:
            List of open orders
        """
        if not self._client:
            raise BinanceClientError("Client not connected")

        if not (self.api_key and self.api_secret):
            raise BinanceClientError("API credentials required for order info")

        try:
            orders = await self._client.futures_get_open_orders(symbol=symbol)
            return orders

        except BinanceAPIException as e:
            self.logger.error(f"API error getting open orders: {e}")
            raise BinanceClientError(f"API error: {e}") from e
        except Exception as e:
            self.logger.error(f"Error getting open orders: {e}")
            raise BinanceClientError(f"Failed to get open orders: {e}") from e

    def is_connected(self) -> bool:
        """Check if the client is connected."""
        return self._is_connected and self._client is not None

    def get_connection_status(self) -> Dict[str, Any]:
        """
        Get detailed connection status.

        Returns:
            Connection status information
        """
        return {
            "connected": self.is_connected(),
            "testnet": self.testnet,
            "last_heartbeat": self._last_heartbeat,
            "connection_retries": self._connection_retries,
            "max_retries": self._max_retries,
        }
