"""
Mock Binance client for deterministic E2E testing.

Provides pre-recorded market data sequences with known ICT patterns
for reproducible integration testing.
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any
from unittest.mock import AsyncMock

import pandas as pd

from trading_bot.data.binance_client import BinanceClient


class MockBinanceClient(BinanceClient):
    """
    Mock Binance client that returns deterministic market data.
    
    Features:
    - Pre-recorded OHLCV data with known patterns
    - Simulated order execution with configurable slippage
    - Deterministic responses for testing
    - Support for different market scenarios
    """

    def __init__(
        self,
        scenario: str = "trending",
        slippage_pct: float = 0.001,
        order_delay_ms: int = 50
    ):
        """
        Initialize mock client.
        
        Args:
            scenario: Market scenario (trending, ranging, volatile, low_liquidity)
            slippage_pct: Simulated slippage percentage
            order_delay_ms: Simulated order execution delay
        """
        # Initialize with dummy keys
        super().__init__(
            api_key="test_key",
            api_secret="test_secret",
            testnet=True
        )
        
        self.scenario = scenario
        self.slippage_pct = slippage_pct
        self.order_delay_ms = order_delay_ms
        
        # Market data storage
        self._klines: Dict[str, pd.DataFrame] = {}
        self._current_prices: Dict[str, Decimal] = {}
        
        # Order tracking
        self._orders: List[Dict[str, Any]] = []
        self._next_order_id = 1
        
        # Position tracking
        self._positions: Dict[str, Dict[str, Any]] = {}
        
        # Account balance
        self._balances: Dict[str, Decimal] = {
            "USDT": Decimal("10000.0"),
            "BTC": Decimal("0.0"),
        }
        
        # Load scenario data
        self._load_scenario_data()

    def _load_scenario_data(self) -> None:
        """Load market data for the selected scenario."""
        if self.scenario == "trending":
            self._load_trending_scenario()
        elif self.scenario == "ranging":
            self._load_ranging_scenario()
        elif self.scenario == "volatile":
            self._load_volatile_scenario()
        elif self.scenario == "low_liquidity":
            self._load_low_liquidity_scenario()
        else:
            raise ValueError(f"Unknown scenario: {self.scenario}")

    def _load_trending_scenario(self) -> None:
        """Load trending market data with clear FVG and order block patterns."""
        # Generate uptrend with ICT patterns
        base_price = 50000.0
        timestamps = []
        opens, highs, lows, closes, volumes = [], [], [], [], []
        
        # Create 500 candles of uptrending data
        for i in range(500):
            timestamp = int((datetime.now() - timedelta(minutes=500-i)).timestamp() * 1000)
            timestamps.append(timestamp)
            
            # Uptrend with occasional pullbacks
            trend_factor = 1.0 + (i * 0.0001)  # Gradual uptrend
            
            # Create FVG pattern every 50 candles
            if i % 50 == 25:
                # Strong bullish move creating FVG
                open_price = base_price * trend_factor
                high_price = open_price * 1.015  # 1.5% gap
                low_price = open_price * 0.999
                close_price = high_price * 0.998
            # Create order block every 50 candles
            elif i % 50 == 40:
                # Consolidation forming order block
                open_price = base_price * trend_factor
                high_price = open_price * 1.002
                low_price = open_price * 0.998
                close_price = open_price * 1.001
            else:
                # Normal candle
                open_price = base_price * trend_factor
                high_price = open_price * 1.003
                low_price = open_price * 0.997
                close_price = open_price * (1.0 + (0.002 if i % 2 == 0 else -0.001))
            
            opens.append(open_price)
            highs.append(high_price)
            lows.append(low_price)
            closes.append(close_price)
            volumes.append(100.0 + (i % 50) * 10.0)  # Varying volume
        
        self._klines["BTCUSDT"] = pd.DataFrame({
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes
        })
        
        self._current_prices["BTCUSDT"] = Decimal(str(closes[-1]))

    def _load_ranging_scenario(self) -> None:
        """Load ranging market data."""
        # Simplified ranging market
        base_price = 50000.0
        timestamps = []
        opens, highs, lows, closes, volumes = [], [], [], [], []
        
        for i in range(500):
            timestamp = int((datetime.now() - timedelta(minutes=500-i)).timestamp() * 1000)
            timestamps.append(timestamp)
            
            # Oscillate between 49500 and 50500
            cycle_pos = (i % 100) / 100.0
            price_range = 1000.0
            current_price = base_price - (price_range / 2) + (price_range * cycle_pos)
            
            opens.append(current_price)
            highs.append(current_price * 1.002)
            lows.append(current_price * 0.998)
            closes.append(current_price * (1.001 if i % 2 == 0 else 0.999))
            volumes.append(80.0 + (i % 20) * 5.0)
        
        self._klines["BTCUSDT"] = pd.DataFrame({
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes
        })
        
        self._current_prices["BTCUSDT"] = Decimal(str(closes[-1]))

    def _load_volatile_scenario(self) -> None:
        """Load high volatility market data."""
        # High volatility scenario
        base_price = 50000.0
        timestamps = []
        opens, highs, lows, closes, volumes = [], [], [], [], []
        
        for i in range(500):
            timestamp = int((datetime.now() - timedelta(minutes=500-i)).timestamp() * 1000)
            timestamps.append(timestamp)
            
            # Large price swings
            swing = (-1 if i % 4 < 2 else 1) * (i % 10) * 0.002
            current_price = base_price * (1.0 + swing)
            
            opens.append(current_price)
            highs.append(current_price * 1.01)  # 1% high
            lows.append(current_price * 0.99)   # 1% low
            closes.append(current_price * (1.005 if i % 2 == 0 else 0.995))
            volumes.append(150.0 + (i % 30) * 15.0)
        
        self._klines["BTCUSDT"] = pd.DataFrame({
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes
        })
        
        self._current_prices["BTCUSDT"] = Decimal(str(closes[-1]))

    def _load_low_liquidity_scenario(self) -> None:
        """Load low liquidity market data."""
        # Low liquidity with wide spreads
        base_price = 50000.0
        timestamps = []
        opens, highs, lows, closes, volumes = [], [], [], [], []
        
        for i in range(500):
            timestamp = int((datetime.now() - timedelta(minutes=500-i)).timestamp() * 1000)
            timestamps.append(timestamp)
            
            current_price = base_price * (1.0 + (i % 20) * 0.0001)
            
            opens.append(current_price)
            highs.append(current_price * 1.005)  # Wider spreads
            lows.append(current_price * 0.995)
            closes.append(current_price * 1.002)
            volumes.append(20.0 + (i % 10) * 2.0)  # Low volume
        
        self._klines["BTCUSDT"] = pd.DataFrame({
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes
        })
        
        self._current_prices["BTCUSDT"] = Decimal(str(closes[-1]))

    async def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> List[List]:
        """Return pre-recorded klines data."""
        await asyncio.sleep(0.01)  # Simulate network delay
        
        if symbol not in self._klines:
            return []
        
        df = self._klines[symbol]
        
        # Apply time filters
        if start_time:
            df = df[df["timestamp"] >= start_time]
        if end_time:
            df = df[df["timestamp"] <= end_time]
        
        # Apply limit
        df = df.tail(limit)
        
        # Convert to Binance format
        klines = []
        for _, row in df.iterrows():
            klines.append([
                row["timestamp"],
                str(row["open"]),
                str(row["high"]),
                str(row["low"]),
                str(row["close"]),
                str(row["volume"]),
                0,  # Close time
                "0",  # Quote asset volume
                0,  # Number of trades
                "0",  # Taker buy base volume
                "0",  # Taker buy quote volume
                "0"  # Ignore
            ])
        
        return klines

    async def get_ticker_price(self, symbol: str) -> Decimal:
        """Return current mock price."""
        await asyncio.sleep(0.01)
        return self._current_prices.get(symbol, Decimal("0"))

    async def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: Decimal,
        price: Optional[Decimal] = None,
        time_in_force: str = "GTC",
        stop_price: Optional[Decimal] = None
    ) -> Dict[str, Any]:
        """Simulate order creation."""
        await asyncio.sleep(self.order_delay_ms / 1000.0)
        
        # Apply slippage to market orders
        execution_price = price
        if order_type == "MARKET":
            current_price = self._current_prices[symbol]
            slippage_mult = Decimal(1.0 + self.slippage_pct) if side == "BUY" else Decimal(1.0 - self.slippage_pct)
            execution_price = current_price * slippage_mult
        
        order = {
            "orderId": self._next_order_id,
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "quantity": str(quantity),
            "price": str(execution_price) if execution_price else "0",
            "status": "FILLED" if order_type == "MARKET" else "NEW",
            "executedQty": str(quantity) if order_type == "MARKET" else "0",
            "cummulativeQuoteQty": str(quantity * execution_price) if execution_price else "0",
            "timeInForce": time_in_force,
            "fills": [
                {
                    "price": str(execution_price) if execution_price else "0",
                    "qty": str(quantity),
                    "commission": "0",
                    "commissionAsset": symbol[-4:] if len(symbol) > 4 else "USDT"
                }
            ] if order_type == "MARKET" else []
        }
        
        self._orders.append(order)
        self._next_order_id += 1
        
        # Update balances for market orders
        if order_type == "MARKET" and execution_price:
            self._update_balances(symbol, side, quantity, execution_price)
        
        return order

    def _update_balances(self, symbol: str, side: str, quantity: Decimal, price: Decimal) -> None:
        """Update account balances after order execution."""
        quote_asset = symbol[-4:] if len(symbol) > 4 else "USDT"
        base_asset = symbol[:-4] if len(symbol) > 4 else "BTC"
        
        if side == "BUY":
            cost = quantity * price
            self._balances[quote_asset] -= cost
            self._balances[base_asset] = self._balances.get(base_asset, Decimal("0")) + quantity
        else:  # SELL
            proceeds = quantity * price
            self._balances[quote_asset] += proceeds
            self._balances[base_asset] -= quantity

    async def get_account(self) -> Dict[str, Any]:
        """Return mock account information."""
        await asyncio.sleep(0.01)
        
        balances = [
            {
                "asset": asset,
                "free": str(balance),
                "locked": "0"
            }
            for asset, balance in self._balances.items()
        ]
        
        return {
            "balances": balances,
            "canTrade": True,
            "canWithdraw": True,
            "canDeposit": True
        }

    async def get_all_orders(self, symbol: str) -> List[Dict[str, Any]]:
        """Return all orders for symbol."""
        await asyncio.sleep(0.01)
        return [o for o in self._orders if o["symbol"] == symbol]

    async def cancel_order(self, symbol: str, order_id: int) -> Dict[str, Any]:
        """Cancel an order."""
        await asyncio.sleep(0.01)
        
        for order in self._orders:
            if order["orderId"] == order_id and order["symbol"] == symbol:
                order["status"] = "CANCELED"
                return order
        
        raise ValueError(f"Order {order_id} not found")

    def reset(self) -> None:
        """Reset mock client state."""
        self._orders = []
        self._next_order_id = 1
        self._positions = {}
        self._balances = {
            "USDT": Decimal("10000.0"),
            "BTC": Decimal("0.0"),
        }
        self._load_scenario_data()
