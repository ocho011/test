"""
Specialized order managers for different order types.

This module provides MarketOrderManager and LimitOrderManager classes
that handle the specific execution logic for market and limit orders.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional
from uuid import uuid4

from ..core.base_component import BaseComponent
from ..core.events import (
    OrderEvent,
    OrderSide,
    OrderStatus,
    OrderType,
    SlippageEvent,
)
from ..data.binance_client import BinanceClient
from .order_executor import OrderRequest


class OrderManagerError(Exception):
    """Custom exception for order manager errors."""

    pass


class MarketOrderManager(BaseComponent):
    """
    Market order manager for immediate execution orders.

    Handles market orders with immediate execution, slippage monitoring,
    and execution quality validation.
    """

    def __init__(
        self,
        binance_client: BinanceClient,
        event_bus=None,
        max_slippage: float = 0.01,  # 1% max slippage
        execution_timeout: float = 30.0,  # 30 seconds
    ):
        """
        Initialize the market order manager.

        Args:
            binance_client: Binance API client
            event_bus: Event bus for communication
            max_slippage: Maximum allowed slippage percentage
            execution_timeout: Maximum time to wait for execution
        """
        super().__init__(name=self.__class__.__name__)
        self.event_bus = event_bus
        self.binance_client = binance_client
        self.max_slippage = max_slippage
        self.execution_timeout = execution_timeout


        # Execution tracking
        self.execution_stats = {
            "total_executed": 0,
            "average_slippage": 0.0,
            "slippage_violations": 0,
            "average_execution_time": 0.0,
        }

    async def _start(self):
        """Start the market order manager (no-op for utility class)."""
        self.logger.info("MarketOrderManager ready")

    async def _stop(self):
        """Stop the market order manager (no-op for utility class)."""
        self.logger.info("MarketOrderManager stopped")

    async def execute_market_order(self, order_request: OrderRequest) -> OrderEvent:
        """
        Execute a market order with immediate execution.

        Args:
            order_request: Market order details

        Returns:
            OrderEvent with execution results

        Raises:
            OrderManagerError: If execution fails or slippage exceeds limits
        """
        if order_request.order_type != OrderType.MARKET:
            raise OrderManagerError("MarketOrderManager only handles market orders")

        start_time = datetime.utcnow()

        self.logger.info(
            f"Executing market order: {order_request.side.value} "
            f"{order_request.quantity} {order_request.symbol}"
        )

        try:
            # Get pre-execution market data
            pre_execution_data = await self._get_market_data(order_request.symbol)

            # Execute market order
            order_event = await self._execute_immediate_order(order_request, pre_execution_data)

            # Calculate execution metrics
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            self._update_execution_stats(order_event, execution_time)

            return order_event

        except Exception as e:
            self.logger.error(f"Market order execution failed: {str(e)}")
            raise OrderManagerError(f"Market order execution failed: {str(e)}") from e

    async def _execute_immediate_order(
        self, order_request: OrderRequest, pre_data: Dict
    ) -> OrderEvent:
        """Execute market order immediately."""
        # Create order event
        order_event = OrderEvent(
            source=self.__class__.__name__,
            client_order_id=order_request.client_order_id,
            symbol=order_request.symbol,
            side=order_request.side,
            order_type=order_request.order_type,
            quantity=order_request.quantity,
            status=OrderStatus.PENDING,
        )

        try:
            # Submit market order
            order_params = {
                "symbol": order_request.symbol,
                "side": order_request.side.value.upper(),
                "type": "MARKET",
                "quantity": str(order_request.quantity),
                "newClientOrderId": order_request.client_order_id,
            }

            if order_request.reduce_only:
                order_params["reduceOnly"] = "true"

            # Update status to submitted
            order_event.status = OrderStatus.SUBMITTED
            await self._emit_event(order_event)

            # Execute order
            response = await self.binance_client.futures_create_order(**order_params)

            # Update order event with response
            order_event.order_id = response.get("orderId")
            order_event.status = OrderStatus(response.get("status", "NEW").lower())
            order_event.filled_quantity = Decimal(str(response.get("executedQty", "0")))

            if response.get("avgPrice"):
                order_event.average_price = Decimal(str(response["avgPrice"]))

            # Validate execution quality
            if order_event.filled_quantity > 0 and order_event.average_price:
                await self._validate_execution_quality(
                    order_event, pre_data, order_request
                )

            # Final status update
            await self._emit_event(order_event)

            return order_event

        except Exception as e:
            order_event.status = OrderStatus.REJECTED
            await self._emit_event(order_event)
            raise

    async def _get_market_data(self, symbol: str) -> Dict:
        """Get current market data for execution quality analysis."""
        try:
            # Get order book for bid/ask spread
            depth = await self.binance_client.futures_order_book(symbol=symbol, limit=5)

            # Get current price
            ticker = await self.binance_client.futures_symbol_ticker(symbol=symbol)

            return {
                "price": Decimal(str(ticker["price"])),
                "bid": Decimal(str(depth["bids"][0][0])) if depth["bids"] else None,
                "ask": Decimal(str(depth["asks"][0][0])) if depth["asks"] else None,
                "bid_size": Decimal(str(depth["bids"][0][1])) if depth["bids"] else None,
                "ask_size": Decimal(str(depth["asks"][0][1])) if depth["asks"] else None,
                "timestamp": datetime.utcnow(),
            }

        except Exception as e:
            self.logger.error(f"Failed to get market data for {symbol}: {str(e)}")
            raise

    async def _validate_execution_quality(
        self, order_event: OrderEvent, pre_data: Dict, order_request: OrderRequest
    ):
        """Validate execution quality and emit slippage events."""
        expected_price = pre_data["bid"] if order_request.side == OrderSide.SELL else pre_data["ask"]

        if not expected_price:
            expected_price = pre_data["price"]

        actual_price = order_event.average_price
        slippage_amount = abs(actual_price - expected_price)
        slippage_percentage = float(slippage_amount / expected_price)

        # Create slippage event
        slippage_event = SlippageEvent(
            source=self.__class__.__name__,
            order_id=order_event.client_order_id,
            symbol=order_event.symbol,
            expected_price=expected_price,
            actual_price=actual_price,
            slippage_amount=slippage_amount,
            slippage_percentage=slippage_percentage,
            quantity=order_event.filled_quantity,
            slippage_cost=slippage_amount * order_event.filled_quantity,
            is_limit_exceeded=slippage_percentage > self.max_slippage,
        )

        await self._emit_event(slippage_event)

        if slippage_event.is_limit_exceeded:
            self.logger.warning(
                f"Slippage limit exceeded: {slippage_percentage:.4f}% > {self.max_slippage:.4f}%"
            )
            raise OrderManagerError(
                f"Slippage limit exceeded: {slippage_percentage:.4f}%"
            )

    def _update_execution_stats(self, order_event: OrderEvent, execution_time: float):
        """Update execution statistics."""
        self.execution_stats["total_executed"] += 1

        # Update average execution time
        total = self.execution_stats["total_executed"]
        current_avg = self.execution_stats["average_execution_time"]
        self.execution_stats["average_execution_time"] = (
            (current_avg * (total - 1)) + execution_time
        ) / total


class LimitOrderManager(BaseComponent):
    """
    Limit order manager for pending order management.

    Handles limit orders with fill monitoring, partial fill tracking,
    and order lifecycle management.
    """

    def __init__(
        self,
        binance_client: BinanceClient,
        event_bus=None,
        monitor_interval: float = 5.0,  # 5 seconds monitoring interval
        max_age: int = 86400,  # 24 hours max age
    ):
        """
        Initialize the limit order manager.

        Args:
            binance_client: Binance API client
            event_bus: Event bus for communication
            monitor_interval: Interval for monitoring order status
            max_age: Maximum age for orders in seconds
        """
        super().__init__(name=self.__class__.__name__)
        self.event_bus = event_bus
        self.binance_client = binance_client
        self.monitor_interval = monitor_interval
        self.max_age = max_age

        # Active order tracking
        self.active_orders: Dict[str, OrderEvent] = {}
        self.monitoring_task: Optional[asyncio.Task] = None


    async def _start(self):
        """Start the limit order manager."""

        # Start monitoring task
        self.monitoring_task = asyncio.create_task(self._monitor_orders())

        self.logger.info("LimitOrderManager started")

    async def _stop(self):
        """Stop the limit order manager."""
        # Cancel monitoring task
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        # Cancel active orders
        await self._cancel_all_orders()

        self.logger.info("LimitOrderManager stopped")

    async def execute_limit_order(self, order_request: OrderRequest) -> OrderEvent:
        """
        Execute a limit order and start monitoring.

        Args:
            order_request: Limit order details

        Returns:
            OrderEvent for the submitted order

        Raises:
            OrderManagerError: If order submission fails
        """
        if order_request.order_type != OrderType.LIMIT:
            raise OrderManagerError("LimitOrderManager only handles limit orders")

        if not order_request.price:
            raise OrderManagerError("Limit orders require a price")

        self.logger.info(
            f"Submitting limit order: {order_request.side.value} "
            f"{order_request.quantity} {order_request.symbol} @ {order_request.price}"
        )

        try:
            # Create order event
            order_event = OrderEvent(
                source=self.__class__.__name__,
                client_order_id=order_request.client_order_id,
                symbol=order_request.symbol,
                side=order_request.side,
                order_type=order_request.order_type,
                quantity=order_request.quantity,
                price=order_request.price,
                status=OrderStatus.PENDING,
            )

            # Submit limit order
            order_params = {
                "symbol": order_request.symbol,
                "side": order_request.side.value.upper(),
                "type": "LIMIT",
                "quantity": str(order_request.quantity),
                "price": str(order_request.price),
                "timeInForce": order_request.time_in_force,
                "newClientOrderId": order_request.client_order_id,
            }

            if order_request.reduce_only:
                order_params["reduceOnly"] = "true"

            # Update status to submitted
            order_event.status = OrderStatus.SUBMITTED
            await self._emit_event(order_event)

            # Execute order
            response = await self.binance_client.futures_create_order(**order_params)

            # Update order event with response
            order_event.order_id = response.get("orderId")
            order_event.status = OrderStatus(response.get("status", "NEW").lower())

            # Add to monitoring
            self.active_orders[order_request.client_order_id] = order_event

            await self._emit_event(order_event)

            self.logger.info(
                f"Limit order submitted: {order_event.client_order_id} "
                f"(Exchange ID: {order_event.order_id})"
            )

            return order_event

        except Exception as e:
            self.logger.error(f"Limit order submission failed: {str(e)}")
            raise OrderManagerError(f"Limit order submission failed: {str(e)}") from e

    async def cancel_order(self, client_order_id: str) -> bool:
        """
        Cancel a specific limit order.

        Args:
            client_order_id: Client order ID to cancel

        Returns:
            True if cancellation successful, False otherwise
        """
        order_event = self.active_orders.get(client_order_id)
        if not order_event:
            self.logger.warning(f"Order {client_order_id} not found")
            return False

        try:
            await self.binance_client.futures_cancel_order(
                symbol=order_event.symbol, orderId=order_event.order_id
            )

            order_event.status = OrderStatus.CANCELLED
            await self._emit_event(order_event)

            # Remove from monitoring
            self.active_orders.pop(client_order_id, None)

            self.logger.info(f"Order {client_order_id} cancelled successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to cancel order {client_order_id}: {str(e)}")
            return False

    async def _monitor_orders(self):
        """Monitor active limit orders for status updates."""
        self.logger.info("Starting order monitoring")

        while True:
            try:
                if self.active_orders:
                    await self._check_order_status()
                    await self._cleanup_old_orders()

                await asyncio.sleep(self.monitor_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in order monitoring: {str(e)}")
                await asyncio.sleep(self.monitor_interval)

        self.logger.info("Order monitoring stopped")

    async def _check_order_status(self):
        """Check status of all active orders."""
        for client_order_id, order_event in list(self.active_orders.items()):
            try:
                # Query order status
                response = await self.binance_client.futures_get_order(
                    symbol=order_event.symbol, orderId=order_event.order_id
                )

                # Update order event
                old_status = order_event.status
                order_event.status = OrderStatus(response.get("status", "NEW").lower())
                order_event.filled_quantity = Decimal(str(response.get("executedQty", "0")))

                if response.get("avgPrice"):
                    order_event.average_price = Decimal(str(response["avgPrice"]))

                # Emit event if status changed
                if order_event.status != old_status:
                    await self._emit_event(order_event)

                    self.logger.info(
                        f"Order {client_order_id} status updated: "
                        f"{old_status.value} -> {order_event.status.value}"
                    )

                # Remove completed orders from monitoring
                if order_event.status in [
                    OrderStatus.FILLED,
                    OrderStatus.CANCELLED,
                    OrderStatus.REJECTED,
                ]:
                    self.active_orders.pop(client_order_id, None)

            except Exception as e:
                self.logger.error(
                    f"Failed to check status for order {client_order_id}: {str(e)}"
                )

    async def _cleanup_old_orders(self):
        """Cancel orders that exceed maximum age."""
        current_time = datetime.utcnow()

        for client_order_id, order_event in list(self.active_orders.items()):
            age = (current_time - order_event.timestamp).total_seconds()

            if age > self.max_age:
                self.logger.warning(
                    f"Order {client_order_id} exceeded max age ({age:.0f}s), cancelling"
                )
                await self.cancel_order(client_order_id)

    async def _cancel_all_orders(self):
        """Cancel all active orders."""
        for client_order_id in list(self.active_orders.keys()):
            await self.cancel_order(client_order_id)

    async def get_active_orders(self) -> List[OrderEvent]:
        """Get list of all active orders."""
        return list(self.active_orders.values())

    async def get_order_count(self) -> int:
        """Get count of active orders."""
        return len(self.active_orders)