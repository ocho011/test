"""
Order execution engine for async order management and coordination.

This module provides the main OrderExecutor class that orchestrates
all order execution activities, coordinates with specialized managers,
and ensures proper order lifecycle management.
"""

import asyncio
import logging
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Set
from uuid import uuid4

from ..core.base_component import BaseComponent
from ..core.events import (
    EventType,
    OrderEvent,
    OrderSide,
    OrderStatus,
    OrderType,
    PositionEvent,
    PositionSide,
    PositionStatus,
    RiskApprovedOrderEvent,
    SignalType,
    SlippageEvent,
    TakeProfitEvent,
)
from ..data.binance_client import BinanceClient


class OrderExecutionError(Exception):
    """Custom exception for order execution errors."""

    pass


class OrderRequest:
    """Order request data structure."""

    def __init__(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: Decimal,
        price: Optional[Decimal] = None,
        stop_price: Optional[Decimal] = None,
        time_in_force: str = "GTC",
        client_order_id: Optional[str] = None,
        reduce_only: bool = False,
    ):
        self.symbol = symbol
        self.side = side
        self.order_type = order_type
        self.quantity = quantity
        self.price = price
        self.stop_price = stop_price
        self.time_in_force = time_in_force
        self.client_order_id = client_order_id or str(uuid4())
        self.reduce_only = reduce_only
        self.created_at = datetime.utcnow()


class OrderExecutor(BaseComponent):
    """
    Main order execution engine for coordinating all order operations.

    Handles async order submission, status tracking, retry logic,
    and coordination with specialized order managers.
    """

    def __init__(
        self,
        binance_client: BinanceClient,
        event_bus=None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        slippage_limit: float = 0.005,  # 0.5% default slippage limit
        dry_run: bool = False,
    ):
        """
        Initialize the order executor.

        Args:
            binance_client: Binance API client for order operations
            event_bus: Event bus for system communication
            max_retries: Maximum retry attempts for failed orders
            retry_delay: Base delay between retries (exponential backoff)
            slippage_limit: Maximum allowed slippage percentage
            dry_run: Enable dry-run mode (log orders without execution)
        """
        super().__init__(name=self.__class__.__name__)
        self.event_bus = event_bus
        self.binance_client = binance_client
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.slippage_limit = slippage_limit
        self.dry_run = dry_run

        # Active order tracking
        self.active_orders: Dict[str, OrderEvent] = {}
        self.pending_orders: Set[str] = set()

        # Execution statistics
        self.execution_stats = {
            "total_orders": 0,
            "successful_orders": 0,
            "failed_orders": 0,
            "retry_attempts": 0,
            "total_slippage": Decimal("0"),
        }

    async def _start(self):
        """Start the order executor and subscribe to events."""
        # Subscribe to RiskApprovedOrderEvent
        if self.event_bus:
            await self.event_bus.subscribe(
                self._handle_risk_approved_order, EventType.RISK_APPROVED_ORDER
            )
            self.logger.info("Subscribed to RiskApprovedOrderEvent")

        self.logger.info("OrderExecutor started")

    async def _stop(self):
        """Stop the order executor."""
        # Cancel any pending orders
        if self.pending_orders:
            self.logger.warning(f"Cancelling {len(self.pending_orders)} pending orders")
            await self._cancel_pending_orders()
        self.logger.info("OrderExecutor stopped")

    async def _emit_event(self, event):
        """Emit an event to the event bus if available."""
        if self.event_bus:
            await self.event_bus.publish(event)

    async def _handle_risk_approved_order(self, event: RiskApprovedOrderEvent):
        """
        Handle risk-approved order events and execute orders automatically.

        Args:
            event: RiskApprovedOrderEvent containing approved signal and quantity

        This method:
        1. Extracts order parameters from the SignalEvent
        2. Creates OrderRequest with approved quantity
        3. In dry-run mode: logs order details without execution
        4. In live mode: executes the order using existing execute_order() method
        """
        try:
            signal = event.signal

            self.logger.info(
                f"Processing risk-approved order: {signal.signal_type.value} "
                f"{signal.symbol} quantity={event.approved_quantity}"
            )

            # Convert signal type to order side
            if signal.signal_type == SignalType.BUY:
                order_side = OrderSide.BUY
            elif signal.signal_type == SignalType.SELL:
                order_side = OrderSide.SELL
            elif signal.signal_type == SignalType.CLOSE_LONG:
                order_side = OrderSide.SELL  # Close long = sell position
            elif signal.signal_type == SignalType.CLOSE_SHORT:
                order_side = OrderSide.BUY  # Close short = buy position
            else:
                self.logger.warning(
                    f"Ignoring signal type {signal.signal_type.value} - "
                    f"not a tradeable signal"
                )
                return

            # Create order request with approved quantity
            # Use MARKET order for immediate execution
            order_request = OrderRequest(
                symbol=signal.symbol,
                side=order_side,
                order_type=OrderType.MARKET,
                quantity=event.approved_quantity,
                # Note: stop_loss and take_profit are managed by PositionTracker
                # not by the order itself
            )

            # Dry-run mode: log order without execution
            if self.dry_run:
                self.logger.info(
                    f"[DRY-RUN] Order would be executed: "
                    f"{order_side.value} {event.approved_quantity} {signal.symbol} @ MARKET"
                )

                # Log stop loss and take profit if available
                if signal.stop_loss or signal.take_profit:
                    sl_str = f"{signal.stop_loss}" if signal.stop_loss else "N/A"
                    tp_str = f"{signal.take_profit}" if signal.take_profit else "N/A"
                    self.logger.info(
                        f"[DRY-RUN] Stop Loss: {sl_str}, Take Profit: {tp_str}"
                    )

                # Emit a simulated order event for testing
                order_event = OrderEvent(
                    source=self.__class__.__name__,
                    client_order_id=order_request.client_order_id,
                    symbol=order_request.symbol,
                    side=order_request.side,
                    order_type=order_request.order_type,
                    quantity=order_request.quantity,
                    price=order_request.price,
                    stop_price=order_request.stop_price,
                    status=OrderStatus.FILLED,  # Simulate successful execution
                    filled_quantity=event.approved_quantity,
                )

                # Emit simulated order event
                await self._emit_event(order_event)

                return

            # Live mode: execute the order
            order_event = await self.execute_order(order_request)

            self.logger.info(
                f"Risk-approved order executed successfully: "
                f"{order_side.value} {event.approved_quantity} {signal.symbol} "
                f"(Order ID: {order_event.client_order_id})"
            )

        except Exception as e:
            self.logger.error(
                f"Failed to execute risk-approved order for {event.signal.symbol}: {str(e)}"
            )

    def _validate_order_request(self, order_request: OrderRequest):
        """
        Validate order request parameters.

        Args:
            order_request: Order request to validate

        Raises:
            OrderExecutionError: If validation fails
        """
        # Validate symbol format (basic check for valid trading pair format)
        # Symbol should end with USDT, USDC, BUSD, or BTC and be at least 6 chars
        if not order_request.symbol or len(order_request.symbol) < 6:
            raise OrderExecutionError(f"Invalid symbol: {order_request.symbol}")

        # Additional check: must end with a valid quote currency
        valid_quote_currencies = ['USDT', 'USDC', 'BUSD', 'BTC', 'ETH']
        if not any(order_request.symbol.endswith(quote) for quote in valid_quote_currencies):
            raise OrderExecutionError(f"Invalid symbol: {order_request.symbol}")

        # Validate quantity is positive
        if order_request.quantity <= 0:
            raise OrderExecutionError(f"Invalid quantity: {order_request.quantity}")

        # Validate side is a valid OrderSide enum value
        if not isinstance(order_request.side, OrderSide):
            raise OrderExecutionError(f"Invalid order side: {order_request.side}")

        # Validate order_type is a valid OrderType enum value
        if not isinstance(order_request.order_type, OrderType):
            raise OrderExecutionError(f"Invalid order type: {order_request.order_type}")

    async def execute_order(self, order_request: OrderRequest) -> OrderEvent:
        """
        Execute an order with retry logic and monitoring.

        Args:
            order_request: Order details to execute

        Returns:
            OrderEvent with execution results

        Raises:
            OrderExecutionError: If order execution fails after all retries
        """
        # Validate order request
        self._validate_order_request(order_request)

        self.logger.info(
            f"Executing {order_request.order_type.value} {order_request.side.value} "
            f"order for {order_request.quantity} {order_request.symbol}"
        )

        # Create initial order event
        order_event = OrderEvent(
            source=self.__class__.__name__,
            client_order_id=order_request.client_order_id,
            symbol=order_request.symbol,
            side=order_request.side,
            order_type=order_request.order_type,
            quantity=order_request.quantity,
            price=order_request.price,
            stop_price=order_request.stop_price,
            status=OrderStatus.PENDING,
        )

        # Track order
        self.active_orders[order_request.client_order_id] = order_event
        self.pending_orders.add(order_request.client_order_id)

        # Emit order event
        await self._emit_event(order_event)

        # Execute with retry logic
        execution_result = await self._execute_with_retry(order_request, order_event)

        # Update statistics
        self._update_execution_stats(execution_result)

        return execution_result

    async def _execute_with_retry(
        self, order_request: OrderRequest, order_event: OrderEvent
    ) -> OrderEvent:
        """Execute order with exponential backoff retry logic."""
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    delay = self.retry_delay * (2 ** (attempt - 1))
                    self.logger.info(
                        f"Retrying order {order_request.client_order_id} "
                        f"(attempt {attempt}/{self.max_retries}) after {delay}s delay"
                    )
                    await asyncio.sleep(delay)
                    self.execution_stats["retry_attempts"] += 1

                # Attempt order execution
                result = await self._execute_single_order(order_request, order_event)

                # Remove from pending orders if successful
                self.pending_orders.discard(order_request.client_order_id)

                return result

            except Exception as e:
                last_exception = e
                self.logger.warning(
                    f"Order execution attempt {attempt + 1} failed: {str(e)}"
                )

                if attempt < self.max_retries:
                    # Update order status for retry
                    order_event.status = OrderStatus.PENDING
                    await self._emit_event(order_event)

        # All retries exhausted
        order_event.status = OrderStatus.REJECTED
        await self._emit_event(order_event)

        self.pending_orders.discard(order_request.client_order_id)

        raise OrderExecutionError(
            f"Order execution failed after {self.max_retries + 1} attempts: "
            f"{str(last_exception)}"
        )

    async def _execute_single_order(
        self, order_request: OrderRequest, order_event: OrderEvent
    ) -> OrderEvent:
        """Execute a single order attempt."""
        try:
            # Update order status to submitted
            order_event.status = OrderStatus.SUBMITTED
            await self._emit_event(order_event)

            # Get current market price for slippage calculation
            current_price = await self._get_current_price(order_request.symbol)

            # Prepare order parameters
            order_params = self._prepare_order_params(order_request)

            # Submit order to exchange
            if order_request.order_type == OrderType.MARKET:
                response = await self.binance_client.futures_create_order(**order_params)
            else:
                response = await self.binance_client.futures_create_order(**order_params)

            # Update order event with response data
            order_event.order_id = response.get("orderId")
            order_event.status = OrderStatus(response.get("status", "NEW").lower())
            order_event.filled_quantity = Decimal(str(response.get("executedQty", "0")))
            
            if response.get("avgPrice"):
                order_event.average_price = Decimal(str(response["avgPrice"]))

            # Calculate slippage for market orders
            if (
                order_request.order_type == OrderType.MARKET
                and order_event.average_price
                and order_event.filled_quantity > 0
            ):
                await self._check_slippage(
                    order_event, current_price, order_event.average_price
                )

            # Emit updated order event
            await self._emit_event(order_event)

            self.logger.info(
                f"Order {order_event.client_order_id} executed successfully "
                f"(Exchange ID: {order_event.order_id})"
            )

            return order_event

        except Exception as e:
            self.logger.error(f"Order execution failed: {str(e)}")
            raise

    async def _get_current_price(self, symbol: str) -> Decimal:
        """Get current market price for a symbol."""
        try:
            ticker = await self.binance_client.futures_symbol_ticker(symbol=symbol)
            return Decimal(str(ticker["price"]))
        except Exception as e:
            self.logger.error(f"Failed to get current price for {symbol}: {str(e)}")
            raise

    def _prepare_order_params(self, order_request: OrderRequest) -> Dict:
        """Prepare order parameters for Binance API."""
        params = {
            "symbol": order_request.symbol,
            "side": order_request.side.value.upper(),
            "type": order_request.order_type.value.upper(),
            "quantity": str(order_request.quantity),
            "timeInForce": order_request.time_in_force,
            "newClientOrderId": order_request.client_order_id,
        }

        if order_request.price:
            params["price"] = str(order_request.price)

        if order_request.stop_price:
            params["stopPrice"] = str(order_request.stop_price)

        if order_request.reduce_only:
            params["reduceOnly"] = "true"

        return params

    async def _check_slippage(
        self, order_event: OrderEvent, expected_price: Decimal, actual_price: Decimal
    ):
        """Check and emit slippage event if limits exceeded."""
        slippage_amount = abs(actual_price - expected_price)
        slippage_percentage = float(slippage_amount / expected_price)

        # Update statistics
        self.execution_stats["total_slippage"] += slippage_amount

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
            is_limit_exceeded=slippage_percentage > self.slippage_limit,
        )

        await self._emit_event(slippage_event)

        if slippage_event.is_limit_exceeded:
            self.logger.warning(
                f"Slippage limit exceeded for order {order_event.client_order_id}: "
                f"{slippage_percentage:.4f}% > {self.slippage_limit:.4f}%"
            )

    async def _cancel_pending_orders(self):
        """Cancel all pending orders."""
        for client_order_id in list(self.pending_orders):
            try:
                order_event = self.active_orders.get(client_order_id)
                if order_event and order_event.order_id:
                    await self.binance_client.futures_cancel_order(
                        symbol=order_event.symbol, orderId=order_event.order_id
                    )
                    order_event.status = OrderStatus.CANCELLED
                    await self._emit_event(order_event)

            except Exception as e:
                self.logger.error(
                    f"Failed to cancel order {client_order_id}: {str(e)}"
                )

        self.pending_orders.clear()

    def _update_execution_stats(self, order_event: OrderEvent):
        """Update execution statistics."""
        self.execution_stats["total_orders"] += 1

        if order_event.status in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED]:
            self.execution_stats["successful_orders"] += 1
        else:
            self.execution_stats["failed_orders"] += 1

    async def get_execution_stats(self) -> Dict:
        """Get execution statistics."""
        stats = self.execution_stats.copy()
        stats["active_orders_count"] = len(self.active_orders)
        stats["pending_orders_count"] = len(self.pending_orders)

        if stats["total_orders"] > 0:
            stats["success_rate"] = (
                stats["successful_orders"] / stats["total_orders"]
            )
            stats["average_slippage"] = float(
                stats["total_slippage"] / stats["total_orders"]
            )
        else:
            stats["success_rate"] = 0.0
            stats["average_slippage"] = 0.0

        return stats

    async def cancel_order(self, client_order_id: str) -> bool:
        """
        Cancel a specific order.

        Args:
            client_order_id: Client order ID to cancel

        Returns:
            True if cancellation successful, False otherwise
        """
        try:
            order_event = self.active_orders.get(client_order_id)
            if not order_event or not order_event.order_id:
                self.logger.warning(f"Order {client_order_id} not found or not submitted")
                return False

            await self.binance_client.futures_cancel_order(
                symbol=order_event.symbol, orderId=order_event.order_id
            )

            order_event.status = OrderStatus.CANCELLED
            await self._emit_event(order_event)

            self.pending_orders.discard(client_order_id)

            self.logger.info(f"Order {client_order_id} cancelled successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to cancel order {client_order_id}: {str(e)}")
            return False