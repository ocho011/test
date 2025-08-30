import asyncio
import logging
import uuid
from typing import Dict

from domain.ports.EventBus import EventBus
from domain.events.OrderEvent import ApprovedTradeOrderEvent, OrderStateChangeEvent

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class AsyncOrderManager:
    """Handles the mechanics of placing, tracking, and cancelling orders with the exchange."""
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.active_orders: Dict[str, asyncio.Task] = {}

    async def start_order_processing(self):
        """Starts the order manager by subscribing to approved order events."""
        logger.info("Order Manager started.")
        await self.event_bus.subscribe("APPROVED_TRADE_ORDER", self._handle_approved_order)
        logger.info("Order Manager subscribed to APPROVED_TRADE_ORDER.")

    async def _handle_approved_order(self, event: ApprovedTradeOrderEvent):
        """Handles an approved trade order by creating a simulated order task."""
        order_id = str(uuid.uuid4())
        logger.info(f"Received approved order {order_id} for {event.symbol}. Simulating placement.")
        
        # Create a task to simulate the order's lifecycle
        order_task = asyncio.create_task(self._simulate_order_lifecycle(order_id, event))
        self.active_orders[order_id] = order_task

    async def _simulate_order_lifecycle(self, order_id: str, order_details: ApprovedTradeOrderEvent):
        """
        Simulates an order being placed and eventually filled.
        In a real system, this would involve WebSocket connections to the exchange.
        """
        try:
            # Simulate time for the order to be placed and sit on the book
            await asyncio.sleep(5) 

            # Simulate the order being filled at the requested price
            logger.info(f"Order {order_id} for {order_details.symbol} has been FILLED.")
            fill_event = OrderStateChangeEvent(
                order_id=order_id,
                symbol=order_details.symbol,
                new_state="FILLED",
                fill_price=order_details.entry_price,
                fill_quantity=order_details.quantity
            )
            await self.event_bus.publish(fill_event)

            # Once filled, the order task is done
            del self.active_orders[order_id]

        except asyncio.CancelledError:
            logger.info(f"Order {order_id} was cancelled.")
            # Publish a cancellation event if needed
            pass

    async def cancel_all_orders(self):
        """Cancels all active (simulated) orders."""
        logger.info(f"Cancelling all {len(self.active_orders)} open orders...")
        for order_id, task in self.active_orders.items():
            task.cancel()
        self.active_orders.clear()
        await asyncio.sleep(0.5)
        logger.info("All open orders cancelled.")