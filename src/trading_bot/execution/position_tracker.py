"""
Position tracker for real-time profit/loss monitoring and position state management.

This module provides comprehensive position tracking with real-time P&L calculations,
position lifecycle management, and integration with market data for accurate monitoring.
"""

import asyncio
import logging
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional
from uuid import uuid4

from ..core.base_component import BaseComponent
from ..core.events import (
    MarketDataEvent,
    OrderEvent,
    OrderSide,
    OrderStatus,
    PositionEvent,
    PositionSide,
    PositionStatus,
)
from ..data.binance_client import BinanceClient


class Position:
    """
    Individual position data structure.

    Represents a single trading position with all relevant tracking data
    including entry details, current status, and P&L calculations.
    """

    def __init__(
        self,
        position_id: str,
        symbol: str,
        side: PositionSide,
        size: Decimal,
        entry_price: Decimal,
        stop_loss: Optional[Decimal] = None,
        take_profit: Optional[Decimal] = None,
    ):
        self.position_id = position_id
        self.symbol = symbol
        self.side = side
        self.size = size
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit

        # Position state
        self.status = PositionStatus.OPENING
        self.current_price: Optional[Decimal] = None
        self.unrealized_pnl: Optional[Decimal] = None
        self.realized_pnl = Decimal("0")

        # Tracking data
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self.orders: List[str] = []  # Associated order IDs

    def calculate_unrealized_pnl(self, current_price: Decimal) -> Decimal:
        """Calculate unrealized P&L based on current price."""
        if self.side == PositionSide.LONG:
            pnl = (current_price - self.entry_price) * self.size
        else:  # SHORT
            pnl = (self.entry_price - current_price) * self.size

        return pnl

    def calculate_risk_reward_ratio(self, current_price: Decimal) -> Optional[float]:
        """Calculate current risk-reward ratio."""
        if not self.stop_loss:
            return None

        risk = abs(self.entry_price - self.stop_loss)
        if risk == 0:
            return None

        if self.side == PositionSide.LONG:
            reward = current_price - self.entry_price
        else:  # SHORT
            reward = self.entry_price - current_price

        return float(reward / risk)

    def update_price(self, current_price: Decimal):
        """Update position with current market price."""
        self.current_price = current_price
        self.unrealized_pnl = self.calculate_unrealized_pnl(current_price)
        self.updated_at = datetime.utcnow()

    def reduce_size(self, quantity: Decimal, exit_price: Decimal) -> Decimal:
        """
        Reduce position size and calculate realized P&L.

        Args:
            quantity: Amount to reduce
            exit_price: Price at which position is reduced

        Returns:
            Realized P&L from the reduction
        """
        if quantity > self.size:
            quantity = self.size

        # Calculate realized P&L for the reduced portion
        if self.side == PositionSide.LONG:
            realized_pnl = (exit_price - self.entry_price) * quantity
        else:  # SHORT
            realized_pnl = (self.entry_price - exit_price) * quantity

        # Update position
        self.size -= quantity
        self.realized_pnl += realized_pnl
        self.updated_at = datetime.utcnow()

        if self.size == 0:
            self.status = PositionStatus.CLOSED

        return realized_pnl

    def to_dict(self) -> Dict:
        """Convert position to dictionary representation."""
        return {
            "position_id": self.position_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "status": self.status.value,
            "size": str(self.size),
            "entry_price": str(self.entry_price),
            "current_price": str(self.current_price) if self.current_price else None,
            "unrealized_pnl": str(self.unrealized_pnl) if self.unrealized_pnl else None,
            "realized_pnl": str(self.realized_pnl),
            "stop_loss": str(self.stop_loss) if self.stop_loss else None,
            "take_profit": str(self.take_profit) if self.take_profit else None,
            "risk_reward_ratio": (
                self.calculate_risk_reward_ratio(self.current_price)
                if self.current_price
                else None
            ),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "orders": self.orders,
        }


class PositionTracker(BaseComponent):
    """
    Real-time position tracking and P&L monitoring system.

    Tracks all active positions, calculates real-time P&L, manages position
    lifecycle, and emits position events for system coordination.
    """

    def __init__(
        self,
        binance_client: BinanceClient,
        event_bus=None,
        update_interval: float = 1.0,  # 1 second update interval
    ):
        """
        Initialize the position tracker.

        Args:
            binance_client: Binance API client for market data
            event_bus: Event bus for communication
            update_interval: Interval for P&L updates in seconds
        """
        super().__init__(name=self.__class__.__name__)
        self.event_bus = event_bus
        self.binance_client = binance_client
        self.update_interval = update_interval

        # Position tracking
        self.positions: Dict[str, Position] = {}
        self.symbol_positions: Dict[str, List[str]] = {}  # symbol -> position_ids

        # Market data cache
        self.current_prices: Dict[str, Decimal] = {}

        # Background tasks
        self.update_task: Optional[asyncio.Task] = None

        # Register event handlers
        self._register_event_handlers()

    def _register_event_handlers(self):
        """Register event handlers for order and market data events."""
        if self.event_bus:
            self.event_bus.subscribe("OrderEvent", self._handle_order_event)
            self.event_bus.subscribe("MarketDataEvent", self._handle_market_data_event)

    async def _start(self):
        """Start the position tracker."""
        # Start P&L update task
        self.update_task = asyncio.create_task(self._update_positions_loop())
        self.logger.info("PositionTracker started")

    async def _stop(self):
        """Stop the position tracker."""
        # Cancel update task
        if self.update_task:
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass
        self.logger.info("PositionTracker stopped")

    async def _emit_event(self, event):
        """Emit an event to the event bus if available."""
        if self.event_bus:
            await self.event_bus.emit(event)

    async def _handle_order_event(self, order_event: OrderEvent):
        """Handle order events to track position changes."""
        try:
            if order_event.status == OrderStatus.FILLED:
                await self._process_filled_order(order_event)
        except Exception as e:
            self.logger.error(f"Error handling order event: {str(e)}")

    async def _handle_market_data_event(self, market_data: MarketDataEvent):
        """Handle market data events to update current prices."""
        try:
            self.current_prices[market_data.symbol] = market_data.price

            # Update positions for this symbol
            if market_data.symbol in self.symbol_positions:
                for position_id in self.symbol_positions[market_data.symbol]:
                    position = self.positions.get(position_id)
                    if position and position.status == PositionStatus.OPEN:
                        position.update_price(market_data.price)

        except Exception as e:
            self.logger.error(f"Error handling market data event: {str(e)}")

    async def _process_filled_order(self, order_event: OrderEvent):
        """Process filled orders to update position tracking."""
        symbol = order_event.symbol
        side = order_event.side
        quantity = order_event.filled_quantity
        price = order_event.average_price

        if not price or quantity <= 0:
            return

        # Determine if this is opening or closing a position
        if order_event.metadata.get("reduce_only", False):
            # This is a position-reducing order
            await self._reduce_positions(symbol, side, quantity, price, order_event.client_order_id)
        else:
            # This is a position-opening order
            await self._open_position(symbol, side, quantity, price, order_event.client_order_id)

    async def _open_position(
        self, symbol: str, side: OrderSide, quantity: Decimal, price: Decimal, order_id: str
    ):
        """Open a new position or add to existing position."""
        position_side = PositionSide.LONG if side == OrderSide.BUY else PositionSide.SHORT

        # Check for existing position in same direction
        existing_position = self._find_position(symbol, position_side, PositionStatus.OPEN)

        if existing_position:
            # Add to existing position (average down/up)
            total_size = existing_position.size + quantity
            weighted_price = (
                (existing_position.entry_price * existing_position.size) + (price * quantity)
            ) / total_size

            existing_position.size = total_size
            existing_position.entry_price = weighted_price
            existing_position.orders.append(order_id)
            existing_position.updated_at = datetime.utcnow()

            self.logger.info(
                f"Added to position {existing_position.position_id}: "
                f"new size {total_size}, new avg price {weighted_price}"
            )

            # Emit position event
            await self._emit_position_event(existing_position)

        else:
            # Create new position
            position_id = str(uuid4())
            position = Position(
                position_id=position_id,
                symbol=symbol,
                side=position_side,
                size=quantity,
                entry_price=price,
            )
            position.status = PositionStatus.OPEN
            position.orders.append(order_id)

            # Store position
            self.positions[position_id] = position

            # Update symbol mapping
            if symbol not in self.symbol_positions:
                self.symbol_positions[symbol] = []
            self.symbol_positions[symbol].append(position_id)

            self.logger.info(
                f"Opened new position {position_id}: "
                f"{side.value} {quantity} {symbol} @ {price}"
            )

            # Emit position event
            await self._emit_position_event(position)

    async def _reduce_positions(
        self, symbol: str, side: OrderSide, quantity: Decimal, price: Decimal, order_id: str
    ):
        """Reduce positions (close/partial close)."""
        # For reducing positions, side is opposite to position side
        position_side = PositionSide.SHORT if side == OrderSide.BUY else PositionSide.LONG

        # Find positions to reduce
        positions_to_reduce = [
            pos for pos in self.positions.values()
            if (pos.symbol == symbol and 
                pos.side == position_side and 
                pos.status == PositionStatus.OPEN)
        ]

        if not positions_to_reduce:
            self.logger.warning(
                f"No open positions found to reduce for {symbol} {position_side.value}"
            )
            return

        # Sort by creation time (FIFO)
        positions_to_reduce.sort(key=lambda p: p.created_at)

        remaining_quantity = quantity

        for position in positions_to_reduce:
            if remaining_quantity <= 0:
                break

            reduction_quantity = min(remaining_quantity, position.size)
            realized_pnl = position.reduce_size(reduction_quantity, price)

            position.orders.append(order_id)

            self.logger.info(
                f"Reduced position {position.position_id} by {reduction_quantity}, "
                f"realized P&L: {realized_pnl}"
            )

            # Emit position event
            await self._emit_position_event(position)

            remaining_quantity -= reduction_quantity

            # Remove from tracking if fully closed
            if position.status == PositionStatus.CLOSED:
                await self._close_position(position.position_id)

        if remaining_quantity > 0:
            self.logger.warning(
                f"Could not reduce full quantity {quantity}, "
                f"remaining: {remaining_quantity}"
            )

    async def _close_position(self, position_id: str):
        """Close and remove a position from tracking."""
        position = self.positions.get(position_id)
        if not position:
            return

        # Remove from symbol mapping
        if position.symbol in self.symbol_positions:
            try:
                self.symbol_positions[position.symbol].remove(position_id)
                if not self.symbol_positions[position.symbol]:
                    del self.symbol_positions[position.symbol]
            except ValueError:
                pass

        # Remove from positions
        del self.positions[position_id]

        self.logger.info(f"Closed position {position_id}")

    def _find_position(
        self, symbol: str, side: PositionSide, status: PositionStatus
    ) -> Optional[Position]:
        """Find position matching criteria."""
        for position in self.positions.values():
            if (position.symbol == symbol and 
                position.side == side and 
                position.status == status):
                return position
        return None

    async def _emit_position_event(self, position: Position):
        """Emit position event."""
        position_event = PositionEvent(
            source=self.__class__.__name__,
            position_id=position.position_id,
            symbol=position.symbol,
            side=position.side,
            status=position.status,
            size=position.size,
            entry_price=position.entry_price,
            current_price=position.current_price,
            unrealized_pnl=position.unrealized_pnl,
            realized_pnl=position.realized_pnl,
            stop_loss=position.stop_loss,
            take_profit=position.take_profit,
            risk_reward_ratio=(
                position.calculate_risk_reward_ratio(position.current_price)
                if position.current_price
                else None
            ),
        )

        await self._emit_event(position_event)

    async def _update_positions_loop(self):
        """Background loop for updating position P&L."""
        self.logger.info("Starting position update loop")

        while True:
            try:
                await self._update_all_positions()
                await asyncio.sleep(self.update_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in position update loop: {str(e)}")
                await asyncio.sleep(self.update_interval)

        self.logger.info("Position update loop stopped")

    async def _update_all_positions(self):
        """Update all active positions with current market data."""
        if not self.positions:
            return

        # Get unique symbols
        symbols = set(pos.symbol for pos in self.positions.values())

        # Update current prices for all symbols
        for symbol in symbols:
            try:
                if symbol not in self.current_prices:
                    ticker = await self.binance_client.futures_symbol_ticker(symbol=symbol)
                    self.current_prices[symbol] = Decimal(str(ticker["price"]))
            except Exception as e:
                self.logger.error(f"Failed to get price for {symbol}: {str(e)}")

        # Update positions
        for position in self.positions.values():
            if position.status == PositionStatus.OPEN:
                current_price = self.current_prices.get(position.symbol)
                if current_price:
                    old_pnl = position.unrealized_pnl
                    position.update_price(current_price)

                    # Emit event if P&L changed significantly (> 1%)
                    if old_pnl and position.unrealized_pnl:
                        pnl_change = abs(position.unrealized_pnl - old_pnl)
                        if old_pnl != 0 and (pnl_change / abs(old_pnl)) > 0.01:
                            await self._emit_position_event(position)

    async def get_positions(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get all positions or positions for a specific symbol."""
        if symbol:
            position_ids = self.symbol_positions.get(symbol, [])
            positions = [self.positions[pid] for pid in position_ids if pid in self.positions]
        else:
            positions = list(self.positions.values())

        return [pos.to_dict() for pos in positions]

    async def get_position(self, position_id: str) -> Optional[Dict]:
        """Get a specific position by ID."""
        position = self.positions.get(position_id)
        return position.to_dict() if position else None

    async def get_total_pnl(self) -> Dict[str, Decimal]:
        """Get total realized and unrealized P&L across all positions."""
        total_realized = Decimal("0")
        total_unrealized = Decimal("0")

        for position in self.positions.values():
            total_realized += position.realized_pnl
            if position.unrealized_pnl:
                total_unrealized += position.unrealized_pnl

        return {
            "realized_pnl": total_realized,
            "unrealized_pnl": total_unrealized,
            "total_pnl": total_realized + total_unrealized,
        }

    async def set_stop_loss(self, position_id: str, stop_price: Decimal) -> bool:
        """Set stop loss for a position."""
        position = self.positions.get(position_id)
        if not position:
            return False

        position.stop_loss = stop_price
        position.updated_at = datetime.utcnow()

        await self._emit_position_event(position)
        return True

    async def set_take_profit(self, position_id: str, take_profit_price: Decimal) -> bool:
        """Set take profit for a position."""
        position = self.positions.get(position_id)
        if not position:
            return False

        position.take_profit = take_profit_price
        position.updated_at = datetime.utcnow()

        await self._emit_position_event(position)
        return True