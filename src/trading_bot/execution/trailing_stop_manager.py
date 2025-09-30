"""
Trailing stop manager for dynamic profit protection.

This module implements trailing stop logic to protect profits while allowing
for continued upside participation in favorable price movements.
"""

import asyncio
import logging
from decimal import Decimal
from typing import Dict, Optional, Set
from uuid import uuid4

from ..core.base_component import BaseComponent
from ..core.events import (
    OrderSide,
    OrderType,
    PositionEvent,
    PositionSide,
    PositionStatus,
    TrailingStopEvent,
)
from .order_executor import OrderExecutor, OrderRequest


class TrailingStopConfig:
    """Configuration for trailing stop logic."""

    def __init__(
        self,
        trail_distance_percentage: float = 0.02,  # 2% trailing distance
        activation_threshold_rr: float = 1.0,  # Activate after 1:1 RR
        min_trail_distance: Optional[Decimal] = None,  # Minimum trail distance in price
        max_trail_distance: Optional[Decimal] = None,  # Maximum trail distance in price
        use_atr_for_distance: bool = False,  # Use ATR for dynamic distance
        atr_multiplier: float = 2.0,  # ATR multiplier for distance calculation
        enable_after_partial_profit: bool = True,  # Enable after first profit target
    ):
        self.trail_distance_percentage = trail_distance_percentage
        self.activation_threshold_rr = activation_threshold_rr
        self.min_trail_distance = min_trail_distance
        self.max_trail_distance = max_trail_distance
        self.use_atr_for_distance = use_atr_for_distance
        self.atr_multiplier = atr_multiplier
        self.enable_after_partial_profit = enable_after_partial_profit


class TrailingStopData:
    """Data structure for tracking trailing stop information."""

    def __init__(
        self,
        position_id: str,
        symbol: str,
        side: PositionSide,
        activation_price: Decimal,
        initial_stop_price: Decimal,
        trail_distance: Decimal,
    ):
        self.position_id = position_id
        self.symbol = symbol
        self.side = side
        self.activation_price = activation_price
        self.current_stop_price = initial_stop_price
        self.trail_distance = trail_distance
        
        # Tracking data
        self.highest_price = activation_price  # For long positions
        self.lowest_price = activation_price   # For short positions
        self.is_active = True
        self.created_at = datetime.utcnow()
        self.last_updated = datetime.utcnow()
        self.adjustment_count = 0

    def update_trail(self, current_price: Decimal) -> Optional[Decimal]:
        """
        Update trailing stop based on current price.
        
        Args:
            current_price: Current market price
            
        Returns:
            New stop price if updated, None if no change
        """
        old_stop = self.current_stop_price
        
        if self.side == PositionSide.LONG:
            # Update highest price seen
            if current_price > self.highest_price:
                self.highest_price = current_price
                
            # Calculate new stop price
            new_stop = self.highest_price - self.trail_distance
            
            # Only move stop up (more favorable)
            if new_stop > self.current_stop_price:
                self.current_stop_price = new_stop
                self.adjustment_count += 1
                self.last_updated = datetime.utcnow()
                return new_stop
                
        else:  # SHORT position
            # Update lowest price seen
            if current_price < self.lowest_price:
                self.lowest_price = current_price
                
            # Calculate new stop price
            new_stop = self.lowest_price + self.trail_distance
            
            # Only move stop down (more favorable)
            if new_stop < self.current_stop_price:
                self.current_stop_price = new_stop
                self.adjustment_count += 1
                self.last_updated = datetime.utcnow()
                return new_stop
                
        return None

    def to_dict(self) -> Dict:
        """Convert trailing stop data to dictionary."""
        return {
            "position_id": self.position_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "activation_price": str(self.activation_price),
            "current_stop_price": str(self.current_stop_price),
            "trail_distance": str(self.trail_distance),
            "highest_price": str(self.highest_price),
            "lowest_price": str(self.lowest_price),
            "is_active": self.is_active,
            "adjustment_count": self.adjustment_count,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
        }


class TrailingStopManager(BaseComponent):
    """
    Trailing stop manager for dynamic profit protection.

    Manages trailing stops to protect profits while allowing continued
    upside participation during favorable price movements.
    """

    def __init__(
        self,
        order_executor: OrderExecutor,
        event_bus=None,
        config: Optional[TrailingStopConfig] = None,
    ):
        """
        Initialize the trailing stop manager.

        Args:
            order_executor: Order executor for stop loss updates
            event_bus: Event bus for communication
            config: Configuration for trailing stop behavior
        """
        super().__init__(name=self.__class__.__name__)
        self.event_bus = event_bus
        self.order_executor = order_executor
        self.config = config or TrailingStopConfig()

        # Active trailing stops
        self.trailing_stops: Dict[str, TrailingStopData] = {}
        self.stop_orders: Dict[str, str] = {}  # position_id -> stop_order_id


        # Register event handlers
        self._register_event_handlers()

    def _register_event_handlers(self):
        """Register event handlers for position updates."""
        if self.event_bus:
            self.event_bus.subscribe("PositionEvent", self._handle_position_event)
            self.event_bus.subscribe("TakeProfitEvent", self._handle_take_profit_event)

    async def _start(self):
        """Start the trailing stop manager."""
        self.logger.info("TrailingStopManager started")

    async def _stop(self):
        """Stop the trailing stop manager."""
        # Cancel all active trailing stops
        await self._cancel_all_trailing_stops()

        self.logger.info("TrailingStopManager stopped")

    async def _handle_position_event(self, position_event: PositionEvent):
        """Handle position events to manage trailing stops."""
        try:
            position_id = position_event.position_id
            
            if position_event.status == PositionStatus.OPEN:
                # Check if position qualifies for trailing stop activation
                if self._should_activate_trailing_stop(position_event):
                    await self._activate_trailing_stop(position_event)
                
                # Update existing trailing stop
                elif position_id in self.trailing_stops and position_event.current_price:
                    await self._update_trailing_stop(position_event)
                    
            elif position_event.status == PositionStatus.CLOSED:
                # Remove trailing stop for closed position
                await self._remove_trailing_stop(position_id)
                
        except Exception as e:
            self.logger.error(f"Error handling position event: {str(e)}")

    async def _handle_take_profit_event(self, take_profit_event):
        """Handle take profit events to potentially activate trailing stops."""
        try:
            if self.config.enable_after_partial_profit:
                position_id = take_profit_event.position_id
                
                # Enable trailing stop after partial profit is taken
                if position_id not in self.trailing_stops:
                    self.logger.info(
                        f"Activating trailing stop for position {position_id} "
                        "after partial profit taking"
                    )
                    # This would require getting current position data
                    # For now, we log the activation trigger
                    
        except Exception as e:
            self.logger.error(f"Error handling take profit event: {str(e)}")

    def _should_activate_trailing_stop(self, position_event: PositionEvent) -> bool:
        """Check if position should activate trailing stop."""
        position_id = position_event.position_id
        
        # Skip if already has trailing stop
        if position_id in self.trailing_stops:
            return False
            
        # Skip if no stop loss or current price
        if not position_event.stop_loss or not position_event.current_price:
            return False
            
        # Check if position has reached activation threshold
        current_rr = self._calculate_risk_reward_ratio(position_event)
        if current_rr is None or current_rr < self.config.activation_threshold_rr:
            return False
            
        return True

    def _calculate_risk_reward_ratio(self, position_event: PositionEvent) -> Optional[float]:
        """Calculate current risk-reward ratio."""
        if not position_event.stop_loss or not position_event.current_price:
            return None

        risk = abs(position_event.entry_price - position_event.stop_loss)
        if risk == 0:
            return None

        if position_event.side == PositionSide.LONG:
            reward = position_event.current_price - position_event.entry_price
        else:  # SHORT
            reward = position_event.entry_price - position_event.current_price

        if reward <= 0:
            return 0.0

        return float(reward / risk)

    async def _activate_trailing_stop(self, position_event: PositionEvent):
        """Activate trailing stop for a position."""
        position_id = position_event.position_id
        current_price = position_event.current_price
        
        # Calculate trail distance
        trail_distance = self._calculate_trail_distance(position_event)
        
        # Calculate initial stop price
        if position_event.side == PositionSide.LONG:
            initial_stop = current_price - trail_distance
        else:  # SHORT
            initial_stop = current_price + trail_distance
            
        # Create trailing stop data
        trailing_stop = TrailingStopData(
            position_id=position_id,
            symbol=position_event.symbol,
            side=position_event.side,
            activation_price=current_price,
            initial_stop_price=initial_stop,
            trail_distance=trail_distance,
        )
        
        # Store trailing stop
        self.trailing_stops[position_id] = trailing_stop
        
        self.logger.info(
            f"Activated trailing stop for position {position_id}: "
            f"activation price {current_price}, initial stop {initial_stop}, "
            f"trail distance {trail_distance}"
        )
        
        # Emit trailing stop event
        await self._emit_trailing_stop_event(trailing_stop, current_price)
        
        # Place initial stop order
        await self._place_stop_order(trailing_stop)

    def _calculate_trail_distance(self, position_event: PositionEvent) -> Decimal:
        """Calculate trailing distance based on configuration."""
        current_price = position_event.current_price
        
        if self.config.use_atr_for_distance:
            # TODO: Implement ATR-based distance calculation
            # For now, use percentage-based calculation
            distance = current_price * Decimal(str(self.config.trail_distance_percentage))
        else:
            distance = current_price * Decimal(str(self.config.trail_distance_percentage))
            
        # Apply min/max constraints
        if self.config.min_trail_distance and distance < self.config.min_trail_distance:
            distance = self.config.min_trail_distance
            
        if self.config.max_trail_distance and distance > self.config.max_trail_distance:
            distance = self.config.max_trail_distance
            
        return distance

    async def _update_trailing_stop(self, position_event: PositionEvent):
        """Update existing trailing stop based on current price."""
        position_id = position_event.position_id
        trailing_stop = self.trailing_stops.get(position_id)
        
        if not trailing_stop or not trailing_stop.is_active:
            return
            
        current_price = position_event.current_price
        new_stop_price = trailing_stop.update_trail(current_price)
        
        if new_stop_price:
            self.logger.info(
                f"Updated trailing stop for position {position_id}: "
                f"new stop price {new_stop_price} (was {trailing_stop.current_stop_price})"
            )
            
            # Emit trailing stop event
            await self._emit_trailing_stop_event(trailing_stop, current_price, new_stop_price)
            
            # Update stop order
            await self._update_stop_order(trailing_stop)

    async def _emit_trailing_stop_event(
        self, 
        trailing_stop: TrailingStopData, 
        current_price: Decimal,
        new_stop_price: Optional[Decimal] = None
    ):
        """Emit trailing stop event."""
        event = TrailingStopEvent(
            source=self.__class__.__name__,
            position_id=trailing_stop.position_id,
            symbol=trailing_stop.symbol,
            new_stop_price=new_stop_price or trailing_stop.current_stop_price,
            previous_stop_price=trailing_stop.current_stop_price if new_stop_price else None,
            current_price=current_price,
            trail_distance=trailing_stop.trail_distance,
            is_active=trailing_stop.is_active,
        )
        
        await self._emit_event(event)

    async def _place_stop_order(self, trailing_stop: TrailingStopData):
        """Place initial stop loss order for trailing stop."""
        try:
            # Determine order side (opposite to position)
            order_side = (
                OrderSide.SELL if trailing_stop.side == PositionSide.LONG 
                else OrderSide.BUY
            )
            
            # Create stop order request
            order_request = OrderRequest(
                symbol=trailing_stop.symbol,
                side=order_side,
                order_type=OrderType.STOP,
                quantity=Decimal("0"),  # This would need to be set based on position size
                stop_price=trailing_stop.current_stop_price,
                client_order_id=f"trail_stop_{trailing_stop.position_id}_{uuid4().hex[:8]}",
                reduce_only=True,
            )
            
            # Note: In a real implementation, you would need to:
            # 1. Get the current position size
            # 2. Execute the stop order
            # 3. Store the order ID for future updates
            
            self.logger.info(
                f"Would place stop order for position {trailing_stop.position_id} "
                f"at {trailing_stop.current_stop_price}"
            )
            
        except Exception as e:
            self.logger.error(
                f"Failed to place stop order for position {trailing_stop.position_id}: {str(e)}"
            )

    async def _update_stop_order(self, trailing_stop: TrailingStopData):
        """Update existing stop order with new price."""
        try:
            position_id = trailing_stop.position_id
            
            # In a real implementation, you would:
            # 1. Cancel the existing stop order
            # 2. Place a new stop order at the updated price
            # 3. Update the stored order ID
            
            self.logger.info(
                f"Would update stop order for position {position_id} "
                f"to {trailing_stop.current_stop_price}"
            )
            
        except Exception as e:
            self.logger.error(
                f"Failed to update stop order for position {position_id}: {str(e)}"
            )

    async def _remove_trailing_stop(self, position_id: str):
        """Remove trailing stop for closed position."""
        trailing_stop = self.trailing_stops.pop(position_id, None)
        if trailing_stop:
            trailing_stop.is_active = False
            
            # Cancel associated stop order
            stop_order_id = self.stop_orders.pop(position_id, None)
            if stop_order_id:
                # Cancel the stop order
                self.logger.info(
                    f"Would cancel stop order {stop_order_id} for closed position {position_id}"
                )
                
            self.logger.info(f"Removed trailing stop for position {position_id}")

    async def _cancel_all_trailing_stops(self):
        """Cancel all active trailing stops."""
        for position_id in list(self.trailing_stops.keys()):
            await self._remove_trailing_stop(position_id)

    async def manually_activate_trailing_stop(
        self,
        position_id: str,
        symbol: str,
        side: PositionSide,
        current_price: Decimal,
        trail_distance: Optional[Decimal] = None,
    ) -> bool:
        """
        Manually activate trailing stop for a position.
        
        Args:
            position_id: Position identifier
            symbol: Trading symbol
            side: Position side
            current_price: Current market price
            trail_distance: Custom trail distance (optional)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if position_id in self.trailing_stops:
                self.logger.warning(f"Trailing stop already active for position {position_id}")
                return False
                
            # Use custom distance or calculate default
            if trail_distance is None:
                trail_distance = current_price * Decimal(str(self.config.trail_distance_percentage))
                
            # Calculate initial stop price
            if side == PositionSide.LONG:
                initial_stop = current_price - trail_distance
            else:
                initial_stop = current_price + trail_distance
                
            # Create trailing stop
            trailing_stop = TrailingStopData(
                position_id=position_id,
                symbol=symbol,
                side=side,
                activation_price=current_price,
                initial_stop_price=initial_stop,
                trail_distance=trail_distance,
            )
            
            self.trailing_stops[position_id] = trailing_stop
            
            self.logger.info(
                f"Manually activated trailing stop for position {position_id}"
            )
            
            # Emit event and place order
            await self._emit_trailing_stop_event(trailing_stop, current_price)
            await self._place_stop_order(trailing_stop)
            
            return True
            
        except Exception as e:
            self.logger.error(
                f"Failed to manually activate trailing stop for position {position_id}: {str(e)}"
            )
            return False

    async def deactivate_trailing_stop(self, position_id: str) -> bool:
        """
        Deactivate trailing stop for a position.
        
        Args:
            position_id: Position identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            trailing_stop = self.trailing_stops.get(position_id)
            if not trailing_stop:
                self.logger.warning(f"No trailing stop found for position {position_id}")
                return False
                
            trailing_stop.is_active = False
            
            # Cancel stop order
            stop_order_id = self.stop_orders.pop(position_id, None)
            if stop_order_id:
                # Would cancel the order here
                pass
                
            self.logger.info(f"Deactivated trailing stop for position {position_id}")
            return True
            
        except Exception as e:
            self.logger.error(
                f"Failed to deactivate trailing stop for position {position_id}: {str(e)}"
            )
            return False

    async def get_trailing_stops(self) -> Dict[str, Dict]:
        """Get all active trailing stops."""
        return {
            position_id: trailing_stop.to_dict()
            for position_id, trailing_stop in self.trailing_stops.items()
        }

    async def get_trailing_stop(self, position_id: str) -> Optional[Dict]:
        """Get specific trailing stop data."""
        trailing_stop = self.trailing_stops.get(position_id)
        return trailing_stop.to_dict() if trailing_stop else None

    async def update_trail_distance(self, position_id: str, new_distance: Decimal) -> bool:
        """
        Update trail distance for active trailing stop.
        
        Args:
            position_id: Position identifier
            new_distance: New trailing distance
            
        Returns:
            True if successful, False otherwise
        """
        try:
            trailing_stop = self.trailing_stops.get(position_id)
            if not trailing_stop:
                return False
                
            trailing_stop.trail_distance = new_distance
            
            self.logger.info(
                f"Updated trail distance for position {position_id} to {new_distance}"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(
                f"Failed to update trail distance for position {position_id}: {str(e)}"
            )
            return False

    async def get_trailing_stop_stats(self) -> Dict:
        """Get trailing stop statistics."""
        active_count = len([ts for ts in self.trailing_stops.values() if ts.is_active])
        total_adjustments = sum(ts.adjustment_count for ts in self.trailing_stops.values())
        
        return {
            "total_trailing_stops": len(self.trailing_stops),
            "active_trailing_stops": active_count,
            "total_adjustments": total_adjustments,
            "average_adjustments": (
                total_adjustments / len(self.trailing_stops) 
                if self.trailing_stops else 0
            ),
        }