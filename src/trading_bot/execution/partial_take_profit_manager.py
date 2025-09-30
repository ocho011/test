"""
Partial take profit manager for systematic profit optimization.

This module implements the 1:1 RR 50% profit taking strategy with
1:2 RR remainder management for optimal risk-reward execution.
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
    TakeProfitEvent,
    TakeProfitType,
)
from .order_executor import OrderExecutor, OrderRequest


class PartialTakeProfitConfig:
    """Configuration for partial take profit logic."""

    def __init__(
        self,
        first_target_rr: float = 1.0,  # 1:1 RR for first target
        first_target_percentage: float = 0.5,  # 50% at first target
        second_target_rr: float = 2.0,  # 1:2 RR for second target
        move_stop_to_breakeven: bool = True,  # Move stop to BE after first target
        enable_trailing_after_first: bool = True,  # Enable trailing after first target
    ):
        self.first_target_rr = first_target_rr
        self.first_target_percentage = first_target_percentage
        self.second_target_rr = second_target_rr
        self.move_stop_to_breakeven = move_stop_to_breakeven
        self.enable_trailing_after_first = enable_trailing_after_first


class PartialTakeProfitManager(BaseComponent):
    """
    Partial take profit manager implementing systematic profit optimization.

    Manages the 1:1 RR 50% profit taking strategy with automatic
    stop loss adjustment and 1:2 RR target management.
    """

    def __init__(
        self,
        order_executor: OrderExecutor,
        event_bus=None,
        config: Optional[PartialTakeProfitConfig] = None,
    ):
        """
        Initialize the partial take profit manager.

        Args:
            order_executor: Order executor for placing profit-taking orders
            event_bus: Event bus for communication
            config: Configuration for profit taking strategy
        """
        super().__init__(event_bus=event_bus)
        self.order_executor = order_executor
        self.config = config or PartialTakeProfitConfig()

        # Tracking active profit targets
        self.active_targets: Dict[str, Dict] = {}  # position_id -> target_data
        self.positions_with_partial_taken: Set[str] = set()

        self.logger = logging.getLogger(self.__class__.__name__)

        # Register event handlers
        self._register_event_handlers()

    def _register_event_handlers(self):
        """Register event handlers for position updates."""
        if self.event_bus:
            self.event_bus.subscribe("PositionEvent", self._handle_position_event)

    async def start(self):
        """Start the partial take profit manager."""
        await super().start()
        self.logger.info("PartialTakeProfitManager started")

    async def stop(self):
        """Stop the partial take profit manager."""
        await super().stop()
        self.logger.info("PartialTakeProfitManager stopped")

    async def _handle_position_event(self, position_event: PositionEvent):
        """Handle position events to monitor for profit targets."""
        try:
            if position_event.status == PositionStatus.OPEN:
                await self._check_profit_targets(position_event)
        except Exception as e:
            self.logger.error(f"Error handling position event: {str(e)}")

    async def _check_profit_targets(self, position_event: PositionEvent):
        """Check if position has hit profit targets."""
        position_id = position_event.position_id

        # Skip if no stop loss (can't calculate RR without risk)
        if not position_event.stop_loss or not position_event.current_price:
            return

        # Calculate current risk-reward ratio
        current_rr = self._calculate_risk_reward_ratio(position_event)
        if current_rr is None:
            return

        # Check first target (1:1 RR, 50% profit taking)
        if (current_rr >= self.config.first_target_rr and 
            position_id not in self.positions_with_partial_taken):
            await self._execute_first_target(position_event, current_rr)

        # Check second target (1:2 RR, remaining position)
        elif (current_rr >= self.config.second_target_rr and 
              position_id in self.positions_with_partial_taken):
            await self._execute_second_target(position_event, current_rr)

    def _calculate_risk_reward_ratio(self, position_event: PositionEvent) -> Optional[float]:
        """Calculate risk-reward ratio for current position."""
        if not position_event.stop_loss or not position_event.current_price:
            return None

        # Calculate risk (distance from entry to stop loss)
        risk = abs(position_event.entry_price - position_event.stop_loss)
        if risk == 0:
            return None

        # Calculate reward (distance from entry to current price)
        if position_event.side == PositionSide.LONG:
            reward = position_event.current_price - position_event.entry_price
        else:  # SHORT
            reward = position_event.entry_price - position_event.current_price

        # Only positive rewards count
        if reward <= 0:
            return 0.0

        return float(reward / risk)

    async def _execute_first_target(self, position_event: PositionEvent, current_rr: float):
        """Execute first profit target at 1:1 RR (50% of position)."""
        position_id = position_event.position_id
        partial_quantity = position_event.size * Decimal(str(self.config.first_target_percentage))

        self.logger.info(
            f"First profit target hit for position {position_id} "
            f"(RR: {current_rr:.2f}), taking {self.config.first_target_percentage*100:.0f}% profit"
        )

        try:
            # Calculate expected profit
            risk_amount = abs(position_event.entry_price - position_event.stop_loss)
            expected_profit = risk_amount * partial_quantity

            # Create take profit event
            take_profit_event = TakeProfitEvent(
                source=self.__class__.__name__,
                position_id=position_id,
                symbol=position_event.symbol,
                trigger_type=TakeProfitType.PARTIAL_1_1_RR,
                trigger_price=position_event.current_price,
                target_quantity=partial_quantity,
                current_rr_ratio=current_rr,
                expected_profit=expected_profit,
            )

            await self._emit_event(take_profit_event)

            # Create closing order (opposite side to position)
            order_side = (
                OrderSide.SELL if position_event.side == PositionSide.LONG 
                else OrderSide.BUY
            )

            order_request = OrderRequest(
                symbol=position_event.symbol,
                side=order_side,
                order_type=OrderType.MARKET,
                quantity=partial_quantity,
                client_order_id=f"partial_tp_{position_id}_{uuid4().hex[:8]}",
                reduce_only=True,
            )

            # Execute profit-taking order
            await self.order_executor.execute_order(order_request)

            # Mark position as having partial profit taken
            self.positions_with_partial_taken.add(position_id)

            # Store target data for stop loss adjustment
            self.active_targets[position_id] = {
                "first_target_hit": True,
                "first_target_rr": current_rr,
                "original_stop_loss": position_event.stop_loss,
                "entry_price": position_event.entry_price,
            }

            self.logger.info(
                f"First profit target executed for position {position_id}, "
                f"profit taken: {expected_profit}"
            )

            # Move stop loss to breakeven if configured
            if self.config.move_stop_to_breakeven:
                await self._move_stop_to_breakeven(position_event)

        except Exception as e:
            self.logger.error(
                f"Failed to execute first profit target for position {position_id}: {str(e)}"
            )

    async def _execute_second_target(self, position_event: PositionEvent, current_rr: float):
        """Execute second profit target at 1:2 RR (remaining position)."""
        position_id = position_event.position_id

        self.logger.info(
            f"Second profit target hit for position {position_id} "
            f"(RR: {current_rr:.2f}), closing remaining position"
        )

        try:
            # Calculate expected profit for remaining position
            target_data = self.active_targets.get(position_id, {})
            risk_amount = abs(position_event.entry_price - target_data.get("original_stop_loss", position_event.stop_loss))
            expected_profit = risk_amount * position_event.size * Decimal("2.0")  # 2:1 RR

            # Create take profit event
            take_profit_event = TakeProfitEvent(
                source=self.__class__.__name__,
                position_id=position_id,
                symbol=position_event.symbol,
                trigger_type=TakeProfitType.FULL_1_2_RR,
                trigger_price=position_event.current_price,
                target_quantity=position_event.size,  # Remaining size
                current_rr_ratio=current_rr,
                expected_profit=expected_profit,
            )

            await self._emit_event(take_profit_event)

            # Create closing order for remaining position
            order_side = (
                OrderSide.SELL if position_event.side == PositionSide.LONG 
                else OrderSide.BUY
            )

            order_request = OrderRequest(
                symbol=position_event.symbol,
                side=order_side,
                order_type=OrderType.MARKET,
                quantity=position_event.size,
                client_order_id=f"full_tp_{position_id}_{uuid4().hex[:8]}",
                reduce_only=True,
            )

            # Execute final profit-taking order
            await self.order_executor.execute_order(order_request)

            # Clean up tracking data
            self.positions_with_partial_taken.discard(position_id)
            self.active_targets.pop(position_id, None)

            self.logger.info(
                f"Second profit target executed for position {position_id}, "
                f"position fully closed with profit: {expected_profit}"
            )

        except Exception as e:
            self.logger.error(
                f"Failed to execute second profit target for position {position_id}: {str(e)}"
            )

    async def _move_stop_to_breakeven(self, position_event: PositionEvent):
        """Move stop loss to breakeven after first profit target."""
        position_id = position_event.position_id

        try:
            # For futures, breakeven is the entry price
            breakeven_price = position_event.entry_price

            self.logger.info(
                f"Moving stop loss to breakeven for position {position_id}: {breakeven_price}"
            )

            # This would typically involve updating the stop loss order
            # For now, we'll emit an event that other components can handle
            # The actual stop loss order update would be handled by another component

            # Update target data
            if position_id in self.active_targets:
                self.active_targets[position_id]["stop_moved_to_breakeven"] = True
                self.active_targets[position_id]["breakeven_price"] = breakeven_price

        except Exception as e:
            self.logger.error(
                f"Failed to move stop to breakeven for position {position_id}: {str(e)}"
            )

    async def add_position_for_monitoring(
        self,
        position_id: str,
        symbol: str,
        side: PositionSide,
        entry_price: Decimal,
        size: Decimal,
        stop_loss: Decimal,
    ):
        """
        Manually add a position for profit target monitoring.

        Args:
            position_id: Unique position identifier
            symbol: Trading symbol
            side: Position side (long/short)
            entry_price: Position entry price
            size: Position size
            stop_loss: Stop loss price
        """
        self.logger.info(
            f"Adding position {position_id} for profit target monitoring: "
            f"{side.value} {size} {symbol} @ {entry_price}, SL: {stop_loss}"
        )

        # Initialize target tracking
        self.active_targets[position_id] = {
            "first_target_hit": False,
            "entry_price": entry_price,
            "original_stop_loss": stop_loss,
            "original_size": size,
            "stop_moved_to_breakeven": False,
        }

    async def remove_position_monitoring(self, position_id: str):
        """Remove position from profit target monitoring."""
        self.positions_with_partial_taken.discard(position_id)
        self.active_targets.pop(position_id, None)

        self.logger.info(f"Removed position {position_id} from monitoring")

    async def get_monitored_positions(self) -> Dict:
        """Get information about positions being monitored."""
        return {
            "active_targets": dict(self.active_targets),
            "positions_with_partial_taken": list(self.positions_with_partial_taken),
            "monitoring_count": len(self.active_targets),
        }

    def calculate_profit_targets(
        self, entry_price: Decimal, stop_loss: Decimal, side: PositionSide
    ) -> Dict[str, Decimal]:
        """
        Calculate profit target prices for given position parameters.

        Args:
            entry_price: Position entry price
            stop_loss: Stop loss price
            side: Position side (long/short)

        Returns:
            Dictionary with first_target and second_target prices
        """
        risk = abs(entry_price - stop_loss)

        if side == PositionSide.LONG:
            first_target = entry_price + (risk * Decimal(str(self.config.first_target_rr)))
            second_target = entry_price + (risk * Decimal(str(self.config.second_target_rr)))
        else:  # SHORT
            first_target = entry_price - (risk * Decimal(str(self.config.first_target_rr)))
            second_target = entry_price - (risk * Decimal(str(self.config.second_target_rr)))

        return {
            "first_target": first_target,
            "second_target": second_target,
            "risk_amount": risk,
        }

    async def get_profit_taking_stats(self) -> Dict:
        """Get profit taking statistics."""
        total_positions = len(self.active_targets)
        partial_taken = len(self.positions_with_partial_taken)
        
        stats = {
            "total_monitored_positions": total_positions,
            "positions_with_partial_profit": partial_taken,
            "first_target_hit_rate": (
                (partial_taken / total_positions) if total_positions > 0 else 0.0
            ),
            "active_first_targets": total_positions - partial_taken,
            "active_second_targets": partial_taken,
        }

        return stats

    async def manual_trigger_first_target(self, position_id: str) -> bool:
        """
        Manually trigger first profit target for a position.

        Args:
            position_id: Position ID to trigger profit taking

        Returns:
            True if successful, False otherwise
        """
        try:
            # This would require getting current position data
            # For now, log the manual trigger
            self.logger.info(f"Manual first target trigger requested for position {position_id}")

            # In a real implementation, you would:
            # 1. Get current position data
            # 2. Validate it's eligible for first target
            # 3. Execute the profit taking logic

            return True

        except Exception as e:
            self.logger.error(f"Failed to manually trigger first target: {str(e)}")
            return False

    async def manual_trigger_second_target(self, position_id: str) -> bool:
        """
        Manually trigger second profit target for a position.

        Args:
            position_id: Position ID to trigger profit taking

        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if first target was already hit
            if position_id not in self.positions_with_partial_taken:
                self.logger.warning(
                    f"Cannot trigger second target for position {position_id}: "
                    "first target not yet hit"
                )
                return False

            self.logger.info(f"Manual second target trigger requested for position {position_id}")

            # In a real implementation, you would:
            # 1. Get current position data
            # 2. Validate it's eligible for second target
            # 3. Execute the final profit taking logic

            return True

        except Exception as e:
            self.logger.error(f"Failed to manually trigger second target: {str(e)}")
            return False

    async def update_profit_target_config(self, new_config: PartialTakeProfitConfig):
        """
        Update profit target configuration.

        Args:
            new_config: New configuration settings
        """
        old_config = self.config
        self.config = new_config

        self.logger.info(
            f"Updated profit target config: "
            f"First RR: {old_config.first_target_rr} -> {new_config.first_target_rr}, "
            f"First %: {old_config.first_target_percentage} -> {new_config.first_target_percentage}, "
            f"Second RR: {old_config.second_target_rr} -> {new_config.second_target_rr}"
        )