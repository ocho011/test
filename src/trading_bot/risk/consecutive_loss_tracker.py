"""
Consecutive loss tracker for risk management.

This module tracks consecutive losing trades and implements automatic trading halt
when the configured threshold is reached, with pattern analysis and recovery conditions.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from ..core.base_component import BaseComponent


class TradeResult(Enum):
    """Trade outcome classification."""

    WIN = "win"
    LOSS = "loss"
    BREAKEVEN = "breakeven"


class TradeRecord(BaseModel):
    """Record of a completed trade."""

    trade_id: str = Field(..., description="Unique trade identifier")
    symbol: str = Field(..., description="Trading symbol")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    entry_price: Decimal = Field(..., description="Trade entry price")
    exit_price: Decimal = Field(..., description="Trade exit price")
    quantity: Decimal = Field(..., description="Trade quantity")
    pnl: Decimal = Field(..., description="Profit/Loss amount")
    pnl_percentage: float = Field(..., description="P&L as percentage")
    result: TradeResult = Field(..., description="Trade outcome")
    commission: Decimal = Field(default=Decimal("0"), description="Trading commission")


class LossStreak(BaseModel):
    """Record of a consecutive loss streak."""

    start_timestamp: datetime = Field(..., description="When streak started")
    end_timestamp: Optional[datetime] = Field(None, description="When streak ended")
    trade_count: int = Field(..., description="Number of consecutive losses")
    total_loss: Decimal = Field(..., description="Total loss amount")
    total_loss_percentage: float = Field(..., description="Total loss as percentage")
    is_active: bool = Field(default=True, description="Whether streak is ongoing")


class ConsecutiveLossTracker(BaseComponent):
    """
    Track consecutive losing trades and implement protective measures.

    Monitors trade outcomes and automatically halts trading when too many
    consecutive losses occur, with pattern analysis and recovery tracking.
    """

    def __init__(
        self,
        name: str = "consecutive_loss_tracker",
        max_consecutive_losses: int = 3,
        loss_threshold_percentage: float = 0.01,  # 1% minimum loss to count
        cooldown_period_hours: int = 24,  # Hours to wait before allowing trading
        streak_reset_hours: int = 48,  # Hours without trading to reset streak
    ):
        """
        Initialize consecutive loss tracker.

        Args:
            name: Component name
            max_consecutive_losses: Maximum consecutive losses before halt
            loss_threshold_percentage: Minimum loss percentage to count as loss
            cooldown_period_hours: Hours to wait after halt before resuming
            streak_reset_hours: Hours without activity to reset streak
        """
        super().__init__(name)

        self.max_consecutive_losses = max_consecutive_losses
        self.loss_threshold_percentage = loss_threshold_percentage
        self.cooldown_period = timedelta(hours=cooldown_period_hours)
        self.streak_reset_period = timedelta(hours=streak_reset_hours)

        # Current state
        self.current_consecutive_losses = 0
        self.current_streak: Optional[LossStreak] = None
        self.trading_halted = False
        self.halt_timestamp: Optional[datetime] = None
        self.halt_reason: Optional[str] = None

        # Trade history
        self.trade_history: List[TradeRecord] = []
        self.loss_streaks: List[LossStreak] = []

        # Statistics
        self.total_trades = 0
        self.total_wins = 0
        self.total_losses = 0
        self.total_breakevens = 0
        self.longest_loss_streak = 0

    async def _start(self) -> None:
        """Start consecutive loss tracking."""
        self.logger.info("Starting consecutive loss tracker")
        self.logger.info(f"Max consecutive losses: {self.max_consecutive_losses}")
        self.logger.info(f"Loss threshold: {self.loss_threshold_percentage:.1%}")
        self.logger.info(f"Cooldown period: {self.cooldown_period}")

    async def _stop(self) -> None:
        """Stop consecutive loss tracking."""
        self.logger.info("Stopping consecutive loss tracker")

    def record_trade(self, trade: TradeRecord) -> None:
        """
        Record a completed trade and update loss tracking.

        Args:
            trade: Completed trade record
        """
        self.logger.debug(
            f"Recording trade {trade.trade_id}: {trade.result.value}, P&L: {trade.pnl}"
        )

        # Add to history
        self.trade_history.append(trade)
        self.total_trades += 1

        # Clean old history (keep last 1000 trades)
        if len(self.trade_history) > 1000:
            self.trade_history = self.trade_history[-1000:]

        # Update statistics and consecutive loss tracking
        self._update_statistics(trade)
        self._update_consecutive_loss_tracking(trade)

        # Check for trading halt conditions
        self._check_halt_conditions()

        # Check for automatic recovery conditions
        self._check_recovery_conditions()

    def _update_statistics(self, trade: TradeRecord) -> None:
        """Update trade statistics."""
        if trade.result == TradeResult.WIN:
            self.total_wins += 1
        elif trade.result == TradeResult.LOSS:
            self.total_losses += 1
        else:
            self.total_breakevens += 1

    def _update_consecutive_loss_tracking(self, trade: TradeRecord) -> None:
        """Update consecutive loss tracking based on trade result."""
        if (
            trade.result == TradeResult.LOSS
            and abs(trade.pnl_percentage) >= self.loss_threshold_percentage
        ):
            # Consecutive loss detected
            self.current_consecutive_losses += 1

            if self.current_streak is None:
                # Start new loss streak
                self.current_streak = LossStreak(
                    start_timestamp=trade.timestamp,
                    trade_count=1,
                    total_loss=abs(trade.pnl),
                    total_loss_percentage=abs(trade.pnl_percentage),
                )
            else:
                # Continue existing streak
                self.current_streak.trade_count += 1
                self.current_streak.total_loss += abs(trade.pnl)
                self.current_streak.total_loss_percentage += abs(trade.pnl_percentage)

            # Update longest streak record
            if self.current_consecutive_losses > self.longest_loss_streak:
                self.longest_loss_streak = self.current_consecutive_losses

            self.logger.warning(
                f"Consecutive loss #{self.current_consecutive_losses} recorded. "
                f"P&L: {trade.pnl:.2f} ({trade.pnl_percentage:.2%})"
            )

        else:
            # Win or breakeven - reset consecutive loss count
            if self.current_consecutive_losses > 0:
                self.logger.info(
                    f"Consecutive loss streak ended at {self.current_consecutive_losses} losses"
                )

                # End current streak
                if self.current_streak:
                    self.current_streak.end_timestamp = trade.timestamp
                    self.current_streak.is_active = False
                    self.loss_streaks.append(self.current_streak)
                    self.current_streak = None

                # Reset consecutive loss count
                self.current_consecutive_losses = 0

    def _check_halt_conditions(self) -> None:
        """Check if trading should be halted due to consecutive losses."""
        if (
            self.current_consecutive_losses >= self.max_consecutive_losses
            and not self.trading_halted
        ):

            reason = (
                f"Maximum consecutive losses reached: {self.current_consecutive_losses} "
                f"(limit: {self.max_consecutive_losses})"
            )

            self._halt_trading(reason)

    def _halt_trading(self, reason: str) -> None:
        """Halt trading due to consecutive losses."""
        self.trading_halted = True
        self.halt_reason = reason
        self.halt_timestamp = datetime.utcnow()

        self.logger.critical(f"TRADING HALTED: {reason}")

        # Emit risk event (in real implementation would use event bus)
        self.logger.error(f"Risk event: Trading halted due to consecutive losses")

    def _check_recovery_conditions(self) -> None:
        """Check if trading can be automatically resumed."""
        if not self.trading_halted or not self.halt_timestamp:
            return

        current_time = datetime.utcnow()

        # Check if cooldown period has passed
        if current_time - self.halt_timestamp >= self.cooldown_period:
            self.logger.info("Cooldown period completed, checking recovery conditions")

            # Additional recovery conditions can be implemented here
            # For now, automatically resume after cooldown
            self._resume_trading("Cooldown period completed")

    def _resume_trading(self, reason: str) -> None:
        """Resume trading after recovery."""
        if self.trading_halted:
            self.logger.info(f"TRADING RESUMED: {reason}")
            self.trading_halted = False
            self.halt_reason = None
            self.halt_timestamp = None

    def check_trading_allowed(self) -> Tuple[bool, Optional[str]]:
        """
        Check if trading is currently allowed.

        Returns:
            Tuple of (allowed, reason_if_not_allowed)
        """
        if self.trading_halted:
            return False, self.halt_reason

        # Check if we're too close to the limit
        if self.current_consecutive_losses >= self.max_consecutive_losses - 1:
            return (
                False,
                f"Near consecutive loss limit ({self.current_consecutive_losses}/{self.max_consecutive_losses})",
            )

        return True, None

    def get_current_status(self) -> Dict[str, any]:
        """Get current consecutive loss tracking status."""
        return {
            "current_consecutive_losses": self.current_consecutive_losses,
            "max_consecutive_losses": self.max_consecutive_losses,
            "trading_halted": self.trading_halted,
            "halt_reason": self.halt_reason,
            "halt_timestamp": (
                self.halt_timestamp.isoformat() if self.halt_timestamp else None
            ),
            "current_streak": (
                {
                    "start_timestamp": (
                        self.current_streak.start_timestamp.isoformat()
                        if self.current_streak
                        else None
                    ),
                    "trade_count": (
                        self.current_streak.trade_count if self.current_streak else 0
                    ),
                    "total_loss": (
                        float(self.current_streak.total_loss)
                        if self.current_streak
                        else 0.0
                    ),
                    "total_loss_percentage": (
                        self.current_streak.total_loss_percentage
                        if self.current_streak
                        else 0.0
                    ),
                }
                if self.current_streak
                else None
            ),
            "statistics": {
                "total_trades": self.total_trades,
                "total_wins": self.total_wins,
                "total_losses": self.total_losses,
                "total_breakevens": self.total_breakevens,
                "win_rate": (
                    self.total_wins / self.total_trades
                    if self.total_trades > 0
                    else 0.0
                ),
                "longest_loss_streak": self.longest_loss_streak,
            },
            "recent_trades": len(self.trade_history),
            "loss_streaks_count": len(self.loss_streaks),
        }

    def get_loss_pattern_analysis(self) -> Dict[str, any]:
        """Analyze patterns in consecutive losses."""
        if not self.loss_streaks:
            return {"analysis": "No loss streaks recorded yet"}

        # Calculate streak statistics
        completed_streaks = [s for s in self.loss_streaks if not s.is_active]
        if not completed_streaks:
            return {"analysis": "No completed loss streaks for analysis"}

        streak_lengths = [s.trade_count for s in completed_streaks]
        streak_losses = [float(s.total_loss) for s in completed_streaks]

        avg_streak_length = sum(streak_lengths) / len(streak_lengths)
        avg_streak_loss = sum(streak_losses) / len(streak_losses)

        # Time between streaks
        time_between_streaks = []
        for i in range(1, len(completed_streaks)):
            if (
                completed_streaks[i - 1].end_timestamp
                and completed_streaks[i].start_timestamp
            ):
                time_diff = (
                    completed_streaks[i].start_timestamp
                    - completed_streaks[i - 1].end_timestamp
                )
                time_between_streaks.append(time_diff.total_seconds() / 3600)  # Hours

        avg_time_between = (
            sum(time_between_streaks) / len(time_between_streaks)
            if time_between_streaks
            else 0
        )

        return {
            "total_streaks": len(completed_streaks),
            "average_streak_length": avg_streak_length,
            "max_streak_length": max(streak_lengths),
            "min_streak_length": min(streak_lengths),
            "average_streak_loss": avg_streak_loss,
            "total_streak_loss": sum(streak_losses),
            "average_hours_between_streaks": avg_time_between,
            "recent_streaks": [
                {
                    "start": s.start_timestamp.isoformat(),
                    "end": s.end_timestamp.isoformat() if s.end_timestamp else None,
                    "trade_count": s.trade_count,
                    "total_loss": float(s.total_loss),
                    "total_loss_percentage": s.total_loss_percentage,
                }
                for s in completed_streaks[-5:]  # Last 5 streaks
            ],
        }

    def reset_streak(self, reason: str = "Manual reset") -> None:
        """Manually reset current consecutive loss streak."""
        if self.current_consecutive_losses > 0:
            self.logger.warning(f"Manually resetting loss streak: {reason}")

            # End current streak
            if self.current_streak:
                self.current_streak.end_timestamp = datetime.utcnow()
                self.current_streak.is_active = False
                self.loss_streaks.append(self.current_streak)
                self.current_streak = None

            # Reset count
            self.current_consecutive_losses = 0

            # Resume trading if halted
            if self.trading_halted:
                self._resume_trading(f"Manual streak reset: {reason}")

    def force_resume_trading(self, reason: str = "Manual override") -> None:
        """Force resume trading (emergency override)."""
        self.logger.warning(f"FORCED TRADING RESUME: {reason}")
        self._resume_trading(reason)

    def update_settings(
        self,
        max_consecutive_losses: Optional[int] = None,
        loss_threshold_percentage: Optional[float] = None,
        cooldown_period_hours: Optional[int] = None,
    ) -> None:
        """Update tracker settings."""
        if max_consecutive_losses is not None:
            self.max_consecutive_losses = max_consecutive_losses
            self.logger.info(
                f"Updated max consecutive losses to {max_consecutive_losses}"
            )

        if loss_threshold_percentage is not None:
            self.loss_threshold_percentage = loss_threshold_percentage
            self.logger.info(
                f"Updated loss threshold to {loss_threshold_percentage:.1%}"
            )

        if cooldown_period_hours is not None:
            self.cooldown_period = timedelta(hours=cooldown_period_hours)
            self.logger.info(
                f"Updated cooldown period to {cooldown_period_hours} hours"
            )

    def get_recent_trades(self, limit: int = 20) -> List[Dict[str, any]]:
        """Get recent trade history."""
        recent_trades = self.trade_history[-limit:]
        return [
            {
                "trade_id": trade.trade_id,
                "symbol": trade.symbol,
                "timestamp": trade.timestamp.isoformat(),
                "entry_price": float(trade.entry_price),
                "exit_price": float(trade.exit_price),
                "quantity": float(trade.quantity),
                "pnl": float(trade.pnl),
                "pnl_percentage": trade.pnl_percentage,
                "result": trade.result.value,
                "commission": float(trade.commission),
            }
            for trade in recent_trades
        ]
