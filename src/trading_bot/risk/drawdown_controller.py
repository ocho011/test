"""
Drawdown controller for risk management.

This module implements drawdown monitoring and control with daily and monthly limits,
automatic trading halt functionality, and recovery tracking.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from ..core.base_component import BaseComponent


class DrawdownPeriod(Enum):
    """Drawdown monitoring periods."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class DrawdownLimit(BaseModel):
    """Drawdown limit configuration."""

    period: DrawdownPeriod = Field(..., description="Time period for limit")
    percentage: float = Field(
        ..., ge=0.01, le=0.5, description="Maximum drawdown percentage"
    )
    enabled: bool = Field(default=True, description="Whether limit is active")


class DrawdownStatus(Enum):
    """Current drawdown status."""

    NORMAL = "normal"  # Within limits
    WARNING = "warning"  # Approaching limits
    LIMIT_REACHED = "limit_reached"  # Limit exceeded
    TRADING_HALTED = "trading_halted"  # Trading stopped due to drawdown


class DrawdownRecord(BaseModel):
    """Record of drawdown event."""

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    period: DrawdownPeriod = Field(..., description="Period type")
    peak_balance: Decimal = Field(..., description="Peak balance before drawdown")
    current_balance: Decimal = Field(..., description="Current balance")
    drawdown_amount: Decimal = Field(..., description="Absolute drawdown amount")
    drawdown_percentage: float = Field(..., description="Drawdown as percentage")
    limit_percentage: float = Field(..., description="Configured limit")
    status: DrawdownStatus = Field(..., description="Drawdown status")


class DrawdownController(BaseComponent):
    """
    Monitor and control trading drawdowns.

    Implements daily and monthly drawdown limits with automatic trading halt
    functionality and recovery tracking.
    """

    def __init__(
        self,
        name: str = "drawdown_controller",
        daily_limit: float = 0.05,  # 5% daily drawdown limit
        monthly_limit: float = 0.15,  # 15% monthly drawdown limit
        warning_threshold: float = 0.8,  # Warn at 80% of limit
        recovery_threshold: float = 0.5,  # Allow trading when recovered to 50% of limit
    ):
        """
        Initialize drawdown controller.

        Args:
            name: Component name
            daily_limit: Maximum daily drawdown (0.05 = 5%)
            monthly_limit: Maximum monthly drawdown (0.15 = 15%)
            warning_threshold: Warning threshold as fraction of limit
            recovery_threshold: Recovery threshold to resume trading
        """
        super().__init__(name)

        self.limits = {
            DrawdownPeriod.DAILY: DrawdownLimit(
                period=DrawdownPeriod.DAILY, percentage=daily_limit
            ),
            DrawdownPeriod.MONTHLY: DrawdownLimit(
                period=DrawdownPeriod.MONTHLY, percentage=monthly_limit
            ),
        }

        self.warning_threshold = warning_threshold
        self.recovery_threshold = recovery_threshold

        # State tracking
        self.daily_peak_balance: Optional[Decimal] = None
        self.monthly_peak_balance: Optional[Decimal] = None
        self.current_balance: Optional[Decimal] = None

        # Trading halt state
        self.trading_halted = False
        self.halt_reason: Optional[str] = None
        self.halt_timestamp: Optional[datetime] = None

        # History tracking
        self.drawdown_history: List[DrawdownRecord] = []
        self.balance_history: List[Tuple[datetime, Decimal]] = []

        # Date tracking for period resets
        self.last_daily_reset = datetime.utcnow().date()
        self.last_monthly_reset = datetime.utcnow().replace(day=1).date()

    async def _start(self) -> None:
        """Start drawdown monitoring."""
        self.logger.info("Starting drawdown controller")
        self.logger.info(
            f"Daily limit: {self.limits[DrawdownPeriod.DAILY].percentage:.1%}"
        )
        self.logger.info(
            f"Monthly limit: {self.limits[DrawdownPeriod.MONTHLY].percentage:.1%}"
        )

    async def _stop(self) -> None:
        """Stop drawdown monitoring."""
        self.logger.info("Stopping drawdown controller")

    def update_balance(self, new_balance: Decimal) -> List[DrawdownRecord]:
        """
        Update current balance and check drawdown limits.

        Args:
            new_balance: Current account balance

        Returns:
            List of drawdown violations detected
        """
        current_time = datetime.utcnow()
        self.current_balance = new_balance

        # Add to balance history
        self.balance_history.append((current_time, new_balance))

        # Clean old history (keep 1 month)
        cutoff_time = current_time - timedelta(days=31)
        self.balance_history = [
            (ts, balance) for ts, balance in self.balance_history if ts > cutoff_time
        ]

        # Check for period resets
        self._check_period_resets(current_time)

        # Update peak balances
        self._update_peak_balances(new_balance)

        # Check drawdown limits
        violations = self._check_drawdown_limits()

        # Log balance update
        self.logger.debug(f"Balance updated: ${new_balance:.2f}")

        return violations

    def _check_period_resets(self, current_time: datetime) -> None:
        """Check if we need to reset periods and peak balances."""
        current_date = current_time.date()

        # Check daily reset
        if current_date > self.last_daily_reset:
            self.logger.info(f"Daily period reset: {current_date}")
            self.daily_peak_balance = self.current_balance
            self.last_daily_reset = current_date

            # Clear daily trading halt if it exists
            if (
                self.trading_halted
                and self.halt_reason
                and "daily" in self.halt_reason.lower()
            ):
                self._resume_trading("Daily period reset")

        # Check monthly reset
        monthly_reset_date = current_time.replace(day=1).date()
        if monthly_reset_date > self.last_monthly_reset:
            self.logger.info(f"Monthly period reset: {monthly_reset_date}")
            self.monthly_peak_balance = self.current_balance
            self.last_monthly_reset = monthly_reset_date

            # Clear monthly trading halt if it exists
            if (
                self.trading_halted
                and self.halt_reason
                and "monthly" in self.halt_reason.lower()
            ):
                self._resume_trading("Monthly period reset")

    def _update_peak_balances(self, current_balance: Decimal) -> None:
        """Update peak balances for drawdown calculation."""
        # Initialize peaks if not set
        if self.daily_peak_balance is None:
            self.daily_peak_balance = current_balance
        if self.monthly_peak_balance is None:
            self.monthly_peak_balance = current_balance

        # Update peaks if balance is higher
        if current_balance > self.daily_peak_balance:
            self.daily_peak_balance = current_balance
            self.logger.debug(f"New daily peak: ${current_balance:.2f}")

        if current_balance > self.monthly_peak_balance:
            self.monthly_peak_balance = current_balance
            self.logger.debug(f"New monthly peak: ${current_balance:.2f}")

    def _check_drawdown_limits(self) -> List[DrawdownRecord]:
        """Check current drawdown against all limits."""
        violations = []

        if self.current_balance is None:
            return violations

        # Check daily drawdown
        if self.daily_peak_balance and self.limits[DrawdownPeriod.DAILY].enabled:
            daily_record = self._calculate_drawdown(
                DrawdownPeriod.DAILY,
                self.daily_peak_balance,
                self.current_balance,
                self.limits[DrawdownPeriod.DAILY].percentage,
            )
            violations.append(daily_record)

            # Check for limit violation
            if daily_record.status in [
                DrawdownStatus.LIMIT_REACHED,
                DrawdownStatus.TRADING_HALTED,
            ]:
                self._handle_limit_violation(daily_record)

        # Check monthly drawdown
        if self.monthly_peak_balance and self.limits[DrawdownPeriod.MONTHLY].enabled:
            monthly_record = self._calculate_drawdown(
                DrawdownPeriod.MONTHLY,
                self.monthly_peak_balance,
                self.current_balance,
                self.limits[DrawdownPeriod.MONTHLY].percentage,
            )
            violations.append(monthly_record)

            # Check for limit violation
            if monthly_record.status in [
                DrawdownStatus.LIMIT_REACHED,
                DrawdownStatus.TRADING_HALTED,
            ]:
                self._handle_limit_violation(monthly_record)

        # Add to history
        self.drawdown_history.extend(violations)

        # Keep only recent history (last 100 records)
        if len(self.drawdown_history) > 100:
            self.drawdown_history = self.drawdown_history[-100:]

        return violations

    def _calculate_drawdown(
        self,
        period: DrawdownPeriod,
        peak_balance: Decimal,
        current_balance: Decimal,
        limit_percentage: float,
    ) -> DrawdownRecord:
        """Calculate drawdown for a specific period."""
        drawdown_amount = peak_balance - current_balance
        drawdown_percentage = (
            float(drawdown_amount / peak_balance) if peak_balance > 0 else 0.0
        )

        # Determine status
        if drawdown_percentage >= limit_percentage:
            status = DrawdownStatus.LIMIT_REACHED
        elif drawdown_percentage >= limit_percentage * self.warning_threshold:
            status = DrawdownStatus.WARNING
        else:
            status = DrawdownStatus.NORMAL

        # If already halted for this period, maintain halt status
        if (
            self.trading_halted
            and self.halt_reason
            and period.value in self.halt_reason.lower()
        ):
            status = DrawdownStatus.TRADING_HALTED

        return DrawdownRecord(
            period=period,
            peak_balance=peak_balance,
            current_balance=current_balance,
            drawdown_amount=drawdown_amount,
            drawdown_percentage=drawdown_percentage,
            limit_percentage=limit_percentage,
            status=status,
        )

    def _handle_limit_violation(self, record: DrawdownRecord) -> None:
        """Handle drawdown limit violation."""
        if record.status == DrawdownStatus.LIMIT_REACHED and not self.trading_halted:
            self._halt_trading(
                f"{record.period.value.title()} drawdown limit exceeded: {record.drawdown_percentage:.1%}"
            )

        # Log violation
        self.logger.warning(
            f"{record.period.value.title()} drawdown violation: "
            f"{record.drawdown_percentage:.1%} (limit: {record.limit_percentage:.1%})"
        )

    def _halt_trading(self, reason: str) -> None:
        """Halt trading due to drawdown violation."""
        self.trading_halted = True
        self.halt_reason = reason
        self.halt_timestamp = datetime.utcnow()

        self.logger.critical(f"TRADING HALTED: {reason}")

        # Emit risk event
        # Note: In a real implementation, this would be published through the event bus

    def _resume_trading(self, reason: str) -> None:
        """Resume trading after recovery or period reset."""
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

        # Check if we're recovered enough to resume trading
        current_violations = self._check_drawdown_limits()
        for violation in current_violations:
            if (
                violation.drawdown_percentage
                >= violation.limit_percentage * self.recovery_threshold
            ):
                # Still too close to limit
                continue

        return True, None

    def get_current_status(self) -> Dict[str, any]:
        """Get current drawdown status."""
        if not self.current_balance:
            return {"status": "not_initialized"}

        daily_drawdown = (
            self._calculate_drawdown(
                DrawdownPeriod.DAILY,
                self.daily_peak_balance or self.current_balance,
                self.current_balance,
                self.limits[DrawdownPeriod.DAILY].percentage,
            )
            if self.daily_peak_balance
            else None
        )

        monthly_drawdown = (
            self._calculate_drawdown(
                DrawdownPeriod.MONTHLY,
                self.monthly_peak_balance or self.current_balance,
                self.current_balance,
                self.limits[DrawdownPeriod.MONTHLY].percentage,
            )
            if self.monthly_peak_balance
            else None
        )

        return {
            "current_balance": float(self.current_balance),
            "trading_halted": self.trading_halted,
            "halt_reason": self.halt_reason,
            "halt_timestamp": (
                self.halt_timestamp.isoformat() if self.halt_timestamp else None
            ),
            "daily": {
                "peak_balance": (
                    float(self.daily_peak_balance) if self.daily_peak_balance else None
                ),
                "drawdown_percentage": (
                    daily_drawdown.drawdown_percentage if daily_drawdown else 0.0
                ),
                "limit_percentage": self.limits[DrawdownPeriod.DAILY].percentage,
                "status": daily_drawdown.status.value if daily_drawdown else "normal",
            },
            "monthly": {
                "peak_balance": (
                    float(self.monthly_peak_balance)
                    if self.monthly_peak_balance
                    else None
                ),
                "drawdown_percentage": (
                    monthly_drawdown.drawdown_percentage if monthly_drawdown else 0.0
                ),
                "limit_percentage": self.limits[DrawdownPeriod.MONTHLY].percentage,
                "status": (
                    monthly_drawdown.status.value if monthly_drawdown else "normal"
                ),
            },
        }

    def update_limits(
        self, daily_limit: Optional[float] = None, monthly_limit: Optional[float] = None
    ) -> None:
        """Update drawdown limits."""
        if daily_limit is not None:
            self.limits[DrawdownPeriod.DAILY].percentage = daily_limit
            self.logger.info(f"Updated daily drawdown limit to {daily_limit:.1%}")

        if monthly_limit is not None:
            self.limits[DrawdownPeriod.MONTHLY].percentage = monthly_limit
            self.logger.info(f"Updated monthly drawdown limit to {monthly_limit:.1%}")

    def force_resume_trading(self, reason: str = "Manual override") -> None:
        """Force resume trading (emergency override)."""
        self.logger.warning(f"FORCED TRADING RESUME: {reason}")
        self._resume_trading(reason)

    def get_drawdown_history(self, limit: int = 50) -> List[Dict[str, any]]:
        """Get recent drawdown history."""
        recent_history = self.drawdown_history[-limit:]
        return [
            {
                "timestamp": record.timestamp.isoformat(),
                "period": record.period.value,
                "peak_balance": float(record.peak_balance),
                "current_balance": float(record.current_balance),
                "drawdown_amount": float(record.drawdown_amount),
                "drawdown_percentage": record.drawdown_percentage,
                "limit_percentage": record.limit_percentage,
                "status": record.status.value,
            }
            for record in recent_history
        ]
