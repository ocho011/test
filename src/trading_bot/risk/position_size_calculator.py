"""
Position size calculator for risk management.

This module implements position sizing logic based on account risk percentage,
Kelly Criterion optimization, and stop-loss considerations for optimal capital allocation.
"""

import logging
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, validator


class PositionSizeMethod(Enum):
    """Position sizing methods available."""

    FIXED_RISK = "fixed_risk"  # Fixed percentage of account
    KELLY_CRITERION = "kelly_criterion"  # Kelly formula optimization
    VOLATILITY_ADJUSTED = "volatility_adjusted"  # ATR-based adjustment


class PositionSizeRequest(BaseModel):
    """Request for position size calculation."""

    account_balance: Decimal = Field(..., description="Current account balance")
    risk_percentage: float = Field(
        default=0.02, ge=0.001, le=0.1, description="Risk percentage (0.001-0.1)"
    )
    entry_price: Decimal = Field(..., gt=0, description="Planned entry price")
    stop_loss_price: Decimal = Field(..., gt=0, description="Stop loss price")
    symbol: str = Field(..., description="Trading symbol")
    method: PositionSizeMethod = Field(
        default=PositionSizeMethod.FIXED_RISK, description="Sizing method"
    )

    # Kelly Criterion specific parameters
    win_rate: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Historical win rate for Kelly"
    )
    avg_win_loss_ratio: Optional[float] = Field(
        None, gt=0, description="Average win/loss ratio for Kelly"
    )

    # Volatility adjustment parameters
    current_atr: Optional[Decimal] = Field(
        None, gt=0, description="Current ATR for volatility adjustment"
    )
    avg_atr: Optional[Decimal] = Field(
        None, gt=0, description="Average ATR for comparison"
    )

    @validator("stop_loss_price")
    def validate_stop_loss(cls, v, values):
        """Ensure stop loss makes sense relative to entry price."""
        if "entry_price" in values and values["entry_price"] is not None:
            entry_price = values["entry_price"]
            # Stop loss should be different from entry price
            if abs(v - entry_price) < entry_price * Decimal(
                "0.001"
            ):  # 0.1% minimum difference
                raise ValueError("Stop loss too close to entry price")
        return v


class PositionSizeResult(BaseModel):
    """Result of position size calculation."""

    position_size: Decimal = Field(..., description="Calculated position size")
    position_value: Decimal = Field(..., description="Total position value")
    risk_amount: Decimal = Field(..., description="Amount at risk")
    risk_percentage_actual: float = Field(..., description="Actual risk percentage")
    stop_loss_distance: Decimal = Field(..., description="Distance to stop loss")
    stop_loss_percentage: float = Field(..., description="Stop loss as percentage")
    method_used: PositionSizeMethod = Field(
        ..., description="Method used for calculation"
    )

    # Kelly Criterion specific results
    kelly_percentage: Optional[float] = Field(
        None, description="Kelly optimal percentage"
    )
    kelly_adjusted: Optional[bool] = Field(None, description="Whether Kelly was capped")

    # Additional metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional calculation data"
    )


class PositionSizeCalculator:
    """
    Calculate optimal position sizes based on risk management parameters.

    Supports multiple position sizing methods:
    - Fixed Risk: Fixed percentage of account balance
    - Kelly Criterion: Mathematical optimization based on win rate and win/loss ratio
    - Volatility Adjusted: ATR-based position size adjustment
    """

    def __init__(
        self,
        max_position_risk: float = 0.02,
        kelly_max_percentage: float = 0.25,
        volatility_multiplier: float = 1.0,
    ):
        """
        Initialize position size calculator.

        Args:
            max_position_risk: Maximum risk per position (0.02 = 2%)
            kelly_max_percentage: Maximum Kelly percentage to prevent over-leveraging
            volatility_multiplier: Multiplier for volatility adjustments
        """
        self.max_position_risk = max_position_risk
        self.kelly_max_percentage = kelly_max_percentage
        self.volatility_multiplier = volatility_multiplier
        self.logger = logging.getLogger("trading_bot.risk.position_size_calculator")

    def calculate_position_size(
        self, request: PositionSizeRequest
    ) -> PositionSizeResult:
        """
        Calculate position size based on the specified method.

        Args:
            request: Position size calculation parameters

        Returns:
            Position size calculation result

        Raises:
            ValueError: If calculation parameters are invalid
        """
        self.logger.debug(
            f"Calculating position size for {request.symbol} using {request.method}"
        )

        # Calculate base stop loss metrics
        stop_loss_distance = abs(request.entry_price - request.stop_loss_price)
        stop_loss_percentage = float(stop_loss_distance / request.entry_price)

        # Route to appropriate calculation method
        if request.method == PositionSizeMethod.FIXED_RISK:
            result = self._calculate_fixed_risk(
                request, stop_loss_distance, stop_loss_percentage
            )
        elif request.method == PositionSizeMethod.KELLY_CRITERION:
            result = self._calculate_kelly_criterion(
                request, stop_loss_distance, stop_loss_percentage
            )
        elif request.method == PositionSizeMethod.VOLATILITY_ADJUSTED:
            result = self._calculate_volatility_adjusted(
                request, stop_loss_distance, stop_loss_percentage
            )
        else:
            raise ValueError(f"Unknown position sizing method: {request.method}")

        # Validate result doesn't exceed maximum risk
        if result.risk_percentage_actual > self.max_position_risk:
            self.logger.warning(
                f"Calculated risk {result.risk_percentage_actual:.1%} exceeds maximum "
                f"{self.max_position_risk:.1%}, adjusting position size"
            )
            result = self._cap_position_size(result, request.account_balance)

        self.logger.info(
            f"Position size calculated: {result.position_size} units, "
            f"Risk: {result.risk_percentage_actual:.2%}, "
            f"Value: ${result.position_value}"
        )

        return result

    def _calculate_fixed_risk(
        self,
        request: PositionSizeRequest,
        stop_loss_distance: Decimal,
        stop_loss_percentage: float,
    ) -> PositionSizeResult:
        """Calculate position size using fixed risk percentage."""
        risk_amount = request.account_balance * Decimal(str(request.risk_percentage))
        position_size = risk_amount / stop_loss_distance
        position_value = position_size * request.entry_price

        return PositionSizeResult(
            position_size=position_size,
            position_value=position_value,
            risk_amount=risk_amount,
            risk_percentage_actual=request.risk_percentage,
            stop_loss_distance=stop_loss_distance,
            stop_loss_percentage=stop_loss_percentage,
            method_used=PositionSizeMethod.FIXED_RISK,
            metadata={
                "fixed_risk_percentage": request.risk_percentage,
                "calculation_method": "risk_amount / stop_loss_distance",
            },
        )

    def _calculate_kelly_criterion(
        self,
        request: PositionSizeRequest,
        stop_loss_distance: Decimal,
        stop_loss_percentage: float,
    ) -> PositionSizeResult:
        """Calculate position size using Kelly Criterion optimization."""
        if request.win_rate is None or request.avg_win_loss_ratio is None:
            self.logger.warning(
                "Kelly Criterion requires win_rate and avg_win_loss_ratio, falling back to fixed risk"
            )
            return self._calculate_fixed_risk(
                request, stop_loss_distance, stop_loss_percentage
            )

        # Kelly Formula: f* = (bp - q) / b
        # where: f* = fraction of capital to wager
        #        b = odds received on the wager (win/loss ratio)
        #        p = probability of winning (win rate)
        #        q = probability of losing (1 - p)

        p = request.win_rate  # Probability of winning
        q = 1.0 - p  # Probability of losing
        b = request.avg_win_loss_ratio  # Odds (average win / average loss)

        kelly_percentage = (b * p - q) / b
        kelly_percentage = max(0.0, kelly_percentage)  # Ensure non-negative

        # Cap Kelly percentage to prevent over-leveraging
        kelly_adjusted = False
        if kelly_percentage > self.kelly_max_percentage:
            kelly_percentage = self.kelly_max_percentage
            kelly_adjusted = True
            self.logger.info(
                f"Kelly percentage capped at {self.kelly_max_percentage:.1%}"
            )

        # Calculate position size based on Kelly percentage
        risk_amount = request.account_balance * Decimal(str(kelly_percentage))
        position_size = risk_amount / stop_loss_distance
        position_value = position_size * request.entry_price

        return PositionSizeResult(
            position_size=position_size,
            position_value=position_value,
            risk_amount=risk_amount,
            risk_percentage_actual=kelly_percentage,
            stop_loss_distance=stop_loss_distance,
            stop_loss_percentage=stop_loss_percentage,
            method_used=PositionSizeMethod.KELLY_CRITERION,
            kelly_percentage=kelly_percentage,
            kelly_adjusted=kelly_adjusted,
            metadata={
                "win_rate": request.win_rate,
                "avg_win_loss_ratio": request.avg_win_loss_ratio,
                "kelly_raw": (b * p - q) / b,
                "kelly_formula": f"({b} * {p} - {q}) / {b}",
            },
        )

    def _calculate_volatility_adjusted(
        self,
        request: PositionSizeRequest,
        stop_loss_distance: Decimal,
        stop_loss_percentage: float,
    ) -> PositionSizeResult:
        """Calculate position size with volatility adjustment."""
        if request.current_atr is None or request.avg_atr is None:
            self.logger.warning(
                "Volatility adjustment requires current_atr and avg_atr, falling back to fixed risk"
            )
            return self._calculate_fixed_risk(
                request, stop_loss_distance, stop_loss_percentage
            )

        # Calculate volatility ratio
        volatility_ratio = float(request.current_atr / request.avg_atr)

        # Adjust risk based on volatility
        # Higher volatility = lower position size
        volatility_adjustment = 1.0 / (volatility_ratio * self.volatility_multiplier)
        adjusted_risk_percentage = request.risk_percentage * volatility_adjustment

        # Ensure adjusted risk doesn't exceed maximum
        adjusted_risk_percentage = min(adjusted_risk_percentage, self.max_position_risk)

        risk_amount = request.account_balance * Decimal(str(adjusted_risk_percentage))
        position_size = risk_amount / stop_loss_distance
        position_value = position_size * request.entry_price

        return PositionSizeResult(
            position_size=position_size,
            position_value=position_value,
            risk_amount=risk_amount,
            risk_percentage_actual=adjusted_risk_percentage,
            stop_loss_distance=stop_loss_distance,
            stop_loss_percentage=stop_loss_percentage,
            method_used=PositionSizeMethod.VOLATILITY_ADJUSTED,
            metadata={
                "base_risk_percentage": request.risk_percentage,
                "volatility_ratio": volatility_ratio,
                "volatility_adjustment": volatility_adjustment,
                "current_atr": float(request.current_atr),
                "avg_atr": float(request.avg_atr),
                "volatility_multiplier": self.volatility_multiplier,
            },
        )

    def _cap_position_size(
        self, result: PositionSizeResult, account_balance: Decimal
    ) -> PositionSizeResult:
        """Cap position size to maximum allowed risk."""
        max_risk_amount = account_balance * Decimal(str(self.max_position_risk))
        adjustment_factor = max_risk_amount / result.risk_amount

        return PositionSizeResult(
            position_size=result.position_size * adjustment_factor,
            position_value=result.position_value * adjustment_factor,
            risk_amount=max_risk_amount,
            risk_percentage_actual=self.max_position_risk,
            stop_loss_distance=result.stop_loss_distance,
            stop_loss_percentage=result.stop_loss_percentage,
            method_used=result.method_used,
            kelly_percentage=result.kelly_percentage,
            kelly_adjusted=result.kelly_adjusted,
            metadata={
                **result.metadata,
                "position_capped": True,
                "original_risk_percentage": result.risk_percentage_actual,
                "adjustment_factor": float(adjustment_factor),
            },
        )

    def validate_position_size(
        self, result: PositionSizeResult, min_position_value: Decimal = Decimal("10")
    ) -> bool:
        """
        Validate that calculated position size meets minimum requirements.

        Args:
            result: Position size calculation result
            min_position_value: Minimum position value required

        Returns:
            True if position size is valid, False otherwise
        """
        if result.position_value < min_position_value:
            self.logger.warning(
                f"Position value ${result.position_value} below minimum ${min_position_value}"
            )
            return False

        if result.position_size <= 0:
            self.logger.error("Position size must be positive")
            return False

        if result.risk_percentage_actual > self.max_position_risk:
            self.logger.error(
                f"Risk percentage {result.risk_percentage_actual:.1%} exceeds maximum {self.max_position_risk:.1%}"
            )
            return False

        return True

    def get_position_size_summary(self, result: PositionSizeResult) -> Dict[str, Any]:
        """
        Get human-readable summary of position size calculation.

        Args:
            result: Position size calculation result

        Returns:
            Dictionary with summary information
        """
        return {
            "position_size": f"{result.position_size:.8f}",
            "position_value": f"${result.position_value:.2f}",
            "risk_amount": f"${result.risk_amount:.2f}",
            "risk_percentage": f"{result.risk_percentage_actual:.2%}",
            "stop_loss_distance": f"${result.stop_loss_distance:.8f}",
            "stop_loss_percentage": f"{result.stop_loss_percentage:.2%}",
            "method": result.method_used.value,
            "kelly_info": (
                {
                    "kelly_percentage": (
                        f"{result.kelly_percentage:.2%}"
                        if result.kelly_percentage
                        else None
                    ),
                    "kelly_adjusted": result.kelly_adjusted,
                }
                if result.kelly_percentage
                else None
            ),
            "metadata": result.metadata,
        }
