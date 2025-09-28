"""
Risk management system integration module.

This module coordinates all risk management components and provides unified
risk assessment and control for trading operations.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from ..core.base_component import BaseComponent
from .consecutive_loss_tracker import ConsecutiveLossTracker, TradeRecord
from .drawdown_controller import DrawdownController
from .position_size_calculator import (
    PositionSizeCalculator,
    PositionSizeRequest,
    PositionSizeResult,
)
from .volatility_filter import VolatilityFilter


class RiskDecision(Enum):
    """Risk assessment decision types."""

    ALLOW = "allow"
    RESTRICT = "restrict"
    HALT = "halt"


class RiskAssessment(BaseModel):
    """Comprehensive risk assessment result."""

    decision: RiskDecision = Field(..., description="Overall risk decision")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in decision")
    reasons: List[str] = Field(default_factory=list, description="Reasons for decision")
    restrictions: List[str] = Field(
        default_factory=list, description="Active restrictions"
    )

    # Component assessments
    position_size_result: Optional[PositionSizeResult] = Field(
        None, description="Position sizing result"
    )
    drawdown_status: Dict[str, Any] = Field(
        default_factory=dict, description="Drawdown status"
    )
    loss_tracker_status: Dict[str, Any] = Field(
        default_factory=dict, description="Loss tracker status"
    )
    volatility_status: Dict[str, Any] = Field(
        default_factory=dict, description="Volatility status"
    )

    # Risk metrics
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Overall risk score")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Assessment timestamp"
    )


class TradeRequest(BaseModel):
    """Trading request for risk assessment."""

    symbol: str = Field(..., description="Trading symbol")
    entry_price: Decimal = Field(..., gt=0, description="Planned entry price")
    stop_loss_price: Decimal = Field(..., gt=0, description="Stop loss price")
    account_balance: Decimal = Field(..., gt=0, description="Current account balance")
    risk_percentage: float = Field(
        default=0.02, ge=0.001, le=0.1, description="Risk percentage"
    )

    # Optional parameters for advanced sizing
    win_rate: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Historical win rate"
    )
    avg_win_loss_ratio: Optional[float] = Field(
        None, gt=0, description="Average win/loss ratio"
    )
    current_atr: Optional[Decimal] = Field(None, gt=0, description="Current ATR")
    avg_atr: Optional[Decimal] = Field(None, gt=0, description="Average ATR")


class RiskManager(BaseComponent):
    """
    Unified risk management system.

    Coordinates position sizing, drawdown control, consecutive loss tracking,
    and volatility filtering to provide comprehensive risk management.
    """

    def __init__(
        self,
        name: str = "risk_manager",
        max_risk_score: float = 0.8,
        risk_score_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize risk manager.

        Args:
            name: Component name
            max_risk_score: Maximum allowed risk score (0.8 = 80%)
            risk_score_weights: Weights for risk score calculation
        """
        super().__init__(name)

        self.max_risk_score = max_risk_score

        # Default weights for risk score calculation
        self.risk_weights = risk_score_weights or {
            "drawdown": 0.3,
            "consecutive_losses": 0.25,
            "volatility": 0.25,
            "position_risk": 0.2,
        }

        # Initialize components
        self.position_calculator = PositionSizeCalculator()
        self.drawdown_controller = DrawdownController()
        self.loss_tracker = ConsecutiveLossTracker()
        self.volatility_filter = VolatilityFilter()

        # Risk state tracking
        self.risk_assessments: List[RiskAssessment] = []
        self.active_restrictions: Dict[str, str] = {}
        self.last_assessment_time: Optional[datetime] = None

        # Performance metrics
        self.total_assessments = 0
        self.blocked_trades = 0
        self.allowed_trades = 0

    async def _start(self) -> None:
        """Start risk management system."""
        self.logger.info("Starting risk management system")

        # Start all components
        (
            await self.position_calculator.start()
            if hasattr(self.position_calculator, "start")
            else None
        )
        await self.drawdown_controller.start()
        await self.loss_tracker.start()
        await self.volatility_filter.start()

        self.logger.info("Risk management system started successfully")

    async def _stop(self) -> None:
        """Stop risk management system."""
        self.logger.info("Stopping risk management system")

        # Stop all components
        (
            await self.position_calculator.stop()
            if hasattr(self.position_calculator, "stop")
            else None
        )
        await self.drawdown_controller.stop()
        await self.loss_tracker.stop()
        await self.volatility_filter.stop()

        self.logger.info("Risk management system stopped")

    def assess_trade_risk(self, trade_request: TradeRequest) -> RiskAssessment:
        """
        Assess risk for a proposed trade.

        Args:
            trade_request: Trading request parameters

        Returns:
            Comprehensive risk assessment
        """
        self.total_assessments += 1
        assessment_time = datetime.utcnow()

        self.logger.debug(f"Assessing trade risk for {trade_request.symbol}")

        # Create position size request
        position_request = PositionSizeRequest(
            account_balance=trade_request.account_balance,
            risk_percentage=trade_request.risk_percentage,
            entry_price=trade_request.entry_price,
            stop_loss_price=trade_request.stop_loss_price,
            symbol=trade_request.symbol,
            win_rate=trade_request.win_rate,
            avg_win_loss_ratio=trade_request.avg_win_loss_ratio,
            current_atr=trade_request.current_atr,
            avg_atr=trade_request.avg_atr,
        )

        # Calculate position size
        position_result = self.position_calculator.calculate_position_size(
            position_request
        )

        # Get component status
        drawdown_allowed, drawdown_reason = (
            self.drawdown_controller.check_trading_allowed()
        )
        drawdown_status = self.drawdown_controller.get_current_status()

        loss_allowed, loss_reason = self.loss_tracker.check_trading_allowed()
        loss_status = self.loss_tracker.get_current_status()

        volatility_allowed, volatility_reason = (
            self.volatility_filter.check_trading_allowed(trade_request.symbol)
        )
        volatility_status = self.volatility_filter.get_volatility_status(
            trade_request.symbol
        )

        # Calculate risk score
        risk_score = self._calculate_risk_score(
            position_result, drawdown_status, loss_status, volatility_status
        )

        # Determine overall decision
        decision, reasons, restrictions = self._determine_decision(
            position_result,
            (drawdown_allowed, drawdown_reason),
            (loss_allowed, loss_reason),
            (volatility_allowed, volatility_reason),
            risk_score,
        )

        # Calculate confidence based on agreement between components
        confidence = self._calculate_confidence(
            drawdown_allowed, loss_allowed, volatility_allowed, risk_score
        )

        # Create assessment
        assessment = RiskAssessment(
            decision=decision,
            confidence=confidence,
            reasons=reasons,
            restrictions=restrictions,
            position_size_result=position_result,
            drawdown_status=drawdown_status,
            loss_tracker_status=loss_status,
            volatility_status=volatility_status,
            risk_score=risk_score,
            timestamp=assessment_time,
        )

        # Store assessment
        self.risk_assessments.append(assessment)
        self.last_assessment_time = assessment_time

        # Clean old assessments (keep last 100)
        if len(self.risk_assessments) > 100:
            self.risk_assessments = self.risk_assessments[-100:]

        # Update statistics
        if decision == RiskDecision.ALLOW:
            self.allowed_trades += 1
        else:
            self.blocked_trades += 1

        # Log assessment
        self.logger.info(
            f"Trade risk assessment: {decision.value} "
            f"(confidence: {confidence:.1%}, risk: {risk_score:.1%})"
        )

        if decision != RiskDecision.ALLOW:
            self.logger.warning(f"Trade blocked: {', '.join(reasons)}")

        return assessment

    def _calculate_risk_score(
        self,
        position_result: PositionSizeResult,
        drawdown_status: Dict[str, Any],
        loss_status: Dict[str, Any],
        volatility_status: Dict[str, Any],
    ) -> float:
        """Calculate overall risk score from component status."""

        # Position risk (0-1 based on risk percentage)
        position_risk = min(
            position_result.risk_percentage_actual / 0.05, 1.0
        )  # Normalize to 5% max

        # Drawdown risk
        drawdown_risk = 0.0
        if drawdown_status.get("daily", {}).get("drawdown_percentage"):
            daily_dd = drawdown_status["daily"]["drawdown_percentage"]
            daily_limit = drawdown_status["daily"]["limit_percentage"]
            drawdown_risk = max(drawdown_risk, daily_dd / daily_limit)

        if drawdown_status.get("monthly", {}).get("drawdown_percentage"):
            monthly_dd = drawdown_status["monthly"]["drawdown_percentage"]
            monthly_limit = drawdown_status["monthly"]["limit_percentage"]
            drawdown_risk = max(drawdown_risk, monthly_dd / monthly_limit)

        # Consecutive loss risk
        loss_risk = 0.0
        if loss_status.get("current_consecutive_losses") and loss_status.get(
            "max_consecutive_losses"
        ):
            loss_risk = (
                loss_status["current_consecutive_losses"]
                / loss_status["max_consecutive_losses"]
            )

        # Volatility risk
        volatility_risk = 0.0
        if isinstance(volatility_status, dict) and "current_state" in volatility_status:
            state = volatility_status["current_state"]
            volatility_map = {"low": 0.1, "normal": 0.3, "high": 0.7, "extreme": 1.0}
            volatility_risk = volatility_map.get(state, 0.5)

        # Calculate weighted risk score
        total_risk = (
            self.risk_weights["drawdown"] * drawdown_risk
            + self.risk_weights["consecutive_losses"] * loss_risk
            + self.risk_weights["volatility"] * volatility_risk
            + self.risk_weights["position_risk"] * position_risk
        )

        return min(total_risk, 1.0)

    def _determine_decision(
        self,
        position_result: PositionSizeResult,
        drawdown_check: Tuple[bool, Optional[str]],
        loss_check: Tuple[bool, Optional[str]],
        volatility_check: Tuple[bool, Optional[str]],
        risk_score: float,
    ) -> Tuple[RiskDecision, List[str], List[str]]:
        """Determine overall trading decision."""

        reasons = []
        restrictions = []

        drawdown_allowed, drawdown_reason = drawdown_check
        loss_allowed, loss_reason = loss_check
        volatility_allowed, volatility_reason = volatility_check

        # Check for hard stops
        if not drawdown_allowed:
            reasons.append(f"Drawdown limit: {drawdown_reason}")
            restrictions.append("trading_halted_drawdown")

        if not loss_allowed:
            reasons.append(f"Loss limit: {loss_reason}")
            restrictions.append("trading_halted_losses")

        if not volatility_allowed:
            reasons.append(f"Volatility filter: {volatility_reason}")
            restrictions.append("trading_restricted_volatility")

        # Check risk score
        if risk_score > self.max_risk_score:
            reasons.append(
                f"Risk score too high: {risk_score:.1%} > {self.max_risk_score:.1%}"
            )
            restrictions.append("risk_score_exceeded")

        # Determine decision
        if not drawdown_allowed or not loss_allowed:
            decision = RiskDecision.HALT
        elif not volatility_allowed or risk_score > self.max_risk_score:
            decision = RiskDecision.RESTRICT
        else:
            decision = RiskDecision.ALLOW

        return decision, reasons, restrictions

    def _calculate_confidence(
        self,
        drawdown_allowed: bool,
        loss_allowed: bool,
        volatility_allowed: bool,
        risk_score: float,
    ) -> float:
        """Calculate confidence in risk assessment."""

        # Count agreeing components
        total_components = 3

        # Simple agreement: if multiple components agree on restriction
        restrictions = [not drawdown_allowed, not loss_allowed, not volatility_allowed]
        restriction_count = sum(restrictions)

        if restriction_count == 0:
            # All allow - high confidence if risk score is also low
            confidence = 0.9 if risk_score < 0.5 else 0.7
        elif restriction_count == total_components:
            # All restrict - very high confidence
            confidence = 0.95
        else:
            # Mixed signals - lower confidence
            confidence = 0.6

        # Adjust for risk score alignment
        if risk_score > 0.8 and restriction_count > 0:
            confidence = min(confidence + 0.1, 1.0)
        elif risk_score < 0.3 and restriction_count == 0:
            confidence = min(confidence + 0.1, 1.0)

        return confidence

    def update_balance(self, new_balance: Decimal) -> None:
        """Update account balance across all components."""
        self.logger.debug(f"Updating balance to ${new_balance}")

        # Update drawdown controller
        violations = self.drawdown_controller.update_balance(new_balance)

        # Log any violations
        for violation in violations:
            if violation.status.value in ["limit_reached", "trading_halted"]:
                self.logger.critical(
                    f"Drawdown violation: {violation.period.value} "
                    f"{violation.drawdown_percentage:.1%} (limit: {violation.limit_percentage:.1%})"
                )

    def update_volatility(
        self, symbol: str, atr_value: Decimal, price: Decimal
    ) -> None:
        """Update volatility information."""
        volatility_state = self.volatility_filter.update_atr(symbol, atr_value, price)

        self.logger.debug(f"Updated volatility for {symbol}: {volatility_state.value}")

    def record_trade_result(self, trade: TradeRecord) -> None:
        """Record completed trade for loss tracking."""
        self.loss_tracker.record_trade(trade)

        # Record trade for volatility filter daily limits
        self.volatility_filter.record_trade(trade.symbol)

        self.logger.debug(
            f"Recorded trade result: {trade.trade_id} - {trade.result.value}"
        )

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "risk_manager": {
                "total_assessments": self.total_assessments,
                "allowed_trades": self.allowed_trades,
                "blocked_trades": self.blocked_trades,
                "block_rate": self.blocked_trades / max(self.total_assessments, 1),
                "last_assessment": (
                    self.last_assessment_time.isoformat()
                    if self.last_assessment_time
                    else None
                ),
                "active_restrictions": self.active_restrictions,
                "max_risk_score": self.max_risk_score,
            },
            "drawdown_controller": self.drawdown_controller.get_current_status(),
            "loss_tracker": self.loss_tracker.get_current_status(),
            "volatility_filter": self.volatility_filter.get_volatility_status(),
            "risk_weights": self.risk_weights,
        }

    def get_recent_assessments(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent risk assessments."""
        recent = self.risk_assessments[-limit:]
        return [
            {
                "timestamp": assessment.timestamp.isoformat(),
                "decision": assessment.decision.value,
                "confidence": assessment.confidence,
                "risk_score": assessment.risk_score,
                "reasons": assessment.reasons,
                "restrictions": assessment.restrictions,
            }
            for assessment in recent
        ]

    def force_allow_trading(self, reason: str = "Manual override") -> None:
        """Force allow trading (emergency override)."""
        self.logger.warning(f"FORCED TRADING ALLOWED: {reason}")

        # Override all components
        self.drawdown_controller.force_resume_trading(reason)
        self.loss_tracker.force_resume_trading(reason)
        # Note: VolatilityFilter has per-symbol overrides

        self.active_restrictions.clear()

    def update_risk_settings(
        self,
        max_risk_score: Optional[float] = None,
        risk_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """Update risk manager settings."""
        if max_risk_score is not None:
            self.max_risk_score = max_risk_score
            self.logger.info(f"Updated max risk score to {max_risk_score:.1%}")

        if risk_weights is not None:
            self.risk_weights.update(risk_weights)
            self.logger.info(f"Updated risk weights: {risk_weights}")
