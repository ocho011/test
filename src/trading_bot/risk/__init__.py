"""
Risk management module for trading bot.

This module provides comprehensive risk management capabilities including:
- Position sizing with multiple calculation methods
- Drawdown monitoring and control
- Consecutive loss tracking
- Volatility-based filtering
- Unified risk assessment and management
"""

from .consecutive_loss_tracker import (
    ConsecutiveLossTracker,
    LossStreak,
    TradeRecord,
    TradeResult,
)
from .drawdown_controller import (
    DrawdownController,
    DrawdownLimit,
    DrawdownPeriod,
    DrawdownRecord,
    DrawdownStatus,
)
from .position_size_calculator import (
    PositionSizeCalculator,
    PositionSizeMethod,
    PositionSizeRequest,
    PositionSizeResult,
)
from .risk_manager import RiskAssessment, RiskDecision, RiskManager, TradeRequest
from .volatility_filter import (
    VolatilityFilter,
    VolatilityReading,
    VolatilityState,
    VolatilityThresholds,
)

__all__ = [
    # Position sizing
    "PositionSizeCalculator",
    "PositionSizeMethod",
    "PositionSizeRequest",
    "PositionSizeResult",
    # Drawdown control
    "DrawdownController",
    "DrawdownPeriod",
    "DrawdownLimit",
    "DrawdownStatus",
    "DrawdownRecord",
    # Loss tracking
    "ConsecutiveLossTracker",
    "TradeResult",
    "TradeRecord",
    "LossStreak",
    # Volatility filtering
    "VolatilityFilter",
    "VolatilityState",
    "VolatilityReading",
    "VolatilityThresholds",
    # Risk management
    "RiskManager",
    "RiskDecision",
    "RiskAssessment",
    "TradeRequest",
]
