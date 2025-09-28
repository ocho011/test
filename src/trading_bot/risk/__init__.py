"""
Risk management module for trading bot.

This module provides comprehensive risk management capabilities including:
- Position sizing with multiple calculation methods
- Drawdown monitoring and control
- Consecutive loss tracking
- Volatility-based filtering
- Unified risk assessment and management
"""

from .position_size_calculator import (
    PositionSizeCalculator,
    PositionSizeMethod,
    PositionSizeRequest,
    PositionSizeResult
)

from .drawdown_controller import (
    DrawdownController,
    DrawdownPeriod,
    DrawdownLimit,
    DrawdownStatus,
    DrawdownRecord
)

from .consecutive_loss_tracker import (
    ConsecutiveLossTracker,
    TradeResult,
    TradeRecord,
    LossStreak
)

from .volatility_filter import (
    VolatilityFilter,
    VolatilityState,
    VolatilityReading,
    VolatilityThresholds
)

from .risk_manager import (
    RiskManager,
    RiskDecision,
    RiskAssessment,
    TradeRequest
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
    "TradeRequest"
]