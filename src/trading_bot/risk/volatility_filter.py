"""
Volatility filter for risk management.

This module implements ATR-based volatility filtering to detect abnormal market conditions
and control trading frequency during high volatility periods.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from ..core.base_component import BaseComponent


class VolatilityState(Enum):
    """Market volatility states."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"


class VolatilityReading(BaseModel):
    """ATR volatility reading."""

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    symbol: str = Field(..., description="Trading symbol")
    atr_value: Decimal = Field(..., description="Current ATR value")
    atr_period: int = Field(..., description="ATR calculation period")
    price: Decimal = Field(..., description="Current price")
    atr_percentage: float = Field(..., description="ATR as percentage of price")


class VolatilityThresholds(BaseModel):
    """Volatility classification thresholds."""

    low_threshold: float = Field(
        default=0.5, description="Multiplier for low volatility"
    )
    high_threshold: float = Field(
        default=1.5, description="Multiplier for high volatility"
    )
    extreme_threshold: float = Field(
        default=2.5, description="Multiplier for extreme volatility"
    )


class VolatilityFilter(BaseComponent):
    """
    Filter trading based on ATR volatility measurements.

    Monitors market volatility using Average True Range (ATR) and restricts
    trading during abnormal volatility conditions with daily trade limits.
    """

    def __init__(
        self,
        name: str = "volatility_filter",
        atr_period: int = 14,
        atr_lookback_periods: int = 50,
        daily_trade_limit: int = 5,
        volatility_thresholds: Optional[VolatilityThresholds] = None,
    ):
        """
        Initialize volatility filter.

        Args:
            name: Component name
            atr_period: Period for ATR calculation
            atr_lookback_periods: Number of periods for average ATR calculation
            daily_trade_limit: Maximum trades per day during high volatility
            volatility_thresholds: Custom volatility thresholds
        """
        super().__init__(name)

        self.atr_period = atr_period
        self.atr_lookback_periods = atr_lookback_periods
        self.daily_trade_limit = daily_trade_limit

        self.thresholds = volatility_thresholds or VolatilityThresholds()

        # State tracking
        self.volatility_readings: Dict[str, List[VolatilityReading]] = {}
        self.current_volatility_state: Dict[str, VolatilityState] = {}
        self.daily_trade_counts: Dict[str, int] = {}  # Symbol -> count
        self.last_reset_date = datetime.utcnow().date()

        # Trading restrictions
        self.trading_restricted: Dict[str, bool] = {}
        self.restriction_reasons: Dict[str, str] = {}

    async def _start(self) -> None:
        """Start volatility filtering."""
        self.logger.info("Starting volatility filter")
        self.logger.info(f"ATR period: {self.atr_period}")
        self.logger.info(f"Daily trade limit: {self.daily_trade_limit}")
        self.logger.info(f"Volatility thresholds: {self.thresholds}")

    async def _stop(self) -> None:
        """Stop volatility filtering."""
        self.logger.info("Stopping volatility filter")

    def update_atr(
        self, symbol: str, atr_value: Decimal, price: Decimal
    ) -> VolatilityState:
        """
        Update ATR reading and determine volatility state.

        Args:
            symbol: Trading symbol
            atr_value: Current ATR value
            price: Current price

        Returns:
            Current volatility state for the symbol
        """
        # Create volatility reading
        reading = VolatilityReading(
            symbol=symbol,
            atr_value=atr_value,
            atr_period=self.atr_period,
            price=price,
            atr_percentage=float(atr_value / price) * 100,
        )

        # Initialize symbol tracking if needed
        if symbol not in self.volatility_readings:
            self.volatility_readings[symbol] = []
            self.current_volatility_state[symbol] = VolatilityState.NORMAL
            self.daily_trade_counts[symbol] = 0
            self.trading_restricted[symbol] = False

        # Add reading to history
        self.volatility_readings[symbol].append(reading)

        # Keep only recent readings
        max_readings = (
            self.atr_lookback_periods * 2
        )  # Keep extra for rolling calculations
        if len(self.volatility_readings[symbol]) > max_readings:
            self.volatility_readings[symbol] = self.volatility_readings[symbol][
                -max_readings:
            ]

        # Calculate volatility state
        volatility_state = self._calculate_volatility_state(symbol)
        self.current_volatility_state[symbol] = volatility_state

        # Update trading restrictions
        self._update_trading_restrictions(symbol, volatility_state)

        # Check daily reset
        self._check_daily_reset()

        self.logger.debug(
            f"{symbol} volatility update: ATR={atr_value:.8f} ({reading.atr_percentage:.2f}%), "
            f"State={volatility_state.value}"
        )

        return volatility_state

    def _calculate_volatility_state(self, symbol: str) -> VolatilityState:
        """Calculate current volatility state based on ATR history."""
        readings = self.volatility_readings[symbol]

        if len(readings) < self.atr_lookback_periods:
            # Not enough data, assume normal
            return VolatilityState.NORMAL

        # Get recent readings for average calculation
        recent_readings = readings[-self.atr_lookback_periods :]
        current_reading = readings[-1]

        # Calculate average ATR percentage
        avg_atr_percentage = sum(r.atr_percentage for r in recent_readings) / len(
            recent_readings
        )

        # Calculate volatility ratio (current vs average)
        volatility_ratio = current_reading.atr_percentage / avg_atr_percentage

        # Classify volatility state
        if volatility_ratio <= self.thresholds.low_threshold:
            return VolatilityState.LOW
        elif volatility_ratio <= self.thresholds.high_threshold:
            return VolatilityState.NORMAL
        elif volatility_ratio <= self.thresholds.extreme_threshold:
            return VolatilityState.HIGH
        else:
            return VolatilityState.EXTREME

    def _update_trading_restrictions(
        self, symbol: str, volatility_state: VolatilityState
    ) -> None:
        """Update trading restrictions based on volatility state."""
        previous_restricted = self.trading_restricted.get(symbol, False)

        if volatility_state == VolatilityState.EXTREME:
            # Extreme volatility - halt trading
            self.trading_restricted[symbol] = True
            self.restriction_reasons[symbol] = (
                f"Extreme volatility detected (state: {volatility_state.value})"
            )

            if not previous_restricted:
                self.logger.warning(
                    f"Trading restricted for {symbol}: {self.restriction_reasons[symbol]}"
                )

        elif volatility_state == VolatilityState.HIGH:
            # High volatility - check daily trade limit
            if self.daily_trade_counts[symbol] >= self.daily_trade_limit:
                self.trading_restricted[symbol] = True
                self.restriction_reasons[symbol] = (
                    f"Daily trade limit reached during high volatility "
                    f"({self.daily_trade_counts[symbol]}/{self.daily_trade_limit})"
                )

                if not previous_restricted:
                    self.logger.warning(
                        f"Trading restricted for {symbol}: {self.restriction_reasons[symbol]}"
                    )
            else:
                # High volatility but within trade limit
                self.trading_restricted[symbol] = False
                self.restriction_reasons[symbol] = ""

        else:
            # Low or normal volatility - allow trading
            self.trading_restricted[symbol] = False
            self.restriction_reasons[symbol] = ""

            if previous_restricted:
                self.logger.info(
                    f"Trading restrictions lifted for {symbol} (volatility: {volatility_state.value})"
                )

    def _check_daily_reset(self) -> None:
        """Check if daily trade counts need to be reset."""
        current_date = datetime.utcnow().date()

        if current_date > self.last_reset_date:
            self.logger.info(f"Daily reset: {current_date}")

            # Reset daily trade counts
            for symbol in self.daily_trade_counts:
                self.daily_trade_counts[symbol] = 0

            # Re-evaluate restrictions without daily limits
            for symbol in self.current_volatility_state:
                self._update_trading_restrictions(
                    symbol, self.current_volatility_state[symbol]
                )

            self.last_reset_date = current_date

    def check_trading_allowed(self, symbol: str) -> Tuple[bool, Optional[str]]:
        """
        Check if trading is allowed for a symbol.

        Args:
            symbol: Trading symbol to check

        Returns:
            Tuple of (allowed, reason_if_not_allowed)
        """
        if symbol not in self.trading_restricted:
            # Symbol not tracked yet, allow trading
            return True, None

        if self.trading_restricted[symbol]:
            return False, self.restriction_reasons[symbol]

        # Check if we're approaching daily limit during high volatility
        current_state = self.current_volatility_state.get(
            symbol, VolatilityState.NORMAL
        )
        if current_state == VolatilityState.HIGH:
            remaining_trades = self.daily_trade_limit - self.daily_trade_counts[symbol]
            if remaining_trades <= 1:
                return (
                    False,
                    f"Approaching daily trade limit during high volatility ({remaining_trades} remaining)",
                )

        return True, None

    def record_trade(self, symbol: str) -> None:
        """
        Record a trade execution for daily limit tracking.

        Args:
            symbol: Trading symbol
        """
        if symbol not in self.daily_trade_counts:
            self.daily_trade_counts[symbol] = 0

        self.daily_trade_counts[symbol] += 1

        self.logger.debug(
            f"Trade recorded for {symbol}: {self.daily_trade_counts[symbol]}/{self.daily_trade_limit}"
        )

        # Re-check restrictions after trade
        current_state = self.current_volatility_state.get(
            symbol, VolatilityState.NORMAL
        )
        self._update_trading_restrictions(symbol, current_state)

    def get_volatility_status(self, symbol: Optional[str] = None) -> Dict[str, any]:
        """
        Get current volatility status.

        Args:
            symbol: Specific symbol to check, or None for all symbols

        Returns:
            Dictionary with volatility status information
        """
        if symbol:
            # Single symbol status
            if symbol not in self.volatility_readings:
                return {"error": f"No volatility data for {symbol}"}

            readings = self.volatility_readings[symbol]
            latest_reading = readings[-1] if readings else None

            return {
                "symbol": symbol,
                "current_state": self.current_volatility_state.get(
                    symbol, VolatilityState.NORMAL
                ).value,
                "trading_restricted": self.trading_restricted.get(symbol, False),
                "restriction_reason": self.restriction_reasons.get(symbol, ""),
                "daily_trades": self.daily_trade_counts.get(symbol, 0),
                "daily_limit": self.daily_trade_limit,
                "latest_reading": (
                    {
                        "atr_value": (
                            float(latest_reading.atr_value) if latest_reading else None
                        ),
                        "atr_percentage": (
                            latest_reading.atr_percentage if latest_reading else None
                        ),
                        "price": (
                            float(latest_reading.price) if latest_reading else None
                        ),
                        "timestamp": (
                            latest_reading.timestamp.isoformat()
                            if latest_reading
                            else None
                        ),
                    }
                    if latest_reading
                    else None
                ),
                "readings_count": len(readings),
            }
        else:
            # All symbols status
            return {
                "symbols": {
                    sym: {
                        "state": self.current_volatility_state.get(
                            sym, VolatilityState.NORMAL
                        ).value,
                        "restricted": self.trading_restricted.get(sym, False),
                        "daily_trades": self.daily_trade_counts.get(sym, 0),
                    }
                    for sym in self.volatility_readings.keys()
                },
                "daily_limit": self.daily_trade_limit,
                "thresholds": {
                    "low": self.thresholds.low_threshold,
                    "high": self.thresholds.high_threshold,
                    "extreme": self.thresholds.extreme_threshold,
                },
                "total_symbols_tracked": len(self.volatility_readings),
                "total_restricted": sum(
                    1 for restricted in self.trading_restricted.values() if restricted
                ),
            }

    def get_volatility_analysis(self, symbol: str, periods: int = 20) -> Dict[str, any]:
        """
        Get detailed volatility analysis for a symbol.

        Args:
            symbol: Trading symbol
            periods: Number of recent periods to analyze

        Returns:
            Dictionary with volatility analysis
        """
        if symbol not in self.volatility_readings:
            return {"error": f"No volatility data for {symbol}"}

        readings = self.volatility_readings[symbol][-periods:]

        if not readings:
            return {"error": f"No recent readings for {symbol}"}

        # Calculate statistics
        atr_percentages = [r.atr_percentage for r in readings]

        avg_atr_percentage = sum(atr_percentages) / len(atr_percentages)
        min_atr_percentage = min(atr_percentages)
        max_atr_percentage = max(atr_percentages)

        # Calculate volatility distribution
        state_counts = {}
        for reading in readings:
            # Recalculate state for each reading (for historical analysis)
            if len(readings) >= self.atr_lookback_periods:
                avg_for_period = avg_atr_percentage  # Simplified
                ratio = reading.atr_percentage / avg_for_period

                if ratio <= self.thresholds.low_threshold:
                    state = VolatilityState.LOW
                elif ratio <= self.thresholds.high_threshold:
                    state = VolatilityState.NORMAL
                elif ratio <= self.thresholds.extreme_threshold:
                    state = VolatilityState.HIGH
                else:
                    state = VolatilityState.EXTREME

                state_counts[state.value] = state_counts.get(state.value, 0) + 1

        return {
            "symbol": symbol,
            "analysis_periods": len(readings),
            "current_state": self.current_volatility_state.get(
                symbol, VolatilityState.NORMAL
            ).value,
            "statistics": {
                "avg_atr_percentage": avg_atr_percentage,
                "min_atr_percentage": min_atr_percentage,
                "max_atr_percentage": max_atr_percentage,
                "atr_range": max_atr_percentage - min_atr_percentage,
                "current_atr_percentage": readings[-1].atr_percentage,
                "volatility_ratio": readings[-1].atr_percentage / avg_atr_percentage,
            },
            "state_distribution": state_counts,
            "recent_readings": [
                {
                    "timestamp": r.timestamp.isoformat(),
                    "atr_value": float(r.atr_value),
                    "atr_percentage": r.atr_percentage,
                    "price": float(r.price),
                }
                for r in readings[-10:]  # Last 10 readings
            ],
        }

    def update_thresholds(
        self,
        low: Optional[float] = None,
        high: Optional[float] = None,
        extreme: Optional[float] = None,
    ) -> None:
        """Update volatility thresholds."""
        if low is not None:
            self.thresholds.low_threshold = low
            self.logger.info(f"Updated low volatility threshold to {low}")

        if high is not None:
            self.thresholds.high_threshold = high
            self.logger.info(f"Updated high volatility threshold to {high}")

        if extreme is not None:
            self.thresholds.extreme_threshold = extreme
            self.logger.info(f"Updated extreme volatility threshold to {extreme}")

        # Re-evaluate all symbols with new thresholds
        for symbol in self.current_volatility_state:
            new_state = self._calculate_volatility_state(symbol)
            self.current_volatility_state[symbol] = new_state
            self._update_trading_restrictions(symbol, new_state)

    def update_daily_trade_limit(self, new_limit: int) -> None:
        """Update daily trade limit."""
        old_limit = self.daily_trade_limit
        self.daily_trade_limit = new_limit

        self.logger.info(f"Updated daily trade limit from {old_limit} to {new_limit}")

        # Re-evaluate restrictions for all symbols
        for symbol in self.current_volatility_state:
            current_state = self.current_volatility_state[symbol]
            self._update_trading_restrictions(symbol, current_state)

    def force_allow_trading(self, symbol: str, reason: str = "Manual override") -> None:
        """Force allow trading for a symbol (emergency override)."""
        self.logger.warning(f"FORCED TRADING ALLOWED for {symbol}: {reason}")
        self.trading_restricted[symbol] = False
        self.restriction_reasons[symbol] = f"Override: {reason}"
