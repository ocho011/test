"""
Traditional technical indicator-based strategy implementation.

This module implements traditional technical analysis using RSI, MACD,
Bollinger Bands, and other classical indicators for signal generation.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
from decimal import Decimal
import uuid

from .base_strategy import AbstractStrategy
from ..core.events import MarketDataEvent
from ..signals.signal_generator import (
    GeneratedSignal,
    SignalDirection,
    SignalType,
    PatternCombination
)
from ..analysis.technical_indicators import TechnicalIndicators


class TraditionalIndicatorStrategy(AbstractStrategy):
    """
    Traditional technical indicator-based trading strategy.

    Uses classical technical analysis indicators including:
    - RSI (Relative Strength Index)
    - MACD (Moving Average Convergence Divergence)
    - Bollinger Bands
    - Stochastic Oscillator
    - Moving Averages (SMA/EMA)
    """

    DEFAULT_PARAMETERS = {
        "rsi_period": 14,
        "rsi_overbought": 70,
        "rsi_oversold": 30,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "bb_period": 20,
        "bb_std": 2.0,
        "sma_fast": 50,
        "sma_slow": 200,
        "use_rsi": True,
        "use_macd": True,
        "use_bb": True,
        "use_ma_cross": True,
        "min_confidence": 0.5,
        "atr_period": 14,
        "risk_reward_ratio": 2.0
    }

    def __init__(
        self,
        name: str = "Traditional_Strategy",
        version: str = "1.0.0",
        parameters: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize traditional indicator strategy.

        Args:
            name: Strategy name
            version: Strategy version
            parameters: Strategy parameters (uses defaults if not provided)
        """
        params = self.DEFAULT_PARAMETERS.copy()
        if parameters:
            params.update(parameters)

        super().__init__(
            name=name,
            version=version,
            description="Traditional technical indicators (RSI, MACD, Bollinger Bands, MA)",
            parameters=params
        )

        self.indicators = TechnicalIndicators()

    async def generate_signals(
        self,
        df: pd.DataFrame,
        current_event: Optional[MarketDataEvent] = None
    ) -> List[GeneratedSignal]:
        """
        Generate signals based on traditional technical indicators.

        Args:
            df: OHLCV DataFrame with market data
            current_event: Optional real-time market data event

        Returns:
            List of generated signals

        Raises:
            ValueError: If data is invalid or insufficient
        """
        if df.empty:
            self.logger.warning("Empty DataFrame provided, no signals generated")
            return []

        min_required = max(
            self.parameters["rsi_period"],
            self.parameters["macd_slow"] + self.parameters["macd_signal"],
            self.parameters["bb_period"],
            self.parameters["sma_slow"]
        )

        if len(df) < min_required:
            raise ValueError(
                f"Insufficient data: need at least {min_required} bars, got {len(df)}"
            )

        # Validate required columns
        required_cols = ["open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        try:
            signals = []

            # Calculate all indicators
            indicators_data = self._calculate_indicators(df)

            # Get current values (last row)
            current_price = float(df["close"].iloc[-1])
            
            # Analyze each indicator for signals
            if self.parameters["use_rsi"]:
                rsi_signal = self._analyze_rsi(
                    indicators_data["rsi"],
                    current_price,
                    df
                )
                if rsi_signal:
                    signals.append(rsi_signal)

            if self.parameters["use_macd"]:
                macd_signal = self._analyze_macd(
                    indicators_data["macd"],
                    indicators_data["macd_signal"],
                    indicators_data["macd_histogram"],
                    current_price,
                    df
                )
                if macd_signal:
                    signals.append(macd_signal)

            if self.parameters["use_bb"]:
                bb_signal = self._analyze_bollinger_bands(
                    indicators_data["bb_upper"],
                    indicators_data["bb_middle"],
                    indicators_data["bb_lower"],
                    current_price,
                    df
                )
                if bb_signal:
                    signals.append(bb_signal)

            if self.parameters["use_ma_cross"]:
                ma_signal = self._analyze_ma_crossover(
                    indicators_data["sma_fast"],
                    indicators_data["sma_slow"],
                    current_price,
                    df
                )
                if ma_signal:
                    signals.append(ma_signal)

            self.logger.info(
                f"Generated {len(signals)} traditional indicator signals"
            )

            return signals

        except Exception as e:
            self.logger.error(f"Error generating traditional signals: {e}", exc_info=True)
            raise

    def _calculate_indicators(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate all technical indicators."""
        indicators = {}

        # RSI
        if self.parameters["use_rsi"]:
            indicators["rsi"] = self.indicators.rsi(
                df["close"],
                period=self.parameters["rsi_period"]
            )

        # MACD
        if self.parameters["use_macd"]:
            macd_data = self.indicators.macd(
                df["close"],
                fast=self.parameters["macd_fast"],
                slow=self.parameters["macd_slow"],
                signal=self.parameters["macd_signal"]
            )
            indicators["macd"] = macd_data["macd"]
            indicators["macd_signal"] = macd_data["signal"]
            indicators["macd_histogram"] = macd_data["histogram"]

        # Bollinger Bands
        if self.parameters["use_bb"]:
            bb_data = self.indicators.bollinger_bands(
                df["close"],
                period=self.parameters["bb_period"],
                std_dev=self.parameters["bb_std"]
            )
            indicators["bb_upper"] = bb_data["upper"]
            indicators["bb_middle"] = bb_data["middle"]
            indicators["bb_lower"] = bb_data["lower"]

        # Moving Averages
        if self.parameters["use_ma_cross"]:
            indicators["sma_fast"] = self.indicators.sma(
                df["close"],
                period=self.parameters["sma_fast"]
            )
            indicators["sma_slow"] = self.indicators.sma(
                df["close"],
                period=self.parameters["sma_slow"]
            )

        # ATR for stop loss calculation
        indicators["atr"] = self.indicators.atr(
            df["high"],
            df["low"],
            df["close"],
            period=self.parameters["atr_period"]
        )

        return indicators

    def _analyze_rsi(
        self,
        rsi: pd.Series,
        current_price: float,
        df: pd.DataFrame
    ) -> Optional[GeneratedSignal]:
        """Analyze RSI for signals."""
        current_rsi = float(rsi.iloc[-1])
        prev_rsi = float(rsi.iloc[-2])

        # RSI oversold → bullish signal
        if current_rsi < self.parameters["rsi_oversold"] and prev_rsi >= self.parameters["rsi_oversold"]:
            return self._create_signal(
                direction=SignalDirection.LONG,
                signal_type=SignalType.BUY,
                current_price=current_price,
                df=df,
                reason=f"RSI oversold crossover: {current_rsi:.2f}",
                confidence=self._calculate_rsi_confidence(current_rsi, SignalDirection.LONG)
            )

        # RSI overbought → bearish signal
        elif current_rsi > self.parameters["rsi_overbought"] and prev_rsi <= self.parameters["rsi_overbought"]:
            return self._create_signal(
                direction=SignalDirection.SHORT,
                signal_type=SignalType.SELL,
                current_price=current_price,
                df=df,
                reason=f"RSI overbought crossover: {current_rsi:.2f}",
                confidence=self._calculate_rsi_confidence(current_rsi, SignalDirection.SHORT)
            )

        return None

    def _analyze_macd(
        self,
        macd: pd.Series,
        signal: pd.Series,
        histogram: pd.Series,
        current_price: float,
        df: pd.DataFrame
    ) -> Optional[GeneratedSignal]:
        """Analyze MACD for signals."""
        current_hist = float(histogram.iloc[-1])
        prev_hist = float(histogram.iloc[-2])

        # MACD bullish crossover
        if current_hist > 0 and prev_hist <= 0:
            return self._create_signal(
                direction=SignalDirection.LONG,
                signal_type=SignalType.BUY,
                current_price=current_price,
                df=df,
                reason=f"MACD bullish crossover: histogram {current_hist:.4f}",
                confidence=min(0.8, abs(current_hist) * 10 + 0.5)
            )

        # MACD bearish crossover
        elif current_hist < 0 and prev_hist >= 0:
            return self._create_signal(
                direction=SignalDirection.SHORT,
                signal_type=SignalType.SELL,
                current_price=current_price,
                df=df,
                reason=f"MACD bearish crossover: histogram {current_hist:.4f}",
                confidence=min(0.8, abs(current_hist) * 10 + 0.5)
            )

        return None

    def _analyze_bollinger_bands(
        self,
        upper: pd.Series,
        middle: pd.Series,
        lower: pd.Series,
        current_price: float,
        df: pd.DataFrame
    ) -> Optional[GeneratedSignal]:
        """Analyze Bollinger Bands for signals."""
        current_upper = float(upper.iloc[-1])
        current_lower = float(lower.iloc[-1])
        current_middle = float(middle.iloc[-1])

        # Price touches lower band → bullish reversal
        if current_price <= current_lower:
            return self._create_signal(
                direction=SignalDirection.LONG,
                signal_type=SignalType.BUY,
                current_price=current_price,
                df=df,
                reason=f"Price at lower Bollinger Band: {current_price:.2f} <= {current_lower:.2f}",
                confidence=0.65
            )

        # Price touches upper band → bearish reversal
        elif current_price >= current_upper:
            return self._create_signal(
                direction=SignalDirection.SHORT,
                signal_type=SignalType.SELL,
                current_price=current_price,
                df=df,
                reason=f"Price at upper Bollinger Band: {current_price:.2f} >= {current_upper:.2f}",
                confidence=0.65
            )

        return None

    def _analyze_ma_crossover(
        self,
        fast_ma: pd.Series,
        slow_ma: pd.Series,
        current_price: float,
        df: pd.DataFrame
    ) -> Optional[GeneratedSignal]:
        """Analyze moving average crossover for signals."""
        current_fast = float(fast_ma.iloc[-1])
        current_slow = float(slow_ma.iloc[-1])
        prev_fast = float(fast_ma.iloc[-2])
        prev_slow = float(slow_ma.iloc[-2])

        # Golden cross (fast MA crosses above slow MA)
        if current_fast > current_slow and prev_fast <= prev_slow:
            return self._create_signal(
                direction=SignalDirection.LONG,
                signal_type=SignalType.BUY,
                current_price=current_price,
                df=df,
                reason=f"Golden cross: MA{self.parameters['sma_fast']} crossed above MA{self.parameters['sma_slow']}",
                confidence=0.75
            )

        # Death cross (fast MA crosses below slow MA)
        elif current_fast < current_slow and prev_fast >= prev_slow:
            return self._create_signal(
                direction=SignalDirection.SHORT,
                signal_type=SignalType.SELL,
                current_price=current_price,
                df=df,
                reason=f"Death cross: MA{self.parameters['sma_fast']} crossed below MA{self.parameters['sma_slow']}",
                confidence=0.75
            )

        return None

    def _calculate_rsi_confidence(self, rsi_value: float, direction: SignalDirection) -> float:
        """Calculate confidence based on RSI value."""
        if direction == SignalDirection.LONG:
            # More oversold = higher confidence
            distance = self.parameters["rsi_oversold"] - rsi_value
            confidence = 0.5 + (distance / self.parameters["rsi_oversold"]) * 0.3
        else:
            # More overbought = higher confidence
            distance = rsi_value - self.parameters["rsi_overbought"]
            confidence = 0.5 + (distance / (100 - self.parameters["rsi_overbought"])) * 0.3

        return min(0.9, max(0.5, confidence))

    def _create_signal(
        self,
        direction: SignalDirection,
        signal_type: SignalType,
        current_price: float,
        df: pd.DataFrame,
        reason: str,
        confidence: float
    ) -> GeneratedSignal:
        """Create a GeneratedSignal with stop loss and take profit."""
        atr = float(self.indicators.atr(
            df["high"],
            df["low"],
            df["close"],
            period=self.parameters["atr_period"]
        ).iloc[-1])

        # Calculate stop loss and take profit
        if direction == SignalDirection.LONG:
            stop_loss = Decimal(str(current_price - (atr * 1.5)))
            take_profit = Decimal(str(current_price + (atr * 1.5 * self.parameters["risk_reward_ratio"])))
        else:
            stop_loss = Decimal(str(current_price + (atr * 1.5)))
            take_profit = Decimal(str(current_price - (atr * 1.5 * self.parameters["risk_reward_ratio"])))

        risk_reward = self.parameters["risk_reward_ratio"]

        return GeneratedSignal(
            signal_id=str(uuid.uuid4()),
            direction=direction,
            signal_type=signal_type,
            symbol="BTCUSDT",  # TODO: Make dynamic
            timeframe="1h",  # TODO: Make dynamic
            entry_price=Decimal(str(current_price)),
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence_score=confidence,
            pattern_combination=PatternCombination.STRUCTURE_ONLY,
            contributing_patterns={"strategy": "traditional_indicators"},
            reasoning=reason,
            timestamp=datetime.now(),
            validity_duration=timedelta(hours=4),
            risk_reward_ratio=risk_reward
        )

    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """Validate traditional strategy parameters."""
        try:
            # RSI parameters
            if "rsi_period" in params:
                if not isinstance(params["rsi_period"], int) or params["rsi_period"] < 2:
                    return False

            if "rsi_overbought" in params:
                if not isinstance(params["rsi_overbought"], (int, float)) or not (50 <= params["rsi_overbought"] <= 100):
                    return False

            if "rsi_oversold" in params:
                if not isinstance(params["rsi_oversold"], (int, float)) or not (0 <= params["rsi_oversold"] <= 50):
                    return False

            # MACD parameters
            for key in ["macd_fast", "macd_slow", "macd_signal"]:
                if key in params:
                    if not isinstance(params[key], int) or params[key] < 1:
                        return False

            # Bollinger Bands
            if "bb_period" in params:
                if not isinstance(params["bb_period"], int) or params["bb_period"] < 2:
                    return False

            if "bb_std" in params:
                if not isinstance(params["bb_std"], (int, float)) or params["bb_std"] <= 0:
                    return False

            # Moving averages
            for key in ["sma_fast", "sma_slow"]:
                if key in params:
                    if not isinstance(params[key], int) or params[key] < 1:
                        return False

            # Boolean flags
            for flag in ["use_rsi", "use_macd", "use_bb", "use_ma_cross"]:
                if flag in params and not isinstance(params[flag], bool):
                    return False

            return True
        except Exception:
            return False

    def get_parameter_schema(self) -> Dict[str, Any]:
        """Get traditional strategy parameter schema."""
        return {
            "rsi_period": {
                "type": "int",
                "default": 14,
                "min": 2,
                "max": 100,
                "description": "RSI calculation period"
            },
            "rsi_overbought": {
                "type": "int",
                "default": 70,
                "min": 50,
                "max": 100,
                "description": "RSI overbought threshold"
            },
            "rsi_oversold": {
                "type": "int",
                "default": 30,
                "min": 0,
                "max": 50,
                "description": "RSI oversold threshold"
            },
            "macd_fast": {
                "type": "int",
                "default": 12,
                "min": 1,
                "max": 50,
                "description": "MACD fast period"
            },
            "macd_slow": {
                "type": "int",
                "default": 26,
                "min": 1,
                "max": 100,
                "description": "MACD slow period"
            },
            "macd_signal": {
                "type": "int",
                "default": 9,
                "min": 1,
                "max": 50,
                "description": "MACD signal period"
            },
            "bb_period": {
                "type": "int",
                "default": 20,
                "min": 2,
                "max": 100,
                "description": "Bollinger Bands period"
            },
            "bb_std": {
                "type": "float",
                "default": 2.0,
                "min": 0.1,
                "max": 5.0,
                "description": "Bollinger Bands standard deviation"
            },
            "sma_fast": {
                "type": "int",
                "default": 50,
                "min": 1,
                "max": 500,
                "description": "Fast moving average period"
            },
            "sma_slow": {
                "type": "int",
                "default": 200,
                "min": 1,
                "max": 500,
                "description": "Slow moving average period"
            },
            "use_rsi": {
                "type": "bool",
                "default": True,
                "description": "Enable RSI signals"
            },
            "use_macd": {
                "type": "bool",
                "default": True,
                "description": "Enable MACD signals"
            },
            "use_bb": {
                "type": "bool",
                "default": True,
                "description": "Enable Bollinger Bands signals"
            },
            "use_ma_cross": {
                "type": "bool",
                "default": True,
                "description": "Enable MA crossover signals"
            },
            "risk_reward_ratio": {
                "type": "float",
                "default": 2.0,
                "min": 0.5,
                "max": 10.0,
                "description": "Risk/reward ratio for position sizing"
            }
        }
