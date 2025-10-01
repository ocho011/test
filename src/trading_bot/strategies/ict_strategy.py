"""
ICT (Inner Circle Trader) strategy implementation.

This module implements ICT methodology using order blocks, fair value gaps,
and market structure analysis for signal generation.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
from decimal import Decimal

from .base_strategy import AbstractStrategy
from ..core.events import MarketDataEvent
from ..signals.signal_generator import SignalGenerator, GeneratedSignal, PatternInput
from ..analysis.ict_analyzer import (
    OrderBlockDetector,
    FairValueGapAnalyzer,
    MarketStructureAnalyzer
)


class ICTStrategy(AbstractStrategy):
    """
    ICT methodology-based trading strategy.

    Uses Inner Circle Trader concepts including:
    - Order blocks (bullish/bearish)
    - Fair value gaps (FVG)
    - Market structure (highs/lows)
    - Break of structure (BOS)
    - Change of character (CHoCH)
    """

    DEFAULT_PARAMETERS = {
        "lookback_period": 50,
        "min_confirmation_bars": 3,
        "min_confidence_threshold": 0.6,
        "max_signals_per_timeframe": 3,
        "use_order_blocks": True,
        "use_fvg": True,
        "use_market_structure": True,
        "fvg_min_gap_percent": 0.1,
        "ob_displacement_threshold": 1.5
    }

    def __init__(
        self,
        name: str = "ICT_Strategy",
        version: str = "1.0.0",
        parameters: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize ICT strategy.

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
            description="ICT methodology with order blocks, FVG, and market structure",
            parameters=params
        )

        # Initialize ICT analyzers
        self.order_block_detector = OrderBlockDetector(
            lookback_period=self.parameters["lookback_period"],
            min_confirmation_bars=self.parameters["min_confirmation_bars"]
        )

        self.fvg_analyzer = FairValueGapAnalyzer(
            min_gap_size_pips=self.parameters["fvg_min_gap_percent"] * 100  # Convert percent to pips
        )

        self.market_structure_analyzer = MarketStructureAnalyzer(
            lookback_period=self.parameters["lookback_period"]
        )

        self.signal_generator = SignalGenerator(
            min_confidence_threshold=self.parameters["min_confidence_threshold"],
            max_signals_per_timeframe=self.parameters["max_signals_per_timeframe"]
        )

    async def generate_signals(
        self,
        df: pd.DataFrame,
        current_event: Optional[MarketDataEvent] = None
    ) -> List[GeneratedSignal]:
        """
        Generate ICT-based trading signals.

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

        if len(df) < self.parameters["lookback_period"]:
            raise ValueError(
                f"Insufficient data: need at least {self.parameters['lookback_period']} bars, "
                f"got {len(df)}"
            )

        # Validate required columns
        required_cols = ["open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        try:
            # Extract current price and timestamp from DataFrame
            current_price = Decimal(str(df["close"].iloc[-1]))
            timestamp = datetime.now() if not isinstance(df.index[-1], datetime) else df.index[-1]
            timeframe = "1h"  # Default timeframe, can be parameterized
            symbol = "BTCUSDT"  # Default symbol, can be parameterized

            # Initialize empty pattern collections
            order_blocks_list = []
            fair_value_gaps_list = []
            market_structures_list = []

            # Detect order blocks
            if self.parameters["use_order_blocks"]:
                order_blocks_list = self.order_block_detector.analyze(df)
                self.logger.debug(f"Detected {len(order_blocks_list)} order blocks")

            # Analyze fair value gaps
            if self.parameters["use_fvg"]:
                fair_value_gaps_list = self.fvg_analyzer.analyze(df)
                self.logger.debug(f"Detected {len(fair_value_gaps_list)} fair value gaps")

            # Analyze market structure
            if self.parameters["use_market_structure"]:
                market_structure = self.market_structure_analyzer.analyze(df)
                # Combine all market structure components into market_structures list
                market_structures_list = [market_structure]
                self.logger.debug(
                    f"Detected market structure: "
                    f"{market_structure.get('total_bos', 0)} BOS, "
                    f"{market_structure.get('total_choch', 0)} CHoCH"
                )

            # Create PatternInput with correct structure
            pattern_input = PatternInput(
                order_blocks=order_blocks_list,
                fair_value_gaps=fair_value_gaps_list,
                market_structures=market_structures_list,
                timeframe=timeframe,
                symbol=symbol,
                current_price=current_price,
                timestamp=timestamp
            )

            # Generate signals from patterns
            signals = self.signal_generator.generate_signals(pattern_input)

            self.logger.info(
                f"Generated {len(signals)} ICT signals from pattern analysis"
            )

            return signals

        except Exception as e:
            self.logger.error(f"Error generating ICT signals: {e}", exc_info=True)
            raise

    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """
        Validate ICT strategy parameters.

        Args:
            params: Parameters to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            # Validate lookback period
            if "lookback_period" in params:
                if not isinstance(params["lookback_period"], int) or params["lookback_period"] < 10:
                    return False

        # Validate min_confirmation_bars
            if "min_confirmation_bars" in params:
                if not isinstance(params["min_confirmation_bars"], int) or params["min_confirmation_bars"] < 1:
                    return False

            # Validate confidence threshold
            if "min_confidence_threshold" in params:
                threshold = params["min_confidence_threshold"]
                if not isinstance(threshold, (int, float)) or not (0.0 <= threshold <= 1.0):
                    return False

            # Validate max signals
            if "max_signals_per_timeframe" in params:
                if not isinstance(params["max_signals_per_timeframe"], int) or params["max_signals_per_timeframe"] < 1:
                    return False

            # Validate boolean flags
            for flag in ["use_order_blocks", "use_fvg", "use_market_structure"]:
                if flag in params and not isinstance(params[flag], bool):
                    return False

            # Validate FVG gap percent
            if "fvg_min_gap_percent" in params:
                gap = params["fvg_min_gap_percent"]
                if not isinstance(gap, (int, float)) or gap < 0:
                    return False

            # Validate displacement threshold
            if "ob_displacement_threshold" in params:
                threshold = params["ob_displacement_threshold"]
                if not isinstance(threshold, (int, float)) or threshold < 1.0:
                    return False

            return True
        except Exception:
            return False

    def get_parameter_schema(self) -> Dict[str, Any]:
        """
        Get ICT strategy parameter schema.

        Returns:
            Parameter schema definition
        """
        return {
            "lookback_period": {
                "type": "int",
                "default": 50,
                "min": 10,
                "max": 500,
                "description": "Number of bars to analyze for pattern detection"
            },
            "min_confirmation_bars": {
                "type": "int",
                "default": 3,
                "min": 1,
                "max": 10,
                "description": "Minimum bars required to confirm order blocks"
            },
            "min_confidence_threshold": {
                "type": "float",
                "default": 0.6,
                "min": 0.0,
                "max": 1.0,
                "description": "Minimum confidence score for signal generation"
            },
            "max_signals_per_timeframe": {
                "type": "int",
                "default": 3,
                "min": 1,
                "max": 10,
                "description": "Maximum number of signals to generate per timeframe"
            },
            "use_order_blocks": {
                "type": "bool",
                "default": True,
                "description": "Enable order block detection"
            },
            "use_fvg": {
                "type": "bool",
                "default": True,
                "description": "Enable fair value gap analysis"
            },
            "use_market_structure": {
                "type": "bool",
                "default": True,
                "description": "Enable market structure analysis"
            },
            "fvg_min_gap_percent": {
                "type": "float",
                "default": 0.1,
                "min": 0.0,
                "max": 5.0,
                "description": "Minimum gap percentage for FVG detection"
            },
            "ob_displacement_threshold": {
                "type": "float",
                "default": 1.5,
                "min": 1.0,
                "max": 5.0,
                "description": "ATR multiplier for order block displacement"
            }
        }

    async def _start(self) -> None:
        """Initialize ICT strategy components."""
        await super()._start()
        self.logger.info(
            f"ICT Strategy initialized with: "
            f"OB={self.parameters['use_order_blocks']}, "
            f"FVG={self.parameters['use_fvg']}, "
            f"MS={self.parameters['use_market_structure']}"
        )

    async def _stop(self) -> None:
        """Cleanup ICT strategy resources."""
        await super()._stop()
        self.logger.info("ICT Strategy stopped")
