"""
ICT (Inner Circle Trader) Technical Analysis Engine

This module implements ICT trading concepts including:
- Order Block Detection (OrderBlockDetector)
- Fair Value Gap Analysis (FairValueGapAnalyzer)
- Market Structure Analysis (MarketStructureAnalyzer)
- Multi-timeframe Management (TimeFrameManager)
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PatternType(Enum):
    """ICT Pattern Types"""

    ORDER_BLOCK = "order_block"
    FAIR_VALUE_GAP = "fair_value_gap"
    BREAK_OF_STRUCTURE = "break_of_structure"
    CHANGE_OF_CHARACTER = "change_of_character"


class TrendDirection(Enum):
    """Market Trend Direction"""

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class SwingPoint:
    """Represents a swing high or swing low point"""

    index: int
    price: float
    timestamp: pd.Timestamp
    is_high: bool  # True for swing high, False for swing low
    confirmation_bars: int = 0


@dataclass
class OrderBlock:
    """Represents an Order Block pattern"""

    start_index: int
    end_index: int
    high_price: float
    low_price: float
    timestamp: pd.Timestamp
    direction: TrendDirection
    confidence: float
    is_valid: bool = True
    mitigation_level: Optional[float] = None


@dataclass
class FairValueGap:
    """Represents a Fair Value Gap pattern"""

    start_index: int
    end_index: int
    gap_high: float
    gap_low: float
    timestamp: pd.Timestamp
    direction: TrendDirection
    gap_size: float
    is_filled: bool = False
    fill_percentage: float = 0.0


class OrderBlockDetector:
    """
    Detects Order Blocks based on swing highs/lows with ICT methodology

    Order Blocks are institutional order zones where large orders were placed,
    identified by swing points that show strong rejection or continuation.
    """

    def __init__(self, lookback_period: int = 5, min_confirmation_bars: int = 3):
        """
        Initialize OrderBlockDetector

        Args:
            lookback_period: Number of bars to look back for swing point identification
            min_confirmation_bars: Minimum bars required to confirm swing point
        """
        self.lookback_period = lookback_period
        self.min_confirmation_bars = min_confirmation_bars
        self.logger = logging.getLogger(f"{__name__}.OrderBlockDetector")

    def find_swing_points(self, data: pd.DataFrame) -> List[SwingPoint]:
        """
        Identify swing highs and lows in price data

        Args:
            data: DataFrame with OHLC columns

        Returns:
            List of SwingPoint objects
        """
        if len(data) < self.lookback_period * 2 + 1:
            return []

        swing_points = []
        highs = data["high"].values
        lows = data["low"].values
        timestamps = data.index

        # Find swing highs
        for i in range(self.lookback_period, len(data) - self.lookback_period):
            # Check if current high is highest in the lookback window
            window_start = i - self.lookback_period
            window_end = i + self.lookback_period + 1

            if highs[i] == np.max(highs[window_start:window_end]):
                # Confirm swing high with additional bars
                confirmation_bars = 0
                for j in range(
                    i + 1, min(i + self.min_confirmation_bars + 1, len(data))
                ):
                    if highs[j] < highs[i]:
                        confirmation_bars += 1
                    else:
                        break

                if confirmation_bars >= self.min_confirmation_bars:
                    swing_points.append(
                        SwingPoint(
                            index=i,
                            price=highs[i],
                            timestamp=timestamps[i],
                            is_high=True,
                            confirmation_bars=confirmation_bars,
                        )
                    )

        # Find swing lows
        for i in range(self.lookback_period, len(data) - self.lookback_period):
            window_start = i - self.lookback_period
            window_end = i + self.lookback_period + 1

            if lows[i] == np.min(lows[window_start:window_end]):
                # Confirm swing low with additional bars
                confirmation_bars = 0
                for j in range(
                    i + 1, min(i + self.min_confirmation_bars + 1, len(data))
                ):
                    if lows[j] > lows[i]:
                        confirmation_bars += 1
                    else:
                        break

                if confirmation_bars >= self.min_confirmation_bars:
                    swing_points.append(
                        SwingPoint(
                            index=i,
                            price=lows[i],
                            timestamp=timestamps[i],
                            is_high=False,
                            confirmation_bars=confirmation_bars,
                        )
                    )

        return sorted(swing_points, key=lambda x: x.index)

    def identify_order_blocks(
        self, data: pd.DataFrame, swing_points: List[SwingPoint]
    ) -> List[OrderBlock]:
        """
        Identify Order Blocks from swing points

        Args:
            data: DataFrame with OHLC data
            swing_points: List of identified swing points

        Returns:
            List of OrderBlock objects
        """
        order_blocks = []

        for swing_point in swing_points:
            # Look for the candle that broke the swing point (displacement candle)
            displacement_candle = self._find_displacement_candle(data, swing_point)

            if displacement_candle is not None:
                # Find the last opposite colored candle before displacement
                order_block_candle = self._find_order_block_candle(
                    data, swing_point, displacement_candle
                )

                if order_block_candle is not None:
                    order_block = self._create_order_block(
                        data, swing_point, order_block_candle
                    )
                    if order_block:
                        order_blocks.append(order_block)

        return order_blocks

    def _find_displacement_candle(
        self, data: pd.DataFrame, swing_point: SwingPoint
    ) -> Optional[int]:
        """
        Find the candle that broke through the swing point (displacement)

        Args:
            data: OHLC DataFrame
            swing_point: The swing point to analyze

        Returns:
            Index of displacement candle or None
        """
        start_search = swing_point.index + 1
        max_search = min(start_search + 20, len(data))  # Limit search to 20 bars

        for i in range(start_search, max_search):
            if swing_point.is_high:
                # For swing high, look for bullish break (close above swing high)
                if data.iloc[i]["close"] > swing_point.price:
                    return i
            else:
                # For swing low, look for bearish break (close below swing low)
                if data.iloc[i]["close"] < swing_point.price:
                    return i

        return None

    def _find_order_block_candle(
        self, data: pd.DataFrame, swing_point: SwingPoint, displacement_index: int
    ) -> Optional[int]:
        """
        Find the order block candle (last opposite candle before displacement)

        Args:
            data: OHLC DataFrame
            swing_point: The swing point
            displacement_index: Index of displacement candle

        Returns:
            Index of order block candle or None
        """
        displacement_candle = data.iloc[displacement_index]
        is_displacement_bullish = (
            displacement_candle["close"] > displacement_candle["open"]
        )

        # Look backwards from displacement to find last opposite colored candle
        for i in range(displacement_index - 1, swing_point.index - 1, -1):
            candle = data.iloc[i]
            is_candle_bullish = candle["close"] > candle["open"]

            # If displacement is bullish, look for last bearish candle
            # If displacement is bearish, look for last bullish candle
            if is_displacement_bullish and not is_candle_bullish:
                return i
            elif not is_displacement_bullish and is_candle_bullish:
                return i

        return None

    def _create_order_block(
        self, data: pd.DataFrame, swing_point: SwingPoint, order_block_index: int
    ) -> Optional[OrderBlock]:
        """
        Create OrderBlock object from identified candle

        Args:
            data: OHLC DataFrame
            swing_point: The swing point
            order_block_index: Index of order block candle

        Returns:
            OrderBlock object or None
        """
        try:
            ob_candle = data.iloc[order_block_index]

            # Determine direction based on swing point type
            direction = (
                TrendDirection.BULLISH
                if not swing_point.is_high
                else TrendDirection.BEARISH
            )

            # Calculate confidence based on various factors
            confidence = self._calculate_order_block_confidence(
                data, swing_point, order_block_index
            )

            return OrderBlock(
                start_index=order_block_index,
                end_index=order_block_index,  # Single candle order block
                high_price=float(ob_candle["high"]),
                low_price=float(ob_candle["low"]),
                timestamp=ob_candle.name,
                direction=direction,
                confidence=confidence,
                is_valid=True,
            )

        except Exception as e:
            self.logger.error(f"Error creating order block: {e}")
            return None

    def _calculate_order_block_confidence(
        self, data: pd.DataFrame, swing_point: SwingPoint, order_block_index: int
    ) -> float:
        """
        Calculate confidence score for order block

        Args:
            data: OHLC DataFrame
            swing_point: The swing point
            order_block_index: Index of order block candle

        Returns:
            Confidence score between 0 and 1
        """
        confidence_factors = []

        # Factor 1: Confirmation bars strength
        confirmation_strength = min(swing_point.confirmation_bars / 5.0, 1.0)
        confidence_factors.append(confirmation_strength)

        # Factor 2: Order block candle size relative to recent average
        ob_candle = data.iloc[order_block_index]
        candle_size = abs(ob_candle["high"] - ob_candle["low"])

        # Calculate average candle size for last 20 periods
        recent_data = data.iloc[max(0, order_block_index - 20) : order_block_index]
        if len(recent_data) > 0:
            avg_size = (recent_data["high"] - recent_data["low"]).mean()
            size_factor = min(candle_size / avg_size, 2.0) / 2.0  # Normalize to 0-1
            confidence_factors.append(size_factor)

        # Factor 3: Volume confirmation (if available)
        if "volume" in data.columns:
            ob_volume = ob_candle["volume"]
            recent_avg_volume = data.iloc[
                max(0, order_block_index - 20) : order_block_index
            ]["volume"].mean()
            if recent_avg_volume > 0:
                volume_factor = min(ob_volume / recent_avg_volume, 3.0) / 3.0
                confidence_factors.append(volume_factor)

        # Return weighted average
        return np.mean(confidence_factors) if confidence_factors else 0.5

    def analyze(self, data: pd.DataFrame) -> List[OrderBlock]:
        """
        Main analysis method to detect order blocks

        Args:
            data: DataFrame with OHLC data

        Returns:
            List of detected OrderBlock objects
        """
        try:
            self.logger.info(f"Analyzing {len(data)} bars for order blocks")

            # Find swing points
            swing_points = self.find_swing_points(data)
            self.logger.info(f"Found {len(swing_points)} swing points")

            # Identify order blocks
            order_blocks = self.identify_order_blocks(data, swing_points)
            self.logger.info(f"Identified {len(order_blocks)} order blocks")

            return order_blocks

        except Exception as e:
            self.logger.error(f"Error in order block analysis: {e}")
            return []


class FairValueGapAnalyzer:
    """
    Analyzes Fair Value Gaps (FVG) - price gaps that represent inefficient pricing

    Fair Value Gaps occur when there's a gap between the high of one candle
    and the low of another candle, with a candle in between that doesn't overlap.
    """

    def __init__(
        self, min_gap_size_pips: float = 5.0, invalidation_threshold: float = 0.5
    ):
        """
        Initialize FairValueGapAnalyzer

        Args:
            min_gap_size_pips: Minimum gap size in pips to consider
            invalidation_threshold: Percentage of gap that needs to be filled for invalidation
        """
        self.min_gap_size_pips = min_gap_size_pips
        self.invalidation_threshold = invalidation_threshold
        self.logger = logging.getLogger(f"{__name__}.FairValueGapAnalyzer")

    def detect_gaps(self, data: pd.DataFrame) -> List[FairValueGap]:
        """
        Detect Fair Value Gaps in price data

        Args:
            data: DataFrame with OHLC columns

        Returns:
            List of FairValueGap objects
        """
        if len(data) < 3:
            return []

        gaps = []

        for i in range(1, len(data) - 1):
            # Check for bullish FVG (gap up)
            bullish_gap = self._check_bullish_gap(data, i)
            if bullish_gap:
                gaps.append(bullish_gap)

            # Check for bearish FVG (gap down)
            bearish_gap = self._check_bearish_gap(data, i)
            if bearish_gap:
                gaps.append(bearish_gap)

        return gaps

    def _check_bullish_gap(
        self, data: pd.DataFrame, middle_index: int
    ) -> Optional[FairValueGap]:
        """
        Check for bullish Fair Value Gap at given index

        Args:
            data: OHLC DataFrame
            middle_index: Index of middle candle

        Returns:
            FairValueGap object if found, None otherwise
        """
        try:
            left_candle = data.iloc[middle_index - 1]
            middle_candle = data.iloc[middle_index]
            right_candle = data.iloc[middle_index + 1]

            # For bullish FVG: left.high < right.low with middle candle not filling the gap
            if (
                left_candle["high"] < right_candle["low"]
                and middle_candle["low"] > left_candle["high"]
                and middle_candle["high"] < right_candle["low"]
            ):

                gap_low = left_candle["high"]
                gap_high = right_candle["low"]
                gap_size = gap_high - gap_low

                # Check minimum gap size
                if gap_size >= self.min_gap_size_pips * self._get_pip_value(data):
                    return FairValueGap(
                        start_index=middle_index - 1,
                        end_index=middle_index + 1,
                        gap_high=gap_high,
                        gap_low=gap_low,
                        timestamp=middle_candle.name,
                        direction=TrendDirection.BULLISH,
                        gap_size=gap_size,
                        is_filled=False,
                        fill_percentage=0.0,
                    )

        except Exception as e:
            self.logger.error(f"Error checking bullish gap: {e}")

        return None

    def _check_bearish_gap(
        self, data: pd.DataFrame, middle_index: int
    ) -> Optional[FairValueGap]:
        """
        Check for bearish Fair Value Gap at given index

        Args:
            data: OHLC DataFrame
            middle_index: Index of middle candle

        Returns:
            FairValueGap object if found, None otherwise
        """
        try:
            left_candle = data.iloc[middle_index - 1]
            middle_candle = data.iloc[middle_index]
            right_candle = data.iloc[middle_index + 1]

            # For bearish FVG: left.low > right.high with middle candle not filling the gap
            if (
                left_candle["low"] > right_candle["high"]
                and middle_candle["high"] < left_candle["low"]
                and middle_candle["low"] > right_candle["high"]
            ):

                gap_high = left_candle["low"]
                gap_low = right_candle["high"]
                gap_size = gap_high - gap_low

                # Check minimum gap size
                if gap_size >= self.min_gap_size_pips * self._get_pip_value(data):
                    return FairValueGap(
                        start_index=middle_index - 1,
                        end_index=middle_index + 1,
                        gap_high=gap_high,
                        gap_low=gap_low,
                        timestamp=middle_candle.name,
                        direction=TrendDirection.BEARISH,
                        gap_size=gap_size,
                        is_filled=False,
                        fill_percentage=0.0,
                    )

        except Exception as e:
            self.logger.error(f"Error checking bearish gap: {e}")

        return None

    def _get_pip_value(self, data: pd.DataFrame) -> float:
        """
        Estimate pip value based on price levels

        Args:
            data: OHLC DataFrame

        Returns:
            Estimated pip value
        """
        # Simple estimation based on price level
        avg_price = data["close"].mean()
        if avg_price > 100:  # Likely JPY pair or stock
            return 0.01
        elif avg_price > 1:  # Major currency pairs
            return 0.0001
        else:  # Crypto or other
            return avg_price * 0.0001

    def update_gap_status(
        self, gaps: List[FairValueGap], current_data: pd.DataFrame
    ) -> List[FairValueGap]:
        """
        Update the fill status of existing gaps

        Args:
            gaps: List of existing gaps
            current_data: Current price data

        Returns:
            Updated list of gaps
        """
        updated_gaps = []

        for gap in gaps:
            if gap.is_filled:
                updated_gaps.append(gap)
                continue

            # Check fill percentage
            fill_percentage = self._calculate_fill_percentage(gap, current_data)
            gap.fill_percentage = fill_percentage

            # Mark as filled if threshold exceeded
            if fill_percentage >= self.invalidation_threshold:
                gap.is_filled = True

            updated_gaps.append(gap)

        return updated_gaps

    def _calculate_fill_percentage(
        self, gap: FairValueGap, data: pd.DataFrame
    ) -> float:
        """
        Calculate how much of the gap has been filled

        Args:
            gap: FairValueGap object
            data: Current price data

        Returns:
            Fill percentage (0.0 to 1.0)
        """
        try:
            # Get price data after gap formation
            gap_end_index = gap.end_index
            if gap_end_index >= len(data):
                return 0.0

            post_gap_data = data.iloc[gap_end_index:]
            if len(post_gap_data) == 0:
                return 0.0

            gap_range = gap.gap_high - gap.gap_low

            if gap.direction == TrendDirection.BULLISH:
                # For bullish gap, check how far price has moved back down into the gap
                lowest_fill = post_gap_data["low"].min()
                if lowest_fill >= gap.gap_high:
                    return 0.0
                elif lowest_fill <= gap.gap_low:
                    return 1.0
                else:
                    filled_amount = gap.gap_high - lowest_fill
                    return filled_amount / gap_range

            else:  # Bearish gap
                # For bearish gap, check how far price has moved back up into the gap
                highest_fill = post_gap_data["high"].max()
                if highest_fill <= gap.gap_low:
                    return 0.0
                elif highest_fill >= gap.gap_high:
                    return 1.0
                else:
                    filled_amount = highest_fill - gap.gap_low
                    return filled_amount / gap_range

        except Exception as e:
            self.logger.error(f"Error calculating fill percentage: {e}")
            return 0.0

    def analyze(
        self, data: pd.DataFrame, existing_gaps: Optional[List[FairValueGap]] = None
    ) -> List[FairValueGap]:
        """
        Main analysis method to detect and update Fair Value Gaps

        Args:
            data: DataFrame with OHLC data
            existing_gaps: List of previously detected gaps to update

        Returns:
            List of FairValueGap objects (new + updated existing)
        """
        try:
            self.logger.info(f"Analyzing {len(data)} bars for Fair Value Gaps")

            # Detect new gaps
            new_gaps = self.detect_gaps(data)
            self.logger.info(f"Found {len(new_gaps)} new gaps")

            # Update existing gaps if provided
            if existing_gaps:
                updated_existing = self.update_gap_status(existing_gaps, data)
                # Combine and remove duplicates
                all_gaps = updated_existing + new_gaps
            else:
                all_gaps = new_gaps

            return all_gaps

        except Exception as e:
            self.logger.error(f"Error in Fair Value Gap analysis: {e}")
            return existing_gaps or []


@dataclass
class MarketStructure:
    """Represents market structure information"""

    structure_type: str  # "BOS" or "CHoCH"
    direction: TrendDirection
    break_level: float
    break_index: int
    timestamp: pd.Timestamp
    previous_structure: Optional["MarketStructure"] = None
    strength: float = 0.0
    pattern_type: str = (
        "market_structure"  # Add pattern_type for validation compatibility
    )
    confidence_score: float = (
        0.8  # Add confidence_score for signal generation compatibility
    )
    break_price: float = None  # Add break_price alias for compatibility

    def __post_init__(self):
        """Set break_price to break_level for compatibility"""
        if self.break_price is None:
            self.break_price = self.break_level


class MarketStructureAnalyzer:
    """
    Analyzes Market Structure changes using ICT concepts

    BOS (Break of Structure): Price breaks previous swing high/low in same direction
    CHoCH (Change of Character): Price breaks swing in opposite direction (trend change)
    """

    def __init__(self, lookback_period: int = 10, min_structure_size: float = 0.001):
        """
        Initialize MarketStructureAnalyzer

        Args:
            lookback_period: Number of bars to look back for structure analysis
            min_structure_size: Minimum price movement to consider as structure
        """
        self.lookback_period = lookback_period
        self.min_structure_size = min_structure_size
        self.logger = logging.getLogger(f"{__name__}.MarketStructureAnalyzer")
        self.current_trend = TrendDirection.NEUTRAL
        self.last_swing_high = None
        self.last_swing_low = None

    def analyze_structure(
        self, data: pd.DataFrame, swing_points: List[SwingPoint]
    ) -> List[MarketStructure]:
        """
        Analyze market structure changes from swing points

        Args:
            data: DataFrame with OHLC data
            swing_points: List of swing points

        Returns:
            List of MarketStructure objects
        """
        if len(swing_points) < 2:
            return []

        structures = []
        current_trend = TrendDirection.NEUTRAL

        # Sort swing points by index
        sorted_swings = sorted(swing_points, key=lambda x: x.index)

        for i in range(1, len(sorted_swings)):
            current_swing = sorted_swings[i]
            previous_swing = sorted_swings[i - 1]

            # Determine structure break
            structure = self._analyze_swing_break(
                data, current_swing, previous_swing, current_trend, sorted_swings[:i]
            )

            if structure:
                structures.append(structure)
                current_trend = structure.direction

        return structures

    def _analyze_swing_break(
        self,
        data: pd.DataFrame,
        current_swing: SwingPoint,
        previous_swing: SwingPoint,
        current_trend: TrendDirection,
        historical_swings: List[SwingPoint],
    ) -> Optional[MarketStructure]:
        """
        Analyze if current swing breaks structure

        Args:
            data: OHLC DataFrame
            current_swing: Current swing point
            previous_swing: Previous swing point
            current_trend: Current market trend
            historical_swings: All previous swing points

        Returns:
            MarketStructure object if structure break detected
        """
        try:
            # Find relevant structural levels to break
            if current_swing.is_high:
                # For swing highs, look for previous swing highs to break
                relevant_swings = [s for s in historical_swings if s.is_high]
                if not relevant_swings:
                    return None

                # Find the most recent significant swing high
                last_high = max(relevant_swings, key=lambda x: x.price)

                # Check if current high breaks previous high
                if current_swing.price > last_high.price + self.min_structure_size:
                    # Determine if BOS or CHoCH
                    if (
                        current_trend == TrendDirection.BULLISH
                        or current_trend == TrendDirection.NEUTRAL
                    ):
                        structure_type = "BOS"  # Break of Structure (continuation)
                        direction = TrendDirection.BULLISH
                    else:
                        structure_type = "CHoCH"  # Change of Character (reversal)
                        direction = TrendDirection.BULLISH

                    return MarketStructure(
                        structure_type=structure_type,
                        direction=direction,
                        break_level=last_high.price,
                        break_index=current_swing.index,
                        timestamp=current_swing.timestamp,
                        strength=self._calculate_structure_strength(
                            current_swing, last_high, data
                        ),
                    )

            else:
                # For swing lows, look for previous swing lows to break
                relevant_swings = [s for s in historical_swings if not s.is_high]
                if not relevant_swings:
                    return None

                # Find the most recent significant swing low
                last_low = min(relevant_swings, key=lambda x: x.price)

                # Check if current low breaks previous low
                if current_swing.price < last_low.price - self.min_structure_size:
                    # Determine if BOS or CHoCH
                    if (
                        current_trend == TrendDirection.BEARISH
                        or current_trend == TrendDirection.NEUTRAL
                    ):
                        structure_type = "BOS"  # Break of Structure (continuation)
                        direction = TrendDirection.BEARISH
                    else:
                        structure_type = "CHoCH"  # Change of Character (reversal)
                        direction = TrendDirection.BEARISH

                    return MarketStructure(
                        structure_type=structure_type,
                        direction=direction,
                        break_level=last_low.price,
                        break_index=current_swing.index,
                        timestamp=current_swing.timestamp,
                        strength=self._calculate_structure_strength(
                            current_swing, last_low, data
                        ),
                    )

        except Exception as e:
            self.logger.error(f"Error analyzing swing break: {e}")

        return None

    def _calculate_structure_strength(
        self, breaking_swing: SwingPoint, broken_swing: SwingPoint, data: pd.DataFrame
    ) -> float:
        """
        Calculate the strength of structure break

        Args:
            breaking_swing: Swing that breaks structure
            broken_swing: Swing that was broken
            data: OHLC DataFrame

        Returns:
            Strength score between 0 and 1
        """
        try:
            strength_factors = []

            # Factor 1: Size of the break
            if breaking_swing.is_high:
                break_size = breaking_swing.price - broken_swing.price
            else:
                break_size = broken_swing.price - breaking_swing.price

            # Normalize break size against recent price movements
            recent_range = self._get_recent_price_range(data, broken_swing.index)
            if recent_range > 0:
                size_factor = min(break_size / recent_range, 2.0) / 2.0
                strength_factors.append(size_factor)

            # Factor 2: Volume confirmation (if available)
            if "volume" in data.columns:
                breaking_volume = data.iloc[breaking_swing.index]["volume"]
                avg_volume = data.iloc[
                    max(0, breaking_swing.index - 20) : breaking_swing.index
                ]["volume"].mean()
                if avg_volume > 0:
                    volume_factor = min(breaking_volume / avg_volume, 3.0) / 3.0
                    strength_factors.append(volume_factor)

            # Factor 3: Time between swings (freshness)
            time_diff = breaking_swing.index - broken_swing.index
            if time_diff > 0:
                # Shorter time = higher strength (more immediate break)
                time_factor = max(0, 1 - (time_diff / 100.0))  # Normalize to 100 bars
                strength_factors.append(time_factor)

            # Factor 4: Confirmation bars
            confirmation_factor = min(breaking_swing.confirmation_bars / 5.0, 1.0)
            strength_factors.append(confirmation_factor)

            return np.mean(strength_factors) if strength_factors else 0.5

        except Exception as e:
            self.logger.error(f"Error calculating structure strength: {e}")
            return 0.5

    def _get_recent_price_range(
        self, data: pd.DataFrame, index: int, lookback: int = 20
    ) -> float:
        """
        Get recent price range for normalization

        Args:
            data: OHLC DataFrame
            index: Current index
            lookback: Number of bars to look back

        Returns:
            Price range over lookback period
        """
        try:
            start_idx = max(0, index - lookback)
            recent_data = data.iloc[start_idx : index + 1]

            if len(recent_data) == 0:
                return 0.0

            high_range = recent_data["high"].max()
            low_range = recent_data["low"].min()
            return high_range - low_range

        except Exception as e:
            self.logger.error(f"Error getting recent price range: {e}")
            return 0.0

    def detect_structural_breaks(
        self, data: pd.DataFrame
    ) -> Dict[str, List[MarketStructure]]:
        """
        Detect structural breaks using price action

        Args:
            data: DataFrame with OHLC data

        Returns:
            Dictionary with 'BOS' and 'CHoCH' lists
        """
        try:
            # First, get swing points using OrderBlockDetector
            ob_detector = OrderBlockDetector(
                lookback_period=self.lookback_period, min_confirmation_bars=3
            )
            swing_points = ob_detector.find_swing_points(data)

            # Analyze structure from swing points
            structures = self.analyze_structure(data, swing_points)

            # Separate BOS and CHoCH
            bos_structures = [s for s in structures if s.structure_type == "BOS"]
            choch_structures = [s for s in structures if s.structure_type == "CHoCH"]

            return {
                "BOS": bos_structures,
                "CHoCH": choch_structures,
                "all_structures": structures,
            }

        except Exception as e:
            self.logger.error(f"Error detecting structural breaks: {e}")
            return {"BOS": [], "CHoCH": [], "all_structures": []}

    def get_current_trend(self, structures: List[MarketStructure]) -> TrendDirection:
        """
        Get current market trend based on recent structures

        Args:
            structures: List of market structures

        Returns:
            Current trend direction
        """
        if not structures:
            return TrendDirection.NEUTRAL

        # Get the most recent structure
        recent_structure = max(structures, key=lambda x: x.break_index)
        return recent_structure.direction

    def analyze(
        self, data: pd.DataFrame
    ) -> Dict[str, Union[List[MarketStructure], TrendDirection]]:
        """
        Main analysis method for market structure

        Args:
            data: DataFrame with OHLC data

        Returns:
            Dictionary containing structure analysis results
        """
        try:
            self.logger.info(f"Analyzing market structure for {len(data)} bars")

            # Detect structural breaks
            structure_results = self.detect_structural_breaks(data)

            # Determine current trend
            current_trend = self.get_current_trend(structure_results["all_structures"])

            # Compile results
            results = {
                "BOS": structure_results["BOS"],
                "CHoCH": structure_results["CHoCH"],
                "all_structures": structure_results["all_structures"],
                "current_trend": current_trend,
                "total_bos": len(structure_results["BOS"]),
                "total_choch": len(structure_results["CHoCH"]),
            }

            self.logger.info(
                f"Found {results['total_bos']} BOS and {results['total_choch']} CHoCH. "
                f"Current trend: {current_trend.value}"
            )

            return results

        except Exception as e:
            self.logger.error(f"Error in market structure analysis: {e}")
            return {
                "BOS": [],
                "CHoCH": [],
                "all_structures": [],
                "current_trend": TrendDirection.NEUTRAL,
                "total_bos": 0,
                "total_choch": 0,
            }


@dataclass
class TimeFrameData:
    """Represents data for a specific timeframe"""

    timeframe: str
    data: pd.DataFrame
    last_update: pd.Timestamp
    analysis_results: Dict = None


class TimeFrameManager:
    """
    Manages multi-timeframe analysis and data synchronization

    Handles different timeframes simultaneously and provides synchronized analysis
    across multiple time horizons for higher timeframe bias confirmation.
    """

    def __init__(self, supported_timeframes: Optional[List[str]] = None):
        """
        Initialize TimeFrameManager

        Args:
            supported_timeframes: List of supported timeframes (e.g., ['1m', '5m', '15m', '1h', '4h', '1d'])
        """
        self.supported_timeframes = supported_timeframes or [
            "1m",
            "5m",
            "15m",
            "1h",
            "4h",
            "1d",
        ]
        self.timeframe_data: Dict[str, TimeFrameData] = {}
        self.logger = logging.getLogger(f"{__name__}.TimeFrameManager")

        # ICT analyzers for each timeframe
        self.analyzers = {
            "order_block": OrderBlockDetector(),
            "fair_value_gap": FairValueGapAnalyzer(),
            "market_structure": MarketStructureAnalyzer(),
        }

    def add_timeframe_data(self, timeframe: str, data: pd.DataFrame) -> bool:
        """
        Add or update data for a specific timeframe

        Args:
            timeframe: Timeframe identifier (e.g., '1h', '4h')
            data: OHLC DataFrame for the timeframe

        Returns:
            True if data was added successfully
        """
        try:
            if timeframe not in self.supported_timeframes:
                self.logger.warning(f"Timeframe {timeframe} not in supported list")
                return False

            # Validate data format
            required_columns = ["open", "high", "low", "close"]
            if not all(col in data.columns for col in required_columns):
                self.logger.error(f"Data for {timeframe} missing required OHLC columns")
                return False

            # Store timeframe data
            self.timeframe_data[timeframe] = TimeFrameData(
                timeframe=timeframe,
                data=data.copy(),
                last_update=pd.Timestamp.now(),
                analysis_results={},
            )

            self.logger.info(f"Added {len(data)} bars for timeframe {timeframe}")
            return True

        except Exception as e:
            self.logger.error(f"Error adding timeframe data for {timeframe}: {e}")
            return False

    def get_timeframe_data(self, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Get data for specific timeframe

        Args:
            timeframe: Timeframe identifier

        Returns:
            DataFrame with OHLC data or None
        """
        if timeframe in self.timeframe_data:
            return self.timeframe_data[timeframe].data
        return None

    def analyze_timeframe(
        self, timeframe: str, analyzer_types: Optional[List[str]] = None
    ) -> Dict:
        """
        Analyze specific timeframe with ICT patterns

        Args:
            timeframe: Timeframe to analyze
            analyzer_types: List of analyzer types to run (default: all)

        Returns:
            Dictionary with analysis results
        """
        if timeframe not in self.timeframe_data:
            self.logger.error(f"No data available for timeframe {timeframe}")
            return {}

        analyzer_types = analyzer_types or list(self.analyzers.keys())
        data = self.timeframe_data[timeframe].data
        results = {}

        try:
            for analyzer_type in analyzer_types:
                if analyzer_type in self.analyzers:
                    self.logger.info(f"Running {analyzer_type} analysis on {timeframe}")

                    analyzer = self.analyzers[analyzer_type]
                    analysis_result = analyzer.analyze(data)
                    results[analyzer_type] = analysis_result

            # Store results
            self.timeframe_data[timeframe].analysis_results = results
            self.logger.info(f"Completed analysis for {timeframe}")

            return results

        except Exception as e:
            self.logger.error(f"Error analyzing timeframe {timeframe}: {e}")
            return {}

    def analyze_all_timeframes(
        self, analyzer_types: Optional[List[str]] = None
    ) -> Dict[str, Dict]:
        """
        Analyze all available timeframes

        Args:
            analyzer_types: List of analyzer types to run

        Returns:
            Dictionary with results for each timeframe
        """
        all_results = {}

        for timeframe in self.timeframe_data.keys():
            results = self.analyze_timeframe(timeframe, analyzer_types)
            if results:
                all_results[timeframe] = results

        return all_results

    def get_higher_timeframe_bias(
        self, current_timeframe: str
    ) -> Dict[str, TrendDirection]:
        """
        Get bias from higher timeframes for confluence

        Args:
            current_timeframe: Current timeframe being analyzed

        Returns:
            Dictionary with bias from each higher timeframe
        """
        timeframe_hierarchy = {
            "1m": 0,
            "5m": 1,
            "15m": 2,
            "30m": 3,
            "1h": 4,
            "4h": 5,
            "12h": 6,
            "1d": 7,
            "1w": 8,
            "1M": 9,
        }

        current_rank = timeframe_hierarchy.get(current_timeframe, 0)
        higher_timeframes = [
            tf
            for tf, rank in timeframe_hierarchy.items()
            if rank > current_rank and tf in self.timeframe_data
        ]

        bias_results = {}

        for htf in higher_timeframes:
            htf_data = self.timeframe_data[htf]
            if (
                htf_data.analysis_results
                and "market_structure" in htf_data.analysis_results
            ):
                structure_results = htf_data.analysis_results["market_structure"]
                bias_results[htf] = structure_results.get(
                    "current_trend", TrendDirection.NEUTRAL
                )
            else:
                # Analyze if not done yet
                structure_analysis = self.analyzers["market_structure"].analyze(
                    htf_data.data
                )
                bias_results[htf] = structure_analysis.get(
                    "current_trend", TrendDirection.NEUTRAL
                )

        return bias_results

    def check_confluence(self, timeframe: str, pattern_type: str) -> Dict[str, any]:
        """
        Check confluence across multiple timeframes

        Args:
            timeframe: Primary timeframe
            pattern_type: Type of pattern to check confluence for

        Returns:
            Dictionary with confluence analysis
        """
        confluence_results = {
            "primary_timeframe": timeframe,
            "pattern_type": pattern_type,
            "confluence_score": 0.0,
            "supporting_timeframes": [],
            "conflicting_timeframes": [],
            "higher_timeframe_bias": {},
        }

        try:
            # Get higher timeframe bias
            htf_bias = self.get_higher_timeframe_bias(timeframe)
            confluence_results["higher_timeframe_bias"] = htf_bias

            # Get primary timeframe trend
            primary_data = self.timeframe_data.get(timeframe)
            if (
                not primary_data
                or "market_structure" not in primary_data.analysis_results
            ):
                return confluence_results

            primary_trend = primary_data.analysis_results["market_structure"].get(
                "current_trend", TrendDirection.NEUTRAL
            )

            # Check confluence
            supporting_count = 0
            conflicting_count = 0

            for htf, bias in htf_bias.items():
                if bias == primary_trend:
                    confluence_results["supporting_timeframes"].append(htf)
                    supporting_count += 1
                elif bias != TrendDirection.NEUTRAL:
                    confluence_results["conflicting_timeframes"].append(htf)
                    conflicting_count += 1

            # Calculate confluence score
            total_timeframes = len(htf_bias)
            if total_timeframes > 0:
                confluence_results["confluence_score"] = (
                    supporting_count / total_timeframes
                )

            self.logger.info(
                f"Confluence for {timeframe} {pattern_type}: "
                f"{confluence_results['confluence_score']:.2f} "
                f"({supporting_count}/{total_timeframes} supporting)"
            )

        except Exception as e:
            self.logger.error(f"Error checking confluence: {e}")

        return confluence_results

    def get_synchronized_signals(self, min_confluence_score: float = 0.6) -> List[Dict]:
        """
        Get trading signals that have confluence across timeframes

        Args:
            min_confluence_score: Minimum confluence score required

        Returns:
            List of synchronized signals
        """
        signals = []

        try:
            for timeframe in self.timeframe_data.keys():
                tf_data = self.timeframe_data[timeframe]

                if not tf_data.analysis_results:
                    continue

                # Check market structure signals
                if "market_structure" in tf_data.analysis_results:
                    structure_results = tf_data.analysis_results["market_structure"]

                    # Check for recent BOS or CHoCH
                    for structure_type in ["BOS", "CHoCH"]:
                        structures = structure_results.get(structure_type, [])

                        if structures:
                            # Get most recent structure
                            recent_structure = max(
                                structures, key=lambda x: x.break_index
                            )

                            # Check confluence
                            confluence = self.check_confluence(
                                timeframe, structure_type
                            )

                            if confluence["confluence_score"] >= min_confluence_score:
                                signals.append(
                                    {
                                        "timeframe": timeframe,
                                        "signal_type": structure_type,
                                        "direction": recent_structure.direction.value,
                                        "confluence_score": confluence[
                                            "confluence_score"
                                        ],
                                        "strength": recent_structure.strength,
                                        "supporting_timeframes": confluence[
                                            "supporting_timeframes"
                                        ],
                                        "timestamp": recent_structure.timestamp,
                                    }
                                )

            # Sort by confluence score
            signals.sort(key=lambda x: x["confluence_score"], reverse=True)

        except Exception as e:
            self.logger.error(f"Error getting synchronized signals: {e}")

        return signals

    def update_realtime_data(self, timeframe: str, new_candle: Dict) -> bool:
        """
        Update with new realtime candle data

        Args:
            timeframe: Timeframe to update
            new_candle: Dictionary with OHLC data

        Returns:
            True if update successful
        """
        try:
            if timeframe not in self.timeframe_data:
                self.logger.error(f"Timeframe {timeframe} not initialized")
                return False

            # Convert new candle to DataFrame row
            new_row = pd.DataFrame([new_candle])

            # Append to existing data
            self.timeframe_data[timeframe].data = pd.concat(
                [self.timeframe_data[timeframe].data, new_row], ignore_index=False
            )

            # Update timestamp
            self.timeframe_data[timeframe].last_update = pd.Timestamp.now()

            return True

        except Exception as e:
            self.logger.error(f"Error updating realtime data for {timeframe}: {e}")
            return False

    def get_memory_usage(self) -> Dict[str, int]:
        """
        Get memory usage statistics

        Returns:
            Dictionary with memory usage per timeframe
        """
        usage = {}
        for timeframe, tf_data in self.timeframe_data.items():
            usage[timeframe] = len(tf_data.data) if tf_data.data is not None else 0
        return usage

    def cleanup_old_data(self, max_bars_per_timeframe: int = 1000):
        """
        Clean up old data to manage memory usage

        Args:
            max_bars_per_timeframe: Maximum bars to keep per timeframe
        """
        for timeframe, tf_data in self.timeframe_data.items():
            if len(tf_data.data) > max_bars_per_timeframe:
                # Keep only the most recent bars
                tf_data.data = tf_data.data.tail(max_bars_per_timeframe).copy()
                self.logger.info(
                    f"Cleaned up {timeframe} data to {max_bars_per_timeframe} bars"
                )

    def analyze(self, analyzer_types: Optional[List[str]] = None) -> Dict[str, Dict]:
        """
        Main analysis method for multi-timeframe analysis

        Args:
            analyzer_types: List of analyzer types to run

        Returns:
            Complete multi-timeframe analysis results
        """
        try:
            self.logger.info(
                f"Starting multi-timeframe analysis for {len(self.timeframe_data)} timeframes"
            )

            # Analyze all timeframes
            timeframe_results = self.analyze_all_timeframes(analyzer_types)

            # Get synchronized signals
            synchronized_signals = self.get_synchronized_signals()

            # Compile final results
            results = {
                "timeframe_results": timeframe_results,
                "synchronized_signals": synchronized_signals,
                "memory_usage": self.get_memory_usage(),
                "analysis_timestamp": pd.Timestamp.now(),
                "active_timeframes": list(self.timeframe_data.keys()),
            }

            self.logger.info(
                f"Multi-timeframe analysis complete. "
                f"Found {len(synchronized_signals)} synchronized signals"
            )

            return results

        except Exception as e:
            self.logger.error(f"Error in multi-timeframe analysis: {e}")
            return {
                "timeframe_results": {},
                "synchronized_signals": [],
                "memory_usage": self.get_memory_usage(),
                "analysis_timestamp": pd.Timestamp.now(),
                "active_timeframes": list(self.timeframe_data.keys()),
            }


@dataclass
class PatternValidationResult:
    """Result of pattern validation analysis"""

    pattern_id: str
    pattern_type: PatternType
    confidence_score: float  # 0.0 to 1.0
    reliability_grade: str  # A, B, C, D, F
    success_probability: float  # Historical success rate
    risk_reward_ratio: float
    backtest_performance: Dict[str, float]
    validation_timestamp: pd.Timestamp
    is_valid: bool
    warnings: List[str] = None


@dataclass
class BacktestResult:
    """Individual backtest result for a pattern"""

    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    profit_loss: float
    profit_loss_pct: float
    duration_bars: int
    was_successful: bool
    max_favorable_excursion: float
    max_adverse_excursion: float


class PatternValidationEngine:
    """
    Pattern Validation System for ICT Analysis

    Validates detected ICT patterns and evaluates their reliability with:
    - Pattern confidence scoring algorithm
    - Historical backtest-based validation
    - Pattern success rate statistical analysis
    - False signal filtering logic
    - Pattern quality grading system
    """

    def __init__(
        self,
        lookback_bars: int = 500,
        min_sample_size: int = 10,
        confidence_threshold: float = 0.6,
    ):
        """
        Initialize Pattern Validation Engine

        Args:
            lookback_bars: Number of historical bars for backtesting
            min_sample_size: Minimum number of patterns needed for statistical significance
            confidence_threshold: Minimum confidence score for pattern validity
        """
        self.lookback_bars = lookback_bars
        self.min_sample_size = min_sample_size
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger(__name__)

        # Performance tracking
        self.pattern_performance_history: Dict[str, List[BacktestResult]] = {}
        self.reliability_cache: Dict[str, float] = {}

    def validate_pattern(
        self,
        pattern: Union[OrderBlock, FairValueGap, MarketStructure],
        data: pd.DataFrame,
        pattern_type: PatternType,
    ) -> PatternValidationResult:
        """
        Validate a single pattern using comprehensive analysis

        Args:
            pattern: Pattern object to validate
            data: Historical OHLC data
            pattern_type: Type of pattern being validated

        Returns:
            PatternValidationResult with validation metrics
        """
        try:
            pattern_id = self._generate_pattern_id(pattern, pattern_type)

            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(
                pattern, data, pattern_type
            )

            # Perform historical backtest
            backtest_results = self._backtest_pattern(pattern, data, pattern_type)

            # Calculate success probability
            success_probability = self._calculate_success_probability(backtest_results)

            # Calculate risk-reward ratio
            risk_reward_ratio = self._calculate_risk_reward_ratio(backtest_results)

            # Generate reliability grade
            reliability_grade = self._generate_reliability_grade(
                confidence_score, success_probability, risk_reward_ratio
            )

            # Compile backtest performance metrics
            backtest_performance = self._compile_backtest_metrics(backtest_results)

            # Generate warnings
            warnings = self._generate_validation_warnings(
                pattern, backtest_results, confidence_score
            )

            # Determine overall validity
            is_valid = (
                confidence_score >= self.confidence_threshold
                and len(backtest_results) >= self.min_sample_size
                and success_probability >= 0.4
            )

            return PatternValidationResult(
                pattern_id=pattern_id,
                pattern_type=pattern_type,
                confidence_score=confidence_score,
                reliability_grade=reliability_grade,
                success_probability=success_probability,
                risk_reward_ratio=risk_reward_ratio,
                backtest_performance=backtest_performance,
                validation_timestamp=pd.Timestamp.now(),
                is_valid=is_valid,
                warnings=warnings,
            )

        except Exception as e:
            self.logger.error(f"Error validating pattern: {e}")
            return self._create_error_result(pattern_type)

    def validate_pattern_batch(
        self, patterns: List[Tuple], data: pd.DataFrame
    ) -> List[PatternValidationResult]:
        """
        Validate multiple patterns in batch for efficiency

        Args:
            patterns: List of (pattern, pattern_type) tuples
            data: Historical OHLC data

        Returns:
            List of validation results
        """
        results = []

        try:
            self.logger.info(f"Starting batch validation of {len(patterns)} patterns")

            for pattern, pattern_type in patterns:
                result = self.validate_pattern(pattern, data, pattern_type)
                results.append(result)

            # Calculate batch statistics
            valid_patterns = sum(1 for r in results if r.is_valid)
            avg_confidence = np.mean([r.confidence_score for r in results])

            self.logger.info(
                f"Batch validation complete. {valid_patterns}/{len(patterns)} valid patterns. "
                f"Average confidence: {avg_confidence:.3f}"
            )

        except Exception as e:
            self.logger.error(f"Error in batch validation: {e}")

        return results

    def _calculate_confidence_score(
        self,
        pattern: Union[OrderBlock, FairValueGap, MarketStructure],
        data: pd.DataFrame,
        pattern_type: PatternType,
    ) -> float:
        """Calculate pattern confidence score based on multiple factors"""
        try:
            confidence_factors = []

            if pattern_type == PatternType.ORDER_BLOCK:
                # Order Block specific confidence factors
                ob = pattern

                # Volume confirmation
                volume_score = self._assess_volume_confirmation(ob, data)
                confidence_factors.append(("volume", volume_score, 0.25))

                # Price action quality
                price_action_score = self._assess_price_action_quality(ob, data)
                confidence_factors.append(("price_action", price_action_score, 0.3))

                # Time proximity
                time_score = self._assess_time_proximity(ob, data)
                confidence_factors.append(("time", time_score, 0.2))

                # Pattern size relative to ATR
                size_score = self._assess_pattern_size(ob, data)
                confidence_factors.append(("size", size_score, 0.25))

            elif pattern_type == PatternType.FAIR_VALUE_GAP:
                # Fair Value Gap specific confidence factors
                fvg = pattern

                # Gap size significance
                gap_size_score = self._assess_gap_size_significance(fvg, data)
                confidence_factors.append(("gap_size", gap_size_score, 0.3))

                # Market context
                context_score = self._assess_market_context(fvg, data)
                confidence_factors.append(("context", context_score, 0.25))

                # Imbalance quality
                imbalance_score = self._assess_imbalance_quality(fvg, data)
                confidence_factors.append(("imbalance", imbalance_score, 0.25))

                # Unfilled duration
                duration_score = self._assess_unfilled_duration(fvg, data)
                confidence_factors.append(("duration", duration_score, 0.2))

            elif pattern_type in [
                PatternType.BREAK_OF_STRUCTURE,
                PatternType.CHANGE_OF_CHARACTER,
            ]:
                # Market Structure specific confidence factors
                ms = pattern

                # Structure significance
                structure_score = self._assess_structure_significance(ms, data)
                confidence_factors.append(("structure", structure_score, 0.35))

                # Momentum confirmation
                momentum_score = self._assess_momentum_confirmation(ms, data)
                confidence_factors.append(("momentum", momentum_score, 0.3))

                # Multi-timeframe alignment
                mtf_score = self._assess_mtf_alignment(ms, data)
                confidence_factors.append(("mtf_alignment", mtf_score, 0.35))

            # Calculate weighted confidence score
            total_score = 0.0
            total_weight = 0.0

            for factor_name, score, weight in confidence_factors:
                total_score += score * weight
                total_weight += weight

            confidence_score = total_score / total_weight if total_weight > 0 else 0.0

            # Apply confidence bounds
            confidence_score = max(0.0, min(1.0, confidence_score))

            return confidence_score

        except Exception as e:
            self.logger.error(f"Error calculating confidence score: {e}")
            return 0.0

    def _backtest_pattern(
        self,
        pattern: Union[OrderBlock, FairValueGap, MarketStructure],
        data: pd.DataFrame,
        pattern_type: PatternType,
    ) -> List[BacktestResult]:
        """Perform historical backtest on similar patterns"""
        try:
            backtest_results = []

            # Find historical occurrences of similar patterns
            similar_patterns = self._find_similar_historical_patterns(
                pattern, data, pattern_type
            )

            for hist_pattern in similar_patterns:
                # Simulate trade execution
                entry_result = self._simulate_pattern_entry(
                    hist_pattern, data, pattern_type
                )
                if entry_result:
                    backtest_results.append(entry_result)

            return backtest_results

        except Exception as e:
            self.logger.error(f"Error in pattern backtest: {e}")
            return []

    def _calculate_success_probability(
        self, backtest_results: List[BacktestResult]
    ) -> float:
        """Calculate success probability from backtest results"""
        if not backtest_results:
            return 0.0

        successful_trades = sum(
            1 for result in backtest_results if result.was_successful
        )
        return successful_trades / len(backtest_results)

    def _calculate_risk_reward_ratio(
        self, backtest_results: List[BacktestResult]
    ) -> float:
        """Calculate average risk-reward ratio from backtest results"""
        if not backtest_results:
            return 0.0

        winning_trades = [r for r in backtest_results if r.was_successful]
        losing_trades = [r for r in backtest_results if not r.was_successful]

        if not winning_trades or not losing_trades:
            return 0.0

        avg_win = np.mean([r.profit_loss for r in winning_trades])
        avg_loss = np.mean([abs(r.profit_loss) for r in losing_trades])

        return avg_win / avg_loss if avg_loss > 0 else 0.0

    def _generate_reliability_grade(
        self,
        confidence_score: float,
        success_probability: float,
        risk_reward_ratio: float,
    ) -> str:
        """Generate reliability grade (A-F) based on metrics"""
        # Weighted scoring
        composite_score = (
            confidence_score * 0.4
            + success_probability * 0.4
            + min(risk_reward_ratio / 2.0, 1.0) * 0.2  # Cap R:R contribution
        )

        if composite_score >= 0.85:
            return "A"
        elif composite_score >= 0.75:
            return "B"
        elif composite_score >= 0.60:
            return "C"
        elif composite_score >= 0.45:
            return "D"
        else:
            return "F"

    def _compile_backtest_metrics(
        self, backtest_results: List[BacktestResult]
    ) -> Dict[str, float]:
        """Compile comprehensive backtest performance metrics"""
        if not backtest_results:
            return {}

        profits = [r.profit_loss for r in backtest_results]
        win_rate = sum(1 for r in backtest_results if r.was_successful) / len(
            backtest_results
        )

        return {
            "total_trades": len(backtest_results),
            "win_rate": win_rate,
            "total_profit": sum(profits),
            "average_profit": np.mean(profits),
            "profit_factor": self._calculate_profit_factor(backtest_results),
            "max_drawdown": self._calculate_max_drawdown(backtest_results),
            "sharpe_ratio": self._calculate_sharpe_ratio(profits),
            "average_duration_bars": np.mean(
                [r.duration_bars for r in backtest_results]
            ),
        }

    def _generate_validation_warnings(
        self,
        pattern: Union[OrderBlock, FairValueGap, MarketStructure],
        backtest_results: List[BacktestResult],
        confidence_score: float,
    ) -> List[str]:
        """Generate validation warnings for pattern quality issues"""
        warnings = []

        if confidence_score < self.confidence_threshold:
            warnings.append(f"Low confidence score: {confidence_score:.3f}")

        if len(backtest_results) < self.min_sample_size:
            warnings.append(
                f"Insufficient historical data: {len(backtest_results)} samples"
            )

        if backtest_results:
            win_rate = sum(1 for r in backtest_results if r.was_successful) / len(
                backtest_results
            )
            if win_rate < 0.4:
                warnings.append(f"Low historical win rate: {win_rate:.1%}")

        return warnings

    # Helper methods for confidence scoring
    def _assess_volume_confirmation(
        self, pattern: OrderBlock, data: pd.DataFrame
    ) -> float:
        """Assess volume confirmation for order blocks"""
        try:
            pattern_data = data.iloc[pattern.start_index : pattern.end_index + 1]
            if "volume" not in pattern_data.columns:
                return 0.5  # Neutral score if no volume data

            avg_volume = data["volume"].rolling(20).mean().iloc[pattern.end_index]
            pattern_volume = pattern_data["volume"].mean()

            volume_ratio = pattern_volume / avg_volume if avg_volume > 0 else 1.0
            return min(volume_ratio / 2.0, 1.0)  # Cap at 1.0

        except Exception:
            return 0.5

    def _assess_price_action_quality(
        self, pattern: OrderBlock, data: pd.DataFrame
    ) -> float:
        """Assess price action quality for order blocks"""
        try:
            pattern_data = data.iloc[pattern.start_index : pattern.end_index + 1]

            # Check for clean rejection or strong move from the zone
            price_range = pattern_data["high"].max() - pattern_data["low"].min()
            atr = (
                data["close"]
                .rolling(14)
                .apply(lambda x: np.mean(np.abs(x.diff())))
                .iloc[pattern.end_index]
            )

            if atr == 0:
                return 0.5

            relative_size = price_range / atr
            return min(relative_size / 3.0, 1.0)  # Normalize to 0-1

        except Exception:
            return 0.5

    def _assess_time_proximity(self, pattern: OrderBlock, data: pd.DataFrame) -> float:
        """Assess time proximity factor for order blocks"""
        try:
            current_index = len(data) - 1
            bars_since_pattern = current_index - pattern.end_index

            # Fresher patterns get higher scores
            if bars_since_pattern <= 10:
                return 1.0
            elif bars_since_pattern <= 50:
                return 0.8
            elif bars_since_pattern <= 100:
                return 0.6
            else:
                return 0.4

        except Exception:
            return 0.5

    def _assess_pattern_size(self, pattern: OrderBlock, data: pd.DataFrame) -> float:
        """Assess pattern size relative to ATR"""
        try:
            pattern_size = abs(pattern.high_price - pattern.low_price)
            atr = (
                data["close"]
                .rolling(14)
                .apply(lambda x: np.mean(np.abs(x.diff())))
                .iloc[pattern.end_index]
            )

            if atr == 0:
                return 0.5

            size_ratio = pattern_size / atr

            # Optimal size is 1-3x ATR
            if 1.0 <= size_ratio <= 3.0:
                return 1.0
            elif 0.5 <= size_ratio < 1.0 or 3.0 < size_ratio <= 5.0:
                return 0.7
            else:
                return 0.3

        except Exception:
            return 0.5

    def _assess_gap_size_significance(
        self, pattern: FairValueGap, data: pd.DataFrame
    ) -> float:
        """Assess Fair Value Gap size significance"""
        try:
            atr = (
                data["close"]
                .rolling(14)
                .apply(lambda x: np.mean(np.abs(x.diff())))
                .iloc[pattern.end_index]
            )

            if atr == 0:
                return 0.5

            gap_ratio = pattern.gap_size / atr

            # Significant gaps are 0.5-2x ATR
            if 0.5 <= gap_ratio <= 2.0:
                return 1.0
            elif 0.2 <= gap_ratio < 0.5 or 2.0 < gap_ratio <= 4.0:
                return 0.6
            else:
                return 0.2

        except Exception:
            return 0.5

    def _assess_market_context(
        self, pattern: FairValueGap, data: pd.DataFrame
    ) -> float:
        """Assess market context for Fair Value Gap"""
        try:
            # Check if gap aligns with trend
            trend_lookback = 20
            start_idx = max(0, pattern.start_index - trend_lookback)
            trend_data = data.iloc[start_idx : pattern.start_index]

            if len(trend_data) < 5:
                return 0.5

            trend_slope = (
                trend_data["close"].iloc[-1] - trend_data["close"].iloc[0]
            ) / len(trend_data)

            gap_bullish = pattern.direction == TrendDirection.BULLISH
            trend_bullish = trend_slope > 0

            # Gap aligning with trend gets higher score
            return 0.8 if gap_bullish == trend_bullish else 0.4

        except Exception:
            return 0.5

    def _assess_imbalance_quality(
        self, pattern: FairValueGap, data: pd.DataFrame
    ) -> float:
        """Assess imbalance quality for Fair Value Gap"""
        try:
            gap_data = data.iloc[pattern.start_index : pattern.end_index + 1]

            # Check for clean gap with no overlap
            if pattern.direction == TrendDirection.BULLISH:
                max_low = gap_data["low"].max()
                gap_overlap = max(0, max_low - pattern.gap_low)
            else:
                min_high = gap_data["high"].min()
                gap_overlap = max(0, pattern.gap_high - min_high)

            overlap_ratio = (
                gap_overlap / pattern.gap_size if pattern.gap_size > 0 else 0
            )
            return max(0, 1.0 - overlap_ratio)

        except Exception:
            return 0.5

    def _assess_unfilled_duration(
        self, pattern: FairValueGap, data: pd.DataFrame
    ) -> float:
        """Assess unfilled duration for Fair Value Gap"""
        try:
            current_index = len(data) - 1
            bars_unfilled = current_index - pattern.end_index

            # Longer unfilled duration indicates stronger imbalance
            if bars_unfilled >= 50:
                return 1.0
            elif bars_unfilled >= 20:
                return 0.8
            elif bars_unfilled >= 10:
                return 0.6
            else:
                return 0.4

        except Exception:
            return 0.5

    def _assess_structure_significance(
        self, pattern: MarketStructure, data: pd.DataFrame
    ) -> float:
        """Assess structure significance for market structure patterns"""
        try:
            # Assess the significance of the structural break
            structure_range = abs(
                pattern.break_price - pattern.previous_structure_price
            )
            atr = (
                data["close"]
                .rolling(14)
                .apply(lambda x: np.mean(np.abs(x.diff())))
                .iloc[pattern.end_index]
            )

            if atr == 0:
                return 0.5

            significance_ratio = structure_range / atr
            return min(significance_ratio / 2.0, 1.0)

        except Exception:
            return 0.5

    def _assess_momentum_confirmation(
        self, pattern: MarketStructure, data: pd.DataFrame
    ) -> float:
        """Assess momentum confirmation for market structure patterns"""
        try:
            # Calculate momentum around the break
            break_data = data.iloc[
                max(0, pattern.start_index - 5) : pattern.end_index + 5
            ]

            if len(break_data) < 5:
                return 0.5

            momentum = (
                break_data["close"].iloc[-1] - break_data["close"].iloc[0]
            ) / len(break_data)
            atr = (
                data["close"]
                .rolling(14)
                .apply(lambda x: np.mean(np.abs(x.diff())))
                .iloc[pattern.end_index]
            )

            if atr == 0:
                return 0.5

            momentum_strength = abs(momentum) / atr
            return min(momentum_strength, 1.0)

        except Exception:
            return 0.5

    def _assess_mtf_alignment(
        self, pattern: MarketStructure, data: pd.DataFrame
    ) -> float:
        """Assess multi-timeframe alignment for market structure patterns"""
        # Simplified assessment - would need multiple timeframe data for full implementation
        return 0.6  # Neutral score

    # Helper methods for backtesting
    def _find_similar_historical_patterns(
        self,
        pattern: Union[OrderBlock, FairValueGap, MarketStructure],
        data: pd.DataFrame,
        pattern_type: PatternType,
    ) -> List:
        """Find similar historical patterns for backtesting"""
        # Simplified implementation - would need full pattern matching logic
        similar_patterns = []

        try:
            lookback_data = (
                data.iloc[-self.lookback_bars :]
                if len(data) > self.lookback_bars
                else data
            )

            # For demonstration, return a few synthetic patterns
            for i in range(min(15, len(lookback_data) // 20)):
                idx = i * 20
                if idx < len(lookback_data) - 10:
                    similar_patterns.append(
                        {
                            "start_index": idx,
                            "end_index": idx + 5,
                            "pattern_data": lookback_data.iloc[idx : idx + 10],
                        }
                    )

        except Exception as e:
            self.logger.error(f"Error finding similar patterns: {e}")

        return similar_patterns[:10]  # Limit to 10 for performance

    def _simulate_pattern_entry(
        self, historical_pattern: Dict, data: pd.DataFrame, pattern_type: PatternType
    ) -> Optional[BacktestResult]:
        """Simulate trade entry and exit for historical pattern"""
        try:
            start_idx = historical_pattern["start_index"]
            pattern_data = historical_pattern["pattern_data"]

            if len(pattern_data) < 5:
                return None

            # Simple simulation logic
            entry_price = pattern_data["close"].iloc[-1]
            entry_time = pattern_data.index[-1]

            # Look for exit in next 20 bars
            exit_idx = start_idx + len(pattern_data) + 20
            if exit_idx >= len(data):
                return None

            exit_data = data.iloc[start_idx + len(pattern_data) : exit_idx]
            if len(exit_data) == 0:
                return None

            # Simple exit strategy: take profit at 2:1 or stop loss at 1:1
            atr = (
                data["close"]
                .rolling(14)
                .apply(lambda x: np.mean(np.abs(x.diff())))
                .iloc[start_idx]
            )
            if atr == 0:
                return None

            take_profit = entry_price + (2 * atr)  # 2:1 R:R
            stop_loss = entry_price - atr

            # Find exit point
            exit_price = None
            exit_time = None
            was_successful = False

            for i, (timestamp, row) in enumerate(exit_data.iterrows()):
                if row["high"] >= take_profit:
                    exit_price = take_profit
                    exit_time = timestamp
                    was_successful = True
                    break
                elif row["low"] <= stop_loss:
                    exit_price = stop_loss
                    exit_time = timestamp
                    was_successful = False
                    break

            if exit_price is None:
                # Exit at last available price
                exit_price = exit_data["close"].iloc[-1]
                exit_time = exit_data.index[-1]
                was_successful = exit_price > entry_price

            profit_loss = exit_price - entry_price
            profit_loss_pct = profit_loss / entry_price if entry_price > 0 else 0
            duration_bars = len(exit_data)

            # Calculate excursions
            mfe = exit_data["high"].max() - entry_price
            mae = entry_price - exit_data["low"].min()

            return BacktestResult(
                entry_time=entry_time,
                exit_time=exit_time,
                entry_price=entry_price,
                exit_price=exit_price,
                profit_loss=profit_loss,
                profit_loss_pct=profit_loss_pct,
                duration_bars=duration_bars,
                was_successful=was_successful,
                max_favorable_excursion=mfe,
                max_adverse_excursion=mae,
            )

        except Exception as e:
            self.logger.error(f"Error simulating pattern entry: {e}")
            return None

    def _calculate_profit_factor(self, backtest_results: List[BacktestResult]) -> float:
        """Calculate profit factor from backtest results"""
        winning_trades = [r for r in backtest_results if r.was_successful]
        losing_trades = [r for r in backtest_results if not r.was_successful]

        if not losing_trades:
            return float("inf") if winning_trades else 0.0

        total_wins = sum(r.profit_loss for r in winning_trades)
        total_losses = sum(abs(r.profit_loss) for r in losing_trades)

        return total_wins / total_losses if total_losses > 0 else 0.0

    def _calculate_max_drawdown(self, backtest_results: List[BacktestResult]) -> float:
        """Calculate maximum drawdown from backtest results"""
        if not backtest_results:
            return 0.0

        cumulative_returns = []
        running_total = 0.0

        for result in backtest_results:
            running_total += result.profit_loss_pct
            cumulative_returns.append(running_total)

        peak = cumulative_returns[0]
        max_drawdown = 0.0

        for ret in cumulative_returns:
            if ret > peak:
                peak = ret
            drawdown = peak - ret
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        return max_drawdown

    def _calculate_sharpe_ratio(self, profits: List[float]) -> float:
        """Calculate Sharpe ratio from profit list"""
        if not profits or len(profits) < 2:
            return 0.0

        mean_return = np.mean(profits)
        std_return = np.std(profits)

        return mean_return / std_return if std_return > 0 else 0.0

    def _generate_pattern_id(
        self,
        pattern: Union[OrderBlock, FairValueGap, MarketStructure],
        pattern_type: PatternType,
    ) -> str:
        """Generate unique pattern ID"""
        try:
            timestamp = getattr(pattern, "timestamp", pd.Timestamp.now())
            return f"{pattern_type.value}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        except Exception:
            return (
                f"{pattern_type.value}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
            )

    def _create_error_result(
        self, pattern_type: PatternType
    ) -> PatternValidationResult:
        """Create error result for failed validation"""
        return PatternValidationResult(
            pattern_id=f"error_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}",
            pattern_type=pattern_type,
            confidence_score=0.0,
            reliability_grade="F",
            success_probability=0.0,
            risk_reward_ratio=0.0,
            backtest_performance={},
            validation_timestamp=pd.Timestamp.now(),
            is_valid=False,
            warnings=["Validation failed due to error"],
        )
