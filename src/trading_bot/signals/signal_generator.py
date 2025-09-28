"""
SignalGenerator - Core ICT Signal Generation System

This module implements the core signal generation logic using ICT pattern combinations.
It takes inputs from Order Block detection, Fair Value Gap analysis, and Market Structure
analysis to generate high-confidence trading signals.

Key Features:
- ICT pattern combination analysis
- Signal type determination (Long/Short)
- Pattern priority weighting
- Multi-timeframe confluence consideration
- Signal confidence scoring
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

from ..analysis.ict_analyzer import (
    FairValueGap,
    MarketStructure,
    OrderBlock,
    PatternType,
    TrendDirection,
)
from ..core.events import SignalEvent, SignalType


class SignalDirection(Enum):
    """Signal direction enumeration"""

    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


class PatternCombination(Enum):
    """ICT Pattern combination types"""

    ORDER_BLOCK_ONLY = "order_block_only"
    FVG_ONLY = "fvg_only"
    STRUCTURE_ONLY = "structure_only"
    ORDER_BLOCK_FVG = "order_block_fvg"
    ORDER_BLOCK_STRUCTURE = "order_block_structure"
    FVG_STRUCTURE = "fvg_structure"
    TRIPLE_CONFLUENCE = "triple_confluence"


@dataclass
class PatternInput:
    """Input pattern data for signal generation"""

    order_blocks: List[OrderBlock]
    fair_value_gaps: List[FairValueGap]
    market_structures: List[MarketStructure]
    timeframe: str
    symbol: str
    current_price: Decimal
    timestamp: datetime


@dataclass
class GeneratedSignal:
    """Generated signal with all metadata"""

    signal_id: str
    direction: SignalDirection
    signal_type: SignalType
    symbol: str
    timeframe: str
    entry_price: Decimal
    stop_loss: Optional[Decimal]
    take_profit: Optional[Decimal]
    confidence_score: float
    pattern_combination: PatternCombination
    contributing_patterns: Dict[str, any]
    reasoning: str
    timestamp: datetime
    validity_duration: timedelta
    risk_reward_ratio: Optional[float] = None


class SignalGenerator:
    """
    Core Signal Generator using ICT Pattern Combinations

    Analyzes combinations of Order Blocks, Fair Value Gaps, and Market Structure
    to generate high-confidence trading signals with appropriate risk management levels.
    """

    def __init__(
        self,
        min_confidence_threshold: float = 0.6,
        max_signals_per_timeframe: int = 3,
        pattern_weights: Optional[Dict[PatternType, float]] = None,
        risk_reward_ratios: Optional[Dict[PatternCombination, float]] = None,
    ):
        """
        Initialize SignalGenerator

        Args:
            min_confidence_threshold: Minimum confidence score for signal generation
            max_signals_per_timeframe: Maximum concurrent signals per timeframe
            pattern_weights: Weighting for different pattern types
            risk_reward_ratios: Target R:R ratios for different pattern combinations
        """
        self.min_confidence_threshold = min_confidence_threshold
        self.max_signals_per_timeframe = max_signals_per_timeframe
        self.logger = logging.getLogger(f"{__name__}.SignalGenerator")

        # Default pattern weights (can be configured)
        self.pattern_weights = pattern_weights or {
            PatternType.ORDER_BLOCK: 0.35,
            PatternType.FAIR_VALUE_GAP: 0.30,
            PatternType.BREAK_OF_STRUCTURE: 0.20,
            PatternType.CHANGE_OF_CHARACTER: 0.15,
        }

        # Default risk-reward ratios by pattern combination
        self.risk_reward_ratios = risk_reward_ratios or {
            PatternCombination.ORDER_BLOCK_ONLY: 1.5,
            PatternCombination.FVG_ONLY: 1.2,
            PatternCombination.STRUCTURE_ONLY: 1.0,
            PatternCombination.ORDER_BLOCK_FVG: 2.0,
            PatternCombination.ORDER_BLOCK_STRUCTURE: 2.5,
            PatternCombination.FVG_STRUCTURE: 1.8,
            PatternCombination.TRIPLE_CONFLUENCE: 3.0,
        }

        # Active signals tracking
        self.active_signals: Dict[str, GeneratedSignal] = {}

        # Pattern priority matrix
        self.pattern_priority_matrix = self._initialize_priority_matrix()

    def generate_signals(self, pattern_input: PatternInput) -> List[GeneratedSignal]:
        """
        Generate trading signals from ICT pattern inputs

        Args:
            pattern_input: Input patterns and market data

        Returns:
            List of generated trading signals
        """
        try:
            self.logger.info(
                f"Generating signals for {pattern_input.symbol} {pattern_input.timeframe}"
            )

            # Validate input patterns
            if not self._validate_pattern_input(pattern_input):
                return []

            # Clean up expired signals
            self._cleanup_expired_signals()

            # Check if we can generate more signals for this timeframe
            if not self._can_generate_signals(pattern_input.timeframe):
                self.logger.info(
                    f"Maximum signals reached for {pattern_input.timeframe}"
                )
                return []

            # Analyze pattern combinations
            pattern_combinations = self._analyze_pattern_combinations(pattern_input)

            # Generate signals for each valid combination
            generated_signals = []
            for combination in pattern_combinations:
                signal = self._generate_signal_from_combination(
                    combination, pattern_input
                )
                if signal and signal.confidence_score >= self.min_confidence_threshold:
                    generated_signals.append(signal)

            # Filter and rank signals
            final_signals = self._filter_and_rank_signals(
                generated_signals, pattern_input
            )

            # Update active signals tracking
            for signal in final_signals:
                self.active_signals[signal.signal_id] = signal

            self.logger.info(f"Generated {len(final_signals)} signals")
            return final_signals

        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return []

    def _validate_pattern_input(self, pattern_input: PatternInput) -> bool:
        """Validate pattern input data"""
        try:
            # Check if we have at least one pattern
            total_patterns = (
                len(pattern_input.order_blocks)
                + len(pattern_input.fair_value_gaps)
                + len(pattern_input.market_structures)
            )

            if total_patterns == 0:
                self.logger.warning("No patterns provided for signal generation")
                return False

            # Validate current price
            if pattern_input.current_price <= 0:
                self.logger.error("Invalid current price")
                return False

            # Validate timestamp
            if pattern_input.timestamp is None:
                self.logger.error("Missing timestamp")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating pattern input: {e}")
            return False

    def _analyze_pattern_combinations(
        self, pattern_input: PatternInput
    ) -> List[Dict[str, any]]:
        """
        Analyze valid pattern combinations for signal generation

        Args:
            pattern_input: Input pattern data

        Returns:
            List of valid pattern combinations with metadata
        """
        combinations = []

        try:
            # Get recent and valid patterns
            recent_obs = self._filter_recent_patterns(pattern_input.order_blocks)
            recent_fvgs = self._filter_recent_patterns(pattern_input.fair_value_gaps)
            recent_structures = self._filter_recent_patterns(
                pattern_input.market_structures
            )

            # Single pattern combinations
            for ob in recent_obs:
                combinations.append(
                    {
                        "type": PatternCombination.ORDER_BLOCK_ONLY,
                        "patterns": {"order_block": ob},
                        "confidence_base": ob.confidence,
                    }
                )

            for fvg in recent_fvgs:
                if not fvg.is_filled:  # Only unfilled gaps
                    combinations.append(
                        {
                            "type": PatternCombination.FVG_ONLY,
                            "patterns": {"fair_value_gap": fvg},
                            "confidence_base": self._calculate_fvg_confidence(fvg),
                        }
                    )

            for structure in recent_structures:
                combinations.append(
                    {
                        "type": PatternCombination.STRUCTURE_ONLY,
                        "patterns": {"market_structure": structure},
                        "confidence_base": structure.confidence_score,
                    }
                )

            # Double pattern combinations
            for ob in recent_obs:
                for fvg in recent_fvgs:
                    if not fvg.is_filled and self._patterns_align(ob, fvg):
                        combinations.append(
                            {
                                "type": PatternCombination.ORDER_BLOCK_FVG,
                                "patterns": {
                                    "order_block": ob,
                                    "fair_value_gap": fvg,
                                },
                                "confidence_base": (
                                    ob.confidence + self._calculate_fvg_confidence(fvg)
                                )
                                / 2,
                            }
                        )

                for structure in recent_structures:
                    if self._patterns_align(ob, structure):
                        combinations.append(
                            {
                                "type": PatternCombination.ORDER_BLOCK_STRUCTURE,
                                "patterns": {
                                    "order_block": ob,
                                    "market_structure": structure,
                                },
                                "confidence_base": (
                                    ob.confidence + structure.confidence_score
                                )
                                / 2,
                            }
                        )

            for fvg in recent_fvgs:
                for structure in recent_structures:
                    if not fvg.is_filled and self._patterns_align(fvg, structure):
                        combinations.append(
                            {
                                "type": PatternCombination.FVG_STRUCTURE,
                                "patterns": {
                                    "fair_value_gap": fvg,
                                    "market_structure": structure,
                                },
                                "confidence_base": (
                                    self._calculate_fvg_confidence(fvg)
                                    + structure.confidence_score
                                )
                                / 2,
                            }
                        )

            # Triple pattern combinations
            for ob in recent_obs:
                for fvg in recent_fvgs:
                    for structure in recent_structures:
                        if (
                            not fvg.is_filled
                            and self._patterns_align(ob, fvg)
                            and self._patterns_align(ob, structure)
                            and self._patterns_align(fvg, structure)
                        ):
                            combinations.append(
                                {
                                    "type": PatternCombination.TRIPLE_CONFLUENCE,
                                    "patterns": {
                                        "order_block": ob,
                                        "fair_value_gap": fvg,
                                        "market_structure": structure,
                                    },
                                    "confidence_base": (
                                        ob.confidence
                                        + self._calculate_fvg_confidence(fvg)
                                        + structure.confidence_score
                                    )
                                    / 3,
                                }
                            )

            # Sort by confidence
            combinations.sort(key=lambda x: x["confidence_base"], reverse=True)
            return combinations[:10]  # Limit to top 10 combinations

        except Exception as e:
            self.logger.error(f"Error analyzing pattern combinations: {e}")
            return []

    def _generate_signal_from_combination(
        self, combination: Dict[str, any], pattern_input: PatternInput
    ) -> Optional[GeneratedSignal]:
        """
        Generate a trading signal from a pattern combination

        Args:
            combination: Pattern combination data
            pattern_input: Input pattern data

        Returns:
            Generated signal or None if generation fails
        """
        try:
            # Determine signal direction
            signal_direction = self._determine_signal_direction(combination)
            if signal_direction == SignalDirection.NEUTRAL:
                return None

            # Calculate entry price and levels
            entry_price, stop_loss, take_profit = self._calculate_signal_levels(
                combination, pattern_input, signal_direction
            )

            # Calculate confidence score
            confidence_score = self._calculate_signal_confidence(
                combination, pattern_input
            )

            # Generate signal ID
            signal_id = self._generate_signal_id(pattern_input, combination)

            # Map to SignalType
            signal_type = (
                SignalType.BUY
                if signal_direction == SignalDirection.LONG
                else SignalType.SELL
            )

            # Calculate risk-reward ratio
            risk_reward_ratio = self._calculate_risk_reward_ratio(
                entry_price, stop_loss, take_profit
            )

            # Generate reasoning
            reasoning = self._generate_signal_reasoning(combination, pattern_input)

            # Set validity duration based on pattern combination
            validity_duration = self._get_validity_duration(combination["type"])

            return GeneratedSignal(
                signal_id=signal_id,
                direction=signal_direction,
                signal_type=signal_type,
                symbol=pattern_input.symbol,
                timeframe=pattern_input.timeframe,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence_score=confidence_score,
                pattern_combination=combination["type"],
                contributing_patterns=combination["patterns"],
                reasoning=reasoning,
                timestamp=pattern_input.timestamp,
                validity_duration=validity_duration,
                risk_reward_ratio=risk_reward_ratio,
            )

        except Exception as e:
            self.logger.error(f"Error generating signal from combination: {e}")
            return None

    def _determine_signal_direction(
        self, combination: Dict[str, any]
    ) -> SignalDirection:
        """Determine signal direction from pattern combination"""
        try:
            patterns = combination["patterns"]
            directions = []

            # Collect directions from each pattern
            if "order_block" in patterns:
                ob = patterns["order_block"]
                if ob.direction == TrendDirection.BULLISH:
                    directions.append(SignalDirection.LONG)
                elif ob.direction == TrendDirection.BEARISH:
                    directions.append(SignalDirection.SHORT)

            if "fair_value_gap" in patterns:
                fvg = patterns["fair_value_gap"]
                if fvg.direction == TrendDirection.BULLISH:
                    directions.append(SignalDirection.LONG)
                elif fvg.direction == TrendDirection.BEARISH:
                    directions.append(SignalDirection.SHORT)

            if "market_structure" in patterns:
                structure = patterns["market_structure"]
                if structure.direction == TrendDirection.BULLISH:
                    directions.append(SignalDirection.LONG)
                elif structure.direction == TrendDirection.BEARISH:
                    directions.append(SignalDirection.SHORT)

            # Determine consensus direction
            if not directions:
                return SignalDirection.NEUTRAL

            long_count = directions.count(SignalDirection.LONG)
            short_count = directions.count(SignalDirection.SHORT)

            if long_count > short_count:
                return SignalDirection.LONG
            elif short_count > long_count:
                return SignalDirection.SHORT
            else:
                return SignalDirection.NEUTRAL

        except Exception as e:
            self.logger.error(f"Error determining signal direction: {e}")
            return SignalDirection.NEUTRAL

    def _calculate_signal_levels(
        self,
        combination: Dict[str, any],
        pattern_input: PatternInput,
        direction: SignalDirection,
    ) -> Tuple[Decimal, Optional[Decimal], Optional[Decimal]]:
        """Calculate entry, stop loss, and take profit levels"""
        try:
            patterns = combination["patterns"]
            current_price = pattern_input.current_price

            # Default levels
            entry_price = current_price
            stop_loss = None
            take_profit = None

            # Calculate ATR for level calculations
            atr = self._estimate_atr(pattern_input)

            if "order_block" in patterns:
                ob = patterns["order_block"]
                if direction == SignalDirection.LONG:
                    entry_price = Decimal(str(ob.low_price))
                    stop_loss = entry_price - Decimal(str(atr))
                else:
                    entry_price = Decimal(str(ob.high_price))
                    stop_loss = entry_price + Decimal(str(atr))

            elif "fair_value_gap" in patterns:
                fvg = patterns["fair_value_gap"]
                if direction == SignalDirection.LONG:
                    entry_price = Decimal(str(fvg.gap_low))
                    stop_loss = entry_price - Decimal(str(atr * 0.5))
                else:
                    entry_price = Decimal(str(fvg.gap_high))
                    stop_loss = entry_price + Decimal(str(atr * 0.5))

            # Calculate take profit using R:R ratio
            target_ratio = self.risk_reward_ratios.get(combination["type"], 2.0)
            if stop_loss:
                risk_amount = abs(entry_price - stop_loss)
                if direction == SignalDirection.LONG:
                    take_profit = entry_price + (
                        risk_amount * Decimal(str(target_ratio))
                    )
                else:
                    take_profit = entry_price - (
                        risk_amount * Decimal(str(target_ratio))
                    )

            return entry_price, stop_loss, take_profit

        except Exception as e:
            self.logger.error(f"Error calculating signal levels: {e}")
            return pattern_input.current_price, None, None

    def _calculate_signal_confidence(
        self, combination: Dict[str, any], pattern_input: PatternInput
    ) -> float:
        """Calculate overall signal confidence score"""
        try:
            base_confidence = combination["confidence_base"]

            # Confluence bonus
            pattern_count = len(combination["patterns"])
            confluence_multiplier = 1.0 + (pattern_count - 1) * 0.15

            # Time proximity bonus (fresher patterns get higher scores)
            time_bonus = self._calculate_time_proximity_bonus(
                combination, pattern_input
            )

            # Pattern alignment bonus
            alignment_bonus = self._calculate_pattern_alignment_bonus(combination)

            # Calculate final confidence
            final_confidence = (
                base_confidence
                * confluence_multiplier
                * (1 + time_bonus + alignment_bonus)
            )

            # Cap at 1.0
            return min(final_confidence, 1.0)

        except Exception as e:
            self.logger.error(f"Error calculating signal confidence: {e}")
            return 0.0

    def _generate_signal_reasoning(
        self, combination: Dict[str, any], pattern_input: PatternInput
    ) -> str:
        """Generate human-readable reasoning for the signal"""
        try:
            patterns = combination["patterns"]
            combination_type = combination["type"]

            reasoning_parts = []

            if combination_type == PatternCombination.TRIPLE_CONFLUENCE:
                reasoning_parts.append("Triple confluence detected:")
            elif "ORDER_BLOCK" in combination_type.value.upper():
                reasoning_parts.append("Order Block confluence:")
            elif "FVG" in combination_type.value.upper():
                reasoning_parts.append("Fair Value Gap setup:")
            elif "STRUCTURE" in combination_type.value.upper():
                reasoning_parts.append("Market structure break:")

            if "order_block" in patterns:
                ob = patterns["order_block"]
                reasoning_parts.append(
                    f"Order Block ({ob.direction.value}) at {ob.low_price:.5f}-{ob.high_price:.5f}"
                )

            if "fair_value_gap" in patterns:
                fvg = patterns["fair_value_gap"]
                reasoning_parts.append(
                    f"Unfilled FVG ({fvg.direction.value}) at {fvg.gap_low:.5f}-{fvg.gap_high:.5f}"
                )

            if "market_structure" in patterns:
                structure = patterns["market_structure"]
                reasoning_parts.append(
                    f"{structure.structure_type} ({structure.direction.value}) at {structure.break_level:.5f}"
                )

            reasoning_parts.append(f"Confidence: {combination['confidence_base']:.1%}")

            return " | ".join(reasoning_parts)

        except Exception as e:
            self.logger.error(f"Error generating signal reasoning: {e}")
            return f"ICT signal from {combination['type'].value}"

    # Helper methods
    def _filter_recent_patterns(self, patterns: List, max_age_hours: int = 24) -> List:
        """Filter patterns by recency"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            recent_patterns = []

            for pattern in patterns:
                pattern_time = getattr(pattern, "timestamp", datetime.now())
                if pattern_time >= cutoff_time:
                    recent_patterns.append(pattern)

            return recent_patterns
        except Exception:
            return patterns

    def _patterns_align(
        self,
        pattern1: Union[OrderBlock, FairValueGap, MarketStructure],
        pattern2: Union[OrderBlock, FairValueGap, MarketStructure],
    ) -> bool:
        """Check if two patterns have directional alignment"""
        try:
            dir1 = getattr(pattern1, "direction", TrendDirection.NEUTRAL)
            dir2 = getattr(pattern2, "direction", TrendDirection.NEUTRAL)
            return dir1 == dir2 and dir1 != TrendDirection.NEUTRAL
        except Exception:
            return False

    def _calculate_fvg_confidence(self, fvg: FairValueGap) -> float:
        """Calculate confidence score for Fair Value Gap"""
        try:
            # Base confidence from gap size
            base_confidence = min(fvg.gap_size / 0.001, 1.0)  # Normalize gap size

            # Bonus for unfilled gaps
            if not fvg.is_filled:
                base_confidence *= 1.2

            # Penalty for partially filled gaps
            if fvg.fill_percentage > 0:
                base_confidence *= 1.0 - fvg.fill_percentage * 0.5

            return min(base_confidence, 1.0)
        except Exception:
            return 0.5

    def _calculate_time_proximity_bonus(
        self, combination: Dict[str, any], pattern_input: PatternInput
    ) -> float:
        """Calculate bonus for recent patterns"""
        try:
            current_time = pattern_input.timestamp
            patterns = combination["patterns"]

            total_age = 0
            pattern_count = 0

            for pattern in patterns.values():
                pattern_time = getattr(pattern, "timestamp", current_time)
                age_hours = (current_time - pattern_time).total_seconds() / 3600
                total_age += age_hours
                pattern_count += 1

            if pattern_count == 0:
                return 0.0

            avg_age_hours = total_age / pattern_count

            # Fresher patterns get higher bonuses
            if avg_age_hours <= 1:
                return 0.3
            elif avg_age_hours <= 4:
                return 0.2
            elif avg_age_hours <= 12:
                return 0.1
            else:
                return 0.0

        except Exception:
            return 0.0

    def _calculate_pattern_alignment_bonus(self, combination: Dict[str, any]) -> float:
        """Calculate bonus for well-aligned patterns"""
        try:
            patterns = combination["patterns"]
            pattern_count = len(patterns)

            if pattern_count <= 1:
                return 0.0

            # Check directional alignment
            directions = []
            for pattern in patterns.values():
                direction = getattr(pattern, "direction", TrendDirection.NEUTRAL)
                if direction != TrendDirection.NEUTRAL:
                    directions.append(direction)

            if len(directions) <= 1:
                return 0.0

            # All patterns aligned = higher bonus
            if len(set(directions)) == 1:
                return 0.1 * (pattern_count - 1)
            else:
                return 0.0

        except Exception:
            return 0.0

    def _estimate_atr(self, pattern_input: PatternInput, period: int = 14) -> float:
        """Estimate ATR for level calculations"""
        try:
            # Simple ATR estimation based on current price
            # In a real implementation, this would use historical data
            return float(pattern_input.current_price) * 0.02  # 2% of current price
        except Exception:
            return 0.001

    def _calculate_risk_reward_ratio(
        self,
        entry_price: Decimal,
        stop_loss: Optional[Decimal],
        take_profit: Optional[Decimal],
    ) -> Optional[float]:
        """Calculate risk-reward ratio"""
        try:
            if not stop_loss or not take_profit:
                return None

            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)

            if risk == 0:
                return None

            return float(reward / risk)
        except Exception:
            return None

    def _generate_signal_id(
        self, pattern_input: PatternInput, combination: Dict[str, any]
    ) -> str:
        """Generate unique signal ID"""
        try:
            timestamp_str = pattern_input.timestamp.strftime("%Y%m%d_%H%M%S")
            combination_str = combination["type"].value
            return f"{pattern_input.symbol}_{pattern_input.timeframe}_{combination_str}_{timestamp_str}"
        except Exception:
            return f"signal_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def _get_validity_duration(self, combination_type: PatternCombination) -> timedelta:
        """Get signal validity duration based on pattern combination"""
        duration_map = {
            PatternCombination.ORDER_BLOCK_ONLY: timedelta(hours=4),
            PatternCombination.FVG_ONLY: timedelta(hours=2),
            PatternCombination.STRUCTURE_ONLY: timedelta(hours=6),
            PatternCombination.ORDER_BLOCK_FVG: timedelta(hours=6),
            PatternCombination.ORDER_BLOCK_STRUCTURE: timedelta(hours=8),
            PatternCombination.FVG_STRUCTURE: timedelta(hours=4),
            PatternCombination.TRIPLE_CONFLUENCE: timedelta(hours=12),
        }
        return duration_map.get(combination_type, timedelta(hours=4))

    def _filter_and_rank_signals(
        self, signals: List[GeneratedSignal], pattern_input: PatternInput
    ) -> List[GeneratedSignal]:
        """Filter and rank signals by quality"""
        try:
            # Filter by minimum confidence
            qualified_signals = [
                s
                for s in signals
                if s.confidence_score >= self.min_confidence_threshold
            ]

            # Rank by confidence score
            qualified_signals.sort(key=lambda x: x.confidence_score, reverse=True)

            # Limit by max signals per timeframe
            return qualified_signals[: self.max_signals_per_timeframe]

        except Exception as e:
            self.logger.error(f"Error filtering and ranking signals: {e}")
            return signals

    def _can_generate_signals(self, timeframe: str) -> bool:
        """Check if we can generate more signals for timeframe"""
        try:
            current_signals = [
                s for s in self.active_signals.values() if s.timeframe == timeframe
            ]
            return len(current_signals) < self.max_signals_per_timeframe
        except Exception:
            return True

    def _cleanup_expired_signals(self):
        """Remove expired signals from active tracking"""
        try:
            current_time = datetime.now()
            expired_ids = []

            for signal_id, signal in self.active_signals.items():
                expiry_time = signal.timestamp + signal.validity_duration
                if current_time > expiry_time:
                    expired_ids.append(signal_id)

            for signal_id in expired_ids:
                del self.active_signals[signal_id]

            if expired_ids:
                self.logger.info(f"Cleaned up {len(expired_ids)} expired signals")

        except Exception as e:
            self.logger.error(f"Error cleaning up expired signals: {e}")

    def _initialize_priority_matrix(self) -> Dict[PatternCombination, float]:
        """Initialize pattern combination priority matrix"""
        return {
            PatternCombination.TRIPLE_CONFLUENCE: 1.0,
            PatternCombination.ORDER_BLOCK_STRUCTURE: 0.9,
            PatternCombination.ORDER_BLOCK_FVG: 0.8,
            PatternCombination.FVG_STRUCTURE: 0.7,
            PatternCombination.ORDER_BLOCK_ONLY: 0.6,
            PatternCombination.STRUCTURE_ONLY: 0.5,
            PatternCombination.FVG_ONLY: 0.4,
        }

    def get_active_signals(
        self, timeframe: Optional[str] = None
    ) -> List[GeneratedSignal]:
        """Get currently active signals"""
        try:
            if timeframe:
                return [
                    s for s in self.active_signals.values() if s.timeframe == timeframe
                ]
            else:
                return list(self.active_signals.values())
        except Exception:
            return []

    def cancel_signal(self, signal_id: str) -> bool:
        """Cancel an active signal"""
        try:
            if signal_id in self.active_signals:
                del self.active_signals[signal_id]
                self.logger.info(f"Cancelled signal {signal_id}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error cancelling signal {signal_id}: {e}")
            return False

    def create_signal_event(self, signal: GeneratedSignal) -> SignalEvent:
        """Convert GeneratedSignal to SignalEvent for publishing"""
        try:
            return SignalEvent(
                source="SignalGenerator",
                symbol=signal.symbol,
                signal_type=signal.signal_type,
                confidence=signal.confidence_score,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                strategy_name=f"ICT_{signal.pattern_combination.value}",
                reasoning=signal.reasoning,
                metadata={
                    "signal_id": signal.signal_id,
                    "timeframe": signal.timeframe,
                    "pattern_combination": signal.pattern_combination.value,
                    "risk_reward_ratio": signal.risk_reward_ratio,
                    "validity_duration": signal.validity_duration.total_seconds(),
                    "contributing_patterns": str(signal.contributing_patterns),
                },
            )
        except Exception as e:
            self.logger.error(f"Error creating signal event: {e}")
            return None
