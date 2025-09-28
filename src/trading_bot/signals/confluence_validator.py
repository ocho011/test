"""
Confluence Validator for Multi-Condition Signal Verification

This module implements comprehensive confluence validation for trading signals,
ensuring signals meet multiple ICT criteria across different timeframes and
pattern types before being considered valid for execution.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from ..analysis.ict_analyzer import (
    OrderBlockDetector,
    FairValueGapAnalyzer,
    MarketStructureAnalyzer,
    TimeFrameManager,
    PatternValidationEngine,
    OrderBlock,
    FairValueGap,
    MarketStructure
)


class ConfluenceType(Enum):
    """Types of confluence validation criteria"""
    STRUCTURAL = "structural"  # Market structure alignment
    PATTERN = "pattern"  # Multiple pattern confirmation
    TEMPORAL = "temporal"  # Time-based confluence
    VOLUME = "volume"  # Volume-based confirmation
    MOMENTUM = "momentum"  # Momentum alignment
    SENTIMENT = "sentiment"  # Market sentiment alignment


class ConfluenceLevel(Enum):
    """Levels of confluence strength"""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


@dataclass
class ConfluenceResult:
    """Result of confluence validation"""
    is_valid: bool
    confluence_level: ConfluenceLevel
    confidence_score: float
    met_criteria: List[ConfluenceType]
    failed_criteria: List[ConfluenceType]
    details: Dict[str, Any]
    validation_timestamp: datetime


@dataclass
class ConfluenceConfig:
    """Configuration for confluence validation"""
    required_confluences: List[ConfluenceType]
    minimum_confluence_count: int = 3
    structural_weight: float = 0.25
    pattern_weight: float = 0.25
    temporal_weight: float = 0.20
    volume_weight: float = 0.15
    momentum_weight: float = 0.10
    sentiment_weight: float = 0.05
    confidence_threshold: float = 0.65
    strict_mode: bool = False


class ConfluenceValidator:
    """
    Multi-condition signal validation system for ICT-based trading signals.

    This validator ensures trading signals meet multiple confluence criteria
    across different dimensions before being considered valid for execution.
    """

    def __init__(
        self,
        order_block_detector: OrderBlockDetector,
        fvg_analyzer: FairValueGapAnalyzer,
        structure_analyzer: MarketStructureAnalyzer,
        timeframe_manager: TimeFrameManager,
        pattern_validator: PatternValidationEngine,
        config: Optional[ConfluenceConfig] = None
    ):
        self.order_block_detector = order_block_detector
        self.fvg_analyzer = fvg_analyzer
        self.structure_analyzer = structure_analyzer
        self.timeframe_manager = timeframe_manager
        self.pattern_validator = pattern_validator
        self.config = config or ConfluenceConfig(
            required_confluences=[
                ConfluenceType.STRUCTURAL,
                ConfluenceType.PATTERN,
                ConfluenceType.TEMPORAL
            ]
        )

        self.logger = logging.getLogger(__name__)

    def validate_signal_confluence(
        self,
        signal_data: Dict[str, Any],
        market_data: Dict[str, pd.DataFrame]
    ) -> ConfluenceResult:
        """
        Validate signal confluence across multiple dimensions.

        Args:
            signal_data: Signal information including patterns and levels
            market_data: Multi-timeframe market data

        Returns:
            ConfluenceResult with validation outcome and details
        """
        try:
            validation_results = {}
            met_criteria = []
            failed_criteria = []
            confidence_scores = {}

            # Validate structural confluence
            if ConfluenceType.STRUCTURAL in self.config.required_confluences:
                structural_result = self._validate_structural_confluence(
                    signal_data, market_data
                )
                validation_results["structural"] = structural_result
                confidence_scores["structural"] = structural_result["confidence"]

                if structural_result["valid"]:
                    met_criteria.append(ConfluenceType.STRUCTURAL)
                else:
                    failed_criteria.append(ConfluenceType.STRUCTURAL)

            # Validate pattern confluence
            if ConfluenceType.PATTERN in self.config.required_confluences:
                pattern_result = self._validate_pattern_confluence(
                    signal_data, market_data
                )
                validation_results["pattern"] = pattern_result
                confidence_scores["pattern"] = pattern_result["confidence"]

                if pattern_result["valid"]:
                    met_criteria.append(ConfluenceType.PATTERN)
                else:
                    failed_criteria.append(ConfluenceType.PATTERN)

            # Validate temporal confluence
            if ConfluenceType.TEMPORAL in self.config.required_confluences:
                temporal_result = self._validate_temporal_confluence(
                    signal_data, market_data
                )
                validation_results["temporal"] = temporal_result
                confidence_scores["temporal"] = temporal_result["confidence"]

                if temporal_result["valid"]:
                    met_criteria.append(ConfluenceType.TEMPORAL)
                else:
                    failed_criteria.append(ConfluenceType.TEMPORAL)

            # Validate volume confluence
            if ConfluenceType.VOLUME in self.config.required_confluences:
                volume_result = self._validate_volume_confluence(
                    signal_data, market_data
                )
                validation_results["volume"] = volume_result
                confidence_scores["volume"] = volume_result["confidence"]

                if volume_result["valid"]:
                    met_criteria.append(ConfluenceType.VOLUME)
                else:
                    failed_criteria.append(ConfluenceType.VOLUME)

            # Validate momentum confluence
            if ConfluenceType.MOMENTUM in self.config.required_confluences:
                momentum_result = self._validate_momentum_confluence(
                    signal_data, market_data
                )
                validation_results["momentum"] = momentum_result
                confidence_scores["momentum"] = momentum_result["confidence"]

                if momentum_result["valid"]:
                    met_criteria.append(ConfluenceType.MOMENTUM)
                else:
                    failed_criteria.append(ConfluenceType.MOMENTUM)

            # Validate sentiment confluence
            if ConfluenceType.SENTIMENT in self.config.required_confluences:
                sentiment_result = self._validate_sentiment_confluence(
                    signal_data, market_data
                )
                validation_results["sentiment"] = sentiment_result
                confidence_scores["sentiment"] = sentiment_result["confidence"]

                if sentiment_result["valid"]:
                    met_criteria.append(ConfluenceType.SENTIMENT)
                else:
                    failed_criteria.append(ConfluenceType.SENTIMENT)

            # Calculate overall confidence score
            overall_confidence = self._calculate_overall_confidence(confidence_scores)

            # Determine confluence level
            confluence_level = self._determine_confluence_level(
                len(met_criteria), overall_confidence
            )

            # Check if signal is valid
            is_valid = self._is_signal_valid(met_criteria, overall_confidence)

            return ConfluenceResult(
                is_valid=is_valid,
                confluence_level=confluence_level,
                confidence_score=overall_confidence,
                met_criteria=met_criteria,
                failed_criteria=failed_criteria,
                details=validation_results,
                validation_timestamp=datetime.now()
            )

        except Exception as e:
            self.logger.error(f"Error in confluence validation: {e}")
            return ConfluenceResult(
                is_valid=False,
                confluence_level=ConfluenceLevel.WEAK,
                confidence_score=0.0,
                met_criteria=[],
                failed_criteria=list(self.config.required_confluences),
                details={"error": str(e)},
                validation_timestamp=datetime.now()
            )

    def _validate_structural_confluence(
        self,
        signal_data: Dict[str, Any],
        market_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Validate market structure confluence across timeframes"""
        try:
            timeframes = list(market_data.keys())
            structural_alignments = []

            signal_direction = signal_data.get("direction", "unknown")

            for timeframe in timeframes:
                data = market_data[timeframe]

                # Analyze market structure
                structure_analysis = self.structure_analyzer.analyze_structure(data)

                if not structure_analysis:
                    continue

                latest_structure = structure_analysis[-1]

                # Check if structure aligns with signal direction
                structure_bias = latest_structure.bias

                alignment_score = 0.0
                if signal_direction == "long" and structure_bias == "bullish":
                    alignment_score = 1.0
                elif signal_direction == "short" and structure_bias == "bearish":
                    alignment_score = 1.0
                elif structure_bias == "neutral":
                    alignment_score = 0.5

                structural_alignments.append({
                    "timeframe": timeframe,
                    "bias": structure_bias,
                    "alignment_score": alignment_score,
                    "structure_type": latest_structure.structure_type
                })

            # Calculate overall structural confidence
            if structural_alignments:
                avg_alignment = np.mean([s["alignment_score"] for s in structural_alignments])
                structural_confidence = avg_alignment
            else:
                structural_confidence = 0.0

            is_valid = structural_confidence >= 0.6

            return {
                "valid": is_valid,
                "confidence": structural_confidence,
                "alignments": structural_alignments,
                "details": {
                    "signal_direction": signal_direction,
                    "structure_count": len(structural_alignments)
                }
            }

        except Exception as e:
            self.logger.error(f"Error in structural confluence validation: {e}")
            return {
                "valid": False,
                "confidence": 0.0,
                "alignments": [],
                "details": {"error": str(e)}
            }

    def _validate_pattern_confluence(
        self,
        signal_data: Dict[str, Any],
        market_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Validate pattern confluence across multiple pattern types"""
        try:
            pattern_confirmations = []
            signal_patterns = signal_data.get("patterns", [])

            if not signal_patterns:
                return {
                    "valid": False,
                    "confidence": 0.0,
                    "confirmations": [],
                    "details": {"error": "No patterns in signal data"}
                }

            for timeframe, data in market_data.items():
                # Check Order Block patterns
                order_blocks = self.order_block_detector.detect_order_blocks(data)
                if order_blocks:
                    ob_confirmation = self._check_order_block_confluence(
                        order_blocks, signal_data, timeframe
                    )
                    if ob_confirmation["valid"]:
                        pattern_confirmations.append(ob_confirmation)

                # Check Fair Value Gap patterns
                fvgs = self.fvg_analyzer.detect_fair_value_gaps(data)
                if fvgs:
                    fvg_confirmation = self._check_fvg_confluence(
                        fvgs, signal_data, timeframe
                    )
                    if fvg_confirmation["valid"]:
                        pattern_confirmations.append(fvg_confirmation)

            # Calculate pattern confluence confidence
            if pattern_confirmations:
                pattern_confidence = min(1.0, len(pattern_confirmations) / 3.0)
            else:
                pattern_confidence = 0.0

            is_valid = len(pattern_confirmations) >= 2

            return {
                "valid": is_valid,
                "confidence": pattern_confidence,
                "confirmations": pattern_confirmations,
                "details": {
                    "pattern_count": len(pattern_confirmations),
                    "signal_patterns": signal_patterns
                }
            }

        except Exception as e:
            self.logger.error(f"Error in pattern confluence validation: {e}")
            return {
                "valid": False,
                "confidence": 0.0,
                "confirmations": [],
                "details": {"error": str(e)}
            }

    def _validate_temporal_confluence(
        self,
        signal_data: Dict[str, Any],
        market_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Validate temporal confluence and timing alignment"""
        try:
            temporal_factors = []

            # Check session alignment
            signal_time = signal_data.get("timestamp", datetime.now())
            session_alignment = self._check_session_alignment(signal_time)
            temporal_factors.append(session_alignment)

            # Check market hours
            market_hours_alignment = self._check_market_hours_alignment(signal_time)
            temporal_factors.append(market_hours_alignment)

            # Check news time avoidance
            news_alignment = self._check_news_time_alignment(signal_time)
            temporal_factors.append(news_alignment)

            # Calculate temporal confidence
            valid_factors = [f for f in temporal_factors if f["valid"]]
            temporal_confidence = len(valid_factors) / len(temporal_factors)

            is_valid = temporal_confidence >= 0.67

            return {
                "valid": is_valid,
                "confidence": temporal_confidence,
                "factors": temporal_factors,
                "details": {
                    "signal_time": signal_time.isoformat(),
                    "valid_factors": len(valid_factors)
                }
            }

        except Exception as e:
            self.logger.error(f"Error in temporal confluence validation: {e}")
            return {
                "valid": False,
                "confidence": 0.0,
                "factors": [],
                "details": {"error": str(e)}
            }

    def _validate_volume_confluence(
        self,
        signal_data: Dict[str, Any],
        market_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Validate volume-based confluence"""
        try:
            volume_confirmations = []

            for timeframe, data in market_data.items():
                if "volume" not in data.columns:
                    continue

                # Calculate volume metrics
                recent_volume = data["volume"].tail(20).mean()
                volume_sma = data["volume"].rolling(50).mean().iloc[-1]

                volume_ratio = recent_volume / volume_sma if volume_sma > 0 else 0

                # Check volume confirmation
                volume_confirmation = {
                    "timeframe": timeframe,
                    "volume_ratio": volume_ratio,
                    "valid": volume_ratio > 1.2  # Above average volume
                }

                volume_confirmations.append(volume_confirmation)

            # Calculate volume confidence
            valid_volumes = [v for v in volume_confirmations if v["valid"]]
            volume_confidence = len(valid_volumes) / max(1, len(volume_confirmations))

            is_valid = volume_confidence >= 0.5

            return {
                "valid": is_valid,
                "confidence": volume_confidence,
                "confirmations": volume_confirmations,
                "details": {
                    "valid_timeframes": len(valid_volumes)
                }
            }

        except Exception as e:
            self.logger.error(f"Error in volume confluence validation: {e}")
            return {
                "valid": False,
                "confidence": 0.0,
                "confirmations": [],
                "details": {"error": str(e)}
            }

    def _validate_momentum_confluence(
        self,
        signal_data: Dict[str, Any],
        market_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Validate momentum confluence across timeframes"""
        try:
            momentum_confirmations = []
            signal_direction = signal_data.get("direction", "unknown")

            for timeframe, data in market_data.items():
                # Calculate momentum indicators
                rsi = self._calculate_rsi(data["close"])
                macd_line, macd_signal = self._calculate_macd(data["close"])

                # Check momentum alignment
                momentum_aligned = False

                if signal_direction == "long":
                    momentum_aligned = (
                        rsi.iloc[-1] > 40 and rsi.iloc[-1] < 70 and
                        macd_line.iloc[-1] > macd_signal.iloc[-1]
                    )
                elif signal_direction == "short":
                    momentum_aligned = (
                        rsi.iloc[-1] < 60 and rsi.iloc[-1] > 30 and
                        macd_line.iloc[-1] < macd_signal.iloc[-1]
                    )

                momentum_confirmations.append({
                    "timeframe": timeframe,
                    "rsi": rsi.iloc[-1],
                    "macd_bullish": macd_line.iloc[-1] > macd_signal.iloc[-1],
                    "aligned": momentum_aligned
                })

            # Calculate momentum confidence
            aligned_momentum = [m for m in momentum_confirmations if m["aligned"]]
            momentum_confidence = len(aligned_momentum) / max(1, len(momentum_confirmations))

            is_valid = momentum_confidence >= 0.6

            return {
                "valid": is_valid,
                "confidence": momentum_confidence,
                "confirmations": momentum_confirmations,
                "details": {
                    "signal_direction": signal_direction,
                    "aligned_timeframes": len(aligned_momentum)
                }
            }

        except Exception as e:
            self.logger.error(f"Error in momentum confluence validation: {e}")
            return {
                "valid": False,
                "confidence": 0.0,
                "confirmations": [],
                "details": {"error": str(e)}
            }

    def _validate_sentiment_confluence(
        self,
        signal_data: Dict[str, Any],
        market_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Validate market sentiment confluence"""
        try:
            # For now, implement basic sentiment based on price action
            sentiment_factors = []

            for timeframe, data in market_data.items():
                # Calculate basic sentiment from price action
                recent_closes = data["close"].tail(10)
                price_trend = (recent_closes.iloc[-1] - recent_closes.iloc[0]) / recent_closes.iloc[0]

                sentiment = "bullish" if price_trend > 0.001 else "bearish" if price_trend < -0.001 else "neutral"

                sentiment_factors.append({
                    "timeframe": timeframe,
                    "sentiment": sentiment,
                    "price_trend": price_trend
                })

            # Calculate sentiment confidence (simplified)
            signal_direction = signal_data.get("direction", "unknown")
            aligned_sentiment = 0

            for factor in sentiment_factors:
                if (signal_direction == "long" and factor["sentiment"] == "bullish") or \
                   (signal_direction == "short" and factor["sentiment"] == "bearish"):
                    aligned_sentiment += 1

            sentiment_confidence = aligned_sentiment / max(1, len(sentiment_factors))
            is_valid = sentiment_confidence >= 0.5

            return {
                "valid": is_valid,
                "confidence": sentiment_confidence,
                "factors": sentiment_factors,
                "details": {
                    "signal_direction": signal_direction,
                    "aligned_sentiment": aligned_sentiment
                }
            }

        except Exception as e:
            self.logger.error(f"Error in sentiment confluence validation: {e}")
            return {
                "valid": False,
                "confidence": 0.0,
                "factors": [],
                "details": {"error": str(e)}
            }

    def _check_order_block_confluence(
        self,
        order_blocks: List[OrderBlock],
        signal_data: Dict[str, Any],
        timeframe: str
    ) -> Dict[str, Any]:
        """Check Order Block confluence with signal"""
        signal_direction = signal_data.get("direction", "unknown")
        signal_entry = signal_data.get("entry_price", 0)

        relevant_obs = []
        for ob in order_blocks:
            if signal_direction == "long" and ob.block_type == "demand":
                if signal_entry >= ob.low and signal_entry <= ob.high:
                    relevant_obs.append(ob)
            elif signal_direction == "short" and ob.block_type == "supply":
                if signal_entry >= ob.low and signal_entry <= ob.high:
                    relevant_obs.append(ob)

        return {
            "valid": len(relevant_obs) > 0,
            "timeframe": timeframe,
            "pattern_type": "order_block",
            "relevant_patterns": len(relevant_obs)
        }

    def _check_fvg_confluence(
        self,
        fvgs: List[FairValueGap],
        signal_data: Dict[str, Any],
        timeframe: str
    ) -> Dict[str, Any]:
        """Check Fair Value Gap confluence with signal"""
        signal_direction = signal_data.get("direction", "unknown")
        signal_entry = signal_data.get("entry_price", 0)

        relevant_fvgs = []
        for fvg in fvgs:
            if signal_direction == "long" and fvg.gap_type == "bullish":
                if signal_entry >= fvg.low and signal_entry <= fvg.high:
                    relevant_fvgs.append(fvg)
            elif signal_direction == "short" and fvg.gap_type == "bearish":
                if signal_entry >= fvg.low and signal_entry <= fvg.high:
                    relevant_fvgs.append(fvg)

        return {
            "valid": len(relevant_fvgs) > 0,
            "timeframe": timeframe,
            "pattern_type": "fair_value_gap",
            "relevant_patterns": len(relevant_fvgs)
        }

    def _check_session_alignment(self, signal_time: datetime) -> Dict[str, Any]:
        """Check if signal timing aligns with optimal trading sessions"""
        # London session: 8:00-17:00 GMT
        # New York session: 13:00-22:00 GMT
        # Overlap: 13:00-17:00 GMT (optimal time)

        hour_utc = signal_time.hour

        # Check if within optimal overlap period
        in_overlap = 13 <= hour_utc <= 17
        in_london = 8 <= hour_utc <= 17
        in_newyork = 13 <= hour_utc <= 22

        session_score = 1.0 if in_overlap else 0.7 if (in_london or in_newyork) else 0.3

        return {
            "valid": session_score >= 0.7,
            "score": session_score,
            "details": {
                "hour_utc": hour_utc,
                "in_overlap": in_overlap,
                "in_london": in_london,
                "in_newyork": in_newyork
            }
        }

    def _check_market_hours_alignment(self, signal_time: datetime) -> Dict[str, Any]:
        """Check if signal is during active market hours"""
        # Avoid weekends and major holidays
        weekday = signal_time.weekday()  # 0=Monday, 6=Sunday

        is_weekday = weekday < 5  # Monday to Friday

        return {
            "valid": is_weekday,
            "score": 1.0 if is_weekday else 0.0,
            "details": {
                "weekday": weekday,
                "is_weekend": not is_weekday
            }
        }

    def _check_news_time_alignment(self, signal_time: datetime) -> Dict[str, Any]:
        """Check if signal avoids high-impact news times"""
        # Simplified: avoid first and last 30 minutes of major sessions
        hour_utc = signal_time.hour
        minute = signal_time.minute

        # Avoid 8:00-8:30 GMT (London open)
        # Avoid 13:00-13:30 GMT (NY open)
        # Avoid 16:30-17:00 GMT (London close)

        avoid_times = [
            (8, 0, 8, 30),   # London open
            (13, 0, 13, 30), # NY open
            (16, 30, 17, 0)  # London close
        ]

        in_avoid_period = any(
            (start_h < hour_utc < end_h) or
            (start_h == hour_utc and minute >= start_m) or
            (end_h == hour_utc and minute <= end_m)
            for start_h, start_m, end_h, end_m in avoid_times
        )

        return {
            "valid": not in_avoid_period,
            "score": 0.0 if in_avoid_period else 1.0,
            "details": {
                "hour_utc": hour_utc,
                "minute": minute,
                "in_avoid_period": in_avoid_period
            }
        }

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(
        self,
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        return macd_line, signal_line

    def _calculate_overall_confidence(self, confidence_scores: Dict[str, float]) -> float:
        """Calculate weighted overall confidence score"""
        weights = {
            "structural": self.config.structural_weight,
            "pattern": self.config.pattern_weight,
            "temporal": self.config.temporal_weight,
            "volume": self.config.volume_weight,
            "momentum": self.config.momentum_weight,
            "sentiment": self.config.sentiment_weight
        }

        weighted_score = 0.0
        total_weight = 0.0

        for criterion, score in confidence_scores.items():
            weight = weights.get(criterion, 0.0)
            weighted_score += score * weight
            total_weight += weight

        return weighted_score / total_weight if total_weight > 0 else 0.0

    def _determine_confluence_level(
        self,
        met_criteria_count: int,
        confidence_score: float
    ) -> ConfluenceLevel:
        """Determine confluence strength level"""
        if met_criteria_count >= 5 and confidence_score >= 0.85:
            return ConfluenceLevel.VERY_STRONG
        elif met_criteria_count >= 4 and confidence_score >= 0.75:
            return ConfluenceLevel.STRONG
        elif met_criteria_count >= 3 and confidence_score >= 0.65:
            return ConfluenceLevel.MODERATE
        else:
            return ConfluenceLevel.WEAK

    def _is_signal_valid(
        self,
        met_criteria: List[ConfluenceType],
        confidence_score: float
    ) -> bool:
        """Determine if signal meets validation requirements"""
        criteria_met = len(met_criteria) >= self.config.minimum_confluence_count
        confidence_met = confidence_score >= self.config.confidence_threshold

        if self.config.strict_mode:
            required_met = all(
                req in met_criteria for req in self.config.required_confluences
            )
            return criteria_met and confidence_met and required_met
        else:
            return criteria_met and confidence_met