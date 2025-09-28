"""
Time-Based Bias Filter for ICT Trading Signals

This module implements sophisticated time-based filtering to ensure trading signals
align with optimal market sessions, bias periods, and temporal trading patterns
based on ICT methodology.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta, time
import pytz

import pandas as pd
import numpy as np


class SessionType(Enum):
    """Trading session types"""
    ASIAN = "asian"
    LONDON = "london"
    NEW_YORK = "new_york"
    SYDNEY = "sydney"


class BiasDirection(Enum):
    """Market bias directions"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    RANGING = "ranging"


class TimeFilter(Enum):
    """Time-based filter types"""
    SESSION_FILTER = "session_filter"
    BIAS_FILTER = "bias_filter"
    NEWS_FILTER = "news_filter"
    VOLATILITY_FILTER = "volatility_filter"
    LIQUIDITY_FILTER = "liquidity_filter"


@dataclass
class SessionInfo:
    """Trading session information"""
    session_type: SessionType
    start_time: time
    end_time: time
    timezone: str
    volatility_score: float
    liquidity_score: float
    optimal_for_direction: Optional[BiasDirection] = None


@dataclass
class BiasWindow:
    """Time window with specific market bias"""
    start_time: datetime
    end_time: datetime
    bias_direction: BiasDirection
    confidence: float
    session: SessionType
    rationale: str


@dataclass
class FilterResult:
    """Result of bias filtering"""
    is_allowed: bool
    filter_score: float
    active_filters: List[TimeFilter]
    failed_filters: List[TimeFilter]
    current_session: SessionType
    current_bias: BiasDirection
    details: Dict[str, Any]
    filter_timestamp: datetime


@dataclass
class BiasFilterConfig:
    """Configuration for bias filtering"""
    timezone: str = "UTC"
    allowed_sessions: List[SessionType] = None
    minimum_session_score: float = 0.6
    respect_bias_windows: bool = True
    avoid_news_times: bool = True
    require_liquidity: bool = True
    session_overlap_bonus: float = 0.2
    bias_confidence_threshold: float = 0.65
    news_avoidance_minutes: int = 30


class BiasFilter:
    """
    Time-based bias filtering system for ICT trading signals.

    This filter ensures signals are only generated during optimal time periods
    based on session characteristics, market bias, and temporal patterns.
    """

    def __init__(self, config: Optional[BiasFilterConfig] = None):
        self.config = config or BiasFilterConfig()

        # Set default allowed sessions if not specified
        if self.config.allowed_sessions is None:
            self.config.allowed_sessions = [
                SessionType.LONDON,
                SessionType.NEW_YORK
            ]

        self.logger = logging.getLogger(__name__)

        # Define trading sessions (UTC times)
        self.sessions = {
            SessionType.SYDNEY: SessionInfo(
                session_type=SessionType.SYDNEY,
                start_time=time(22, 0),  # 22:00 UTC
                end_time=time(6, 0),     # 06:00 UTC next day
                timezone="Australia/Sydney",
                volatility_score=0.4,
                liquidity_score=0.3
            ),
            SessionType.ASIAN: SessionInfo(
                session_type=SessionType.ASIAN,
                start_time=time(0, 0),   # 00:00 UTC
                end_time=time(9, 0),     # 09:00 UTC
                timezone="Asia/Tokyo",
                volatility_score=0.5,
                liquidity_score=0.4
            ),
            SessionType.LONDON: SessionInfo(
                session_type=SessionType.LONDON,
                start_time=time(8, 0),   # 08:00 UTC
                end_time=time(17, 0),    # 17:00 UTC
                timezone="Europe/London",
                volatility_score=0.8,
                liquidity_score=0.9,
                optimal_for_direction=BiasDirection.BULLISH
            ),
            SessionType.NEW_YORK: SessionInfo(
                session_type=SessionType.NEW_YORK,
                start_time=time(13, 0),  # 13:00 UTC
                end_time=time(22, 0),    # 22:00 UTC
                timezone="America/New_York",
                volatility_score=0.9,
                liquidity_score=1.0,
                optimal_for_direction=BiasDirection.BEARISH
            )
        }

        # High-impact news times to avoid (UTC)
        self.news_times = [
            time(8, 30),   # London open
            time(13, 30),  # NY open
            time(14, 30),  # US economic data
            time(16, 30),  # London close
            time(20, 0),   # US close data
        ]

    def filter_signal(
        self,
        signal_data: Dict[str, Any],
        current_time: Optional[datetime] = None
    ) -> FilterResult:
        """
        Filter signal based on time-based bias criteria.

        Args:
            signal_data: Signal information including direction and timing
            current_time: Current time for filtering (defaults to now)

        Returns:
            FilterResult with filtering outcome and details
        """
        if current_time is None:
            current_time = datetime.now(pytz.UTC)

        try:
            filter_results = {}
            active_filters = []
            failed_filters = []

            # Apply session filter
            session_result = self._apply_session_filter(signal_data, current_time)
            filter_results["session"] = session_result
            if session_result["passed"]:
                active_filters.append(TimeFilter.SESSION_FILTER)
            else:
                failed_filters.append(TimeFilter.SESSION_FILTER)

            # Apply bias filter
            bias_result = self._apply_bias_filter(signal_data, current_time)
            filter_results["bias"] = bias_result
            if bias_result["passed"]:
                active_filters.append(TimeFilter.BIAS_FILTER)
            else:
                failed_filters.append(TimeFilter.BIAS_FILTER)

            # Apply news filter
            if self.config.avoid_news_times:
                news_result = self._apply_news_filter(signal_data, current_time)
                filter_results["news"] = news_result
                if news_result["passed"]:
                    active_filters.append(TimeFilter.NEWS_FILTER)
                else:
                    failed_filters.append(TimeFilter.NEWS_FILTER)

            # Apply volatility filter
            volatility_result = self._apply_volatility_filter(signal_data, current_time)
            filter_results["volatility"] = volatility_result
            if volatility_result["passed"]:
                active_filters.append(TimeFilter.VOLATILITY_FILTER)
            else:
                failed_filters.append(TimeFilter.VOLATILITY_FILTER)

            # Apply liquidity filter
            if self.config.require_liquidity:
                liquidity_result = self._apply_liquidity_filter(signal_data, current_time)
                filter_results["liquidity"] = liquidity_result
                if liquidity_result["passed"]:
                    active_filters.append(TimeFilter.LIQUIDITY_FILTER)
                else:
                    failed_filters.append(TimeFilter.LIQUIDITY_FILTER)

            # Calculate overall filter score
            filter_score = self._calculate_filter_score(filter_results)

            # Determine current session and bias
            current_session = self._get_current_session(current_time)
            current_bias = self._determine_current_bias(current_time, signal_data)

            # Determine if signal is allowed
            is_allowed = self._is_signal_allowed(active_filters, failed_filters, filter_score)

            return FilterResult(
                is_allowed=is_allowed,
                filter_score=filter_score,
                active_filters=active_filters,
                failed_filters=failed_filters,
                current_session=current_session,
                current_bias=current_bias,
                details=filter_results,
                filter_timestamp=current_time
            )

        except Exception as e:
            self.logger.error(f"Error in bias filtering: {e}")
            return FilterResult(
                is_allowed=False,
                filter_score=0.0,
                active_filters=[],
                failed_filters=list(TimeFilter),
                current_session=SessionType.ASIAN,  # Default
                current_bias=BiasDirection.NEUTRAL,
                details={"error": str(e)},
                filter_timestamp=current_time
            )

    def _apply_session_filter(
        self,
        signal_data: Dict[str, Any],
        current_time: datetime
    ) -> Dict[str, Any]:
        """Apply trading session filter"""
        try:
            current_session = self._get_current_session(current_time)
            session_info = self.sessions[current_session]

            # Check if current session is allowed
            session_allowed = current_session in self.config.allowed_sessions

            # Calculate session score
            base_score = session_info.volatility_score * 0.6 + session_info.liquidity_score * 0.4

            # Check for session overlaps (bonus)
            overlap_bonus = 0.0
            if self._is_session_overlap(current_time):
                overlap_bonus = self.config.session_overlap_bonus

            session_score = min(1.0, base_score + overlap_bonus)

            # Check minimum session score requirement
            score_requirement_met = session_score >= self.config.minimum_session_score

            # Overall session filter result
            passed = session_allowed and score_requirement_met

            return {
                "passed": passed,
                "score": session_score,
                "current_session": current_session.value,
                "session_allowed": session_allowed,
                "score_requirement_met": score_requirement_met,
                "overlap_bonus": overlap_bonus,
                "details": {
                    "volatility_score": session_info.volatility_score,
                    "liquidity_score": session_info.liquidity_score,
                    "minimum_required": self.config.minimum_session_score
                }
            }

        except Exception as e:
            self.logger.error(f"Error in session filter: {e}")
            return {
                "passed": False,
                "score": 0.0,
                "error": str(e)
            }

    def _apply_bias_filter(
        self,
        signal_data: Dict[str, Any],
        current_time: datetime
    ) -> Dict[str, Any]:
        """Apply market bias filter"""
        try:
            signal_direction = signal_data.get("direction", "unknown")
            current_bias = self._determine_current_bias(current_time, signal_data)

            # Check bias alignment
            bias_aligned = False
            bias_confidence = 0.5

            if signal_direction == "long" and current_bias == BiasDirection.BULLISH:
                bias_aligned = True
                bias_confidence = 0.8
            elif signal_direction == "short" and current_bias == BiasDirection.BEARISH:
                bias_aligned = True
                bias_confidence = 0.8
            elif current_bias == BiasDirection.NEUTRAL:
                bias_aligned = True
                bias_confidence = 0.6
            elif current_bias == BiasDirection.RANGING:
                # Ranging markets can be traded with proper setup
                bias_aligned = True
                bias_confidence = 0.4

            # Check confidence threshold
            confidence_met = bias_confidence >= self.config.bias_confidence_threshold

            # Overall bias filter result
            passed = bias_aligned and (not self.config.respect_bias_windows or confidence_met)

            return {
                "passed": passed,
                "score": bias_confidence,
                "signal_direction": signal_direction,
                "current_bias": current_bias.value,
                "bias_aligned": bias_aligned,
                "confidence_met": confidence_met,
                "details": {
                    "bias_confidence_threshold": self.config.bias_confidence_threshold,
                    "respect_bias_windows": self.config.respect_bias_windows
                }
            }

        except Exception as e:
            self.logger.error(f"Error in bias filter: {e}")
            return {
                "passed": False,
                "score": 0.0,
                "error": str(e)
            }

    def _apply_news_filter(
        self,
        signal_data: Dict[str, Any],
        current_time: datetime
    ) -> Dict[str, Any]:
        """Apply news time avoidance filter"""
        try:
            current_utc_time = current_time.time()
            avoidance_minutes = self.config.news_avoidance_minutes

            # Check if current time is within avoidance window of news times
            in_news_window = False
            closest_news_time = None
            minutes_to_news = float('inf')

            for news_time in self.news_times:
                # Calculate time difference in minutes
                current_minutes = current_utc_time.hour * 60 + current_utc_time.minute
                news_minutes = news_time.hour * 60 + news_time.minute

                time_diff = abs(current_minutes - news_minutes)

                # Handle day boundary (e.g., 23:45 to 00:15)
                if time_diff > 12 * 60:  # More than 12 hours apart
                    time_diff = 24 * 60 - time_diff

                if time_diff < minutes_to_news:
                    minutes_to_news = time_diff
                    closest_news_time = news_time

                if time_diff <= avoidance_minutes:
                    in_news_window = True

            # Filter passes if not in news window
            passed = not in_news_window

            return {
                "passed": passed,
                "score": 1.0 if passed else 0.0,
                "in_news_window": in_news_window,
                "closest_news_time": closest_news_time.strftime("%H:%M") if closest_news_time else None,
                "minutes_to_news": int(minutes_to_news),
                "details": {
                    "avoidance_minutes": avoidance_minutes,
                    "news_times_utc": [t.strftime("%H:%M") for t in self.news_times]
                }
            }

        except Exception as e:
            self.logger.error(f"Error in news filter: {e}")
            return {
                "passed": False,
                "score": 0.0,
                "error": str(e)
            }

    def _apply_volatility_filter(
        self,
        signal_data: Dict[str, Any],
        current_time: datetime
    ) -> Dict[str, Any]:
        """Apply volatility-based filter"""
        try:
            current_session = self._get_current_session(current_time)
            session_info = self.sessions[current_session]

            # Base volatility score from session
            volatility_score = session_info.volatility_score

            # Apply time-based volatility adjustments
            hour_utc = current_time.hour

            # First hour of major sessions typically has higher volatility
            if hour_utc in [8, 13]:  # London and NY opens
                volatility_score = min(1.0, volatility_score + 0.2)

            # Last hour of sessions can be volatile but less predictable
            elif hour_utc in [16, 21]:  # London and NY closes
                volatility_score = max(0.3, volatility_score - 0.1)

            # Overlap periods have enhanced volatility
            if self._is_session_overlap(current_time):
                volatility_score = min(1.0, volatility_score + 0.15)

            # Filter passes if volatility is adequate
            passed = volatility_score >= 0.5

            return {
                "passed": passed,
                "score": volatility_score,
                "base_session_volatility": session_info.volatility_score,
                "time_adjustment_applied": True,
                "is_session_overlap": self._is_session_overlap(current_time),
                "details": {
                    "hour_utc": hour_utc,
                    "session": current_session.value
                }
            }

        except Exception as e:
            self.logger.error(f"Error in volatility filter: {e}")
            return {
                "passed": False,
                "score": 0.0,
                "error": str(e)
            }

    def _apply_liquidity_filter(
        self,
        signal_data: Dict[str, Any],
        current_time: datetime
    ) -> Dict[str, Any]:
        """Apply liquidity-based filter"""
        try:
            current_session = self._get_current_session(current_time)
            session_info = self.sessions[current_session]

            # Base liquidity score from session
            liquidity_score = session_info.liquidity_score

            # Overlap periods have enhanced liquidity
            if self._is_session_overlap(current_time):
                liquidity_score = min(1.0, liquidity_score + 0.2)

            # Weekend or off-hours penalty
            weekday = current_time.weekday()  # 0=Monday, 6=Sunday
            if weekday >= 5:  # Weekend
                liquidity_score = max(0.1, liquidity_score - 0.5)

            # Filter passes if liquidity is adequate
            passed = liquidity_score >= 0.6

            return {
                "passed": passed,
                "score": liquidity_score,
                "base_session_liquidity": session_info.liquidity_score,
                "is_weekend": weekday >= 5,
                "is_session_overlap": self._is_session_overlap(current_time),
                "details": {
                    "weekday": weekday,
                    "session": current_session.value
                }
            }

        except Exception as e:
            self.logger.error(f"Error in liquidity filter: {e}")
            return {
                "passed": False,
                "score": 0.0,
                "error": str(e)
            }

    def _get_current_session(self, current_time: datetime) -> SessionType:
        """Determine current trading session"""
        current_utc_time = current_time.time()
        current_hour = current_utc_time.hour

        # Check each session
        for session_type, session_info in self.sessions.items():
            start_hour = session_info.start_time.hour
            end_hour = session_info.end_time.hour

            # Handle sessions that cross midnight
            if start_hour > end_hour:  # Session crosses midnight
                if current_hour >= start_hour or current_hour < end_hour:
                    return session_type
            else:  # Normal session
                if start_hour <= current_hour < end_hour:
                    return session_type

        # Default to Asian session if no match
        return SessionType.ASIAN

    def _is_session_overlap(self, current_time: datetime) -> bool:
        """Check if current time is during session overlap"""
        current_hour = current_time.hour

        # London-NY overlap: 13:00-17:00 UTC
        london_ny_overlap = 13 <= current_hour < 17

        # Sydney-Asian overlap: 22:00-06:00 UTC (crosses midnight)
        sydney_asian_overlap = current_hour >= 22 or current_hour < 6

        return london_ny_overlap or sydney_asian_overlap

    def _determine_current_bias(
        self,
        current_time: datetime,
        signal_data: Dict[str, Any]
    ) -> BiasDirection:
        """Determine current market bias based on time and context"""
        try:
            current_session = self._get_current_session(current_time)
            session_info = self.sessions[current_session]

            # Start with session's optimal bias if available
            if session_info.optimal_for_direction:
                base_bias = session_info.optimal_for_direction
            else:
                base_bias = BiasDirection.NEUTRAL

            # Time-based bias adjustments
            hour_utc = current_time.hour

            # London morning typically bullish (8-12 UTC)
            if 8 <= hour_utc < 12:
                return BiasDirection.BULLISH

            # NY afternoon typically bearish (15-19 UTC)
            elif 15 <= hour_utc < 19:
                return BiasDirection.BEARISH

            # Overlap periods can be ranging
            elif self._is_session_overlap(current_time):
                return BiasDirection.RANGING

            # Off-hours tend to be neutral
            elif hour_utc < 8 or hour_utc > 22:
                return BiasDirection.NEUTRAL

            return base_bias

        except Exception as e:
            self.logger.error(f"Error determining current bias: {e}")
            return BiasDirection.NEUTRAL

    def _calculate_filter_score(self, filter_results: Dict[str, Dict[str, Any]]) -> float:
        """Calculate overall filter score from individual filter results"""
        scores = []
        weights = {
            "session": 0.30,
            "bias": 0.25,
            "news": 0.20,
            "volatility": 0.15,
            "liquidity": 0.10
        }

        total_weight = 0.0
        weighted_sum = 0.0

        for filter_name, result in filter_results.items():
            if "score" in result and filter_name in weights:
                weight = weights[filter_name]
                score = result["score"]

                weighted_sum += score * weight
                total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _is_signal_allowed(
        self,
        active_filters: List[TimeFilter],
        failed_filters: List[TimeFilter],
        filter_score: float
    ) -> bool:
        """Determine if signal should be allowed based on filter results"""
        # Critical filters that must pass
        critical_filters = [TimeFilter.SESSION_FILTER]

        # Check if all critical filters passed
        critical_passed = all(f in active_filters for f in critical_filters)

        # Check if minimum score is met
        score_threshold = 0.6
        score_met = filter_score >= score_threshold

        # Signal is allowed if critical filters pass and score is adequate
        return critical_passed and score_met

    def get_optimal_trading_windows(
        self,
        date: datetime,
        signal_direction: Optional[str] = None
    ) -> List[BiasWindow]:
        """Get optimal trading windows for a given date"""
        try:
            windows = []

            # London session window
            london_start = datetime.combine(date.date(), time(8, 0))
            london_end = datetime.combine(date.date(), time(17, 0))

            windows.append(BiasWindow(
                start_time=london_start,
                end_time=london_end,
                bias_direction=BiasDirection.BULLISH,
                confidence=0.8,
                session=SessionType.LONDON,
                rationale="London session with high liquidity and volatility"
            ))

            # NY session window
            ny_start = datetime.combine(date.date(), time(13, 0))
            ny_end = datetime.combine(date.date(), time(22, 0))

            windows.append(BiasWindow(
                start_time=ny_start,
                end_time=ny_end,
                bias_direction=BiasDirection.BEARISH,
                confidence=0.8,
                session=SessionType.NEW_YORK,
                rationale="New York session with maximum liquidity"
            ))

            # Overlap window (premium)
            overlap_start = datetime.combine(date.date(), time(13, 0))
            overlap_end = datetime.combine(date.date(), time(17, 0))

            windows.append(BiasWindow(
                start_time=overlap_start,
                end_time=overlap_end,
                bias_direction=BiasDirection.RANGING,
                confidence=0.9,
                session=SessionType.LONDON,  # Considered London for overlap
                rationale="London-NY overlap with maximum liquidity and volatility"
            ))

            return windows

        except Exception as e:
            self.logger.error(f"Error getting optimal trading windows: {e}")
            return []

    def get_session_info(self, session_type: SessionType) -> Optional[SessionInfo]:
        """Get information about a specific trading session"""
        return self.sessions.get(session_type)

    def is_optimal_time_for_direction(
        self,
        current_time: datetime,
        signal_direction: str
    ) -> Tuple[bool, float]:
        """Check if current time is optimal for given signal direction"""
        try:
            current_bias = self._determine_current_bias(current_time, {"direction": signal_direction})

            optimal = False
            confidence = 0.0

            if signal_direction == "long" and current_bias == BiasDirection.BULLISH:
                optimal = True
                confidence = 0.8
            elif signal_direction == "short" and current_bias == BiasDirection.BEARISH:
                optimal = True
                confidence = 0.8
            elif current_bias in [BiasDirection.NEUTRAL, BiasDirection.RANGING]:
                optimal = True
                confidence = 0.5

            return optimal, confidence

        except Exception as e:
            self.logger.error(f"Error checking optimal time: {e}")
            return False, 0.0