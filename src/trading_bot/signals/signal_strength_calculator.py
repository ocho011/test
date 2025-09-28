"""
Signal Strength Calculator for ICT Trading Signals

This module implements comprehensive signal strength scoring based on multiple
factors including pattern quality, market context, risk-reward ratios, and
historical performance metrics.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..analysis.ict_analyzer import PatternValidationEngine


class StrengthCategory(Enum):
    """Categories for signal strength assessment"""

    PATTERN_QUALITY = "pattern_quality"
    MARKET_CONTEXT = "market_context"
    RISK_REWARD = "risk_reward"
    VOLUME_PROFILE = "volume_profile"
    MOMENTUM = "momentum"
    STRUCTURAL_BIAS = "structural_bias"
    HISTORICAL_PERFORMANCE = "historical_performance"


class StrengthLevel(Enum):
    """Signal strength levels"""

    VERY_WEAK = "very_weak"
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


@dataclass
class StrengthScore:
    """Individual strength score for a category"""

    category: StrengthCategory
    score: float  # 0.0 to 1.0
    weight: float
    details: Dict[str, Any]
    rationale: str


@dataclass
class SignalStrength:
    """Overall signal strength assessment"""

    overall_score: float  # 0.0 to 1.0
    strength_level: StrengthLevel
    category_scores: List[StrengthScore]
    weighted_score: float
    confidence_interval: Tuple[float, float]
    calculation_timestamp: datetime
    details: Dict[str, Any]


@dataclass
class StrengthConfig:
    """Configuration for signal strength calculation"""

    pattern_quality_weight: float = 0.25
    market_context_weight: float = 0.20
    risk_reward_weight: float = 0.20
    volume_profile_weight: float = 0.15
    momentum_weight: float = 0.10
    structural_bias_weight: float = 0.10
    historical_performance_weight: float = 0.00  # Disabled initially
    minimum_risk_reward_ratio: float = 1.5
    volume_threshold: float = 1.2
    momentum_threshold: float = 0.6


class SignalStrengthCalculator:
    """
    Comprehensive signal strength scoring system for ICT trading signals.

    This calculator evaluates signal strength across multiple dimensions
    to provide a robust assessment of signal quality and probability of success.
    """

    def __init__(
        self,
        pattern_validator: PatternValidationEngine,
        config: Optional[StrengthConfig] = None,
    ):
        self.pattern_validator = pattern_validator
        self.config = config or StrengthConfig()
        self.logger = logging.getLogger(__name__)

        # Historical performance tracking (simplified for now)
        self.performance_history = []

    def calculate_signal_strength(
        self,
        signal_data: Dict[str, Any],
        market_data: Dict[str, pd.DataFrame],
        confluence_result: Optional[Dict[str, Any]] = None,
    ) -> SignalStrength:
        """
        Calculate comprehensive signal strength score.

        Args:
            signal_data: Signal information including patterns and levels
            market_data: Multi-timeframe market data
            confluence_result: Optional confluence validation result

        Returns:
            SignalStrength with detailed scoring breakdown
        """
        try:
            category_scores = []

            # Calculate pattern quality score
            pattern_score = self._calculate_pattern_quality_score(
                signal_data, market_data
            )
            category_scores.append(pattern_score)

            # Calculate market context score
            context_score = self._calculate_market_context_score(
                signal_data, market_data
            )
            category_scores.append(context_score)

            # Calculate risk-reward score
            risk_reward_score = self._calculate_risk_reward_score(signal_data)
            category_scores.append(risk_reward_score)

            # Calculate volume profile score
            volume_score = self._calculate_volume_profile_score(
                signal_data, market_data
            )
            category_scores.append(volume_score)

            # Calculate momentum score
            momentum_score = self._calculate_momentum_score(signal_data, market_data)
            category_scores.append(momentum_score)

            # Calculate structural bias score
            structural_score = self._calculate_structural_bias_score(
                signal_data, market_data
            )
            category_scores.append(structural_score)

            # Calculate historical performance score (if enabled)
            if self.config.historical_performance_weight > 0:
                historical_score = self._calculate_historical_performance_score(
                    signal_data
                )
                category_scores.append(historical_score)

            # Calculate weighted overall score
            weighted_score = self._calculate_weighted_score(category_scores)

            # Calculate unweighted average score
            overall_score = np.mean([score.score for score in category_scores])

            # Determine strength level
            strength_level = self._determine_strength_level(weighted_score)

            # Calculate confidence interval
            confidence_interval = self._calculate_confidence_interval(
                category_scores, confluence_result
            )

            return SignalStrength(
                overall_score=overall_score,
                strength_level=strength_level,
                category_scores=category_scores,
                weighted_score=weighted_score,
                confidence_interval=confidence_interval,
                calculation_timestamp=datetime.now(),
                details={
                    "signal_id": signal_data.get("id", "unknown"),
                    "pattern_count": len(signal_data.get("patterns", [])),
                    "timeframes_analyzed": len(market_data),
                    "confluence_available": confluence_result is not None,
                },
            )

        except Exception as e:
            self.logger.error(f"Error calculating signal strength: {e}")
            return self._create_error_strength_result(str(e))

    def _calculate_pattern_quality_score(
        self, signal_data: Dict[str, Any], market_data: Dict[str, pd.DataFrame]
    ) -> StrengthScore:
        """Calculate pattern quality score based on ICT pattern integrity"""
        try:
            patterns = signal_data.get("patterns", [])
            quality_scores = []
            details = {}

            if not patterns:
                return StrengthScore(
                    category=StrengthCategory.PATTERN_QUALITY,
                    score=0.0,
                    weight=self.config.pattern_quality_weight,
                    details={"error": "No patterns found"},
                    rationale="No patterns available for analysis",
                )

            # Evaluate each pattern type
            for pattern in patterns:
                pattern_type = pattern.get("type", "unknown")
                pattern_strength = pattern.get("strength", 0.5)

                if pattern_type == "order_block":
                    ob_quality = self._evaluate_order_block_quality(
                        pattern, market_data
                    )
                    quality_scores.append(ob_quality)
                    details["order_block_quality"] = ob_quality

                elif pattern_type == "fair_value_gap":
                    fvg_quality = self._evaluate_fvg_quality(pattern, market_data)
                    quality_scores.append(fvg_quality)
                    details["fvg_quality"] = fvg_quality

                elif pattern_type == "market_structure":
                    structure_quality = self._evaluate_structure_quality(
                        pattern, market_data
                    )
                    quality_scores.append(structure_quality)
                    details["structure_quality"] = structure_quality

                # Add base pattern strength
                quality_scores.append(pattern_strength)

            # Calculate overall pattern quality
            if quality_scores:
                pattern_quality = np.mean(quality_scores)
            else:
                pattern_quality = 0.5

            # Apply pattern diversity bonus
            unique_patterns = len(set(p.get("type", "unknown") for p in patterns))
            diversity_bonus = min(0.2, unique_patterns * 0.05)
            pattern_quality = min(1.0, pattern_quality + diversity_bonus)

            details.update(
                {
                    "pattern_count": len(patterns),
                    "unique_pattern_types": unique_patterns,
                    "diversity_bonus": diversity_bonus,
                    "individual_scores": quality_scores,
                }
            )

            rationale = f"Pattern quality based on {len(patterns)} patterns with {unique_patterns} unique types"

            return StrengthScore(
                category=StrengthCategory.PATTERN_QUALITY,
                score=pattern_quality,
                weight=self.config.pattern_quality_weight,
                details=details,
                rationale=rationale,
            )

        except Exception as e:
            self.logger.error(f"Error in pattern quality calculation: {e}")
            return StrengthScore(
                category=StrengthCategory.PATTERN_QUALITY,
                score=0.0,
                weight=self.config.pattern_quality_weight,
                details={"error": str(e)},
                rationale="Error in pattern quality analysis",
            )

    def _calculate_market_context_score(
        self, signal_data: Dict[str, Any], market_data: Dict[str, pd.DataFrame]
    ) -> StrengthScore:
        """Calculate market context score based on current market conditions"""
        try:
            context_factors = []
            details = {}

            # Evaluate market volatility
            volatility_score = self._evaluate_volatility_context(market_data)
            context_factors.append(volatility_score)
            details["volatility"] = volatility_score

            # Evaluate trend strength
            trend_score = self._evaluate_trend_strength(market_data)
            context_factors.append(trend_score)
            details["trend_strength"] = trend_score

            # Evaluate market phase
            phase_score = self._evaluate_market_phase(market_data)
            context_factors.append(phase_score)
            details["market_phase"] = phase_score

            # Evaluate session context
            session_score = self._evaluate_session_context(signal_data)
            context_factors.append(session_score)
            details["session_context"] = session_score

            # Calculate overall context score
            context_score = np.mean(context_factors)

            details["individual_factors"] = context_factors

            rationale = (
                "Market context based on volatility, trend, phase, and session analysis"
            )

            return StrengthScore(
                category=StrengthCategory.MARKET_CONTEXT,
                score=context_score,
                weight=self.config.market_context_weight,
                details=details,
                rationale=rationale,
            )

        except Exception as e:
            self.logger.error(f"Error in market context calculation: {e}")
            return StrengthScore(
                category=StrengthCategory.MARKET_CONTEXT,
                score=0.0,
                weight=self.config.market_context_weight,
                details={"error": str(e)},
                rationale="Error in market context analysis",
            )

    def _calculate_risk_reward_score(
        self, signal_data: Dict[str, Any]
    ) -> StrengthScore:
        """Calculate risk-reward score based on trade setup metrics"""
        try:
            entry_price = signal_data.get("entry_price", 0)
            stop_loss = signal_data.get("stop_loss", 0)
            take_profit = signal_data.get("take_profit", 0)

            if not all([entry_price, stop_loss, take_profit]):
                return StrengthScore(
                    category=StrengthCategory.RISK_REWARD,
                    score=0.0,
                    weight=self.config.risk_reward_weight,
                    details={"error": "Missing price levels"},
                    rationale="Incomplete price level information",
                )

            # Calculate risk and reward
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)

            if risk <= 0:
                risk_reward_ratio = 0.0
            else:
                risk_reward_ratio = reward / risk

            # Score based on risk-reward ratio
            if risk_reward_ratio >= self.config.minimum_risk_reward_ratio:
                # Scale score based on how much the ratio exceeds minimum
                excess_ratio = risk_reward_ratio - self.config.minimum_risk_reward_ratio
                rr_score = min(1.0, 0.7 + (excess_ratio * 0.1))
            else:
                # Penalize signals with poor risk-reward
                rr_score = max(
                    0.0,
                    risk_reward_ratio / self.config.minimum_risk_reward_ratio * 0.5,
                )

            # Calculate stop loss distance as percentage of entry price
            stop_loss_percentage = (risk / entry_price) * 100 if entry_price > 0 else 0

            details = {
                "risk_reward_ratio": risk_reward_ratio,
                "risk_amount": risk,
                "reward_amount": reward,
                "stop_loss_percentage": stop_loss_percentage,
                "minimum_required_ratio": self.config.minimum_risk_reward_ratio,
            }

            rationale = f"Risk-reward ratio of {risk_reward_ratio:.2f} with {stop_loss_percentage:.2f}% stop loss"

            return StrengthScore(
                category=StrengthCategory.RISK_REWARD,
                score=rr_score,
                weight=self.config.risk_reward_weight,
                details=details,
                rationale=rationale,
            )

        except Exception as e:
            self.logger.error(f"Error in risk-reward calculation: {e}")
            return StrengthScore(
                category=StrengthCategory.RISK_REWARD,
                score=0.0,
                weight=self.config.risk_reward_weight,
                details={"error": str(e)},
                rationale="Error in risk-reward analysis",
            )

    def _calculate_volume_profile_score(
        self, signal_data: Dict[str, Any], market_data: Dict[str, pd.DataFrame]
    ) -> StrengthScore:
        """Calculate volume profile score"""
        try:
            volume_factors = []
            details = {}

            for timeframe, data in market_data.items():
                if "volume" not in data.columns:
                    continue

                # Calculate volume metrics
                recent_volume = data["volume"].tail(10).mean()
                avg_volume = data["volume"].rolling(50).mean().iloc[-1]

                if avg_volume > 0:
                    volume_ratio = recent_volume / avg_volume
                    volume_factors.append(volume_ratio)

                    details[f"{timeframe}_volume_ratio"] = volume_ratio
                    details[f"{timeframe}_recent_volume"] = recent_volume
                    details[f"{timeframe}_avg_volume"] = avg_volume

            if not volume_factors:
                return StrengthScore(
                    category=StrengthCategory.VOLUME_PROFILE,
                    score=0.5,
                    weight=self.config.volume_profile_weight,
                    details={"warning": "No volume data available"},
                    rationale="Volume data not available for analysis",
                )

            # Calculate average volume ratio across timeframes
            avg_volume_ratio = np.mean(volume_factors)

            # Score based on volume threshold
            if avg_volume_ratio >= self.config.volume_threshold:
                volume_score = min(
                    1.0,
                    0.6 + (avg_volume_ratio - self.config.volume_threshold) * 0.2,
                )
            else:
                volume_score = max(
                    0.0, avg_volume_ratio / self.config.volume_threshold * 0.6
                )

            details.update(
                {
                    "average_volume_ratio": avg_volume_ratio,
                    "volume_threshold": self.config.volume_threshold,
                    "timeframes_analyzed": len(volume_factors),
                }
            )

            rationale = f"Volume analysis across {len(volume_factors)} timeframes with {avg_volume_ratio:.2f} average ratio"

            return StrengthScore(
                category=StrengthCategory.VOLUME_PROFILE,
                score=volume_score,
                weight=self.config.volume_profile_weight,
                details=details,
                rationale=rationale,
            )

        except Exception as e:
            self.logger.error(f"Error in volume profile calculation: {e}")
            return StrengthScore(
                category=StrengthCategory.VOLUME_PROFILE,
                score=0.0,
                weight=self.config.volume_profile_weight,
                details={"error": str(e)},
                rationale="Error in volume profile analysis",
            )

    def _calculate_momentum_score(
        self, signal_data: Dict[str, Any], market_data: Dict[str, pd.DataFrame]
    ) -> StrengthScore:
        """Calculate momentum score based on technical indicators"""
        try:
            momentum_factors = []
            details = {}
            signal_direction = signal_data.get("direction", "unknown")

            for timeframe, data in market_data.items():
                # Calculate RSI
                rsi = self._calculate_rsi(data["close"])
                current_rsi = rsi.iloc[-1]

                # Calculate MACD
                macd_line, macd_signal = self._calculate_macd(data["close"])
                macd_bullish = macd_line.iloc[-1] > macd_signal.iloc[-1]

                # Calculate momentum alignment score
                momentum_score = 0.0

                if signal_direction == "long":
                    # For long signals, prefer RSI between 40-70 and bullish MACD
                    if 40 <= current_rsi <= 70:
                        momentum_score += 0.5
                    elif 30 <= current_rsi <= 80:
                        momentum_score += 0.3

                    if macd_bullish:
                        momentum_score += 0.5

                elif signal_direction == "short":
                    # For short signals, prefer RSI between 30-60 and bearish MACD
                    if 30 <= current_rsi <= 60:
                        momentum_score += 0.5
                    elif 20 <= current_rsi <= 70:
                        momentum_score += 0.3

                    if not macd_bullish:
                        momentum_score += 0.5

                momentum_factors.append(momentum_score)
                details[f"{timeframe}_momentum"] = {
                    "rsi": current_rsi,
                    "macd_bullish": macd_bullish,
                    "momentum_score": momentum_score,
                }

            if not momentum_factors:
                return StrengthScore(
                    category=StrengthCategory.MOMENTUM,
                    score=0.5,
                    weight=self.config.momentum_weight,
                    details={"warning": "No momentum data calculated"},
                    rationale="Insufficient data for momentum analysis",
                )

            # Calculate average momentum score
            avg_momentum = np.mean(momentum_factors)

            details.update(
                {
                    "average_momentum": avg_momentum,
                    "signal_direction": signal_direction,
                    "timeframes_analyzed": len(momentum_factors),
                }
            )

            rationale = f"Momentum analysis for {signal_direction} signal across {len(momentum_factors)} timeframes"

            return StrengthScore(
                category=StrengthCategory.MOMENTUM,
                score=avg_momentum,
                weight=self.config.momentum_weight,
                details=details,
                rationale=rationale,
            )

        except Exception as e:
            self.logger.error(f"Error in momentum calculation: {e}")
            return StrengthScore(
                category=StrengthCategory.MOMENTUM,
                score=0.0,
                weight=self.config.momentum_weight,
                details={"error": str(e)},
                rationale="Error in momentum analysis",
            )

    def _calculate_structural_bias_score(
        self, signal_data: Dict[str, Any], market_data: Dict[str, pd.DataFrame]
    ) -> StrengthScore:
        """Calculate structural bias alignment score"""
        try:
            bias_factors = []
            details = {}
            signal_direction = signal_data.get("direction", "unknown")

            for timeframe, data in market_data.items():
                # Simple trend analysis using moving averages
                short_ma = data["close"].rolling(20).mean().iloc[-1]
                long_ma = data["close"].rolling(50).mean().iloc[-1]
                current_price = data["close"].iloc[-1]

                # Determine structural bias
                if short_ma > long_ma and current_price > short_ma:
                    structural_bias = "bullish"
                elif short_ma < long_ma and current_price < short_ma:
                    structural_bias = "bearish"
                else:
                    structural_bias = "neutral"

                # Calculate alignment score
                alignment_score = 0.0
                if signal_direction == "long" and structural_bias == "bullish":
                    alignment_score = 1.0
                elif signal_direction == "short" and structural_bias == "bearish":
                    alignment_score = 1.0
                elif structural_bias == "neutral":
                    alignment_score = 0.5

                bias_factors.append(alignment_score)
                details[f"{timeframe}_bias"] = {
                    "structural_bias": structural_bias,
                    "alignment_score": alignment_score,
                    "short_ma": short_ma,
                    "long_ma": long_ma,
                    "current_price": current_price,
                }

            if not bias_factors:
                return StrengthScore(
                    category=StrengthCategory.STRUCTURAL_BIAS,
                    score=0.5,
                    weight=self.config.structural_bias_weight,
                    details={"warning": "No structural bias data calculated"},
                    rationale="Insufficient data for structural bias analysis",
                )

            # Calculate average structural alignment
            avg_bias_score = np.mean(bias_factors)

            details.update(
                {
                    "average_bias_score": avg_bias_score,
                    "signal_direction": signal_direction,
                    "timeframes_analyzed": len(bias_factors),
                }
            )

            rationale = f"Structural bias alignment for {signal_direction} signal across {len(bias_factors)} timeframes"

            return StrengthScore(
                category=StrengthCategory.STRUCTURAL_BIAS,
                score=avg_bias_score,
                weight=self.config.structural_bias_weight,
                details=details,
                rationale=rationale,
            )

        except Exception as e:
            self.logger.error(f"Error in structural bias calculation: {e}")
            return StrengthScore(
                category=StrengthCategory.STRUCTURAL_BIAS,
                score=0.0,
                weight=self.config.structural_bias_weight,
                details={"error": str(e)},
                rationale="Error in structural bias analysis",
            )

    def _calculate_historical_performance_score(
        self, signal_data: Dict[str, Any]
    ) -> StrengthScore:
        """Calculate score based on historical performance of similar signals"""
        try:
            # This is a placeholder for historical performance tracking
            # In a real implementation, this would analyze past signal performance

            signal_patterns = signal_data.get("patterns", [])
            pattern_types = [p.get("type", "unknown") for p in signal_patterns]

            # Simplified scoring based on pattern type performance
            # This would be replaced with actual historical data
            performance_scores = {
                "order_block": 0.75,
                "fair_value_gap": 0.70,
                "market_structure": 0.65,
                "liquidity_sweep": 0.80,
            }

            type_scores = []
            for pattern_type in pattern_types:
                type_scores.append(performance_scores.get(pattern_type, 0.60))

            if type_scores:
                historical_score = np.mean(type_scores)
            else:
                historical_score = 0.60  # Default score

            details = {
                "pattern_types": pattern_types,
                "individual_scores": type_scores,
                "historical_data_points": len(self.performance_history),
            }

            rationale = (
                f"Historical performance based on {len(pattern_types)} pattern types"
            )

            return StrengthScore(
                category=StrengthCategory.HISTORICAL_PERFORMANCE,
                score=historical_score,
                weight=self.config.historical_performance_weight,
                details=details,
                rationale=rationale,
            )

        except Exception as e:
            self.logger.error(f"Error in historical performance calculation: {e}")
            return StrengthScore(
                category=StrengthCategory.HISTORICAL_PERFORMANCE,
                score=0.0,
                weight=self.config.historical_performance_weight,
                details={"error": str(e)},
                rationale="Error in historical performance analysis",
            )

    def _evaluate_order_block_quality(
        self, pattern: Dict[str, Any], market_data: Dict[str, pd.DataFrame]
    ) -> float:
        """Evaluate Order Block pattern quality"""
        # Base quality from pattern strength
        base_quality = pattern.get("strength", 0.5)

        # Additional quality factors
        touches = pattern.get("touches", 0)
        volume_confirmation = pattern.get("volume_confirmation", False)

        # Quality adjustments
        quality_score = base_quality

        # Bonus for multiple touches (shows respect)
        if touches > 1:
            quality_score += min(0.2, touches * 0.05)

        # Bonus for volume confirmation
        if volume_confirmation:
            quality_score += 0.1

        return min(1.0, quality_score)

    def _evaluate_fvg_quality(
        self, pattern: Dict[str, Any], market_data: Dict[str, pd.DataFrame]
    ) -> float:
        """Evaluate Fair Value Gap pattern quality"""
        # Base quality from pattern strength
        base_quality = pattern.get("strength", 0.5)

        # Gap size as percentage of price
        gap_size_pct = pattern.get("gap_size_percent", 0)

        # Quality adjustments
        quality_score = base_quality

        # Optimal gap size bonus (not too small, not too large)
        if 0.1 <= gap_size_pct <= 1.0:
            quality_score += 0.15
        elif 0.05 <= gap_size_pct <= 2.0:
            quality_score += 0.05

        return min(1.0, quality_score)

    def _evaluate_structure_quality(
        self, pattern: Dict[str, Any], market_data: Dict[str, pd.DataFrame]
    ) -> float:
        """Evaluate Market Structure pattern quality"""
        # Base quality from pattern strength
        base_quality = pattern.get("strength", 0.5)

        # Structure type importance
        structure_type = pattern.get("structure_type", "unknown")

        # Quality adjustments based on structure type
        quality_score = base_quality

        if structure_type in ["BOS", "CHoCH"]:
            quality_score += 0.1

        return min(1.0, quality_score)

    def _evaluate_volatility_context(
        self, market_data: Dict[str, pd.DataFrame]
    ) -> float:
        """Evaluate current market volatility context"""
        volatility_scores = []

        for timeframe, data in market_data.items():
            # Calculate ATR-based volatility
            high_low = data["high"] - data["low"]
            atr = high_low.rolling(14).mean()
            current_atr = atr.iloc[-1]
            avg_atr = atr.rolling(50).mean().iloc[-1]

            if avg_atr > 0:
                volatility_ratio = current_atr / avg_atr
                # Optimal volatility is moderate (not too low, not too high)
                if 0.8 <= volatility_ratio <= 1.5:
                    volatility_scores.append(1.0)
                elif 0.5 <= volatility_ratio <= 2.0:
                    volatility_scores.append(0.7)
                else:
                    volatility_scores.append(0.3)

        return np.mean(volatility_scores) if volatility_scores else 0.5

    def _evaluate_trend_strength(self, market_data: Dict[str, pd.DataFrame]) -> float:
        """Evaluate current trend strength"""
        trend_scores = []

        for timeframe, data in market_data.items():
            # Calculate trend strength using ADX-like measure
            close_prices = data["close"]

            # Simple trend strength calculation
            short_ma = close_prices.rolling(10).mean()
            long_ma = close_prices.rolling(30).mean()

            # Calculate slope of moving averages
            short_slope = (short_ma.iloc[-1] - short_ma.iloc[-5]) / 5
            long_slope = (long_ma.iloc[-1] - long_ma.iloc[-10]) / 10

            # Normalize slopes
            price_range = data["close"].rolling(50).std().iloc[-1]
            if price_range > 0:
                short_slope_norm = abs(short_slope) / price_range
                long_slope_norm = abs(long_slope) / price_range

                trend_strength = (short_slope_norm + long_slope_norm) / 2
                trend_scores.append(min(1.0, trend_strength))

        return np.mean(trend_scores) if trend_scores else 0.5

    def _evaluate_market_phase(self, market_data: Dict[str, pd.DataFrame]) -> float:
        """Evaluate current market phase (trending vs ranging)"""
        phase_scores = []

        for timeframe, data in market_data.items():
            # Calculate ranging vs trending using price action
            high_prices = data["high"].tail(20)
            low_prices = data["low"].tail(20)

            # Check for clear directional movement
            recent_high = high_prices.max()
            recent_low = low_prices.min()

            current_price = data["close"].iloc[-1]
            price_position = (current_price - recent_low) / (recent_high - recent_low)

            # Prefer clear directional bias (not stuck in middle)
            if price_position < 0.3 or price_position > 0.7:
                phase_scores.append(0.8)
            else:
                phase_scores.append(0.4)

        return np.mean(phase_scores) if phase_scores else 0.5

    def _evaluate_session_context(self, signal_data: Dict[str, Any]) -> float:
        """Evaluate trading session context"""
        signal_time = signal_data.get("timestamp", datetime.now())
        hour_utc = signal_time.hour

        # Score based on trading session quality
        # London session: 8:00-17:00 GMT
        # New York session: 13:00-22:00 GMT
        # Overlap: 13:00-17:00 GMT (best)

        if 13 <= hour_utc <= 17:  # Overlap period
            return 1.0
        elif 8 <= hour_utc <= 17 or 13 <= hour_utc <= 22:  # Main sessions
            return 0.7
        else:  # Off-hours
            return 0.3

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
        signal: int = 9,
    ) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        return macd_line, signal_line

    def _calculate_weighted_score(self, category_scores: List[StrengthScore]) -> float:
        """Calculate weighted overall score"""
        weighted_sum = sum(score.score * score.weight for score in category_scores)
        total_weight = sum(score.weight for score in category_scores)
        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _determine_strength_level(self, weighted_score: float) -> StrengthLevel:
        """Determine strength level based on score"""
        if weighted_score >= 0.85:
            return StrengthLevel.VERY_STRONG
        elif weighted_score >= 0.70:
            return StrengthLevel.STRONG
        elif weighted_score >= 0.55:
            return StrengthLevel.MODERATE
        elif weighted_score >= 0.35:
            return StrengthLevel.WEAK
        else:
            return StrengthLevel.VERY_WEAK

    def _calculate_confidence_interval(
        self,
        category_scores: List[StrengthScore],
        confluence_result: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, float]:
        """Calculate confidence interval for the strength score"""
        scores = [score.score for score in category_scores]
        mean_score = np.mean(scores)
        std_score = np.std(scores)

        # Simple confidence interval calculation
        confidence_width = std_score * 1.96  # 95% confidence interval

        lower_bound = max(0.0, mean_score - confidence_width)
        upper_bound = min(1.0, mean_score + confidence_width)

        # Adjust based on confluence result if available
        if confluence_result and confluence_result.get("is_valid"):
            confluence_confidence = confluence_result.get("confidence_score", 0.5)
            # Narrow the interval if confluence is high
            adjustment = (1.0 - confluence_confidence) * 0.1
            lower_bound = max(lower_bound, mean_score - adjustment)
            upper_bound = min(upper_bound, mean_score + adjustment)

        return (lower_bound, upper_bound)

    def _create_error_strength_result(self, error_message: str) -> SignalStrength:
        """Create error result for signal strength calculation"""
        return SignalStrength(
            overall_score=0.0,
            strength_level=StrengthLevel.VERY_WEAK,
            category_scores=[],
            weighted_score=0.0,
            confidence_interval=(0.0, 0.0),
            calculation_timestamp=datetime.now(),
            details={"error": error_message},
        )
