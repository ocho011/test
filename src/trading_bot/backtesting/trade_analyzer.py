"""
TradeAnalyzer: Trade-level analysis and pattern recognition.

Provides detailed analysis of individual trades including hold duration,
entry/exit patterns, consecutive wins/losses, time-of-day performance,
and trade quality scoring.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Tuple
from collections import defaultdict

import pandas as pd
import numpy as np


class TradePattern:
    """Enumeration of recognized trade patterns."""
    MOMENTUM = "momentum"
    REVERSAL = "reversal"
    BREAKOUT = "breakout"
    PULLBACK = "pullback"
    UNKNOWN = "unknown"


class TradeAnalyzer:
    """
    Trade-level analyzer for detailed trade analysis and pattern recognition.

    Analyzes:
    - Individual trade P&L and holding periods
    - Entry and exit patterns
    - Consecutive wins/losses
    - Time-of-day and day-of-week performance
    - Trade frequency optimization
    - Trade quality scoring
    """

    def __init__(self):
        """Initialize trade analyzer."""
        pass

    def analyze(self, trades: List[Dict]) -> Dict:
        """
        Perform comprehensive trade-level analysis.

        Args:
            trades: List of trade dictionaries

        Returns:
            Dictionary with detailed trade analytics
        """
        if not trades:
            return self._empty_results()

        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(trades)
        df["pnl"] = df["pnl"].apply(lambda x: Decimal(x) if x else Decimal("0"))
        df["entry_time"] = pd.to_datetime(df["entry_time"])
        df["exit_time"] = pd.to_datetime(df["exit_time"])

        # Calculate holding duration
        df["holding_duration"] = (df["exit_time"] - df["entry_time"]).dt.total_seconds() / 3600  # hours

        # Trade outcomes
        df["is_winner"] = df["pnl"] > 0

        # Basic statistics
        basic_stats = self._calculate_basic_statistics(df)

        # Holding period analysis
        holding_analysis = self._analyze_holding_periods(df)

        # Entry/Exit pattern analysis
        pattern_analysis = self._analyze_entry_exit_patterns(df)

        # Consecutive analysis
        consecutive_analysis = self._analyze_consecutive_trades(df)

        # Time-based analysis
        time_analysis = self._analyze_by_time_of_day(df)
        dow_analysis = self._analyze_by_day_of_week(df)

        # Trade frequency optimization
        frequency_analysis = self._analyze_trade_frequency(df)

        # Trade quality scoring
        quality_scores = self._calculate_trade_quality_scores(trades)

        # MAE/MFE analysis
        mae_mfe_analysis = self._analyze_mae_mfe(trades)

        return {
            **basic_stats,
            **holding_analysis,
            **pattern_analysis,
            **consecutive_analysis,
            **time_analysis,
            **dow_analysis,
            **frequency_analysis,
            "quality_scores": quality_scores,
            **mae_mfe_analysis,
        }

    def _calculate_basic_statistics(self, df: pd.DataFrame) -> Dict:
        """Calculate basic trade statistics."""
        pnl_series = df["pnl"].astype(float)

        return {
            "total_trades": len(df),
            "winning_trades": int(df["is_winner"].sum()),
            "losing_trades": int((~df["is_winner"]).sum()),
            "avg_trade_pnl": float(pnl_series.mean()),
            "median_trade_pnl": float(pnl_series.median()),
            "std_trade_pnl": float(pnl_series.std()),
            "best_trade_pnl": float(pnl_series.max()),
            "worst_trade_pnl": float(pnl_series.min()),
            "avg_winning_trade": float(pnl_series[df["is_winner"]].mean()) if df["is_winner"].any() else 0.0,
            "avg_losing_trade": float(pnl_series[~df["is_winner"]].mean()) if (~df["is_winner"]).any() else 0.0,
        }

    def _analyze_holding_periods(self, df: pd.DataFrame) -> Dict:
        """Analyze trade holding period patterns."""
        holding_hours = df["holding_duration"]

        winning_holds = holding_hours[df["is_winner"]]
        losing_holds = holding_hours[~df["is_winner"]]

        return {
            "avg_holding_hours": float(holding_hours.mean()),
            "median_holding_hours": float(holding_hours.median()),
            "min_holding_hours": float(holding_hours.min()),
            "max_holding_hours": float(holding_hours.max()),
            "avg_winning_hold_hours": float(winning_holds.mean()) if len(winning_holds) > 0 else 0.0,
            "avg_losing_hold_hours": float(losing_holds.mean()) if len(losing_holds) > 0 else 0.0,
            "hold_duration_correlation_with_pnl": float(df["holding_duration"].corr(df["pnl"].astype(float))),
        }

    def _analyze_entry_exit_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze entry and exit patterns."""
        # Extract hour of day for entry and exit
        df["entry_hour"] = df["entry_time"].dt.hour
        df["exit_hour"] = df["exit_time"].dt.hour

        # Best entry hours (by average P&L)
        entry_performance = df.groupby("entry_hour")["pnl"].apply(lambda x: float(x.astype(float).mean()))
        best_entry_hours = entry_performance.nlargest(3).index.tolist()

        # Best exit hours
        exit_performance = df.groupby("exit_hour")["pnl"].apply(lambda x: float(x.astype(float).mean()))
        best_exit_hours = exit_performance.nlargest(3).index.tolist()

        # Side analysis
        if "side" in df.columns:
            side_performance = df.groupby("side").agg({
                "pnl": lambda x: float(x.astype(float).mean()),
                "trade_id": "count"
            })
            side_stats = {
                f"{side}_avg_pnl": perf
                for side, perf in side_performance["pnl"].items()
            }
            side_counts = {
                f"{side}_trade_count": int(count)
                for side, count in side_performance["trade_id"].items()
            }
        else:
            side_stats = {}
            side_counts = {}

        return {
            "best_entry_hours": best_entry_hours,
            "best_exit_hours": best_exit_hours,
            **side_stats,
            **side_counts,
        }

    def _analyze_consecutive_trades(self, df: pd.DataFrame) -> Dict:
        """Analyze consecutive wins and losses."""
        # Sort by entry time
        df_sorted = df.sort_values("entry_time")

        # Calculate consecutive runs
        consecutive_wins = []
        consecutive_losses = []

        current_win_streak = 0
        current_loss_streak = 0

        for is_winner in df_sorted["is_winner"]:
            if is_winner:
                current_win_streak += 1
                if current_loss_streak > 0:
                    consecutive_losses.append(current_loss_streak)
                    current_loss_streak = 0
            else:
                current_loss_streak += 1
                if current_win_streak > 0:
                    consecutive_wins.append(current_win_streak)
                    current_win_streak = 0

        # Add final streaks
        if current_win_streak > 0:
            consecutive_wins.append(current_win_streak)
        if current_loss_streak > 0:
            consecutive_losses.append(current_loss_streak)

        return {
            "max_consecutive_wins": max(consecutive_wins) if consecutive_wins else 0,
            "max_consecutive_losses": max(consecutive_losses) if consecutive_losses else 0,
            "avg_consecutive_wins": float(np.mean(consecutive_wins)) if consecutive_wins else 0.0,
            "avg_consecutive_losses": float(np.mean(consecutive_losses)) if consecutive_losses else 0.0,
            "consecutive_win_streaks": len(consecutive_wins),
            "consecutive_loss_streaks": len(consecutive_losses),
        }

    def _analyze_by_time_of_day(self, df: pd.DataFrame) -> Dict:
        """Analyze performance by time of day."""
        df["hour"] = df["entry_time"].dt.hour

        # Group into time periods
        df["time_period"] = df["hour"].apply(self._classify_time_period)

        period_performance = df.groupby("time_period").agg({
            "pnl": lambda x: float(x.astype(float).mean()),
            "is_winner": lambda x: float(x.mean() * 100),  # win rate
            "trade_id": "count"
        }).to_dict()

        return {
            "time_period_avg_pnl": {
                period: pnl
                for period, pnl in period_performance["pnl"].items()
            },
            "time_period_win_rate": {
                period: wr
                for period, wr in period_performance["is_winner"].items()
            },
            "time_period_trade_count": {
                period: int(count)
                for period, count in period_performance["trade_id"].items()
            },
        }

    def _analyze_by_day_of_week(self, df: pd.DataFrame) -> Dict:
        """Analyze performance by day of week."""
        df["day_of_week"] = df["entry_time"].dt.day_name()

        dow_performance = df.groupby("day_of_week").agg({
            "pnl": lambda x: float(x.astype(float).mean()),
            "is_winner": lambda x: float(x.mean() * 100),
            "trade_id": "count"
        }).to_dict()

        return {
            "day_of_week_avg_pnl": dow_performance["pnl"],
            "day_of_week_win_rate": dow_performance["is_winner"],
            "day_of_week_trade_count": {k: int(v) for k, v in dow_performance["trade_id"].items()},
        }

    def _analyze_trade_frequency(self, df: pd.DataFrame) -> Dict:
        """Analyze trade frequency and its impact on performance."""
        # Calculate trades per day
        df["date"] = df["entry_time"].dt.date
        daily_trade_counts = df.groupby("date").size()

        # Performance by frequency
        df["trades_on_day"] = df["date"].map(daily_trade_counts)

        frequency_performance = df.groupby("trades_on_day").agg({
            "pnl": lambda x: float(x.astype(float).mean()),
            "is_winner": lambda x: float(x.mean() * 100),
        }).to_dict()

        return {
            "avg_trades_per_day": float(daily_trade_counts.mean()),
            "median_trades_per_day": float(daily_trade_counts.median()),
            "max_trades_per_day": int(daily_trade_counts.max()),
            "min_trades_per_day": int(daily_trade_counts.min()),
            "frequency_avg_pnl": frequency_performance["pnl"],
            "frequency_win_rate": frequency_performance["is_winner"],
        }

    def _calculate_trade_quality_scores(self, trades: List[Dict]) -> List[Dict]:
        """
        Calculate quality scores for individual trades.

        Score factors:
        - Risk-reward ratio achievement
        - Holding duration optimality
        - MAE/MFE ratio (efficiency of exit)
        - P&L relative to average
        - Stop loss/take profit usage
        """
        quality_scores = []

        for trade in trades:
            score_components = {}

            # P&L score (normalized to 0-100)
            pnl = float(Decimal(trade.get("pnl", "0")))
            pnl_pct = trade.get("pnl_percentage", 0)

            if pnl_pct:
                if pnl_pct > 0:
                    pnl_score = min(100, pnl_pct * 10)  # 10% = 100 points
                else:
                    pnl_score = max(0, 100 + pnl_pct * 10)  # -10% = 0 points
            else:
                pnl_score = 50

            score_components["pnl_score"] = pnl_score

            # Risk management score
            has_stop_loss = trade.get("stop_loss") is not None
            has_take_profit = trade.get("take_profit") is not None

            risk_mgmt_score = 0
            if has_stop_loss:
                risk_mgmt_score += 50
            if has_take_profit:
                risk_mgmt_score += 50

            score_components["risk_management_score"] = risk_mgmt_score

            # MAE/MFE efficiency score
            mae = abs(float(Decimal(trade.get("mae", "0"))))
            mfe = abs(float(Decimal(trade.get("mfe", "0"))))

            if mfe > 0:
                efficiency = (mfe - mae) / mfe * 100
                efficiency_score = max(0, min(100, 50 + efficiency))
            else:
                efficiency_score = 50

            score_components["efficiency_score"] = efficiency_score

            # Overall quality score (weighted average)
            overall_score = (
                pnl_score * 0.4 +
                risk_mgmt_score * 0.3 +
                efficiency_score * 0.3
            )

            quality_scores.append({
                "trade_id": trade["trade_id"],
                "overall_quality_score": overall_score,
                **score_components,
            })

        return quality_scores

    def _analyze_mae_mfe(self, trades: List[Dict]) -> Dict:
        """Analyze Maximum Adverse Excursion and Maximum Favorable Excursion."""
        if not trades:
            return {}

        mae_values = [abs(float(Decimal(t.get("mae", "0")))) for t in trades]
        mfe_values = [abs(float(Decimal(t.get("mfe", "0")))) for t in trades]

        winning_trades = [t for t in trades if Decimal(t.get("pnl", "0")) > 0]
        losing_trades = [t for t in trades if Decimal(t.get("pnl", "0")) < 0]

        winning_mae = [abs(float(Decimal(t.get("mae", "0")))) for t in winning_trades]
        winning_mfe = [abs(float(Decimal(t.get("mfe", "0")))) for t in winning_trades]

        losing_mae = [abs(float(Decimal(t.get("mae", "0")))) for t in losing_trades]
        losing_mfe = [abs(float(Decimal(t.get("mfe", "0")))) for t in losing_trades]

        return {
            "avg_mae": float(np.mean(mae_values)) if mae_values else 0.0,
            "avg_mfe": float(np.mean(mfe_values)) if mfe_values else 0.0,
            "avg_winning_mae": float(np.mean(winning_mae)) if winning_mae else 0.0,
            "avg_winning_mfe": float(np.mean(winning_mfe)) if winning_mfe else 0.0,
            "avg_losing_mae": float(np.mean(losing_mae)) if losing_mae else 0.0,
            "avg_losing_mfe": float(np.mean(losing_mfe)) if losing_mfe else 0.0,
            "mfe_mae_ratio": (
                float(np.mean(mfe_values) / np.mean(mae_values))
                if mae_values and np.mean(mae_values) != 0 else 0.0
            ),
        }

    @staticmethod
    def _classify_time_period(hour: int) -> str:
        """Classify hour into trading session period."""
        if 0 <= hour < 6:
            return "asia_session"
        elif 6 <= hour < 14:
            return "europe_session"
        elif 14 <= hour < 22:
            return "us_session"
        else:
            return "late_us_session"

    def _empty_results(self) -> Dict:
        """Return empty results when no trades available."""
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "message": "No trades to analyze",
        }