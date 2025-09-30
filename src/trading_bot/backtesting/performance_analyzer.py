"""
PerformanceAnalyzer: Comprehensive performance metrics calculation for backtesting.

Calculates Sharpe ratio, maximum drawdown, win rate, average profit/loss ratio,
annualized returns, volatility, and statistical significance metrics.
"""

import math
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np


class PerformanceAnalyzer:
    """
    Performance metrics analyzer for backtest results.

    Calculates key performance indicators including:
    - Sharpe Ratio
    - Maximum Drawdown
    - Win Rate
    - Average Profit/Loss Ratio
    - Annualized Returns
    - Volatility
    - Statistical significance metrics
    """

    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize performance analyzer.

        Args:
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculation (default: 2%)
        """
        self.risk_free_rate = risk_free_rate

    def analyze(
        self,
        equity_curve: List[Dict],
        trades: List[Dict],
        initial_capital: Decimal,
        benchmark_returns: Optional[List[float]] = None,
    ) -> Dict:
        """
        Perform comprehensive performance analysis.

        Args:
            equity_curve: List of equity curve points
            trades: List of closed trades
            initial_capital: Starting capital
            benchmark_returns: Optional benchmark returns for comparison

        Returns:
            Dictionary with all performance metrics
        """
        if not equity_curve or not trades:
            return self._empty_results()

        # Convert equity curve to DataFrame
        df_equity = pd.DataFrame(equity_curve)
        df_equity["equity"] = df_equity["equity"].apply(Decimal)

        # Calculate returns
        returns = self._calculate_returns(df_equity)

        # Core metrics
        sharpe = self.calculate_sharpe_ratio(returns)
        sortino = self.calculate_sortino_ratio(returns)
        max_dd, max_dd_duration = self.calculate_max_drawdown(df_equity)
        calmar = self.calculate_calmar_ratio(returns, max_dd)

        # Trade statistics
        win_rate = self.calculate_win_rate(trades)
        avg_profit_loss = self.calculate_avg_profit_loss_ratio(trades)
        profit_factor = self.calculate_profit_factor(trades)

        # Return metrics
        total_return = self.calculate_total_return(df_equity, initial_capital)
        ann_return = self.calculate_annualized_return(df_equity, initial_capital)
        ann_volatility = self.calculate_annualized_volatility(returns)

        # Advanced metrics
        var_95 = self.calculate_value_at_risk(returns, confidence=0.95)
        cvar_95 = self.calculate_conditional_var(returns, confidence=0.95)

        # Benchmark comparison
        benchmark_metrics = {}
        if benchmark_returns:
            benchmark_metrics = self._calculate_benchmark_comparison(returns, benchmark_returns)

        # Statistical significance
        statistical_tests = self._perform_statistical_tests(trades)

        # Time-based analysis
        time_analysis = self._analyze_by_time_period(df_equity, trades)

        return {
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "max_drawdown": max_dd,
            "max_drawdown_duration_days": max_dd_duration,
            "win_rate": win_rate,
            "avg_profit_loss_ratio": avg_profit_loss,
            "profit_factor": profit_factor,
            "total_return_pct": total_return,
            "annualized_return_pct": ann_return,
            "annualized_volatility_pct": ann_volatility,
            "value_at_risk_95": var_95,
            "conditional_var_95": cvar_95,
            **benchmark_metrics,
            **statistical_tests,
            **time_analysis,
        }

    def calculate_sharpe_ratio(
        self,
        returns: pd.Series,
        periods_per_year: int = 252,
    ) -> float:
        """
        Calculate Sharpe ratio.

        Args:
            returns: Series of period returns
            periods_per_year: Number of trading periods per year

        Returns:
            Sharpe ratio (annualized)
        """
        if len(returns) < 2:
            return 0.0

        # Calculate excess returns
        daily_rf = (1 + self.risk_free_rate) ** (1 / periods_per_year) - 1
        excess_returns = returns - daily_rf

        # Sharpe ratio
        if excess_returns.std() == 0:
            return 0.0

        sharpe = excess_returns.mean() / excess_returns.std()
        return float(sharpe * np.sqrt(periods_per_year))

    def calculate_sortino_ratio(
        self,
        returns: pd.Series,
        periods_per_year: int = 252,
    ) -> float:
        """
        Calculate Sortino ratio (uses downside deviation).

        Args:
            returns: Series of period returns
            periods_per_year: Number of trading periods per year

        Returns:
            Sortino ratio (annualized)
        """
        if len(returns) < 2:
            return 0.0

        # Calculate excess returns
        daily_rf = (1 + self.risk_free_rate) ** (1 / periods_per_year) - 1
        excess_returns = returns - daily_rf

        # Downside deviation (only negative returns)
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0:
            return float('inf') if excess_returns.mean() > 0 else 0.0

        downside_dev = downside_returns.std()
        if downside_dev == 0:
            return 0.0

        sortino = excess_returns.mean() / downside_dev
        return float(sortino * np.sqrt(periods_per_year))

    def calculate_max_drawdown(
        self,
        equity_curve: pd.DataFrame,
    ) -> Tuple[float, int]:
        """
        Calculate maximum drawdown and its duration.

        Args:
            equity_curve: DataFrame with equity values

        Returns:
            Tuple of (max_drawdown_pct, duration_days)
        """
        equity = equity_curve["equity"].astype(float)

        # Calculate running maximum
        running_max = equity.expanding().max()

        # Calculate drawdown
        drawdown = (equity - running_max) / running_max * 100

        # Maximum drawdown
        max_dd = abs(drawdown.min())

        # Calculate duration
        max_dd_duration = 0
        current_dd_duration = 0

        for dd in drawdown:
            if dd < 0:
                current_dd_duration += 1
                max_dd_duration = max(max_dd_duration, current_dd_duration)
            else:
                current_dd_duration = 0

        return float(max_dd), max_dd_duration

    def calculate_calmar_ratio(
        self,
        returns: pd.Series,
        max_drawdown: float,
    ) -> float:
        """
        Calculate Calmar ratio (annualized return / max drawdown).

        Args:
            returns: Series of period returns
            max_drawdown: Maximum drawdown percentage

        Returns:
            Calmar ratio
        """
        if max_drawdown == 0:
            return 0.0

        ann_return = float(returns.mean() * 252 * 100)  # Annualized in percentage
        return ann_return / max_drawdown

    def calculate_win_rate(self, trades: List[Dict]) -> float:
        """
        Calculate win rate (percentage of profitable trades).

        Args:
            trades: List of trade dictionaries

        Returns:
            Win rate percentage
        """
        if not trades:
            return 0.0

        winning_trades = sum(
            1 for trade in trades
            if trade.get("pnl") and Decimal(trade["pnl"]) > 0
        )

        return (winning_trades / len(trades)) * 100

    def calculate_avg_profit_loss_ratio(self, trades: List[Dict]) -> float:
        """
        Calculate average profit/loss ratio.

        Args:
            trades: List of trade dictionaries

        Returns:
            Average profit/loss ratio
        """
        if not trades:
            return 0.0

        winning_trades = [
            Decimal(trade["pnl"])
            for trade in trades
            if trade.get("pnl") and Decimal(trade["pnl"]) > 0
        ]

        losing_trades = [
            abs(Decimal(trade["pnl"]))
            for trade in trades
            if trade.get("pnl") and Decimal(trade["pnl"]) < 0
        ]

        if not winning_trades or not losing_trades:
            return 0.0

        avg_win = sum(winning_trades) / len(winning_trades)
        avg_loss = sum(losing_trades) / len(losing_trades)

        if avg_loss == 0:
            return float('inf')

        return float(avg_win / avg_loss)

    def calculate_profit_factor(self, trades: List[Dict]) -> float:
        """
        Calculate profit factor (gross profit / gross loss).

        Args:
            trades: List of trade dictionaries

        Returns:
            Profit factor
        """
        if not trades:
            return 0.0

        gross_profit = sum(
            Decimal(trade["pnl"])
            for trade in trades
            if trade.get("pnl") and Decimal(trade["pnl"]) > 0
        )

        gross_loss = abs(sum(
            Decimal(trade["pnl"])
            for trade in trades
            if trade.get("pnl") and Decimal(trade["pnl"]) < 0
        ))

        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0

        return float(gross_profit / gross_loss)

    def calculate_total_return(
        self,
        equity_curve: pd.DataFrame,
        initial_capital: Decimal,
    ) -> float:
        """Calculate total return percentage."""
        final_equity = equity_curve.iloc[-1]["equity"]
        return float(((final_equity - initial_capital) / initial_capital) * 100)

    def calculate_annualized_return(
        self,
        equity_curve: pd.DataFrame,
        initial_capital: Decimal,
    ) -> float:
        """Calculate annualized return percentage."""
        total_return = self.calculate_total_return(equity_curve, initial_capital)

        # Calculate years
        start_date = equity_curve.iloc[0]["timestamp"]
        end_date = equity_curve.iloc[-1]["timestamp"]
        years = (end_date - start_date).days / 365.25

        if years == 0:
            return 0.0

        # Annualized return
        ann_return = ((1 + total_return / 100) ** (1 / years) - 1) * 100
        return float(ann_return)

    def calculate_annualized_volatility(
        self,
        returns: pd.Series,
        periods_per_year: int = 252,
    ) -> float:
        """Calculate annualized volatility (standard deviation of returns)."""
        if len(returns) < 2:
            return 0.0

        return float(returns.std() * np.sqrt(periods_per_year) * 100)

    def calculate_value_at_risk(
        self,
        returns: pd.Series,
        confidence: float = 0.95,
    ) -> float:
        """
        Calculate Value at Risk (VaR) at given confidence level.

        Args:
            returns: Series of returns
            confidence: Confidence level (e.g., 0.95 for 95%)

        Returns:
            VaR as percentage
        """
        if len(returns) < 10:
            return 0.0

        return float(np.percentile(returns, (1 - confidence) * 100) * 100)

    def calculate_conditional_var(
        self,
        returns: pd.Series,
        confidence: float = 0.95,
    ) -> float:
        """
        Calculate Conditional Value at Risk (CVaR/Expected Shortfall).

        Args:
            returns: Series of returns
            confidence: Confidence level

        Returns:
            CVaR as percentage
        """
        if len(returns) < 10:
            return 0.0

        var_threshold = np.percentile(returns, (1 - confidence) * 100)
        tail_returns = returns[returns <= var_threshold]

        if len(tail_returns) == 0:
            return 0.0

        return float(tail_returns.mean() * 100)

    def _calculate_returns(self, equity_curve: pd.DataFrame) -> pd.Series:
        """Calculate period-over-period returns."""
        equity = equity_curve["equity"].astype(float)
        returns = equity.pct_change().dropna()
        return returns

    def _calculate_benchmark_comparison(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: List[float],
    ) -> Dict:
        """Calculate metrics comparing strategy to benchmark."""
        if len(strategy_returns) != len(benchmark_returns):
            return {}

        benchmark_series = pd.Series(benchmark_returns)

        # Alpha and Beta
        covariance = np.cov(strategy_returns, benchmark_series)[0][1]
        benchmark_variance = np.var(benchmark_series)

        beta = covariance / benchmark_variance if benchmark_variance != 0 else 0

        strategy_return = strategy_returns.mean() * 252
        benchmark_return = benchmark_series.mean() * 252
        alpha = strategy_return - (self.risk_free_rate + beta * (benchmark_return - self.risk_free_rate))

        # Information Ratio
        excess_returns = strategy_returns - benchmark_series
        tracking_error = excess_returns.std() * np.sqrt(252)
        information_ratio = (excess_returns.mean() * 252) / tracking_error if tracking_error != 0 else 0

        return {
            "alpha": float(alpha),
            "beta": float(beta),
            "information_ratio": float(information_ratio),
            "tracking_error": float(tracking_error),
        }

    def _perform_statistical_tests(self, trades: List[Dict]) -> Dict:
        """Perform statistical significance tests."""
        if len(trades) < 30:
            return {"statistical_significance": "insufficient_data"}

        pnl_values = [
            float(Decimal(trade["pnl"]))
            for trade in trades
            if trade.get("pnl")
        ]

        if not pnl_values:
            return {"statistical_significance": "no_pnl_data"}

        # T-test for mean different from zero
        t_statistic = np.mean(pnl_values) / (np.std(pnl_values) / np.sqrt(len(pnl_values)))

        # Approximate p-value (two-tailed)
        from scipy import stats
        p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), len(pnl_values) - 1))

        return {
            "t_statistic": float(t_statistic),
            "p_value": float(p_value),
            "statistical_significance": "significant" if p_value < 0.05 else "not_significant",
        }

    def _analyze_by_time_period(
        self,
        equity_curve: pd.DataFrame,
        trades: List[Dict],
    ) -> Dict:
        """Analyze performance by time periods (daily, monthly, yearly)."""
        df = equity_curve.copy()
        df["date"] = pd.to_datetime(df["timestamp"])
        df["equity"] = df["equity"].astype(float)

        # Daily returns
        df["daily_return"] = df["equity"].pct_change()

        # Monthly aggregation
        monthly = df.groupby(df["date"].dt.to_period("M")).agg({
            "equity": "last",
            "daily_return": "sum"
        })

        # Yearly aggregation
        yearly = df.groupby(df["date"].dt.year).agg({
            "equity": "last",
            "daily_return": "sum"
        })

        return {
            "best_day_return": float(df["daily_return"].max() * 100) if len(df) > 0 else 0.0,
            "worst_day_return": float(df["daily_return"].min() * 100) if len(df) > 0 else 0.0,
            "best_month_return": float(monthly["daily_return"].max() * 100) if len(monthly) > 0 else 0.0,
            "worst_month_return": float(monthly["daily_return"].min() * 100) if len(monthly) > 0 else 0.0,
            "positive_months": int((monthly["daily_return"] > 0).sum()) if len(monthly) > 0 else 0,
            "negative_months": int((monthly["daily_return"] < 0).sum()) if len(monthly) > 0 else 0,
        }

    def _empty_results(self) -> Dict:
        """Return empty results for cases with no data."""
        return {
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_duration_days": 0,
            "win_rate": 0.0,
            "avg_profit_loss_ratio": 0.0,
            "profit_factor": 0.0,
            "total_return_pct": 0.0,
            "annualized_return_pct": 0.0,
            "annualized_volatility_pct": 0.0,
            "value_at_risk_95": 0.0,
            "conditional_var_95": 0.0,
        }