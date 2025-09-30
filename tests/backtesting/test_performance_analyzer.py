"""
Tests for PerformanceAnalyzer: Performance metrics calculation.
"""

import pytest
import pandas as pd
import numpy as np
from decimal import Decimal
from datetime import datetime, timedelta

from trading_bot.backtesting.performance_analyzer import PerformanceAnalyzer


@pytest.fixture
def performance_analyzer():
    """Create a PerformanceAnalyzer instance."""
    return PerformanceAnalyzer()


@pytest.fixture
def sample_equity_curve():
    """Generate sample equity curve data."""
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(252)]  # 1 year

    # Simulate equity curve with some volatility and overall growth
    initial_equity = 10000
    returns = np.random.normal(0.001, 0.02, len(dates))  # 0.1% daily return, 2% volatility
    cumulative_returns = (1 + returns).cumprod()
    equity_values = initial_equity * cumulative_returns

    df = pd.DataFrame({
        "timestamp": dates,
        "equity": equity_values
    })
    df.set_index("timestamp", inplace=True)

    return df


@pytest.fixture
def sample_trades():
    """Generate sample trade data."""
    trades = []

    for i in range(100):
        # Mix of winning and losing trades
        pnl = Decimal(str(np.random.normal(100, 500)))  # Some profit/loss variation

        trades.append({
            "trade_id": i + 1,
            "symbol": "BTCUSDT",
            "side": "LONG" if i % 2 == 0 else "SHORT",
            "entry_price": Decimal("50000"),
            "exit_price": Decimal("50000") + pnl,
            "entry_time": datetime(2024, 1, 1) + timedelta(hours=i),
            "exit_time": datetime(2024, 1, 1) + timedelta(hours=i+2),
            "size": Decimal("0.1"),
            "pnl": pnl,
            "pnl_percentage": float(pnl / Decimal("5000")),  # As percentage
            "commission": Decimal("5"),
            "exit_reason": "take_profit" if pnl > 0 else "stop_loss"
        })

    return trades


class TestPerformanceAnalyzer:
    """Test suite for PerformanceAnalyzer."""

    def test_initialization(self, performance_analyzer):
        """Test analyzer initialization."""
        assert performance_analyzer is not None
        assert performance_analyzer.risk_free_rate == 0.02  # Default 2%

    def test_sharpe_ratio_calculation(self, performance_analyzer):
        """Test Sharpe ratio calculation."""
        # Create returns with known characteristics
        returns = pd.Series([0.01, 0.02, -0.01, 0.015, 0.005] * 50)  # 250 trading days

        sharpe = performance_analyzer.calculate_sharpe_ratio(returns, periods_per_year=252)

        assert isinstance(sharpe, float)
        assert sharpe > 0  # Positive returns should have positive Sharpe

        # Test with negative returns
        negative_returns = pd.Series([-0.01, -0.02, -0.015] * 84)
        sharpe_negative = performance_analyzer.calculate_sharpe_ratio(negative_returns, periods_per_year=252)

        assert sharpe_negative < 0  # Negative returns should have negative Sharpe

    def test_sortino_ratio_calculation(self, performance_analyzer):
        """Test Sortino ratio calculation (downside deviation)."""
        # Create returns with asymmetric distribution
        returns = pd.Series([0.02, 0.01, -0.03, 0.015, -0.01] * 50)

        sortino = performance_analyzer.calculate_sortino_ratio(returns, periods_per_year=252)

        assert isinstance(sortino, float)
        # Sortino should be different from Sharpe (uses only downside deviation)

    def test_max_drawdown_calculation(self, performance_analyzer, sample_equity_curve):
        """Test maximum drawdown calculation."""
        max_dd, max_dd_duration = performance_analyzer.calculate_max_drawdown(sample_equity_curve)

        assert isinstance(max_dd, float)
        assert isinstance(max_dd_duration, int)
        assert max_dd <= 0  # Drawdown is negative or zero
        assert max_dd >= -100  # Should not exceed -100%
        assert max_dd_duration >= 0

    def test_calmar_ratio_calculation(self, performance_analyzer, sample_equity_curve):
        """Test Calmar ratio calculation."""
        calmar = performance_analyzer.calculate_calmar_ratio(sample_equity_curve, periods_per_year=252)

        assert isinstance(calmar, float)
        # Calmar ratio should be defined (annualized return / max drawdown)

    def test_value_at_risk_calculation(self, performance_analyzer):
        """Test VaR calculation at different confidence levels."""
        returns = pd.Series(np.random.normal(0.001, 0.02, 1000))

        # Test 95% VaR
        var_95 = performance_analyzer.calculate_value_at_risk(returns, confidence=0.95)
        assert isinstance(var_95, float)
        assert var_95 < 0  # VaR is typically negative (representing loss)

        # Test 99% VaR (should be more extreme)
        var_99 = performance_analyzer.calculate_value_at_risk(returns, confidence=0.99)
        assert var_99 < var_95  # 99% VaR should be more extreme than 95%

    def test_conditional_var_calculation(self, performance_analyzer):
        """Test Conditional VaR (Expected Shortfall) calculation."""
        returns = pd.Series(np.random.normal(0.001, 0.02, 1000))

        # Test CVaR at 95% confidence
        cvar_95 = performance_analyzer.calculate_conditional_var(returns, confidence=0.95)

        assert isinstance(cvar_95, float)
        assert cvar_95 < 0  # CVaR represents expected loss in tail

        # CVaR should be more extreme than VaR
        var_95 = performance_analyzer.calculate_value_at_risk(returns, confidence=0.95)
        assert cvar_95 < var_95

    def test_win_rate_calculation(self, performance_analyzer, sample_trades):
        """Test win rate calculation."""
        win_rate = performance_analyzer.calculate_win_rate(sample_trades)

        assert isinstance(win_rate, float)
        assert 0 <= win_rate <= 100  # Win rate should be percentage

    def test_profit_factor_calculation(self, performance_analyzer, sample_trades):
        """Test profit factor calculation."""
        profit_factor = performance_analyzer.calculate_profit_factor(sample_trades)

        assert isinstance(profit_factor, float)
        assert profit_factor >= 0  # Profit factor is non-negative
        # Profit factor > 1 means profitable strategy

    def test_average_trade_metrics(self, performance_analyzer, sample_trades):
        """Test average trade P&L calculations."""
        avg_trade = performance_analyzer.calculate_average_trade(sample_trades)

        assert isinstance(avg_trade, float)

        # Test average winning and losing trades
        avg_win = performance_analyzer.calculate_average_winning_trade(sample_trades)
        avg_loss = performance_analyzer.calculate_average_losing_trade(sample_trades)

        assert avg_win > 0  # Average win should be positive
        assert avg_loss < 0  # Average loss should be negative

    def test_recovery_factor_calculation(self, performance_analyzer, sample_equity_curve):
        """Test recovery factor calculation."""
        recovery_factor = performance_analyzer.calculate_recovery_factor(sample_equity_curve)

        assert isinstance(recovery_factor, float)
        # Recovery factor = Net profit / Max drawdown

    def test_comprehensive_analysis(self, performance_analyzer, sample_equity_curve, sample_trades):
        """Test comprehensive analysis with all metrics."""
        analysis = performance_analyzer.analyze(
            equity_curve=sample_equity_curve,
            trades=sample_trades,
            initial_capital=Decimal("10000")
        )

        # Verify all key metrics are present
        assert "sharpe_ratio" in analysis
        assert "sortino_ratio" in analysis
        assert "calmar_ratio" in analysis
        assert "max_drawdown" in analysis
        assert "max_drawdown_duration_days" in analysis
        assert "total_return_pct" in analysis
        assert "annualized_return_pct" in analysis
        assert "annualized_volatility_pct" in analysis
        assert "win_rate" in analysis
        assert "profit_factor" in analysis
        assert "avg_trade" in analysis
        assert "value_at_risk_95" in analysis
        assert "conditional_var_95" in analysis

        # Verify types
        assert isinstance(analysis["sharpe_ratio"], float)
        assert isinstance(analysis["max_drawdown"], float)
        assert isinstance(analysis["win_rate"], float)

    def test_statistical_significance(self, performance_analyzer, sample_trades):
        """Test statistical significance testing for returns."""
        returns = pd.Series([float(t["pnl"]) / 10000 for t in sample_trades])

        t_stat, p_value = performance_analyzer.test_statistical_significance(returns)

        assert isinstance(t_stat, float)
        assert isinstance(p_value, float)
        assert 0 <= p_value <= 1  # p-value range

    def test_benchmark_comparison(self, performance_analyzer):
        """Test performance comparison against benchmark."""
        strategy_returns = pd.Series(np.random.normal(0.001, 0.02, 252))
        benchmark_returns = pd.Series(np.random.normal(0.0005, 0.015, 252))

        alpha, beta = performance_analyzer.calculate_alpha_beta(
            strategy_returns=strategy_returns,
            benchmark_returns=benchmark_returns
        )

        assert isinstance(alpha, float)
        assert isinstance(beta, float)
        # Alpha represents excess return over benchmark
        # Beta represents correlation with benchmark

    def test_time_based_analysis(self, performance_analyzer, sample_trades):
        """Test daily/monthly/yearly performance analysis."""
        time_analysis = performance_analyzer.analyze_by_time_period(sample_trades)

        assert "daily_returns" in time_analysis
        assert "monthly_returns" in time_analysis
        assert "yearly_returns" in time_analysis

        # Verify structure
        assert isinstance(time_analysis["daily_returns"], dict)

    def test_edge_case_no_trades(self, performance_analyzer):
        """Test analysis with no trades."""
        empty_trades = []

        # Should handle gracefully
        win_rate = performance_analyzer.calculate_win_rate(empty_trades)
        assert win_rate == 0.0

    def test_edge_case_all_winning_trades(self, performance_analyzer):
        """Test with all winning trades."""
        winning_trades = [
            {"pnl": Decimal("100")} for _ in range(10)
        ]

        win_rate = performance_analyzer.calculate_win_rate(winning_trades)
        assert win_rate == 100.0

        profit_factor = performance_analyzer.calculate_profit_factor(winning_trades)
        assert profit_factor == float('inf')  # No losses, infinite profit factor

    def test_edge_case_all_losing_trades(self, performance_analyzer):
        """Test with all losing trades."""
        losing_trades = [
            {"pnl": Decimal("-100")} for _ in range(10)
        ]

        win_rate = performance_analyzer.calculate_win_rate(losing_trades)
        assert win_rate == 0.0

        profit_factor = performance_analyzer.calculate_profit_factor(losing_trades)
        assert profit_factor == 0.0  # No profits