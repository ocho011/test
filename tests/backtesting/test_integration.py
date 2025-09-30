"""
Integration tests for backtesting system: End-to-end workflow tests.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal

from trading_bot.backtesting.backtest_engine import BacktestEngine
from trading_bot.backtesting.performance_analyzer import PerformanceAnalyzer
from trading_bot.backtesting.trade_analyzer import TradeAnalyzer
from trading_bot.backtesting.monte_carlo_simulator import MonteCarloSimulator
from trading_bot.backtesting.visualization import BacktestVisualizer
from trading_bot.execution.position_tracker import PositionSide


@pytest.fixture
def complete_backtest_system(mock_binance_client):
    """Create complete backtesting system with all components."""

    return {
        "engine": BacktestEngine(
            binance_client=mock_binance_client,
            initial_capital=Decimal("10000"),
            commission_rate=Decimal("0.001"),
            slippage_rate=Decimal("0.0005"),
            name="test_backtest_engine"
        ),
        "performance_analyzer": PerformanceAnalyzer(),
        "trade_analyzer": TradeAnalyzer(),
        "monte_carlo": MonteCarloSimulator(n_simulations=1000),
        "visualizer": BacktestVisualizer(output_dir="/tmp/backtest_results")
    }


@pytest.fixture
def sample_historical_data():
    """Generate realistic historical data for integration testing."""
    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(hours=i) for i in range(24 * 200)]  # 200 days

    # Simulate realistic price movements
    base_price = 50000
    returns = np.random.normal(0.0001, 0.02, len(dates))  # Small drift, 2% volatility
    prices = base_price * (1 + returns).cumprod()

    data = {
        "open_time": dates,
        "open": prices * 0.999,
        "high": prices * 1.002,
        "low": prices * 0.998,
        "close": prices,
        "volume": np.random.uniform(1000, 5000, len(dates))
    }

    df = pd.DataFrame(data)
    df.set_index("open_time", inplace=True)
    return df


class TestBacktestingIntegration:
    """Integration tests for complete backtesting workflow."""

    @pytest.mark.asyncio
    async def test_complete_backtest_workflow(self, complete_backtest_system, sample_historical_data):
        """Test complete backtest workflow from data loading to analysis."""
        engine = complete_backtest_system["engine"]
        performance_analyzer = complete_backtest_system["performance_analyzer"]
        trade_analyzer = complete_backtest_system["trade_analyzer"]

        # 1. Load historical data
        engine.historical_data = sample_historical_data

        # 2. Define and register strategy
        async def momentum_strategy(event):
            """Simple momentum strategy for testing."""
            bar = event["data"]
            symbol = event["symbol"]

            # Calculate simple momentum
            if len(engine.historical_data) > 20:
                recent_data = engine.historical_data.iloc[-20:]
                momentum = (bar["close"] - recent_data["close"].iloc[0]) / recent_data["close"].iloc[0]

                # Entry signal: strong momentum
                if momentum > 0.02 and symbol not in engine.positions:
                    await engine._open_position(
                        symbol=symbol,
                        side=PositionSide.LONG,
                        size=Decimal("0.1"),
                        entry_price=Decimal(str(bar["close"])),
                        stop_loss=Decimal(str(bar["close"] * 0.98)),
                        take_profit=Decimal(str(bar["close"] * 1.03))
                    )

                # Exit signal: momentum reversal
                elif momentum < -0.01 and symbol in engine.positions:
                    await engine._close_position(symbol, Decimal(str(bar["close"])), "momentum_reversal")

        engine.register_strategy_callback(momentum_strategy)

        # 3. Run backtest
        backtest_results = await engine.run_backtest("BTCUSDT")

        # 4. Analyze performance
        equity_df = pd.DataFrame(backtest_results["equity_curve"])
        equity_df["timestamp"] = pd.to_datetime(equity_df["timestamp"])
        equity_df.set_index("timestamp", inplace=True)

        performance_metrics = performance_analyzer.analyze(
            equity_curve=equity_df,
            trades=backtest_results["trades"],
            initial_capital=Decimal("10000")
        )

        # 5. Analyze trades
        if backtest_results["trades"]:
            trade_analysis = trade_analyzer.analyze(backtest_results["trades"])
        else:
            trade_analysis = {}

        # Verify workflow completion
        assert backtest_results is not None
        assert "total_trades" in backtest_results
        assert "final_equity" in backtest_results

        assert performance_metrics is not None
        assert "sharpe_ratio" in performance_metrics
        assert "max_drawdown" in performance_metrics

        # Verify data consistency
        assert backtest_results["initial_capital"] == Decimal("10000")
        assert len(backtest_results["equity_curve"]) > 0

    @pytest.mark.asyncio
    async def test_backtest_with_monte_carlo(self, complete_backtest_system, sample_historical_data):
        """Test backtest with Monte Carlo simulation."""
        engine = complete_backtest_system["engine"]
        monte_carlo = complete_backtest_system["monte_carlo"]

        # Run backtest
        engine.historical_data = sample_historical_data

        # Simple strategy
        async def buy_hold_strategy(event):
            bar = event["data"]
            symbol = event["symbol"]

            if symbol not in engine.positions and len(engine.trades) == 0:  # Buy once
                await engine._open_position(
                    symbol=symbol,
                    side=PositionSide.LONG,
                    size=Decimal("1.0"),
                    entry_price=Decimal(str(bar["close"])),
                    stop_loss=None,
                    take_profit=None
                )

        engine.register_strategy_callback(buy_hold_strategy)
        backtest_results = await engine.run_backtest("BTCUSDT")

        # Calculate returns
        equity_df = pd.DataFrame(backtest_results["equity_curve"])
        equity_df["equity"] = equity_df["equity"].apply(float)
        returns = equity_df["equity"].pct_change().dropna().tolist()

        # Run Monte Carlo simulation
        if len(returns) > 30:  # Need sufficient data
            mc_results = monte_carlo.simulate_returns(
                historical_returns=returns,
                n_periods=30,
                initial_capital=Decimal("10000")
            )

            # Verify Monte Carlo results
            assert "mean_final_capital" in mc_results
            assert "var_95" in mc_results
            assert "cvar_95" in mc_results
            assert "simulated_paths" in mc_results

            # Calculate VaR
            var_95 = monte_carlo.calculate_var(returns, confidence_level=0.95)
            assert isinstance(var_95, float)

    @pytest.mark.asyncio
    async def test_backtest_with_visualization(self, complete_backtest_system, sample_historical_data):
        """Test backtest with visualization and export."""
        engine = complete_backtest_system["engine"]
        performance_analyzer = complete_backtest_system["performance_analyzer"]
        visualizer = complete_backtest_system["visualizer"]

        # Run backtest
        engine.historical_data = sample_historical_data

        async def simple_strategy(event):
            bar = event["data"]
            symbol = event["symbol"]

            # Random entry for testing
            if symbol not in engine.positions and np.random.random() > 0.95:
                await engine._open_position(
                    symbol=symbol,
                    side=PositionSide.LONG,
                    size=Decimal("0.1"),
                    entry_price=Decimal(str(bar["close"])),
                    stop_loss=Decimal(str(bar["close"] * 0.98)),
                    take_profit=Decimal(str(bar["close"] * 1.02))
                )

        engine.register_strategy_callback(simple_strategy)
        backtest_results = await engine.run_backtest("BTCUSDT")

        # Analyze performance
        equity_df = pd.DataFrame(backtest_results["equity_curve"])
        equity_df["timestamp"] = pd.to_datetime(equity_df["timestamp"])
        equity_df.set_index("timestamp", inplace=True)

        performance_metrics = performance_analyzer.analyze(
            equity_curve=equity_df,
            trades=backtest_results["trades"],
            initial_capital=Decimal("10000")
        )

        # Export results
        if backtest_results["trades"]:
            # Test CSV export
            csv_paths = visualizer.export_to_csv(
                equity_curve=backtest_results["equity_curve"],
                trades=backtest_results["trades"],
                performance_metrics=performance_metrics,
                prefix="integration_test"
            )

            assert "equity_curve" in csv_paths
            assert "trades" in csv_paths
            assert "metrics" in csv_paths

            # Test JSON export
            json_path = visualizer.export_to_json(
                results=backtest_results,
                filename="integration_test_results.json"
            )

            assert json_path is not None
            assert "integration_test_results.json" in json_path

    def test_performance_consistency(self, complete_backtest_system):
        """Test consistency between different analysis methods."""
        performance_analyzer = complete_backtest_system["performance_analyzer"]

        # Create consistent test data
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))

        # Calculate metrics multiple times
        sharpe_1 = performance_analyzer.calculate_sharpe_ratio(returns, periods_per_year=252)
        sharpe_2 = performance_analyzer.calculate_sharpe_ratio(returns, periods_per_year=252)

        # Should be identical (deterministic calculation)
        assert sharpe_1 == sharpe_2

    @pytest.mark.asyncio
    async def test_multi_symbol_backtest(self, complete_backtest_system):
        """Test backtesting with multiple symbols."""
        engine = complete_backtest_system["engine"]

        # Create data for multiple symbols
        symbols = ["BTCUSDT", "ETHUSDT"]

        for symbol in symbols:
            # Generate data
            dates = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(24 * 200)]
            prices = 50000 * (1 + np.random.normal(0.0001, 0.02, len(dates))).cumprod()

            data = {
                "open_time": dates,
                "open": prices * 0.999,
                "high": prices * 1.002,
                "low": prices * 0.998,
                "close": prices,
                "volume": np.random.uniform(1000, 5000, len(dates))
            }

            df = pd.DataFrame(data)
            df.set_index("open_time", inplace=True)
            engine.historical_data = df

            # Simple strategy
            async def multi_symbol_strategy(event):
                bar = event["data"]
                sym = event["symbol"]

                if sym not in engine.positions:
                    await engine._open_position(
                        symbol=sym,
                        side=PositionSide.LONG,
                        size=Decimal("0.1"),
                        entry_price=Decimal(str(bar["close"])),
                        stop_loss=Decimal(str(bar["close"] * 0.98)),
                        take_profit=Decimal(str(bar["close"] * 1.02))
                    )

            engine.register_strategy_callback(multi_symbol_strategy)
            results = await engine.run_backtest(symbol)

            # Verify results for each symbol
            assert results is not None
            assert results["total_trades"] >= 0

    def test_edge_case_no_trades_workflow(self, complete_backtest_system, sample_historical_data):
        """Test complete workflow when strategy generates no trades."""
        performance_analyzer = complete_backtest_system["performance_analyzer"]
        trade_analyzer = complete_backtest_system["trade_analyzer"]

        # Simulate backtest with no trades
        backtest_results = {
            "initial_capital": Decimal("10000"),
            "final_equity": Decimal("10000"),
            "total_return_pct": 0.0,
            "total_trades": 0,
            "trades": [],
            "equity_curve": [
                {"timestamp": datetime(2024, 1, 1), "equity": Decimal("10000")},
                {"timestamp": datetime(2024, 7, 1), "equity": Decimal("10000")}
            ]
        }

        # Should handle gracefully
        equity_df = pd.DataFrame(backtest_results["equity_curve"])
        equity_df["timestamp"] = pd.to_datetime(equity_df["timestamp"])
        equity_df.set_index("timestamp", inplace=True)

        performance_metrics = performance_analyzer.analyze(
            equity_curve=equity_df,
            trades=backtest_results["trades"],
            initial_capital=Decimal("10000")
        )

        trade_analysis = trade_analyzer.analyze(backtest_results["trades"])

        # Verify graceful handling
        assert performance_metrics["total_return_pct"] == 0.0
        assert trade_analysis["total_trades"] == 0