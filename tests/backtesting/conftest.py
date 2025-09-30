"""
Pytest configuration and shared fixtures for backtesting tests.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal


@pytest.fixture(scope="session")
def test_config():
    """Common test configuration."""
    return {
        "initial_capital": "10000",
        "commission_rate": "0.001",
        "slippage_rate": "0.0005",
        "risk_free_rate": 0.02
    }


@pytest.fixture
def sample_kline_data():
    """Generate sample kline (OHLCV) data."""
    def _generate_klines(
        start_date=datetime(2024, 1, 1),
        periods=1000,
        interval_hours=1,
        base_price=50000,
        volatility=0.02
    ):
        dates = [start_date + timedelta(hours=i * interval_hours) for i in range(periods)]

        # Simulate realistic price movements
        returns = np.random.normal(0.0001, volatility, periods)
        prices = base_price * (1 + returns).cumprod()

        data = {
            "open_time": dates,
            "open": prices * (1 - np.random.uniform(0, 0.005, periods)),
            "high": prices * (1 + np.random.uniform(0, 0.01, periods)),
            "low": prices * (1 - np.random.uniform(0, 0.01, periods)),
            "close": prices,
            "volume": np.random.uniform(1000, 5000, periods)
        }

        df = pd.DataFrame(data)
        df.set_index("open_time", inplace=True)
        return df

    return _generate_klines


@pytest.fixture
def sample_trade_data():
    """Generate sample trade data."""
    def _generate_trades(
        num_trades=100,
        base_price=50000,
        avg_pnl=100,
        pnl_std=500
    ):
        trades = []

        start_time = datetime(2024, 1, 1)

        for i in range(num_trades):
            # Mix of winning and losing trades
            pnl = Decimal(str(np.random.normal(avg_pnl, pnl_std)))
            size = Decimal("0.1")
            entry_price = Decimal(str(base_price + np.random.uniform(-1000, 1000)))
            exit_price = entry_price + pnl / size

            # Calculate MAE and MFE
            mae = pnl if pnl < 0 else Decimal(str(np.random.uniform(-100, 0)))
            mfe = pnl if pnl > 0 else Decimal(str(np.random.uniform(0, 100)))

            trades.append({
                "trade_id": i + 1,
                "symbol": "BTCUSDT",
                "side": "LONG" if i % 2 == 0 else "SHORT",
                "entry_price": entry_price,
                "exit_price": exit_price,
                "entry_time": start_time + timedelta(hours=i * 4),
                "exit_time": start_time + timedelta(hours=i * 4 + 2),
                "size": size,
                "pnl": pnl,
                "pnl_percentage": float(pnl / (entry_price * size) * 100),
                "commission": Decimal("5"),
                "mae": mae,
                "mfe": mfe,
                "stop_loss": entry_price * Decimal("0.98") if i % 3 == 0 else None,
                "take_profit": entry_price * Decimal("1.02") if i % 3 == 1 else None,
                "exit_reason": "take_profit" if pnl > 0 else "stop_loss"
            })

        return trades

    return _generate_trades


@pytest.fixture
def sample_equity_curve_data():
    """Generate sample equity curve data."""
    def _generate_equity_curve(
        initial_equity=10000,
        periods=252,
        daily_return=0.001,
        volatility=0.02
    ):
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(periods)]

        # Simulate equity curve
        returns = np.random.normal(daily_return, volatility, periods)
        cumulative_returns = (1 + returns).cumprod()
        equity_values = initial_equity * cumulative_returns

        df = pd.DataFrame({
            "timestamp": dates,
            "equity": equity_values,
            "cash": equity_values * 0.5,  # Assume 50% in cash
            "position_value": equity_values * 0.5
        })
        df.set_index("timestamp", inplace=True)

        return df

    return _generate_equity_curve


@pytest.fixture
def mock_binance_client():
    """Mock BinanceClient for testing."""
    class MockBinanceClient:
        async def get_historical_klines(self, symbol, interval, start_time, end_time=None):
            # Generate mock data
            periods = 1000
            dates = [start_time + timedelta(hours=i) for i in range(periods)]

            prices = 50000 * (1 + np.random.normal(0.0001, 0.02, periods)).cumprod()

            data = {
                "open_time": dates,
                "open": prices * 0.999,
                "high": prices * 1.002,
                "low": prices * 0.998,
                "close": prices,
                "volume": np.random.uniform(1000, 5000, periods)
            }

            df = pd.DataFrame(data)
            df.set_index("open_time", inplace=True)
            return df

    return MockBinanceClient()


@pytest.fixture
def realistic_market_scenario():
    """Generate realistic market scenarios for testing."""
    def _generate_scenario(scenario_type="trending"):
        periods = 1000
        dates = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(periods)]

        if scenario_type == "trending":
            # Upward trending market
            trend = np.linspace(0, 0.5, periods)
            noise = np.random.normal(0, 0.02, periods)
            returns = trend / periods + noise
            prices = 50000 * (1 + returns).cumprod()

        elif scenario_type == "ranging":
            # Range-bound market
            prices = 50000 + 2000 * np.sin(np.linspace(0, 10 * np.pi, periods))
            prices += np.random.normal(0, 500, periods)

        elif scenario_type == "volatile":
            # High volatility market
            returns = np.random.normal(0, 0.05, periods)
            prices = 50000 * (1 + returns).cumprod()

        elif scenario_type == "crash":
            # Market crash scenario
            normal_periods = int(periods * 0.7)
            crash_periods = periods - normal_periods

            normal_returns = np.random.normal(0.0005, 0.02, normal_periods)
            crash_returns = np.random.normal(-0.05, 0.03, crash_periods)

            returns = np.concatenate([normal_returns, crash_returns])
            prices = 50000 * (1 + returns).cumprod()

        else:
            # Default: random walk
            returns = np.random.normal(0.0001, 0.02, periods)
            prices = 50000 * (1 + returns).cumprod()

        data = {
            "open_time": dates,
            "open": prices * 0.999,
            "high": prices * 1.002,
            "low": prices * 0.998,
            "close": prices,
            "volume": np.random.uniform(1000, 5000, periods)
        }

        df = pd.DataFrame(data)
        df.set_index("open_time", inplace=True)
        return df

    return _generate_scenario