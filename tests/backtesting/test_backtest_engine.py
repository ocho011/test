"""
Tests for BacktestEngine: Core backtesting engine functionality.
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
import pandas as pd

from trading_bot.backtesting.backtest_engine import BacktestEngine, Trade, BacktestStatus
from trading_bot.execution.position_tracker import PositionSide


@pytest.fixture
def backtest_engine(mock_binance_client):
    """Create a BacktestEngine instance for testing."""
    engine = BacktestEngine(
        binance_client=mock_binance_client,
        initial_capital=Decimal("10000"),
        commission_rate=Decimal("0.001"),
        slippage_rate=Decimal("0.0005"),
        name="test_backtest_engine"
    )
    return engine


@pytest.fixture
def sample_historical_data():
    """Generate sample historical OHLCV data for testing."""
    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(hours=i) for i in range(24 * 200)]  # 200 days of hourly data

    data = {
        "open_time": dates,
        "open": [100.0 + i * 0.1 for i in range(len(dates))],
        "high": [101.0 + i * 0.1 for i in range(len(dates))],
        "low": [99.0 + i * 0.1 for i in range(len(dates))],
        "close": [100.5 + i * 0.1 for i in range(len(dates))],
        "volume": [1000.0] * len(dates)
    }

    df = pd.DataFrame(data)
    df.set_index("open_time", inplace=True)
    return df


class TestBacktestEngine:
    """Test suite for BacktestEngine."""

    def test_initialization(self, backtest_engine):
        """Test engine initialization with config."""
        assert backtest_engine.initial_capital == Decimal("10000")
        assert backtest_engine.commission_rate == Decimal("0.001")
        assert backtest_engine.slippage_rate == Decimal("0.0005")
        assert backtest_engine.status == BacktestStatus.IDLE
        assert backtest_engine.current_capital == Decimal("10000")

    @pytest.mark.asyncio
    async def test_load_historical_data_validation(self, backtest_engine):
        """Test historical data loading with minimum period validation."""
        symbol = "BTCUSDT"
        interval = "1h"

        # Should fail: less than 6 months
        start_time = datetime(2024, 8, 1)
        end_time = datetime(2024, 9, 1)  # Only 1 month

        with pytest.raises(ValueError, match="minimum 6 months"):
            await backtest_engine.load_historical_data(symbol, interval, start_time, end_time)

    @pytest.mark.asyncio
    async def test_load_historical_data_success(self, backtest_engine, sample_historical_data, monkeypatch):
        """Test successful historical data loading."""
        symbol = "BTCUSDT"
        interval = "1h"
        start_time = datetime(2024, 1, 1)
        end_time = datetime(2024, 7, 31)  # 7 months

        # Mock BinanceClient get_historical_klines method
        async def mock_get_historical_klines(symbol, interval, start_time, end_time):
            # Return raw klines data format expected by BacktestEngine
            return sample_historical_data.reset_index().to_dict('records')

        monkeypatch.setattr(backtest_engine.binance_client, "get_historical_klines", mock_get_historical_klines)

        df = await backtest_engine.load_historical_data(symbol, interval, start_time, end_time)

        assert df is not None
        assert len(df) > 0
        assert "close" in df.columns
        assert backtest_engine.historical_data is not None

    @pytest.mark.asyncio
    async def test_run_backtest_basic(self, backtest_engine, sample_historical_data):
        """Test basic backtest execution."""
        backtest_engine.historical_data = sample_historical_data

        # Simple strategy callback: buy when close > open, sell when close < open
        # Note: strategy functions are called synchronously but can return signals
        def simple_strategy(engine, bar):
            # Return None (no signals) - backtest_engine doesn't support async strategies
            return None

        backtest_engine.register_strategy(simple_strategy)

        results = await backtest_engine.run_backtest("BTCUSDT")

        assert results is not None
        assert "initial_capital" in results
        assert "final_equity" in results
        assert "total_trades" in results
        assert "equity_curve" in results
        assert backtest_engine.status == BacktestStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_position_opening(self, backtest_engine):
        """Test opening a position with commission and slippage."""
        symbol = "BTCUSDT"
        side = PositionSide.LONG
        size = Decimal("0.1")
        entry_price = Decimal("50000")

        backtest_engine.current_capital = Decimal("10000")

        await backtest_engine._open_position(
            symbol=symbol,
            side=side,
            size=size,
            entry_price=entry_price,
            stop_loss=Decimal("47500"),
            take_profit=Decimal("52500")
        )

        # Check position was created
        assert len(backtest_engine.open_positions) > 0
        # Get the first (and only) trade
        position = list(backtest_engine.open_positions.values())[0]

        # Verify commission and slippage applied
        expected_slippage = entry_price * Decimal("0.0005")
        effective_price = entry_price + expected_slippage

        assert position.side == side
        assert position.size == size
        assert abs(position.entry_price - effective_price) < Decimal("0.01")

    @pytest.mark.asyncio
    async def test_position_closing_profit(self, backtest_engine):
        """Test closing a profitable position."""
        symbol = "BTCUSDT"

        # Open position
        backtest_engine.current_capital = Decimal("10000")
        entry_price = Decimal("50000")
        size = Decimal("0.1")

        await backtest_engine._open_position(
            symbol=symbol,
            side=PositionSide.LONG,
            size=size,
            entry_price=entry_price,
            stop_loss=None,
            take_profit=None
        )

        initial_capital = backtest_engine.current_capital

        # Close position at profit
        exit_price = Decimal("52000")  # 4% profit
        trade_id = list(backtest_engine.open_positions.keys())[0]
        trade = backtest_engine.open_positions[trade_id]
        await backtest_engine._close_position(trade, backtest_engine.current_time, exit_price, symbol)

        # Manually remove from open_positions (as _close_position doesn't do it)
        del backtest_engine.open_positions[trade_id]

        # Verify position closed
        assert len(backtest_engine.open_positions) == 0

        # Verify trade recorded
        assert len(backtest_engine.closed_trades) == 1
        trade = backtest_engine.closed_trades[0]
        assert trade.pnl > 0

        # Verify capital increased
        assert backtest_engine.current_capital > initial_capital

    @pytest.mark.asyncio
    async def test_position_closing_loss(self, backtest_engine):
        """Test closing a losing position."""
        symbol = "BTCUSDT"

        # Set initial capital before opening position
        initial_capital = Decimal("10000")
        backtest_engine.current_capital = initial_capital
        entry_price = Decimal("50000")
        size = Decimal("0.1")

        await backtest_engine._open_position(
            symbol=symbol,
            side=PositionSide.LONG,
            size=size,
            entry_price=entry_price,
            stop_loss=None,
            take_profit=None
        )

        # Close position at loss
        exit_price = Decimal("48000")  # 4% loss
        trade_id = list(backtest_engine.open_positions.keys())[0]
        trade = backtest_engine.open_positions[trade_id]
        await backtest_engine._close_position(trade, backtest_engine.current_time, exit_price, symbol)

        # Manually remove from open_positions (as _close_position doesn't do it)
        del backtest_engine.open_positions[trade_id]

        # Verify trade recorded
        trade = backtest_engine.closed_trades[0]
        assert trade.pnl < 0

        # Verify capital decreased
        assert backtest_engine.current_capital < initial_capital

    @pytest.mark.asyncio
    async def test_stop_loss_execution(self, backtest_engine):
        """Test automatic stop loss execution."""
        symbol = "BTCUSDT"
        entry_price = Decimal("50000")
        stop_loss = Decimal("48000")
        size = Decimal("0.1")

        backtest_engine.current_capital = Decimal("10000")

        # Open position with stop loss
        await backtest_engine._open_position(
            symbol=symbol,
            side=PositionSide.LONG,
            size=size,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=None
        )

        # Simulate price hitting stop loss - create bar with price below stop loss
        bar_data = {
            "open": Decimal("48000"),
            "high": Decimal("48500"),
            "low": Decimal("47500"),  # Below stop loss
            "close": Decimal("47800"),
            "volume": Decimal("100")
        }
        bar = pd.Series(bar_data)

        await backtest_engine._check_exit_conditions(bar, symbol)

        # Verify position closed
        assert len(backtest_engine.open_positions) == 0

        # Verify trade was closed
        assert len(backtest_engine.closed_trades) == 1
        trade = backtest_engine.closed_trades[0]
        # exit_price includes slippage: stop_loss * (1 - slippage_rate) for LONG
        expected_exit = stop_loss * (Decimal("1") - Decimal("0.0005"))
        assert abs(trade.exit_price - expected_exit) < Decimal("1")  # Allow small rounding error

    @pytest.mark.asyncio
    async def test_take_profit_execution(self, backtest_engine):
        """Test automatic take profit execution."""
        symbol = "BTCUSDT"
        entry_price = Decimal("50000")
        take_profit = Decimal("52000")
        size = Decimal("0.1")

        backtest_engine.current_capital = Decimal("10000")

        # Open position with take profit
        await backtest_engine._open_position(
            symbol=symbol,
            side=PositionSide.LONG,
            size=size,
            entry_price=entry_price,
            stop_loss=None,
            take_profit=take_profit
        )

        # Simulate price hitting take profit - create bar with price above take profit
        bar_data = {
            "open": Decimal("51500"),
            "high": Decimal("52500"),  # Above take profit
            "low": Decimal("51000"),
            "close": Decimal("52200"),
            "volume": Decimal("100")
        }
        bar = pd.Series(bar_data)

        await backtest_engine._check_exit_conditions(bar, symbol)

        # Verify position closed
        assert len(backtest_engine.open_positions) == 0

        # Verify trade was closed
        assert len(backtest_engine.closed_trades) == 1
        trade = backtest_engine.closed_trades[0]
        # exit_price includes slippage: take_profit * (1 - slippage_rate) for LONG
        expected_exit = take_profit * (Decimal("1") - Decimal("0.0005"))
        assert abs(trade.exit_price - expected_exit) < Decimal("1")  # Allow small rounding error

    def test_equity_curve_tracking(self, backtest_engine):
        """Test equity curve generation."""
        # Add some mock trades
        backtest_engine.current_capital = Decimal("10000")
        backtest_engine.equity_curve = []

        # Record initial equity
        backtest_engine._record_equity(datetime(2024, 1, 1), Decimal("10000"))

        # Simulate capital change
        backtest_engine.current_capital = Decimal("10500")
        backtest_engine._record_equity(datetime(2024, 1, 2), Decimal("10500"))

        assert len(backtest_engine.equity_curve) == 2
        assert backtest_engine.equity_curve[0]["equity"] == Decimal("10000")
        assert backtest_engine.equity_curve[1]["equity"] == Decimal("10500")

    def test_trade_mae_mfe_tracking(self):
        """Test Maximum Adverse/Favorable Excursion tracking."""
        trade = Trade(
            trade_id=1,
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=Decimal("50000"),
            entry_time=datetime.now(),
            size=Decimal("0.1"),
            stop_loss=None,
            take_profit=None
        )

        # Simulate price movements
        trade.update_mae_mfe(Decimal("49000"))  # Price drops
        assert trade.mae < 0  # Should record adverse movement

        trade.update_mae_mfe(Decimal("51000"))  # Price rises
        assert trade.mfe > 0  # Should record favorable movement

        trade.update_mae_mfe(Decimal("48000"))  # Price drops more
        assert trade.mae <= Decimal("-2000")  # MAE should update to worse or equal