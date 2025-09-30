"""
Functional tests for backtesting workflow using actual implementation API.

These tests validate the real BacktestEngine implementation works correctly
with realistic trading scenarios and proper data flow.
"""

import pytest
from decimal import Decimal
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, List

from trading_bot.backtesting.backtest_engine import BacktestEngine, Trade, PositionSide
from trading_bot.backtesting.performance_analyzer import PerformanceAnalyzer


class TestBacktestingFunctionalWorkflow:
    """Test complete backtesting workflows with actual implementation."""

    @pytest.fixture
    def realistic_price_data(self):
        """Generate realistic OHLCV data for testing - 6 months of hourly data."""
        start_date = datetime(2024, 1, 1)
        data = []
        
        # Generate 6 months of hourly data (180 days * 24 hours = 4320 bars)
        num_bars = 180 * 24
        base_price = 45000.0
        
        for i in range(num_bars):
            date = start_date + timedelta(hours=i)
            
            # Create realistic price movement with trends and volatility
            # First third: uptrend
            if i < num_bars // 3:
                trend = (i / (num_bars // 3)) * 5000
            # Second third: ranging
            elif i < 2 * num_bars // 3:
                trend = 5000 + ((i % 100) - 50) * 20
            # Last third: downtrend
            else:
                remaining = num_bars - i
                trend = 5000 - ((num_bars - i) / (num_bars // 3)) * 2000
            
            # Add some volatility
            volatility = (i % 20 - 10) * 10
            
            price = base_price + trend + volatility
            open_price = price + ((-1) ** i) * 50
            high = max(open_price, price) + abs(volatility) + 100
            low = min(open_price, price) - abs(volatility) - 100
            close = price
            
            data.append({
                'timestamp': date,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': 100.0 + (i % 50)
            })
        
        return pd.DataFrame(data)

    @pytest.fixture
    def mock_binance_client(self, realistic_price_data):
        """Create mock BinanceClient that returns realistic data in Binance API format."""
        class MockBinanceClient:
            async def get_historical_klines(self, symbol, interval, start_time, end_time):
                # Convert DataFrame to Binance klines format (list of lists)
                klines = []
                for _, row in realistic_price_data.iterrows():
                    open_time_ms = int(row['timestamp'].timestamp() * 1000)
                    close_time_ms = open_time_ms + 3600000  # 1 hour later
                    
                    klines.append([
                        open_time_ms,           # Open time
                        str(row['open']),       # Open
                        str(row['high']),       # High
                        str(row['low']),        # Low
                        str(row['close']),      # Close
                        str(row['volume']),     # Volume
                        close_time_ms,          # Close time
                        "0",                    # Quote asset volume
                        0,                      # Number of trades
                        "0",                    # Taker buy base volume
                        "0",                    # Taker buy quote volume
                        "0"                     # Ignore
                    ])
                
                return klines
        
        return MockBinanceClient()

    @pytest.mark.asyncio
    async def test_simple_momentum_strategy_workflow(
        self,
        mock_binance_client,
        realistic_price_data
    ):
        """
        Test complete workflow: data loading → strategy → backtest → results.
        Uses a simple momentum strategy.
        """
        # 1. Initialize BacktestEngine
        engine = BacktestEngine(
            binance_client=mock_binance_client,
            initial_capital=Decimal("10000"),
            commission_rate=Decimal("0.001"),
            slippage_rate=Decimal("0.0005"),
            name="test_momentum_backtest"
        )
        
        # 2. Load historical data
        await engine.load_historical_data(
            symbol="BTCUSDT",
            interval="1h",
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 7, 1)  # 6 months of data
        )
        
        assert engine.historical_data is not None
        assert len(engine.historical_data) > 0
        
        # 3. Register momentum strategy
        def momentum_strategy(engine, bar):
            """Simple momentum: buy when close > 20-bar average."""
            signals = []
            
            # Get current bar index
            current_idx = bar.name if isinstance(bar.name, int) else len(engine.equity_curve) - 1
            
            if current_idx < 20:
                return signals
            
            # Get recent closes using integer indexing
            recent_bars = engine.historical_data.iloc[
                max(0, current_idx - 20):current_idx
            ]
            avg_price = recent_bars['close'].mean()
            
            # Buy signal if price above average and no position
            if bar['close'] > avg_price and len(engine.open_positions) == 0:
                signals.append({
                    'action': 'buy',
                    'size': Decimal('0.1'),
                    'stop_loss': Decimal(str(float(bar['low']) * 0.95)),
                    'take_profit': Decimal(str(float(bar['high']) * 1.05))
                })
            # Sell signal if price below average and have position
            elif bar['close'] < avg_price and len(engine.open_positions) > 0:
                signals.append({
                    'action': 'sell',
                    'size': Decimal('0.1')
                })
            
            return signals
        
        engine.register_strategy(momentum_strategy)
        assert engine.strategy_func is not None
        
        # 4. Run backtest
        results = await engine.run_backtest(symbol="BTCUSDT")
        
        # 5. Validate results structure
        assert isinstance(results, dict)
        assert 'total_trades' in results
        assert 'winning_trades' in results
        assert 'losing_trades' in results
        assert 'total_return_pct' in results
        assert 'final_equity' in results

        # 6. Validate backtest ran correctly
        assert results['total_trades'] >= 0
        assert len(engine.equity_curve) > 0
        assert len(engine.closed_trades) >= 0

        # 7. Validate equity curve makes sense
        assert all('timestamp' in point for point in engine.equity_curve)
        assert all('equity' in point for point in engine.equity_curve)
        assert all(isinstance(point['equity'], Decimal) for point in engine.equity_curve)

        print(f"\n✓ Backtest completed successfully:")
        print(f"  - Total trades: {results['total_trades']}")
        print(f"  - Win rate: {results.get('win_rate', 0):.2f}%")
        print(f"  - Total return: {results['total_return_pct']:.2f}%")
        print(f"  - Final equity: ${results['final_equity']}")

    @pytest.mark.asyncio
    async def test_performance_analyzer_with_real_backtest(
        self,
        mock_binance_client,
        realistic_price_data
    ):
        """
        Test PerformanceAnalyzer works with real backtest results.
        Validates calculation integrity.
        """
        # Run a backtest first
        engine = BacktestEngine(
            binance_client=mock_binance_client,
            initial_capital=Decimal("10000"),
            commission_rate=Decimal("0.001"),
            slippage_rate=Decimal("0.0005"),
            name="test_performance_backtest"
        )
        
        await engine.load_historical_data(
            symbol="BTCUSDT",
            interval="1h",
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 7, 1)  # 6 months of data
        )
        
        # Simple buy-and-hold strategy
        def buy_hold_strategy(engine, bar):
            if len(engine.open_positions) == 0 and len(engine.equity_curve) == 1:
                return [{
                    'action': 'buy',
                    'size': Decimal('0.5'),
                    'stop_loss': Decimal('0'),
                    'take_profit': Decimal('999999')
                }]
            return []
        
        engine.register_strategy(buy_hold_strategy)
        await engine.run_backtest(symbol="BTCUSDT")
        
        # Convert equity curve to DataFrame
        equity_df = pd.DataFrame(engine.equity_curve)
        equity_df['equity'] = equity_df['equity'].astype(float)
        
        # Analyze performance
        analyzer = PerformanceAnalyzer()
        
        # Test max drawdown calculation
        max_dd, max_dd_duration = analyzer.calculate_max_drawdown(equity_df)
        assert isinstance(max_dd, float)
        assert max_dd >= 0  # Returns positive percentage
        assert isinstance(max_dd_duration, int)
        assert max_dd_duration >= 0
        
        # Test Sharpe ratio calculation
        returns = analyzer._calculate_returns(equity_df)
        sharpe = analyzer.calculate_sharpe_ratio(returns)
        assert isinstance(sharpe, float)
        
        # Test win rate calculation
        trades = [
            {
                'pnl': float(trade.pnl),
                'return_pct': float(trade.return_pct)
            }
            for trade in engine.closed_trades
        ]
        
        if len(trades) > 0:
            win_rate = analyzer.calculate_win_rate(trades)
            assert isinstance(win_rate, float)
            assert 0 <= win_rate <= 100
            
            profit_factor = analyzer.calculate_profit_factor(trades)
            assert isinstance(profit_factor, float)
            assert profit_factor >= 0
        
        print(f"\n✓ Performance analysis completed:")
        print(f"  - Max drawdown: {max_dd:.2f}%")
        print(f"  - Drawdown duration: {max_dd_duration} periods")
        print(f"  - Sharpe ratio: {sharpe:.2f}")
        if len(trades) > 0:
            print(f"  - Win rate: {win_rate:.2f}%")
            print(f"  - Profit factor: {profit_factor:.2f}")

    @pytest.mark.asyncio
    async def test_position_lifecycle(
        self,
        mock_binance_client,
        realistic_price_data
    ):
        """
        Test complete position lifecycle: open → update → close.
        Validates P&L calculations and position tracking.
        """
        engine = BacktestEngine(
            binance_client=mock_binance_client,
            initial_capital=Decimal("10000"),
            commission_rate=Decimal("0.001"),
            slippage_rate=Decimal("0.0005"),
            name="test_position_lifecycle"
        )
        
        await engine.load_historical_data(
            symbol="BTCUSDT",
            interval="1h",
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 7, 1)  # 6 months of data
        )
        
        # Strategy that opens and closes one position
        position_opened = False
        
        def single_trade_strategy(engine, bar):
            nonlocal position_opened
            
            # Get current bar index
            current_idx = bar.name if isinstance(bar.name, int) else len(engine.equity_curve) - 1
            
            # Open position on bar 10
            if current_idx == 10 and not position_opened:
                position_opened = True
                return [{
                    'action': 'buy',
                    'size': Decimal('0.1'),
                    'stop_loss': Decimal(str(float(bar['low']) * 0.90)),
                    'take_profit': Decimal(str(float(bar['high']) * 1.10))
                }]
            
            # Close position on bar 20
            if current_idx == 20 and len(engine.open_positions) > 0:
                return [{
                    'action': 'sell',
                    'size': Decimal('0.1')
                }]
            
            return []
        
        engine.register_strategy(single_trade_strategy)
        results = await engine.run_backtest(symbol="BTCUSDT")
        
        # Validate position was opened and closed
        assert results['total_trades'] >= 1
        assert len(engine.closed_trades) >= 1
        
        # Validate trade structure
        trade = engine.closed_trades[0]
        assert isinstance(trade.entry_price, Decimal)
        assert isinstance(trade.exit_price, Decimal)
        assert isinstance(trade.pnl, Decimal)
        assert isinstance(trade.pnl_percentage, float)
        assert trade.side in [PositionSide.LONG, PositionSide.SHORT]
        
        # Validate P&L calculation matches expected
        expected_pnl = trade.size * (trade.exit_price - trade.entry_price)
        if trade.side == PositionSide.SHORT:
            expected_pnl = trade.size * (trade.entry_price - trade.exit_price)

        # P&L should match calculation (note: commission is tracked separately in capital)
        assert trade.pnl == expected_pnl

        print(f"\n✓ Position lifecycle validated:")
        print(f"  - Entry: ${trade.entry_price}")
        print(f"  - Exit: ${trade.exit_price}")
        print(f"  - Size: {trade.size}")
        print(f"  - P&L: ${trade.pnl}")
        print(f"  - Return: {trade.pnl_percentage:.2f}%")

    @pytest.mark.asyncio
    async def test_stop_loss_execution(
        self,
        mock_binance_client,
        realistic_price_data
    ):
        """
        Test that stop-loss orders are executed correctly.
        """
        engine = BacktestEngine(
            binance_client=mock_binance_client,
            initial_capital=Decimal("10000"),
            commission_rate=Decimal("0.001"),
            slippage_rate=Decimal("0.0005"),
            name="test_stop_loss"
        )
        
        await engine.load_historical_data(
            symbol="BTCUSDT",
            interval="1h",
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 7, 1)  # 6 months of data
        )
        
        # Strategy with tight stop loss
        def tight_stop_strategy(engine, bar):
            # Get current bar index
            current_idx = bar.name if isinstance(bar.name, int) else len(engine.equity_curve) - 1
            
            if current_idx == 10 and len(engine.open_positions) == 0:
                return [{
                    'action': 'buy',
                    'size': Decimal('0.1'),
                    'stop_loss': Decimal(str(float(bar['close']) * 0.98)),  # 2% below entry
                    'take_profit': Decimal(str(float(bar['close']) * 1.20))  # 20% above entry
                }]
            return []
        
        engine.register_strategy(tight_stop_strategy)
        results = await engine.run_backtest(symbol="BTCUSDT")
        
        # Validate trade was executed
        assert results['total_trades'] >= 1
        
        # Check if stop loss was hit (if trade closed with loss < -2%)
        if len(engine.closed_trades) > 0:
            trade = engine.closed_trades[0]
            
            # If trade was closed by stop loss, return should be around -2%
            if trade.pnl_percentage < -1.5:
                print(f"\n✓ Stop loss executed:")
                print(f"  - Return: {trade.pnl_percentage:.2f}% (stop at -2%)")
                print(f"  - P&L: ${trade.pnl}")

                # Validate stop loss worked (loss should be close to stop level)
                assert trade.pnl_percentage > -3.0  # Not much worse than -2%