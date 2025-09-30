"""
BacktestEngine: Core backtesting engine for historical strategy validation.

Provides event-driven simulation of trading strategies with historical data,
position tracking, and P&L calculation.
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional
from enum import Enum

import pandas as pd

from ..core.base_component import BaseComponent
from ..core.events import EventType
from ..data.binance_client import BinanceClient
from ..execution.position_tracker import Position, PositionSide, PositionStatus


class BacktestStatus(Enum):
    """Backtest execution status."""
    IDLE = "idle"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class Trade:
    """Individual trade record for backtesting."""

    def __init__(
        self,
        trade_id: str,
        symbol: str,
        side: PositionSide,
        entry_time: datetime,
        entry_price: Decimal,
        size: Decimal,
        stop_loss: Optional[Decimal] = None,
        take_profit: Optional[Decimal] = None,
    ):
        self.trade_id = trade_id
        self.symbol = symbol
        self.side = side
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.size = size
        self.stop_loss = stop_loss
        self.take_profit = take_profit

        # Exit information
        self.exit_time: Optional[datetime] = None
        self.exit_price: Optional[Decimal] = None
        self.pnl: Optional[Decimal] = None
        self.pnl_percentage: Optional[float] = None
        self.mae: Decimal = Decimal("0")  # Maximum Adverse Excursion
        self.mfe: Decimal = Decimal("0")  # Maximum Favorable Excursion

    def close_trade(self, exit_time: datetime, exit_price: Decimal):
        """Close the trade and calculate P&L."""
        self.exit_time = exit_time
        self.exit_price = exit_price

        if self.side == PositionSide.LONG:
            self.pnl = (exit_price - self.entry_price) * self.size
        else:  # SHORT
            self.pnl = (self.entry_price - exit_price) * self.size

        self.pnl_percentage = float((self.pnl / (self.entry_price * self.size)) * 100)

    def update_excursions(self, current_price: Decimal):
        """Update MAE and MFE during trade lifetime."""
        if self.side == PositionSide.LONG:
            excursion = current_price - self.entry_price
        else:  # SHORT
            excursion = self.entry_price - current_price

        # Update Maximum Favorable Excursion
        if excursion > self.mfe:
            self.mfe = excursion

        # Update Maximum Adverse Excursion
        if excursion < self.mae:
            self.mae = excursion

    def update_mae_mfe(self, current_price: Decimal):
        """Alias for update_excursions for test compatibility."""
        self.update_excursions(current_price)

    def to_dict(self) -> Dict:
        """Convert trade to dictionary."""
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "entry_time": self.entry_time.isoformat(),
            "entry_price": str(self.entry_price),
            "size": str(self.size),
            "stop_loss": str(self.stop_loss) if self.stop_loss else None,
            "take_profit": str(self.take_profit) if self.take_profit else None,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "exit_price": str(self.exit_price) if self.exit_price else None,
            "pnl": str(self.pnl) if self.pnl else None,
            "pnl_percentage": self.pnl_percentage,
            "mae": str(self.mae),
            "mfe": str(self.mfe),
        }


class BacktestEngine(BaseComponent):
    """
    Core backtesting engine for historical strategy validation.

    Features:
    - Historical data loading and validation (minimum 6 months)
    - Event-driven simulation with time-based processing
    - Position tracking and P&L calculation
    - Integration with strategy signal generators
    - Memory-optimized pandas DataFrame processing
    """

    def __init__(
        self,
        binance_client: BinanceClient,
        initial_capital: Decimal = Decimal("10000"),
        commission_rate: Decimal = Decimal("0.001"),  # 0.1% per trade
        slippage_rate: Decimal = Decimal("0.0005"),  # 0.05% slippage
        **kwargs
    ):
        super().__init__(**kwargs)
        self.binance_client = binance_client
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate

        # State management
        self.status = BacktestStatus.IDLE
        self.current_capital = initial_capital
        self.equity_curve: List[Dict] = []

        # Position and trade tracking
        self.open_positions: Dict[str, Trade] = {}
        self.closed_trades: List[Trade] = []
        self.trade_counter = 0

        # Historical data
        self.historical_data: Optional[pd.DataFrame] = None
        self.current_time: Optional[datetime] = None
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

        # Strategy callback
        self.strategy_func: Optional[Callable] = None

    async def _start(self):
        """Start the backtest engine."""
        self.logger.info("BacktestEngine started")
        self.status = BacktestStatus.IDLE

    async def _stop(self):
        """Stop the backtest engine."""
        self.logger.info("BacktestEngine stopped")
        self.status = BacktestStatus.IDLE
        self.historical_data = None

    async def load_historical_data(
        self,
        symbol: str,
        interval: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Load historical kline data for backtesting.

        Args:
            symbol: Trading pair symbol
            interval: Kline interval (1m, 5m, 15m, 1h, 4h, 1d)
            start_time: Start datetime for data
            end_time: End datetime for data (default: now)

        Returns:
            DataFrame with OHLCV data

        Raises:
            ValueError: If data period is less than 6 months
        """
        if end_time is None:
            end_time = datetime.utcnow()

        # Validate minimum 6 months of data
        min_period = timedelta(days=180)
        if end_time - start_time < min_period:
            raise ValueError(
                f"Backtesting requires minimum 6 months of data. "
                f"Provided: {(end_time - start_time).days} days"
            )

        self.logger.info(
            f"Loading historical data for {symbol} from {start_time} to {end_time}"
        )

        # Fetch historical klines
        klines = await self.binance_client.get_historical_klines(
            symbol=symbol,
            interval=interval,
            start_time=int(start_time.timestamp() * 1000),
            end_time=int(end_time.timestamp() * 1000),
        )

        # Convert to DataFrame
        df = pd.DataFrame(
            klines,
            columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades", "taker_buy_base",
                "taker_buy_quote", "ignore"
            ]
        )

        # Convert timestamps to datetime
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

        # Convert price columns to Decimal for precision
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col].apply(Decimal)

        df["volume"] = df["volume"].apply(Decimal)

        # Set index to open_time
        df.set_index("open_time", inplace=True)

        self.historical_data = df
        self.start_time = start_time
        self.end_time = end_time

        self.logger.info(f"Loaded {len(df)} klines for backtesting")
        return df

    def register_strategy(self, strategy_func: Callable):
        """
        Register a strategy function for signal generation.

        The strategy function should accept (engine, current_bar) and return
        trading signals as a list of dicts with 'action', 'size', 'stop_loss', 'take_profit'.

        Example:
            def my_strategy(engine, bar):
                signals = []
                if bar['close'] > bar['open']:
                    signals.append({
                        'action': 'buy',
                        'size': Decimal('0.1'),
                        'stop_loss': bar['low'],
                        'take_profit': bar['high'] * Decimal('1.02')
                    })
                return signals
        """
        self.strategy_func = strategy_func
        self.logger.info("Strategy function registered")

    async def run_backtest(self, symbol: str) -> Dict:
        """
        Execute the backtest simulation.

        Args:
            symbol: Trading pair symbol

        Returns:
            Dictionary with backtest results
        """
        if self.historical_data is None:
            raise ValueError("Historical data not loaded. Call load_historical_data() first")

        if self.strategy_func is None:
            raise ValueError("Strategy not registered. Call register_strategy() first")

        self.logger.info(f"Starting backtest for {symbol}")
        self.status = BacktestStatus.RUNNING

        # Reset state
        self.current_capital = self.initial_capital
        self.open_positions = {}
        self.closed_trades = []
        self.equity_curve = []
        self.trade_counter = 0

        # Iterate through historical data
        for timestamp, bar in self.historical_data.iterrows():
            self.current_time = timestamp

            # Update open positions with current price
            await self._update_positions(bar)

            # Check for stop loss / take profit hits
            await self._check_exit_conditions(bar, symbol)

            # Generate signals from strategy
            signals = self.strategy_func(self, bar)

            # Process signals
            if signals:
                await self._process_signals(signals, bar, symbol)

            # Record equity curve point
            self._record_equity(timestamp, bar["close"])

            # Emit progress event periodically
            if len(self.equity_curve) % 100 == 0:
                await self._emit_progress()

        self.status = BacktestStatus.COMPLETED
        self.logger.info("Backtest completed")

        # Close any remaining open positions
        await self._close_all_positions(self.historical_data.iloc[-1], symbol)

        return self._generate_results()

    async def _update_positions(self, bar: pd.Series):
        """Update open positions with current price information."""
        for trade in self.open_positions.values():
            trade.update_excursions(bar["close"])

    async def _check_exit_conditions(self, bar: pd.Series, symbol: str):
        """Check if stop loss or take profit conditions are met."""
        closed_positions = []

        for trade_id, trade in self.open_positions.items():
            should_close = False
            exit_price = None

            # Check stop loss
            if trade.stop_loss:
                if trade.side == PositionSide.LONG and bar["low"] <= trade.stop_loss:
                    should_close = True
                    exit_price = trade.stop_loss
                elif trade.side == PositionSide.SHORT and bar["high"] >= trade.stop_loss:
                    should_close = True
                    exit_price = trade.stop_loss

            # Check take profit
            if trade.take_profit and not should_close:
                if trade.side == PositionSide.LONG and bar["high"] >= trade.take_profit:
                    should_close = True
                    exit_price = trade.take_profit
                elif trade.side == PositionSide.SHORT and bar["low"] <= trade.take_profit:
                    should_close = True
                    exit_price = trade.take_profit

            if should_close:
                await self._close_position(trade, self.current_time, exit_price, symbol)
                closed_positions.append(trade_id)

        # Remove closed positions
        for trade_id in closed_positions:
            del self.open_positions[trade_id]

    async def _process_signals(self, signals: List[Dict], bar: pd.Series, symbol: str):
        """Process trading signals from strategy."""
        for signal in signals:
            action = signal.get("action")
            size = signal.get("size", Decimal("0"))
            stop_loss = signal.get("stop_loss")
            take_profit = signal.get("take_profit")

            if action in ["buy", "long"]:
                await self._open_position(
                    symbol, PositionSide.LONG, size, bar["close"],
                    stop_loss, take_profit
                )
            elif action in ["sell", "short"]:
                await self._open_position(
                    symbol, PositionSide.SHORT, size, bar["close"],
                    stop_loss, take_profit
                )

    async def _open_position(
        self,
        symbol: str,
        side: PositionSide,
        size: Decimal,
        entry_price: Decimal,
        stop_loss: Optional[Decimal] = None,
        take_profit: Optional[Decimal] = None,
    ):
        """Open a new position."""
        # Apply slippage
        if side == PositionSide.LONG:
            entry_price = entry_price * (Decimal("1") + self.slippage_rate)
        else:
            entry_price = entry_price * (Decimal("1") - self.slippage_rate)

        # Calculate cost including commission
        cost = size * entry_price
        commission = cost * self.commission_rate
        total_cost = cost + commission

        # Check if we have enough capital
        if total_cost > self.current_capital:
            self.logger.warning(f"Insufficient capital for trade: {total_cost} > {self.current_capital}")
            return

        # Create trade
        self.trade_counter += 1
        trade_id = f"trade_{self.trade_counter}"

        trade = Trade(
            trade_id=trade_id,
            symbol=symbol,
            side=side,
            entry_time=self.current_time,
            entry_price=entry_price,
            size=size,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

        self.open_positions[trade_id] = trade
        self.current_capital -= total_cost

        self.logger.debug(f"Opened {side.value} position: {trade_id} at {entry_price}")

    async def _close_position(
        self,
        trade: Trade,
        exit_time: datetime,
        exit_price: Decimal,
        symbol: str,
    ):
        """Close an existing position."""
        # Apply slippage
        if trade.side == PositionSide.LONG:
            exit_price = exit_price * (Decimal("1") - self.slippage_rate)
        else:
            exit_price = exit_price * (Decimal("1") + self.slippage_rate)

        # Calculate proceeds including commission
        proceeds = trade.size * exit_price
        commission = proceeds * self.commission_rate
        net_proceeds = proceeds - commission

        # Close trade and calculate P&L
        trade.close_trade(exit_time, exit_price)

        # Update capital
        self.current_capital += net_proceeds

        # Move to closed trades
        self.closed_trades.append(trade)

        self.logger.debug(
            f"Closed position {trade.trade_id}: P&L = {trade.pnl} ({trade.pnl_percentage:.2f}%)"
        )

    async def _close_all_positions(self, final_bar: pd.Series, symbol: str):
        """Close all remaining open positions at end of backtest."""
        for trade in list(self.open_positions.values()):
            await self._close_position(
                trade, self.current_time, final_bar["close"], symbol
            )
        self.open_positions.clear()

    def _record_equity(self, timestamp: datetime, current_price: Decimal):
        """Record equity curve point."""
        # Calculate total equity (capital + unrealized P&L)
        unrealized_pnl = sum(
            trade.pnl if trade.pnl else Decimal("0")
            for trade in self.open_positions.values()
        )

        total_equity = self.current_capital + unrealized_pnl

        self.equity_curve.append({
            "timestamp": timestamp,
            "equity": total_equity,
            "cash": self.current_capital,
            "unrealized_pnl": unrealized_pnl,
            "open_positions": len(self.open_positions),
        })

    def _generate_results(self) -> Dict:
        """Generate backtest results summary."""
        total_trades = len(self.closed_trades)
        winning_trades = [t for t in self.closed_trades if t.pnl and t.pnl > 0]
        losing_trades = [t for t in self.closed_trades if t.pnl and t.pnl < 0]

        final_equity = self.equity_curve[-1]["equity"] if self.equity_curve else self.initial_capital
        total_return = ((final_equity - self.initial_capital) / self.initial_capital) * 100

        return {
            "initial_capital": str(self.initial_capital),
            "final_equity": str(final_equity),
            "total_return_pct": float(total_return),
            "total_trades": total_trades,
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": len(winning_trades) / total_trades * 100 if total_trades > 0 else 0,
            "equity_curve": self.equity_curve,
            "trades": [t.to_dict() for t in self.closed_trades],
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
        }

    async def _emit_progress(self):
        """Emit backtest progress event."""
        # Note: Event publishing disabled - SYSTEM_STATUS event type not available
        # Progress can be monitored through equity_curve length
        pass