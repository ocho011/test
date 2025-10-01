"""
Strategy performance tracking and monitoring.

This module provides comprehensive performance tracking for trading
strategies with real-time metrics and historical analysis.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass, field
from collections import defaultdict
import statistics


@dataclass
class TradeRecord:
    """Record of a single trade execution."""
    
    trade_id: str
    strategy_name: str
    timestamp: datetime
    direction: str  # 'LONG' or 'SHORT'
    entry_price: Decimal
    exit_price: Optional[Decimal] = None
    quantity: Decimal = Decimal('0')
    pnl: Optional[Decimal] = None
    pnl_percent: Optional[float] = None
    duration: Optional[timedelta] = None
    exit_reason: Optional[str] = None
    signal_confidence: float = 0.0
    is_closed: bool = False


@dataclass
class StrategyMetrics:
    """Performance metrics for a strategy."""
    
    strategy_name: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    total_pnl: Decimal = Decimal('0')
    total_pnl_percent: float = 0.0
    average_pnl: Decimal = Decimal('0')
    average_win: Decimal = Decimal('0')
    average_loss: Decimal = Decimal('0')
    
    best_trade: Decimal = Decimal('0')
    worst_trade: Decimal = Decimal('0')
    
    sharpe_ratio: Optional[float] = None
    max_drawdown: Decimal = Decimal('0')
    max_drawdown_percent: float = 0.0
    
    average_trade_duration: Optional[timedelta] = None
    average_confidence: float = 0.0
    
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    
    first_trade_time: Optional[datetime] = None
    last_trade_time: Optional[datetime] = None
    
    additional_metrics: Dict[str, Any] = field(default_factory=dict)


class StrategyPerformanceTracker:
    """
    Comprehensive strategy performance tracking system.

    Tracks and analyzes performance metrics for all strategies:
    - Win/loss statistics
    - PnL tracking (absolute and percentage)
    - Risk metrics (Sharpe ratio, max drawdown)
    - Trade analysis (duration, confidence correlation)
    - Comparative analysis across strategies
    """

    def __init__(self):
        """Initialize the performance tracker."""
        self.logger = logging.getLogger("trading_bot.performance_tracker")
        
        # Per-strategy tracking
        self._trades: Dict[str, List[TradeRecord]] = defaultdict(list)
        self._open_trades: Dict[str, List[TradeRecord]] = defaultdict(list)
        self._metrics_cache: Dict[str, StrategyMetrics] = {}
        self._cache_valid: Dict[str, bool] = defaultdict(lambda: False)
        
        self.logger.info("StrategyPerformanceTracker initialized")

    def record_trade(
        self,
        strategy_name: str,
        entry_price: Decimal,
        exit_price: Decimal,
        quantity: Decimal,
        is_winning: bool,
        entry_time: Optional[datetime] = None,
        exit_time: Optional[datetime] = None,
    ) -> None:
        """
        Record a completed trade (convenience method for testing).
        
        Args:
            strategy_name: Name of the strategy
            entry_price: Entry price
            exit_price: Exit price
            quantity: Trade quantity
            is_winning: Whether the trade was profitable
            entry_time: Entry timestamp (defaults to now)
            exit_time: Exit timestamp (defaults to now)
        """
        import uuid
        
        if entry_time is None:
            entry_time = datetime.now()
        if exit_time is None:
            exit_time = datetime.now()
        
        # Calculate P&L
        pnl = (exit_price - entry_price) * quantity
        
        # Calculate P&L percentage
        pnl_percent = float((exit_price - entry_price) / entry_price * 100) if entry_price != 0 else 0.0
        
        # Determine direction from price change
        direction = 'LONG' if exit_price > entry_price else 'SHORT'
        
        # Calculate duration
        duration = exit_time - entry_time if entry_time and exit_time else None
        
        # Create trade record with correct field names
        trade = TradeRecord(
            trade_id=str(uuid.uuid4()),
            strategy_name=strategy_name,
            timestamp=entry_time,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            pnl=pnl,
            pnl_percent=pnl_percent,
            duration=duration,
            is_closed=True,
        )
        
        # Store trade
        if strategy_name not in self._trades:
            self._trades[strategy_name] = []
        self._trades[strategy_name].append(trade)
        
        # Invalidate cache
        self._invalidate_cache(strategy_name)
        
        self.logger.debug(
            f"Recorded trade for {strategy_name}: "
            f"Entry={entry_price}, Exit={exit_price}, P&L={pnl}"
        )

    def record_trade_entry(
        self,
        trade_id: str,
        strategy_name: str,
        direction: str,
        entry_price: Decimal,
        quantity: Decimal,
        signal_confidence: float = 0.0
    ) -> None:
        """
        Record a new trade entry.

        Args:
            trade_id: Unique trade identifier
            strategy_name: Name of strategy that generated the trade
            direction: Trade direction ('LONG' or 'SHORT')
            entry_price: Entry price
            quantity: Trade quantity
            signal_confidence: Confidence score of the signal (0-1)
        """
        trade = TradeRecord(
            trade_id=trade_id,
            strategy_name=strategy_name,
            timestamp=datetime.now(),
            direction=direction,
            entry_price=entry_price,
            quantity=quantity,
            signal_confidence=signal_confidence,
            is_closed=False
        )
        
        self._open_trades[strategy_name].append(trade)
        self._invalidate_cache(strategy_name)
        
        self.logger.info(
            f"Recorded trade entry: {trade_id} for {strategy_name} "
            f"({direction} @ {entry_price})"
        )

    def record_trade_exit(
        self,
        trade_id: str,
        strategy_name: str,
        exit_price: Decimal,
        exit_reason: str = "unknown"
    ) -> None:
        """
        Record a trade exit and calculate PnL.

        Args:
            trade_id: Trade identifier
            strategy_name: Strategy name
            exit_price: Exit price
            exit_reason: Reason for exit (e.g., 'take_profit', 'stop_loss', 'manual')
        """
        # Find the open trade
        open_trades = self._open_trades[strategy_name]
        trade_index = next(
            (i for i, t in enumerate(open_trades) if t.trade_id == trade_id),
            None
        )
        
        if trade_index is None:
            self.logger.warning(f"Trade {trade_id} not found in open trades")
            return
        
        trade = open_trades.pop(trade_index)
        
        # Calculate PnL
        trade.exit_price = exit_price
        trade.duration = datetime.now() - trade.timestamp
        trade.exit_reason = exit_reason
        trade.is_closed = True
        
        if trade.direction == 'LONG':
            pnl = (exit_price - trade.entry_price) * trade.quantity
        else:  # SHORT
            pnl = (trade.entry_price - exit_price) * trade.quantity
        
        trade.pnl = pnl
        trade.pnl_percent = float((pnl / (trade.entry_price * trade.quantity)) * 100)
        
        # Add to closed trades
        self._trades[strategy_name].append(trade)
        self._invalidate_cache(strategy_name)
        
        self.logger.info(
            f"Recorded trade exit: {trade_id} for {strategy_name} "
            f"(PnL: {pnl:.2f}, {trade.pnl_percent:.2f}%)"
        )

    def get_strategy_metrics(self, strategy_name: str) -> StrategyMetrics:
        """
        Get comprehensive metrics for a strategy.

        Args:
            strategy_name: Strategy name

        Returns:
            Strategy performance metrics
        """
        if self._cache_valid[strategy_name]:
            return self._metrics_cache[strategy_name]
        
        metrics = self._calculate_metrics(strategy_name)
        self._metrics_cache[strategy_name] = metrics
        self._cache_valid[strategy_name] = True
        
        return metrics

    def get_metrics(self, strategy_name: str) -> StrategyMetrics:
        """
        Get metrics for a specific strategy (alias for get_strategy_metrics).
        
        Args:
            strategy_name: Strategy name
            
        Returns:
            StrategyMetrics for the strategy
        """
        return self.get_strategy_metrics(strategy_name)

    def _calculate_metrics(self, strategy_name: str) -> StrategyMetrics:
        """Calculate all performance metrics for a strategy."""
        trades = self._trades[strategy_name]
        
        if not trades:
            return StrategyMetrics(strategy_name=strategy_name)
        
        # Basic statistics
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.pnl and t.pnl > 0)
        losing_trades = sum(1 for t in trades if t.pnl and t.pnl < 0)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
        
        # PnL statistics
        total_pnl = sum(t.pnl for t in trades if t.pnl)
        pnls = [float(t.pnl) for t in trades if t.pnl]
        average_pnl = Decimal(str(statistics.mean(pnls))) if pnls else Decimal('0')
        
        wins = [float(t.pnl) for t in trades if t.pnl and t.pnl > 0]
        losses = [float(t.pnl) for t in trades if t.pnl and t.pnl < 0]
        
        average_win = Decimal(str(statistics.mean(wins))) if wins else Decimal('0')
        average_loss = Decimal(str(statistics.mean(losses))) if losses else Decimal('0')
        
        best_trade = max((t.pnl for t in trades if t.pnl), default=Decimal('0'))
        worst_trade = min((t.pnl for t in trades if t.pnl), default=Decimal('0'))
        
        # Calculate Sharpe ratio (simplified)
        sharpe_ratio = None
        if len(pnls) > 1:
            mean_return = statistics.mean(pnls)
            std_return = statistics.stdev(pnls)
            if std_return > 0:
                sharpe_ratio = mean_return / std_return
        
        # Max drawdown calculation
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        
        for pnl in pnls:
            cumulative += pnl
            if cumulative > peak:
                peak = cumulative
            drawdown = peak - cumulative
            if drawdown > max_dd:
                max_dd = drawdown
        
        # Calculate percentage-based metrics
        total_capital_used = sum(
            float(t.entry_price * t.quantity) for t in trades
        )
        total_pnl_percent = (
            float(total_pnl) / total_capital_used * 100
            if total_capital_used > 0 else 0.0
        )
        
        max_dd_percent = (
            (max_dd / peak * 100) if peak > 0 else 0.0
        )
        
        # Trade duration analysis
        durations = [t.duration for t in trades if t.duration]
        avg_duration = None
        if durations:
            avg_duration = sum(durations, timedelta()) / len(durations)
        
        # Confidence analysis
        confidences = [t.signal_confidence for t in trades]
        avg_confidence = statistics.mean(confidences) if confidences else 0.0
        
        # Consecutive wins/losses
        current_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        
        for trade in trades:
            if trade.pnl and trade.pnl > 0:
                current_streak = max(1, current_streak + 1) if current_streak >= 0 else 1
            elif trade.pnl and trade.pnl < 0:
                current_streak = min(-1, current_streak - 1) if current_streak <= 0 else -1
            
            if current_streak > 0:
                max_win_streak = max(max_win_streak, current_streak)
            elif current_streak < 0:
                max_loss_streak = max(max_loss_streak, abs(current_streak))
        
        # Create metrics object
        metrics = StrategyMetrics(
            strategy_name=strategy_name,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            total_pnl_percent=total_pnl_percent,
            average_pnl=average_pnl,
            average_win=average_win,
            average_loss=average_loss,
            best_trade=best_trade,
            worst_trade=worst_trade,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=Decimal(str(max_dd)),
            max_drawdown_percent=max_dd_percent,
            average_trade_duration=avg_duration,
            average_confidence=avg_confidence,
            consecutive_wins=current_streak if current_streak > 0 else 0,
            consecutive_losses=abs(current_streak) if current_streak < 0 else 0,
            max_consecutive_wins=max_win_streak,
            max_consecutive_losses=max_loss_streak,
            first_trade_time=trades[0].timestamp if trades else None,
            last_trade_time=trades[-1].timestamp if trades else None
        )
        
        return metrics

    def get_all_metrics(self) -> Dict[str, StrategyMetrics]:
        """
        Get metrics for all tracked strategies.

        Returns:
            Dictionary mapping strategy names to metrics
        """
        return {
            strategy_name: self.get_strategy_metrics(strategy_name)
            for strategy_name in self._trades.keys()
        }

    def compare_strategies(
        self,
        strategy_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compare performance across multiple strategies.

        Args:
            strategy_names: List of strategies to compare (all if None)

        Returns:
            Comparison dictionary with rankings and relative performance
        """
        if strategy_names is None:
            strategy_names = list(self._trades.keys())
        
        metrics_list = [
            self.get_strategy_metrics(name)
            for name in strategy_names
            if name in self._trades
        ]
        
        if not metrics_list:
            return {}
        
        # Rank by different metrics
        rankings = {
            "by_total_pnl": sorted(
                metrics_list,
                key=lambda m: m.total_pnl,
                reverse=True
            ),
            "by_win_rate": sorted(
                metrics_list,
                key=lambda m: m.win_rate,
                reverse=True
            ),
            "by_sharpe_ratio": sorted(
                [m for m in metrics_list if m.sharpe_ratio is not None],
                key=lambda m: m.sharpe_ratio,
                reverse=True
            ),
            "by_avg_pnl": sorted(
                metrics_list,
                key=lambda m: m.average_pnl,
                reverse=True
            )
        }
        
        return {
            "strategies_compared": len(metrics_list),
            "rankings": {
                key: [m.strategy_name for m in ranked_list]
                for key, ranked_list in rankings.items()
            },
            "best_overall": rankings["by_total_pnl"][0].strategy_name if rankings["by_total_pnl"] else None,
            "metrics": {m.strategy_name: m for m in metrics_list}
        }

    def get_recent_trades(
        self,
        strategy_name: str,
        limit: int = 10
    ) -> List[TradeRecord]:
        """
        Get most recent trades for a strategy.

        Args:
            strategy_name: Strategy name
            limit: Maximum number of trades to return

        Returns:
            List of recent trade records
        """
        trades = self._trades.get(strategy_name, [])
        return sorted(trades, key=lambda t: t.timestamp, reverse=True)[:limit]

    def get_open_trades(self, strategy_name: Optional[str] = None) -> List[TradeRecord]:
        """
        Get currently open trades.

        Args:
            strategy_name: Optional strategy filter

        Returns:
            List of open trade records
        """
        if strategy_name:
            return self._open_trades.get(strategy_name, []).copy()
        
        # Return all open trades
        all_open = []
        for trades in self._open_trades.values():
            all_open.extend(trades)
        return all_open

    def clear_history(self, strategy_name: Optional[str] = None) -> None:
        """
        Clear trade history.

        Args:
            strategy_name: Clear specific strategy (all if None)
        """
        if strategy_name:
            self._trades[strategy_name].clear()
            self._invalidate_cache(strategy_name)
            self.logger.info(f"Cleared history for strategy: {strategy_name}")
        else:
            self._trades.clear()
            self._metrics_cache.clear()
            self._cache_valid.clear()
            self.logger.info("Cleared all strategy histories")

    def _invalidate_cache(self, strategy_name: str) -> None:
        """Invalidate cached metrics for a strategy."""
        self._cache_valid[strategy_name] = False

    def export_metrics(
        self,
        strategy_name: str,
        format: str = "dict"
    ) -> Any:
        """
        Export strategy metrics in specified format.

        Args:
            strategy_name: Strategy to export
            format: Export format ('dict', 'json')

        Returns:
            Exported metrics
        """
        metrics = self.get_strategy_metrics(strategy_name)
        
        if format == "dict":
            return {
                "strategy_name": metrics.strategy_name,
                "total_trades": metrics.total_trades,
                "win_rate": metrics.win_rate,
                "total_pnl": float(metrics.total_pnl),
                "average_pnl": float(metrics.average_pnl),
                "sharpe_ratio": metrics.sharpe_ratio,
                "max_drawdown": float(metrics.max_drawdown),
                "best_trade": float(metrics.best_trade),
                "worst_trade": float(metrics.worst_trade)
            }
        
        # Add more formats as needed
        return metrics
