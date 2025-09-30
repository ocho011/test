"""
Backtesting module for trading strategy validation and performance analysis.

This module provides comprehensive backtesting capabilities including:
- Historical data-based strategy testing
- Performance metrics calculation (Sharpe ratio, drawdown, win rate)
- Trade-level analysis and pattern recognition
- Walk-forward optimization for parameter tuning
- Monte Carlo simulation for risk analysis
- Result visualization and export functionality
"""

from .backtest_engine import BacktestEngine
from .performance_analyzer import PerformanceAnalyzer
from .trade_analyzer import TradeAnalyzer
from .walk_forward_optimizer import WalkForwardOptimizer
from .monte_carlo_simulator import MonteCarloSimulator
from .visualization import BacktestVisualizer

__all__ = [
    "BacktestEngine",
    "PerformanceAnalyzer",
    "TradeAnalyzer",
    "WalkForwardOptimizer",
    "MonteCarloSimulator",
    "BacktestVisualizer",
]