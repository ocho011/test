"""
BacktestVisualizer: Visualization and export functionality for backtest results.

Provides matplotlib-based charts, plotly interactive visualizations, and
export capabilities to CSV/JSON/Excel formats with automated report generation.
"""

import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np


class BacktestVisualizer:
    """
    Visualization and export handler for backtest results.

    Features:
    - Matplotlib-based static charts (equity curve, drawdown, trade distribution)
    - Plotly interactive charts for web dashboards
    - CSV/JSON/Excel export
    - Automated performance report generation
    - Web dashboard API endpoint support
    """

    def __init__(self, output_dir: str = "backtest_results"):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_to_csv(
        self,
        equity_curve: List[Dict],
        trades: List[Dict],
        performance_metrics: Dict,
        prefix: str = "backtest",
    ) -> Dict[str, str]:
        """
        Export backtest results to CSV files.

        Args:
            equity_curve: Equity curve data
            trades: Trade data
            performance_metrics: Performance metrics
            prefix: Filename prefix

        Returns:
            Dictionary with file paths
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Export equity curve
        equity_df = pd.DataFrame(equity_curve)
        equity_path = self.output_dir / f"{prefix}_equity_{timestamp}.csv"
        equity_df.to_csv(equity_path, index=False)

        # Export trades
        trades_df = pd.DataFrame(trades)
        trades_path = self.output_dir / f"{prefix}_trades_{timestamp}.csv"
        trades_df.to_csv(trades_path, index=False)

        # Export performance metrics
        metrics_df = pd.DataFrame([performance_metrics])
        metrics_path = self.output_dir / f"{prefix}_metrics_{timestamp}.csv"
        metrics_df.to_csv(metrics_path, index=False)

        return {
            "equity_curve": str(equity_path),
            "trades": str(trades_path),
            "metrics": str(metrics_path),
        }

    def export_to_json(
        self,
        results: Dict,
        filename: Optional[str] = None,
    ) -> str:
        """
        Export complete backtest results to JSON.

        Args:
            results: Complete backtest results dictionary
            filename: Custom filename (optional)

        Returns:
            File path
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_results_{timestamp}.json"

        filepath = self.output_dir / filename

        # Convert Decimal objects to strings for JSON serialization
        json_results = self._prepare_for_json(results)

        with open(filepath, 'w') as f:
            json.dump(json_results, f, indent=2)

        return str(filepath)

    def export_to_excel(
        self,
        equity_curve: List[Dict],
        trades: List[Dict],
        performance_metrics: Dict,
        trade_analysis: Optional[Dict] = None,
        filename: Optional[str] = None,
    ) -> str:
        """
        Export backtest results to Excel with multiple sheets.

        Args:
            equity_curve: Equity curve data
            trades: Trade data
            performance_metrics: Performance metrics
            trade_analysis: Trade analysis results (optional)
            filename: Custom filename (optional)

        Returns:
            File path
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_report_{timestamp}.xlsx"

        filepath = self.output_dir / filename

        # Create Excel writer
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Equity curve sheet
            equity_df = pd.DataFrame(equity_curve)
            equity_df.to_excel(writer, sheet_name='Equity Curve', index=False)

            # Trades sheet
            trades_df = pd.DataFrame(trades)
            trades_df.to_excel(writer, sheet_name='Trades', index=False)

            # Performance metrics sheet
            metrics_df = pd.DataFrame([performance_metrics])
            metrics_df = metrics_df.T
            metrics_df.columns = ['Value']
            metrics_df.to_excel(writer, sheet_name='Performance Metrics')

            # Trade analysis sheet (if provided)
            if trade_analysis:
                analysis_df = pd.DataFrame([trade_analysis])
                analysis_df = analysis_df.T
                analysis_df.columns = ['Value']
                analysis_df.to_excel(writer, sheet_name='Trade Analysis')

        return str(filepath)

    def generate_performance_report(
        self,
        backtest_results: Dict,
        performance_metrics: Dict,
        trade_analysis: Optional[Dict] = None,
        filename: Optional[str] = None,
    ) -> str:
        """
        Generate comprehensive performance report in markdown format.

        Args:
            backtest_results: Backtest results
            performance_metrics: Performance metrics
            trade_analysis: Trade analysis (optional)
            filename: Custom filename (optional)

        Returns:
            File path
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_report_{timestamp}.md"

        filepath = self.output_dir / filename

        # Build report content
        report_lines = []

        # Header
        report_lines.append("# Backtesting Performance Report")
        report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Executive Summary
        report_lines.append("## Executive Summary\n")
        report_lines.append(f"- **Initial Capital**: {backtest_results.get('initial_capital', 'N/A')}")
        report_lines.append(f"- **Final Equity**: {backtest_results.get('final_equity', 'N/A')}")
        report_lines.append(f"- **Total Return**: {backtest_results.get('total_return_pct', 0):.2f}%")
        report_lines.append(f"- **Total Trades**: {backtest_results.get('total_trades', 0)}")
        report_lines.append(f"- **Win Rate**: {backtest_results.get('win_rate', 0):.2f}%\n")

        # Performance Metrics
        report_lines.append("## Performance Metrics\n")
        report_lines.append("### Risk-Adjusted Returns")
        report_lines.append(f"- **Sharpe Ratio**: {performance_metrics.get('sharpe_ratio', 0):.3f}")
        report_lines.append(f"- **Sortino Ratio**: {performance_metrics.get('sortino_ratio', 0):.3f}")
        report_lines.append(f"- **Calmar Ratio**: {performance_metrics.get('calmar_ratio', 0):.3f}\n")

        report_lines.append("### Risk Metrics")
        report_lines.append(f"- **Maximum Drawdown**: {performance_metrics.get('max_drawdown', 0):.2f}%")
        report_lines.append(f"- **Max DD Duration**: {performance_metrics.get('max_drawdown_duration_days', 0)} days")
        report_lines.append(f"- **Value at Risk (95%)**: {performance_metrics.get('value_at_risk_95', 0):.2f}%")
        report_lines.append(f"- **Conditional VaR (95%)**: {performance_metrics.get('conditional_var_95', 0):.2f}%\n")

        report_lines.append("### Trade Statistics")
        report_lines.append(f"- **Profit Factor**: {performance_metrics.get('profit_factor', 0):.2f}")
        report_lines.append(f"- **Avg Profit/Loss Ratio**: {performance_metrics.get('avg_profit_loss_ratio', 0):.2f}")
        report_lines.append(f"- **Annualized Return**: {performance_metrics.get('annualized_return_pct', 0):.2f}%")
        report_lines.append(f"- **Annualized Volatility**: {performance_metrics.get('annualized_volatility_pct', 0):.2f}%\n")

        # Trade Analysis (if available)
        if trade_analysis:
            report_lines.append("## Trade Analysis\n")

            report_lines.append("### Holding Period Analysis")
            report_lines.append(f"- **Avg Holding Duration**: {trade_analysis.get('avg_holding_hours', 0):.1f} hours")
            report_lines.append(f"- **Avg Winning Hold**: {trade_analysis.get('avg_winning_hold_hours', 0):.1f} hours")
            report_lines.append(f"- **Avg Losing Hold**: {trade_analysis.get('avg_losing_hold_hours', 0):.1f} hours\n")

            report_lines.append("### Consecutive Trades")
            report_lines.append(f"- **Max Consecutive Wins**: {trade_analysis.get('max_consecutive_wins', 0)}")
            report_lines.append(f"- **Max Consecutive Losses**: {trade_analysis.get('max_consecutive_losses', 0)}\n")

        # Period Analysis
        if 'start_time' in backtest_results and 'end_time' in backtest_results:
            report_lines.append("## Testing Period\n")
            report_lines.append(f"- **Start Date**: {backtest_results['start_time']}")
            report_lines.append(f"- **End Date**: {backtest_results['end_time']}\n")

        # Footer
        report_lines.append("\n---\n")
        report_lines.append("*This report was automatically generated by BacktestVisualizer*")

        # Write report
        with open(filepath, 'w') as f:
            f.write('\n'.join(report_lines))

        return str(filepath)

    def create_dashboard_data(
        self,
        backtest_results: Dict,
        performance_metrics: Dict,
        trade_analysis: Optional[Dict] = None,
    ) -> Dict:
        """
        Create JSON data structure optimized for web dashboard consumption.

        Args:
            backtest_results: Backtest results
            performance_metrics: Performance metrics
            trade_analysis: Trade analysis (optional)

        Returns:
            Dashboard-ready data structure
        """
        dashboard_data = {
            "summary": {
                "initial_capital": backtest_results.get("initial_capital"),
                "final_equity": backtest_results.get("final_equity"),
                "total_return_pct": backtest_results.get("total_return_pct"),
                "total_trades": backtest_results.get("total_trades"),
                "winning_trades": backtest_results.get("winning_trades"),
                "losing_trades": backtest_results.get("losing_trades"),
                "win_rate": backtest_results.get("win_rate"),
            },
            "performance": {
                "sharpe_ratio": performance_metrics.get("sharpe_ratio"),
                "sortino_ratio": performance_metrics.get("sortino_ratio"),
                "max_drawdown": performance_metrics.get("max_drawdown"),
                "profit_factor": performance_metrics.get("profit_factor"),
                "annualized_return_pct": performance_metrics.get("annualized_return_pct"),
            },
            "equity_curve": backtest_results.get("equity_curve", []),
            "trades": backtest_results.get("trades", []),
        }

        if trade_analysis:
            dashboard_data["trade_analysis"] = trade_analysis

        return dashboard_data

    def _prepare_for_json(self, obj):
        """Recursively convert Decimal objects to strings for JSON serialization."""
        if isinstance(obj, Decimal):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: self._prepare_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_json(item) for item in obj]
        elif isinstance(obj, datetime):
            return obj.isoformat()
        else:
            return obj

    def get_summary_statistics(self, backtest_results: Dict, performance_metrics: Dict) -> Dict:
        """
        Extract key summary statistics for quick display.

        Args:
            backtest_results: Backtest results
            performance_metrics: Performance metrics

        Returns:
            Summary statistics dictionary
        """
        return {
            "returns": {
                "total_return_pct": backtest_results.get("total_return_pct", 0),
                "annualized_return_pct": performance_metrics.get("annualized_return_pct", 0),
            },
            "risk": {
                "sharpe_ratio": performance_metrics.get("sharpe_ratio", 0),
                "max_drawdown_pct": performance_metrics.get("max_drawdown", 0),
                "volatility_pct": performance_metrics.get("annualized_volatility_pct", 0),
            },
            "trades": {
                "total": backtest_results.get("total_trades", 0),
                "win_rate_pct": backtest_results.get("win_rate", 0),
                "profit_factor": performance_metrics.get("profit_factor", 0),
            },
        }