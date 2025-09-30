"""
MonteCarloSimulator: Monte Carlo simulation for portfolio risk analysis.

Performs probabilistic return simulation using bootstrap resampling to estimate
Value at Risk, Conditional VaR, maximum loss probabilities, and stress scenarios.
"""

from decimal import Decimal
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


class MonteCarloSimulator:
    """
    Monte Carlo simulator for risk analysis and probabilistic forecasting.

    Features:
    - Bootstrap resampling of historical returns
    - VaR and CVaR calculation
    - Maximum loss probability estimation
    - Performance distribution generation
    - Portfolio stress testing
    """

    def __init__(self, n_simulations: int = 10000, random_seed: Optional[int] = None):
        """
        Initialize Monte Carlo simulator.

        Args:
            n_simulations: Number of simulation runs (default: 10,000)
            random_seed: Random seed for reproducibility
        """
        self.n_simulations = n_simulations
        if random_seed is not None:
            np.random.seed(random_seed)

    def simulate_returns(
        self,
        historical_returns: List[float],
        n_periods: int,
        initial_capital: Decimal = Decimal("10000"),
    ) -> Dict:
        """
        Simulate future returns using bootstrap resampling.

        Args:
            historical_returns: List of historical period returns
            n_periods: Number of periods to simulate
            initial_capital: Starting capital

        Returns:
            Dictionary with simulation results
        """
        if len(historical_returns) < 30:
            raise ValueError("Need at least 30 historical returns for reliable simulation")

        returns_array = np.array(historical_returns)

        # Run simulations
        simulated_paths = []
        final_capitals = []

        for _ in range(self.n_simulations):
            # Bootstrap sample returns
            sampled_returns = np.random.choice(returns_array, size=n_periods, replace=True)

            # Calculate equity curve for this path
            capital = float(initial_capital)
            path = [capital]

            for ret in sampled_returns:
                capital *= (1 + ret)
                path.append(capital)

            simulated_paths.append(path)
            final_capitals.append(capital)

        # Calculate statistics
        final_capitals_array = np.array(final_capitals)

        # Value at Risk (95% and 99%)
        var_95 = np.percentile(final_capitals_array, 5)
        var_99 = np.percentile(final_capitals_array, 1)

        # Conditional VaR (Expected Shortfall)
        cvar_95 = final_capitals_array[final_capitals_array <= var_95].mean()
        cvar_99 = final_capitals_array[final_capitals_array <= var_99].mean()

        # Maximum loss probabilities
        prob_loss = (final_capitals_array < float(initial_capital)).sum() / self.n_simulations
        prob_loss_10pct = (final_capitals_array < float(initial_capital) * 0.9).sum() / self.n_simulations
        prob_loss_20pct = (final_capitals_array < float(initial_capital) * 0.8).sum() / self.n_simulations

        # Distribution statistics
        mean_final = final_capitals_array.mean()
        std_final = final_capitals_array.std()
        median_final = np.percentile(final_capitals_array, 50)

        # Calculate percentiles
        percentiles = {
            f"p{p}": float(np.percentile(final_capitals_array, p))
            for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
        }

        return {
            "n_simulations": self.n_simulations,
            "n_periods": n_periods,
            "initial_capital": str(initial_capital),
            "mean_final_capital": float(mean_final),
            "median_final_capital": float(median_final),
            "std_final_capital": float(std_final),
            "var_95": float(var_95),
            "var_99": float(var_99),
            "cvar_95": float(cvar_95),
            "cvar_99": float(cvar_99),
            "prob_any_loss": float(prob_loss),
            "prob_loss_10pct": float(prob_loss_10pct),
            "prob_loss_20pct": float(prob_loss_20pct),
            "percentiles": percentiles,
            "simulated_paths": simulated_paths[:100],  # Store first 100 paths for visualization
        }

    def calculate_var(
        self,
        returns: List[float],
        confidence_level: float = 0.95,
        horizon_days: int = 1,
    ) -> float:
        """
        Calculate Value at Risk.

        Args:
            returns: Historical returns
            confidence_level: Confidence level (0.95 for 95%)
            horizon_days: Time horizon in days

        Returns:
            VaR as percentage
        """
        returns_array = np.array(returns)
        var = np.percentile(returns_array, (1 - confidence_level) * 100)

        # Scale by horizon (assuming i.i.d. returns)
        var_scaled = var * np.sqrt(horizon_days)

        return float(var_scaled * 100)

    def calculate_cvar(
        self,
        returns: List[float],
        confidence_level: float = 0.95,
        horizon_days: int = 1,
    ) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall).

        Args:
            returns: Historical returns
            confidence_level: Confidence level
            horizon_days: Time horizon in days

        Returns:
            CVaR as percentage
        """
        returns_array = np.array(returns)
        var_threshold = np.percentile(returns_array, (1 - confidence_level) * 100)

        # Expected shortfall: average of returns worse than VaR
        tail_returns = returns_array[returns_array <= var_threshold]

        if len(tail_returns) == 0:
            return 0.0

        cvar = tail_returns.mean()

        # Scale by horizon
        cvar_scaled = cvar * np.sqrt(horizon_days)

        return float(cvar_scaled * 100)

    def stress_test(
        self,
        historical_returns: List[float],
        initial_capital: Decimal,
        scenarios: Optional[Dict[str, List[float]]] = None,
    ) -> Dict:
        """
        Perform stress testing under various market scenarios.

        Args:
            historical_returns: Historical returns
            initial_capital: Starting capital
            scenarios: Dictionary of scenario names to return sequences

        Returns:
            Dictionary with stress test results
        """
        if scenarios is None:
            # Create default stress scenarios
            returns_array = np.array(historical_returns)
            mean_return = returns_array.mean()
            std_return = returns_array.std()

            scenarios = {
                "market_crash": [mean_return - 3 * std_return] * 20,
                "prolonged_downturn": [mean_return - std_return] * 60,
                "high_volatility": np.random.normal(mean_return, std_return * 2, 30).tolist(),
                "black_swan": [mean_return - 5 * std_return] * 5,
            }

        results = {}

        for scenario_name, scenario_returns in scenarios.items():
            capital = float(initial_capital)

            for ret in scenario_returns:
                capital *= (1 + ret)

            final_capital = capital
            total_return = (final_capital / float(initial_capital) - 1) * 100
            drawdown = ((final_capital - float(initial_capital)) / float(initial_capital)) * 100

            results[scenario_name] = {
                "initial_capital": str(initial_capital),
                "final_capital": float(final_capital),
                "total_return_pct": float(total_return),
                "drawdown_pct": float(drawdown) if drawdown < 0 else 0.0,
                "n_periods": len(scenario_returns),
            }

        return results

    def estimate_max_loss_probability(
        self,
        historical_returns: List[float],
        loss_threshold_pct: float,
        horizon_days: int = 30,
    ) -> float:
        """
        Estimate probability of exceeding maximum loss threshold.

        Args:
            historical_returns: Historical returns
            loss_threshold_pct: Loss threshold as percentage (e.g., 10 for 10%)
            horizon_days: Time horizon

        Returns:
            Probability of exceeding loss threshold
        """
        returns_array = np.array(historical_returns)

        # Simulate returns over horizon
        simulated_losses = []

        for _ in range(self.n_simulations):
            # Sample returns for horizon period
            period_returns = np.random.choice(returns_array, size=horizon_days, replace=True)

            # Calculate cumulative return
            cumulative_return = np.prod(1 + period_returns) - 1

            # Record if loss exceeds threshold
            if cumulative_return < -(loss_threshold_pct / 100):
                simulated_losses.append(1)
            else:
                simulated_losses.append(0)

        probability = np.mean(simulated_losses)

        return float(probability)

    def generate_confidence_intervals(
        self,
        historical_returns: List[float],
        n_periods: int,
        initial_capital: Decimal,
        confidence_levels: List[float] = [0.68, 0.95, 0.99],
    ) -> Dict:
        """
        Generate confidence intervals for future equity curves.

        Args:
            historical_returns: Historical returns
            n_periods: Number of periods to project
            initial_capital: Starting capital
            confidence_levels: List of confidence levels

        Returns:
            Dictionary with confidence interval bounds
        """
        returns_array = np.array(historical_returns)

        # Simulate paths
        simulated_finals = []

        for _ in range(self.n_simulations):
            sampled_returns = np.random.choice(returns_array, size=n_periods, replace=True)
            final_capital = float(initial_capital) * np.prod(1 + sampled_returns)
            simulated_finals.append(final_capital)

        finals_array = np.array(simulated_finals)

        # Calculate confidence intervals
        intervals = {}

        for level in confidence_levels:
            lower_pct = (1 - level) / 2 * 100
            upper_pct = (1 - (1 - level) / 2) * 100

            intervals[f"ci_{int(level*100)}"] = {
                "lower": float(np.percentile(finals_array, lower_pct)),
                "upper": float(np.percentile(finals_array, upper_pct)),
                "confidence_level": level,
            }

        return {
            "initial_capital": str(initial_capital),
            "n_periods": n_periods,
            "confidence_intervals": intervals,
            "expected_value": float(finals_array.mean()),
        }