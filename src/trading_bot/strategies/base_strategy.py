"""
Abstract base class for trading strategies.

This module provides the AbstractStrategy class that defines the standard
interface for all trading strategies in the system.
"""

from abc import abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd

from ..core.base_component import BaseComponent
from ..core.events import MarketDataEvent
from ..signals.signal_generator import GeneratedSignal


class AbstractStrategy(BaseComponent):
    """
    Abstract base class for all trading strategies.

    Provides standardized interface for strategy implementation,
    parameter management, and signal generation. All strategies
    must extend this class and implement the required abstract methods.
    """

    def __init__(
        self,
        name: str,
        version: str,
        description: str,
        parameters: Dict[str, Any]
    ):
        """
        Initialize the strategy.

        Args:
            name: Unique strategy identifier
            version: Strategy version string (e.g., "1.0.0")
            description: Human-readable strategy description
            parameters: Strategy-specific configuration parameters
        """
        super().__init__(name)
        self.version = version
        self.description = description
        self.parameters = parameters.copy()
        self._validate_parameters(self.parameters)

    @abstractmethod
    async def generate_signals(
        self,
        df: pd.DataFrame,
        current_event: Optional[MarketDataEvent] = None
    ) -> List[GeneratedSignal]:
        """
        Generate trading signals based on market data.

        Args:
            df: OHLCV DataFrame with market data (columns: open, high, low, close, volume, timestamp)
            current_event: Optional real-time market data event for live trading

        Returns:
            List of generated signals (can be empty if no signals)

        Raises:
            ValueError: If data is invalid or insufficient
        """
        pass

    @abstractmethod
    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """
        Validate strategy parameters.

        Args:
            params: Parameter dictionary to validate

        Returns:
            True if parameters are valid

        Raises:
            ValueError: If parameters are invalid with description of issue
        """
        pass

    @abstractmethod
    def get_parameter_schema(self) -> Dict[str, Any]:
        """
        Get parameter schema definition.

        Returns:
            Dictionary describing parameter names, types, defaults, and constraints
            Format:
            {
                "param_name": {
                    "type": "int|float|str|bool",
                    "default": value,
                    "min": min_value,  # optional
                    "max": max_value,  # optional
                    "description": "Parameter description"
                }
            }
        """
        pass

    def get_config(self) -> Dict[str, Any]:
        """
        Get current strategy configuration.

        Returns:
            Dictionary with strategy metadata and current parameters
        """
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "parameters": self.parameters.copy(),
            "state": self.state.value,
            "parameter_schema": self.get_parameter_schema()
        }

    def update_parameters(self, params: Dict[str, Any]) -> None:
        """
        Update strategy parameters with validation.

        Args:
            params: New parameters to apply

        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If strategy is currently running
        """
        if self.is_running():
            raise RuntimeError(
                f"Cannot update parameters while strategy {self.name} is running. "
                "Stop the strategy first."
            )

        # Validate new parameters
        self.validate_parameters(params)

        # Update parameters
        self.parameters.update(params)
        self.logger.info(f"Updated parameters for strategy {self.name}: {params}")

    def _validate_parameters(self, params: Dict[str, Any]) -> None:
        """
        Internal parameter validation wrapper.

        Args:
            params: Parameters to validate

        Raises:
            ValueError: If validation fails
        """
        try:
            if not self.validate_parameters(params):
                raise ValueError(f"Parameter validation failed for strategy {self.name}")
        except Exception as e:
            self.logger.error(f"Parameter validation error: {e}")
            raise

    async def _start(self) -> None:
        """
        Strategy-specific startup logic.

        Override this method to implement custom initialization.
        Default implementation logs startup.
        """
        self.logger.info(
            f"Strategy {self.name} v{self.version} started with parameters: {self.parameters}"
        )

    async def _stop(self) -> None:
        """
        Strategy-specific shutdown logic.

        Override this method to implement custom cleanup.
        Default implementation logs shutdown.
        """
        self.logger.info(f"Strategy {self.name} stopped")

    def __repr__(self) -> str:
        """String representation of the strategy."""
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"version='{self.version}', "
            f"state={self.state.value})"
        )
