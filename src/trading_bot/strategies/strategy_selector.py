"""
Strategy selector for runtime strategy switching.

This module provides safe runtime strategy switching with state
preservation and validation.
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
import asyncio

from .base_strategy import AbstractStrategy
from .strategy_registry import get_registry
from ..core.base_component import BaseComponent, ComponentState
from ..signals.signal_generator import GeneratedSignal


class StrategySelector(BaseComponent):
    """
    Manages runtime strategy switching with safety checks.

    Features:
    - Safe strategy switching (only when no active positions)
    - Strategy state preservation
    - Validation before switching
    - Automatic cleanup of old strategy
    - Event notification for strategy changes
    """

    def __init__(
        self,
        name: str = "StrategySelector",
        registry: Optional['StrategyRegistry'] = None,
        allow_mid_trade_switch: bool = False
    ):
        """
        Initialize the strategy selector.

        Args:
            name: Component name
            registry: Optional strategy registry (uses global if None)
            allow_mid_trade_switch: Allow switching strategies mid-trade
        """
        super().__init__(name)

        self._current_strategy: Optional[AbstractStrategy] = None
        self._previous_strategy: Optional[AbstractStrategy] = None
        self._switch_history: List[Dict[str, Any]] = []
        self._allow_mid_trade_switch = allow_mid_trade_switch
        self.registry = registry if registry is not None else get_registry()

    @property
    def current_strategy(self) -> Optional[AbstractStrategy]:
        """Get the currently active strategy."""
        return self._current_strategy

    @property
    def current_strategy_name(self) -> Optional[str]:
        """Get the name of the currently active strategy."""
        return self._current_strategy.name if self._current_strategy else None

    async def set_strategy(
        self,
        strategy_name: str,
        parameters: Optional[Dict[str, Any]] = None,
        force: bool = False
    ) -> None:
        """
        Set or switch to a different strategy.

        Args:
            strategy_name: Name of strategy to activate (must be registered)
            parameters: Strategy-specific parameters
            force: Force switch even if current strategy is running

        Raises:
            ValueError: If strategy is not registered or switch is unsafe
            RuntimeError: If switch fails
        """
        if not self.registry.is_registered(strategy_name):
            raise ValueError(
                f"Strategy '{strategy_name}' is not registered. "
                f"Available: {self.registry.list_strategies()}"
            )

        # Check if we're already using this strategy
        if self._current_strategy and self._current_strategy.name == strategy_name:
            self.logger.info(f"Already using strategy '{strategy_name}', no switch needed")
            
            # Update parameters if provided
            if parameters:
                self._current_strategy.update_parameters(parameters)
            
            return

        # Safety check: don't switch during active trading unless forced
        if self._current_strategy and self._current_strategy.is_running() and not force:
            if not self._allow_mid_trade_switch:
                raise RuntimeError(
                    "Cannot switch strategy while running"
                )

        try:
            # Store previous strategy
            self._previous_strategy = self._current_strategy

            # Stop current strategy if running
            if self._current_strategy and self._current_strategy.is_running():
                self.logger.info(f"Stopping current strategy: {self._current_strategy.name}")
                await self._current_strategy.stop()

            # Create new strategy instance
            self.logger.info(f"Creating new strategy: {strategy_name}")
            new_strategy = self.registry.create_strategy(
                strategy_name,
                parameters=parameters
            )

            # Start new strategy if selector is running
            if self.is_running():
                self.logger.info(f"Starting new strategy: {strategy_name}")
                await new_strategy.start()

            # Update current strategy
            self._current_strategy = new_strategy

            # Record switch in history
            self._switch_history.append({
                "timestamp": datetime.now(),
                "from_strategy": self._previous_strategy.name if self._previous_strategy else None,
                "to_strategy": strategy_name,
                "parameters": parameters,
                "forced": force
            })

            self.logger.info(
                f"Successfully switched to strategy: {strategy_name} "
                f"(force={force}, parameters={bool(parameters)})"
            )

        except Exception as e:
            self.logger.error(f"Failed to switch strategy: {e}", exc_info=True)
            
            # Attempt rollback to previous strategy
            if self._previous_strategy:
                self.logger.warning("Attempting to rollback to previous strategy")
                try:
                    self._current_strategy = self._previous_strategy
                    if self.is_running():
                        await self._current_strategy.start()
                except Exception as rollback_error:
                    self.logger.error(f"Rollback failed: {rollback_error}")
            
            raise RuntimeError(f"Strategy switch failed: {e}")

    async def switch_strategy(
        self,
        strategy_name: str,
        parameters: Optional[Dict[str, Any]] = None,
        wait_for_completion: bool = True
    ) -> None:
        """
        Switch to a different strategy safely.

        Alias for set_strategy with additional waiting logic.

        Args:
            strategy_name: Strategy to switch to
            parameters: Strategy parameters
            wait_for_completion: Wait for current strategy to complete operations

        Raises:
            ValueError: If strategy is not registered
            RuntimeError: If switch fails
        """
        if wait_for_completion and self._current_strategy:
            if self._current_strategy.is_running():
                self.logger.info("Waiting for current strategy to stop...")
                await self._current_strategy.stop()
                await self._current_strategy.wait_for_stop(timeout=30.0)

        await self.set_strategy(strategy_name, parameters, force=False)

    async def get_current_strategy_config(self) -> Optional[Dict[str, Any]]:
        """
        Get configuration of the currently active strategy.

        Returns:
            Strategy configuration or None if no strategy is active
        """
        if not self._current_strategy:
            return None

        return self._current_strategy.get_config()

    async def generate_signals(self, *args, **kwargs) -> List[GeneratedSignal]:
        """
        Generate signals using the current strategy.

        Args:
            *args: Positional arguments passed to strategy.generate_signals()
            **kwargs: Keyword arguments passed to strategy.generate_signals()

        Returns:
            List of generated signals

        Raises:
            RuntimeError: If no strategy is active
        """
        if not self._current_strategy:
            raise RuntimeError("No active strategy. Set a strategy first using set_strategy()")

        if not self._current_strategy.is_running():
            raise RuntimeError(
                f"Strategy {self._current_strategy.name} is not running. "
                "Start it first using start()"
            )

        try:
            signals = await self._current_strategy.generate_signals(*args, **kwargs)
            return signals

        except Exception as e:
            self.logger.error(f"Signal generation failed: {e}", exc_info=True)
            raise

    def get_switch_history(self) -> List[Dict[str, Any]]:
        """
        Get history of strategy switches.

        Returns:
            List of switch records with timestamps and details
        """
        return self._switch_history.copy()

    def clear_switch_history(self) -> None:
        """Clear the strategy switch history."""
        self._switch_history.clear()
        self.logger.info("Cleared strategy switch history")

    def allow_mid_trade_switching(self, allow: bool) -> None:
        """
        Configure whether mid-trade strategy switching is allowed.

        Args:
            allow: True to allow switching during active trading (dangerous)

        Warning:
            Allowing mid-trade switching can lead to inconsistent state
            and should only be used in specific scenarios.
        """
        self._allow_mid_trade_switch = allow
        self.logger.warning(
            f"Mid-trade strategy switching: {'ENABLED' if allow else 'DISABLED'}"
        )

    async def _start(self) -> None:
        """Start the strategy selector and current strategy."""
        if self._current_strategy:
            self.logger.info(f"Starting current strategy: {self._current_strategy.name}")
            await self._current_strategy.start()
        else:
            self.logger.warning("No strategy set, selector started but no strategy is active")

    async def _stop(self) -> None:
        """Stop the strategy selector and current strategy."""
        if self._current_strategy and self._current_strategy.is_running():
            self.logger.info(f"Stopping current strategy: {self._current_strategy.name}")
            await self._current_strategy.stop()

    def __repr__(self) -> str:
        """String representation of the selector."""
        strategy_name = self._current_strategy.name if self._current_strategy else "None"
        return (
            f"StrategySelector("
            f"current={strategy_name}, "
            f"switches={len(self._switch_history)}, "
            f"state={self.state.value})"
        )
