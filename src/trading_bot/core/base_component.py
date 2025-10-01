"""
Base component interface for all trading bot components.

This module provides the BaseComponent abstract class that defines
the standard interface and lifecycle management for all system components.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional


class ComponentState(Enum):
    """Component lifecycle states."""

    INITIALIZED = "initialized"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class BaseComponent(ABC):
    """
    Abstract base class for all trading bot components.

    Provides standardized lifecycle management, logging, and error handling
    for all system components including strategies, executors, and analyzers.
    """

    def __init__(self, name: str):
        """
        Initialize the component.

        Args:
            name: Unique name identifier for this component
        """
        self.name = name
        self.state = ComponentState.INITIALIZED
        self.ComponentState = ComponentState  # Make accessible as instance attribute
        self.logger = logging.getLogger(f"trading_bot.{name}")
        self._start_event = asyncio.Event()
        self._stop_event = asyncio.Event()
        self._error: Optional[Exception] = None

    @property
    def _state(self) -> ComponentState:
        """Alias for state (for backward compatibility)."""
        return self.state
    
    @_state.setter
    def _state(self, value: ComponentState) -> None:
        """Alias for state (for backward compatibility)."""
        self.state = value

    @abstractmethod
    async def _start(self) -> None:
        """
        Component-specific startup logic.

        Subclasses must implement this method to define their
        initialization and startup behavior.
        """
        pass

    @abstractmethod
    async def _stop(self) -> None:
        """
        Component-specific shutdown logic.

        Subclasses must implement this method to define their
        cleanup and shutdown behavior.
        """
        pass

    async def start(self) -> None:
        """
        Start the component with proper state management.

        This method handles the component lifecycle transition from
        initialized/stopped to running state with error handling.
        """
        if self.state in [ComponentState.RUNNING, ComponentState.STARTING]:
            self.logger.warning(f"Component {self.name} is already running or starting")
            return

        try:
            self.logger.info(f"Starting component: {self.name}")
            self.state = ComponentState.STARTING

            await self._start()

            self.state = ComponentState.RUNNING
            self._start_event.set()
            self.logger.info(f"Component {self.name} started successfully")

        except Exception as e:
            self.state = ComponentState.ERROR
            self._error = e
            self.logger.error(f"Failed to start component {self.name}: {e}")
            raise

    async def stop(self) -> None:
        """
        Stop the component with graceful shutdown.

        This method handles the component lifecycle transition from
        running to stopped state with proper cleanup.
        """
        if self.state in [ComponentState.STOPPED, ComponentState.STOPPING]:
            self.logger.warning(f"Component {self.name} is already stopped or stopping")
            return

        try:
            self.logger.info(f"Stopping component: {self.name}")
            self.state = ComponentState.STOPPING

            await self._stop()

            self.state = ComponentState.STOPPED
            self._stop_event.set()
            self._start_event.clear()
            self.logger.info(f"Component {self.name} stopped successfully")

        except Exception as e:
            self.state = ComponentState.ERROR
            self._error = e
            self.logger.error(f"Failed to stop component {self.name}: {e}")
            raise

    def is_running(self) -> bool:
        """
        Check if the component is currently running.

        Returns:
            True if component is in RUNNING state, False otherwise
        """
        return self.state == ComponentState.RUNNING

    def is_healthy(self) -> bool:
        """
        Check if the component is in a healthy state.

        Returns:
            True if component is not in ERROR state, False otherwise
        """
        return self.state != ComponentState.ERROR

    def get_error(self) -> Optional[Exception]:
        """
        Get the last error that occurred in this component.

        Returns:
            The last exception that put the component in ERROR state,
            None if no error occurred
        """
        return self._error

    async def wait_for_start(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for the component to complete startup.

        Args:
            timeout: Maximum time to wait in seconds, None for no timeout

        Returns:
            True if component started successfully, False if timeout
        """
        try:
            await asyncio.wait_for(self._start_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def wait_for_stop(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for the component to complete shutdown.

        Args:
            timeout: Maximum time to wait in seconds, None for no timeout

        Returns:
            True if component stopped successfully, False if timeout
        """
        try:
            await asyncio.wait_for(self._stop_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    def __repr__(self) -> str:
        """String representation of the component."""
        return (
            f"{self.__class__.__name__}(name='{self.name}', state={self.state.value})"
        )
