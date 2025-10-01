"""
Strategy registry for plugin-based strategy management.

This module provides centralized strategy registration, discovery,
and instantiation capabilities for the trading bot.
"""

import logging
from typing import Dict, Type, List, Optional, Any
from threading import Lock

from .base_strategy import AbstractStrategy


class StrategyRegistry:
    """
    Centralized registry for trading strategies.

    Provides plugin-like architecture for strategy management with:
    - Strategy registration and discovery
    - Version management
    - Strategy instantiation with validation
    - Thread-safe operations
    """

    _instance = None
    _lock = Lock()

    def __new__(cls):
        """Singleton pattern for global registry access."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the strategy registry."""
        if self._initialized:
            return

        self._strategies: Dict[str, Type[AbstractStrategy]] = {}
        self._strategy_metadata: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger("trading_bot.strategy_registry")
        self._initialized = True

        self.logger.info("StrategyRegistry initialized")

    def register(
        self,
        strategy_class: Type[AbstractStrategy],
        name: Optional[str] = None,
        version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register a strategy class in the registry.

        Args:
            strategy_class: Strategy class to register
            name: Optional custom name (uses class name if not provided)
            version: Optional version string
            metadata: Optional additional metadata

        Raises:
            ValueError: If strategy is invalid or already registered
            TypeError: If strategy_class is not a subclass of AbstractStrategy
        """
        if not issubclass(strategy_class, AbstractStrategy):
            raise TypeError(
                f"{strategy_class.__name__} must be a subclass of AbstractStrategy"
            )

        strategy_name = name or strategy_class.__name__

        with self._lock:
            if strategy_name in self._strategies:
                self.logger.warning(
                    f"Strategy '{strategy_name}' already registered, overwriting"
                )

            self._strategies[strategy_name] = strategy_class

            # Store metadata
            self._strategy_metadata[strategy_name] = {
                "class_name": strategy_class.__name__,
                "module": strategy_class.__module__,
                "version": version,
                "metadata": metadata or {}
            }

            self.logger.info(
                f"Registered strategy: {strategy_name} "
                f"(class: {strategy_class.__name__}, version: {version})"
            )

    def unregister(self, name: str) -> None:
        """
        Unregister a strategy from the registry.

        Args:
            name: Strategy name to unregister

        Raises:
            KeyError: If strategy is not registered
        """
        with self._lock:
            if name not in self._strategies:
                raise KeyError(f"Strategy '{name}' is not registered")

            del self._strategies[name]
            del self._strategy_metadata[name]

            self.logger.info(f"Unregistered strategy: {name}")

    def get_strategy_class(self, name: str) -> Type[AbstractStrategy]:
        """
        Get a registered strategy class.

        Args:
            name: Strategy name

        Returns:
            Strategy class

        Raises:
            KeyError: If strategy is not registered
        """
        if name not in self._strategies:
            available = ", ".join(self._strategies.keys())
            raise KeyError(
                f"Strategy '{name}' is not registered. "
                f"Available strategies: {available}"
            )

        return self._strategies[name]

    def create_strategy(
        self,
        name: str,
        parameters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> AbstractStrategy:
        """
        Create a strategy instance from the registry.

        Args:
            name: Registered strategy name
            parameters: Strategy-specific parameters
            **kwargs: Additional arguments passed to strategy constructor

        Returns:
            Instantiated strategy

        Raises:
            KeyError: If strategy is not registered
            ValueError: If instantiation fails
        """
        strategy_class = self.get_strategy_class(name)

        try:
            if parameters:
                kwargs["parameters"] = parameters

            strategy = strategy_class(**kwargs)

            self.logger.info(
                f"Created strategy instance: {name} "
                f"(parameters: {len(parameters or {})} params)"
            )

            return strategy

        except Exception as e:
            self.logger.error(
                f"Failed to create strategy '{name}': {e}",
                exc_info=True
            )
            raise ValueError(f"Failed to create strategy '{name}': {e}")

    def list_strategies(self) -> List[str]:
        """
        Get list of all registered strategy names.

        Returns:
            List of strategy names
        """
        return list(self._strategies.keys())

    def get_strategy_info(self, name: str) -> Dict[str, Any]:
        """
        Get detailed information about a registered strategy.

        Args:
            name: Strategy name

        Returns:
            Dictionary with strategy information

        Raises:
            KeyError: If strategy is not registered
        """
        if name not in self._strategies:
            raise KeyError(f"Strategy '{name}' is not registered")

        strategy_class = self._strategies[name]
        metadata = self._strategy_metadata[name]

        # Create a temporary instance to get parameter schema and description
        try:
            temp_instance = strategy_class()
            parameter_schema = temp_instance.get_parameter_schema()
            description = temp_instance.description if hasattr(temp_instance, 'description') else ""
        except Exception:
            parameter_schema = {}
            description = ""

        return {
            "name": name,
            "class_name": metadata["class_name"],
            "module": metadata["module"],
            "version": metadata.get("version"),
            "description": description,
            "parameter_schema": parameter_schema,
            "metadata": metadata.get("metadata", {})
        }

    def get_all_strategies_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all registered strategies.

        Returns:
            Dictionary mapping strategy names to their information
        """
        return {
            name: self.get_strategy_info(name)
            for name in self._strategies.keys()
        }

    def is_registered(self, name: str) -> bool:
        """
        Check if a strategy is registered.

        Args:
            name: Strategy name

        Returns:
            True if registered, False otherwise
        """
        return name in self._strategies

    def clear(self) -> None:
        """
        Clear all registered strategies.

        Warning: This removes all strategies from the registry.
        Use with caution.
        """
        with self._lock:
            count = len(self._strategies)
            self._strategies.clear()
            self._strategy_metadata.clear()

            self.logger.warning(f"Cleared registry: removed {count} strategies")

    def auto_register_builtin_strategies(self) -> None:
        """
        Automatically register built-in strategies.

        This is a convenience method to register all strategies
        included in the strategies module.
        """
        from .ict_strategy import ICTStrategy
        from .traditional_strategy import TraditionalIndicatorStrategy

        # Register with short names (primary)
        self.register(ICTStrategy, name="ICT", version="1.0.0")
        self.register(
            TraditionalIndicatorStrategy,
            name="Traditional",
            version="1.0.0"
        )
        
        # Register with full class names (for compatibility)
        self.register(ICTStrategy, name="ICTStrategy", version="1.0.0")
        self.register(
            TraditionalIndicatorStrategy,
            name="TraditionalIndicatorStrategy",
            version="1.0.0"
        )

        self.logger.info("Auto-registered built-in strategies")

    def __repr__(self) -> str:
        """String representation of the registry."""
        return (
            f"StrategyRegistry("
            f"strategies={len(self._strategies)}, "
            f"registered={list(self._strategies.keys())})"
        )


# Global registry instance
_registry = StrategyRegistry()


def get_registry() -> StrategyRegistry:
    """
    Get the global strategy registry instance.

    Returns:
        Global StrategyRegistry instance
    """
    return _registry
