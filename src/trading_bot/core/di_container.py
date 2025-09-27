"""
Dependency injection container for managing component dependencies.

This module provides a DI container for automatic dependency resolution
and lifecycle management of components in the trading system.
"""

import asyncio
import inspect
import logging
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Type, TypeVar, Union
from weakref import WeakValueDictionary

T = TypeVar("T")


class Lifetime(Enum):
    """Dependency lifetime management options."""

    SINGLETON = "singleton"  # Single instance per container
    TRANSIENT = "transient"  # New instance per resolution
    SCOPED = "scoped"  # Single instance per scope


class DependencyDescriptor:
    """Describes how to create and manage a dependency."""

    def __init__(
        self,
        factory: Union[Type[T], Callable[..., T]],
        lifetime: Lifetime = Lifetime.TRANSIENT,
        name: Optional[str] = None,
        dependencies: Optional[List[str]] = None,
    ):
        """
        Initialize dependency descriptor.

        Args:
            factory: Class or factory function to create instances
            lifetime: Instance lifetime management
            name: Optional name for named dependencies
            dependencies: List of dependency names this depends on
        """
        self.factory = factory
        self.lifetime = lifetime
        self.name = name
        self.dependencies = dependencies or []
        self.instance: Optional[T] = None

    def __repr__(self) -> str:
        return f"DependencyDescriptor(factory={self.factory.__name__}, lifetime={self.lifetime.value})"


class CircularDependencyError(Exception):
    """Raised when circular dependencies are detected."""

    pass


class DependencyNotFoundError(Exception):
    """Raised when a required dependency cannot be resolved."""

    pass


class DIScope:
    """Represents a dependency injection scope for scoped instances."""

    def __init__(self, name: str):
        self.name = name
        self.instances: Dict[str, Any] = {}
        self.created_at = asyncio.get_event_loop().time()

    def get_instance(self, key: str) -> Optional[Any]:
        """Get scoped instance by key."""
        return self.instances.get(key)

    def set_instance(self, key: str, instance: Any) -> None:
        """Set scoped instance by key."""
        self.instances[key] = instance

    def clear(self) -> None:
        """Clear all scoped instances."""
        self.instances.clear()


class DIContainer:
    """
    Dependency injection container for automatic dependency resolution.

    Provides registration, resolution, and lifecycle management for
    components with support for circular dependency detection.
    """

    def __init__(self, name: str = "default"):
        """
        Initialize the DI container.

        Args:
            name: Container name for identification
        """
        self.name = name
        self.logger = logging.getLogger(f"trading_bot.di.{name}")

        # Dependency management
        self._descriptors: Dict[Type, DependencyDescriptor] = {}
        self._named_descriptors: Dict[str, DependencyDescriptor] = {}
        self._singletons: WeakValueDictionary = WeakValueDictionary()

        # Scope management
        self._current_scope: Optional[DIScope] = None
        self._scopes: Dict[str, DIScope] = {}

        # Circular dependency detection
        self._resolution_stack: Set[str] = set()

        # Built-in registrations
        self._register_builtin_dependencies()

    def _register_builtin_dependencies(self) -> None:
        """Register built-in dependencies like the container itself."""
        self.register_instance(DIContainer, self, name="di_container")

    def register_type(
        self,
        interface: Type[T],
        implementation: Type[T],
        lifetime: Lifetime = Lifetime.TRANSIENT,
        name: Optional[str] = None,
    ) -> None:
        """
        Register a type mapping in the container.

        Args:
            interface: Interface or base type
            implementation: Concrete implementation type
            lifetime: Instance lifetime management
            name: Optional name for named registration
        """
        descriptor = DependencyDescriptor(
            factory=implementation, lifetime=lifetime, name=name
        )

        if name:
            self._named_descriptors[name] = descriptor
        else:
            self._descriptors[interface] = descriptor

        self.logger.debug(
            f"Registered {implementation.__name__} for {interface.__name__} ({lifetime.value})"
        )

    def register_factory(
        self,
        interface: Type[T],
        factory: Callable[..., T],
        lifetime: Lifetime = Lifetime.TRANSIENT,
        name: Optional[str] = None,
    ) -> None:
        """
        Register a factory function for creating instances.

        Args:
            interface: Interface type the factory creates
            factory: Factory function
            lifetime: Instance lifetime management
            name: Optional name for named registration
        """
        descriptor = DependencyDescriptor(factory=factory, lifetime=lifetime, name=name)

        if name:
            self._named_descriptors[name] = descriptor
        else:
            self._descriptors[interface] = descriptor

        self.logger.debug(
            f"Registered factory for {interface.__name__} ({lifetime.value})"
        )

    def register_instance(
        self, interface: Type[T], instance: T, name: Optional[str] = None
    ) -> None:
        """
        Register a pre-created instance.

        Args:
            interface: Interface type
            instance: Pre-created instance
            name: Optional name for named registration
        """
        descriptor = DependencyDescriptor(
            factory=lambda: instance, lifetime=Lifetime.SINGLETON, name=name
        )
        descriptor.instance = instance

        if name:
            self._named_descriptors[name] = descriptor
        else:
            self._descriptors[interface] = descriptor

        self.logger.debug(f"Registered instance for {interface.__name__}")

    def resolve(self, interface: Type[T], name: Optional[str] = None) -> T:
        """
        Resolve a dependency from the container.

        Args:
            interface: Interface type to resolve
            name: Optional name for named resolution

        Returns:
            Instance of the requested type

        Raises:
            DependencyNotFoundError: If dependency cannot be found
            CircularDependencyError: If circular dependency detected
        """
        key = name if name else interface.__name__

        # Check for circular dependencies
        if key in self._resolution_stack:
            cycle = list(self._resolution_stack) + [key]
            raise CircularDependencyError(
                f"Circular dependency detected: {' -> '.join(cycle)}"
            )

        try:
            self._resolution_stack.add(key)
            return self._resolve_internal(interface, name)
        finally:
            self._resolution_stack.discard(key)

    def _resolve_internal(self, interface: Type[T], name: Optional[str] = None) -> T:
        """Internal resolution logic."""
        # Get descriptor
        descriptor = self._get_descriptor(interface, name)
        if not descriptor:
            raise DependencyNotFoundError(
                f"No registration found for {interface.__name__}"
            )

        # Handle different lifetimes
        if descriptor.lifetime == Lifetime.SINGLETON:
            return self._resolve_singleton(descriptor, interface, name)
        elif descriptor.lifetime == Lifetime.SCOPED:
            return self._resolve_scoped(descriptor, interface, name)
        else:  # TRANSIENT
            return self._create_instance(descriptor)

    def _get_descriptor(
        self, interface: Type[T], name: Optional[str] = None
    ) -> Optional[DependencyDescriptor]:
        """Get descriptor for interface/name combination."""
        if name:
            return self._named_descriptors.get(name)
        else:
            return self._descriptors.get(interface)

    def _resolve_singleton(
        self, descriptor: DependencyDescriptor, interface: Type[T], name: Optional[str]
    ) -> T:
        """Resolve singleton instance."""
        key = name if name else interface.__name__

        # Check if already created
        if descriptor.instance is not None:
            return descriptor.instance

        # Check weak reference cache
        if key in self._singletons:
            return self._singletons[key]

        # Create new singleton
        instance = self._create_instance(descriptor)
        descriptor.instance = instance
        self._singletons[key] = instance

        return instance

    def _resolve_scoped(
        self, descriptor: DependencyDescriptor, interface: Type[T], name: Optional[str]
    ) -> T:
        """Resolve scoped instance."""
        if not self._current_scope:
            # Fall back to singleton behavior if no scope
            return self._resolve_singleton(descriptor, interface, name)

        key = name if name else interface.__name__
        instance = self._current_scope.get_instance(key)

        if instance is None:
            instance = self._create_instance(descriptor)
            self._current_scope.set_instance(key, instance)

        return instance

    def _create_instance(self, descriptor: DependencyDescriptor) -> Any:
        """Create a new instance using the descriptor."""
        factory = descriptor.factory

        try:
            # Get constructor parameters
            if inspect.isclass(factory):
                sig = inspect.signature(factory.__init__)
                params = list(sig.parameters.values())[1:]  # Skip 'self'
            else:
                sig = inspect.signature(factory)
                params = list(sig.parameters.values())

            # Resolve dependencies
            kwargs = {}
            for param in params:
                if param.annotation != inspect.Parameter.empty:
                    # Try to resolve by type
                    try:
                        kwargs[param.name] = self.resolve(param.annotation)
                    except DependencyNotFoundError:
                        # Try to resolve by name
                        try:
                            kwargs[param.name] = self.resolve(
                                param.annotation, param.name
                            )
                        except DependencyNotFoundError:
                            if param.default == inspect.Parameter.empty:
                                raise DependencyNotFoundError(
                                    f"Cannot resolve parameter '{param.name}' of type {param.annotation}"
                                )

            # Create instance
            instance = factory(**kwargs)
            self.logger.debug(f"Created instance of {factory.__name__}")
            return instance

        except Exception as e:
            self.logger.error(f"Failed to create instance of {factory.__name__}: {e}")
            raise

    def create_scope(self, name: str) -> DIScope:
        """
        Create a new dependency injection scope.

        Args:
            name: Scope name

        Returns:
            New scope instance
        """
        scope = DIScope(name)
        self._scopes[name] = scope
        return scope

    def enter_scope(self, scope: Union[str, DIScope]) -> DIScope:
        """
        Enter a dependency injection scope.

        Args:
            scope: Scope name or instance

        Returns:
            The entered scope
        """
        if isinstance(scope, str):
            if scope not in self._scopes:
                self._scopes[scope] = DIScope(scope)
            scope_instance = self._scopes[scope]
        else:
            scope_instance = scope

        self._current_scope = scope_instance
        self.logger.debug(f"Entered scope: {scope_instance.name}")
        return scope_instance

    def exit_scope(self) -> None:
        """Exit the current dependency injection scope."""
        if self._current_scope:
            self.logger.debug(f"Exited scope: {self._current_scope.name}")
            self._current_scope = None

    def clear_scope(self, scope_name: str) -> None:
        """
        Clear a specific scope and its instances.

        Args:
            scope_name: Name of scope to clear
        """
        if scope_name in self._scopes:
            self._scopes[scope_name].clear()
            del self._scopes[scope_name]
            self.logger.debug(f"Cleared scope: {scope_name}")

    def is_registered(self, interface: Type[T], name: Optional[str] = None) -> bool:
        """
        Check if a type is registered in the container.

        Args:
            interface: Interface type to check
            name: Optional name to check

        Returns:
            True if registered, False otherwise
        """
        return self._get_descriptor(interface, name) is not None

    def get_registrations(self) -> Dict[str, DependencyDescriptor]:
        """
        Get all current registrations.

        Returns:
            Dictionary of all registrations
        """
        all_registrations = {}

        # Add type-based registrations
        for interface, descriptor in self._descriptors.items():
            all_registrations[interface.__name__] = descriptor

        # Add named registrations
        for name, descriptor in self._named_descriptors.items():
            all_registrations[f"named:{name}"] = descriptor

        return all_registrations

    def validate_dependencies(self) -> List[str]:
        """
        Validate all registered dependencies for circular references.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        for interface, descriptor in self._descriptors.items():
            try:
                self._resolution_stack.clear()
                self.resolve(interface)
            except CircularDependencyError as e:
                errors.append(f"{interface.__name__}: {str(e)}")
            except DependencyNotFoundError as e:
                errors.append(f"{interface.__name__}: {str(e)}")

        for name, descriptor in self._named_descriptors.items():
            try:
                self._resolution_stack.clear()
                # We need the interface type to resolve, but named descriptors don't store it
                # So we'll skip validation for named dependencies for now
                pass
            except Exception as e:
                errors.append(f"named:{name}: {str(e)}")

        return errors

    def dispose(self) -> None:
        """Dispose of the container and clean up resources."""
        self._current_scope = None
        self._scopes.clear()
        self._singletons.clear()
        self._descriptors.clear()
        self._named_descriptors.clear()
        self._resolution_stack.clear()

        self.logger.info(f"DI Container '{self.name}' disposed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.dispose()
