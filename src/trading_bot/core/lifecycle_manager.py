"""
Component lifecycle management system for orchestrating startup and shutdown.

This module provides the ComponentLifecycleManager for managing the lifecycle
of all system components with proper dependency ordering and health monitoring.
"""

import asyncio
import logging
from collections import defaultdict, deque
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from .base_component import BaseComponent, ComponentState
from .di_container import DIContainer
from .events import BaseEvent, EventType, RiskEvent, RiskEventType, RiskSeverity
from .message_hub import MessageHub


class StartupOrder(Enum):
    """Component startup order priorities."""
    INFRASTRUCTURE = 1    # Core infrastructure (DI, EventBus, etc.)
    DATA = 2              # Data providers and connections
    ANALYSIS = 3          # Analysis and strategy components
    EXECUTION = 4         # Order execution and risk management
    NOTIFICATION = 5      # Monitoring and notification systems


class ComponentDependency:
    """Represents a dependency relationship between components."""

    def __init__(self, component: BaseComponent, dependencies: List[str]):
        """
        Initialize component dependency.

        Args:
            component: The component instance
            dependencies: List of component names this component depends on
        """
        self.component = component
        self.dependencies = dependencies
        self.startup_order = StartupOrder.ANALYSIS  # Default order
        self.health_check_interval = 30.0  # seconds
        self.restart_attempts = 0
        self.max_restart_attempts = 3


class ComponentLifecycleManager(BaseComponent):
    """
    Manages the lifecycle of all system components.

    Handles component registration, dependency resolution, ordered startup/shutdown,
    health monitoring, and automatic restart capabilities.
    """

    def __init__(self, di_container: DIContainer, message_hub: MessageHub):
        super().__init__("ComponentLifecycleManager")

        self.di_container = di_container
        self.message_hub = message_hub

        # Component management
        self.components: Dict[str, ComponentDependency] = {}
        self.startup_order: List[List[str]] = [[] for _ in StartupOrder]
        self.running_components: Set[str] = set()

        # Health monitoring
        self.health_monitor_task: Optional[asyncio.Task] = None
        self.health_check_interval = 10.0  # seconds

        # Shutdown management
        self.shutdown_timeout = 30.0  # seconds
        self.force_shutdown = False

        # Statistics
        self.stats = {
            'components_started': 0,
            'components_stopped': 0,
            'startup_failures': 0,
            'restart_attempts': 0,
            'health_check_failures': 0
        }

    async def _start(self) -> None:
        """Start the lifecycle manager."""
        # Register with message hub
        await self.message_hub.register_component(
            self, capabilities=["lifecycle_management", "health_monitoring"]
        )

        # Start health monitoring
        self.health_monitor_task = asyncio.create_task(self._health_monitor_loop())

        self.logger.info("ComponentLifecycleManager started")

    async def _stop(self) -> None:
        """Stop the lifecycle manager."""
        # Stop health monitoring
        if self.health_monitor_task:
            self.health_monitor_task.cancel()
            try:
                await self.health_monitor_task
            except asyncio.CancelledError:
                pass

        self.logger.info("ComponentLifecycleManager stopped")

    def register_component(
        self,
        component: BaseComponent,
        dependencies: Optional[List[str]] = None,
        startup_order: StartupOrder = StartupOrder.ANALYSIS,
        health_check_interval: float = 30.0
    ) -> None:
        """
        Register a component for lifecycle management.

        Args:
            component: Component to register
            dependencies: List of component names this depends on
            startup_order: Startup order priority
            health_check_interval: Health check interval in seconds
        """
        dependency = ComponentDependency(component, dependencies or [])
        dependency.startup_order = startup_order
        dependency.health_check_interval = health_check_interval

        self.components[component.name] = dependency

        # Add to startup order list
        order_index = startup_order.value - 1
        if component.name not in self.startup_order[order_index]:
            self.startup_order[order_index].append(component.name)

        self.logger.info(
            f"Registered component {component.name} with order {startup_order.name} "
            f"and dependencies: {dependencies}"
        )

    def unregister_component(self, component_name: str) -> None:
        """
        Unregister a component from lifecycle management.

        Args:
            component_name: Name of component to unregister
        """
        if component_name in self.components:
            dependency = self.components[component_name]

            # Remove from startup order
            order_index = dependency.startup_order.value - 1
            if component_name in self.startup_order[order_index]:
                self.startup_order[order_index].remove(component_name)

            # Remove from components
            del self.components[component_name]
            self.running_components.discard(component_name)

            self.logger.info(f"Unregistered component {component_name}")

    async def start_all_components(self) -> bool:
        """
        Start all registered components in dependency order.

        Returns:
            True if all components started successfully, False otherwise
        """
        self.logger.info("Starting all components...")

        # Resolve startup order with dependencies
        startup_sequence = self._resolve_startup_order()
        if not startup_sequence:
            self.logger.error("Failed to resolve component startup order")
            return False

        total_components = len(startup_sequence)
        started_count = 0

        # Start components in resolved order
        for component_name in startup_sequence:
            success = await self._start_component(component_name)
            if success:
                started_count += 1
                self.stats['components_started'] += 1
            else:
                self.stats['startup_failures'] += 1
                self.logger.error(f"Failed to start component {component_name}")

                # Decide whether to continue or abort
                if self._is_critical_component(component_name):
                    self.logger.error("Critical component failed to start, aborting startup")
                    await self._stop_started_components()
                    return False

        success_rate = started_count / total_components if total_components > 0 else 0
        self.logger.info(
            f"Component startup completed: {started_count}/{total_components} "
            f"components started successfully ({success_rate:.1%})"
        )

        return started_count == total_components

    async def stop_all_components(self, force: bool = False) -> bool:
        """
        Stop all running components in reverse dependency order.

        Args:
            force: Force shutdown even if components don't stop gracefully

        Returns:
            True if all components stopped successfully, False otherwise
        """
        self.logger.info("Stopping all components...")
        self.force_shutdown = force

        # Get shutdown order (reverse of startup)
        shutdown_sequence = self._resolve_shutdown_order()
        total_components = len(shutdown_sequence)
        stopped_count = 0

        # Stop components in resolved order
        for component_name in shutdown_sequence:
            if component_name in self.running_components:
                success = await self._stop_component(component_name)
                if success:
                    stopped_count += 1
                    self.stats['components_stopped'] += 1

        success_rate = stopped_count / total_components if total_components > 0 else 1
        self.logger.info(
            f"Component shutdown completed: {stopped_count}/{total_components} "
            f"components stopped successfully ({success_rate:.1%})"
        )

        return stopped_count == total_components

    async def restart_component(self, component_name: str) -> bool:
        """
        Restart a specific component.

        Args:
            component_name: Name of component to restart

        Returns:
            True if restart was successful, False otherwise
        """
        if component_name not in self.components:
            self.logger.error(f"Component {component_name} not registered")
            return False

        dependency = self.components[component_name]
        dependency.restart_attempts += 1
        self.stats['restart_attempts'] += 1

        self.logger.info(f"Restarting component {component_name} (attempt {dependency.restart_attempts})")

        # Stop component
        await self._stop_component(component_name)

        # Wait a moment before restart
        await asyncio.sleep(1.0)

        # Start component
        success = await self._start_component(component_name)

        if success:
            dependency.restart_attempts = 0  # Reset on successful restart
            self.logger.info(f"Component {component_name} restarted successfully")
        else:
            self.logger.error(f"Failed to restart component {component_name}")

        return success

    async def _start_component(self, component_name: str) -> bool:
        """Start a specific component."""
        if component_name not in self.components:
            return False

        dependency = self.components[component_name]
        component = dependency.component

        # Check if dependencies are running
        for dep_name in dependency.dependencies:
            if dep_name not in self.running_components:
                self.logger.error(f"Dependency {dep_name} not running for {component_name}")
                return False

        try:
            self.logger.info(f"Starting component: {component_name}")
            await component.start()

            # Wait for component to be fully started
            started = await component.wait_for_start(timeout=30.0)
            if started and component.is_running():
                self.running_components.add(component_name)
                self.logger.info(f"Component {component_name} started successfully")

                # Register with message hub if not already registered
                if hasattr(component, 'handle_event'):
                    await self.message_hub.subscribe_component(component)

                return True
            else:
                self.logger.error(f"Component {component_name} failed to start properly")
                return False

        except Exception as e:
            self.logger.error(f"Error starting component {component_name}: {e}")
            return False

    async def _stop_component(self, component_name: str) -> bool:
        """Stop a specific component."""
        if component_name not in self.components:
            return False

        dependency = self.components[component_name]
        component = dependency.component

        try:
            self.logger.info(f"Stopping component: {component_name}")
            await component.stop()

            # Wait for component to stop
            stopped = await component.wait_for_stop(timeout=self.shutdown_timeout)
            if stopped or self.force_shutdown:
                self.running_components.discard(component_name)
                self.logger.info(f"Component {component_name} stopped successfully")

                # Unregister from message hub
                await self.message_hub.unregister_component(component)

                return True
            else:
                self.logger.error(f"Component {component_name} failed to stop gracefully")
                return False

        except Exception as e:
            self.logger.error(f"Error stopping component {component_name}: {e}")
            return False

    def _resolve_startup_order(self) -> List[str]:
        """Resolve component startup order based on dependencies."""
        # Create dependency graph
        graph = {}
        in_degree = defaultdict(int)

        for name, dependency in self.components.items():
            graph[name] = dependency.dependencies.copy()
            for dep in dependency.dependencies:
                in_degree[dep] += 1
            if name not in in_degree:
                in_degree[name] = 0

        # Topological sort with priority-based ordering
        result = []
        queue = deque()

        # Group by startup order priority and add to queue
        for order in StartupOrder:
            order_index = order.value - 1
            for component_name in self.startup_order[order_index]:
                if component_name in in_degree and in_degree[component_name] == 0:
                    queue.append(component_name)

        while queue:
            current = queue.popleft()
            result.append(current)

            # Reduce in-degree for dependents
            for name, dependencies in graph.items():
                if current in dependencies:
                    in_degree[name] -= 1
                    if in_degree[name] == 0:
                        queue.append(name)

        # Check for circular dependencies
        if len(result) != len(self.components):
            self.logger.error("Circular dependency detected in component graph")
            return []

        return result

    def _resolve_shutdown_order(self) -> List[str]:
        """Resolve component shutdown order (reverse of startup)."""
        startup_order = self._resolve_startup_order()
        return list(reversed(startup_order))

    def _is_critical_component(self, component_name: str) -> bool:
        """Check if a component is critical for system operation."""
        if component_name not in self.components:
            return False

        dependency = self.components[component_name]
        # Infrastructure components are considered critical
        return dependency.startup_order == StartupOrder.INFRASTRUCTURE

    async def _stop_started_components(self) -> None:
        """Stop all components that have already been started."""
        for component_name in list(self.running_components):
            await self._stop_component(component_name)

    async def _health_monitor_loop(self) -> None:
        """Background health monitoring loop."""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._check_component_health()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health monitor loop: {e}")

    async def _check_component_health(self) -> None:
        """Check health of all running components."""
        unhealthy_components = []

        for component_name in list(self.running_components):
            if component_name not in self.components:
                continue

            dependency = self.components[component_name]
            component = dependency.component

            if not component.is_healthy():
                unhealthy_components.append(component_name)
                self.stats['health_check_failures'] += 1

                # Attempt restart if under limit
                if dependency.restart_attempts < dependency.max_restart_attempts:
                    self.logger.warning(f"Component {component_name} unhealthy, attempting restart")
                    await self.restart_component(component_name)
                else:
                    self.logger.error(
                        f"Component {component_name} repeatedly unhealthy, "
                        f"max restart attempts ({dependency.max_restart_attempts}) exceeded"
                    )

                    # Publish risk event
                    risk_event = RiskEvent(
                        source=self.name,
                        risk_type=RiskEventType.RISK_CHECK_FAILED,
                        severity=RiskSeverity.CRITICAL,
                        current_value=component.state.value,
                        limit_value=ComponentState.RUNNING.value,
                        description=f"Component {component_name} failed health checks repeatedly",
                        action_required=True,
                        suggested_action=f"Manual intervention required for component {component_name}"
                    )

                    await self.message_hub.publish_event(risk_event)

    def get_component_status(self) -> Dict[str, Dict[str, any]]:
        """Get status of all registered components."""
        status = {}

        for name, dependency in self.components.items():
            component = dependency.component
            status[name] = {
                'state': component.state.value,
                'is_running': component.is_running(),
                'is_healthy': component.is_healthy(),
                'startup_order': dependency.startup_order.name,
                'dependencies': dependency.dependencies,
                'restart_attempts': dependency.restart_attempts,
                'error': str(component.get_error()) if component.get_error() else None
            }

        return status

    def get_system_stats(self) -> Dict[str, any]:
        """Get system-wide statistics."""
        return {
            'total_components': len(self.components),
            'running_components': len(self.running_components),
            'failed_components': len([
                name for name, dep in self.components.items()
                if not dep.component.is_healthy()
            ]),
            'stats': self.stats.copy()
        }