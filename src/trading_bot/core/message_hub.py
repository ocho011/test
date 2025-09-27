"""
Central message hub for coordinating communication between components.

This module provides the MessageHub class that orchestrates all component
communication, message routing, and system-wide monitoring.
"""

import asyncio
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from weakref import WeakKeyDictionary

from .base_component import BaseComponent, ComponentState
from .event_bus import EventBus
from .events import BaseEvent, EventType, RiskEvent, RiskEventType, RiskSeverity


class ComponentRegistry:
    """Registry for tracking registered components and their capabilities."""

    def __init__(self):
        self.components: WeakKeyDictionary[BaseComponent, Dict[str, Any]] = (
            WeakKeyDictionary()
        )
        self.components_by_name: Dict[str, BaseComponent] = {}
        self.capabilities: Dict[str, Set[BaseComponent]] = defaultdict(set)

    def register(
        self, component: BaseComponent, capabilities: Optional[List[str]] = None
    ) -> None:
        """Register a component with optional capabilities."""
        self.components[component] = {
            "name": component.name,
            "registered_at": datetime.utcnow(),
            "capabilities": capabilities or [],
        }
        self.components_by_name[component.name] = component

        # Register capabilities
        if capabilities:
            for capability in capabilities:
                self.capabilities[capability].add(component)

    def unregister(self, component: BaseComponent) -> None:
        """Unregister a component."""
        if component in self.components:
            # Remove from capabilities
            component_info = self.components[component]
            for capability in component_info.get("capabilities", []):
                self.capabilities[capability].discard(component)

            # Remove from registries
            del self.components[component]
            if component.name in self.components_by_name:
                del self.components_by_name[component.name]

    def get_by_name(self, name: str) -> Optional[BaseComponent]:
        """Get component by name."""
        return self.components_by_name.get(name)

    def get_by_capability(self, capability: str) -> Set[BaseComponent]:
        """Get all components with a specific capability."""
        return self.capabilities.get(capability, set()).copy()

    def get_all(self) -> List[BaseComponent]:
        """Get all registered components."""
        return list(self.components.keys())


class MessageRouter:
    """Routes messages between components based on rules and patterns."""

    def __init__(self):
        self.routing_rules: List[Dict[str, Any]] = []
        self.message_transformers: Dict[str, Any] = {}

    def add_routing_rule(
        self,
        source_pattern: str,
        target_pattern: str,
        event_types: Optional[List[EventType]] = None,
        condition: Optional[callable] = None,
    ) -> None:
        """Add a message routing rule."""
        rule = {
            "source_pattern": source_pattern,
            "target_pattern": target_pattern,
            "event_types": event_types or [],
            "condition": condition,
        }
        self.routing_rules.append(rule)

    def should_route(self, event: BaseEvent, source: str, target: str) -> bool:
        """Check if an event should be routed from source to target."""
        for rule in self.routing_rules:
            if self._matches_pattern(
                source, rule["source_pattern"]
            ) and self._matches_pattern(target, rule["target_pattern"]):

                # Check event type filter
                if rule["event_types"] and event.event_type not in rule["event_types"]:
                    continue

                # Check custom condition
                if rule["condition"] and not rule["condition"](event, source, target):
                    continue

                return True

        return False

    def _matches_pattern(self, name: str, pattern: str) -> bool:
        """Check if a name matches a pattern (supports wildcards)."""
        if pattern == "*":
            return True
        return name == pattern


class BackpressureManager:
    """Manages backpressure and flow control for message processing."""

    def __init__(self, warning_threshold: int = 80, critical_threshold: int = 95):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.component_metrics: Dict[str, Dict[str, Any]] = defaultdict(dict)

    def update_metrics(
        self, component_name: str, queue_size: int, max_queue_size: int
    ) -> None:
        """Update component metrics for backpressure monitoring."""
        usage_percent = (queue_size / max_queue_size) * 100 if max_queue_size > 0 else 0

        self.component_metrics[component_name].update(
            {
                "queue_size": queue_size,
                "max_queue_size": max_queue_size,
                "usage_percent": usage_percent,
                "last_updated": datetime.utcnow(),
            }
        )

    def get_backpressure_level(self, component_name: str) -> str:
        """Get backpressure level for a component."""
        metrics = self.component_metrics.get(component_name, {})
        usage_percent = metrics.get("usage_percent", 0)

        if usage_percent >= self.critical_threshold:
            return "critical"
        elif usage_percent >= self.warning_threshold:
            return "warning"
        else:
            return "normal"

    def get_overloaded_components(self) -> List[str]:
        """Get list of components experiencing critical backpressure."""
        overloaded = []
        for component_name, metrics in self.component_metrics.items():
            if metrics.get("usage_percent", 0) >= self.critical_threshold:
                overloaded.append(component_name)
        return overloaded


class MessageHub(BaseComponent):
    """
    Central hub for coordinating all component communication.

    Manages the event bus, component registry, message routing,
    and system-wide communication monitoring.
    """

    def __init__(self, max_queue_size: int = 1000):
        super().__init__("MessageHub")

        # Core components
        self.event_bus = EventBus(max_queue_size=max_queue_size)
        self.component_registry = ComponentRegistry()
        self.message_router = MessageRouter()
        self.backpressure_manager = BackpressureManager()

        # Monitoring
        self.health_check_interval = 30.0  # seconds
        self.health_check_task: Optional[asyncio.Task] = None

        # Statistics
        self.stats = {
            "messages_routed": 0,
            "routing_errors": 0,
            "backpressure_events": 0,
            "component_failures": 0,
        }

    async def _start(self) -> None:
        """Start the message hub and its components."""
        await self.event_bus.start()

        # Start health monitoring
        self.health_check_task = asyncio.create_task(self._health_monitor())

        self.logger.info("MessageHub started successfully")

    async def _stop(self) -> None:
        """Stop the message hub and clean up resources."""
        # Stop health monitoring
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass

        # Stop event bus
        await self.event_bus.stop()

        self.logger.info("MessageHub stopped successfully")

    async def register_component(
        self, component: BaseComponent, capabilities: Optional[List[str]] = None
    ) -> None:
        """
        Register a component with the message hub.

        Args:
            component: Component to register
            capabilities: List of capabilities this component provides
        """
        self.component_registry.register(component, capabilities)
        self.logger.info(
            f"Registered component: {component.name} with capabilities: {capabilities}"
        )

    async def unregister_component(self, component: BaseComponent) -> None:
        """
        Unregister a component from the message hub.

        Args:
            component: Component to unregister
        """
        self.component_registry.unregister(component)
        self.logger.info(f"Unregistered component: {component.name}")

    async def publish_event(self, event: BaseEvent) -> bool:
        """
        Publish an event through the message hub.

        Args:
            event: Event to publish

        Returns:
            True if event was published successfully
        """
        success = await self.event_bus.publish(event)
        if success:
            self.stats["messages_routed"] += 1
        else:
            self.stats["routing_errors"] += 1

        return success

    async def subscribe_component(
        self,
        component: BaseComponent,
        event_types: Optional[List[EventType]] = None,
        priority_filter=None,
        source_filter: Optional[str] = None,
    ) -> None:
        """
        Subscribe a component to events.

        Args:
            component: Component to subscribe
            event_types: Event types to subscribe to
            priority_filter: Minimum priority level
            source_filter: Filter by source component
        """

        async def component_handler(event: BaseEvent):
            """Wrapper handler for component event processing."""
            try:
                if hasattr(component, "handle_event"):
                    await component.handle_event(event)
            except Exception as e:
                self.logger.error(f"Error handling event in {component.name}: {e}")
                await self._handle_component_error(component, e)

        await self.event_bus.subscribe(
            handler=component_handler,
            event_types=event_types,
            priority_filter=priority_filter,
            source_filter=source_filter,
        )

        self.logger.info(f"Subscribed component {component.name} to events")

    async def get_component_by_name(self, name: str) -> Optional[BaseComponent]:
        """Get a registered component by name."""
        return self.component_registry.get_by_name(name)

    async def get_components_by_capability(self, capability: str) -> Set[BaseComponent]:
        """Get all components with a specific capability."""
        return self.component_registry.get_by_capability(capability)

    async def broadcast_to_capability(self, capability: str, event: BaseEvent) -> int:
        """
        Broadcast an event to all components with a specific capability.

        Args:
            capability: Target capability
            event: Event to broadcast

        Returns:
            Number of components that received the event
        """
        components = await self.get_components_by_capability(capability)
        count = 0

        for component in components:
            try:
                if hasattr(component, "handle_event"):
                    await component.handle_event(event)
                    count += 1
            except Exception as e:
                self.logger.error(f"Error broadcasting to {component.name}: {e}")

        return count

    async def _health_monitor(self) -> None:
        """Background task for monitoring component health."""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._check_component_health()
                await self._check_backpressure()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health monitor: {e}")

    async def _check_component_health(self) -> None:
        """Check health of all registered components."""
        for component in self.component_registry.get_all():
            if not component.is_healthy():
                self.stats["component_failures"] += 1

                # Create risk event for unhealthy component
                risk_event = RiskEvent(
                    source=self.name,
                    risk_type=RiskEventType.RISK_CHECK_FAILED,
                    severity=RiskSeverity.WARNING,
                    current_value=component.state.value,
                    limit_value=ComponentState.RUNNING.value,
                    description=f"Component {component.name} is in unhealthy state: {component.state.value}",
                    action_required=True,
                    suggested_action=f"Investigate and restart component {component.name}",
                )

                await self.publish_event(risk_event)

    async def _check_backpressure(self) -> None:
        """Check for backpressure issues in the system."""
        # Update event bus queue metrics
        queue_sizes = self.event_bus.get_queue_sizes()
        total_queue_size = sum(queue_sizes.values())

        if total_queue_size > 0:
            self.backpressure_manager.update_metrics(
                "EventBus",
                total_queue_size,
                self.event_bus.max_queue_size * len(queue_sizes),
            )

        # Check for overloaded components
        overloaded = self.backpressure_manager.get_overloaded_components()
        if overloaded:
            self.stats["backpressure_events"] += 1

            risk_event = RiskEvent(
                source=self.name,
                risk_type=RiskEventType.EXPOSURE_LIMIT,
                severity=RiskSeverity.CRITICAL,
                current_value=len(overloaded),
                limit_value=0,
                description=f"Components experiencing critical backpressure: {', '.join(overloaded)}",
                action_required=True,
                suggested_action="Reduce message load or scale affected components",
            )

            await self.publish_event(risk_event)

    async def _handle_component_error(
        self, component: BaseComponent, error: Exception
    ) -> None:
        """Handle errors from component event processing."""
        self.logger.error(f"Component {component.name} error: {error}")

        risk_event = RiskEvent(
            source=self.name,
            risk_type=RiskEventType.RISK_CHECK_FAILED,
            severity=RiskSeverity.WARNING,
            current_value=str(error),
            limit_value="None",
            description=f"Error in component {component.name}: {str(error)}",
            action_required=False,
            suggested_action="Check component logs and error handling",
        )

        await self.publish_event(risk_event)

    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        return {
            "message_hub": self.stats.copy(),
            "event_bus": self.event_bus.get_stats(),
            "queue_sizes": self.event_bus.get_queue_sizes(),
            "registered_components": len(self.component_registry.get_all()),
            "capabilities": {
                cap: len(components)
                for cap, components in self.component_registry.capabilities.items()
            },
        }

    async def shutdown_all_components(self, timeout: float = 30.0) -> None:
        """
        Gracefully shutdown all registered components.

        Args:
            timeout: Maximum time to wait for shutdown
        """
        components = self.component_registry.get_all()
        self.logger.info(f"Shutting down {len(components)} components")

        # Stop all components concurrently
        tasks = []
        for component in components:
            if component.is_running():
                task = asyncio.create_task(component.stop())
                tasks.append(task)

        if tasks:
            try:
                await asyncio.wait_for(asyncio.gather(*tasks), timeout=timeout)
                self.logger.info("All components shut down successfully")
            except asyncio.TimeoutError:
                self.logger.warning("Component shutdown timed out")
                # Cancel remaining tasks
                for task in tasks:
                    if not task.done():
                        task.cancel()
