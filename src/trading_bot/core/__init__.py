"""
Core trading bot components and infrastructure.

This module provides the foundational classes and systems for the trading bot,
including the base component interface, event system, and dependency injection.
"""

from .base_component import BaseComponent, ComponentState
from .di_container import DIContainer, Lifetime
from .event_bus import EventBus, EventSubscription
from .events import (
    BaseEvent,
    EventType,
    EventPriority,
    MarketDataEvent,
    SignalEvent,
    SignalType,
    OrderEvent,
    OrderSide,
    OrderType,
    OrderStatus,
    RiskEvent,
    RiskEventType,
    RiskSeverity,
    create_event_from_dict,
    serialize_event,
)
from .lifecycle_manager import ComponentLifecycleManager, StartupOrder
from .message_hub import MessageHub

__all__ = [
    # Base infrastructure
    "BaseComponent",
    "ComponentState",

    # Dependency injection
    "DIContainer",
    "Lifetime",

    # Event system
    "EventBus",
    "EventSubscription",
    "BaseEvent",
    "EventType",
    "EventPriority",

    # Event types
    "MarketDataEvent",
    "SignalEvent",
    "SignalType",
    "OrderEvent",
    "OrderSide",
    "OrderType",
    "OrderStatus",
    "RiskEvent",
    "RiskEventType",
    "RiskSeverity",

    # Event utilities
    "create_event_from_dict",
    "serialize_event",

    # System management
    "ComponentLifecycleManager",
    "StartupOrder",
    "MessageHub",
]