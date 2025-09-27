"""
Event bus implementation for publish-subscribe messaging pattern.

This module provides the EventBus class for asynchronous event distribution
using asyncio queues and publish-subscribe pattern for loose coupling.
"""

import asyncio
import logging
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union
from weakref import WeakSet

from .events import BaseEvent, EventType, EventPriority


class EventSubscription:
    """Represents an event subscription with filtering capabilities."""

    def __init__(
        self,
        handler: Callable[[BaseEvent], Any],
        event_types: Optional[Set[EventType]] = None,
        priority_filter: Optional[EventPriority] = None,
        source_filter: Optional[str] = None
    ):
        """
        Initialize event subscription.

        Args:
            handler: Async function to handle events
            event_types: Set of event types to subscribe to (None for all)
            priority_filter: Minimum priority level for events
            source_filter: Filter events from specific source component
        """
        self.handler = handler
        self.event_types = event_types
        self.priority_filter = priority_filter
        self.source_filter = source_filter
        self.active = True

    def matches(self, event: BaseEvent) -> bool:
        """
        Check if an event matches this subscription's filters.

        Args:
            event: Event to check against filters

        Returns:
            True if event matches all filters, False otherwise
        """
        if not self.active:
            return False

        # Event type filter
        if self.event_types is not None and event.event_type not in self.event_types:
            return False

        # Priority filter
        if self.priority_filter is not None and event.priority.value < self.priority_filter.value:
            return False

        # Source filter
        if self.source_filter is not None and event.source != self.source_filter:
            return False

        return True


class EventBus:
    """
    Asynchronous event bus for publish-subscribe messaging.

    Provides event routing, filtering, and queue management for
    decoupled component communication in the trading system.
    """

    def __init__(self, max_queue_size: int = 1000):
        """
        Initialize the event bus.

        Args:
            max_queue_size: Maximum size for event queues
        """
        self.max_queue_size = max_queue_size
        self.logger = logging.getLogger(__name__)

        # Subscription management
        self._subscriptions: List[EventSubscription] = []
        self._subscription_lock = asyncio.Lock()

        # Event queues by priority
        self._priority_queues: Dict[EventPriority, asyncio.Queue] = {
            priority: asyncio.Queue(maxsize=max_queue_size)
            for priority in EventPriority
        }

        # Processing control
        self._running = False
        self._processor_task: Optional[asyncio.Task] = None
        self._weak_handlers: WeakSet = WeakSet()

        # Statistics
        self._stats = {
            'events_published': 0,
            'events_processed': 0,
            'events_dropped': 0,
            'handler_errors': 0
        }

    async def start(self) -> None:
        """Start the event bus processing."""
        if self._running:
            self.logger.warning("EventBus is already running")
            return

        self._running = True
        self._processor_task = asyncio.create_task(self._process_events())
        self.logger.info("EventBus started")

    async def stop(self) -> None:
        """Stop the event bus and clean up resources."""
        if not self._running:
            return

        self._running = False

        # Cancel processor task
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass

        # Clear queues
        for queue in self._priority_queues.values():
            while not queue.empty():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

        self.logger.info("EventBus stopped")

    async def publish(self, event: BaseEvent) -> bool:
        """
        Publish an event to the bus.

        Args:
            event: Event to publish

        Returns:
            True if event was queued successfully, False if queue is full
        """
        if not self._running:
            self.logger.warning("Attempting to publish event while EventBus is not running")
            return False

        try:
            # Get appropriate priority queue
            queue = self._priority_queues[event.priority]

            # Try to put event in queue (non-blocking)
            queue.put_nowait(event)

            self._stats['events_published'] += 1
            self.logger.debug(f"Published event: {event.event_type.value} from {event.source}")
            return True

        except asyncio.QueueFull:
            self._stats['events_dropped'] += 1
            self.logger.error(f"Event queue full, dropping event: {event.event_type.value}")
            return False

    async def subscribe(
        self,
        handler: Callable[[BaseEvent], Any],
        event_types: Optional[Union[EventType, List[EventType]]] = None,
        priority_filter: Optional[EventPriority] = None,
        source_filter: Optional[str] = None
    ) -> EventSubscription:
        """
        Subscribe to events with optional filtering.

        Args:
            handler: Async function to handle events
            event_types: Event type(s) to subscribe to
            priority_filter: Minimum priority level
            source_filter: Filter by source component

        Returns:
            EventSubscription object for managing the subscription
        """
        # Normalize event_types to set
        if event_types is None:
            event_type_set = None
        elif isinstance(event_types, EventType):
            event_type_set = {event_types}
        else:
            event_type_set = set(event_types)

        subscription = EventSubscription(
            handler=handler,
            event_types=event_type_set,
            priority_filter=priority_filter,
            source_filter=source_filter
        )

        async with self._subscription_lock:
            self._subscriptions.append(subscription)
            self._weak_handlers.add(handler)

        self.logger.info(f"Added subscription for {event_type_set or 'all'} events")
        return subscription

    async def unsubscribe(self, subscription: EventSubscription) -> None:
        """
        Remove a subscription from the event bus.

        Args:
            subscription: Subscription to remove
        """
        async with self._subscription_lock:
            subscription.active = False
            if subscription in self._subscriptions:
                self._subscriptions.remove(subscription)

        self.logger.info("Removed event subscription")

    async def _process_events(self) -> None:
        """Main event processing loop with priority handling."""
        self.logger.info("Event processor started")

        while self._running:
            try:
                # Process events by priority (highest first)
                event = await self._get_next_event()
                if event:
                    await self._dispatch_event(event)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in event processor: {e}")
                await asyncio.sleep(0.1)  # Brief pause on error

        self.logger.info("Event processor stopped")

    async def _get_next_event(self) -> Optional[BaseEvent]:
        """
        Get the next event to process, prioritizing by importance.

        Returns:
            Next event to process or None if no events available
        """
        # Check queues in priority order (highest priority first)
        for priority in sorted(EventPriority, key=lambda p: p.value, reverse=True):
            queue = self._priority_queues[priority]
            if not queue.empty():
                try:
                    return queue.get_nowait()
                except asyncio.QueueEmpty:
                    continue

        # If no events available, wait for any event
        try:
            # Create tasks for all queues
            tasks = []
            for priority, queue in self._priority_queues.items():
                task = asyncio.create_task(queue.get())
                task.priority = priority
                tasks.append(task)

            # Wait for first event
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED, timeout=1.0
            )

            # Cancel pending tasks
            for task in pending:
                task.cancel()

            # Return result from completed task
            if done:
                completed_task = done.pop()
                return await completed_task

        except asyncio.TimeoutError:
            pass

        return None

    async def _dispatch_event(self, event: BaseEvent) -> None:
        """
        Dispatch an event to all matching subscribers.

        Args:
            event: Event to dispatch
        """
        dispatched_count = 0

        async with self._subscription_lock:
            # Create a copy to avoid modification during iteration
            subscriptions = self._subscriptions.copy()

        for subscription in subscriptions:
            if subscription.matches(event):
                try:
                    # Handle both sync and async handlers
                    result = subscription.handler(event)
                    if asyncio.iscoroutine(result):
                        await result

                    dispatched_count += 1

                except Exception as e:
                    self._stats['handler_errors'] += 1
                    self.logger.error(f"Error in event handler: {e}")

        self._stats['events_processed'] += 1
        self.logger.debug(f"Dispatched event to {dispatched_count} handlers")

    def get_stats(self) -> Dict[str, int]:
        """
        Get event bus statistics.

        Returns:
            Dictionary containing processing statistics
        """
        return self._stats.copy()

    def get_queue_sizes(self) -> Dict[str, int]:
        """
        Get current queue sizes by priority.

        Returns:
            Dictionary mapping priority names to queue sizes
        """
        return {
            priority.name: queue.qsize()
            for priority, queue in self._priority_queues.items()
        }

    def is_running(self) -> bool:
        """Check if the event bus is running."""
        return self._running

    async def wait_for_empty_queues(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for all event queues to be empty.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if all queues are empty, False if timeout
        """
        start_time = asyncio.get_event_loop().time()

        while True:
            if all(queue.empty() for queue in self._priority_queues.values()):
                return True

            if timeout and (asyncio.get_event_loop().time() - start_time) > timeout:
                return False

            await asyncio.sleep(0.1)

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()