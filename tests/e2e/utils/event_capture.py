"""
Event capture utility for E2E testing.

Records all events published during test execution for verification.
"""

import asyncio
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from trading_bot.core.event_bus import EventBus
from trading_bot.core.events import EventType


class EventCapture:
    """
    Captures and records all events from EventBus for testing.
    
    Features:
    - Record all events with timestamps
    - Filter events by type
    - Verify event ordering
    - Assert event payloads
    - Detect duplicate or missing events
    """

    def __init__(self, event_bus: EventBus):
        """
        Initialize event capture.
        
        Args:
            event_bus: EventBus instance to monitor
        """
        self.event_bus = event_bus
        self.events: List[Dict[str, Any]] = []
        self.events_by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._subscription_ids: List[str] = []
        self._is_capturing = False

    async def start(self, event_types: Optional[Set[str]] = None) -> None:
        """
        Start capturing events.
        
        Args:
            event_types: Set of event types to capture (None for all)
        """
        if self._is_capturing:
            return
        
        self._is_capturing = True
        
        # Subscribe to all events if no filter specified
        if event_types is None:
            # Subscribe to common event types using EventType enum
            event_types = {
                EventType.MARKET_DATA,
                EventType.SIGNAL,
                EventType.ORDER,
                EventType.RISK,
                EventType.POSITION,
                EventType.CANDLE_CLOSED,
                EventType.RISK_APPROVED_ORDER,
            }

        # Subscribe to each event type
        for event_type in event_types:
            subscription = await self.event_bus.subscribe(
                self._capture_event,
                event_types=event_type
            )
            self._subscription_ids.append(subscription)

    async def stop(self) -> None:
        """Stop capturing events and unsubscribe."""
        if not self._is_capturing:
            return

        # Unsubscribe from all events
        for subscription in self._subscription_ids:
            await self.event_bus.unsubscribe(subscription)

        self._subscription_ids.clear()
        self._is_capturing = False

    async def _capture_event(self, event) -> None:
        """
        Capture an event.

        Args:
            event: Event to capture (BaseEvent instance)
        """
        # Store event with both EventType enum and string key for compatibility
        event_type_str = event.event_type.value if hasattr(event.event_type, 'value') else str(event.event_type)

        event_record = {
            "type": event_type_str,
            "data": event.dict() if hasattr(event, 'dict') else event,
            "timestamp": datetime.now(),
            "event_id": str(event.event_id) if hasattr(event, 'event_id') else None,
            "source": event.source if hasattr(event, 'source') else None
        }

        self.events.append(event_record)
        self.events_by_type[event_type_str].append(event_record)
        self.events_by_type[event.event_type].append(event_record)

    def get_events(self, event_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get captured events.

        Args:
            event_type: Filter by event type string (e.g., "market_data") or None for all

        Returns:
            List of event records
        """
        if event_type:
            # Try both string lookup and EventType enum lookup
            events = self.events_by_type.get(event_type, [])
            if not events:
                # Try finding by EventType enum
                for key, value in self.events_by_type.items():
                    if hasattr(key, 'value') and key.value == event_type:
                        return value
            return events
        return self.events

    def get_event_count(self, event_type: Optional[str] = None) -> int:
        """
        Get count of captured events.
        
        Args:
            event_type: Filter by event type (None for all)
            
        Returns:
            Event count
        """
        if event_type:
            return len(self.events_by_type.get(event_type, []))
        return len(self.events)

    def assert_event_exists(
        self,
        event_type: str,
        data_matcher: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Assert that an event with matching type and data exists.
        
        Args:
            event_type: Expected event type
            data_matcher: Dict of expected data fields (partial match)
            
        Returns:
            Matching event record
            
        Raises:
            AssertionError: If no matching event found
        """
        events = self.get_events(event_type)
        
        if not events:
            raise AssertionError(f"No events of type '{event_type}' found")
        
        if data_matcher is None:
            return events[0]
        
        # Find event matching data criteria
        for event in events:
            match = True
            for key, expected_value in data_matcher.items():
                if key not in event["data"]:
                    match = False
                    break
                if event["data"][key] != expected_value:
                    match = False
                    break
            
            if match:
                return event
        
        raise AssertionError(
            f"No event of type '{event_type}' found matching data: {data_matcher}"
        )

    def assert_event_sequence(self, expected_sequence: List[str]) -> None:
        """
        Assert that events occurred in expected order.
        
        Args:
            expected_sequence: List of event types in expected order
            
        Raises:
            AssertionError: If sequence doesn't match
        """
        actual_sequence = [e["type"] for e in self.events]
        
        # Find subsequence
        seq_idx = 0
        for event_type in actual_sequence:
            if seq_idx < len(expected_sequence) and event_type == expected_sequence[seq_idx]:
                seq_idx += 1
        
        if seq_idx != len(expected_sequence):
            raise AssertionError(
                f"Expected event sequence {expected_sequence} not found. "
                f"Actual sequence: {actual_sequence}"
            )

    def assert_no_duplicate_events(self, event_type: str, unique_field: str) -> None:
        """
        Assert no duplicate events based on unique field.
        
        Args:
            event_type: Event type to check
            unique_field: Field that should be unique
            
        Raises:
            AssertionError: If duplicates found
        """
        events = self.get_events(event_type)
        seen_values = set()
        duplicates = []
        
        for event in events:
            if unique_field not in event["data"]:
                continue
            
            value = event["data"][unique_field]
            if value in seen_values:
                duplicates.append(value)
            seen_values.add(value)
        
        if duplicates:
            raise AssertionError(
                f"Duplicate events found for {event_type} on field {unique_field}: {duplicates}"
            )

    def get_timing_between_events(
        self,
        start_event_type: str,
        end_event_type: str
    ) -> Optional[float]:
        """
        Get time elapsed between two event types.
        
        Args:
            start_event_type: Starting event type
            end_event_type: Ending event type
            
        Returns:
            Time in seconds, or None if events not found
        """
        start_events = self.get_events(start_event_type)
        end_events = self.get_events(end_event_type)
        
        if not start_events or not end_events:
            return None
        
        start_time = start_events[0]["timestamp"]
        end_time = end_events[0]["timestamp"]
        
        return (end_time - start_time).total_seconds()

    def clear(self) -> None:
        """Clear all captured events."""
        self.events.clear()
        self.events_by_type.clear()

    def __repr__(self) -> str:
        """String representation."""
        return f"EventCapture(captured={len(self.events)}, types={len(self.events_by_type)})"
