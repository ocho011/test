"""
Signal Event Publisher for ICT Trading Signals

This module implements comprehensive signal event publishing to integrate
trading signals with the broader event-driven architecture of the trading bot.
"""

import asyncio
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

from ..core.events import EventType, SignalEvent
from .bias_filter import FilterResult
from .confluence_validator import ConfluenceResult
from .signal_strength_calculator import SignalStrength


class PublishingMode(Enum):
    """Signal publishing modes"""

    IMMEDIATE = "immediate"
    BATCH = "batch"
    SCHEDULED = "scheduled"
    CONDITIONAL = "conditional"


class EventPriority(Enum):
    """Event priority levels"""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class PublishingStatus(Enum):
    """Publishing status"""

    PENDING = "pending"
    PUBLISHED = "published"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DELAYED = "delayed"


@dataclass
class SignalEventData:
    """Enhanced signal event data for publishing"""

    signal_id: str
    signal_type: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence_score: float
    strength_score: float
    confluence_level: str
    filter_passed: bool
    patterns: List[Dict[str, Any]]
    timeframes: List[str]
    risk_reward_ratio: float
    session_info: Dict[str, Any]
    metadata: Dict[str, Any]
    created_at: datetime
    expires_at: Optional[datetime] = None


@dataclass
class PublishingConfig:
    """Configuration for signal event publishing"""

    mode: PublishingMode = PublishingMode.IMMEDIATE
    batch_size: int = 10
    batch_timeout_seconds: int = 30
    retry_attempts: int = 3
    retry_delay_seconds: int = 5
    require_confluence_validation: bool = True
    require_strength_calculation: bool = True
    require_bias_filtering: bool = True
    minimum_confidence_score: float = 0.65
    minimum_strength_score: float = 0.60
    priority_based_on_strength: bool = True
    enable_persistence: bool = True
    max_concurrent_publishes: int = 5


@dataclass
class PublishingResult:
    """Result of signal publishing"""

    success: bool
    signal_id: str
    event_id: Optional[str]
    status: PublishingStatus
    published_at: Optional[datetime]
    error_message: Optional[str]
    retry_count: int
    metadata: Dict[str, Any]


class SignalEventPublisher:
    """
    Comprehensive signal event publishing system for ICT trading signals.

    This publisher integrates with the event-driven architecture to ensure
    signals are properly distributed to downstream systems with appropriate
    validation, filtering, and persistence.
    """

    def __init__(
        self,
        config: Optional[PublishingConfig] = None,
        event_handlers: Optional[List[Callable]] = None,
    ):
        self.config = config or PublishingConfig()
        self.event_handlers = event_handlers or []
        self.logger = logging.getLogger(__name__)

        # Internal state
        self.pending_events = []
        self.published_events = {}
        self.failed_events = {}
        self.publishing_semaphore = asyncio.Semaphore(
            self.config.max_concurrent_publishes
        )

        # Batch processing
        self.batch_queue = []
        self.batch_timer = None

    def publish_signal(
        self,
        signal_data: Dict[str, Any],
        confluence_result: Optional[ConfluenceResult] = None,
        strength_result: Optional[SignalStrength] = None,
        filter_result: Optional[FilterResult] = None,
        priority: EventPriority = EventPriority.NORMAL,
    ) -> PublishingResult:
        """
        Publish a trading signal as an event.

        Args:
            signal_data: Core signal information
            confluence_result: Optional confluence validation result
            strength_result: Optional signal strength calculation
            filter_result: Optional bias filtering result
            priority: Event priority level

        Returns:
            PublishingResult indicating success/failure
        """
        try:
            # Validate signal before publishing
            validation_result = self._validate_signal_for_publishing(
                signal_data, confluence_result, strength_result, filter_result
            )

            if not validation_result["is_valid"]:
                return PublishingResult(
                    success=False,
                    signal_id=signal_data.get("id", "unknown"),
                    event_id=None,
                    status=PublishingStatus.FAILED,
                    published_at=None,
                    error_message=validation_result["error_message"],
                    retry_count=0,
                    metadata=validation_result,
                )

            # Create enhanced signal event data
            event_data = self._create_signal_event_data(
                signal_data, confluence_result, strength_result, filter_result
            )

            # Determine publishing mode and execute
            if self.config.mode == PublishingMode.IMMEDIATE:
                return self._publish_immediately(event_data, priority)
            elif self.config.mode == PublishingMode.BATCH:
                return self._add_to_batch(event_data, priority)
            elif self.config.mode == PublishingMode.SCHEDULED:
                return self._schedule_publish(event_data, priority)
            elif self.config.mode == PublishingMode.CONDITIONAL:
                return self._publish_conditionally(event_data, priority)
            else:
                return self._publish_immediately(event_data, priority)

        except Exception as e:
            self.logger.error(f"Error publishing signal: {e}")
            return PublishingResult(
                success=False,
                signal_id=signal_data.get("id", "unknown"),
                event_id=None,
                status=PublishingStatus.FAILED,
                published_at=None,
                error_message=str(e),
                retry_count=0,
                metadata={"error": str(e)},
            )

    async def publish_signal_async(
        self,
        signal_data: Dict[str, Any],
        confluence_result: Optional[ConfluenceResult] = None,
        strength_result: Optional[SignalStrength] = None,
        filter_result: Optional[FilterResult] = None,
        priority: EventPriority = EventPriority.NORMAL,
    ) -> PublishingResult:
        """Async version of signal publishing"""
        async with self.publishing_semaphore:
            return await asyncio.get_event_loop().run_in_executor(
                None,
                self.publish_signal,
                signal_data,
                confluence_result,
                strength_result,
                filter_result,
                priority,
            )

    def _validate_signal_for_publishing(
        self,
        signal_data: Dict[str, Any],
        confluence_result: Optional[ConfluenceResult],
        strength_result: Optional[SignalStrength],
        filter_result: Optional[FilterResult],
    ) -> Dict[str, Any]:
        """Validate signal meets publishing requirements"""
        validation_errors = []

        # Check required signal data
        required_fields = [
            "id",
            "direction",
            "entry_price",
            "stop_loss",
            "take_profit",
        ]
        for field in required_fields:
            if field not in signal_data or signal_data[field] is None:
                validation_errors.append(f"Missing required field: {field}")

        # Check confluence validation requirement
        if self.config.require_confluence_validation and confluence_result is None:
            validation_errors.append("Confluence validation required but not provided")

        if confluence_result and not confluence_result.is_valid:
            validation_errors.append("Signal failed confluence validation")

        # Check strength calculation requirement
        if self.config.require_strength_calculation and strength_result is None:
            validation_errors.append("Strength calculation required but not provided")

        if strength_result:
            if strength_result.weighted_score < self.config.minimum_strength_score:
                validation_errors.append(
                    f"Signal strength {strength_result.weighted_score:.2f} below minimum {self.config.minimum_strength_score}"
                )

        # Check bias filtering requirement
        if self.config.require_bias_filtering and filter_result is None:
            validation_errors.append("Bias filtering required but not provided")

        if filter_result and not filter_result.is_allowed:
            validation_errors.append("Signal failed bias filtering")

        # Check confidence score
        confidence_score = signal_data.get("confidence_score", 0.0)
        if confidence_score < self.config.minimum_confidence_score:
            validation_errors.append(
                f"Confidence score {confidence_score:.2f} below minimum {self.config.minimum_confidence_score}"
            )

        is_valid = len(validation_errors) == 0

        return {
            "is_valid": is_valid,
            "error_message": (
                "; ".join(validation_errors) if validation_errors else None
            ),
            "validation_errors": validation_errors,
        }

    def _create_signal_event_data(
        self,
        signal_data: Dict[str, Any],
        confluence_result: Optional[ConfluenceResult],
        strength_result: Optional[SignalStrength],
        filter_result: Optional[FilterResult],
    ) -> SignalEventData:
        """Create comprehensive signal event data"""
        # Extract core signal information
        signal_id = signal_data.get("id", str(uuid4()))
        signal_type = signal_data.get("type", "ict_signal")
        direction = signal_data["direction"]
        entry_price = signal_data["entry_price"]
        stop_loss = signal_data["stop_loss"]
        take_profit = signal_data["take_profit"]

        # Calculate risk-reward ratio
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        risk_reward_ratio = reward / risk if risk > 0 else 0.0

        # Extract scores
        confidence_score = signal_data.get("confidence_score", 0.0)
        strength_score = strength_result.weighted_score if strength_result else 0.0
        confluence_level = (
            confluence_result.confluence_level.value if confluence_result else "unknown"
        )
        filter_passed = filter_result.is_allowed if filter_result else False

        # Extract patterns and timeframes
        patterns = signal_data.get("patterns", [])
        timeframes = signal_data.get("timeframes", [])

        # Extract session information
        session_info = {}
        if filter_result:
            session_info = {
                "current_session": filter_result.current_session.value,
                "current_bias": filter_result.current_bias.value,
                "filter_score": filter_result.filter_score,
            }

        # Create metadata
        metadata = {
            "created_by": "signal_generator",
            "validation_timestamp": datetime.now().isoformat(),
            "confluence_details": (
                asdict(confluence_result) if confluence_result else None
            ),
            "strength_details": (asdict(strength_result) if strength_result else None),
            "filter_details": asdict(filter_result) if filter_result else None,
            "original_signal_data": signal_data,
        }

        # Calculate expiration time
        expires_at = signal_data.get("expires_at")
        if expires_at is None and "validity_minutes" in signal_data:
            expires_at = datetime.now() + timedelta(
                minutes=signal_data["validity_minutes"]
            )

        return SignalEventData(
            signal_id=signal_id,
            signal_type=signal_type,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence_score=confidence_score,
            strength_score=strength_score,
            confluence_level=confluence_level,
            filter_passed=filter_passed,
            patterns=patterns,
            timeframes=timeframes,
            risk_reward_ratio=risk_reward_ratio,
            session_info=session_info,
            metadata=metadata,
            created_at=datetime.now(),
            expires_at=expires_at,
        )

    def _publish_immediately(
        self, event_data: SignalEventData, priority: EventPriority
    ) -> PublishingResult:
        """Publish signal event immediately"""
        try:
            # Create SignalEvent
            signal_event = SignalEvent(
                id=str(uuid4()),
                timestamp=datetime.now(),
                event_type=EventType.SIGNAL_GENERATED,
                signal_type=event_data.signal_type,
                direction=event_data.direction,
                entry_price=event_data.entry_price,
                stop_loss=event_data.stop_loss,
                take_profit=event_data.take_profit,
                confidence_score=event_data.confidence_score,
                patterns=event_data.patterns,
                timeframes=event_data.timeframes,
                metadata={
                    **event_data.metadata,
                    "priority": priority.value,
                    "strength_score": event_data.strength_score,
                    "confluence_level": event_data.confluence_level,
                    "filter_passed": event_data.filter_passed,
                    "session_info": event_data.session_info,
                },
            )

            # Publish to event handlers
            for handler in self.event_handlers:
                try:
                    handler(signal_event)
                except Exception as e:
                    self.logger.error(f"Error in event handler: {e}")

            # Store in published events
            self.published_events[event_data.signal_id] = {
                "event": signal_event,
                "published_at": datetime.now(),
                "priority": priority,
            }

            return PublishingResult(
                success=True,
                signal_id=event_data.signal_id,
                event_id=signal_event.id,
                status=PublishingStatus.PUBLISHED,
                published_at=datetime.now(),
                error_message=None,
                retry_count=0,
                metadata={
                    "priority": priority.value,
                    "handlers_count": len(self.event_handlers),
                },
            )

        except Exception as e:
            self.logger.error(f"Error in immediate publishing: {e}")
            return PublishingResult(
                success=False,
                signal_id=event_data.signal_id,
                event_id=None,
                status=PublishingStatus.FAILED,
                published_at=None,
                error_message=str(e),
                retry_count=0,
                metadata={"error": str(e)},
            )

    def _add_to_batch(
        self, event_data: SignalEventData, priority: EventPriority
    ) -> PublishingResult:
        """Add signal to batch queue"""
        try:
            self.batch_queue.append(
                {
                    "event_data": event_data,
                    "priority": priority,
                    "added_at": datetime.now(),
                }
            )

            # Check if batch is full
            if len(self.batch_queue) >= self.config.batch_size:
                self._process_batch()

            # Set batch timer if not already set
            elif self.batch_timer is None:
                self.batch_timer = asyncio.get_event_loop().call_later(
                    self.config.batch_timeout_seconds, self._process_batch
                )

            return PublishingResult(
                success=True,
                signal_id=event_data.signal_id,
                event_id=None,
                status=PublishingStatus.PENDING,
                published_at=None,
                error_message=None,
                retry_count=0,
                metadata={
                    "batch_position": len(self.batch_queue),
                    "batch_size": self.config.batch_size,
                },
            )

        except Exception as e:
            self.logger.error(f"Error adding to batch: {e}")
            return PublishingResult(
                success=False,
                signal_id=event_data.signal_id,
                event_id=None,
                status=PublishingStatus.FAILED,
                published_at=None,
                error_message=str(e),
                retry_count=0,
                metadata={"error": str(e)},
            )

    def _process_batch(self):
        """Process all signals in batch queue"""
        if not self.batch_queue:
            return

        try:
            # Sort by priority
            priority_order = {
                EventPriority.CRITICAL: 0,
                EventPriority.HIGH: 1,
                EventPriority.NORMAL: 2,
                EventPriority.LOW: 3,
            }

            sorted_batch = sorted(
                self.batch_queue,
                key=lambda x: priority_order.get(x["priority"], 2),
            )

            # Publish each signal
            for item in sorted_batch:
                event_data = item["event_data"]
                priority = item["priority"]

                result = self._publish_immediately(event_data, priority)
                if not result.success:
                    self.logger.warning(
                        f"Failed to publish signal {event_data.signal_id} in batch: {result.error_message}"
                    )

            # Clear batch queue
            self.batch_queue.clear()

            # Reset timer
            if self.batch_timer:
                self.batch_timer.cancel()
                self.batch_timer = None

        except Exception as e:
            self.logger.error(f"Error processing batch: {e}")

    def _schedule_publish(
        self, event_data: SignalEventData, priority: EventPriority
    ) -> PublishingResult:
        """Schedule signal for future publishing"""
        # This is a placeholder for scheduled publishing
        # In a real implementation, this would integrate with a task scheduler
        return self._publish_immediately(event_data, priority)

    def _publish_conditionally(
        self, event_data: SignalEventData, priority: EventPriority
    ) -> PublishingResult:
        """Publish signal based on conditional logic"""
        # Check conditions (example: only high-confidence signals during off-hours)
        current_hour = datetime.now().hour

        # During main trading hours, publish normally
        if 8 <= current_hour <= 22:
            return self._publish_immediately(event_data, priority)

        # During off-hours, only publish high-confidence signals
        elif event_data.confidence_score >= 0.8:
            return self._publish_immediately(event_data, priority)

        else:
            return PublishingResult(
                success=False,
                signal_id=event_data.signal_id,
                event_id=None,
                status=PublishingStatus.DELAYED,
                published_at=None,
                error_message="Signal delayed due to conditional filtering",
                retry_count=0,
                metadata={
                    "condition": "off_hours_low_confidence",
                    "current_hour": current_hour,
                    "confidence_score": event_data.confidence_score,
                },
            )

    def add_event_handler(self, handler: Callable):
        """Add event handler for signal events"""
        self.event_handlers.append(handler)

    def remove_event_handler(self, handler: Callable):
        """Remove event handler"""
        if handler in self.event_handlers:
            self.event_handlers.remove(handler)

    def get_published_events(self) -> Dict[str, Any]:
        """Get all published events"""
        return self.published_events.copy()

    def get_failed_events(self) -> Dict[str, Any]:
        """Get all failed events"""
        return self.failed_events.copy()

    def get_pending_events(self) -> List[Dict[str, Any]]:
        """Get all pending events"""
        return self.batch_queue.copy()

    def clear_event_history(self):
        """Clear event history"""
        self.published_events.clear()
        self.failed_events.clear()

    def get_publishing_stats(self) -> Dict[str, Any]:
        """Get publishing statistics"""
        return {
            "published_count": len(self.published_events),
            "failed_count": len(self.failed_events),
            "pending_count": len(self.batch_queue),
            "handlers_count": len(self.event_handlers),
            "config": asdict(self.config),
        }
