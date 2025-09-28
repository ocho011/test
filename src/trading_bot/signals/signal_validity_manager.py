"""
Signal Validity and Timeout Management System

This module implements comprehensive signal lifecycle management including:
- Signal validity timeout tracking
- Automatic signal expiration
- Signal state management
- Performance-based validity adjustment
- Memory-efficient cleanup
- Signal archival system
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Set
import asyncio
import logging
from threading import Lock
import weakref

logger = logging.getLogger(__name__)


class SignalState(Enum):
    """Signal lifecycle states"""
    ACTIVE = "active"
    EXPIRED = "expired"
    EXECUTED = "executed"
    CANCELLED = "cancelled"
    ARCHIVED = "archived"


class ValidityReason(Enum):
    """Reasons for signal validity changes"""
    TIMEOUT = "timeout"
    MARKET_CONDITION_CHANGE = "market_condition_change"
    MANUAL_CANCELLATION = "manual_cancellation"
    EXECUTION = "execution"
    PERFORMANCE_BASED = "performance_based"
    SYSTEM_SHUTDOWN = "system_shutdown"


@dataclass
class SignalValidityConfig:
    """Configuration for signal validity management"""
    # Base timeout settings
    default_timeout_minutes: int = 60
    min_timeout_minutes: int = 5
    max_timeout_minutes: int = 480  # 8 hours

    # Performance-based adjustments
    enable_performance_adjustment: bool = True
    performance_lookback_signals: int = 50
    good_performance_threshold: float = 0.65
    poor_performance_threshold: float = 0.35
    performance_timeout_multiplier: float = 1.5

    # Cleanup and archival
    cleanup_interval_minutes: int = 15
    archive_after_days: int = 7
    max_active_signals: int = 1000

    # Market condition adjustments
    volatility_timeout_multiplier: float = 0.8
    low_liquidity_timeout_multiplier: float = 0.6

    # Execution tracking
    track_execution_performance: bool = True
    execution_timeout_buffer_minutes: int = 5


@dataclass
class SignalValidityInfo:
    """Information about signal validity status"""
    signal_id: str
    created_at: datetime
    expires_at: datetime
    state: SignalState
    original_timeout_minutes: int
    adjusted_timeout_minutes: int
    validity_reason: Optional[ValidityReason] = None
    performance_score: Optional[float] = None
    market_conditions: Dict[str, Any] = field(default_factory=dict)
    execution_info: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SignalValidityManager:
    """
    Comprehensive signal validity and timeout management system

    Manages the lifecycle of trading signals including:
    - Automatic timeout handling
    - Performance-based validity adjustments
    - Market condition adaptations
    - Signal state transitions
    - Cleanup and archival
    """

    def __init__(self, config: Optional[SignalValidityConfig] = None):
        self.config = config or SignalValidityConfig()
        self._active_signals: Dict[str, SignalValidityInfo] = {}
        self._expired_signals: Dict[str, SignalValidityInfo] = {}
        self._archived_signals: Dict[str, SignalValidityInfo] = {}
        self._performance_history: List[Dict[str, Any]] = []
        self._lock = Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._callbacks: Dict[SignalState, List[Callable]] = {
            state: [] for state in SignalState
        }
        self._running = False

    def start(self):
        """Start the validity manager background processes"""
        if self._running:
            return

        self._running = True
        logger.info("Starting signal validity manager")

        # Start cleanup task
        try:
            loop = asyncio.get_running_loop()
            self._cleanup_task = loop.create_task(self._cleanup_loop())
        except RuntimeError:
            # No running loop, cleanup will be called manually
            logger.warning("No asyncio loop running, cleanup must be called manually")

    def stop(self):
        """Stop the validity manager and cleanup resources"""
        if not self._running:
            return

        self._running = False
        logger.info("Stopping signal validity manager")

        # Cancel cleanup task
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()

        # Mark all active signals as cancelled due to system shutdown
        with self._lock:
            for signal_id in list(self._active_signals.keys()):
                self._expire_signal(signal_id, ValidityReason.SYSTEM_SHUTDOWN)

    def register_signal(
        self,
        signal_id: str,
        signal_data: Dict[str, Any],
        custom_timeout_minutes: Optional[int] = None
    ) -> SignalValidityInfo:
        """
        Register a new signal for validity tracking

        Args:
            signal_id: Unique identifier for the signal
            signal_data: Signal data for performance analysis
            custom_timeout_minutes: Custom timeout override

        Returns:
            SignalValidityInfo object
        """
        with self._lock:
            if signal_id in self._active_signals:
                raise ValueError(f"Signal {signal_id} already registered")

            # Check signal limit
            if len(self._active_signals) >= self.config.max_active_signals:
                self._force_cleanup_oldest()

            # Calculate timeout
            base_timeout = custom_timeout_minutes or self.config.default_timeout_minutes
            adjusted_timeout = self._calculate_adjusted_timeout(
                base_timeout, signal_data
            )

            # Create validity info
            now = datetime.now()
            validity_info = SignalValidityInfo(
                signal_id=signal_id,
                created_at=now,
                expires_at=now + timedelta(minutes=adjusted_timeout),
                state=SignalState.ACTIVE,
                original_timeout_minutes=base_timeout,
                adjusted_timeout_minutes=adjusted_timeout,
                market_conditions=self._extract_market_conditions(signal_data),
                metadata={
                    'signal_type': signal_data.get('signal_type'),
                    'confidence': signal_data.get('confidence'),
                    'strength_score': signal_data.get('strength_score')
                }
            )

            self._active_signals[signal_id] = validity_info
            logger.debug(f"Registered signal {signal_id} with {adjusted_timeout}min timeout")

            # Trigger callback
            self._trigger_callbacks(SignalState.ACTIVE, validity_info)

            return validity_info

    def extend_signal_validity(
        self,
        signal_id: str,
        additional_minutes: int,
        reason: str = "manual_extension"
    ) -> bool:
        """
        Extend the validity period of an active signal

        Args:
            signal_id: Signal to extend
            additional_minutes: Additional time to add
            reason: Reason for extension

        Returns:
            True if extended successfully
        """
        with self._lock:
            if signal_id not in self._active_signals:
                return False

            validity_info = self._active_signals[signal_id]
            if validity_info.state != SignalState.ACTIVE:
                return False

            # Apply limits
            new_total_minutes = validity_info.adjusted_timeout_minutes + additional_minutes
            new_total_minutes = min(new_total_minutes, self.config.max_timeout_minutes)

            # Update expiry
            validity_info.expires_at = validity_info.created_at + timedelta(
                minutes=new_total_minutes
            )
            validity_info.adjusted_timeout_minutes = new_total_minutes
            validity_info.metadata['extension_reason'] = reason
            validity_info.metadata['extended_at'] = datetime.now().isoformat()

            logger.debug(f"Extended signal {signal_id} by {additional_minutes} minutes")
            return True

    def mark_signal_executed(
        self,
        signal_id: str,
        execution_data: Dict[str, Any]
    ) -> bool:
        """
        Mark a signal as executed and track performance

        Args:
            signal_id: Signal that was executed
            execution_data: Execution details for performance tracking

        Returns:
            True if marked successfully
        """
        with self._lock:
            if signal_id not in self._active_signals:
                return False

            validity_info = self._active_signals[signal_id]
            validity_info.state = SignalState.EXECUTED
            validity_info.validity_reason = ValidityReason.EXECUTION
            validity_info.execution_info = {
                'executed_at': datetime.now().isoformat(),
                'execution_price': execution_data.get('price'),
                'execution_volume': execution_data.get('volume'),
                'execution_type': execution_data.get('type', 'market'),
                **execution_data
            }

            # Track for performance analysis
            if self.config.track_execution_performance:
                self._track_signal_performance(validity_info, execution_data)

            # Move to expired (will be archived later)
            self._expired_signals[signal_id] = validity_info
            del self._active_signals[signal_id]

            # Trigger callback
            self._trigger_callbacks(SignalState.EXECUTED, validity_info)

            logger.debug(f"Marked signal {signal_id} as executed")
            return True

    def cancel_signal(
        self,
        signal_id: str,
        reason: str = "manual_cancellation"
    ) -> bool:
        """
        Manually cancel an active signal

        Args:
            signal_id: Signal to cancel
            reason: Cancellation reason

        Returns:
            True if cancelled successfully
        """
        with self._lock:
            if signal_id not in self._active_signals:
                return False

            return self._expire_signal(signal_id, ValidityReason.MANUAL_CANCELLATION)

    def get_signal_validity(self, signal_id: str) -> Optional[SignalValidityInfo]:
        """Get current validity information for a signal"""
        with self._lock:
            # Check active signals first
            if signal_id in self._active_signals:
                return self._active_signals[signal_id]

            # Check expired signals
            if signal_id in self._expired_signals:
                return self._expired_signals[signal_id]

            # Check archived signals
            if signal_id in self._archived_signals:
                return self._archived_signals[signal_id]

            return None

    def get_active_signals(self) -> List[SignalValidityInfo]:
        """Get all currently active signals"""
        with self._lock:
            return list(self._active_signals.values())

    def get_expired_signals(
        self,
        include_executed: bool = True,
        include_cancelled: bool = True
    ) -> List[SignalValidityInfo]:
        """Get expired signals with filtering options"""
        with self._lock:
            signals = []
            for validity_info in self._expired_signals.values():
                if validity_info.state == SignalState.EXECUTED and not include_executed:
                    continue
                if validity_info.state == SignalState.CANCELLED and not include_cancelled:
                    continue
                signals.append(validity_info)
            return signals

    def cleanup_expired_signals(self) -> int:
        """
        Manually trigger cleanup of expired signals

        Returns:
            Number of signals cleaned up
        """
        return self._cleanup_expired_signals()

    def register_state_callback(
        self,
        state: SignalState,
        callback: Callable[[SignalValidityInfo], None]
    ):
        """Register a callback for signal state changes"""
        self._callbacks[state].append(callback)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for timeout adjustments"""
        with self._lock:
            if not self._performance_history:
                return {}

            recent_signals = self._performance_history[-self.config.performance_lookback_signals:]

            total_signals = len(recent_signals)
            executed_signals = sum(1 for s in recent_signals if s.get('executed', False))
            expired_signals = total_signals - executed_signals

            execution_rate = executed_signals / total_signals if total_signals > 0 else 0

            avg_time_to_execution = 0
            if executed_signals > 0:
                execution_times = [
                    s.get('time_to_execution', 0)
                    for s in recent_signals
                    if s.get('executed', False)
                ]
                avg_time_to_execution = sum(execution_times) / len(execution_times)

            return {
                'total_signals': total_signals,
                'executed_signals': executed_signals,
                'expired_signals': expired_signals,
                'execution_rate': execution_rate,
                'avg_time_to_execution_minutes': avg_time_to_execution,
                'performance_adjustment_active': (
                    execution_rate < self.config.poor_performance_threshold or
                    execution_rate > self.config.good_performance_threshold
                ) if self.config.enable_performance_adjustment else False
            }

    def _calculate_adjusted_timeout(
        self,
        base_timeout: int,
        signal_data: Dict[str, Any]
    ) -> int:
        """Calculate timeout with performance and market condition adjustments"""
        adjusted_timeout = base_timeout

        # Performance-based adjustment
        if self.config.enable_performance_adjustment:
            stats = self.get_performance_stats()
            execution_rate = stats.get('execution_rate', 0.5)

            if execution_rate < self.config.poor_performance_threshold:
                # Poor performance, extend timeout
                adjusted_timeout *= self.config.performance_timeout_multiplier
            elif execution_rate > self.config.good_performance_threshold:
                # Good performance, can use shorter timeout
                adjusted_timeout /= self.config.performance_timeout_multiplier

        # Market condition adjustments
        volatility = signal_data.get('volatility', 'normal')
        if volatility == 'high':
            adjusted_timeout *= self.config.volatility_timeout_multiplier

        liquidity = signal_data.get('liquidity', 'normal')
        if liquidity == 'low':
            adjusted_timeout *= self.config.low_liquidity_timeout_multiplier

        # Apply limits
        adjusted_timeout = max(self.config.min_timeout_minutes, adjusted_timeout)
        adjusted_timeout = min(self.config.max_timeout_minutes, adjusted_timeout)

        return int(adjusted_timeout)

    def _extract_market_conditions(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant market conditions from signal data"""
        return {
            'volatility': signal_data.get('volatility'),
            'liquidity': signal_data.get('liquidity'),
            'trend_strength': signal_data.get('trend_strength'),
            'session': signal_data.get('session'),
            'major_news': signal_data.get('major_news', False)
        }

    def _expire_signal(self, signal_id: str, reason: ValidityReason) -> bool:
        """Internal method to expire a signal"""
        if signal_id not in self._active_signals:
            return False

        validity_info = self._active_signals[signal_id]
        validity_info.state = SignalState.EXPIRED
        validity_info.validity_reason = reason

        # Move to expired signals
        self._expired_signals[signal_id] = validity_info
        del self._active_signals[signal_id]

        # Trigger callback
        self._trigger_callbacks(SignalState.EXPIRED, validity_info)

        logger.debug(f"Expired signal {signal_id} due to {reason.value}")
        return True

    def _track_signal_performance(
        self,
        validity_info: SignalValidityInfo,
        execution_data: Dict[str, Any]
    ):
        """Track signal performance for timeout optimization"""
        time_to_execution = (
            datetime.now() - validity_info.created_at
        ).total_seconds() / 60  # Convert to minutes

        performance_record = {
            'signal_id': validity_info.signal_id,
            'executed': True,
            'time_to_execution': time_to_execution,
            'original_timeout': validity_info.original_timeout_minutes,
            'adjusted_timeout': validity_info.adjusted_timeout_minutes,
            'confidence': validity_info.metadata.get('confidence'),
            'strength_score': validity_info.metadata.get('strength_score'),
            'market_conditions': validity_info.market_conditions,
            'execution_price': execution_data.get('price'),
            'created_at': validity_info.created_at.isoformat()
        }

        self._performance_history.append(performance_record)

        # Keep only recent history
        max_history = self.config.performance_lookback_signals * 2
        if len(self._performance_history) > max_history:
            self._performance_history = self._performance_history[-max_history:]

    def _trigger_callbacks(self, state: SignalState, validity_info: SignalValidityInfo):
        """Trigger registered callbacks for state changes"""
        for callback in self._callbacks[state]:
            try:
                callback(validity_info)
            except Exception as e:
                logger.error(f"Error in signal state callback: {e}")

    def _cleanup_expired_signals(self) -> int:
        """Clean up expired signals and archive old ones"""
        cleaned_count = 0
        now = datetime.now()
        archive_cutoff = now - timedelta(days=self.config.archive_after_days)

        with self._lock:
            # Check for newly expired active signals
            expired_ids = []
            for signal_id, validity_info in self._active_signals.items():
                if now >= validity_info.expires_at:
                    expired_ids.append(signal_id)

            # Expire active signals that have timed out
            for signal_id in expired_ids:
                self._expire_signal(signal_id, ValidityReason.TIMEOUT)
                cleaned_count += 1

            # Archive old expired signals
            archived_ids = []
            for signal_id, validity_info in self._expired_signals.items():
                if validity_info.created_at < archive_cutoff:
                    archived_ids.append(signal_id)

            for signal_id in archived_ids:
                validity_info = self._expired_signals[signal_id]
                validity_info.state = SignalState.ARCHIVED
                self._archived_signals[signal_id] = validity_info
                del self._expired_signals[signal_id]

                # Trigger callback
                self._trigger_callbacks(SignalState.ARCHIVED, validity_info)
                cleaned_count += 1

        if cleaned_count > 0:
            logger.debug(f"Cleaned up {cleaned_count} signals")

        return cleaned_count

    def _force_cleanup_oldest(self):
        """Force cleanup of oldest signals when limit is reached"""
        if not self._active_signals:
            return

        # Find oldest signal
        oldest_id = min(
            self._active_signals.keys(),
            key=lambda x: self._active_signals[x].created_at
        )

        logger.warning(f"Force expiring oldest signal {oldest_id} due to limit")
        self._expire_signal(oldest_id, ValidityReason.TIMEOUT)

    async def _cleanup_loop(self):
        """Background cleanup loop"""
        while self._running:
            try:
                self._cleanup_expired_signals()
                await asyncio.sleep(self.config.cleanup_interval_minutes * 60)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying