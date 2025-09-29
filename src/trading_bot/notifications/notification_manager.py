"""
NotificationManager for ICT Trading System

This module implements comprehensive notification management with priority handling,
spam prevention, and integration with the existing signal event system.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
from collections import defaultdict, deque
import json

from .discord_bot import DiscordBotManager
from ..signals.signal_event_publisher import SignalEventData, EventPriority
from ..core.events import SignalEvent


class NotificationType(Enum):
    """Types of notifications supported by the system."""
    
    TRADE_ENTRY = "trade_entry"
    TRADE_EXIT = "trade_exit"
    POSITION_UPDATE = "position_update"
    SYSTEM_ERROR = "system_error"
    SYSTEM_WARNING = "system_warning"
    DAILY_REPORT = "daily_report"
    WEEKLY_REPORT = "weekly_report"
    MONTHLY_REPORT = "monthly_report"
    SIGNAL_GENERATED = "signal_generated"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    RISK_ALERT = "risk_alert"
    CONNECTION_STATUS = "connection_status"


class NotificationPriority(Enum):
    """Notification priority levels."""
    
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    URGENT = 5


@dataclass
class NotificationConfig:
    """Configuration for notification behavior."""
    
    # Spam prevention settings
    rate_limit_window: int = 60  # seconds
    max_notifications_per_window: int = 10
    duplicate_suppression_window: int = 300  # seconds
    
    # Priority-based rate limiting
    priority_rate_limits: Dict[NotificationPriority, int] = field(default_factory=lambda: {
        NotificationPriority.LOW: 5,
        NotificationPriority.NORMAL: 8,
        NotificationPriority.HIGH: 12,
        NotificationPriority.CRITICAL: 20,
        NotificationPriority.URGENT: 50
    })
    
    # Retry configuration
    max_retries: int = 3
    retry_delay: int = 5  # seconds
    retry_backoff_multiplier: float = 2.0
    
    # Channel routing
    channel_routing: Dict[NotificationType, str] = field(default_factory=lambda: {
        NotificationType.TRADE_ENTRY: "trading",
        NotificationType.TRADE_EXIT: "trading",
        NotificationType.POSITION_UPDATE: "trading",
        NotificationType.SIGNAL_GENERATED: "trading",
        NotificationType.ORDER_FILLED: "trading",
        NotificationType.ORDER_CANCELLED: "trading",
        NotificationType.SYSTEM_ERROR: "alerts",
        NotificationType.SYSTEM_WARNING: "alerts",
        NotificationType.RISK_ALERT: "alerts",
        NotificationType.CONNECTION_STATUS: "alerts",
        NotificationType.DAILY_REPORT: "reports",
        NotificationType.WEEKLY_REPORT: "reports",
        NotificationType.MONTHLY_REPORT: "reports",
    })


@dataclass
class NotificationData:
    """Data structure for a notification."""
    
    id: str
    type: NotificationType
    priority: NotificationPriority
    title: str
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    retry_count: int = 0
    max_retries: int = 3
    channel_override: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert notification to dictionary."""
        return {
            'id': self.id,
            'type': self.type.value,
            'priority': self.priority.value,
            'title': self.title,
            'message': self.message,
            'data': self.data,
            'timestamp': self.timestamp.isoformat(),
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'channel_override': self.channel_override
        }


@dataclass
class NotificationResult:
    """Result of a notification attempt."""
    
    success: bool
    notification_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    error_message: Optional[str] = None
    channel_used: Optional[str] = None
    retry_scheduled: bool = False


class NotificationQueue:
    """Priority queue for notifications with spam prevention."""
    
    def __init__(self, config: NotificationConfig):
        self.config = config
        self.queues = {
            priority: deque() for priority in NotificationPriority
        }
        self.recent_notifications = deque()
        self.duplicate_hashes = {}
        self.rate_limit_counters = defaultdict(list)
        self.logger = logging.getLogger(__name__)
    
    def add_notification(self, notification: NotificationData) -> bool:
        """
        Add notification to queue with spam prevention.
        
        Args:
            notification: Notification to add
            
        Returns:
            bool: True if notification was added, False if blocked by spam prevention
        """
        # Check for duplicates
        if self._is_duplicate(notification):
            self.logger.debug(f"Blocked duplicate notification: {notification.id}")
            return False
        
        # Check rate limits
        if self._exceeds_rate_limit(notification):
            self.logger.warning(f"Rate limit exceeded for notification: {notification.id}")
            return False
        
        # Add to appropriate priority queue
        self.queues[notification.priority].append(notification)
        
        # Track for spam prevention
        self._track_notification(notification)
        
        self.logger.debug(f"Added notification to queue: {notification.id} (priority: {notification.priority})")
        return True
    
    def get_next_notification(self) -> Optional[NotificationData]:
        """Get the next notification to process (highest priority first)."""
        for priority in sorted(NotificationPriority, key=lambda x: x.value, reverse=True):
            if self.queues[priority]:
                return self.queues[priority].popleft()
        return None
    
    def get_queue_size(self) -> Dict[NotificationPriority, int]:
        """Get the size of each priority queue."""
        return {priority: len(queue) for priority, queue in self.queues.items()}
    
    def _is_duplicate(self, notification: NotificationData) -> bool:
        """Check if notification is a duplicate within the suppression window."""
        # Create hash based on type, title, and key data
        content_hash = hash((
            notification.type.value,
            notification.title,
            str(sorted(notification.data.items()) if notification.data else "")
        ))
        
        now = datetime.utcnow()
        
        # Check if we've seen this exact notification recently
        if content_hash in self.duplicate_hashes:
            last_seen = self.duplicate_hashes[content_hash]
            if (now - last_seen).total_seconds() < self.config.duplicate_suppression_window:
                return True
        
        # Update hash tracking
        self.duplicate_hashes[content_hash] = now
        
        # Clean old hashes
        cutoff = now - timedelta(seconds=self.config.duplicate_suppression_window)
        self.duplicate_hashes = {
            h: ts for h, ts in self.duplicate_hashes.items()
            if ts > cutoff
        }
        
        return False
    
    def _exceeds_rate_limit(self, notification: NotificationData) -> bool:
        """Check if adding this notification would exceed rate limits."""
        now = datetime.utcnow()
        priority = notification.priority
        
        # Clean old timestamps
        cutoff = now - timedelta(seconds=self.config.rate_limit_window)
        self.rate_limit_counters[priority] = [
            ts for ts in self.rate_limit_counters[priority] if ts > cutoff
        ]
        
        # Check if we're at the limit
        current_count = len(self.rate_limit_counters[priority])
        limit = self.config.priority_rate_limits.get(priority, 10)
        
        return current_count >= limit
    
    def _track_notification(self, notification: NotificationData):
        """Track notification for rate limiting."""
        now = datetime.utcnow()
        self.rate_limit_counters[notification.priority].append(now)
        self.recent_notifications.append({
            'id': notification.id,
            'timestamp': now,
            'type': notification.type.value,
            'priority': notification.priority.value
        })
        
        # Keep only recent notifications for monitoring
        cutoff = now - timedelta(hours=1)
        while (self.recent_notifications and 
               self.recent_notifications[0]['timestamp'] < cutoff):
            self.recent_notifications.popleft()


class NotificationManager:
    """
    Main notification manager that handles all notification processing.
    
    Integrates with Discord bot, manages notification queues, implements
    spam prevention, and provides retry logic for failed notifications.
    """
    
    def __init__(self, discord_manager: DiscordBotManager, config: NotificationConfig = None):
        """
        Initialize the NotificationManager.
        
        Args:
            discord_manager: Discord bot manager instance
            config: Notification configuration
        """
        self.discord_manager = discord_manager
        self.config = config or NotificationConfig()
        self.queue = NotificationQueue(self.config)
        self.logger = logging.getLogger(__name__)
        
        self._running = False
        self._processing_task: Optional[asyncio.Task] = None
        self._retry_queue: List[NotificationData] = []
        self._failed_notifications: List[Dict[str, Any]] = []
        
        # Statistics tracking
        self._stats = {
            'total_sent': 0,
            'total_failed': 0,
            'total_suppressed': 0,
            'rate_limited': 0,
            'by_type': defaultdict(int),
            'by_priority': defaultdict(int)
        }
    
    async def start(self):
        """Start the notification manager processing loop."""
        if self._running:
            self.logger.warning("NotificationManager is already running")
            return
        
        self._running = True
        self._processing_task = asyncio.create_task(self._processing_loop())
        self.logger.info("NotificationManager started")
    
    async def stop(self):
        """Stop the notification manager."""
        if not self._running:
            return
        
        self._running = False
        
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("NotificationManager stopped")
    
    async def send_notification(self, notification: NotificationData) -> NotificationResult:
        """
        Send a notification immediately (bypass queue).
        
        Args:
            notification: Notification to send
            
        Returns:
            NotificationResult: Result of the notification attempt
        """
        return await self._send_notification_to_discord(notification)
    
    def queue_notification(self, notification: NotificationData) -> bool:
        """
        Add notification to the processing queue.
        
        Args:
            notification: Notification to queue
            
        Returns:
            bool: True if notification was queued, False if blocked
        """
        added = self.queue.add_notification(notification)
        
        if added:
            self._stats['by_type'][notification.type.value] += 1
            self._stats['by_priority'][notification.priority.value] += 1
        else:
            self._stats['total_suppressed'] += 1
        
        return added
    
    def create_trade_notification(
        self,
        trade_type: str,  # "entry" or "exit"
        symbol: str,
        side: str,  # "buy" or "sell"
        quantity: float,
        price: float,
        pnl: Optional[float] = None,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> NotificationData:
        """Create a trading notification."""
        notification_type = (
            NotificationType.TRADE_ENTRY if trade_type == "entry" 
            else NotificationType.TRADE_EXIT
        )
        
        title = f"📈 Trade {trade_type.title()}: {symbol}"
        
        if trade_type == "entry":
            message = f"{side.upper()} {quantity} {symbol} at ${price:.4f}"
        else:
            pnl_text = f" | PnL: ${pnl:.2f}" if pnl is not None else ""
            message = f"Closed {side.upper()} {quantity} {symbol} at ${price:.4f}{pnl_text}"
        
        data = {
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            'trade_type': trade_type
        }
        
        if pnl is not None:
            data['pnl'] = pnl
        
        if additional_data:
            data.update(additional_data)
        
        return NotificationData(
            id=f"trade_{trade_type}_{symbol}_{datetime.utcnow().timestamp()}",
            type=notification_type,
            priority=NotificationPriority.HIGH,
            title=title,
            message=message,
            data=data
        )
    
    def create_signal_notification(
        self,
        signal_data: SignalEventData,
        additional_info: Optional[str] = None
    ) -> NotificationData:
        """Create a notification from signal event data."""
        title = f"🎯 Signal Generated: {signal_data.symbol}"
        message = f"{signal_data.signal_type} signal for {signal_data.symbol}"
        
        if additional_info:
            message += f" | {additional_info}"
        
        return NotificationData(
            id=f"signal_{signal_data.signal_id}",
            type=NotificationType.SIGNAL_GENERATED,
            priority=NotificationPriority.NORMAL,
            title=title,
            message=message,
            data={
                'signal_id': signal_data.signal_id,
                'symbol': signal_data.symbol,
                'signal_type': signal_data.signal_type,
                'confidence': signal_data.confidence,
                'strength': signal_data.strength
            }
        )
    
    def create_system_notification(
        self,
        level: str,  # "error", "warning", "info"
        title: str,
        message: str,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> NotificationData:
        """Create a system notification."""
        if level == "error":
            notification_type = NotificationType.SYSTEM_ERROR
            priority = NotificationPriority.CRITICAL
            title = f"❌ {title}"
        elif level == "warning":
            notification_type = NotificationType.SYSTEM_WARNING
            priority = NotificationPriority.HIGH
            title = f"⚠️ {title}"
        else:
            notification_type = NotificationType.CONNECTION_STATUS
            priority = NotificationPriority.NORMAL
            title = f"ℹ️ {title}"
        
        data = {'level': level}
        if additional_data:
            data.update(additional_data)
        
        return NotificationData(
            id=f"system_{level}_{datetime.utcnow().timestamp()}",
            type=notification_type,
            priority=priority,
            title=title,
            message=message,
            data=data
        )
    
    async def _processing_loop(self):
        """Main processing loop for queued notifications."""
        while self._running:
            try:
                # Process retry queue first
                await self._process_retry_queue()
                
                # Process regular queue
                notification = self.queue.get_next_notification()
                if notification:
                    result = await self._send_notification_to_discord(notification)
                    
                    if not result.success and notification.retry_count < notification.max_retries:
                        # Schedule retry
                        notification.retry_count += 1
                        self._retry_queue.append(notification)
                        self.logger.info(f"Scheduled retry for notification {notification.id} (attempt {notification.retry_count})")
                    elif not result.success:
                        # Max retries exceeded
                        self._failed_notifications.append({
                            'notification': notification.to_dict(),
                            'final_error': result.error_message,
                            'timestamp': datetime.utcnow().isoformat()
                        })
                        self._stats['total_failed'] += 1
                        self.logger.error(f"Notification {notification.id} failed permanently after {notification.retry_count} retries")
                
                # Small delay to prevent CPU spinning
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error in notification processing loop: {e}", exc_info=True)
                await asyncio.sleep(1)
    
    async def _process_retry_queue(self):
        """Process notifications waiting for retry."""
        if not self._retry_queue:
            return
        
        # Find notifications ready for retry
        now = datetime.utcnow()
        ready_for_retry = []
        
        for notification in self._retry_queue[:]:
            retry_delay = (self.config.retry_delay * 
                          (self.config.retry_backoff_multiplier ** (notification.retry_count - 1)))
            
            if (now - notification.timestamp).total_seconds() >= retry_delay:
                ready_for_retry.append(notification)
                self._retry_queue.remove(notification)
        
        # Retry notifications
        for notification in ready_for_retry:
            result = await self._send_notification_to_discord(notification)
            
            if not result.success and notification.retry_count < notification.max_retries:
                notification.retry_count += 1
                self._retry_queue.append(notification)
            elif not result.success:
                self._failed_notifications.append({
                    'notification': notification.to_dict(),
                    'final_error': result.error_message,
                    'timestamp': datetime.utcnow().isoformat()
                })
                self._stats['total_failed'] += 1
    
    async def _send_notification_to_discord(self, notification: NotificationData) -> NotificationResult:
        """
        Send notification to Discord via the bot manager.
        
        Args:
            notification: Notification to send
            
        Returns:
            NotificationResult: Result of the send attempt
        """
        try:
            # Determine channel
            channel = (notification.channel_override or 
                      self.config.channel_routing.get(notification.type, "alerts"))
            
            # Check if Discord bot is available
            if not self.discord_manager.is_running():
                return NotificationResult(
                    success=False,
                    notification_id=notification.id,
                    error_message="Discord bot is not running",
                    channel_used=channel
                )
            
            client = self.discord_manager.get_client()
            if not client:
                return NotificationResult(
                    success=False,
                    notification_id=notification.id,
                    error_message="Discord client not available",
                    channel_used=channel
                )
            
            # Send message
            success = await client.send_message(
                channel_type=channel,
                content=f"**{notification.title}**\n{notification.message}"
            )
            
            if success:
                self._stats['total_sent'] += 1
                return NotificationResult(
                    success=True,
                    notification_id=notification.id,
                    channel_used=channel
                )
            else:
                return NotificationResult(
                    success=False,
                    notification_id=notification.id,
                    error_message="Failed to send message to Discord",
                    channel_used=channel
                )
        
        except Exception as e:
            self.logger.error(f"Error sending notification {notification.id}: {e}")
            return NotificationResult(
                success=False,
                notification_id=notification.id,
                error_message=str(e),
                channel_used=channel if 'channel' in locals() else None
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get notification statistics."""
        queue_sizes = self.queue.get_queue_size()
        
        return {
            'statistics': dict(self._stats),
            'queue_sizes': {str(k): v for k, v in queue_sizes.items()},
            'retry_queue_size': len(self._retry_queue),
            'failed_notifications_count': len(self._failed_notifications),
            'is_running': self._running,
            'discord_connected': self.discord_manager.is_running()
        }
    
    def get_failed_notifications(self) -> List[Dict[str, Any]]:
        """Get list of failed notifications."""
        return self._failed_notifications.copy()
    
    def clear_failed_notifications(self):
        """Clear the failed notifications list."""
        self._failed_notifications.clear()