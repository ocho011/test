"""
Discord Notification System for ICT Trading Bot

This package provides comprehensive Discord integration for real-time trading notifications,
system health monitoring, and automated performance reporting.
"""

from .discord_bot import DiscordBotManager, TradingBotDiscordClient
from .notification_manager import (
    NotificationManager,
    NotificationConfig,
    NotificationData,
    NotificationType,
    NotificationPriority,
    NotificationResult
)
from .discord_formatter import DiscordFormatter, MessageColor, EmojiSet
from .system_health_checker import (
    SystemHealthChecker,
    PerformanceReporter,
    HealthStatus,
    ComponentType,
    HealthMetric,
    ComponentHealth,
    PerformanceMetrics
)
from .discord_integration import (
    DiscordNotificationSystem,
    create_discord_system,
    get_global_discord_system,
    set_global_discord_system
)

__all__ = [
    # Core Discord bot components
    "DiscordBotManager",
    "TradingBotDiscordClient",
    
    # Notification management
    "NotificationManager",
    "NotificationConfig",
    "NotificationData",
    "NotificationType",
    "NotificationPriority",
    "NotificationResult",
    
    # Message formatting
    "DiscordFormatter",
    "MessageColor",
    "EmojiSet",
    
    # Health monitoring and reporting
    "SystemHealthChecker",
    "PerformanceReporter",
    "HealthStatus",
    "ComponentType",
    "HealthMetric",
    "ComponentHealth",
    "PerformanceMetrics",
    
    # Unified integration
    "DiscordNotificationSystem",
    "create_discord_system",
    "get_global_discord_system",
    "set_global_discord_system",
]