"""
Discord Integration Module for ICT Trading System

This module provides a unified interface for Discord notifications,
integrating all Discord-related components for easy use throughout the trading bot.
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from datetime import datetime

from ..config.bot_config import BotConfig
from .discord_bot import DiscordBotManager, TradingBotDiscordClient
from .notification_manager import NotificationManager, NotificationConfig, NotificationData
from .discord_formatter import DiscordFormatter
from .system_health_checker import SystemHealthChecker, PerformanceReporter


class DiscordNotificationSystem:
    """
    Unified Discord notification system for the ICT trading bot.
    
    This class provides a single interface for all Discord-related functionality,
    including bot management, notification processing, health monitoring, and
    performance reporting.
    """
    
    def __init__(self, bot_config: BotConfig, notification_config: Optional[NotificationConfig] = None):
        """
        Initialize the Discord notification system.
        
        Args:
            bot_config: Bot configuration containing Discord settings
            notification_config: Notification configuration (optional)
        """
        self.bot_config = bot_config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.discord_manager = DiscordBotManager(bot_config)
        self.notification_config = notification_config or NotificationConfig()
        self.notification_manager = NotificationManager(self.discord_manager, self.notification_config)
        self.formatter = DiscordFormatter()
        self.health_checker = SystemHealthChecker(self.notification_manager)
        self.performance_reporter = PerformanceReporter(self.notification_manager)
        
        self._initialized = False
        self._running = False
    
    async def initialize(self) -> bool:
        """
        Initialize all Discord components.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        if self._initialized:
            self.logger.warning("Discord notification system already initialized")
            return True
        
        try:
            self.logger.info("Initializing Discord notification system...")
            
            # Start Discord bot
            if not await self.discord_manager.start():
                self.logger.error("Failed to start Discord bot")
                return False
            
            self.logger.info("Discord bot started successfully")
            
            # Initialize notification manager
            await self.notification_manager.start()
            self.logger.info("Notification manager started")
            
            self._initialized = True
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Discord notification system: {e}")
            return False
    
    async def start_monitoring(self) -> bool:
        """
        Start health monitoring and performance reporting.
        
        Returns:
            bool: True if monitoring started successfully, False otherwise
        """
        if not self._initialized:
            self.logger.error("Discord notification system not initialized")
            return False
        
        try:
            # Start health monitoring
            await self.health_checker.start_monitoring()
            self.logger.info("Health monitoring started")
            
            # Start performance reporting
            await self.performance_reporter.start_reporting()
            self.logger.info("Performance reporting started")
            
            self._running = True
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start monitoring: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown all Discord components gracefully."""
        self.logger.info("Shutting down Discord notification system...")
        
        try:
            # Stop monitoring components
            if hasattr(self.health_checker, '_running') and self.health_checker._running:
                await self.health_checker.stop_monitoring()
            
            if hasattr(self.performance_reporter, '_running') and self.performance_reporter._running:
                await self.performance_reporter.stop_reporting()
            
            # Stop notification manager
            if hasattr(self.notification_manager, '_running') and self.notification_manager._running:
                await self.notification_manager.stop()
            
            # Stop Discord bot
            if self.discord_manager.is_running():
                await self.discord_manager.stop()
            
            self._running = False
            self._initialized = False
            
            self.logger.info("Discord notification system shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    # Convenience methods for sending notifications
    
    def send_trade_notification(
        self,
        trade_type: str,  # "entry" or "exit"
        symbol: str,
        side: str,  # "buy" or "sell"
        quantity: float,
        price: float,
        pnl: Optional[float] = None,
        additional_data: Optional[Dict[str, Any]] = None,
        include_chart: bool = False,
        chart_data: Optional[bytes] = None
    ) -> bool:
        """
        Send a trading notification.
        
        Args:
            trade_type: "entry" or "exit"
            symbol: Trading symbol
            side: "buy" or "sell"
            quantity: Trade quantity
            price: Execution price
            pnl: Profit/loss (for exits)
            additional_data: Additional trade data
            include_chart: Whether to include chart image
            chart_data: Chart image data as bytes
            
        Returns:
            bool: True if notification was queued successfully
        """
        try:
            notification = self.notification_manager.create_trade_notification(
                trade_type=trade_type,
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price,
                pnl=pnl,
                additional_data=additional_data
            )
            
            return self.notification_manager.queue_notification(notification)
            
        except Exception as e:
            self.logger.error(f"Failed to send trade notification: {e}")
            return False
    
    def send_signal_notification(
        self,
        symbol: str,
        signal_type: str,
        confidence: float,
        strength: Optional[float] = None,
        additional_info: Optional[str] = None,
        additional_data: Optional[Dict[str, Any]] = None,
        include_chart: bool = False,
        chart_data: Optional[bytes] = None
    ) -> bool:
        """
        Send a signal notification.
        
        Args:
            symbol: Trading symbol
            signal_type: Type of signal
            confidence: Signal confidence (0-1)
            strength: Signal strength
            additional_info: Additional info text
            additional_data: Additional signal data
            include_chart: Whether to include chart image
            chart_data: Chart image data as bytes
            
        Returns:
            bool: True if notification was queued successfully
        """
        try:
            # Create signal event data structure
            from ..signals.signal_event_publisher import SignalEventData
            
            signal_data = SignalEventData(
                signal_id=f"signal_{symbol}_{datetime.utcnow().timestamp()}",
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                strength=strength or 0.0,
                timestamp=datetime.utcnow()
            )
            
            notification = self.notification_manager.create_signal_notification(
                signal_data=signal_data,
                additional_info=additional_info
            )
            
            # Add additional data if provided
            if additional_data:
                notification.data.update(additional_data)
            
            return self.notification_manager.queue_notification(notification)
            
        except Exception as e:
            self.logger.error(f"Failed to send signal notification: {e}")
            return False
    
    def send_system_notification(
        self,
        level: str,  # "error", "warning", "info"
        title: str,
        message: str,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Send a system notification.
        
        Args:
            level: Notification level ("error", "warning", "info")
            title: Notification title
            message: Notification message
            additional_data: Additional data
            
        Returns:
            bool: True if notification was queued successfully
        """
        try:
            notification = self.notification_manager.create_system_notification(
                level=level,
                title=title,
                message=message,
                additional_data=additional_data
            )
            
            return self.notification_manager.queue_notification(notification)
            
        except Exception as e:
            self.logger.error(f"Failed to send system notification: {e}")
            return False
    
    def send_custom_notification(self, notification_data: NotificationData) -> bool:
        """
        Send a custom notification.
        
        Args:
            notification_data: Custom notification data
            
        Returns:
            bool: True if notification was queued successfully
        """
        try:
            return self.notification_manager.queue_notification(notification_data)
        except Exception as e:
            self.logger.error(f"Failed to send custom notification: {e}")
            return False
    
    # Status and monitoring methods
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'initialized': self._initialized,
            'running': self._running,
            'discord_bot': self.discord_manager.is_running(),
            'notification_manager': self.notification_manager.get_statistics(),
            'health_checker': self.health_checker.get_health_summary(),
            'performance_reporter': self.performance_reporter.get_reporting_status(),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test Discord connection and send test message."""
        try:
            # Test Discord connection
            connection_status = await self.discord_manager.test_connection()
            
            # Send test notification if connected
            if connection_status.get('connected', False):
                test_notification = self.notification_manager.create_system_notification(
                    level="info",
                    title="Discord Connection Test",
                    message="This is a test message to verify Discord integration is working properly.",
                    additional_data={'test': True}
                )
                
                queued = self.notification_manager.queue_notification(test_notification)
                connection_status['test_message_sent'] = queued
            
            return connection_status
            
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return {
                'connected': False,
                'error': str(e),
                'test_message_sent': False
            }
    
    def get_notification_statistics(self) -> Dict[str, Any]:
        """Get notification statistics."""
        return self.notification_manager.get_statistics()
    
    def get_failed_notifications(self) -> list:
        """Get list of failed notifications."""
        return self.notification_manager.get_failed_notifications()
    
    def clear_failed_notifications(self):
        """Clear failed notifications list."""
        self.notification_manager.clear_failed_notifications()
    
    async def force_health_check(self) -> Dict[str, Any]:
        """Force an immediate health check."""
        try:
            health_results = await self.health_checker.check_all_components()
            return {
                component.value: {
                    'status': health.status.value,
                    'last_check': health.last_check.isoformat(),
                    'metrics': [
                        {
                            'name': metric.name,
                            'value': metric.value,
                            'unit': metric.unit,
                            'status': metric.status.value
                        }
                        for metric in health.metrics
                    ],
                    'error_message': health.error_message
                }
                for component, health in health_results.items()
            }
        except Exception as e:
            self.logger.error(f"Forced health check failed: {e}")
            return {'error': str(e)}
    
    async def generate_performance_report(self, report_type: str = "daily") -> Dict[str, Any]:
        """
        Generate and send performance report.
        
        Args:
            report_type: "daily", "weekly", or "monthly"
            
        Returns:
            dict: Performance metrics
        """
        try:
            if report_type == "daily":
                metrics = await self.performance_reporter.generate_daily_report()
            elif report_type == "weekly":
                metrics = await self.performance_reporter.generate_weekly_report()
            elif report_type == "monthly":
                metrics = await self.performance_reporter.generate_monthly_report()
            else:
                raise ValueError(f"Invalid report type: {report_type}")
            
            await self.performance_reporter.send_performance_report(metrics, report_type)
            
            return metrics.to_dict()
            
        except Exception as e:
            self.logger.error(f"Failed to generate {report_type} report: {e}")
            return {'error': str(e)}
    
    # Configuration methods
    
    def update_notification_config(self, new_config: NotificationConfig):
        """Update notification configuration."""
        self.notification_config = new_config
        self.notification_manager.config = new_config
        self.notification_manager.queue.config = new_config
    
    def get_discord_channels(self) -> Dict[str, Any]:
        """Get information about configured Discord channels."""
        if self.discord_manager.is_running():
            client = self.discord_manager.get_client()
            if client:
                return asyncio.run(client.get_channel_info())
        return {}


# Convenience function for easy integration
async def create_discord_system(bot_config: BotConfig, 
                              notification_config: Optional[NotificationConfig] = None,
                              start_monitoring: bool = True) -> DiscordNotificationSystem:
    """
    Create and initialize a Discord notification system.
    
    Args:
        bot_config: Bot configuration
        notification_config: Notification configuration (optional)
        start_monitoring: Whether to start health monitoring and reporting
        
    Returns:
        DiscordNotificationSystem: Initialized Discord system
    """
    system = DiscordNotificationSystem(bot_config, notification_config)
    
    if await system.initialize():
        if start_monitoring:
            await system.start_monitoring()
        return system
    else:
        raise RuntimeError("Failed to initialize Discord notification system")


# Global system instance (optional, for singleton pattern)
_global_discord_system: Optional[DiscordNotificationSystem] = None


def get_global_discord_system() -> Optional[DiscordNotificationSystem]:
    """Get the global Discord system instance."""
    return _global_discord_system


def set_global_discord_system(system: DiscordNotificationSystem):
    """Set the global Discord system instance."""
    global _global_discord_system
    _global_discord_system = system