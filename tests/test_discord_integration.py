"""
Test script for Discord notification system integration.

This script provides basic testing functionality for the Discord notification system
without requiring a full trading bot setup.
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class MockBotConfig:
    """Mock bot configuration for testing."""
    
    def __init__(self):
        self.discord = {
            'token': os.getenv('DISCORD_BOT_TOKEN'),
            'channel_ids': {
                'trading': os.getenv('DISCORD_TRADING_CHANNEL_ID'),
                'alerts': os.getenv('DISCORD_ALERTS_CHANNEL_ID'),
                'reports': os.getenv('DISCORD_REPORTS_CHANNEL_ID'),
                'debug': os.getenv('DISCORD_DEBUG_CHANNEL_ID'),
            }
        }


async def test_discord_bot_connection():
    """Test basic Discord bot connection."""
    logger.info("Testing Discord bot connection...")
    
    try:
        from src.trading_bot.notifications.discord_bot import DiscordBotManager
        
        config = MockBotConfig()
        if not config.discord['token']:
            logger.error("DISCORD_BOT_TOKEN environment variable not set")
            return False
        
        manager = DiscordBotManager(config)
        
        # Start bot
        success = await manager.start()
        if not success:
            logger.error("Failed to start Discord bot")
            return False
        
        logger.info("Discord bot started successfully")
        
        # Test connection
        connection_info = await manager.test_connection()
        logger.info(f"Connection info: {connection_info}")
        
        # Stop bot
        await manager.stop()
        logger.info("Discord bot stopped")
        
        return True
        
    except Exception as e:
        logger.error(f"Discord bot connection test failed: {e}")
        return False


async def test_notification_manager():
    """Test notification manager functionality."""
    logger.info("Testing notification manager...")
    
    try:
        from src.trading_bot.notifications.discord_bot import DiscordBotManager
        from src.trading_bot.notifications.notification_manager import NotificationManager, NotificationConfig
        
        config = MockBotConfig()
        if not config.discord['token']:
            logger.error("DISCORD_BOT_TOKEN environment variable not set")
            return False
        
        # Create components
        discord_manager = DiscordBotManager(config)
        notification_config = NotificationConfig()
        notification_manager = NotificationManager(discord_manager, notification_config)
        
        # Start Discord bot
        if not await discord_manager.start():
            logger.error("Failed to start Discord bot")
            return False
        
        # Start notification manager
        await notification_manager.start()
        
        # Test creating notifications
        trade_notification = notification_manager.create_trade_notification(
            trade_type="entry",
            symbol="BTCUSDT",
            side="buy",
            quantity=0.001,
            price=45000.0,
            additional_data={'strategy': 'ICT_OB_FVG'}
        )
        
        system_notification = notification_manager.create_system_notification(
            level="info",
            title="Test System Notification",
            message="This is a test system notification from the integration test.",
            additional_data={'test': True}
        )
        
        # Queue notifications
        trade_queued = notification_manager.queue_notification(trade_notification)
        system_queued = notification_manager.queue_notification(system_notification)
        
        logger.info(f"Trade notification queued: {trade_queued}")
        logger.info(f"System notification queued: {system_queued}")
        
        # Wait a bit for processing
        await asyncio.sleep(5)
        
        # Check statistics
        stats = notification_manager.get_statistics()
        logger.info(f"Notification statistics: {stats}")
        
        # Cleanup
        await notification_manager.stop()
        await discord_manager.stop()
        
        return True
        
    except Exception as e:
        logger.error(f"Notification manager test failed: {e}")
        return False


async def test_discord_formatter():
    """Test Discord message formatting."""
    logger.info("Testing Discord formatter...")
    
    try:
        from src.trading_bot.notifications.discord_formatter import DiscordFormatter
        from src.trading_bot.notifications.notification_manager import NotificationData, NotificationType, NotificationPriority
        
        formatter = DiscordFormatter()
        
        # Test trade notification formatting
        trade_notification = NotificationData(
            id="test_trade_1",
            type=NotificationType.TRADE_ENTRY,
            priority=NotificationPriority.HIGH,
            title="Trade Entry: BTCUSDT",
            message="BUY 0.001 BTCUSDT at $45000.00",
            data={
                'symbol': 'BTCUSDT',
                'side': 'buy',
                'quantity': 0.001,
                'price': 45000.0,
                'strategy': 'ICT_OB_FVG'
            }
        )
        
        formatted = formatter.format_notification(trade_notification)
        logger.info("Trade notification formatted successfully")
        logger.info(f"Embed title: {formatted['embed'].title}")
        
        # Test system notification formatting
        system_notification = NotificationData(
            id="test_system_1",
            type=NotificationType.SYSTEM_ERROR,
            priority=NotificationPriority.CRITICAL,
            title="System Error Test",
            message="This is a test system error notification",
            data={
                'component': 'test_component',
                'error_code': 'TEST_001',
                'level': 'error'
            }
        )
        
        formatted_system = formatter.format_notification(system_notification)
        logger.info("System notification formatted successfully")
        logger.info(f"System embed title: {formatted_system['embed'].title}")
        
        return True
        
    except Exception as e:
        logger.error(f"Discord formatter test failed: {e}")
        return False


async def test_health_checker():
    """Test system health checker."""
    logger.info("Testing system health checker...")
    
    try:
        from src.trading_bot.notifications.discord_bot import DiscordBotManager
        from src.trading_bot.notifications.notification_manager import NotificationManager
        from src.trading_bot.notifications.system_health_checker import SystemHealthChecker
        
        config = MockBotConfig()
        discord_manager = DiscordBotManager(config)
        notification_manager = NotificationManager(discord_manager)
        health_checker = SystemHealthChecker(notification_manager)
        
        # Run health check
        health_results = await health_checker.check_all_components()
        logger.info(f"Health check completed for {len(health_results)} components")
        
        for component, health in health_results.items():
            logger.info(f"{component.value}: {health.status.value} ({len(health.metrics)} metrics)")
        
        # Get health summary
        summary = health_checker.get_health_summary()
        logger.info(f"Health summary: {summary}")
        
        return True
        
    except Exception as e:
        logger.error(f"Health checker test failed: {e}")
        return False


async def test_full_integration():
    """Test full Discord integration system."""
    logger.info("Testing full Discord integration...")
    
    try:
        from src.trading_bot.notifications.discord_integration import DiscordNotificationSystem, NotificationConfig
        
        config = MockBotConfig()
        if not config.discord['token']:
            logger.error("DISCORD_BOT_TOKEN environment variable not set")
            return False
        
        # Create Discord system
        notification_config = NotificationConfig()
        discord_system = DiscordNotificationSystem(config, notification_config)
        
        # Initialize
        if not await discord_system.initialize():
            logger.error("Failed to initialize Discord system")
            return False
        
        logger.info("Discord system initialized successfully")
        
        # Start monitoring
        if not await discord_system.start_monitoring():
            logger.warning("Failed to start monitoring (this is expected without full bot setup)")
        
        # Test connection
        connection_status = await discord_system.test_connection()
        logger.info(f"Connection test result: {connection_status}")
        
        # Send test notifications
        trade_sent = discord_system.send_trade_notification(
            trade_type="entry",
            symbol="ETHUSDT",
            side="buy",
            quantity=0.1,
            price=3000.0,
            additional_data={'test': True}
        )
        logger.info(f"Trade notification sent: {trade_sent}")
        
        signal_sent = discord_system.send_signal_notification(
            symbol="BTCUSDT",
            signal_type="ORDER_BLOCK",
            confidence=0.85,
            strength=7.5,
            additional_info="Strong bullish signal detected"
        )
        logger.info(f"Signal notification sent: {signal_sent}")
        
        system_sent = discord_system.send_system_notification(
            level="info",
            title="Integration Test Complete",
            message="Discord integration test completed successfully!",
            additional_data={'test_time': datetime.utcnow().isoformat()}
        )
        logger.info(f"System notification sent: {system_sent}")
        
        # Wait for processing
        await asyncio.sleep(10)
        
        # Get system status
        status = discord_system.get_system_status()
        logger.info(f"System status: {status}")
        
        # Get statistics
        stats = discord_system.get_notification_statistics()
        logger.info(f"Notification statistics: {stats}")
        
        # Shutdown
        await discord_system.shutdown()
        logger.info("Discord system shutdown complete")
        
        return True
        
    except Exception as e:
        logger.error(f"Full integration test failed: {e}")
        return False


async def main():
    """Run all tests."""
    logger.info("Starting Discord integration tests...")
    
    # Check environment variables
    required_env_vars = ['DISCORD_BOT_TOKEN']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        logger.error("Please set these variables in your .env file or environment:")
        logger.error("DISCORD_BOT_TOKEN=your_discord_bot_token")
        logger.error("DISCORD_TRADING_CHANNEL_ID=channel_id (optional)")
        logger.error("DISCORD_ALERTS_CHANNEL_ID=channel_id (optional)")
        logger.error("DISCORD_REPORTS_CHANNEL_ID=channel_id (optional)")
        return
    
    tests = [
        ("Discord Formatter", test_discord_formatter),
        ("Health Checker", test_health_checker),
        ("Discord Bot Connection", test_discord_bot_connection),
        ("Notification Manager", test_notification_manager),
        ("Full Integration", test_full_integration),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running {test_name} test...")
        logger.info(f"{'='*50}")
        
        try:
            result = await test_func()
            results[test_name] = result
            logger.info(f"{test_name} test: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            logger.error(f"{test_name} test failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("All tests passed! Discord integration is working correctly.")
    else:
        logger.warning(f"{total - passed} tests failed. Please check the logs above for details.")


if __name__ == "__main__":
    asyncio.run(main())