"""
Discord Bot for ICT Trading System

This module implements the Discord bot client with basic connection and setup functionality.
Provides the foundation for sending trading notifications and system updates to Discord.
"""

import asyncio
import logging
from typing import Optional
import discord
from discord.ext import commands
import os
from datetime import datetime

from ..config.bot_config import BotConfig


class TradingBotDiscordClient(discord.Client):
    """
    Discord client for the ICT trading bot.
    
    Handles connection, authentication, and provides the foundation
    for sending notifications to Discord channels.
    """
    
    def __init__(self, config: BotConfig, *args, **kwargs):
        """
        Initialize the Discord client.
        
        Args:
            config: Bot configuration containing Discord token and settings
        """
        # Set up intents for the bot
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        
        super().__init__(intents=intents, *args, **kwargs)
        
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.is_ready = False
        self.start_time = datetime.utcnow()
        
        # Discord channels for different notification types
        self.channels = {
            'trading': None,      # Main trading notifications
            'alerts': None,       # System alerts and errors
            'reports': None,      # Daily/weekly performance reports
            'debug': None,        # Debug information (optional)
        }
    
    async def on_ready(self):
        """Called when the bot has successfully connected to Discord."""
        self.logger.info(f"Discord bot logged in as {self.user} (ID: {self.user.id})")
        self.is_ready = True
        
        # Initialize channels
        await self._initialize_channels()
        
        # Send startup notification
        await self._send_startup_notification()
    
    async def on_error(self, event, *args, **kwargs):
        """Handle Discord client errors."""
        self.logger.error(f"Discord client error in event {event}", exc_info=True)
    
    async def on_disconnect(self):
        """Handle Discord disconnect."""
        self.logger.warning("Discord bot disconnected")
        self.is_ready = False
    
    async def on_resumed(self):
        """Handle Discord reconnection."""
        self.logger.info("Discord bot reconnected")
        self.is_ready = True
    
    async def _initialize_channels(self):
        """Initialize Discord channels for notifications."""
        try:
            # Get channel IDs from config
            channel_ids = self.config.discord.get('channel_ids', {})
            
            for channel_type, channel_id in channel_ids.items():
                if channel_id:
                    channel = self.get_channel(int(channel_id))
                    if channel:
                        self.channels[channel_type] = channel
                        self.logger.info(f"Initialized {channel_type} channel: {channel.name}")
                    else:
                        self.logger.warning(f"Could not find {channel_type} channel with ID {channel_id}")
        
        except Exception as e:
            self.logger.error(f"Error initializing channels: {e}")
    
    async def _send_startup_notification(self):
        """Send a notification when the bot starts up."""
        if self.channels.get('alerts'):
            embed = discord.Embed(
                title="🚀 Trading Bot Started",
                description="ICT Trading Bot has successfully connected to Discord",
                color=0x00ff00,
                timestamp=datetime.utcnow()
            )
            embed.add_field(name="Bot User", value=str(self.user), inline=True)
            embed.add_field(name="Startup Time", value=self.start_time.strftime("%Y-%m-%d %H:%M:%S UTC"), inline=True)
            
            try:
                await self.channels['alerts'].send(embed=embed)
            except Exception as e:
                self.logger.error(f"Failed to send startup notification: {e}")
    
    async def send_message(self, channel_type: str, content: str = None, embed: discord.Embed = None) -> bool:
        """
        Send a message to a specific channel type.
        
        Args:
            channel_type: Type of channel ('trading', 'alerts', 'reports', 'debug')
            content: Text content of the message
            embed: Discord embed object
            
        Returns:
            bool: True if message was sent successfully, False otherwise
        """
        if not self.is_ready:
            self.logger.warning("Discord bot is not ready, cannot send message")
            return False
        
        channel = self.channels.get(channel_type)
        if not channel:
            self.logger.warning(f"No channel configured for type: {channel_type}")
            return False
        
        try:
            await channel.send(content=content, embed=embed)
            return True
        except Exception as e:
            self.logger.error(f"Failed to send message to {channel_type} channel: {e}")
            return False
    
    async def get_channel_info(self) -> dict:
        """Get information about configured channels."""
        info = {}
        for channel_type, channel in self.channels.items():
            if channel:
                info[channel_type] = {
                    'name': channel.name,
                    'id': channel.id,
                    'guild': channel.guild.name if hasattr(channel, 'guild') else None
                }
            else:
                info[channel_type] = None
        return info


class DiscordBotManager:
    """
    Manager class for the Discord bot client.
    
    Handles bot lifecycle, connection management, and provides
    a high-level interface for bot operations.
    """
    
    def __init__(self, config: BotConfig):
        """
        Initialize the Discord bot manager.
        
        Args:
            config: Bot configuration
        """
        self.config = config
        self.client: Optional[TradingBotDiscordClient] = None
        self.logger = logging.getLogger(__name__)
        self._running = False
    
    async def start(self) -> bool:
        """
        Start the Discord bot.
        
        Returns:
            bool: True if bot started successfully, False otherwise
        """
        if self._running:
            self.logger.warning("Discord bot is already running")
            return True
        
        try:
            # Validate configuration
            if not self._validate_config():
                return False
            
            # Create client instance
            self.client = TradingBotDiscordClient(self.config)
            
            # Start the bot
            token = self.config.discord.get('token')
            if not token:
                self.logger.error("Discord bot token not found in configuration")
                return False
            
            # Start the bot in a background task
            asyncio.create_task(self.client.start(token))
            
            # Wait for bot to be ready
            max_wait = 30  # seconds
            wait_time = 0
            while not self.client.is_ready and wait_time < max_wait:
                await asyncio.sleep(1)
                wait_time += 1
            
            if self.client.is_ready:
                self._running = True
                self.logger.info("Discord bot started successfully")
                return True
            else:
                self.logger.error("Discord bot failed to start within timeout period")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to start Discord bot: {e}")
            return False
    
    async def stop(self):
        """Stop the Discord bot."""
        if not self._running:
            return
        
        try:
            if self.client:
                await self.client.close()
            self._running = False
            self.logger.info("Discord bot stopped")
        except Exception as e:
            self.logger.error(f"Error stopping Discord bot: {e}")
    
    def is_running(self) -> bool:
        """Check if the bot is running."""
        return self._running and self.client and self.client.is_ready
    
    def get_client(self) -> Optional[TradingBotDiscordClient]:
        """Get the Discord client instance."""
        return self.client if self.is_running() else None
    
    def _validate_config(self) -> bool:
        """Validate Discord configuration."""
        if not self.config.discord:
            self.logger.error("Discord configuration not found")
            return False
        
        if not self.config.discord.get('token'):
            self.logger.error("Discord bot token not configured")
            return False
        
        return True
    
    async def test_connection(self) -> dict:
        """
        Test the Discord bot connection and return status information.
        
        Returns:
            dict: Connection status and information
        """
        status = {
            'connected': False,
            'user': None,
            'guilds': [],
            'channels': {},
            'latency': None,
            'uptime': None
        }
        
        if self.is_running() and self.client:
            status['connected'] = True
            status['user'] = str(self.client.user)
            status['guilds'] = [guild.name for guild in self.client.guilds]
            status['channels'] = await self.client.get_channel_info()
            status['latency'] = round(self.client.latency * 1000, 2)  # ms
            
            if self.client.start_time:
                uptime = datetime.utcnow() - self.client.start_time
                status['uptime'] = str(uptime)
        
        return status