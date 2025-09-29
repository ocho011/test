"""
DiscordFormatter for ICT Trading System

This module implements comprehensive Discord message formatting with rich embeds,
color coding, chart image attachments, and professional trading notification templates.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum
import discord
from io import BytesIO
import base64

from .notification_manager import NotificationData, NotificationType, NotificationPriority


class MessageColor(Enum):
    """Color scheme for different types of messages."""
    
    # Trading colors
    BUY_GREEN = 0x00ff00
    SELL_RED = 0xff0000
    PROFIT_GREEN = 0x32cd32
    LOSS_RED = 0xdc143c
    NEUTRAL_BLUE = 0x1e90ff
    
    # System colors
    INFO_BLUE = 0x3498db
    WARNING_ORANGE = 0xf39c12
    ERROR_RED = 0xe74c3c
    SUCCESS_GREEN = 0x2ecc71
    
    # Report colors
    DAILY_PURPLE = 0x9b59b6
    WEEKLY_INDIGO = 0x6c5ce7
    MONTHLY_GOLD = 0xf1c40f


class EmojiSet:
    """Emoji constants for consistent formatting."""
    
    # Trading emojis
    TRADE_ENTRY = "📈"
    TRADE_EXIT = "📉"
    PROFIT = "💰"
    LOSS = "💸"
    SIGNAL = "🎯"
    
    # System emojis
    INFO = "ℹ️"
    WARNING = "⚠️"
    ERROR = "❌"
    SUCCESS = "✅"
    ROBOT = "🤖"
    
    # Reporting emojis
    CHART = "📊"
    CALENDAR = "📅"
    CLOCK = "🕐"
    MONEY = "💵"
    PERCENTAGE = "📈"
    
    # Directional emojis
    UP_ARROW = "⬆️"
    DOWN_ARROW = "⬇️"
    RIGHT_ARROW = "➡️"
    
    # Status emojis
    ONLINE = "🟢"
    OFFLINE = "🔴"
    CONNECTING = "🟡"


class DiscordFormatter:
    """
    Advanced Discord message formatter for trading notifications.
    
    Provides rich embed creation, color coding, emoji integration,
    and specialized formatting for different notification types.
    """
    
    def __init__(self):
        """Initialize the Discord formatter."""
        self.logger = logging.getLogger(__name__)
        self.max_embed_length = 6000  # Discord limit
        self.max_field_length = 1024  # Discord field limit
        self.max_title_length = 256   # Discord title limit
        self.max_description_length = 4096  # Discord description limit
    
    def format_notification(self, notification: NotificationData, 
                          include_chart: bool = False,
                          chart_data: Optional[bytes] = None) -> Dict[str, Any]:
        """
        Format a notification into Discord message components.
        
        Args:
            notification: Notification data to format
            include_chart: Whether to include chart image
            chart_data: Chart image data as bytes
            
        Returns:
            dict: Dictionary containing 'content', 'embed', and optional 'file'
        """
        try:
            # Route to specific formatters based on notification type
            if notification.type in [NotificationType.TRADE_ENTRY, NotificationType.TRADE_EXIT]:
                return self._format_trade_notification(notification, include_chart, chart_data)
            elif notification.type == NotificationType.SIGNAL_GENERATED:
                return self._format_signal_notification(notification, include_chart, chart_data)
            elif notification.type in [NotificationType.SYSTEM_ERROR, NotificationType.SYSTEM_WARNING]:
                return self._format_system_notification(notification)
            elif notification.type in [NotificationType.DAILY_REPORT, NotificationType.WEEKLY_REPORT, NotificationType.MONTHLY_REPORT]:
                return self._format_report_notification(notification)
            elif notification.type in [NotificationType.ORDER_FILLED, NotificationType.ORDER_CANCELLED]:
                return self._format_order_notification(notification)
            else:
                return self._format_generic_notification(notification)
                
        except Exception as e:
            self.logger.error(f"Error formatting notification {notification.id}: {e}")
            return self._format_error_fallback(notification)
    
    def _format_trade_notification(self, notification: NotificationData, 
                                 include_chart: bool = False,
                                 chart_data: Optional[bytes] = None) -> Dict[str, Any]:
        """Format trading entry/exit notifications."""
        data = notification.data
        is_entry = notification.type == NotificationType.TRADE_ENTRY
        
        # Determine colors and emojis
        side = data.get('side', '').lower()
        if is_entry:
            color = MessageColor.BUY_GREEN.value if side == 'buy' else MessageColor.SELL_RED.value
            emoji = EmojiSet.TRADE_ENTRY
        else:
            pnl = data.get('pnl', 0)
            color = MessageColor.PROFIT_GREEN.value if pnl >= 0 else MessageColor.LOSS_RED.value
            emoji = EmojiSet.TRADE_EXIT
        
        # Create embed
        embed = discord.Embed(
            title=f"{emoji} {notification.title}",
            description=self._truncate_text(notification.message, self.max_description_length),
            color=color,
            timestamp=notification.timestamp
        )
        
        # Add trading details
        symbol = data.get('symbol', 'Unknown')
        quantity = data.get('quantity', 0)
        price = data.get('price', 0)
        
        embed.add_field(name="Symbol", value=f"`{symbol}`", inline=True)
        embed.add_field(name="Side", value=f"`{side.upper()}`", inline=True)
        embed.add_field(name="Quantity", value=f"`{quantity:,.4f}`", inline=True)
        embed.add_field(name="Price", value=f"`${price:,.4f}`", inline=True)
        
        # Add PnL for exits
        if not is_entry and 'pnl' in data:
            pnl = data['pnl']
            pnl_emoji = EmojiSet.PROFIT if pnl >= 0 else EmojiSet.LOSS
            embed.add_field(
                name="P&L", 
                value=f"{pnl_emoji} `${pnl:,.2f}`", 
                inline=True
            )
        
        # Add additional trading info
        if 'entry_price' in data:
            embed.add_field(name="Entry Price", value=f"`${data['entry_price']:,.4f}`", inline=True)
        
        if 'stop_loss' in data:
            embed.add_field(name="Stop Loss", value=f"`${data['stop_loss']:,.4f}`", inline=True)
        
        if 'take_profit' in data:
            embed.add_field(name="Take Profit", value=f"`${data['take_profit']:,.4f}`", inline=True)
        
        # Add footer
        embed.set_footer(text=f"Notification ID: {notification.id}")
        
        result = {"embed": embed}
        
        # Add chart if provided
        if include_chart and chart_data:
            try:
                file = discord.File(BytesIO(chart_data), filename=f"chart_{symbol}.png")
                embed.set_image(url=f"attachment://chart_{symbol}.png")
                result["file"] = file
            except Exception as e:
                self.logger.warning(f"Failed to attach chart: {e}")
        
        return result
    
    def _format_signal_notification(self, notification: NotificationData,
                                  include_chart: bool = False,
                                  chart_data: Optional[bytes] = None) -> Dict[str, Any]:
        """Format signal generation notifications."""
        data = notification.data
        
        # Create embed
        embed = discord.Embed(
            title=f"{EmojiSet.SIGNAL} {notification.title}",
            description=self._truncate_text(notification.message, self.max_description_length),
            color=MessageColor.NEUTRAL_BLUE.value,
            timestamp=notification.timestamp
        )
        
        # Add signal details
        symbol = data.get('symbol', 'Unknown')
        signal_type = data.get('signal_type', 'Unknown')
        confidence = data.get('confidence', 0)
        strength = data.get('strength', 0)
        
        embed.add_field(name="Symbol", value=f"`{symbol}`", inline=True)
        embed.add_field(name="Signal Type", value=f"`{signal_type}`", inline=True)
        embed.add_field(name="Confidence", value=f"`{confidence:.1%}`", inline=True)
        
        if strength:
            embed.add_field(name="Strength", value=f"`{strength:.2f}`", inline=True)
        
        # Add signal-specific data
        if 'price' in data:
            embed.add_field(name="Current Price", value=f"`${data['price']:,.4f}`", inline=True)
        
        if 'target_price' in data:
            embed.add_field(name="Target Price", value=f"`${data['target_price']:,.4f}`", inline=True)
        
        if 'stop_loss' in data:
            embed.add_field(name="Suggested SL", value=f"`${data['stop_loss']:,.4f}`", inline=True)
        
        # Add timeframe if available
        if 'timeframe' in data:
            embed.add_field(name="Timeframe", value=f"`{data['timeframe']}`", inline=True)
        
        embed.set_footer(text=f"Signal ID: {data.get('signal_id', notification.id)}")
        
        result = {"embed": embed}
        
        # Add chart if provided
        if include_chart and chart_data:
            try:
                file = discord.File(BytesIO(chart_data), filename=f"signal_{symbol}.png")
                embed.set_image(url=f"attachment://signal_{symbol}.png")
                result["file"] = file
            except Exception as e:
                self.logger.warning(f"Failed to attach chart: {e}")
        
        return result
    
    def _format_system_notification(self, notification: NotificationData) -> Dict[str, Any]:
        """Format system error/warning notifications."""
        is_error = notification.type == NotificationType.SYSTEM_ERROR
        
        color = MessageColor.ERROR_RED.value if is_error else MessageColor.WARNING_ORANGE.value
        emoji = EmojiSet.ERROR if is_error else EmojiSet.WARNING
        
        embed = discord.Embed(
            title=f"{emoji} {notification.title}",
            description=self._truncate_text(notification.message, self.max_description_length),
            color=color,
            timestamp=notification.timestamp
        )
        
        # Add system details
        if 'component' in notification.data:
            embed.add_field(name="Component", value=f"`{notification.data['component']}`", inline=True)
        
        if 'error_code' in notification.data:
            embed.add_field(name="Error Code", value=f"`{notification.data['error_code']}`", inline=True)
        
        if 'level' in notification.data:
            embed.add_field(name="Level", value=f"`{notification.data['level'].upper()}`", inline=True)
        
        # Add stack trace or additional details
        if 'details' in notification.data:
            details = str(notification.data['details'])
            if len(details) > 500:
                details = details[:500] + "..."
            embed.add_field(name="Details", value=f"```\n{details}\n```", inline=False)
        
        embed.set_footer(text=f"Notification ID: {notification.id}")
        
        return {"embed": embed}
    
    def _format_report_notification(self, notification: NotificationData) -> Dict[str, Any]:
        """Format performance report notifications."""
        data = notification.data
        
        # Determine color based on report type
        if notification.type == NotificationType.DAILY_REPORT:
            color = MessageColor.DAILY_PURPLE.value
            emoji = f"{EmojiSet.CALENDAR} Daily"
        elif notification.type == NotificationType.WEEKLY_REPORT:
            color = MessageColor.WEEKLY_INDIGO.value
            emoji = f"{EmojiSet.CALENDAR} Weekly"
        else:
            color = MessageColor.MONTHLY_GOLD.value
            emoji = f"{EmojiSet.CALENDAR} Monthly"
        
        embed = discord.Embed(
            title=f"{emoji} {notification.title}",
            description=self._truncate_text(notification.message, self.max_description_length),
            color=color,
            timestamp=notification.timestamp
        )
        
        # Add performance metrics
        if 'total_pnl' in data:
            pnl = data['total_pnl']
            pnl_emoji = EmojiSet.PROFIT if pnl >= 0 else EmojiSet.LOSS
            embed.add_field(name="Total P&L", value=f"{pnl_emoji} `${pnl:,.2f}`", inline=True)
        
        if 'total_trades' in data:
            embed.add_field(name="Total Trades", value=f"`{data['total_trades']}`", inline=True)
        
        if 'win_rate' in data:
            embed.add_field(name="Win Rate", value=f"`{data['win_rate']:.1%}`", inline=True)
        
        if 'profit_factor' in data:
            embed.add_field(name="Profit Factor", value=f"`{data['profit_factor']:.2f}`", inline=True)
        
        if 'sharpe_ratio' in data:
            embed.add_field(name="Sharpe Ratio", value=f"`{data['sharpe_ratio']:.2f}`", inline=True)
        
        if 'max_drawdown' in data:
            embed.add_field(name="Max Drawdown", value=f"`{data['max_drawdown']:.1%}`", inline=True)
        
        # Add top performing symbols
        if 'top_symbols' in data:
            top_symbols = data['top_symbols'][:5]  # Limit to top 5
            symbols_text = "\n".join([f"`{symbol}: ${pnl:,.2f}`" for symbol, pnl in top_symbols])
            embed.add_field(name="Top Performers", value=symbols_text, inline=False)
        
        # Add period information
        if 'period_start' in data and 'period_end' in data:
            period_text = f"{data['period_start']} to {data['period_end']}"
            embed.add_field(name="Period", value=f"`{period_text}`", inline=False)
        
        embed.set_footer(text=f"Report generated at {notification.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        
        return {"embed": embed}
    
    def _format_order_notification(self, notification: NotificationData) -> Dict[str, Any]:
        """Format order filled/cancelled notifications."""
        data = notification.data
        is_filled = notification.type == NotificationType.ORDER_FILLED
        
        color = MessageColor.SUCCESS_GREEN.value if is_filled else MessageColor.WARNING_ORANGE.value
        emoji = EmojiSet.SUCCESS if is_filled else EmojiSet.WARNING
        
        embed = discord.Embed(
            title=f"{emoji} {notification.title}",
            description=self._truncate_text(notification.message, self.max_description_length),
            color=color,
            timestamp=notification.timestamp
        )
        
        # Add order details
        if 'symbol' in data:
            embed.add_field(name="Symbol", value=f"`{data['symbol']}`", inline=True)
        
        if 'order_type' in data:
            embed.add_field(name="Order Type", value=f"`{data['order_type']}`", inline=True)
        
        if 'side' in data:
            embed.add_field(name="Side", value=f"`{data['side'].upper()}`", inline=True)
        
        if 'quantity' in data:
            embed.add_field(name="Quantity", value=f"`{data['quantity']:,.4f}`", inline=True)
        
        if 'price' in data:
            embed.add_field(name="Price", value=f"`${data['price']:,.4f}`", inline=True)
        
        if 'order_id' in data:
            embed.add_field(name="Order ID", value=f"`{data['order_id']}`", inline=True)
        
        # Add reason for cancellation
        if not is_filled and 'reason' in data:
            embed.add_field(name="Cancellation Reason", value=f"`{data['reason']}`", inline=False)
        
        embed.set_footer(text=f"Notification ID: {notification.id}")
        
        return {"embed": embed}
    
    def _format_generic_notification(self, notification: NotificationData) -> Dict[str, Any]:
        """Format generic notifications that don't have specialized formatting."""
        # Choose color based on priority
        color_map = {
            NotificationPriority.LOW: MessageColor.INFO_BLUE.value,
            NotificationPriority.NORMAL: MessageColor.NEUTRAL_BLUE.value,
            NotificationPriority.HIGH: MessageColor.WARNING_ORANGE.value,
            NotificationPriority.CRITICAL: MessageColor.ERROR_RED.value,
            NotificationPriority.URGENT: MessageColor.ERROR_RED.value
        }
        
        color = color_map.get(notification.priority, MessageColor.INFO_BLUE.value)
        
        embed = discord.Embed(
            title=f"{EmojiSet.INFO} {notification.title}",
            description=self._truncate_text(notification.message, self.max_description_length),
            color=color,
            timestamp=notification.timestamp
        )
        
        # Add any additional data fields
        for key, value in notification.data.items():
            if isinstance(value, (str, int, float)):
                embed.add_field(
                    name=key.replace('_', ' ').title(),
                    value=f"`{value}`",
                    inline=True
                )
        
        embed.add_field(name="Priority", value=f"`{notification.priority.name}`", inline=True)
        embed.set_footer(text=f"Notification ID: {notification.id}")
        
        return {"embed": embed}
    
    def _format_error_fallback(self, notification: NotificationData) -> Dict[str, Any]:
        """Fallback formatter for when main formatting fails."""
        embed = discord.Embed(
            title=f"{EmojiSet.ERROR} Formatting Error",
            description=f"Failed to format notification properly.\n\nOriginal message: {notification.message}",
            color=MessageColor.ERROR_RED.value,
            timestamp=notification.timestamp
        )
        
        embed.add_field(name="Notification Type", value=f"`{notification.type.value}`", inline=True)
        embed.add_field(name="Priority", value=f"`{notification.priority.name}`", inline=True)
        embed.set_footer(text=f"Notification ID: {notification.id}")
        
        return {"embed": embed}
    
    def create_custom_embed(
        self,
        title: str,
        description: str,
        color: Union[int, MessageColor] = MessageColor.INFO_BLUE,
        fields: Optional[List[Dict[str, Union[str, bool]]]] = None,
        footer: Optional[str] = None,
        thumbnail_url: Optional[str] = None,
        image_url: Optional[str] = None
    ) -> discord.Embed:
        """
        Create a custom Discord embed.
        
        Args:
            title: Embed title
            description: Embed description
            color: Embed color (int or MessageColor enum)
            fields: List of field dictionaries with name, value, and inline keys
            footer: Footer text
            thumbnail_url: Thumbnail image URL
            image_url: Main image URL
            
        Returns:
            discord.Embed: Configured embed object
        """
        if isinstance(color, MessageColor):
            color = color.value
        
        embed = discord.Embed(
            title=self._truncate_text(title, self.max_title_length),
            description=self._truncate_text(description, self.max_description_length),
            color=color,
            timestamp=datetime.utcnow()
        )
        
        if fields:
            for field in fields:
                embed.add_field(
                    name=field.get('name', 'Field'),
                    value=self._truncate_text(str(field.get('value', '')), self.max_field_length),
                    inline=field.get('inline', True)
                )
        
        if footer:
            embed.set_footer(text=self._truncate_text(footer, 2048))
        
        if thumbnail_url:
            embed.set_thumbnail(url=thumbnail_url)
        
        if image_url:
            embed.set_image(url=image_url)
        
        return embed
    
    def _truncate_text(self, text: str, max_length: int) -> str:
        """Truncate text to fit Discord limits."""
        if len(text) <= max_length:
            return text
        return text[:max_length - 3] + "..."
    
    def format_table(self, headers: List[str], rows: List[List[str]], 
                    max_width: int = 1000) -> str:
        """
        Format data as a table using code blocks.
        
        Args:
            headers: List of column headers
            rows: List of rows (each row is a list of strings)
            max_width: Maximum width for the table
            
        Returns:
            str: Formatted table string
        """
        if not headers or not rows:
            return "```\nNo data to display\n```"
        
        # Calculate column widths
        col_widths = [len(header) for header in headers]
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(str(cell)))
        
        # Create format string
        total_width = sum(col_widths) + len(headers) * 3 - 1
        if total_width > max_width:
            # Adjust column widths proportionally
            scale_factor = (max_width - len(headers) * 3 + 1) / total_width
            col_widths = [max(5, int(w * scale_factor)) for w in col_widths]
        
        # Build table
        lines = []
        
        # Header
        header_line = " | ".join(f"{header:<{col_widths[i]}}" for i, header in enumerate(headers))
        lines.append(header_line)
        lines.append("-" * len(header_line))
        
        # Rows
        for row in rows:
            row_line = " | ".join(
                f"{str(cell):<{col_widths[i]}}"[:col_widths[i]] 
                for i, cell in enumerate(row)
            )
            lines.append(row_line)
        
        return f"```\n{chr(10).join(lines)}\n```"
    
    def format_code_block(self, code: str, language: str = "") -> str:
        """Format code with syntax highlighting."""
        return f"```{language}\n{code}\n```"
    
    def format_inline_code(self, code: str) -> str:
        """Format inline code."""
        return f"`{code}`"
    
    def format_progress_bar(self, current: float, total: float, width: int = 20) -> str:
        """Create a text-based progress bar."""
        if total == 0:
            percentage = 0
        else:
            percentage = min(100, max(0, (current / total) * 100))
        
        filled = int((percentage / 100) * width)
        empty = width - filled
        
        bar = "█" * filled + "░" * empty
        return f"[{bar}] {percentage:.1f}%"