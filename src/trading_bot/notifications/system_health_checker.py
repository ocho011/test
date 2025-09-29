"""
SystemHealthChecker and PerformanceReporter for ICT Trading System

This module implements comprehensive system health monitoring and automated
performance reporting capabilities for the trading bot.
"""

import asyncio
import logging
import psutil
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import json

from .notification_manager import NotificationManager, NotificationData, NotificationType, NotificationPriority


class HealthStatus(Enum):
    """System health status levels."""
    
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    OFFLINE = "offline"


class ComponentType(Enum):
    """Types of system components to monitor."""
    
    DISCORD_BOT = "discord_bot"
    EXCHANGE_API = "exchange_api"
    SIGNAL_GENERATOR = "signal_generator"
    ORDER_EXECUTOR = "order_executor"
    DATABASE = "database"
    MEMORY = "memory"
    CPU = "cpu"
    DISK = "disk"
    NETWORK = "network"


@dataclass
class HealthMetric:
    """Individual health metric data."""
    
    component: ComponentType
    name: str
    value: float
    unit: str
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    status: HealthStatus = HealthStatus.HEALTHY
    message: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def evaluate_status(self) -> HealthStatus:
        """Evaluate health status based on thresholds."""
        if self.threshold_critical and self.value >= self.threshold_critical:
            self.status = HealthStatus.CRITICAL
        elif self.threshold_warning and self.value >= self.threshold_warning:
            self.status = HealthStatus.WARNING
        else:
            self.status = HealthStatus.HEALTHY
        
        return self.status


@dataclass
class ComponentHealth:
    """Health status for a system component."""
    
    component: ComponentType
    status: HealthStatus
    metrics: List[HealthMetric]
    last_check: datetime
    error_message: Optional[str] = None
    uptime: Optional[float] = None  # seconds
    
    def get_overall_status(self) -> HealthStatus:
        """Get overall component status from metrics."""
        if not self.metrics:
            return HealthStatus.OFFLINE
        
        statuses = [metric.status for metric in self.metrics]
        
        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.WARNING in statuses:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY


class SystemHealthChecker:
    """
    Comprehensive system health monitoring.
    
    Monitors various system components including Discord bot connection,
    exchange API status, memory usage, CPU usage, and other critical metrics.
    """
    
    def __init__(self, notification_manager: NotificationManager):
        """
        Initialize the system health checker.
        
        Args:
            notification_manager: NotificationManager instance for alerts
        """
        self.notification_manager = notification_manager
        self.logger = logging.getLogger(__name__)
        
        # Health monitoring configuration
        self.check_interval = 60  # seconds
        self.component_checks = {}
        self.health_history = {}
        self.alert_cooldowns = {}  # Prevent alert spam
        self.cooldown_period = 300  # 5 minutes
        
        # System resource thresholds
        self.memory_warning_threshold = 80.0  # %
        self.memory_critical_threshold = 95.0  # %
        self.cpu_warning_threshold = 80.0  # %
        self.cpu_critical_threshold = 95.0  # %
        self.disk_warning_threshold = 85.0  # %
        self.disk_critical_threshold = 95.0  # %
        
        self._running = False
        self._monitoring_task: Optional[asyncio.Task] = None
        self.start_time = datetime.utcnow()
    
    async def start_monitoring(self):
        """Start the health monitoring loop."""
        if self._running:
            self.logger.warning("Health monitoring is already running")
            return
        
        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("System health monitoring started")
    
    async def stop_monitoring(self):
        """Stop the health monitoring."""
        if not self._running:
            return
        
        self._running = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("System health monitoring stopped")
    
    async def check_all_components(self) -> Dict[ComponentType, ComponentHealth]:
        """
        Check health of all system components.
        
        Returns:
            dict: Component health status for each component
        """
        health_results = {}
        
        # Check each component type
        for component_type in ComponentType:
            try:
                health = await self._check_component(component_type)
                health_results[component_type] = health
                
                # Send alerts if needed
                await self._process_health_alerts(health)
                
            except Exception as e:
                self.logger.error(f"Error checking {component_type.value}: {e}")
                health_results[component_type] = ComponentHealth(
                    component=component_type,
                    status=HealthStatus.OFFLINE,
                    metrics=[],
                    last_check=datetime.utcnow(),
                    error_message=str(e)
                )
        
        return health_results
    
    async def _check_component(self, component: ComponentType) -> ComponentHealth:
        """Check health of a specific component."""
        if component == ComponentType.DISCORD_BOT:
            return await self._check_discord_bot()
        elif component == ComponentType.EXCHANGE_API:
            return await self._check_exchange_api()
        elif component == ComponentType.MEMORY:
            return self._check_memory()
        elif component == ComponentType.CPU:
            return self._check_cpu()
        elif component == ComponentType.DISK:
            return self._check_disk()
        elif component == ComponentType.NETWORK:
            return self._check_network()
        else:
            # Generic component check
            return ComponentHealth(
                component=component,
                status=HealthStatus.HEALTHY,
                metrics=[],
                last_check=datetime.utcnow()
            )
    
    async def _check_discord_bot(self) -> ComponentHealth:
        """Check Discord bot health."""
        metrics = []
        status = HealthStatus.HEALTHY
        error_message = None
        
        try:
            # Check if Discord bot is running
            discord_manager = getattr(self.notification_manager, 'discord_manager', None)
            if not discord_manager:
                status = HealthStatus.OFFLINE
                error_message = "Discord manager not available"
            elif not discord_manager.is_running():
                status = HealthStatus.CRITICAL
                error_message = "Discord bot is not running"
            else:
                # Get connection info
                connection_info = await discord_manager.test_connection()
                
                if connection_info.get('connected', False):
                    # Check latency
                    latency = connection_info.get('latency', 0)
                    latency_metric = HealthMetric(
                        component=ComponentType.DISCORD_BOT,
                        name="latency",
                        value=latency,
                        unit="ms",
                        threshold_warning=500.0,
                        threshold_critical=1000.0
                    )
                    latency_metric.evaluate_status()
                    metrics.append(latency_metric)
                    
                    # Check guild count
                    guild_count = len(connection_info.get('guilds', []))
                    guild_metric = HealthMetric(
                        component=ComponentType.DISCORD_BOT,
                        name="guild_count",
                        value=guild_count,
                        unit="count"
                    )
                    metrics.append(guild_metric)
                    
                    # Overall status from metrics
                    status = max([m.status for m in metrics], key=lambda x: x.value)
                else:
                    status = HealthStatus.CRITICAL
                    error_message = "Discord bot not connected"
        
        except Exception as e:
            status = HealthStatus.CRITICAL
            error_message = f"Discord bot check failed: {e}"
        
        return ComponentHealth(
            component=ComponentType.DISCORD_BOT,
            status=status,
            metrics=metrics,
            last_check=datetime.utcnow(),
            error_message=error_message
        )
    
    async def _check_exchange_api(self) -> ComponentHealth:
        """Check exchange API connectivity."""
        # This is a placeholder - would need to integrate with actual exchange client
        metrics = []
        
        # Simulate API response time check
        api_response_metric = HealthMetric(
            component=ComponentType.EXCHANGE_API,
            name="response_time",
            value=150.0,  # ms - would be actual measurement
            unit="ms",
            threshold_warning=500.0,
            threshold_critical=2000.0
        )
        api_response_metric.evaluate_status()
        metrics.append(api_response_metric)
        
        return ComponentHealth(
            component=ComponentType.EXCHANGE_API,
            status=api_response_metric.status,
            metrics=metrics,
            last_check=datetime.utcnow()
        )
    
    def _check_memory(self) -> ComponentHealth:
        """Check system memory usage."""
        metrics = []
        
        try:
            memory = psutil.virtual_memory()
            
            memory_metric = HealthMetric(
                component=ComponentType.MEMORY,
                name="usage_percent",
                value=memory.percent,
                unit="%",
                threshold_warning=self.memory_warning_threshold,
                threshold_critical=self.memory_critical_threshold
            )
            memory_metric.evaluate_status()
            metrics.append(memory_metric)
            
            # Available memory
            available_gb = memory.available / (1024**3)
            available_metric = HealthMetric(
                component=ComponentType.MEMORY,
                name="available_gb",
                value=available_gb,
                unit="GB"
            )
            metrics.append(available_metric)
            
        except Exception as e:
            self.logger.error(f"Memory check failed: {e}")
            return ComponentHealth(
                component=ComponentType.MEMORY,
                status=HealthStatus.OFFLINE,
                metrics=[],
                last_check=datetime.utcnow(),
                error_message=str(e)
            )
        
        overall_status = max([m.status for m in metrics], key=lambda x: x.value)
        
        return ComponentHealth(
            component=ComponentType.MEMORY,
            status=overall_status,
            metrics=metrics,
            last_check=datetime.utcnow()
        )
    
    def _check_cpu(self) -> ComponentHealth:
        """Check CPU usage."""
        metrics = []
        
        try:
            # Get CPU usage (averaged over 1 second)
            cpu_percent = psutil.cpu_percent(interval=1)
            
            cpu_metric = HealthMetric(
                component=ComponentType.CPU,
                name="usage_percent",
                value=cpu_percent,
                unit="%",
                threshold_warning=self.cpu_warning_threshold,
                threshold_critical=self.cpu_critical_threshold
            )
            cpu_metric.evaluate_status()
            metrics.append(cpu_metric)
            
            # CPU count
            cpu_count_metric = HealthMetric(
                component=ComponentType.CPU,
                name="core_count",
                value=psutil.cpu_count(),
                unit="cores"
            )
            metrics.append(cpu_count_metric)
            
        except Exception as e:
            self.logger.error(f"CPU check failed: {e}")
            return ComponentHealth(
                component=ComponentType.CPU,
                status=HealthStatus.OFFLINE,
                metrics=[],
                last_check=datetime.utcnow(),
                error_message=str(e)
            )
        
        overall_status = max([m.status for m in metrics], key=lambda x: x.value)
        
        return ComponentHealth(
            component=ComponentType.CPU,
            status=overall_status,
            metrics=metrics,
            last_check=datetime.utcnow()
        )
    
    def _check_disk(self) -> ComponentHealth:
        """Check disk usage."""
        metrics = []
        
        try:
            disk = psutil.disk_usage('/')
            
            disk_percent = (disk.used / disk.total) * 100
            disk_metric = HealthMetric(
                component=ComponentType.DISK,
                name="usage_percent",
                value=disk_percent,
                unit="%",
                threshold_warning=self.disk_warning_threshold,
                threshold_critical=self.disk_critical_threshold
            )
            disk_metric.evaluate_status()
            metrics.append(disk_metric)
            
            # Free space in GB
            free_gb = disk.free / (1024**3)
            free_metric = HealthMetric(
                component=ComponentType.DISK,
                name="free_gb",
                value=free_gb,
                unit="GB"
            )
            metrics.append(free_metric)
            
        except Exception as e:
            self.logger.error(f"Disk check failed: {e}")
            return ComponentHealth(
                component=ComponentType.DISK,
                status=HealthStatus.OFFLINE,
                metrics=[],
                last_check=datetime.utcnow(),
                error_message=str(e)
            )
        
        overall_status = max([m.status for m in metrics], key=lambda x: x.value)
        
        return ComponentHealth(
            component=ComponentType.DISK,
            status=overall_status,
            metrics=metrics,
            last_check=datetime.utcnow()
        )
    
    def _check_network(self) -> ComponentHealth:
        """Check network connectivity."""
        metrics = []
        
        try:
            # Get network I/O stats
            net_io = psutil.net_io_counters()
            
            # Calculate bytes per second (if we have previous measurement)
            current_time = time.time()
            if hasattr(self, '_last_net_check'):
                time_diff = current_time - self._last_net_check['time']
                bytes_sent_per_sec = (net_io.bytes_sent - self._last_net_check['bytes_sent']) / time_diff
                bytes_recv_per_sec = (net_io.bytes_recv - self._last_net_check['bytes_recv']) / time_diff
                
                # Convert to MB/s
                sent_mbps = bytes_sent_per_sec / (1024**2)
                recv_mbps = bytes_recv_per_sec / (1024**2)
                
                sent_metric = HealthMetric(
                    component=ComponentType.NETWORK,
                    name="sent_mbps",
                    value=sent_mbps,
                    unit="MB/s"
                )
                metrics.append(sent_metric)
                
                recv_metric = HealthMetric(
                    component=ComponentType.NETWORK,
                    name="recv_mbps",
                    value=recv_mbps,
                    unit="MB/s"
                )
                metrics.append(recv_metric)
            
            # Store current values for next check
            self._last_net_check = {
                'time': current_time,
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv
            }
            
        except Exception as e:
            self.logger.error(f"Network check failed: {e}")
            return ComponentHealth(
                component=ComponentType.NETWORK,
                status=HealthStatus.OFFLINE,
                metrics=[],
                last_check=datetime.utcnow(),
                error_message=str(e)
            )
        
        return ComponentHealth(
            component=ComponentType.NETWORK,
            status=HealthStatus.HEALTHY,
            metrics=metrics,
            last_check=datetime.utcnow()
        )
    
    async def _process_health_alerts(self, health: ComponentHealth):
        """Process health status and send alerts if needed."""
        component = health.component
        current_time = datetime.utcnow()
        
        # Check if we're in cooldown period
        last_alert = self.alert_cooldowns.get(component)
        if last_alert and (current_time - last_alert).total_seconds() < self.cooldown_period:
            return
        
        # Send alert for critical or warning status
        if health.status in [HealthStatus.CRITICAL, HealthStatus.WARNING]:
            level = "error" if health.status == HealthStatus.CRITICAL else "warning"
            
            title = f"System Health Alert: {component.value.replace('_', ' ').title()}"
            message = f"Component status: {health.status.value.upper()}"
            
            if health.error_message:
                message += f"\nError: {health.error_message}"
            
            # Add metric details
            if health.metrics:
                problematic_metrics = [m for m in health.metrics if m.status != HealthStatus.HEALTHY]
                if problematic_metrics:
                    message += "\n\nProblematic metrics:"
                    for metric in problematic_metrics:
                        message += f"\n- {metric.name}: {metric.value}{metric.unit}"
            
            notification = self.notification_manager.create_system_notification(
                level=level,
                title=title,
                message=message,
                additional_data={
                    'component': component.value,
                    'status': health.status.value,
                    'metrics': [
                        {
                            'name': m.name,
                            'value': m.value,
                            'unit': m.unit,
                            'status': m.status.value
                        } for m in health.metrics
                    ]
                }
            )
            
            self.notification_manager.queue_notification(notification)
            self.alert_cooldowns[component] = current_time
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                # Perform health checks
                health_results = await self.check_all_components()
                
                # Store health history
                timestamp = datetime.utcnow()
                self.health_history[timestamp] = health_results
                
                # Clean old history (keep last 24 hours)
                cutoff = timestamp - timedelta(hours=24)
                self.health_history = {
                    ts: data for ts, data in self.health_history.items()
                    if ts > cutoff
                }
                
                # Wait for next check
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in health monitoring loop: {e}", exc_info=True)
                await asyncio.sleep(self.check_interval)
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get current health summary."""
        if not self.health_history:
            return {
                'overall_status': 'unknown',
                'components': {},
                'last_check': None,
                'uptime': (datetime.utcnow() - self.start_time).total_seconds()
            }
        
        latest_check = max(self.health_history.keys())
        latest_health = self.health_history[latest_check]
        
        # Determine overall status
        statuses = [health.status for health in latest_health.values()]
        if HealthStatus.CRITICAL in statuses:
            overall_status = 'critical'
        elif HealthStatus.WARNING in statuses:
            overall_status = 'warning'
        elif HealthStatus.OFFLINE in statuses:
            overall_status = 'degraded'
        else:
            overall_status = 'healthy'
        
        return {
            'overall_status': overall_status,
            'components': {
                comp.value: {
                    'status': health.status.value,
                    'last_check': health.last_check.isoformat(),
                    'metrics_count': len(health.metrics),
                    'error_message': health.error_message
                }
                for comp, health in latest_health.items()
            },
            'last_check': latest_check.isoformat(),
            'uptime': (datetime.utcnow() - self.start_time).total_seconds(),
            'monitoring_active': self._running
        }


@dataclass
class PerformanceMetrics:
    """Performance metrics for a specific period."""
    
    period_start: datetime
    period_end: datetime
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    max_drawdown: float = 0.0
    sharpe_ratio: Optional[float] = None
    profit_factor: Optional[float] = None
    win_rate: float = 0.0
    
    def calculate_derived_metrics(self):
        """Calculate derived performance metrics."""
        if self.total_trades > 0:
            self.win_rate = self.winning_trades / self.total_trades
        
        if self.winning_trades > 0:
            self.average_win = self.gross_profit / self.winning_trades
        
        if self.losing_trades > 0:
            self.average_loss = abs(self.gross_loss) / self.losing_trades
        
        if self.gross_loss != 0:
            self.profit_factor = self.gross_profit / abs(self.gross_loss)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'period_start': self.period_start.isoformat(),
            'period_end': self.period_end.isoformat(),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'total_pnl': round(self.total_pnl, 2),
            'gross_profit': round(self.gross_profit, 2),
            'gross_loss': round(self.gross_loss, 2),
            'largest_win': round(self.largest_win, 2),
            'largest_loss': round(self.largest_loss, 2),
            'average_win': round(self.average_win, 2),
            'average_loss': round(self.average_loss, 2),
            'max_consecutive_wins': self.max_consecutive_wins,
            'max_consecutive_losses': self.max_consecutive_losses,
            'max_drawdown': round(self.max_drawdown, 4),
            'sharpe_ratio': round(self.sharpe_ratio, 3) if self.sharpe_ratio else None,
            'profit_factor': round(self.profit_factor, 3) if self.profit_factor else None,
            'win_rate': round(self.win_rate, 4)
        }


class PerformanceReporter:
    """
    Automated performance reporting system.
    
    Generates and sends daily, weekly, and monthly performance reports
    with comprehensive trading statistics and analysis.
    """
    
    def __init__(self, notification_manager: NotificationManager):
        """
        Initialize the performance reporter.
        
        Args:
            notification_manager: NotificationManager instance for sending reports
        """
        self.notification_manager = notification_manager
        self.logger = logging.getLogger(__name__)
        
        # Reporting configuration
        self.daily_report_time = "09:00"  # UTC
        self.weekly_report_day = 0  # Monday
        self.monthly_report_day = 1  # First day of month
        
        self._running = False
        self._reporting_task: Optional[asyncio.Task] = None
        
        # In a real implementation, this would connect to a trading database
        self.trade_history = []  # Placeholder for trade data
    
    async def start_reporting(self):
        """Start the automated reporting system."""
        if self._running:
            self.logger.warning("Performance reporting is already running")
            return
        
        self._running = True
        self._reporting_task = asyncio.create_task(self._reporting_loop())
        self.logger.info("Performance reporting started")
    
    async def stop_reporting(self):
        """Stop the automated reporting."""
        if not self._running:
            return
        
        self._running = False
        
        if self._reporting_task:
            self._reporting_task.cancel()
            try:
                await self._reporting_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Performance reporting stopped")
    
    async def generate_daily_report(self, date: Optional[datetime] = None) -> PerformanceMetrics:
        """Generate daily performance report."""
        if date is None:
            date = datetime.utcnow().date()
        
        period_start = datetime.combine(date, datetime.min.time())
        period_end = period_start + timedelta(days=1)
        
        return await self._calculate_metrics(period_start, period_end)
    
    async def generate_weekly_report(self, date: Optional[datetime] = None) -> PerformanceMetrics:
        """Generate weekly performance report."""
        if date is None:
            date = datetime.utcnow()
        
        # Find the start of the week (Monday)
        days_since_monday = date.weekday()
        week_start = date - timedelta(days=days_since_monday)
        period_start = datetime.combine(week_start.date(), datetime.min.time())
        period_end = period_start + timedelta(days=7)
        
        return await self._calculate_metrics(period_start, period_end)
    
    async def generate_monthly_report(self, date: Optional[datetime] = None) -> PerformanceMetrics:
        """Generate monthly performance report."""
        if date is None:
            date = datetime.utcnow()
        
        # First day of the month
        period_start = datetime(date.year, date.month, 1)
        
        # First day of next month
        if date.month == 12:
            period_end = datetime(date.year + 1, 1, 1)
        else:
            period_end = datetime(date.year, date.month + 1, 1)
        
        return await self._calculate_metrics(period_start, period_end)
    
    async def _calculate_metrics(self, start_time: datetime, end_time: datetime) -> PerformanceMetrics:
        """Calculate performance metrics for a given period."""
        # In a real implementation, this would query the trading database
        # For now, we'll return sample data
        
        metrics = PerformanceMetrics(
            period_start=start_time,
            period_end=end_time,
            total_trades=25,
            winning_trades=15,
            losing_trades=10,
            total_pnl=1250.75,
            gross_profit=2100.50,
            gross_loss=-849.75,
            largest_win=325.80,
            largest_loss=-156.40,
            max_consecutive_wins=4,
            max_consecutive_losses=3,
            max_drawdown=0.08,
            sharpe_ratio=1.45
        )
        
        metrics.calculate_derived_metrics()
        return metrics
    
    async def send_performance_report(self, metrics: PerformanceMetrics, report_type: str):
        """Send performance report notification."""
        # Determine notification type
        if report_type == "daily":
            notification_type = NotificationType.DAILY_REPORT
        elif report_type == "weekly":
            notification_type = NotificationType.WEEKLY_REPORT
        else:
            notification_type = NotificationType.MONTHLY_REPORT
        
        # Create notification
        title = f"{report_type.title()} Performance Report"
        
        # Format message
        message = f"Trading performance for {metrics.period_start.strftime('%Y-%m-%d')} to {metrics.period_end.strftime('%Y-%m-%d')}"
        
        # Add key metrics to message
        if metrics.total_trades > 0:
            pnl_emoji = "📈" if metrics.total_pnl >= 0 else "📉"
            message += f"\n{pnl_emoji} Total P&L: ${metrics.total_pnl:,.2f}"
            message += f"\n🎯 Win Rate: {metrics.win_rate:.1%}"
            message += f"\n📊 Total Trades: {metrics.total_trades}"
        else:
            message += "\nNo trades executed during this period."
        
        notification = NotificationData(
            id=f"{report_type}_report_{metrics.period_start.strftime('%Y%m%d')}",
            type=notification_type,
            priority=NotificationPriority.NORMAL,
            title=title,
            message=message,
            data=metrics.to_dict()
        )
        
        self.notification_manager.queue_notification(notification)
        self.logger.info(f"Sent {report_type} performance report")
    
    async def _reporting_loop(self):
        """Main reporting loop that schedules reports."""
        last_daily = None
        last_weekly = None
        last_monthly = None
        
        while self._running:
            try:
                now = datetime.utcnow()
                today = now.date()
                
                # Check for daily report
                if last_daily != today and now.hour >= 9:  # 9 AM UTC
                    metrics = await self.generate_daily_report()
                    await self.send_performance_report(metrics, "daily")
                    last_daily = today
                
                # Check for weekly report (Monday)
                if now.weekday() == 0 and last_weekly != today and now.hour >= 9:
                    metrics = await self.generate_weekly_report()
                    await self.send_performance_report(metrics, "weekly")
                    last_weekly = today
                
                # Check for monthly report (1st of month)
                if now.day == 1 and last_monthly != today and now.hour >= 9:
                    metrics = await self.generate_monthly_report()
                    await self.send_performance_report(metrics, "monthly")
                    last_monthly = today
                
                # Sleep for an hour before next check
                await asyncio.sleep(3600)
                
            except Exception as e:
                self.logger.error(f"Error in reporting loop: {e}", exc_info=True)
                await asyncio.sleep(3600)
    
    def get_reporting_status(self) -> Dict[str, Any]:
        """Get current reporting status."""
        return {
            'reporting_active': self._running,
            'daily_report_time': self.daily_report_time,
            'weekly_report_day': self.weekly_report_day,
            'monthly_report_day': self.monthly_report_day,
            'trade_history_count': len(self.trade_history)
        }