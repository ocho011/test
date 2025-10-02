"""
Health monitoring system.

Central health monitoring system that aggregates health checks from multiple
components, collects system metrics, and provides overall system health status.
"""

import asyncio
import logging
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from ..core.base_component import BaseComponent
from .health_checker import (
    HealthChecker,
    ComponentHealthChecker,
    ResourceHealthChecker,
)
from .models import (
    HealthStatus,
    ComponentHealth,
    SystemHealth,
    MetricsSnapshot,
)


logger = logging.getLogger(__name__)


class HealthMonitor(BaseComponent):
    """
    Central health monitoring system.

    Aggregates health checks from all registered components and provides
    system-wide health status and metrics.
    """

    def __init__(
        self,
        check_interval_seconds: float = 30.0,
        enable_auto_checks: bool = True
    ):
        """
        Initialize health monitor.

        Args:
            check_interval_seconds: Interval between automatic health checks
            enable_auto_checks: Whether to enable automatic periodic checks
        """
        super().__init__("HealthMonitor")
        self.check_interval_seconds = check_interval_seconds
        self.enable_auto_checks = enable_auto_checks

        # Health checkers registry
        self.checkers: Dict[str, HealthChecker] = {}

        # Latest health status
        self.latest_health: Optional[SystemHealth] = None
        self.start_time: Optional[datetime] = None

        # Resource checker
        self.resource_checker = ResourceHealthChecker()

        # Background task
        self._check_task: Optional[asyncio.Task] = None

    async def _start(self) -> None:
        """Start the health monitor."""
        self.start_time = datetime.utcnow()

        # Register system resource checker
        self.register_checker(self.resource_checker)

        # Start automatic checks if enabled
        if self.enable_auto_checks:
            self._check_task = asyncio.create_task(self._check_loop())

        self.logger.info("Health monitor started")

    async def _stop(self) -> None:
        """Stop the health monitor."""
        # Stop automatic checks
        if self._check_task and not self._check_task.done():
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass

        self.logger.info("Health monitor stopped")

    def register_checker(self, checker: HealthChecker) -> None:
        """
        Register a health checker.

        Args:
            checker: HealthChecker instance to register
        """
        self.checkers[checker.name] = checker
        self.logger.info(f"Registered health checker: {checker.name}")

    def register_component(self, component: BaseComponent) -> None:
        """
        Register a component for health monitoring.

        Args:
            component: Component to monitor
        """
        checker = ComponentHealthChecker(component)
        self.register_checker(checker)

    def unregister_checker(self, name: str) -> None:
        """
        Unregister a health checker.

        Args:
            name: Name of the checker to unregister
        """
        if name in self.checkers:
            del self.checkers[name]
            self.logger.info(f"Unregistered health checker: {name}")

    async def check_all(self) -> SystemHealth:
        """
        Perform health checks on all registered components.

        Returns:
            SystemHealth with aggregated status
        """
        self.logger.debug("Performing system-wide health check")

        # Run all health checks concurrently
        check_tasks = [
            checker.check_with_timeout()
            for checker in self.checkers.values()
        ]

        results = await asyncio.gather(*check_tasks, return_exceptions=True)

        # Build component health map
        components: Dict[str, ComponentHealth] = {}

        for checker_name, result in zip(self.checkers.keys(), results):
            if isinstance(result, Exception):
                # Handle checker exception
                components[checker_name] = ComponentHealth(
                    name=checker_name,
                    status=HealthStatus.UNKNOWN,
                    message=f"Health check failed: {str(result)}",
                    checks=[],
                    metrics={}
                )
            else:
                # Normal health check result
                components[checker_name] = ComponentHealth(
                    name=checker_name,
                    status=result.status,
                    message=result.message,
                    checks=[result],
                    metrics=result.details
                )

        # Determine overall system status
        overall_status = self._aggregate_status(components)

        # Calculate uptime
        uptime_seconds = 0.0
        if self.start_time:
            uptime_seconds = (datetime.utcnow() - self.start_time).total_seconds()

        # Create system health
        system_health = SystemHealth(
            status=overall_status,
            components=components,
            timestamp=datetime.utcnow(),
            uptime_seconds=uptime_seconds,
        )

        self.latest_health = system_health
        return system_health

    def _aggregate_status(
        self,
        components: Dict[str, ComponentHealth]
    ) -> HealthStatus:
        """
        Aggregate component statuses into overall system status.

        Args:
            components: Map of component health statuses

        Returns:
            Aggregated system health status
        """
        if not components:
            return HealthStatus.UNKNOWN

        # Count components by status
        status_counts = {
            HealthStatus.HEALTHY: 0,
            HealthStatus.DEGRADED: 0,
            HealthStatus.UNHEALTHY: 0,
            HealthStatus.UNKNOWN: 0,
        }

        for component in components.values():
            status_counts[component.status] += 1

        # Determine overall status based on counts
        if status_counts[HealthStatus.UNHEALTHY] > 0:
            return HealthStatus.UNHEALTHY
        elif status_counts[HealthStatus.DEGRADED] > 0:
            return HealthStatus.DEGRADED
        elif status_counts[HealthStatus.UNKNOWN] > 0:
            return HealthStatus.DEGRADED  # Treat unknown as degraded
        else:
            return HealthStatus.HEALTHY

    async def get_health(self) -> SystemHealth:
        """
        Get current system health.

        Returns cached health if recent, otherwise performs new check.

        Returns:
            SystemHealth instance
        """
        # If no cached health or stale, perform new check
        if self.latest_health is None or self._is_health_stale():
            return await self.check_all()

        return self.latest_health

    def _is_health_stale(self) -> bool:
        """Check if cached health is stale."""
        if self.latest_health is None:
            return True

        age = datetime.utcnow() - self.latest_health.timestamp
        return age > timedelta(seconds=self.check_interval_seconds)

    async def get_metrics(self) -> MetricsSnapshot:
        """
        Collect current system metrics.

        Returns:
            MetricsSnapshot with current metrics
        """
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            process = psutil.Process()

            # Build metrics snapshot
            metrics = MetricsSnapshot(
                timestamp=datetime.utcnow(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / (1024 * 1024),
                memory_available_mb=memory.available / (1024 * 1024),
                disk_percent=disk.percent,
                disk_used_gb=disk.used / (1024 * 1024 * 1024),
                disk_available_gb=disk.free / (1024 * 1024 * 1024),
                open_file_descriptors=process.num_fds() if hasattr(process, 'num_fds') else 0,
                thread_count=process.num_threads(),
            )

            return metrics

        except Exception as e:
            self.logger.error(f"Failed to collect metrics: {e}")
            return MetricsSnapshot()

    async def _check_loop(self) -> None:
        """Background task for periodic health checks."""
        self.logger.info(
            f"Starting automatic health checks "
            f"(interval: {self.check_interval_seconds}s)"
        )

        try:
            while True:
                try:
                    await self.check_all()
                except Exception as e:
                    self.logger.error(f"Health check failed: {e}")

                await asyncio.sleep(self.check_interval_seconds)

        except asyncio.CancelledError:
            self.logger.info("Health check loop cancelled")
            raise

    def get_component_status(self, name: str) -> Optional[ComponentHealth]:
        """
        Get health status for a specific component.

        Args:
            name: Component name

        Returns:
            ComponentHealth or None if not found
        """
        if self.latest_health and name in self.latest_health.components:
            return self.latest_health.components[name]
        return None

    def is_system_healthy(self) -> bool:
        """
        Check if system is currently healthy.

        Returns:
            True if system is healthy, False otherwise
        """
        if self.latest_health is None:
            return False
        return self.latest_health.is_healthy

    def is_system_ready(self) -> bool:
        """
        Check if system is ready to accept traffic.

        Returns:
            True if system is ready, False otherwise
        """
        if self.latest_health is None:
            return False
        return self.latest_health.is_ready
