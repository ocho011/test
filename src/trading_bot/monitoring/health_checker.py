"""
Health checker implementations.

Provides various health check implementations for different component types
including base health checkers, database health checks, external service checks,
and system resource monitoring.
"""

import asyncio
import logging
import psutil
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Dict, Any

from ..core.base_component import BaseComponent
from .models import HealthStatus, HealthCheckResult, ComponentHealth


logger = logging.getLogger(__name__)


class HealthChecker(ABC):
    """Base class for health checkers."""

    def __init__(self, name: str, timeout_seconds: float = 5.0):
        """
        Initialize health checker.

        Args:
            name: Name of the component being checked
            timeout_seconds: Timeout for health check operations
        """
        self.name = name
        self.timeout_seconds = timeout_seconds
        self.logger = logging.getLogger(f"{__name__}.{name}")

    @abstractmethod
    async def check_health(self) -> HealthCheckResult:
        """
        Perform health check.

        Returns:
            HealthCheckResult with status and details
        """
        pass

    async def check_with_timeout(self) -> HealthCheckResult:
        """
        Perform health check with timeout.

        Returns:
            HealthCheckResult, or UNHEALTHY if timeout occurs
        """
        try:
            return await asyncio.wait_for(
                self.check_health(),
                timeout=self.timeout_seconds
            )
        except asyncio.TimeoutError:
            self.logger.warning(f"Health check timeout for {self.name}")
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Health check timed out after {self.timeout_seconds}s",
                details={"timeout": True}
            )
        except Exception as e:
            self.logger.error(f"Health check error for {self.name}: {e}")
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                details={"error": str(e), "error_type": type(e).__name__}
            )


class ComponentHealthChecker(HealthChecker):
    """Health checker for BaseComponent instances."""

    def __init__(
        self,
        component: BaseComponent,
        timeout_seconds: float = 5.0
    ):
        """
        Initialize component health checker.

        Args:
            component: Component to check
            timeout_seconds: Timeout for health check operations
        """
        super().__init__(component.name, timeout_seconds)
        self.component = component

    async def check_health(self) -> HealthCheckResult:
        """Check component health."""
        try:
            # Check if component is running
            if not self.component.is_running():
                return HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    message="Component is not running",
                    details={"running": False}
                )

            # Component is running and operational
            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                message="Component is operational",
                details={
                    "running": True,
                    "component_type": type(self.component).__name__
                }
            )

        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Component check failed: {str(e)}",
                details={"error": str(e)}
            )


class DatabaseHealthChecker(HealthChecker):
    """Health checker for database connections."""

    def __init__(
        self,
        name: str,
        connection_test_func,
        timeout_seconds: float = 3.0
    ):
        """
        Initialize database health checker.

        Args:
            name: Name of the database
            connection_test_func: Async function to test connection
            timeout_seconds: Timeout for connection test
        """
        super().__init__(name, timeout_seconds)
        self.connection_test_func = connection_test_func

    async def check_health(self) -> HealthCheckResult:
        """Check database connection health."""
        try:
            # Test database connection
            await self.connection_test_func()

            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                message="Database connection is healthy",
                details={"connected": True}
            )

        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Database connection failed: {str(e)}",
                details={"connected": False, "error": str(e)}
            )


class ExternalServiceHealthChecker(HealthChecker):
    """Health checker for external service connections."""

    def __init__(
        self,
        name: str,
        service_test_func,
        timeout_seconds: float = 5.0,
        degraded_threshold: float = 0.7
    ):
        """
        Initialize external service health checker.

        Args:
            name: Name of the external service
            service_test_func: Async function to test service
            timeout_seconds: Timeout for service test
            degraded_threshold: Response time threshold for degraded status
        """
        super().__init__(name, timeout_seconds)
        self.service_test_func = service_test_func
        self.degraded_threshold = degraded_threshold

    async def check_health(self) -> HealthCheckResult:
        """Check external service health."""
        try:
            start_time = datetime.utcnow()

            # Test service connection
            await self.service_test_func()

            response_time = (datetime.utcnow() - start_time).total_seconds()

            # Determine status based on response time
            if response_time > self.timeout_seconds * self.degraded_threshold:
                status = HealthStatus.DEGRADED
                message = f"Service responding slowly ({response_time:.2f}s)"
            else:
                status = HealthStatus.HEALTHY
                message = "Service is responding normally"

            return HealthCheckResult(
                status=status,
                message=message,
                details={
                    "reachable": True,
                    "response_time_seconds": response_time
                }
            )

        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Service unreachable: {str(e)}",
                details={"reachable": False, "error": str(e)}
            )


class ResourceHealthChecker(HealthChecker):
    """Health checker for system resources (CPU, memory, disk)."""

    def __init__(
        self,
        name: str = "system_resources",
        cpu_threshold: float = 90.0,
        memory_threshold: float = 90.0,
        disk_threshold: float = 90.0
    ):
        """
        Initialize resource health checker.

        Args:
            name: Name for this checker
            cpu_threshold: CPU usage percentage for degraded status
            memory_threshold: Memory usage percentage for degraded status
            disk_threshold: Disk usage percentage for degraded status
        """
        super().__init__(name, timeout_seconds=2.0)
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.disk_threshold = disk_threshold

    async def check_health(self) -> HealthCheckResult:
        """Check system resource health."""
        try:
            # Get resource usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            # Determine overall status
            issues = []

            if cpu_percent > self.cpu_threshold:
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")

            if memory.percent > self.memory_threshold:
                issues.append(f"High memory usage: {memory.percent:.1f}%")

            if disk.percent > self.disk_threshold:
                issues.append(f"High disk usage: {disk.percent:.1f}%")

            # Set status based on issues
            if issues:
                status = HealthStatus.DEGRADED
                message = "; ".join(issues)
            else:
                status = HealthStatus.HEALTHY
                message = "System resources are within normal limits"

            return HealthCheckResult(
                status=status,
                message=message,
                details={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_used_mb": memory.used / (1024 * 1024),
                    "memory_available_mb": memory.available / (1024 * 1024),
                    "disk_percent": disk.percent,
                    "disk_used_gb": disk.used / (1024 * 1024 * 1024),
                    "disk_free_gb": disk.free / (1024 * 1024 * 1024),
                }
            )

        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNKNOWN,
                message=f"Failed to check resources: {str(e)}",
                details={"error": str(e)}
            )
