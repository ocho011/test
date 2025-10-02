"""
Health monitoring data models.

Defines the data structures used for health checks, component status,
system health aggregation, and metrics collection.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any


class HealthStatus(str, Enum):
    """Health status enumeration."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a single health check."""

    status: HealthStatus
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "status": self.status.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
        }


@dataclass
class ComponentHealth:
    """Health information for a single component."""

    name: str
    status: HealthStatus
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    checks: List[HealthCheckResult] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "checks": [check.to_dict() for check in self.checks],
            "metrics": self.metrics,
        }

    @property
    def is_healthy(self) -> bool:
        """Check if component is healthy."""
        return self.status == HealthStatus.HEALTHY


@dataclass
class SystemHealth:
    """Overall system health information."""

    status: HealthStatus
    components: Dict[str, ComponentHealth]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    version: str = "1.0.0"
    environment: str = "production"
    uptime_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
            "environment": self.environment,
            "uptime_seconds": self.uptime_seconds,
            "components": {
                name: comp.to_dict()
                for name, comp in self.components.items()
            },
            "summary": {
                "total_components": len(self.components),
                "healthy": sum(1 for c in self.components.values() if c.status == HealthStatus.HEALTHY),
                "degraded": sum(1 for c in self.components.values() if c.status == HealthStatus.DEGRADED),
                "unhealthy": sum(1 for c in self.components.values() if c.status == HealthStatus.UNHEALTHY),
                "unknown": sum(1 for c in self.components.values() if c.status == HealthStatus.UNKNOWN),
            }
        }

    @property
    def is_healthy(self) -> bool:
        """Check if entire system is healthy."""
        return self.status == HealthStatus.HEALTHY

    @property
    def is_ready(self) -> bool:
        """Check if system is ready to accept traffic."""
        # System is ready if no components are unhealthy
        return not any(
            comp.status == HealthStatus.UNHEALTHY
            for comp in self.components.values()
        )


@dataclass
class MetricsSnapshot:
    """Snapshot of system metrics at a point in time."""

    timestamp: datetime = field(default_factory=datetime.utcnow)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_available_mb: float = 0.0
    disk_percent: float = 0.0
    disk_used_gb: float = 0.0
    disk_available_gb: float = 0.0
    open_file_descriptors: int = 0
    thread_count: int = 0

    # Component-specific metrics
    event_queue_size: int = 0
    event_processing_rate: float = 0.0
    active_positions: int = 0
    pending_orders: int = 0

    # Network metrics
    network_errors: int = 0
    api_call_count: int = 0
    api_error_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "system": {
                "cpu_percent": self.cpu_percent,
                "memory_percent": self.memory_percent,
                "memory_used_mb": self.memory_used_mb,
                "memory_available_mb": self.memory_available_mb,
                "disk_percent": self.disk_percent,
                "disk_used_gb": self.disk_used_gb,
                "disk_available_gb": self.disk_available_gb,
                "open_file_descriptors": self.open_file_descriptors,
                "thread_count": self.thread_count,
            },
            "trading": {
                "event_queue_size": self.event_queue_size,
                "event_processing_rate": self.event_processing_rate,
                "active_positions": self.active_positions,
                "pending_orders": self.pending_orders,
            },
            "network": {
                "network_errors": self.network_errors,
                "api_call_count": self.api_call_count,
                "api_error_count": self.api_error_count,
            }
        }
