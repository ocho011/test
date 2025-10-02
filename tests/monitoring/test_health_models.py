"""
Tests for health monitoring models.
"""

import pytest
from datetime import datetime

from src.trading_bot.monitoring.models import (
    HealthStatus,
    HealthCheckResult,
    ComponentHealth,
    SystemHealth,
    MetricsSnapshot,
)


class TestHealthStatus:
    """Tests for HealthStatus enum."""

    def test_health_status_values(self):
        """Test health status enum values."""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        assert HealthStatus.UNKNOWN.value == "unknown"


class TestHealthCheckResult:
    """Tests for HealthCheckResult model."""

    def test_create_health_check_result(self):
        """Test creating a health check result."""
        result = HealthCheckResult(
            status=HealthStatus.HEALTHY,
            message="All systems operational",
            details={"cpu": 25.5, "memory": 50.0}
        )

        assert result.status == HealthStatus.HEALTHY
        assert result.message == "All systems operational"
        assert result.details["cpu"] == 25.5
        assert result.details["memory"] == 50.0
        assert isinstance(result.timestamp, datetime)

    def test_health_check_result_to_dict(self):
        """Test converting health check result to dictionary."""
        result = HealthCheckResult(
            status=HealthStatus.DEGRADED,
            message="High CPU usage",
            details={"cpu_percent": 85.5}
        )

        result_dict = result.to_dict()

        assert result_dict["status"] == "degraded"
        assert result_dict["message"] == "High CPU usage"
        assert result_dict["details"]["cpu_percent"] == 85.5
        assert "timestamp" in result_dict


class TestComponentHealth:
    """Tests for ComponentHealth model."""

    def test_create_component_health(self):
        """Test creating component health."""
        check = HealthCheckResult(
            status=HealthStatus.HEALTHY,
            message="Component operational"
        )

        component = ComponentHealth(
            name="test_component",
            status=HealthStatus.HEALTHY,
            message="Operating normally",
            checks=[check],
            metrics={"uptime": 3600}
        )

        assert component.name == "test_component"
        assert component.status == HealthStatus.HEALTHY
        assert component.message == "Operating normally"
        assert len(component.checks) == 1
        assert component.metrics["uptime"] == 3600

    def test_component_health_is_healthy(self):
        """Test component health status check."""
        healthy_component = ComponentHealth(
            name="test",
            status=HealthStatus.HEALTHY,
            message="OK"
        )
        assert healthy_component.is_healthy is True

        degraded_component = ComponentHealth(
            name="test",
            status=HealthStatus.DEGRADED,
            message="Slow"
        )
        assert degraded_component.is_healthy is False

    def test_component_health_to_dict(self):
        """Test converting component health to dictionary."""
        component = ComponentHealth(
            name="test_component",
            status=HealthStatus.HEALTHY,
            message="OK",
            checks=[],
            metrics={"requests": 100}
        )

        component_dict = component.to_dict()

        assert component_dict["name"] == "test_component"
        assert component_dict["status"] == "healthy"
        assert component_dict["message"] == "OK"
        assert component_dict["checks"] == []
        assert component_dict["metrics"]["requests"] == 100


class TestSystemHealth:
    """Tests for SystemHealth model."""

    def test_create_system_health(self):
        """Test creating system health."""
        component1 = ComponentHealth(
            name="component1",
            status=HealthStatus.HEALTHY,
            message="OK"
        )
        component2 = ComponentHealth(
            name="component2",
            status=HealthStatus.DEGRADED,
            message="Slow"
        )

        system = SystemHealth(
            status=HealthStatus.DEGRADED,
            components={"component1": component1, "component2": component2},
            version="1.0.0",
            environment="test",
            uptime_seconds=3600.0
        )

        assert system.status == HealthStatus.DEGRADED
        assert len(system.components) == 2
        assert system.version == "1.0.0"
        assert system.environment == "test"
        assert system.uptime_seconds == 3600.0

    def test_system_health_is_healthy(self):
        """Test system health status check."""
        healthy_system = SystemHealth(
            status=HealthStatus.HEALTHY,
            components={}
        )
        assert healthy_system.is_healthy is True

        degraded_system = SystemHealth(
            status=HealthStatus.DEGRADED,
            components={}
        )
        assert degraded_system.is_healthy is False

    def test_system_health_is_ready(self):
        """Test system readiness check."""
        # System with no unhealthy components is ready
        component1 = ComponentHealth(
            name="c1",
            status=HealthStatus.HEALTHY,
            message="OK"
        )
        component2 = ComponentHealth(
            name="c2",
            status=HealthStatus.DEGRADED,
            message="Slow"
        )
        ready_system = SystemHealth(
            status=HealthStatus.DEGRADED,
            components={"c1": component1, "c2": component2}
        )
        assert ready_system.is_ready is True

        # System with unhealthy component is not ready
        unhealthy_component = ComponentHealth(
            name="c3",
            status=HealthStatus.UNHEALTHY,
            message="Failed"
        )
        not_ready_system = SystemHealth(
            status=HealthStatus.UNHEALTHY,
            components={"c1": component1, "c3": unhealthy_component}
        )
        assert not_ready_system.is_ready is False

    def test_system_health_to_dict(self):
        """Test converting system health to dictionary."""
        component1 = ComponentHealth(
            name="c1",
            status=HealthStatus.HEALTHY,
            message="OK"
        )
        component2 = ComponentHealth(
            name="c2",
            status=HealthStatus.DEGRADED,
            message="Slow"
        )

        system = SystemHealth(
            status=HealthStatus.DEGRADED,
            components={"c1": component1, "c2": component2},
            version="1.0.0",
            environment="test",
            uptime_seconds=100.0
        )

        system_dict = system.to_dict()

        assert system_dict["status"] == "degraded"
        assert system_dict["version"] == "1.0.0"
        assert system_dict["environment"] == "test"
        assert system_dict["uptime_seconds"] == 100.0
        assert len(system_dict["components"]) == 2
        assert system_dict["summary"]["total_components"] == 2
        assert system_dict["summary"]["healthy"] == 1
        assert system_dict["summary"]["degraded"] == 1
        assert system_dict["summary"]["unhealthy"] == 0


class TestMetricsSnapshot:
    """Tests for MetricsSnapshot model."""

    def test_create_metrics_snapshot(self):
        """Test creating metrics snapshot."""
        metrics = MetricsSnapshot(
            cpu_percent=45.5,
            memory_percent=60.0,
            memory_used_mb=4096.0,
            memory_available_mb=2048.0,
            disk_percent=70.0,
            event_queue_size=100,
            active_positions=5
        )

        assert metrics.cpu_percent == 45.5
        assert metrics.memory_percent == 60.0
        assert metrics.event_queue_size == 100
        assert metrics.active_positions == 5

    def test_metrics_snapshot_to_dict(self):
        """Test converting metrics snapshot to dictionary."""
        metrics = MetricsSnapshot(
            cpu_percent=45.5,
            memory_percent=60.0,
            event_queue_size=100,
            api_call_count=1000
        )

        metrics_dict = metrics.to_dict()

        assert metrics_dict["system"]["cpu_percent"] == 45.5
        assert metrics_dict["system"]["memory_percent"] == 60.0
        assert metrics_dict["trading"]["event_queue_size"] == 100
        assert metrics_dict["network"]["api_call_count"] == 1000
        assert "timestamp" in metrics_dict
