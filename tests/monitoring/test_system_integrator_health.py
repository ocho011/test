"""
Tests for SystemIntegrator health endpoints.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from src.trading_bot.system_integrator import SystemIntegrator
from src.trading_bot.monitoring.models import (
    HealthStatus,
    ComponentHealth,
    SystemHealth,
    MetricsSnapshot,
)


class TestSystemIntegratorHealth:
    """Tests for SystemIntegrator health endpoints."""

    @pytest.mark.asyncio
    async def test_get_health_no_monitor(self):
        """Test get_health when monitor is not initialized."""
        integrator = SystemIntegrator()
        integrator.health_monitor = None

        health = await integrator.get_health()

        assert health["status"] == "unknown"
        assert "not initialized" in health["message"].lower()

    @pytest.mark.asyncio
    async def test_get_health_success(self):
        """Test successful health check."""
        integrator = SystemIntegrator()

        # Mock health monitor
        mock_health_monitor = AsyncMock()
        system_health = SystemHealth(
            status=HealthStatus.HEALTHY,
            components={
                "component1": ComponentHealth(
                    name="component1",
                    status=HealthStatus.HEALTHY,
                    message="OK"
                )
            },
            uptime_seconds=100.0
        )
        mock_health_monitor.get_health.return_value = system_health

        integrator.health_monitor = mock_health_monitor

        health = await integrator.get_health()

        assert health["status"] == "healthy"
        assert "component1" in health["components"]
        assert health["uptime_seconds"] == 100.0

    @pytest.mark.asyncio
    async def test_get_readiness_no_monitor(self):
        """Test get_readiness when monitor is not initialized."""
        integrator = SystemIntegrator()
        integrator.health_monitor = None

        readiness = await integrator.get_readiness()

        assert readiness["ready"] is False
        assert "not initialized" in readiness["message"].lower()

    @pytest.mark.asyncio
    async def test_get_readiness_ready(self):
        """Test readiness when system is ready."""
        integrator = SystemIntegrator()

        # Mock health monitor
        mock_health_monitor = AsyncMock()
        system_health = SystemHealth(
            status=HealthStatus.HEALTHY,
            components={
                "c1": ComponentHealth("c1", HealthStatus.HEALTHY, "OK"),
                "c2": ComponentHealth("c2", HealthStatus.DEGRADED, "Slow"),
            },
            uptime_seconds=100.0
        )
        mock_health_monitor.get_health.return_value = system_health

        integrator.health_monitor = mock_health_monitor

        readiness = await integrator.get_readiness()

        assert readiness["ready"] is True
        assert readiness["status"] == "healthy"
        assert readiness["components_summary"]["total"] == 2
        assert readiness["components_summary"]["healthy"] == 1
        assert readiness["components_summary"]["degraded"] == 1

    @pytest.mark.asyncio
    async def test_get_readiness_not_ready(self):
        """Test readiness when system is not ready."""
        integrator = SystemIntegrator()

        # Mock health monitor
        mock_health_monitor = AsyncMock()
        system_health = SystemHealth(
            status=HealthStatus.UNHEALTHY,
            components={
                "c1": ComponentHealth("c1", HealthStatus.HEALTHY, "OK"),
                "c2": ComponentHealth("c2", HealthStatus.UNHEALTHY, "Failed"),
            },
            uptime_seconds=100.0
        )
        mock_health_monitor.get_health.return_value = system_health

        integrator.health_monitor = mock_health_monitor

        readiness = await integrator.get_readiness()

        assert readiness["ready"] is False
        assert readiness["status"] == "unhealthy"
        assert readiness["components_summary"]["unhealthy"] == 1

    @pytest.mark.asyncio
    async def test_get_liveness_running(self):
        """Test liveness when system is running."""
        integrator = SystemIntegrator()

        # Mock is_running
        with patch.object(integrator, 'is_running', return_value=True):
            liveness = await integrator.get_liveness()

            assert liveness["alive"] is True
            assert liveness["status"] == "running"
            assert "timestamp" in liveness

    @pytest.mark.asyncio
    async def test_get_liveness_stopped(self):
        """Test liveness when system is stopped."""
        integrator = SystemIntegrator()

        # Mock is_running
        with patch.object(integrator, 'is_running', return_value=False):
            liveness = await integrator.get_liveness()

            assert liveness["alive"] is False
            assert liveness["status"] == "stopped"

    @pytest.mark.asyncio
    async def test_get_metrics_no_monitor(self):
        """Test get_metrics when monitor is not initialized."""
        integrator = SystemIntegrator()
        integrator.health_monitor = None

        metrics = await integrator.get_metrics()

        assert "error" in metrics

    @pytest.mark.asyncio
    async def test_get_metrics_success(self):
        """Test successful metrics collection."""
        integrator = SystemIntegrator()

        # Mock health monitor
        mock_health_monitor = AsyncMock()
        metrics_snapshot = MetricsSnapshot(
            cpu_percent=45.5,
            memory_percent=60.0,
            event_queue_size=100,
            active_positions=5
        )
        mock_health_monitor.get_metrics.return_value = metrics_snapshot

        integrator.health_monitor = mock_health_monitor

        metrics = await integrator.get_metrics()

        assert metrics["system"]["cpu_percent"] == 45.5
        assert metrics["system"]["memory_percent"] == 60.0
        assert metrics["trading"]["event_queue_size"] == 100
        assert metrics["trading"]["active_positions"] == 5

    @pytest.mark.asyncio
    async def test_get_component_health_no_monitor(self):
        """Test get_component_health when monitor is not initialized."""
        integrator = SystemIntegrator()
        integrator.health_monitor = None

        component_health = await integrator.get_component_health("test")

        assert "error" in component_health
        assert "not initialized" in component_health["error"].lower()

    @pytest.mark.asyncio
    async def test_get_component_health_found(self):
        """Test get_component_health when component is found."""
        integrator = SystemIntegrator()

        # Mock health monitor
        mock_health_monitor = Mock()
        component_health = ComponentHealth(
            name="test_component",
            status=HealthStatus.HEALTHY,
            message="OK"
        )
        mock_health_monitor.get_component_status.return_value = component_health

        integrator.health_monitor = mock_health_monitor

        result = await integrator.get_component_health("test_component")

        assert result["name"] == "test_component"
        assert result["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_get_component_health_not_found(self):
        """Test get_component_health when component is not found."""
        integrator = SystemIntegrator()

        # Mock health monitor
        mock_health_monitor = Mock()
        mock_health_monitor.get_component_status.return_value = None

        integrator.health_monitor = mock_health_monitor

        result = await integrator.get_component_health("non_existent")

        assert "error" in result
        assert "not found" in result["error"].lower()
