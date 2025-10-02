"""
Tests for HealthMonitor system.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from src.trading_bot.core.base_component import BaseComponent
from src.trading_bot.monitoring.health_monitor import HealthMonitor
from src.trading_bot.monitoring.health_checker import (
    HealthChecker,
    ComponentHealthChecker,
)
from src.trading_bot.monitoring.models import (
    HealthStatus,
    HealthCheckResult,
)


class MockComponent(BaseComponent):
    """Mock component for testing."""

    def __init__(self, name: str, running: bool = True):
        super().__init__(name)
        self._running = running

    async def _start(self):
        """Mock start."""
        pass

    async def _stop(self):
        """Mock stop."""
        pass

    def is_running(self):
        """Mock running check."""
        return self._running


class MockHealthChecker(HealthChecker):
    """Mock health checker."""

    def __init__(self, name: str, status: HealthStatus = HealthStatus.HEALTHY):
        super().__init__(name)
        self.status = status

    async def check_health(self):
        """Mock check."""
        return HealthCheckResult(
            status=self.status,
            message=f"Mock check for {self.name}"
        )


class TestHealthMonitor:
    """Tests for HealthMonitor."""

    @pytest.mark.asyncio
    async def test_health_monitor_initialization(self):
        """Test health monitor initialization."""
        monitor = HealthMonitor(
            check_interval_seconds=10.0,
            enable_auto_checks=False
        )

        assert monitor.check_interval_seconds == 10.0
        assert monitor.enable_auto_checks is False
        assert len(monitor.checkers) == 0
        assert monitor.latest_health is None

    @pytest.mark.asyncio
    async def test_health_monitor_start_stop(self):
        """Test starting and stopping health monitor."""
        monitor = HealthMonitor(enable_auto_checks=False)

        await monitor.start()
        assert monitor.is_running() is True
        assert monitor.start_time is not None

        await monitor.stop()
        assert monitor.is_running() is False

    @pytest.mark.asyncio
    async def test_register_checker(self):
        """Test registering a health checker."""
        monitor = HealthMonitor(enable_auto_checks=False)
        await monitor.start()

        checker = MockHealthChecker("test_checker", HealthStatus.HEALTHY)
        monitor.register_checker(checker)

        assert "test_checker" in monitor.checkers
        assert monitor.checkers["test_checker"] == checker

        await monitor.stop()

    @pytest.mark.asyncio
    async def test_register_component(self):
        """Test registering a component for health monitoring."""
        monitor = HealthMonitor(enable_auto_checks=False)
        await monitor.start()

        component = MockComponent("test_component", running=True)
        monitor.register_component(component)

        assert "test_component" in monitor.checkers
        assert isinstance(monitor.checkers["test_component"], ComponentHealthChecker)

        await monitor.stop()

    @pytest.mark.asyncio
    async def test_unregister_checker(self):
        """Test unregistering a health checker."""
        monitor = HealthMonitor(enable_auto_checks=False)
        await monitor.start()

        checker = MockHealthChecker("test_checker")
        monitor.register_checker(checker)
        assert "test_checker" in monitor.checkers

        monitor.unregister_checker("test_checker")
        assert "test_checker" not in monitor.checkers

        await monitor.stop()

    @pytest.mark.asyncio
    async def test_check_all_healthy(self):
        """Test checking all components when all are healthy."""
        monitor = HealthMonitor(enable_auto_checks=False)
        await monitor.start()

        # Register multiple healthy checkers
        monitor.register_checker(MockHealthChecker("checker1", HealthStatus.HEALTHY))
        monitor.register_checker(MockHealthChecker("checker2", HealthStatus.HEALTHY))
        monitor.register_checker(MockHealthChecker("checker3", HealthStatus.HEALTHY))

        system_health = await monitor.check_all()

        assert system_health.status == HealthStatus.HEALTHY
        assert len(system_health.components) == 4  # 3 + resource checker
        assert system_health.uptime_seconds > 0

        await monitor.stop()

    @pytest.mark.asyncio
    async def test_check_all_degraded(self):
        """Test checking all components when some are degraded."""
        monitor = HealthMonitor(enable_auto_checks=False)
        await monitor.start()

        monitor.register_checker(MockHealthChecker("checker1", HealthStatus.HEALTHY))
        monitor.register_checker(MockHealthChecker("checker2", HealthStatus.DEGRADED))
        monitor.register_checker(MockHealthChecker("checker3", HealthStatus.HEALTHY))

        system_health = await monitor.check_all()

        assert system_health.status == HealthStatus.DEGRADED
        assert len(system_health.components) == 4

        await monitor.stop()

    @pytest.mark.asyncio
    async def test_check_all_unhealthy(self):
        """Test checking all components when some are unhealthy."""
        monitor = HealthMonitor(enable_auto_checks=False)
        await monitor.start()

        monitor.register_checker(MockHealthChecker("checker1", HealthStatus.HEALTHY))
        monitor.register_checker(MockHealthChecker("checker2", HealthStatus.UNHEALTHY))
        monitor.register_checker(MockHealthChecker("checker3", HealthStatus.DEGRADED))

        system_health = await monitor.check_all()

        assert system_health.status == HealthStatus.UNHEALTHY
        assert len(system_health.components) == 4

        await monitor.stop()

    @pytest.mark.asyncio
    async def test_aggregate_status(self):
        """Test status aggregation logic."""
        monitor = HealthMonitor(enable_auto_checks=False)
        await monitor.start()

        from src.trading_bot.monitoring.models import ComponentHealth

        # All healthy
        components = {
            "c1": ComponentHealth("c1", HealthStatus.HEALTHY, "OK"),
            "c2": ComponentHealth("c2", HealthStatus.HEALTHY, "OK"),
        }
        assert monitor._aggregate_status(components) == HealthStatus.HEALTHY

        # One degraded
        components["c2"].status = HealthStatus.DEGRADED
        assert monitor._aggregate_status(components) == HealthStatus.DEGRADED

        # One unhealthy
        components["c2"].status = HealthStatus.UNHEALTHY
        assert monitor._aggregate_status(components) == HealthStatus.UNHEALTHY

        # One unknown
        components = {
            "c1": ComponentHealth("c1", HealthStatus.HEALTHY, "OK"),
            "c2": ComponentHealth("c2", HealthStatus.UNKNOWN, "?"),
        }
        assert monitor._aggregate_status(components) == HealthStatus.DEGRADED

        await monitor.stop()

    @pytest.mark.asyncio
    async def test_get_health_cached(self):
        """Test getting cached health status."""
        monitor = HealthMonitor(
            check_interval_seconds=10.0,
            enable_auto_checks=False
        )
        await monitor.start()

        monitor.register_checker(MockHealthChecker("checker1", HealthStatus.HEALTHY))

        # First call should check
        health1 = await monitor.get_health()
        assert health1 is not None

        # Second call should return cached
        health2 = await monitor.get_health()
        assert health2 == health1  # Same object

        await monitor.stop()

    @pytest.mark.asyncio
    async def test_get_health_stale(self):
        """Test getting health status when cache is stale."""
        monitor = HealthMonitor(
            check_interval_seconds=0.1,  # Very short interval
            enable_auto_checks=False
        )
        await monitor.start()

        monitor.register_checker(MockHealthChecker("checker1", HealthStatus.HEALTHY))

        # First call
        health1 = await monitor.get_health()
        assert health1 is not None

        # Wait for cache to become stale
        await asyncio.sleep(0.2)

        # Second call should perform new check
        health2 = await monitor.get_health()
        assert health2 is not health1  # Different object

        await monitor.stop()

    @pytest.mark.asyncio
    async def test_get_metrics(self):
        """Test getting system metrics."""
        monitor = HealthMonitor(enable_auto_checks=False)
        await monitor.start()

        with patch('psutil.cpu_percent', return_value=45.5), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_usage') as mock_disk, \
             patch('psutil.Process') as mock_process:

            mock_memory.return_value = Mock(
                percent=60.0,
                used=4096 * 1024 * 1024,
                available=2048 * 1024 * 1024
            )
            mock_disk.return_value = Mock(
                percent=70.0,
                used=100 * 1024 * 1024 * 1024,
                free=50 * 1024 * 1024 * 1024
            )
            mock_process.return_value.num_threads.return_value = 10

            metrics = await monitor.get_metrics()

            assert metrics.cpu_percent == 45.5
            assert metrics.memory_percent == 60.0
            assert metrics.disk_percent == 70.0
            assert metrics.thread_count == 10

        await monitor.stop()

    @pytest.mark.asyncio
    async def test_get_component_status(self):
        """Test getting status for specific component."""
        monitor = HealthMonitor(enable_auto_checks=False)
        await monitor.start()

        monitor.register_checker(MockHealthChecker("test_component", HealthStatus.HEALTHY))
        await monitor.check_all()

        component_health = monitor.get_component_status("test_component")
        assert component_health is not None
        assert component_health.name == "test_component"
        assert component_health.status == HealthStatus.HEALTHY

        # Non-existent component
        missing_health = monitor.get_component_status("non_existent")
        assert missing_health is None

        await monitor.stop()

    @pytest.mark.asyncio
    async def test_is_system_healthy(self):
        """Test system healthy check."""
        monitor = HealthMonitor(enable_auto_checks=False)
        await monitor.start()

        monitor.register_checker(MockHealthChecker("checker1", HealthStatus.HEALTHY))
        await monitor.check_all()

        assert monitor.is_system_healthy() is True

        # Add unhealthy checker
        monitor.register_checker(MockHealthChecker("checker2", HealthStatus.UNHEALTHY))
        await monitor.check_all()

        assert monitor.is_system_healthy() is False

        await monitor.stop()

    @pytest.mark.asyncio
    async def test_is_system_ready(self):
        """Test system readiness check."""
        monitor = HealthMonitor(enable_auto_checks=False)
        await monitor.start()

        # Healthy and degraded components - system is ready
        monitor.register_checker(MockHealthChecker("checker1", HealthStatus.HEALTHY))
        monitor.register_checker(MockHealthChecker("checker2", HealthStatus.DEGRADED))
        await monitor.check_all()

        assert monitor.is_system_ready() is True

        # Add unhealthy component - system is not ready
        monitor.register_checker(MockHealthChecker("checker3", HealthStatus.UNHEALTHY))
        await monitor.check_all()

        assert monitor.is_system_ready() is False

        await monitor.stop()

    @pytest.mark.asyncio
    async def test_auto_check_loop(self):
        """Test automatic health check loop."""
        monitor = HealthMonitor(
            check_interval_seconds=0.1,
            enable_auto_checks=True
        )

        monitor.register_checker(MockHealthChecker("checker1", HealthStatus.HEALTHY))

        await monitor.start()

        # Wait for multiple check cycles
        await asyncio.sleep(0.3)

        # Should have updated health multiple times
        assert monitor.latest_health is not None

        await monitor.stop()
