"""
Tests for health checker implementations.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from src.trading_bot.core.base_component import BaseComponent
from src.trading_bot.monitoring.health_checker import (
    HealthChecker,
    ComponentHealthChecker,
    DatabaseHealthChecker,
    ExternalServiceHealthChecker,
    ResourceHealthChecker,
)
from src.trading_bot.monitoring.models import HealthStatus


class MockHealthChecker(HealthChecker):
    """Mock health checker for testing base class."""

    def __init__(self, name: str, return_status: HealthStatus = HealthStatus.HEALTHY):
        super().__init__(name)
        self.return_status = return_status
        self.check_called = False

    async def check_health(self):
        """Mock health check."""
        self.check_called = True
        from src.trading_bot.monitoring.models import HealthCheckResult
        return HealthCheckResult(
            status=self.return_status,
            message="Mock health check"
        )


class TestHealthChecker:
    """Tests for HealthChecker base class."""

    @pytest.mark.asyncio
    async def test_check_with_timeout_success(self):
        """Test successful health check with timeout."""
        checker = MockHealthChecker("test", HealthStatus.HEALTHY)
        result = await checker.check_with_timeout()

        assert checker.check_called is True
        assert result.status == HealthStatus.HEALTHY
        assert result.message == "Mock health check"

    @pytest.mark.asyncio
    async def test_check_with_timeout_timeout(self):
        """Test health check timeout."""
        async def slow_check():
            await asyncio.sleep(10)
            from src.trading_bot.monitoring.models import HealthCheckResult
            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                message="Should not reach"
            )

        checker = MockHealthChecker("test")
        checker.check_health = slow_check
        checker.timeout_seconds = 0.1

        result = await checker.check_with_timeout()

        assert result.status == HealthStatus.UNHEALTHY
        assert "timed out" in result.message.lower()
        assert result.details.get("timeout") is True

    @pytest.mark.asyncio
    async def test_check_with_timeout_exception(self):
        """Test health check with exception."""
        async def failing_check():
            raise ValueError("Test error")

        checker = MockHealthChecker("test")
        checker.check_health = failing_check

        result = await checker.check_with_timeout()

        assert result.status == HealthStatus.UNHEALTHY
        assert "Test error" in result.message
        assert result.details.get("error") == "Test error"


class TestComponentHealthChecker:
    """Tests for ComponentHealthChecker."""

    @pytest.mark.asyncio
    async def test_check_running_component(self):
        """Test checking a running component."""
        # Create mock component
        component = Mock(spec=BaseComponent)
        component.name = "test_component"
        component.is_running.return_value = True

        checker = ComponentHealthChecker(component)
        result = await checker.check_health()

        assert result.status == HealthStatus.HEALTHY
        assert "operational" in result.message.lower()
        assert result.details["running"] is True

    @pytest.mark.asyncio
    async def test_check_stopped_component(self):
        """Test checking a stopped component."""
        component = Mock(spec=BaseComponent)
        component.name = "test_component"
        component.is_running.return_value = False

        checker = ComponentHealthChecker(component)
        result = await checker.check_health()

        assert result.status == HealthStatus.UNHEALTHY
        assert "not running" in result.message.lower()
        assert result.details["running"] is False

    @pytest.mark.asyncio
    async def test_check_component_exception(self):
        """Test component check with exception."""
        component = Mock(spec=BaseComponent)
        component.name = "test_component"
        component.is_running.side_effect = RuntimeError("Component error")

        checker = ComponentHealthChecker(component)
        result = await checker.check_health()

        assert result.status == HealthStatus.UNHEALTHY
        assert "Component error" in result.message


class TestDatabaseHealthChecker:
    """Tests for DatabaseHealthChecker."""

    @pytest.mark.asyncio
    async def test_check_healthy_database(self):
        """Test checking healthy database connection."""
        async def successful_connection():
            return True

        checker = DatabaseHealthChecker(
            name="test_db",
            connection_test_func=successful_connection
        )
        result = await checker.check_health()

        assert result.status == HealthStatus.HEALTHY
        assert "healthy" in result.message.lower()
        assert result.details["connected"] is True

    @pytest.mark.asyncio
    async def test_check_failed_database(self):
        """Test checking failed database connection."""
        async def failed_connection():
            raise ConnectionError("Database unavailable")

        checker = DatabaseHealthChecker(
            name="test_db",
            connection_test_func=failed_connection
        )
        result = await checker.check_health()

        assert result.status == HealthStatus.UNHEALTHY
        assert "Database unavailable" in result.message
        assert result.details["connected"] is False


class TestExternalServiceHealthChecker:
    """Tests for ExternalServiceHealthChecker."""

    @pytest.mark.asyncio
    async def test_check_healthy_service(self):
        """Test checking healthy external service."""
        async def fast_service():
            await asyncio.sleep(0.01)
            return True

        checker = ExternalServiceHealthChecker(
            name="test_service",
            service_test_func=fast_service,
            timeout_seconds=1.0,
            degraded_threshold=0.5
        )
        result = await checker.check_health()

        assert result.status == HealthStatus.HEALTHY
        assert result.details["reachable"] is True
        assert "response_time_seconds" in result.details

    @pytest.mark.asyncio
    async def test_check_degraded_service(self):
        """Test checking degraded (slow) external service."""
        async def slow_service():
            await asyncio.sleep(0.4)
            return True

        checker = ExternalServiceHealthChecker(
            name="test_service",
            service_test_func=slow_service,
            timeout_seconds=1.0,
            degraded_threshold=0.3  # 0.3 * 1.0 = 0.3s threshold
        )
        result = await checker.check_health()

        assert result.status == HealthStatus.DEGRADED
        assert "slowly" in result.message.lower()
        assert result.details["reachable"] is True

    @pytest.mark.asyncio
    async def test_check_unreachable_service(self):
        """Test checking unreachable external service."""
        async def failing_service():
            raise ConnectionError("Service unavailable")

        checker = ExternalServiceHealthChecker(
            name="test_service",
            service_test_func=failing_service
        )
        result = await checker.check_health()

        assert result.status == HealthStatus.UNHEALTHY
        assert "unreachable" in result.message.lower()
        assert result.details["reachable"] is False


class TestResourceHealthChecker:
    """Tests for ResourceHealthChecker."""

    @pytest.mark.asyncio
    async def test_check_healthy_resources(self):
        """Test checking healthy system resources."""
        with patch('psutil.cpu_percent', return_value=30.0), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_usage') as mock_disk:

            mock_memory.return_value = Mock(
                percent=40.0,
                used=4096 * 1024 * 1024,
                available=6144 * 1024 * 1024
            )
            mock_disk.return_value = Mock(
                percent=50.0,
                used=100 * 1024 * 1024 * 1024,
                free=100 * 1024 * 1024 * 1024
            )

            checker = ResourceHealthChecker(
                cpu_threshold=80.0,
                memory_threshold=80.0,
                disk_threshold=80.0
            )
            result = await checker.check_health()

            assert result.status == HealthStatus.HEALTHY
            assert "normal" in result.message.lower()
            assert result.details["cpu_percent"] == 30.0
            assert result.details["memory_percent"] == 40.0

    @pytest.mark.asyncio
    async def test_check_degraded_resources(self):
        """Test checking degraded system resources."""
        with patch('psutil.cpu_percent', return_value=95.0), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_usage') as mock_disk:

            mock_memory.return_value = Mock(
                percent=92.0,
                used=14 * 1024 * 1024 * 1024,
                available=1 * 1024 * 1024 * 1024
            )
            mock_disk.return_value = Mock(
                percent=50.0,
                used=100 * 1024 * 1024 * 1024,
                free=100 * 1024 * 1024 * 1024
            )

            checker = ResourceHealthChecker(
                cpu_threshold=90.0,
                memory_threshold=90.0,
                disk_threshold=90.0
            )
            result = await checker.check_health()

            assert result.status == HealthStatus.DEGRADED
            assert "High CPU" in result.message
            assert "High memory" in result.message

    @pytest.mark.asyncio
    async def test_check_resources_exception(self):
        """Test resource check with exception."""
        with patch('psutil.cpu_percent', side_effect=RuntimeError("psutil error")):
            checker = ResourceHealthChecker()
            result = await checker.check_health()

            assert result.status == HealthStatus.UNKNOWN
            assert "Failed to check" in result.message
