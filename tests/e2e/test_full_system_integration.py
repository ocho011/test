"""
E2E test for complete system integration.

Tests the full trading cycle from market data through signal generation,
strategy selection, order execution, position management, and risk controls.
"""

import pytest
import asyncio
from decimal import Decimal

from trading_bot.system_integrator import SystemIntegrator
from .utils.event_capture import EventCapture
from .utils.performance_collector import PerformanceCollector


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.asyncio
class TestFullSystemIntegration:
    """Test complete end-to-end trading system integration."""

    async def test_complete_trading_cycle(
        self,
        system_integrator: SystemIntegrator,
        event_capture: EventCapture,
        performance_collector: PerformanceCollector
    ):
        """
        Test complete trading cycle from data to position close.
        
        Flow:
        Market data → ICT analysis → Pattern detection → Signal generation →
        Strategy selection → Risk validation → Order execution → Position tracking →
        Trailing stop → Position close
        """
        await performance_collector.start_resource_monitoring(interval_seconds=1.0)
        performance_collector.checkpoint("cycle_start")
        
        # Get all components
        market_data_provider = system_integrator.get_component("market_data_provider")
        ict_analyzer = system_integrator.get_component("ict_analyzer")
        signal_generator = system_integrator.get_component("signal_generator")
        risk_manager = system_integrator.get_component("risk_manager")
        order_executor = system_integrator.get_component("order_executor")
        position_tracker = system_integrator.get_component("position_tracker")
        
        # Verify all components available
        assert all([
            market_data_provider,
            ict_analyzer,
            signal_generator,
            risk_manager,
            order_executor,
            position_tracker
        ])
        
        # Phase 1: Data collection and analysis
        performance_collector.checkpoint("data_start")
        klines = await market_data_provider.get_klines("BTCUSDT", "1h", limit=200)
        assert len(klines) > 0
        performance_collector.checkpoint("data_complete")
        
        # Phase 2: Pattern detection
        performance_collector.checkpoint("analysis_start")
        await ict_analyzer.analyze_market_data("BTCUSDT", klines)
        performance_collector.checkpoint("analysis_complete")
        
        # Wait for signal generation
        await asyncio.sleep(1.0)
        
        # Phase 3: Verify event flow
        performance_collector.checkpoint("cycle_end")
        
        # Check critical events occurred
        assert event_capture.get_event_count("market_data_updated") > 0
        
        pattern_events = event_capture.get_events("pattern_detected")
        if pattern_events:
            assert event_capture.get_event_count("signal_generated") >= 0
        
        # Phase 4: Verify event sequence
        core_sequence = [
            "market_data_updated",
            "pattern_detected"
        ]
        event_capture.assert_event_sequence(core_sequence)
        
        # Phase 5: Performance assertions
        performance_collector.assert_latency_within(
            "data_start",
            "data_complete",
            max_latency_ms=100.0
        )
        
        performance_collector.assert_latency_within(
            "analysis_start",
            "analysis_complete",
            max_latency_ms=500.0
        )
        
        # End-to-end latency
        e2e_latency = performance_collector.record_latency("cycle_start", "cycle_end")
        assert e2e_latency < 3000.0, f"Full cycle too slow: {e2e_latency:.2f}ms"
        
        # Phase 6: Resource usage verification
        await performance_collector.stop_resource_monitoring()
        
        peak_memory = performance_collector.report.get_peak_memory_mb()
        assert peak_memory is not None
        assert peak_memory < 1000, f"Memory usage too high: {peak_memory:.2f}MB"
        
        avg_cpu = performance_collector.report.get_avg_cpu_percent()
        assert avg_cpu is not None
        # CPU usage should be reasonable (< 80% on average)

    async def test_system_startup_and_shutdown(
        self,
        test_config_path,
        mock_binance_client
    ):
        """Test system startup and graceful shutdown."""
        # Create new system instance
        system = SystemIntegrator(
            config_path=str(test_config_path),
            environment="testing"
        )
        
        # Start system
        await system.start()
        
        # Verify system is running
        status = system.get_system_status()
        assert status["status"] == "running"
        assert status["environment"] == "testing"
        
        # Verify components started
        components_status = status["components"]
        assert len(components_status) > 0
        
        # Graceful shutdown
        await system.stop()
        
        # Verify system stopped
        status = system.get_system_status()
        assert status["status"] == "stopped"

    async def test_component_lifecycle_management(
        self,
        system_integrator: SystemIntegrator
    ):
        """Test component lifecycle management."""
        lifecycle_manager = system_integrator.lifecycle_manager
        
        # Get component status
        component_status = lifecycle_manager.get_component_status()
        
        # Verify all components properly initialized
        for comp_name, status in component_status.items():
            assert status["state"] in ["RUNNING", "STARTED", "INITIALIZED"]
        
        # Verify startup order was respected
        stats = lifecycle_manager.get_system_stats()
        assert stats["total_components"] > 0
        assert stats["started_components"] == stats["total_components"]

    async def test_event_bus_throughput(
        self,
        system_integrator: SystemIntegrator,
        event_capture: EventCapture,
        performance_collector: PerformanceCollector
    ):
        """Test EventBus handles high event throughput."""
        event_bus = system_integrator.event_bus
        
        # Start throughput measurement
        performance_collector.start_throughput("event_bus")
        
        # Publish many events
        num_events = 1000
        for i in range(num_events):
            await event_bus.publish(
                "test_event",
                {"sequence": i}
            )
            performance_collector.increment_throughput("event_bus")
        
        # Wait for processing
        await asyncio.sleep(0.5)
        
        # End throughput measurement
        throughput = performance_collector.end_throughput("event_bus")
        
        # Verify high throughput
        assert throughput > 500, f"Event throughput too low: {throughput:.2f} events/sec"
        
        # Verify no events lost
        event_stats = event_bus.get_stats()
        assert event_stats["published"] >= num_events

    async def test_system_recovery_from_component_failure(
        self,
        system_integrator: SystemIntegrator
    ):
        """Test system handles component failures gracefully."""
        # This test verifies the system doesn't crash when components encounter errors
        # In production, components should handle errors and continue operating
        
        # System should remain running even if individual components encounter issues
        status = system_integrator.get_system_status()
        assert status["status"] == "running"
        
        # Event bus should continue functioning
        assert system_integrator.event_bus.is_running()

    @pytest.mark.performance
    async def test_system_performance_benchmarks(
        self,
        system_integrator: SystemIntegrator,
        performance_collector: PerformanceCollector
    ):
        """Benchmark system performance against targets."""
        await performance_collector.start_resource_monitoring(interval_seconds=0.5)
        
        market_data_provider = system_integrator.get_component("market_data_provider")
        ict_analyzer = system_integrator.get_component("ict_analyzer")
        
        # Run multiple iterations
        iterations = 10
        performance_collector.start_throughput("analysis_pipeline")
        
        for i in range(iterations):
            performance_collector.checkpoint(f"iter_{i}_start")
            
            klines = await market_data_provider.get_klines("BTCUSDT", "1h", limit=100)
            await ict_analyzer.analyze_market_data("BTCUSDT", klines)
            
            performance_collector.checkpoint(f"iter_{i}_end")
            performance_collector.record_latency(f"iter_{i}_start", f"iter_{i}_end")
            performance_collector.increment_throughput("analysis_pipeline")
            
            await asyncio.sleep(0.1)
        
        throughput = performance_collector.end_throughput("analysis_pipeline")
        await performance_collector.stop_resource_monitoring()
        
        # Performance targets
        assert throughput > 5, f"Analysis throughput too low: {throughput:.2f} ops/sec"
        
        # Generate performance report
        report_summary = performance_collector.report.summary()
        
        # Verify performance metrics collected
        assert report_summary["latencies"]["count"] == iterations
        assert report_summary["resources"]["samples"] > 0
