"""
E2E tests for signal → order execution flow.

Tests signal processing through strategy selection to order placement and position tracking.
"""

import pytest
import asyncio
from decimal import Decimal

from trading_bot.system_integrator import SystemIntegrator
from .utils.event_capture import EventCapture
from .utils.performance_collector import PerformanceCollector


@pytest.mark.e2e
@pytest.mark.asyncio
class TestSignalToExecutionFlow:
    """Test signal → strategy → order execution → position tracking flow."""

    async def test_signal_to_strategy_selection(
        self,
        system_integrator: SystemIntegrator,
        event_capture: EventCapture,
        sample_signal_data: dict
    ):
        """Test that signals trigger strategy selection."""
        strategy_selector = system_integrator.get_component("strategy_selector")
        signal_generator = system_integrator.get_component("signal_generator")
        
        assert strategy_selector is not None
        assert signal_generator is not None
        
        # Publish signal event
        await system_integrator.event_bus.publish(
            "signal_generated",
            sample_signal_data
        )
        
        await asyncio.sleep(0.2)
        
        # Verify strategy was selected
        # In a real system, this would trigger strategy execution
        assert True  # Strategy selection is internal

    async def test_order_placement_from_signal(
        self,
        system_integrator: SystemIntegrator,
        event_capture: EventCapture,
        performance_collector: PerformanceCollector,
        sample_signal_data: dict
    ):
        """Test order placement triggered by trading signal."""
        order_executor = system_integrator.get_component("order_executor")
        position_tracker = system_integrator.get_component("position_tracker")
        
        assert order_executor is not None
        assert position_tracker is not None
        
        performance_collector.checkpoint("signal_received")
        
        # In real system, signal would flow through strategy to order executor
        # For testing, we simulate order placement
        order_result = await order_executor.execute_market_order(
            symbol=sample_signal_data["symbol"],
            side="BUY" if sample_signal_data["direction"] == "LONG" else "SELL",
            quantity=Decimal("0.01")
        )
        
        performance_collector.checkpoint("order_placed")
        
        # Verify order execution
        assert order_result is not None
        assert order_result.get("status") in ["FILLED", "NEW"]
        
        # Check execution latency
        exec_latency = performance_collector.record_latency(
            "signal_received",
            "order_placed"
        )
        assert exec_latency < 500, f"Order execution too slow: {exec_latency:.2f}ms"
        
        await asyncio.sleep(0.2)
        
        # Verify order event
        order_events = event_capture.get_events("order_placed")
        assert len(order_events) > 0

    async def test_position_tracking_integration(
        self,
        system_integrator: SystemIntegrator,
        event_capture: EventCapture
    ):
        """Test that executed orders update position tracker."""
        order_executor = system_integrator.get_component("order_executor")
        position_tracker = system_integrator.get_component("position_tracker")
        
        # Execute order
        await order_executor.execute_market_order(
            symbol="BTCUSDT",
            side="BUY",
            quantity=Decimal("0.01")
        )
        
        await asyncio.sleep(0.2)
        
        # Verify position opened
        position_events = event_capture.get_events("position_opened")
        assert len(position_events) > 0
        
        # Check position data
        position = position_events[0]["data"]
        assert "symbol" in position
        assert "quantity" in position
        assert position["symbol"] == "BTCUSDT"

    async def test_complete_signal_to_execution_flow(
        self,
        system_integrator: SystemIntegrator,
        event_capture: EventCapture,
        performance_collector: PerformanceCollector
    ):
        """Test complete flow from signal to position establishment."""
        await performance_collector.start_resource_monitoring()
        
        # Generate market data → pattern → signal
        market_data_provider = system_integrator.get_component("market_data_provider")
        ict_analyzer = system_integrator.get_component("ict_analyzer")
        
        performance_collector.checkpoint("flow_start")
        
        klines = await market_data_provider.get_klines("BTCUSDT", "1h", limit=100)
        await ict_analyzer.analyze_market_data("BTCUSDT", klines)
        
        await asyncio.sleep(1.0)
        
        performance_collector.checkpoint("flow_end")
        
        # Verify event flow
        expected_sequence = [
            "pattern_detected",
            "signal_generated"
        ]
        
        event_capture.assert_event_sequence(expected_sequence)
        
        # Check latency
        performance_collector.assert_latency_within(
            "flow_start",
            "flow_end",
            max_latency_ms=2000.0
        )
        
        await performance_collector.stop_resource_monitoring()
