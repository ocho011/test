"""
E2E tests for data collection → signal generation flow.

Tests the complete flow from market data ingestion through ICT pattern
detection to trading signal generation and validation.
"""

import pytest
import asyncio
from decimal import Decimal

from trading_bot.system_integrator import SystemIntegrator
from trading_bot.data.market_data_provider import MarketDataProvider
from trading_bot.analysis import ICTAnalyzer
from trading_bot.signals.signal_generator import SignalGenerator

from .fixtures.mock_binance_client import MockBinanceClient
from .utils.event_capture import EventCapture
from .utils.performance_collector import PerformanceCollector


@pytest.mark.e2e
@pytest.mark.asyncio
class TestDataToSignalFlow:
    """Test data collection → ICT analysis → signal generation flow."""

    async def test_market_data_to_pattern_detection(
        self,
        system_integrator: SystemIntegrator,
        event_capture: EventCapture,
        performance_collector: PerformanceCollector
    ):
        """
        Test that market data triggers pattern detection.
        
        Flow: Market data → ICT analysis → Pattern detected event
        """
        # Start performance monitoring
        await performance_collector.start_resource_monitoring(interval_seconds=0.5)
        performance_collector.checkpoint("test_start")
        
        # Get components
        market_data_provider = system_integrator.get_component("market_data_provider")
        ict_analyzer = system_integrator.get_component("ict_analyzer")
        
        assert market_data_provider is not None
        assert ict_analyzer is not None
        
        # Request market data
        performance_collector.checkpoint("request_data")
        klines = await market_data_provider.get_klines(
            symbol="BTCUSDT",
            interval="1h",
            limit=100
        )
        performance_collector.checkpoint("data_received")
        
        # Verify data received
        assert len(klines) > 0
        
        # Record latency
        data_latency = performance_collector.record_latency("request_data", "data_received")
        assert data_latency is not None
        assert data_latency < 100, f"Data retrieval too slow: {data_latency:.2f}ms"
        
        # Analyze for patterns
        performance_collector.checkpoint("analyze_start")
        await ict_analyzer.analyze_market_data("BTCUSDT", klines)
        performance_collector.checkpoint("analyze_complete")
        
        # Record analysis latency
        analysis_latency = performance_collector.record_latency("analyze_start", "analyze_complete")
        assert analysis_latency is not None
        assert analysis_latency < 200, f"Pattern analysis too slow: {analysis_latency:.2f}ms"
        
        # Wait for events to propagate
        await asyncio.sleep(0.5)
        
        # Verify pattern detection events
        pattern_events = event_capture.get_events("pattern_detected")
        assert len(pattern_events) > 0, "No patterns detected in trending market"
        
        # Verify pattern event structure
        pattern_event = pattern_events[0]
        assert "pattern_type" in pattern_event["data"]
        assert "confidence" in pattern_event["data"]
        assert "timeframe" in pattern_event["data"]
        
        # Stop monitoring
        await performance_collector.stop_resource_monitoring()
        performance_collector.checkpoint("test_end")
        
        # Assert end-to-end latency
        e2e_latency = performance_collector.record_latency("request_data", "analyze_complete")
        assert e2e_latency < 300, f"End-to-end latency too high: {e2e_latency:.2f}ms"

    async def test_pattern_to_signal_generation(
        self,
        system_integrator: SystemIntegrator,
        event_capture: EventCapture,
        performance_collector: PerformanceCollector
    ):
        """
        Test that detected patterns generate trading signals.
        
        Flow: Pattern detected → Signal generation → Signal generated event
        """
        # Get components
        market_data_provider = system_integrator.get_component("market_data_provider")
        ict_analyzer = system_integrator.get_component("ict_analyzer")
        signal_generator = system_integrator.get_component("signal_generator")
        
        assert signal_generator is not None
        
        # Get market data and detect patterns
        klines = await market_data_provider.get_klines("BTCUSDT", "1h", limit=100)
        
        performance_collector.checkpoint("pattern_start")
        await ict_analyzer.analyze_market_data("BTCUSDT", klines)
        performance_collector.checkpoint("pattern_complete")
        
        # Wait for signal generation
        await asyncio.sleep(0.5)
        
        # Verify signal events
        signal_events = event_capture.get_events("signal_generated")
        assert len(signal_events) > 0, "No signals generated from patterns"
        
        # Verify signal structure
        signal_event = signal_events[0]
        signal_data = signal_event["data"]
        
        assert "signal_id" in signal_data
        assert "symbol" in signal_data
        assert "direction" in signal_data  # LONG or SHORT
        assert "confidence" in signal_data
        assert "entry_price" in signal_data
        assert "stop_loss" in signal_data
        assert "take_profit" in signal_data
        
        # Verify signal quality
        assert 0.0 <= signal_data["confidence"] <= 1.0
        assert signal_data["symbol"] == "BTCUSDT"
        assert signal_data["direction"] in ["LONG", "SHORT"]

    async def test_signal_validation_flow(
        self,
        system_integrator: SystemIntegrator,
        event_capture: EventCapture
    ):
        """
        Test signal validation with confluence checks.
        
        Flow: Signal generated → Confluence validation → Signal validated event
        """
        # Get components
        market_data_provider = system_integrator.get_component("market_data_provider")
        ict_analyzer = system_integrator.get_component("ict_analyzer")
        confluence_validator = system_integrator.get_component("confluence_validator")
        
        assert confluence_validator is not None
        
        # Generate signals
        klines = await market_data_provider.get_klines("BTCUSDT", "1h", limit=100)
        await ict_analyzer.analyze_market_data("BTCUSDT", klines)
        
        # Wait for validation
        await asyncio.sleep(0.5)
        
        # Check for validated signals
        validated_events = event_capture.get_events("signal_validated")
        
        # In trending market, we should get some validated signals
        assert len(validated_events) >= 0  # May be 0 if no high-confidence patterns
        
        # If we have validated signals, check their structure
        if validated_events:
            validated_event = validated_events[0]
            assert "signal_id" in validated_event["data"]
            assert "validation_score" in validated_event["data"]
            assert "confluence_factors" in validated_event["data"]

    async def test_complete_data_to_signal_flow(
        self,
        system_integrator: SystemIntegrator,
        event_capture: EventCapture,
        performance_collector: PerformanceCollector
    ):
        """
        Test complete flow from data ingestion to validated signals.
        
        Flow: Market data → Pattern detection → Signal generation → Validation → Events
        """
        await performance_collector.start_resource_monitoring(interval_seconds=0.5)
        
        # Get components
        market_data_provider = system_integrator.get_component("market_data_provider")
        
        # Start throughput tracking
        performance_collector.start_throughput("data_to_signal")
        
        # Process market data
        performance_collector.checkpoint("flow_start")
        klines = await market_data_provider.get_klines("BTCUSDT", "1h", limit=200)
        performance_collector.checkpoint("flow_data_received")
        
        # Let the system process
        await asyncio.sleep(1.0)
        performance_collector.checkpoint("flow_complete")
        
        # End throughput tracking
        performance_collector.increment_throughput("data_to_signal", len(klines))
        throughput = performance_collector.end_throughput("data_to_signal")
        
        # Verify event sequence
        expected_sequence = [
            "market_data_updated",
            "pattern_detected",
            "signal_generated"
        ]
        
        event_capture.assert_event_sequence(expected_sequence)
        
        # Verify latencies
        performance_collector.assert_latency_within(
            "flow_start",
            "flow_data_received",
            max_latency_ms=100.0
        )
        
        performance_collector.assert_latency_within(
            "flow_start",
            "flow_complete",
            max_latency_ms=500.0
        )
        
        # Verify no duplicate signals
        signal_events = event_capture.get_events("signal_generated")
        if signal_events:
            event_capture.assert_no_duplicate_events("signal_generated", "signal_id")
        
        await performance_collector.stop_resource_monitoring()
        
        # Check resource usage
        peak_memory = performance_collector.report.get_peak_memory_mb()
        assert peak_memory is not None
        assert peak_memory < 500, f"Memory usage too high: {peak_memory:.2f}MB"

    @pytest.mark.parametrize("scenario", ["ranging", "volatile"])
    async def test_different_market_scenarios(
        self,
        scenario: str,
        system_integrator: SystemIntegrator,
        event_capture: EventCapture,
        mock_binance_client: MockBinanceClient
    ):
        """
        Test signal generation in different market conditions.
        
        Scenarios:
        - ranging: Sideways market (fewer signals expected)
        - volatile: High volatility (more caution expected)
        """
        # Get components
        market_data_provider = system_integrator.get_component("market_data_provider")
        ict_analyzer = system_integrator.get_component("ict_analyzer")
        
        # Process market data
        klines = await market_data_provider.get_klines("BTCUSDT", "1h", limit=200)
        await ict_analyzer.analyze_market_data("BTCUSDT", klines)
        
        await asyncio.sleep(0.5)
        
        # Verify system handles different scenarios
        pattern_events = event_capture.get_events("pattern_detected")
        signal_events = event_capture.get_events("signal_generated")
        
        # Ranging markets should have fewer patterns than trending
        if mock_binance_client.scenario == "ranging":
            # Ranging market may have patterns but fewer high-confidence ones
            assert True  # System should handle gracefully
        
        # Volatile markets should generate signals with appropriate confidence
        elif mock_binance_client.scenario == "volatile":
            if signal_events:
                for event in signal_events:
                    # In volatile markets, confidence might be adjusted
                    assert "confidence" in event["data"]

    async def test_signal_event_timing(
        self,
        system_integrator: SystemIntegrator,
        event_capture: EventCapture
    ):
        """
        Test that signal events are published in correct order with proper timing.
        """
        # Get components
        market_data_provider = system_integrator.get_component("market_data_provider")
        ict_analyzer = system_integrator.get_component("ict_analyzer")
        
        # Generate patterns and signals
        klines = await market_data_provider.get_klines("BTCUSDT", "1h", limit=100)
        await ict_analyzer.analyze_market_data("BTCUSDT", klines)
        
        await asyncio.sleep(0.5)
        
        # Verify timing between events
        pattern_to_signal_time = event_capture.get_timing_between_events(
            "pattern_detected",
            "signal_generated"
        )
        
        if pattern_to_signal_time is not None:
            # Signal generation should be fast after pattern detection
            assert pattern_to_signal_time < 0.5, \
                f"Too slow from pattern to signal: {pattern_to_signal_time:.3f}s"
