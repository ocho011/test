"""
Comprehensive tests for StrategyCoordinator component.

Tests multi-timeframe ICT analysis coordination, confluence detection,
and strategy execution integration.
"""

import asyncio
from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from trading_bot.analysis import AnalysisResult, ICTAnalyzer, TrendDirection
from trading_bot.analysis.ict_analyzer import (
    FairValueGap,
    MarketStructure,
    OrderBlock,
)
from trading_bot.core.event_bus import EventBus
from trading_bot.core.events import (
    CandleClosedEvent,
    EventPriority,
    EventType,
    SignalEvent,
    SignalType,
)
from trading_bot.strategies.strategy_coordinator import StrategyCoordinator
from trading_bot.strategies.strategy_system_integration import (
    IntegratedStrategySystem,
)


# Test Fixtures
@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV DataFrame for testing."""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="5min")
    np.random.seed(42)

    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": 50000 + np.cumsum(np.random.randn(100) * 10),
            "high": 50000 + np.cumsum(np.random.randn(100) * 10) + 20,
            "low": 50000 + np.cumsum(np.random.randn(100) * 10) - 20,
            "close": 50000 + np.cumsum(np.random.randn(100) * 10),
            "volume": np.random.randint(100, 1000, 100),
        }
    )

    return df


@pytest.fixture
def mock_event_bus():
    """Create mock EventBus."""
    bus = Mock(spec=EventBus)
    bus.subscribe = AsyncMock(return_value="subscription-123")
    bus.unsubscribe = AsyncMock()
    bus.publish = AsyncMock()
    bus.start = AsyncMock()
    bus.stop = AsyncMock()
    bus.is_running = Mock(return_value=True)
    return bus


@pytest.fixture
def mock_ict_analyzer():
    """Create mock ICTAnalyzer."""
    analyzer = Mock(spec=ICTAnalyzer)

    # Create sample analysis result
    def create_analysis_result(df, timeframe):
        return AnalysisResult(
            timestamp=pd.Timestamp.now(),
            timeframe=timeframe,
            order_blocks=[
                OrderBlock(
                    start_index=10,
                    end_index=15,
                    high_price=51000.0,
                    low_price=50500.0,
                    timestamp=pd.Timestamp.now(),
                    direction=TrendDirection.BULLISH,
                    confidence=0.85,
                )
            ],
            fair_value_gaps=[
                FairValueGap(
                    start_index=20,
                    end_index=22,
                    gap_high=50800.0,
                    gap_low=50600.0,
                    timestamp=pd.Timestamp.now(),
                    direction=TrendDirection.BULLISH,
                    gap_size=200.0,
                )
            ],
            market_structures=[
                MarketStructure(
                    structure_type="BOS",
                    direction=TrendDirection.BULLISH,
                    break_level=51200.0,
                    break_index=30,
                    timestamp=pd.Timestamp.now(),
                    strength=0.9,
                )
            ],
            indicators={},
            pattern_validations=[],
            confluence_signals=[],
            analysis_duration=0.5,
            memory_usage={"peak_mb": 50.0, "current_mb": 45.0},
            summary={},
        )

    analyzer.analyze_comprehensive = Mock(side_effect=create_analysis_result)
    return analyzer


@pytest.fixture
def mock_strategy_system():
    """Create mock IntegratedStrategySystem."""
    system = Mock(spec=IntegratedStrategySystem)
    system.generate_signals = AsyncMock(
        return_value=[
            {
                "direction": "BUY",
                "confidence": 0.75,
                "entry_price": Decimal("51000"),
                "stop_loss": Decimal("50500"),
                "take_profit": Decimal("52000"),
                "quantity": Decimal("0.1"),
                "strategy_name": "ICTStrategy",
                "reasoning": "Bullish order block + FVG confluence",
            }
        ]
    )
    return system


@pytest.fixture
async def coordinator(mock_event_bus, mock_ict_analyzer, mock_strategy_system):
    """Create StrategyCoordinator instance."""
    coord = StrategyCoordinator(
        event_bus=mock_event_bus,
        ict_analyzer=mock_ict_analyzer,
        strategy_system=mock_strategy_system,
        subscribed_intervals=["5m", "15m", "4h", "1d"],
        min_confluence_timeframes=2,
    )
    return coord


# Initialization Tests
class TestStrategyCoordinatorInitialization:
    """Test StrategyCoordinator initialization."""

    def test_initialization_with_defaults(
        self, mock_event_bus, mock_ict_analyzer, mock_strategy_system
    ):
        """Test initialization with default parameters."""
        coord = StrategyCoordinator(
            event_bus=mock_event_bus,
            ict_analyzer=mock_ict_analyzer,
            strategy_system=mock_strategy_system,
        )

        assert coord.event_bus == mock_event_bus
        assert coord.ict_analyzer == mock_ict_analyzer
        assert coord.strategy_system == mock_strategy_system
        assert coord.subscribed_intervals == ["5m", "15m", "4h", "1d"]
        assert coord.min_confluence_timeframes == 2
        assert coord.timeframe_analysis == {}
        assert not coord.is_running()

    def test_initialization_with_custom_intervals(
        self, mock_event_bus, mock_ict_analyzer, mock_strategy_system
    ):
        """Test initialization with custom intervals."""
        custom_intervals = ["1m", "5m", "1h"]
        coord = StrategyCoordinator(
            event_bus=mock_event_bus,
            ict_analyzer=mock_ict_analyzer,
            strategy_system=mock_strategy_system,
            subscribed_intervals=custom_intervals,
            min_confluence_timeframes=3,
        )

        assert coord.subscribed_intervals == custom_intervals
        assert coord.min_confluence_timeframes == 3


# Start/Stop Tests
class TestStrategyCoordinatorStartStop:
    """Test StrategyCoordinator start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_subscribes_to_events(self, coordinator, mock_event_bus):
        """Test that start subscribes to CandleClosedEvent."""
        await coordinator.start()

        mock_event_bus.subscribe.assert_called_once()
        call_args = mock_event_bus.subscribe.call_args
        assert call_args.kwargs["event_type"] == EventType.CANDLE_CLOSED
        assert call_args.kwargs["priority"] == EventPriority.HIGH
        assert callable(call_args.kwargs["handler"])
        assert coordinator.is_running()

    @pytest.mark.asyncio
    async def test_start_when_already_running(self, coordinator, mock_event_bus):
        """Test that starting when already running is handled gracefully."""
        await coordinator.start()
        mock_event_bus.subscribe.reset_mock()

        await coordinator.start()

        # Should not subscribe again
        mock_event_bus.subscribe.assert_not_called()

    @pytest.mark.asyncio
    async def test_stop_unsubscribes_from_events(self, coordinator, mock_event_bus):
        """Test that stop unsubscribes from events."""
        await coordinator.start()
        await coordinator.stop()

        mock_event_bus.unsubscribe.assert_called_once_with("subscription-123")
        assert not coordinator.is_running()

    @pytest.mark.asyncio
    async def test_stop_when_not_running(self, coordinator, mock_event_bus):
        """Test that stopping when not running is handled gracefully."""
        await coordinator.stop()

        # Should not try to unsubscribe
        mock_event_bus.unsubscribe.assert_not_called()


# Candle Event Handling Tests
class TestCandleEventHandling:
    """Test CandleClosedEvent handling."""

    @pytest.mark.asyncio
    async def test_handle_candle_closed_runs_analysis(
        self, coordinator, sample_ohlcv_data, mock_ict_analyzer
    ):
        """Test that candle closed event triggers ICT analysis."""
        event = CandleClosedEvent(
            source="TestMarketData",
            symbol="BTCUSDT",
            interval="5m",
            df=sample_ohlcv_data,
            timestamp=datetime.utcnow(),
        )

        await coordinator._handle_candle_closed(event)

        mock_ict_analyzer.analyze_comprehensive.assert_called_once_with(
            sample_ohlcv_data, "5m"
        )
        assert "5m" in coordinator.timeframe_analysis

    @pytest.mark.asyncio
    async def test_handle_candle_closed_ignores_unsubscribed_intervals(
        self, coordinator, sample_ohlcv_data, mock_ict_analyzer
    ):
        """Test that unsubscribed intervals are ignored."""
        event = CandleClosedEvent(
            source="TestMarketData",
            symbol="BTCUSDT",
            interval="1m",  # Not in subscribed intervals
            df=sample_ohlcv_data,
            timestamp=datetime.utcnow(),
        )

        await coordinator._handle_candle_closed(event)

        mock_ict_analyzer.analyze_comprehensive.assert_not_called()
        assert "1m" not in coordinator.timeframe_analysis

    @pytest.mark.asyncio
    async def test_handle_multiple_timeframes(
        self, coordinator, sample_ohlcv_data, mock_ict_analyzer
    ):
        """Test handling events for multiple timeframes."""
        intervals = ["5m", "15m", "4h"]

        for interval in intervals:
            event = CandleClosedEvent(
                source="TestMarketData",
                symbol="BTCUSDT",
                interval=interval,
                df=sample_ohlcv_data,
                timestamp=datetime.utcnow(),
            )
            await coordinator._handle_candle_closed(event)

        assert len(coordinator.timeframe_analysis) == 3
        for interval in intervals:
            assert interval in coordinator.timeframe_analysis

    @pytest.mark.asyncio
    async def test_handle_candle_closed_error_handling(
        self, coordinator, sample_ohlcv_data, mock_ict_analyzer
    ):
        """Test error handling in candle event processing."""
        mock_ict_analyzer.analyze_comprehensive.side_effect = Exception(
            "Analysis failed"
        )

        event = CandleClosedEvent(
            source="TestMarketData",
            symbol="BTCUSDT",
            interval="5m",
            df=sample_ohlcv_data,
            timestamp=datetime.utcnow(),
        )

        # Should not raise exception
        await coordinator._handle_candle_closed(event)

        # Should not store failed analysis
        assert "5m" not in coordinator.timeframe_analysis


# Confluence Detection Tests
class TestConfluenceDetection:
    """Test multi-timeframe confluence detection."""

    @pytest.mark.asyncio
    async def test_confluence_with_insufficient_timeframes(
        self, coordinator, sample_ohlcv_data
    ):
        """Test that confluence is not met with insufficient timeframes."""
        # Add only 1 timeframe (need 2)
        event = CandleClosedEvent(
            source="TestMarketData",
            symbol="BTCUSDT",
            interval="5m",
            df=sample_ohlcv_data,
            timestamp=datetime.utcnow(),
        )
        await coordinator._handle_candle_closed(event)

        assert not coordinator._check_confluence()

    @pytest.mark.asyncio
    async def test_confluence_with_aligned_trends(
        self, coordinator, sample_ohlcv_data, mock_ict_analyzer
    ):
        """Test confluence detection with aligned bullish trends."""
        # Create analyzer that returns bullish trends
        def create_bullish_result(df, timeframe):
            return AnalysisResult(
                timestamp=pd.Timestamp.now(),
                timeframe=timeframe,
                order_blocks=[],
                fair_value_gaps=[],
                market_structures=[
                    MarketStructure(
                structure_type="BOS",
                direction=TrendDirection.BULLISH,
                break_level=51200.0,
                break_index=30,
                timestamp=pd.Timestamp.now(),
                strength=0.9,
            )
                ],
                indicators={},
                pattern_validations=[],
                confluence_signals=[],
                analysis_duration=0.5,
                memory_usage={"peak_mb": 50.0, "current_mb": 45.0},
                summary={},
            )

        mock_ict_analyzer.analyze_comprehensive = Mock(
            side_effect=create_bullish_result
        )

        # Add 2 timeframes with bullish trends
        for interval in ["5m", "15m"]:
            event = CandleClosedEvent(
                source="TestMarketData",
                symbol="BTCUSDT",
                interval=interval,
                df=sample_ohlcv_data,
                timestamp=datetime.utcnow(),
            )
            await coordinator._handle_candle_closed(event)

        assert coordinator._check_confluence()

    @pytest.mark.asyncio
    async def test_confluence_with_mixed_trends(
        self, coordinator, sample_ohlcv_data, mock_ict_analyzer
    ):
        """Test confluence detection with mixed trends."""
        results_generated = []

        def create_mixed_result(df, timeframe):
            # Alternate between bullish and bearish
            direction = (
                TrendDirection.BULLISH
                if len(results_generated) % 2 == 0
                else TrendDirection.BEARISH
            )
            results_generated.append(direction)

            return AnalysisResult(
                timestamp=pd.Timestamp.now(),
                timeframe=timeframe,
                order_blocks=[],
                fair_value_gaps=[],
                market_structures=[
                    MarketStructure(
                        structure_type="BOS",
                        direction=direction,
                        break_level=51200.0,
                        break_index=30,
                        timestamp=pd.Timestamp.now(),
                        strength=0.9,
                    )
                ],
                indicators={},
                pattern_validations=[],
                confluence_signals=[],
                analysis_duration=0.5,
                memory_usage={"peak_mb": 50.0, "current_mb": 45.0},
                summary={},
            )

        mock_ict_analyzer.analyze_comprehensive = Mock(side_effect=create_mixed_result)

        # Add 2 timeframes with opposite trends
        for interval in ["5m", "15m"]:
            event = CandleClosedEvent(
                source="TestMarketData",
                symbol="BTCUSDT",
                interval=interval,
                df=sample_ohlcv_data,
                timestamp=datetime.utcnow(),
            )
            await coordinator._handle_candle_closed(event)

        # 50/50 split should not meet confluence (need >50%)
        assert not coordinator._check_confluence()

    @pytest.mark.asyncio
    async def test_confluence_with_majority_alignment(
        self, coordinator, sample_ohlcv_data, mock_ict_analyzer
    ):
        """Test confluence with majority trend alignment."""
        results_generated = []

        def create_mostly_bullish_result(df, timeframe):
            # 3 bullish, 1 bearish
            direction = (
                TrendDirection.BEARISH
                if len(results_generated) == 2
                else TrendDirection.BULLISH
            )
            results_generated.append(direction)

            return AnalysisResult(
                timestamp=pd.Timestamp.now(),
                timeframe=timeframe,
                order_blocks=[],
                fair_value_gaps=[],
                market_structures=[
                    MarketStructure(
                        structure_type="BOS",
                        direction=direction,
                        break_level=51200.0,
                        break_index=30,
                        timestamp=pd.Timestamp.now(),
                        strength=0.9,
                    )
                ],
                indicators={},
                pattern_validations=[],
                confluence_signals=[],
                analysis_duration=0.5,
                memory_usage={"peak_mb": 50.0, "current_mb": 45.0},
                summary={},
            )

        mock_ict_analyzer.analyze_comprehensive = Mock(
            side_effect=create_mostly_bullish_result
        )

        # Add 4 timeframes (3 bullish, 1 bearish = 75% alignment)
        for interval in ["5m", "15m", "4h", "1d"]:
            event = CandleClosedEvent(
                source="TestMarketData",
                symbol="BTCUSDT",
                interval=interval,
                df=sample_ohlcv_data,
                timestamp=datetime.utcnow(),
            )
            await coordinator._handle_candle_closed(event)

        # 75% > 50% should meet confluence
        assert coordinator._check_confluence()


# Signal Generation Tests
class TestSignalGeneration:
    """Test signal generation and publishing."""

    @pytest.mark.asyncio
    async def test_generate_and_publish_signals(
        self,
        coordinator,
        sample_ohlcv_data,
        mock_strategy_system,
        mock_event_bus,
    ):
        """Test signal generation and publishing."""
        await coordinator._generate_and_publish_signals("BTCUSDT", sample_ohlcv_data)

        mock_strategy_system.generate_signals.assert_called_once_with(
            sample_ohlcv_data
        )
        mock_event_bus.publish.assert_called_once()

        # Verify published event is SignalEvent
        published_event = mock_event_bus.publish.call_args[0][0]
        assert isinstance(published_event, SignalEvent)
        assert published_event.symbol == "BTCUSDT"
        assert published_event.signal_type == SignalType.BUY
        assert published_event.confidence == 0.75

    @pytest.mark.asyncio
    async def test_signal_conversion_buy(self, coordinator):
        """Test conversion of BUY signal to SignalEvent."""
        signal = {
            "direction": "BUY",
            "confidence": 0.8,
            "entry_price": Decimal("51000"),
            "stop_loss": Decimal("50500"),
            "take_profit": Decimal("52000"),
            "quantity": Decimal("0.1"),
            "strategy_name": "ICTStrategy",
            "reasoning": "Test reasoning",
        }

        event = coordinator._convert_to_signal_event("BTCUSDT", signal)

        assert event.symbol == "BTCUSDT"
        assert event.signal_type == SignalType.BUY
        assert event.confidence == 0.8
        assert event.entry_price == Decimal("51000")
        assert event.stop_loss == Decimal("50500")
        assert event.take_profit == Decimal("52000")

    @pytest.mark.asyncio
    async def test_signal_conversion_sell(self, coordinator):
        """Test conversion of SELL signal to SignalEvent."""
        signal = {
            "direction": "SELL",
            "confidence": 0.7,
            "entry_price": Decimal("51000"),
            "strategy_name": "ICTStrategy",
        }

        event = coordinator._convert_to_signal_event("BTCUSDT", signal)

        assert event.signal_type == SignalType.SELL
        assert event.confidence == 0.7

    @pytest.mark.asyncio
    async def test_signal_generation_error_handling(
        self, coordinator, sample_ohlcv_data, mock_strategy_system, mock_event_bus
    ):
        """Test error handling in signal generation."""
        mock_strategy_system.generate_signals.side_effect = Exception(
            "Strategy failed"
        )

        # Should not raise exception
        await coordinator._generate_and_publish_signals("BTCUSDT", sample_ohlcv_data)

        # Should not publish any events
        mock_event_bus.publish.assert_not_called()


# Status and Query Tests
class TestStatusAndQuery:
    """Test status retrieval and query methods."""

    def test_get_timeframe_analysis(self, coordinator, sample_ohlcv_data):
        """Test retrieving specific timeframe analysis."""
        # Manually add analysis result
        result = Mock(spec=AnalysisResult)
        coordinator.timeframe_analysis["5m"] = result

        retrieved = coordinator.get_timeframe_analysis("5m")
        assert retrieved == result

        # Non-existent timeframe
        assert coordinator.get_timeframe_analysis("1h") is None

    def test_get_all_timeframe_analysis(self, coordinator):
        """Test retrieving all timeframe analyses."""
        result1 = Mock(spec=AnalysisResult)
        result2 = Mock(spec=AnalysisResult)
        coordinator.timeframe_analysis["5m"] = result1
        coordinator.timeframe_analysis["15m"] = result2

        all_results = coordinator.get_all_timeframe_analysis()

        assert len(all_results) == 2
        assert all_results["5m"] == result1
        assert all_results["15m"] == result2

    def test_get_status(self, coordinator):
        """Test status retrieval."""
        status = coordinator.get_status()

        assert "running" in status
        assert "subscribed_intervals" in status
        assert "min_confluence_timeframes" in status
        assert "analyzed_timeframes" in status
        assert "timeframe_count" in status
        assert status["subscribed_intervals"] == ["5m", "15m", "4h", "1d"]
        assert status["min_confluence_timeframes"] == 2

    @pytest.mark.asyncio
    async def test_status_after_start(self, coordinator):
        """Test status after coordinator is started."""
        await coordinator.start()
        status = coordinator.get_status()

        assert status["running"] is True
        assert status["subscription_count"] == 1


# Integration Tests
class TestStrategyCoordinatorIntegration:
    """Integration tests with real event flow."""

    @pytest.mark.asyncio
    async def test_full_workflow_with_confluence(
        self,
        coordinator,
        sample_ohlcv_data,
        mock_ict_analyzer,
        mock_strategy_system,
        mock_event_bus,
    ):
        """Test full workflow from candle event to signal publication."""

        # Setup: Create analyzer that returns bullish trends
        def create_bullish_result(df, timeframe):
            return AnalysisResult(
                timestamp=pd.Timestamp.now(),
                timeframe=timeframe,
                order_blocks=[],
                fair_value_gaps=[],
                market_structures=[
                    MarketStructure(
                structure_type="BOS",
                direction=TrendDirection.BULLISH,
                break_level=51200.0,
                break_index=30,
                timestamp=pd.Timestamp.now(),
                strength=0.9,
            )
                ],
                indicators={},
                pattern_validations=[],
                confluence_signals=[],
                analysis_duration=0.5,
                memory_usage={},
                summary={},
            )

        mock_ict_analyzer.analyze_comprehensive = Mock(
            side_effect=create_bullish_result
        )

        await coordinator.start()

        # Simulate 2 candle closed events (meet confluence threshold)
        for interval in ["5m", "15m"]:
            event = CandleClosedEvent(
                source="TestMarketData",
                symbol="BTCUSDT",
                interval=interval,
                df=sample_ohlcv_data,
                timestamp=datetime.utcnow(),
            )
            await coordinator._handle_candle_closed(event)

        # Verify: Analysis was run for both timeframes
        assert mock_ict_analyzer.analyze_comprehensive.call_count == 2

        # Verify: Signal was generated and published
        mock_strategy_system.generate_signals.assert_called_once()
        mock_event_bus.publish.assert_called_once()

        # Verify: Published event is correct
        published_event = mock_event_bus.publish.call_args[0][0]
        assert isinstance(published_event, SignalEvent)
        assert published_event.symbol == "BTCUSDT"

    @pytest.mark.asyncio
    async def test_no_signal_without_confluence(
        self,
        coordinator,
        sample_ohlcv_data,
        mock_ict_analyzer,
        mock_strategy_system,
        mock_event_bus,
    ):
        """Test that no signal is generated without confluence."""
        await coordinator.start()

        # Simulate only 1 candle event (insufficient for confluence)
        event = CandleClosedEvent(
            source="TestMarketData",
            symbol="BTCUSDT",
            interval="5m",
            df=sample_ohlcv_data,
            timestamp=datetime.utcnow(),
        )
        await coordinator._handle_candle_closed(event)

        # Verify: Analysis was run
        mock_ict_analyzer.analyze_comprehensive.assert_called_once()

        # Verify: No signal was generated
        mock_strategy_system.generate_signals.assert_not_called()
        mock_event_bus.publish.assert_not_called()
