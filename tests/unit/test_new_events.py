"""
Unit tests for new event types: CandleClosedEvent and RiskApprovedOrderEvent.

Tests cover:
- Event instantiation and validation
- Field validation and constraints
- Serialization and deserialization
- Event type mapping integration
"""

import json
from datetime import datetime
from decimal import Decimal

import pandas as pd
import pytest

from trading_bot.core.events import (
    CandleClosedEvent,
    EventPriority,
    EventType,
    RiskApprovedOrderEvent,
    SignalEvent,
    SignalType,
    create_event_from_dict,
    serialize_event,
)


class TestCandleClosedEvent:
    """Tests for CandleClosedEvent."""

    @pytest.fixture
    def sample_df(self):
        """Create sample OHLCV DataFrame."""
        return pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0],
                "high": [102.0, 103.0, 104.0],
                "low": [99.0, 100.0, 101.0],
                "close": [101.0, 102.0, 103.0],
                "volume": [1000.0, 1100.0, 1200.0],
            }
        )

    def test_event_creation(self, sample_df):
        """Test basic CandleClosedEvent creation."""
        event = CandleClosedEvent(
            symbol="BTCUSDT",
            interval="5m",
            df=sample_df,
            source="test_aggregator",
        )

        assert event.symbol == "BTCUSDT"
        assert event.interval == "5m"
        assert len(event.df) == 3
        assert event.event_type == EventType.CANDLE_CLOSED
        assert event.priority == EventPriority.HIGH
        assert event.source == "test_aggregator"
        assert isinstance(event.timestamp, datetime)

    def test_interval_validation(self, sample_df):
        """Test interval validation accepts valid values."""
        valid_intervals = ["1m", "5m", "15m", "1h", "4h", "1d", "1w"]

        for interval in valid_intervals:
            event = CandleClosedEvent(
                symbol="BTCUSDT", interval=interval, df=sample_df, source="test"
            )
            assert event.interval == interval

    def test_invalid_interval_raises_error(self, sample_df):
        """Test that invalid interval raises validation error."""
        with pytest.raises(ValueError, match="Invalid interval"):
            CandleClosedEvent(
                symbol="BTCUSDT", interval="invalid", df=sample_df, source="test"
            )

    def test_dataframe_serialization(self, sample_df):
        """Test DataFrame serialization to dict."""
        event = CandleClosedEvent(
            symbol="BTCUSDT", interval="5m", df=sample_df, source="test"
        )

        event_dict = event.model_dump()

        # DataFrame is not auto-serialized by model_dump, need custom serializer
        assert "df" in event_dict
        assert isinstance(event_dict["df"], pd.DataFrame)
        assert len(event_dict["df"]) == 3

    def test_priority_is_high(self, sample_df):
        """Test that CandleClosedEvent has HIGH priority by default."""
        event = CandleClosedEvent(
            symbol="BTCUSDT", interval="5m", df=sample_df, source="test"
        )

        assert event.priority == EventPriority.HIGH

    def test_multiple_symbols(self, sample_df):
        """Test events for different symbols."""
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]

        for symbol in symbols:
            event = CandleClosedEvent(
                symbol=symbol, interval="5m", df=sample_df, source="test"
            )
            assert event.symbol == symbol

    def test_serialize_event_function(self, sample_df):
        """Test serialize_event helper function."""
        event = CandleClosedEvent(
            symbol="BTCUSDT", interval="15m", df=sample_df, source="test"
        )

        serialized = serialize_event(event)

        assert serialized["symbol"] == "BTCUSDT"
        assert serialized["interval"] == "15m"
        # EventType enum is not auto-converted to string in model_dump
        assert serialized["event_type"] == EventType.CANDLE_CLOSED
        assert isinstance(serialized["df"], pd.DataFrame)


class TestRiskApprovedOrderEvent:
    """Tests for RiskApprovedOrderEvent."""

    @pytest.fixture
    def sample_signal(self):
        """Create sample SignalEvent."""
        return SignalEvent(
            symbol="BTCUSDT",
            signal_type=SignalType.BUY,
            confidence=0.85,
            entry_price=Decimal("50000.00"),
            stop_loss=Decimal("49000.00"),
            take_profit=Decimal("52000.00"),
            quantity=Decimal("0.1"),
            strategy_name="test_strategy",
            source="strategy_coordinator",
        )

    def test_event_creation(self, sample_signal):
        """Test basic RiskApprovedOrderEvent creation."""
        event = RiskApprovedOrderEvent(
            signal=sample_signal,
            approved_quantity=Decimal("0.08"),
            risk_params={
                "max_position_size": Decimal("1.0"),
                "stop_loss_pct": 0.02,
                "risk_per_trade": 0.01,
            },
            source="risk_manager",
        )

        assert event.signal.symbol == "BTCUSDT"
        assert event.signal.signal_type == SignalType.BUY
        assert event.approved_quantity == Decimal("0.08")
        assert event.risk_params["stop_loss_pct"] == 0.02
        assert event.event_type == EventType.RISK_APPROVED_ORDER
        assert event.priority == EventPriority.NORMAL
        assert event.source == "risk_manager"

    def test_approved_quantity_validation(self, sample_signal):
        """Test that approved_quantity must be positive."""
        with pytest.raises(ValueError):
            RiskApprovedOrderEvent(
                signal=sample_signal,
                approved_quantity=Decimal("-0.1"),
                risk_params={},
                source="test",
            )

        with pytest.raises(ValueError):
            RiskApprovedOrderEvent(
                signal=sample_signal,
                approved_quantity=Decimal("0"),
                risk_params={},
                source="test",
            )

    def test_quantity_reduced_from_signal(self, sample_signal):
        """Test that approved quantity can be less than signal quantity."""
        # Signal suggests 0.1
        assert sample_signal.quantity == Decimal("0.1")

        # Risk manager approves only 0.05
        event = RiskApprovedOrderEvent(
            signal=sample_signal,
            approved_quantity=Decimal("0.05"),
            risk_params={"reason": "position_limit"},
            source="risk_manager",
        )

        assert event.approved_quantity < sample_signal.quantity
        assert event.approved_quantity == Decimal("0.05")

    def test_risk_params_storage(self, sample_signal):
        """Test that risk_params can store various risk metrics."""
        risk_params = {
            "max_position_size": Decimal("2.0"),
            "current_exposure": Decimal("1.5"),
            "stop_loss_pct": 0.02,
            "risk_reward_ratio": 2.0,
            "account_balance": Decimal("10000.00"),
            "risk_per_trade": 0.01,
        }

        event = RiskApprovedOrderEvent(
            signal=sample_signal,
            approved_quantity=Decimal("0.1"),
            risk_params=risk_params,
            source="risk_manager",
        )

        assert event.risk_params == risk_params
        assert event.risk_params["risk_reward_ratio"] == 2.0

    def test_priority_is_normal(self, sample_signal):
        """Test that RiskApprovedOrderEvent has NORMAL priority by default."""
        event = RiskApprovedOrderEvent(
            signal=sample_signal,
            approved_quantity=Decimal("0.1"),
            risk_params={},
            source="test",
        )

        assert event.priority == EventPriority.NORMAL

    def test_signal_preservation(self, sample_signal):
        """Test that original signal is preserved in the event."""
        event = RiskApprovedOrderEvent(
            signal=sample_signal,
            approved_quantity=Decimal("0.1"),
            risk_params={},
            source="test",
        )

        # All signal fields should be accessible
        assert event.signal.symbol == "BTCUSDT"
        assert event.signal.signal_type == SignalType.BUY
        assert event.signal.confidence == 0.85
        assert event.signal.entry_price == Decimal("50000.00")
        assert event.signal.stop_loss == Decimal("49000.00")
        assert event.signal.strategy_name == "test_strategy"

    def test_serialization(self, sample_signal):
        """Test event serialization."""
        event = RiskApprovedOrderEvent(
            signal=sample_signal,
            approved_quantity=Decimal("0.1"),
            risk_params={"max_size": Decimal("1.0")},
            source="test",
        )

        event_dict = event.model_dump()

        assert event_dict["event_type"] == EventType.RISK_APPROVED_ORDER
        assert "signal" in event_dict
        assert event_dict["signal"]["symbol"] == "BTCUSDT"
        assert event_dict["approved_quantity"] == Decimal("0.1")


class TestEventTypeMapping:
    """Tests for event type mapping and deserialization."""

    def test_candle_closed_in_mapping(self):
        """Test that CandleClosedEvent is in EVENT_TYPE_MAPPING."""
        from trading_bot.core.events import EVENT_TYPE_MAPPING

        assert EventType.CANDLE_CLOSED in EVENT_TYPE_MAPPING
        assert EVENT_TYPE_MAPPING[EventType.CANDLE_CLOSED] == CandleClosedEvent

    def test_risk_approved_order_in_mapping(self):
        """Test that RiskApprovedOrderEvent is in EVENT_TYPE_MAPPING."""
        from trading_bot.core.events import EVENT_TYPE_MAPPING

        assert EventType.RISK_APPROVED_ORDER in EVENT_TYPE_MAPPING
        assert (
            EVENT_TYPE_MAPPING[EventType.RISK_APPROVED_ORDER] == RiskApprovedOrderEvent
        )

    def test_create_event_from_dict_candle(self):
        """Test creating CandleClosedEvent from dictionary."""
        # Use DataFrame directly since pydantic doesn't auto-convert list to DataFrame
        df = pd.DataFrame([
            {"open": 100.0, "high": 102.0, "low": 99.0, "close": 101.0, "volume": 1000.0}
        ])

        event_data = {
            "event_type": EventType.CANDLE_CLOSED,
            "symbol": "BTCUSDT",
            "interval": "5m",
            "df": df,
            "source": "test",
        }

        event = create_event_from_dict(event_data)

        assert isinstance(event, CandleClosedEvent)
        assert event.symbol == "BTCUSDT"
        assert event.interval == "5m"

    def test_create_event_from_dict_risk_approved(self):
        """Test creating RiskApprovedOrderEvent from dictionary."""
        # Create proper signal object first
        signal = SignalEvent(
            symbol="BTCUSDT",
            signal_type=SignalType.BUY,
            confidence=0.85,
            strategy_name="test",
            source="test",
        )

        event_data = {
            "event_type": EventType.RISK_APPROVED_ORDER,
            "signal": signal,
            "approved_quantity": Decimal("0.1"),
            "risk_params": {"max_size": Decimal("1.0")},
            "source": "test",
        }

        event = create_event_from_dict(event_data)

        assert isinstance(event, RiskApprovedOrderEvent)
        assert event.approved_quantity == Decimal("0.1")


class TestEventIntegration:
    """Integration tests for event workflow."""

    def test_candle_to_signal_to_order_workflow(self):
        """Test event flow: Candle → Signal → RiskApproved."""
        # 1. CandleClosedEvent
        df = pd.DataFrame(
            {
                "open": [50000.0],
                "high": [51000.0],
                "low": [49500.0],
                "close": [50800.0],
                "volume": [100.0],
            }
        )

        candle_event = CandleClosedEvent(
            symbol="BTCUSDT", interval="5m", df=df, source="aggregator"
        )

        # 2. SignalEvent (generated from candle analysis)
        signal_event = SignalEvent(
            symbol=candle_event.symbol,
            signal_type=SignalType.BUY,
            confidence=0.9,
            entry_price=Decimal("50800.00"),
            stop_loss=Decimal("49800.00"),
            take_profit=Decimal("52800.00"),
            quantity=Decimal("0.1"),
            strategy_name="ict_strategy",
            source="strategy_coordinator",
        )

        # 3. RiskApprovedOrderEvent
        risk_event = RiskApprovedOrderEvent(
            signal=signal_event,
            approved_quantity=Decimal("0.08"),  # Reduced by risk manager
            risk_params={
                "max_position_size": Decimal("1.0"),
                "current_exposure": Decimal("0.92"),
                "risk_per_trade": 0.01,
            },
            source="risk_manager",
        )

        # Verify workflow
        assert candle_event.event_type == EventType.CANDLE_CLOSED
        assert signal_event.event_type == EventType.SIGNAL
        assert risk_event.event_type == EventType.RISK_APPROVED_ORDER

        # Verify data flows through
        assert risk_event.signal.symbol == candle_event.symbol
        assert risk_event.approved_quantity < signal_event.quantity
