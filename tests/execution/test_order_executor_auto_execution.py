"""
Tests for OrderExecutor automatic order execution from RiskApprovedOrderEvent.
"""

import asyncio
from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.trading_bot.core.event_bus import EventBus
from src.trading_bot.core.events import (
    EventType,
    OrderSide,
    OrderStatus,
    OrderType,
    RiskApprovedOrderEvent,
    SignalEvent,
    SignalType,
)
from src.trading_bot.data.binance_client import BinanceClient
from src.trading_bot.execution.order_executor import OrderExecutor


@pytest.fixture
def mock_binance_client():
    """Create a mock Binance client."""
    client = AsyncMock(spec=BinanceClient)

    # Mock futures_create_order response
    client.futures_create_order = AsyncMock(
        return_value={
            "orderId": "12345",
            "status": "FILLED",
            "executedQty": "0.1",
            "avgPrice": "50000.00",
        }
    )

    # Mock futures_symbol_ticker response
    client.futures_symbol_ticker = AsyncMock(
        return_value={"price": "50000.00"}
    )

    return client


@pytest.fixture
def event_bus():
    """Create an event bus for testing."""
    return EventBus()


@pytest.fixture
async def order_executor(mock_binance_client, event_bus):
    """Create an OrderExecutor instance."""
    executor = OrderExecutor(
        binance_client=mock_binance_client,
        event_bus=event_bus,
        max_retries=2,
        retry_delay=0.1,
    )
    await executor.start()
    yield executor
    await executor.stop()


@pytest.mark.asyncio
async def test_subscribe_to_risk_approved_order_event(order_executor, event_bus):
    """Test that OrderExecutor subscribes to RiskApprovedOrderEvent on start."""
    # Check that the subscription exists
    assert len(event_bus._subscriptions) > 0

    # Check that at least one subscription is for RISK_APPROVED_ORDER event type
    has_risk_subscription = False
    for sub in event_bus._subscriptions:
        if sub.event_types and EventType.RISK_APPROVED_ORDER in sub.event_types:
            has_risk_subscription = True
            break

    assert has_risk_subscription, "OrderExecutor should subscribe to RISK_APPROVED_ORDER events"


@pytest.mark.asyncio
async def test_handle_buy_signal(order_executor, mock_binance_client):
    """Test automatic order execution for BUY signal."""
    # Create a BUY signal
    signal = SignalEvent(
        source="TestStrategy",
        symbol="BTCUSDT",
        signal_type=SignalType.BUY,
        confidence=0.85,
        entry_price=Decimal("50000"),
        stop_loss=Decimal("49000"),
        take_profit=Decimal("52000"),
        strategy_name="TestStrategy",
    )

    # Create RiskApprovedOrderEvent
    risk_approved_event = RiskApprovedOrderEvent(
        source="RiskManager",
        signal=signal,
        approved_quantity=Decimal("0.1"),
        risk_params={"max_position_size": Decimal("1.0")},
    )

    # Handle the event
    await order_executor._handle_risk_approved_order(risk_approved_event)

    # Verify order was submitted
    mock_binance_client.futures_create_order.assert_called_once()
    call_args = mock_binance_client.futures_create_order.call_args[1]

    assert call_args["symbol"] == "BTCUSDT"
    assert call_args["side"] == "BUY"
    assert call_args["type"] == "MARKET"
    assert call_args["quantity"] == "0.1"


@pytest.mark.asyncio
async def test_handle_sell_signal(order_executor, mock_binance_client):
    """Test automatic order execution for SELL signal."""
    # Create a SELL signal
    signal = SignalEvent(
        source="TestStrategy",
        symbol="ETHUSDT",
        signal_type=SignalType.SELL,
        confidence=0.80,
        entry_price=Decimal("3000"),
        stop_loss=Decimal("3100"),
        take_profit=Decimal("2800"),
        strategy_name="TestStrategy",
    )

    # Create RiskApprovedOrderEvent
    risk_approved_event = RiskApprovedOrderEvent(
        source="RiskManager",
        signal=signal,
        approved_quantity=Decimal("1.5"),
        risk_params={"max_position_size": Decimal("5.0")},
    )

    # Handle the event
    await order_executor._handle_risk_approved_order(risk_approved_event)

    # Verify SELL order was submitted
    mock_binance_client.futures_create_order.assert_called_once()
    call_args = mock_binance_client.futures_create_order.call_args[1]

    assert call_args["symbol"] == "ETHUSDT"
    assert call_args["side"] == "SELL"
    assert call_args["type"] == "MARKET"
    assert call_args["quantity"] == "1.5"


@pytest.mark.asyncio
async def test_handle_close_long_signal(order_executor, mock_binance_client):
    """Test automatic order execution for CLOSE_LONG signal."""
    # Create a CLOSE_LONG signal
    signal = SignalEvent(
        source="TestStrategy",
        symbol="BTCUSDT",
        signal_type=SignalType.CLOSE_LONG,
        confidence=0.75,
        strategy_name="TestStrategy",
    )

    # Create RiskApprovedOrderEvent
    risk_approved_event = RiskApprovedOrderEvent(
        source="RiskManager",
        signal=signal,
        approved_quantity=Decimal("0.5"),
        risk_params={},
    )

    # Handle the event
    await order_executor._handle_risk_approved_order(risk_approved_event)

    # Verify SELL order was submitted (close long = sell)
    mock_binance_client.futures_create_order.assert_called_once()
    call_args = mock_binance_client.futures_create_order.call_args[1]

    assert call_args["side"] == "SELL"
    assert call_args["quantity"] == "0.5"


@pytest.mark.asyncio
async def test_handle_close_short_signal(order_executor, mock_binance_client):
    """Test automatic order execution for CLOSE_SHORT signal."""
    # Create a CLOSE_SHORT signal
    signal = SignalEvent(
        source="TestStrategy",
        symbol="ETHUSDT",
        signal_type=SignalType.CLOSE_SHORT,
        confidence=0.70,
        strategy_name="TestStrategy",
    )

    # Create RiskApprovedOrderEvent
    risk_approved_event = RiskApprovedOrderEvent(
        source="RiskManager",
        signal=signal,
        approved_quantity=Decimal("2.0"),
        risk_params={},
    )

    # Handle the event
    await order_executor._handle_risk_approved_order(risk_approved_event)

    # Verify BUY order was submitted (close short = buy)
    mock_binance_client.futures_create_order.assert_called_once()
    call_args = mock_binance_client.futures_create_order.call_args[1]

    assert call_args["side"] == "BUY"
    assert call_args["quantity"] == "2.0"


@pytest.mark.asyncio
async def test_ignore_hold_signal(order_executor, mock_binance_client):
    """Test that HOLD signals are ignored and no order is created."""
    # Create a HOLD signal
    signal = SignalEvent(
        source="TestStrategy",
        symbol="BTCUSDT",
        signal_type=SignalType.HOLD,
        confidence=0.60,
        strategy_name="TestStrategy",
    )

    # Create RiskApprovedOrderEvent
    risk_approved_event = RiskApprovedOrderEvent(
        source="RiskManager",
        signal=signal,
        approved_quantity=Decimal("0.1"),
        risk_params={},
    )

    # Handle the event
    await order_executor._handle_risk_approved_order(risk_approved_event)

    # Verify no order was submitted
    mock_binance_client.futures_create_order.assert_not_called()


@pytest.mark.asyncio
async def test_uses_approved_quantity(order_executor, mock_binance_client):
    """Test that approved quantity from risk manager is used, not signal quantity."""
    # Create signal with one quantity
    signal = SignalEvent(
        source="TestStrategy",
        symbol="BTCUSDT",
        signal_type=SignalType.BUY,
        confidence=0.85,
        quantity=Decimal("1.0"),  # Signal suggests 1.0
        strategy_name="TestStrategy",
    )

    # Risk manager approves different quantity
    risk_approved_event = RiskApprovedOrderEvent(
        source="RiskManager",
        signal=signal,
        approved_quantity=Decimal("0.3"),  # Risk manager approves only 0.3
        risk_params={"max_position_size": Decimal("0.5")},
    )

    # Handle the event
    await order_executor._handle_risk_approved_order(risk_approved_event)

    # Verify approved quantity was used
    call_args = mock_binance_client.futures_create_order.call_args[1]
    assert call_args["quantity"] == "0.3"  # Should use approved, not signal quantity


@pytest.mark.asyncio
async def test_uses_market_order_type(order_executor, mock_binance_client):
    """Test that MARKET order type is always used for immediate execution."""
    signal = SignalEvent(
        source="TestStrategy",
        symbol="BTCUSDT",
        signal_type=SignalType.BUY,
        confidence=0.85,
        strategy_name="TestStrategy",
    )

    risk_approved_event = RiskApprovedOrderEvent(
        source="RiskManager",
        signal=signal,
        approved_quantity=Decimal("0.1"),
        risk_params={},
    )

    # Handle the event
    await order_executor._handle_risk_approved_order(risk_approved_event)

    # Verify MARKET order type was used
    call_args = mock_binance_client.futures_create_order.call_args[1]
    assert call_args["type"] == "MARKET"


@pytest.mark.asyncio
async def test_error_handling_for_failed_execution(order_executor, mock_binance_client):
    """Test that execution errors are caught and logged without crashing."""
    # Make the order execution fail
    mock_binance_client.futures_create_order.side_effect = Exception("Exchange error")

    signal = SignalEvent(
        source="TestStrategy",
        symbol="BTCUSDT",
        signal_type=SignalType.BUY,
        confidence=0.85,
        strategy_name="TestStrategy",
    )

    risk_approved_event = RiskApprovedOrderEvent(
        source="RiskManager",
        signal=signal,
        approved_quantity=Decimal("0.1"),
        risk_params={},
    )

    # Handle the event - should not raise exception
    await order_executor._handle_risk_approved_order(risk_approved_event)

    # Verify attempt was made
    assert mock_binance_client.futures_create_order.called


@pytest.mark.asyncio
async def test_event_bus_integration(event_bus, mock_binance_client):
    """Test full integration with event bus for automatic order execution."""
    # Start the event bus first
    await event_bus.start()

    # Create executor with event bus
    executor = OrderExecutor(
        binance_client=mock_binance_client,
        event_bus=event_bus,
    )
    await executor.start()

    # Create and emit RiskApprovedOrderEvent
    signal = SignalEvent(
        source="TestStrategy",
        symbol="BTCUSDT",
        signal_type=SignalType.BUY,
        confidence=0.85,
        strategy_name="TestStrategy",
    )

    risk_approved_event = RiskApprovedOrderEvent(
        source="RiskManager",
        signal=signal,
        approved_quantity=Decimal("0.1"),
        risk_params={},
    )

    # Publish the event through event bus
    await event_bus.publish(risk_approved_event)

    # Give event bus time to process
    await asyncio.sleep(0.1)

    # Verify order was executed
    mock_binance_client.futures_create_order.assert_called_once()

    await executor.stop()
    await event_bus.stop()


@pytest.mark.asyncio
async def test_multiple_signals_processed(order_executor, mock_binance_client):
    """Test that multiple risk-approved signals are processed correctly."""
    signals = [
        SignalEvent(
            source="Strategy1",
            symbol="BTCUSDT",
            signal_type=SignalType.BUY,
            confidence=0.85,
            strategy_name="Strategy1",
        ),
        SignalEvent(
            source="Strategy2",
            symbol="ETHUSDT",
            signal_type=SignalType.SELL,
            confidence=0.80,
            strategy_name="Strategy2",
        ),
        SignalEvent(
            source="Strategy3",
            symbol="BNBUSDT",
            signal_type=SignalType.BUY,
            confidence=0.75,
            strategy_name="Strategy3",
        ),
    ]

    # Process each signal
    for signal in signals:
        risk_approved_event = RiskApprovedOrderEvent(
            source="RiskManager",
            signal=signal,
            approved_quantity=Decimal("0.1"),
            risk_params={},
        )
        await order_executor._handle_risk_approved_order(risk_approved_event)

    # Verify all three orders were executed
    assert mock_binance_client.futures_create_order.call_count == 3

    # Verify correct symbols
    calls = mock_binance_client.futures_create_order.call_args_list
    symbols = [call[1]["symbol"] for call in calls]
    assert "BTCUSDT" in symbols
    assert "ETHUSDT" in symbols
    assert "BNBUSDT" in symbols


@pytest.mark.asyncio
async def test_execution_stats_updated(order_executor, mock_binance_client):
    """Test that execution statistics are updated after auto execution."""
    signal = SignalEvent(
        source="TestStrategy",
        symbol="BTCUSDT",
        signal_type=SignalType.BUY,
        confidence=0.85,
        strategy_name="TestStrategy",
    )

    risk_approved_event = RiskApprovedOrderEvent(
        source="RiskManager",
        signal=signal,
        approved_quantity=Decimal("0.1"),
        risk_params={},
    )

    # Get initial stats
    initial_stats = await order_executor.get_execution_stats()
    initial_count = initial_stats["total_orders"]

    # Execute order
    await order_executor._handle_risk_approved_order(risk_approved_event)

    # Get updated stats
    updated_stats = await order_executor.get_execution_stats()

    # Verify stats were updated
    assert updated_stats["total_orders"] == initial_count + 1
    assert updated_stats["successful_orders"] >= initial_stats["successful_orders"]
