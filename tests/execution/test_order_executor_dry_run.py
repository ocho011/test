"""
Tests for OrderExecutor dry-run mode functionality.

Tests verify that dry-run mode properly:
1. Logs order details without executing
2. Emits simulated OrderEvents
3. Does not call Binance API
4. Handles stop loss and take profit logging
"""

import asyncio
from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, call
from uuid import uuid4

import pytest

from src.trading_bot.core.event_bus import EventBus
from src.trading_bot.core.events import (
    EventType,
    OrderEvent,
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

    # Mock futures_create_order response (should NOT be called in dry-run)
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
async def dry_run_executor(mock_binance_client, event_bus):
    """Create an OrderExecutor instance in dry-run mode."""
    executor = OrderExecutor(
        binance_client=mock_binance_client,
        event_bus=event_bus,
        max_retries=2,
        retry_delay=0.1,
        dry_run=True,  # Enable dry-run mode
    )
    await executor.start()
    yield executor
    await executor.stop()


@pytest.fixture
async def live_executor(mock_binance_client, event_bus):
    """Create an OrderExecutor instance in live mode for comparison."""
    executor = OrderExecutor(
        binance_client=mock_binance_client,
        event_bus=event_bus,
        max_retries=2,
        retry_delay=0.1,
        dry_run=False,  # Live mode
    )
    await executor.start()
    yield executor
    await executor.stop()


@pytest.mark.asyncio
async def test_dry_run_initialization(dry_run_executor):
    """Test that dry-run mode is properly initialized."""
    assert dry_run_executor.dry_run is True


@pytest.mark.asyncio
async def test_live_mode_initialization(live_executor):
    """Test that live mode is properly initialized."""
    assert live_executor.dry_run is False


@pytest.mark.asyncio
async def test_dry_run_no_api_call_for_buy(dry_run_executor, mock_binance_client):
    """Test that dry-run mode does NOT call Binance API for BUY orders."""
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

    risk_approved_event = RiskApprovedOrderEvent(
        source="RiskManager",
        signal=signal,
        approved_quantity=Decimal("0.1"),
        risk_params={},
    )

    # Handle the event in dry-run mode
    await dry_run_executor._handle_risk_approved_order(risk_approved_event)

    # Verify NO API call was made
    mock_binance_client.futures_create_order.assert_not_called()


@pytest.mark.asyncio
async def test_dry_run_no_api_call_for_sell(dry_run_executor, mock_binance_client):
    """Test that dry-run mode does NOT call Binance API for SELL orders."""
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

    risk_approved_event = RiskApprovedOrderEvent(
        source="RiskManager",
        signal=signal,
        approved_quantity=Decimal("1.5"),
        risk_params={},
    )

    # Handle the event in dry-run mode
    await dry_run_executor._handle_risk_approved_order(risk_approved_event)

    # Verify NO API call was made
    mock_binance_client.futures_create_order.assert_not_called()


@pytest.mark.asyncio
async def test_dry_run_emits_simulated_order_event(dry_run_executor, event_bus):
    """Test that dry-run mode emits a simulated OrderEvent."""
    # Start the event bus first
    await event_bus.start()

    # Create a list to capture emitted events
    captured_events = []

    async def capture_event(event):
        captured_events.append(event)

    # Subscribe to OrderEvent
    await event_bus.subscribe(capture_event, EventType.ORDER)

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
    await dry_run_executor._handle_risk_approved_order(risk_approved_event)

    # Give event bus time to process
    await asyncio.sleep(0.1)

    # Verify an OrderEvent was emitted
    assert len(captured_events) > 0

    # Find the simulated FILLED order event
    filled_events = [e for e in captured_events if e.status == OrderStatus.FILLED]
    assert len(filled_events) == 1

    order_event = filled_events[0]
    assert order_event.symbol == "BTCUSDT"
    assert order_event.side == OrderSide.BUY
    assert order_event.order_type == OrderType.MARKET
    assert order_event.quantity == Decimal("0.1")
    assert order_event.filled_quantity == Decimal("0.1")
    assert order_event.status == OrderStatus.FILLED

    # Clean up
    await event_bus.stop()


@pytest.mark.asyncio
async def test_dry_run_logs_order_details(dry_run_executor, caplog):
    """Test that dry-run mode logs order details in the specified format."""
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

    risk_approved_event = RiskApprovedOrderEvent(
        source="RiskManager",
        signal=signal,
        approved_quantity=Decimal("0.1"),
        risk_params={},
    )

    # Capture logs
    with caplog.at_level("INFO"):
        await dry_run_executor._handle_risk_approved_order(risk_approved_event)

    # Verify log contains dry-run marker and order details
    log_messages = [record.message for record in caplog.records]

    # Check for main order log
    order_log = [msg for msg in log_messages if "[DRY-RUN] Order would be executed:" in msg]
    assert len(order_log) == 1
    assert "buy" in order_log[0].lower()
    assert "0.1" in order_log[0]
    assert "BTCUSDT" in order_log[0]
    assert "@ MARKET" in order_log[0]

    # Check for stop loss / take profit log
    sl_tp_log = [msg for msg in log_messages if "[DRY-RUN] Stop Loss:" in msg]
    assert len(sl_tp_log) == 1
    assert "49000" in sl_tp_log[0]
    assert "52000" in sl_tp_log[0]


@pytest.mark.asyncio
async def test_dry_run_logs_without_stop_loss_take_profit(dry_run_executor, caplog):
    """Test dry-run logging when stop loss and take profit are not provided."""
    signal = SignalEvent(
        source="TestStrategy",
        symbol="ETHUSDT",
        signal_type=SignalType.SELL,
        confidence=0.80,
        strategy_name="TestStrategy",
        # No stop_loss or take_profit
    )

    risk_approved_event = RiskApprovedOrderEvent(
        source="RiskManager",
        signal=signal,
        approved_quantity=Decimal("1.5"),
        risk_params={},
    )

    # Capture logs
    with caplog.at_level("INFO"):
        await dry_run_executor._handle_risk_approved_order(risk_approved_event)

    # Verify main order log exists
    log_messages = [record.message for record in caplog.records]
    order_log = [msg for msg in log_messages if "[DRY-RUN] Order would be executed:" in msg]
    assert len(order_log) == 1

    # Verify NO stop loss / take profit log (since both are None)
    sl_tp_log = [msg for msg in log_messages if "[DRY-RUN] Stop Loss:" in msg]
    assert len(sl_tp_log) == 0


@pytest.mark.asyncio
async def test_dry_run_logs_partial_sl_tp(dry_run_executor, caplog):
    """Test dry-run logging when only stop loss or take profit is provided."""
    signal = SignalEvent(
        source="TestStrategy",
        symbol="BNBUSDT",
        signal_type=SignalType.BUY,
        confidence=0.75,
        stop_loss=Decimal("400"),  # Only stop loss, no take profit
        strategy_name="TestStrategy",
    )

    risk_approved_event = RiskApprovedOrderEvent(
        source="RiskManager",
        signal=signal,
        approved_quantity=Decimal("2.0"),
        risk_params={},
    )

    # Capture logs
    with caplog.at_level("INFO"):
        await dry_run_executor._handle_risk_approved_order(risk_approved_event)

    # Verify stop loss / take profit log with N/A for missing value
    log_messages = [record.message for record in caplog.records]
    sl_tp_log = [msg for msg in log_messages if "[DRY-RUN] Stop Loss:" in msg]
    assert len(sl_tp_log) == 1
    assert "400" in sl_tp_log[0]
    assert "N/A" in sl_tp_log[0]


@pytest.mark.asyncio
async def test_dry_run_vs_live_behavior(dry_run_executor, live_executor, mock_binance_client):
    """Test behavior difference between dry-run and live mode."""
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

    # Test dry-run mode
    await dry_run_executor._handle_risk_approved_order(risk_approved_event)
    dry_run_call_count = mock_binance_client.futures_create_order.call_count

    # Reset mock
    mock_binance_client.futures_create_order.reset_mock()

    # Test live mode
    await live_executor._handle_risk_approved_order(risk_approved_event)
    live_call_count = mock_binance_client.futures_create_order.call_count

    # Verify dry-run made NO API calls, live mode made ONE API call
    assert dry_run_call_count == 0
    assert live_call_count == 1


@pytest.mark.asyncio
async def test_dry_run_close_long_signal(dry_run_executor, mock_binance_client, caplog):
    """Test dry-run mode for CLOSE_LONG signal."""
    signal = SignalEvent(
        source="TestStrategy",
        symbol="BTCUSDT",
        signal_type=SignalType.CLOSE_LONG,
        confidence=0.75,
        strategy_name="TestStrategy",
    )

    risk_approved_event = RiskApprovedOrderEvent(
        source="RiskManager",
        signal=signal,
        approved_quantity=Decimal("0.5"),
        risk_params={},
    )

    with caplog.at_level("INFO"):
        await dry_run_executor._handle_risk_approved_order(risk_approved_event)

    # Verify no API call
    mock_binance_client.futures_create_order.assert_not_called()

    # Verify correct side in logs (SELL for close long)
    log_messages = [record.message for record in caplog.records]
    order_log = [msg for msg in log_messages if "[DRY-RUN] Order would be executed:" in msg]
    assert len(order_log) == 1
    assert "sell" in order_log[0].lower()


@pytest.mark.asyncio
async def test_dry_run_close_short_signal(dry_run_executor, mock_binance_client, caplog):
    """Test dry-run mode for CLOSE_SHORT signal."""
    signal = SignalEvent(
        source="TestStrategy",
        symbol="ETHUSDT",
        signal_type=SignalType.CLOSE_SHORT,
        confidence=0.70,
        strategy_name="TestStrategy",
    )

    risk_approved_event = RiskApprovedOrderEvent(
        source="RiskManager",
        signal=signal,
        approved_quantity=Decimal("2.0"),
        risk_params={},
    )

    with caplog.at_level("INFO"):
        await dry_run_executor._handle_risk_approved_order(risk_approved_event)

    # Verify no API call
    mock_binance_client.futures_create_order.assert_not_called()

    # Verify correct side in logs (BUY for close short)
    log_messages = [record.message for record in caplog.records]
    order_log = [msg for msg in log_messages if "[DRY-RUN] Order would be executed:" in msg]
    assert len(order_log) == 1
    assert "buy" in order_log[0].lower()


@pytest.mark.asyncio
async def test_dry_run_multiple_orders(dry_run_executor, mock_binance_client):
    """Test that multiple orders in dry-run mode do not call API."""
    signals = [
        SignalEvent(
            source=f"Strategy{i}",
            symbol="BTCUSDT",
            signal_type=SignalType.BUY,
            confidence=0.85,
            strategy_name=f"Strategy{i}",
        )
        for i in range(5)
    ]

    # Process each signal
    for signal in signals:
        risk_approved_event = RiskApprovedOrderEvent(
            source="RiskManager",
            signal=signal,
            approved_quantity=Decimal("0.1"),
            risk_params={},
        )
        await dry_run_executor._handle_risk_approved_order(risk_approved_event)

    # Verify NO API calls were made for any order
    assert mock_binance_client.futures_create_order.call_count == 0


@pytest.mark.asyncio
async def test_dry_run_event_bus_integration(event_bus, mock_binance_client):
    """Test full integration with event bus in dry-run mode."""
    # Start the event bus
    await event_bus.start()

    # Create executor in dry-run mode
    executor = OrderExecutor(
        binance_client=mock_binance_client,
        event_bus=event_bus,
        dry_run=True,
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

    # Verify NO order was executed via API
    mock_binance_client.futures_create_order.assert_not_called()

    await executor.stop()
    await event_bus.stop()


@pytest.mark.asyncio
async def test_dry_run_ignores_hold_signal(dry_run_executor, mock_binance_client, caplog):
    """Test that dry-run mode also ignores HOLD signals."""
    signal = SignalEvent(
        source="TestStrategy",
        symbol="BTCUSDT",
        signal_type=SignalType.HOLD,
        confidence=0.60,
        strategy_name="TestStrategy",
    )

    risk_approved_event = RiskApprovedOrderEvent(
        source="RiskManager",
        signal=signal,
        approved_quantity=Decimal("0.1"),
        risk_params={},
    )

    with caplog.at_level("INFO"):
        await dry_run_executor._handle_risk_approved_order(risk_approved_event)

    # Verify no API call
    mock_binance_client.futures_create_order.assert_not_called()

    # Verify no dry-run order log (should be ignored before dry-run logic)
    log_messages = [record.message for record in caplog.records]
    order_log = [msg for msg in log_messages if "[DRY-RUN] Order would be executed:" in msg]
    assert len(order_log) == 0


@pytest.mark.asyncio
async def test_dry_run_with_error_handling(dry_run_executor, mock_binance_client):
    """Test that dry-run mode handles errors gracefully (though no API calls should happen)."""
    # Make the event bus fail (simulate error scenario)
    dry_run_executor.event_bus = None

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

    # Should not raise exception even with event bus error
    await dry_run_executor._handle_risk_approved_order(risk_approved_event)

    # Verify no API call was made
    mock_binance_client.futures_create_order.assert_not_called()
