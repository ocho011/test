"""
PositionTracker 테스트
"""

import pytest
from decimal import Decimal
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from src.trading_bot.execution.position_tracker import (
    PositionTracker,
    Position
)
from src.trading_bot.core.events import (
    PositionEvent,
    EventType,
    PositionStatus,
    PositionSide
)
from src.trading_bot.data.binance_client import BinanceClient


@pytest.fixture
def mock_binance_client():
    """모의 바이낸스 클라이언트"""
    client = MagicMock(spec=BinanceClient)
    client.futures_symbol_ticker = AsyncMock(return_value={"price": "50000.0"})
    return client


@pytest.fixture
def position_tracker(mock_binance_client):
    """PositionTracker 인스턴스"""
    return PositionTracker(binance_client=mock_binance_client)


@pytest.fixture
def sample_position():
    """샘플 포지션"""
    return Position(
        position_id="test_position_001",
        symbol="BTCUSDT",
        side=PositionSide.LONG,
        size=Decimal("0.1"),
        entry_price=Decimal("50000"),
        stop_loss=Decimal("49000"),
        take_profit=Decimal("52000")
    )


@pytest.mark.asyncio
async def test_add_position(position_tracker):
    """포지션 추가 테스트 - 이벤트 기반"""
    from src.trading_bot.core.events import OrderEvent, OrderSide, OrderType, OrderStatus
    
    # Given - Create a FILLED order event to trigger position creation
    order_event = OrderEvent(
        source="test",
        client_order_id="test_order_001",
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0.1"),
        status=OrderStatus.FILLED,
        filled_quantity=Decimal("0.1"),
        average_price=Decimal("50000")
    )
    
    # When
    await position_tracker._handle_order_event(order_event)
    
    # Then
    assert len(position_tracker.positions) == 1
    position = list(position_tracker.positions.values())[0]
    assert position.symbol == "BTCUSDT"
    assert position.side == PositionSide.LONG
    assert position.size == Decimal("0.1")
    assert position.entry_price == Decimal("50000")



@pytest.mark.asyncio
async def test_calculate_unrealized_pnl_long(sample_position):
    """롱 포지션 미실현 PnL 계산 테스트"""
    # Given
    current_price = Decimal("51000")  # 1000$ 상승
    
    # When
    pnl = sample_position.calculate_unrealized_pnl(current_price)
    
    # Then
    # (51000 - 50000) * 0.1 * 10 = 100$
    expected_pnl = Decimal("100")
    assert pnl == expected_pnl


@pytest.mark.asyncio
async def test_calculate_unrealized_pnl_short():
    """숏 포지션 미실현 PnL 계산 테스트"""
    # Given
    position = Position(
        position_id="test_position_002",
        symbol="BTCUSDT",
        side=PositionSide.SHORT,
        size=Decimal("0.1"),
        entry_price=Decimal("50000"),
        stop_loss=Decimal("51000"),
        take_profit=Decimal("48000")
    )
    current_price = Decimal("49000")  # 1000$ 하락
    
    # When
    pnl = position.calculate_unrealized_pnl(current_price)
    
    # Then
    # (50000 - 49000) * 0.1 * 10 = 100$
    expected_pnl = Decimal("100")
    assert pnl == expected_pnl



@pytest.mark.asyncio
async def test_close_position(position_tracker):
    """포지션 청산 테스트 - 이벤트 기반"""
    from src.trading_bot.core.events import OrderEvent, OrderSide, OrderType, OrderStatus
    
    # Given - Open a position first
    open_order = OrderEvent(
        source="test",
        client_order_id="test_order_001",
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0.1"),
        status=OrderStatus.FILLED,
        filled_quantity=Decimal("0.1"),
        average_price=Decimal("50000")
    )
    await position_tracker._handle_order_event(open_order)
    
    # When - Close the position with opposite side order
    close_order = OrderEvent(
        source="test",
        client_order_id="test_order_002",
        symbol="BTCUSDT",
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        quantity=Decimal("0.1"),
        status=OrderStatus.FILLED,
        filled_quantity=Decimal("0.1"),
        average_price=Decimal("51500"),
        metadata={"reduce_only": True}
    )
    await position_tracker._handle_order_event(close_order)
    
    # Then
    assert len(position_tracker.positions) == 0


@pytest.mark.asyncio
async def test_get_total_pnl(position_tracker):
    """총 PnL 계산 테스트 - 이벤트 기반"""
    from src.trading_bot.core.events import OrderEvent, OrderSide, OrderType, OrderStatus
    
    # Given - Create two positions via events
    btc_order = OrderEvent(
        source="test",
        client_order_id="test_order_btc",
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0.1"),
        status=OrderStatus.FILLED,
        filled_quantity=Decimal("0.1"),
        average_price=Decimal("50000")
    )
    await position_tracker._handle_order_event(btc_order)
    
    eth_order = OrderEvent(
        source="test",
        client_order_id="test_order_eth",
        symbol="ETHUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("1"),
        status=OrderStatus.FILLED,
        filled_quantity=Decimal("1"),
        average_price=Decimal("3000")
    )
    await position_tracker._handle_order_event(eth_order)
    
    # Update prices
    btc_position = list(position_tracker.positions.values())[0]
    btc_position.update_price(Decimal("51000"))
    
    eth_position = list(position_tracker.positions.values())[1]
    eth_position.update_price(Decimal("3100"))
    
    # When
    total_pnl = await position_tracker.get_total_pnl()
    
    # Then
    # BTC: (51000 - 50000) * 0.1 = 100
    # ETH: (3100 - 3000) * 1 = 100
    # Total unrealized: 200
    assert total_pnl["unrealized_pnl"] == Decimal("200")


@pytest.mark.asyncio
async def test_get_positions_by_symbol(position_tracker):
    """심볼별 포지션 조회 테스트 - 이벤트 기반"""
    from src.trading_bot.core.events import OrderEvent, OrderSide, OrderType, OrderStatus
    
    # Given - Create two positions for different symbols
    btc_order = OrderEvent(
        source="test",
        client_order_id="test_order_btc",
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0.1"),
        status=OrderStatus.FILLED,
        filled_quantity=Decimal("0.1"),
        average_price=Decimal("50000")
    )
    await position_tracker._handle_order_event(btc_order)
    
    eth_order = OrderEvent(
        source="test",
        client_order_id="test_order_eth",
        symbol="ETHUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("1"),
        status=OrderStatus.FILLED,
        filled_quantity=Decimal("1"),
        average_price=Decimal("3000")
    )
    await position_tracker._handle_order_event(eth_order)
    
    # When
    btc_positions = await position_tracker.get_positions("BTCUSDT")
    eth_positions = await position_tracker.get_positions("ETHUSDT")
    
    # Then
    assert len(btc_positions) == 1
    assert len(eth_positions) == 1
    assert btc_positions[0]["symbol"] == "BTCUSDT"
    assert eth_positions[0]["symbol"] == "ETHUSDT"




@pytest.mark.asyncio
async def test_position_tracker_start_stop(position_tracker):
    """PositionTracker 시작/중지 테스트"""
    # When
    await position_tracker.start()
    assert position_tracker.is_running()
    
    await position_tracker.stop()
    assert not position_tracker.is_running()


@pytest.mark.asyncio
async def test_position_event_publishing(position_tracker):
    """포지션 이벤트 발행 테스트 - 이벤트 기반"""
    from src.trading_bot.core.events import OrderEvent, OrderSide, OrderType, OrderStatus
    
    # Given
    events_published = []
    
    async def mock_emit_event(event):
        events_published.append(event)
    
    position_tracker._emit_event = mock_emit_event
    
    # When - Create a position via order event
    order_event = OrderEvent(
        source="test",
        client_order_id="test_order_001",
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0.1"),
        status=OrderStatus.FILLED,
        filled_quantity=Decimal("0.1"),
        average_price=Decimal("50000")
    )
    await position_tracker._handle_order_event(order_event)
    
    # Then
    assert len(events_published) >= 1
    from src.trading_bot.core.events import PositionEvent
    assert all(isinstance(event, PositionEvent) for event in events_published)
    assert events_published[0].status == PositionStatus.OPEN