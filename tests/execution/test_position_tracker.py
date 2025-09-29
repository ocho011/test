"""
PositionTracker 테스트
"""

import pytest
from decimal import Decimal
from datetime import datetime

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


@pytest.fixture
def position_tracker():
    """PositionTracker 인스턴스"""
    return PositionTracker()


@pytest.fixture
def sample_position():
    """샘플 포지션"""
    return Position(
        symbol="BTCUSDT",
        side=PositionSide.LONG,
        size=Decimal("0.1"),
        entry_price=Decimal("50000"),
        leverage=10,
        margin_required=Decimal("500"),
        stop_loss=Decimal("49000"),
        take_profit=Decimal("52000")
    )


@pytest.mark.asyncio
async def test_add_position(position_tracker, sample_position):
    """포지션 추가 테스트"""
    # When
    position_id = await position_tracker.add_position(sample_position)
    
    # Then
    assert position_id is not None
    assert len(position_tracker.positions) == 1
    assert position_tracker.positions[position_id] == sample_position


@pytest.mark.asyncio
async def test_update_position_price(position_tracker, sample_position):
    """포지션 가격 업데이트 테스트"""
    # Given
    position_id = await position_tracker.add_position(sample_position)
    new_price = Decimal("51000")
    
    # When
    await position_tracker.update_position_price(position_id, new_price)
    
    # Then
    updated_position = position_tracker.positions[position_id]
    assert updated_position.current_price == new_price


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
        symbol="BTCUSDT",
        side=PositionSide.SHORT,
        size=Decimal("0.1"),
        entry_price=Decimal("50000"),
        leverage=10,
        margin_required=Decimal("500")
    )
    current_price = Decimal("49000")  # 1000$ 하락
    
    # When
    pnl = position.calculate_unrealized_pnl(current_price)
    
    # Then
    # (50000 - 49000) * 0.1 * 10 = 100$
    expected_pnl = Decimal("100")
    assert pnl == expected_pnl


@pytest.mark.asyncio
async def test_calculate_roe(sample_position):
    """ROE 계산 테스트"""
    # Given
    current_price = Decimal("52000")  # 2000$ 상승
    
    # When
    roe = sample_position.calculate_roe(current_price)
    
    # Then
    # PnL = (52000 - 50000) * 0.1 * 10 = 200$
    # ROE = 200 / 500 * 100 = 40%
    expected_roe = Decimal("40")
    assert roe == expected_roe


@pytest.mark.asyncio
async def test_close_position(position_tracker, sample_position):
    """포지션 청산 테스트"""
    # Given
    position_id = await position_tracker.add_position(sample_position)
    close_price = Decimal("51500")
    
    # When
    closed_position = await position_tracker.close_position(position_id, close_price)
    
    # Then
    assert closed_position.status == PositionStatus.CLOSED
    assert closed_position.exit_price == close_price
    assert closed_position.realized_pnl is not None
    assert position_id not in position_tracker.positions


@pytest.mark.asyncio
async def test_get_total_pnl(position_tracker):
    """총 PnL 계산 테스트"""
    # Given
    position1 = Position(
        symbol="BTCUSDT",
        side=PositionSide.LONG,
        size=Decimal("0.1"),
        entry_price=Decimal("50000"),
        leverage=10,
        margin_required=Decimal("500")
    )
    position2 = Position(
        symbol="ETHUSDT",
        side=PositionSide.LONG,
        size=Decimal("1"),
        entry_price=Decimal("3000"),
        leverage=5,
        margin_required=Decimal("600")
    )
    
    position_id1 = await position_tracker.add_position(position1)
    position_id2 = await position_tracker.add_position(position2)
    
    # 가격 업데이트
    await position_tracker.update_position_price(position_id1, Decimal("51000"))
    await position_tracker.update_position_price(position_id2, Decimal("3100"))
    
    # When
    total_pnl = position_tracker.get_total_pnl()
    
    # Then
    # BTC: (51000 - 50000) * 0.1 * 10 = 100$
    # ETH: (3100 - 3000) * 1 * 5 = 500$
    # Total: 600$
    expected_total = Decimal("600")
    assert total_pnl == expected_total


@pytest.mark.asyncio
async def test_get_positions_by_symbol(position_tracker):
    """심볼별 포지션 조회 테스트"""
    # Given
    btc_position = Position(
        symbol="BTCUSDT",
        side=PositionSide.LONG,
        size=Decimal("0.1"),
        entry_price=Decimal("50000"),
        leverage=10,
        margin_required=Decimal("500")
    )
    eth_position = Position(
        symbol="ETHUSDT",
        side=PositionSide.LONG,
        size=Decimal("1"),
        entry_price=Decimal("3000"),
        leverage=5,
        margin_required=Decimal("600")
    )
    
    await position_tracker.add_position(btc_position)
    await position_tracker.add_position(eth_position)
    
    # When
    btc_positions = position_tracker.get_positions_by_symbol("BTCUSDT")
    eth_positions = position_tracker.get_positions_by_symbol("ETHUSDT")
    
    # Then
    assert len(btc_positions) == 1
    assert len(eth_positions) == 1
    assert btc_positions[0].symbol == "BTCUSDT"
    assert eth_positions[0].symbol == "ETHUSDT"


@pytest.mark.asyncio
async def test_position_risk_level():
    """포지션 리스크 레벨 테스트"""
    # Given
    position = Position(
        symbol="BTCUSDT",
        side=PositionSide.LONG,
        size=Decimal("0.1"),
        entry_price=Decimal("50000"),
        leverage=10,
        margin_required=Decimal("500"),
        stop_loss=Decimal("49000")  # 2% 손실
    )
    
    # When
    risk_level_profitable = position.get_risk_level(Decimal("51000"))  # 수익 상태
    risk_level_loss = position.get_risk_level(Decimal("49500"))  # 손실 상태
    risk_level_critical = position.get_risk_level(Decimal("49100"))  # 임계 상태
    
    # Then
    assert risk_level_profitable == "LOW"
    assert risk_level_loss == "MEDIUM"
    assert risk_level_critical == "HIGH"


@pytest.mark.asyncio
async def test_position_margin_level():
    """포지션 마진 레벨 테스트"""
    # Given
    position = Position(
        symbol="BTCUSDT",
        side=PositionSide.LONG,
        size=Decimal("0.1"),
        entry_price=Decimal("50000"),
        leverage=10,
        margin_required=Decimal("500")
    )
    
    # When
    margin_level_safe = position.get_margin_level(Decimal("51000"))  # 수익 상태
    margin_level_danger = position.get_margin_level(Decimal("49000"))  # 10% 손실
    
    # Then
    assert margin_level_safe > 100  # 안전한 마진 레벨
    assert margin_level_danger < 100  # 위험한 마진 레벨


@pytest.mark.asyncio
async def test_position_tracker_start_stop(position_tracker):
    """PositionTracker 시작/중지 테스트"""
    # When
    await position_tracker.start()
    assert position_tracker.running
    
    await position_tracker.stop()
    assert not position_tracker.running


@pytest.mark.asyncio
async def test_position_event_publishing(position_tracker, sample_position):
    """포지션 이벤트 발행 테스트"""
    # Given
    events_published = []
    
    async def mock_publish_event(event):
        events_published.append(event)
    
    position_tracker.publish_event = mock_publish_event
    
    # When
    position_id = await position_tracker.add_position(sample_position)
    await position_tracker.update_position_price(position_id, Decimal("51000"))
    
    # Then
    assert len(events_published) == 2  # 추가 + 업데이트
    assert all(isinstance(event, PositionEvent) for event in events_published)
    assert events_published[0].status == PositionStatus.OPEN.value
    assert events_published[1].status == PositionStatus.OPEN.value