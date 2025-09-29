"""
SlippageController 및 OrderRetryHandler 테스트
"""

import pytest
import asyncio
from decimal import Decimal
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from src.trading_bot.execution.slippage_controller import (
    SlippageController,
    OrderRetryHandler,
    SlippageConfig,
    RetryConfig,
    SlippageType,
    RetryStrategy,
    SlippageData,
    OrderRetryData
)


@pytest.fixture
def slippage_config():
    """슬리피지 설정"""
    return SlippageConfig(
        max_slippage_percentage=0.1,
        excessive_slippage_threshold=0.5,
        monitoring_enabled=True,
        auto_cancel_on_excessive=True,
        slippage_tolerance_by_symbol={
            "BTCUSDT": 0.05,
            "ETHUSDT": 0.08
        }
    )


@pytest.fixture
def retry_config():
    """재시도 설정"""
    return RetryConfig(
        max_retries=3,
        base_delay=1.0,
        max_delay=30.0,
        backoff_multiplier=2.0,
        strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        retry_on_slippage=True,
        retry_on_timeout=True,
        retry_on_network_error=True
    )


@pytest.fixture
def slippage_controller(slippage_config):
    """SlippageController 인스턴스"""
    return SlippageController(slippage_config)


@pytest.fixture
def retry_handler(retry_config):
    """OrderRetryHandler 인스턴스"""
    return OrderRetryHandler(retry_config)


@pytest.fixture
def sample_order_request():
    """샘플 주문 요청"""
    return MagicMock(order_id="test_order_001")


@pytest.mark.asyncio
async def test_calculate_slippage_buy_negative(slippage_controller):
    """매수 시 불리한 슬리피지 계산 테스트"""
    # Given
    expected_price = Decimal("50000")
    executed_price = Decimal("50100")  # 100$ 더 비싸게 체결
    side = "BUY"
    
    # When
    slippage_amount, slippage_percentage, slippage_type = slippage_controller.calculate_slippage(
        expected_price, executed_price, side
    )
    
    # Then
    assert slippage_amount == Decimal("100")
    assert slippage_percentage == Decimal("0.2")  # 0.2%
    assert slippage_type == SlippageType.NEGATIVE


@pytest.mark.asyncio
async def test_calculate_slippage_sell_negative(slippage_controller):
    """매도 시 불리한 슬리피지 계산 테스트"""
    # Given
    expected_price = Decimal("50000")
    executed_price = Decimal("49900")  # 100$ 더 싸게 체결
    side = "SELL"
    
    # When
    slippage_amount, slippage_percentage, slippage_type = slippage_controller.calculate_slippage(
        expected_price, executed_price, side
    )
    
    # Then
    assert slippage_amount == Decimal("100")
    assert slippage_percentage == Decimal("0.2")  # 0.2%
    assert slippage_type == SlippageType.NEGATIVE


@pytest.mark.asyncio
async def test_calculate_slippage_positive(slippage_controller):
    """유리한 슬리피지 계산 테스트"""
    # Given
    expected_price = Decimal("50000")
    executed_price = Decimal("49900")  # 100$ 더 싸게 체결 (매수 시)
    side = "BUY"
    
    # When
    slippage_amount, slippage_percentage, slippage_type = slippage_controller.calculate_slippage(
        expected_price, executed_price, side
    )
    
    # Then
    assert slippage_amount == Decimal("-100")
    assert slippage_percentage == Decimal("-0.2")  # -0.2%
    assert slippage_type == SlippageType.POSITIVE


@pytest.mark.asyncio
async def test_calculate_slippage_excessive(slippage_controller):
    """과도한 슬리피지 계산 테스트"""
    # Given
    expected_price = Decimal("50000")
    executed_price = Decimal("50300")  # 300$ 더 비싸게 체결 (0.6% 슬리피지)
    side = "BUY"
    
    # When
    slippage_amount, slippage_percentage, slippage_type = slippage_controller.calculate_slippage(
        expected_price, executed_price, side
    )
    
    # Then
    assert slippage_amount == Decimal("300")
    assert slippage_percentage == Decimal("0.6")  # 0.6%
    assert slippage_type == SlippageType.EXCESSIVE


@pytest.mark.asyncio
async def test_is_slippage_acceptable_global_limit(slippage_controller):
    """전역 슬리피지 허용 한계 테스트"""
    # Given
    symbol = "ADAUSDT"  # 심볼별 설정이 없는 경우
    slippage_percentage = 0.08  # 0.08%
    
    # When
    is_acceptable = slippage_controller.is_slippage_acceptable(symbol, slippage_percentage)
    
    # Then
    assert is_acceptable  # 0.1% 한계 내


@pytest.mark.asyncio
async def test_is_slippage_acceptable_symbol_specific(slippage_controller):
    """심볼별 슬리피지 허용 한계 테스트"""
    # Given
    symbol = "BTCUSDT"  # 심볼별 설정: 0.05%
    slippage_percentage = 0.08  # 0.08%
    
    # When
    is_acceptable = slippage_controller.is_slippage_acceptable(symbol, slippage_percentage)
    
    # Then
    assert not is_acceptable  # 0.05% 한계 초과


@pytest.mark.asyncio
async def test_validate_order_execution_acceptable(slippage_controller):
    """허용 가능한 주문 실행 검증 테스트"""
    # Given
    slippage_controller.publish_event = AsyncMock()
    
    order_id = "test_order_001"
    symbol = "BTCUSDT"
    expected_price = Decimal("50000")
    executed_price = Decimal("50020")  # 0.04% 슬리피지
    side = "BUY"
    
    # When
    is_acceptable, slippage_data = await slippage_controller.validate_order_execution(
        order_id, symbol, expected_price, executed_price, side
    )
    
    # Then
    assert is_acceptable
    assert isinstance(slippage_data, SlippageData)
    assert slippage_data.slippage_type == SlippageType.NEGATIVE
    slippage_controller.publish_event.assert_called_once()


@pytest.mark.asyncio
async def test_validate_order_execution_excessive(slippage_controller):
    """과도한 슬리피지 주문 실행 검증 테스트"""
    # Given
    slippage_controller.publish_event = AsyncMock()
    
    order_id = "test_order_001"
    symbol = "BTCUSDT"
    expected_price = Decimal("50000")
    executed_price = Decimal("50300")  # 0.6% 슬리피지 (과도함)
    side = "BUY"
    
    # When
    is_acceptable, slippage_data = await slippage_controller.validate_order_execution(
        order_id, symbol, expected_price, executed_price, side
    )
    
    # Then
    assert not is_acceptable  # 과도한 슬리피지로 거부
    assert slippage_data.slippage_type == SlippageType.EXCESSIVE


@pytest.mark.asyncio
async def test_get_symbol_slippage_stats(slippage_controller):
    """심볼별 슬리피지 통계 테스트"""
    # Given
    slippage_controller.publish_event = AsyncMock()
    
    # 여러 주문 실행
    for i in range(5):
        await slippage_controller.validate_order_execution(
            f"order_{i}",
            "BTCUSDT",
            Decimal("50000"),
            Decimal("50020"),  # 일정한 슬리피지
            "BUY"
        )
    
    # When
    stats = slippage_controller.get_symbol_slippage_stats("BTCUSDT")
    
    # Then
    assert stats is not None
    assert stats['total_orders'] == 5
    assert stats['negative_slippage_count'] == 5
    assert stats['average_slippage'] == Decimal("0.04")


@pytest.mark.asyncio
async def test_get_recent_slippage_data(slippage_controller):
    """최근 슬리피지 데이터 조회 테스트"""
    # Given
    slippage_controller.publish_event = AsyncMock()
    
    # 최근 데이터 추가
    await slippage_controller.validate_order_execution(
        "recent_order",
        "BTCUSDT",
        Decimal("50000"),
        Decimal("50020"),
        "BUY"
    )
    
    # When
    recent_data = slippage_controller.get_recent_slippage_data(hours=1)
    
    # Then
    assert len(recent_data) == 1
    assert recent_data[0].order_id == "recent_order"


@pytest.mark.asyncio
async def test_should_retry_max_retries_exceeded(retry_handler):
    """최대 재시도 횟수 초과 테스트"""
    # Given
    error_type = "network"
    retry_count = 5  # max_retries (3) 초과
    
    # When
    should_retry = retry_handler.should_retry(error_type, retry_count)
    
    # Then
    assert not should_retry


@pytest.mark.asyncio
async def test_should_retry_slippage_disabled(retry_handler):
    """슬리피지 재시도 비활성화 테스트"""
    # Given
    retry_handler.config.retry_on_slippage = False
    error_type = "slippage"
    retry_count = 1
    
    # When
    should_retry = retry_handler.should_retry(error_type, retry_count)
    
    # Then
    assert not should_retry


@pytest.mark.asyncio
async def test_calculate_delay_exponential_backoff(retry_handler):
    """지수 백오프 지연 계산 테스트"""
    # Given
    retry_count = 2
    
    # When
    delay = retry_handler.calculate_delay(retry_count)
    
    # Then
    # base_delay (1.0) * backoff_multiplier (2.0) ^ retry_count (2) = 4.0
    expected_delay = 4.0
    assert delay == expected_delay


@pytest.mark.asyncio
async def test_calculate_delay_linear_backoff(retry_handler):
    """선형 백오프 지연 계산 테스트"""
    # Given
    retry_handler.config.strategy = RetryStrategy.LINEAR_BACKOFF
    retry_count = 2
    
    # When
    delay = retry_handler.calculate_delay(retry_count)
    
    # Then
    # base_delay (1.0) * (retry_count + 1) = 3.0
    expected_delay = 3.0
    assert delay == expected_delay


@pytest.mark.asyncio
async def test_calculate_delay_immediate(retry_handler):
    """즉시 재시도 지연 계산 테스트"""
    # Given
    retry_handler.config.strategy = RetryStrategy.IMMEDIATE
    retry_count = 2
    
    # When
    delay = retry_handler.calculate_delay(retry_count)
    
    # Then
    assert delay == 0.0


@pytest.mark.asyncio
async def test_schedule_retry_success(retry_handler, sample_order_request):
    """재시도 스케줄링 성공 테스트"""
    # Given
    retry_handler.publish_event = AsyncMock()
    error_type = "network"
    error_message = "Connection timeout"
    
    # When
    result = await retry_handler.schedule_retry(sample_order_request, error_type, error_message)
    
    # Then
    assert result is True
    assert len(retry_handler.retry_queue) == 1
    
    # 재시도 데이터 확인
    order_id = str(id(sample_order_request))
    retry_data = retry_handler.retry_queue[order_id]
    assert retry_data.retry_count == 1
    assert retry_data.retry_history[0]['error'] == error_message


@pytest.mark.asyncio
async def test_schedule_retry_max_exceeded(retry_handler, sample_order_request):
    """최대 재시도 횟수 초과 시 스케줄링 테스트"""
    # Given
    order_id = str(id(sample_order_request))
    retry_data = OrderRetryData(sample_order_request, datetime.now())
    retry_data.retry_count = 5  # 최대값 초과
    retry_handler.retry_queue[order_id] = retry_data
    
    # When
    result = await retry_handler.schedule_retry(sample_order_request, "network", "Error")
    
    # Then
    assert result is False
    assert order_id not in retry_handler.retry_queue  # 큐에서 제거됨


@pytest.mark.asyncio
async def test_cancel_retry(retry_handler, sample_order_request):
    """재시도 취소 테스트"""
    # Given
    order_id = str(id(sample_order_request))
    retry_data = OrderRetryData(sample_order_request, datetime.now())
    retry_handler.retry_queue[order_id] = retry_data
    
    # When
    result = retry_handler.cancel_retry(order_id)
    
    # Then
    assert result is True
    assert order_id not in retry_handler.retry_queue


@pytest.mark.asyncio
async def test_get_retry_status(retry_handler, sample_order_request):
    """재시도 상태 조회 테스트"""
    # Given
    order_id = str(id(sample_order_request))
    retry_data = OrderRetryData(sample_order_request, datetime.now())
    retry_data.retry_count = 2
    retry_handler.retry_queue[order_id] = retry_data
    
    # When
    status = retry_handler.get_retry_status(order_id)
    
    # Then
    assert status is not None
    assert status['retry_count'] == 2
    assert status['max_retries'] == retry_handler.config.max_retries
    assert status['is_cancelled'] is False


@pytest.mark.asyncio
async def test_get_all_pending_retries(retry_handler):
    """모든 대기 중인 재시도 조회 테스트"""
    # Given
    # 두 개의 재시도 항목 추가
    order1 = MagicMock(order_id="order1")
    order2 = MagicMock(order_id="order2")
    
    retry_data1 = OrderRetryData(order1, datetime.now())
    retry_data2 = OrderRetryData(order2, datetime.now())
    
    retry_handler.retry_queue["order1"] = retry_data1
    retry_handler.retry_queue["order2"] = retry_data2
    
    # 하나는 취소
    retry_data2.cancel()
    
    # When
    pending_retries = retry_handler.get_all_pending_retries()
    
    # Then
    assert len(pending_retries) == 1  # 취소되지 않은 것만
    assert pending_retries[0] is not None


@pytest.mark.asyncio
async def test_slippage_controller_start_stop(slippage_controller):
    """SlippageController 시작/중지 테스트"""
    # When
    await slippage_controller.start()
    assert slippage_controller.running
    
    await slippage_controller.stop()
    assert not slippage_controller.running


@pytest.mark.asyncio
async def test_retry_handler_start_stop(retry_handler):
    """OrderRetryHandler 시작/중지 테스트"""
    # When
    await retry_handler.start()
    assert retry_handler.running
    
    await retry_handler.stop()
    assert not retry_handler.running
    assert len(retry_handler.retry_queue) == 0  # 모든 재시도 취소됨