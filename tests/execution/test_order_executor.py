"""
OrderExecutor 테스트
"""

import pytest
import asyncio
from decimal import Decimal
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from src.trading_bot.execution.order_executor import (
    OrderExecutor,
    OrderRequest,
    OrderExecutionError
)
from src.trading_bot.core.events import OrderEvent, EventType
from src.trading_bot.data.binance_client import BinanceClient


@pytest.fixture
def mock_binance_client():
    """모의 바이낸스 클라이언트"""
    client = MagicMock(spec=BinanceClient)
    client.place_order = AsyncMock()
    client.get_order_status = AsyncMock()
    client.cancel_order = AsyncMock()
    return client


@pytest.fixture
def order_executor(mock_binance_client):
    """OrderExecutor 인스턴스"""
    return OrderExecutor(binance_client=mock_binance_client)


@pytest.fixture
def sample_order_request():
    """샘플 주문 요청"""
    return OrderRequest(
        symbol="BTCUSDT",
        side="BUY",
        order_type="MARKET",
        quantity=Decimal("0.1"),
        price=None,
        time_in_force="GTC",
        client_order_id="test_order_001"
    )


@pytest.mark.asyncio
async def test_execute_order_success(order_executor, mock_binance_client, sample_order_request):
    """주문 실행 성공 테스트"""
    # Given
    mock_response = {
        'orderId': '12345',
        'clientOrderId': 'test_order_001',
        'symbol': 'BTCUSDT',
        'status': 'FILLED',
        'executedQty': '0.1',
        'fills': [{'price': '50000.0', 'qty': '0.1'}]
    }
    mock_binance_client.place_order.return_value = mock_response
    
    # When
    result = await order_executor.execute_order(sample_order_request)
    
    # Then
    assert isinstance(result, OrderEvent)
    assert result.symbol == "BTCUSDT"
    assert result.order_id == "12345"
    assert result.status == "FILLED"
    assert result.executed_quantity == 0.1
    mock_binance_client.place_order.assert_called_once()


@pytest.mark.asyncio
async def test_execute_order_failure(order_executor, mock_binance_client, sample_order_request):
    """주문 실행 실패 테스트"""
    # Given
    mock_binance_client.place_order.side_effect = Exception("Network error")
    
    # When & Then
    with pytest.raises(OrderExecutionError) as exc_info:
        await order_executor.execute_order(sample_order_request)
    
    assert "Network error" in str(exc_info.value)


@pytest.mark.asyncio
async def test_execute_order_with_retry(order_executor, mock_binance_client, sample_order_request):
    """재시도를 통한 주문 실행 테스트"""
    # Given
    order_executor.max_retries = 2
    mock_response = {
        'orderId': '12345',
        'clientOrderId': 'test_order_001',
        'symbol': 'BTCUSDT',
        'status': 'FILLED',
        'executedQty': '0.1',
        'fills': [{'price': '50000.0', 'qty': '0.1'}]
    }
    
    # 첫 번째 호출은 실패, 두 번째 호출은 성공
    mock_binance_client.place_order.side_effect = [
        Exception("Temporary error"),
        mock_response
    ]
    
    # When
    result = await order_executor.execute_order(sample_order_request)
    
    # Then
    assert isinstance(result, OrderEvent)
    assert result.order_id == "12345"
    assert mock_binance_client.place_order.call_count == 2


@pytest.mark.asyncio
async def test_monitor_order_status(order_executor, mock_binance_client):
    """주문 상태 모니터링 테스트"""
    # Given
    order_id = "12345"
    symbol = "BTCUSDT"
    mock_binance_client.get_order_status.return_value = {
        'orderId': order_id,
        'status': 'FILLED',
        'executedQty': '0.1'
    }
    
    # When
    status = await order_executor.monitor_order_status(order_id, symbol)
    
    # Then
    assert status['status'] == 'FILLED'
    mock_binance_client.get_order_status.assert_called_once_with(order_id, symbol)


@pytest.mark.asyncio
async def test_cancel_order(order_executor, mock_binance_client):
    """주문 취소 테스트"""
    # Given
    order_id = "12345"
    symbol = "BTCUSDT"
    mock_binance_client.cancel_order.return_value = {
        'orderId': order_id,
        'status': 'CANCELED'
    }
    
    # When
    result = await order_executor.cancel_order(order_id, symbol)
    
    # Then
    assert result['status'] == 'CANCELED'
    mock_binance_client.cancel_order.assert_called_once_with(order_id, symbol)


@pytest.mark.asyncio
async def test_validate_order_request_invalid_symbol(order_executor):
    """잘못된 심볼로 주문 요청 검증 테스트"""
    # Given
    invalid_request = OrderRequest(
        symbol="INVALID",
        side="BUY",
        order_type="MARKET",
        quantity=Decimal("0.1")
    )
    
    # When & Then
    with pytest.raises(OrderExecutionError) as exc_info:
        await order_executor.execute_order(invalid_request)
    
    assert "Invalid symbol" in str(exc_info.value)


@pytest.mark.asyncio
async def test_validate_order_request_invalid_quantity(order_executor):
    """잘못된 수량으로 주문 요청 검증 테스트"""
    # Given
    invalid_request = OrderRequest(
        symbol="BTCUSDT",
        side="BUY",
        order_type="MARKET",
        quantity=Decimal("0")  # 0 수량
    )
    
    # When & Then
    with pytest.raises(OrderExecutionError) as exc_info:
        await order_executor.execute_order(invalid_request)
    
    assert "Invalid quantity" in str(exc_info.value)


@pytest.mark.asyncio
async def test_order_executor_start_stop(order_executor):
    """OrderExecutor 시작/중지 테스트"""
    # When
    await order_executor.start()
    assert order_executor.running
    
    await order_executor.stop()
    assert not order_executor.running


@pytest.mark.asyncio
async def test_order_timeout_handling(order_executor, mock_binance_client, sample_order_request):
    """주문 타임아웃 처리 테스트"""
    # Given
    order_executor.order_timeout = 0.1  # 짧은 타임아웃 설정
    
    async def slow_response(*args, **kwargs):
        await asyncio.sleep(0.2)  # 타임아웃보다 오래 걸림
        return {'orderId': '12345', 'status': 'NEW'}
    
    mock_binance_client.place_order.side_effect = slow_response
    
    # When & Then
    with pytest.raises(OrderExecutionError) as exc_info:
        await order_executor.execute_order(sample_order_request)
    
    assert "timeout" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_partial_fill_handling(order_executor, mock_binance_client, sample_order_request):
    """부분 체결 처리 테스트"""
    # Given
    mock_response = {
        'orderId': '12345',
        'clientOrderId': 'test_order_001',
        'symbol': 'BTCUSDT',
        'status': 'PARTIALLY_FILLED',
        'executedQty': '0.05',  # 절반만 체결
        'origQty': '0.1',
        'fills': [{'price': '50000.0', 'qty': '0.05'}]
    }
    mock_binance_client.place_order.return_value = mock_response
    
    # When
    result = await order_executor.execute_order(sample_order_request)
    
    # Then
    assert result.status == "PARTIALLY_FILLED"
    assert result.executed_quantity == 0.05
    assert result.remaining_quantity == 0.05