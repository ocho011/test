"""
슬리피지 제어 및 주문 재시도 핸들러

이 모듈은 슬리피지 모니터링, 제어 및 주문 재시도 로직을 구현합니다.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, Field

from ..core.base_component import BaseComponent
from ..core.events import EventType, OrderEvent, SlippageEvent


class SlippageType(Enum):
    """슬리피지 유형"""
    POSITIVE = "positive"  # 유리한 슬리피지
    NEGATIVE = "negative"  # 불리한 슬리피지
    EXCESSIVE = "excessive"  # 과도한 슬리피지


class RetryStrategy(Enum):
    """재시도 전략"""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    IMMEDIATE = "immediate"
    NO_RETRY = "no_retry"


@dataclass
class SlippageData:
    """슬리피지 데이터"""
    order_id: str
    symbol: str
    expected_price: Decimal
    executed_price: Decimal
    slippage_amount: Decimal
    slippage_percentage: Decimal
    slippage_type: SlippageType
    timestamp: datetime


class SlippageConfig(BaseModel):
    """슬리피지 제어 설정"""
    max_slippage_percentage: float = Field(default=0.1, description="최대 허용 슬리피지 (%)")
    excessive_slippage_threshold: float = Field(default=0.5, description="과도한 슬리피지 임계값 (%)")
    monitoring_enabled: bool = Field(default=True, description="슬리피지 모니터링 활성화")
    auto_cancel_on_excessive: bool = Field(default=True, description="과도한 슬리피지 시 자동 취소")
    slippage_tolerance_by_symbol: Dict[str, float] = Field(default_factory=dict, description="심볼별 슬리피지 허용치")


class RetryConfig(BaseModel):
    """재시도 설정"""
    max_retries: int = Field(default=3, description="최대 재시도 횟수")
    base_delay: float = Field(default=1.0, description="기본 지연 시간 (초)")
    max_delay: float = Field(default=30.0, description="최대 지연 시간 (초)")
    backoff_multiplier: float = Field(default=2.0, description="백오프 배수")
    strategy: RetryStrategy = Field(default=RetryStrategy.EXPONENTIAL_BACKOFF, description="재시도 전략")
    retry_on_slippage: bool = Field(default=True, description="슬리피지 발생 시 재시도")
    retry_on_timeout: bool = Field(default=True, description="타임아웃 시 재시도")
    retry_on_network_error: bool = Field(default=True, description="네트워크 오류 시 재시도")


class OrderRetryData:
    """주문 재시도 데이터"""
    
    def __init__(self, order_request: Any, original_timestamp: datetime):
        self.order_request = order_request
        self.original_timestamp = original_timestamp
        self.retry_count = 0
        self.last_retry_timestamp: Optional[datetime] = None
        self.retry_history: List[Dict[str, Any]] = []
        self.is_cancelled = False
        
    def add_retry_attempt(self, error: str, delay: float) -> None:
        """재시도 시도 기록 추가"""
        self.retry_count += 1
        self.last_retry_timestamp = datetime.now()
        self.retry_history.append({
            'attempt': self.retry_count,
            'timestamp': self.last_retry_timestamp,
            'error': error,
            'delay': delay
        })
        
    def cancel(self) -> None:
        """재시도 취소"""
        self.is_cancelled = True


class SlippageController(BaseComponent):
    """슬리피지 제어기"""

    def __init__(self, config: SlippageConfig):
        super().__init__(name=self.__class__.__name__)
        self.config = config
        self.slippage_history: List[SlippageData] = []
        self.symbol_slippage_stats: Dict[str, Dict[str, Any]] = {}

    async def _start(self) -> None:
        """컴포넌트 시작"""
        self.logger.info("SlippageController started")

    async def _stop(self) -> None:
        """컴포넌트 중지"""
        self.logger.info("SlippageController stopped")
        
    def calculate_slippage(
        self,
        expected_price: Decimal,
        executed_price: Decimal,
        side: str
    ) -> tuple[Decimal, Decimal, SlippageType]:
        """슬리피지 계산"""
        if side.upper() == "BUY":
            # 매수 시: 실행가격이 예상보다 높으면 불리한 슬리피지
            slippage_amount = executed_price - expected_price
        else:
            # 매도 시: 실행가격이 예상보다 낮으면 불리한 슬리피지
            slippage_amount = expected_price - executed_price
            
        slippage_percentage = (slippage_amount / expected_price) * 100
        
        # 슬리피지 유형 결정
        if slippage_amount > 0:
            if slippage_percentage > self.config.excessive_slippage_threshold:
                slippage_type = SlippageType.EXCESSIVE
            else:
                slippage_type = SlippageType.NEGATIVE
        else:
            slippage_type = SlippageType.POSITIVE
            
        return slippage_amount, slippage_percentage, slippage_type
        
    def is_slippage_acceptable(self, symbol: str, slippage_percentage: float) -> bool:
        """슬리피지 허용 여부 확인"""
        # 심볼별 허용치가 설정되어 있으면 사용
        if symbol in self.config.slippage_tolerance_by_symbol:
            max_allowed = self.config.slippage_tolerance_by_symbol[symbol]
        else:
            max_allowed = self.config.max_slippage_percentage
            
        return abs(slippage_percentage) <= max_allowed
        
    async def validate_order_execution(
        self,
        order_id: str,
        symbol: str,
        expected_price: Decimal,
        executed_price: Decimal,
        side: str
    ) -> tuple[bool, Optional[SlippageData]]:
        """주문 실행 검증"""
        if not self.config.monitoring_enabled:
            return True, None
            
        # 슬리피지 계산
        slippage_amount, slippage_percentage, slippage_type = self.calculate_slippage(
            expected_price, executed_price, side
        )
        
        # 슬리피지 데이터 생성
        slippage_data = SlippageData(
            order_id=order_id,
            symbol=symbol,
            expected_price=expected_price,
            executed_price=executed_price,
            slippage_amount=slippage_amount,
            slippage_percentage=slippage_percentage,
            slippage_type=slippage_type,
            timestamp=datetime.now()
        )
        
        # 슬리피지 기록 저장
        self.slippage_history.append(slippage_data)
        self._update_symbol_stats(symbol, slippage_data)
        
        # 슬리피지 이벤트 발행
        await self.publish_event(SlippageEvent(
            symbol=symbol,
            order_id=order_id,
            expected_price=float(expected_price),
            executed_price=float(executed_price),
            slippage_amount=float(slippage_amount),
            slippage_percentage=float(slippage_percentage),
            slippage_type=slippage_type.value,
            timestamp=slippage_data.timestamp
        ))
        
        # 과도한 슬리피지 처리
        if slippage_type == SlippageType.EXCESSIVE and self.config.auto_cancel_on_excessive:
            self.logger.warning(
                f"Excessive slippage detected for {symbol}: {slippage_percentage:.2f}%. "
                f"Order {order_id} execution rejected."
            )
            return False, slippage_data
            
        # 슬리피지 허용 여부 확인
        is_acceptable = self.is_slippage_acceptable(symbol, slippage_percentage)
        
        if not is_acceptable:
            self.logger.warning(
                f"Slippage exceeds tolerance for {symbol}: {slippage_percentage:.2f}%. "
                f"Order {order_id} execution rejected."
            )
            
        return is_acceptable, slippage_data
        
    def _update_symbol_stats(self, symbol: str, slippage_data: SlippageData) -> None:
        """심볼별 슬리피지 통계 업데이트"""
        if symbol not in self.symbol_slippage_stats:
            self.symbol_slippage_stats[symbol] = {
                'total_orders': 0,
                'total_slippage': Decimal('0'),
                'positive_slippage_count': 0,
                'negative_slippage_count': 0,
                'excessive_slippage_count': 0,
                'average_slippage': Decimal('0'),
                'max_slippage': Decimal('0'),
                'min_slippage': Decimal('0')
            }
            
        stats = self.symbol_slippage_stats[symbol]
        stats['total_orders'] += 1
        stats['total_slippage'] += slippage_data.slippage_percentage
        
        if slippage_data.slippage_type == SlippageType.POSITIVE:
            stats['positive_slippage_count'] += 1
        elif slippage_data.slippage_type == SlippageType.NEGATIVE:
            stats['negative_slippage_count'] += 1
        elif slippage_data.slippage_type == SlippageType.EXCESSIVE:
            stats['excessive_slippage_count'] += 1
            
        # 평균, 최대, 최소 슬리피지 업데이트
        stats['average_slippage'] = stats['total_slippage'] / stats['total_orders']
        stats['max_slippage'] = max(stats['max_slippage'], slippage_data.slippage_percentage)
        stats['min_slippage'] = min(stats['min_slippage'], slippage_data.slippage_percentage)
        
    def get_symbol_slippage_stats(self, symbol: str) -> Optional[Dict[str, Any]]:
        """심볼별 슬리피지 통계 조회"""
        return self.symbol_slippage_stats.get(symbol)
        
    def get_recent_slippage_data(self, hours: int = 24) -> List[SlippageData]:
        """최근 슬리피지 데이터 조회"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            data for data in self.slippage_history
            if data.timestamp >= cutoff_time
        ]


class OrderRetryHandler(BaseComponent):
    """주문 재시도 핸들러"""
    
    def __init__(self, config: RetryConfig):
        super().__init__(name=self.__class__.__name__)
        self.config = config
        self.retry_queue: Dict[str, OrderRetryData] = {}

    async def _start(self) -> None:
        """컴포넌트 시작"""
        self.logger.info("OrderRetryHandler started")

    async def _stop(self) -> None:
        """컴포넌트 중지"""
        # 대기 중인 재시도 취소
        for retry_data in self.retry_queue.values():
            retry_data.cancel()
        self.retry_queue.clear()
        self.logger.info("OrderRetryHandler stopped")
        
    def should_retry(self, error_type: str, retry_count: int) -> bool:
        """재시도 여부 결정"""
        if retry_count >= self.config.max_retries:
            return False
            
        if error_type == "slippage" and not self.config.retry_on_slippage:
            return False
        elif error_type == "timeout" and not self.config.retry_on_timeout:
            return False
        elif error_type == "network" and not self.config.retry_on_network_error:
            return False
            
        return True
        
    def calculate_delay(self, retry_count: int) -> float:
        """재시도 지연 시간 계산"""
        if self.config.strategy == RetryStrategy.IMMEDIATE:
            return 0.0
        elif self.config.strategy == RetryStrategy.NO_RETRY:
            return float('inf')
        elif self.config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.config.base_delay * (retry_count + 1)
        else:  # EXPONENTIAL_BACKOFF
            delay = self.config.base_delay * (self.config.backoff_multiplier ** retry_count)
            
        return min(delay, self.config.max_delay)
        
    async def schedule_retry(
        self,
        order_request: Any,
        error_type: str,
        error_message: str
    ) -> bool:
        """주문 재시도 스케줄링"""
        order_id = getattr(order_request, 'order_id', str(id(order_request)))
        
        # 기존 재시도 데이터 조회 또는 생성
        if order_id not in self.retry_queue:
            self.retry_queue[order_id] = OrderRetryData(
                order_request=order_request,
                original_timestamp=datetime.now()
            )
            
        retry_data = self.retry_queue[order_id]
        
        # 재시도 가능 여부 확인
        if not self.should_retry(error_type, retry_data.retry_count):
            self.logger.info(
                f"Max retries reached or retry not allowed for order {order_id}. "
                f"Error type: {error_type}, Retry count: {retry_data.retry_count}"
            )
            self.retry_queue.pop(order_id, None)
            return False
            
        # 지연 시간 계산
        delay = self.calculate_delay(retry_data.retry_count)
        
        # 재시도 기록
        retry_data.add_retry_attempt(error_message, delay)
        
        self.logger.info(
            f"Scheduling retry {retry_data.retry_count}/{self.config.max_retries} "
            f"for order {order_id} in {delay:.2f} seconds. Error: {error_message}"
        )
        
        # 지연 후 재시도 실행
        asyncio.create_task(self._execute_retry(order_id, delay))
        
        return True
        
    async def _execute_retry(self, order_id: str, delay: float) -> None:
        """재시도 실행"""
        await asyncio.sleep(delay)
        
        retry_data = self.retry_queue.get(order_id)
        if not retry_data or retry_data.is_cancelled:
            return
            
        # Note: Retry event publishing disabled - generic Event class not available
        # Event publishing can be re-enabled with proper OrderEvent structure if needed
        pass
        
        self.logger.info(f"Executing retry {retry_data.retry_count} for order {order_id}")
        
    def cancel_retry(self, order_id: str) -> bool:
        """재시도 취소"""
        retry_data = self.retry_queue.get(order_id)
        if retry_data:
            retry_data.cancel()
            self.retry_queue.pop(order_id, None)
            self.logger.info(f"Cancelled retry for order {order_id}")
            return True
        return False
        
    def get_retry_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """재시도 상태 조회"""
        retry_data = self.retry_queue.get(order_id)
        if not retry_data:
            return None
            
        return {
            'order_id': order_id,
            'retry_count': retry_data.retry_count,
            'max_retries': self.config.max_retries,
            'original_timestamp': retry_data.original_timestamp,
            'last_retry_timestamp': retry_data.last_retry_timestamp,
            'is_cancelled': retry_data.is_cancelled,
            'retry_history': retry_data.retry_history
        }
        
    def get_all_pending_retries(self) -> List[Dict[str, Any]]:
        """모든 대기 중인 재시도 조회"""
        return [
            self.get_retry_status(order_id)
            for order_id in self.retry_queue.keys()
            if not self.retry_queue[order_id].is_cancelled
        ]