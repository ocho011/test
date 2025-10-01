# 테스트 안정화 계획 - 2025년 세션

## 현재 상태 (2025-10-01)
- BacktestEngine: 11/11 passing ✅
- MarketDataProvider: 18/21 passing (3 skipped)
- OrderExecutor: 5/13 passing (8 skipped)
- PositionTracker: 3/12 passing (9 skipped)
- **총 19개 스킵된 테스트**

## Phase 1: test_order_executor.py (8개 수정)

### 파일: tests/execution/test_order_executor.py

#### 1. test_monitor_order_status (라인 121-139) - 제거
- 이유: OrderExecutor에 monitor_order_status 메서드 존재하지 않음
- 액션: 테스트 전체 제거

#### 2. test_cancel_order (라인 142-159) - 수정
- 현재: `cancel_order(order_id, symbol)`
- 실제: `cancel_order(client_order_id)`
- 참고: src/trading_bot/execution/order_executor.py:419-449

#### 3. test_order_timeout_handling (라인 209-226) - 제거
- 이유: OrderExecutor에 order_timeout 속성 없음

#### 4. test_partial_fill_handling (라인 229-252) - 수정
- 이유: OrderEvent에 remaining_quantity 속성 없음
- 액션: filled_quantity만 사용하도록 수정

#### 5. test_order_status_updates (라인 162-182) - 확인 필요
- 스킵 사유 확인 필요

#### 6. test_order_rejection (라인 185-206) - 확인 필요
- 스킵 사유 확인 필요

#### 7-8. 추가 스킵 테스트들 확인 필요

## Phase 2: test_position_tracker.py (9개 수정)

### 파일: tests/execution/test_position_tracker.py

#### 이벤트 기반 아키텍처 패턴
PositionTracker는 add_position/update_position_price 같은 직접 메서드가 없고
OrderEvent를 받아서 처리하는 이벤트 기반 아키텍처 사용

#### Position 클래스 실제 시그니처:
```python
Position(
    position_id: str,
    symbol: str,
    side: PositionSide,
    size: Decimal,
    entry_price: Decimal,
    stop_loss: Optional[Decimal] = None,
    take_profit: Optional[Decimal] = None
)
# leverage, margin_required 파라미터 없음
```

#### 수정 필요 테스트들:

##### 1. test_add_position (라인 51-62) - 재작성
- 현재: `position_tracker.add_position(position)` 직접 호출
- 변경: OrderEvent 생성 → `handle_order_event()` 호출로 변경
- 패턴:
```python
order_event = OrderEvent(
    order_id="test_order_1",
    client_order_id="client_1",
    symbol="BTCUSDT",
    side=OrderSide.BUY,
    order_type=OrderType.MARKET,
    quantity=Decimal("1.0"),
    price=Decimal("50000"),
    status=OrderStatus.FILLED,
    filled_quantity=Decimal("1.0"),
    timestamp=datetime.now(timezone.utc)
)
await position_tracker.handle_order_event(order_event)
```

##### 2. test_update_position_price (라인 64-78) - 재작성
- 현재: `position_tracker.update_position_price(position_id, new_price)`
- 변경: 가격 업데이트는 새로운 주문 이벤트로 처리
- 또는: 실시간 가격 업데이트 로직 확인 필요

##### 3. test_calculate_roe (라인 119-133) - 제거
- 이유: Position 클래스에 calculate_roe 메서드 없음
- 액션: 테스트 제거 또는 다른 방식으로 ROE 계산 검증

##### 4. test_close_position (라인 136-152) - 재작성
- 현재: `position_tracker.close_position(position_id, close_price)`
- 변경: SELL OrderEvent로 포지션 종료
```python
close_order_event = OrderEvent(
    order_id="close_order_1",
    client_order_id="client_close_1",
    symbol="BTCUSDT",
    side=OrderSide.SELL,  # 반대 방향
    order_type=OrderType.MARKET,
    quantity=position.size,
    price=close_price,
    status=OrderStatus.FILLED,
    filled_quantity=position.size,
    timestamp=datetime.now(timezone.utc)
)
await position_tracker.handle_order_event(close_order_event)
```

##### 5. test_get_total_pnl (라인 154-191) - 재작성
- 이벤트 기반으로 포지션 생성 후 PnL 계산 검증

##### 6. test_get_positions_by_symbol (라인 194-227) - 재작성
- 이벤트 기반으로 여러 포지션 생성 후 필터링 검증

##### 7. test_position_risk_level (라인 230-253) - 제거
- 이유: get_risk_level 메서드 없음
- 액션: 테스트 제거

##### 8. test_position_margin_level (라인 256-277) - 제거
- 이유: get_margin_level 메서드 없음
- 액션: 테스트 제거

##### 9. test_position_event_publishing (라인 290-310) - 재작성
- 이벤트 기반 아키텍처로 수정
- PositionEvent 발행 검증

## Phase 3: test_market_data_provider.py (3개 수정)

### 파일: tests/data/test_market_data_provider.py

#### 1. test_reconnection_logic (라인 266-285) - 제거
- 이유: MarketDataProvider에 `_connect_websocket`, `_handle_reconnection` 메서드 없음
- 액션: 테스트 전체 제거
- 참고: 실제 구현에서는 binance-connector-python의 자동 재연결 기능 사용

#### 2. test_subscription_management_during_reconnection (라인 287-302) - 제거
- 이유: `_connected` 속성, `_handle_reconnection` 메서드 없음
- 액션: 테스트 전체 제거

#### 3. test_subscription_status_tracking (라인 316-332) - 제거
- 이유: StreamSubscription에 `is_active()`, `pause()`, `resume()` 메서드 없음
- 액션: 테스트 전체 제거
- 참고: StreamSubscription은 단순 데이터 클래스 (dataclass)

## 테스트 헬퍼 함수 패턴

### OrderEvent 생성 헬퍼
```python
def create_order_event(
    order_id: str,
    symbol: str,
    side: OrderSide,
    quantity: Decimal,
    price: Decimal,
    status: OrderStatus = OrderStatus.FILLED
) -> OrderEvent:
    return OrderEvent(
        order_id=order_id,
        client_order_id=f"client_{order_id}",
        symbol=symbol,
        side=side,
        order_type=OrderType.MARKET,
        quantity=quantity,
        price=price,
        status=status,
        filled_quantity=quantity if status == OrderStatus.FILLED else Decimal("0"),
        timestamp=datetime.now(timezone.utc)
    )
```

### Position 생성 헬퍼 (이벤트 기반)
```python
async def create_position_via_event(
    position_tracker: PositionTracker,
    symbol: str,
    side: OrderSide,
    size: Decimal,
    entry_price: Decimal
) -> str:
    """이벤트를 통해 포지션 생성하고 position_id 반환"""
    order_event = create_order_event(
        order_id=f"open_{symbol}_{datetime.now().timestamp()}",
        symbol=symbol,
        side=side,
        quantity=size,
        price=entry_price,
        status=OrderStatus.FILLED
    )
    await position_tracker.handle_order_event(order_event)

    # 생성된 포지션 찾기
    positions = position_tracker.get_positions_by_symbol(symbol)
    return positions[-1].position_id if positions else None
```

## 실행 순서

1. **OrderExecutor 테스트 수정**
   ```bash
   pytest tests/execution/test_order_executor.py -v
   # 목표: 13/13 passing
   ```

2. **PositionTracker 테스트 수정**
   ```bash
   pytest tests/execution/test_position_tracker.py -v
   # 목표: 12/12 passing (또는 불필요한 테스트 제거 후 적절한 수)
   ```

3. **MarketDataProvider 테스트 수정**
   ```bash
   pytest tests/data/test_market_data_provider.py -v
   # 목표: 18/18 passing (3개 제거)
   ```

4. **전체 테스트 스위트 실행**
   ```bash
   pytest tests/ -v --tb=short
   # 목표: 100% passing, 0 skipped
   ```

5. **Task #11 (전략 추상화) 진행**

## 참고 파일들
- `src/trading_bot/execution/order_executor.py` - OrderExecutor 실제 구현
- `src/trading_bot/execution/position_tracker.py` - PositionTracker 실제 구현
- `src/trading_bot/data/market_data_provider.py` - MarketDataProvider 실제 구현
- `src/trading_bot/core/events.py` - OrderEvent, PositionEvent 정의
- `src/trading_bot/core/types.py` - Position, OrderRequest 등 타입 정의

## 주요 아키텍처 패턴

### 이벤트 기반 아키텍처
PositionTracker는 직접적인 CRUD 메서드(`add_position`, `update_position`, `close_position`)가 없습니다.
대신 OrderEvent를 받아서 내부적으로 포지션을 관리합니다:

```
OrderEvent → PositionTracker.handle_order_event() → 내부 포지션 업데이트 → PositionEvent 발행
```

### 실제 구현 확인 방법
테스트를 수정하기 전에 반드시 실제 구현을 확인:
```python
# 메서드 존재 여부 확인
import inspect
print(inspect.getmembers(OrderExecutor, predicate=inspect.ismethod))

# 클래스 시그니처 확인
print(inspect.signature(Position.__init__))
```

## 이전 작업 완료 기록

### Task #1-3: MarketDataProvider, Position, OrderRequest (완료)
- MarketDataProvider: 18/21 passing (3 skipped)
- Position: leverage 파라미터 추가 완료
- OrderRequest: enum 값 검증 완료

### Task #4: BacktestEngine (완료 ✅)
- 11/11 passing (100%)
- 주요 수정 사항:
  - Strategy 함수 시그니처 수정 (engine, bar)
  - 포지션 종료 후 수동 삭제 추가
  - 슬리피지 고려한 exit_price 검증
  - Async/sync 패턴 수정
  - Initial capital 타이밍 수정
