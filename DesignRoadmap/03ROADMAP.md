# Async ICT Trading System - Development Roadmap

## 1. 프로젝트 목표

- ICT(Inner Circle Trader) 이론에 기반하여 바이낸스 선물 시장에서 작동하는 완전 자동화된 비동기 거래 시스템 구축.
- `asyncio`를 활용하여 실시간 데이터 처리에 대한 고성능 및 확장성 확보.

## 2. 현재까지 완료된 상태 (As-Is)

- **분석 엔진 (Phase 1-3) 개발 완료**:
    - **시장 구조 분석**: `AsyncStructureBreakDetector` (BOS/CHoCH 탐지)
    - **오더 블록 분석**: `AsyncOrderBlockDetector`
    - **유동성 분석**: `AsyncLiquidityDetector` (유동성 풀, Liquidity Sweep 탐지)
    - **가격 불균형 분석**: `AsyncFVGDetector` (Fair Value Gap 탐지)
    - **시간 기반 분석**: `AsyncKillZoneManager` (런던/뉴욕 세션 분석)

- **통합/실행 프레임워크 (Phase 4) 기본 골격 구현**:
    - `AsyncTradingOrchestrator`를 중심으로 시스템의 전체 구조가 마련됨.
    - `AsyncStrategyCoordinator`, `AsyncRiskManager`, `AsyncOrderManager`의 기본 클래스만 생성되어 있으며, 내부 로직은 비어있는 상태.

## 3. 향후 개발 계획 (To-Be)

Phase 4 (통합 및 최적화)의 상세 구현을 아래 순서대로 진행합니다.

### **Step 1: 전략 종합 (Strategy Coordination)**

- **목표**: `AsyncStrategyCoordinator` 구현. 여러 분석 모듈에서 오는 신호(이벤트)들을 종합하여 의미 있는 거래 기회를 포착.
- **주요 작업**:
    1.  `EventBus`로부터 `FVGEvent`, `OrderBlockEvent`, `LiquidityEvent` 등 다양한 이벤트를 구독.
    2.  여러 조건이 결합되는 시나리오 정의 (예: "HTF 오더블록에 가격이 도달하고, LTF에서 FVG가 형성될 때").
    3.  정의된 시나리오 충족 시, `PreliminaryTradeDecision` (예비 거래 결정) 이벤트를 생성하여 `EventBus`에 발행.

### **Step 2: 리스크 관리 (Risk Management)**

- **목표**: `AsyncRiskManager` 구현. 예비 거래 결정에 대해 리스크를 평가하고 최종 거래 여부를 승인.
- **주요 작업**:
    1.  `PreliminaryTradeDecision` 이벤트를 구독.
    2.  포지션 사이징(Position Sizing) 계산: 계좌 잔액, 설정된 리스크 비율(예: 1회 거래당 1%)에 따라 거래 규모 결정.
    3.  계좌 전체 리스크 평가: 현재 총 노출, 최대 손실 허용 범위(Max Drawdown) 등과 비교하여 거래 승인/거부.
    4.  승인 시, 거래 규모와 손절/익절 가격이 포함된 `ApprovedTradeOrder` 이벤트를 발행.

### **Step 3: 주문 실행 (Order Execution)**

- **목표**: `AsyncOrderManager` 구현. 승인된 거래를 실제 거래소(바이낸스)에 전송하고 관리.
- **주요 작업**:
    1.  `ApprovedTradeOrder` 이벤트를 구독.
    2.  `infrastructure/binance/`의 API 클라이언트를 사용하여 실제 Market/Limit 주문 실행.
    3.  주문 상태(체결, 부분 체결, 취소)를 비동기적으로 추적.
    4.  주문 상태 변경 시 `OrderStateChangeEvent`를 발행하여 시스템의 다른 부분에 알림.

### **Step 4: 전체 통합 및 테스트**

- **목표**: `AsyncTradingOrchestrator` 내에서 Step 1~3의 컴포넌트들이 원활하게 연동되는지 확인.
- **주요 작업**:
    1.  모의(Mock) 이벤트를 사용하여 전체 거래 흐름(신호 발생 → 전략 종합 → 리스크 평가 → 주문 실행) 테스트.
    2.  실시간 데이터 스트림에 연결하여 시스템이 안정적으로 작동하는지 검증.
