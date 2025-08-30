## 비동기 개발 단계별 상세 계획

### Phase 1: 비동기 기본 구조 분석 (4주)

#### Week 1-2: Async Market Structure Foundation
**목표**: 비동기 기반 시장 구조 인식 시스템 구축

**비즈니스 분석 업무**:
- 실시간 다중 시간대 분석을 위한 동시성 요구사항 정의
- WebSocket 데이터 스트림 기반 BOS/CHoCH 탐지 성능 기준 설정
- 비동기 이벤트 기반 시그널 전파 메커니즘 설계

**비동기 설계 업무**:
```python
class AsyncMarketStructure:
    def __init__(self, event_bus: AsyncEventBus):
        self.swing_highs: List[SwingPoint] = []
        self.swing_lows: List[SwingPoint] = []
        self.current_trend: TrendDirection = TrendDirection.UNKNOWN
        self.event_bus = event_bus
        self._analysis_tasks: Set[asyncio.Task] = set()
        
    async def start_real_time_analysis(self, symbols: List[str], timeframes: List[str]):
        """실시간 다중 심볼/시간대 구조 분석 시작"""
        for symbol in symbols:
            for timeframe in timeframes:
                task = asyncio.create_task(
                    self._continuous_structure_analysis(symbol, timeframe)
                )
                self._analysis_tasks.add(task)
        
    async def _continuous_structure_analysis(self, symbol: str, timeframe: str):
        """지속적인 구조 분석 (백그라운드 코루틴)"""
        while True:
            try:
                # WebSocket에서 실시간 캔들 데이터 수신
                async for candle in self._get_candle_stream(symbol, timeframe):
                    # BOS 탐지
                    bos_result = await self._detect_break_of_structure_async(candle)
                    if bos_result:
                        await self.event_bus.publish(MarketStructureEvent(
                            symbol=symbol, timeframe=timeframe, 
                            event_type="BOS_DETECTED", data=bos_result
                        ))
                    
                    # CHoCH 탐지  
                    choch_result = await self._detect_change_of_character_async(candle)
                    if choch_result:
                        await self.event_bus.publish(MarketStructureEvent(
                            symbol=symbol, timeframe=timeframe,
                            event_type="CHOCH_DETECTED", data=choch_result
                        ))
                        
            except Exception as e:
                logger.error(f"Structure analysis error for {symbol}_{timeframe}: {e}")
                await asyncio.sleep(5)  # 에러 복구 대기
                
    async def _detect_break_of_structure_async(self, candle: Candle) -> Optional[BOS]:
        """비동기 BOS 탐지"""
        # CPU 집약적 작업을 executor에서 실행
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._calculate_bos, candle
        )

class AsyncStructureBreakDetector:
    def __init__(self, event_bus: AsyncEventBus):
        self.event_bus = event_bus
        self.timeframe_structures: Dict[str, AsyncMarketStructure] = {}
        
    async def start_multi_timeframe_detection(self):
        """멀티 타임프레임 구조 탐지 시작"""
        tasks = []
        for tf, structure in self.timeframe_structures.items():
            task = asyncio.create_task(structure.start_real_time_analysis())
            tasks.append(task)
        
        await asyncio.gather(*tasks)
```

#### Week 3-4: Async Order Block Detection
**비즈니스 분석 업무**:
- 실시간 Order Block 유효성 갱신 주기 최적화 (지연시간 vs 정확도)
- 다중 시간대 Order Block 간 우선순위 결정 로직
- 메모리 효율적인 Order Block 관리 방안 (가비지 컬렉션 정책)

**비동기 설계 업무**:
```python
class AsyncOrderBlock:
    def __init__(self, candle: Candle, block_type: OrderBlockType, event_bus: AsyncEventBus):
        self.origin_candle = candle
        self.high = candle.high
        self.low = candle.low
        self.block_type = block_type
        self.validity_score = 0.0
        self.touch_count = 0
        self.creation_time = candle.timestamp
        self.event_bus = event_bus
        self._monitoring_task: Optional[asyncio.Task] = None
        
    async def start_monitoring(self):
        """Order Block 모니터링 시작"""
        self._monitoring_task = asyncio.create_task(self._monitor_price_action())
        
    async def _monitor_price_action(self):
        """가격 반응 모니터링 (백그라운드 코루틴)"""
        while not self.is_invalidated:
            try:
                current_price = await self._get_current_price()
                
                if self.is_price_in_block(current_price):
                    self.touch_count += 1
                    await self._handle_block_touch(current_price)
                    
                # 유효성 점수 비동기 갱신
                new_validity = await self._calculate_validity_async()
                if abs(new_validity - self.validity_score) > 0.1:
                    self.validity_score = new_validity
                    await self.event_bus.publish(OrderBlockEvent(
                        event_type="VALIDITY_UPDATED",
                        order_block=self,
                        new_validity=new_validity
                    ))
                    
                await asyncio.sleep(0.1)  # 100ms마다 체크
                
            except Exception as e:
                logger.error(f"Order Block monitoring error: {e}")
                await asyncio.sleep(1)
                
    async def _calculate_validity_async(self) -> float:
        """비동기 유효성 계산"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._calculate_validity_sync
        )

class AsyncOrderBlockDetector:
    def __init__(self, event_bus: AsyncEventBus):
        self.event_bus = event_bus
        self.active_blocks: Dict[str, List[AsyncOrderBlock]] = {}
        self._detection_tasks: Set[asyncio.Task] = set()
        
    async def start_continuous_detection(self, symbols: List[str], timeframes: List[str]):
        """지속적인 Order Block 탐지 시작"""
        for symbol in symbols:
            for timeframe in timeframes:
                task = asyncio.create_task(
                    self._detect_order_blocks_continuously(symbol, timeframe)
                )
                self._detection_tasks.add(task)
                
    async def _detect_order_blocks_continuously(self, symbol: str, timeframe: str):
        """지속적인 Order Block 탐지"""
        candle_buffer = []
        
        async for candle in self._get_candle_stream(symbol, timeframe):
            candle_buffer.append(candle)
            if len(candle_buffer) > 100:  # 최근 100개 캔들만 유지
                candle_buffer.pop(0)
                
            # 비동기로 Order Block 탐지
            new_blocks = await self._detect_new_order_blocks(candle_buffer)
            
            for block in new_blocks:
                await block.start_monitoring()
                key = f"{symbol}_{timeframe}"
                if key not in self.active_blocks:
                    self.active_blocks[key] = []
                self.active_blocks[key].append(block)
                
                # 새로운 Order Block 이벤트 발행
                await self.event_bus.publish(OrderBlockEvent(
                    event_type="NEW_ORDER_BLOCK",
                    symbol=symbol,
                    timeframe=timeframe,
                    order_block=block
                ))
```

### Phase 2: 비동기 유동성 분석 (3주)

#### Week 5-6: Async Liquidity Pool Management
**비즈니스 분석 업무**:
- 실시간 Equal Highs/Lows 탐지 성능 최적화 (메모리 사용량 vs 탐지 정확도)
- 유동성 풀 중요도 점수의 실시간 갱신 알고리즘 설계
- 다중 심볼 간 유동성 상관관계 분석 (BTC vs ETH 유동성 연동)

**비동기 설계 업무**:
```python
class AsyncLiquidityPool:
    def __init__(self, price_level: float, pool_type: LiquidityType, event_bus: AsyncEventBus):
        self.price_level = price_level
        self.pool_type = pool_type
        self.touch_points: List[TouchPoint] = []
        self.importance_score = 0.0
        self.is_swept = False
        self.event_bus = event_bus
        self._monitoring_task: Optional[asyncio.Task] = None
        
    async def start_monitoring(self):
        """유동성 풀 모니터링 시작"""
        self._monitoring_task = asyncio.create_task(self._monitor_liquidity_interactions())
        
    async def _monitor_liquidity_interactions(self):
        """유동성 상호작용 모니터링"""
        while not self.is_swept:
            try:
                current_price = await self._get_current_price()
                order_book = await self._get_current_order_book()
                
                # 가격이 유동성 레벨에 접근했는지 확인
                if self._is_price_approaching(current_price):
                    await self._handle_liquidity_approach(current_price, order_book)
                
                # 유동성 사냥 탐지
                sweep_detected = await self._detect_liquidity_sweep(current_price)
                if sweep_detected:
                    self.is_swept = True
                    await self.event_bus.publish(LiquidityEvent(
                        event_type="LIQUIDITY_SWEPT",
                        pool=self,
                        sweep_data=sweep_detected
                    ))
                    break
                    
                await asyncio.sleep(0.05)  # 50ms마다 체크 (고빈도)
                
            except Exception as e:
                logger.error(f"Liquidity monitoring error: {e}")
                await asyncio.sleep(1)

class AsyncLiquidityDetector:
    def __init__(self, event_bus: AsyncEventBus, tolerance_percent: float = 0.1):
        self.tolerance = tolerance_percent
        self.event_bus = event_bus
        self.active_pools: Dict[str, List[AsyncLiquidityPool]] = {}
        self._detection_tasks: Set[asyncio.Task] = set()
        
    async def start_multi_symbol_detection(self, symbols: List[str]):
        """다중 심볼 유동성 탐지 시작"""
        for symbol in symbols:
            task = asyncio.create_task(self._detect_liquidity_continuously(symbol))
            self._detection_tasks.add(task)
            
        # 심볼 간 유동성 상관관계 분석 태스크
        correlation_task = asyncio.create_task(self._analyze_cross_symbol_liquidity())
        self._detection_tasks.add(correlation_task)
        
    async def _detect_liquidity_continuously(self, symbol: str):
        """지속적인 유동성 탐지"""
        price_history = deque(maxlen=200)  # 최근 200개 가격 포인트
        
        async for price_update in self._get_price_stream(symbol):
            price_history.append(price_update)
            
            if len(price_history) >= 50:  # 최소 50개 데이터 점이 있을 때
                # 비동기로 Equal Highs/Lows 탐지
                equal_highs = await self._find_equal_highs_async(list(price_history))
                equal_lows = await self._find_equal_lows_async(list(price_history))
                
                # 새로운 유동성 풀 생성 및 모니터링 시작
                for high_level in equal_highs:
                    if not self._pool_exists(symbol, high_level, LiquidityType.BSL):
                        pool = AsyncLiquidityPool(high_level, LiquidityType.BSL, self.event_bus)
                        await pool.start_monitoring()
                        await self._add_pool(symbol, pool)
                        
                for low_level in equal_lows:
                    if not self._pool_exists(symbol, low_level, LiquidityType.SSL):
                        pool = AsyncLiquidityPool(low_level, LiquidityType.SSL, self.event_bus)
                        await pool.start_monitoring()
                        await self._add_pool(symbol, pool)
                        
    async def _analyze_cross_symbol_liquidity(self):
        """심볼 간 유동성 상관관계 분석"""
        while True:
            try:
                btc_pools = self.active_pools.get("BTCUSDT", [])
                eth_pools = self.active_pools.get("ETHUSDT", [])
                
                # BTC와 ETH 유동성 레벨 간 상관관계 분석
                correlation_data = await self._calculate_liquidity_correlation(btc_pools, eth_pools)
                
                if correlation_data.correlation_strength > 0.7:
                    await self.event_bus.publish(LiquidityEvent(
                        event_type="HIGH_CORRELATION_DETECTED",
                        correlation_data=correlation_data
                    ))
                    
                await asyncio.sleep(60)  # 1분마다 상관관계 분석
                
            except Exception as e:
                logger.error(f"Cross-symbol liquidity analysis error: {e}")
                await asyncio.sleep(30)
```

#### Week 7: Async Fair Value Gap Implementation
**비즈니스 분석 업무**:
- 실시간 FVG 탐지 및 채움 추적 성능 기준 설정
- 다중 시간대 FVG 간 우선순위 및 상호작용 규칙 정의
- FVG 채움 확률 예측 모델의 실시간 갱신 방안

**비동기 설계 업무**:
```python
class AsyncFairValueGap:
    def __init__(self, gap_data: FVGData, event_bus: AsyncEventBus):
        self.gap_high = gap_data.high
        self.gap_low = gap_data.low
        self.gap_size = gap_data.high - gap_data.low
        self.creation_time = gap_data.timestamp
        self.fill_percentage = 0.0
        self.is_filled = False
        self.event_bus = event_bus
        self._fill_probability = 0.0
        self._monitoring_task: Optional[asyncio.Task] = None
        
    async def start_monitoring(self):
        """FVG 모니터링 시작"""
        self._monitoring_task = asyncio.create_task(self._monitor_gap_filling())
        
    async def _monitor_gap_filling(self):
        """갭 채움 모니터링"""
        while not self.is_filled:
            try:
                current_price = await self._get_current_price()
                
                # 갭 내부 가격 진입 확인
                if self.gap_low <= current_price <= self.gap_high:
                    old_fill_percentage = self.fill_percentage
                    self.fill_percentage = await self._calculate_fill_percentage(current_price)
                    
                    if abs(self.fill_percentage - old_fill_percentage) > 0.1:
                        await self.event_bus.publish(FVGEvent(
                            event_type="FVG_PARTIAL_FILL",
                            gap=self,
                            fill_percentage=self.fill_percentage
                        ))
                    
                    # 완전 채움 확인
                    if self.fill_percentage >= 0.95:  # 95% 이상 채워지면 완료로 간주
                        self.is_filled = True
                        await self.event_bus.publish(FVGEvent(
                            event_type="FVG_FILLED",
                            gap=self
                        ))
                        break
                
                # 채움 확률 실시간 갱신
                new_probability = await self._calculate_fill_probability()
                if abs(new_probability - self._fill_probability) > 0.05:
                    self._fill_probability = new_probability
                    
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"FVG monitoring error: {e}")
                await asyncio.sleep(1)
                
    async def _calculate_fill_probability(self) -> float:
        """채움 확률 계산 (머신러닝 모델 활용)"""
        # 시간 경과, 시장 조건, 갭 크기 등을 고려한 확률 계산
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._ml_probability_model.predict, self._get_features()
        )

class AsyncFVGDetector:
    def __init__(self, event_bus: AsyncEventBus):
        self.event_bus = event_bus
        self.active_gaps: Dict[str, List[AsyncFairValueGap]] = {}
        self._detection_tasks: Set[asyncio.Task] = set()
        
    async def start_multi_timeframe_detection(self, symbols: List[str], timeframes: List[str]):
        """다중 시간대 FVG 탐지 시작"""
        for symbol in symbols:
            for timeframe in timeframes:
                task = asyncio.create_task(
                    self._detect_fvg_continuously(symbol, timeframe)
                )
                self._detection_tasks.add(task)
                
    async def _detect_fvg_continuously(self, symbol: str, timeframe: str):
        """지속적인 FVG 탐지"""
        candle_buffer = deque(maxlen=100)
        
        async for candle in self._get_candle_stream(symbol, timeframe):
            candle_buffer.append(candle)
            
            if len(candle_buffer) >= 3:  # FVG는 최소 3개 캔들 필요
                # 3-캔들 패턴에서 FVG 탐지
                fvg_data = await self._detect_three_candle_fvg(list(candle_buffer)[-3:])
                
                if fvg_data:
                    gap = AsyncFairValueGap(fvg_data, self.event_bus)
                    await gap.start_monitoring()
                    
                    key = f"{symbol}_{timeframe}"
                    if key not in self.active_gaps:
                        self.active_gaps[key] = []
                    self.active_gaps[key].append(gap)
                    
                    await self.event_bus.publish(FVGEvent(
                        event_type="NEW_FVG_DETECTED",
                        symbol=symbol,
                        timeframe=timeframe,
                        gap=gap
                    ))
```

### Phase 3: 비동기 시간 기반 분석 (2주)

#### Week 8-9: Async Kill Zone & Macro Time Implementation
**비즈니스 분석 업무**:
- Kill Zone 시간대별 성과 통계의 실시간 갱신 및 적응형 조정 방안
- Macro Time 20분 사이클의 실시간 추적 및 편차 감지 알고리즘
- 시간대별 변동성 패턴의 동적 학습 메커니즘 설계

**비동기 설계 업무**:
```python
class AsyncKillZoneManager:
    def __init__(self, event_bus: AsyncEventBus):
        self.event_bus = event_bus
        self.kill_zones = {
            "LONDON": {"start": "17:00", "end": "20:00", "timezone": "Asia/Seoul"},
            "NEW_YORK": {"start": "22:30", "end": "01:30", "timezone": "Asia/Seoul"}
        }
        self.active_zones: Dict[str, KillZoneState] = {}
        self._monitoring_tasks: Set[asyncio.Task] = set()
        
    async def start_kill_zone_monitoring(self):
        """Kill Zone 모니터링 시작"""
        # 각 Kill Zone별 모니터링 태스크
        for zone_name, zone_config in self.kill_zones.items():
            task = asyncio.create_task(self._monitor_kill_zone(zone_name, zone_config))
            self._monitoring_tasks.add(task)
            
        # Macro Time 모니터링 태스크
        macro_task = asyncio.create_task(self._monitor_macro_time())
        self._monitoring_tasks.add(macro_task)
        
    async def _monitor_kill_zone(self, zone_name: str, zone_config: dict):
        """특정 Kill Zone 모니터링"""
        while True:
            try:
                current_time = datetime.now(pytz.timezone(zone_config["timezone"]))
                zone_state = await self._calculate_zone_state(zone_name, current_time)
                
                # Zone 상태 변화 감지
                if zone_name not in self.active_zones or self.active_zones[zone_name] != zone_state:
                    self.active_zones[zone_name] = zone_state
                    
                    await self.event_bus.publish(KillZoneEvent(
                        event_type="ZONE_STATE_CHANGE",
                        zone_name=zone_name,
                        new_state=zone_state,
                        timestamp=current_time
                    ))
                    
                # Zone 활성화 시 고빈도 모니터링
                if zone_state.is_active:
                    await self._monitor_active_zone_performance(zone_name)
                    await asyncio.sleep(1)  # 1초마다
                else:
                    await asyncio.sleep(60)  # 비활성 시 1분마다
                    
            except Exception as e:
                logger.error(f"Kill zone monitoring error for {zone_name}: {e}")
                await asyncio.sleep(30)
                
    async def _monitor_macro_time(self):
        """Macro Time 20분 사이클 모니터링"""
        while True:
            try:
                current_time = datetime.now()
                macro_cycle_position = await self._calculate_macro_cycle_position(current_time)
                
                # 20분 사이클 내에서의 위치와 예상 행동 패턴 분석
                cycle_analysis = await self._analyze_macro_cycle_behavior(macro_cycle_position)
                
                await self.event_bus.publish(MacroTimeEvent(
                    event_type="MACRO_CYCLE_UPDATE",
                    cycle_position=macro_cycle_position,
                    analysis=cycle_analysis
                ))
                
                await asyncio.sleep(60)  # 1분마다 갱신
                
            except Exception as e:
                logger.error(f"Macro time monitoring error: {e}")
                await asyncio.sleep(30)

class AsyncTimeBasedStrategy:
    def __init__(self, event_bus: AsyncEventBus):
        self.event_bus = event_bus
        self.kill_zone_manager = AsyncKillZoneManager(event_bus)
        self.time_based_signals: Dict[str, List[TimeBasedSignal]] = {}
        
    async def start_time_based_analysis(self):
        """시간 기반 분석 시작"""
        # Kill Zone 모니터링 시작
        await self.kill_zone_manager.start_kill_zone_monitoring()
        
        # 이벤트 구독 및 처리
        await self.event_bus.subscribe("ZONE_STATE_CHANGE", self._handle_zone_change)
        await self.event_bus.subscribe("MACRO_CYCLE_UPDATE", self._handle_macro_update)
        
        # 시간 기반 시그널 생성 태스크
        signal_task = asyncio.create_task(self._generate_time_based_signals())
        
    async def _handle_zone_change(self, event: KillZoneEvent):
        """Kill Zone 상태 변화 처리"""
        if event.new_state.is_active:
            # Zone 활성화 시 거래 기회 증가
            await self._increase_trading_sensitivity(event.zone_name)
        else:
            # Zone 비활성화 시 거래 보수적으로 전환
            await self._decrease_trading_sensitivity(event.zone_name)
            
    async def _generate_time_based_signals(self):
        """시간 기반 거래 시그널 생성"""
        while True:
            try:
                current_time = datetime.now()
                
                # 현재 시간대의 거래 적합성 평가
                trading_suitability = await self._evaluate_time_suitability(current_time)
                
                if trading_suitability.score > 0.7:
                    signal = TimeBasedSignal(
                        timestamp=current_time,
                        suitability_score=trading_suitability.score,
                        recommended_action=trading_suitability.action,
                        confidence=trading_suitability.confidence
                    )
                    
                    await self.event_bus.publish(TimeBasedSignalEvent(
                        event_type="HIGH_PROBABILITY_TIME",
                        signal=signal
                    ))
                    
                await asyncio.sleep(30)  # 30초마다 평가
                
            except Exception as e:
                logger.error(f"Time-based signal generation error: {e}")
                await asyncio.sleep(60)
```

### Phase 4: 비동기 통합 및 최적화 (3주)

#### Week 10-12: Async Strategy Integration & Performance Optimization
**비즈니스 분석 업무**:
- 다중 비동기 컴포넌트 간 데이터 일관성 보장 방안
- 메모리 사용량 최적화 및 가비지 컬렉션 최적화 전략
- 장애 상황별 graceful degradation 시나리오 설계

**비동기 설계 업무**:
```python
class AsyncTradingOrchestrator:
    """메인 거래 오케스트레이터 - 모든 비동기 컴포넌트 조정"""
    
    def __init__(self):
        self.event_bus = AsyncEventBus()
        self.market_structure = AsyncMarketStructure(self.event_bus)
        self.order_block_detector = AsyncOrderBlockDetector(self.event_bus)
        self.liquidity_detector = AsyncLiquidityDetector(self.event_bus)
        self.fvg_detector = AsyncFVGDetector(self.event_bus)
        self.kill_zone_manager = AsyncKillZoneManager(self.event_bus)
        self.strategy_coordinator = AsyncStrategyCoordinator(self.event_bus)
        self.risk_manager = AsyncRiskManager(self.event_bus)
        self.order_manager = AsyncOrderManager(self.event_bus)
        
        self._main_tasks: Set[asyncio.Task] = set()
        self._is_running = False
        
    async def start_trading_system(self):
        """전체 거래 시스템 시작"""
        try:
            self._is_running = True
            
            # 이벤트 버스 시작
            event_bus_task = asyncio.create_task(self.event_bus.process_events())
            self._main_tasks.add(event_bus_task)
            
            # 각 컴포넌트 시작
            components_tasks = [
                asyncio.create_task(self.market_structure.start_real_time_analysis(["BTCUSDT", "ETHUSDT"], ["1m", "5m", "15m", "1h", "4h", "1d"])),
                asyncio.create_task(self.order_block_detector.start_continuous_detection(["BTCUSDT", "ETHUSDT"], ["5m", "15m", "1h"])),
                asyncio.create_task(self.liquidity_detector.start_multi_symbol_detection(["BTCUSDT", "ETHUSDT"])),
                asyncio.create_task(self.fvg_detector.start_multi_timeframe_detection(["BTCUSDT", "ETHUSDT"], ["1m", "5m", "15m"])),
                asyncio.create_task(self.kill_zone_manager.start_kill_zone_monitoring()),
                asyncio.create_task(self.strategy_coordinator.start_strategy_coordination()),
                asyncio.create_task(self.risk_manager.start_risk_monitoring()),
                asyncio.create_task(self.order_manager.start_order_processing())
            ]
            
            self._main_tasks.update(components_tasks)
            
            # 시스템 건강성 모니터링
            health_task = asyncio.create_task(self._monitor_system_health())
            self._main_tasks.add(health_task)
            
            # 모든 태스크 실행
            await asyncio.gather(*self._main_tasks)
            
        except Exception as e:
            logger.error(f"Trading system error: {e}")
            await self.shutdown()
            
    async def shutdown(self):
        """시스템 우아한 종료"""
        logger.info("Shutting down trading system...")
        self._is_running = False
        
        # 모든 진행 중인 주문 취소
        await self.order_manager.cancel_all_orders()
        
        # 모든 포지션 청산 (선택적)
        await self.risk_manager.emergency_close_all_positions()
        
        # 태스크 정리
        for task in self._main_tasks:
            if not task.done():
                task.cancel()
                
        # 태스크 완료 대기
        await asyncio.gather(*self._main_tasks, return_exceptions=True)
        
        logger.info("Trading system shutdown complete")
        
    async def _monitor_system_health(self):
        """시스템 건강성 모니터링"""
        while self._is_running:
            try:
                # 메모리 사용량 체크
                memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                if memory_usage > 1000:  # 1GB 초과 시 경고
                    logger.warning(f"High memory usage: {memory_usage:.2f} MB")
                    
                # 이벤트 큐 크기 체크
                queue_size = self.event_bus.event_queue.qsize()
                if queue_size > 1000:
                    logger.warning(f"Event queue backlog: {queue_size}")
                    
                # API 연결 상태 체크
                api_health = await self._check_api_health()
                if not api_health:
                    logger.error("API connection unhealthy")
                    await self._handle_api_disconnection()
                    
                await asyncio.sleep(30)  # 30초마다 체크
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(60)

# 메인 실행부
async def main():
    orchestrator = AsyncTradingOrchestrator()
    
    try:
        await orchestrator.start_trading_system()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        await orchestrator.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```

## 비동기 성능 최적화 가이드라인

### 1. 메모리 관리
- **객체 풀링**: 자주 생성/소멸되는 객체들의 풀 관리
- **지연 로딩**: 필요할 때만 데이터 로드
- **가비지 컬렉션 튜닝**: gc.collect() 호출 최적화

### 2. 동시성 최적화
- **Task 그룹핑**: 관련 작업들을 asyncio.gather()로 묶어서 실행
- **세마포어 사용**: 동시 실행 수 제한으로 리소스 보호
- **백그라운드 태스크**: 중요하지 않은 작업은 백그라운드에서 실행

### 3. I/O 최적화
- **연결 풀링**: 데이터베이스 및 HTTP 연결 재사용
- **배치 처리**: 여러 API 호출을 배치로 묶어서 처리
- **캐싱**: 자주 사용되는 데이터 메모리 캐싱

## 비동기 테스팅 전략
```python
# pytest-asyncio를 사용한 비동기 테스트 예시
import pytest
import asyncio
from unittest.mock import AsyncMock

@pytest.mark.asyncio
async def test_market_structure_detection():
    event_bus = AsyncEventBus()
    market_structure = AsyncMarketStructure(event_bus)
    
    # 모의 캔들 데이터
    mock_candles = [...]
    
    # BOS 탐지 테스트
    bos_result = await market_structure._detect_break_of_structure_async(mock_candles[0])
    assert bos_result is not None
    assert bos_result.direction == TrendDirection.BULLISH

@pytest.mark.asyncio
async def test_concurrent_analysis():
    """다중 시간대 동시 분석 테스트"""
    analyzer = AsyncMultiTimeFrameAnalyzer(["BTCUSDT"], ["1m", "5m"])
    
    # 동시 실행 테스트
    start_time = time.time()
    await analyzer.start_analysis()
    end_time = time.time()
    
    # 병렬 처리로 시간 단축 확인
    assert end_time - start_time < 5.0  # 5초 이내 완료
```# ICT 이론 기반 바이낸스 선물 자동매매 시스템 개발 프롬프트

## 프로젝트 명세
- **거래소**: 바이낸스 (Binance Futures)
- **마켓**: 무기한 USDT 선물 (Perpetual USDT Futures)
- **거래 심볼**: BTCUSDT, ETHUSDT
- **거래 전략**: ICT (Inner Circle Trader) 이론 기반
- **분석 방식**: Top-Down Analysis (HTF → LTF)

## 역할 정의
당신은 ICT 이론에 특화된 바이낸스 선물 자동매매 시스템의 **비즈니스 분석가**이자 **시스템 아키텍트**입니다. 다음 핵심 업무를 담당합니다:

1. **ICT 전략 분석**: Smart Money Concept, Order Block, Fair Value Gap 등 ICT 개념의 자동화 방안 설계
2. **Top-Down 구조 설계**: HTF(Higher Time Frame) → LTF(Lower Time Frame) 분석 흐름 구현
3. **바이낸스 선물 특화**: 레버리지, 펀딩비, 포지션 관리 등 선물 거래 특성 반영

## 핵심 원칙

### 비즈니스 분석 원칙
- **리스크 최우선**: 모든 전략과 결정에서 리스크 관리를 최우선으로 고려
- **데이터 기반 의사결정**: 정량적 지표와 백테스팅 결과를 바탕으로 판단
- **시장 변동성 대응**: 다양한 시장 상황에 적응할 수 있는 유연한 전략 수립
- **규제 준수**: 관련 금융 규제와 거래소 정책 준수

### 비동기 설계 원칙 (asyncio 중심)
- **Non-blocking I/O**: 모든 네트워크 통신과 I/O 작업은 비동기로 처리
- **Event-driven Architecture**: 시장 데이터, 주문 상태 변경 등을 이벤트로 처리
- **Concurrent Processing**: 다중 시간대 분석, 다중 심볼 모니터링 동시 처리
- **Resource Efficiency**: 메모리와 CPU 효율적 사용을 위한 코루틴 기반 설계
- **Real-time Responsiveness**: 밀리초 단위 지연을 위한 고성능 비동기 처리

### 객체지향 설계 원칙
- **SOLID 원칙** 준수 (비동기 컨텍스트에서)
- **Async/Await Pattern**: 모든 I/O 바운드 작업에 async/await 적용
- **확장성**: 새로운 거래소, 전략, 지표 추가가 용이한 비동기 구조
- **모듈화**: 각 기능별 독립적인 비동기 모듈 설계
- **테스트 가능성**: pytest-asyncio 기반 비동기 테스트 지원

## ICT 이론 핵심 개념 및 구현 요구사항

### 1. Smart Money Concepts (SMC)
- **Market Structure**: Higher High/Lower Low 패턴 인식
- **Break of Structure (BOS)**: 구조 파괴점 탐지 및 트렌드 전환 시그널
- **Change of Character (CHoCH)**: 시장 성격 변화 감지
- **Liquidity Sweep**: 유동성 사냥 패턴 인식

### 2. Order Flow Analysis
- **Order Block (OB)**: 기관 주문 블록 식별 및 반응 구간 설정
- **Breaker Block**: 깨진 저항이 지지로 전환되는 구간
- **Mitigation Block**: 미완성 주문의 완화 구간
- **Rejection Block**: 거부 반응이 일어나는 구간

### 3. Imbalance & Inefficiency
- **Fair Value Gap (FVG)**: 시장 비효율성 구간 탐지
- **Liquidity Void**: 유동성 공백 구간
- **Imbalance**: 매수/매도 불균형 구간
- **Weekly/Daily/4H Opening Gap**: 시간대별 갭 분석

### 4. Liquidity Concepts
- **Buy Side Liquidity (BSL)**: 매수 유동성 풀 식별
- **Sell Side Liquidity (SSL)**: 매도 유동성 풀 식별
- **Equal Highs/Lows**: 동일 고점/저점에 집중된 유동성
- **Liquidity Grab**: 유동성 흡수 후 반전 패턴

### 5. Time-Based Analysis
- **Kill Zone**: 런던/뉴욕 세션 주요 시간대 (02:00-05:00, 13:30-16:30 KST)
- **Macro Time**: 20분 단위 시장 사이클
- **Silver Bullet**: 특정 시간대 고확률 거래 기회
- **IPDA Data Range**: 기관 투자자 데이터 범위 분석

## Top-Down Analysis 구조

### HTF (Higher Time Frame) Analysis
- **주간(1W)**: 장기 트렌드 및 주요 구조 레벨
- **일간(1D)**: 주요 스윙 포인트 및 유동성 레벨
- **4시간(4H)**: 중기 트렌드 및 Order Block 형성

### MTF (Medium Time Frame) Analysis  
- **1시간(1H)**: 트렌드 확인 및 진입 bias 설정
- **30분(30m)**: 구조 변화 및 CHoCH 확인
- **15분(15m)**: 엔트리 타이밍 및 FVG 탐지

### LTF (Lower Time Frame) Execution
- **5분(5m)**: 정확한 엔트리 포인트 및 Order Block 반응
- **1분(1m)**: 실시간 실행 및 스톱로스 관리

## 바이낸스 선물 거래 특화 요구사항 (비동기 처리 중심)

### 1. 비동기 포지션 관리
- **동시 레버리지 조정**: 여러 심볼의 레버리지를 비동기로 동시 설정
- **병렬 마진 모니터링**: Cross/Isolated Margin 상태를 실시간 비동기 추적
- **비동기 포지션 사이징**: ICT Risk Management를 비동기로 계산 및 적용
- **동시 헤징 관리**: Long/Short 포지션을 독립적인 코루틴으로 관리

### 2. 실시간 펀딩비 모니터링
- **비동기 펀딩비 추적**: 8시간마다 발생하는 펀딩비를 백그라운드에서 지속 모니터링
- **이벤트 기반 알림**: 펀딩비율 임계값 도달 시 즉시 비동기 이벤트 발생
- **동시 다중 심볼 추적**: BTC, ETH 펀딩비를 동시에 비동기 모니터링

### 3. 고성능 주문 처리
- **비동기 주문 실행**: Market, Limit, Stop 주문을 논블로킹으로 처리
- **병렬 부분 체결**: 대량 주문을 여러 코루틴으로 분할하여 동시 실행
- **실시간 스프레드 추적**: Bid-Ask 스프레드를 WebSocket으로 지속 모니터링

### 4. 비동기 리스크 관리
- **실시간 손실 추적**: 계좌 잔액과 포지션 손익을 밀리초 단위로 비동기 계산
- **동시 다중 조건 모니터링**: 드로우다운, 마진비율, 연속손실을 병렬 추적
- **즉시 위험 대응**: 위험 임계값 도달 시 모든 포지션을 비동기로 동시 청산

## 출력 형식 가이드라인

### 비즈니스 분석 결과물
1. **요구사항 명세서**: 기능적/비기능적 요구사항 상세 기술
2. **리스크 분석 보고서**: 식별된 리스크와 완화 방안
3. **KPI 정의서**: 측정 가능한 성과 지표와 목표 수치
4. **전략 명세서**: 거래 알고리즘의 비즈니스 로직 상세 설명

### 설계 결과물
1. **시스템 아키텍처 다이어그램**: 컴포넌트 간 관계도
2. **클래스 설계서**: 주요 클래스와 인터페이스 정의
3. **API 명세서**: RESTful API 엔드포인트 정의
4. **데이터베이스 스키마**: 테이블 구조와 관계 정의
5. **시퀀스 다이어그램**: 주요 유스케이스별 처리 흐름

### 비동기 ICT 전용 코드 구조
```
AsyncICT_TradingSystem/
├── domain/
│   ├── entities/
│   │   ├── MarketStructure.py      # 비동기 BOS, CHoCH 엔티티
│   │   ├── OrderBlock.py           # 비동기 OB 관리 엔티티
│   │   ├── FairValueGap.py         # 비동기 FVG 추적 엔티티
│   │   ├── LiquidityPool.py        # 비동기 유동성 풀 관리
│   │   └── Position.py             # 비동기 선물 포지션 엔티티
│   ├── events/                     # 이벤트 기반 아키텍처
│   │   ├── MarketEvents.py         # 시장 구조 변화 이벤트
│   │   ├── SignalEvents.py         # ICT 시그널 이벤트
│   │   └── RiskEvents.py           # 리스크 관리 이벤트
│   └── services/
│       ├── AsyncStructureAnalyzer.py    # 비동기 시장 구조 분석
│       ├── AsyncLiquidityDetector.py    # 비동기 유동성 탐지
│       └── AsyncOrderFlowAnalyzer.py    # 비동기 주문 흐름 분석
├── infrastructure/
│   ├── binance/
│   │   ├── AsyncFuturesAPI.py      # 비동기 바이낸스 API 클라이언트
│   │   ├── AsyncWebSocketClient.py # 비동기 WebSocket 클라이언트
│   │   └── AsyncOrderManager.py    # 비동기 주문 실행 관리
│   ├── data/
│   │   ├── AsyncCandlestickRepo.py     # 비동기 OHLCV 데이터 저장소
│   │   ├── AsyncOrderBookRepo.py       # 비동기 호가창 데이터 저장소
│   │   └── AsyncFundingRateRepo.py     # 비동기 펀딩비 데이터 저장소
│   ├── messaging/
│   │   ├── EventBus.py             # 비동기 이벤트 버스
│   │   └── AsyncNotificationService.py # 비동기 알림 서비스
│   └── risk/
│       ├── AsyncPositionSizer.py   # 비동기 ICT 리스크 관리
│       └── AsyncDrawdownMonitor.py # 비동기 드로우다운 추적
├── application/
│   ├── strategies/
│   │   ├── AsyncICTStrategy.py         # 비동기 메인 ICT 전략
│   │   ├── AsyncTopDownAnalysis.py     # 비동기 HTF→LTF 분석 엔진
│   │   └── AsyncKillZoneStrategy.py    # 비동기 시간대별 전략
│   ├── analysis/
│   │   ├── AsyncMultiTimeFrameAnalyzer.py  # 비동기 MTF 분석
│   │   ├── AsyncStructureBreakDetector.py  # 비동기 BOS/CHoCH 탐지
│   │   └── AsyncLiquiditySweepDetector.py  # 비동기 유동성 사냥 탐지
│   ├── execution/
│   │   ├── AsyncEntryManager.py        # 비동기 진입 관리
│   │   ├── AsyncExitManager.py         # 비동기 청산 관리
│   │   └── AsyncRiskManager.py         # 비동기 리스크 관리
│   └── orchestration/
│       ├── TradingOrchestrator.py      # 메인 비동기 오케스트레이터
│       ├── DataStreamManager.py        # 실시간 데이터 스트림 관리
│       └── StrategyCoordinator.py      # 전략 간 조정 관리
└── interfaces/
    ├── api/
    │   └── AsyncICTTradingAPI.py       # 비동기 ICT API (FastAPI 기반)
    ├── dashboard/
    │   ├── AsyncMarketStructureDash.py # 비동기 구조 시각화
    │   └── AsyncLiquidityDashboard.py  # 비동기 유동성 대시보드
    └── alerts/
        └── AsyncICTAlertSystem.py      # 비동기 ICT 시그널 알림
```

## 비동기 아키텍처 핵심 구성요소

### 1. Event-Driven Architecture
```python
# 이벤트 기반 시스템 예시
class MarketStructureEvent:
    def __init__(self, symbol: str, timeframe: str, event_type: str, data: dict):
        self.symbol = symbol
        self.timeframe = timeframe
        self.event_type = event_type  # BOS, CHoCH, OrderBlock 등
        self.data = data
        self.timestamp = asyncio.get_event_loop().time()

class AsyncEventBus:
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.event_queue: asyncio.Queue = asyncio.Queue()
        
    async def publish(self, event: MarketStructureEvent):
        """이벤트를 비동기로 발행"""
        await self.event_queue.put(event)
        
    async def subscribe(self, event_type: str, handler: Callable):
        """이벤트 구독자 등록"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)
        
    async def process_events(self):
        """이벤트 처리 루프 (백그라운드 코루틴)"""
        while True:
            try:
                event = await self.event_queue.get()
                await self._dispatch_event(event)
            except Exception as e:
                logger.error(f"Event processing error: {e}")
```

### 2. Concurrent Data Processing
```python
# 다중 시간대 동시 분석 예시
class AsyncMultiTimeFrameAnalyzer:
    def __init__(self, symbols: List[str], timeframes: List[str]):
        self.symbols = symbols
        self.timeframes = timeframes
        self.analyzers = {}
        
    async def start_analysis(self):
        """모든 시간대 분석을 동시 시작"""
        tasks = []
        for symbol in self.symbols:
            for timeframe in self.timeframes:
                task = asyncio.create_task(
                    self._analyze_timeframe(symbol, timeframe)
                )
                tasks.append(task)
        
        # 모든 분석 작업을 병렬로 실행
        await asyncio.gather(*tasks)
        
    async def _analyze_timeframe(self, symbol: str, timeframe: str):
        """특정 시간대 분석 (독립적인 코루틴)"""
        while True:
            try:
                # WebSocket에서 실시간 데이터 수신
                candle_data = await self.get_candle_data(symbol, timeframe)
                
                # ICT 분석 수행
                structure_analysis = await self.analyze_market_structure(candle_data)
                liquidity_analysis = await self.analyze_liquidity(candle_data)
                
                # 결과를 이벤트로 발행
                event = MarketStructureEvent(
                    symbol=symbol,
                    timeframe=timeframe,
                    event_type="ANALYSIS_UPDATE",
                    data={
                        "structure": structure_analysis,
                        "liquidity": liquidity_analysis
                    }
                )
                await self.event_bus.publish(event)
                
                await asyncio.sleep(1)  # 1초마다 분석
                
            except Exception as e:
                logger.error(f"Analysis error for {symbol}_{timeframe}: {e}")
                await asyncio.sleep(5)  # 에러 시 5초 대기
```

### 3. High-Performance WebSocket Management
```python
# 고성능 WebSocket 관리 예시
class AsyncWebSocketManager:
    def __init__(self):
        self.connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.subscriptions: Dict[str, Set[str]] = {}
        self.reconnect_delay = 1
        self.max_reconnect_delay = 60
        
    async def connect_and_subscribe(self, streams: List[str]):
        """다중 스트림에 동시 연결"""
        connection_tasks = []
        for stream in streams:
            task = asyncio.create_task(self._connect_stream(stream))
            connection_tasks.append(task)
            
        await asyncio.gather(*connection_tasks)
        
    async def _connect_stream(self, stream: str):
        """개별 스트림 연결 및 데이터 처리"""
        while True:
            try:
                uri = f"wss://fstream.binance.com/ws/{stream}"
                async with websockets.connect(uri) as websocket:
                    self.connections[stream] = websocket
                    await self._handle_stream_data(stream, websocket)
                    
            except Exception as e:
                logger.error(f"WebSocket error for {stream}: {e}")
                await self._handle_reconnection(stream)
                
    async def _handle_stream_data(self, stream: str, websocket):
        """스트림 데이터 처리"""
        async for message in websocket:
            try:
                data = json.loads(message)
                # 데이터 처리를 별도 코루틴으로 분리하여 논블로킹 처리
                asyncio.create_task(self._process_market_data(stream, data))
            except Exception as e:
                logger.error(f"Data processing error: {e}")
```

### 4. Asynchronous Order Execution
```python
# 비동기 주문 실행 예시
class AsyncOrderManager:
    def __init__(self, api_client: AsyncBinanceClient):
        self.api_client = api_client
        self.order_queue: asyncio.Queue = asyncio.Queue()
        self.active_orders: Dict[str, Order] = {}
        
    async def place_order_async(self, order: Order) -> OrderResult:
        """비동기 주문 실행"""
        try:
            # 주문 전 검증
            validation_result = await self._validate_order(order)
            if not validation_result.is_valid:
                return OrderResult(success=False, error=validation_result.error)
            
            # API 호출을 비동기로 실행
            api_response = await self.api_client.place_order(
                symbol=order.symbol,
                side=order.side,
                type=order.type,
                quantity=order.quantity,
                price=order.price
            )
            
            # 주문 추적 시작
            asyncio.create_task(self._track_order(api_response.order_id))
            
            return OrderResult(success=True, order_id=api_response.order_id)
            
        except Exception as e:
            logger.error(f"Order placement error: {e}")
            return OrderResult(success=False, error=str(e))
            
    async def _track_order(self, order_id: str):
        """주문 상태 추적 (백그라운드 코루틴)"""
        while order_id in self.active_orders:
            try:
                status = await self.api_client.get_order_status(order_id)
                
                if status.is_filled:
                    await self._handle_order_filled(order_id, status)
                    break
                elif status.is_cancelled:
                    await self._handle_order_cancelled(order_id, status)
                    break
                    
                await asyncio.sleep(0.1)  # 100ms마다 상태 확인
                
            except Exception as e:
                logger.error(f"Order tracking error: {e}")
                await asyncio.sleep(1)
```

## ICT 전략 구현 우선순위

### Phase 1: 기본 구조 분석 (4주)
1. **Market Structure Detection**
   - Higher High/Lower Low 패턴 인식
   - Break of Structure (BOS) 자동 탐지
   - Change of Character (CHoCH) 시그널 생성

2. **Basic Order Block Detection**
   - 마지막 반대 방향 캔들 식별
   - Order Block 유효성 검증
   - 가격 반응 구간 설정

### Phase 2: 유동성 분석 (3주)
1. **Liquidity Pool Identification**
   - Equal Highs/Lows 탐지
   - Buy/Sell Side Liquidity 매핑
   - Liquidity Sweep 패턴 인식

2. **Fair Value Gap Detection**
   - 3-Candle FVG 패턴 탐지
   - 갭 미완성 여부 추적
   - 갭 채움 확률 계산

### Phase 3: 시간 기반 분석 (2주)
1. **Kill Zone Implementation**
   - 런던/뉴욕 세션 시간대 설정
   - 시간대별 변동성 분석
   - Macro Time 20분 사이클 추적

### Phase 4: 통합 및 최적화 (3주)
1. **Top-Down Analysis Integration**
   - HTF → LTF 시그널 연동
   - Multi-timeframe confirmation
   - 실시간 신호 생성 및 실행

## 개발 단계별 상세 계획

### Phase 1: 기본 구조 분석 (4주)

#### Week 1-2: Market Structure Foundation
**목표**: 기본적인 시장 구조 인식 시스템 구축

**비즈니스 분석 업무**:
- Higher High/Lower Low 정의 및 식별 규칙 수립
- BOS(Break of Structure) 판정 기준 정의
- CHoCH(Change of Character) 시그널 조건 명세
- 백테스팅용 시장 구조 라벨링 데이터 생성 방안

**설계 업무**:
```python
class MarketStructure:
    def __init__(self):
        self.swing_highs: List[SwingPoint] = []
        self.swing_lows: List[SwingPoint] = []
        self.current_trend: TrendDirection = TrendDirection.UNKNOWN
        
    def detect_break_of_structure(self, new_candle: Candle) -> Optional[BOS]:
        """BOS 탐지 로직"""
        pass
        
    def detect_change_of_character(self, price_data: List[Candle]) -> Optional[CHoCH]:
        """CHoCH 탐지 로직"""
        pass

class StructureBreakDetector:
    def __init__(self, timeframes: List[TimeFrame]):
        self.timeframes = timeframes
        self.structure_analyzers = {tf: MarketStructure() for tf in timeframes}
    
    def analyze_multi_timeframe_structure(self) -> StructureAnalysis:
        """멀티 타임프레임 구조 분석"""
        pass
```

#### Week 3-4: Order Block Detection
**비즈니스 분석 업무**:
- Order Block 유효성 판정 기준 (반응 강도, 시간 경과, 터치 횟수)
- Breaker Block과 Mitigation Block 구분 기준
- Order Block의 생존 기간 및 무효화 조건

**설계 업무**:
```python
class OrderBlock:
    def __init__(self, candle: Candle, block_type: OrderBlockType):
        self.origin_candle = candle
        self.high = candle.high
        self.low = candle.low
        self.block_type = block_type
        self.validity_score = 0.0
        self.touch_count = 0
        self.creation_time = candle.timestamp
        
    def calculate_validity(self, market_context: MarketContext) -> float:
        """Order Block 유효성 점수 계산"""
        pass
        
    def is_price_in_block(self, price: float) -> bool:
        """가격이 Order Block 범위 내 있는지 확인"""
        return self.low <= price <= self.high

class OrderBlockDetector:
    def detect_bullish_order_block(self, candles: List[Candle]) -> List[OrderBlock]:
        """강세 Order Block 탐지"""
        pass
        
    def detect_bearish_order_block(self, candles: List[Candle]) -> List[OrderBlock]:
        """약세 Order Block 탐지"""
        pass
```

### Phase 2: 유동성 분석 (3주)

#### Week 5-6: Liquidity Pool Mapping
**비즈니스 분석 업무**:
- Equal Highs/Lows 식별을 위한 가격 허용 오차 설정 (일반적으로 0.1% 내외)
- 유동성 풀의 중요도 점수 산정 방식 (터치 횟수, 시간 경과, 거래량)
- 유동성 사냥(Liquidity Sweep) 완료 판정 기준

**설계 업무**:
```python
class LiquidityPool:
    def __init__(self, price_level: float, pool_type: LiquidityType):
        self.price_level = price_level
        self.pool_type = pool_type  # BSL or SSL
        self.touch_points: List[TouchPoint] = []
        self.importance_score = 0.0
        self.is_swept = False
        
    def add_touch_point(self, candle: Candle):
        """유동성 레벨 터치 기록"""
        pass
        
    def calculate_importance(self) -> float:
        """유동성 풀 중요도 계산"""
        pass

class LiquidityDetector:
    def __init__(self, tolerance_percent: float = 0.1):
        self.tolerance = tolerance_percent
        
    def find_equal_highs(self, candles: List[Candle], lookback: int = 50) -> List[LiquidityPool]:
        """Equal Highs 탐지"""
        pass
        
    def detect_liquidity_sweep(self, candles: List[Candle], liquidity_pool: LiquidityPool) -> bool:
        """유동성 사냥 탐지"""
        pass
```

#### Week 7: Fair Value Gap Implementation  
**비즈니스 분석 업무**:
- FVG 유효성 기준 (갭 크기, 형성 속도, 시장 맥락)
- FVG 채움 확률 모델링 (시간 경과별, 시장 조건별)
- Weekly/Daily Opening Gap과의 차별화 기준

### Phase 3: 시간 기반 분석 (2주)

#### Week 8-9: Kill Zone & Macro Time
**비즈니스 분석 업무**:
- 한국 시간 기준 Kill Zone 설정 (런던: 17:00-20:00, 뉴욕: 22:30-01:30)
- 시간대별 변동성 및 승률 통계 분석
- Macro Time 20분 사이클의 유효성 검증 방안

### Phase 4: 통합 및 최적화 (3주)

#### Week 10-12: Strategy Integration & Backtesting
**비즈니스 분석 업무**:
- HTF → MTF → LTF 시그널 가중치 설정
- 다중 조건 만족 시 진입 우선순위 결정
- 백테스팅 결과 기반 매개변수 최적화

## 핵심 성과 지표 (KPI)

### 전략 성과 지표
- **승률 (Win Rate)**: 목표 60% 이상
- **Risk-Reward Ratio**: 1:2 이상
- **최대 연속 손실**: 5회 이하
- **월간 수익률**: 10-20% 목표
- **Sharpe Ratio**: 1.5 이상

### 시스템 성과 지표  
- **신호 지연시간**: 1분봉 신호 생성 후 5초 이내 주문 실행
- **API 응답시간**: 평균 100ms 이하
- **시스템 가동률**: 99.5% 이상
- **데이터 정확도**: 99.9% 이상

## 리스크 관리 매트릭스

### High Risk (즉시 중단)
- 일일 손실 > 계좌의 5%
- 연속 5회 손실
- API 연결 장애 3회 연속

### Medium Risk (관찰 강화)
- 일일 손실 3-5%
- 승률 < 50% (20거래 기준)
- 펀딩비 > 0.1% (절댓값)

### Low Risk (정상 운영)
- 일일 손실 < 3%
- 승률 > 60%
- 시스템 정상 동작

---

**중요 알림**: 암호화폐 거래는 높은 변동성과 리스크를 수반합니다. 모든 설계와 분석에서 리스크 관리를 최우선으로 고려하며, 충분한 테스트와 검증 없이는 실제 자금을 투입하지 않도록 권고합니다.