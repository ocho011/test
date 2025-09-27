# ICT Trading Bot

ICT (Inner Circle Trader) 이론을 기반으로 한 자동매매 시스템입니다.

## 주요 특징

- **전략**: ICT Order Block + Fair Value Gap (FVG) 조합
- **거래소**: Binance Futures (USDT 무기한 계약)
- **리스크 관리**: 거래당 2% 위험 관리
- **목표 수익률**: 월 10-15%
- **아키텍처**: 이벤트 기반 비동기 시스템

## 기술 스택

- **언어**: Python 3.9+
- **주요 라이브러리**:
  - python-binance: 공식 바이낸스 API
  - asyncio/aiohttp: 비동기 처리
  - pydantic: 설정 관리 및 데이터 검증
  - pandas/numpy: 데이터 분석
  - discord.py: 알림 시스템

## 프로젝트 구조

```
src/trading_bot/
├── core/          # 핵심 시스템 (EventBus, BaseComponent)
├── strategies/    # 거래 전략 (ICTStrategy)
├── analysis/      # 기술적 분석 (OrderBlock, FVG 감지)
├── risk/          # 리스크 관리 (포지션 사이징, 드로우다운)
├── execution/     # 주문 실행 (OrderExecutor, PositionManager)
├── notifications/ # 알림 시스템 (Discord)
├── config/        # 설정 관리
└── utils/         # 유틸리티 함수

tests/
├── unit/          # 단위 테스트
└── integration/   # 통합 테스트

config/            # 설정 파일
logs/              # 로그 파일
data/              # 히스토리컬 데이터
```

## 설치 및 실행

1. **의존성 설치**:
```bash
pip install -e .
```

2. **개발 환경 설정**:
```bash
pip install -e ".[dev]"
```

3. **환경 변수 설정**:
```bash
cp .env.example .env
# .env 파일에 API 키 설정
```

4. **테스트 실행**:
```bash
pytest
```

## 라이선스

MIT License

## 주의사항

이 시스템은 교육 및 연구 목적으로 제작되었습니다. 실제 거래에 사용하기 전에 충분한 백테스팅과 페이퍼 트레이딩을 권장합니다.