# Docker 설치 가이드 (macOS)

## 방법 1: Docker Desktop 설치 (권장)

Docker Desktop은 macOS용 공식 Docker 애플리케이션으로 GUI와 CLI를 모두 제공합니다.

### 1-1. 직접 다운로드 설치

1. **Docker Desktop 다운로드**
   - 웹사이트: https://www.docker.com/products/docker-desktop
   - Apple Silicon (M1/M2/M3): "Mac with Apple chip" 선택
   - Intel Mac: "Mac with Intel chip" 선택

2. **설치 파일 실행**
   ```bash
   # 다운로드한 .dmg 파일을 더블클릭
   # Docker.app을 Applications 폴더로 드래그
   ```

3. **Docker Desktop 실행**
   - Applications 폴더에서 Docker 아이콘을 더블클릭
   - 첫 실행 시 권한 요청에 동의
   - 메뉴바에 Docker 고래 아이콘이 나타날 때까지 대기

4. **설치 확인**
   ```bash
   docker --version
   docker-compose --version
   ```

### 1-2. Homebrew로 설치 (선호하는 경우)

```bash
# Homebrew 설치 (미설치 시)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Docker Desktop 설치
brew install --cask docker

# Docker Desktop 실행
open /Applications/Docker.app

# 설치 확인
docker --version
docker-compose --version
```

## 방법 2: Colima 설치 (경량 대안)

Docker Desktop 대신 경량 CLI 기반 솔루션을 원하는 경우:

```bash
# Colima와 Docker CLI 설치
brew install colima docker docker-compose

# Colima 시작 (기본 설정)
colima start

# 또는 리소스 지정하여 시작
colima start --cpu 4 --memory 8 --disk 60

# 설치 확인
docker --version
docker-compose --version
docker ps
```

## 설치 후 초기 설정

### 1. Docker 실행 확인

```bash
# Docker 데몬이 실행 중인지 확인
docker info

# Hello World 테스트
docker run hello-world
```

예상 출력:
```
Hello from Docker!
This message shows that your installation appears to be working correctly.
```

### 2. 리소스 설정 (Docker Desktop)

Docker Desktop > Preferences (⚙️) > Resources:
- **CPUs**: 4 이상 권장
- **Memory**: 4GB 이상 권장 (8GB 이상 이상적)
- **Disk**: 60GB 이상 권장

### 3. 자동 시작 설정 (선택)

Docker Desktop > Preferences > General:
- ✅ "Start Docker Desktop when you log in" 체크

## 프로젝트에서 Docker 사용

### 1. 환경 변수 설정

```bash
# 프로젝트 루트로 이동
cd /Users/osangwon/github/test

# 환경 변수 템플릿 복사
cp .env.template .env

# 실제 API 키로 수정
vim .env  # 또는 nano .env
```

`.env` 파일 예시:
```bash
TRADING_BINANCE_API_KEY=your_actual_binance_api_key_here
TRADING_BINANCE_API_SECRET=your_actual_binance_secret_here
TRADING_DISCORD_BOT_TOKEN=your_actual_discord_token_here
TRADING_DISCORD_CHANNEL_ID=your_actual_channel_id_here
TRADING_ENV=development
```

### 2. Docker 이미지 빌드

```bash
# 개발 환경 빌드
./scripts/docker_build.sh build dev

# 또는 직접 docker-compose 사용
docker-compose -f docker-compose.yml -f docker-compose.dev.yml build
```

### 3. 컨테이너 실행

```bash
# 개발 환경 시작
./scripts/docker_build.sh start dev

# 로그 확인
./scripts/docker_build.sh logs

# 또는 직접 docker-compose 사용
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
docker-compose logs -f trading-bot
```

### 4. 헬스체크 확인

```bash
# API 헬스체크
curl http://localhost:8000/health

# Docker 헬스 상태 확인
docker-compose ps
docker inspect --format='{{.State.Health.Status}}' ict-trading-bot
```

### 5. 컨테이너 관리

```bash
# 컨테이너 중지
./scripts/docker_build.sh stop

# 컨테이너 재시작
./scripts/docker_build.sh restart dev

# 컨테이너 내부 접속
docker-compose exec trading-bot /bin/bash

# 리소스 사용량 모니터링
docker stats ict-trading-bot

# 컨테이너 및 이미지 정리
./scripts/docker_build.sh clean
```

## 문제 해결

### Docker Desktop이 시작되지 않는 경우

```bash
# Docker Desktop 완전히 종료
pkill -SIGHUP -f Docker

# 다시 시작
open /Applications/Docker.app

# 또는 재설치
brew uninstall --cask docker
brew install --cask docker
```

### "Cannot connect to Docker daemon" 오류

```bash
# Docker 데몬 실행 확인
docker info

# Docker Desktop을 수동으로 시작
open /Applications/Docker.app

# Colima 사용 시
colima start
```

### 권한 오류

```bash
# 현재 사용자를 docker 그룹에 추가 (Linux)
sudo usermod -aG docker $USER

# macOS에서는 일반적으로 불필요하지만, 필요한 경우
sudo chown -R $USER:$USER ~/.docker
```

### 디스크 공간 부족

```bash
# 사용하지 않는 이미지, 컨테이너 정리
docker system prune -a

# 볼륨까지 포함하여 정리
docker system prune -a --volumes

# 디스크 사용량 확인
docker system df
```

### 빌드 캐시 문제

```bash
# 캐시 없이 재빌드
docker-compose build --no-cache

# 빌드 캐시 정리
docker builder prune
```

## 유용한 Docker 명령어

```bash
# 실행 중인 컨테이너 목록
docker ps

# 모든 컨테이너 목록 (중지된 것 포함)
docker ps -a

# 이미지 목록
docker images

# 특정 컨테이너 로그
docker logs -f ict-trading-bot

# 컨테이너 내부에서 명령 실행
docker exec -it ict-trading-bot python --version

# 컨테이너 재시작
docker restart ict-trading-bot

# 컨테이너 중지
docker stop ict-trading-bot

# 컨테이너 삭제
docker rm ict-trading-bot

# 이미지 삭제
docker rmi ict-trading-bot
```

## 다음 단계

Docker 설치 및 설정이 완료되면:

1. ✅ Docker 설치 완료
2. ✅ 환경 변수 설정 (.env)
3. ✅ 이미지 빌드
4. ✅ 컨테이너 실행
5. ✅ 헬스체크 확인
6. 📖 [Docker 사용 가이드](./docker.md) 참조
7. 🚀 Task 12.5 나머지 테스트 수행

## 추가 리소스

- [Docker 공식 문서](https://docs.docker.com/)
- [Docker Desktop for Mac](https://docs.docker.com/desktop/install/mac-install/)
- [Docker Compose 문서](https://docs.docker.com/compose/)
- [Dockerfile 모범 사례](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)
