# 🚀 Real-time Event-Driven Stock Analysis Architecture

**최종 업데이트**: 2026-05-04  
**시스템 아키텍처**: Producer-Consumer Pattern with Redis Pub/Sub

---

## 📋 목차

1. [아키텍처 개요](#아키텍처-개요)
2. [컴포넌트 설명](#컴포넌트-설명)
3. [데이터 흐름](#데이터-흐름)
4. [설치 및 실행](#설치-및-실행)
5. [모니터링](#모니터링)
6. [트러블슈팅](#트러블슈팅)
7. [성능 고려사항](#성능-고려사항)
8. [향후 개선](#향후-개선)

---

## 🏗️ 아키텍처 개요

### Before: 폴링 기반 모놀리식 구조

```
┌─────────────────────────────────┐
│   Monolithic Stock App (app.py) │
├─────────────────────────────────┤
│                                 │
│  ┌──────────────────────────┐   │
│  │ 1. 데이터 수집           │   │
│  │    (Polling/REST API)    │   │
│  └──────────────────────────┘   │
│                │                 │
│                ↓                 │
│  ┌──────────────────────────┐   │
│  │ 2. 분석 로직             │   │
│  │    (기술적/감정 분석)    │   │
│  └──────────────────────────┘   │
│                │                 │
│                ↓                 │
│  ┌──────────────────────────┐   │
│  │ 3. 알림 발송             │   │
│  │    (Telegram/Email)      │   │
│  └──────────────────────────┘   │
│                                 │
│  ⚠️ 문제점:                     │
│  - 모든 기능이 하나의 프로세스   │
│  - 분석 중에는 데이터 수집 중단   │
│  - 확장성 제한                  │
│  - 부하 분산 어려움              │
│                                 │
└─────────────────────────────────┘
```

### After: 이벤트 기반 분산 구조

```
┌────────────────────────────────────────────────────────────────┐
│   Real-time Event-Driven Architecture                         │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────────┐                ┌──────────────────┐     │
│  │  Data Ingestor   │                │ Analysis Worker  │     │
│  │   (Producer)     │                │   (Consumer)     │     │
│  │                  │                │                  │     │
│  │ ┌──────────────┐ │  market_updates │ ┌──────────────┐ │     │
│  │ │ WebSocket/   │ │  channel        │ │ Subscriber   │ │     │
│  │ │ REST API     ├─┼────────┬────────┤ │ & Analyzer   │ │     │
│  │ │ Listener     │ │        │        │ │ (stock_ai)  │ │     │
│  │ │              │ │    Redis Pub/Sub │ │              │ │     │
│  │ │ Market Data  │ │  Message Broker  │ │ Telegram     │ │     │
│  │ │ Parser       │ │        │        │ │ Notification │ │     │
│  │ └──────────────┘ │        │        │ └──────────────┘ │     │
│  └──────────────────┘        │        └──────────────────┘     │
│                              ↓                                 │
│                       ┌─────────────┐                         │
│                       │   Redis     │                         │
│                       │             │                         │
│                       │  Channel:   │                         │
│                       │  market_    │                         │
│                       │  updates    │                         │
│                       │             │                         │
│                       │  JSON msgs  │                         │
│                       │  {ticker,   │                         │
│                       │   price,    │                         │
│                       │   change,%} │                         │
│                       └─────────────┘                         │
│                                                                │
│  ✅ 장점:                                                      │
│  - Producer/Consumer 독립 프로세스                            │
│  - 동시 데이터 수집 & 분석                                    │
│  - 쉬운 확장성 (Consumer 추가 가능)                          │
│  - 부하 분산 가능                                            │
│  - 메시지 큐 덕에 안정적                                      │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## 🔧 컴포넌트 설명

### 1️⃣ **Data Ingestor (Producer)**

**역할**: 시장 데이터 수집 및 Redis 발행

**위치**: `data_ingestor.py`

**특징**:
- 비동기 WebSocket/REST API 클라이언트
- 수신 데이터 파싱 및 정규화
- 자동 재접속 로직 (최대 10회 재시도)
- 에러 격리 (개별 종목 오류가 전체 시스템 영향 없음)

**동작 흐름**:
```python
1. market_data_provider (WebSocket/API)
   ↓
2. parse_market_data()
   ↓
3. MarketUpdate 객체 생성
   ↓
4. redis.publish("market_updates", json)
```

**설정 예**:
```python
ingestor = DataIngestor(
    redis_host="localhost",
    redis_port=6379,
    watch_tickers=["005930.KS", "AAPL", "MSFT"]
)
await ingestor.run(use_simulation=True, interval=5.0)
```

**메시지 포맷** (Redis Pub/Sub):
```json
{
  "ticker": "005930.KS",
  "timestamp": "2026-05-04T14:30:00",
  "price": 70000.50,
  "volume": 1000000,
  "change": 2.50,
  "high": 71000.00,
  "low": 69500.00,
  "open_": 68500.00,
  "market": "KRX"
}
```

---

### 2️⃣ **Analysis Worker (Consumer)**

**역할**: Redis 구독, 실시간 분석, 알림 발송

**위치**: `analysis_worker.py`

**특징**:
- Redis 채널 구독 (블로킹 없음)
- stock_ai.py 핵심 함수 호출
- 기술적 지표, 감정 분석 통합
- Telegram 봇 비동기 알림

**동작 흐름**:
```python
1. Redis 채널 구독 ("market_updates")
   ↓
2. 메시지 수신
   ↓
3. analyze_update(market_data)
   ├─ get_stock_data() → 기술적 지표
   ├─ analyze_technicals() → RSI, MACD 등
   └─ get_advanced_sentiment() → 뉴스 감정 분석
   ↓
4. 신호 결정 (BUY/SELL/HOLD)
   ↓
5. send_telegram_notification()
```

**신호 생성 로직** (예시):
```python
if rsi > 70:                 # 과매수
    signal = "SELL"
    confidence = 0.7
    reason = "RSI 과매수"
elif rsi < 30:               # 과매도
    signal = "BUY"
    confidence = 0.7
    reason = "RSI 과매도"
elif sentiment_score > 2:     # 긍정 뉴스
    signal = "BUY"
    confidence = min(base + 0.2, 1.0)
    reason += " + 긍정 뉴스"
```

**설정 예**:
```python
worker = AnalysisWorker(
    redis_host="localhost",
    redis_port=6379,
    telegram_bot_token="123456:ABC-DEF...",
    telegram_chat_id="1234567890"
)
await worker.run()
```

---

### 3️⃣ **Redis (메시지 브로커)**

**역할**: Producer → Consumer 간 메시지 중간 전달

**특징**:
- Pub/Sub 패턴 (1:N 메시지 전달)
- 메모리 기반 저장소 (빠른 성능)
- 자동 메시지 만료
- TTL 관리 가능

**채널 구조**:
```
Redis
├── Channel: market_updates
│   ├── Subscriber 1 (AnalysisWorker #1)
│   ├── Subscriber 2 (AnalysisWorker #2)
│   └── Subscriber 3 (Other Consumer)
│
├── Channel: analysis_results (선택적)
│   └── Subscriber (Logging/Storage)
│
└── Channel: alerts
    └── Subscriber (Alert Manager)
```

---

### 4️⃣ **Telegram 알림**

**역할**: 사용자에게 실시간 거래 신호 전달

**설정 방법**:
```bash
# 1. Telegram BotFather와 채팅
#    /newbot → Bot Name, Bot Username 입력
#    → Bot Token 획득 (예: 123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11)

# 2. 자신의 Chat ID 획득
#    @userinfobot 검색 → /start
#    → Chat ID 확인 (예: 1234567890)

# 3. 환경 변수 설정
export TELEGRAM_BOT_TOKEN="123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"
export TELEGRAM_CHAT_ID="1234567890"

# 4. docker-compose 실행
docker-compose up -d
```

**알림 포맷**:
```
🟢 **BUY 신호**
종목: 005930.KS
현재가: 70,000.50
변동: +2.50%
이유: RSI 과매도 + 긍정 뉴스
시간: 14:30:00
```

---

## 📊 데이터 흐름

### 실시간 데이터 처리 흐름

```
시간축 →

거래소 WebSocket
  │
  ├─ Tick 1 (14:30:00)
  │  └─ 삼성전자 70,000 (+1.5%)
  │     ↓
  │  [Ingestor] → Parse → Normalize
  │     ↓
  │  Redis.publish("market_updates", {ticker, price, ...})
  │     ↓
  │  [Worker 1] → Subscribe & Analyze
  │     ├─ RSI: 35 (과매도)
  │     ├─ Sentiment: +2.3 (긍정)
  │     └─ Signal: BUY (신뢰도 0.75)
  │        ↓
  │     Telegram: "🟢 BUY 신호: 삼성전자 70,000 (+1.5%)"
  │
  │
  ├─ Tick 2 (14:30:05)
  │  └─ AAPL 180.25 (+0.5%)
  │     ↓
  │  [Ingestor] → Parse → Normalize
  │     ↓
  │  Redis.publish("market_updates", {ticker, price, ...})
  │     ↓
  │  [Worker 2] → Subscribe & Analyze
  │     ├─ MACD: 신호선 교차 (상승)
  │     ├─ Sentiment: 0.0 (중립)
  │     └─ Signal: HOLD
  │        ↓
  │     Telegram: (알림 미발송)
  │
  │
  └─ Tick 3 (14:30:10)
     └─ 현대차 95,500 (-3.2%)
        ↓
     [Ingestor] → Parse → Normalize
        ↓
     Redis.publish("market_updates", {ticker, price, ...})
        ↓
     [Worker 3] → Subscribe & Analyze
        ├─ RSI: 72 (과매수)
        ├─ Sentiment: -1.8 (부정)
        └─ Signal: SELL (신뢰도 0.85)
           ↓
        Telegram: "🔴 SELL 신호: 현대차 95,500 (-3.2%)"


동시성:
- Ingestor는 계속 데이터 수집
- Worker는 각자 독립적으로 분석 실행
- 메시지 손실 없음 (Redis 큐 덕분)
```

---

## 🚀 설치 및 실행

### 사전 요구사항

- Docker & Docker Compose
- Python 3.9+
- Redis (또는 Docker)
- Telegram Bot Token & Chat ID

### 빠른 시작 (Docker Compose)

#### 1️⃣ **환경 설정**

```bash
cd /path/to/Auto\ Stock\ Analyzer

# .env 파일 생성
cp .env.example .env

# .env 파일 수정
nano .env
# 또는
vim .env

# 필수 항목:
# TELEGRAM_BOT_TOKEN=your_bot_token
# TELEGRAM_CHAT_ID=your_chat_id
```

#### 2️⃣ **서비스 실행**

```bash
# 모든 서비스 시작
docker-compose up -d

# 로그 확인
docker-compose logs -f

# 개별 서비스 로그
docker-compose logs -f data_ingestor
docker-compose logs -f analysis_worker
docker-compose logs -f redis
```

#### 3️⃣ **모니터링**

```bash
# Redis 데이터 확인 (웹 UI)
open http://localhost:8081

# Redis CLI 접속
docker-compose exec redis redis-cli

# 채널 모니터링
> PUBSUB CHANNELS
> PUBSUB NUMSUB market_updates
> SUBSCRIBE market_updates
```

#### 4️⃣ **종료**

```bash
# 서비스 중지
docker-compose stop

# 서비스 제거
docker-compose down

# 볼륨 포함 제거
docker-compose down -v
```

---

### 로컬 실행 (개발 환경)

```bash
# 1. Redis 설치 (Mac)
brew install redis

# 2. Redis 시작
redis-server

# 3. Python 환경 설정
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 4. 터미널 1: Data Ingestor 실행
python data_ingestor.py

# 5. 터미널 2: Analysis Worker 실행
TELEGRAM_BOT_TOKEN="..." TELEGRAM_CHAT_ID="..." python analysis_worker.py

# 6. 터미널 3: Redis 모니터링
redis-cli
> SUBSCRIBE market_updates
```

---

## 📊 모니터링

### Redis Commander 웹 UI

**URL**: http://localhost:8081

**확인 사항**:
- `market_updates` 채널의 메시지 흐름
- Redis 메모리 사용량
- 연결된 클라이언트 수

### 로그 분석

```bash
# Data Ingestor 로그 (메시지 발행)
docker-compose logs data_ingestor | grep "📤 발행"

# Analysis Worker 로그 (분석 결과)
docker-compose logs analysis_worker | grep "🔍 분석 결과"

# 에러 추적
docker-compose logs analysis_worker | grep ERROR
```

### Prometheus 메트릭 (선택)

```bash
# Prometheus 엔드포인트 추가 (향후 확장)
/metrics 엔드포인트 구현:
- Ingestor: 발행 메시지 수, 평균 지연
- Worker: 분석 시간, 신호 발생 빈도
- Redis: 채널 구독자 수, 메시지 레이트
```

---

## 🔧 트러블슈팅

### 문제 1: Redis 연결 실패

```
❌ Error: ConnectionRefusedError: [Errno 111] Connection refused
```

**해결책**:
```bash
# Redis 실행 상태 확인
docker ps | grep redis

# Redis 시작
docker-compose up -d redis

# Redis 헬스 체크
docker-compose exec redis redis-cli ping
# PONG 응답 → 정상
```

### 문제 2: 메시지 도착하지 않음

```
❌ 분석 결과가 없음
```

**진단**:
```bash
# 1. Data Ingestor가 메시지 발행 중인지 확인
docker-compose logs data_ingestor | grep "📤 발행"

# 2. Redis 채널 모니터링
docker-compose exec redis redis-cli
> PUBSUB CHANNELS
# 결과: market_updates

# 3. 메시지 직접 확인
> SUBSCRIBE market_updates
# 메시지가 보이면 정상

# 4. Worker가 구독 중인지 확인
docker-compose logs analysis_worker | grep "👂 메시지"
```

**해결책**:
```bash
# Data Ingestor 재시작
docker-compose restart data_ingestor

# 또는
docker-compose stop data_ingestor
docker-compose up -d data_ingestor
```

### 문제 3: Telegram 알림이 오지 않음

```
❌ Telegram 메시지 미수신
```

**진단**:
```bash
# 1. 설정 확인
docker-compose config | grep TELEGRAM

# 2. 로그 확인
docker-compose logs analysis_worker | grep Telegram

# 3. 봇 토큰 검증
# Telegram에서 직접 테스트:
curl "https://api.telegram.org/bot{BOT_TOKEN}/getMe"
# 정상 응답: {"ok":true, "result": {...}}
```

**해결책**:
```bash
# 1. 환경 변수 다시 설정
export TELEGRAM_BOT_TOKEN="..."
export TELEGRAM_CHAT_ID="..."

# 2. docker-compose 재시작
docker-compose down
docker-compose up -d

# 3. 테스트 메시지 발송
docker-compose exec analysis_worker python -c "
import asyncio
from analysis_worker import TelegramNotifier
notifier = TelegramNotifier('TOKEN', 'CHAT_ID')
asyncio.run(notifier.send_message('테스트 메시지'))
"
```

### 문제 4: 높은 CPU 사용률

```
❌ CPU 사용률 > 80%
```

**원인**: 과도한 분석 부하

**해결책**:
```bash
# 1. 폴링 간격 증가
# docker-compose.yml 수정:
environment:
  - POLLING_INTERVAL=10  # 5초 → 10초

# 2. 분석 워커 복제
# 새로운 worker 컨테이너 추가:
analysis_worker_2:
  build: ...
  depends_on: ...
  # 동일한 설정

docker-compose up -d analysis_worker_2

# 3. 모니터링할 종목 줄이기
WATCH_TICKERS=005930.KS,000660.KS  # 전체 → 2개만
```

---

## ⚡ 성능 고려사항

### 처리량 (Throughput)

```
목표: 1,000+ ticks/초 처리

현재 구현:
- Data Ingestor: 최대 100 ticks/초 (시뮬레이션)
- Analysis Worker: ~5초/분석 (stock_ai.py 포함)
  (단일 worker 기준)

최적화 방안:
1. Redis 배치 발행
   - 여러 메시지를 한 번에 발행
   - 네트워크 왕복 감소

2. Consumer 스케일링
   - 여러 Worker 프로세스 배포
   - 같은 종목 → 같은 worker (순서 보장)
   - 다른 종목 → 다른 worker (병렬 처리)

3. 분석 최적화
   - 캐시: 기술적 지표 재계산 방지
   - 배치: 여러 종목 한 번에 분석
   - 비동기: 블로킹 작업 제거
```

### 지연 시간 (Latency)

```
목표: <100ms (데이터 수신 → 알림 발송)

현재:
- Ingestor → Redis: ~1ms
- Redis 큐: ~0ms
- Worker 분석: ~5,000ms (stock_ai.py)
- Telegram API: ~500ms

개선 방안:
1. 간단한 분석만 실시간 수행
   - RSI, MACD 등 캐시된 지표
   - 복잡한 분석은 배경에서 실행

2. 비동기 알림
   - 분석 완료 전에 수신 확인 반환
   - Telegram 발송은 별도 큐에서 처리

3. Redis 최적화
   - 메모리 설정: maxmemory 512MB
   - 제거 정책: allkeys-lru
```

### 메모리 사용량

```
Redis 메모리:
- 메시지 크기: ~200 bytes
- TTL: 60초 (기본)
- 최대 메시지: 100/초 × 60초 = 6,000개
- 예상: ~1.2 MB

설정 (docker-compose.yml):
command: redis-server --maxmemory 512mb --maxmemory-policy allkeys-lru

정책 설명:
- maxmemory: 최대 메모리 512MB
- maxmemory-policy: LRU로 오래된 키 자동 삭제
```

---

## 🔮 향후 개선

### 1️⃣ **다중 브로커 지원**

```python
# RabbitMQ, Kafka 등으로 확장
class KafkaIngestor(DataIngestor):
    async def publish_update(self, update):
        await kafka_producer.send("market-updates", update)

class KafkaWorker(AnalysisWorker):
    async def listen(self):
        async for msg in kafka_consumer:
            await self.process_message(msg)
```

### 2️⃣ **데이터베이스 저장**

```python
# 분석 결과 영구 저장
from sqlalchemy import create_engine
engine = create_engine("postgresql://...")

# 로깅
await log_analysis_result(ticker, signal, confidence, timestamp)
```

### 3️⃣ **대시보드**

```
실시간 분석 결과 시각화:
- 활성 신호 맵
- 종목별 RSI/MACD
- Telegram 알림 히스토리
- Worker 처리량 메트릭
```

### 4️⃣ **ML 기반 신호**

```python
# scikit-learn, TensorFlow 모델 통합
from sklearn.ensemble import RandomForestClassifier

predictor = RandomForestClassifier()
prediction = predictor.predict(features)
confidence = predictor.predict_proba(features).max()
```

### 5️⃣ **분산 처리**

```
Kubernetes 배포:
- Redis: StatefulSet
- Data Ingestor: Deployment (1개)
- Analysis Worker: Deployment (자동 스케일)
- 메트릭: Prometheus + Grafana
```

---

## 📚 참고 자료

- [Redis Pub/Sub](https://redis.io/topics/pubsub)
- [Asyncio Documentation](https://docs.python.org/3/library/asyncio.html)
- [Docker Compose](https://docs.docker.com/compose/)
- [Telegram Bot API](https://core.telegram.org/bots/api)
- [Event-Driven Architecture](https://www.nginx.com/blog/event-driven-architecture/)

---

**End of Documentation**
