## 🚀 Real-time Event-Driven 주식 분석 시스템 - 빠른 시작 가이드

### 📖 개요

이 프로젝트는 **실시간 데이터 스트리밍 + 이벤트 기반 워커 모델**을 사용하여 
주식 시세를 실시간으로 분석하고 거래 신호를 즉시 전달합니다.

```
Data Ingestor (수집) → Redis Pub/Sub → Analysis Worker (분석) → Telegram (알림)
```

### ⚡ 빠른 시작 (3단계)

#### 1️⃣ **준비**

```bash
# 1. .env 파일 생성
cp .env.example .env

# 2. Telegram 봇 설정
#    BotFather → /newbot → Bot Token 획득
#    @userinfobot → /start → Chat ID 획득

# 3. .env 파일 수정
nano .env

# 필수 설정:
# TELEGRAM_BOT_TOKEN=your_bot_token
# TELEGRAM_CHAT_ID=your_chat_id
```

#### 2️⃣ **실행**

```bash
# Docker 필요 (설치: https://www.docker.com)

# 모든 서비스 시작
docker-compose up -d

# 로그 확인
docker-compose logs -f
```

#### 3️⃣ **모니터링**

```bash
# 웹 UI에서 Redis 확인
open http://localhost:8081

# 또는 CLI로 확인
docker-compose exec redis redis-cli
> SUBSCRIBE market_updates
```

### 📁 파일 구조

```
Auto Stock Analyzer/
├── data_ingestor.py              ← Producer (데이터 수집)
├── analysis_worker.py            ← Consumer (분석 & 알림)
├── docker-compose.yml            ← Docker 오케스트레이션
├── Dockerfile.ingestor           ← Ingestor 이미지
├── Dockerfile.worker             ← Worker 이미지
├── .env.example                  ← 환경 설정 템플릿
├── REALTIME_ARCHITECTURE.md      ← 상세 문서
└── requirements.txt              ← Python 의존성
```

### 🔧 로컬 개발 (Docker 없이)

```bash
# 1. Python 환경 설정
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Redis 설치 및 실행
# Mac: brew install redis && redis-server
# Linux: sudo apt install redis && redis-server
# Windows: WSL2 추천 또는 WSL에서 redis-server

# 3. 터미널 1: Data Ingestor
python data_ingestor.py

# 4. 터미널 2: Analysis Worker
export TELEGRAM_BOT_TOKEN="your_token"
export TELEGRAM_CHAT_ID="your_chat_id"
python analysis_worker.py
```

### 📊 시스템 구조

```
┌─────────────────────────────────────────────────────────┐
│  Real-time Event-Driven Stock Analysis System          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Data Ingestor (Producer)                              │
│  ├─ WebSocket/REST API 수신                            │
│  ├─ 시세 데이터 파싱                                    │
│  └─ Redis Pub/Sub 발행                                 │
│       │                                                 │
│       ↓                                                 │
│  Redis (Message Broker)                                │
│  └─ Channel: market_updates                            │
│       │                                                 │
│       ↓                                                 │
│  Analysis Worker (Consumer) [x N]                      │
│  ├─ Redis 구독                                         │
│  ├─ stock_ai.py 분석 실행                              │
│  │  ├─ 기술적 지표 (RSI, MACD, etc)                   │
│  │  ├─ 감정 분석 (뉴스 분석)                          │
│  │  └─ 신호 생성 (BUY/SELL/HOLD)                     │
│  └─ Telegram 알림 발송                                 │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 🔄 데이터 흐름 예시

```
14:30:00 → 삼성전자 70,000원 (+2.5%) 수신
         → Ingestor 파싱
         → Redis: market_updates 채널에 발행
         → Worker 구독 & 분석 시작
           ├─ RSI: 35 (과매도 ✓)
           ├─ Sentiment: +2.3 (긍정 뉴스 ✓)
           └─ Signal: BUY (신뢰도 75%)
         → Telegram: "🟢 BUY 신호: 삼성전자 70,000 (+2.5%)"
```

### ⚙️ 설정 옵션

**docker-compose.yml 에서 수정**:

```yaml
# 1. 모니터링할 종목 변경
environment:
  - WATCH_TICKERS=005930.KS,000660.KS,035720.KS

# 2. 데이터 수집 간격 변경
command: python data_ingestor.py --interval=10

# 3. Worker 복제 (병렬 처리 증가)
analysis_worker_2:
  service: analysis_worker
  ...

# 4. Redis 메모리 제한
command: redis-server --maxmemory 512mb --maxmemory-policy allkeys-lru
```

### 📱 Telegram 설정

**1. 봇 생성**
```
1. Telegram에서 @BotFather 검색
2. /newbot 입력
3. 봇 이름 입력 (예: My Stock Analyzer)
4. 봇 사용자명 입력 (예: my_stock_bot)
5. Bot Token 획득 (예: 123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11)
```

**2. Chat ID 획득**
```
1. Telegram에서 @userinfobot 검색
2. /start 입력
3. Chat ID 확인 (예: 1234567890)
```

**3. 환경 변수 설정**
```bash
export TELEGRAM_BOT_TOKEN="123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"
export TELEGRAM_CHAT_ID="1234567890"
```

### 🛠️ 트러블슈팅

| 문제 | 해결책 |
|------|--------|
| Redis 연결 오류 | `docker-compose logs redis` 확인, `docker-compose restart redis` |
| 메시지 미수신 | `docker-compose exec redis redis-cli SUBSCRIBE market_updates` |
| Telegram 알림 미전송 | `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` 확인 |
| 높은 CPU 사용 | 폴링 간격 증가 또는 모니터링 종목 축소 |

### 📊 성능 목표

- **처리량**: 100+ ticks/초 (시뮬레이션)
- **지연**: <100ms (수신 → 알림)
- **신뢰성**: 메시지 손실 없음 (Redis Pub/Sub)
- **확장성**: Worker 추가로 병렬 처리 증가

### 📚 상세 문서

더 자세한 내용은 [REALTIME_ARCHITECTURE.md](REALTIME_ARCHITECTURE.md) 참고

### 🎯 다음 단계

1. **커스텀 분석 로직 추가**
   ```python
   # analysis_worker.py에서 analyze_update() 수정
   custom_signal = my_custom_analysis(update_data)
   ```

2. **데이터베이스 저장**
   ```python
   # 분석 결과를 PostgreSQL/MongoDB에 저장
   await save_analysis_result(ticker, signal, timestamp)
   ```

3. **대시보드 추가**
   ```bash
   # Streamlit 대시보드로 실시간 결과 시각화
   streamlit run dashboard.py
   ```

4. **알림 채널 확장**
   ```python
   # Slack, Discord, Email 등 추가
   await slack_notifier.send_signal(signal)
   ```

### 📞 지원

문제 발생 시:
1. `docker-compose logs` 확인
2. [REALTIME_ARCHITECTURE.md](REALTIME_ARCHITECTURE.md)의 트러블슈팅 참고
3. 이슈 생성

---

**Happy Trading! 🚀**
