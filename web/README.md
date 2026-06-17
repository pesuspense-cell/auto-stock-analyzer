# ASA — Next.js 마이그레이션

기존 Streamlit 앱(`stock_ai.py` 외)을 건드리지 않고 `web/` 에 격리한 신규 아키텍처.

```
web/
├── backend/   [경로 A] FastAPI — stock_ai/src 전 로직 재사용 (Render/Railway 배포)
└── frontend/  Next.js (App Router) + TypeScript + Tailwind CSS
    ├── app/api/v1/   [경로 B] Next.js API Routes — I/O 로직을 TS로 직접 처리
    ├── lib/supabase/ Supabase 클라이언트 (browser/server/service-role)
    ├── lib/cache.ts  Supabase 캐싱 레이어 (시세 3초 / 환율 5분 TTL)
    └── supabase/     schema.sql + 종목 시드 스크립트
```

## 두 가지 백엔드 경로 (중요)

이 저장소에는 **상호 보완적인 두 백엔드 경로**가 있습니다. `stock_ai.py`(5,837줄)는
Python/pandas 수치연산이라 Vercel 런타임에서 실행 불가하고, 백테스트·ASA는 수 분
소요라 서버리스 타임아웃에 걸립니다. 따라서:

| 영역 | 권장 경로 |
|------|-----------|
| 시세·환율·검색·사용자/포트폴리오·캐싱 | **경로 B** (Vercel API Routes + Supabase) — 완전 통합 |
| 지표·펀더멘털 스코어 | 경로 B로 점진 TS 포팅 (또는 경로 A 유지) |
| 백테스트·ASA(장시간) | **경로 A** Python 워커 유지 (서버리스 불가) — 결과를 `jobs` 테이블에 기록 |

### 경로 B 셋업 (Next.js + Supabase)

```bash
cd web/frontend
cp .env.local.example .env.local      # Supabase URL/anon/service_role 키 입력
npm install
# 1) Supabase SQL Editor 에서 supabase/schema.sql 실행 (stocks/캐시/jobs)
# 2) supabase/migrations/001_supabase_auth.sql 실행
#    → 레거시 테이블 보존(legacy_*) + Auth 기반 사용자 테이블 + RLS + 백필 트리거
# 3) 종목 마스터 시드 (8,922종목)
node supabase/seed-stocks.mjs
npm run dev                            # http://localhost:3000
# 확인: GET /api/v1/quote/AAPL · /api/v1/market/rates · /api/v1/stocks/search?q=삼성
```

### 인증 — Supabase Auth 전환 (기존 database.py 대체)

- **흐름**: `components/AuthGate.tsx` 가 `@supabase/ssr` 브라우저 클라이언트로
  `signUp`/`signInWithPassword`/`signOut`. 세션은 쿠키에 저장되고 `middleware.ts` 가
  매 요청마다 갱신 → Route Handler 에서 `auth.uid()` 로 RLS 가 적용된다.
- **기존 사용자**: werkzeug 해시는 이전 불가 → **동일 이메일로 재가입**하면
  `migration 001` 의 `handle_new_user` 트리거가 `legacy_*` 데이터(포트폴리오·매도이력·
  추천이력)를 새 uuid 계정으로 **자동 이관**한다.
- **포트폴리오 API**(쿠키 세션, RLS): `GET/POST /api/v1/portfolio`,
  `DELETE /api/v1/portfolio/[id]`, `POST /api/v1/portfolio/[id]/sell`,
  `GET /api/v1/portfolio/trades` — 모두 `app/api/v1/portfolio/**` 에 TS로 이식됨.
- **하이브리드 라우팅**: `next.config.mjs` 의 rewrites 는 *afterFiles* 라 로컬 Route 가
  우선. 로컬에 없는 경로(`/analysis`·`/news`·`/fundamental`·`/asa`·`/backtest`)만
  FastAPI(경로 A)로 폴백된다 → 점진 이전 가능.

> ✅ (작업 #3에서 해소) ASA·백테스트는 이제 `jobs` 큐 + Python 워커가 신규 Supabase
> 포트폴리오를 참조해 처리한다. 아래 "ASA·백테스트 비동기 Job 큐" 섹션 참고.

### 마이그레이션 트리거 검증

```bash
# Supabase SQL Editor 에서 (migration 001 적용 후):
#   supabase/verify_migration.sql 실행
# → 레거시 사용자 재가입 시 데이터 자동 이관을 BEGIN/ROLLBACK 안에서 실제 트리거로 검증.
#   "✅ PASS — 트리거 백필 정상 동작" NOTICE 가 뜨면 정상. 모든 변경은 롤백됨(비파괴).
```

### ASA·백테스트 비동기 Job 큐 (작업 #3)

장시간(수 분) 작업을 **Supabase `jobs` 테이블 큐 + Python 워커**로 분리 — 서버리스
타임아웃 회피 + 신규 Supabase 포트폴리오 참조(레거시 DB 의존 제거).

```
Next.js API (enqueue)            Python 워커 (자립)               프론트
POST /api/v1/asa/run        ─┐                                   useJob() 폴링
POST /api/v1/backtest/run    ├─▶ jobs(status='pending') ──▶ 워커가 SKIP LOCKED 로
GET  /api/v1/jobs/[id]      ─┘                               원자적 클레임(→processing)
                                                            · ASA: public.portfolios 조회
                                                              → live_screener 실행
                                                            · 백테스트: Screener+Engine
                                                            → result 기록, status='completed'
                              ◀── 폴링으로 결과 수신 ──────────────┘
```

**워커 실행** (FastAPI 환경, 레거시 로직 재사용):
```bash
cd web/backend
pip install -r requirements.txt   # + 레포 루트 requirements (yfinance 등)
SUPABASE_DB_URL=postgresql://... python -m worker.jobs_worker
```
- Supabase SQL Editor 에서 `supabase/migrations/002_jobs_queue.sql` 먼저 실행.
- 워커는 **직접 DB 연결**(service 권한)이라 RLS 우회 → 모든 사용자 jobs/portfolios 접근.
- 중복 처리 방지: `FOR UPDATE SKIP LOCKED` 원자적 클레임 (워커 수평 확장 안전).
- 결과의 numpy/datetime 타입은 `_json_default` 로 jsonb 직렬화.
- 프론트 [hooks/useJob.ts](frontend/hooks/useJob.ts) + [JobStatusBanner](frontend/components/JobStatusBanner.tsx)
  가 pending(큐)→processing(처리)→completed(결과)/error 를 추적·표시.
- ASA/백테스트 탭은 jobs.user_id(RLS) 때문에 **로그인 필요** → `AuthGate` 로 감쌈.

> ✅ 이로써 README 상단의 "ASA가 레거시 DB 포트폴리오 참조" 갭 해소됨.

### 기술지표 TS 포팅 (작업 #2)

`stock_ai.py` 의 `_add_indicators()` + `generate_signals()`(12모듈 채점)를 **순수 TS**로 이식:
- [lib/indicators/ta.ts](frontend/lib/indicators/ta.ts) — RSI·MACD·볼린저·스토캐스틱·ADX·CCI·
  Williams%R·ROC·OBV·MFI·Z-Score·VWAP 등 pandas 공식 1:1 (ewm/rolling/std ddof=1 의미 보존).
- [lib/indicators/signals.ts](frontend/lib/indicators/signals.ts) — 12모듈 채점(-10~+10) + 라벨.
- [app/api/v1/indicators/[ticker]/route.ts](frontend/app/api/v1/indicators/[ticker]/route.ts) —
  OHLC(Yahoo) → 계산 → `market_cache`(장중 60초/장외 30분 TTL) → 타입 JSON.
- **Polling 트래픽 방어**: [lib/market-hours.ts](frontend/lib/market-hours.ts) `pollingIntervalMs()` →
  장중 15초 / 장외 평일 5분 / **주말 0(정지)**. `IndicatorsPanel` 의 SWR `refreshInterval`(함수형)에 적용.
- **수치 검증**: 260봉 고정 데이터로 Python 원본과 대조 → 21개 지표 + 신호점수(7.3/강력매수) **완전 일치**.
- ⚠️ 모듈 11(다이버전스)은 별도 `detect_divergence()` 의존이라 본 포팅에서 기여 0(최대 ±1.5 편차 가능).

### 하이브리드 라우팅 주의 (검증 중 발견·수정)

Next.js 배열 형태 `rewrites` 는 **afterFiles** 단계 — *정적 라우트 다음, **동적 라우트 이전***에
평가된다. 따라서 catch-all `/api/:path*` 프록시는 동적 로컬 라우트(`quote/[ticker]`,
`portfolio/[id]`)를 **가린다(shadow)**. → `next.config.mjs` 는 catch-all 대신 **FastAPI 폴백
경로만 명시적으로 나열**한다(`FASTAPI_FALLBACK_PATHS`). 로컬로 이전 완료 시 목록에서 제거.

## 1. 백엔드 실행 (포트 8000)

```bash
cd web/backend
cp .env.example .env            # SUPABASE_DB_URL 등 입력
pip install -r requirements.txt # + 레포 루트 requirements.txt (yfinance 등) 필요
uvicorn app.main:app --reload --port 8000
# Swagger 문서: http://localhost:8000/docs
```

> `app/bootstrap.py` 가 레포 루트를 `sys.path` 에 등록하여 `stock_ai.py` / `src/*` 를
> 코드 복제 없이 직접 import 한다. `stock_ai.py` 는 streamlit 의존이 없어 안전하다.

## 2. 프론트엔드 실행 (포트 3000)

```bash
cd web/frontend
cp .env.local.example .env.local   # BACKEND_URL 확인
npm install
npm run dev                         # http://localhost:3000
```

`next.config.mjs` 의 rewrites 가 `/api/*` 를 백엔드로 프록시하므로 CORS 없이 개발 가능.

## 3. 구현 현황

| 영역 | 상태 |
|------|------|
| 인증 (register/login/logout/me) | ✅ `src/database.py` 재사용 |
| 시장 현황 (overview/rates) | ✅ 5분 부분 갱신(SWR) |
| 종목 검색 (한글/영문/티커) | ✅ resolve_ticker |
| 종합 분석 (signals/hybrid/realtime) | ✅ 파이프라인 이식 |
| 차트 OHLC / 실시간가 | ✅ 엔드포인트 |
| 포트폴리오 CRUD/매도/매매이력 | ✅ Bearer 인증 |
| 뉴스 (기사/감성/섹터/AI요약) | ✅ KR·US·ETF 분기 이식 |
| 펀더멘털 (지표/ETF/수급/AI 리포트) | ✅ generate_financial_report + 규칙기반 결론 |
| ASA 추천 | ✅ live_screener 백그라운드 잡 + 폴링 |
| 백테스트 | ✅ Screener+Engine 잡 + 폴링, metrics 재계산 |
| 차트 캔들 시각화 (lightweight-charts) | ⏳ 후속 (엔드포인트 `/analysis/{t}/ohlc` 준비됨) |

## 4. 배포 (목표)

- **Frontend → Vercel**: `BACKEND_URL` 환경변수에 백엔드 URL 지정.
- **Backend → Render / Railway**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`,
  환경변수 `SUPABASE_DB_URL`, `CORS_ORIGINS=https://<vercel-domain>`.
