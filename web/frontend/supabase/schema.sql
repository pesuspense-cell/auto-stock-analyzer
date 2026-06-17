-- ════════════════════════════════════════════════════════════════════════════
--  ASA — Supabase 스키마 (Next.js + Vercel 통합 아키텍처)
--  Supabase SQL Editor 에 그대로 실행. Supabase Auth(auth.users) 기반으로
--  기존 database.py 의 커스텀 users/session_token 인증을 대체한다.
-- ════════════════════════════════════════════════════════════════════════════

-- ── 0. 확장 ──────────────────────────────────────────────────────────────────
create extension if not exists "pgcrypto";   -- gen_random_uuid()

-- ════════════════════════════════════════════════════════════════════════════
--  A. 종목 마스터 (stock_metadata.json 8,922종목 시드)
-- ════════════════════════════════════════════════════════════════════════════
create table if not exists public.stocks (
  ticker      text primary key,                 -- "005930.KS", "AAPL"
  name        text not null,                    -- "삼성전자"
  name_kr     text,                             -- 미국주 한글 별칭 (예: "엔비디아")
  market      text,                             -- KOSPI / KOSDAQ / NASDAQ / NYSE / AMEX
  is_etf      boolean not null default false,
  updated_at  timestamptz not null default now()
);
-- 한글/영문/티커 통합 검색용 (ilike 가속). pg_trgm 사용.
create extension if not exists "pg_trgm";
create index if not exists stocks_name_trgm    on public.stocks using gin (name gin_trgm_ops);
create index if not exists stocks_name_kr_trgm on public.stocks using gin (name_kr gin_trgm_ops);

-- ════════════════════════════════════════════════════════════════════════════
--  B. 캐싱 레이어 (외부 주가 API 오버헤드 차단)
--  TTL 판정은 fetched_at 기준으로 애플리케이션(lib/cache.ts)에서 수행한다.
-- ════════════════════════════════════════════════════════════════════════════

-- 단일 종목 시세 (장중 3초 / 장외 60초 TTL)
create table if not exists public.quote_cache (
  ticker       text primary key references public.stocks(ticker) on delete cascade,
  price        double precision not null,
  prev_close   double precision,
  change_pct   double precision,
  volume       double precision,
  currency     text,                            -- KRW / USD
  is_realtime  boolean not null default false,
  fetched_at   timestamptz not null default now()
);

-- 환율 (5분 TTL)
create table if not exists public.exchange_rate_cache (
  pair        text primary key,                 -- "USD/KRW"
  rate        double precision not null,
  change_pct  double precision,
  fetched_at  timestamptz not null default now()
);

-- 시장 무버스/섹터ETF 등 구조화 페이로드 (5~10분 TTL)
create table if not exists public.market_cache (
  scope       text primary key,                 -- "overview" / "sector_etf"
  payload     jsonb not null,
  fetched_at  timestamptz not null default now()
);

-- ════════════════════════════════════════════════════════════════════════════
--  C. 사용자 데이터
--  NOTE: 사용자 소유 테이블(user_settings / portfolios / trade_history /
--        recommendation_history)은 기존 인스턴스의 레거시 테이블과 이름이 충돌하므로
--        migrations/001_supabase_auth.sql 에서 "레거시 보존 → 재생성 → 백필" 로 처리한다.
--        (신규 인스턴스라면 001 만 실행해도 동일하게 생성됨)
-- ════════════════════════════════════════════════════════════════════════════

-- 장시간 작업(백테스트/ASA) 결과 — Python 워커가 기록, Next.js가 폴링 조회
create table if not exists public.jobs (
  id          uuid primary key default gen_random_uuid(),
  user_id     uuid references auth.users(id) on delete cascade,
  kind        text not null,                     -- "backtest" / "asa"
  status      text not null default 'running',   -- running / done / error
  params      jsonb,
  result      jsonb,
  error       text,
  created_at  timestamptz not null default now(),
  updated_at  timestamptz not null default now()
);
create index if not exists jobs_user on public.jobs(user_id);

-- ════════════════════════════════════════════════════════════════════════════
--  D. RLS (Row Level Security)
--  · 캐시/종목 테이블: 공개 읽기, 쓰기는 service_role(API Route)만 → RLS로 막음
--  · 사용자 테이블: 본인 행만 CRUD
-- ════════════════════════════════════════════════════════════════════════════

-- 공개 읽기 테이블
alter table public.stocks               enable row level security;
alter table public.quote_cache          enable row level security;
alter table public.exchange_rate_cache  enable row level security;
alter table public.market_cache         enable row level security;

create policy "public read stocks"  on public.stocks              for select using (true);
create policy "public read quotes"  on public.quote_cache         for select using (true);
create policy "public read rates"   on public.exchange_rate_cache for select using (true);
create policy "public read market"  on public.market_cache        for select using (true);
-- (INSERT/UPDATE 정책 없음 → service_role 키만 쓰기 가능. RLS는 service_role을 우회함)

-- 사용자 소유 테이블 (jobs) — 나머지 사용자 테이블 RLS는 migration 001 에서 설정
alter table public.jobs enable row level security;
create policy "own jobs" on public.jobs
  for all using (auth.uid() = user_id) with check (auth.uid() = user_id);
