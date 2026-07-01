-- ════════════════════════════════════════════════════════════════════════════
--  007 — 백테스트 시세 DB 캐시 (price_bars)
--
--  과거 일봉 OHLCV 를 캐시해 백테스트가 매 실행마다 유니버스 전체(최대 400종목)를
--  재다운로드하던 것을 read-through(캐시 우선 + 미스분만 수집·적재)로 바꾼다.
--  → 반복 실행 속도·안정성(yfinance 429 차단 무관)·메모리 개선. (price_cache.py 사용)
--
--  · 접근은 워커/백엔드의 직접 psycopg2 연결(SUPABASE_DB_URL) 전용 — 유저 소유 데이터가
--    아니므로 RLS 는 '차단만'(정책 없음) 걸어 PostREST(anon/authenticated) 노출을 막는다.
--    직접 연결(postgres 롤)은 RLS 를 우회하므로 워커 동작에는 영향 없음.
--
--  Supabase SQL Editor 에서 실행. idempotent(여러 번 실행해도 안전).
-- ════════════════════════════════════════════════════════════════════════════

create table if not exists public.price_bars (
  ticker     text not null,
  date       date not null,
  open       double precision,
  high       double precision,
  low        double precision,
  close      double precision,
  volume     double precision,
  source     text,
  updated_at timestamptz not null default now(),
  primary key (ticker, date)
);

-- 종목 단독 커버리지 조회(min/max/count) 및 범위 스캔 가속.
-- PK(ticker,date)로 대부분 커버되지만, 날짜 우선 스캔용 보조 인덱스도 둔다.
create index if not exists price_bars_date_idx on public.price_bars (date);

-- 캐시 테이블은 PostgREST(anon/authenticated)로 노출할 필요가 없다 → RLS enable + 정책 없음.
-- 워커/백엔드의 직접 연결(postgres 롤)은 RLS 를 우회하므로 read/upsert 정상 동작.
alter table public.price_bars enable row level security;
