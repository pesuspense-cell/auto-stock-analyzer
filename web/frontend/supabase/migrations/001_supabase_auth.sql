-- ════════════════════════════════════════════════════════════════════════════
--  001 — Supabase Auth 전환 (기존 인스턴스 재사용, 비파괴)
--
--  기존 database.py 가 만든 레거시 테이블(정수 user_id + werkzeug 해시)을 보존하고,
--  Supabase Auth(auth.users, uuid) 기반 테이블을 새로 만든 뒤,
--  사용자가 동일 이메일로 가입하면 레거시 데이터를 자동 백필한다.
--
--  ⚠️ 레거시 비밀번호 해시(werkzeug)는 Supabase Auth로 이전 불가 → 기존 사용자는
--     "동일 이메일로 재가입(또는 비밀번호 재설정)" 해야 하며, 그 시점에 데이터가 이관된다.
--
--  이 스크립트는 idempotent 하다(여러 번 실행해도 안전). Supabase SQL Editor 에서 실행.
-- ════════════════════════════════════════════════════════════════════════════

create extension if not exists "pgcrypto";

-- ── 1) 레거시 보존: 기존 테이블을 legacy_* 로 이름 변경 (데이터 보존, 되돌리기 가능) ──
--    user_id 가 정수형인 "구" 테이블만 대상으로 한다(이미 마이그레이션된 경우 skip).
do $$
begin
  if to_regclass('public.users') is not null
     and exists (select 1 from information_schema.columns
                 where table_schema='public' and table_name='users' and column_name='password_hash') then
    alter table if exists public.portfolios            rename to legacy_portfolios;
    alter table if exists public.trade_history         rename to legacy_trade_history;
    alter table if exists public.recommendation_history rename to legacy_recommendation_history;
    alter table if exists public.users                 rename to legacy_users;
    raise notice '레거시 테이블을 legacy_* 로 보존했습니다.';
  end if;
end $$;

-- ── 2) Auth 기반 테이블 생성 ──────────────────────────────────────────────────
create table if not exists public.user_settings (
  user_id        uuid primary key references auth.users(id) on delete cascade,
  gemini_api_key text,
  groq_api_key   text,
  dart_api_key   text,
  watchlist      jsonb not null default '[]'::jsonb,
  cash_balance   double precision not null default 0,   -- 예수금(현금 잔고). 기존 인스턴스는 003 마이그레이션으로 추가
  updated_at     timestamptz not null default now()
);

create table if not exists public.portfolios (
  id          uuid primary key default gen_random_uuid(),
  user_id     uuid not null references auth.users(id) on delete cascade,
  ticker      text not null,
  avg_price   double precision not null,
  quantity    double precision not null default 1,
  added_at    timestamptz not null default now(),
  unique (user_id, ticker)
);
create index if not exists portfolios_user on public.portfolios(user_id);

create table if not exists public.trade_history (
  id          uuid primary key default gen_random_uuid(),
  user_id     uuid not null references auth.users(id) on delete cascade,
  ticker      text not null,
  buy_price   double precision not null,
  sell_price  double precision not null,
  quantity    double precision not null,
  net_profit  double precision not null,
  return_rate double precision not null,
  traded_at   timestamptz not null default now()
);
create index if not exists trade_history_user on public.trade_history(user_id);

create table if not exists public.recommendation_history (
  id              uuid primary key default gen_random_uuid(),
  user_id         uuid not null references auth.users(id) on delete cascade,
  investment_amt  bigint not null,
  risk_profile    text not null default '중립형',
  recommendations jsonb not null,
  created_at      timestamptz not null default now()
);

-- ── 3) RLS (본인 행만 CRUD) ───────────────────────────────────────────────────
alter table public.user_settings          enable row level security;
alter table public.portfolios             enable row level security;
alter table public.trade_history          enable row level security;
alter table public.recommendation_history enable row level security;

drop policy if exists "own settings"   on public.user_settings;
drop policy if exists "own portfolios" on public.portfolios;
drop policy if exists "own trades"     on public.trade_history;
drop policy if exists "own recs"       on public.recommendation_history;

create policy "own settings" on public.user_settings
  for all using (auth.uid() = user_id) with check (auth.uid() = user_id);
create policy "own portfolios" on public.portfolios
  for all using (auth.uid() = user_id) with check (auth.uid() = user_id);
create policy "own trades" on public.trade_history
  for all using (auth.uid() = user_id) with check (auth.uid() = user_id);
create policy "own recs" on public.recommendation_history
  for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

-- ── 4) 레거시 데이터 백필 함수 (이메일 매칭) ──────────────────────────────────
--     legacy_* 테이블이 있을 때만 동작. 신규 인스턴스에서는 no-op.
create or replace function public.migrate_legacy_user(p_uid uuid, p_email text)
returns void
language plpgsql
security definer
set search_path = public
as $$
begin
  if to_regclass('public.legacy_users') is null then
    return;
  end if;

  -- 포트폴리오 이관
  insert into public.portfolios (user_id, ticker, avg_price, quantity, added_at)
  select p_uid, lp.ticker, lp.avg_price, lp.quantity, coalesce(lp.added_at, now())
  from public.legacy_portfolios lp
  join public.legacy_users lu on lu.id = lp.user_id
  where lower(lu.email) = lower(p_email)
  on conflict (user_id, ticker) do nothing;

  -- 매도 이력 이관
  insert into public.trade_history
    (user_id, ticker, buy_price, sell_price, quantity, net_profit, return_rate, traded_at)
  select p_uid, lt.ticker, lt.buy_price, lt.sell_price, lt.quantity,
         lt.net_profit, lt.return_rate, coalesce(lt.traded_at, now())
  from public.legacy_trade_history lt
  join public.legacy_users lu on lu.id = lt.user_id
  where lower(lu.email) = lower(p_email);

  -- 추천 이력 이관
  insert into public.recommendation_history
    (user_id, investment_amt, risk_profile, recommendations, created_at)
  select p_uid, lr.investment_amt, lr.risk_profile, lr.recommendations,
         coalesce(lr.created_at, now())
  from public.legacy_recommendation_history lr
  join public.legacy_users lu on lu.id = lr.user_id
  where lower(lu.email) = lower(p_email);
exception when others then
  -- 백필 실패가 가입 자체를 막지 않도록 방어
  raise warning 'migrate_legacy_user 실패(%): %', p_email, sqlerrm;
end $$;

-- ── 5) 신규 가입 트리거: user_settings 생성 + 레거시 자동 이관 ─────────────────
create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
  insert into public.user_settings (user_id) values (new.id)
  on conflict (user_id) do nothing;
  perform public.migrate_legacy_user(new.id, new.email);
  return new;
end $$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
  after insert on auth.users
  for each row execute function public.handle_new_user();
