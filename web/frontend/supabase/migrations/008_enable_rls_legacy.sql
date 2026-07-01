-- ════════════════════════════════════════════════════════════════════════════
--  008 — 레거시 테이블 RLS 활성화 (Supabase advisor: "RLS Disabled in Public")
--
--  001 에서 구 인증 스키마를 legacy_* 로 이름만 바꿔 보존했다. 이 테이블들은
--  public 스키마에 있어 PostgREST(anon/authenticated 키)로 자동 노출되는데, RLS 가
--  꺼져 있어 email·password_hash 등 민감 데이터가 API 로 새어나갈 수 있다(advisor 경고).
--
--  · 이 테이블을 읽는 유일한 주체는 migrate_legacy_user() — security definer 라 RLS 를
--    우회한다. 따라서 정책 없이 RLS 만 켜면(= 전면 차단) 마이그레이션은 그대로 동작하고
--    API 직접 접근만 막힌다. 007(price_bars)과 동일한 'RLS enable + 정책 없음' 방식.
--
--  · 신규 인스턴스엔 legacy_* 가 없을 수 있으므로 to_regclass 가드로 존재할 때만 적용.
--
--  Supabase SQL Editor 에서 실행. idempotent(여러 번 실행해도 안전).
-- ════════════════════════════════════════════════════════════════════════════

do $$
declare
  t text;
begin
  foreach t in array array[
    'public.legacy_users',
    'public.legacy_portfolios',
    'public.legacy_trade_history',
    'public.legacy_recommendation_history'
  ]
  loop
    if to_regclass(t) is not null then
      execute format('alter table %s enable row level security', t);
      raise notice 'RLS 활성화: %', t;
    else
      raise notice 'skip(없음): %', t;
    end if;
  end loop;
end $$;
