-- ════════════════════════════════════════════════════════════════════════════
--  검증 스크립트 — Werkzeug 레거시 사용자 → Supabase Auth 재가입 자동 이관 트리거
--
--  migrations/001_supabase_auth.sql 을 먼저 적용한 뒤, Supabase SQL Editor 에서 실행.
--  전체가 BEGIN/ROLLBACK 으로 감싸져 있어 **아무것도 영구 저장되지 않는다**(비파괴).
--  실제 on_auth_user_created 트리거를 발화시켜 handle_new_user → migrate_legacy_user
--  경로를 end-to-end 로 검증하고, RAISE NOTICE 로 합격/불합격을 출력한다.
-- ════════════════════════════════════════════════════════════════════════════

begin;

do $$
declare
  v_uid     uuid := gen_random_uuid();
  v_email   text := 'legacy-verify-' || floor(random() * 1000000)::int || '@example.com';
  v_legacy  int;
  n_pf      int;
  n_tr      int;
  n_set     int;
begin
  -- ── 0) 선행조건 확인 ──────────────────────────────────────────────────────
  if to_regclass('public.legacy_users') is null then
    raise notice 'ℹ️  legacy_users 가 없습니다(신규 인스턴스). 트리거의 user_settings 생성만 검증합니다.';
  else
    -- ── 1) 레거시 사용자/데이터 시드 (테스트용, 곧 롤백됨) ────────────────────
    insert into public.legacy_users (email, password_hash)
    values (v_email, 'werkzeug$hash$dummy') returning id into v_legacy;

    insert into public.legacy_portfolios (user_id, ticker, avg_price, quantity)
    values (v_legacy, '005930.KS', 70000, 10),
           (v_legacy, 'AAPL',        150,  5);

    insert into public.legacy_trade_history
      (user_id, ticker, buy_price, sell_price, quantity, net_profit, return_rate)
    values (v_legacy, 'TSLA', 200, 250, 3, 150, 25);
  end if;

  -- ── 2) 동일 이메일로 Supabase Auth 가입 시뮬레이션 ────────────────────────
  --     → on_auth_user_created 트리거 발화 → handle_new_user → migrate_legacy_user
  insert into auth.users
    (instance_id, id, aud, role, email, encrypted_password,
     email_confirmed_at, created_at, updated_at,
     raw_app_meta_data, raw_user_meta_data)
  values
    ('00000000-0000-0000-0000-000000000000', v_uid, 'authenticated', 'authenticated',
     v_email, '', now(), now(), now(), '{}'::jsonb, '{}'::jsonb);

  -- ── 3) 검증 ───────────────────────────────────────────────────────────────
  select count(*) into n_set from public.user_settings  where user_id = v_uid;
  select count(*) into n_pf  from public.portfolios     where user_id = v_uid;
  select count(*) into n_tr  from public.trade_history  where user_id = v_uid;

  raise notice '──────────────────────────────────────────────';
  raise notice ' user_settings 생성 (기대 1)      : %', n_set;
  if to_regclass('public.legacy_users') is not null then
    raise notice ' portfolios 이관 (기대 2)         : %', n_pf;
    raise notice ' trade_history 이관 (기대 1)      : %', n_tr;
    if n_set = 1 and n_pf = 2 and n_tr = 1 then
      raise notice ' ✅ PASS — 트리거 백필 정상 동작';
    else
      raise warning ' ❌ FAIL — 기대값과 불일치 (트리거/함수 로직 점검 필요)';
    end if;
  else
    if n_set = 1 then
      raise notice ' ✅ PASS — user_settings 자동 생성 정상 (이관 대상 없음)';
    else
      raise warning ' ❌ FAIL — user_settings 미생성';
    end if;
  end if;
  raise notice '──────────────────────────────────────────────';
end $$;

-- 모든 시드/가입/이관 데이터를 되돌린다 (영구 저장 없음).
rollback;
