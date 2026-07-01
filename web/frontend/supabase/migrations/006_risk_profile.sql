-- ════════════════════════════════════════════════════════════════════════════
--  006 — 사용자별 매매 위험성향(risk profile) 선택
--
--  각 사용자가 '내 포트폴리오' 탭에서 매매 알고리즘 성향을 선택한다. 이 값은
--  ① 백테스트 실행 엔진(backtest.py vs backtest_v5_5_active.py)과
--  ② 라이브 시그널봇(signal_bot.py)의 진입 커트라인·종목 캡·서킷브레이커·디펜스
--     눌림목 허용 여부를 함께 분기한다.
--
--    risk_profile = 'safe'       → 안전투자형 v4.6 (기본값, 강한 방어)
--    risk_profile = 'aggressive' → 위험감수형 v5.5 (야수커트 70pt·종목캡 33%·당일손절 2종목·
--                                  디펜스 눌림목 10%캡 허용)
--
--  Supabase SQL Editor 에서 실행. idempotent(여러 번 실행해도 안전).
-- ════════════════════════════════════════════════════════════════════════════

alter table public.user_settings
  add column if not exists risk_profile text not null default 'safe';

-- 허용값만 들어가도록 제약(없으면 추가). safe/aggressive 외 차단.
do $$
begin
  if not exists (
    select 1 from pg_constraint where conname = 'user_settings_risk_profile_chk'
  ) then
    alter table public.user_settings
      add constraint user_settings_risk_profile_chk
      check (risk_profile in ('safe', 'aggressive'));
  end if;
end $$;
