-- ════════════════════════════════════════════════════════════════════════════
--  004 — 종목별 실제 손절가/익절가 + 시그널 봇 알림 on/off 환경설정
--
--  ① portfolios.stop_loss / take_profit : 사용자가 MTS에 건 실제 손절·익절가(원).
--     비워두면(null) 봇이 진입가+현재ATR로 자동 산출(기존 동작).
--  ② user_settings.alert_prefs : 신호 종류별 알림 on/off (웹 UI에서 제어).
--     예) {"BUY":true,"SELL_TP":true,"SELL_SL":true,"SELL_TS":true,"SELL_REBAL":false}
--     키가 없으면 봇의 .env 기본값을 따른다.
--
--  Supabase SQL Editor 에서 실행. idempotent(여러 번 실행해도 안전).
-- ════════════════════════════════════════════════════════════════════════════

alter table public.portfolios
  add column if not exists stop_loss   double precision,
  add column if not exists take_profit double precision;

alter table public.user_settings
  add column if not exists alert_prefs jsonb not null default '{}'::jsonb;
