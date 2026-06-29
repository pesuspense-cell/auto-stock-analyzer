-- ════════════════════════════════════════════════════════════════════════════
--  005 — 사용자별 텔레그램 알림 연동 (다중 사용자 1:N 발송)
--
--  각 사용자가 자신의 텔레그램으로 알림을 받도록 user_settings 에 연동 정보를 둔다.
--    telegram_chat_id    : 연동된 텔레그램 숫자 chat id (없으면 미연동)
--    telegram_enabled    : 알림 수신 동의 여부(on/off)
--    telegram_link_token : 딥링크 연동용 일회용 토큰(연동 완료 시 null 로 소거)
--
--  연동 흐름(딥링크): 웹에서 토큰 발급 → t.me/<bot>?start=<token> → 사용자가 [시작] →
--  봇(telegram_link_bot.py)이 토큰으로 사용자를 찾아 chat_id 저장 + enabled=true.
--
--  Supabase SQL Editor 에서 실행. idempotent.
-- ════════════════════════════════════════════════════════════════════════════

alter table public.user_settings
  add column if not exists telegram_chat_id    text,
  add column if not exists telegram_enabled    boolean not null default false,
  add column if not exists telegram_link_token text;

-- 토큰으로 사용자 역조회(봇 연동 처리) 가속 + 토큰 유일성 보장
create unique index if not exists user_settings_tg_token
  on public.user_settings (telegram_link_token)
  where telegram_link_token is not null;
