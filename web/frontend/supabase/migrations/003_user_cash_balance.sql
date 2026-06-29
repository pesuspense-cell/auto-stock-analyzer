-- ════════════════════════════════════════════════════════════════════════════
--  003 — 사용자 현금 잔고(예수금) 컬럼 추가
--
--  포트폴리오 총자산(현금 + 보유 평가액) 계산과 시그널 봇 계좌 연동을 위해
--  user_settings 에 현금 잔고를 보관한다. 기존 인스턴스용 비파괴 마이그레이션.
--  Supabase SQL Editor 에서 실행. idempotent(여러 번 실행해도 안전).
-- ════════════════════════════════════════════════════════════════════════════

alter table public.user_settings
  add column if not exists cash_balance double precision not null default 0;
