-- ════════════════════════════════════════════════════════════════════════════
--  002 — jobs 비동기 큐 정비 (ASA / 백테스트 백그라운드 워커용)
--
--  상태 어휘: pending → processing → completed | error
--  Next.js API 는 pending 으로 인서트(큐잉), Python 워커가 SKIP LOCKED 로 원자적 클레임.
--  schema.sql 의 jobs 테이블이 이미 존재한다는 전제(없으면 schema.sql 먼저 실행).
-- ════════════════════════════════════════════════════════════════════════════

-- 기본 상태를 'pending' 으로 (기존 default 'running' 대체)
alter table public.jobs alter column status set default 'pending';

-- 워커 클레임 가속용 부분 인덱스 (pending 만)
create index if not exists jobs_pending_idx
  on public.jobs (created_at)
  where status = 'pending';

-- 상태 점검용 보조 인덱스
create index if not exists jobs_status_kind_idx on public.jobs (status, kind);

-- (선택) 상태값 무결성 — 잘못된 상태 인서트 방지
do $$
begin
  if not exists (
    select 1 from pg_constraint where conname = 'jobs_status_chk'
  ) then
    alter table public.jobs
      add constraint jobs_status_chk
      check (status in ('pending', 'processing', 'completed', 'error'));
  end if;
end $$;

comment on table public.jobs is
  'ASA/백테스트 비동기 작업 큐. Next.js=enqueue(pending), Python 워커=처리(processing→completed/error).';
