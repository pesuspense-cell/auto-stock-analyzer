"""e2e_live.py — 워커 ↔ 실 Supabase 라이브 E2E 검증.

실행:
  cd web/backend
  SUPABASE_DB_URL=postgresql://... python -m worker.e2e_live           # 기본(빠름·안전)
  SUPABASE_DB_URL=postgresql://... python -m worker.e2e_live --full    # 전체 처리 경로까지

기본 모드(네트워크 비의존, 비파괴):
  Phase 1  스키마·migration 002 검증 (status default='pending', 인덱스, CHECK 제약, portfolios)
  Phase 2  워커 계약 DB 왕복 — sentinel job 으로 claim→_finish(완료/에러) 라운드트립.
           · 실제 worker._finish / _as_json / _json_default 를 호출 → numpy/datetime jsonb 직렬화 검증.
           · claim 은 kind='__e2e_probe__' 로 스코프 → 실사용자 jobs 를 절대 건드리지 않음.
           · CHECK 제약이 잘못된 status 를 거부하는지 확인.
           · 모든 sentinel 행은 종료 시 삭제(비파괴).

--full 모드(추가): 실제 `python -m worker.jobs_worker` 서브프로세스를 띄우고 asa job(NULL user)을
  큐잉 → completed/error 까지 폴링. live_screener 전시장 스캔이라 수 분·네트워크 소요.

종료코드: 모든 단언 통과 0, 실패 1.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import psycopg2
import psycopg2.extras

PROBE_KIND = "__e2e_probe__"

# ── 출력 헬퍼 ────────────────────────────────────────────────────────────────
_PASS = 0
_FAIL = 0


def _ok(msg: str) -> None:
    global _PASS
    _PASS += 1
    print(f"  ✅ {msg}")


def _bad(msg: str) -> None:
    global _FAIL
    _FAIL += 1
    print(f"  ❌ {msg}")


def _check(cond: bool, ok_msg: str, bad_msg: str | None = None) -> bool:
    if cond:
        _ok(ok_msg)
    else:
        _bad(bad_msg or ok_msg)
    return cond


def _section(title: str) -> None:
    print(f"\n── {title} " + "─" * max(0, 60 - len(title)))


# ── DB URL 해석 ──────────────────────────────────────────────────────────────
def _resolve_db_url() -> str:
    url = os.getenv("SUPABASE_DB_URL", "").strip()
    if url:
        return url
    # backend/.env 폴백
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("SUPABASE_DB_URL"):
                v = line.split("=", 1)[1].strip().strip('"').strip("'")
                if v:
                    return v
    raise SystemExit(
        "❌ SUPABASE_DB_URL 이 없습니다. 환경변수로 지정하거나 web/backend/.env 에 넣어주세요.\n"
        "   예) SUPABASE_DB_URL=postgresql://postgres:<pw>@db.<ref>.supabase.co:5432/postgres"
    )


def _redact(url: str) -> str:
    import re
    return re.sub(r"://([^:]+):[^@]+@", r"://\1:<redacted>@", url)


# ── Phase 1: 스키마 / migration ──────────────────────────────────────────────
def phase1_schema(conn) -> bool:
    """jobs 테이블 존재 여부를 반환 — False 면 후속 Phase 2/3 를 건너뛴다."""
    _section("Phase 1 — 스키마 · migration 002 검증 (read-only)")
    with conn.cursor() as cur:
        # jobs 컬럼
        cur.execute(
            """
            SELECT column_name, column_default, is_nullable
              FROM information_schema.columns
             WHERE table_schema='public' AND table_name='jobs'
            """
        )
        cols = {r[0]: (r[1], r[2]) for r in cur.fetchall()}
        jobs_exists = bool(cols)
        _check(jobs_exists, "public.jobs 테이블 존재", "public.jobs 테이블이 없음 → schema.sql 먼저 실행")
        for c in ("id", "user_id", "kind", "status", "params", "result", "error", "created_at", "updated_at"):
            _check(c in cols, f"jobs.{c} 컬럼 존재", f"jobs.{c} 컬럼 누락")

        # status default = 'pending' (migration 002)
        default = (cols.get("status") or ("", ""))[0] or ""
        _check(
            "pending" in default,
            f"status 기본값이 'pending' (migration 002 적용됨) — '{default}'",
            f"status 기본값이 'pending' 아님 → migration 002 미적용 (현재: '{default}')",
        )

        # 부분 인덱스
        cur.execute("SELECT indexname FROM pg_indexes WHERE schemaname='public' AND tablename='jobs'")
        idx = {r[0] for r in cur.fetchall()}
        _check("jobs_pending_idx" in idx, "jobs_pending_idx 부분 인덱스 존재", "jobs_pending_idx 누락 → migration 002 미적용")
        _check("jobs_status_kind_idx" in idx, "jobs_status_kind_idx 인덱스 존재", "jobs_status_kind_idx 누락")

        # CHECK 제약
        cur.execute("SELECT 1 FROM pg_constraint WHERE conname='jobs_status_chk'")
        _check(cur.fetchone() is not None, "jobs_status_chk CHECK 제약 존재", "jobs_status_chk 누락 → migration 002 미적용")

        # portfolios
        cur.execute("SELECT to_regclass('public.portfolios')")
        _check(cur.fetchone()[0] is not None, "public.portfolios 테이블 존재 (워커 _load_portfolio 대상)", "public.portfolios 없음")
    conn.rollback()
    return jobs_exists


# ── Phase 2: 워커 계약 DB 왕복 ───────────────────────────────────────────────
def _claim_scoped(conn, kind: str) -> dict | None:
    """worker._claim_one 의 스코프 버전 — kind 로 한정해 실사용자 jobs 를 건드리지 않는다."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            UPDATE public.jobs
               SET status='processing', updated_at=now()
             WHERE id = (
                 SELECT id FROM public.jobs
                  WHERE status='pending' AND kind=%s
                  ORDER BY created_at
                  FOR UPDATE SKIP LOCKED
                  LIMIT 1
             )
            RETURNING id, user_id, kind, params, status
            """,
            (kind,),
        )
        row = cur.fetchone()
    conn.commit()
    return dict(row) if row else None


def _enqueue_probe(conn, params: dict) -> str:
    jid = str(uuid.uuid4())
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO public.jobs (id, user_id, kind, status, params) VALUES (%s, NULL, %s, 'pending', %s)",
            (jid, PROBE_KIND, psycopg2.extras.Json(params)),
        )
    conn.commit()
    return jid


def _status(conn, jid: str) -> tuple[str, dict | None, str | None]:
    with conn.cursor() as cur:
        cur.execute("SELECT status, result, error FROM public.jobs WHERE id=%s", (jid,))
        r = cur.fetchone()
    return (r[0], r[1], r[2]) if r else ("<gone>", None, None)


def phase2_roundtrip(conn, worker) -> None:
    _section("Phase 2 — 워커 계약 DB 왕복 (claim → _finish, 비파괴)")
    import numpy as np

    # 2-1) 완료 경로 + numpy/datetime jsonb 직렬화
    jid = _enqueue_probe(conn, {"e2e": True, "ts": datetime.now(timezone.utc).isoformat()})
    claimed = _claim_scoped(conn, PROBE_KIND)
    _check(claimed is not None and str(claimed["id"]) == jid,
           "SKIP LOCKED 원자적 claim → 우리 sentinel 반환", "claim 이 sentinel 을 반환하지 못함")
    _check(bool(claimed) and claimed["status"] == "processing",
           "claim 후 status='processing' 전이", "status 가 processing 으로 바뀌지 않음")

    # 실제 worker._finish + _as_json + _json_default 호출 (numpy/datetime 포함)
    result = {
        "output": "e2e-probe-ok",
        "np_int": np.int64(42),
        "np_float": np.float64(3.14),
        "np_arr": np.array([1, 2, 3]),
        "when": datetime.now(timezone.utc),
    }
    worker._finish(conn, jid, result=result)
    st, res, err = _status(conn, jid)
    _check(st == "completed", "_finish(result=...) → status='completed'", f"status 가 completed 아님: {st}")
    _check(res is not None and res.get("output") == "e2e-probe-ok", "result jsonb 저장/회수 정상", "result jsonb 불일치")
    _check(bool(res) and res.get("np_int") == 42 and abs((res.get("np_float") or 0) - 3.14) < 1e-9,
           "numpy 스칼라 jsonb 직렬화 (_json_default) 정상", "numpy 직렬화 실패")
    _check(bool(res) and res.get("np_arr") == [1, 2, 3], "numpy ndarray → JSON 배열 직렬화 정상", "ndarray 직렬화 실패")

    # 2-2) 에러 경로
    jid2 = _enqueue_probe(conn, {"e2e": "err"})
    _claim_scoped(conn, PROBE_KIND)
    worker._finish(conn, jid2, error="의도된 e2e 에러 메시지")
    st2, _, err2 = _status(conn, jid2)
    _check(st2 == "error", "_finish(error=...) → status='error'", f"status 가 error 아님: {st2}")
    _check(err2 == "의도된 e2e 에러 메시지", "error 텍스트 저장 정상", f"error 텍스트 불일치: {err2!r}")

    # 2-3) CHECK 제약이 잘못된 status 를 거부하는지
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO public.jobs (user_id, kind, status, params) VALUES (NULL, %s, 'bogus', %s)",
                (PROBE_KIND, psycopg2.extras.Json({})),
            )
        conn.commit()
        _bad("CHECK 제약이 잘못된 status('bogus')를 허용함 (거부 기대)")
    except psycopg2.errors.CheckViolation:
        conn.rollback()
        _ok("jobs_status_chk 가 잘못된 status('bogus')를 거부 (IntegrityError)")
    except Exception as e:  # noqa: BLE001
        conn.rollback()
        _bad(f"예상치 못한 예외(CHECK 검증): {type(e).__name__}: {e}")


def _cleanup(conn) -> int:
    conn.rollback()  # 이전 단계가 실패 트랜잭션을 남겼을 수 있음 — 먼저 정리
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('public.jobs')")
        if cur.fetchone()[0] is None:
            conn.rollback()
            return 0
        cur.execute("DELETE FROM public.jobs WHERE kind=%s", (PROBE_KIND,))
        n = cur.rowcount
    conn.commit()
    return n


# ── Phase 3 (--full): 실제 워커 서브프로세스 ─────────────────────────────────
def phase3_full(conn, db_url: str, timeout: float) -> None:
    import subprocess

    _section(f"Phase 3 — 실제 워커 서브프로세스 라이브 (--full, timeout={timeout:.0f}s)")
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM public.jobs WHERE status='pending'")
        pending = cur.fetchone()[0]
    conn.rollback()
    if pending:
        print(f"  ⚠️ 기존 pending job {pending}건 — 실제 워커는 이들도 처리합니다(테스트 DB 권장).")

    env = {**os.environ, "SUPABASE_DB_URL": db_url, "JOBS_POLL_INTERVAL": "1"}
    backend_dir = Path(__file__).resolve().parents[1]
    proc = subprocess.Popen(
        [sys.executable, "-m", "worker.jobs_worker"],
        cwd=str(backend_dir), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    print(f"  워커 PID={proc.pid} 기동")
    try:
        # asa job (NULL user → 빈 포트폴리오) 큐잉
        jid = str(uuid.uuid4())
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO public.jobs (id, user_id, kind, status, params) VALUES (%s, NULL, 'asa', 'pending', %s)",
                (jid, psycopg2.extras.Json({"cash": 1_000_000})),
            )
        conn.commit()
        print(f"  asa job 큐잉: {jid}")

        deadline = time.time() + timeout
        last = None
        while time.time() < deadline:
            if proc.poll() is not None:
                _bad(f"워커 프로세스가 조기 종료(exit={proc.returncode})")
                break
            st, res, err = _status(conn, jid)
            if st != last:
                print(f"  status={st}")
                last = st
            if st in ("completed", "error"):
                break
            time.sleep(2)

        st, res, err = _status(conn, jid)
        if st == "completed":
            head = (str((res or {}).get("output", ""))[:200]).replace("\n", " ")
            _ok(f"실제 워커가 asa job 완료 → result.output[:200]: {head!r}")
        elif st == "error":
            _bad(f"워커가 asa job 을 error 처리: {err}")
        else:
            _bad(f"timeout 내 미완료 (마지막 status={st})")
    finally:
        proc.terminate()
        try:
            out, _ = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, _ = proc.communicate()
        # 워커 로그 일부
        tail = "\n".join((out or "").strip().splitlines()[-8:])
        if tail:
            print("  ── 워커 로그(말미) ──")
            for ln in tail.splitlines():
                print(f"    {ln}")
        # cleanup
        with conn.cursor() as cur:
            cur.execute("DELETE FROM public.jobs WHERE id=%s", (jid,))
        conn.commit()


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description="워커 ↔ 실 Supabase 라이브 E2E")
    ap.add_argument("--full", action="store_true", help="실제 워커 서브프로세스로 asa job 처리까지 (수 분·네트워크)")
    ap.add_argument("--timeout", type=float, default=600, help="--full job 완료 대기 한도(초)")
    args = ap.parse_args()

    db_url = _resolve_db_url()
    print(f"\U0001f50c 대상 Supabase: {_redact(db_url)}")

    try:
        conn = psycopg2.connect(dsn=db_url, connect_timeout=10)
        conn.autocommit = False
    except Exception as e:  # noqa: BLE001
        print(f"❌ DB 연결 실패: {type(e).__name__}: {e}")
        return 1

    with conn.cursor() as cur:
        cur.execute("SELECT version()")
        print(f"  ✅ 연결 성공 — {cur.fetchone()[0].split(',')[0]}")
    conn.rollback()

    # 실제 워커 모듈 import (worker._finish / _as_json / _json_default 테스트)
    from worker import jobs_worker as worker  # noqa: WPS433

    try:
        jobs_exists = phase1_schema(conn)
        if not jobs_exists:
            _section("Phase 2/3 — 건너뜀")
            print("  ⏭️  public.jobs 테이블이 없어 워커 계약 검증을 건너뜁니다.")
            print("     → frontend/supabase/schema.sql (jobs 포함) 적용 후 migrations/002_jobs_queue.sql 실행.")
        else:
            phase2_roundtrip(conn, worker)
            if args.full:
                phase3_full(conn, db_url, args.timeout)
    finally:
        n = _cleanup(conn)
        print(f"\n\U0001f9f9 정리: sentinel job {n}건 삭제")
        conn.close()

    print(f"\n{'='*64}\n결과: PASS={_PASS}  FAIL={_FAIL}")
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
