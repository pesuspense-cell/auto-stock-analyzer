"""jobs_worker.py — Supabase `jobs` 큐를 처리하는 Python 백그라운드 워커.

아키텍처(작업 #3):
  Next.js API → jobs(status='pending') 인서트 → [이 워커] 폴링·클레임 → 연산 →
  결과를 jobs.result 에 기록하고 status='completed'(또는 'error') 로 갱신.

핵심:
  · 원자적 클레임: `FOR UPDATE SKIP LOCKED` 로 여러 워커가 같은 작업을 중복 처리하지 않는다.
  · 신규 Supabase 포트폴리오(public.portfolios, uuid)를 직접 참조 → 레거시 DB 의존 제거.
  · ASA = live_screener, 백테스트 = StockScreener+BacktestEngine (기존 로직 재사용).

실행:
  cd web/backend
  SUPABASE_DB_URL=postgresql://... python -m worker.jobs_worker

Realtime 구독 대안: supabase-py 의 realtime 채널로 INSERT 이벤트를 받을 수도 있으나,
장시간(수 분) 작업 특성상 폴링 + SKIP LOCKED 가 단순하고 견고하다.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import time
from datetime import datetime, timezone

import psycopg2
import psycopg2.extras

from app import bootstrap  # noqa: F401  (sys.path: stock_ai/live_screener/backtest)
from app.services.asa_service import build_positions, run_asa
from app.services.backtest_service import run_backtest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [worker] %(levelname)s: %(message)s",
)
logger = logging.getLogger("jobs_worker")

POLL_INTERVAL = float(os.getenv("JOBS_POLL_INTERVAL", "3"))
_running = True


def _db_url() -> str:
    url = os.getenv("SUPABASE_DB_URL", "")
    if not url:
        raise RuntimeError("SUPABASE_DB_URL 환경변수가 필요합니다.")
    return url


def _json_default(o):
    """numpy/pandas 타입을 JSON 직렬화 가능한 형태로 변환."""
    try:
        import numpy as np
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass
    if isinstance(o, datetime):
        return o.isoformat()
    return str(o)


def _as_json(payload: dict):
    return psycopg2.extras.Json(payload, dumps=lambda x: json.dumps(x, default=_json_default))


# ── 작업 처리기 ───────────────────────────────────────────────────────────────

def _load_portfolio(cur, user_id) -> list[dict]:
    cur.execute(
        "SELECT ticker, avg_price, quantity FROM public.portfolios WHERE user_id = %s",
        (user_id,),
    )
    return [
        {"ticker": r["ticker"], "avg_price": float(r["avg_price"]), "quantity": float(r["quantity"])}
        for r in cur.fetchall()
    ]


def _process(conn, job: dict) -> dict:
    """job 종류별 연산 수행 → result dict 반환."""
    kind = job["kind"]
    params = job["params"] or {}

    if kind == "asa":
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            items = _load_portfolio(cur, job["user_id"]) if job["user_id"] else []
        cash = float(params.get("cash", 1_000_000))
        logger.info("ASA 실행 — 보유 %d종목, 예치금 %s", len(items), f"{cash:,.0f}")
        positions = build_positions(items)
        output = run_asa(cash, positions)
        return {"output": output}

    if kind == "backtest":
        logger.info("백테스트 실행 — %s", {k: params.get(k) for k in ("markets", "start_date", "end_date")})
        return run_backtest(params)

    raise ValueError(f"알 수 없는 작업 종류: {kind}")


def _claim_one(conn) -> dict | None:
    """pending 작업 1건을 원자적으로 클레임(→processing)하고 반환."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            UPDATE public.jobs
               SET status = 'processing', updated_at = now()
             WHERE id = (
                 SELECT id FROM public.jobs
                  WHERE status = 'pending'
                  ORDER BY created_at
                  FOR UPDATE SKIP LOCKED
                  LIMIT 1
             )
            RETURNING id, user_id, kind, params
            """
        )
        row = cur.fetchone()
    conn.commit()
    return dict(row) if row else None


def _finish(conn, job_id, *, result: dict | None = None, error: str | None = None) -> None:
    with conn.cursor() as cur:
        if error is not None:
            cur.execute(
                "UPDATE public.jobs SET status='error', error=%s, updated_at=now() WHERE id=%s",
                (error[:4000], job_id),
            )
        else:
            cur.execute(
                "UPDATE public.jobs SET status='completed', result=%s, error=NULL, updated_at=now() WHERE id=%s",
                (_as_json(result or {}), job_id),
            )
    conn.commit()


# ── 메인 루프 ─────────────────────────────────────────────────────────────────

def _stop(*_):
    global _running
    _running = False
    logger.info("종료 신호 수신 — 현재 작업 완료 후 정지합니다.")


def main() -> None:
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    conn = psycopg2.connect(dsn=_db_url(), connect_timeout=10)
    conn.autocommit = False
    logger.info("워커 시작 — jobs 큐 폴링 (interval=%.1fs)", POLL_INTERVAL)

    try:
        while _running:
            try:
                job = _claim_one(conn)
            except psycopg2.OperationalError as e:
                logger.warning("DB 연결 오류 — 재연결: %s", e)
                try:
                    conn.close()
                except Exception:
                    pass
                time.sleep(POLL_INTERVAL)
                conn = psycopg2.connect(dsn=_db_url(), connect_timeout=10)
                conn.autocommit = False
                continue

            if not job:
                time.sleep(POLL_INTERVAL)
                continue

            jid = job["id"]
            logger.info("작업 클레임: %s (%s)", jid, job["kind"])
            t0 = time.time()
            try:
                result = _process(conn, job)
                _finish(conn, jid, result=result)
                logger.info("✅ 완료: %s (%.1fs)", jid, time.time() - t0)
            except Exception as e:
                conn.rollback()
                logger.exception("❌ 실패: %s", jid)
                try:
                    _finish(conn, jid, error=str(e))
                except Exception:
                    logger.exception("실패 상태 기록조차 실패: %s", jid)
    finally:
        try:
            conn.close()
        except Exception:
            pass
        logger.info("워커 정지 완료.")


if __name__ == "__main__":
    main()
