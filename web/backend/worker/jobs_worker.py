"""jobs_worker.py — Supabase `jobs` 큐를 처리하는 Python 백그라운드 워커.

아키텍처:
  Next.js API → jobs(status='pending') 인서트 → [이 워커] 폴링·클레임 → 연산 →
  결과를 jobs.result 에 기록하고 status='completed'(또는 'error') 로 갱신.

레인(lane) 분리 — 인터랙티브 지연 방지:
  · batch       : backtest, asa (수 분 단위 장기 작업)
  · interactive : analysis, news, fundamental, fundamental_ai (사용자 대기 → 빠른 응답 필요)
  배치 작업이 워커를 점유해도 인터랙티브 작업이 뒤에 줄 서지 않도록, 레인별로 독립
  스레드(+독립 DB 커넥션)가 자기 레인의 kind 만 클레임한다. interactive 레인은 동시
  처리량을 위해 N개 스레드를 띄운다(JOBS_INTERACTIVE_CONCURRENCY, 기본 3).

핵심:
  · 원자적 클레임: `FOR UPDATE SKIP LOCKED` + `kind = ANY(%s)` 로 레인별 중복 없이 클레임.
  · 신규 Supabase 포트폴리오(public.portfolios, uuid)를 직접 참조 → 레거시 DB 의존 제거.
  · 무거운 분석/뉴스/펀더멘털 연산을 API(FastAPI)에서 이 워커로 이관 → API 는 경량 유지.

실행:
  cd web/backend
  SUPABASE_DB_URL=postgresql://... python -m worker.jobs_worker
"""
from __future__ import annotations

import json
import logging
import math
import os
import signal
import threading
import time
from datetime import datetime, timedelta, timezone

import psycopg2
import psycopg2.extras

from app import bootstrap  # noqa: F401  (sys.path: stock_ai/live_screener/backtest)
from app.core.config import settings
from app.services import (
    analysis_service,
    fundamental_service,
    news_service,
    portfolio_service,
    stock_lists,
)
from app.services.asa_service import build_positions, run_asa
from app.services.backtest_service import run_backtest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [worker] %(levelname)s: %(message)s",
)
logger = logging.getLogger("jobs_worker")

POLL_INTERVAL = float(os.getenv("JOBS_POLL_INTERVAL", "1"))
# 고아 작업 회수 임계값(초). 워커가 작업을 'processing' 으로 클레임한 뒤 재시작·OOM 등으로
# 죽으면 그 작업은 영원히 'processing' 에 남아 프론트가 무한 폴링한다(스피너가 끝나지 않음).
# 정상 작업의 최대 소요는 ~140s(AI 리포트) 수준이므로, 이 값보다 오래 'processing' 인 작업은
# 고아로 간주해 'error' 로 회수한다(→ UI 에 에러+다시시도 노출). 기본 10분.
STALE_PROCESSING_SEC = int(os.getenv("JOBS_STALE_SEC", "600"))
# 리퍼 점검 주기(초).
REAPER_INTERVAL = float(os.getenv("JOBS_REAPER_INTERVAL", "60"))
_STALE_ERROR_MSG = "작업이 제한 시간 내 완료되지 않았습니다 (워커 재시작 추정). 다시 시도해 주세요."
# 인터랙티브 동시 처리 스레드 수. 분석/뉴스 작업은 대부분 네트워크(yfinance·LLM) 대기라
# 스레드 동시성이 지연을 실질적으로 줄인다. 다만 동시 pandas 연산은 메모리를 먹으므로
# Render starter(512MB)에선 2가 안전. 메모리 여유가 있으면 환경변수로 올린다.
INTERACTIVE_CONCURRENCY = max(1, int(os.getenv("JOBS_INTERACTIVE_CONCURRENCY", "2")))

# 종료(SIGTERM) 시 진행 중 작업이 끝날 때까지 기다리는 graceful drain 시간(초).
# Render 재배포는 SIGTERM 을 보낸다. 짧은 작업(뉴스·펀더멘털 ~수초)은 이 시간 안에 끝나
# 정상 완료되고, 그래도 안 끝난 무거운 작업(분석·백테스트)은 _requeue_inflight 가 pending 으로
# 되돌려 다음 인스턴스가 재처리한다. 기본 90초.
DRAIN_TIMEOUT = float(os.getenv("JOBS_DRAIN_TIMEOUT", "90"))

# 레인 정의 — kind → 레인. 배치는 1스레드, 인터랙티브는 N스레드.
BATCH_KINDS = ("backtest", "asa")
INTERACTIVE_KINDS = ("analysis", "news", "fundamental", "fundamental_ai", "portfolio_analysis", "investors")

_running = True

# 현재 각 레인이 잡고 처리 중인 job id (lane → job_id). 종료 시 미완료분을 pending 으로 되돌리는 데 쓴다.
_inflight: dict[str, str] = {}
_inflight_lock = threading.Lock()


def _db_url() -> str:
    url = os.getenv("SUPABASE_DB_URL", "")
    if not url:
        raise RuntimeError("SUPABASE_DB_URL 환경변수가 필요합니다.")
    return url


def _json_default(o):
    """numpy/pandas 타입을 JSON 직렬화 가능한 형태로 변환(잔여 폴백)."""
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


def _json_safe(o):
    """JSON/Postgres 안전 변환 — numpy 타입 정규화 + NaN/Infinity → null.

    PostgreSQL jsonb 는 NaN/Infinity 토큰을 거부하므로(파이썬 json 은 기본 허용),
    payload 를 재귀적으로 훑어 비유한(非有限) float 을 None 으로 바꾼다.
    (뉴스 sector_performance 의 avg_chg=NaN 으로 인한 작업 실패 수정)
    """
    try:
        import numpy as np
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.floating):
            o = float(o)
        elif isinstance(o, np.ndarray):
            return [_json_safe(x) for x in o.tolist()]
    except Exception:
        pass
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, datetime):
        return o.isoformat()
    return o


def _as_json(payload: dict):
    return psycopg2.extras.Json(_json_safe(payload), dumps=lambda x: json.dumps(x, default=_json_default))


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


def _sname(ticker: str) -> str:
    return stock_lists.ticker_name_map().get(ticker, ticker)


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
        return {"output": run_asa(cash, positions)}

    if kind == "backtest":
        logger.info("백테스트 실행 — %s", {k: params.get(k) for k in ("markets", "start_date", "end_date")})
        return run_backtest(params)

    if kind == "analysis":
        ticker = params["ticker"]
        period = params.get("period", "6mo")
        use_llm = bool(params.get("useLlm", False))
        logger.info("분석 실행 — %s (period=%s, llm=%s)", ticker, period, use_llm)
        return analysis_service.analyze(
            ticker, period, use_llm,
            settings.gemini_api_key, settings.groq_api_key, _sname(ticker),
        )

    if kind == "news":
        ticker = params["ticker"]
        cname = stock_lists.ticker_name_map().get(ticker, "")
        logger.info("뉴스 실행 — %s", ticker)
        data = news_service.news(ticker, settings.gemini_api_key, settings.groq_api_key, cname)
        if not data.get("etf_meta"):
            data["etf_meta"] = None
        return data

    if kind == "fundamental":
        ticker = params["ticker"]
        logger.info("펀더멘털 실행 — %s", ticker)
        return fundamental_service.fundamental(ticker)

    if kind == "investors":
        # 투자자별 매매동향(수급) — 일일 갱신. 분기 캐시되는 fundamental 과 분리해 신선도 유지.
        ticker = params["ticker"]
        logger.info("수급 실행 — %s", ticker)
        return {
            "investors": fundamental_service.investors(ticker),
            "history": fundamental_service.investor_history(ticker),
        }

    if kind == "fundamental_ai":
        ticker = params["ticker"]
        use_llm = bool(params.get("useLlm", True))
        logger.info("AI 재무리포트 실행 — %s (llm=%s)", ticker, use_llm)
        return fundamental_service.ai_report(
            ticker, settings.gemini_api_key, settings.groq_api_key, use_llm, _sname(ticker),
        )

    if kind == "portfolio_analysis":
        raw_items = params.get("items", []) or []
        items = [
            {"ticker": i["ticker"], "avg_price": float(i["avgPrice"]), "quantity": float(i["quantity"])}
            for i in raw_items
        ]
        prices = {i["ticker"]: float(i["price"]) for i in raw_items if i.get("price") is not None}
        name_map = {i["ticker"]: (i.get("name") or i["ticker"]) for i in raw_items}
        logger.info("포트폴리오 분석 — %d종목", len(items))
        return portfolio_service.analyze(items, prices, name_map)

    raise ValueError(f"알 수 없는 작업 종류: {kind}")


# 클레임 우선순위 — 작은 값일수록 먼저 처리. 사용자가 화면에서 대기 중인 작업
# (analysis·portfolio_analysis)을 차트탭의 투기적 prefetch(news·fundamental)보다
# 먼저 집어, 2스레드밖에 없어도 화면용 분석이 prefetch 뒤에 줄 서지 않게 한다.
_CLAIM_PRIORITY_SQL = """
            CASE kind
                WHEN 'analysis' THEN 0
                WHEN 'portfolio_analysis' THEN 0
                WHEN 'fundamental_ai' THEN 1
                ELSE 2
            END, created_at
"""


def _claim_one(conn, kinds: tuple[str, ...]) -> dict | None:
    """주어진 레인(kinds)의 pending 작업 1건을 원자적으로 클레임(→processing).

    우선순위(_CLAIM_PRIORITY_SQL) → created_at 순. 동일 우선순위는 FIFO."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            f"""
            UPDATE public.jobs
               SET status = 'processing', updated_at = now()
             WHERE id = (
                 SELECT id FROM public.jobs
                  WHERE status = 'pending' AND kind = ANY(%s)
                  ORDER BY {_CLAIM_PRIORITY_SQL}
                  FOR UPDATE SKIP LOCKED
                  LIMIT 1
             )
            RETURNING id, user_id, kind, params
            """,
            (list(kinds),),
        )
        row = cur.fetchone()
    conn.commit()
    return dict(row) if row else None


_KST = timezone(timedelta(hours=9))


def _quarter_key(now: datetime | None = None) -> str:
    """현재 분기 키(KST 기준, 예: 2026Q2). fundamental 캐시 scope 에 사용 — 분기당 1회 갱신."""
    n = now or datetime.now(_KST)
    return f"{n.year}Q{(n.month - 1) // 3 + 1}"


def _cache_scope(job: dict) -> str | None:
    """kind 별 market_cache scope 키. 캐시 대상이 아니면 None.

    재조회 즉시응답(enqueue 단계)을 위한 키. 각 enqueue 라우트가 동일 규칙으로 키를
    만들어 캐시를 조회하므로(둘이 어긋나면 캐시 미스), 형식을 반드시 일치시킬 것.
      · news        → 일 단위 TTL 로 갱신(scope 고정)
      · fundamental → 분기 키 포함 → 분기 경계에서만 자동 갱신
      · analysis    → period·LLM 에 따라 결과가 달라지므로 scope 에 함께 인코딩
    """
    kind = job["kind"]
    params = job.get("params") or {}
    ticker = params.get("ticker")
    if not ticker:
        return None
    tkr = str(ticker).upper()
    if kind == "news":
        return f"news:{tkr}"
    if kind == "investors":
        # 일 단위 키 — KST 날짜가 바뀌면 scope 가 달라져 자동 재수집(수급은 매일 변함).
        return f"investors:{tkr}:{datetime.now(_KST).strftime('%Y%m%d')}"
    if kind == "fundamental":
        return f"fundamental:{tkr}:{_quarter_key()}"
    if kind == "analysis":
        period = params.get("period", "6mo")
        lane = "llm" if params.get("useLlm", False) else "base"
        return f"analysis:{tkr}:{period}:{lane}"
    return None


def _cache_result(conn, job: dict, result: dict) -> None:
    """analysis/news/fundamental 결과를 market_cache 에 upsert → 다음 조회는 enqueue 단계에서 즉시응답."""
    scope = _cache_scope(job)
    if not scope:
        return
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO public.market_cache (scope, payload, fetched_at)
                VALUES (%s, %s, now())
                ON CONFLICT (scope) DO UPDATE SET payload = EXCLUDED.payload, fetched_at = now()
                """,
                (scope, _as_json(result)),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        logger.warning("market_cache 저장 실패: %s", scope)


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


# ── 고아 작업 리퍼 ────────────────────────────────────────────────────────────

def _reap_stale(conn) -> int:
    """STALE_PROCESSING_SEC 보다 오래 'processing' 에 갇힌 고아 작업을 'error' 로 회수.

    워커가 작업을 클레임(processing)한 직후 죽으면(OOM·재배포 등) 그 작업을 완료/실패로
    옮길 주체가 사라져 프론트가 무한 폴링한다. 클레임 시점에 updated_at 을 now() 로 찍으므로
    (_claim_one), 그 뒤로 오래 갱신이 없으면 진행 중 죽은 고아로 판단할 수 있다.
    반환: 회수한 작업 수."""
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE public.jobs
               SET status='error', error=%s, updated_at=now()
             WHERE status='processing'
               AND updated_at < now() - make_interval(secs => %s)
            RETURNING id
            """,
            (_STALE_ERROR_MSG, STALE_PROCESSING_SEC),
        )
        n = cur.rowcount
    conn.commit()
    return n


def _reaper_loop() -> None:
    """독립 커넥션으로 주기적으로 고아 작업을 회수한다(기동 시 1회 + REAPER_INTERVAL 마다)."""
    conn = psycopg2.connect(dsn=_db_url(), connect_timeout=10)
    conn.autocommit = False
    logger.info("[reaper] 시작 — 임계값 %ds, 점검주기 %.0fs", STALE_PROCESSING_SEC, REAPER_INTERVAL)
    try:
        while _running:
            try:
                n = _reap_stale(conn)
                if n:
                    logger.warning("[reaper] 고아 작업 %d건 회수(error 처리)", n)
            except psycopg2.OperationalError as e:
                logger.warning("[reaper] DB 연결 오류 — 재연결: %s", e)
                try:
                    conn.close()
                except Exception:
                    pass
                conn = psycopg2.connect(dsn=_db_url(), connect_timeout=10)
                conn.autocommit = False
            except Exception:
                conn.rollback()
                logger.exception("[reaper] 회수 중 오류")
            # REAPER_INTERVAL 동안 대기하되, 종료 신호에 빠르게 반응하도록 잘게 나눠 잔다.
            slept = 0.0
            while _running and slept < REAPER_INTERVAL:
                time.sleep(min(1.0, REAPER_INTERVAL - slept))
                slept += 1.0
    finally:
        try:
            conn.close()
        except Exception:
            pass
        logger.info("[reaper] 정지.")


# ── 레인 워커 루프 ────────────────────────────────────────────────────────────

def _lane_loop(lane: str, kinds: tuple[str, ...]) -> None:
    """한 스레드 = 한 DB 커넥션 = 한 레인. 자기 레인 작업만 클레임·처리한다."""
    conn = psycopg2.connect(dsn=_db_url(), connect_timeout=10)
    conn.autocommit = False
    logger.info("[%s] 레인 워커 시작 — kinds=%s", lane, kinds)
    try:
        while _running:
            try:
                job = _claim_one(conn, kinds)
            except psycopg2.OperationalError as e:
                logger.warning("[%s] DB 연결 오류 — 재연결: %s", lane, e)
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
            with _inflight_lock:
                _inflight[lane] = jid
            logger.info("[%s] 작업 클레임: %s (%s)", lane, jid, job["kind"])
            t0 = time.time()
            try:
                result = _process(conn, job)
                _cache_result(conn, job, result)
                _finish(conn, jid, result=result)
                logger.info("[%s] ✅ 완료: %s (%.1fs)", lane, jid, time.time() - t0)
            except Exception as e:
                conn.rollback()
                logger.exception("[%s] ❌ 실패: %s", lane, jid)
                try:
                    _finish(conn, jid, error=str(e))
                except Exception:
                    logger.exception("[%s] 실패 상태 기록조차 실패: %s", lane, jid)
            finally:
                with _inflight_lock:
                    _inflight.pop(lane, None)
    finally:
        try:
            conn.close()
        except Exception:
            pass
        logger.info("[%s] 레인 워커 정지.", lane)


# ── 메인 ──────────────────────────────────────────────────────────────────────

def _stop(*_):
    global _running
    _running = False
    logger.info("종료 신호 수신 — 진행 중 작업을 %ds 까지 기다린 뒤 미완료분은 재큐합니다.", int(DRAIN_TIMEOUT))


def _requeue_inflight() -> None:
    """graceful drain 후에도 끝나지 않은 in-flight 작업을 'processing'→'pending' 으로 되돌린다.

    재배포(SIGTERM)로 워커가 내려갈 때, 잡고 있던 무거운 작업을 잃지 않고 새 인스턴스가 즉시
    다시 집게 한다(리퍼의 10분 안전망보다 빠른 능동 회수). 이미 완료된 작업은 status 조건으로 건너뛴다."""
    with _inflight_lock:
        ids = list({v for v in _inflight.values()})
    if not ids:
        return
    try:
        conn = psycopg2.connect(dsn=_db_url(), connect_timeout=10)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE public.jobs SET status='pending', updated_at=now() "
                    "WHERE id = ANY(%s) AND status='processing'",
                    (ids,),
                )
                n = cur.rowcount
            conn.commit()
            if n:
                logger.info("[shutdown] 미완료 in-flight %d건을 pending 으로 재큐", n)
        finally:
            conn.close()
    except Exception:
        logger.exception("[shutdown] in-flight 재큐 실패 — 리퍼가 추후 회수")


def main() -> None:
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    threads: list[threading.Thread] = []
    # batch 레인 1스레드
    threads.append(threading.Thread(target=_lane_loop, args=("batch", BATCH_KINDS), daemon=True))
    # interactive 레인 N스레드
    for i in range(INTERACTIVE_CONCURRENCY):
        threads.append(
            threading.Thread(target=_lane_loop, args=(f"interactive-{i + 1}", INTERACTIVE_KINDS), daemon=True)
        )
    # 고아 작업 리퍼 1스레드 — 죽은 워커가 남긴 'processing' 고아를 회수해 무한 폴링 방지.
    threads.append(threading.Thread(target=_reaper_loop, daemon=True))

    logger.info(
        "워커 시작 — batch 1 + interactive %d + reaper 1 스레드 (poll=%.1fs)",
        INTERACTIVE_CONCURRENCY, POLL_INTERVAL,
    )
    for t in threads:
        t.start()

    # 메인 스레드는 종료 신호를 기다린다(시그널 수신 위해 살아있어야 함).
    try:
        while _running:
            time.sleep(0.5)
    except KeyboardInterrupt:
        _stop()

    # graceful drain — 공통 deadline 안에서 각 스레드가 진행 중 작업을 끝내도록 기다린다.
    deadline = time.time() + DRAIN_TIMEOUT
    for t in threads:
        t.join(timeout=max(0.1, deadline - time.time()))
    # drain 후에도 남은 미완료 작업은 pending 으로 되돌려 다음 인스턴스가 재처리.
    _requeue_inflight()
    logger.info("워커 정지 완료.")


if __name__ == "__main__":
    main()
