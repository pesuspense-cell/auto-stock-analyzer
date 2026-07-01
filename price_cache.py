"""price_cache.py — 백테스트 시세 DB 캐시 (read-through).

과거 일봉 OHLCV 를 Supabase(Postgres) `public.price_bars` 테이블에 캐시한다.
백테스트가 매 실행마다 유니버스 전체(최대 400종목)를 재다운로드하던 것을,
    ① 캐시 우선 조회(coverage 판정)
    ② 미스 종목만 소스(FDR)로 수집 후 upsert
    ③ 전체를 캐시에서 벌크 read
하는 read-through 로 바꿔 반복 실행 속도·안정성(yfinance 429 무관)·메모리(필요 구간만)를 개선한다.

- 과거 시세는 정적이라 한 번 적재하면 재사용. 최근 며칠만 refresh 로 갱신.
- `SUPABASE_DB_URL` 미설정(로컬 단독 실행 등)이면 캐시 비활성 → 호출측이 직접 소스 폴백.
- 워커/백엔드는 SUPABASE_DB_URL(pooler)로 직접 psycopg2 접속하므로 RLS 무관(테이블은 RLS 차단만).
"""
from __future__ import annotations

import logging
import math
import os

import pandas as pd

logger = logging.getLogger("price_cache")

_COLS = ("Open", "High", "Low", "Close", "Volume")


def cache_enabled() -> bool:
    """캐시 사용 가능 여부 — SUPABASE_DB_URL 이 설정돼 있으면 True."""
    return bool(os.getenv("SUPABASE_DB_URL", "").strip())


def _connect():
    import psycopg2
    return psycopg2.connect(dsn=os.getenv("SUPABASE_DB_URL", ""), connect_timeout=10)


def _num(v):
    try:
        if v is None:
            return None
        x = float(v)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def coverage_bulk(tickers: list[str]) -> dict[str, tuple]:
    """종목별 캐시 커버리지 {ticker: (min_date, max_date, count)}. 실패 시 빈 dict."""
    if not tickers:
        return {}
    out: dict[str, tuple] = {}
    con = None
    try:
        con = _connect()
        with con.cursor() as cur:
            cur.execute(
                "SELECT ticker, min(date), max(date), count(*) "
                "FROM public.price_bars WHERE ticker = ANY(%s) GROUP BY ticker",
                (list(tickers),),
            )
            for tk, mn, mx, cnt in cur.fetchall():
                out[tk] = (mn, mx, int(cnt))
    except Exception as e:
        logger.warning("coverage_bulk 실패: %s", e)
    finally:
        if con is not None:
            con.close()
    return out


def read_bars_bulk(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    """[start,end] 구간의 종목별 OHLCV DataFrame(Datetime index). 실패/빈값 시 빈 dict."""
    out: dict[str, pd.DataFrame] = {}
    if not tickers:
        return out
    con = None
    try:
        con = _connect()
        with con.cursor() as cur:
            cur.execute(
                "SELECT ticker, date, open, high, low, close, volume "
                "FROM public.price_bars "
                "WHERE ticker = ANY(%s) AND date BETWEEN %s AND %s "
                "ORDER BY ticker, date",
                (list(tickers), str(start)[:10], str(end)[:10]),
            )
            rows = cur.fetchall()
    except Exception as e:
        logger.warning("read_bars_bulk 실패: %s", e)
        return out
    finally:
        if con is not None:
            con.close()
    if not rows:
        return out
    df = pd.DataFrame(rows, columns=["ticker", "date", *_COLS])
    df["date"] = pd.to_datetime(df["date"])
    for c in _COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for tk, g in df.groupby("ticker", sort=False):
        out[str(tk)] = g.drop(columns=["ticker"]).set_index("date").sort_index()
    return out


def upsert_many(items: list[tuple[str, pd.DataFrame]], source: str = "fdr") -> int:
    """[(ticker, OHLCV df), ...] 를 price_bars 에 벌크 upsert. 적재 행수 반환."""
    rows: list[tuple] = []
    for ticker, df in items:
        if df is None or df.empty:
            continue
        for dt, r in df.iterrows():
            try:
                d = pd.Timestamp(dt).date()
            except Exception:
                continue
            rows.append((
                ticker, d,
                _num(r.get("Open")), _num(r.get("High")), _num(r.get("Low")),
                _num(r.get("Close")), _num(r.get("Volume")), source,
            ))
    if not rows:
        return 0
    con = None
    try:
        import psycopg2.extras
        con = _connect()
        with con.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                "INSERT INTO public.price_bars "
                "(ticker, date, open, high, low, close, volume, source) VALUES %s "
                "ON CONFLICT (ticker, date) DO UPDATE SET "
                "open=EXCLUDED.open, high=EXCLUDED.high, low=EXCLUDED.low, "
                "close=EXCLUDED.close, volume=EXCLUDED.volume, "
                "source=EXCLUDED.source, updated_at=now()",
                rows, page_size=2000,
            )
        con.commit()
        return len(rows)
    except Exception as e:
        logger.warning("upsert_many 실패: %s", e)
        return 0
    finally:
        if con is not None:
            con.close()


def covers(cov: tuple | None, start, end, *, recent_slack_days: int = 7,
           min_rows: int = 30) -> bool:
    """캐시 커버리지 튜플(min,max,count)이 [start,end] 를 충분히 덮는가.

    - 시작: 캐시 최소일 ≤ 요청 시작일(웜업 포함)
    - 종료: 캐시 최대일 ≥ min(요청 종료일, 오늘) − recent_slack_days (최근 며칠 미갱신 허용)
    - 행수: min_rows 이상(빈껍데기 방지)
    """
    if not cov:
        return False
    mn, mx, cnt = cov
    if cnt < min_rows or mn is None or mx is None:
        return False
    mn = pd.Timestamp(mn)
    mx = pd.Timestamp(mx)
    start = pd.Timestamp(start)
    need_end = min(pd.Timestamp(end), pd.Timestamp.now().normalize()) - pd.Timedelta(days=recent_slack_days)
    return mn <= start and mx >= need_end
