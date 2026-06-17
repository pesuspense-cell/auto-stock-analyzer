"""backtest_service.py — 백테스트 (StockScreener + BacktestEngine).

ui/backtest_tab.py 의 _run_and_display 데이터 산출부를 이식.
스크리닝 → 엔진 실행은 수 분 소요되므로 백그라운드 잡 + 폴링으로 처리하고,
_report() 가 print 만 하므로 metrics 는 equity_curve/trade_log 에서 직접 계산한다.
"""
from __future__ import annotations

import contextlib
import io
import logging
import threading
import uuid
from datetime import datetime, timezone

import pandas as pd

from app import bootstrap  # noqa: F401

logger = logging.getLogger(__name__)

_JOBS: dict[str, dict] = {}
_LOCK = threading.Lock()


def _metrics(equity_curve: list[dict], trade_log: list[dict]) -> dict:
    """backtest._report() 의 성적표 계산식을 재현 (print 대신 dict 반환)."""
    if not equity_curve:
        return {}
    ec = equity_curve
    last = ec[-1]

    max_peak = float("-inf")
    mdd = 0.0
    for r in ec:
        asset = r["total_asset"]
        if asset > max_peak:
            max_peak = asset
        if max_peak > 0:
            dd = (max_peak - asset) / max_peak * 100
            mdd = max(mdd, dd)

    try:
        n_years = (pd.Timestamp(ec[-1]["date"]) - pd.Timestamp(ec[0]["date"])).days / 365.25
        cagr = ((last["total_asset"] / ec[0]["invested"]) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0.0
    except Exception:
        cagr = 0.0

    sells = [t for t in trade_log if t["action"] in ("SELL_TP", "SELL_TS", "SELL_SL")]
    wins = [t for t in sells if (t["price"] - t["entry_price"]) > 0]
    win_rate = len(wins) / len(sells) * 100 if sells else 0.0
    sl_count = sum(1 for t in trade_log if t["action"] == "SELL_SL")

    return {
        "total_invested": last["invested"],
        "final_asset": last["total_asset"],
        "cash": last["cash"],
        "return_pct": last["return_pct"],
        "cagr": round(cagr, 2),
        "mdd": round(-mdd, 2),
        "win_rate": round(win_rate, 1),
        "total_sells": len(sells),
        "sl_count": sl_count,
    }


def _fetch_benchmark(ticker: str | None, equity_curve: list[dict]) -> list[dict]:
    if not ticker or not equity_curve:
        return []
    try:
        import yfinance as yf
        raw = yf.download(ticker, start=equity_curve[0]["date"], end=equity_curve[-1]["date"],
                          auto_adjust=True, progress=False)
        if raw.empty:
            return []
        close = raw["Close"].dropna()
        first = float(close.iloc[0])
        return [{"date": d.strftime("%Y-%m-%d"), "return_pct": (float(p) / first - 1) * 100}
                for d, p in close.items()]
    except Exception:
        return []


def run_backtest(params: dict) -> dict:
    """스크리닝 + 백테스트 실행 → 결과 dict 반환.

    순수 실행 함수 — jobs 워커(Supabase)와 FastAPI 인메모리 잡이 공용한다.
    스크리닝 통과 종목이 없으면 ValueError.
    """
    from backtest import BacktestEngine, StockScreener

    screener = StockScreener(universe_per_market=params["universe_n"])
    selected = screener.screen(markets=params["markets"], top_n=params["top_n"]) or []
    if not selected:
        raise ValueError("스크리닝 통과 종목이 없습니다.")

    name_map = {s["ticker"]: s["name"] for s in selected}
    log_buf = io.StringIO()
    engine = BacktestEngine(
        initial_capital=params["initial_capital"],
        start_date=params["start_date"],
        end_date=params["end_date"],
        deposit_schedule=params["deposit_schedule"],
        ticker_name_map=name_map,
    )
    with contextlib.redirect_stdout(log_buf):
        engine.run()

    return {
        "metrics": _metrics(engine.equity_curve, engine.trade_log),
        "equity_curve": engine.equity_curve,
        "trade_log": engine.trade_log,
        "selected_stocks": [
            {"ticker": s["ticker"], "name": s["name"], "score": s["score"],
             "close": s["close"], "volume": s["volume"]}
            for s in selected
        ],
        "benchmark": _fetch_benchmark(params.get("benchmark_ticker"), engine.equity_curve),
        "benchmark_label": params.get("benchmark_label", ""),
        "log_text": log_buf.getvalue(),
    }


def _run(job_id: str, params: dict) -> None:
    try:
        result = run_backtest(params)
        with _LOCK:
            _JOBS[job_id].update(status="done", result=result,
                                 finished_at=datetime.now(timezone.utc).isoformat())
    except ValueError as e:
        with _LOCK:
            _JOBS[job_id].update(status="error", error=str(e))
    except Exception as e:
        logger.exception("[backtest] job %s 실패", job_id)
        with _LOCK:
            _JOBS[job_id].update(status="error", error=str(e),
                                 finished_at=datetime.now(timezone.utc).isoformat())


def start_job(params: dict) -> str:
    job_id = uuid.uuid4().hex
    with _LOCK:
        _JOBS[job_id] = {"status": "running", "result": None, "error": "",
                         "started_at": datetime.now(timezone.utc).isoformat()}
    threading.Thread(target=_run, args=(job_id, params), daemon=True).start()
    return job_id


def get_job(job_id: str) -> dict | None:
    with _LOCK:
        job = _JOBS.get(job_id)
        return dict(job) if job else None
