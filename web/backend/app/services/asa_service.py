"""asa_service.py — ASA 추천 (live_screener 인프로세스 실행).

render_asa_tab 의 핵심 로직 이식:
  - 보유 종목 → ATR 기반 SL/TP 자동 산출 → positions dict
  - live_screener.MY_CURRENT_BALANCE 교체 후 main() 실행, stdout 캡처

live_screener.main() 은 수 분 소요되므로 백그라운드 스레드 + 인메모리 잡 스토어로
비동기 실행하고, 프론트는 job_id 로 폴링한다.
"""
from __future__ import annotations

import io
import logging
import threading
import uuid
from contextlib import redirect_stdout
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

from app import bootstrap  # noqa: F401
from app.services import stock_lists

logger = logging.getLogger(__name__)

# ── 인메모리 잡 스토어 ───────────────────────────────────────────────
_JOBS: dict[str, dict] = {}
_LOCK = threading.Lock()


def _is_us(ticker: str) -> bool:
    code = ticker.split(".")[0]
    return not (len(code) == 6 and code[0].isdigit())


def _name(ticker: str) -> str:
    nm = stock_lists.ticker_name_map().get(ticker)
    if nm:
        return nm
    code = ticker.split(".")[0]
    tm = stock_lists.ticker_name_map()
    return tm.get(f"{code}.KS") or tm.get(f"{code}.KQ") or code


def build_positions(items: list[dict]) -> dict:
    """보유 종목 → ATR(14, 2.5x) 기반 SL / 25% TP 포지션 dict (render_asa_tab 이식)."""
    positions: dict = {}
    if not items:
        return positions

    tickers = [str(it.get("ticker", "")) for it in items]
    sl_tp: dict[str, tuple[float, float]] = {}
    try:
        query = tickers if len(tickers) > 1 else tickers[0]
        raw = yf.download(query, period="60d", auto_adjust=True, progress=False)
        for t in tickers:
            try:
                avg = float(next((x.get("avg_price") or 0) for x in items if str(x.get("ticker")) == t))
                tdf = raw.xs(t, axis=1, level=1).dropna(how="all") if isinstance(raw.columns, pd.MultiIndex) else raw.dropna(how="all")
                if len(tdf) >= 14 and "High" in tdf.columns:
                    tr = pd.concat([
                        tdf["High"] - tdf["Low"],
                        (tdf["High"] - tdf["Close"].shift(1)).abs(),
                        (tdf["Low"] - tdf["Close"].shift(1)).abs(),
                    ], axis=1).max(axis=1)
                    atr = float(tr.ewm(alpha=1.0 / 14, adjust=False, min_periods=14).mean().iloc[-1])
                    sl = max(avg - 2.5 * atr, avg * 0.92)
                else:
                    sl = avg * 0.92
                sl_tp[t] = (round(sl), round(avg * 1.25))
            except Exception:
                avg_f = float(next(((x.get("avg_price") or 0) for x in items if str(x.get("ticker")) == t), 0))
                sl_tp[t] = (round(avg_f * 0.92), round(avg_f * 1.25))
    except Exception:
        for it in items:
            avg_f = float(it.get("avg_price") or 0)
            sl_tp[str(it.get("ticker", ""))] = (round(avg_f * 0.92), round(avg_f * 1.25))

    for it in items:
        t = str(it.get("ticker", ""))
        avg = float(it.get("avg_price") or 0)
        qty = int(float(it.get("quantity") or 0))
        sl, tp = sl_tp.get(t, (round(avg * 0.92), round(avg * 1.25)))
        base = _name(t)
        positions[t] = {
            "name": f"{base} (USD)" if _is_us(t) else base,
            "entry_price": avg, "quantity": qty,
            "sl": float(sl), "tp": float(tp),
        }
    return positions


def run_asa(cash: float, positions: dict) -> str:
    """ASA(live_screener) 인프로세스 실행 → stdout 텍스트 반환.

    순수 실행 함수 — jobs 워커(Supabase)와 FastAPI 인메모리 잡이 공용한다.
    """
    import live_screener as ls
    ls.MY_CURRENT_BALANCE = {"cash": float(cash), "positions": positions}

    class _NoopThread:
        def join(self):
            pass

    ls.show_popup_after_close = lambda: _NoopThread()  # type: ignore[attr-defined]

    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            ls.main()
    except SystemExit:
        pass
    return buf.getvalue() or "(출력 없음)"


def _run(job_id: str, cash: float, positions: dict) -> None:
    try:
        output = run_asa(cash, positions)
        with _LOCK:
            _JOBS[job_id].update(
                status="done", output=output,
                finished_at=datetime.now(timezone.utc).isoformat(),
            )
    except Exception as e:
        logger.exception("[asa] job %s 실패", job_id)
        with _LOCK:
            _JOBS[job_id].update(status="error", error=str(e),
                                 finished_at=datetime.now(timezone.utc).isoformat())


def start_job(cash: float, items: list[dict]) -> str:
    """ASA 분석 잡 시작 → job_id 반환."""
    job_id = uuid.uuid4().hex
    with _LOCK:
        _JOBS[job_id] = {
            "status": "running", "output": "", "error": "",
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
    positions = build_positions(items)
    threading.Thread(target=_run, args=(job_id, cash, positions), daemon=True).start()
    return job_id


def get_job(job_id: str) -> dict | None:
    with _LOCK:
        job = _JOBS.get(job_id)
        return dict(job) if job else None
