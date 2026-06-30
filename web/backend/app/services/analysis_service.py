"""analysis_service.py — 차트 분석 탭 종합 파이프라인.

app.py 의 `elif _data_ready:` 블록(주가·재무 병렬 로딩 → 신호·뉴스 →
하이브리드 신호 → 리스크 조정)을 streamlit 의존 없이 그대로 옮긴 것.
무거운 동기 호출이므로 라우터에서 run_in_threadpool 로 감싸 실행한다.
"""
from __future__ import annotations

import concurrent.futures
import logging
import os
import time
from datetime import datetime, timezone, timedelta

import httpx
import pandas as pd
import yfinance as yf

from app import bootstrap  # noqa: F401
from app.core.cache import ttl_cache

from src.indicators import (
    get_stock_data,
    generate_signals,
    get_advanced_analysis,
    calculate_expected_return,
    get_enhanced_hybrid_signal,
    check_volume_anomaly,
    check_dead_time,
    check_breakout_signal,
    adjust_risk_conservative,
)
from src.fundamental import get_fundamental_data, calculate_fundamental_score
from src.utils import _flatten_columns

logger = logging.getLogger(__name__)

_KST = timezone(timedelta(hours=9))

# yfinance 차단 환경(Render) 게이트 — stock_ai._YF_DISABLED 와 동일 의미.
# 켜지면 bench_returns 가 막힐 yf 호출 대신 FDR 로 지수를 받아 cold 분석 지연을 줄인다.
_YF_DISABLED = os.getenv("ASA_DISABLE_YFINANCE", "").strip().lower() in ("1", "true", "yes", "on")


def _now_kst() -> datetime:
    return datetime.now(_KST)


# ── 캐시 래퍼 (app.py @st.cache_data 대응) ──────────────────────────

@ttl_cache(ttl=3600)
def stock_data(ticker: str, period: str) -> pd.DataFrame:
    return get_stock_data(ticker, period)


@ttl_cache(ttl=3600)
def fundamental(ticker: str) -> dict:
    return get_fundamental_data(ticker)


@ttl_cache(ttl=300)
def dead_time(ticker: str) -> dict:
    return check_dead_time(ticker)


@ttl_cache(ttl=3600)
def bench_returns(ticker: str) -> pd.Series:
    is_kr = ticker.endswith((".KS", ".KQ"))
    # yfinance 차단 환경: ^KS11/^GSPC yf.download 가 타임아웃까지 헛대기하므로 FDR 로 직행.
    if _YF_DISABLED:
        code = "KS11" if is_kr else "US500"   # FDR 지수 코드(KOSPI / S&P500)
        try:
            import FinanceDataReader as fdr
            from datetime import timedelta as _td
            start = (datetime.now() - _td(days=240)).strftime("%Y-%m-%d")
            d = fdr.DataReader(code, start)
            if d is not None and not d.empty and "Close" in d.columns:
                return d["Close"].pct_change().dropna()
        except Exception as e:
            logger.warning("[bench] FDR %s 실패: %s", code, e)
        return pd.Series(dtype=float)
    sym = "^KS11" if is_kr else "^GSPC"
    try:
        d = _flatten_columns(yf.download(sym, period="6mo", auto_adjust=True, progress=False))
        return d["Close"].pct_change().dropna()
    except Exception:
        return pd.Series(dtype=float)


# 네이버 실시간 폴링(무인증, delayTime=0) — Yahoo 의 ~15분 지연 대체용.
_NAVER_POLL = "https://polling.finance.naver.com/api/realtime/domestic/stock"
_NAVER_HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://m.stock.naver.com/"}


def _naver_realtime(ticker: str) -> dict | None:
    """국내(.KS/.KQ) 실시간 현재가 — 네이버 폴링. 실패/비국내면 None."""
    if not ticker.endswith((".KS", ".KQ")):
        return None
    code = ticker.split(".")[0]
    try:
        r = httpx.get(f"{_NAVER_POLL}/{code}", headers=_NAVER_HEADERS, timeout=5.0)
        if r.status_code != 200:
            return None
        datas = r.json().get("datas") or []
        d = datas[0] if datas else None
        if not d:
            return None
        price = float(str(d.get("closePrice", "")).replace(",", "") or 0)
        if price <= 0:
            return None
        is_open = d.get("marketStatus") == "OPEN"
        return {
            "price": price, "ts": _now_kst().strftime("%H:%M:%S"),
            "is_realtime": is_open, "stale": not is_open,
            "stale_msg": "" if is_open else "장 마감 후 종가입니다.",
        }
    except Exception as e:
        logger.warning("[naver-rt] %s: %s", ticker, e)
        return None


@ttl_cache(ttl=60)
def realtime_price(ticker: str) -> dict:
    """app.py _realtime_price_1m 의 핵심 로직 이식. 국내는 네이버 실시간 우선."""
    now_k = _now_kst()
    ts = now_k.strftime("%H:%M:%S")
    is_kr = ticker.endswith((".KS", ".KQ"))

    # 국내(.KS/.KQ)는 네이버 실시간(지연 0) 우선 — 실패 시 아래 yfinance 폴백.
    if is_kr:
        nv = _naver_realtime(ticker)
        if nv:
            return nv

    after_close = is_kr and (now_k.hour * 60 + now_k.minute) >= (15 * 60 + 30)

    try:
        fi = yf.Ticker(ticker).fast_info
        p = float(fi.last_price)
        if p > 0:
            return {"price": p, "ts": ts, "is_realtime": not after_close,
                    "stale": after_close,
                    "stale_msg": "장 마감 후 종가입니다." if after_close else ""}
    except Exception:
        pass

    try:
        df = _flatten_columns(
            yf.download(ticker, period="1d", interval="1m", auto_adjust=True, progress=False)
        )
        if not df.empty and "Close" in df.columns:
            close_s = df["Close"].dropna()
            if not close_s.empty:
                p = float(close_s.iloc[-1])
                if p > 0:
                    return {"price": p, "ts": ts, "is_realtime": not after_close,
                            "stale": after_close, "stale_msg": ""}
    except Exception:
        pass

    try:
        d = _flatten_columns(yf.download(ticker, period="2d", auto_adjust=True, progress=False))
        if not d.empty and "Close" in d.columns:
            p = float(d["Close"].dropna().iloc[-1])
            if p > 0:
                return {"price": p, "ts": ts, "is_realtime": False, "stale": True,
                        "stale_msg": "장이 열리지 않은 상태입니다. 가장 최근 종가를 표시합니다."}
    except Exception:
        pass

    return {"price": 0.0, "ts": ts, "is_realtime": False, "stale": False, "stale_msg": ""}


# ── OHLC (차트용) ───────────────────────────────────────────────────

def ohlc(ticker: str, period: str) -> list[dict]:
    df = stock_data(ticker, period)
    if df is None or df.empty or "Close" not in df.columns:
        return []
    out = []
    for idx, row in df.iterrows():
        try:
            out.append({
                "date": idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx),
                "open": float(row.get("Open", 0) or 0),
                "high": float(row.get("High", 0) or 0),
                "low": float(row.get("Low", 0) or 0),
                "close": float(row.get("Close", 0) or 0),
                "volume": float(row.get("Volume", 0) or 0),
            })
        except Exception:
            continue
    return out


# ── 종합 분석 파이프라인 (app.py _data_ready 블록 이식) ──────────────

def _empty_advanced() -> dict:
    return {"trend_score": 50.0, "momentum_score": 50.0, "volume_score": 50.0,
            "divergence": {}, "zscore": None, "vpvr": {}, "ichimoku": {}, "summary_items": []}


def _news_sentiment(ticker: str, use_llm: bool, gemini: str, groq: str) -> dict:
    """뉴스 감성 — LLM 우선, 실패 시 키워드 폴백 (app.py _news_sentiment_* 대응)."""
    try:
        if use_llm and (gemini or groq):
            from src.news_logic import analyze_news_fast
            return analyze_news_fast(
                ticker=ticker, company_name="",
                api_key=gemini, groq_api_key=groq,
                max_news=12, deep_n=5,
            ) or {}
        from src.news_logic import get_naver_news, analyze_news_sentiment_keywords
        news = get_naver_news(ticker, max_items=12) or []
        return analyze_news_sentiment_keywords(news, ticker, "") or {}
    except Exception as e:
        logger.warning("[news] %s: %s", ticker, e)
        return {}


def analyze(ticker: str, period: str, use_llm: bool = False,
            gemini: str = "", groq: str = "", sname: str = "") -> dict:
    """동기 종합 분석. 라우터에서 run_in_threadpool 로 호출."""
    # 단계별 소요시간 진단 — cold 분석이 느린 지점을 워커 로그로 못박는다(t_* 초). 캐시 warm 이면
    # 각 단계가 0초에 가깝다(@ttl_cache 적중). 비정상적으로 큰 단계가 그 시점의 병목.
    _t = time.perf_counter()
    _timings: dict[str, float] = {}

    def _lap(name: str) -> None:
        nonlocal _t
        now = time.perf_counter()
        _timings[name] = now - _t
        _t = now

    data = stock_data(ticker, period)
    _lap("stock_data")
    fund_info = fundamental(ticker)
    _lap("fundamental")

    if data is None or data.empty or "Close" not in data.columns:
        raise ValueError(f"'{ticker}' 데이터를 불러올 수 없습니다.")

    close_raw = data["Close"]
    close = close_raw.iloc[:, 0] if isinstance(close_raw, pd.DataFrame) else close_raw

    # 신호·재무 + 뉴스 병렬 계산
    def _compute():
        try:
            va = check_volume_anomaly(data)
            sig = generate_signals(data)
            if va.get("is_halted"):
                sig = {"score": 0, "label": "거래 정지/주의", "badge": "⛔",
                       "reasons": [va.get("reason", "")], "_halted": True}
            adv = get_advanced_analysis(data)
            exp = calculate_expected_return(data, sig, ticker=ticker,
                                            benchmark_returns=bench_returns(ticker))
            last = float(close.iloc[-1]) if not close.empty else 0.0
            fsd = calculate_fundamental_score(fund_info, last)
            return va, sig, adv, exp, fsd
        except Exception as e:
            logger.warning("[compute] %s: %s", ticker, e)
            return {}, {"score": 0, "label": "분석 오류", "badge": "⚠️", "reasons": []}, {}, {}, {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        f_comp = pool.submit(_compute)
        f_news = pool.submit(_news_sentiment, ticker, use_llm, gemini, groq)
        vol_anomaly, signals, advanced, expected, fund_score_data = f_comp.result()
        news_result = f_news.result()
    _lap("compute+news")

    news_score = news_result.get("score", 0.0) if isinstance(news_result, dict) else 0.0
    tech_score = signals.get("score", 0) if isinstance(signals, dict) else 0
    dt = dead_time(ticker)
    breakout = check_breakout_signal(data)
    _lap("deadtime+breakout")

    # 점수 정규화 (app.py 와 동일)
    tech5 = max(-5, min(5, round(tech_score / 2)))
    news1 = max(-1.0, min(1.0, news_score / 5.0))
    raw_fund = fund_score_data.get("fund_score", 0)
    fund_100 = max(0, min(100, int((raw_fund + 8) / 16 * 100)))
    rsi = (float(data["RSI"].iloc[-1])
           if "RSI" in data.columns and not data["RSI"].isna().iloc[-1] else 0.0)

    hybrid = get_enhanced_hybrid_signal(
        tech_score=tech5, news_score=news1, fund_score=fund_100,
        vol_anomaly=vol_anomaly, dead_time=dt, breakout=breakout,
        advanced=advanced, period=period, rsi=rsi,
    )

    if expected:
        exp_ret = expected.get("expected_return_pct", 0.0)
        sharpe = expected.get("sharpe", 1.0)
        if exp_ret >= 50.0 and sharpe < 0.5:
            hybrid.setdefault("warnings", []).append(
                "⚠️ 샤프지수 주의: 예상 수익률이 높지만 샤프지수가 낮아 리스크 대비 수익이 불안정합니다."
            )
            if hybrid.get("label") in ("강력 매수", "매수 추천"):
                hybrid["label"], hybrid["badge"] = "주의", "🟡"

    risk_adj = adjust_risk_conservative(expected) if expected else {}
    realtime = realtime_price(ticker)
    _lap("hybrid+realtime")

    total = sum(_timings.values())
    logger.info(
        "[analyze] %s %s total=%.1fs | %s",
        ticker, period, total,
        " ".join(f"{k}={v:.1f}" for k, v in _timings.items()),
    )

    return {
        "ticker": ticker,
        "sname": sname,
        "period": period,
        "data_ready": True,
        "signals": signals,
        "hybrid": hybrid,
        "realtime": realtime,
        "advanced": advanced or _empty_advanced(),
        "expected": expected or None,
        "risk_adj": risk_adj,
        "fund_score_data": fund_score_data,
        "fund_info": fund_info,
        "news_result": news_result,
        "dead_time": dt,
        "breakout": breakout,
        "vol_anomaly": vol_anomaly,
        "tech_score": float(tech_score),
        "news_score": float(news_score),
    }
