"""stock_lists.py — 종목 사전 로딩/검색 (app.py 의 _krx_stocks/_us_stocks/_etf_stocks 대체).

기존 @st.cache_data(ttl=86400) 래퍼를 ttl_cache 로 치환한다.
"""
from __future__ import annotations

from app import bootstrap  # noqa: F401
from app.core.cache import ttl_cache

from src.utils import (
    get_krx_stock_list,
    get_krx_etf_list,
    get_us_stock_list,
    get_top_kospi_stocks,
    get_top_kosdaq_stocks,
    get_top_us_stocks,
    get_top_nasdaq_stocks,
    resolve_ticker,
    is_etf_ticker,
)

_DAY = 86400


@ttl_cache(ttl=_DAY)
def krx_stocks() -> dict:
    return get_krx_stock_list() or {}


@ttl_cache(ttl=3600)
def etf_stocks() -> dict:
    return get_krx_etf_list() or {}


@ttl_cache(ttl=_DAY)
def us_stocks() -> dict:
    return get_us_stock_list() or {}


@ttl_cache(ttl=_DAY)
def all_stocks_merged() -> dict:
    """세 종목 사전 합산 (app.py _all_stocks_merged 대응)."""
    result: dict = {}
    result.update(krx_stocks())
    result.update(etf_stocks())
    result.update(us_stocks())
    return result


@ttl_cache(ttl=_DAY)
def ticker_name_map() -> dict[str, str]:
    """ticker → 표시명 (app.py _ticker_name_map 대응)."""
    result: dict[str, str] = {}
    for display, ticker in krx_stocks().items():
        result[ticker] = display.split(" (")[0].strip()
    for display, ticker in etf_stocks().items():
        result[ticker] = display.split(" (")[0].strip()
    for display, ticker in us_stocks().items():
        parts = display.split(" / ")
        result[ticker] = parts[0].strip() if len(parts) > 1 else display.split(" (")[0].strip()
    return result


def top_stocks(market: str) -> dict:
    """추천 탭의 _get_full_stocks 대응."""
    if market == "KOSPI":
        return get_top_kospi_stocks(500)
    if market == "KOSDAQ":
        return get_top_kosdaq_stocks(500)
    if market == "KOSPI + KOSDAQ":
        return {**get_top_kospi_stocks(500), **get_top_kosdaq_stocks(500)}
    if market == "미국 주식 (나스닥)":
        return get_top_nasdaq_stocks(500)
    return get_top_us_stocks(503)


def search(query: str, top_n: int = 10) -> list[dict]:
    """한글/영문/티커 통합 검색 (resolve_ticker)."""
    return resolve_ticker(query, top_n=top_n) or []


def is_etf(ticker: str) -> bool:
    if is_etf_ticker(ticker):
        return True
    if not (ticker.endswith(".KS") or ticker.endswith(".KQ")):
        return False
    code = ticker.replace(".KS", "").replace(".KQ", "").strip().zfill(6)
    values = etf_stocks().values()
    return f"{code}.KS" in values or f"{code}.KQ" in values
