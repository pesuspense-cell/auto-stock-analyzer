"""market_service.py — 시장 현황 탭 데이터 (app.py _movers/_full_movers/_rates 대체)."""
from __future__ import annotations

import pandas as pd

from app import bootstrap  # noqa: F401
from app.core.cache import ttl_cache

from src.utils import (
    get_full_market_movers,
    get_exchange_rates,
    INDICES,
)
from src.indicators import get_stock_data  # noqa: F401  (필요 시 확장용)


def _movers_from_df(df: pd.DataFrame) -> list[dict]:
    if df is None or df.empty:
        return []
    out = []
    for _, r in df.iterrows():
        out.append({
            "name": str(r.get("종목명", "")),
            "ticker": str(r.get("티커", "")),
            "price": float(r.get("현재가", 0) or 0),
            "change_pct": float(r.get("등락률(%)", 0) or 0),
        })
    return out


@ttl_cache(ttl=300)
def full_movers(top_n: int = 10) -> dict:
    """KOSPI+KOSDAQ 상승/하락 상위 (get_full_market_movers)."""
    gainers_df, losers_df = get_full_market_movers(top_n=top_n)
    return {
        "gainers": _movers_from_df(gainers_df),
        "losers": _movers_from_df(losers_df),
    }


@ttl_cache(ttl=300)
def exchange_rates() -> list[dict]:
    """환율 (get_exchange_rates) → 평탄화."""
    raw = get_exchange_rates() or {}
    return [
        {"pair": name, "rate": float(v.get("rate", 0)), "change_pct": float(v.get("change", 0))}
        for name, v in raw.items()
    ]


def overview(top_n: int = 10) -> dict:
    fm = full_movers(top_n)
    return {
        "indices": [],  # INDICES 심볼 가격은 routers/market 에서 _index_data 로 보강
        "gainers": fm["gainers"],
        "losers": fm["losers"],
    }


def index_symbols() -> dict:
    return dict(INDICES)
