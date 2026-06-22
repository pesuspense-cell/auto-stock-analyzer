"""portfolio_service.py — 포트폴리오 진단 + 매수/매도 지침.

src/portfolio_optimizer 의 섹터 분류 → 시장/섹터 모멘텀 → 리밸런싱 가이드(HHI·조건
매트릭스)를 묶어 보유 종목에 대한 매수/매도/유지 지침을 생성한다.
시장 모멘텀 스캔(scan_market_momentum)은 전 사용자 공통이라 10분 캐시한다.
"""
from __future__ import annotations

import logging

from app import bootstrap  # noqa: F401
from app.core.cache import ttl_cache

from src.portfolio_optimizer import (
    classify_sectors,
    scan_market_momentum,
    build_rebalancing_guide,
)

logger = logging.getLogger(__name__)


@ttl_cache(ttl=600)
def market_momentum() -> dict:
    """KOSPI/KOSDAQ 추세 + 섹터 ETF 모멘텀 (10분 캐시 — 사용자 무관)."""
    try:
        return scan_market_momentum() or {}
    except Exception as e:
        logger.warning("[portfolio] 시장 모멘텀 스캔 실패: %s", e)
        return {"market_status": "데이터 없음", "sector_scores": []}


def analyze(items: list[dict], prices: dict[str, float], name_map: dict[str, str]) -> dict:
    """보유 종목 진단 + 리밸런싱/매매 지침.

    items    : [{"ticker", "avg_price", "quantity"}, ...]
    prices   : {ticker: current_price}
    name_map : {ticker: 표시명}
    """
    if not items:
        return {"empty": True, "sectors": {}, "guide": {}, "item_values": [], "total_value": 0.0}

    sector_data = classify_sectors(items, prices)
    momentum = market_momentum()
    guide = build_rebalancing_guide(sector_data, momentum, name_map)

    return {
        "empty": False,
        "total_value": sector_data.get("total_value", 0.0),
        "sectors": sector_data.get("sectors", {}),
        "item_values": sector_data.get("item_values", []),
        "unknown_tickers": sector_data.get("unknown_tickers", []),
        "market_status": momentum.get("market_status", ""),
        "kospi_above_ma": momentum.get("kospi_above_ma"),
        "kosdaq_above_ma": momentum.get("kosdaq_above_ma"),
        "guide": guide,
    }
