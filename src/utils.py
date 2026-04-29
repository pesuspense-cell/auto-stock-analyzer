"""
utils.py - 공통 유틸리티 모듈
종목 사전·지수 상수, 종목 리스트 로딩, 시장 데이터, 추천 분석 등 공용 기능
"""
from stock_ai import (
    # ── 상수 ──────────────────────────────────────────────────────────────────
    KOSPI_STOCKS,
    KOSDAQ_STOCKS,
    US_STOCKS,
    EXCHANGE_PAIRS,
    INDICES,

    # ── 시장 데이터 ───────────────────────────────────────────────────────────
    get_market_movers,
    get_full_market_movers,
    get_exchange_rates,
    get_investor_trading_naver,

    # ── 종목 추천 ──────────────────────────────────────────────────────────────
    get_recommendations,

    # ── 종목 리스트 로딩 ──────────────────────────────────────────────────────
    get_krx_stock_list,
    get_krx_etf_list,
    get_us_stock_list,
    get_top_kospi_stocks,
    get_top_kosdaq_stocks,
    get_top_us_stocks,
    get_top_nasdaq_stocks,

    # ── ETF 유틸리티 ─────────────────────────────────────────────────────────
    is_etf_ticker,
    _ETF_PORTFOLIO_MAP,
)

__all__ = [
    "KOSPI_STOCKS",
    "US_STOCKS",
    "EXCHANGE_PAIRS",
    "INDICES",
    "get_market_movers",
    "get_full_market_movers",
    "get_exchange_rates",
    "get_investor_trading_naver",
    "get_recommendations",
    "get_krx_stock_list",
    "get_krx_etf_list",
    "get_us_stock_list",
    "get_top_kospi_stocks",
    "get_top_kosdaq_stocks",
    "get_top_us_stocks",
    "get_top_nasdaq_stocks",
    "is_etf_ticker",
    "_ETF_PORTFOLIO_MAP",
]
