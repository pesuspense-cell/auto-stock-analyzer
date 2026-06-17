"""news_service.py — 뉴스 & 관련 종목 탭 데이터.

ui/layouts.py 의 render_news_tab 데이터 수집부(st.* 제거)를 이식.
KR/US/ETF 분기에 따라 뉴스 수집 → 감성 분석 → 섹터 성과를 병렬 처리한다.
"""
from __future__ import annotations

import concurrent.futures
import logging
from datetime import datetime

import yfinance as yf

from app import bootstrap  # noqa: F401
from app.core.cache import ttl_cache
from app.services import stock_lists

from src.news_logic import (
    get_naver_news,
    analyze_news_sentiment_keywords,
    analyze_news_sentiment_llm,
    summarize_article_llm,
    get_related_sector_performance,
    get_etf_news_with_holdings,
    analyze_etf_news_sentiment,
    fetch_naver_news_fast,
)
from src.fundamental import get_etf_fundamental_data

logger = logging.getLogger(__name__)


def _naver(ticker: str) -> list[dict]:
    try:
        items = fetch_naver_news_fast(ticker, max_items=12)
        if items:
            return items
    except Exception:
        pass
    return get_naver_news(ticker, max_items=12) or []


def _us_news(ticker: str) -> list[dict]:
    out: list[dict] = []
    try:
        for it in (yf.Ticker(ticker).news or [])[:10]:
            c = it.get("content", it)
            ts_raw = c.get("pubDate", it.get("providerPublishTime", ""))
            out.append({
                "title": c.get("title", it.get("title", "제목 없음")),
                "link": (c.get("canonicalUrl", {}).get("url") or it.get("link", "#")),
                "publisher": (c.get("provider", {}).get("displayName") or it.get("publisher", "")),
                "pub_date": (
                    datetime.fromtimestamp(ts_raw).strftime("%Y-%m-%d %H:%M")
                    if isinstance(ts_raw, (int, float)) else str(ts_raw)
                ),
            })
    except Exception as e:
        logger.warning("[us_news] %s: %s", ticker, e)
    return out


@ttl_cache(ttl=300)
def sector_perf(ticker: str) -> dict:
    try:
        return get_related_sector_performance(ticker) or {}
    except Exception:
        return {}


@ttl_cache(ttl=600)
def news(ticker: str, gemini: str = "", groq: str = "", company_name: str = "") -> dict:
    """뉴스 + 감성 + 섹터 성과 통합 (render_news_tab 이식)."""
    is_etf = stock_lists.is_etf(ticker)
    is_kr = ticker.endswith((".KS", ".KQ"))
    cname = company_name if company_name and company_name != ticker else ""

    etf_meta: dict = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        f_sector = pool.submit(sector_perf, ticker)

        if is_etf:
            etf_data = get_etf_fundamental_data(ticker) or {}
            raw_news = get_etf_news_with_holdings(ticker, etf_data, max_items=15) or []
            f_sent = pool.submit(analyze_etf_news_sentiment, ticker, etf_data, raw_news) if raw_news else None
            etf_meta = {
                "sector": etf_data.get("sector", ""),
                "holdings": [h.get("name", h.get("ticker", ""))
                             for h in etf_data.get("top_holdings", [])[:5]],
            }
        else:
            raw_news = _naver(ticker) if is_kr else _us_news(ticker)

            def _do_sent(rn=raw_news):
                if gemini or groq:
                    return analyze_news_sentiment_llm(rn, ticker, gemini, groq, cname)
                return analyze_news_sentiment_keywords(rn, ticker, cname)

            f_sent = pool.submit(_do_sent) if raw_news else None

        sec = f_sector.result() or {}
        sent = f_sent.result() if f_sent else {}

    return {
        "ticker": ticker,
        "is_etf": is_etf,
        "articles": raw_news,
        "sentiment": sent or {},
        "sector_performance": sec,
        "etf_meta": etf_meta,
    }


def summarize(title: str, link: str, ticker: str, gemini: str = "", groq: str = "") -> dict:
    """단일 기사 AI 요약 (summarize_article_llm)."""
    try:
        return summarize_article_llm(title, link, ticker, gemini, groq) or {}
    except Exception as e:
        logger.warning("[summarize] %s: %s", link, e)
        return {"summary": "요약을 생성할 수 없습니다.", "sentiment": "중립",
                "score": 0.0, "key_points": [], "investment_implication": ""}
