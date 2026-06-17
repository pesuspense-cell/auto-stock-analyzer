"""stocks.py — 종목 검색/목록 라우터 (사이드바 대응)."""
from __future__ import annotations

from fastapi import APIRouter, Query
from fastapi.concurrency import run_in_threadpool

from app.schemas.stocks import StockHit, StockSearchResponse
from app.services import stock_lists

router = APIRouter(prefix="/stocks", tags=["stocks"])


@router.get("/search", response_model=StockSearchResponse)
async def search(q: str = Query(min_length=1), limit: int = 10):
    hits = await run_in_threadpool(stock_lists.search, q, limit)
    results = [
        StockHit(
            display=(f"{h.get('name_kr') or h.get('name', '')} ({h['ticker']})"),
            name=h.get("name_kr") or h.get("name", ""),
            ticker=h["ticker"],
        )
        for h in hits if h.get("ticker")
    ]
    return StockSearchResponse(query=q, results=results)


@router.get("/list")
async def stock_list(market: str = "KOSPI"):
    """시장별 상위 종목 사전 (추천/스크리너용)."""
    data = await run_in_threadpool(stock_lists.top_stocks, market)
    return {"market": market, "stocks": data}
