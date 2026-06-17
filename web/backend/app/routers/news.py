"""news.py — 뉴스 & 관련 종목 라우터."""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool

from app.core.config import settings
from app.schemas.news import (
    NewsResponse, SummarizeRequest, SummarizeResponse,
)
from app.services import news_service, stock_lists

router = APIRouter(prefix="/news", tags=["news"])


@router.get("/{ticker}", response_model=NewsResponse)
async def get_news(ticker: str):
    cname = stock_lists.ticker_name_map().get(ticker, "")
    data = await run_in_threadpool(
        news_service.news, ticker,
        settings.gemini_api_key, settings.groq_api_key, cname,
    )
    # etf_meta 가 빈 dict 이면 None 으로 (비-ETF)
    if not data.get("etf_meta"):
        data["etf_meta"] = None
    return NewsResponse(**data)


@router.post("/summarize", response_model=SummarizeResponse)
async def summarize(body: SummarizeRequest):
    data = await run_in_threadpool(
        news_service.summarize, body.title, body.link, body.ticker,
        settings.gemini_api_key, settings.groq_api_key,
    )
    return SummarizeResponse(**data)
