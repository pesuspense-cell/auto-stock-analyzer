"""fundamental.py — 펀더멘털 & 기관 라우터."""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool

from app.core.config import settings
from app.schemas.fundamental import AiReportResponse, FundamentalResponse
from app.services import fundamental_service, stock_lists

router = APIRouter(prefix="/fundamental", tags=["fundamental"])


@router.get("/{ticker}", response_model=FundamentalResponse)
async def get_fundamental(ticker: str):
    data = await run_in_threadpool(fundamental_service.fundamental, ticker)
    return FundamentalResponse(**data)


@router.post("/{ticker}/ai-report", response_model=AiReportResponse)
async def ai_report(ticker: str, use_llm: bool = True):
    sname = stock_lists.ticker_name_map().get(ticker, ticker)
    data = await run_in_threadpool(
        fundamental_service.ai_report, ticker,
        settings.gemini_api_key, settings.groq_api_key, use_llm, sname,
    )
    return AiReportResponse(**data)
