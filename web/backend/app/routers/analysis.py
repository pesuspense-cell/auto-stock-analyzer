"""analysis.py — 차트 분석 라우터 (앱의 핵심 파이프라인)."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.concurrency import run_in_threadpool

from app.core.config import settings
from app.schemas.analysis import (
    AnalysisRequest, AnalysisResponse, OhlcPoint, RealtimePrice,
)
from app.services import analysis_service, stock_lists

router = APIRouter(prefix="/analysis", tags=["analysis"])


@router.post("", response_model=AnalysisResponse)
async def analyze(body: AnalysisRequest):
    sname = stock_lists.ticker_name_map().get(body.ticker, body.ticker)
    try:
        result = await run_in_threadpool(
            analysis_service.analyze,
            body.ticker, body.period.value, body.use_llm,
            settings.gemini_api_key, settings.groq_api_key, sname,
        )
    except ValueError as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(e))
    return AnalysisResponse(**result)


@router.get("/{ticker}/ohlc", response_model=list[OhlcPoint])
async def ohlc(ticker: str, period: str = Query("6mo")):
    data = await run_in_threadpool(analysis_service.ohlc, ticker, period)
    return [OhlcPoint(**p) for p in data]


@router.get("/{ticker}/realtime", response_model=RealtimePrice)
async def realtime(ticker: str):
    data = await run_in_threadpool(analysis_service.realtime_price, ticker)
    return RealtimePrice(**data)
