"""market.py — 시장 현황 라우터."""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool

from app.schemas.market import MarketOverview, ExchangeRate
from app.services import market_service

router = APIRouter(prefix="/market", tags=["market"])


@router.get("/overview", response_model=MarketOverview)
async def overview(top_n: int = 10):
    data = await run_in_threadpool(market_service.overview, top_n)
    return MarketOverview(**data)


@router.get("/rates", response_model=list[ExchangeRate])
async def rates():
    data = await run_in_threadpool(market_service.exchange_rates)
    return [ExchangeRate(**r) for r in data]
