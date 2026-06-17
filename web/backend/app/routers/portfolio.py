"""portfolio.py — 포트폴리오/매매 라우터 (🔒 Bearer 인증).

src/database.py 의 CRUD 를 그대로 사용하고, 실시간가·종목명으로 항목을 보강한다.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool

from app import bootstrap  # noqa: F401
from app.core.security import get_current_user
from app.schemas.common import OkResponse
from app.schemas.portfolio import (
    PortfolioAddRequest, PortfolioItem, SellRequest, SellResponse, TradeItem,
)
from app.services import analysis_service, stock_lists

from src.database import (
    upsert_portfolio, get_portfolio, delete_portfolio_item,
    sell_item, get_trade_history,
)

router = APIRouter(prefix="/portfolio", tags=["portfolio"])


def _enrich(item: dict) -> dict:
    """보유 항목에 종목명·현재가·수익률 주입."""
    ticker = item["ticker"]
    name = stock_lists.ticker_name_map().get(ticker, ticker)
    price = analysis_service.realtime_price(ticker).get("price", 0.0)
    avg = float(item["avg_price"])
    qty = float(item["quantity"])
    return {
        **item,
        "name": name,
        "current_price": price or None,
        "return_pct": ((price / avg - 1) * 100) if (price and avg) else None,
        "eval_amount": (price * qty) if price else None,
    }


@router.get("", response_model=list[PortfolioItem])
async def list_portfolio(user: dict = Depends(get_current_user)):
    rows = await run_in_threadpool(get_portfolio, user["id"])
    enriched = await run_in_threadpool(lambda: [_enrich(r) for r in rows])
    return [PortfolioItem(**r) for r in enriched]


@router.post("", response_model=OkResponse)
async def add(body: PortfolioAddRequest, user: dict = Depends(get_current_user)):
    await run_in_threadpool(
        upsert_portfolio, user["id"], body.ticker, body.avg_price, body.quantity
    )
    return OkResponse(ok=True)


@router.delete("/{item_id}", response_model=OkResponse)
async def delete(item_id: int, user: dict = Depends(get_current_user)):
    result = await run_in_threadpool(delete_portfolio_item, item_id, user["id"])
    if not result.get("ok"):
        raise HTTPException(status.HTTP_404_NOT_FOUND, result.get("error", "삭제 실패"))
    return OkResponse(ok=True)


@router.post("/{item_id}/sell", response_model=SellResponse)
async def sell(item_id: int, body: SellRequest, user: dict = Depends(get_current_user)):
    result = await run_in_threadpool(
        sell_item, user["id"], item_id, body.sell_price, body.quantity
    )
    if not result.get("ok"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, result.get("error", "매도 실패"))
    return SellResponse(**result)


@router.get("/trades", response_model=list[TradeItem])
async def trades(user: dict = Depends(get_current_user)):
    rows = await run_in_threadpool(get_trade_history, user["id"])
    return [TradeItem(**r) for r in rows]
