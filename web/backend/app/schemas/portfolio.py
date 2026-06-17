"""portfolio.py — 포트폴리오/매매이력 스키마."""
from __future__ import annotations

from pydantic import BaseModel, Field


class PortfolioAddRequest(BaseModel):
    ticker: str
    avg_price: float = Field(gt=0)
    quantity: float = Field(default=1.0, gt=0)


class SellRequest(BaseModel):
    sell_price: float = Field(gt=0)
    quantity: float | None = Field(default=None, gt=0)  # None → 전량 매도


class PortfolioItem(BaseModel):
    id: int
    ticker: str
    avg_price: float
    quantity: float
    added_at: str
    # 실시간 평가 (서비스에서 주입)
    name: str | None = None
    current_price: float | None = None
    return_pct: float | None = None
    eval_amount: float | None = None


class TradeItem(BaseModel):
    id: int
    ticker: str
    buy_price: float
    sell_price: float
    quantity: float
    net_profit: float
    return_rate: float
    traded_at: str


class SellResponse(BaseModel):
    ok: bool
    net_profit: float | None = None
    return_rate: float | None = None
    error: str | None = None
