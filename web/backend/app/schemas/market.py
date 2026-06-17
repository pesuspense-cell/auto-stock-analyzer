"""market.py — 시장 현황 탭 스키마."""
from __future__ import annotations

from pydantic import BaseModel


class MoverItem(BaseModel):
    name: str
    ticker: str
    price: float
    change_pct: float


class IndexQuote(BaseModel):
    name: str
    symbol: str
    price: float
    change_pct: float


class ExchangeRate(BaseModel):
    pair: str
    rate: float
    change_pct: float | None = None


class SectorEtfItem(BaseModel):
    country: str           # 국내 / 미국
    tag: str               # 테마 (예: 💻 반도체)
    name: str
    ticker: str
    price_label: str       # "1,234₩" / "$12.34"
    change_pct: float


class MarketOverview(BaseModel):
    indices: list[IndexQuote] = []
    gainers: list[MoverItem] = []
    losers: list[MoverItem] = []


class UsdKrwPoint(BaseModel):
    date: str
    close: float
