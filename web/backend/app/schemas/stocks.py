"""stocks.py — 종목 검색/목록 스키마."""
from __future__ import annotations

from pydantic import BaseModel


class StockHit(BaseModel):
    display: str           # 사용자에게 보이는 라벨 (예: "삼성전자 (005930)")
    name: str              # 종목명
    ticker: str            # yfinance 티커 (예: "005930.KS")


class StockSearchResponse(BaseModel):
    query: str
    results: list[StockHit] = []
