"""common.py — 공통 응답 래퍼 및 열거형."""
from __future__ import annotations

from enum import Enum
from typing import Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class Market(str, Enum):
    KOSPI = "KOSPI"
    KOSDAQ = "KOSDAQ"
    KOSPI_KOSDAQ = "KOSPI + KOSDAQ"
    US_NASDAQ = "미국 주식 (나스닥)"
    US_SP500 = "미국 주식 (S&P500)"


class Period(str, Enum):
    """yfinance 기간 — 기존 사이드바 옵션과 동일."""
    p1mo = "1mo"
    p3mo = "3mo"
    p6mo = "6mo"
    p1y = "1y"
    p2y = "2y"


class RiskProfile(str, Enum):
    conservative = "보수형"
    neutral = "중립형"
    aggressive = "공격형"


class ApiResponse(BaseModel, Generic[T]):
    """일관된 응답 봉투."""
    ok: bool = True
    data: T | None = None
    error: str | None = None


class OkResponse(BaseModel):
    ok: bool = True
    error: str | None = None
