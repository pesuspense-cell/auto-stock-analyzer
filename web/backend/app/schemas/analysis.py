"""analysis.py — 차트 분석 탭의 종합 분석 결과 스키마.

기존 app.py 의 `_state` dict 와 분석 파이프라인 산출물을 1:1로 옮긴다.
내부 dict 구조가 유연하므로(stock_ai 의 dict 반환) 핵심 필드만 명시하고
나머지는 dict 로 통과시킨다.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.schemas.common import Period


class AnalysisRequest(BaseModel):
    ticker: str
    period: Period = Period.p6mo
    use_llm: bool = False


class Signal(BaseModel):
    score: float = 0.0
    label: str = ""
    badge: str = ""
    reasons: list[str] = []


class HybridSignal(BaseModel):
    hybrid_score: float = 0.0
    combined_score: float = 50.0
    label: str = ""
    badge: str = ""
    reasons: list[str] = []
    warnings: list[str] = []


class RealtimePrice(BaseModel):
    price: float = 0.0
    ts: str = ""
    is_realtime: bool = False
    stale: bool = False
    stale_msg: str = ""


class OhlcPoint(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class AnalysisResponse(BaseModel):
    ticker: str
    sname: str = ""
    period: str
    data_ready: bool = True

    signals: Signal = Field(default_factory=Signal)
    hybrid: HybridSignal = Field(default_factory=HybridSignal)
    realtime: RealtimePrice = Field(default_factory=RealtimePrice)

    # stock_ai 의 자유 형식 dict — 구조가 풍부하여 통과형으로 노출
    advanced: dict[str, Any] = {}
    expected: dict[str, Any] | None = None
    risk_adj: dict[str, Any] = {}
    fund_score_data: dict[str, Any] = {}
    fund_info: dict[str, Any] = {}
    news_result: dict[str, Any] = {}
    dead_time: dict[str, Any] = {}
    breakout: dict[str, Any] = {}
    vol_anomaly: dict[str, Any] = {}

    tech_score: float = 0.0
    news_score: float = 0.0
