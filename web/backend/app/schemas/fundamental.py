"""fundamental.py — 펀더멘털 탭 스키마."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class FundamentalResponse(BaseModel):
    ticker: str
    is_etf: bool = False
    fund_info: dict[str, Any] = {}
    fund_score_data: dict[str, Any] = {}
    etf_data: dict[str, Any] = {}
    etf_score: dict[str, Any] = {}
    investors: dict[str, Any] = {}
    investor_history: list[dict[str, Any]] = []
    insiders: list[dict[str, Any]] = []


class QuickAssessment(BaseModel):
    score10: float = 5.0
    verdict: str = "중립"
    reasons: list[str] = []
    summary: str = ""
    has_fund: bool = False


class AiReportResponse(BaseModel):
    ok: bool = False
    report: str = ""
    provider: str = ""
    error: str = ""
    quick_assessment: QuickAssessment = QuickAssessment()
