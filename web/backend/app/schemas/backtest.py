"""backtest.py — 백테스트 스키마."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class BacktestRequest(BaseModel):
    markets: list[str] = Field(default_factory=lambda: ["KOSPI", "KOSDAQ"])
    universe_n: int = Field(default=200, ge=50, le=500)
    top_n: int = Field(default=20, ge=5, le=50)
    initial_capital: float = Field(default=10_000_000, ge=1_000_000)
    start_date: str = "2020-01-01"
    end_date: str = "2024-12-31"
    # {"YYYY-MM-DD": 금액} 추가 입금 일정
    deposit_schedule: dict[str, float] = {}
    benchmark_ticker: str | None = "^KS11"
    benchmark_label: str = "^KS11 (KOSPI)"


class BacktestJobStarted(BaseModel):
    job_id: str
    status: str = "running"


class BacktestMetrics(BaseModel):
    total_invested: float = 0.0
    final_asset: float = 0.0
    cash: float = 0.0
    return_pct: float = 0.0
    cagr: float = 0.0
    mdd: float = 0.0
    win_rate: float = 0.0
    total_sells: int = 0
    sl_count: int = 0


class BacktestResult(BaseModel):
    metrics: BacktestMetrics = BacktestMetrics()
    equity_curve: list[dict[str, Any]] = []
    trade_log: list[dict[str, Any]] = []
    selected_stocks: list[dict[str, Any]] = []
    benchmark: list[dict[str, Any]] = []
    benchmark_label: str = ""
    log_text: str = ""


class BacktestJobStatus(BaseModel):
    job_id: str
    status: str           # running | done | error
    error: str = ""
    result: BacktestResult | None = None
