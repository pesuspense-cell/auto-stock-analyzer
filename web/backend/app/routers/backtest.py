"""backtest.py — 백테스트 라우터 (백그라운드 잡 + 폴링)."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from fastapi.concurrency import run_in_threadpool

from app.schemas.backtest import (
    BacktestJobStarted, BacktestJobStatus, BacktestRequest, BacktestResult,
)
from app.services import backtest_service

router = APIRouter(prefix="/backtest", tags=["backtest"])


@router.post("/run", response_model=BacktestJobStarted)
async def run(body: BacktestRequest):
    job_id = await run_in_threadpool(backtest_service.start_job, body.model_dump())
    return BacktestJobStarted(job_id=job_id)


@router.get("/jobs/{job_id}", response_model=BacktestJobStatus)
async def job_status(job_id: str):
    job = backtest_service.get_job(job_id)
    if not job:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "잡을 찾을 수 없습니다.")
    result = job.get("result")
    return BacktestJobStatus(
        job_id=job_id, status=job["status"], error=job.get("error", ""),
        result=BacktestResult(**result) if result else None,
    )
