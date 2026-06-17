"""asa.py — ASA 추천 라우터 (백그라운드 잡 + 폴링).

로그인 사용자의 포트폴리오를 보유 포지션으로 반영한다(선택). 미인증 시 신규 매수 신호만.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool

from app import bootstrap  # noqa: F401
from app.core.security import get_optional_user
from app.schemas.asa import AsaJobStarted, AsaJobStatus, AsaRunRequest
from app.services import asa_service

from src.database import get_portfolio

router = APIRouter(prefix="/asa", tags=["asa"])


@router.post("/run", response_model=AsaJobStarted)
async def run(body: AsaRunRequest, user: dict | None = Depends(get_optional_user)):
    items = await run_in_threadpool(get_portfolio, user["id"]) if user else []
    job_id = await run_in_threadpool(asa_service.start_job, body.cash, items)
    return AsaJobStarted(job_id=job_id)


@router.get("/jobs/{job_id}", response_model=AsaJobStatus)
async def job_status(job_id: str):
    job = asa_service.get_job(job_id)
    if not job:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "잡을 찾을 수 없습니다.")
    return AsaJobStatus(
        job_id=job_id, status=job["status"],
        output=job.get("output", ""), error=job.get("error", ""),
    )
