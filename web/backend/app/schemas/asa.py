"""asa.py — ASA 추천 스키마."""
from __future__ import annotations

from pydantic import BaseModel, Field


class AsaRunRequest(BaseModel):
    cash: float = Field(default=1_000_000, ge=0)


class AsaJobStarted(BaseModel):
    job_id: str
    status: str = "running"


class AsaJobStatus(BaseModel):
    job_id: str
    status: str           # running | done | error
    output: str = ""
    error: str = ""
