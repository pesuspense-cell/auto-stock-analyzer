"""main.py — FastAPI 진입점. CORS·라우터 등록·DB 초기화.

실행:  cd web/backend && uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import bootstrap  # noqa: F401  — 반드시 가장 먼저 (sys.path 등록)
from app.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 기존 Streamlit 앱과 동일한 Supabase 스키마 초기화
    try:
        from src.database import init_db
        init_db()
        logger.info("[startup] DB 초기화 완료")
    except Exception as e:
        logger.warning("[startup] DB 초기화 건너뜀: %s", e)
    yield


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["health"])
async def health():
    return {"status": "ok", "service": settings.app_name}


# ── 라우터 등록 ─────────────────────────────────────────────────────
from app.routers import (  # noqa: E402
    auth, market, stocks, analysis, portfolio,
    news, fundamental, asa, backtest,
)

for r in (
    auth.router, market.router, stocks.router, analysis.router, portfolio.router,
    news.router, fundamental.router, asa.router, backtest.router,
):
    app.include_router(r, prefix=settings.api_v1_prefix)
