"""auth.py — 인증 라우터. src/database.py 의 순수 함수를 그대로 노출."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool

from app import bootstrap  # noqa: F401
from app.core.security import get_current_user
from app.schemas.auth import (
    LoginRequest, LoginResponse, RegisterRequest, UserResponse,
)
from app.schemas.common import OkResponse
from app.core.security import _bearer  # HTTPBearer 인스턴스 재사용
from fastapi.security import HTTPAuthorizationCredentials

from src.database import register_user, login_user, logout_user

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=OkResponse)
async def register(body: RegisterRequest):
    result = await run_in_threadpool(register_user, body.email, body.password)
    if not result.get("ok"):
        raise HTTPException(status.HTTP_409_CONFLICT, result.get("error", "회원가입 실패"))
    return OkResponse(ok=True)


@router.post("/login", response_model=LoginResponse)
async def login(body: LoginRequest):
    result = await run_in_threadpool(login_user, body.email, body.password)
    if not result.get("ok"):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, result.get("error", "로그인 실패"))
    return LoginResponse(**result)


@router.post("/logout", response_model=OkResponse)
async def logout(creds: HTTPAuthorizationCredentials | None = Depends(_bearer)):
    if creds and creds.credentials:
        await run_in_threadpool(logout_user, creds.credentials)
    return OkResponse(ok=True)


@router.get("/me", response_model=UserResponse)
async def me(user: dict = Depends(get_current_user)):
    return UserResponse(id=user["id"], email=user["email"])
