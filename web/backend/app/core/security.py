"""security.py — 인증 의존성.

기존 src/database.py 의 토큰 + 슬라이딩 만료 로직을 그대로 사용한다.
- get_current_user: Bearer 토큰 필수 (만료/무효 → 401)
- get_optional_user: 토큰 없어도 통과 (None 반환)
- require_app_password: 진입 게이트 (헤더 X-App-Password)
"""
from __future__ import annotations

from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app import bootstrap  # noqa: F401  (sys.path 등록)
from app.core.config import settings
from src.database import get_user_by_token

_bearer = HTTPBearer(auto_error=False)


def get_optional_user(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> dict | None:
    if creds is None or not creds.credentials:
        return None
    return get_user_by_token(creds.credentials)


def get_current_user(
    user: dict | None = Depends(get_optional_user),
) -> dict:
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="인증이 필요하거나 세션이 만료되었습니다.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def require_app_password(x_app_password: str = Header(default="")) -> None:
    """진입 게이트 — 기존 _APP_PASSWORD 검증을 헤더로 대체."""
    if x_app_password != settings.app_password:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="앱 비밀번호가 올바르지 않습니다.",
        )
