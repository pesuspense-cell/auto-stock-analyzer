"""auth.py — 인증 요청/응답 스키마."""
from __future__ import annotations

from pydantic import BaseModel, EmailStr, Field


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=4, max_length=128)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class LoginResponse(BaseModel):
    ok: bool = True
    token: str
    user_id: int
    email: str


class UserResponse(BaseModel):
    id: int
    email: str
