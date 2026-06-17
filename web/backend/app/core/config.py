"""config.py — pydantic-settings 기반 환경 설정.

기존 app.py 의 _SECRETS_KEY_MAP / 환경변수 로딩 로직을 단일 Settings 로 통합한다.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ── 앱 ──────────────────────────────────────────────────────────
    app_name: str = "ASA Service API"
    api_v1_prefix: str = "/api/v1"
    app_password: str = "qnwkehlwk"          # 진입 게이트

    # ── DB ──────────────────────────────────────────────────────────
    supabase_db_url: str = ""

    # ── 외부 키 ─────────────────────────────────────────────────────
    gemini_api_key: str = ""
    groq_api_key: str = ""
    dart_api_key: str = ""
    krx_id: str = ""
    krx_pw: str = ""

    # ── CORS ────────────────────────────────────────────────────────
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return get_settings_uncached()


def get_settings_uncached() -> Settings:
    s = Settings()
    # database.py / 일부 src 모듈이 환경변수에서 직접 읽으므로 역방향 동기화
    import os

    for env_key, value in (
        ("SUPABASE_DB_URL", s.supabase_db_url),
        ("GEMINI_API_KEY", s.gemini_api_key),
        ("GROQ_API_KEY", s.groq_api_key),
        ("DART_API_KEY", s.dart_api_key),
        ("KRX_ID", s.krx_id),
        ("KRX_PW", s.krx_pw),
    ):
        if value and not os.environ.get(env_key):
            os.environ[env_key] = value
    return s


settings = get_settings()
