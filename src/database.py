"""
database.py - Supabase(PostgreSQL) 기반 사용자 인증 및 포트폴리오 관리

연결 정보는 SUPABASE_DB_URL 환경 변수 또는 st.secrets["SUPABASE_DB_URL"]에서 읽는다.
형식: postgresql://[user]:[password]@[host]:[port]/[dbname]
"""
import logging
import os
import secrets
from contextlib import contextmanager
from datetime import timedelta

import psycopg2
import psycopg2.errors
import psycopg2.extras
import psycopg2.pool
from werkzeug.security import check_password_hash, generate_password_hash

SESSION_TIMEOUT_HOURS = 1

logger = logging.getLogger(__name__)

_pool: psycopg2.pool.ThreadedConnectionPool | None = None


# ── 연결 관리 ─────────────────────────────────────────────────────────────────

def _db_url() -> str:
    """SUPABASE_DB_URL을 환경 변수 → st.secrets 순서로 읽는다."""
    url = os.getenv("SUPABASE_DB_URL", "")
    if not url:
        try:
            import streamlit as st
            # st.secrets는 dict-like; KeyError 방어
            url = st.secrets["SUPABASE_DB_URL"]
        except (KeyError, AttributeError, Exception):
            pass
    if not url:
        raise RuntimeError(
            "SUPABASE_DB_URL이 설정되지 않았습니다. "
            "Streamlit Secrets 또는 환경 변수에 SUPABASE_DB_URL을 추가하세요."
        )
    return url


def _get_pool() -> psycopg2.pool.ThreadedConnectionPool:
    global _pool
    if _pool is None:
        _pool = psycopg2.pool.ThreadedConnectionPool(
            1, 10,
            dsn=_db_url(),
            connect_timeout=10,   # 타임아웃 10초 — Cannot assign requested address 방어
        )
    return _pool


@contextmanager
def _conn():
    """스레드 안전 커넥션 풀에서 연결을 대여하고 반납한다.

    pool_pre_ping 동등 로직: 커넥션을 꺼낼 때 SELECT 1로 생존 여부를 확인하고,
    죽은 연결이면 해당 연결을 폐기하고 풀 전체를 재생성한 뒤 새 연결을 반환한다.
    """
    global _pool
    pool = _get_pool()
    con = pool.getconn()
    try:
        # ── pre-ping: 연결 유효성 검사 ──────────────────────────────────────
        try:
            with con.cursor() as _cur:
                _cur.execute("SELECT 1")
        except psycopg2.OperationalError:
            # 죽은 연결 폐기 → 풀 재생성 → 새 연결
            pool.putconn(con, close=True)
            _pool = None
            pool = _get_pool()
            con = pool.getconn()
        yield con
    finally:
        pool.putconn(con)


# ── 스키마 초기화 ─────────────────────────────────────────────────────────────

def init_db() -> None:
    """users / portfolios / recommendation_history 테이블을 PostgreSQL에 생성한다 (없을 때만)."""
    with _conn() as con:
        with con.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id            SERIAL      PRIMARY KEY,
                    email         TEXT        UNIQUE NOT NULL,
                    password_hash TEXT        NOT NULL,
                    session_token TEXT,
                    last_activity TIMESTAMPTZ,
                    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            # 기존 테이블에 last_activity 컬럼이 없는 경우 마이그레이션
            cur.execute("""
                ALTER TABLE users ADD COLUMN IF NOT EXISTS last_activity TIMESTAMPTZ
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS portfolios (
                    id        SERIAL           PRIMARY KEY,
                    user_id   INTEGER          NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    ticker    TEXT             NOT NULL,
                    avg_price DOUBLE PRECISION NOT NULL,
                    quantity  DOUBLE PRECISION NOT NULL DEFAULT 1.0,
                    added_at  TIMESTAMPTZ      NOT NULL DEFAULT NOW()
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS recommendation_history (
                    id              SERIAL      PRIMARY KEY,
                    user_id         INTEGER     NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    investment_amt  BIGINT      NOT NULL,
                    risk_profile    TEXT        NOT NULL DEFAULT '중립형',
                    recommendations JSONB       NOT NULL,
                    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
        con.commit()
    logger.info("[DB] Supabase PostgreSQL 연결 및 스키마 초기화 완료")


# ── 인증 ─────────────────────────────────────────────────────────────────────

def register_user(email: str, password: str) -> dict:
    """회원가입. 성공 → {"ok": True}, 중복 → {"ok": False, "error": ...}"""
    pw_hash = generate_password_hash(password)
    with _conn() as con:
        try:
            with con.cursor() as cur:
                cur.execute(
                    "INSERT INTO users (email, password_hash) VALUES (%s, %s)",
                    (email.strip().lower(), pw_hash),
                )
            con.commit()
            return {"ok": True}
        except psycopg2.errors.UniqueViolation:
            con.rollback()
            return {"ok": False, "error": "이미 사용 중인 이메일입니다"}


def login_user(email: str, password: str) -> dict:
    """로그인. 성공 → {"ok": True, "token": ..., "user_id": ..., "email": ...}"""
    with _conn() as con:
        with con.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT id, email, password_hash FROM users WHERE email = %s",
                (email.strip().lower(),),
            )
            row = cur.fetchone()
    if not row or not check_password_hash(row["password_hash"], password):
        return {"ok": False, "error": "이메일 또는 비밀번호가 올바르지 않습니다"}
    token = secrets.token_hex(32)
    with _conn() as con:
        with con.cursor() as cur:
            cur.execute(
                "UPDATE users SET session_token = %s, last_activity = NOW() WHERE id = %s",
                (token, row["id"]),
            )
        con.commit()
    return {"ok": True, "token": token, "user_id": row["id"], "email": row["email"]}


def logout_user(token: str) -> None:
    with _conn() as con:
        with con.cursor() as cur:
            cur.execute(
                "UPDATE users SET session_token = NULL, last_activity = NULL"
                " WHERE session_token = %s",
                (token,),
            )
        con.commit()


def get_user_by_token(token: str) -> dict | None:
    """토큰으로 사용자 조회. 만료됐거나 없으면 None.

    - last_activity 기준 SESSION_TIMEOUT_HOURS 초과 시 토큰 삭제 후 None 반환
    - 유효하면 last_activity = NOW() 로 갱신 (Sliding Expiration)
    """
    if not token:
        return None
    with _conn() as con:
        with con.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT id, email,"
                " (last_activity IS NULL OR last_activity < NOW() - %s) AS expired"
                " FROM users WHERE session_token = %s",
                (timedelta(hours=SESSION_TIMEOUT_HOURS), token),
            )
            row = cur.fetchone()
            if not row:
                return None
            if row["expired"]:
                cur.execute(
                    "UPDATE users SET session_token = NULL, last_activity = NULL"
                    " WHERE id = %s",
                    (row["id"],),
                )
                con.commit()
                return None
            cur.execute(
                "UPDATE users SET last_activity = NOW() WHERE id = %s",
                (row["id"],),
            )
            con.commit()
    return {"id": row["id"], "email": row["email"]}


# ── 포트폴리오 CRUD ───────────────────────────────────────────────────────────

def add_portfolio(user_id: int, ticker: str, avg_price: float, quantity: float = 1.0) -> dict:
    with _conn() as con:
        with con.cursor() as cur:
            cur.execute(
                "INSERT INTO portfolios (user_id, ticker, avg_price, quantity)"
                " VALUES (%s, %s, %s, %s)",
                (user_id, ticker.upper().strip(), float(avg_price), float(quantity)),
            )
        con.commit()
    return {"ok": True}


def get_portfolio(user_id: int) -> list[dict]:
    with _conn() as con:
        with con.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT id, ticker, avg_price, quantity, added_at"
                " FROM portfolios WHERE user_id = %s ORDER BY added_at DESC",
                (user_id,),
            )
            rows = cur.fetchall()
    return [
        {
            **dict(r),
            "added_at": r["added_at"].isoformat() if r.get("added_at") else "",
        }
        for r in rows
    ]


def delete_portfolio_item(item_id: int, user_id: int) -> dict:
    with _conn() as con:
        with con.cursor() as cur:
            cur.execute(
                "DELETE FROM portfolios WHERE id = %s AND user_id = %s",
                (item_id, user_id),
            )
            rowcount = cur.rowcount
        con.commit()
    return {"ok": True} if rowcount > 0 else {"ok": False, "error": "항목을 찾을 수 없습니다"}


# ── AI 추천 이력 ──────────────────────────────────────────────────────────────

def save_recommendation(
    user_id: int,
    investment_amt: int,
    risk_profile: str,
    recommendations: list[dict],
) -> dict:
    """AI 추천 결과를 recommendation_history 테이블에 저장."""
    import json as _json
    with _conn() as con:
        with con.cursor() as cur:
            cur.execute(
                "INSERT INTO recommendation_history"
                " (user_id, investment_amt, risk_profile, recommendations)"
                " VALUES (%s, %s, %s, %s)",
                (user_id, investment_amt, risk_profile,
                 _json.dumps(recommendations, ensure_ascii=False)),
            )
        con.commit()
    return {"ok": True}


def get_recommendation_history(user_id: int, limit: int = 5) -> list[dict]:
    """최근 추천 이력 조회 (최신순)."""
    with _conn() as con:
        with con.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT id, investment_amt, risk_profile, recommendations, created_at"
                " FROM recommendation_history"
                " WHERE user_id = %s"
                " ORDER BY created_at DESC LIMIT %s",
                (user_id, limit),
            )
            rows = cur.fetchall()
    return [
        {
            **dict(r),
            "created_at": r["created_at"].isoformat() if r.get("created_at") else "",
        }
        for r in rows
    ]
