"""
database.py - Supabase(PostgreSQL) 기반 사용자 인증 및 포트폴리오 관리

연결 정보는 SUPABASE_DB_URL 환경 변수 또는 st.secrets["SUPABASE_DB_URL"]에서 읽는다.
형식: postgresql://[user]:[password]@[host]:[port]/[dbname]
"""
import logging
import os
import secrets
from contextlib import contextmanager

import psycopg2
import psycopg2.errors
import psycopg2.extras
import psycopg2.pool
from werkzeug.security import check_password_hash, generate_password_hash

logger = logging.getLogger(__name__)

_pool: psycopg2.pool.ThreadedConnectionPool | None = None


# ── 연결 관리 ─────────────────────────────────────────────────────────────────

def _db_url() -> str:
    """SUPABASE_DB_URL을 환경 변수 → st.secrets 순서로 읽는다."""
    url = os.getenv("SUPABASE_DB_URL", "")
    if not url:
        try:
            import streamlit as st
            url = st.secrets.get("SUPABASE_DB_URL", "")
        except Exception:
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
        _pool = psycopg2.pool.ThreadedConnectionPool(1, 10, dsn=_db_url())
    return _pool


@contextmanager
def _conn():
    """스레드 안전 커넥션 풀에서 연결을 대여하고 반납한다."""
    pool = _get_pool()
    con = pool.getconn()
    try:
        yield con
    finally:
        pool.putconn(con)


# ── 스키마 초기화 ─────────────────────────────────────────────────────────────

def init_db() -> None:
    """users / portfolios 테이블을 PostgreSQL에 생성한다 (없을 때만)."""
    with _conn() as con:
        with con.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id            SERIAL      PRIMARY KEY,
                    email         TEXT        UNIQUE NOT NULL,
                    password_hash TEXT        NOT NULL,
                    session_token TEXT,
                    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
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
                "UPDATE users SET session_token = %s WHERE id = %s",
                (token, row["id"]),
            )
        con.commit()
    return {"ok": True, "token": token, "user_id": row["id"], "email": row["email"]}


def logout_user(token: str) -> None:
    with _conn() as con:
        with con.cursor() as cur:
            cur.execute(
                "UPDATE users SET session_token = NULL WHERE session_token = %s",
                (token,),
            )
        con.commit()


def get_user_by_token(token: str) -> dict | None:
    """토큰으로 사용자 조회. 없으면 None."""
    if not token:
        return None
    with _conn() as con:
        with con.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT id, email FROM users WHERE session_token = %s",
                (token,),
            )
            row = cur.fetchone()
    return dict(row) if row else None


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
