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
            cur.execute("""
                CREATE TABLE IF NOT EXISTS trade_history (
                    id          SERIAL           PRIMARY KEY,
                    user_id     INTEGER          NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    ticker      TEXT             NOT NULL,
                    buy_price   DOUBLE PRECISION NOT NULL,
                    sell_price  DOUBLE PRECISION NOT NULL,
                    quantity    DOUBLE PRECISION NOT NULL,
                    net_profit  DOUBLE PRECISION NOT NULL,
                    return_rate DOUBLE PRECISION NOT NULL,
                    traded_at   TIMESTAMPTZ      NOT NULL DEFAULT NOW()
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


def upsert_portfolio(user_id: int, ticker: str, avg_price: float, quantity: float = 1.0) -> dict:
    """추가 매수 통합 — 이미 보유 중이면 가중평균 단가·수량 합산, 없으면 신규 추가.

    반환: {"ok": True, "merged": bool}
    """
    ticker = ticker.upper().strip()
    with _conn() as con:
        with con.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT id, avg_price, quantity FROM portfolios"
                " WHERE user_id = %s AND ticker = %s",
                (user_id, ticker),
            )
            existing = cur.fetchone()
        with con.cursor() as cur:
            if existing:
                old_qty   = float(existing["quantity"])
                old_price = float(existing["avg_price"])
                new_qty   = old_qty + float(quantity)
                new_price = (old_price * old_qty + float(avg_price) * float(quantity)) / new_qty
                cur.execute(
                    "UPDATE portfolios SET avg_price = %s, quantity = %s WHERE id = %s",
                    (new_price, new_qty, existing["id"]),
                )
            else:
                cur.execute(
                    "INSERT INTO portfolios (user_id, ticker, avg_price, quantity)"
                    " VALUES (%s, %s, %s, %s)",
                    (user_id, ticker, float(avg_price), float(quantity)),
                )
        con.commit()
    return {"ok": True, "merged": existing is not None}


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


def sell_item(
    user_id: int,
    item_id: int,
    sell_price: float,
    quantity: float | None = None,
) -> dict:
    """포트폴리오 종목 매도 — trade_history에 기록하고 portfolios에서 수량 차감(또는 삭제).

    quantity가 None이면 보유 전량 매도.
    반환: {"ok": True, "net_profit": float, "return_rate": float} 또는 {"ok": False, "error": str}
    """
    with _conn() as con:
        with con.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT ticker, avg_price, quantity FROM portfolios WHERE id = %s AND user_id = %s",
                (item_id, user_id),
            )
            row = cur.fetchone()
    if not row:
        return {"ok": False, "error": "항목을 찾을 수 없습니다"}

    ticker    = row["ticker"]
    buy_price = float(row["avg_price"])
    total_qty = float(row["quantity"])
    sell_qty  = min(float(quantity), total_qty) if quantity is not None else total_qty
    if sell_qty <= 0:
        return {"ok": False, "error": "매도 수량이 0 이하입니다"}

    net_profit  = (sell_price - buy_price) * sell_qty
    return_rate = (sell_price / buy_price - 1) * 100 if buy_price else 0.0

    with _conn() as con:
        with con.cursor() as cur:
            cur.execute(
                "INSERT INTO trade_history"
                " (user_id, ticker, buy_price, sell_price, quantity, net_profit, return_rate)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (user_id, ticker, buy_price, float(sell_price),
                 sell_qty, net_profit, return_rate),
            )
            remaining = total_qty - sell_qty
            if remaining <= 0.001:
                cur.execute(
                    "DELETE FROM portfolios WHERE id = %s AND user_id = %s",
                    (item_id, user_id),
                )
            else:
                cur.execute(
                    "UPDATE portfolios SET quantity = %s WHERE id = %s AND user_id = %s",
                    (remaining, item_id, user_id),
                )
        con.commit()
    return {"ok": True, "net_profit": net_profit, "return_rate": return_rate}


def get_trade_history(user_id: int, limit: int = 100) -> list[dict]:
    """매도 이력 최신순 조회."""
    with _conn() as con:
        with con.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT id, ticker, buy_price, sell_price, quantity,"
                " net_profit, return_rate, traded_at"
                " FROM trade_history WHERE user_id = %s"
                " ORDER BY traded_at DESC LIMIT %s",
                (user_id, limit),
            )
            rows = cur.fetchall()
    return [
        {
            **dict(r),
            "traded_at": r["traded_at"].isoformat() if r.get("traded_at") else "",
        }
        for r in rows
    ]


def get_trade_summary(user_id: int) -> dict:
    """누적 매도 통계 — 통화 혼재 시 UI에서 환율 적용 필요.

    반환: {total_sell_amount, total_net_profit, total_buy_cost, trade_count}
    """
    with _conn() as con:
        with con.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT"
                "  COALESCE(SUM(sell_price * quantity), 0) AS total_sell_amount,"
                "  COALESCE(SUM(net_profit),            0) AS total_net_profit,"
                "  COALESCE(SUM(buy_price  * quantity), 0) AS total_buy_cost,"
                "  COUNT(*)                               AS trade_count"
                " FROM trade_history WHERE user_id = %s",
                (user_id,),
            )
            row = cur.fetchone()
    return {
        "total_sell_amount": float(row["total_sell_amount"]),
        "total_net_profit":  float(row["total_net_profit"]),
        "total_buy_cost":    float(row["total_buy_cost"]),
        "trade_count":       int(row["trade_count"]),
    }


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
