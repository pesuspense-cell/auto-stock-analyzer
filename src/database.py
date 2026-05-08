"""
database.py - SQLite 기반 사용자 인증 및 포트폴리오 관리
사용처: flask_api.py (HTTP 래퍼) / app.py (직접 호출)
"""
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from werkzeug.security import check_password_hash, generate_password_hash

DB_PATH = Path(__file__).parent.parent / "stock_app.db"


@contextmanager
def _conn():
    con = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    try:
        yield con
    finally:
        con.close()


def init_db() -> None:
    with _conn() as con:
        con.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                email         TEXT    UNIQUE NOT NULL,
                password_hash TEXT    NOT NULL,
                session_token TEXT,
                created_at    TEXT    NOT NULL
            );
            CREATE TABLE IF NOT EXISTS portfolios (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id   INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                ticker    TEXT    NOT NULL,
                avg_price REAL    NOT NULL,
                quantity  REAL    NOT NULL DEFAULT 1.0,
                added_at  TEXT    NOT NULL
            );
        """)
        con.commit()


# ── 인증 ─────────────────────────────────────────────────────────────────────

def register_user(email: str, password: str) -> dict:
    """회원가입. 성공 → {"ok": True}, 중복 → {"ok": False, "error": ...}"""
    pw_hash = generate_password_hash(password)
    try:
        with _conn() as con:
            con.execute(
                "INSERT INTO users (email, password_hash, created_at) VALUES (?, ?, ?)",
                (email.strip().lower(), pw_hash, datetime.now().isoformat()),
            )
            con.commit()
        return {"ok": True}
    except sqlite3.IntegrityError:
        return {"ok": False, "error": "이미 사용 중인 이메일입니다"}


def login_user(email: str, password: str) -> dict:
    """로그인. 성공 → {"ok": True, "token": ..., "user_id": ..., "email": ...}"""
    with _conn() as con:
        row = con.execute(
            "SELECT id, email, password_hash FROM users WHERE email = ?",
            (email.strip().lower(),),
        ).fetchone()
    if not row or not check_password_hash(row["password_hash"], password):
        return {"ok": False, "error": "이메일 또는 비밀번호가 올바르지 않습니다"}
    token = secrets.token_hex(32)
    with _conn() as con:
        con.execute(
            "UPDATE users SET session_token = ? WHERE id = ?",
            (token, row["id"]),
        )
        con.commit()
    return {"ok": True, "token": token, "user_id": row["id"], "email": row["email"]}


def logout_user(token: str) -> None:
    with _conn() as con:
        con.execute(
            "UPDATE users SET session_token = NULL WHERE session_token = ?",
            (token,),
        )
        con.commit()


def get_user_by_token(token: str) -> dict | None:
    """토큰으로 사용자 조회. 없으면 None."""
    if not token:
        return None
    with _conn() as con:
        row = con.execute(
            "SELECT id, email FROM users WHERE session_token = ?",
            (token,),
        ).fetchone()
    return dict(row) if row else None


# ── 포트폴리오 CRUD ───────────────────────────────────────────────────────────

def add_portfolio(user_id: int, ticker: str, avg_price: float, quantity: float = 1.0) -> dict:
    with _conn() as con:
        con.execute(
            "INSERT INTO portfolios (user_id, ticker, avg_price, quantity, added_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (user_id, ticker.upper().strip(), float(avg_price), float(quantity),
             datetime.now().isoformat()),
        )
        con.commit()
    return {"ok": True}


def get_portfolio(user_id: int) -> list[dict]:
    with _conn() as con:
        rows = con.execute(
            "SELECT id, ticker, avg_price, quantity, added_at"
            " FROM portfolios WHERE user_id = ? ORDER BY added_at DESC",
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def delete_portfolio_item(item_id: int, user_id: int) -> dict:
    with _conn() as con:
        cur = con.execute(
            "DELETE FROM portfolios WHERE id = ? AND user_id = ?",
            (item_id, user_id),
        )
        con.commit()
    return {"ok": True} if cur.rowcount > 0 else {"ok": False, "error": "항목을 찾을 수 없습니다"}
