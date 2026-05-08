"""
flask_api.py - Flask REST API 서버 (포트 5000)
Streamlit(포트 8501)에서 requests로 호출하는 백엔드 인증/포트폴리오 API

실행: python flask_api.py
"""
from functools import wraps

from flask import Flask, jsonify, request

from src.database import (
    add_portfolio,
    delete_portfolio_item,
    get_portfolio,
    get_user_by_token,
    init_db,
    login_user,
    logout_user,
    register_user,
)

app = Flask(__name__)


# ── 인증 미들웨어 ──────────────────────────────────────────────────────────────

def _require_auth(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        token = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
        user = get_user_by_token(token)
        if not user:
            return jsonify({"error": "Unauthorized"}), 401
        request.user = user
        return f(*args, **kwargs)
    return wrapper


def _token_from_header() -> str:
    return request.headers.get("Authorization", "").removeprefix("Bearer ").strip()


# ── 인증 라우트 ────────────────────────────────────────────────────────────────

@app.post("/register")
def register():
    d = request.get_json(silent=True) or {}
    email    = d.get("email", "").strip()
    password = d.get("password", "")
    if not email or len(password) < 6:
        return jsonify({"error": "이메일과 비밀번호(6자 이상)를 입력하세요"}), 400
    result = register_user(email, password)
    if result["ok"]:
        return jsonify({"message": "회원가입 성공"}), 201
    return jsonify(result), 409


@app.post("/login")
def login():
    d = request.get_json(silent=True) or {}
    result = login_user(d.get("email", ""), d.get("password", ""))
    if result["ok"]:
        return jsonify(result), 200
    return jsonify(result), 401


@app.post("/logout")
@_require_auth
def logout():
    logout_user(_token_from_header())
    return jsonify({"message": "로그아웃 성공"}), 200


# ── 포트폴리오 라우트 ──────────────────────────────────────────────────────────

@app.get("/portfolio")
@_require_auth
def get_portfolio_route():
    items = get_portfolio(request.user["id"])
    return jsonify(items), 200


@app.post("/portfolio/add")
@_require_auth
def add_portfolio_route():
    d = request.get_json(silent=True) or {}
    ticker    = d.get("ticker", "").upper().strip()
    avg_price = d.get("avg_price")
    quantity  = float(d.get("quantity", 1.0))
    if not ticker or avg_price is None:
        return jsonify({"error": "ticker와 avg_price는 필수입니다"}), 400
    add_portfolio(request.user["id"], ticker, float(avg_price), quantity)
    return jsonify({"message": f"{ticker} 포트폴리오에 추가되었습니다"}), 201


@app.delete("/portfolio/<int:item_id>")
@_require_auth
def delete_portfolio_route(item_id: int):
    result = delete_portfolio_item(item_id, request.user["id"])
    if result["ok"]:
        return jsonify(result), 200
    return jsonify(result), 404


# ── 진입점 ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    init_db()
    print("✅ DB 초기화 완료")
    print("🚀 Flask API 서버 시작 → http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
