"""
supabase_account.py — 웹앱 로그인 계정의 포트폴리오·예수금을 Supabase에서 읽어
시그널 봇 계좌(holdings) 구조로 변환한다.

자동 주문이 아니라 알림 전용이므로, 봇은 service_role(secret) 키로 PostgREST/Admin
API를 통해 대상 계정의 보유종목(portfolios)과 예수금(user_settings.cash_balance)을
읽기만 한다. 손절가(sl)·익절가(tp)는 portfolios에 저장돼 있지 않으므로 봇이 진입가 +
현재 ATR로 산출한다(signal_bot.monitor_positions).

자격(URL/secret key)은 다음 순서로 자동 탐색한다:
  1) 환경변수 SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY (루트 .env)
  2) web/frontend/.env.local 의 NEXT_PUBLIC_SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY
→ 시크릿을 중복 보관하지 않고 프론트엔드 .env.local 을 그대로 재사용한다.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).parent

# KRX 전체 상장목록(코드→한글명) 캐시 — Supabase stocks 에 없는 종목(ETF·신규상장 등) 보강용.
_krx_name_cache: dict[str, str] | None = None


def _krx_name_map() -> dict[str, str]:
    """FinanceDataReader 로 KOSPI·KOSDAQ·ETF 전체 상장목록의 코드→한글명을 1회 로드(프로세스 캐시).

    실패해도 빈 dict 를 반환해 호출부가 티커 폴백하도록 한다(이름 미해석은 치명적 아님).
    """
    global _krx_name_cache
    if _krx_name_cache is not None:
        return _krx_name_cache
    m: dict[str, str] = {}
    try:
        import FinanceDataReader as fdr
        for market in ("KOSPI", "KOSDAQ", "ETF/KR"):
            try:
                df = fdr.StockListing(market)
                code_col = "Code" if "Code" in df.columns else "Symbol"
                for _, row in df.iterrows():
                    code = str(row.get(code_col, "")).strip().zfill(6)
                    name = str(row.get("Name", "")).strip()
                    if code and name and name.lower() != "nan":
                        m.setdefault(code, name)
            except Exception:
                continue
    except Exception:
        pass
    _krx_name_cache = m
    return m


def _discover_creds() -> tuple[str, str]:
    url = os.getenv("SUPABASE_URL") or os.getenv("NEXT_PUBLIC_SUPABASE_URL") or ""
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_SECRET_KEY") or ""
    if url and key:
        return url, key
    envlocal = _ROOT / "web" / "frontend" / ".env.local"
    if envlocal.exists():
        d: dict[str, str] = {}
        for line in envlocal.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                d[k.strip()] = v.strip()
        url = url or d.get("NEXT_PUBLIC_SUPABASE_URL") or d.get("SUPABASE_URL") or ""
        key = key or d.get("SUPABASE_SERVICE_ROLE_KEY") or d.get("SUPABASE_SECRET_KEY") or ""
    return url, key


class SupabaseAccount:
    def __init__(self, url: str | None = None, key: str | None = None):
        if not (url and key):
            url, key = _discover_creds()
        self.url = (url or "").rstrip("/")
        self.key = key or ""
        self.enabled = bool(self.url and self.key)

    def _get(self, path: str):
        req = urllib.request.Request(
            f"{self.url}/{path}",
            headers={"apikey": self.key, "Authorization": f"Bearer {self.key}"},
        )
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read().decode())

    def _patch(self, path: str, body: dict) -> None:
        req = urllib.request.Request(
            f"{self.url}/{path}", method="PATCH",
            data=json.dumps(body).encode(),
            headers={
                "apikey": self.key, "Authorization": f"Bearer {self.key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=20) as r:
            r.read()

    def link_telegram(self, token: str, chat_id: str) -> str | None:
        """딥링크 토큰으로 사용자를 찾아 chat_id 저장 + 알림 ON + 토큰 소거.

        연동된 user_id 반환(토큰 무효/만료면 None). bot.py·telegram_link_bot.py 공용.
        """
        if not (self.enabled and token):
            return None
        rows = self._get(
            f"rest/v1/user_settings?telegram_link_token=eq.{urllib.parse.quote(token)}&select=user_id"
        )
        if not rows:
            return None
        uid = rows[0]["user_id"]
        self._patch(f"rest/v1/user_settings?user_id=eq.{uid}", {
            "telegram_chat_id": str(chat_id),
            "telegram_enabled": True,
            "telegram_link_token": None,
        })
        return uid

    def list_notify_users(self) -> list[dict]:
        """텔레그램 연동(chat_id 등록) + 수신 ON 사용자 목록 [{user_id, chat_id}].

        005 마이그레이션(telegram_* 컬럼) 적용 전이면 빈 목록을 반환해 봇이 env 폴백만
        쓰도록 한다.
        """
        try:
            rows = self._get(
                "rest/v1/user_settings"
                "?telegram_enabled=eq.true&telegram_chat_id=not.is.null"
                "&select=user_id,telegram_chat_id"
            )
        except Exception:
            return []
        return [
            {"user_id": r["user_id"], "chat_id": r["telegram_chat_id"]}
            for r in rows if r.get("telegram_chat_id")
        ]

    def resolve_user_id(self, email: str) -> str | None:
        """Auth Admin API로 이메일 → user_id 해석."""
        data = self._get("auth/v1/admin/users?per_page=200")
        users = data.get("users", data) if isinstance(data, dict) else data
        for u in users:
            if (u.get("email") or "").lower() == email.strip().lower():
                return u["id"]
        return None

    def _resolve_names(self, tickers: list[str]) -> dict[str, str]:
        """티커 → 한글 종목명. ① Supabase stocks 우선, ② 미해석분은 KRX 상장목록(FDR)에서 보강."""
        if not tickers:
            return {}
        names: dict[str, str] = {}
        # ① Supabase stocks 테이블 (한글 별칭 name_kr 우선)
        val = "(" + ",".join(f'"{t}"' for t in tickers) + ")"
        path = f"rest/v1/stocks?ticker=in.{urllib.parse.quote(val)}&select=ticker,name,name_kr"
        try:
            for s in self._get(path):
                nm = s.get("name_kr") or s.get("name")
                if nm:
                    names[s["ticker"]] = nm
        except Exception:
            pass
        # ② Supabase 에 없던 종목(ETF·신규상장 등)은 KRX 전체 상장목록의 한글명으로 보강
        unresolved = [t for t in tickers if t not in names]
        if unresolved:
            krx = _krx_name_map()
            if krx:
                for t in unresolved:
                    nm = krx.get(t.split(".")[0].strip().zfill(6))
                    if nm:
                        names[t] = nm
        return names

    def load_holdings(self, user_id: str | None = None, email: str | None = None) -> dict:
        """대상 계정의 예수금·보유종목·알림설정을 시그널 봇 holdings 구조로 반환.

        반환: {
          "cash": float,
          "positions": {ticker: {name, entry_price, quantity, [sl], [tp]}},
          "alert_prefs": {신호종류: bool},   # 웹 UI 알림 on/off (없으면 빈 dict)
          "user_id": str,
        }
        포지션의 sl/tp 는 portfolios 에 저장돼 있으면 채우고(실제 MTS 손절·익절가),
        없으면(null) 키 자체를 넣지 않아 봇이 진입가+현재ATR로 산출하도록 한다.
        """
        if not self.enabled:
            raise RuntimeError("Supabase 자격(URL/secret key) 미설정 — .env 또는 web/frontend/.env.local 확인")
        if not user_id:
            if not email:
                raise ValueError("user_id 또는 email 중 하나는 필요합니다.")
            user_id = self.resolve_user_id(email)
            if not user_id:
                raise RuntimeError(f"이메일 '{email}' 에 해당하는 계정을 찾을 수 없습니다.")

        # 004 마이그레이션(alert_prefs / stop_loss·take_profit) 적용 전이면 확장 컬럼 select 가
        # 실패하므로 기본 컬럼으로 폴백 — 마이그레이션 전후 모두 동작하도록.
        try:
            cs = self._get(
                f"rest/v1/user_settings?user_id=eq.{user_id}&select=cash_balance,alert_prefs"
            )
        except Exception:
            cs = self._get(f"rest/v1/user_settings?user_id=eq.{user_id}&select=cash_balance")
        cash        = float(cs[0]["cash_balance"]) if cs else 0.0
        alert_prefs = (cs[0].get("alert_prefs") if cs else None) or {}

        try:
            rows = self._get(
                f"rest/v1/portfolios?user_id=eq.{user_id}"
                "&select=ticker,avg_price,quantity,stop_loss,take_profit&order=added_at.desc"
            )
        except Exception:
            rows = self._get(
                f"rest/v1/portfolios?user_id=eq.{user_id}"
                "&select=ticker,avg_price,quantity&order=added_at.desc"
            )
        names = self._resolve_names([r["ticker"] for r in rows])

        positions: dict[str, dict] = {}
        for r in rows:
            t = r["ticker"]
            pos = {
                "name":        names.get(t, t),
                "entry_price": float(r["avg_price"]),
                "quantity":    float(r["quantity"]),
            }
            if r.get("stop_loss") is not None:
                pos["sl"] = float(r["stop_loss"])      # 실제 MTS 손절가
            if r.get("take_profit") is not None:
                pos["tp"] = float(r["take_profit"])    # 실제 MTS 익절가
            positions[t] = pos
        return {
            "cash": cash, "positions": positions,
            "alert_prefs": alert_prefs, "user_id": user_id,
        }


if __name__ == "__main__":
    # 빠른 자체 점검: python supabase_account.py [user_id|email]
    import sys
    acc = SupabaseAccount()
    print("enabled:", acc.enabled, "| url set:", bool(acc.url))
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        h = acc.load_holdings(user_id=arg if "-" in arg else None,
                              email=None if "-" in arg else arg)
        print(f"예수금: {h['cash']:,.0f}원 · 보유 {len(h['positions'])}종목")
        for t, p in h["positions"].items():
            print(f"  {t} {p['name']}  진입 {p['entry_price']:,.0f} × {p['quantity']:g}")
