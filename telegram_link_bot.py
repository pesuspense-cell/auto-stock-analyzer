"""
telegram_link_bot.py — 텔레그램 딥링크 계정 연동 리스너 (다중 사용자)

장기 실행하며 getUpdates 롱폴링으로 '/start <token>' 메시지를 처리한다.
웹에서 발급한 일회용 토큰으로 Supabase user_settings 를 역조회해, 메시지를 보낸
텔레그램 chat_id 를 그 사용자에 저장하고 알림을 켠 뒤(토큰 소거) 완료 메시지를 보낸다.

  웹: "텔레그램 연동" → 토큰 발급(user_settings.telegram_link_token) → t.me/<bot>?start=<token>
  사용자: 링크 진입 후 [시작] → 봇이 '/start <token>' + chat_id 수신
  이 리스너: token→user 매칭 → telegram_chat_id 저장 + telegram_enabled=true + token=null

실행:  python telegram_link_bot.py
필요:  .env 의 TELEGRAM_BOT_TOKEN, Supabase 자격(루트 .env 또는 web/frontend/.env.local)
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

from supabase_account import SupabaseAccount

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S",
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
)
logger = logging.getLogger("telegram_link_bot")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_API             = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
OFFSET_FILE        = Path(__file__).parent / ".telegram_offset"


class TelegramLinker:
    def __init__(self):
        self.acc = SupabaseAccount()
        if not TELEGRAM_BOT_TOKEN:
            raise RuntimeError("TELEGRAM_BOT_TOKEN 미설정 — .env 확인")
        if not self.acc.enabled:
            raise RuntimeError("Supabase 자격 미설정 — .env / web/frontend/.env.local 확인")

    # ── Telegram API ─────────────────────────────────────────────────────────
    def _tg_get(self, method: str, params: dict) -> dict:
        url = f"{TG_API}/{method}?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url, timeout=60) as r:
            return json.load(r)

    def _tg_send(self, chat_id: str | int, text: str) -> None:
        data = urllib.parse.urlencode({"chat_id": chat_id, "text": text, "parse_mode": "HTML"}).encode()
        try:
            with urllib.request.urlopen(f"{TG_API}/sendMessage", data=data, timeout=15) as r:
                json.load(r)
        except Exception as e:
            logger.warning(f"답장 발송 실패(chat {chat_id}): {e}")

    # ── Supabase REST (service_role) ──────────────────────────────────────────
    def _sb(self, method: str, path: str, body: dict | None = None) -> list:
        req = urllib.request.Request(
            f"{self.acc.url}/{path}", method=method,
            data=json.dumps(body).encode() if body is not None else None,
            headers={
                "apikey": self.acc.key, "Authorization": f"Bearer {self.acc.key}",
                "Content-Type": "application/json", "Prefer": "return=representation",
            },
        )
        with urllib.request.urlopen(req, timeout=20) as r:
            txt = r.read().decode()
            return json.loads(txt) if txt else []

    def _link_user(self, token: str, chat_id: str) -> bool:
        """토큰→사용자 매칭 후 chat_id 저장 + 알림 ON + 토큰 소거. 성공 시 True."""
        rows = self._sb("GET", f"rest/v1/user_settings?telegram_link_token=eq.{urllib.parse.quote(token)}&select=user_id")
        if not rows:
            return False
        uid = rows[0]["user_id"]
        self._sb("PATCH", f"rest/v1/user_settings?user_id=eq.{uid}", {
            "telegram_chat_id": str(chat_id),
            "telegram_enabled": True,
            "telegram_link_token": None,
        })
        logger.info(f"연동 완료: user {uid[:8]}… ← chat {chat_id}")
        return True

    # ── 메시지 처리 ────────────────────────────────────────────────────────────
    def _handle(self, update: dict) -> None:
        msg = update.get("message") or update.get("edited_message") or {}
        text = (msg.get("text") or "").strip()
        chat = msg.get("chat", {})
        chat_id = chat.get("id")
        if not chat_id or not text.startswith("/start"):
            return
        parts = text.split(maxsplit=1)
        token = parts[1].strip() if len(parts) > 1 else ""
        if not token:
            self._tg_send(chat_id,
                "👋 안녕하세요. 계정 연동은 <b>웹사이트의 '텔레그램 연동' 버튼</b>으로 "
                "생성된 링크를 통해 진행해 주세요.")
            return
        try:
            ok = self._link_user(token, str(chat_id))
        except Exception as e:
            logger.error(f"연동 처리 오류: {e}")
            self._tg_send(chat_id, "⚠️ 연동 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.")
            return
        if ok:
            self._tg_send(chat_id,
                "✅ <b>텔레그램 연동 완료!</b>\n이제 이 채팅으로 회원님 계좌 기준 매매 시그널을 보내드립니다.\n"
                "<i>알림 on/off 는 웹 '시그널 봇 알림 설정'에서 바꿀 수 있습니다.</i>")
        else:
            self._tg_send(chat_id,
                "⚠️ 유효하지 않거나 만료된 연동 링크입니다. 웹에서 '텔레그램 연동'을 다시 눌러 새 링크를 받아주세요.")

    # ── 롱폴링 루프 ────────────────────────────────────────────────────────────
    def _load_offset(self) -> int:
        try:
            return int(OFFSET_FILE.read_text().strip())
        except Exception:
            return 0

    def _save_offset(self, offset: int) -> None:
        try:
            OFFSET_FILE.write_text(str(offset))
        except Exception:
            pass

    def run(self) -> None:
        offset = self._load_offset()
        logger.info("텔레그램 연동 리스너 시작 — '/start <token>' 대기 중 (Ctrl+C 종료)")
        while True:
            try:
                resp = self._tg_get("getUpdates", {"offset": offset, "timeout": 50})
                for upd in resp.get("result", []):
                    offset = upd["update_id"] + 1
                    try:
                        self._handle(upd)
                    except Exception as e:
                        logger.error(f"업데이트 처리 오류: {e}")
                    self._save_offset(offset)
            except KeyboardInterrupt:
                logger.info("종료(Ctrl+C)")
                break
            except urllib.error.HTTPError as e:
                logger.error(f"Telegram HTTP 오류: {e} — 5초 후 재시도")
                time.sleep(5)
            except Exception as e:
                logger.error(f"폴링 오류: {e} — 5초 후 재시도")
                time.sleep(5)


if __name__ == "__main__":
    TelegramLinker().run()
