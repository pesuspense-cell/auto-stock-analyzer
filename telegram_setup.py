"""
telegram_setup.py — Telegram 봇 설정 마법사

실행:  python telegram_setup.py

단계:
  1. 봇 토큰 입력 (BotFather에서 발급)
  2. 봇 유효성 검증 (getMe)
  3. Chat ID 자동 조회 (getUpdates)  ← 봇에게 /start 메시지 먼저 보내야 함
  4. 테스트 메시지 발송
  5. .env 파일 저장
"""
import asyncio
import os
import sys
from pathlib import Path

try:
    import aiohttp
except ImportError:
    print("❌ aiohttp 미설치. 다음 명령 실행: pip install aiohttp")
    sys.exit(1)

try:
    from dotenv import load_dotenv, set_key
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False

ENV_FILE = Path(__file__).parent / ".env"
BASE_URL = "https://api.telegram.org"


# ─── Telegram API helpers ───────────────────────────────────────────────────

async def api_get(session: aiohttp.ClientSession, token: str, method: str, **params):
    url = f"{BASE_URL}/bot{token}/{method}"
    async with session.get(url, params=params or None) as resp:
        return await resp.json()


async def api_post(session: aiohttp.ClientSession, token: str, method: str, json: dict):
    url = f"{BASE_URL}/bot{token}/{method}"
    async with session.post(url, json=json) as resp:
        return await resp.json()


# ─── 단계별 함수 ────────────────────────────────────────────────────────────

async def step1_verify_token(session: aiohttp.ClientSession, token: str) -> dict | None:
    """봇 토큰 검증"""
    print("\n🔍 봇 토큰 검증 중...")
    try:
        resp = await api_get(session, token, "getMe")
        if resp.get("ok"):
            bot = resp["result"]
            print(f"  ✅ 봇 확인: @{bot['username']} ({bot['first_name']})")
            return bot
        else:
            print(f"  ❌ 토큰 오류: {resp.get('description', '알 수 없음')}")
            return None
    except Exception as e:
        print(f"  ❌ 연결 실패: {e}")
        return None


async def step2_get_chat_id(session: aiohttp.ClientSession, token: str) -> str | None:
    """Chat ID 자동 조회"""
    print("\n📬 Chat ID 조회 중...")
    print("  → Telegram에서 해당 봇에게 /start 를 보낸 뒤 Enter를 누르세요.")
    input("     (준비되면 Enter) ")

    resp = await api_get(session, token, "getUpdates", limit=10, timeout=0)
    if not resp.get("ok"):
        print(f"  ❌ getUpdates 오류: {resp.get('description')}")
        return None

    updates = resp.get("result", [])
    if not updates:
        print("  ⚠️  수신된 메시지 없음.")
        print("     봇에게 /start 메시지를 보내고 다시 시도하세요.")
        return None

    # 가장 최근 메시지에서 chat_id 추출
    last = updates[-1]
    msg = last.get("message") or last.get("channel_post") or {}
    chat = msg.get("chat", {})
    chat_id = str(chat.get("id", ""))
    chat_name = chat.get("title") or chat.get("username") or chat.get("first_name", "")

    if chat_id:
        print(f"  ✅ Chat ID: {chat_id}  ({chat_name})")
        return chat_id
    else:
        print("  ❌ Chat ID를 추출할 수 없습니다.")
        return None


async def step3_send_test(session: aiohttp.ClientSession, token: str, chat_id: str) -> bool:
    """테스트 메시지 발송"""
    print("\n📤 테스트 메시지 발송 중...")
    html = (
        "✅ <b>Auto Stock Analyzer 연결 성공!</b>\n\n"
        "이 봇은 다음 신호를 실시간으로 알려드립니다:\n"
        "🟢 <b>BUY</b>  — 매수 신호\n"
        "🔴 <b>SELL</b> — 매도 신호\n"
        "🟡 <b>WAIT</b> — 관망 신호\n"
        "⚠️ 급등락 경보\n\n"
        "<i>설정이 완료되었습니다. docker-compose up -d 로 시스템을 시작하세요.</i>"
    )
    resp = await api_post(session, token, "sendMessage", {
        "chat_id": chat_id,
        "text": html,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    })
    if resp.get("ok"):
        print("  ✅ 테스트 메시지 발송 성공!")
        return True
    else:
        print(f"  ❌ 발송 실패: {resp.get('description')}")
        return False


def step4_save_env(token: str, chat_id: str):
    """환경 변수 .env 파일에 저장"""
    print("\n💾 .env 파일 저장 중...")

    if HAS_DOTENV and ENV_FILE.exists():
        set_key(str(ENV_FILE), "TELEGRAM_BOT_TOKEN", token)
        set_key(str(ENV_FILE), "TELEGRAM_CHAT_ID", chat_id)
        print(f"  ✅ 기존 .env 업데이트: {ENV_FILE}")
    else:
        # 새로 작성 (기존 내용 유지)
        lines = []
        if ENV_FILE.exists():
            existing = ENV_FILE.read_text(encoding="utf-8").splitlines()
            for line in existing:
                if not line.startswith("TELEGRAM_BOT_TOKEN=") and \
                   not line.startswith("TELEGRAM_CHAT_ID="):
                    lines.append(line)
        else:
            # .env.example 복사
            example = Path(__file__).parent / ".env.example"
            if example.exists():
                lines = example.read_text(encoding="utf-8").splitlines()

        lines.append(f"TELEGRAM_BOT_TOKEN={token}")
        lines.append(f"TELEGRAM_CHAT_ID={chat_id}")
        ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"  ✅ .env 생성: {ENV_FILE}")


# ─── 메인 ───────────────────────────────────────────────────────────────────

async def main():
    print("=" * 60)
    print("  Telegram 봇 설정 마법사")
    print("=" * 60)
    print()
    print("📋 사전 준비:")
    print("  1. Telegram에서 @BotFather 검색")
    print("  2. /newbot 입력 → 봇 이름 / 사용자명 입력")
    print("  3. Bot Token 복사 (예: 123456:ABC-DEF...)")
    print()

    # 기존 토큰 확인
    if HAS_DOTENV and ENV_FILE.exists():
        load_dotenv(ENV_FILE)
    existing_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    existing_chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

    # 토큰 입력
    if existing_token and existing_token != "your_bot_token_here":
        use_existing = input(
            f"기존 토큰 발견: {existing_token[:20]}... 재사용? [Y/n] "
        ).strip().lower()
        token = existing_token if use_existing != "n" else input("새 Bot Token: ").strip()
    else:
        token = input("Bot Token 입력: ").strip()

    if not token:
        print("❌ 토큰이 없습니다. 종료합니다.")
        sys.exit(1)

    timeout = aiohttp.ClientTimeout(total=15)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # 1. 토큰 검증
        bot_info = await step1_verify_token(session, token)
        if not bot_info:
            print("\n❌ 유효하지 않은 Bot Token입니다.")
            sys.exit(1)

        # 2. Chat ID 조회
        if existing_chat_id and existing_chat_id != "your_chat_id_here":
            use_existing = input(
                f"\n기존 Chat ID 발견: {existing_chat_id}. 재사용? [Y/n] "
            ).strip().lower()
            if use_existing != "n":
                chat_id = existing_chat_id
                print(f"  ✅ Chat ID 재사용: {chat_id}")
            else:
                chat_id = await step2_get_chat_id(session, token)
        else:
            chat_id = await step2_get_chat_id(session, token)

        if not chat_id:
            # 수동 입력 fallback
            chat_id = input("\nChat ID를 직접 입력하세요 (없으면 Enter 스킵): ").strip()
            if not chat_id:
                print("❌ Chat ID 없이 계속할 수 없습니다.")
                sys.exit(1)

        # 3. 테스트 메시지
        ok = await step3_send_test(session, token, chat_id)
        if not ok:
            print("\n⚠️  테스트 실패했지만 설정은 저장합니다.")

    # 4. .env 저장
    step4_save_env(token, chat_id)

    print("\n" + "=" * 60)
    print("  설정 완료!")
    print("=" * 60)
    print()
    print("다음 명령으로 시스템 시작:")
    print("  docker-compose up -d")
    print()
    print("또는 로컬 실행:")
    print("  python analysis_worker.py")


if __name__ == "__main__":
    asyncio.run(main())
