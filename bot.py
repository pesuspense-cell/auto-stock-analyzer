"""
bot.py — 텔레그램 대화형 트레이딩 비서

명령어 (한국어):
  /분석 [종목명]  — 종목 분석 리포트
  /관심 [종목명]  — 관심종목 추가
  /현황          — 관심종목 및 포지션 요약
  /도움          — 명령어 안내

영어 별칭 (텔레그램 커맨드 메뉴 등록용):
  /analyze, /watch, /status, /help, /start

실행:
  python bot.py

필수 환경 변수 (.env 또는 export):
  TELEGRAM_BOT_TOKEN=<BotFather 발급 토큰>
  TELEGRAM_CHAT_ID=<허용할 Chat ID, 비워두면 전체 허용>

주의:
  - 한국어 명령어(/분석 등)는 Telegram API 레벨에서 'command'로 인식 안 되므로
    MessageHandler + Regex 방식으로 처리함
  - 분석 함수(get_stock_data 등)는 동기 블로킹 → asyncio.to_thread 로 래핑
"""
from __future__ import annotations

import asyncio
import html
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# ── 프로젝트 모듈 ─────────────────────────────────────────────────────────────
from strategy import TradingStrategy

try:
    from stock_ai import (
        get_stock_data,
        generate_signals,
        get_krx_stock_list,
    )
    HAS_STOCK_AI = True
except ImportError:
    HAS_STOCK_AI = False

try:
    import redis.asyncio as aioredis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False

# ── 설정 ─────────────────────────────────────────────────────────────────────
TOKEN          = os.getenv("TELEGRAM_BOT_TOKEN", "")
ALLOWED_CHAT   = os.getenv("TELEGRAM_CHAT_ID", "")   # 비워두면 전체 허용
REDIS_HOST     = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT     = int(os.getenv("REDIS_PORT", 6379))
WATCHLIST_FILE = Path(__file__).parent / "watchlist.json"

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("bot")

strategy = TradingStrategy()

# ── 종목명 → ticker 캐시 ──────────────────────────────────────────────────────
_NAME_MAP: dict[str, str] = {}
_NAME_MAP_LOADED = False


def _load_name_map() -> None:
    global _NAME_MAP, _NAME_MAP_LOADED
    if _NAME_MAP_LOADED:
        return
    try:
        krx = get_krx_stock_list()        # {"삼성전자 (005930)": "005930.KS", ...}
        for display, ticker in krx.items():
            name = display.split(" (")[0].strip()
            _NAME_MAP[name] = ticker
            _NAME_MAP[name.replace(" ", "")] = ticker
    except Exception as e:
        logger.warning(f"KRX 종목 목록 로드 실패: {e}")
    _NAME_MAP_LOADED = True


def resolve_ticker(query: str) -> tuple[str, str] | None:
    """
    종목명·코드 → (ticker, display_name).
    예: "삼성전자" → ("005930.KS", "삼성전자")
        "005930"   → ("005930.KS", "005930")
        "AAPL"     → ("AAPL", "AAPL")
    """
    q = query.strip()

    # 6자리 숫자 → KS 자동 추가
    if re.fullmatch(r"\d{6}", q):
        return f"{q}.KS", q

    # 이미 ticker 형식 (005930.KS / 005930.KQ / AAPL)
    if re.fullmatch(r"\d{6}\.(KS|KQ)", q) or re.fullmatch(r"[A-Z]{1,5}", q):
        return q, q

    _load_name_map()

    # 정확히 일치
    if q in _NAME_MAP:
        return _NAME_MAP[q], q

    # 공백 제거 후 일치
    nospace = q.replace(" ", "")
    if nospace in _NAME_MAP:
        return _NAME_MAP[nospace], q

    # 부분 일치 (첫 번째 결과)
    for name, ticker in _NAME_MAP.items():
        if q in name:
            return ticker, name

    return None


# ── 관심종목 I/O (watchlist.json ↔ bot) ────────────────────────────────────
def _load_watchlist() -> list[dict]:
    try:
        if WATCHLIST_FILE.exists():
            return json.loads(WATCHLIST_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return []


def _save_watchlist(wl: list[dict]) -> None:
    WATCHLIST_FILE.write_text(
        json.dumps(wl, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ── HTML 포맷 헬퍼 ────────────────────────────────────────────────────────────
def h(text) -> str:
    """HTML 이스케이프"""
    return html.escape(str(text))


def _price_str(price: float, ticker: str) -> str:
    is_krx = ticker.endswith(".KS") or ticker.endswith(".KQ")
    return f"{price:,.0f}{'원' if is_krx else '$'}"


async def _send(msg, text: str) -> None:
    """4096자 초과 시 자동 분할 발송"""
    for i in range(0, max(len(text), 1), 4096):
        await msg.reply_text(text[i : i + 4096], parse_mode=ParseMode.HTML)


# ── 분석 리포트 포맷 ─────────────────────────────────────────────────────────
def _build_report(
    ticker: str,
    display: str,
    data,
    signals: dict,
    atr: float,
    kill_switch: bool,
    entry_ok: bool,
    tp: float,
    sl: float,
) -> str:
    last = data.iloc[-1]
    prev = data.iloc[-2]
    price     = float(last["Close"])
    prev_close = float(prev["Close"])
    chg_pct   = (price - prev_close) / prev_close * 100 if prev_close else 0
    is_krx    = ticker.endswith(".KS") or ticker.endswith(".KQ")

    chg_arrow = "📈" if chg_pct >= 0 else "📉"
    chg_sign  = "+" if chg_pct >= 0 else ""

    rsi       = float(last.get("RSI", 0))
    macd_hist = float(last.get("MACD_Hist", 0))
    bb_pct    = float(last.get("BB_PCT", 0.5))
    adx       = float(last.get("ADX", 0))
    sma5      = float(last.get("SMA_5", 0))
    sma20     = float(last.get("SMA_20", 0))
    sma60     = float(last.get("SMA_60", 0))

    rsi_status = "과매수 ⚠️" if rsi > 70 else ("과매도 ⚠️" if rsi < 30 else "중립")
    macd_dir   = "상승" if macd_hist > 0 else "하락"
    bb_pos     = "상단 근접" if bb_pct > 0.8 else ("하단 근접" if bb_pct < 0.2 else "중앙")

    label  = signals.get("label", "HOLD")
    badge  = signals.get("badge", "⚪")
    score  = signals.get("score_int", 0)
    reasons = signals.get("reasons", [])

    atr_str = f"{atr:,.0f}" if is_krx else f"{atr:.4f}"
    tp_str  = _price_str(tp, ticker) if tp > 0 else "N/A"
    sl_str  = _price_str(sl, ticker) if sl > 0 else "N/A"

    lines = [
        f"📊 <b>{h(display)}</b>  <code>{h(ticker)}</code>",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"{chg_arrow} <b>현재가</b>  <b>{h(_price_str(price, ticker))}</b>"
        f"  ({chg_sign}{chg_pct:.2f}%)",
        "",
        "📈 <b>기술적 지표</b>",
        f"├ RSI: <code>{rsi:.1f}</code>  {rsi_status}",
        f"├ MACD Hist: <code>{macd_hist:.2f}</code>  ({macd_dir})",
        f"├ SMA 5/20/60: <code>{sma5:,.0f}</code> / <code>{sma20:,.0f}</code> / <code>{sma60:,.0f}</code>",
        f"├ 볼린저밴드: {bb_pos}  (<code>{bb_pct:.0%}</code>)",
        f"└ ADX: <code>{adx:.1f}</code>",
        "",
        "🎯 <b>매매 전략 신호</b>",
        f"├ 종합: <b>{h(badge + ' ' + label)}</b>  (점수 {score:+d})",
        f"├ Kill-Switch: {'🔴 발동 — 신규 진입 차단' if kill_switch else '🟢 비활성'}",
        f"├ ATR: <code>{atr_str}</code>",
        f"├ 익절 목표가: <code>{tp_str}</code>",
        f"└ 손절 가격: <code>{sl_str}</code>",
    ]

    if entry_ok:
        lines += ["", "✅ <b>진입 신호 감지</b>  — SMA 골든크로스 + 상승 기울기"]

    if reasons:
        lines += ["", "📋 <b>신호 근거</b>"]
        for r in reasons[:5]:
            lines.append(f"• {h(str(r))}")

    lines += ["", f"⏰ <i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>"]
    return "\n".join(lines)


# ── 인증 확인 ────────────────────────────────────────────────────────────────
def _is_allowed(update: Update) -> bool:
    if not ALLOWED_CHAT:
        return True
    return str(update.effective_chat.id) == ALLOWED_CHAT


# ── 명령어 파싱 (한국어/영어 혼용) ───────────────────────────────────────────
def _parse_args(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    """
    '/분석 삼성전자'  또는  '/analyze 삼성전자' 모두에서 '삼성전자' 추출.
    """
    text = (update.effective_message.text or "").strip()
    parts = text.split(maxsplit=1)
    if len(parts) > 1:
        return parts[1].strip()
    return " ".join(context.args or [])


# ── 핸들러: /분석 ─────────────────────────────────────────────────────────────
async def analyze_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_allowed(update):
        return

    msg   = update.effective_message
    query = _parse_args(update, context)

    if not query:
        await msg.reply_text(
            "사용법: <b>/분석</b> [종목명 또는 티커]\n"
            "예)  /분석 삼성전자\n"
            "     /분석 005930\n"
            "     /분석 AAPL",
            parse_mode=ParseMode.HTML,
        )
        return

    await msg.chat.send_action(ChatAction.TYPING)

    resolved = resolve_ticker(query)
    if not resolved:
        await msg.reply_text(
            f"❌ 종목을 찾을 수 없습니다: <code>{h(query)}</code>\n"
            "티커 코드로 다시 시도해보세요 (예: 005930, 000660.KS)",
            parse_mode=ParseMode.HTML,
        )
        return

    ticker, display = resolved

    try:
        # 블로킹 I/O → 별도 스레드
        data = await asyncio.to_thread(get_stock_data, ticker, "3mo")
        await msg.chat.send_action(ChatAction.TYPING)

        if data is None or data.empty or len(data) < 30:
            await msg.reply_text(
                f"❌ 데이터 부족: <code>{h(ticker)}</code>\n"
                "상장 기간이 짧거나 올바르지 않은 티커일 수 있습니다.",
                parse_mode=ParseMode.HTML,
            )
            return

        signals = await asyncio.to_thread(generate_signals, data)
        await msg.chat.send_action(ChatAction.TYPING)

        close = data["Close"]
        high  = data["High"]
        low   = data["Low"]

        sma5_curr, sma5_prev   = strategy.compute_sma(close, 5)
        sma20_curr, _          = strategy.compute_sma(close, 20)
        atr                    = strategy.compute_atr(high, low, close)
        kill_switch            = strategy.check_kill_switch(close)
        entry_ok               = (
            strategy.is_entry_signal(sma5_curr, sma20_curr, sma5_prev)
            and not kill_switch
        )
        price = float(close.iloc[-1])
        tp = strategy.get_exit_price(price, atr, is_profit=True)  if atr > 0 else -1.0
        sl = strategy.get_exit_price(price, atr, is_profit=False) if atr > 0 else -1.0

        report = _build_report(ticker, display, data, signals, atr, kill_switch, entry_ok, tp, sl)
        await _send(msg, report)

    except Exception as e:
        logger.error(f"분석 오류 [{ticker}]: {e}", exc_info=True)
        await msg.reply_text(
            f"❌ 분석 중 오류 발생\n<code>{h(str(e)[:200])}</code>",
            parse_mode=ParseMode.HTML,
        )


# ── 핸들러: /관심 ─────────────────────────────────────────────────────────────
async def watch_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_allowed(update):
        return

    msg   = update.effective_message
    query = _parse_args(update, context)

    if not query:
        await msg.reply_text(
            "사용법: <b>/관심</b> [종목명 또는 티커]\n예) /관심 삼성전자",
            parse_mode=ParseMode.HTML,
        )
        return

    await msg.chat.send_action(ChatAction.TYPING)

    resolved = resolve_ticker(query)
    if not resolved:
        await msg.reply_text(
            f"❌ 종목을 찾을 수 없습니다: <code>{h(query)}</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    ticker, display = resolved
    wl = _load_watchlist()

    # 중복 확인
    if any(w["ticker"] == ticker for w in wl):
        await msg.reply_text(
            f"ℹ️  <b>{h(display)}</b> (<code>{h(ticker)}</code>)은 이미 관심종목입니다.",
            parse_mode=ParseMode.HTML,
        )
        return

    wl.append({"name": display, "ticker": ticker})
    _save_watchlist(wl)
    logger.info(f"관심종목 추가: {ticker}")

    await msg.reply_text(
        f"✅ 관심종목 추가: <b>{h(display)}</b> (<code>{h(ticker)}</code>)\n"
        f"현재 관심종목: <b>{len(wl)}개</b>",
        parse_mode=ParseMode.HTML,
    )


# ── 핸들러: /현황 ─────────────────────────────────────────────────────────────
async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_allowed(update):
        return

    msg = update.effective_message
    await msg.chat.send_action(ChatAction.TYPING)

    wl = _load_watchlist()

    # Redis 포지션 조회
    positions: dict[str, dict] = {}
    if HAS_REDIS:
        try:
            r = await aioredis.from_url(
                f"redis://{REDIS_HOST}:{REDIS_PORT}/0", decode_responses=True
            )
            for item in wl:
                t = item["ticker"]
                pos = await r.hgetall(f"position:{t}")
                if pos.get("in_position") == "1":
                    positions[t] = pos
            await r.aclose()
        except Exception as e:
            logger.warning(f"Redis 조회 실패: {e}")

    if not wl:
        await msg.reply_text(
            "📋 관심종목이 없습니다.\n<b>/관심 [종목명]</b> 으로 추가하세요.",
            parse_mode=ParseMode.HTML,
        )
        return

    # 관심종목별 빠른 신호 조회 (병렬)
    async def _quick_signal(item: dict) -> tuple[str, str, str, float]:
        ticker  = item["ticker"]
        display = item["name"]
        try:
            data = await asyncio.to_thread(get_stock_data, ticker, "1mo")
            if data is None or data.empty:
                return ticker, display, "⚪ 조회 실패", 0.0
            sig   = generate_signals(data)
            price = float(data["Close"].iloc[-1])
            return ticker, display, f"{sig.get('badge','⚪')} {sig.get('label','')}", price
        except Exception:
            return ticker, display, "⚪ 오류", 0.0

    await msg.chat.send_action(ChatAction.TYPING)
    results = await asyncio.gather(*[_quick_signal(w) for w in wl])

    lines = [
        f"📋 <b>현황</b>  <i>{datetime.now().strftime('%m/%d %H:%M')}</i>",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"👀 <b>관심종목 {len(wl)}개</b>",
    ]

    for i, (ticker, display, sig_label, price) in enumerate(results):
        is_krx    = ticker.endswith(".KS") or ticker.endswith(".KQ")
        price_str = f"{price:,.0f}{'원' if is_krx else '$'}" if price else "—"
        connector = "└" if i == len(results) - 1 else "├"
        pos_mark  = " 📂" if ticker in positions else ""
        lines.append(
            f"{connector} <b>{h(display)}</b>  {h(sig_label)}  "
            f"<code>{price_str}</code>{pos_mark}"
        )

    # 보유 포지션 상세
    if positions:
        lines += ["", f"📂 <b>보유 포지션 {len(positions)}개</b>"]
        for i, (ticker, pos) in enumerate(positions.items()):
            entry  = float(pos.get("entry_price", 0))
            rebuy  = int(pos.get("rebuy_count", 0))
            name   = next((w["name"] for w in wl if w["ticker"] == ticker), ticker)
            connector = "└" if i == len(positions) - 1 else "├"
            lines.append(
                f"{connector} <b>{h(name)}</b>  진입가 <code>{_price_str(entry, ticker)}</code>"
                f"  분할매수 <code>{rebuy}회</code>"
            )

    await _send(msg, "\n".join(lines))


# ── 핸들러: /도움 ─────────────────────────────────────────────────────────────
async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_allowed(update):
        return

    text = (
        "🤖 <b>트레이딩 비서 명령어</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "📊 <b>/분석</b> [종목명]\n"
        "   기술적 지표 + 전략 신호 리포트\n"
        "   예) /분석 삼성전자\n\n"
        "⭐ <b>/관심</b> [종목명]\n"
        "   관심종목 추가\n"
        "   예) /관심 SK하이닉스\n\n"
        "📋 <b>/현황</b>\n"
        "   관심종목 신호 요약 + 보유 포지션\n\n"
        "💡 티커 직접 입력도 가능합니다\n"
        "   예) /분석 005930  /분석 AAPL"
    )
    await update.effective_message.reply_text(text, parse_mode=ParseMode.HTML)


# ── Application 설정 및 실행 ──────────────────────────────────────────────────
def main() -> None:
    if not TOKEN:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN 환경변수가 없습니다.\n"
            ".env 파일에 TELEGRAM_BOT_TOKEN=... 을 설정하세요."
        )

    app = Application.builder().token(TOKEN).build()

    # 한국어 명령어 (MessageHandler + Regex)
    for pattern, handler in [
        (r"^/분석",  analyze_cmd),
        (r"^/관심",  watch_cmd),
        (r"^/현황",  status_cmd),
        (r"^/도움",  help_cmd),
    ]:
        app.add_handler(MessageHandler(filters.TEXT & filters.Regex(pattern), handler))

    # 영어 별칭 (텔레그램 커맨드 메뉴 등록용)
    app.add_handler(CommandHandler(["analyze", "a"], analyze_cmd))
    app.add_handler(CommandHandler(["watch",   "w"], watch_cmd))
    app.add_handler(CommandHandler(["status",  "s"], status_cmd))
    app.add_handler(CommandHandler(["help", "start"], help_cmd))

    logger.info("=" * 60)
    logger.info("  트레이딩 비서 봇 시작")
    logger.info("=" * 60)
    logger.info("명령어: /분석 /관심 /현황 /도움")
    logger.info("영어:   /analyze /watch /status /help")

    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
