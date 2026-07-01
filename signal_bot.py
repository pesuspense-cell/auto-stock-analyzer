"""
signal_bot.py — 실시간 매매 시그널 텔레그램 알림 봇 (자동 주문 없음 / 알림 전용)

검증 완료된 백테스트 v4 엔진(live_screener.py 계승)의 매수·매도·리밸런싱 조건을
장중 주기적으로 감시하고, 신호가 포착되면 텔레그램으로 '행동 가이드라인' 메시지를
즉시 전송한다. 증권사 자동 주문은 일절 수행하지 않으며, 사용자가 알림을 보고 직접
MTS로 매매한다.

┌──────────────────────────────────────────────────────────────────────────┐
│ [v4 엔진 실시간 데이터(일봉+장중 형성봉)] → [조건 감시: 골든크로스/눌림목/청산]│
│   → [비중 계산: 총자산 15%캡·노출 45%상한] → [텔레그램 알림] → [수동 매매]    │
└──────────────────────────────────────────────────────────────────────────┘

감시 신호:
  🚨 BUY        — SMA 골든크로스(야수) / 거래량 눌림목(디펜스) 신규 진입
  💵✅ SELL_TP  — 익절 목표가 도달
  💵❌ SELL_SL  — 손절선 이탈
  🎯  TS_ARM    — 수익 +12% 돌파 → 손절선 본전(+0.5%)으로 상향 권고(이익 보호 발동)
  🎯✅ SELL_TS  — 트레일링 스톱 청산(고점 대비 하락)
  ♻️  SELL_REBAL— 총 주식노출 45% 상한 초과 → 부분매도 리밸런싱

보유 종목(매도/리밸런싱 판정 기준)은 live_screener.MY_CURRENT_BALANCE 에서 읽는다.
매수 체결 후에는 그 파일의 positions 를 직접 갱신하세요(자동 추종 아님). 봇은 휘발성
런타임 상태(고점가·트레일링 발동 여부·중복발송 방지 키)만 signal_bot_state.json 에 저장한다.

실행:
  pip install -r requirements.txt           # python-telegram-bot, schedule, pytz 포함
  # .env 에 TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID 설정(telegram_setup.py 마법사 가능)
  python signal_bot.py                       # 장중 자동 감시 루프
  python signal_bot.py --once                # 즉시 1회 점검(테스트)
  python signal_bot.py --test                # 텔레그램 연결 테스트 메시지만 발송
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, time as dtime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

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

# ── 검증된 v4 엔진 로직 재사용 (단일 출처) ───────────────────────────────────
from live_screener import (
    MarketAnalyzer, run_screening, calc_buy_plan, _build_universe, _prepare_df,
    MY_CURRENT_BALANCE,
    MAX_POS_BULL, MAX_POS_BEAR, MAX_GROSS_EXPOSURE, SL_SLIPPAGE_PCT,
    ATR_SL_MULT_BULL, BULL_SL_FLOOR_PCT, SL_FALLBACK_PCT,
    RiskConfig, risk_config, RISK_SAFE,
)
from strategy import TradingStrategy
from supabase_account import SupabaseAccount

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
)
logger = logging.getLogger("signal_bot")

# ══════════════════════════════════════════════════════════════════════════════
#  설정
# ══════════════════════════════════════════════════════════════════════════════

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")

POLL_INTERVAL_MIN  = int(os.getenv("SIGNAL_POLL_MIN", "5"))      # 감시 주기(분)
UNIVERSE_N         = int(os.getenv("SIGNAL_UNIVERSE_N", "100"))  # 시장별 스크리닝 유니버스 상위 N

# ── [2단계] 계좌 데이터 소스 — Supabase 로그인 계정 / manual(MY_CURRENT_BALANCE) ──
# supabase: 웹앱 로그인 계정의 예수금·보유종목을 Supabase에서 읽어 봇 계좌로 사용.
#           대상 계정은 SIGNAL_ACCOUNT_USER_ID(uuid) 또는 SIGNAL_ACCOUNT_EMAIL 로 지정.
# manual:   live_screener.MY_CURRENT_BALANCE(수동 잔고)를 그대로 사용(폴백 포함).
ACCOUNT_SOURCE  = os.getenv("SIGNAL_ACCOUNT_SOURCE", "supabase").lower()
ACCOUNT_USER_ID = os.getenv("SIGNAL_ACCOUNT_USER_ID", "")
ACCOUNT_EMAIL   = os.getenv("SIGNAL_ACCOUNT_EMAIL", "")


# ── [알림 on/off] 신호 종류별 발송 토글 (.env, 기본 전부 ON) ─────────────────
# 노이즈가 많은 신호(예: 풀매수 상태의 리밸런싱)를 종류별로 끌 수 있다. 꺼진 종류는
# 신호를 만들지도·중복방지 키를 남기지도 않으므로, 나중에 다시 켜면 즉시 정상 발송된다.
def _env_bool(name: str, default: bool = True) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on", "y")

# 트레일링 발동(TS_ARM)·트레일링 청산(SELL_TS)은 SIGNAL_ENABLE_SELL_TS 하나로 묶어 제어.
SIGNAL_TYPE_ENABLED: dict[str, bool] = {
    "BUY":        _env_bool("SIGNAL_ENABLE_BUY",     True),
    "SELL_TP":    _env_bool("SIGNAL_ENABLE_SELL_TP", True),
    "SELL_SL":    _env_bool("SIGNAL_ENABLE_SELL_SL", True),
    "TS_ARM":     _env_bool("SIGNAL_ENABLE_SELL_TS", True),
    "SELL_TS":    _env_bool("SIGNAL_ENABLE_SELL_TS", True),
    "SELL_REBAL": _env_bool("SIGNAL_ENABLE_REBAL",   True),
}


def _disabled_types_label() -> str:
    """꺼진 신호 종류 요약(로그·시작 알림용). 대표 5종 기준."""
    off = [k for k in ("BUY", "SELL_TP", "SELL_SL", "SELL_TS", "SELL_REBAL")
           if not SIGNAL_TYPE_ENABLED.get(k, True)]
    return ", ".join(off) if off else "없음(전부 ON)"


def resolve_enabled(prefs: dict | None) -> dict:
    """이 사용자용 신호 on/off 맵을 만든다(.env 기본값 위에 user_settings.alert_prefs 덮어씀).

    SIGNAL_TYPE_ENABLED(전역)를 변형하지 않고 '복사본'을 반환하므로 다중 사용자에서
    사용자별 설정이 서로 섞이지 않는다. 명시된 키만 덮어쓰고(없으면 .env 기본값 유지)
    SELL_TS 키는 트레일링 발동(TS_ARM)에도 함께 적용된다.
    """
    enabled = dict(SIGNAL_TYPE_ENABLED)
    for key in ("BUY", "SELL_TP", "SELL_SL", "SELL_TS", "SELL_REBAL"):
        if prefs and key in prefs:
            val = bool(prefs[key])
            enabled[key] = val
            if key == "SELL_TS":
                enabled["TS_ARM"] = val
    return enabled


def _user_substate(state: dict, user_id: str) -> dict:
    """사용자별 런타임/중복방지 상태 하위 dict(없으면 생성). 사용자 간 dedup 격리."""
    users = state.setdefault("users", {})
    return users.setdefault(user_id, {"runtime": {}, "sent": []})

# 장 운영 시간(한국 KST). 09:00 ~ 15:30.
MARKET_OPEN  = dtime(9, 0)
MARKET_CLOSE = dtime(15, 30)
KST_TZ       = "Asia/Seoul"

STATE_FILE = Path(__file__).parent / "signal_bot_state.json"

# ── 트레일링 스톱 파라미터 — 발동 마진은 위험성향(RiskConfig.trail_activate_profit)이
#    결정한다. 아래 상수는 cfg 미전달 시 폴백(안전형 v4.6 = +18%).
TRAIL_ACTIVATE_PROFIT = 0.18   # (폴백) 수익 +18% 최초 도달 시 트레일링 ON
TRAIL_BREAKEVEN_PCT   = 0.005  # 활성화 시 손절선을 본전 +0.5%(수수료 보전선)로 상향

_strategy = TradingStrategy()
_account: SupabaseAccount | None = None


def load_account() -> dict:
    """봇이 감시할 계좌(holdings)를 로드.

    SIGNAL_ACCOUNT_SOURCE=supabase 이면 웹앱 로그인 계정(Supabase)의 예수금·보유종목을
    읽어 사용하고, manual 이거나 로드 실패 시 live_screener.MY_CURRENT_BALANCE(수동 잔고)로
    안전하게 폴백한다. 반환 holdings 의 positions 는 sl/tp 가 없을 수 있으며(Supabase),
    그 경우 monitor_positions 가 진입가+현재ATR로 손절/익절선을 산출한다.
    """
    global _account
    if ACCOUNT_SOURCE == "supabase" and (ACCOUNT_USER_ID or ACCOUNT_EMAIL):
        try:
            if _account is None:
                _account = SupabaseAccount()
            if _account.enabled:
                h = _account.load_holdings(
                    user_id=ACCOUNT_USER_ID or None,
                    email=ACCOUNT_EMAIL or None,
                )
                logger.info(
                    f"계좌 로드(Supabase): 예수금 {h['cash']:,.0f}원 · 보유 {len(h['positions'])}종목"
                )
                return h
            logger.warning("Supabase 자격 미설정 → 수동 잔고(MY_CURRENT_BALANCE)로 폴백")
        except Exception as e:
            logger.error(f"Supabase 계좌 로드 실패 → 수동 잔고로 폴백: {e}")
    return MY_CURRENT_BALANCE


def _now_kst() -> datetime:
    try:
        import pytz
        return datetime.now(pytz.timezone(KST_TZ))
    except Exception:
        return datetime.now()


def _is_market_hours(now: datetime | None = None) -> bool:
    now = now or _now_kst()
    if now.weekday() >= 5:                       # 토(5)·일(6) 휴장
        return False
    return MARKET_OPEN <= now.time() <= MARKET_CLOSE


def _won(x: float) -> str:
    return f"{x:,.0f}원"


def _man(x: float) -> str:
    """원 → '약 NNN만 원' 가독 표기."""
    return f"약 {x/10_000:,.0f}만 원"


# ══════════════════════════════════════════════════════════════════════════════
#  텔레그램 발송기 (python-telegram-bot 우선, 실패 시 HTTP 폴백)
# ══════════════════════════════════════════════════════════════════════════════

class TelegramNotifier:
    """동기 루프(schedule)에서 호출하는 텔레그램 발송기.

    python-telegram-bot(v20+, async)을 우선 사용하되 단일 이벤트 루프를 재사용해
    매 발송마다 루프를 새로 만드는 오버헤드를 피한다. 라이브러리 미설치 시 표준
    urllib HTTP 호출로 폴백한다(추가 의존성 없이도 동작).
    """

    def __init__(self, token: str, chat_id: str = "", dry_run: bool = False):
        self.token   = token
        self.chat_id = chat_id                       # 기본(폴백) chat_id — 다중사용자에선 send(chat_id=)로 지정
        self.dry_run = dry_run or not token          # 토큰만 있으면 발송 가능(수신자는 호출 시 지정)
        self._bot    = None
        self._loop   = None
        if self.dry_run:
            logger.warning("⚠️ 텔레그램 토큰 미설정(DRY-RUN) — 메시지를 콘솔에만 출력합니다.")
            return
        try:
            import asyncio
            from telegram import Bot
            self._loop = asyncio.new_event_loop()
            self._bot  = Bot(token=token)
            self._mode = "ptb"
            logger.info("텔레그램 발송기: python-telegram-bot 사용")
        except Exception as e:
            self._mode = "http"
            logger.info(f"python-telegram-bot 미사용({e}) → HTTP 폴백 사용")

    def send(self, text: str, chat_id: str | None = None) -> bool:
        target = chat_id or self.chat_id
        if self.dry_run or not target:
            print(f"\n──────── [DRY-RUN 텔레그램 메시지 → chat {target or '미지정'}] ────────")
            print(text.replace("<b>", "").replace("</b>", "")
                      .replace("<i>", "").replace("</i>", ""))
            print("────────────────────────────────────────\n")
            return True
        try:
            if self._mode == "ptb":
                self._loop.run_until_complete(
                    self._bot.send_message(
                        chat_id=target, text=text,
                        parse_mode="HTML", disable_web_page_preview=True,
                    )
                )
            else:
                self._send_http(text, target)
            return True
        except Exception as e:
            logger.error(f"텔레그램 발송 실패(chat {target}): {e}")
            return False

    def _send_http(self, text: str, chat_id: str) -> None:
        import urllib.parse
        import urllib.request
        url  = f"https://api.telegram.org/bot{self.token}/sendMessage"
        data = urllib.parse.urlencode({
            "chat_id": chat_id, "text": text,
            "parse_mode": "HTML", "disable_web_page_preview": "true",
        }).encode()
        with urllib.request.urlopen(url, data=data, timeout=15) as r:
            res = json.load(r)
            if not res.get("ok"):
                raise RuntimeError(res.get("description", "unknown"))


# ══════════════════════════════════════════════════════════════════════════════
#  런타임 상태 (고점가·트레일링 발동·중복발송 방지)
# ══════════════════════════════════════════════════════════════════════════════

def load_state() -> dict:
    state: dict = {}
    if STATE_FILE.exists():
        try:
            state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("상태 파일 손상 — 새로 시작합니다.")
    # 다중사용자 구조 정규화: 사용자별 상태는 state["users"][user_id] = {runtime, sent}
    state.setdefault("users", {})
    state.setdefault("session", {})
    return state


def save_state(state: dict) -> None:
    try:
        STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logger.error(f"상태 저장 실패: {e}")


def _dedup(state: dict, key: str) -> bool:
    """이미 발송한 신호면 True. 처음이면 기록하고 False."""
    if key in state["sent"]:
        return True
    state["sent"].append(key)
    # 너무 커지지 않게 최근 500개만 유지
    if len(state["sent"]) > 500:
        state["sent"] = state["sent"][-500:]
    return False


# ══════════════════════════════════════════════════════════════════════════════
#  실시간 데이터 — 보유 종목 일봉+장중 형성봉
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_position_bar(ticker: str, cache: dict | None = None) -> dict | None:
    """보유 종목의 최신 봉(장중이면 당일 형성봉) + ATR 을 반환.

    cache(사이클 단위) 제공 시 같은 종목을 여러 사용자가 보유해도 1회만 다운로드한다.
    """
    if cache is not None and ticker in cache:
        return cache[ticker]
    bar: dict | None = None
    try:
        raw = yf.download(ticker, period="40d", interval="1d",
                          auto_adjust=True, progress=False)
        df = _prepare_df(raw, ticker)        # SMA/RSI/ATR 부착 (live_screener 동일)
        if df is not None and not df.empty:
            row = df.iloc[-1]
            close = float(row["Close"])
            atr   = float(row.get("ATR", np.nan))
            bar = {
                "close": close,
                "high":  float(row.get("High", close)),
                "low":   float(row.get("Low",  close)),
                "open":  float(row.get("Open", close)),
                "atr":   atr if (not np.isnan(atr) and atr > 0) else -1.0,
            }
    except Exception as e:
        logger.warning(f"  {ticker} 현재가 조회 실패: {e}")
    if cache is not None:
        cache[ticker] = bar
    return bar


# ══════════════════════════════════════════════════════════════════════════════
#  청산 감시 (SELL_SL / SELL_TS / SELL_TP) — backtest._try_sell v4 로직 재현
# ══════════════════════════════════════════════════════════════════════════════

def monitor_positions(
    holdings: dict, state: dict, enabled: dict, bar_cache: dict | None = None,
    cfg: RiskConfig | None = None,
) -> tuple[list[dict], dict[str, float]]:
    """보유 종목별 청산 조건을 판정. (신호 리스트, 종목별 현재가맵)을 반환.

    enabled: 이 사용자용 신호 on/off 맵. bar_cache: 사이클 단위 시세 캐시(다중사용자 절약).
    cfg: 위험성향(트레일링 발동 마진·휩소가드·당일 손절 서킷브레이커 분기). 미지정 시 안전형.
    """
    cfg = cfg or RISK_SAFE
    positions = holdings.get("positions", {})
    signals:   list[dict] = []
    price_map: dict[str, float] = {}
    runtime = state["runtime"]
    today   = _now_kst().strftime("%Y-%m-%d")
    # [v4.6-5] 당일 SELL_SL 카운터(사용자별) — 새 날이면 리셋
    sl_day = state.setdefault("sl_day", {})
    if today not in sl_day:
        sl_day.clear()
        sl_day[today] = 0

    for ticker, info in positions.items():
        bar = _fetch_position_bar(ticker, bar_cache)
        if bar is None:
            continue
        price_map[ticker] = bar["close"]

        entry = float(info["entry_price"])
        name  = info.get("name", ticker)
        high, low, open_, close, atr = (
            bar["high"], bar["low"], bar["open"], bar["close"], bar["atr"]
        )
        # 손절·익절선: 수동 잔고는 저장값(sl/tp) 사용, Supabase 계정 포지션은 미저장이므로
        # 진입가 + 현재 ATR로 v4 엔진과 동일하게 산출(야수 손절식 + 본전대비 +25% 익절).
        sl = float(info.get("sl") or 0)
        tp = float(info.get("tp") or 0)
        if sl <= 0:
            sl = (max(entry - ATR_SL_MULT_BULL * atr, entry * BULL_SL_FLOOR_PCT)
                  if atr > 0 else entry * SL_FALLBACK_PCT)
        if tp <= 0:
            tp = entry * 1.25

        rt = runtime.setdefault(ticker, {"peak": entry, "trailing_active": False})
        rt["peak"] = max(float(rt.get("peak", entry)), high)
        # [v4.6-6] 휩소 가드용 최초 관측일(≈진입 인지일) 기록
        rt.setdefault("first_seen", today)

        # ── 트레일링 활성화 — 수익 +18%(cfg) 최초 도달 시 손절선 본전+0.5%로 상향 ──
        breakeven_floor = entry * (1.0 + TRAIL_BREAKEVEN_PCT) / (1.0 - SL_SLIPPAGE_PCT)
        if not rt["trailing_active"] and high >= entry * (1.0 + cfg.trail_activate_profit):
            rt["trailing_active"] = True
            sl = max(sl, breakeven_floor)
            if enabled["TS_ARM"] and not _dedup(state, f"{ticker}:TS_ARM"):
                signals.append({
                    "type": "TS_ARM", "ticker": ticker, "name": name,
                    "entry": entry, "current": close, "new_sl": sl,
                    "pnl": (close - entry) / entry * 100,
                })

        # 트레일링 손절선 — 고점 기준, 본전 아래로는 내려가지 않음(이익 보호 전용)
        trailing_sl = (
            _strategy.get_trailing_stop(rt["peak"], atr)
            if (rt["trailing_active"] and atr > 0) else -1.0
        )
        if trailing_sl > 0:
            trailing_sl = max(trailing_sl, breakeven_floor)

        hard_sl      = sl if sl > 0 else float("-inf")
        ts_val       = trailing_sl if trailing_sl > 0 else float("-inf")
        effective_sl = max(hard_sl, ts_val)

        # ── 손절/트레일링 우선 ────────────────────────────────────────────────
        # [v4.6-6] 휩소 가드: 진입 후 cfg.whipsaw_guard_days 거래일 이내는 장중 저가(low)
        # 흔들림으로 손절선이 터지지 않도록 '종가(close)' 기준으로만 손절 판정한다.
        guard_bars = int(np.busday_count(rt.get("first_seen", today), today))
        in_whipsaw = 0 <= guard_bars <= cfg.whipsaw_guard_days
        sl_trigger = close if in_whipsaw else low
        if effective_sl > float("-inf") and sl_trigger <= effective_sl:
            if in_whipsaw:
                fill = close                          # 휩소 가드: 종가 기준 청산
            else:
                fill = effective_sl * (1.0 - SL_SLIPPAGE_PCT)
                if open_ < effective_sl:
                    fill = min(fill, open_)
                fill = max(fill, low)
            is_ts = rt["trailing_active"] and fill >= entry
            stype = "SELL_TS" if is_ts else "SELL_SL"
            # [v4.6-5] 당일 SELL_SL 서킷브레이커 — 한도 초과 시 발송 보류(보유 지속, 익일 재판정).
            # 이익보호(SELL_TS)는 제한 대상이 아니다.
            if stype == "SELL_SL" and sl_day[today] >= cfg.max_daily_sl:
                continue
            if enabled.get(stype, True) and not _dedup(state, f"{ticker}:{stype}"):
                signals.append({
                    "type": stype, "ticker": ticker, "name": name,
                    "entry": entry, "current": close, "fill": fill,
                    "stop": effective_sl, "pnl": (fill - entry) / entry * 100,
                    "qty": int(info.get("quantity", 0)),
                })
                if stype == "SELL_SL":
                    sl_day[today] = sl_day[today] + 1
            continue

        # ── 익절 목표가 도달 ─────────────────────────────────────────────────
        if tp > 0 and high >= tp:
            fill = open_ if open_ > tp else tp
            fill = min(fill, high)
            if enabled["SELL_TP"] and not _dedup(state, f"{ticker}:SELL_TP"):
                signals.append({
                    "type": "SELL_TP", "ticker": ticker, "name": name,
                    "entry": entry, "current": close, "fill": fill, "tp": tp,
                    "pnl": (fill - entry) / entry * 100,
                    "qty": int(info.get("quantity", 0)),
                })

    return signals, price_map


def check_rebalance(
    holdings: dict, price_map: dict[str, float], state: dict,
    cfg: RiskConfig | None = None, is_bear: bool = False,
) -> dict | None:
    """총 주식노출이 모드별 상한(야수 55%/디펜스 30%)을 초과하면 부분매도 리밸런싱 권고."""
    cfg = cfg or RISK_SAFE
    positions = holdings.get("positions", {})
    cash      = float(holdings.get("cash", 0))
    invested  = sum(
        int(p.get("quantity", 0)) * price_map.get(t, p.get("entry_price", 0))
        for t, p in positions.items()
    )
    total_asset = cash + invested
    if total_asset <= 0:
        return None
    gross_pct = cfg.max_gross_exposure_bear if is_bear else cfg.max_gross_exposure_bull
    cap = gross_pct * total_asset
    if invested <= cap:
        return None

    over = invested - cap
    # 평가액 최대 종목부터 트림 대상 선정
    biggest = max(
        positions.items(),
        key=lambda kv: int(kv[1].get("quantity", 0)) * price_map.get(kv[0], 0),
        default=None,
    )
    if not biggest:
        return None
    ticker, info = biggest
    price = price_map.get(ticker, 0)
    if price <= 0:
        return None
    sell_qty = min(int(info.get("quantity", 0)), int(over / price) + 1)
    if sell_qty <= 0:
        return None

    today = _now_kst().strftime("%Y-%m-%d")
    if _dedup(state, f"{today}:REBAL:{ticker}"):
        return None
    return {
        "type": "SELL_REBAL", "ticker": ticker, "name": info.get("name", ticker),
        "exposure_pct": invested / total_asset * 100, "sell_qty": sell_qty,
        "price": price, "trim_value": sell_qty * price,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  매수 감시 (BUY) — run_screening + calc_buy_plan 재사용
# ══════════════════════════════════════════════════════════════════════════════

def screen_candidates(
    is_bear: bool, cfg: RiskConfig | None = None, market: MarketAnalyzer | None = None,
) -> list[dict]:
    """매수 후보 스크리닝 — 같은 위험성향(프로파일) 안에선 유저 공통이라 프로파일당 1회 수행.

    held_tickers 는 비워서 전체 후보를 뽑고, 보유 종목 제외·수량 계산은 사용자별로 처리한다.
    cfg/market 로 디펜스 눌림목 정책(v4.6-4)과 지수 5일선 필터(v5-1)를 적용한다.
    """
    universe = _build_universe(UNIVERSE_N)
    return run_screening(universe, is_bear, held_tickers=set(), cfg=cfg, market=market)


def build_buy_signals(
    candidates: list[dict], holdings: dict, price_map: dict[str, float],
    is_bear: bool, state: dict, cfg: RiskConfig | None = None,
) -> list[dict]:
    """사용자 계좌(시드머니) 기준으로 매수 후보에 대한 맞춤 수량·예산을 계산한다."""
    cfg = cfg or RISK_SAFE
    positions     = holdings.get("positions", {})
    cash          = float(holdings.get("cash", 0))
    held_tickers  = set(positions.keys())
    invested      = sum(
        int(p.get("quantity", 0)) * price_map.get(t, p.get("entry_price", 0))
        for t, p in positions.items()
    )
    total_asset   = cash + invested
    max_positions = MAX_POS_BEAR if is_bear else MAX_POS_BULL
    n_positions   = len(positions)
    if n_positions >= max_positions:
        return []

    signals: list[dict] = []
    today   = _now_kst().strftime("%Y-%m-%d")
    for cand in candidates:
        if cand["ticker"] in held_tickers:
            continue
        if n_positions + len(signals) >= max_positions:
            break
        if _dedup(state, f"{today}:BUY:{cand['ticker']}"):
            continue
        plan = calc_buy_plan(
            cand, cash=cash, total_asset=total_asset,
            current_invested=invested, n_positions=n_positions + len(signals),
            max_positions=max_positions, cfg=cfg, is_bear=is_bear,
        )
        if plan is None or plan["qty"] <= 0:
            continue
        signals.append({
            "type": "BUY", "ticker": cand["ticker"], "name": cand["name"],
            "signal": cand["signal"], "score": cand["score"], "close": cand["close"],
            **plan,
        })
    return signals


# ══════════════════════════════════════════════════════════════════════════════
#  메시지 포맷팅 (사용자 행동 가이드라인)
# ══════════════════════════════════════════════════════════════════════════════

def fmt_buy(s: dict) -> str:
    is_bull = s["signal"] == "SMA_GOLDEN_CROSS"
    head    = "🚨 [야수 시그널 포착 - 매수 추천]" if is_bull else "🚨 [디펜스 시그널 포착 - 눌림목 매수]"
    close, sl, tp, qty = s["close"], s["sl"], s["tp"], s["qty"]
    sl_pct  = (sl - close) / close * 100
    tp_pct  = (tp - close) / close * 100
    return (
        f"<b>{head}</b>\n"
        f"- 종목: {s['name']} ({s['ticker']})\n"
        f"- 현재가: {_won(close)}\n"
        f"- 신호강도: {s['score']:.0f}점\n"
        f"- 추천 비중: 총자산의 15% 캡 적용 "
        f"(목표 수량: {qty}주 / 투입 예산 {_man(s['alloc_cash'])}, 비중 {s['alloc_pct']*100:.1f}%)\n"
        f"- 리스크: 손절 시 계좌 타격 {_man(s['risk_amt'])}(총자산 {s['risk_pct']:.2f}%)\n"
        f"- 대응 가이드: 매수 즉시 MTS에 "
        f"[익절 지정가: {_won(tp)}({tp_pct:+.1f}%)], "
        f"[손절 감시가: {_won(sl)}({sl_pct:+.1f}%)] 예약 설정을 걸어두세요."
    )


def fmt_sell_tp(s: dict) -> str:
    return (
        f"<b>💵✅ [익절 시그널 - 목표가 도달]</b>\n"
        f"- 종목: {s['name']} ({s['ticker']})\n"
        f"- 현재가: {_won(s['current'])} / 진입가: {_won(s['entry'])} "
        f"(손익 <b>{s['pnl']:+.1f}%</b>)\n"
        f"- 대응 가이드: 익절 목표가 도달 — 보유 {s['qty']}주 전량 익절 매도하세요."
    )


def fmt_sell_sl(s: dict) -> str:
    return (
        f"<b>💵❌ [손절 시그널 - 손절선 이탈]</b>\n"
        f"- 종목: {s['name']} ({s['ticker']})\n"
        f"- 현재가: {_won(s['current'])} / 진입가: {_won(s['entry'])} "
        f"(손익 <b>{s['pnl']:+.1f}%</b>)\n"
        f"- 손절선: {_won(s['stop'])}\n"
        f"- 대응 가이드: 손절선 이탈 — 지체 없이 보유 {s['qty']}주 전량 손절하세요. (잽은 작게)"
    )


def fmt_ts_arm(s: dict) -> str:
    return (
        f"<b>🎯 [트레일링 발동 - 본전 방어선 상향]</b>\n"
        f"- 종목: {s['name']} ({s['ticker']})\n"
        f"- 현재가: {_won(s['current'])} (수익 <b>{s['pnl']:+.1f}%</b>, +12% 돌파)\n"
        f"- 대응 가이드: MTS 손절가를 <b>본전+0.5% = {_won(s['new_sl'])}</b>로 올리세요. "
        f"이제 이 종목은 마이너스 청산이 불가합니다(이익 보호 전용)."
    )


def fmt_sell_ts(s: dict) -> str:
    return (
        f"<b>🎯✅ [트레일링 스톱 - 이익 확정 매도]</b>\n"
        f"- 종목: {s['name']} ({s['ticker']})\n"
        f"- 현재가: {_won(s['current'])} / 진입가: {_won(s['entry'])} "
        f"(손익 <b>{s['pnl']:+.1f}%</b>)\n"
        f"- 트레일링선: {_won(s['stop'])}\n"
        f"- 대응 가이드: 고점 대비 하락 — 트레일링 스톱 {s['qty']}주 전량 매도해 이익을 확정하세요."
    )


def fmt_rebal(s: dict) -> str:
    return (
        f"<b>♻️ [리밸런싱 - 노출 45% 상한 초과]</b>\n"
        f"- 현재 총 주식노출: <b>{s['exposure_pct']:.1f}%</b> (상한 45% 초과)\n"
        f"- 트림 대상: {s['name']} ({s['ticker']})\n"
        f"- 대응 가이드: {s['name']}을 약 <b>{s['sell_qty']}주</b>"
        f"({_man(s['trim_value'])}) 부분매도해 노출을 45%로 낮추세요. (핵심 방패)"
    )


FORMATTERS = {
    "BUY": fmt_buy, "SELL_TP": fmt_sell_tp, "SELL_SL": fmt_sell_sl,
    "TS_ARM": fmt_ts_arm, "SELL_TS": fmt_sell_ts, "SELL_REBAL": fmt_rebal,
}


# ══════════════════════════════════════════════════════════════════════════════
#  1회 점검 사이클
# ══════════════════════════════════════════════════════════════════════════════

def _notify_targets() -> list[dict]:
    """알림 발송 대상 [{user_id, chat_id}].

    ① Supabase에서 텔레그램 연동·수신 ON 사용자(list_notify_users) 전체.
    ② env 폴백: 소유자가 웹 연동을 안 했어도 SIGNAL_ACCOUNT_USER_ID→TELEGRAM_CHAT_ID 로 발송
       (로컬 단독 운용 호환).
    """
    global _account
    if _account is None:
        try:
            _account = SupabaseAccount()
        except Exception:
            _account = None

    targets: list[dict] = []
    seen: set[str] = set()
    if _account and _account.enabled:
        try:
            for u in _account.list_notify_users():
                uid, chat = u.get("user_id"), u.get("chat_id")
                if uid and chat and uid not in seen:
                    targets.append({
                        "user_id": uid, "chat_id": str(chat),
                        "risk_profile": u.get("risk_profile") or "safe",
                    })
                    seen.add(uid)
        except Exception as e:
            logger.error(f"알림 대상 조회 실패: {e}")
    if ACCOUNT_USER_ID and TELEGRAM_CHAT_ID and ACCOUNT_USER_ID not in seen:
        targets.append({
            "user_id": ACCOUNT_USER_ID, "chat_id": TELEGRAM_CHAT_ID,
            "risk_profile": os.getenv("SIGNAL_ACCOUNT_RISK_PROFILE", "safe"),
        })
    return targets


def run_cycle(notifier: TelegramNotifier, state: dict) -> int:
    """1회 감시 사이클 (1:N) — 장세·매수후보는 공통 1회, 청산/리밸런싱/매수는 사용자별 발송.

    각 사용자는 자신의 보유종목·예수금(시드머니)·알림설정으로 맞춤 계산되어, 자신의
    텔레그램 chat 으로만 알림을 받는다. 중복방지·고점추적 상태는 사용자별로 격리된다.
    """
    global _account
    targets = _notify_targets()
    if not targets:
        logger.info("  알림 대상 사용자 없음(텔레그램 연동·수신 ON 사용자 0명)")
        return 0

    # ── 지수 데이터 1회 로드 → 위험성향(safe/aggressive)별로 장세·후보 1회씩 계산 ──
    # 같은 프로파일 유저는 장세·매수후보를 공유하므로 프로파일당 최대 1회만 스크리닝한다.
    market = MarketAnalyzer()
    market.load()

    from collections import defaultdict
    by_profile: dict[str, list[dict]] = defaultdict(list)
    for tgt in targets:
        by_profile[str(tgt.get("risk_profile") or "safe").lower()].append(tgt)

    # profile → (cfg, is_bear, candidates)
    screen_cache: dict[str, tuple] = {}
    for profile, plist in by_profile.items():
        cfg     = risk_config(profile)
        is_bear = market.is_bear_market(cfg)
        candidates = screen_candidates(is_bear, cfg, market)
        screen_cache[profile] = (cfg, is_bear, candidates)
        logger.info(
            f"[{cfg.name}] 장세: {'🐻 디펜스(하락장)' if is_bear else '🐂 야수(상승장)'} "
            f"(스코어커트 {cfg.market_score_cutline:.0f}pt) · 대상 {len(plist)}명 · 후보 {len(candidates)}개"
        )

    bar_cache: dict = {}        # 사이클 단위 시세 캐시(여러 사용자가 같은 종목 보유 시 1회만 다운로드)
    sent_count = 0
    use_supabase = bool(_account and _account.enabled)

    for tgt in targets:
        uid, chat_id = tgt["user_id"], tgt["chat_id"]
        cfg, is_bear, candidates = screen_cache[str(tgt.get("risk_profile") or "safe").lower()]
        try:
            holdings = _account.load_holdings(user_id=uid) if use_supabase else load_account()
        except Exception as e:
            logger.error(f"  [{uid[:8]}] 계좌 로드 실패 — 건너뜀: {e}")
            continue

        enabled = resolve_enabled(holdings.get("alert_prefs") or {})
        ustate  = _user_substate(state, uid)

        # 매도 완료 종목의 런타임/중복키 정리 (사용자별)
        held = set(holdings.get("positions", {}).keys())
        for t in list(ustate["runtime"].keys()):
            if t not in held:
                ustate["runtime"].pop(t, None)
                ustate["sent"] = [k for k in ustate["sent"] if not k.startswith(f"{t}:")]

        # 1) 청산 감시  2) 리밸런싱  3) 매수(사용자 시드머니 기준 맞춤 수량) — 모두 cfg 반영
        sell_signals, price_map = monitor_positions(holdings, ustate, enabled, bar_cache, cfg)
        if enabled["SELL_REBAL"]:
            rebal = check_rebalance(holdings, price_map, ustate, cfg, is_bear)
            if rebal:
                sell_signals.append(rebal)
        buy_signals = (
            build_buy_signals(candidates, holdings, price_map, is_bear, ustate, cfg)
            if enabled["BUY"] else []
        )

        # 발송 — 이 사용자의 chat 으로만 (청산·리밸런싱 우선, 그 다음 매수)
        user_sent = 0
        for s in sell_signals + buy_signals:
            if not enabled.get(s["type"], True):
                continue
            if notifier.send(FORMATTERS[s["type"]](s), chat_id=chat_id):
                user_sent += 1
                sent_count += 1
                logger.info(f"  📤 [{uid[:8]}] {s['type']}: {s.get('name', s.get('ticker'))}")
        if user_sent == 0:
            logger.info(f"  [{uid[:8]}] 신규 신호 없음")

    save_state(state)
    return sent_count


# ══════════════════════════════════════════════════════════════════════════════
#  스케줄러 루프
# ══════════════════════════════════════════════════════════════════════════════

def _scheduled_job(notifier: TelegramNotifier, state: dict) -> None:
    now = _now_kst()
    if not _is_market_hours(now):
        # 장 마감 직후 1회 마감 안내
        today = now.strftime("%Y-%m-%d")
        if now.weekday() < 5 and now.time() > MARKET_CLOSE \
           and state["session"].get("closed_date") != today:
            state["session"]["closed_date"] = today
            save_state(state)
            notifier.send("🔔 <b>장 마감</b> — 오늘 시그널 감시를 종료합니다. 내일 09:00에 재개합니다.")
        return

    # 장 시작 1회 안내
    today = now.strftime("%Y-%m-%d")
    if state["session"].get("opened_date") != today:
        state["session"]["opened_date"] = today
        save_state(state)
        notifier.send(f"🔔 <b>장 시작</b> — 시그널 감시를 시작합니다 ({POLL_INTERVAL_MIN}분 주기).")

    logger.info(f"━━━ 감시 사이클 [{now.strftime('%H:%M')}] ━━━")
    try:
        run_cycle(notifier, state)
    except Exception as e:
        logger.error(f"사이클 오류: {e}", exc_info=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="실시간 매매 시그널 텔레그램 봇")
    parser.add_argument("--once", action="store_true", help="즉시 1회 점검 후 종료(테스트)")
    parser.add_argument("--test", action="store_true", help="텔레그램 연결 테스트 메시지만 발송")
    args = parser.parse_args()

    notifier = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
    state    = load_state()
    logger.info(f"알림 설정 — 꺼진 신호: {_disabled_types_label()}")

    if args.test:
        ok = notifier.send(
            "✅ <b>시그널 봇 연결 테스트</b>\n검증된 v4 엔진 기반 실시간 시그널 봇이 정상 연결되었습니다."
        )
        print("발송 성공" if ok else "발송 실패")
        return

    if args.once:
        logger.info("── 1회 점검 모드 ──")
        run_cycle(notifier, state)
        return

    # ── 정규 스케줄 루프 ─────────────────────────────────────────────────────
    try:
        import schedule
    except ImportError:
        logger.error("schedule 미설치 — 'pip install schedule pytz' 후 재실행하세요.")
        sys.exit(1)

    notifier.send(
        f"🤖 <b>시그널 봇 가동</b>\n"
        f"장중(09:00~15:30) {POLL_INTERVAL_MIN}분 주기로 v4 엔진 매매 시그널을 감시합니다.\n"
        f"🔕 꺼진 알림: {_disabled_types_label()}\n"
        f"<i>자동 주문 없음 — 알림을 보고 직접 MTS로 매매하세요.</i>"
    )
    logger.info(f"스케줄러 시작 — {POLL_INTERVAL_MIN}분 주기, 장중 09:00~15:30 KST")

    schedule.every(POLL_INTERVAL_MIN).minutes.do(_scheduled_job, notifier=notifier, state=state)
    _scheduled_job(notifier, state)          # 시작 즉시 1회 실행

    import time as _time
    while True:
        try:
            schedule.run_pending()
            _time.sleep(20)
        except KeyboardInterrupt:
            logger.info("사용자 종료(Ctrl+C) — 봇을 멈춥니다.")
            break
        except Exception as e:
            logger.error(f"루프 오류: {e}", exc_info=True)
            _time.sleep(30)


if __name__ == "__main__":
    main()
