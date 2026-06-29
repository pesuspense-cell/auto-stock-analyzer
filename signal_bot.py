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

# 장 운영 시간(한국 KST). 09:00 ~ 15:30.
MARKET_OPEN  = dtime(9, 0)
MARKET_CLOSE = dtime(15, 30)
KST_TZ       = "Asia/Seoul"

STATE_FILE = Path(__file__).parent / "signal_bot_state.json"

# ── [v4] 트레일링 스톱 파라미터 (backtest.py 와 동일) ────────────────────────
TRAIL_ACTIVATE_PROFIT = 0.12   # 수익 +12% 최초 도달 시 트레일링 즉시 ON
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

    def __init__(self, token: str, chat_id: str, dry_run: bool = False):
        self.token   = token
        self.chat_id = chat_id
        self.dry_run = dry_run or not (token and chat_id)
        self._bot    = None
        self._loop   = None
        if self.dry_run:
            logger.warning("⚠️ 텔레그램 미설정(DRY-RUN) — 메시지를 콘솔에만 출력합니다.")
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

    def send(self, text: str) -> bool:
        if self.dry_run:
            print("\n──────── [DRY-RUN 텔레그램 메시지] ────────")
            print(text.replace("<b>", "").replace("</b>", "")
                      .replace("<i>", "").replace("</i>", ""))
            print("────────────────────────────────────────\n")
            return True
        try:
            if self._mode == "ptb":
                self._loop.run_until_complete(
                    self._bot.send_message(
                        chat_id=self.chat_id, text=text,
                        parse_mode="HTML", disable_web_page_preview=True,
                    )
                )
            else:
                self._send_http(text)
            return True
        except Exception as e:
            logger.error(f"텔레그램 발송 실패: {e}")
            return False

    def _send_http(self, text: str) -> None:
        import urllib.parse
        import urllib.request
        url  = f"https://api.telegram.org/bot{self.token}/sendMessage"
        data = urllib.parse.urlencode({
            "chat_id": self.chat_id, "text": text,
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
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("상태 파일 손상 — 새로 시작합니다.")
    return {"runtime": {}, "sent": [], "session": {}}


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

def _fetch_position_bar(ticker: str) -> dict | None:
    """보유 종목의 최신 봉(장중이면 당일 형성봉) + ATR 을 반환."""
    try:
        raw = yf.download(ticker, period="40d", interval="1d",
                          auto_adjust=True, progress=False)
        df = _prepare_df(raw, ticker)        # SMA/RSI/ATR 부착 (live_screener 동일)
        if df is None or df.empty:
            return None
        row = df.iloc[-1]
        close = float(row["Close"])
        atr   = float(row.get("ATR", np.nan))
        return {
            "close": close,
            "high":  float(row.get("High", close)),
            "low":   float(row.get("Low",  close)),
            "open":  float(row.get("Open", close)),
            "atr":   atr if (not np.isnan(atr) and atr > 0) else -1.0,
        }
    except Exception as e:
        logger.warning(f"  {ticker} 현재가 조회 실패: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  청산 감시 (SELL_SL / SELL_TS / SELL_TP) — backtest._try_sell v4 로직 재현
# ══════════════════════════════════════════════════════════════════════════════

def monitor_positions(holdings: dict, state: dict) -> tuple[list[dict], dict[str, float]]:
    """보유 종목별 청산 조건을 판정. (신호 리스트, 종목별 현재가맵)을 반환."""
    positions = holdings.get("positions", {})
    signals:   list[dict] = []
    price_map: dict[str, float] = {}
    runtime = state["runtime"]

    for ticker, info in positions.items():
        bar = _fetch_position_bar(ticker)
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

        # ── [v4] 트레일링 활성화 — 수익 +12% 최초 도달 시 손절선 본전+0.5%로 상향 ──
        breakeven_floor = entry * (1.0 + TRAIL_BREAKEVEN_PCT) / (1.0 - SL_SLIPPAGE_PCT)
        if not rt["trailing_active"] and high >= entry * (1.0 + TRAIL_ACTIVATE_PROFIT):
            rt["trailing_active"] = True
            sl = max(sl, breakeven_floor)
            if not _dedup(state, f"{ticker}:TS_ARM"):
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

        # ── 손절/트레일링 우선 (보수적 체결: 손절선 -1% 슬리피지, 갭하락 시 시초가) ──
        if effective_sl > float("-inf") and low <= effective_sl:
            fill = effective_sl * (1.0 - SL_SLIPPAGE_PCT)
            if open_ < effective_sl:
                fill = min(fill, open_)
            fill = max(fill, low)
            is_ts = rt["trailing_active"] and fill >= entry
            stype = "SELL_TS" if is_ts else "SELL_SL"
            if not _dedup(state, f"{ticker}:{stype}"):
                signals.append({
                    "type": stype, "ticker": ticker, "name": name,
                    "entry": entry, "current": close, "fill": fill,
                    "stop": effective_sl, "pnl": (fill - entry) / entry * 100,
                    "qty": int(info.get("quantity", 0)),
                })
            continue

        # ── 익절 목표가 도달 ─────────────────────────────────────────────────
        if tp > 0 and high >= tp:
            fill = open_ if open_ > tp else tp
            fill = min(fill, high)
            if not _dedup(state, f"{ticker}:SELL_TP"):
                signals.append({
                    "type": "SELL_TP", "ticker": ticker, "name": name,
                    "entry": entry, "current": close, "fill": fill, "tp": tp,
                    "pnl": (fill - entry) / entry * 100,
                    "qty": int(info.get("quantity", 0)),
                })

    return signals, price_map


def check_rebalance(holdings: dict, price_map: dict[str, float], state: dict) -> dict | None:
    """총 주식노출이 총자산의 45%를 초과하면 최대 평가종목 부분매도 리밸런싱 권고."""
    positions = holdings.get("positions", {})
    cash      = float(holdings.get("cash", 0))
    invested  = sum(
        int(p.get("quantity", 0)) * price_map.get(t, p.get("entry_price", 0))
        for t, p in positions.items()
    )
    total_asset = cash + invested
    if total_asset <= 0:
        return None
    cap = MAX_GROSS_EXPOSURE * total_asset
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

def screen_buys(
    holdings: dict, price_map: dict[str, float], is_bear: bool, state: dict
) -> list[dict]:
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

    universe   = _build_universe(UNIVERSE_N)
    candidates = run_screening(universe, is_bear, held_tickers)

    signals: list[dict] = []
    today   = _now_kst().strftime("%Y-%m-%d")
    for cand in candidates:
        if n_positions + len(signals) >= max_positions:
            break
        if _dedup(state, f"{today}:BUY:{cand['ticker']}"):
            continue
        plan = calc_buy_plan(
            cand, cash=cash, total_asset=total_asset,
            current_invested=invested, n_positions=n_positions + len(signals),
            max_positions=max_positions,
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

def run_cycle(notifier: TelegramNotifier, state: dict) -> int:
    """1회 감시 사이클 — 청산 → 리밸런싱 → 매수 순으로 신호를 점검·발송."""
    holdings = load_account()
    sent_count = 0

    # 보유 종목이 사라졌으면(매도 완료) 관련 런타임/중복키 정리
    held = set(holdings.get("positions", {}).keys())
    for t in list(state["runtime"].keys()):
        if t not in held:
            state["runtime"].pop(t, None)
            state["sent"] = [k for k in state["sent"] if not k.startswith(f"{t}:")]

    # ── 장세 판별 ───────────────────────────────────────────────────────────
    market = MarketAnalyzer()
    market.load()
    is_bear = market.is_bear_market()
    logger.info(f"장세: {'🐻 디펜스(하락장)' if is_bear else '🐂 야수(상승장)'}")

    # ── 1) 청산 감시 (보유 종목) ─────────────────────────────────────────────
    sell_signals, price_map = monitor_positions(holdings, state)

    # ── 2) 리밸런싱 감시 ─────────────────────────────────────────────────────
    rebal = check_rebalance(holdings, price_map, state)
    if rebal:
        sell_signals.append(rebal)

    # ── 3) 매수 감시 (매수잠금 없을 때) ──────────────────────────────────────
    buy_signals = screen_buys(holdings, price_map, is_bear, state)

    # ── 발송 (청산·리밸런싱 우선, 그 다음 매수) ──────────────────────────────
    for s in sell_signals + buy_signals:
        msg = FORMATTERS[s["type"]](s)
        if notifier.send(msg):
            sent_count += 1
            logger.info(f"  📤 {s['type']} 발송: {s.get('name', s.get('ticker'))}")

    save_state(state)
    if sent_count == 0:
        logger.info("  포착된 신규 신호 없음")
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
