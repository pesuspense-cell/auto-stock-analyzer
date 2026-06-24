"""
live_scheduler.py — 실시간 자동매매 스케줄러

매일 09:00 ~ 15:20 KST, 5분 간격으로 시장 데이터를 수집하고
매수·매도 신호 발생 시 Telegram 알림을 발송합니다.

실행:
  python live_scheduler.py

필수 환경 변수 (.env):
  TELEGRAM_BOT_TOKEN=<BotFather 발급 토큰>
  TELEGRAM_CHAT_ID=<chat id>

설정:
  SCAN_MARKETS       — 대상 마켓 목록
  UNIVERSE_N         — 마켓별 유니버스 크기
  VOLUME_TOP_N       — 거래대금 상위 N (매수 후보 1차 필터)
  ACCOUNT_CAPITAL    — 리스크 사이징 기준 총자산
  MAX_RISK_PER_TRADE — 종목당 손절 시 허용 손실 비율(총자산 2%) [개편1]
  RSI_HARD_LIMIT     — 이 RSI 이상 종목 신규 진입 금지 [개편2]

리스크 관리(backtest.py 동일): ATR 고정 리스크 배정 / RSI 하드필터 /
연쇄손절 시장필터 락아웃 / 보수적 손절 체결.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytz
import requests
import yfinance as yf
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from backtest import StockScreener
from stock_ai import _add_indicators, _flatten_columns
from strategy import TradingStrategy

# ══════════════════════════════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════════════════════════════

KST = pytz.timezone("Asia/Seoul")

MARKET_OPEN       = (9,  0)   # 09:00 KST
MARKET_CLOSE      = (15, 20)  # 15:20 KST — 마지막 스캔
SCAN_INTERVAL_MIN = 5         # 스캔 주기 (분)

SCAN_MARKETS      = ["KOSPI", "KOSDAQ"]
UNIVERSE_N        = 100       # 마켓별 유니버스 크기
VOLUME_TOP_N      = 30        # 거래대금 상위 N (1차 필터)

# ── [개편1] 리스크 관리 파라미터 (backtest.py 동일 규칙) ─────────────────────
ACCOUNT_CAPITAL    = 10_000_000  # 리스크 사이징 기준 총자산 (실제 계좌 규모로 조정)
MAX_RISK_PER_TRADE = 0.02        # 단일 종목 손절 시 허용 손실 = 총자산의 2%
MAX_POSITION_PCT   = 0.20        # 단일 종목 평가액 상한 = 총자산의 20% (몰빵 방지)
SL_FALLBACK_PCT    = 0.93        # ATR 미산출 시 손절 폴백 (-7%)
SL_SLIPPAGE_PCT    = 0.01        # [개편4] 손절 체결 보수 슬리피지 (-1%)

# ── [개편2] RSI 하드필터 ─────────────────────────────────────────────────────
RSI_HARD_LIMIT     = 70.0        # 종목 RSI_14 이 값 이상이면 신규 진입 전면 금지

# ── [개편3] 연쇄손절 → 시장 필터 연동 락아웃 ────────────────────────────────
SL_LOOKBACK_DAYS   = 5           # 연쇄손절 집계 윈도우(일)
SL_LOCKOUT_COUNT   = 3           # 윈도우 내 SELL_SL 발생 임계 횟수

POSITIONS_FILE = Path(__file__).parent / "live_positions.json"
ALERTED_FILE   = Path(__file__).parent / "live_alerted.json"
LOCKOUT_FILE   = Path(__file__).parent / "live_lockout.json"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID",   "")

# ══════════════════════════════════════════════════════════════════════════════
# 로깅
# ══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("live_scheduler")

# ══════════════════════════════════════════════════════════════════════════════
# Telegram 발송
# ══════════════════════════════════════════════════════════════════════════════

def send_telegram(text: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram 설정 없음 — 알림 생략")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        resp = requests.post(
            url,
            json={
                "chat_id":                  TELEGRAM_CHAT_ID,
                "text":                     text,
                "parse_mode":               "HTML",
                "disable_web_page_preview": True,
            },
            timeout=10,
        )
        ok = resp.json().get("ok", False)
        if not ok:
            logger.warning(f"Telegram 응답 오류: {resp.text[:200]}")
        return ok
    except Exception as e:
        logger.warning(f"Telegram 발송 실패: {e}")
        return False

# ══════════════════════════════════════════════════════════════════════════════
# 포지션 영속화 (JSON 파일)
# ══════════════════════════════════════════════════════════════════════════════

def load_positions() -> dict:
    if POSITIONS_FILE.exists():
        try:
            return json.loads(POSITIONS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_positions(positions: dict) -> None:
    POSITIONS_FILE.write_text(
        json.dumps(positions, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

# ══════════════════════════════════════════════════════════════════════════════
# 당일 알림 중복 방지
# ══════════════════════════════════════════════════════════════════════════════

def load_alerted() -> dict:
    today = datetime.now(KST).strftime("%Y-%m-%d")
    if ALERTED_FILE.exists():
        try:
            data = json.loads(ALERTED_FILE.read_text(encoding="utf-8"))
            if data.get("date") == today:
                return data.get("keys", {})
        except Exception:
            pass
    return {}


def save_alerted(alerted: dict) -> None:
    today = datetime.now(KST).strftime("%Y-%m-%d")
    ALERTED_FILE.write_text(
        json.dumps({"date": today, "keys": alerted}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

# ══════════════════════════════════════════════════════════════════════════════
# [개편3] 연쇄손절 락아웃 상태 영속화
# ══════════════════════════════════════════════════════════════════════════════

def load_lockout() -> dict:
    """{"sl_dates": [ISO date,...], "locked": bool} 형태의 락아웃 상태."""
    if LOCKOUT_FILE.exists():
        try:
            data = json.loads(LOCKOUT_FILE.read_text(encoding="utf-8"))
            return {"sl_dates": list(data.get("sl_dates", [])),
                    "locked":   bool(data.get("locked", False))}
        except Exception:
            pass
    return {"sl_dates": [], "locked": False}


def save_lockout(state: dict) -> None:
    LOCKOUT_FILE.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI — backtest.py / live_screener.py 와 동일."""
    delta    = close.diff()
    gain     = delta.where(delta > 0, 0.0)
    loss     = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)

# ══════════════════════════════════════════════════════════════════════════════
# KOSDAQ 시장 상황 (하락장 여부)
# ══════════════════════════════════════════════════════════════════════════════

_bear_cache: dict = {"ts": 0.0, "value": False}   # 1시간 캐시


def is_bear_market() -> bool:
    now_ts = time.time()
    if now_ts - _bear_cache["ts"] < 3600:          # 캐시 유효
        return _bear_cache["value"]

    try:
        raw = yf.download("^KQ11", period="2mo", auto_adjust=True, progress=False)
        if raw.empty or len(raw) < 20:
            _bear_cache.update({"ts": now_ts, "value": False})
            return False

        close = (
            raw["Close"].iloc[:, 0].dropna()
            if isinstance(raw.columns, pd.MultiIndex)
            else raw["Close"].dropna()
        )
        sma20   = float(close.rolling(20).mean().iloc[-1])
        current = float(close.iloc[-1])
        result  = current < sma20

        status = "하락장 ⚠️" if result else "상승장 ✅"
        logger.info(f"KOSDAQ 시장: {status}  (현재 {current:,.0f} / SMA20 {sma20:,.0f})")
        _bear_cache.update({"ts": now_ts, "value": result})
        return result

    except Exception as e:
        logger.warning(f"KOSDAQ 지수 조회 실패: {e}")
        _bear_cache.update({"ts": now_ts, "value": False})
        return False


_recover_cache: dict = {"ts": 0.0, "value": False}   # 1시간 캐시


def market_recovered() -> bool:
    """[개편3] KOSPI 또는 KOSDAQ 중 하나라도 종가가 자기 20일선 위 안착 → 락아웃 해제 조건."""
    now_ts = time.time()
    if now_ts - _recover_cache["ts"] < 3600:
        return _recover_cache["value"]
    result = False
    for sym in ("^KS11", "^KQ11"):
        try:
            raw = yf.download(sym, period="2mo", auto_adjust=True, progress=False)
            if raw.empty or len(raw) < 20:
                continue
            close = (raw["Close"].iloc[:, 0] if isinstance(raw.columns, pd.MultiIndex)
                     else raw["Close"]).dropna()
            sma20 = float(close.rolling(20).mean().iloc[-1])
            if float(close.iloc[-1]) > sma20:
                result = True
                break
        except Exception as e:
            logger.warning(f"{sym} 20일선 조회 실패: {e}")
    _recover_cache.update({"ts": now_ts, "value": result})
    return result


def resolve_stop_loss(price: float, atr: float, tag: str) -> float:
    """[개편1] 신호별 손절가 — backtest.py _resolve_stop_loss 동일."""
    if atr > 0:
        if tag == "SMA_GOLDEN_CROSS":
            return max(price - 2.5 * atr, price * 0.92)
        return price - 2.0 * atr          # 추세추종 등
    return price * SL_FALLBACK_PCT


def calc_position_qty(price: float, stop_loss: float, total_asset: float) -> int:
    """[개편1] ATR 손절 기반 고정 리스크 수량 — 손절 시 손실 = 총자산의 2% 고정."""
    risk_per_share = price - stop_loss
    if risk_per_share <= 0 or price <= 0:
        return 0
    qty = int(total_asset * MAX_RISK_PER_TRADE / risk_per_share)
    qty = min(qty, int(total_asset * MAX_POSITION_PCT / price))   # 20% 비중 캡
    return max(qty, 0)

# ══════════════════════════════════════════════════════════════════════════════
# 종목 데이터 다운로드 + 지표 계산
# ══════════════════════════════════════════════════════════════════════════════

def fetch_ticker_data(tickers: list[str]) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    CHUNK = 30
    for i in range(0, len(tickers), CHUNK):
        chunk = tickers[i: i + CHUNK]
        try:
            raw = yf.download(
                chunk if len(chunk) > 1 else chunk[0],
                period="3mo",
                auto_adjust=True,
                progress=False,
            )
            if raw.empty:
                continue
            for ticker in chunk:
                try:
                    if isinstance(raw.columns, pd.MultiIndex):
                        df = raw.xs(ticker, axis=1, level=1).dropna(how="all")
                    else:
                        df = raw.dropna(how="all")
                    if len(df) < 30:
                        continue
                    df = _flatten_columns(df)
                    df = _add_indicators(df)
                    df["RSI_14"] = _calc_rsi(df["Close"])   # [개편2] 과열 하드필터용
                    result[ticker] = df
                except Exception:
                    continue
        except Exception as e:
            logger.warning(f"배치 다운로드 실패: {e}")
    return result

# ══════════════════════════════════════════════════════════════════════════════
# 매수 후보 스크리닝 (SMA 골든크로스 / 추세 추종)
# ══════════════════════════════════════════════════════════════════════════════

def screen_buy_candidates(
    ticker_data: dict[str, pd.DataFrame],
    strategy: TradingStrategy,
) -> dict[str, str]:
    # 1단계: 거래대금 상위 N
    turnover: list[tuple[str, float]] = []
    for ticker, df in ticker_data.items():
        if df.empty:
            continue
        row    = df.iloc[-1]
        close  = float(row.get("Close",  0) or 0)
        volume = float(row.get("Volume", 0) or 0)
        turnover.append((ticker, close * volume))
    turnover.sort(key=lambda x: x[1], reverse=True)
    top_tickers = [t for t, _ in turnover[:VOLUME_TOP_N]]

    # 2단계: SMA 크로스 + 모멘텀
    candidates: dict[str, str] = {}
    for ticker in top_tickers:
        df = ticker_data.get(ticker)
        if df is None or len(df) < strategy.kill_window + 60:
            continue
        if strategy.check_kill_switch(df["Close"]):
            continue
        row       = df.iloc[-1]
        prev      = df.iloc[-2]
        sma5      = float(row.get("SMA_5",  np.nan))
        sma20     = float(row.get("SMA_20", np.nan))
        prev_sma5 = float(prev.get("SMA_5", np.nan))
        if any(np.isnan(v) for v in [sma5, sma20, prev_sma5]):
            continue
        # [개편2] RSI 하드필터 — 과열 종목 신규 진입 금지(상투 추격 차단)
        stock_rsi = float(row.get("RSI_14", np.nan))
        if not np.isnan(stock_rsi) and stock_rsi >= RSI_HARD_LIMIT:
            continue
        if strategy.is_entry_signal(sma5, sma20, prev_sma5):
            tag = "SMA_GOLDEN_CROSS" if prev_sma5 <= sma20 else "SMA_TREND_FOLLOW"
            candidates[ticker] = tag

    return candidates

# ══════════════════════════════════════════════════════════════════════════════
# 매도 신호 체크 (보유 포지션)
# ══════════════════════════════════════════════════════════════════════════════

def check_sell_signals(
    positions: dict,
    ticker_data: dict[str, pd.DataFrame],
    strategy: TradingStrategy,
) -> list[dict]:
    signals = []
    for ticker, pos in list(positions.items()):
        df = ticker_data.get(ticker)
        if df is None or df.empty:
            continue
        row   = df.iloc[-1]
        price = float(row.get("Close", 0))
        high  = float(row.get("High",  price))
        low   = float(row.get("Low",   price))
        open_ = float(row.get("Open",  price))
        atr   = float(row.get("ATR",   np.nan))
        if price <= 0:
            continue

        atr_val     = atr if not np.isnan(atr) and atr > 0 else -1.0
        entry_price = float(pos.get("entry_price", 0))
        # 트레일링은 '직전 최고가' 기준 (당일 고점-손절 동시 발생 방지)
        prev_peak   = float(pos.get("peak_price", price))
        take_profit = float(pos.get("take_profit", -1))
        stop_loss   = float(pos.get("stop_loss",   -1))

        action = reason = None
        fill_price = price

        # [개편4] 보수적 체결: 손절을 익절보다 먼저 검사, 장중 저가가 손절선 터치 시 청산
        trailing_sl = strategy.get_trailing_stop(prev_peak, atr_val) if atr_val > 0 else -1.0
        hard_sl     = stop_loss   if stop_loss   > 0 else float("-inf")
        ts_val      = trailing_sl if trailing_sl > 0 else float("-inf")
        effective   = max(hard_sl, ts_val)

        if effective > float("-inf") and low <= effective:
            fill_price = effective * (1.0 - SL_SLIPPAGE_PCT)
            if open_ < effective:               # 갭하락: 시초가가 손절선 하회
                fill_price = min(fill_price, open_)
            fill_price = max(fill_price, low)   # 당일 최저가 밑으로는 체결 불가
            if trailing_sl > 0 and trailing_sl >= stop_loss:
                action = "SELL_TS"
                reason = f"트레일링 스톱 (최고가 {prev_peak:,.0f} → 청산선 {effective:,.0f}, 체결 {fill_price:,.0f})"
            else:
                action = "SELL_SL"
                reason = f"손절선 터치 (손절선 {stop_loss:,.0f}, 체결 {fill_price:,.0f})"
        elif take_profit > 0 and high >= take_profit:
            action = "SELL_TP"
            fill_price = take_profit
            reason = f"익절 목표가 도달 ({take_profit:,.0f})"
        else:
            # 미청산 — 당일 고가를 peak 에 반영 (다음 틱 트레일링 추적용)
            positions[ticker]["peak_price"] = max(prev_peak, high)

        if action:
            pnl = (fill_price - entry_price) / entry_price * 100 if entry_price > 0 else 0.0
            signals.append({
                "ticker":      ticker,
                "name":        pos.get("name", ticker),
                "action":      action,
                "price":       fill_price,
                "entry_price": entry_price,
                "pnl_pct":     pnl,
                "reason":      reason,
            })

    return signals

# ══════════════════════════════════════════════════════════════════════════════
# 1회 스캔 실행
# ══════════════════════════════════════════════════════════════════════════════

def run_scan(
    universe_map: dict[str, str],
    strategy: TradingStrategy,
    positions: dict,
    alerted: dict,
    lockout: dict,
    bear: bool,
) -> tuple[dict, dict, dict]:
    now_str = datetime.now(KST).strftime("%H:%M")
    today_str = datetime.now(KST).strftime("%Y-%m-%d")
    logger.info(f"=== 스캔 {now_str} ({'하락장' if bear else '상승장'}) ===")

    # 유니버스 + 보유 종목 모두 포함
    tickers = list(universe_map.keys())
    for t in positions:
        if t not in universe_map:
            tickers.append(t)

    ticker_data = fetch_ticker_data(tickers)
    if not ticker_data:
        logger.warning("데이터 없음 — 스캔 스킵")
        return positions, alerted, lockout

    # ── 매도 우선 ──────────────────────────────────────────────────────────────
    sell_signals = check_sell_signals(positions, ticker_data, strategy)
    for sig in sell_signals:
        ticker    = sig["ticker"]
        alert_key = f"sell_{ticker}_{sig['action']}"
        if alert_key not in alerted:
            icon = {"SELL_TP": "💵✅", "SELL_TS": "🎯✅", "SELL_SL": "💵❌"}.get(sig["action"], "🔴")
            text = (
                f"{icon} <b>[매도 신호] {sig['name']} ({ticker})</b>\n"
                f"유형: {sig['action']}\n"
                f"체결가: {sig['price']:,.0f}원\n"
                f"진입가: {sig['entry_price']:,.0f}원\n"
                f"손익: {sig['pnl_pct']:+.1f}%\n"
                f"사유: {sig['reason']}"
            )
            if send_telegram(text):
                alerted[alert_key] = now_str
                logger.info(f"매도 알림 발송: {ticker} {sig['action']}")
        # [개편3] 연쇄손절 집계 — SELL_SL 발생 일자 기록
        if sig["action"] == "SELL_SL":
            lockout["sl_dates"].append(today_str)
        del positions[ticker]

    # ── [개편3] 시장 필터 연동 락아웃 평가 ──────────────────────────────────────
    cutoff = (datetime.now(KST) - timedelta(days=SL_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    lockout["sl_dates"] = [d for d in lockout["sl_dates"] if d >= cutoff]
    if lockout["locked"]:
        if market_recovered():
            lockout["locked"] = False
            logger.info("🔓 락아웃 해제 — 지수 20일선 안착")
            send_telegram("🔓 <b>락아웃 해제</b> — KOSPI/KOSDAQ 20일선 안착, 신규 매수 재개")
    elif len(lockout["sl_dates"]) >= SL_LOCKOUT_COUNT:
        lockout["locked"] = True
        logger.info(f"🔒 락아웃 발동 — 최근 {SL_LOOKBACK_DAYS}일 SELL_SL {len(lockout['sl_dates'])}회")
        send_telegram(
            f"🔒 <b>연쇄손절 락아웃 발동</b>\n"
            f"최근 {SL_LOOKBACK_DAYS}일 SELL_SL {len(lockout['sl_dates'])}회 → "
            f"지수(KOSPI/KOSDAQ) 20일선 안착까지 신규 매수 전면 중단(현금 보유)."
        )

    # ── 매수 후보 스크리닝 (락아웃 중이면 전면 스킵) ────────────────────────────
    if lockout["locked"]:
        logger.info("락아웃 활성 — 신규 매수 스킵")
        return positions, alerted, lockout

    buy_candidates = screen_buy_candidates(ticker_data, strategy)
    new_buys = {t: tag for t, tag in buy_candidates.items() if t not in positions}

    if new_buys:
        for ticker, tag in new_buys.items():
            alert_key = f"buy_{ticker}_{today_str}"
            if alert_key in alerted:
                continue  # 당일 동일 종목 중복 알림 방지

            df = ticker_data.get(ticker)
            if df is None:
                continue
            row     = df.iloc[-1]
            price   = float(row.get("Close", 0))
            atr_raw = float(row.get("ATR",   np.nan))
            if price <= 0:
                continue

            atr_val = atr_raw if not np.isnan(atr_raw) and atr_raw > 0 else -1.0
            # [개편1] 신호별 손절가 + ATR 고정 리스크 사이징
            sl = resolve_stop_loss(price, atr_val, tag)
            if sl <= 0 or sl >= price:
                continue
            tp = price * 1.25 if tag == "SMA_GOLDEN_CROSS" else (
                strategy.get_exit_price(price, atr_val, is_profit=True) if atr_val > 0 else price * 1.25
            )
            qty = calc_position_qty(price, sl, ACCOUNT_CAPITAL)
            if qty <= 0:
                continue
            invest   = qty * price
            risk_amt = (price - sl) * qty

            market_label = "⚠️ 하락장" if bear else "✅ 상승장"
            tag_label    = {"SMA_GOLDEN_CROSS": "SMA 골든크로스", "SMA_TREND_FOLLOW": "SMA 추세 추종"}.get(tag, tag)
            text = (
                f"🟢 <b>[매수 신호] {universe_map.get(ticker, ticker)} ({ticker})</b>\n"
                f"신호: {tag_label}\n"
                f"현재가: {price:,.0f}원\n"
                f"추천 수량: 약 {qty}주 (투입 {invest:,.0f}원 / 비중 {invest / ACCOUNT_CAPITAL * 100:.0f}%)\n"
                f"익절 목표: {tp:,.0f}원\n"
                f"손절선: {sl:,.0f}원\n"
                f"리스크: {risk_amt:,.0f}원 (총자산 {risk_amt / ACCOUNT_CAPITAL * 100:.2f}%)\n"
                f"시장: {market_label}"
            )
            if send_telegram(text):
                alerted[alert_key] = now_str
                logger.info(f"매수 알림 발송: {ticker} ({tag})")

            # 포지션 등록
            positions[ticker] = {
                "name":        universe_map.get(ticker, ticker),
                "entry_price": price,
                "peak_price":  price,
                "take_profit": tp,
                "stop_loss":   sl,
                "signal_tag":  tag,
                "quantity":    qty,
                "entry_time":  datetime.now(KST).isoformat(),
            }
    else:
        logger.info("매수 신호 없음")

    return positions, alerted, lockout

# ══════════════════════════════════════════════════════════════════════════════
# 타이밍 헬퍼
# ══════════════════════════════════════════════════════════════════════════════

def sleep_until_next_tick(interval_min: int = SCAN_INTERVAL_MIN) -> None:
    now        = datetime.now(KST)
    elapsed_s  = now.second + (now.minute % interval_min) * 60
    wait_s     = interval_min * 60 - elapsed_s
    if wait_s < 10:
        wait_s += interval_min * 60
    next_tick  = now + timedelta(seconds=wait_s)
    logger.info(f"다음 스캔: {next_tick.strftime('%H:%M:%S')}  ({wait_s}초 후)")
    time.sleep(wait_s)


def sleep_until_market_open() -> None:
    now       = datetime.now(KST)
    open_time = now.replace(
        hour=MARKET_OPEN[0], minute=MARKET_OPEN[1], second=0, microsecond=0
    )
    if now >= open_time:
        return
    wait_s = (open_time - now).total_seconds()
    logger.info(f"장 시작(09:00)까지 {wait_s / 60:.0f}분 대기")
    time.sleep(wait_s)

# ══════════════════════════════════════════════════════════════════════════════
# 메인 루프
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("❌ TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID가 .env에 없습니다. 종료합니다.")
        sys.exit(1)

    strategy = TradingStrategy()
    screener = StockScreener(universe_per_market=UNIVERSE_N)

    print("=" * 60)
    print("  Auto Stock Analyzer — 실시간 스케줄러")
    print("=" * 60)
    print(f"  마켓  : {SCAN_MARKETS}")
    print(f"  주기  : {SCAN_INTERVAL_MIN}분  (09:00 ~ 15:20 KST, 평일)")
    print(f"  리스크: 종목당 총자산 {MAX_RISK_PER_TRADE:.0%} 고정 / 비중 상한 {MAX_POSITION_PCT:.0%} "
          f"(기준 자본 {ACCOUNT_CAPITAL:,.0f}원)")
    print(f"  필터  : RSI≥{RSI_HARD_LIMIT:.0f} 진입금지 / 연쇄손절 {SL_LOCKOUT_COUNT}회→20일선 안착까지 락아웃")
    print(f"  텔레그램 chat_id: {TELEGRAM_CHAT_ID}")
    print("=" * 60)

    send_telegram(
        "🚀 <b>Auto Stock Analyzer 시작</b>\n"
        f"09:00~15:20 KST, {SCAN_INTERVAL_MIN}분 주기 스캔을 시작합니다."
    )

    universe_map:  dict[str, str] = {}
    universe_date: str            = ""
    positions  = load_positions()
    alerted:   dict[str, str]     = {}
    lockout    = load_lockout()   # [개편3] 연쇄손절 락아웃 상태 복원

    while True:
        now = datetime.now(KST)

        # ── 주말 스킵 ──────────────────────────────────────────────────────────
        if now.weekday() >= 5:
            logger.info("주말 — 1시간 후 재확인")
            time.sleep(3600)
            continue

        open_time  = now.replace(hour=MARKET_OPEN[0],  minute=MARKET_OPEN[1],  second=0, microsecond=0)
        close_time = now.replace(hour=MARKET_CLOSE[0], minute=MARKET_CLOSE[1], second=0, microsecond=0)

        # ── 장 시작 전 대기 ────────────────────────────────────────────────────
        if now < open_time:
            sleep_until_market_open()
            continue

        # ── 장 종료 후: 다음날 09:00까지 대기 ────────────────────────────────
        if now > close_time:
            tomorrow_open = (now + timedelta(days=1)).replace(
                hour=MARKET_OPEN[0], minute=MARKET_OPEN[1], second=0, microsecond=0
            )
            # 다음날이 주말이면 월요일로 밀기
            while tomorrow_open.weekday() >= 5:
                tomorrow_open += timedelta(days=1)
            wait_s = (tomorrow_open - now).total_seconds()
            logger.info(f"장 마감 — {tomorrow_open.strftime('%m/%d %H:%M')}까지 {wait_s/3600:.1f}시간 대기")
            save_positions(positions)
            alerted = {}
            time.sleep(wait_s)
            continue

        # ── 유니버스 (하루 1회 갱신) ──────────────────────────────────────────
        today_str = now.strftime("%Y-%m-%d")
        if universe_date != today_str or not universe_map:
            logger.info("유니버스 갱신 중...")
            universe    = screener.build_universe(SCAN_MARKETS)
            universe_map = {t: n for t, n in universe}
            universe_date = today_str
            alerted     = load_alerted()
            logger.info(f"유니버스 {len(universe_map)}종목 로드")

        # ── KOSDAQ 시장 상황 (1시간 캐시) ────────────────────────────────────
        bear = is_bear_market()

        # ── 스캔 실행 ─────────────────────────────────────────────────────────
        try:
            positions, alerted, lockout = run_scan(
                universe_map, strategy, positions, alerted, lockout, bear
            )
        except Exception as e:
            logger.error(f"스캔 오류: {e}", exc_info=True)

        save_positions(positions)
        save_alerted(alerted)
        save_lockout(lockout)

        sleep_until_next_tick()


if __name__ == "__main__":
    main()
