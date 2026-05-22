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
  SCAN_MARKETS      — 대상 마켓 목록
  UNIVERSE_N        — 마켓별 유니버스 크기
  VOLUME_TOP_N      — 거래대금 상위 N (매수 후보 1차 필터)
  POSITION_PCT      — 정상 베팅 비중 (총자산 대비)
  POSITION_PCT_BEAR — 하락장 베팅 비중 (KOSDAQ < SMA20 시 적용)
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

POSITION_PCT      = 0.10      # 정상 베팅 비중
POSITION_PCT_BEAR = 0.05      # 하락장 베팅 비중

POSITIONS_FILE = Path(__file__).parent / "live_positions.json"
ALERTED_FILE   = Path(__file__).parent / "live_alerted.json"

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
        atr   = float(row.get("ATR",   np.nan))
        if price <= 0:
            continue

        atr_val     = atr if not np.isnan(atr) and atr > 0 else -1.0
        entry_price = float(pos.get("entry_price", 0))
        peak_price  = max(float(pos.get("peak_price", price)), price)
        take_profit = float(pos.get("take_profit", -1))
        stop_loss   = float(pos.get("stop_loss",   -1))

        positions[ticker]["peak_price"] = peak_price  # 최고가 갱신

        action = reason = None

        if take_profit > 0 and price >= take_profit:
            action = "SELL_TP"
            reason = f"익절 목표가 도달 ({take_profit:,.0f})"
        else:
            trailing_sl = strategy.get_trailing_stop(peak_price, atr_val) if atr_val > 0 else -1.0
            hard_sl     = stop_loss   if stop_loss   > 0 else float("-inf")
            ts_val      = trailing_sl if trailing_sl > 0 else float("-inf")
            effective   = max(hard_sl, ts_val)
            if effective > float("-inf") and price <= effective:
                if trailing_sl > 0 and trailing_sl >= stop_loss:
                    action = "SELL_TS"
                    reason = f"트레일링 스톱 (최고가 {peak_price:,.0f} → 청산선 {effective:,.0f})"
                else:
                    action = "SELL_SL"
                    reason = f"손절가 도달 ({stop_loss:,.0f})"

        if action:
            pnl = (price - entry_price) / entry_price * 100 if entry_price > 0 else 0.0
            signals.append({
                "ticker":      ticker,
                "name":        pos.get("name", ticker),
                "action":      action,
                "price":       price,
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
    bear: bool,
) -> tuple[dict, dict]:
    now_str = datetime.now(KST).strftime("%H:%M")
    logger.info(f"=== 스캔 {now_str} ({'하락장' if bear else '상승장'}) ===")

    # 유니버스 + 보유 종목 모두 포함
    tickers = list(universe_map.keys())
    for t in positions:
        if t not in universe_map:
            tickers.append(t)

    ticker_data = fetch_ticker_data(tickers)
    if not ticker_data:
        logger.warning("데이터 없음 — 스캔 스킵")
        return positions, alerted

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
                f"현재가: {sig['price']:,.0f}원\n"
                f"진입가: {sig['entry_price']:,.0f}원\n"
                f"손익: {sig['pnl_pct']:+.1f}%\n"
                f"사유: {sig['reason']}"
            )
            if send_telegram(text):
                alerted[alert_key] = now_str
                logger.info(f"매도 알림 발송: {ticker} {sig['action']}")
        del positions[ticker]

    # ── 매수 후보 스크리닝 ──────────────────────────────────────────────────────
    buy_candidates = screen_buy_candidates(ticker_data, strategy)
    new_buys = {t: tag for t, tag in buy_candidates.items() if t not in positions}

    if new_buys:
        today_str    = datetime.now(KST).strftime("%Y-%m-%d")
        sizing_label = f"{POSITION_PCT_BEAR:.0%}" if bear else f"{POSITION_PCT:.0%}"

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
            tp = strategy.get_exit_price(price, atr_val, is_profit=True)  if atr_val > 0 else -1.0
            sl = strategy.get_exit_price(price, atr_val, is_profit=False) if atr_val > 0 else -1.0

            market_label = "⚠️ 하락장 (베팅 축소)" if bear else "✅ 상승장"
            tag_label    = {"SMA_GOLDEN_CROSS": "SMA 골든크로스", "SMA_TREND_FOLLOW": "SMA 추세 추종"}.get(tag, tag)
            text = (
                f"🟢 <b>[매수 신호] {universe_map.get(ticker, ticker)} ({ticker})</b>\n"
                f"신호: {tag_label}\n"
                f"현재가: {price:,.0f}원\n"
                f"익절 목표: {tp:,.0f}원\n" if tp > 0 else ""
                f"손절선: {sl:,.0f}원\n"   if sl > 0 else ""
                f"시장: {market_label}\n"
                f"베팅 비중: {sizing_label}"
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
                "entry_time":  datetime.now(KST).isoformat(),
            }
    else:
        logger.info("매수 신호 없음")

    return positions, alerted

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
    print(f"  베팅  : 정상 {POSITION_PCT:.0%}  / 하락장 {POSITION_PCT_BEAR:.0%}")
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
            positions, alerted = run_scan(universe_map, strategy, positions, alerted, bear)
        except Exception as e:
            logger.error(f"스캔 오류: {e}", exc_info=True)

        save_positions(positions)
        save_alerted(alerted)

        sleep_until_next_tick()


if __name__ == "__main__":
    main()
