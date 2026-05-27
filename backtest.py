"""
backtest.py — 날짜 중심(Date-Driven) 백테스팅 엔진 (고승률 국장 최적화 버전)

매 거래일마다 전체 유니버스에서 실시간으로 종목을 선정한 뒤
매수·매도를 집행합니다.

수정 사항:
  - SMA_TREND_FOLLOW(추세 추종) 로직 완전 제거 (오직 신선한 골든크로스만)
  - 거래대금 스파이크 필터 추가 (당일 거래대금이 최근 20일 평균의 3배 이상 폭발 시만 진입)
  - 펀드급 리스크 관리 반영 (동시 진입 락, 최대 3종목 제한, 예수금 하드캡)
"""
from __future__ import annotations

import csv
import logging
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import yfinance as yf

# Windows 콘솔에서 유니코드 출력 강제 설정
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

from strategy import TradingStrategy
from stock_ai import _add_indicators, _flatten_columns

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    level=logging.WARNING,
)
logger = logging.getLogger("backtest")

# ══════════════════════════════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════════════════════════════

SCREEN_MARKETS    = ["KOSPI", "KOSDAQ"]   # 대상 마켓
SCREEN_UNIVERSE_N = 200                   # 마켓별 후보 종목 수 (시가총액 상위)
SCREEN_TOP_N      = 20                    # StockScreener 단독 실행 시 최종 선정 수

VOLUME_TOP_N      = 100                   # 1단계 필터: 일별 거래대금 상위 N
NEWS_CANDIDATE_N  = 15                    # 3단계 필터: 뉴스 분석 대상 상위 N

START_DATE = "2020-01-01"
END_DATE   = "2024-12-31"

INITIAL_CAPITAL = 10_000_000   # 1,000만 원

# 추가 입금 스케줄
DEPOSIT_SCHEDULE: dict[str, int] = {
    "2021-01-04": 5_000_000,
    "2022-01-03": 5_000_000,
    "2023-01-02": 5_000_000,
}

RESULT_DIR = Path(__file__).parent / "backtest_results"

# ══════════════════════════════════════════════════════════════════════════════
# 펀드 회계 시뮬레이터
# ══════════════════════════════════════════════════════════════════════════════

class TradingSimulator:
    def __init__(self, initial_capital: float):
        self.cash             = initial_capital
        self.portfolio: dict[str, int] = {}
        self.invested_capital = initial_capital
        self.total_units      = initial_capital
        self.unit_price       = 1.0

    def get_total_asset(self, current_market_prices: dict[str, float]) -> float:
        stock_value = sum(
            qty * current_market_prices.get(code, 0)
            for code, qty in self.portfolio.items()
        )
        return self.cash + stock_value

    def update_unit_price(self, current_market_prices: dict[str, float]) -> float:
        total_asset = self.get_total_asset(current_market_prices)
        if self.total_units > 0:
            self.unit_price = total_asset / self.total_units
        return self.unit_price

    def deposit(self, amount: float, current_market_prices: dict[str, float]) -> None:
        self.update_unit_price(current_market_prices)
        new_units = amount / self.unit_price
        self.cash             += amount
        self.invested_capital += amount
        self.total_units      += new_units
        print(f"  💰 {amount:>12,.0f}원 추가 입금  "
              f"(기준가 {self.unit_price:.4f} / 발행좌수 {new_units:,.2f})")

    def execute_buy(self, code: str, price: float, quantity: int) -> bool:
        cost = price * quantity
        if self.cash < cost:
            return False
        self.cash -= cost
        self.portfolio[code] = self.portfolio.get(code, 0) + quantity
        return True

    def execute_sell(self, code: str, price: float, quantity: int) -> bool:
        if self.portfolio.get(code, 0) < quantity:
            return False
        self.cash += price * quantity
        self.portfolio[code] -= quantity
        if self.portfolio[code] == 0:
            del self.portfolio[code]
        return True

    def get_current_return(self, current_market_prices: dict[str, float]) -> float:
        self.update_unit_price(current_market_prices)
        return (self.unit_price - 1.0) * 100


# ══════════════════════════════════════════════════════════════════════════════
# 종목별 포지션 상태
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PositionState:
    in_position:   bool  = False
    entry_price:   float = 0.0
    rebuy_count:   int   = 0
    take_profit:   float = -1.0
    stop_loss:     float = -1.0
    peak_price:    float = 0.0
    signal_tag:    str   = ""
    allocated_pct: float = 0.0

    def reset(self) -> None:
        self.in_position   = False
        self.entry_price   = 0.0
        self.rebuy_count   = 0
        self.take_profit   = -1.0
        self.stop_loss     = -1.0
        self.peak_price    = 0.0
        self.signal_tag    = ""
        self.allocated_pct = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 백테스팅 엔진 (날짜 중심)
# ══════════════════════════════════════════════════════════════════════════════

class BacktestEngine:
    WARMUP_DAYS       = 300   # 지표 안정화에 필요한 워밍업 일수
    MAX_POSITIONS     = 3     # ★ [안전장치 1] 최대 보유 종목 수 제한

    def __init__(
        self,
        initial_capital: float,
        start_date: str,
        end_date: str,
        markets: list[str] | None = None,
        universe_n: int = 200,
        volume_top_n: int = 100,
        news_candidate_n: int = 15,
        deposit_schedule: dict[str, float] | None = None,
        ticker_name_map: dict[str, str] | None = None,
    ):
        self.markets          = markets or ["KOSPI", "KOSDAQ"]
        self.universe_n       = universe_n
        self.volume_top_n     = volume_top_n
        self.news_candidate_n = news_candidate_n
        self.start_date       = pd.Timestamp(start_date)
        self.end_date         = pd.Timestamp(end_date)
        self.deposit_schedule = deposit_schedule or {}
        self._ticker_names: dict[str, str] = ticker_name_map or {}
        self._market_factor: float = 1.0

        self.simulator    = TradingSimulator(initial_capital)
        self.strategy     = TradingStrategy()
        self.positions:    dict[str, PositionState] = {}
        self.trade_log:    list[dict] = []
        self.equity_curve: list[dict] = []
        self._kosdaq_df:   Optional[pd.DataFrame] = None

    def _load_data(self) -> dict[str, pd.DataFrame]:
        load_start = self.start_date - pd.Timedelta(days=self.WARMUP_DAYS)
        load_end   = self.end_date   + pd.Timedelta(days=2)

        if self._ticker_names:
            all_tickers = list(self._ticker_names.keys())
            print(f"  유니버스 크기: {len(all_tickers)}개 종목 (사전 선정)")
        else:
            screener = StockScreener(universe_per_market=self.universe_n)
            universe = screener.build_universe(self.markets)
            all_tickers = [t for t, _ in universe]
            self._ticker_names = {t: n for t, n in universe}
            print(f"  유니버스 크기: {len(all_tickers)}개 종목")

        ticker_data: dict[str, pd.DataFrame] = {}
        CHUNK = 50

        for i in range(0, len(all_tickers), CHUNK):
            chunk = all_tickers[i: i + CHUNK]
            hi = min(i + CHUNK, len(all_tickers))
            print(f"  [{i + 1}~{hi}] 배치 다운로드 중...", end=" ", flush=True)
            try:
                raw = yf.download(
                    chunk if len(chunk) > 1 else chunk[0],
                    start=load_start.strftime("%Y-%m-%d"),
                    end=load_end.strftime("%Y-%m-%d"),
                    auto_adjust=True,
                    progress=False,
                )
            except Exception as e:
                print(f"오류: {e}")
                continue

            if raw.empty:
                print("데이터 없음")
                continue

            for ticker in chunk:
                try:
                    if isinstance(raw.columns, pd.MultiIndex):
                        df = raw.xs(ticker, axis=1, level=1).dropna(how="all")
                    else:
                        df = raw.dropna(how="all")

                    if df.empty or len(df) < 60:
                        continue

                    df = _flatten_columns(df)
                    df = _add_indicators(df)
                    ticker_data[ticker] = df
                except Exception:
                    continue

            print(f"누적 {len(ticker_data)}개 유효")

        return ticker_data

    def _load_kosdaq_index(self) -> None:
        load_start = self.start_date - pd.Timedelta(days=self.WARMUP_DAYS)
        load_end   = self.end_date   + pd.Timedelta(days=2)
        print("  KOSDAQ 지수(^KQ11) 로딩 중...", end=" ", flush=True)
        try:
            raw = yf.download(
                "^KQ11",
                start=load_start.strftime("%Y-%m-%d"),
                end=load_end.strftime("%Y-%m-%d"),
                auto_adjust=True,
                progress=False,
            )
            if raw.empty:
                print("데이터 없음 (시장 조건 체크 비활성화)")
                return
            df = _flatten_columns(raw)
            df["SMA_20"] = df["Close"].rolling(20).mean()
            self._kosdaq_df = df
            print(f"{len(df)}일치 로드 완료")
        except Exception as e:
            print(f"실패 ({e})")

    def _is_bear_market(self, date: pd.Timestamp) -> bool:
        if self._kosdaq_df is None:
            return False
        locs = self._kosdaq_df.index.get_indexer([date], method="ffill")
        if locs[0] < 0:
            return False
        row   = self._kosdaq_df.iloc[locs[0]]
        close = float(row.get("Close",  np.nan))
        sma20 = float(row.get("SMA_20", np.nan))
        if np.isnan(close) or np.isnan(sma20):
            return False
        return close < sma20

    # ── [수정] 진입 점수 산정: 거래대금 스파이크 강도와 스케일 연동 ────────────────

    def _score_entry(
        self,
        df: pd.DataFrame,
        idx: int,
        spike_ratio: float
    ) -> float:
        """
        골든크로스의 강도를 0~100점으로 환산한다.
        - Base (50): 신선한 골든크로스 기본 점수
        - SMA 갭 보너스 (0~20): SMA5-SMA20 이격률
        - SMA 기울기 보너스 (0~15): SMA5 상승 속도
        - 거래대금 스파이크 보너스 (0~15): ★ 20일 평균 대비 당일 거래대금 비율 연동
        """
        row  = df.iloc[idx]
        prev = df.iloc[idx - 1]

        sma5      = float(row.get("SMA_5",  np.nan))
        sma20     = float(row.get("SMA_20", np.nan))
        prev_sma5 = float(prev.get("SMA_5", np.nan))

        if any(np.isnan(v) for v in [sma5, sma20, prev_sma5]) or sma20 <= 0 or prev_sma5 <= 0:
            return 50.0

        base = 50.0  # 추세 추종이 제거되었으므로 고정 기본값

        gap_pct   = (sma5 - sma20) / sma20 * 100
        gap_bonus = min(20.0, max(0.0, gap_pct * 2.0))

        slope_pct   = (sma5 - prev_sma5) / prev_sma5 * 100
        slope_bonus = min(15.0, max(0.0, slope_pct * 300.0))

        # ★ 거래대금 스파이크 배수별 점수 스케일링 (3배=5점, 5배=10점, 10배 이상=15점 만점)
        if spike_ratio >= 10.0:
            vol_bonus = 15.0
        elif spike_ratio >= 5.0:
            vol_bonus = 10.0
        elif spike_ratio >= 3.0:
            vol_bonus = 5.0
        else:
            vol_bonus = 0.0

        return min(100.0, round(base + gap_bonus + slope_bonus + vol_bonus, 1))

    # ── [수정] 일일 종목 스크리닝: 거래량 급감 20일선 눌림목 + 하락장 전면 금지 ───────────

    def _screen_daily(
        self,
        date: pd.Timestamp,
        ticker_data: dict[str, pd.DataFrame],
    ) -> dict[str, tuple[str, float]]:

        # 1. 대세 하락장 시 매수 전면 금지 (현금 100% 보존 쉴드)
        if self._is_bear_market(date):
            return {}

        # 시가총액/거래대금 상위 유니버스 필터링
        turnover_list: list[tuple[str, float]] = []
        for ticker, df in ticker_data.items():
            if date not in df.index:
                continue
            idx_t = df.index.get_loc(date)
            if idx_t < 1:
                continue
            prev_row = df.iloc[idx_t - 1]
            turnover_list.append((ticker, float(prev_row.get("Close", 0)) * float(prev_row.get("Volume", 0))))

        turnover_list.sort(key=lambda x: x[1], reverse=True)
        top_volume = [t for t, _ in turnover_list[: self.volume_top_n]]

        candidates: dict[str, tuple[str, float]] = {}
        for ticker in top_volume:
            df = ticker_data[ticker]
            if date not in df.index:
                continue
            idx = df.index.get_loc(date)
            if idx < 30:
                continue

            row   = df.iloc[idx]
            close = float(row["Close"])
            vol   = float(row["Volume"])
            sma20 = float(row.get("SMA_20", np.nan))

            if np.isnan(sma20) or sma20 <= 0:
                continue

            # ─── [필승 필터] 세력 등에 올라타는 찐 눌림목 3단 필터 ───

            # ① 조건 1 (주도주 흔적): 최근 10일 이내에 거래대금이 20일 평균 대비 5배 이상 폭발한 적이 있는가?
            has_past_spike = False
            max_spike_ratio = 1.0

            for lookback_idx in range(max(0, idx - 10), idx):
                p_row = df.iloc[lookback_idx]
                p_turnover = float(p_row["Close"]) * float(p_row["Volume"])
                p_avg_turnover = (
                    df["Volume"].iloc[max(0, lookback_idx - 19):lookback_idx + 1]
                    * df["Close"].iloc[max(0, lookback_idx - 19):lookback_idx + 1]
                ).mean()

                p_spike_ratio = p_turnover / p_avg_turnover if p_avg_turnover > 0 else 0.0
                if p_spike_ratio >= 5.0:
                    has_past_spike = True
                    if p_spike_ratio > max_spike_ratio:
                        max_spike_ratio = p_spike_ratio

            if has_past_spike:
                # ② 조건 2 (눌림 확인): 오늘 주가가 20일선까지 조정받아 왔는가? (-1% ~ +2%)
                if (sma20 * 0.99) <= close <= (sma20 * 1.02):

                    # ③ 조건 3 (거래량 숨고르기): 오늘 거래량이 20일 평균의 70% 이하로 말라붙었는가?
                    avg_vol_20 = df["Volume"].iloc[max(0, idx - 19):idx + 1].mean()
                    if vol <= avg_vol_20 * 0.7:
                        tag   = "DRY_VOLUME_NULIM"
                        score = self._score_entry(df, idx, max_spike_ratio)
                        candidates[ticker] = (tag, score)

        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1][1], reverse=True)
        return dict(sorted_candidates[: self.news_candidate_n])

    def _try_sell(
        self,
        date_str: str,
        ticker: str,
        df: pd.DataFrame,
        idx: int,
        current_prices: dict[str, float],
    ) -> None:
        pos = self.positions.get(ticker)
        if pos is None or not pos.in_position:
            return

        row   = df.iloc[idx]
        price = float(row["Close"])
        atr   = float(row.get("ATR", np.nan))
        if np.isnan(atr) or atr <= 0:
            atr = -1.0

        pos.peak_price = max(pos.peak_price, price)

        # 익절 목표가 도달
        if pos.take_profit > 0 and price >= pos.take_profit:
            qty = self.simulator.portfolio.get(ticker, 0)
            if qty and self.simulator.execute_sell(ticker, price, qty):
                pnl_pct = (price - pos.entry_price) / pos.entry_price * 100
                self._log_trade(
                    date_str, ticker, "SELL_TP", price, qty,
                    f"익절 목표가 도달 ({pos.take_profit:,.0f}, P/L: {pnl_pct:+.1f}%)",
                    entry_price=pos.entry_price,
                    signal_tag=pos.signal_tag,
                )
                del self.positions[ticker]
            return

        # 트레일링 스톱 + 하드 손절
        trailing_sl  = self.strategy.get_trailing_stop(pos.peak_price, atr) if atr > 0 else -1.0
        hard_sl      = pos.stop_loss if pos.stop_loss  > 0 else float("-inf")
        ts_val       = trailing_sl   if trailing_sl    > 0 else float("-inf")
        effective_sl = max(hard_sl, ts_val)

        if effective_sl > float("-inf") and price <= effective_sl:
            qty = self.simulator.portfolio.get(ticker, 0)
            if qty and self.simulator.execute_sell(ticker, price, qty):
                pnl_pct = (price - pos.entry_price) / pos.entry_price * 100
                if trailing_sl > 0 and trailing_sl >= pos.stop_loss:
                    action      = "SELL_TS"
                    reason_text = (f"트레일링 스톱 (최고가 {pos.peak_price:,.0f} → "
                                   f"청산선 {effective_sl:,.0f}, P/L: {pnl_pct:+.1f}%)")
                else:
                    action      = "SELL_SL"
                    reason_text = f"손절가 도달 ({pos.stop_loss:,.0f}, P/L: {pnl_pct:+.1f}%)"
                self._log_trade(date_str, ticker, action, price, qty, reason_text,
                                entry_price=pos.entry_price, signal_tag=pos.signal_tag)
                del self.positions[ticker]
            return

        # 분할매수 (물타기)
        if self.strategy.is_rebuy_signal(price, pos.entry_price, pos.rebuy_count):
            total_asset  = self.simulator.get_total_asset(current_prices)
            rebuy_pct    = pos.allocated_pct if pos.allocated_pct > 0 else 0.10
            alloc_cash   = total_asset * rebuy_pct

            # 예수금 실시간 초과 방지 락
            if alloc_cash > self.simulator.cash:
                alloc_cash = self.simulator.cash * 0.9

            qty = int(alloc_cash / price)
            if qty > 0 and self.simulator.execute_buy(ticker, price, qty):
                pos.rebuy_count += 1
                self._log_trade(
                    date_str, ticker, f"REBUY#{pos.rebuy_count}",
                    price, qty,
                    f"{pos.rebuy_count}차 분할매수 [배정 {alloc_cash:,.0f}원]",
                    signal_tag=pos.signal_tag,
                )

    # ── [수정] 매수 집행: 펀드급 4대 리스크 관리 기능 결합 ───────────────────────────

    def _try_buy(
        self,
        date_str: str,
        ticker: str,
        df: pd.DataFrame,
        idx: int,
        current_prices: dict[str, float],
        signal_tag: str = "UNKNOWN",
        score: float = 50.0,
    ) -> None:
        row   = df.iloc[idx]
        price = float(row["Close"])
        atr   = float(row.get("ATR", np.nan))
        if np.isnan(atr) or atr <= 0:
            atr = -1.0

        # ★ [안전장치 1] 최대 보유 종목 수 상한 락 (Max Positions = 3)
        if len(self.positions) >= self.MAX_POSITIONS:
            return

        # ★ [안전장치 2] 실시간 예수금 잔고 검증 (최소 1주 가격도 없으면 차단)
        if self.simulator.cash < price:
            return

        # ★ [자율 비중] 확신도 점수대별 비중 현실화 (보수적 세팅)
        if score >= 85:
            alloc_pct = 0.50  # 역대급 조건: 자산의 50%
        elif score >= 70:
            alloc_pct = 0.25  # 우수한 조건: 자산의 25%
        elif score >= 55:
            alloc_pct = 0.10  # 보통 조건: 자산의 10%
        else:
            return            # 55점 미만 패스

        # ★ [안전장치 4] 하락장 디펜스 쉴드 강화 (베팅 비중을 평소의 30%로 강제 축소)
        if self._is_bear_market(pd.Timestamp(date_str)):
            alloc_pct *= 0.3
            signal_tag += "_BEAR_SHIELD"

        total_asset = self.simulator.get_total_asset(current_prices)
        alloc_cash  = total_asset * alloc_pct

        # ★ [안전장치 3] 가용 현금 실시간 락 (남은 현금을 오버하면 잔고의 90%만 채우도록 하드캡)
        if alloc_cash > self.simulator.cash:
            alloc_cash = self.simulator.cash * 0.9

        qty = int(alloc_cash / price)
        if qty <= 0:
            return

        if not self.simulator.execute_buy(ticker, price, qty):
            return

        tp = self.strategy.get_exit_price(price, atr, is_profit=True)  if atr > 0 else -1.0
        sl = self.strategy.get_exit_price(price, atr, is_profit=False) if atr > 0 else -1.0

        pos = PositionState()
        pos.in_position   = True
        pos.entry_price   = price
        pos.peak_price    = price
        pos.rebuy_count   = 0
        pos.take_profit   = tp
        pos.stop_loss     = sl
        pos.signal_tag    = signal_tag
        pos.allocated_pct = alloc_pct
        self.positions[ticker] = pos

        self._log_trade(
            date_str, ticker, "BUY", price, qty,
            f"점수 {score:.1f}pt → 비중 {alloc_pct:.1%}, 배정 {alloc_cash:,.0f}원",
            signal_tag=signal_tag,
        )

    def _log_trade(self, date: str, ticker: str, action: str,
                   price: float, qty: int, reason: str,
                   entry_price: float = 0.0,
                   signal_tag: str = "") -> None:
        self.trade_log.append({
            "date":        date,
            "ticker":      ticker,
            "name":        self._ticker_names.get(ticker, ticker),
            "action":      action,
            "price":       price,
            "qty":         qty,
            "amount":      price * qty,
            "reason":      reason,
            "entry_price": entry_price,
            "signal_tag":  signal_tag,
        })
        icon = {"BUY": "🛒", "SELL_TP": "💵✅", "SELL_SL": "💵❌", "SELL_TS": "🎯✅"}.get(action, "↩️")
        display = self._ticker_names.get(ticker, ticker)
        print(f"  {icon} {date} [{display}] {action:10s}  "
              f"가격 {price:>10,.0f}  수량 {qty:>5}  ({reason})")

    def run(self) -> None:
        print("\n" + "=" * 65)
        print("  백테스팅 엔진 시작 (고승률 국장 최적화 버전)")
        print("=" * 65)
        print(f"  기간     : {self.start_date.date()} ~ {self.end_date.date()}")
        print(f"  초기자본 : {self.simulator.cash:,.0f}원")
        print(f"  마켓     : {', '.join(self.markets)}")
        print(f"  유니버스 : 마켓별 {self.universe_n}개 후보")
        print(f"  최대 보유 한도: {self.MAX_POSITIONS} 종목 제한 락")
        print("=" * 65)

        print("\n[1] 유니버스 데이터 준비")
        ticker_data = self._load_data()
        self._load_kosdaq_index()
        if not ticker_data:
            print("❌ 유효한 데이터 없음. 종료.")
            return

        all_dates: set[pd.Timestamp] = set()
        for df in ticker_data.values():
            all_dates.update(df.index[(df.index >= self.start_date) & (df.index <= self.end_date)])
        trading_dates = sorted(all_dates)

        deposit_on: dict[str, float] = {}
        for dep_date_str, amount in self.deposit_schedule.items():
            dep_ts = pd.Timestamp(dep_date_str)
            target = next((d for d in trading_dates if d >= dep_ts), None)
            if target:
                key = target.strftime("%Y-%m-%d")
                deposit_on[key] = deposit_on.get(key, 0) + amount

        print(f"\n[2] 백테스트 실행  ({len(trading_dates)}거래일 × 유니버스 {len(ticker_data)}종목)")

        prev_year = None
        for date in trading_dates:
            date_str = date.strftime("%Y-%m-%d")

            if date.year != prev_year:
                print(f"\n  ── {date.year}년 ──────────────────────────────────")
                prev_year = date.year

            current_prices: dict[str, float] = {
                ticker: float(df.loc[date, "Close"])
                for ticker, df in ticker_data.items()
                if date in df.index
            }

            if date_str in deposit_on:
                self.simulator.deposit(deposit_on[date_str], current_prices)

            # ── 매도 우선 ─────────────────────────────────────────────────
            for held_ticker in list(self.positions.keys()):
                df = ticker_data.get(held_ticker)
                if df is None or date not in df.index:
                    continue
                idx = df.index.get_loc(date)
                if idx < 1:
                    continue
                self._try_sell(date_str, held_ticker, df, idx, current_prices)

            # ── 일일 스크리닝 (골든크로스 + 거래대금 폭발 검증) ─────────────────
            candidates = self._screen_daily(date, ticker_data)

            # ── 매수 집행 ─────────────────────────────────────────────────
            for buy_ticker, (signal_tag, score) in candidates.items():
                if buy_ticker in self.positions:
                    continue

                # ★ 하루 루프 내에서 실시간으로 보유 종목 수가 3개 충족되면 루프 자체 탈출
                if len(self.positions) >= self.MAX_POSITIONS:
                    break
                if self.simulator.cash <= 0:
                    break

                df = ticker_data.get(buy_ticker)
                if df is None or date not in df.index:
                    continue
                idx = df.index.get_loc(date)
                self._try_buy(date_str, buy_ticker, df, idx, current_prices, signal_tag, score)

            total_asset = self.simulator.get_total_asset(current_prices)
            return_pct  = self.simulator.get_current_return(current_prices)
            self.equity_curve.append({
                "date":        date_str,
                "total_asset": round(total_asset),
                "return_pct":  round(return_pct, 4),
                "cash":        round(self.simulator.cash),
                "invested":    round(self.simulator.invested_capital),
            })

        print("\n[3] 결과 분석")
        self._report()

    def _report(self) -> None:
        if not self.equity_curve:
            print("기록된 데이터 없음")
            return

        ec   = self.equity_curve
        last = ec[-1]

        initial_investment = ec[0]["invested"]
        final_asset        = last["total_asset"]
        final_return_pct   = last["return_pct"]
        invested           = last["invested"]

        assets = [r["total_asset"] for r in ec]
        peak   = assets[0]
        mdd    = 0.0
        for a in assets:
            if a > peak:
                peak = a
            dd = (peak - a) / peak * 100
            mdd = max(mdd, dd)

        n_days  = (pd.Timestamp(ec[-1]["date"]) - pd.Timestamp(ec[0]["date"])).days
        n_years = n_days / 365.25
        cagr = ((final_asset / initial_investment) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0.0

        buys    = [t for t in self.trade_log if "BUY" in t["action"]]
        sell_tp = [t for t in self.trade_log if t["action"] == "SELL_TP"]
        sell_ts = [t for t in self.trade_log if t["action"] == "SELL_TS"]
        sell_sl = [t for t in self.trade_log if t["action"] == "SELL_SL"]
        all_sells = sell_tp + sell_ts + sell_sl

        def _pnl_pct(t: dict) -> float:
            ep = t.get("entry_price", 0)
            return (t["price"] - ep) / ep * 100 if ep > 0 else 0.0

        win_trades  = [t for t in all_sells if _pnl_pct(t) > 0]
        loss_trades = [t for t in all_sells if _pnl_pct(t) <= 0]
        win_rate    = len(win_trades) / len(all_sells) * 100 if all_sells else 0.0

        avg_profit = (sum(_pnl_pct(t) for t in win_trades)  / len(win_trades))  if win_trades  else 0.0
        avg_loss   = (sum(_pnl_pct(t) for t in loss_trades) / len(loss_trades)) if loss_trades else 0.0
        pl_ratio   = abs(avg_profit / avg_loss) if avg_loss != 0 else float("inf")

        print("\n" + "=" * 65)
        print("  백테스트 결과 요약")
        print("=" * 65)
        print(f"  기간          : {ec[0]['date']} ~ {ec[-1]['date']}  ({n_years:.1f}년)")
        print(f"  초기 투자     : {initial_investment:>15,.0f} 원")
        print(f"  추가 입금 합계: {invested - initial_investment:>15,.0f} 원")
        print(f"  총 투자 원금  : {invested:>15,.0f} 원")
        print(f"  최종 자산     : {final_asset:>15,.0f} 원")
        print(f"  현금 잔고     : {last['cash']:>15,.0f} 원")
        print(f"  보유 주식 가치: {final_asset - last['cash']:>15,.0f} 원")
        print("─" * 65)
        print(f"  펀드 수익률   : {final_return_pct:>+14.2f} %")
        print(f"  CAGR          : {cagr:>+14.2f} %")
        print(f"  최대 낙폭(MDD): {-mdd:>+14.2f} %")
        print("─" * 65)
        print(f"  총 매수 횟수  : {len(buys):>5}회")
        print(f"  익절 (TP)     : {len(sell_tp):>5}회")
        print(f"  트레일링 (TS) : {len(sell_ts):>5}회")
        print(f"  손절 (SL)     : {len(sell_sl):>5}회")
        print(f"  승률          : {win_rate:>14.1f} %")
        print("─" * 65)
        print(f"  평균 익절률   : {avg_profit:>+14.2f} %")
        print(f"  평균 손실률   : {avg_loss:>+14.2f} %")
        print(f"  손익비 (P/L)  : {pl_ratio:>14.2f}")
        print("=" * 65)
        self._print_equity_chart(ec)
        self._print_signal_report()
        self._save_results()

    def _print_equity_chart(self, ec: list[dict], width: int = 60, height: int = 15) -> None:
        returns = [r["return_pct"] for r in ec]
        min_r   = min(returns)
        max_r   = max(returns)
        span    = max_r - min_r if max_r != min_r else 1.0
        step   = max(1, len(ec) // width)
        sample = ec[::step]

        print("\n  수익률 추이 (펀드 기준가 기반)")
        print("  " + "─" * (len(sample) + 4))
        for row_idx in range(height, -1, -1):
            threshold = min_r + span * row_idx / height
            line = ""
            for pt in sample:
                line += "█" if pt["return_pct"] >= threshold else " "
            label = f"{threshold:>+6.1f}%"
            print(f"  {label} │{line}")
        print("  " + "─" * (len(sample) + 4))

    def _print_signal_report(self) -> None:
        from collections import defaultdict
        all_sells = [t for t in self.trade_log if t["action"] in ("SELL_TP", "SELL_TS", "SELL_SL")]
        if not all_sells:
            return

        def _pnl_pct(t: dict) -> float:
            ep = t.get("entry_price", 0)
            return (t["price"] - ep) / ep * 100 if ep > 0 else 0.0

        def _pnl_won(t: dict) -> float:
            ep = t.get("entry_price", 0)
            return (t["price"] - ep) * t.get("qty", 0) if ep > 0 else 0.0

        groups = defaultdict(list)
        for t in all_sells:
            groups[t.get("signal_tag") or "UNKNOWN"].append(t)

        print("\n" + "=" * 65)
        print("  로직별 기여도 및 성적 분석 리포트")
        print("=" * 65)
        for tag, trades in groups.items():
            wins = [t for t in trades if _pnl_pct(t) > 0]
            win_rate  = len(wins) / len(trades) * 100 if trades else 0.0
            total_pnl = sum(_pnl_won(t) for t in trades)
            avg_pct   = sum(_pnl_pct(t) for t in trades) / len(trades) if trades else 0.0
            print(f"  ▶ [{tag}]")
            print(f"     매매 횟수: {len(trades)}회 | 승률: {win_rate:.1f}% | 총손익: {total_pnl:+,.0f}원 | 평균수익률: {avg_pct:+.2f}%")
        print("=" * 65)

    def _save_results(self) -> None:
        RESULT_DIR.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.equity_curve:
            with open(RESULT_DIR / f"equity_{ts}.csv", "w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=self.equity_curve[0].keys())
                w.writeheader()
                w.writerows(self.equity_curve)


# ══════════════════════════════════════════════════════════════════════════════
# 종목 자동 선정 스크리너 (기존과 동일)
# ══════════════════════════════════════════════════════════════════════════════

class StockScreener:
    MARKET_CFG: dict[str, tuple[str, float]] = {
        "KOSPI":   ("get_top_kospi_stocks",  50_000),
        "KOSDAQ":  ("get_top_kosdaq_stocks", 30_000),
        "S&P500":  ("get_top_us_stocks",     200_000),
        "NASDAQ":  ("get_top_nasdaq_stocks", 100_000),
    }

    def __init__(self, universe_per_market: int = 200):
        self.universe_per_market = universe_per_market

    def build_universe(self, markets: list[str]) -> list[tuple[str, str]]:
        from stock_ai import (
            get_top_kospi_stocks, get_top_kosdaq_stocks,
            get_top_us_stocks,   get_top_nasdaq_stocks,
        )
        loaders = {
            "KOSPI":   get_top_kospi_stocks,
            "KOSDAQ":  get_top_kosdaq_stocks,
            "S&P500":  get_top_us_stocks,
            "NASDAQ":  get_top_nasdaq_stocks,
        }
        seen: set[str] = set()
        result: list[tuple[str, str]] = []
        for market in markets:
            fn = loaders.get(market)
            if fn is None:
                continue
            stocks: dict[str, str] = fn(self.universe_per_market)
            for name, ticker in stocks.items():
                if ticker not in seen:
                    seen.add(ticker)
                    result.append((ticker, name))
        return result

    def _score_chunk(
        self,
        chunk: list[tuple[str, str]],
        lookback: str,
        min_volume: float,
    ) -> list[dict]:
        tickers  = [t for t, _ in chunk]
        name_map = {t: n for t, n in chunk}
        scored: list[dict] = []

        try:
            raw = yf.download(
                tickers if len(tickers) > 1 else tickers[0],
                period=lookback,
                auto_adjust=True,
                progress=False,
            )
            if raw.empty:
                return scored

            for ticker in tickers:
                try:
                    if isinstance(raw.columns, pd.MultiIndex):
                        close  = raw["Close"][ticker].dropna()
                        volume = raw["Volume"][ticker].dropna()
                    else:
                        close  = raw["Close"].dropna()
                        volume = raw["Volume"].dropna()

                    if len(close) < 20:
                        continue

                    avg_vol = float(volume.mean())
                    if avg_vol < min_volume:
                        continue

                    sma20   = float(close.rolling(20).mean().iloc[-1])
                    current = float(close.iloc[-1])
                    if current <= sma20:
                        continue

                    momentum   = (current / float(close.iloc[0]) - 1) * 100
                    volatility = float(close.pct_change().std()) * 100 or 0.001
                    vol_score  = math.log10(max(avg_vol, 1))

                    score = (
                        momentum   * 0.5
                        + (1 / volatility) * 10 * 0.3
                        + vol_score        * 0.2
                    )
                    scored.append({
                        "ticker":     ticker,
                        "name":       name_map.get(ticker, ticker),
                        "score":      round(score, 4),
                        "momentum":   round(momentum, 2),
                        "volatility": round(volatility, 4),
                        "avg_volume": round(avg_vol),
                    })
                except Exception:
                    continue
        except Exception:
            pass

        return scored

    def screen(
        self,
        markets: list[str],
        top_n: int = 20,
        lookback: str = "3mo",
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> list[dict]:
        if progress_cb:
            progress_cb(0.0, "종목 유니버스 로딩 중...")

        universe = self.build_universe(markets)
        if not universe:
            return []

        min_vol_map = {t: self.MARKET_CFG.get(m, ("", 50_000))[1]
                       for m in markets
                       for t, _ in self.build_universe([m])}
        default_min_vol = min(v for _, v in self.MARKET_CFG.values())

        CHUNK = 50
        all_scored: list[dict] = []
        total = len(universe)

        if progress_cb:
            progress_cb(2.0, f"{total}개 후보 종목 배치 다운로드 시작...")

        for i in range(0, total, CHUNK):
            chunk = universe[i: i + CHUNK]
            chunk_min_vol = min(
                min_vol_map.get(t, default_min_vol) for t, _ in chunk
            )
            all_scored.extend(self._score_chunk(chunk, lookback, chunk_min_vol))

            if progress_cb:
                pct = 5.0 + (i + len(chunk)) / total * 90.0
                progress_cb(pct, f"스크리닝 {i + len(chunk)}/{total}개 완료...")

        all_scored.sort(key=lambda x: x["score"], reverse=True)
        selected = all_scored[:top_n]

        if progress_cb:
            progress_cb(100.0,
                        f"스크리닝 완료 — {len(all_scored)}개 통과 → 상위 {len(selected)}개 선정")
        return selected


if __name__ == "__main__":
    engine = BacktestEngine(
        initial_capital  = INITIAL_CAPITAL,
        start_date       = START_DATE,
        end_date         = END_DATE,
        markets          = SCREEN_MARKETS,
        universe_n       = SCREEN_UNIVERSE_N,
        volume_top_n     = VOLUME_TOP_N,
        news_candidate_n = NEWS_CANDIDATE_N,
        deposit_schedule = DEPOSIT_SCHEDULE,
    )
    engine.run()
