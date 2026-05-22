"""
backtest.py — 과거 데이터 기반 백테스팅 엔진

TradingSimulator(펀드 회계) + TradingStrategy(매매 로직)를 결합하여
과거 일봉 데이터로 수익률 검증을 수행합니다.

실행:
  python backtest.py

설정:
  TICKERS            — 백테스트 종목 목록 (yfinance 티커)
  START_DATE         — 백테스트 시작일 (YYYY-MM-DD)
  END_DATE           — 백테스트 종료일 (YYYY-MM-DD)
  INITIAL_CAPITAL    — 초기 투자 원금 (원)
  DEPOSIT_SCHEDULE   — 추가 입금 일정 {"YYYY-MM-DD": 금액, ...}
  POSITION_SIZING_PCT— 총자산 대비 1회 매수 비중 (0.0 ~ 1.0)
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
# 설정 (CLI 직접 실행 시 여기를 수정하세요)
# ══════════════════════════════════════════════════════════════════════════════

# 종목은 StockScreener가 자동 선정합니다 — 아래 마켓/스크리닝 파라미터만 조정하세요
SCREEN_MARKETS    = ["KOSPI", "KOSDAQ"]   # 대상 마켓
SCREEN_UNIVERSE_N = 200                   # 마켓별 후보 종목 수 (시가총액 상위)
SCREEN_TOP_N      = 20                    # 최종 선정 종목 수

START_DATE = "2020-01-01"
END_DATE   = "2024-12-31"

INITIAL_CAPITAL    = 10_000_000   # 1,000만 원
POSITION_SIZING_PCT = 0.20        # 총자산의 20%씩 매수

# 인위적 추가 입금 스케줄 (날짜 → 금액)
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
    in_position: bool  = False
    entry_price: float = 0.0
    rebuy_count: int   = 0
    take_profit: float = -1.0
    stop_loss:   float = -1.0

    def reset(self) -> None:
        self.in_position = False
        self.entry_price = 0.0
        self.rebuy_count = 0
        self.take_profit = -1.0
        self.stop_loss   = -1.0


# ══════════════════════════════════════════════════════════════════════════════
# 백테스팅 엔진
# ══════════════════════════════════════════════════════════════════════════════

class BacktestEngine:
    WARMUP_DAYS = 300   # 지표 안정화에 필요한 워밍업 일수

    def __init__(
        self,
        tickers: list[str],
        initial_capital: float,
        start_date: str,
        end_date: str,
        deposit_schedule: dict[str, float] | None = None,
        position_sizing_pct: float = 0.20,
    ):
        self.tickers             = tickers
        self.start_date          = pd.Timestamp(start_date)
        self.end_date            = pd.Timestamp(end_date)
        self.deposit_schedule    = deposit_schedule or {}
        self.position_sizing_pct = position_sizing_pct

        self.simulator  = TradingSimulator(initial_capital)
        self.strategy   = TradingStrategy()
        self.positions  = {t: PositionState() for t in tickers}
        self.trade_log:  list[dict] = []
        self.equity_curve: list[dict] = []

    # ── 데이터 다운로드 & 지표 계산 ──────────────────────────────────────────

    def _load_data(self) -> dict[str, pd.DataFrame]:
        load_start = self.start_date - pd.Timedelta(days=self.WARMUP_DAYS)
        load_end   = self.end_date   + pd.Timedelta(days=2)

        ticker_data: dict[str, pd.DataFrame] = {}
        for ticker in self.tickers:
            print(f"  [{ticker}] 데이터 다운로드 중...", end=" ", flush=True)
            raw = yf.download(
                ticker,
                start=load_start.strftime("%Y-%m-%d"),
                end=load_end.strftime("%Y-%m-%d"),
                auto_adjust=True,
                progress=False,
            )
            if raw.empty or len(raw) < 60:
                print(f"데이터 부족 — 건너뜀")
                continue
            df = _flatten_columns(raw)
            df = _add_indicators(df)
            ticker_data[ticker] = df
            print(f"{len(df)}개 봉 로드 완료")
        return ticker_data

    # ── 매매 결정 로직 ────────────────────────────────────────────────────────

    def _decide(
        self,
        ticker: str,
        df: pd.DataFrame,
        idx: int,
        current_prices: dict[str, float],
    ) -> None:
        row  = df.iloc[idx]
        prev = df.iloc[idx - 1]
        price = float(row["Close"])
        pos   = self.positions[ticker]
        date_str = df.index[idx].strftime("%Y-%m-%d")

        # 지표 유효성 확인
        sma5      = float(row.get("SMA_5",  np.nan))
        sma20     = float(row.get("SMA_20", np.nan))
        prev_sma5 = float(prev.get("SMA_5", np.nan))
        atr       = float(row.get("ATR",    np.nan))

        if any(np.isnan(v) for v in [sma5, sma20, prev_sma5]):
            return
        if np.isnan(atr) or atr <= 0:
            atr = -1.0

        # Kill-Switch: 최근 20봉 변동성 체크
        kill = self.strategy.check_kill_switch(df["Close"].iloc[: idx + 1])

        if not pos.in_position:
            # ── 신규 진입 신호 ─────────────────────────────────────────────
            if kill:
                return
            if not self.strategy.is_entry_signal(sma5, sma20, prev_sma5):
                return

            total_asset = self.simulator.get_total_asset(current_prices)
            alloc_cash  = total_asset * self.position_sizing_pct
            qty = max(1, int(alloc_cash / price))

            if not self.simulator.execute_buy(ticker, price, qty):
                return

            tp = self.strategy.get_exit_price(price, atr, is_profit=True)  if atr > 0 else -1.0
            sl = self.strategy.get_exit_price(price, atr, is_profit=False) if atr > 0 else -1.0

            pos.in_position = True
            pos.entry_price = price
            pos.rebuy_count = 0
            pos.take_profit = tp
            pos.stop_loss   = sl

            self._log_trade(date_str, ticker, "BUY", price, qty, "SMA 골든크로스+기울기 돌파")

        else:
            # ── 익절 ──────────────────────────────────────────────────────
            if pos.take_profit > 0 and price >= pos.take_profit:
                qty = self.simulator.portfolio.get(ticker, 0)
                if qty and self.simulator.execute_sell(ticker, price, qty):
                    self._log_trade(date_str, ticker, "SELL_TP", price, qty,
                                    f"익절 목표가 도달 ({pos.take_profit:,.0f})")
                    pos.reset()
                return

            # ── 손절 ──────────────────────────────────────────────────────
            if pos.stop_loss > 0 and price <= pos.stop_loss:
                qty = self.simulator.portfolio.get(ticker, 0)
                if qty and self.simulator.execute_sell(ticker, price, qty):
                    self._log_trade(date_str, ticker, "SELL_SL", price, qty,
                                    f"손절가 도달 ({pos.stop_loss:,.0f})")
                    pos.reset()
                return

            # ── 분할매수 (물타기) ──────────────────────────────────────────
            if self.strategy.is_rebuy_signal(price, pos.entry_price, pos.rebuy_count):
                total_asset = self.simulator.get_total_asset(current_prices)
                alloc_cash  = total_asset * self.position_sizing_pct
                qty = max(1, int(alloc_cash / price))
                if self.simulator.execute_buy(ticker, price, qty):
                    pos.rebuy_count += 1
                    self._log_trade(date_str, ticker, f"REBUY#{pos.rebuy_count}",
                                    price, qty, f"{pos.rebuy_count}차 분할매수")

    def _log_trade(self, date: str, ticker: str, action: str,
                   price: float, qty: int, reason: str) -> None:
        self.trade_log.append({
            "date":   date,
            "ticker": ticker,
            "action": action,
            "price":  price,
            "qty":    qty,
            "amount": price * qty,
            "reason": reason,
        })
        icon = {"BUY": "🛒", "SELL_TP": "💵✅", "SELL_SL": "💵❌"}.get(action, "↩️")
        print(f"  {icon} {date} [{ticker}] {action:10s}  "
              f"가격 {price:>10,.0f}  수량 {qty:>5}  ({reason})")

    # ── 메인 루프 ────────────────────────────────────────────────────────────

    def run(self) -> None:
        print("\n" + "=" * 65)
        print("  백테스팅 엔진 시작")
        print("=" * 65)
        print(f"  기간     : {self.start_date.date()} ~ {self.end_date.date()}")
        print(f"  초기자본 : {self.simulator.cash:,.0f}원")
        print(f"  종목     : {', '.join(self.tickers)}")
        print(f"  포지션   : 총자산의 {self.position_sizing_pct:.0%}씩 매수")
        print("=" * 65)

        print("\n[1] 데이터 준비")
        ticker_data = self._load_data()
        if not ticker_data:
            print("❌ 유효한 데이터 없음. 종료.")
            return

        # 백테스트 구간에 해당하는 거래일 목록 (전체 종목 합집합)
        all_dates: set[pd.Timestamp] = set()
        for df in ticker_data.values():
            all_dates.update(df.index[(df.index >= self.start_date) & (df.index <= self.end_date)])
        trading_dates = sorted(all_dates)

        # 입금 일정을 실제 거래일로 매핑 (공휴일 → 다음 거래일 이월)
        deposit_on: dict[str, float] = {}
        for dep_date_str, amount in self.deposit_schedule.items():
            dep_ts = pd.Timestamp(dep_date_str)
            target = next((d for d in trading_dates if d >= dep_ts), None)
            if target:
                key = target.strftime("%Y-%m-%d")
                deposit_on[key] = deposit_on.get(key, 0) + amount
                if key != dep_date_str:
                    print(f"  ℹ️  입금일 이월: {dep_date_str} → {key}  "
                          f"({amount:,.0f}원, 공휴일/비거래일)")

        print(f"\n[2] 백테스트 실행  ({len(trading_dates)}거래일)")

        prev_year = None
        for date in trading_dates:
            date_str = date.strftime("%Y-%m-%d")

            # 연도가 바뀌면 구분선 출력
            if date.year != prev_year:
                print(f"\n  ── {date.year}년 ──────────────────────────────────")
                prev_year = date.year

            # 현재 시장가 수집
            current_prices: dict[str, float] = {}
            for ticker, df in ticker_data.items():
                if date in df.index:
                    current_prices[ticker] = float(df.loc[date, "Close"])

            # 추가 입금 처리 (공휴일 이월 적용)
            if date_str in deposit_on:
                self.simulator.deposit(deposit_on[date_str], current_prices)

            # 종목별 매매 판단
            for ticker, df in ticker_data.items():
                if date not in df.index:
                    continue
                idx = df.index.get_loc(date)
                if idx < self.strategy.kill_window + 60:
                    continue
                self._decide(ticker, df, idx, current_prices)

            # 수익률 기록
            total_asset   = self.simulator.get_total_asset(current_prices)
            return_pct    = self.simulator.get_current_return(current_prices)
            invested      = self.simulator.invested_capital
            self.equity_curve.append({
                "date":         date_str,
                "total_asset":  round(total_asset),
                "return_pct":   round(return_pct, 4),
                "cash":         round(self.simulator.cash),
                "invested":     round(invested),
            })

        print("\n[3] 결과 분석")
        self._report(ticker_data)

    # ── 결과 리포트 ──────────────────────────────────────────────────────────

    def _report(self, ticker_data: dict[str, pd.DataFrame]) -> None:
        if not self.equity_curve:
            print("기록된 데이터 없음")
            return

        ec   = self.equity_curve
        last = ec[-1]

        # ── 기본 성과 지표 ─────────────────────────────────────────────────
        initial_investment = ec[0]["invested"]
        final_asset        = last["total_asset"]
        final_return_pct   = last["return_pct"]
        invested           = last["invested"]

        # MDD 계산 (최대 낙폭)
        assets = [r["total_asset"] for r in ec]
        peak   = assets[0]
        mdd    = 0.0
        for a in assets:
            if a > peak:
                peak = a
            dd = (peak - a) / peak * 100
            mdd = max(mdd, dd)

        # CAGR 계산
        n_days = (pd.Timestamp(ec[-1]["date"]) - pd.Timestamp(ec[0]["date"])).days
        n_years = n_days / 365.25
        cagr = ((final_asset / initial_investment) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0.0

        # 매매 통계
        buys     = [t for t in self.trade_log if "BUY" in t["action"]]
        sell_tp  = [t for t in self.trade_log if t["action"] == "SELL_TP"]
        sell_sl  = [t for t in self.trade_log if t["action"] == "SELL_SL"]
        win_rate = len(sell_tp) / (len(sell_tp) + len(sell_sl)) * 100 if (sell_tp or sell_sl) else 0.0

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
        print(f"  익절 횟수     : {len(sell_tp):>5}회")
        print(f"  손절 횟수     : {len(sell_sl):>5}회")
        print(f"  승률          : {win_rate:>14.1f} %")

        # 현재 보유 포지션
        current_prices = {}
        for ticker, df in ticker_data.items():
            latest = df["Close"].dropna()
            if not latest.empty:
                current_prices[ticker] = float(latest.iloc[-1])

        if self.simulator.portfolio:
            print("\n  보유 포지션:")
            for ticker, qty in self.simulator.portfolio.items():
                price = current_prices.get(ticker, 0)
                pos   = self.positions[ticker]
                value = price * qty
                pnl   = (price - pos.entry_price) / pos.entry_price * 100 if pos.entry_price else 0
                print(f"    {ticker:15s} {qty:>6}주  "
                      f"평단 {pos.entry_price:>10,.0f}  "
                      f"현재 {price:>10,.0f}  "
                      f"평가 {value:>12,.0f}  "
                      f"수익률 {pnl:>+6.1f}%")

        # 입금 이벤트 요약
        if self.deposit_schedule:
            print("\n  추가 입금 내역:")
            for d, amt in sorted(self.deposit_schedule.items()):
                print(f"    {d}  +{amt:,.0f}원")

        print("=" * 65)

        # ── ASCII 에쿼티 커브 ──────────────────────────────────────────────
        self._print_equity_chart(ec)

        # ── 결과 파일 저장 ─────────────────────────────────────────────────
        self._save_results()

    def _print_equity_chart(self, ec: list[dict], width: int = 60, height: int = 15) -> None:
        returns = [r["return_pct"] for r in ec]
        min_r   = min(returns)
        max_r   = max(returns)
        span    = max_r - min_r if max_r != min_r else 1.0

        # 월 단위로 샘플링 (최대 width 포인트)
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
        first_date = sample[0]["date"][:7]
        last_date  = sample[-1]["date"][:7]
        padding    = " " * (len(sample) // 2 - len(first_date))
        print(f"         {first_date}{padding}{last_date}")

    def _save_results(self) -> None:
        RESULT_DIR.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        ec_path    = RESULT_DIR / f"equity_{ts}.csv"
        trade_path = RESULT_DIR / f"trades_{ts}.csv"

        if self.equity_curve:
            keys = self.equity_curve[0].keys()
            with open(ec_path, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(self.equity_curve)
            print(f"\n  📄 에쿼티 커브 저장: {ec_path.name}")

        if self.trade_log:
            keys = self.trade_log[0].keys()
            with open(trade_path, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(self.trade_log)
            print(f"  📄 매매 내역 저장: {trade_path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 종목 자동 선정 스크리너
# ══════════════════════════════════════════════════════════════════════════════

class StockScreener:
    """
    전체 종목 유니버스에서 백테스트 대상을 자동 선정.

    선정 기준 (복합 점수):
      - 3개월 모멘텀  (50%) — 단기 상승 추세 강도
      - 변동성 역수    (30%) — 안정적 종목 선호
      - 거래량 점수   (20%) — 유동성

    필터:
      - 가격 > SMA20 (상승 추세 종목만)
      - 일평균 거래량 > min_volume
    """

    # 지원 마켓 → (로더 함수명, 최소 거래량)
    MARKET_CFG: dict[str, tuple[str, float]] = {
        "KOSPI":   ("get_top_kospi_stocks",  50_000),
        "KOSDAQ":  ("get_top_kosdaq_stocks", 30_000),
        "S&P500":  ("get_top_us_stocks",     200_000),
        "NASDAQ":  ("get_top_nasdaq_stocks", 100_000),
    }

    def __init__(self, universe_per_market: int = 200):
        self.universe_per_market = universe_per_market

    # ── 유니버스 로드 ────────────────────────────────────────────────────────

    def build_universe(self, markets: list[str]) -> list[tuple[str, str]]:
        """마켓별 후보 종목 목록 반환. [(ticker, name), ...]"""
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
            stocks: dict[str, str] = fn(self.universe_per_market)  # {name: ticker}
            for name, ticker in stocks.items():
                if ticker not in seen:
                    seen.add(ticker)
                    result.append((ticker, name))
        return result

    # ── 배치 스코어링 ────────────────────────────────────────────────────────

    def _score_chunk(
        self,
        chunk: list[tuple[str, str]],
        lookback: str,
        min_volume: float,
    ) -> list[dict]:
        """청크 단위 배치 다운로드 후 스코어 계산."""
        tickers = [t for t, _ in chunk]
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
                    # MultiIndex: (field, ticker) 또는 단일 종목 flat
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
                        continue  # 상승 추세 아님

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

    # ── 퍼블릭 API ───────────────────────────────────────────────────────────

    def screen(
        self,
        markets: list[str],
        top_n: int = 20,
        lookback: str = "3mo",
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> list[dict]:
        """
        자동 종목 선정 실행.

        Parameters
        ----------
        markets     : ["KOSPI", "KOSDAQ", "S&P500", "NASDAQ"] 중 선택
        top_n       : 최종 선정 종목 수
        lookback    : 스크리닝 기간 (yfinance period 문자열)
        progress_cb : (pct: float, msg: str) → None  (UI 진행률 콜백)

        Returns
        -------
        [{"ticker", "name", "score", "momentum", "volatility", "avg_volume"}, ...]
        sorted by score descending, length <= top_n
        """
        if progress_cb:
            progress_cb(0.0, "종목 유니버스 로딩 중...")

        universe = self.build_universe(markets)
        if not universe:
            return []

        # 마켓별 최소 거래량 매핑 (혼합 마켓 대응)
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


# ══════════════════════════════════════════════════════════════════════════════
# 실행 진입점
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ── 자동 종목 선정 ─────────────────────────────────────────────────────
    print("\n[0] 종목 자동 선정 (스크리닝)")
    screener = StockScreener(universe_per_market=SCREEN_UNIVERSE_N)

    def _cli_progress(pct: float, msg: str) -> None:
        print(f"  [{pct:5.1f}%] {msg}")

    selected = screener.screen(
        markets     = SCREEN_MARKETS,
        top_n       = SCREEN_TOP_N,
        progress_cb = _cli_progress,
    )

    if not selected:
        print("❌ 스크리닝 통과 종목 없음. 종료.")
        sys.exit(1)

    print(f"\n  선정된 {len(selected)}개 종목:")
    for rank, s in enumerate(selected, 1):
        print(f"  {rank:2d}. [{s['ticker']:15s}] {s['name']:20s}  "
              f"모멘텀 {s['momentum']:>+6.1f}%  변동성 {s['volatility']:.2f}%")

    tickers = [s["ticker"] for s in selected]

    # ── 백테스트 실행 ──────────────────────────────────────────────────────
    engine = BacktestEngine(
        tickers             = tickers,
        initial_capital     = INITIAL_CAPITAL,
        start_date          = START_DATE,
        end_date            = END_DATE,
        deposit_schedule    = DEPOSIT_SCHEDULE,
        position_sizing_pct = POSITION_SIZING_PCT,
    )
    engine.run()
