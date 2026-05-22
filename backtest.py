"""
backtest.py — 날짜 중심(Date-Driven) 백테스팅 엔진

매 거래일마다 전체 유니버스에서 실시간으로 종목을 선정한 뒤
매수·매도를 집행합니다.

실행:
  python backtest.py

설정:
  SCREEN_MARKETS     — 대상 마켓 (KOSPI / KOSDAQ / S&P500 / NASDAQ)
  SCREEN_UNIVERSE_N  — 마켓별 유니버스 후보 수
  VOLUME_TOP_N       — 1단계: 일별 거래대금 상위 N
  NEWS_CANDIDATE_N   — 3단계: 뉴스 분석 대상 상위 N (백테스트에서는 통과)
  START_DATE / END_DATE  — 백테스트 기간
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

SCREEN_MARKETS    = ["KOSPI", "KOSDAQ"]   # 대상 마켓
SCREEN_UNIVERSE_N = 200                   # 마켓별 후보 종목 수 (시가총액 상위)
SCREEN_TOP_N      = 20                    # StockScreener 단독 실행 시 최종 선정 수

VOLUME_TOP_N      = 100                   # 1단계 필터: 일별 거래대금 상위 N
NEWS_CANDIDATE_N  = 15                    # 3단계 필터: 뉴스 분석 대상 상위 N

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
    peak_price:  float = 0.0   # 진입 후 최고 종가 (트레일링 스톱 기준)

    def reset(self) -> None:
        self.in_position = False
        self.entry_price = 0.0
        self.rebuy_count = 0
        self.take_profit = -1.0
        self.stop_loss   = -1.0
        self.peak_price  = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 백테스팅 엔진 (날짜 중심)
# ══════════════════════════════════════════════════════════════════════════════

class BacktestEngine:
    WARMUP_DAYS = 300   # 지표 안정화에 필요한 워밍업 일수

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
        position_sizing_pct: float = 0.20,
    ):
        self.markets           = markets or ["KOSPI", "KOSDAQ"]
        self.universe_n        = universe_n
        self.volume_top_n      = volume_top_n
        self.news_candidate_n  = news_candidate_n
        self.start_date        = pd.Timestamp(start_date)
        self.end_date          = pd.Timestamp(end_date)
        self.deposit_schedule  = deposit_schedule or {}
        self.position_sizing_pct = position_sizing_pct

        self.simulator    = TradingSimulator(initial_capital)
        self.strategy     = TradingStrategy()
        self.positions:    dict[str, PositionState] = {}   # 동적 관리
        self.trade_log:    list[dict] = []
        self.equity_curve: list[dict] = []

    # ── 전체 유니버스 데이터 로딩 ────────────────────────────────────────────

    def _load_data(self) -> dict[str, pd.DataFrame]:
        load_start = self.start_date - pd.Timedelta(days=self.WARMUP_DAYS)
        load_end   = self.end_date   + pd.Timedelta(days=2)

        screener = StockScreener(universe_per_market=self.universe_n)
        universe = screener.build_universe(self.markets)
        all_tickers = [t for t, _ in universe]
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

    # ── 일일 종목 스크리닝 ────────────────────────────────────────────────────

    def _screen_daily(
        self,
        date: pd.Timestamp,
        ticker_data: dict[str, pd.DataFrame],
    ) -> list[str]:
        """
        날짜 기준 3단계 매수 후보 선정.

        1단계: 거래대금(종가×거래량) 상위 volume_top_n
        2단계: SMA 골든크로스 + 기울기 신호 (미래 데이터 참조 없음)
        3단계: 뉴스 감성 분석 — 백테스트에서는 news_candidate_n개 그대로 통과
               (실시간 운용 시 뉴스 API 연동 예정)
        """
        # 1단계: 거래대금 상위 N
        turnover_list: list[tuple[str, float]] = []
        for ticker, df in ticker_data.items():
            if date not in df.index:
                continue
            row    = df.loc[date]
            close  = float(row.get("Close",  0) or 0)
            volume = float(row.get("Volume", 0) or 0)
            turnover_list.append((ticker, close * volume))

        turnover_list.sort(key=lambda x: x[1], reverse=True)
        top_volume = [t for t, _ in turnover_list[: self.volume_top_n]]

        # 2단계: 차트·모멘텀 신호
        candidates: list[str] = []
        for ticker in top_volume:
            df = ticker_data[ticker]
            if date not in df.index:
                continue
            idx = df.index.get_loc(date)
            if idx < self.strategy.kill_window + 60:
                continue

            row       = df.iloc[idx]
            prev      = df.iloc[idx - 1]
            sma5      = float(row.get("SMA_5",  np.nan))
            sma20     = float(row.get("SMA_20", np.nan))
            prev_sma5 = float(prev.get("SMA_5", np.nan))

            if any(np.isnan(v) for v in [sma5, sma20, prev_sma5]):
                continue
            if self.strategy.check_kill_switch(df["Close"].iloc[: idx + 1]):
                continue
            if self.strategy.is_entry_signal(sma5, sma20, prev_sma5):
                candidates.append(ticker)

        # 3단계: 뉴스 감성 (백테스트에서는 상위 N개 통과)
        return candidates[: self.news_candidate_n]

    # ── 매도 판단 (보유 포지션 점검 + 분할매수) ──────────────────────────────

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

        # 최고가 갱신 (트레일링 스톱 기준)
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
                )
                del self.positions[ticker]
            return

        # 트레일링 스톱 + 하드 손절 (effective = 더 높은 쪽 적용)
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
                                entry_price=pos.entry_price)
                del self.positions[ticker]
            return

        # 분할매수 (물타기) — 보유 중 하락 시
        if self.strategy.is_rebuy_signal(price, pos.entry_price, pos.rebuy_count):
            total_asset = self.simulator.get_total_asset(current_prices)
            alloc_cash  = total_asset * self.position_sizing_pct
            qty = max(1, int(alloc_cash / price))
            if self.simulator.execute_buy(ticker, price, qty):
                pos.rebuy_count += 1
                self._log_trade(
                    date_str, ticker, f"REBUY#{pos.rebuy_count}",
                    price, qty,
                    f"{pos.rebuy_count}차 분할매수 [배정 {alloc_cash:,.0f}원]",
                )

    # ── 매수 집행 (신규 진입) ─────────────────────────────────────────────────

    def _try_buy(
        self,
        date_str: str,
        ticker: str,
        df: pd.DataFrame,
        idx: int,
        current_prices: dict[str, float],
    ) -> None:
        row   = df.iloc[idx]
        price = float(row["Close"])
        atr   = float(row.get("ATR", np.nan))
        if np.isnan(atr) or atr <= 0:
            atr = -1.0

        total_asset = self.simulator.get_total_asset(current_prices)
        alloc_cash  = total_asset * self.position_sizing_pct
        qty = max(1, int(alloc_cash / price))

        if not self.simulator.execute_buy(ticker, price, qty):
            return

        tp = self.strategy.get_exit_price(price, atr, is_profit=True)  if atr > 0 else -1.0
        sl = self.strategy.get_exit_price(price, atr, is_profit=False) if atr > 0 else -1.0

        pos = PositionState()
        pos.in_position = True
        pos.entry_price = price
        pos.peak_price  = price
        pos.rebuy_count = 0
        pos.take_profit = tp
        pos.stop_loss   = sl
        self.positions[ticker] = pos

        self._log_trade(
            date_str, ticker, "BUY", price, qty,
            f"일일스크리닝 선정 [총자산 {total_asset:,.0f} → 배정 {alloc_cash:,.0f}원]",
        )

    def _log_trade(self, date: str, ticker: str, action: str,
                   price: float, qty: int, reason: str,
                   entry_price: float = 0.0) -> None:
        self.trade_log.append({
            "date":        date,
            "ticker":      ticker,
            "action":      action,
            "price":       price,
            "qty":         qty,
            "amount":      price * qty,
            "reason":      reason,
            "entry_price": entry_price,
        })
        icon = {"BUY": "🛒", "SELL_TP": "💵✅", "SELL_SL": "💵❌", "SELL_TS": "🎯✅"}.get(action, "↩️")
        print(f"  {icon} {date} [{ticker}] {action:10s}  "
              f"가격 {price:>10,.0f}  수량 {qty:>5}  ({reason})")

    # ── 메인 루프 (날짜 중심) ─────────────────────────────────────────────────

    def run(self) -> None:
        print("\n" + "=" * 65)
        print("  백테스팅 엔진 시작 (날짜 중심 / 일일 종목 선정)")
        print("=" * 65)
        print(f"  기간     : {self.start_date.date()} ~ {self.end_date.date()}")
        print(f"  초기자본 : {self.simulator.cash:,.0f}원")
        print(f"  마켓     : {', '.join(self.markets)}")
        print(f"  유니버스 : 마켓별 {self.universe_n}개 후보")
        print(f"  거래대금 필터 : 일별 상위 {self.volume_top_n}개")
        print(f"  뉴스 분석 대상: 상위 {self.news_candidate_n}개 (백테스트 통과)")
        print(f"  포지션   : 총자산의 {self.position_sizing_pct:.0%}씩 동적 배분")
        print(f"  익절 ATR : ×{self.strategy.tp_multiplier:.1f}  "
              f"손절 ATR : ×{self.strategy.sl_multiplier:.1f}  "
              f"트레일링 : ×{self.strategy.trailing_multiplier:.1f}")
        print("=" * 65)

        print("\n[1] 유니버스 데이터 준비")
        ticker_data = self._load_data()
        if not ticker_data:
            print("❌ 유효한 데이터 없음. 종료.")
            return

        # 거래일 목록 (전체 유니버스 합집합)
        all_dates: set[pd.Timestamp] = set()
        for df in ticker_data.values():
            all_dates.update(df.index[(df.index >= self.start_date) & (df.index <= self.end_date)])
        trading_dates = sorted(all_dates)

        # 입금 일정 이월 처리 (공휴일 → 다음 거래일)
        deposit_on: dict[str, float] = {}
        for dep_date_str, amount in self.deposit_schedule.items():
            dep_ts = pd.Timestamp(dep_date_str)
            target = next((d for d in trading_dates if d >= dep_ts), None)
            if target:
                key = target.strftime("%Y-%m-%d")
                deposit_on[key] = deposit_on.get(key, 0) + amount
                if key != dep_date_str:
                    print(f"  ℹ️  입금일 이월: {dep_date_str} → {key}  ({amount:,.0f}원)")

        print(f"\n[2] 백테스트 실행  ({len(trading_dates)}거래일 × 유니버스 {len(ticker_data)}종목)")

        prev_year = None
        for date in trading_dates:
            date_str = date.strftime("%Y-%m-%d")

            if date.year != prev_year:
                print(f"\n  ── {date.year}년 ──────────────────────────────────")
                prev_year = date.year

            # 오늘 전 종목 시장가
            current_prices: dict[str, float] = {
                ticker: float(df.loc[date, "Close"])
                for ticker, df in ticker_data.items()
                if date in df.index
            }

            # 추가 입금 처리
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

            # ── 일일 스크리닝 → 매수 후보 ────────────────────────────────
            candidates = self._screen_daily(date, ticker_data)

            # ── 매수 집행 ─────────────────────────────────────────────────
            for buy_ticker in candidates:
                if buy_ticker in self.positions:
                    continue  # 이미 보유 중
                if self.simulator.cash <= 0:
                    break
                df = ticker_data.get(buy_ticker)
                if df is None or date not in df.index:
                    continue
                idx = df.index.get_loc(date)
                self._try_buy(date_str, buy_ticker, df, idx, current_prices)

            # 수익률 기록
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
        self._report(ticker_data)

    # ── 결과 리포트 ──────────────────────────────────────────────────────────

    def _report(self, ticker_data: dict[str, pd.DataFrame]) -> None:
        if not self.equity_curve:
            print("기록된 데이터 없음")
            return

        ec   = self.equity_curve
        last = ec[-1]

        initial_investment = ec[0]["invested"]
        final_asset        = last["total_asset"]
        final_return_pct   = last["return_pct"]
        invested           = last["invested"]

        # MDD
        assets = [r["total_asset"] for r in ec]
        peak   = assets[0]
        mdd    = 0.0
        for a in assets:
            if a > peak:
                peak = a
            dd = (peak - a) / peak * 100
            mdd = max(mdd, dd)

        # CAGR
        n_days  = (pd.Timestamp(ec[-1]["date"]) - pd.Timestamp(ec[0]["date"])).days
        n_years = n_days / 365.25
        cagr = ((final_asset / initial_investment) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0.0

        # 매매 통계
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
        print(f"  익절 (TP)     : {len(sell_tp):>5}회  목표가 직접 도달")
        print(f"  트레일링 (TS) : {len(sell_ts):>5}회  트레일링 스톱 청산")
        print(f"  손절 (SL)     : {len(sell_sl):>5}회  하드 손절")
        print(f"  승률          : {win_rate:>14.1f} %  (실제 P/L 기준)")
        print("─" * 65)
        print(f"  평균 익절률   : {avg_profit:>+14.2f} %")
        print(f"  평균 손실률   : {avg_loss:>+14.2f} %")
        if pl_ratio == float("inf"):
            print(f"  손익비 (P/L)  : {'∞':>15s}  (손절 거래 없음)")
        else:
            marker = "✅" if pl_ratio >= 1.5 else ("⚠️" if pl_ratio >= 1.0 else "❌")
            print(f"  손익비 (P/L)  : {pl_ratio:>14.2f}  {marker} (>1.5 목표)")

        # 현재 보유 포지션
        current_prices: dict[str, float] = {}
        for ticker, df in ticker_data.items():
            latest = df["Close"].dropna()
            if not latest.empty:
                current_prices[ticker] = float(latest.iloc[-1])

        if self.simulator.portfolio:
            print("\n  보유 포지션:")
            for ticker, qty in self.simulator.portfolio.items():
                price = current_prices.get(ticker, 0)
                pos   = self.positions.get(ticker)
                entry = pos.entry_price if pos else 0
                value = price * qty
                pnl   = (price - entry) / entry * 100 if entry else 0
                print(f"    {ticker:15s} {qty:>6}주  "
                      f"평단 {entry:>10,.0f}  "
                      f"현재 {price:>10,.0f}  "
                      f"평가 {value:>12,.0f}  "
                      f"수익률 {pnl:>+6.1f}%")

        if self.deposit_schedule:
            print("\n  추가 입금 내역:")
            for d, amt in sorted(self.deposit_schedule.items()):
                print(f"    {d}  +{amt:,.0f}원")

        print("=" * 65)
        self._print_equity_chart(ec)
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
# 종목 자동 선정 스크리너 (단독 실행 또는 유니버스 빌드용)
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

    MARKET_CFG: dict[str, tuple[str, float]] = {
        "KOSPI":   ("get_top_kospi_stocks",  50_000),
        "KOSDAQ":  ("get_top_kosdaq_stocks", 30_000),
        "S&P500":  ("get_top_us_stocks",     200_000),
        "NASDAQ":  ("get_top_nasdaq_stocks", 100_000),
    }

    def __init__(self, universe_per_market: int = 200):
        self.universe_per_market = universe_per_market

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
        """
        자동 종목 선정 실행 (단독 스크리닝 용도).

        Parameters
        ----------
        markets     : ["KOSPI", "KOSDAQ", "S&P500", "NASDAQ"] 중 선택
        top_n       : 최종 선정 종목 수
        lookback    : 스크리닝 기간 (yfinance period 문자열)
        progress_cb : (pct: float, msg: str) → None  (UI 진행률 콜백)
        """
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


# ══════════════════════════════════════════════════════════════════════════════
# 실행 진입점
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    engine = BacktestEngine(
        initial_capital     = INITIAL_CAPITAL,
        start_date          = START_DATE,
        end_date            = END_DATE,
        markets             = SCREEN_MARKETS,
        universe_n          = SCREEN_UNIVERSE_N,
        volume_top_n        = VOLUME_TOP_N,
        news_candidate_n    = NEWS_CANDIDATE_N,
        deposit_schedule    = DEPOSIT_SCHEDULE,
        position_sizing_pct = POSITION_SIZING_PCT,
    )
    engine.run()
