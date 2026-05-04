"""
strategy.py - 이벤트 기반 트레이딩 전략 엔진

설계 원칙:
  - 모든 메서드는 Pure Function — 외부 상태 변경 없이 결과값만 반환
  - 포지션 상태(entry_price, current_n 등)는 호출자(Redis)가 관리
  - numpy array / pandas Series / list 모두 허용
  - 입력 데이터 부족 시 False 또는 -1.0 반환 (예외 미전파)
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from typing import Union

logger = logging.getLogger(__name__)

ArrayLike = Union[np.ndarray, pd.Series, list]


class TradingStrategy:
    """
    추세 추종 + 분할매수 + ATR 청산 + 변동성 Kill-Switch 전략 엔진.

    Parameters
    ----------
    threshold_slope : float
        진입 신호 단기 SMA 기울기 임계값 (기본 0.001 = 0.1%/봉)
    step_percent : float
        분할매수 트리거 하락률 (기본 0.03 = 3%)
    max_step : int
        최대 분할매수 횟수 (기본 3)
    atr_multiplier : float
        ATR 기반 익절·손절 배수 (기본 1.5)
    atr_period : int
        ATR 계산 기간 (기본 14)
    kill_window : int
        변동성 Kill-Switch 계산 윈도우 (기본 20)
    """

    def __init__(
        self,
        threshold_slope: float = 0.001,
        step_percent: float = 0.03,
        max_step: int = 3,
        atr_multiplier: float = 1.5,
        atr_period: int = 14,
        kill_window: int = 20,
    ):
        self.threshold_slope = threshold_slope
        self.step_percent = step_percent
        self.max_step = max_step
        self.atr_multiplier = atr_multiplier
        self.atr_period = atr_period
        self.kill_window = kill_window

    # ── 1. 진입 로직: 추세 추종 (Momentum Entry) ─────────────────────────────

    def is_entry_signal(
        self,
        sma_short: float,
        sma_long: float,
        prev_sma_short: float,
        threshold_slope: float | None = None,
    ) -> bool:
        """
        추세 추종 진입 신호 판단.

        조건:
          (1) 단기 SMA > 장기 SMA  → 골든크로스 영역
          (2) 단기 SMA 기울기 > threshold_slope  → 상승 모멘텀 확인

        기울기 = (sma_short - prev_sma_short) / prev_sma_short

        Parameters
        ----------
        sma_short : float
            현재 단기 SMA
        sma_long : float
            현재 장기 SMA
        prev_sma_short : float
            직전 단기 SMA (기울기 계산용)
        threshold_slope : float, optional
            기울기 임계값. None이면 인스턴스 기본값 사용.

        Returns
        -------
        bool
            True = 진입 신호, False = 미진입 또는 입력 오류
        """
        threshold = threshold_slope if threshold_slope is not None else self.threshold_slope

        try:
            if prev_sma_short == 0:
                logger.warning("is_entry_signal: prev_sma_short=0, 기울기 계산 불가")
                return False
            if any(v is None or np.isnan(v) for v in [sma_short, sma_long, prev_sma_short]):
                logger.warning("is_entry_signal: NaN 입력 감지")
                return False

            slope = (sma_short - prev_sma_short) / abs(prev_sma_short)
            cross_ok = sma_short > sma_long
            slope_ok = slope > threshold

            logger.debug(
                f"진입 체크 | SMA({sma_short:.4f}) > SMA_L({sma_long:.4f})={cross_ok} "
                f"| slope={slope:.6f} > {threshold}={slope_ok}"
            )
            return cross_ok and slope_ok

        except (TypeError, ZeroDivisionError) as e:
            logger.warning(f"is_entry_signal 오류: {e}")
            return False

    # ── 2. 분할매수 로직: 스텝형 ─────────────────────────────────────────────

    def is_rebuy_signal(
        self,
        current_price: float,
        entry_price: float,
        current_n: int,
        step_percent: float | None = None,
        max_step: int | None = None,
    ) -> bool:
        """
        스텝형 분할매수(물타기) 신호 판단.

        조건:
          (1) current_n < max_step  → 최대 횟수 미초과
          (2) current_price <= entry_price * (1 - step_percent * (current_n + 1))

        예시 (step_percent=0.03, entry=100):
          current_n=0 → 97원 이하일 때 1차 추가매수
          current_n=1 → 94원 이하일 때 2차 추가매수
          current_n=2 → 91원 이하일 때 3차 추가매수

        Parameters
        ----------
        current_price : float
            현재 가격
        entry_price : float
            최초 진입 가격 (기준 가격)
        current_n : int
            현재까지의 분할매수 횟수 (0 = 아직 없음)
        step_percent : float, optional
            분할매수 트리거 하락률. None이면 인스턴스 기본값.
        max_step : int, optional
            최대 분할매수 횟수. None이면 인스턴스 기본값.

        Returns
        -------
        bool
            True = 분할매수 신호, False = 미해당 또는 입력 오류
        """
        step = step_percent if step_percent is not None else self.step_percent
        max_n = max_step if max_step is not None else self.max_step

        try:
            if entry_price <= 0:
                logger.warning(f"is_rebuy_signal: entry_price={entry_price} <= 0")
                return False
            if current_n < 0:
                logger.warning(f"is_rebuy_signal: current_n={current_n} < 0")
                return False

            within_max = current_n < max_n
            trigger_price = entry_price * (1.0 - step * (current_n + 1))
            price_ok = current_price <= trigger_price

            logger.debug(
                f"분할매수 체크 | n={current_n}/{max_n} | "
                f"trigger={trigger_price:.2f} | current={current_price:.2f} | ok={price_ok}"
            )
            return within_max and price_ok

        except (TypeError, ValueError) as e:
            logger.warning(f"is_rebuy_signal 오류: {e}")
            return False

    # ── 3. 청산 로직: ATR 기반 다이내믹 출구 ────────────────────────────────

    def get_exit_price(
        self,
        current_price: float,
        atr: float,
        is_profit: bool,
        multiplier: float | None = None,
    ) -> float:
        """
        ATR 기반 다이내믹 익절·손절 가격 계산.

        로직:
          익절: current_price + (atr * multiplier)
          손절: current_price - (atr * multiplier)

        Parameters
        ----------
        current_price : float
            현재 가격
        atr : float
            Average True Range 값 (compute_atr()로 산출)
        is_profit : bool
            True → 익절 목표가, False → 손절 목표가
        multiplier : float, optional
            ATR 배수 (권장 범위 1.5~2.0). None이면 인스턴스 기본값.

        Returns
        -------
        float
            목표 가격. 입력 오류 시 -1.0 반환.
        """
        mult = multiplier if multiplier is not None else self.atr_multiplier

        try:
            if atr < 0:
                raise ValueError(f"ATR 음수 불가: {atr}")
            if current_price <= 0:
                raise ValueError(f"current_price <= 0: {current_price}")
            if mult <= 0:
                raise ValueError(f"multiplier <= 0: {mult}")

            offset = atr * mult
            target = current_price + offset if is_profit else current_price - offset

            logger.debug(
                f"{'익절' if is_profit else '손절'} 목표가 | "
                f"{current_price:.2f} {'+'if is_profit else '-'} "
                f"ATR({atr:.4f})×{mult} = {target:.2f}"
            )
            return round(target, 4)

        except (TypeError, ValueError) as e:
            logger.warning(f"get_exit_price 오류: {e}")
            return -1.0

    # ── 4. 리스크 관리: 변동성 Kill-Switch ──────────────────────────────────

    def check_kill_switch(
        self,
        price_series: ArrayLike,
        threshold: float = 0.05,
    ) -> bool:
        """
        변동성 Kill-Switch — 시장 과열 시 신규 진입 차단.

        최근 `kill_window`봉의 변동성 = std(prices) / mean(prices)
        이 값이 threshold를 초과하면 True 반환 → 모든 진입 신호 무효화.

        Parameters
        ----------
        price_series : array-like
            종가 시계열 (numpy array, pandas Series, list 모두 허용)
        threshold : float
            변동성 임계값 (기본 0.05 = 5%). 장세에 따라 0.03~0.10 조정.

        Returns
        -------
        bool
            True = 과변동 → 진입 차단 / False = 정상 범위 → 진입 허용
        """
        try:
            prices = pd.Series(price_series, dtype=float).dropna()

            if len(prices) < self.kill_window:
                logger.warning(
                    f"check_kill_switch: 데이터 부족 "
                    f"({len(prices)}/{self.kill_window}), 차단 안 함"
                )
                return False

            recent = prices.iloc[-self.kill_window:]
            mean_price = recent.mean()

            if mean_price == 0:
                logger.warning("check_kill_switch: mean=0, False 반환")
                return False

            volatility = recent.std(ddof=1) / mean_price
            triggered = bool(volatility > threshold)

            logger.debug(
                f"Kill-Switch | vol={volatility:.4f} > {threshold} = {triggered}"
            )
            return triggered

        except Exception as e:
            logger.warning(f"check_kill_switch 오류: {e}")
            return False

    # ── 헬퍼: ATR 계산 ──────────────────────────────────────────────────────

    def compute_atr(
        self,
        high: ArrayLike,
        low: ArrayLike,
        close: ArrayLike,
        period: int | None = None,
    ) -> float:
        """
        ATR(Average True Range) 계산 — 순수 pandas 구현 (외부 라이브러리 불필요).

        Parameters
        ----------
        high, low, close : array-like
            고가·저가·종가 시계열 (동일 길이)
        period : int, optional
            ATR 기간. None이면 인스턴스 기본값(14) 사용.

        Returns
        -------
        float
            최신 ATR 값. 데이터 부족 시 -1.0 반환.
        """
        n = period if period is not None else self.atr_period

        try:
            df = pd.DataFrame({
                "high":  pd.Series(high,  dtype=float),
                "low":   pd.Series(low,   dtype=float),
                "close": pd.Series(close, dtype=float),
            }).dropna()

            if len(df) < n + 1:
                logger.warning(f"compute_atr: 데이터 부족 ({len(df)} < {n + 1})")
                return -1.0

            prev_close = df["close"].shift(1)
            tr = pd.concat([
                df["high"] - df["low"],
                (df["high"] - prev_close).abs(),
                (df["low"]  - prev_close).abs(),
            ], axis=1).max(axis=1)

            latest = float(tr.ewm(span=n, adjust=False).mean().iloc[-1])

            if np.isnan(latest):
                logger.warning("compute_atr: NaN 반환됨")
                return -1.0

            return round(latest, 4)

        except Exception as e:
            logger.warning(f"compute_atr 오류: {e}")
            return -1.0

    # ── 헬퍼: SMA 계산 ──────────────────────────────────────────────────────

    @staticmethod
    def compute_sma(price_series: ArrayLike, period: int) -> tuple[float, float]:
        """
        현재 SMA와 직전 SMA를 함께 반환 (기울기 계산용).

        Parameters
        ----------
        price_series : array-like
            종가 시계열
        period : int
            이동평균 기간

        Returns
        -------
        tuple[float, float]
            (current_sma, prev_sma). 데이터 부족 시 (-1.0, -1.0).
        """
        try:
            s = pd.Series(price_series, dtype=float).dropna()
            if len(s) < period + 1:
                logger.warning(
                    f"compute_sma({period}): 데이터 부족 ({len(s)} < {period + 1})"
                )
                return -1.0, -1.0
            sma = s.rolling(period).mean()
            return float(sma.iloc[-1]), float(sma.iloc[-2])
        except Exception as e:
            logger.warning(f"compute_sma 오류: {e}")
            return -1.0, -1.0


# ── analysis_worker.py 연동 예제 ────────────────────────────────────────────
#
# Redis 포지션 상태 스키마 (Hash):
#   KEY   : "position:{ticker}"
#   FIELDS: in_position (0/1), entry_price (float), rebuy_count (int)
#
# 사용 예:
#
#   import json
#   import yfinance as yf
#   from strategy import TradingStrategy
#
#   strategy = TradingStrategy(
#       threshold_slope=0.001,   # SMA 기울기 0.1% 이상
#       step_percent=0.03,       # 3% 하락마다 분할매수
#       max_step=3,              # 최대 3회
#       atr_multiplier=1.5,      # ATR × 1.5 익절·손절
#   )
#
#   async def analyze_update(self, update_data: dict) -> dict:
#       ticker = update_data["ticker"]
#
#       # ── OHLCV 데이터 조회 ──────────────────────────────────────────────
#       df = yf.download(ticker, period="3mo", interval="1d", progress=False)
#       if df.empty or len(df) < 30:
#           return {"signal": "HOLD", "reason": "데이터 부족"}
#
#       close  = df["Close"]
#       high   = df["High"]
#       low    = df["Low"]
#       price  = float(close.iloc[-1])
#
#       # ── Kill-Switch 먼저 체크 ──────────────────────────────────────────
#       if strategy.check_kill_switch(close, threshold=0.05):
#           return {"signal": "HOLD", "reason": "변동성 Kill-Switch 발동"}
#
#       # ── ATR 계산 ──────────────────────────────────────────────────────
#       atr = strategy.compute_atr(high, low, close)
#
#       # ── SMA 계산 ──────────────────────────────────────────────────────
#       sma5,  prev5  = strategy.compute_sma(close, 5)
#       sma20, prev20 = strategy.compute_sma(close, 20)
#
#       # ── Redis에서 포지션 상태 조회 ────────────────────────────────────
#       pos = await self.redis.hgetall(f"position:{ticker}")
#       in_position  = pos.get("in_position", "0") == "1"
#       entry_price  = float(pos.get("entry_price", 0))
#       rebuy_count  = int(pos.get("rebuy_count", 0))
#
#       # ── 신호 판단 ─────────────────────────────────────────────────────
#       if not in_position:
#           if strategy.is_entry_signal(sma5, sma20, prev5):
#               tp = strategy.get_exit_price(price, atr, is_profit=True)
#               sl = strategy.get_exit_price(price, atr, is_profit=False)
#               # Redis에 포지션 저장
#               await self.redis.hset(f"position:{ticker}", mapping={
#                   "in_position": "1",
#                   "entry_price": str(price),
#                   "rebuy_count": "0",
#               })
#               return {
#                   "signal": "BUY",
#                   "reason": f"SMA 골든크로스 + 기울기 돌파",
#                   "take_profit": tp,
#                   "stop_loss": sl,
#               }
#       else:
#           if strategy.is_rebuy_signal(price, entry_price, rebuy_count):
#               await self.redis.hincrby(f"position:{ticker}", "rebuy_count", 1)
#               return {"signal": "BUY", "reason": f"분할매수 {rebuy_count + 1}차"}
#
#           tp = strategy.get_exit_price(entry_price, atr, is_profit=True,  multiplier=2.0)
#           sl = strategy.get_exit_price(entry_price, atr, is_profit=False, multiplier=1.0)
#           if price >= tp:
#               await self.redis.delete(f"position:{ticker}")
#               return {"signal": "SELL", "reason": f"익절 목표가 도달 ({tp:.2f})"}
#           if price <= sl:
#               await self.redis.delete(f"position:{ticker}")
#               return {"signal": "SELL", "reason": f"손절가 도달 ({sl:.2f})"}
#
#       return {"signal": "HOLD", "reason": "조건 미충족"}
