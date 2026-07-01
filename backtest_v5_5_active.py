"""
backtest_v5_5_active.py — 듀얼 모드 백테스팅 엔진 [위험감수형(High-Risk) v5.5]

  ※ 본 파일은 backtest.py(안전투자형 v4.6 검증본)의 완전한 사본에서, 하락장 대기 시간을 줄이고
    상승장 회전율·수익률을 극대화하도록 방어 메커니즘을 일부 완화한 '위험감수형' 변형이다.
    유저가 '내 포트폴리오' 탭에서 안전형(v4.6)/위험형(v5.5) 중 선택하며, 엔진은 risk_profile 로 분기한다.

  [v5.5 완화 스펙 4가지 — base 대비 변경점]
    [v5.5-1] 야수 진입 장벽 완화: MARKET_SCORE_BULL_CUTLINE 85→70pt (조금만 돌아도 빠른 야수 전환).
    [v5.5-2] 단일종목 캡 상향: MAX_POSITION_PCT_BULL 22%→33% (주도주 집중 투자).
    [v5.5-3] 서킷브레이커 완화: MAX_DAILY_SL 1→2종목 (변동성 정면 돌파).
    [v5.5-4] 디펜스 눌림목 매수 허용: ALLOW_DEFENSE_NULIM_BUY=True, 종목당 10% 캡(DEFENSE_NULIM_POS_PCT)
             으로 '낙주 매수' 허용(전면 차단 해제).
  그 외 모든 방어·리스크 로직(손절선 판정, 트레일링 +18%, 휩소가드 T+1, 지수5일선필터,
  노출캡, 글리치/시가평가 보정 등)은 안전형 v4.6 과 동일하게 유지한다.

================================================================================
[v4.6 변경점 — 6대 스펙 (base: backtest_v5.py 전체 사본)]
  v5(지수5일선필터·야수커트라인·모드별 노출캡)를 그대로 계승하고 아래 6가지를 추가/강화한다.
  v5 원본은 baseline 비교용으로 보존한다(같은 세션·날짜에 둘 다 돌려 같은 유니버스로 비교).

  [v4.6-1] 지수(Market Index) 5일선 매수 필터  ─ (v5에서 이미 구현, 계승)
    매수 시그널 당일 소속 지수(KOSPI/KOSDAQ) 종가가 5일선(SMA_5) 아래면 신규 매수 Skip.
    차단 횟수는 self._index_filter_pass_count 로 집계 → 성적표 "[지수필터 매수패스]"에 표기.

  [v4.6-2] 본전 보호(트레일링) 발동 마진 상향  +12% → +18%
    TRAIL_ACTIVATE_PROFIT = 0.18. 큰 추세를 더 길게 추종하도록 트레일링 ON 시점을 늦춘다.

  [v4.6-3 철회] 고정 손절선(SL) 폭 축소(-8.5%)  ─ 사용자 철회, 미적용
    과거 통제 백테스트에서 손절폭 하드컷(-9.5%)이 휩쏘·MDD 악화(-20%→-37%)로 기각된 이력을
    수용해 본 스펙은 철회한다. v5의 손절선 판정 로직(ATR×3.0, 야수 하한 -12%)을 그대로 유지하고,
    진입 초기 데미지는 아래 신규 방어 스펙(Spec 5 서킷브레이커, Spec 6 휩소 가드)으로 제어한다.

  [v4.6-4] 디펜스 모드 시 물타기(REBUY) + 눌림목 신규매수 전면 금지
    · 디펜스 모드 중 추가 분할매수(REBUY) 100% 비활성화 → self._defense_rebuy_block_count 집계.
    · 디펜스 모드 중 DRY_VOLUME_NULIM 신규 매수도 전면 차단(매수 단계 자체를 야수 모드로 한정).

  [v4.6-5] 당일 최대 손절 종목 수 제한 = 1개 (계좌 서킷브레이커)
    하루에 SELL_SL(진짜 손절)로 청산 가능한 종목 수를 MAX_DAILY_SL(1)로 제한.
    이미 1종목이 손절됐으면 당일 타 종목의 손절 집행을 보류(보유 지속, 익일 종가 재판정).
    트레일링 익절(SELL_TS)·고정익절(SELL_TP)·리밸런싱은 제한 없음. 대폭락일 동시다발 손절 차단.

  [v4.6-6] 진입 초기 휩소(Whipsaw) 방어  ─ 당일 포함 T+1(2거래일)
    진입 후 2거래일 이내(bars_held ≤ WHIPSAW_GUARD_BARS=1)에는 장중 저가(Low) 흔들림으로
    손절선이 터지지 않도록 실시간 손절을 유예하고 '종가(Close)' 기준으로만 손절 여부를 판정한다.

  ── 이하 v5 계승 사항 ──
  본 파일은 backtest_v5.py 의 완전한 사본에서 위 규칙만 강화한 버전이다.

  [v5-1] 지수(Market Index) Top-Down 매수 필터
    개별 종목 매수 조건이 충족돼도, 그 종목이 속한 시장 지수(KOSPI/KOSDAQ)의
    당일 종가가 '지수 5일 이동평균선(SMA_5)' 아래면 신규 매수를 원천 차단(Skip).
    .KS → KOSPI, .KQ → KOSDAQ 로 종목의 소속 지수를 판별한다.
    로그: "[PASS] 지수 5일선 하회로 인한 매수 잠금 (시장 리스크 방어)"
    패스 횟수는 self._index_filter_pass_count 로 집계해 성적표에 표기.

  [v5-2] 야수 모드 진입 조건/판정 기준 강화 (Market Score 커트라인 도입)
    v4의 'KOSPI·KOSDAQ 중 하나라도 SMA_60 위면 무조건 야수'(98% 야수 과다) 판정을
    폐기하고, 두 지수의 정렬·추세 상태를 0~100점 Market Score 로 환산해
    평균 점수 >= MARKET_SCORE_BULL_CUTLINE(85pt)일 때만 야수 모드로 진입한다.
    커트라인 미만이면 즉시 디펜스 모드:
      · 종목당 투입 비중 캡 <= MAX_POSITION_PCT(15%)
      · 총 주식노출 캡을 MAX_GROSS_EXPOSURE_BEAR(30%)로 강제 축소 → 현금 70% 확보.

  [v5-3] 포트폴리오 노출 상한 상향(45%→55%) + 리밸런싱 단순화
    · 야수 모드 총노출 상한 MAX_GROSS_EXPOSURE_BULL = 55%(기존 45% 대비 상향).
    · 노출 상한 도달 시에도 '당일 상승 중'인 종목은 기계적 부분매도(SELL_REBAL)를
      유예하고 온전히 홀딩 → 우상향 추세주(리츠 등)가 일일 분할매도로 깎이는 현상 방지.
      청산은 익절(TP)·트레일링스톱(TS)·손절(SL) 신호로만 수행. 비상승 종목만 트림.
================================================================================

[야수 모드 - 상승장]  벤치마크 지수 SMA_60 위 (KOSPI·KOSDAQ 모두)
  → SMA_GOLDEN_CROSS: 거래대금 3배+ 폭발 + 5일선 20일선 돌파 → 추격 돌파 매매
  → MAX 8종목 동시 보유, 물타기 금지

[디펜스 모드 - 하락장]  벤치마크 지수 SMA_60 아래 (모든 시장 동시)
  → DRY_VOLUME_NULIM: 10일 내 5배+ 폭발 후 거래량 70% 이하 급감 눌림목
  → MAX 3종목 강한 통제, 물타기 최대 2회 (단, 종목 비중 상한 내에서만)

[리스크 관리 v2 — MDD -99% → -20% 이내 방어 재설계]
  포트폴리오 '파산'의 진짜 원인은 ① 종목 단위 통제만 있고 계좌 전체 통제가 없어
  동시다발 폭락·반복 재진입의 누적 낙폭을 못 막은 점, ② 보유종목 당일 바 누락 시
  평가액을 0으로 계산하던 시가평가 버그(자산곡선 허수 폭락)였다.

  1) 트레일링 스톱 지연 활성화: 진입 즉시 감시(조기 약손실 청산)를 폐지.
     보유 수익률이 최초 +12%(TRAIL_ACTIVATE_PROFIT)에 도달한 뒤에만 트레일링 ON.
     그 전에는 고정 익절(TP)·ATR 손절(SL)로만 판단 → 흔들림 칼손절 차단.
     [v4] 활성화 순간 손절선을 '본전 +0.5%(수수료 보전선)'으로 즉시 상향 → 10%대
     수익을 줬던 종목이 마이너스 손절로 돌변하는 경로를 원천 차단(이익 보호 전용).
     (참고: 손절폭 하드컷 -9.5% 안은 통제 백테스트에서 휩쏘·MDD 악화로 기각 — _resolve_stop_loss 주석)
  2) ATR 고정 리스크 배정 + '15% 하드캡': 수량 = (총자산×2%)/(진입가−손절가),
     단일 종목 평가액 ≤ 총자산 MAX_POSITION_PCT(15%) 강제. RSI≥70 진입 금지.
  3) 드로다운 방어 = 상시 노출상한 + 낙폭 매수잠금 (강제 매도 없음):
     · 총 주식노출 ≤ 총자산 55%(MAX_GROSS_EXPOSURE) 상시 제한 → 곡선 전체 스케일다운.
     · 전고점 대비 -8%(DD_BUY_LOCK) 초과 낙폭이면 신규 매수만 잠금(보유분은
       ATR손절·트레일링·익절로만 정상 청산).
     [폐기] 이산 하드스톱(해제 시 전고점 리베이스→하락장 낙폭 누적, MDD -30%대)과
     CPPI 강제 디리스킹(바닥 sell-low 휩쏘로 전략 edge 파괴, 수익 음전환)은 모두 폐기.
  4) ATR 손절 버퍼 상향(2.5→3.0×ATR, 하한 -8%→-12%)으로 노이즈 칼손절 방지 +
     보수적 체결: 일봉 저가가 손절선 터치 시 '손절선 −1%'(갭하락 시 시초가)로 당일 청산.

- 시가평가 보정: 당일 바 없는 보유종목은 직전 종가로 평가(_last_close 캐리).
- MDD: 실제 자산 가치(max_peak_asset) 기준 낙폭 비율 계산
- UI 완벽 지원: StockScreener.screen() 및 _score_chunk() 연동
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import yfinance as yf

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
# 글로벌 설정
# ══════════════════════════════════════════════════════════════════════════════

SCREEN_MARKETS    = ["KOSPI", "KOSDAQ"]
SCREEN_UNIVERSE_N = 200
VOLUME_TOP_N      = 100
NEWS_CANDIDATE_N  = 15

# 독립 실행(python backtest.py) 기본 구간 — 장기 스모크 테스트용(라이브는 인자로 주입).
START_DATE = "2020-01-01"
END_DATE   = "2024-12-31"

INITIAL_CAPITAL = 10_000_000

DEPOSIT_SCHEDULE: dict[str, int] = {
    "2021-01-04": 5_000_000,
    "2022-01-03": 5_000_000,
    "2023-01-02": 5_000_000,
}

RESULT_DIR = Path(__file__).parent / "backtest_results"


def compute_mdd(assets) -> float:
    """표준 최대낙폭(MDD, %) — 누적 최고점 대비 최대 하락폭을 음수로 반환(예: -18.46).

    공식: drawdown = (asset − cummax(asset)) / cummax(asset);  MDD = min(drawdown) × 100.
    데이터 글리치(NaN·inf·0 이하)는 허수 낙폭(피크/트로프 왜곡)의 원인이므로 사전 제거한다.
    유효 표본이 2개 미만이면 0.0 반환.
    """
    clean = [float(a) for a in assets if a is not None and np.isfinite(a) and a > 0]
    if len(clean) < 2:
        return 0.0
    s        = pd.Series(clean, dtype="float64")
    peak     = s.cummax()
    drawdown = (s - peak) / peak
    mdd      = drawdown.min() * 100.0
    return float(mdd) if np.isfinite(mdd) else 0.0


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
        stock_value = 0.0
        for code, qty in self.portfolio.items():
            price = current_market_prices.get(code, 0.0)
            # NaN/inf 가격이 평가액을 오염시켜 자산곡선이 튀는 것을 방지(글리치 방어)
            if not np.isfinite(price) or price <= 0:
                continue
            stock_value += qty * price
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


@dataclass
class PositionState:
    in_position:     bool  = False
    entry_price:     float = 0.0
    rebuy_count:     int   = 0
    take_profit:     float = -1.0
    stop_loss:       float = -1.0
    peak_price:      float = 0.0
    signal_tag:      str   = ""
    allocated_pct:   float = 0.0
    trailing_active: bool  = False   # [v2-1] +15% 도달 후에만 True
    entry_idx:       int   = -1      # [v4.6-6] 진입 시점 df 인덱스 위치(휩소 가드 보유일수 계산용)

    def reset(self) -> None:
        self.in_position   = False
        self.entry_price   = 0.0
        self.rebuy_count   = 0
        self.take_profit   = -1.0
        self.stop_loss     = -1.0
        self.peak_price      = 0.0
        self.signal_tag      = ""
        self.allocated_pct   = 0.0
        self.trailing_active = False
        self.entry_idx       = -1


# ══════════════════════════════════════════════════════════════════════════════
# 백테스팅 엔진 (듀얼 모드 야수 + 디펜스)
# ══════════════════════════════════════════════════════════════════════════════

class BacktestEngine:
    WARMUP_DAYS     = 300
    MAX_POS_BULL    = 8    # 야수 모드(상승장): 최대 8종목
    MAX_POS_BEAR    = 3    # 디펜스 모드(하락장): 최대 3종목
    MAX_REBUY_COUNT = 2    # 물타기 최대 허용 횟수 (DRY_VOLUME_NULIM 전용)

    # ── [v2-2] ATR 기반 고정 리스크 배정 + 단일종목 하드캡 ───────────────────
    MAX_RISK_PER_TRADE = 0.02   # 단일 종목 손절 시 허용 손실 = 총자산의 2%
    MAX_POSITION_PCT   = 0.15   # 단일 종목 평가액 상한 = 총자산의 15% (디펜스/기본 하드캡)
    # [v4] 야수 모드 한정 단일종목 캡 상향 — 주도주 상승기 익절 효율 극대화.
    # [v5.5-2] 위험감수형: 야수 단일종목 캡 22%→33%로 상향(주도주 집중 투자).
    #   총노출 상한(MAX_GROSS_EXPOSURE_BULL 55%)은 유지되므로 1종목 33%여도 포트폴리오는 55%에서 제한.
    MAX_POSITION_PCT_BULL = 0.33  # 야수(SMA_GOLDEN_CROSS) 진입 시 단일종목 상한 = 33%
    # [v5.5-4] 위험감수형: 디펜스(하락장) 눌림목(DRY_VOLUME_NULIM) 신규 매수를 전면 차단하지 않고
    #   종목당 투입 비중 10% 캡으로 제한 허용('낙주 매수'). 차단 대신 소액 진입으로 변동성 정면 돌파.
    DEFENSE_NULIM_POS_PCT = 0.10  # 디펜스 모드 눌림목 단일종목 상한 = 10%
    ALLOW_DEFENSE_NULIM_BUY = True  # 디펜스 모드 눌림목 신규매수 허용 여부(안전형=False, 위험형=True)
    SL_FALLBACK_PCT    = 0.90   # ATR 미산출 시 손절 폴백 (-10%)

    # ── [v4] ATR 손절 버퍼: v3 값 유지 (확대안은 백테스트로 기각) ─────────────
    # 시도: ATR 배수 ×1.15(3.0→3.45)·하한 -12%→-14% 확대 → 2020~2026 백테스트에서
    # MDD -28.5%→-43%, 수익 +27%→-12% 로 급격히 악화(하락장에서 패자를 더 오래 들고감).
    # 결론: 잦은 손절(잽) 방지는 '버퍼 확대'가 아니라 아래 '재진입 쿨다운'으로 해결.
    ATR_SL_MULT_BULL  = 3.0     # 야수 손절 ATR 배수 (v3 유지)
    BULL_SL_FLOOR_PCT = 0.88    # 야수 손절 하한 -12% (v3 유지)
    ATR_SL_MULT_BEAR  = 3.0     # 디펜스(눌림목) 손절 ATR 배수 (v3 유지)
    SL_SLIPPAGE_PCT   = 0.01    # 손절 체결 시 손절선 -1% 보수 반영

    # ── [v4-튜닝, 기각] 단일종목 최대 손절 폭 하드컷(-9.5%) — 역효과로 폐기 ────
    # 시도: ATR 손절선이 -9.5%보다 깊어도 -9.5%에서 칼손절해 '잽' 데미지를 줄이려 함.
    # 통제 백테스트 결과(동일 유니버스): 트레일링만(-20.2%/MDD -16.5%) → 하드컷 추가 시
    # -37.3%/MDD -35.1%로 급악화. 원인 ① -9.5%는 모멘텀주 정상 되돌림 안쪽이라 휩쏘
    # 손절이 191→260회로 폭증 ② 고정리스크 사이징이 손절폭을 좁히자 종목비중을 16→21%로
    # 역팽창시켜 휩쏘 손실·MDD를 함께 키움. 결론: 손절폭 조이기는 이 전략과 상극 → 미적용.

    # ── [v4] 손절 직후 동일종목 재진입 쿨다운 (뇌동매매·재추격 방지) ──────────
    # 백테스트상 수익 +3.8M·MDD -1.8%p 개선 효과 확인(ATR 확대 대비 안전한 잽 방지책).
    REENTRY_COOLDOWN_DAYS = 7   # SELL_SL 발생 종목은 7일(캘린더, ≈5거래일) 신규진입 금지

    # ── [v4-튜닝] 트레일링 스톱 지연 활성화 + 본전 보전선 ────────────────────
    # v3의 +15% 활성화는 익절 전 흔들림을 피했으나, 10%대 수익을 줬다가 마이너스
    # 손절로 돌변하는 케이스가 남았다. → 활성화 시점을 +12%로 앞당기고(지연 방지),
    # 활성화 즉시 손절선을 '진입가 본전 + 0.5%(수수료 보전선)'으로 끌어올려 수익
    # 보존을 강제한다. 일단 +12% 수익을 준 종목이 손실로 돌변하는 경로를 원천 차단.
    # [v4.6-2] 본전 보호(트레일링) 발동 마진 상향: +12% → +18% (큰 추세 추종 강화)
    TRAIL_ACTIVATE_PROFIT = 0.18  # 수익률 +18% 최초 도달 시 트레일링 즉시 ON
    TRAIL_BREAKEVEN_PCT   = 0.005 # 활성화 시 손절선을 본전 +0.5%(수수료 보전선)로 상향

    # [v4.6-3 철회] 고정 손절선 -8.5% 축소 스펙은 사용자가 철회(과거 백테스트 기각 이력 수용).
    # → v5의 손절선 판정 로직(ATR×3.0 + 야수 하한 -12%)을 그대로 유지한다. 상수 미도입.

    # [v4.6-5] 계좌 서킷브레이커: 하루에 SELL_SL(진짜 손절)로 청산 가능한 최대 종목 수.
    # [v5.5-3] 위험감수형: 서킷브레이커 완화 1→2종목(변동성 정면 돌파). 0 이하면 사실상 해제.
    MAX_DAILY_SL = 2              # 당일 2종목까지 손절 허용(안전형=1, 위험형=2)

    # [v4.6-6] 진입 초기 휩소 방어 윈도: 진입 후 이 바 수 이내는 종가 기준으로만 손절 판정.
    WHIPSAW_GUARD_BARS = 1        # bars_held ≤ 1 → 당일 포함 T+1(2거래일) 장중 손절 유예

    # ── [v2-2] RSI 하드 필터: 이 값 이상이면 신규 진입 전면 금지 ─────────────
    RSI_HARD_LIMIT       = 70.0
    NULIM_PRIORITY_BONUS = 8.0   # 승률 높은 눌림목 전략 확신도 가산점

    # ── [v3] 드로다운 방어 (상시 노출 상한 + 낙폭 매수잠금) ──────────────────
    # 핵심 교훈: 낙폭 바닥에서 강제 매도(CPPI 디리스킹/하드청산)는 휩쏘 손실로
    # 전략 edge 자체를 파괴한다(+2%/거래 → -1.4%). 그래서 '강제 매도'는 폐기하고
    #   ① 상시 총주식노출을 총자산의 MAX_GROSS_EXPOSURE 로 제한(곡선 전체 스케일다운)
    #   ② 전고점 대비 낙폭이 DD_BUY_LOCK 초과 시 '신규 매수만' 잠금
    #      (보유분은 강제청산하지 않고 ATR손절·트레일링·익절로만 정상 청산 → edge 보존)
    # 로 MDD를 비례 축소한다. 전고점(HWM) 추적은 낙폭 매수잠금 판정용.
    # [v5-3] 노출 상한 상향: 야수장 45%→55%. 디펜스장은 30%로 더 조여 현금을 강제 확보한다.
    #   야수(상승) 모드  → MAX_GROSS_EXPOSURE_BULL = 0.55
    #   디펜스(하락) 모드 → MAX_GROSS_EXPOSURE_BEAR = 0.30 (현금 70% 확보, [v5-2] 연동)
    # 일별 모드(self._current_is_bear)에 따라 _gross_exposure_pct() 가 둘 중 하나를 반환.
    MAX_GROSS_EXPOSURE_BULL = 0.55
    MAX_GROSS_EXPOSURE_BEAR = 0.30
    MAX_GROSS_EXPOSURE = MAX_GROSS_EXPOSURE_BULL  # 하위호환(리포트 라벨·정적 참조용 기본값)
    DD_BUY_LOCK        = 1.00   # 전고점 대비 낙폭 매수잠금 (1.00=사실상 OFF; 노출상한이 주 방어)

    # [v5-2] 야수 모드 진입 커트라인 — 두 지수 Market Score 평균이 이 값 이상이어야 야수.
    # (v4의 'SMA_60 위 하나라도 → 야수' 대비 보수적. 미만이면 즉시 디펜스 모드)
    # [v5.5-1] 위험감수형: 진입 장벽 85pt→70pt 하향(시장이 조금만 돌아도 빠르게 야수 전환).
    MARKET_SCORE_BULL_CUTLINE = 70.0

    # ── [v3.1] 데이터 글리치 방어: 단일봉 비정상 가격점프 필터 ────────────────
    # 한국주식 일일 등락 제한은 ±30% → 단일봉 2.5배↑/0.4배↓ 변동은 데이터 오류로 간주하고
    # 직전 '정상가'로 평가(평가액 폭주·유령 리밸런싱·MDD 폭발 차단). 단, 새 레벨이 이틀
    # 연속 유지되면 실제 가격변화로 수용해 특정 종목 평가가 영구 동결되는 것을 방지한다.
    GLITCH_JUMP_HIGH = 2.5
    GLITCH_JUMP_LOW  = 0.4

    BENCHMARK_INDEXMAP = {
        "KOSPI":  "^KS11",
        "KOSDAQ": "^KQ11",
        "S&P500": "^GSPC",
        "NASDAQ": "^IXIC",
    }

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

        self.simulator    = TradingSimulator(initial_capital)
        self.strategy     = TradingStrategy()
        self.positions:    dict[str, PositionState] = {}
        self.trade_log:    list[dict] = []
        self.equity_curve: list[dict] = []
        self._market_index_dfs: dict[str, pd.DataFrame] = {}
        # [v3] 드로다운 방어 상태 (상시 노출상한 + 낙폭 매수잠금)
        self._peak_asset:        float = float(initial_capital)  # 전고점 자산(HWM, 상승만)
        self._exposure_cap_value: float = float("inf")           # 당일 허용 주식 평가액(=MAX_GROSS_EXPOSURE×총자산), 리밸런싱 백스톱용
        self._buy_locked:        bool  = False                   # 낙폭 매수잠금 활성 여부
        # [v2] 시가평가 보정용: 종목별 마지막 '정상' 종가 캐리 (당일 바 누락·글리치 시 사용)
        self._last_close:        dict[str, float] = {}
        # [v3.1] 글리치 필터용: 종목별 마지막 '관측' 종가(글리치 포함) — 새 레벨 재동기 판정
        self._last_raw_close:    dict[str, float] = {}
        # [v4] 손절 직후 재진입 쿨다운: 종목 → 마지막 SELL_SL 일자
        self._sl_block:          dict[str, pd.Timestamp] = {}
        # [v5] 당일 모드 캐시(하루 1회 평가) — 노출상한·캡 산정이 일관되게 같은 모드를 참조하도록.
        self._current_is_bear:   bool = True
        # [v5-1] 지수 5일선 하회로 매수가 패스된 누적 횟수(성적표 표기용)
        self._index_filter_pass_count: int = 0
        # [v4.6-4] 디펜스 모드로 차단된 물타기(REBUY) 누적 횟수(성적표 표기용)
        self._defense_rebuy_block_count: int = 0
        # [v4.6-5] 당일 손절 종목 수(서킷브레이커) — 매일 0으로 리셋 / 누적 차단 횟수
        self._daily_sl_count: int = 0
        self._daily_sl_blocked_count: int = 0

    # ── 데이터 로드 ──────────────────────────────────────────────────────────

    def _load_data(self) -> dict[str, pd.DataFrame]:
        load_start = self.start_date - pd.Timedelta(days=self.WARMUP_DAYS)
        load_end   = self.end_date   + pd.Timedelta(days=2)

        if self._ticker_names:
            all_tickers = list(self._ticker_names.keys())
            print(f"  유니버스 크기: {len(all_tickers)}개 종목 (사전 바인딩)")
        else:
            screener = StockScreener(universe_per_market=self.universe_n)
            universe = screener.build_universe(self.markets)
            all_tickers = [t for t, _ in universe]
            self._ticker_names = {t: n for t, n in universe}
            print(f"  유니버스 크기: {len(all_tickers)}개 종목 자동 구축 완료")

        ticker_data: dict[str, pd.DataFrame] = {}
        CHUNK = 50

        for i in range(0, len(all_tickers), CHUNK):
            chunk = all_tickers[i: i + CHUNK]
            hi = min(i + CHUNK, len(all_tickers))
            print(f"  [{i + 1}~{hi}] 배치 데이터 로드 중...", end=" ", flush=True)
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

                    # SMA_5 보정: stock_ai._add_indicators 미포함 시 인라인 계산
                    if "SMA_5" not in df.columns:
                        df["SMA_5"] = df["Close"].rolling(5).mean()

                    # [개편 2] 종목별 Wilder RSI_14 — 진입 직전 과열 하드필터용
                    # stock_ai 의 SMA식 RSI(컬럼명 "RSI")와 별개로 일관된 Wilder 방식 사용
                    df["RSI_14"] = self._calc_rsi(df["Close"], period=14)

                    ticker_data[ticker] = df
                except Exception:
                    continue

            print(f"누적 {len(ticker_data)}개 성공")

        return ticker_data

    def _load_benchmark_indices(self) -> None:
        load_start = self.start_date - pd.Timedelta(days=self.WARMUP_DAYS)
        load_end   = self.end_date   + pd.Timedelta(days=2)

        for mkt in self.markets:
            idx_symbol = self.BENCHMARK_INDEXMAP.get(mkt)
            if not idx_symbol:
                continue
            print(f"  📊 {mkt} 벤치마크 지수({idx_symbol}) 로드 중...", end=" ", flush=True)
            try:
                raw = yf.download(
                    idx_symbol,
                    start=load_start.strftime("%Y-%m-%d"),
                    end=load_end.strftime("%Y-%m-%d"),
                    auto_adjust=True,
                    progress=False,
                )
                if not raw.empty:
                    df = _flatten_columns(raw)
                    df["SMA_5"]  = df["Close"].rolling(5).mean()
                    df["SMA_20"] = df["Close"].rolling(20).mean()
                    df["SMA_60"] = df["Close"].rolling(60).mean()
                    # [Step 2] 과열 판단용 14일 RSI 사전 계산
                    df["RSI_14"] = self._calc_rsi(df["Close"], period=14)
                    self._market_index_dfs[mkt] = df
                    print(f"{len(df)}일치 완료")
                else:
                    print("데이터 없음")
            except Exception as e:
                print(f"실패 ({e})")

    # ── 장세 판별 ─────────────────────────────────────────────────────────────

    # ── [v5-2] 지수별 Market Score (0~100pt) ──────────────────────────────────

    @staticmethod
    def _score_single_index(row: pd.Series) -> float | None:
        """단일 지수의 정렬·추세 상태를 0~100점으로 환산. 산출 불가 시 None.

        배점(합 100): 종가>5일선 25 / 종가>20일선 25 / 종가>60일선 30 / 5일선>20일선 20.
        → 완전 정배열·전 이평선 위 = 100점. 단기 약세부터 점수가 즉시 깎인다.
        """
        close = float(row.get("Close",  np.nan))
        sma5  = float(row.get("SMA_5",  np.nan))
        sma20 = float(row.get("SMA_20", np.nan))
        sma60 = float(row.get("SMA_60", np.nan))
        if any(np.isnan(v) for v in (close, sma5, sma20, sma60)):
            return None
        score = 0.0
        if close > sma5:
            score += 25.0
        if close > sma20:
            score += 25.0
        if close > sma60:
            score += 30.0
        if sma5 > sma20:
            score += 20.0
        return score

    def _compute_market_score(self, date: pd.Timestamp) -> float:
        """추적 중인 모든 지수(KOSPI·KOSDAQ) Market Score 의 '평균'.

        v4의 'OR(하나라도 위)' 낙관 판정과 달리 평균을 쓰므로, 한 지수만 강해도
        다른 지수가 약하면 점수가 끌어내려져 보수적으로 디펜스 전환된다.
        유효 지수가 없으면 0.0(=디펜스) 반환.
        """
        scores: list[float] = []
        for _, df in self._market_index_dfs.items():
            idx_loc = df.index.get_indexer([date], method="ffill")[0]
            if idx_loc < 0:
                continue
            s = self._score_single_index(df.iloc[idx_loc])
            if s is not None:
                scores.append(s)
        return sum(scores) / len(scores) if scores else 0.0

    def _is_bear_market(self, date: pd.Timestamp) -> bool:
        """[v5-2] Market Score 평균 < 커트라인(85pt)이면 디펜스(하락장, True).

        지수 데이터가 아예 없으면 보수적으로 디펜스(True)로 본다.
        """
        if not self._market_index_dfs:
            return True
        return self._compute_market_score(date) < self.MARKET_SCORE_BULL_CUTLINE

    def _get_max_positions(self, date: pd.Timestamp) -> int:
        """장세별 동적 최대 보유 종목 수: 상승장 8 / 하락장 3"""
        return self.MAX_POS_BEAR if self._is_bear_market(date) else self.MAX_POS_BULL

    # ── [Step 2] RSI 계산 유틸리티 ───────────────────────────────────────────

    @staticmethod
    def _calc_rsi(close_series: pd.Series, period: int = 14) -> pd.Series:
        """Wilder's RSI 계산 (alpha=1/period EWM 스무딩)"""
        delta    = close_series.diff()
        gain     = delta.where(delta > 0, 0.0)
        loss     = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        rs       = avg_gain / avg_loss.replace(0.0, np.nan)
        rsi      = 100.0 - (100.0 / (1.0 + rs))
        return rsi.fillna(50.0)

    def _get_market_rsi(self, date: pd.Timestamp) -> float:
        """진입 시점의 시장 지수 RSI 반환 — 복수 지수 중 최댓값(가장 과열된 지수 기준)"""
        rsi_values: list[float] = []
        for _, df in self._market_index_dfs.items():
            if "RSI_14" not in df.columns:
                continue
            idx_loc = df.index.get_indexer([date], method="ffill")[0]
            if idx_loc < 0:
                continue
            val = float(df["RSI_14"].iloc[idx_loc])
            if not np.isnan(val):
                rsi_values.append(val)
        return max(rsi_values) if rsi_values else 50.0

    # ── [v2-3] 계좌 MDD 기반 2단계 서킷브레이커 ───────────────────────────────

    def _index_above_ma20(self, date: pd.Timestamp) -> bool:
        """KOSPI 또는 KOSDAQ 중 '하나라도' 종가가 자기 20일선 위에 안착했는가 (해제 보조 조건)."""
        for _, df in self._market_index_dfs.items():
            if "SMA_20" not in df.columns:
                continue
            idx_loc = df.index.get_indexer([date], method="ffill")[0]
            if idx_loc < 0:
                continue
            row   = df.iloc[idx_loc]
            close = float(row.get("Close", np.nan))
            sma20 = float(row.get("SMA_20", np.nan))
            if np.isnan(close) or np.isnan(sma20):
                continue
            if close > sma20:
                return True
        return False

    def _mark_to_market(
        self, date: pd.Timestamp, ticker_data: dict[str, pd.DataFrame]
    ) -> dict[str, float]:
        """당일 종가 맵 구성 + 직전 종가 캐리.

        [v2 버그픽스] 보유 종목의 당일 바가 없을 때(거래정지·결측) 평가액을 0으로
        떨어뜨리던 문제를 차단 — 직전 종가(_last_close)로 평가해 자산곡선 허수 폭락 방지.
        """
        prices: dict[str, float] = {}
        for t, df in ticker_data.items():
            if date not in df.index:
                continue
            p = float(df.loc[date, "Close"])
            if not (np.isfinite(p) and p > 0):         # NaN·0·음수 종가 → 직전 정상가 캐리
                if t in self._last_close:
                    prices[t] = self._last_close[t]
                continue

            last = self._last_close.get(t)             # 마지막 '정상가'(글리치 제외 기준)
            raw  = self._last_raw_close.get(t)         # 마지막 '관측가'(글리치 포함)
            self._last_raw_close[t] = p                # 원관측가는 항상 갱신
            jump = last is not None and (
                p > last * self.GLITCH_JUMP_HIGH or p < last * self.GLITCH_JUMP_LOW
            )
            # 비정상 점프지만 직전 관측가와 동일 레벨이면(이틀 연속) 실제 변화로 수용 → 영구 동결 방지
            stable = raw is not None and 0.7 * raw <= p <= 1.3 * raw
            if jump and not stable:                    # 단일봉 글리치 → 평가에서 직전 정상가 사용
                if t in self._last_close:
                    prices[t] = self._last_close[t]    # _last_close 미갱신(오염 방지)
            else:                                      # 정상가 → 채택 및 캐리 갱신
                prices[t] = p
                self._last_close[t] = p
        for t in self.positions:                       # 보유분은 직전가로라도 반드시 평가
            if t not in prices and t in self._last_close:
                prices[t] = self._last_close[t]
        return prices

    def _gross_invested(self, prices: dict[str, float]) -> float:
        """현재 주식 평가액 합계."""
        return sum(
            self.simulator.portfolio.get(t, 0) * prices.get(t, 0.0)
            for t in self.positions
        )

    # ── [v5] 모드별 총노출 상한 비율 ──────────────────────────────────────────

    def _gross_exposure_pct(self) -> float:
        """[v5-2/3] 당일 모드별 총 주식노출 상한 비율.
        디펜스(하락) → 30%(현금 70% 강제 확보) / 야수(상승) → 55%."""
        return (
            self.MAX_GROSS_EXPOSURE_BEAR if self._current_is_bear
            else self.MAX_GROSS_EXPOSURE_BULL
        )

    # ── [v5-1] 지수 5일선 Top-Down 매수 필터 ──────────────────────────────────

    @staticmethod
    def _market_of(ticker: str) -> str | None:
        """종목 코드 접미사로 소속 지수 판별: .KS→KOSPI, .KQ→KOSDAQ, 그 외 None."""
        if ticker.endswith(".KS"):
            return "KOSPI"
        if ticker.endswith(".KQ"):
            return "KOSDAQ"
        return None

    def _index_below_sma5(self, market: str, date: pd.Timestamp) -> bool:
        """[v5-1] 해당 시장 지수의 당일 종가가 5일선(SMA_5) 아래인가.

        지수 데이터가 없으면 보수적으로 True(매수 차단)로 본다.
        시장 미상(None)일 땐 필터 적용 불가 → False(통과)."""
        df = self._market_index_dfs.get(market)
        if df is None:
            return True
        idx_loc = df.index.get_indexer([date], method="ffill")[0]
        if idx_loc < 0:
            return True
        row   = df.iloc[idx_loc]
        close = float(row.get("Close", np.nan))
        sma5  = float(row.get("SMA_5", np.nan))
        if np.isnan(close) or np.isnan(sma5):
            return True
        return close < sma5

    def _exposure_cap_live(self, prices: dict[str, float]) -> float:
        """실시간 노출 상한 평가액 = (모드별 노출비율) × 현재 총자산.
        체결 직전마다 라이브로 재계산 → 같은 날 여러 매수가 누적돼도 상한 불침범 보장."""
        return self._gross_exposure_pct() * self.simulator.get_total_asset(prices)

    def _exposure_room(self, prices: dict[str, float]) -> float:
        """추가 매수 가능 금액(원) = 노출상한 − 현재 주식 평가액 (음수면 0).
        모든 매수 경로(_try_buy·물타기)가 통과해야 하는 하드 게이트의 단일 기준."""
        return max(0.0, self._exposure_cap_live(prices) - self._gross_invested(prices))

    def _is_rising_today(
        self, ticker: str, date: pd.Timestamp, ticker_data: dict[str, pd.DataFrame]
    ) -> bool:
        """[v5-3] 당일 종가가 전일 종가보다 높은가(상승 추세 중인가). 판정 불가 시 False."""
        df = ticker_data.get(ticker)
        if df is None or date not in df.index:
            return False
        idx = df.index.get_loc(date)
        if idx < 1:
            return False
        close = float(df.iloc[idx]["Close"])
        prev  = float(df.iloc[idx - 1]["Close"])
        if not (np.isfinite(close) and np.isfinite(prev) and prev > 0):
            return False
        return close > prev

    def _update_risk_state(
        self, date: pd.Timestamp, date_str: str, total_asset: float,
        prices: dict[str, float], ticker_data: dict[str, pd.DataFrame] | None = None,
    ) -> None:
        """[v3+v5] 전고점(HWM) 추적 + (모드별)노출상한 리밸런싱 + 낙폭 매수잠금.

        - _exposure_cap_value = (모드별 노출비율) × 총자산. [v5-2/3] 야수 55% / 디펜스 30%.
        - 총 주식노출이 상한을 넘으면 초과분을 평가액 큰 종목부터 '부분 매도'로 리밸런싱.
          단 [v5-3] '당일 상승 중'인 종목은 트림 유예 → 우상향 추세주를 일일 분할매도로
          깎지 않는다(청산은 TP/TS/SL 신호로만). 비상승 종목부터만 노출을 줄인다.
        - 전고점 대비 낙폭 > DD_BUY_LOCK 이면 신규 매수 잠금(보유분은 스톱으로만 청산).
        """
        if total_asset > self._peak_asset:
            self._peak_asset = total_asset
        dd = (self._peak_asset - total_asset) / self._peak_asset if self._peak_asset > 0 else 0.0
        cap = self._gross_exposure_pct() * total_asset
        self._exposure_cap_value = cap

        # ── 노출 상한 초과분 부분 매도 리밸런싱 ([v5-3] 상승 종목은 유예) ────────
        invested = self._gross_invested(prices)
        if invested > cap:
            over = invested - cap
            for ticker in sorted(
                self.positions.keys(),
                key=lambda t: self.simulator.portfolio.get(t, 0) * prices.get(t, 0.0),
                reverse=True,
            ):
                if over <= 0:
                    break
                # [v5-3] 당일 상승 중인 추세주는 기계적 부분매도 유예(온전히 홀딩)
                if ticker_data is not None and self._is_rising_today(ticker, date, ticker_data):
                    continue
                price = prices.get(ticker)
                held  = self.simulator.portfolio.get(ticker, 0)
                if not price or held <= 0:
                    continue
                sell_qty = min(held, int(over / price) + 1)
                if sell_qty <= 0:
                    continue
                pos = self.positions[ticker]
                if self.simulator.execute_sell(ticker, price, sell_qty):
                    self._log_trade(
                        date_str, ticker, "SELL_REBAL", price, sell_qty,
                        f"노출상한({self._gross_exposure_pct():.0%}) 리밸런싱 부분매도(비상승 종목)",
                        entry_price=pos.entry_price, signal_tag=pos.signal_tag,
                    )
                    if self.simulator.portfolio.get(ticker, 0) <= 0:
                        self.positions.pop(ticker, None)   # 전량 소진 시 포지션 정리
                    over -= sell_qty * price

        was_locked = self._buy_locked
        self._buy_locked = dd >= self.DD_BUY_LOCK
        if self._buy_locked and not was_locked:
            print(f"  🔒 낙폭 매수잠금 [{date_str}] 전고점 대비 -{dd*100:.1f}% → 신규 매수 중단(보유분 유지)")
        elif was_locked and not self._buy_locked:
            print(f"  🔓 매수잠금 해제 [{date_str}] 낙폭 회복 → 신규 매수 재개")

    # ── 확신도 점수 연산 ──────────────────────────────────────────────────────

    def _score_entry_nulim(self, df: pd.DataFrame, idx: int, max_spike_ratio: float) -> float:
        """DRY_VOLUME_NULIM: 과거 대금 폭발 강도 × 20일선 수렴도 기반 확신도 (50~100pt)"""
        base  = 50.0
        row   = df.iloc[idx]
        close = float(row["Close"])
        sma20 = float(row.get("SMA_20", np.nan))

        gap_pct   = abs(close - sma20) / sma20 * 100 if sma20 > 0 else 5.0
        gap_bonus = max(0.0, 20.0 - (gap_pct * 4.0))

        if max_spike_ratio >= 10.0:
            vol_bonus = 30.0
        elif max_spike_ratio >= 5.0:
            vol_bonus = 20.0
        else:
            vol_bonus = 10.0

        # [개편 2] 승률 높은 눌림목 전략 우선순위 가산 (추격 매매 대비 우대)
        return min(100.0, round(base + gap_bonus + vol_bonus + self.NULIM_PRIORITY_BONUS, 1))

    def _score_entry_golden(self, df: pd.DataFrame, idx: int, vol_ratio: float) -> float:
        """SMA_GOLDEN_CROSS: 거래대금 배율 × 당일 모멘텀 × 이격도 기반 확신도 (50~100pt)"""
        base     = 50.0
        row      = df.iloc[idx]
        prev_row = df.iloc[idx - 1]

        close      = float(row["Close"])
        prev_close = float(prev_row["Close"])
        sma5       = float(row.get("SMA_5",  np.nan))
        sma20      = float(row.get("SMA_20", np.nan))

        if vol_ratio >= 10.0:
            vol_bonus = 35.0
        elif vol_ratio >= 5.0:
            vol_bonus = 25.0
        else:
            vol_bonus = 15.0

        # 당일 상승 모멘텀 확인
        momentum_bonus = 10.0 if close > prev_close else 0.0

        # 5일선-20일선 이격도: 막 교차 시 보너스, 과열 시 패널티
        if sma20 > 0 and not np.isnan(sma5):
            gap_pct = (sma5 - sma20) / sma20 * 100
            if gap_pct > 15.0:
                gap_bonus = -15.0
            elif gap_pct > 5.0:
                gap_bonus = 0.0
            else:
                gap_bonus = 10.0
        else:
            gap_bonus = 0.0

        return min(100.0, round(base + vol_bonus + momentum_bonus + gap_bonus, 1))

    # ── 일일 스크리닝 (듀얼 모드 분기) ────────────────────────────────────────

    def _screen_daily(
        self, date: pd.Timestamp, ticker_data: dict[str, pd.DataFrame]
    ) -> dict[str, tuple[str, float]]:
        """장세에 따라 야수 모드 / 디펜스 모드 스크리너 자동 분기"""
        if self._is_bear_market(date):
            return self._screen_nulim(date, ticker_data)
        else:
            return self._screen_golden_cross(date, ticker_data)

    def _build_turnover_top(
        self,
        date: pd.Timestamp,
        ticker_data: dict[str, pd.DataFrame],
        use_today: bool = False,
    ) -> list[str]:
        """거래대금 기준 상위 N 종목 1차 필터.

        use_today=True: 오늘 거래대금 기준 (골든크로스 폭발 포착용)
        use_today=False: 전일 거래대금 기준 (눌림목 스크리닝 기본값)
        """
        turnover_list: list[tuple[str, float]] = []
        for ticker, df in ticker_data.items():
            if date not in df.index:
                continue
            idx_t = df.index.get_loc(date)
            if use_today:
                row = df.iloc[idx_t]
            else:
                if idx_t < 1:
                    continue
                row = df.iloc[idx_t - 1]
            turnover_list.append(
                (ticker, float(row.get("Close", 0)) * float(row.get("Volume", 0)))
            )
        turnover_list.sort(key=lambda x: x[1], reverse=True)
        return [t for t, _ in turnover_list[: self.volume_top_n]]

    def _screen_nulim(
        self, date: pd.Timestamp, ticker_data: dict[str, pd.DataFrame]
    ) -> dict[str, tuple[str, float]]:
        """[디펜스 모드] DRY_VOLUME_NULIM 거래량 급감 20일선 눌림목 스크리닝"""
        top_volume = self._build_turnover_top(date, ticker_data, use_today=False)
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

            # 조건 1: 오늘 주가가 20일선 부근 수렴 (-1% ~ +2.5%)
            if not (sma20 * 0.99 <= close <= sma20 * 1.025):
                continue

            # 조건 2: 오늘 거래량이 20일 평균의 70% 이하로 급감
            avg_vol_20 = df["Volume"].iloc[max(0, idx - 19): idx + 1].mean()
            if vol > avg_vol_20 * 0.7:
                continue

            # 조건 3: 최근 10일 내 평균 대비 5배+ 거래대금 폭발 흔적 탐색
            has_past_spike  = False
            max_spike_ratio = 1.0

            for lb in range(max(0, idx - 10), idx):
                p_row      = df.iloc[lb]
                p_turnover = float(p_row["Close"]) * float(p_row["Volume"])
                p_avg      = (
                    df["Volume"].iloc[max(0, lb - 19): lb + 1]
                    * df["Close"].iloc[max(0, lb - 19): lb + 1]
                ).mean()
                ratio = p_turnover / p_avg if p_avg > 0 else 0.0
                if ratio >= 5.0:
                    has_past_spike  = True
                    max_spike_ratio = max(max_spike_ratio, ratio)

            if has_past_spike:
                score = self._score_entry_nulim(df, idx, max_spike_ratio)
                candidates[ticker] = ("DRY_VOLUME_NULIM", score)

        sorted_c = sorted(candidates.items(), key=lambda x: x[1][1], reverse=True)
        return dict(sorted_c[: self.news_candidate_n])

    def _screen_golden_cross(
        self, date: pd.Timestamp, ticker_data: dict[str, pd.DataFrame]
    ) -> dict[str, tuple[str, float]]:
        """[야수 모드] 정배열 주도주 거래대금 폭발 + 5일선 위 돌파 흐름 포착"""
        top_volume = self._build_turnover_top(date, ticker_data, use_today=True)
        candidates: dict[str, tuple[str, float]] = {}

        for ticker in top_volume:
            df = ticker_data[ticker]
            if date not in df.index:
                continue
            idx = df.index.get_loc(date)
            if idx < 25:
                continue

            row      = df.iloc[idx]
            prev_row = df.iloc[idx - 1]

            close      = float(row["Close"])
            vol        = float(row["Volume"])
            sma5       = float(row.get("SMA_5",  np.nan))
            sma20      = float(row.get("SMA_20", np.nan))
            prev_close = float(prev_row["Close"])

            if any(np.isnan(v) for v in [sma5, sma20]):
                continue

            # 조건 1: 5일선 > 20일선 정배열 우상향 상태 유지
            if sma5 <= sma20:
                continue

            # 조건 2: 당일 거래대금이 최근 20일 평균 대비 2.5배 이상 강력 폭발
            today_turnover  = close * vol
            avg_turnover_20 = (
                df["Volume"].iloc[max(0, idx - 19): idx + 1]
                * df["Close"].iloc[max(0, idx - 19): idx + 1]
            ).mean()
            if avg_turnover_20 <= 0:
                continue
            vol_ratio = today_turnover / avg_turnover_20
            if vol_ratio < 2.5:
                continue

            # 조건 3: 주가가 전일 종가 및 5일 이동평균선보다 위에 있는 강력한 돌파 흐름
            if close < prev_close or close < sma5:
                continue

            score = self._score_entry_golden(df, idx, vol_ratio)
            candidates[ticker] = ("SMA_GOLDEN_CROSS", score)

        sorted_c = sorted(candidates.items(), key=lambda x: x[1][1], reverse=True)
        return dict(sorted_c[: max(self.news_candidate_n, self.MAX_POS_BULL * 2)])

    # ── [개편 1] 리스크 관리 모듈: 손절가 산정 + 고정 리스크 포지션 사이징 ────

    def _resolve_stop_loss(self, price: float, atr: float, signal_tag: str) -> float:
        """진입가·ATR·전략별 손절가 산정 (포지션 사이징/청산 공통 기준).

        [v2-4] 노이즈 칼손절 방지를 위해 ATR 버퍼를 상향(2.5→3.0×ATR).
        - SMA_GOLDEN_CROSS: max(진입가−3.0×ATR, 진입가×0.88) = 더 타이트한 쪽(하한 -12%)
        - DRY_VOLUME_NULIM 등: 진입가−3.0×ATR
        - ATR 미산출 시: 진입가 × SL_FALLBACK_PCT(-10%)
        """
        if signal_tag == "SMA_GOLDEN_CROSS":
            if atr > 0:
                return max(price - self.ATR_SL_MULT_BULL * atr, price * self.BULL_SL_FLOOR_PCT)
            return price * self.SL_FALLBACK_PCT
        if atr > 0:
            return price - self.ATR_SL_MULT_BEAR * atr
        return price * self.SL_FALLBACK_PCT

    def _calc_position_qty(
        self, price: float, stop_loss: float, total_asset: float, cash: float,
        max_pos_pct: float | None = None,
    ) -> int:
        """ATR 손절 기반 고정 리스크 배정 — 자산 증가 시 몰빵 차단의 핵심.

        수량 = (총자산 × MAX_RISK_PER_TRADE) / (진입가 − 손절가)
        → 손절선 청산 시 손실이 항상 '총자산의 2%'로 고정.
        추가 안전캡: ① 단일 종목 평가액 ≤ 총자산 × max_pos_pct(모드별, 기본 MAX_POSITION_PCT)
                     ② 매수금액 ≤ 가용현금 × 95%
        """
        risk_per_share = price - stop_loss
        if risk_per_share <= 0 or price <= 0:
            return 0
        pos_pct = max_pos_pct if max_pos_pct is not None else self.MAX_POSITION_PCT
        risk_budget = total_asset * self.MAX_RISK_PER_TRADE
        qty = int(risk_budget / risk_per_share)               # 고정 리스크 수량
        qty = min(qty, int((total_asset * pos_pct) / price))  # 비중 상한 캡(모드별)
        qty = min(qty, int((cash * 0.95) / price))            # 현금 하드캡
        return max(qty, 0)

    # ── 매매 집행 ─────────────────────────────────────────────────────────────

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
        close = float(row["Close"])
        high  = float(row.get("High", close))
        low   = float(row.get("Low",  close))
        open_ = float(row.get("Open", close))
        atr   = float(row.get("ATR", np.nan))
        if np.isnan(atr) or atr <= 0:
            atr = -1.0

        # [v4.6-2] 트레일링 스톱 활성화 — 수익률 +18%(TRAIL_ACTIVATE_PROFIT) 최초 도달 시 ON(큰 추세 추종).
        # 활성화 전에는 고정 익절(TP)·ATR 손절(SL)만으로 판단 → 흔들림 조기 약손실 청산 차단.
        # 활성화되는 순간 손절선 자체를 '본전 +0.5%(수수료 보전선)'으로 즉시 상향한다 →
        # 10%대 수익을 줬던 종목이 마이너스 손절로 돌변하는 경로를 ATR 유무와 무관하게 원천 차단.
        # 슬리피지(SL_SLIPPAGE_PCT) 차감 후에도 체결가 ≥ 본전+0.5%가 되도록 손절선을 역보정.
        breakeven_floor = (
            pos.entry_price * (1.0 + self.TRAIL_BREAKEVEN_PCT) / (1.0 - self.SL_SLIPPAGE_PCT)
        )
        if not pos.trailing_active and high >= pos.entry_price * (1.0 + self.TRAIL_ACTIVATE_PROFIT):
            pos.trailing_active = True
            pos.stop_loss       = max(pos.stop_loss, breakeven_floor)  # 손절선 본전+0.5%로 즉시 상향

        # 트레일링은 '직전까지의 최고가' 기준으로 산정 (당일 고점-손절 동시 발생 방지)
        trailing_sl  = (
            self.strategy.get_trailing_stop(pos.peak_price, atr)
            if (pos.trailing_active and atr > 0) else -1.0
        )
        # 트레일링 손절선도 본전+0.5% 아래로는 내려가지 않게 보정(이익 보호 전용 — 마이너스 청산 불가).
        if trailing_sl > 0:
            trailing_sl = max(trailing_sl, breakeven_floor)
        hard_sl      = pos.stop_loss if pos.stop_loss > 0 else float("-inf")
        ts_val       = trailing_sl   if trailing_sl    > 0 else float("-inf")
        effective_sl = max(hard_sl, ts_val)

        # ── [개편 4] 손절/트레일링 우선 체크 — 보수적 체결 ─────────────────────
        # 일봉 저가(Low)가 손절선을 터치했다면 종가가 아니라 '손절선 -1% 슬리피지'로
        # 당일 즉시 청산. 갭하락으로 시초가가 손절선보다 더 낮으면 시초가로 체결.
        #
        # [v4.6-6] 진입 초기 휩소 방어: 진입 후 T+1(2거래일) 이내(bars_held ≤ 1)에는
        # 장중 저가(Low) 흔들림으로 손절선이 터지는 것을 유예하고 '종가(Close)'로만 손절 판정.
        # → 트리거 기준가를 윈도 내에서는 low 대신 close 로 사용하고, 체결도 종가로 한다.
        bars_held        = idx - pos.entry_idx if pos.entry_idx >= 0 else self.WHIPSAW_GUARD_BARS + 1
        in_whipsaw_guard = 0 <= bars_held <= self.WHIPSAW_GUARD_BARS
        sl_trigger       = close if in_whipsaw_guard else low
        if effective_sl > float("-inf") and sl_trigger <= effective_sl:
            if in_whipsaw_guard:
                fill = close                          # 휩소 가드: 종가 기준 청산
            else:
                fill = effective_sl * (1.0 - self.SL_SLIPPAGE_PCT)
                if open_ < effective_sl:              # 갭하락: 시초가가 손절선 하회
                    fill = min(fill, open_)
                fill = max(fill, low)                 # 당일 최저가 밑으로는 체결 불가
            # [v4] 트레일링 활성화(손절선이 본전 위로 상향됨) 후 끊긴 청산은 본전 이상에서
            # 체결되므로 이익보호(SELL_TS)로 분류. 그 외는 진짜 손절(SELL_SL).
            action = "SELL_TS" if (pos.trailing_active and fill >= pos.entry_price) else "SELL_SL"

            # [v4.6-5] 계좌 서킷브레이커: 당일 진짜 손절(SELL_SL)은 MAX_DAILY_SL(1)종목까지만.
            # 이미 한도 도달이면 타 종목의 손절 집행을 당일 보류(보유 지속, 익일 종가 재판정).
            # 이익보호(SELL_TS)는 제한 대상이 아니다.
            if action == "SELL_SL" and self._daily_sl_count >= self.MAX_DAILY_SL:
                self._daily_sl_blocked_count += 1
                pos.peak_price = max(pos.peak_price, high)   # 트레일링 추적용 peak 만 갱신
                return

            qty = self.simulator.portfolio.get(ticker, 0)
            if qty and self.simulator.execute_sell(ticker, fill, qty):
                pnl_pct    = (fill - pos.entry_price) / pos.entry_price * 100
                reason_txt = (
                    f"트레일링스톱 작동 (선: {effective_sl:,.0f})"
                    if action == "SELL_TS"
                    else (f"[휩소가드 종가청산] 손절 ({pos.stop_loss:,.0f})" if in_whipsaw_guard
                          else f"손절선 이탈 ({pos.stop_loss:,.0f})")
                )
                self._log_trade(
                    date_str, ticker, action, fill, qty,
                    f"{reason_txt} [체결 {fill:,.0f} / 손익: {pnl_pct:+.1f}%]",
                    entry_price=pos.entry_price, signal_tag=pos.signal_tag,
                )
                # [v4] 진짜 손절(SELL_SL)만 재진입 쿨다운 등록 + [v4.6-5] 당일 손절 카운트 증가
                if action == "SELL_SL":
                    self._sl_block[ticker] = pd.Timestamp(date_str)
                    self._daily_sl_count += 1
                del self.positions[ticker]
            return

        # ── 고정 익절선 도달 (고가 기준, 목표가 체결) ─────────────────────────
        if pos.take_profit > 0 and high >= pos.take_profit:
            fill = pos.take_profit
            if open_ > pos.take_profit:        # 갭상승: 시초가가 목표가 상회
                fill = open_
            fill = min(fill, high)
            qty  = self.simulator.portfolio.get(ticker, 0)
            if qty and self.simulator.execute_sell(ticker, fill, qty):
                pnl_pct = (fill - pos.entry_price) / pos.entry_price * 100
                self._log_trade(
                    date_str, ticker, "SELL_TP", fill, qty,
                    f"익절 달성 (목표: {pos.take_profit:,.0f}, 손익: {pnl_pct:+.1f}%)",
                    entry_price=pos.entry_price, signal_tag=pos.signal_tag,
                )
                del self.positions[ticker]
            return

        # 미청산 — 당일 고가를 peak 에 반영 (다음 봉 트레일링 추적용)
        pos.peak_price = max(pos.peak_price, high)

        # ── 분할 매수(물타기): DRY_VOLUME_NULIM 전용, 최대 MAX_REBUY_COUNT회 ───
        # GOLDEN_CROSS 추격 매매는 물타기 금지 (달리는 말에서 내리는 전략)
        if pos.signal_tag == "SMA_GOLDEN_CROSS":
            return

        rebuy_ready = (
            pos.rebuy_count < self.MAX_REBUY_COUNT
            and self.strategy.is_rebuy_signal(close, pos.entry_price, pos.rebuy_count)
        )

        # [v4.6-4] 디펜스(하락장) 모드에서는 물타기(REBUY) 100% 비활성화 — 하락장 추가매수가
        # 반등 실패 후 손절될 때 비중 누적 타격이 크다. 실제 물타기 신호가 떴는데 차단된 경우만
        # _defense_rebuy_block_count 로 집계해 성적표에 표기한다(당일 확정 모드 _current_is_bear 참조).
        if self._current_is_bear:
            if rebuy_ready:
                self._defense_rebuy_block_count += 1
            return

        if rebuy_ready:
            # [개편 1 + v3-hard] 물타기도 ① 종목 비중 상한(MAX_POSITION_PCT)과
            # ② 총노출 상한(MAX_GROSS_EXPOSURE)을 동시에 만족해야 한다. 둘 중 더 빡빡한
            # 여력만큼만 추가 매수 → 물타기로 총노출이 45%를 넘던 누수 경로 차단.
            total_asset = self.simulator.get_total_asset(current_prices)
            held_qty    = self.simulator.portfolio.get(ticker, 0)
            cur_value   = held_qty * close
            pos_room    = total_asset * self.MAX_POSITION_PCT - cur_value  # 단일종목 15% 여력
            gross_room  = self._exposure_room(current_prices)             # 총노출 45% 여력
            room        = min(pos_room, gross_room)
            if room < close:
                return  # 종목/총노출 상한 도달 — 추가 매수 불가
            alloc_cash = min(room, self.simulator.cash * 0.9)
            add_qty    = int(alloc_cash / close)
            if add_qty > 0 and self.simulator.execute_buy(ticker, close, add_qty):
                pos.rebuy_count += 1
                self._log_trade(
                    date_str, ticker, f"REBUY#{pos.rebuy_count}", close, add_qty,
                    f"눌림목 분할 추가 물타기 [배정 {alloc_cash:,.0f}원]",
                    signal_tag=pos.signal_tag,
                )

    def _try_buy(
        self,
        date_str: str,
        ticker: str,
        df: pd.DataFrame,
        idx: int,
        current_prices: dict[str, float],
        signal_tag: str = "UNKNOWN",
        score: float = 50.0,
        max_positions: int = 3,
    ) -> None:
        row   = df.iloc[idx]
        price = float(row["Close"])
        atr   = float(row.get("ATR", np.nan))
        if np.isnan(atr) or atr <= 0:
            atr = -1.0

        # [v5.5-4] 디펜스(하락장) 모드 신규 매수 정책:
        #   · ALLOW_DEFENSE_NULIM_BUY=False(안전형 v4.6) → 전면 차단.
        #   · ALLOW_DEFENSE_NULIM_BUY=True(위험형 v5.5) → 눌림목(DRY_VOLUME_NULIM)만 10% 캡으로 허용.
        if self._current_is_bear and not self.ALLOW_DEFENSE_NULIM_BUY:
            return

        # [안전장치 1] 동적 보유 수 상한 제어
        if len(self.positions) >= max_positions:
            return

        # [안전장치 2] 최저가 가용잔고 검증
        if self.simulator.cash < price:
            return

        # [v4] 손절 직후 재진입 쿨다운 — 같은 종목을 SELL_SL 후 N일 내 재매수 금지(뇌동매매 차단)
        last_sl = self._sl_block.get(ticker)
        if last_sl is not None and (pd.Timestamp(date_str) - last_sl).days < self.REENTRY_COOLDOWN_DAYS:
            return

        # 확신도 품질 필터 (배정 비중이 아니라 '진입 자격' 게이트 — 55pt 미만 컷)
        if score < 55:
            return

        # ── [개편 2] RSI 하드 필터 — 과열 구간 신규 진입 전면 금지(Lock) ────────
        # 기존 'RSI>75 비중 절반' 로직 완전 폐기. 종목 자신의 RSI_14 ≥ 70 이면
        # 상투 추격을 원천 차단(진입 자체 금지).
        stock_rsi = float(row.get("RSI_14", np.nan))
        if not np.isnan(stock_rsi) and stock_rsi >= self.RSI_HARD_LIMIT:
            return
        # 야수 모드(추격 돌파)는 시장 지수 과열도 추가 차단
        if signal_tag == "SMA_GOLDEN_CROSS":
            if self._get_market_rsi(pd.Timestamp(date_str)) >= self.RSI_HARD_LIMIT:
                return

        # ── [v5-1] 지수 5일선 Top-Down 필터 — 시장 리스크 방어 ──────────────────
        # 개별 종목 조건이 충족돼도, 소속 지수(KOSPI/KOSDAQ) 종가가 5일선 아래면 진입 금지.
        market = self._market_of(ticker)
        if market is not None and self._index_below_sma5(market, pd.Timestamp(date_str)):
            self._index_filter_pass_count += 1
            print(
                f"  🚫 [PASS] 지수 5일선 하회로 인한 매수 잠금 (시장 리스크 방어) "
                f"[{date_str}] {self._ticker_names.get(ticker, ticker)} ({market})"
            )
            return

        # ── [개편 1] 손절가 먼저 산정 → ATR 기반 고정 리스크 포지션 사이징 ──────
        sl = self._resolve_stop_loss(price, atr, signal_tag)
        if sl <= 0 or sl >= price:
            return

        # 단일종목 캡: 야수(SMA_GOLDEN_CROSS) 33% / 디펜스 눌림목(낙주매수) 10% / 그 외 15%.
        if signal_tag == "SMA_GOLDEN_CROSS":
            pos_pct = self.MAX_POSITION_PCT_BULL
        elif self._current_is_bear:
            pos_pct = self.DEFENSE_NULIM_POS_PCT          # [v5.5-4] 디펜스 눌림목 10% 캡
        else:
            pos_pct = self.MAX_POSITION_PCT
        total_asset = self.simulator.get_total_asset(current_prices)
        qty = self._calc_position_qty(price, sl, total_asset, self.simulator.cash, pos_pct)
        if qty <= 0:
            return

        # [v3-hard] 노출 상한 하드 게이트 — [현재 주식평가액 + 신규매수금액]이
        # [총자산 × MAX_GROSS_EXPOSURE]를 1원이라도 초과하면 주문을 전면 차단한다.
        # 라이브 총자산 기준으로 매 체결 직전 재계산하므로, 같은 날 신호가 동시에 떠도
        # 누적 노출이 상한을 넘는 일이 없다(체결 → 다음 호출 시 room 자동 감소).
        room = self._exposure_room(current_prices)
        if room < price:
            return                       # 노출 여력 없음 → 신규 매수 전면 차단
        qty = min(qty, int(room / price))
        if qty <= 0:
            return

        if not self.simulator.execute_buy(ticker, price, qty):
            return

        # 익절: 야수 모드 +25% 고정 / 그 외 ATR×3.5
        if signal_tag == "SMA_GOLDEN_CROSS":
            tp = price * 1.25
        else:
            tp = self.strategy.get_exit_price(price, atr, is_profit=True) if atr > 0 else -1.0

        pos                = PositionState()
        pos.in_position    = True
        pos.entry_price    = price
        pos.peak_price     = price
        pos.rebuy_count    = 0
        pos.take_profit    = tp
        pos.stop_loss      = sl
        pos.signal_tag     = signal_tag
        pos.entry_idx      = idx          # [v4.6-6] 휩소 가드 보유일수 계산 기준점
        # 실제 투입 비중(%) 기록 — 배정 로직이 아니라 사후 참고/물타기 캡 계산용
        pos.allocated_pct  = (price * qty) / total_asset if total_asset > 0 else 0.0
        self.positions[ticker] = pos

        invest_amt = price * qty
        risk_amt   = (price - sl) * qty
        risk_pct   = risk_amt / total_asset * 100 if total_asset > 0 else 0.0
        mode_label = "야수" if signal_tag == "SMA_GOLDEN_CROSS" else "디펜스"
        self._log_trade(
            date_str, ticker, "BUY", price, qty,
            f"[{mode_label}] {score:.1f}pt | 투입 {invest_amt:,.0f}원({pos.allocated_pct:.1%}) "
            f"| 손절 {sl:,.0f} | 리스크 {risk_amt:,.0f}원(총자산 {risk_pct:.2f}%)",
            signal_tag=signal_tag,
        )

    def _log_trade(
        self,
        date: str,
        ticker: str,
        action: str,
        price: float,
        qty: int,
        reason: str,
        entry_price: float = 0.0,
        signal_tag: str = "",
    ) -> None:
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
        icon = {
            "BUY": "🛒", "SELL_TP": "💵✅", "SELL_SL": "💵❌", "SELL_TS": "🎯✅",
            "SELL_REBAL": "♻️",
        }.get(action, "↩️")
        print(
            f"  {icon} {date} [{self._ticker_names.get(ticker, ticker)}] {action:10s} "
            f"가격 {price:>10,.0f} 수량 {qty:>5} ({reason})"
        )

    # ── 메인 루프 ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        print("\n" + "=" * 70)
        print("  백테스팅 통합 엔진 — 듀얼 모드 (야수 + 디펜스)")
        print("=" * 70)

        ticker_data = self._load_data()
        self._load_benchmark_indices()
        if not ticker_data:
            return

        all_dates: set[pd.Timestamp] = set()
        for df in ticker_data.values():
            all_dates.update(
                df.index[(df.index >= self.start_date) & (df.index <= self.end_date)]
            )
        trading_dates = sorted(all_dates)

        # 추가 입금일 → 실제 거래일 매핑
        deposit_on: dict[str, float] = {}
        for dep_date_str, amount in self.deposit_schedule.items():
            dep_ts = pd.Timestamp(dep_date_str)
            target = next((d for d in trading_dates if d >= dep_ts), None)
            if target:
                deposit_on[target.strftime("%Y-%m-%d")] = amount

        for date in trading_dates:
            date_str = date.strftime("%Y-%m-%d")
            # [v5] 당일 모드를 하루 1회 확정 — 노출상한·캡·물타기가 같은 모드를 일관 참조.
            self._current_is_bear = self._is_bear_market(date)
            # [v4.6-5] 계좌 서킷브레이커: 당일 손절 종목 카운터를 매일 0으로 리셋.
            self._daily_sl_count = 0
            # [v2] 시가평가 보정 — 당일 바 없는 보유종목은 직전가로 평가 (허수 폭락 방지)
            current_prices = self._mark_to_market(date, ticker_data)

            # 추가 입금 처리
            if date_str in deposit_on:
                self.simulator.deposit(deposit_on[date_str], current_prices)

            # 동적 최대 보유 종목 수 결정 (1회 호출로 일관성 유지)
            max_pos = self._get_max_positions(date)

            # 매도 정산 우선 (종목별 TP/SL/트레일링)
            for held_ticker in list(self.positions.keys()):
                df = ticker_data.get(held_ticker)
                if df is not None and date in df.index:
                    self._try_sell(
                        date_str, held_ticker, df,
                        df.index.get_loc(date), current_prices,
                    )

            # [v3] 매도 정산 후 위험 상태 갱신 (전고점·노출상한 리밸런싱·매수잠금)
            # [v5-3] ticker_data 전달 → 당일 상승 종목 리밸런싱 유예 판정에 사용
            total_asset = self.simulator.get_total_asset(current_prices)
            self._update_risk_state(date, date_str, total_asset, current_prices, ticker_data)

            # 스크리닝 및 매수 집행 (낙폭 매수잠금 아닐 때만, 노출은 모드별 상한 내에서)
            # [v5.5-4] 디펜스 모드에서도 ALLOW_DEFENSE_NULIM_BUY=True 면 눌림목 매수를 허용한다
            # (실제 종목 캡/차단은 _try_buy 에서 모드·플래그로 최종 판정).
            if not self._buy_locked and (not self._current_is_bear or self.ALLOW_DEFENSE_NULIM_BUY):
                candidates = self._screen_daily(date, ticker_data)
                for buy_ticker, (signal_tag, score) in candidates.items():
                    if buy_ticker in self.positions:
                        continue
                    if len(self.positions) >= max_pos:
                        break
                    if self.simulator.cash <= 0:
                        break
                    df = ticker_data.get(buy_ticker)
                    if df is not None and date in df.index:
                        self._try_buy(
                            date_str, buy_ticker, df,
                            df.index.get_loc(date), current_prices,
                            signal_tag, score, max_pos,
                        )

            # 자산 곡선 기록 (mode 컬럼 — 매수잠금 여부 병기) — [v5] 당일 확정 모드 사용
            bear      = self._current_is_bear
            base_mode = "BEAR" if bear else "BULL"
            lock_suffix = "_BUYLOCK" if self._buy_locked else ""
            total_asset = self.simulator.get_total_asset(current_prices)
            self.equity_curve.append({
                "date":        date_str,
                "total_asset": round(total_asset),
                "return_pct":  round(self.simulator.get_current_return(current_prices), 4),
                "cash":        round(self.simulator.cash),
                "invested":    round(self.simulator.invested_capital),
                "mode":        f"{base_mode}{lock_suffix}",
            })

        self._report()

    # ── 성적 리포트 ───────────────────────────────────────────────────────────

    def _report(self) -> None:
        if not self.equity_curve:
            return
        ec   = self.equity_curve
        last = ec[-1]

        # ── [Step 1] MDD: 표준 cummax 낙폭 공식(글리치 방어 포함) ──────────────
        # compute_mdd 는 NaN·inf·0이하 자산점을 제거 후 (asset−peak)/peak 의 최솟값을
        # 계산한다. 일시적 데이터 튐(허수 피크/트로프)에 MDD가 폭발하지 않는다.
        # 표기 일관성을 위해 양수 크기로 보관(예: 18.46 → "-18.46%" 출력).
        mdd = abs(compute_mdd([r["total_asset"] for r in ec]))
        # ────────────────────────────────────────────────────────────────────────

        n_years = (
            (pd.Timestamp(ec[-1]["date"]) - pd.Timestamp(ec[0]["date"])).days
        ) / 365.25
        cagr = (
            (last["total_asset"] / ec[0]["invested"]) ** (1 / n_years) - 1
        ) * 100 if n_years > 0 else 0.0

        sell_actions = ("SELL_TP", "SELL_TS", "SELL_SL")
        sells    = [t for t in self.trade_log if t["action"] in sell_actions]
        wins     = [t for t in sells if (t["price"] - t["entry_price"]) > 0]
        win_rate = len(wins) / len(sells) * 100 if sells else 0.0

        bull_days    = sum(1 for r in ec if r.get("mode", "").startswith("BULL"))
        bear_days    = sum(1 for r in ec if r.get("mode", "").startswith("BEAR"))
        buylock_days = sum(1 for r in ec if "_BUYLOCK" in r.get("mode", ""))
        total_days   = len(ec)
        sl_count     = sum(1 for t in self.trade_log if t["action"] == "SELL_SL")
        ts_count     = sum(1 for t in self.trade_log if t["action"] == "SELL_TS")
        tp_count     = sum(1 for t in self.trade_log if t["action"] == "SELL_TP")
        rebal_count  = sum(1 for t in self.trade_log if t["action"] == "SELL_REBAL")
        idx_pass     = self._index_filter_pass_count   # [v5-1] 지수 5일선 하회 매수 패스 횟수
        reb_block    = self._defense_rebuy_block_count # [v4.6-4] 디펜스 모드 물타기 차단 횟수
        sl_cb_block  = self._daily_sl_blocked_count    # [v4.6-5] 당일 손절한도 초과로 보류된 횟수

        mdd_flag = "✅ 목표(-20%) 이내" if mdd <= 20.0 else "⚠️ 목표(-20%) 초과"
        # [v5] 승률 목표 40% 방어 달성 여부 직관 표기
        wr_flag  = "✅ 목표(40%) 달성" if win_rate >= 40.0 else "⚠️ 목표(40%) 미달"

        nulim_label = f"디펜스눌림목허용(캡{self.DEFENSE_NULIM_POS_PCT:.0%})" if self.ALLOW_DEFENSE_NULIM_BUY else "디펜스물타기금지"
        print("\n" + "=" * 70)
        print(f"  [최종] 백테스트 성적표 v5.5 위험감수형 — 야수커트{self.MARKET_SCORE_BULL_CUTLINE:.0f}pt + "
              f"종목캡{self.MAX_POSITION_PCT_BULL:.0%} + 본전보호+{self.TRAIL_ACTIVATE_PROFIT:.0%} + "
              f"{nulim_label} + 당일손절{self.MAX_DAILY_SL}종목 + 휩소가드T+{self.WHIPSAW_GUARD_BARS}")
        print(f"         (검증기간 {ec[0]['date']} ~ {ec[-1]['date']})")
        print("=" * 70)
        print(f"  총 투자 원금       : {last['invested']:>15,.0f} 원")
        print(f"  최종 자산          : {last['total_asset']:>15,.0f} 원")
        print(f"  현금 잔고          : {last['cash']:>15,.0f} 원")
        print(f"  펀드 수익률        : {last['return_pct']:>+14.2f} %")
        print(f"  CAGR               : {cagr:>+14.2f} %")
        print(f"  최대 낙폭(MDD)     : {-mdd:>+14.2f} %  {mdd_flag}")
        print(f"  승률               : {win_rate:>14.1f} %  {wr_flag}  (총 청산 {len(sells)}회)")
        print("  " + "-" * 66)
        print(f"  🚫 지수필터 매수패스: {idx_pass:>5}회  (지수 5일선 하회 → 시장 리스크 방어)")
        print(f"  ⛔ 디펜스 물타기차단 : {reb_block:>5}회  (디펜스 모드 중 REBUY 신호 → 100% 비활성화)")
        print(f"  🧯 당일손절한도 보류 : {sl_cb_block:>5}회  (계좌 서킷브레이커 — 당일 {self.MAX_DAILY_SL}종목 초과 손절 유예)")
        print(f"  🛡️ 디펜스 모드 작동 : {bear_days:>5}일 ({bear_days / total_days * 100:.1f}%)  "
              f"[노출캡 {self.MAX_GROSS_EXPOSURE_BEAR:.0%}·현금강제확보·신규매수금지]")
        print(f"  ⚔️ 야수 모드 기간   : {bull_days:>5}일 ({bull_days / total_days * 100:.1f}%)  "
              f"[노출캡 {self.MAX_GROSS_EXPOSURE_BULL:.0%}]")
        print("  " + "-" * 66)
        print(f"  청산 내역          : 익절 {tp_count}회 / 트레일링 {ts_count}회 / 손절 {sl_count}회")
        print(f"  리밸런싱 부분매도   : {rebal_count}회 (비상승 종목만 / 상승종목은 홀딩 유예)")
        print(f"  낙폭 매수잠금       : {buylock_days}일 (신규매수 중단 일수)")
        print("=" * 70)
        self._print_signal_report()

    def _print_signal_report(self) -> None:
        from collections import defaultdict
        sells = [t for t in self.trade_log if t["action"] in ("SELL_TP", "SELL_TS", "SELL_SL")]
        if not sells:
            return
        groups: dict[str, list] = defaultdict(list)
        for t in sells:
            groups[t.get("signal_tag", "UNKNOWN")].append(t)

        print("\n  ▶ 로직별 성적 상세 리포트")
        for tag, trades in groups.items():
            wins    = [t for t in trades if (t["price"] - t["entry_price"]) > 0]
            avg_pct = (
                sum((t["price"] - t["entry_price"]) / t["entry_price"] * 100 for t in trades)
                / len(trades)
            )
            print(
                f"     [{tag}] 청산: {len(trades)}회 | "
                f"승률: {len(wins) / len(trades) * 100:.1f}% | "
                f"평균손익률: {avg_pct:+.2f}%"
            )


# ══════════════════════════════════════════════════════════════════════════════
# [UI 완전 복원] StockScreener 클래스
# ══════════════════════════════════════════════════════════════════════════════

class StockScreener:
    MARKET_CFG = {
        "KOSPI":  ("get_top_kospi_stocks",  50_000),
        "KOSDAQ": ("get_top_kosdaq_stocks", 30_000),
        "S&P500": ("get_top_sp500_stocks",  100_000),
        "NASDAQ": ("get_top_nasdaq_stocks", 150_000),
    }

    def __init__(self, universe_per_market: int = 200):
        self.universe_per_market = universe_per_market

    def build_universe(self, markets: list[str]) -> list[tuple[str, str]]:
        import stock_ai
        seen:   set[str] = set()
        result: list[tuple[str, str]] = []
        for mkt in markets:
            cfg = self.MARKET_CFG.get(mkt)
            if not cfg:
                continue
            fn = getattr(stock_ai, cfg[0], None)
            if not fn:
                continue
            for name, ticker in fn(self.universe_per_market).items():
                if ticker not in seen:
                    seen.add(ticker)
                    result.append((ticker, name))
        return result

    def _score_chunk(
        self, chunk: list[tuple[str, str]], lookback: int, min_volume: float
    ) -> list[dict]:
        """UI 실시간 스크리닝 게이지용 청크별 병렬 스코어링.

        DRY_VOLUME_NULIM 과 SMA_GOLDEN_CROSS 두 패턴 모두 탐지.
        """
        scored_list: list[dict] = []
        tickers = [t for t, _ in chunk]
        try:
            raw = yf.download(tickers, period=f"{lookback}d", auto_adjust=True, progress=False)
        except Exception:
            return []

        for ticker, name in chunk:
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    df = raw.xs(ticker, axis=1, level=1).dropna(how="all")
                else:
                    df = raw.dropna(how="all")

                if df.empty or len(df) < 20:
                    continue
                df = _flatten_columns(df)
                df = _add_indicators(df)
                if "SMA_5" not in df.columns:
                    df["SMA_5"] = df["Close"].rolling(5).mean()

                idx   = len(df) - 1
                row   = df.iloc[idx]
                close = float(row["Close"])
                vol   = float(row["Volume"])
                sma20 = float(row.get("SMA_20", np.nan))

                if np.isnan(sma20) or vol < min_volume:
                    continue

                avg_vol = df["Volume"].iloc[max(0, idx - 19): idx + 1].mean()

                # ① DRY_VOLUME_NULIM 패턴
                if (sma20 * 0.99) <= close <= (sma20 * 1.03) and vol <= avg_vol * 0.8:
                    score = 60.0 + (10.0 * (avg_vol / (vol if vol > 0 else 1)))
                    scored_list.append({
                        "ticker": ticker,
                        "name":   name,
                        "score":  min(100.0, round(score, 1)),
                        "close":  close,
                        "volume": vol,
                        "signal": "DRY_VOLUME_NULIM",
                    })

                # ② SMA_GOLDEN_CROSS 패턴 (DRY 조건 미충족 시 검사)
                elif idx >= 1:
                    prev_row   = df.iloc[idx - 1]
                    sma5       = float(row.get("SMA_5",  np.nan))
                    prev_sma5  = float(prev_row.get("SMA_5",  np.nan))
                    prev_sma20 = float(prev_row.get("SMA_20", np.nan))
                    today_tv   = close * vol
                    avg_tv     = (
                        df["Volume"].iloc[max(0, idx - 19): idx + 1]
                        * df["Close"].iloc[max(0, idx - 19): idx + 1]
                    ).mean()

                    if (
                        not any(np.isnan(v) for v in [sma5, prev_sma5, prev_sma20])
                        and sma5 > sma20
                        and prev_sma5 <= prev_sma20
                        and avg_tv > 0
                        and today_tv >= avg_tv * 3.0
                    ):
                        vol_ratio = today_tv / avg_tv
                        score = min(100.0, 50.0 + min(35.0, vol_ratio * 5.0) + 15.0)
                        scored_list.append({
                            "ticker": ticker,
                            "name":   name,
                            "score":  round(score, 1),
                            "close":  close,
                            "volume": vol,
                            "signal": "SMA_GOLDEN_CROSS",
                        })

            except Exception:
                continue

        return scored_list

    def screen(
        self,
        markets: list[str],
        top_n: int,
        lookback: int = 60,
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> list[dict]:
        """대시보드에서 단독으로 호출하는 실시간 스크리닝 진입점"""
        if progress_cb:
            progress_cb(0.0, "전체 시장 유니버스 구성 수집 중...")
        universe = self.build_universe(markets)

        default_min_vol = 5000
        all_scored: list[dict] = []
        CHUNK = 30
        total = len(universe)

        if progress_cb:
            progress_cb(5.0, f"총 {total}개 종목 분할 스크리닝 기동...")

        for i in range(0, total, CHUNK):
            chunk = universe[i: i + CHUNK]
            all_scored.extend(self._score_chunk(chunk, lookback, default_min_vol))
            if progress_cb:
                pct = 5.0 + ((i + len(chunk)) / total * 92.0)
                progress_cb(pct, f"진행 중 ({i + len(chunk)}/{total} 완료)")

        all_scored.sort(key=lambda x: x["score"], reverse=True)
        selected = all_scored[:top_n]

        if progress_cb:
            progress_cb(100.0, f"스크리닝 완료 — 상위 {len(selected)}개 선정")
        return selected


# ══════════════════════════════════════════════════════════════════════════════
# 실행 진입점
# ══════════════════════════════════════════════════════════════════════════════

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
