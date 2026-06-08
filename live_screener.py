"""
live_screener.py — 실전 라이브 스크리너 & 잔고 관리 시스템

backtest.py 핵심 로직 완전 계승:
  ▸ 시장 지수 판별: KOSPI·KOSDAQ SMA_60 기준 야수/디펜스 모드 자동 분기
  ▸ 지수 RSI_14 > 75 과열 브레이크 → 베팅 비중 50% 자동 축소
  ▸ ATR 동적 손절선: max(진입가 - 2.5×ATR, 진입가 × 0.92) 중 타이트한 쪽
  ▸ 확신도 점수대별 배분: 85pt→50% / 70pt→35% / 55pt→20%
  ▸ 오후 6시 이후 실행 시 우측 하단 지속성 팝업창 자동 표시

실행 타이밍: 장 마감 후 (15:30~) 또는 다음 날 장 시작 전
"""
from __future__ import annotations

import sys
import threading
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

# ══════════════════════════════════════════════════════════════════════════════
#  ★★★ 수동 잔고 구조체 — 매일 장 마감 후 직접 업데이트하세요 ★★★
#
#  종목코드 형식: KOSPI 종목 → "XXXXXX.KS" / KOSDAQ 종목 → "XXXXXX.KQ"
#  sl: 현재 MTS에 설정된 손절 예약가  /  tp: 목표 익절가
# ══════════════════════════════════════════════════════════════════════════════

MY_CURRENT_BALANCE: dict = {
    "cash": 3_026_638,      # 현재 가용 실제 예수금 (원)
    "positions": {
        "050890.KQ": {
            "name":        "쏠리드",
            "entry_price": 17_080,    # 평균 매수 단가 (원)
            "quantity":    155,       # 보유 수량 (주)
            "sl":          15_884,    # 현재 설정된 손절가 (원)
            "tp":          21_350,    # 현재 설정된 익절가 (원)
        },
        "021240.KS": {
            "name":        "웅진코웨이",
            "entry_price": 77_800,
            "quantity":    159,
            "sl":          71_576,
            "tp":          97_250,
        },
    },
}

# ══════════════════════════════════════════════════════════════════════════════
#  글로벌 설정 — 필요 시 조정 가능
# ══════════════════════════════════════════════════════════════════════════════

UNIVERSE_PER_MARKET: int   = 150    # 시장별 스크리닝 유니버스 상위 N종목
LOOKBACK_DAYS:       int   = 120    # 데이터 로드 기간 (지표 계산용)

RSI_OVERHEAT_THRESHOLD: float = 75.0   # 지수 RSI 과열 브레이크 임계값
MIN_TURNOVER_RATIO:     float = 2.5    # 야수 모드 거래대금 최소 배율
MAX_POS_BULL:           int   = 8      # 야수(상승장) 최대 보유 종목 수
MAX_POS_BEAR:           int   = 3      # 디펜스(하락장) 최대 보유 종목 수

BENCHMARK_SYMBOLS: dict[str, str] = {
    "KOSPI":  "^KS11",
    "KOSDAQ": "^KQ11",
}

# ══════════════════════════════════════════════════════════════════════════════
#  내부 지표 계산 유틸리티 (backtest.py 로직 동일)
# ══════════════════════════════════════════════════════════════════════════════

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """MultiIndex 컬럼을 단일 레벨로 평탄화"""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df


def _add_sma(df: pd.DataFrame) -> pd.DataFrame:
    """단기·중기·장기 이동평균선 추가"""
    df = df.copy()
    df["SMA_5"]  = df["Close"].rolling(5).mean()
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_60"] = df["Close"].rolling(60).mean()
    return df


def _calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI — backtest.py BacktestEngine._calc_rsi() 완전 동일"""
    delta    = close.diff()
    gain     = delta.where(delta > 0, 0.0)
    loss     = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi      = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def _calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """True Range 기반 ATR — Wilder EWM 스무딩"""
    high       = df["High"]
    low        = df["Low"]
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def _prepare_df(raw: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    """다운로드된 raw DataFrame에서 단일 종목 지표 완성 DataFrame 반환"""
    try:
        if isinstance(raw.columns, pd.MultiIndex):
            df = raw.xs(ticker, axis=1, level=1).dropna(how="all").copy()
        else:
            df = _flatten_columns(raw).dropna(how="all").copy()
        if df.empty or len(df) < 25:
            return None
        df = _add_sma(df)
        df["ATR"] = _calc_atr(df)
        return df
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  1. 시장 환경 분석기
# ══════════════════════════════════════════════════════════════════════════════

class MarketAnalyzer:
    """KOSPI·KOSDAQ 벤치마크 지수를 기반으로 장세 및 RSI 과열을 판별한다."""

    def __init__(self) -> None:
        self._index_dfs: dict[str, pd.DataFrame] = {}

    def load(self) -> None:
        print("  📊 벤치마크 지수 데이터 로드 중...")
        for mkt, sym in BENCHMARK_SYMBOLS.items():
            try:
                raw = yf.download(sym, period=f"{LOOKBACK_DAYS}d",
                                  auto_adjust=True, progress=False)
                if raw.empty:
                    print(f"    ⚠️ {mkt}({sym}) 데이터 없음")
                    continue
                df = _flatten_columns(raw).copy()
                df = _add_sma(df)
                df["RSI_14"] = _calc_rsi(df["Close"])
                self._index_dfs[mkt] = df
                last = df.iloc[-1]
                print(
                    f"    {mkt}({sym}) — {len(df)}일치 완료  |  "
                    f"현재가 {float(last['Close']):,.0f}  "
                    f"SMA60 {float(last['SMA_60']):,.0f}  "
                    f"RSI {float(last['RSI_14']):.1f}"
                )
            except Exception as e:
                print(f"    ⚠️ {mkt} 로드 실패: {e}")

    def is_bear_market(self) -> bool:
        """KOSPI·KOSDAQ 중 하나라도 SMA_60 위에 있으면 상승장(야수 모드) — backtest.py 동일"""
        if not self._index_dfs:
            return False
        valid = bear = 0
        for df in self._index_dfs.values():
            if df.empty:
                continue
            row   = df.iloc[-1]
            close = float(row.get("Close", np.nan))
            sma60 = float(row.get("SMA_60", np.nan))
            if np.isnan(close) or np.isnan(sma60):
                continue
            valid += 1
            if close < sma60:
                bear += 1
        if valid > 0 and bear < valid:
            return False
        return True

    def get_market_rsi(self) -> float:
        """복수 지수 중 최댓값 — 가장 과열된 지수 기준으로 브레이크 판정"""
        values = []
        for df in self._index_dfs.values():
            if "RSI_14" not in df.columns or df.empty:
                continue
            val = float(df["RSI_14"].iloc[-1])
            if not np.isnan(val):
                values.append(val)
        return max(values) if values else 50.0

    def get_summary(self) -> dict[str, dict]:
        """지수별 요약 정보 반환"""
        out = {}
        for mkt, df in self._index_dfs.items():
            if df.empty:
                continue
            row = df.iloc[-1]
            out[mkt] = {
                "close": float(row.get("Close", np.nan)),
                "sma60": float(row.get("SMA_60", np.nan)),
                "rsi14": float(row.get("RSI_14", np.nan)),
            }
        return out


# ══════════════════════════════════════════════════════════════════════════════
#  2. 보유 포지션 청산 감시
# ══════════════════════════════════════════════════════════════════════════════

class PositionMonitor:
    """MY_CURRENT_BALANCE["positions"]의 종목 현재가를 조회해 TP/SL 이탈 여부를 판정한다."""

    def __init__(self, positions: dict) -> None:
        self.positions = positions
        self.results:  list[dict] = []   # 전체 포지션 상태 (이탈 여부 포함)

    def check(self) -> None:
        if not self.positions:
            print("  ℹ️  보유 종목 없음")
            return

        tickers = list(self.positions.keys())
        print(f"  📡 보유 종목 현재가 조회 중 ({len(tickers)}개)...")

        try:
            query = tickers if len(tickers) > 1 else tickers[0]
            raw = yf.download(query, period="5d", auto_adjust=True, progress=False)
        except Exception as e:
            print(f"    ⚠️ 현재가 조회 실패: {e}")
            return

        for ticker, info in self.positions.items():
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    s = raw.xs(ticker, axis=1, level=1)["Close"].dropna()
                else:
                    s = _flatten_columns(raw)["Close"].dropna()

                if s.empty:
                    print(f"    ⚠️ {ticker} 가격 데이터 없음 — 건너뜀")
                    continue

                current_price = float(s.iloc[-1])
                entry  = float(info["entry_price"])
                sl     = float(info["sl"])
                tp     = float(info["tp"])
                name   = info.get("name", ticker)
                pnl    = (current_price - entry) / entry * 100

                alert_type: str | None = None
                if current_price <= sl:
                    alert_type = "SELL_SL"
                elif current_price >= tp:
                    alert_type = "SELL_TP"

                self.results.append({
                    "ticker":        ticker,
                    "name":          name,
                    "current_price": current_price,
                    "entry_price":   entry,
                    "quantity":      int(info.get("quantity", 0)),
                    "sl":            sl,
                    "tp":            tp,
                    "pnl_pct":       pnl,
                    "alert_type":    alert_type,
                })
                status = "🚨 매도 조건 충족" if alert_type else "✅ 보유 유지"
                print(
                    f"    [{ticker} {name}]  현재가 {current_price:>10,.0f}  "
                    f"손익 {pnl:+.1f}%  {status}"
                )
            except Exception as e:
                print(f"    ⚠️ {ticker} 처리 실패: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  3. 유니버스 빌드 (stock_ai 연동)
# ══════════════════════════════════════════════════════════════════════════════

def _build_universe(n: int = UNIVERSE_PER_MARKET) -> list[tuple[str, str]]:
    """KOSPI + KOSDAQ 상위 N종목 유니버스 구축. stock_ai 없으면 하드코딩 폴백."""
    try:
        import stock_ai as _sa
        result: list[tuple[str, str]] = []
        seen:   set[str] = set()
        for fn_name in ("get_top_kospi_stocks", "get_top_kosdaq_stocks"):
            fn = getattr(_sa, fn_name, None)
            if fn is None:
                continue
            for name, ticker in fn(n).items():
                if ticker not in seen:
                    seen.add(ticker)
                    result.append((ticker, name))
        if result:
            return result
    except ImportError:
        pass

    # stock_ai 없을 때 기본 유니버스 (KOSPI·KOSDAQ 주요 대형주)
    fallback = [
        ("005930.KS", "삼성전자"),   ("000660.KS", "SK하이닉스"),
        ("035420.KS", "NAVER"),       ("005380.KS", "현대차"),
        ("000270.KS", "기아"),        ("035720.KS", "카카오"),
        ("051910.KS", "LG화학"),      ("006400.KS", "삼성SDI"),
        ("028260.KS", "삼성물산"),    ("096770.KS", "SK이노베이션"),
        ("003670.KS", "포스코퓨처엠"),("207940.KS", "삼성바이오로직스"),
        ("068270.KS", "셀트리온"),    ("105560.KS", "KB금융"),
        ("055550.KS", "신한지주"),    ("086790.KS", "하나금융지주"),
        ("032830.KS", "삼성생명"),    ("017670.KS", "SK텔레콤"),
        ("030200.KS", "KT"),          ("009150.KS", "삼성전기"),
        ("003550.KS", "LG"),          ("034730.KS", "SK"),
        ("011070.KS", "LG이노텍"),    ("247540.KQ", "에코프로비엠"),
        ("086520.KQ", "에코프로"),    ("357780.KQ", "솔브레인"),
        ("293490.KQ", "카카오게임즈"),("263750.KQ", "펄어비스"),
        ("112040.KQ", "위메이드"),    ("041510.KQ", "에스엠"),
        ("035900.KQ", "JYP Ent."),    ("122870.KQ", "와이지엔터테인먼트"),
        ("196170.KQ", "알테오젠"),    ("145020.KQ", "휴젤"),
        ("214150.KQ", "클래시스"),    ("091990.KQ", "셀트리온헬스케어"),
        ("328130.KQ", "루닛"),        ("950130.KQ", "엑스플러스"),
        ("066970.KQ", "엘앤에프"),    ("247540.KQ", "에코프로비엠"),
    ]
    print("  ⚠️ stock_ai 모듈 없음 — 내장 기본 유니버스 사용 (약 40개)")
    return [(t, n) for t, n in fallback]


# ══════════════════════════════════════════════════════════════════════════════
#  4. 스크리닝 신호 탐지 (backtest.py 완전 동일 로직)
# ══════════════════════════════════════════════════════════════════════════════

def _score_golden_cross(df: pd.DataFrame, vol_ratio: float) -> float:
    """[야수 모드] SMA_GOLDEN_CROSS 확신도 점수 (50~100pt) — backtest.py 동일"""
    idx      = len(df) - 1
    row      = df.iloc[idx]
    prev_row = df.iloc[idx - 1]
    close      = float(row["Close"])
    prev_close = float(prev_row["Close"])
    sma5       = float(row.get("SMA_5",  np.nan))
    sma20      = float(row.get("SMA_20", np.nan))

    vol_bonus = 35.0 if vol_ratio >= 10 else (25.0 if vol_ratio >= 5 else 15.0)
    momentum_bonus = 10.0 if close > prev_close else 0.0

    if sma20 > 0 and not np.isnan(sma5):
        gap_pct = (sma5 - sma20) / sma20 * 100
        gap_bonus = -15.0 if gap_pct > 15 else (0.0 if gap_pct > 5 else 10.0)
    else:
        gap_bonus = 0.0

    return min(100.0, round(50.0 + vol_bonus + momentum_bonus + gap_bonus, 1))


def _score_nulim(df: pd.DataFrame, max_spike_ratio: float) -> float:
    """[디펜스 모드] DRY_VOLUME_NULIM 확신도 점수 (50~100pt) — backtest.py 동일"""
    idx   = len(df) - 1
    row   = df.iloc[idx]
    close = float(row["Close"])
    sma20 = float(row.get("SMA_20", np.nan))

    gap_pct   = abs(close - sma20) / sma20 * 100 if sma20 > 0 else 5.0
    gap_bonus = max(0.0, 20.0 - (gap_pct * 4.0))
    vol_bonus = 30.0 if max_spike_ratio >= 10 else (20.0 if max_spike_ratio >= 5 else 10.0)

    return min(100.0, round(50.0 + gap_bonus + vol_bonus, 1))


def _detect_golden_cross(df: pd.DataFrame) -> tuple[bool, float, float]:
    """[야수 모드] 오늘 기준 SMA_GOLDEN_CROSS 신호 탐지
    Returns: (신호여부, vol_ratio, score)
    """
    if len(df) < 25:
        return False, 0.0, 0.0

    idx      = len(df) - 1
    row      = df.iloc[idx]
    prev_row = df.iloc[idx - 1]

    close      = float(row["Close"])
    vol        = float(row["Volume"])
    sma5       = float(row.get("SMA_5",  np.nan))
    sma20      = float(row.get("SMA_20", np.nan))
    prev_close = float(prev_row["Close"])

    if any(np.isnan(v) for v in [sma5, sma20, close, vol]):
        return False, 0.0, 0.0

    # 조건 1: 5일선 > 20일선 정배열
    if sma5 <= sma20:
        return False, 0.0, 0.0

    # 조건 2: 오늘 거래대금이 최근 20일 평균 대비 MIN_TURNOVER_RATIO 배 이상 폭발
    today_tv  = close * vol
    slice_v   = df["Volume"].iloc[max(0, idx - 19): idx + 1]
    slice_c   = df["Close"].iloc[max(0, idx - 19): idx + 1]
    avg_tv_20 = (slice_v * slice_c).mean()
    if avg_tv_20 <= 0:
        return False, 0.0, 0.0
    vol_ratio = today_tv / avg_tv_20
    if vol_ratio < MIN_TURNOVER_RATIO:
        return False, 0.0, 0.0

    # 조건 3: 전일 종가 및 5일선 위에서의 돌파 흐름
    if close < prev_close or close < sma5:
        return False, 0.0, 0.0

    score = _score_golden_cross(df, vol_ratio)
    return True, vol_ratio, score


def _detect_nulim(df: pd.DataFrame) -> tuple[bool, float, float]:
    """[디펜스 모드] 오늘 기준 DRY_VOLUME_NULIM 눌림목 신호 탐지
    Returns: (신호여부, max_spike_ratio, score)
    """
    if len(df) < 30:
        return False, 0.0, 0.0

    idx   = len(df) - 1
    row   = df.iloc[idx]
    close = float(row["Close"])
    vol   = float(row["Volume"])
    sma20 = float(row.get("SMA_20", np.nan))

    if np.isnan(sma20) or sma20 <= 0:
        return False, 0.0, 0.0

    # 조건 1: 주가가 20일선 부근 수렴 (-1% ~ +2.5%)
    if not (sma20 * 0.99 <= close <= sma20 * 1.025):
        return False, 0.0, 0.0

    # 조건 2: 오늘 거래량이 20일 평균 대비 70% 이하로 급감
    avg_vol_20 = df["Volume"].iloc[max(0, idx - 19): idx + 1].mean()
    if vol > avg_vol_20 * 0.7:
        return False, 0.0, 0.0

    # 조건 3: 최근 10일 내 거래대금 5배+ 폭발 흔적
    has_spike    = False
    max_spike    = 1.0
    for lb in range(max(0, idx - 10), idx):
        p_row = df.iloc[lb]
        p_tv  = float(p_row["Close"]) * float(p_row["Volume"])
        slice_v = df["Volume"].iloc[max(0, lb - 19): lb + 1]
        slice_c = df["Close"].iloc[max(0, lb - 19): lb + 1]
        p_avg = (slice_v * slice_c).mean()
        ratio = p_tv / p_avg if p_avg > 0 else 0.0
        if ratio >= 5.0:
            has_spike = True
            max_spike = max(max_spike, ratio)

    if not has_spike:
        return False, 0.0, 0.0

    score = _score_nulim(df, max_spike)
    return True, max_spike, score


def run_screening(
    universe:   list[tuple[str, str]],
    is_bear:    bool,
    held_tickers: set[str],
) -> list[dict]:
    """유니버스 전체를 청크 단위로 다운로드하고 오늘 자 신호를 스크리닝한다."""
    CHUNK = 30
    candidates: list[dict] = []

    for i in range(0, len(universe), CHUNK):
        chunk   = universe[i: i + CHUNK]
        tickers = [t for t, _ in chunk]

        try:
            query = tickers if len(tickers) > 1 else tickers[0]
            raw   = yf.download(query, period=f"{LOOKBACK_DAYS}d",
                                 auto_adjust=True, progress=False)
        except Exception:
            continue

        for ticker, name in chunk:
            if ticker in held_tickers:
                continue

            df = _prepare_df(raw, ticker)
            if df is None:
                continue

            try:
                if is_bear:
                    ok, ratio, score = _detect_nulim(df)
                    signal = "DRY_VOLUME_NULIM"
                else:
                    ok, ratio, score = _detect_golden_cross(df)
                    signal = "SMA_GOLDEN_CROSS"

                if ok and score >= 55.0:
                    row   = df.iloc[-1]
                    close = float(row["Close"])
                    atr   = float(row.get("ATR", np.nan))
                    candidates.append({
                        "ticker": ticker,
                        "name":   name,
                        "close":  close,
                        "atr":    atr,
                        "score":  score,
                        "ratio":  ratio,
                        "signal": signal,
                    })
            except Exception:
                continue

        done = min(i + CHUNK, len(universe))
        print(
            f"    [{done:>4}/{len(universe)}] 진행 중..."
            f"  신호 포착: {len(candidates)}개"
        )

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates


# ══════════════════════════════════════════════════════════════════════════════
#  5. 매수 계획 계산 (배분 비중 + ATR 손절/익절)
# ══════════════════════════════════════════════════════════════════════════════

def calc_buy_plan(
    candidate:     dict,
    cash:          float,
    is_overheated: bool,
    n_positions:   int,
    max_positions: int,
) -> dict | None:
    """확신도 점수 + RSI 과열 여부를 반영해 매수 금액·수량·SL·TP를 계산한다.

    배분 기준 (backtest.py _try_buy 완전 동일):
      85pt 이상 → 50%  /  70pt 이상 → 35%  /  55pt 이상 → 20%
    RSI 과열 시: 모든 비중을 정확히 절반으로 축소
    """
    if n_positions >= max_positions:
        return None

    score = candidate["score"]
    if score >= 85:
        alloc_pct = 0.50
    elif score >= 70:
        alloc_pct = 0.35
    elif score >= 55:
        alloc_pct = 0.20
    else:
        return None

    if is_overheated:
        alloc_pct *= 0.5

    alloc_cash = cash * alloc_pct
    close      = candidate["close"]
    atr        = candidate["atr"]

    qty = int(alloc_cash / close) if close > 0 else 0
    if qty <= 0:
        return None

    # [Step 3] ATR 기반 동적 손절선 — backtest.py _try_buy SMA_GOLDEN_CROSS 블록 동일
    if not np.isnan(atr) and atr > 0:
        sl_atr   = close - 2.5 * atr   # ATR 버퍼 손절
        sl_fixed = close * 0.92         # -8% 고정 하한 안전망
        sl       = max(sl_atr, sl_fixed)   # 더 높은 쪽 = 더 타이트한 손절
    else:
        sl = close * 0.93               # ATR 미산출 시 -7% 폴백

    tp = close * 1.25   # 야수 모드 고정 +25% 익절

    return {
        "alloc_pct":  alloc_pct,
        "alloc_cash": alloc_cash,
        "qty":        qty,
        "sl":         sl,
        "tp":         tp,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  6. 오후 6시 이후 우측 하단 지속성 팝업창
# ══════════════════════════════════════════════════════════════════════════════

def show_popup_after_close() -> threading.Thread:
    """tkinter 팝업창 — 사용자가 직접 닫기 버튼을 누를 때까지 유지된다.

    메인 스레드를 차단하지 않도록 별도 데몬 스레드에서 실행하고,
    프로그램 종료 시 강제 닫힘을 방지하기 위해 join()으로 대기한다.
    헤드리스 서버(디스플레이 없음)에서는 조용히 스킵된다.
    """
    try:
        import tkinter as tk
    except Exception:
        print("  [팝업 생략] 디스플레이 없음 — 헤드리스 서버 환경")
        t = threading.Thread(target=lambda: None, daemon=True)
        t.start()
        return t

    popup_closed = threading.Event()

    def _run() -> None:
        try:
            root = tk.Tk()
        except Exception:
            print("  [팝업 생략] 디스플레이 없음 — 헤드리스 서버 환경")
            return
        root.title("📊 실전 스크리너 알림")

        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()
        w, h = 320, 145
        x = sw - w - 24
        y = sh - h - 72
        root.geometry(f"{w}x{h}+{x}+{y}")

        root.attributes("-topmost", True)
        root.configure(bg="#1e1e2e")
        root.resizable(False, False)

        def _on_close() -> None:
            popup_closed.set()
            root.destroy()

        root.protocol("WM_DELETE_WINDOW", _on_close)

        msg = (
            "🌙 장 마감 분석 완료!\n\n"
            "콘솔 창에서 내일 아침의\n"
            "매매 지침 리포트를 확인하고\n"
            "MTS에 주문을 준비하세요."
        )
        tk.Label(
            root, text=msg,
            bg="#1e1e2e", fg="#cdd6f4",
            font=("맑은 고딕", 9),
            justify="center",
        ).pack(expand=True, pady=(12, 4))

        tk.Button(
            root,
            text="  리포트 확인 완료 (닫기)  ",
            command=_on_close,
            bg="#313244", fg="#cba6f7",
            font=("맑은 고딕", 9, "bold"),
            relief="flat",
            cursor="hand2",
            padx=8, pady=5,
            activebackground="#45475a",
            activeforeground="#f5c2e7",
        ).pack(pady=(0, 12))

        root.mainloop()

    t = threading.Thread(target=_run, daemon=False)
    t.start()
    return t   # 호출자가 필요 시 join() 가능


# ══════════════════════════════════════════════════════════════════════════════
#  7. 통합 콘솔 리포트 출력
# ══════════════════════════════════════════════════════════════════════════════

def print_report(
    today_str:        str,
    index_summary:    dict,
    is_bear:          bool,
    market_rsi:       float,
    is_overheated:    bool,
    position_results: list[dict],
    buy_plans:        list[dict],
    cash:             float,
) -> None:
    SEP  = "═" * 56
    LINE = "─" * 56

    mode_label = "디펜스 모드 기동 중 🛡️" if is_bear else "야수 모드 기동 중 🔥"
    rsi_label  = (
        f"⚠️ 과열 브레이크 작동 — 베팅 비중 50% 축소"
        if is_overheated
        else "✅ 정상 구간"
    )

    print()
    print(SEP)
    print(f"  📢 [실전 지침] {today_str} 장 마감 기준 내일의 매매 전략 리포트")
    print(SEP)

    # ── 시장 환경 ─────────────────────────────────────────────────────────────
    print(f"📊 [시장 환경] 모드: {mode_label}")
    print(f"   지수 RSI  : {market_rsi:.1f}  ({rsi_label})")
    for mkt, info in index_summary.items():
        if any(np.isnan(v) for v in info.values()):
            continue
        diff  = info["close"] - info["sma60"]
        arrow = "▲" if diff > 0 else "▼"
        dist  = diff / info["sma60"] * 100
        print(
            f"   {mkt:6s} : 현재가 {info['close']:>10,.0f}  "
            f"SMA60 {info['sma60']:>10,.0f}  "
            f"{arrow} {dist:+.1f}%  RSI {info['rsi14']:.1f}"
        )

    print(LINE)

    # ── 매도 알림 ─────────────────────────────────────────────────────────────
    sell_alerts = [r for r in position_results if r["alert_type"] is not None]
    hold_list   = [r for r in position_results if r["alert_type"] is None]

    if sell_alerts:
        print("🚨 [내일 아침 시초가 즉시 매도 지침]")
        for r in sell_alerts:
            if r["alert_type"] == "SELL_TP":
                trigger_word = "익절가"
                side_val     = r["tp"]
                breach_word  = "돌파"
            else:
                trigger_word = "손절선"
                side_val     = r["sl"]
                breach_word  = "이탈"
            est_amt = r["current_price"] * r["quantity"]
            print(
                f"  - [{r['ticker']}  {r['name']}]"
                f"  오늘 종가({r['current_price']:,.0f})가"
                f" {trigger_word}({side_val:,.0f}) {breach_word}!"
            )
            print(
                f"    → 즉시 전량 매도 {r['quantity']}주"
                f"  예상 회수금 {est_amt:,.0f}원"
                f"  손익률 {r['pnl_pct']:+.1f}%"
            )
    else:
        print("✅ [매도 지침] 청산 조건에 걸린 종목 없음 — 전 종목 보유 유지")

    if hold_list:
        print(f"\n📋 [보유 유지 현황]  총 {len(hold_list)}종목")
        for r in hold_list:
            pnl_mark = "📈" if r["pnl_pct"] >= 0 else "📉"
            print(
                f"  - [{r['ticker']}  {r['name']}]"
                f"  현재가 {r['current_price']:>10,.0f}"
                f"  미실현 {r['pnl_pct']:+.1f}% {pnl_mark}"
                f"  │  SL {r['sl']:,.0f}  →  TP {r['tp']:,.0f}"
            )

    print(LINE)

    # ── 매수 지침 ─────────────────────────────────────────────────────────────
    if buy_plans:
        print(f"🛒 [내일 아침 시초가 신규 매수 지침]  가용 예수금 {cash:,.0f}원")
        print()
        for bp in buy_plans:
            c = bp["candidate"]
            p = bp["plan"]
            mode_tag = "야수 돌파" if c["signal"] == "SMA_GOLDEN_CROSS" else "디펜스 눌림목"
            atr_str  = f"{c['atr']:,.0f}원" if not np.isnan(c["atr"]) else "산출 불가"
            print(
                f"  - [{c['ticker']}  {c['name']}]"
                f"  {mode_tag} 포착  (확신도 {c['score']:.1f}pt"
                f" / 배정 비중 {p['alloc_pct']:.1%})"
            )
            print(f"    ▶ 추천 매수 금액 : {p['alloc_cash']:>12,.0f} 원")
            print(
                f"    ▶ 예상 매수 수량 : 약 {p['qty']} 주"
                f"  (내일 시초가 부근 진입  /  ATR {atr_str})"
            )
            print(f"    ▶ 진입 후 즉시 설정할 타겟 가격:")
            print(f"       - 🛑 동적 손절가 (SL) : {p['sl']:>10,.0f} 원  (ATR 변동성 반영)")
            print(f"       - 🎯 목표 익절가 (TP) : {p['tp']:>10,.0f} 원  (+25% 고정)")
            print()
    else:
        print("📭 [매수 지침] 오늘 기준 신규 매수 신호 없음 — 내일 관망")

    print(SEP)
    print("  💡 이 리포트는 오늘 장 마감 데이터 기준 자동 산출 값입니다.")
    print("     실제 주문 전 종목 뉴스·공시·시황을 반드시 재확인하세요.")
    print(SEP)


# ══════════════════════════════════════════════════════════════════════════════
#  8. 메인 실행 흐름
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    now       = datetime.now()
    today_str = now.strftime("%Y-%m-%d")
    cash      = float(MY_CURRENT_BALANCE["cash"])
    positions = MY_CURRENT_BALANCE["positions"]
    n_pos     = len(positions)

    print()
    print("═" * 56)
    print("  🚀 실전 라이브 스크리너 & 잔고 관리 시스템")
    print(f"     실행 시각 : {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print("═" * 56)
    print(f"  현재 예수금  : {cash:>15,.0f} 원")
    print(f"  보유 종목 수 : {n_pos}개")
    if 9 <= now.hour < 15 and now.weekday() < 5:
        print()
        print("  ⚠️  장 중 실행이 감지되었습니다.")
        print("     종가 확정 전 데이터로 계산하므로 SL/TP 수치가 불완전할 수 있습니다.")
        print("     장 마감(15:30) 이후 재실행을 권장합니다.")
    print()

    # ── Step 1: 시장 환경 분석 ─────────────────────────────────────────────────
    print("[ 1/4 ] 시장 환경 분석 중...")
    analyzer      = MarketAnalyzer()
    analyzer.load()
    is_bear        = analyzer.is_bear_market()
    market_rsi     = analyzer.get_market_rsi()
    is_overheated  = market_rsi > RSI_OVERHEAT_THRESHOLD
    max_positions  = MAX_POS_BEAR if is_bear else MAX_POS_BULL
    index_summary  = analyzer.get_summary()

    mode_str  = "디펜스 모드" if is_bear else "야수 모드"
    heat_str  = f"  ⚠️  RSI {market_rsi:.1f} > {RSI_OVERHEAT_THRESHOLD:.0f} 과열 브레이크 발동 — 베팅 비중 50% 축소" if is_overheated else ""
    print(f"  → {mode_str}  |  지수 RSI {market_rsi:.1f}  |  최대 보유 {max_positions}종목{heat_str}")
    print()

    # ── Step 2: 보유 포지션 청산 점검 ─────────────────────────────────────────
    print("[ 2/4 ] 보유 포지션 청산 조건 점검 중...")
    monitor = PositionMonitor(positions)
    monitor.check()
    print()

    # ── Step 3: 신규 매수 스크리닝 ────────────────────────────────────────────
    print(f"[ 3/4 ] 신규 매수 신호 스크리닝 중  ({'디펜스' if is_bear else '야수'} 모드)...")
    held_tickers = set(positions.keys())
    universe     = _build_universe(UNIVERSE_PER_MARKET)
    print(f"  유니버스 : {len(universe)}개 종목")
    candidates   = run_screening(universe, is_bear, held_tickers)
    print(f"  → 총 {len(candidates)}개 신호 포착")
    print()

    # ── Step 4: 매수 계획 계산 ────────────────────────────────────────────────
    print("[ 4/4 ] 매수 금액·수량·SL/TP 계산 중...")
    buy_plans: list[dict] = []
    cur_n = n_pos
    for c in candidates:
        if cur_n >= max_positions:
            print(f"  최대 보유 종목 수({max_positions}개) 도달 — 추가 매수 지침 생략")
            break
        plan = calc_buy_plan(
            candidate     = c,
            cash          = cash,
            is_overheated = is_overheated,
            n_positions   = cur_n,
            max_positions = max_positions,
        )
        if plan:
            buy_plans.append({"candidate": c, "plan": plan})
            cur_n += 1
    print(f"  → {len(buy_plans)}개 매수 지침 산출 완료")
    print()

    # ── 통합 리포트 출력 ───────────────────────────────────────────────────────
    print_report(
        today_str        = today_str,
        index_summary    = index_summary,
        is_bear          = is_bear,
        market_rsi       = market_rsi,
        is_overheated    = is_overheated,
        position_results = monitor.results,
        buy_plans        = buy_plans,
        cash             = cash,
    )

    # ── 오후 6시 이후: 우측 하단 지속성 팝업 ────────────────────────────────────
    if now.hour >= 18:
        print()
        print("  [팝업] 오후 6시 이후 실행 감지 — 우측 하단 알림창 표시 중...")
        popup_thread = show_popup_after_close()
        print("  팝업창을 직접 닫을 때까지 프로그램이 유지됩니다.\n")
        popup_thread.join()   # 사용자가 닫을 때까지 메인 스레드 대기
    else:
        print()
        print("  (오후 6시 이전 실행 — 팝업 알림창 생략)")


if __name__ == "__main__":
    main()
