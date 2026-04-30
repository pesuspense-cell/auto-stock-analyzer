"""
stock_ai.py - 주식 분석 핵심 알고리즘 엔진
매수/매도 신호, 추천 종목, 예상 수익률 분석

아키텍처:
  이 파일이 모든 비즈니스 로직의 단일 구현 소스입니다.
  src/ 하위 모듈(indicators.py, fundamental.py, news_logic.py, utils.py)은
  이 파일의 함수를 re-export하는 얇은 래퍼입니다.
  app.py는 src/ 모듈을 통해서만 import합니다.

  src/indicators.py  — 기술적 지표·매매 신호
  src/fundamental.py — 펀더멘털 가치평가
  src/news_logic.py  — 뉴스 수집·필터링·감성 분석 (비동기 배치, 캐싱)
  src/utils.py       — 종목 사전·지수 상수·시장 데이터
"""
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
import hashlib
import time
import re as _re_global
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings('ignore')

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

# ─── 주요 종목 사전 ───────────────────────────────────────────────────────────

KOSPI_STOCKS = {
    "삼성전자":       "005930.KS",
    "SK하이닉스":     "000660.KS",
    "LG에너지솔루션": "373220.KS",
    "삼성바이오로직스": "207940.KS",
    "현대차":         "005380.KS",
    "기아":           "000270.KS",
    "POSCO홀딩스":    "005490.KS",
    "NAVER":          "035420.KS",
    "카카오":         "035720.KS",
    "LG화학":         "051910.KS",
    "삼성SDI":        "006400.KS",
    "셀트리온":       "068270.KS",
}

KOSDAQ_STOCKS = {
    "에코프로비엠":   "247540.KQ",
    "에코프로":       "086520.KQ",
    "HLB":           "028300.KQ",
    "알테오젠":       "196170.KQ",
    "리가켐바이오":   "141080.KQ",
    "삼천당제약":     "000250.KQ",
    "클래시스":       "214150.KQ",
    "포스코DX":       "022100.KQ",
    "두산테스나":     "131220.KQ",
    "메가스터디교육": "215200.KQ",
    "나무기술":       "242040.KQ",
}

US_STOCKS = {
    "Apple":      "AAPL",
    "Microsoft":  "MSFT",
    "NVIDIA":     "NVDA",
    "Amazon":     "AMZN",
    "Tesla":      "TSLA",
    "Meta":       "META",
    "Alphabet":   "GOOGL",
    "Broadcom":   "AVGO",
}

EXCHANGE_PAIRS = {
    "USD/KRW": "USDKRW=X",
    "JPY/KRW": "JPYKRW=X",
    "EUR/KRW": "EURKRW=X",
    "CNY/KRW": "CNYKRW=X",
}

INDICES = {
    "코스피": "^KS11",
    "코스닥": "^KQ11",
    "나스닥": "^IXIC",
    "S&P500": "^GSPC",
}

# ─── 데이터 로드 및 지표 계산 ─────────────────────────────────────────────────

def get_stock_data(ticker: str, period: str = "3mo") -> pd.DataFrame:
    """
    주식 데이터 로드 및 기술적 지표 계산 (pandas-TA 벡터화 엔진).
    지표 목록:
      이동평균 : SMA 5/20/60, EMA 20/50/200
      모멘텀   : RSI(14), 스토캐스틱(14,3), CCI(20), Williams %R(14), ROC(12)
      추세     : MACD(12/26/9), ADX±DI(14), 일목균형표
      변동성   : 볼린저밴드(20,2σ), ATR(14), Z-Score(20)
      거래량   : Volume MA20, OBV, OBV MA20, MFI(14)
    """
    data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if data.empty or len(data) < 20:
        return data

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)

    close  = data["Close"]
    high   = data["High"]
    low    = data["Low"]
    volume = data["Volume"]

    # ── 이동평균선 (SMA / EMA) ─────────────────────────────────────────────────
    data["SMA_5"]  = close.rolling(5).mean()
    data["SMA_20"] = close.rolling(20).mean()
    data["SMA_60"] = close.rolling(60).mean()
    data["EMA_20"] = close.ewm(span=20, adjust=False).mean()
    data["EMA_50"] = close.ewm(span=50, adjust=False).mean()
    if len(data) >= 200:
        data["EMA_200"] = close.ewm(span=200, adjust=False).mean()

    # ── RSI (14) ──────────────────────────────────────────────────────────────
    _delta = close.diff()
    _up    = _delta.clip(lower=0)
    _down  = (-_delta).clip(lower=0)
    _rs    = _up.rolling(14).mean() / _down.rolling(14).mean()
    data["RSI"] = 100.0 - (100.0 / (1.0 + _rs))

    # ── MACD (12/26/9) ────────────────────────────────────────────────────────
    _ema12 = close.ewm(span=12, adjust=False).mean()
    _ema26 = close.ewm(span=26, adjust=False).mean()
    data["MACD"]        = _ema12 - _ema26
    data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
    data["MACD_Hist"]   = data["MACD"] - data["MACD_Signal"]

    # ── 볼린저밴드 (20, 2σ) ───────────────────────────────────────────────────
    _bb_mid = close.rolling(20).mean()
    _bb_std = close.rolling(20).std()
    data["BB_Middle"] = _bb_mid
    data["BB_Upper"]  = _bb_mid + 2 * _bb_std
    data["BB_Lower"]  = _bb_mid - 2 * _bb_std
    data["BB_Width"]  = (data["BB_Upper"] - data["BB_Lower"]) / _bb_mid
    _bb_range = (data["BB_Upper"] - data["BB_Lower"]).replace(0, np.nan)
    data["BB_PCT"]    = (close - data["BB_Lower"]) / _bb_range

    # ── 스토캐스틱 (14, 3) ────────────────────────────────────────────────────
    _low14  = low.rolling(14).min()
    _high14 = high.rolling(14).max()
    _stoch_raw = (close - _low14) / (_high14 - _low14).replace(0, np.nan) * 100
    data["STOCH_K"] = _stoch_raw.rolling(3).mean()
    data["STOCH_D"] = data["STOCH_K"].rolling(3).mean()

    # ── True Range (ATR 공유) ─────────────────────────────────────────────────
    _tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    # ── ADX + 방향성 지수 (14) ────────────────────────────────────────────────
    _h_diff   = high.diff()
    _l_diff   = (-low).diff()
    _plus_dm  = pd.Series(np.where((_h_diff > _l_diff) & (_h_diff > 0), _h_diff, 0.0), index=data.index)
    _minus_dm = pd.Series(np.where((_l_diff > _h_diff) & (_l_diff > 0), _l_diff, 0.0), index=data.index)
    _atr14    = _tr.rolling(14).mean()
    _plus_di  = 100 * _plus_dm.rolling(14).mean() / _atr14
    _minus_di = 100 * _minus_dm.rolling(14).mean() / _atr14
    _di_sum   = (_plus_di + _minus_di).replace(0, np.nan)
    _dx       = (_plus_di - _minus_di).abs() / _di_sum * 100
    data["ADX"]     = _dx.rolling(14).mean()
    data["ADX_POS"] = _plus_di
    data["ADX_NEG"] = _minus_di

    # ── CCI (20) ──────────────────────────────────────────────────────────────
    _tp      = (high + low + close) / 3
    _cci_ma  = _tp.rolling(20).mean()
    _cci_mad = _tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    data["CCI"] = (_tp - _cci_ma) / (0.015 * _cci_mad.replace(0, np.nan))

    # ── Williams %R (14) ──────────────────────────────────────────────────────
    _wr_high = high.rolling(14).max()
    _wr_low  = low.rolling(14).min()
    data["WILLIAMS_R"] = (_wr_high - close) / (_wr_high - _wr_low).replace(0, np.nan) * -100

    # ── ATR (14) ──────────────────────────────────────────────────────────────
    data["ATR"] = _tr.rolling(14).mean()

    # ── ROC (12) ──────────────────────────────────────────────────────────────
    data["ROC"] = (close / close.shift(12) - 1) * 100

    # ── OBV + MA20 ────────────────────────────────────────────────────────────
    _obv_dir = np.sign(close.diff()).fillna(0)
    data["OBV"]      = (volume * _obv_dir).cumsum()
    data["OBV_MA20"] = data["OBV"].rolling(20).mean()

    # ── MFI (14) ──────────────────────────────────────────────────────────────
    _tp2     = (high + low + close) / 3
    _mf      = _tp2 * volume
    _tp_diff = _tp2.diff()
    _pos_mf  = _mf.where(_tp_diff > 0, 0.0).rolling(14).sum()
    _neg_mf  = _mf.where(_tp_diff < 0, 0.0).rolling(14).sum()
    data["MFI"] = 100 - (100 / (1 + _pos_mf / _neg_mf.replace(0, np.nan)))

    # ── 거래량 이평 ───────────────────────────────────────────────────────────
    data["Volume_MA20"] = volume.rolling(20).mean()

    # ── 일목균형표 (Ichimoku Cloud) ────────────────────────────────────────────
    data["ICHI_TENKAN"] = (high.rolling(9).max()  + low.rolling(9).min())  / 2
    data["ICHI_KIJUN"]  = (high.rolling(26).max() + low.rolling(26).min()) / 2
    data["ICHI_SPAN_A"] = ((data["ICHI_TENKAN"] + data["ICHI_KIJUN"]) / 2).shift(26)
    data["ICHI_SPAN_B"] = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    data["ICHI_CHIKOU"] = close.shift(-26)

    # ── Z-Score (20일) ────────────────────────────────────────────────────────
    _z_mean = close.rolling(20).mean()
    _z_std  = close.rolling(20).std()
    data["Z_SCORE"] = (close - _z_mean) / _z_std.replace(0, np.nan)

    # ── VWAP 멀티 타임프레임 (Shannon — 롤링 누적합) ──────────────────────────
    # 수식: (Typical_Price × Volume) 롤링합 / Volume 롤링합
    # 참고: 라이언 섀넌, 《기술적 분석을 이용한 다중 타임프레임 분석》
    _tp_vwap = (high + low + close) / 3  # Typical Price
    _pv      = _tp_vwap * volume
    for _window, _col in [(5, "VWAP_W"), (20, "VWAP_M"), (60, "VWAP_Q")]:
        _cum_pv = _pv.rolling(_window).sum()
        _cum_v  = volume.rolling(_window).sum().replace(0, np.nan)
        data[_col] = _cum_pv / _cum_v

    return data


# ─── 매매 신호 알고리즘 ───────────────────────────────────────────────────────

def generate_signals(data: pd.DataFrame) -> dict:
    """
    12개 모듈 복합 채점으로 매매 신호 생성.
    점수 범위: -10 ~ +10  (양수=매수, 음수=매도)

    모듈별 최대 기여도:
      1. 장기 추세 필터 (EMA 200)        ±1.5  — 시장 국면 판단
      2. MACD 모멘텀                     ±2.5  — 추세 전환 감지
      3. RSI 과매수/과매도               ±2.0  — 단기 모멘텀
      4. 오실레이터 컨센서스             ±2.0  — 스토캐스틱·CCI·Williams%R 합의
         (스토캐스틱 ±0.8 / CCI ±0.6 / Williams%R ±0.6)
      5. 거래량 지표 (OBV + MFI)         ±1.5  — 수급·자금 흐름
      6. ADX 필터 적용 EMA 크로스        ±2.0  — 추세 방향 (신뢰도 보정)
      7. 볼린저밴드 + Squeeze            ±1.0  — 변동성 위치
      8. ROC 모멘텀 확인                 ±0.5  — 가격 변화율 필터
      9. 일목균형표 구름 위치            ±1.8  — 중기 추세 국면 (구름 위·아래·내부)
     10. Z-Score 통계적 위치            ±1.0  — 통계적 과매수/과매도 감지
     11. 다이버전스 (RSI·MACD)          ±1.5  — 추세 전환 조기 경보
     12. VWAP 멀티 타임프레임           ±2.0  — 주간·월간·분기 VWAP 위치 및 스택 정렬
         (참고: 라이언 섀넌, 《기술적 분석을 이용한 다중 타임프레임 분석》)
    이론적 최대: ±20 → cap ±10
    """
    if data.empty or len(data) < 21:
        return {}

    score   = 0.0
    reasons = []
    last    = data.iloc[-1]
    prev    = data.iloc[-2]
    close   = data["Close"]
    price   = float(close.iloc[-1])
    p_chg   = (price - float(close.iloc[-2])) / float(close.iloc[-2])  # 당일 등락률

    # ══ 1. 장기 추세 필터 (EMA 200) — ±1.5 ═══════════════════════════════════
    ema200 = _f(last, "EMA_200")
    if ema200:
        if price > ema200 * 1.02:        # 명확히 상단
            score += 1.5
            reasons.append(f"EMA200 상단 +2% ({price:,.0f} vs {ema200:,.0f}) → 장기 강세장")
        elif price > ema200:
            score += 0.8
            reasons.append(f"EMA200 상단 ({price:,.0f} > {ema200:,.0f}) → 장기 상승 추세")
        elif price < ema200 * 0.98:      # 명확히 하단
            score -= 1.5
            reasons.append(f"EMA200 하단 -2% ({price:,.0f} vs {ema200:,.0f}) → 장기 약세장")
        else:
            score -= 0.8
            reasons.append(f"EMA200 하단 ({price:,.0f} < {ema200:,.0f}) → 장기 하락 추세")

    # ══ 2. MACD 모멘텀 — ±2.5 ════════════════════════════════════════════════
    macd,  sig  = _f(last, "MACD"),  _f(last, "MACD_Signal")
    pmacd, psig = _f(prev, "MACD"),  _f(prev, "MACD_Signal")
    if all(v is not None for v in [macd, sig, pmacd, psig]):
        cross_up   = pmacd < psig and macd > sig
        cross_down = pmacd > psig and macd < sig
        if cross_up:
            score += 2.0; reasons.append("MACD 골든크로스 → 강한 매수 신호")
        elif cross_down:
            score -= 2.0; reasons.append("MACD 데드크로스 → 강한 매도 신호")
        elif macd > sig:
            score += 0.5; reasons.append(f"MACD 매수 우위 ({macd:.3f} > {sig:.3f})")
        else:
            score -= 0.5; reasons.append(f"MACD 매도 우위 ({macd:.3f} < {sig:.3f})")

        # 히스토그램 기울기 (최근 3봉 추세)
        if "MACD_Hist" in data.columns and len(data) >= 4:
            hist3 = data["MACD_Hist"].iloc[-3:].dropna()
            if len(hist3) == 3:
                if float(hist3.iloc[0]) < float(hist3.iloc[1]) < float(hist3.iloc[2]):
                    score += 0.5; reasons.append("MACD 히스토그램 연속 상승 → 모멘텀 강화")
                elif float(hist3.iloc[0]) > float(hist3.iloc[1]) > float(hist3.iloc[2]):
                    score -= 0.5; reasons.append("MACD 히스토그램 연속 하락 → 모멘텀 약화")

    # ══ 3. RSI (14) — ±2.0 ═══════════════════════════════════════════════════
    rsi = _f(last, "RSI")
    if rsi is not None:
        if rsi < 20:
            score += 2.0; reasons.append(f"RSI 극과매도 ({rsi:.1f}) → 강반등 기대")
        elif rsi < 30:
            score += 1.5; reasons.append(f"RSI 과매도 ({rsi:.1f}) → 매수 고려")
        elif rsi < 40:
            score += 0.7; reasons.append(f"RSI 매수권 ({rsi:.1f})")
        elif rsi < 50:
            score += 0.2
        elif rsi > 80:
            score -= 2.0; reasons.append(f"RSI 극과매수 ({rsi:.1f}) → 강한 매도 신호")
        elif rsi > 70:
            score -= 1.5; reasons.append(f"RSI 과매수 ({rsi:.1f}) → 매도 고려")
        elif rsi > 60:
            score -= 0.7; reasons.append(f"RSI 과열 진입 ({rsi:.1f})")
        elif rsi > 50:
            score -= 0.2

    # ══ 4. 오실레이터 컨센서스 — ±2.0 ═══════════════════════════════════════
    osc_parts: list[float] = []

    # 스토캐스틱 (±0.8)
    sk, sd = _f(last, "STOCH_K"), _f(last, "STOCH_D")
    psk    = _f(prev, "STOCH_K")
    if sk is not None and sd is not None:
        if sk < 20 and sd < 20:
            osc_parts.append(0.8)
            reasons.append(f"스토캐스틱 과매도 (K:{sk:.1f} D:{sd:.1f})")
        elif sk > 80 and sd > 80:
            osc_parts.append(-0.8)
            reasons.append(f"스토캐스틱 과매수 (K:{sk:.1f} D:{sd:.1f})")
        elif sk > sd and (psk is None or psk <= sd):
            osc_parts.append(0.4)         # K선이 D선 상향 돌파
        elif sk < sd and (psk is None or psk >= sd):
            osc_parts.append(-0.4)
        elif sk > sd:
            osc_parts.append(0.2)
        else:
            osc_parts.append(-0.2)

    # CCI (±0.6)
    cci = _f(last, "CCI")
    if cci is not None:
        if cci < -200:
            osc_parts.append(0.6); reasons.append(f"CCI 극과매도 ({cci:.0f})")
        elif cci < -100:
            osc_parts.append(0.4); reasons.append(f"CCI 과매도 ({cci:.0f})")
        elif cci > 200:
            osc_parts.append(-0.6); reasons.append(f"CCI 극과매수 ({cci:.0f})")
        elif cci > 100:
            osc_parts.append(-0.4); reasons.append(f"CCI 과매수 ({cci:.0f})")
        elif 0 < cci < 100:
            osc_parts.append(0.1)         # 중립 상승권
        else:
            osc_parts.append(-0.1)

    # Williams %R (±0.6)
    wr = _f(last, "WILLIAMS_R")
    if wr is not None:
        if wr < -90:
            osc_parts.append(0.6); reasons.append(f"Williams %R 극과매도 ({wr:.1f})")
        elif wr < -80:
            osc_parts.append(0.4); reasons.append(f"Williams %R 과매도 ({wr:.1f})")
        elif wr > -10:
            osc_parts.append(-0.6); reasons.append(f"Williams %R 극과매수 ({wr:.1f})")
        elif wr > -20:
            osc_parts.append(-0.4); reasons.append(f"Williams %R 과매수 ({wr:.1f})")
        elif wr > -50:
            osc_parts.append(-0.1)
        else:
            osc_parts.append(0.1)

    if osc_parts:
        osc_sum = sum(osc_parts)
        # 세 오실레이터가 방향 일치 시 보너스 (컨센서스 프리미엄)
        if len(osc_parts) == 3:
            if all(v > 0 for v in osc_parts):
                osc_sum += 0.4; reasons.append("오실레이터 3종 매수 합의 → 신뢰도 상승")
            elif all(v < 0 for v in osc_parts):
                osc_sum -= 0.4; reasons.append("오실레이터 3종 매도 합의 → 신뢰도 상승")
        score += max(-2.0, min(2.0, osc_sum))

    # ══ 5. 거래량 지표 (OBV + MFI + 원시 거래량) — ±1.5 ════════════════════
    vol    = float(data["Volume"].iloc[-1])
    vol_ma = _f(last, "Volume_MA20")

    # OBV 추세 (±0.5)
    obv     = _f(last, "OBV")
    obv_ma  = _f(last, "OBV_MA20")
    if obv is not None and obv_ma is not None:
        if obv > obv_ma:
            score += 0.5; reasons.append("OBV > MA20 → 매집 추세 (수급 긍정)")
        else:
            score -= 0.5; reasons.append("OBV < MA20 → 분산 추세 (수급 부정)")

    # MFI (±0.7)
    mfi = _f(last, "MFI")
    if mfi is not None:
        if mfi < 20:
            score += 0.7; reasons.append(f"MFI 과매도 ({mfi:.1f}) → 자금 유입 기대")
        elif mfi < 30:
            score += 0.4
        elif mfi > 80:
            score -= 0.7; reasons.append(f"MFI 과매수 ({mfi:.1f}) → 자금 이탈 주의")
        elif mfi > 70:
            score -= 0.4

    # 원시 거래량 급증 (±0.3 — 보조 역할)
    if vol_ma and vol_ma > 0:
        ratio = vol / vol_ma
        if ratio > 2.5:
            if p_chg > 0:
                score += 0.3; reasons.append(f"거래량 폭증 ({ratio:.1f}x) + 상승 → 강한 매수세")
            else:
                score -= 0.3; reasons.append(f"거래량 폭증 ({ratio:.1f}x) + 하락 → 강한 매도세")
        elif ratio > 1.5:
            score += 0.2 if p_chg > 0 else -0.2

    # ══ 6. ADX 필터 적용 EMA 크로스 — ±2.0 ══════════════════════════════════
    adx     = _f(last, "ADX")
    adx_pos = _f(last, "ADX_POS")
    adx_neg = _f(last, "ADX_NEG")
    ema20   = _f(last, "EMA_20")
    ema50   = _f(last, "EMA_50")
    pema20  = _f(prev, "EMA_20")
    pema50  = _f(prev, "EMA_50")

    # ADX 강도에 따른 신호 신뢰도 가중치
    if adx is not None:
        if adx > 35:    adx_w = 1.0    # 강한 추세 — 이동평균 신호 신뢰
        elif adx > 25:  adx_w = 0.8
        elif adx > 20:  adx_w = 0.6
        elif adx > 15:  adx_w = 0.4
        else:           adx_w = 0.2    # 횡보장 — 이동평균 신호 약화
    else:
        adx_w = 0.5

    if all(v is not None for v in [ema20, ema50, pema20, pema50]):
        cross_up = pema20 < pema50 and ema20 >= ema50
        cross_dn = pema20 > pema50 and ema20 <= ema50
        adx_label = f"ADX:{adx:.0f}" if adx else ""
        if cross_up:
            contrib = round(2.0 * adx_w, 2)
            score += contrib
            reasons.append(f"EMA20 골든크로스 ({adx_label}) → 상승 전환")
        elif cross_dn:
            contrib = round(2.0 * adx_w, 2)
            score -= contrib
            reasons.append(f"EMA20 데드크로스 ({adx_label}) → 하락 전환")
        elif ema20 > ema50:
            contrib = round(0.8 * adx_w, 2)
            score += contrib
            if adx and adx > 20:
                reasons.append(f"EMA 상승 배열 ({adx_label} 추세 {'강' if adx > 30 else '보통'})")
        else:
            contrib = round(0.8 * adx_w, 2)
            score -= contrib
            if adx and adx > 20:
                reasons.append(f"EMA 하락 배열 ({adx_label})")

    # +DI / -DI 방향성 추가 확인
    if adx_pos is not None and adx_neg is not None and adx is not None and adx > 20:
        if adx_pos > adx_neg:
            score += 0.3; reasons.append(f"+DI({adx_pos:.1f}) > -DI({adx_neg:.1f}) → 상승 우세")
        else:
            score -= 0.3; reasons.append(f"-DI({adx_neg:.1f}) > +DI({adx_pos:.1f}) → 하락 우세")

    # ══ 7. 볼린저밴드 + Squeeze — ±1.0 ══════════════════════════════════════
    bb_u  = _f(last, "BB_Upper")
    bb_l  = _f(last, "BB_Lower")
    bb_m  = _f(last, "BB_Middle")
    bb_w  = _f(last, "BB_Width")    # 밴드 폭 (좁을수록 Squeeze)
    if bb_u and bb_l and bb_m:
        if price < bb_l:
            score += 1.0; reasons.append(f"볼린저 하단 이탈 ({price:,.0f} < {bb_l:,.0f}) → 반등 기대")
        elif price > bb_u:
            score -= 1.0; reasons.append(f"볼린저 상단 돌파 ({price:,.0f} > {bb_u:,.0f}) → 과열 주의")
        else:
            # %B 위치 (0.5 기준 상하)
            bb_pct = (price - bb_l) / (bb_u - bb_l) if (bb_u - bb_l) > 0 else 0.5
            if bb_pct > 0.7:
                score -= 0.3; reasons.append(f"볼린저 상단 70% 이상 위치 (%B:{bb_pct:.2f})")
            elif bb_pct < 0.3:
                score += 0.3; reasons.append(f"볼린저 하단 30% 이하 위치 (%B:{bb_pct:.2f})")

        # Squeeze 감지 (밴드 폭 역대 최소 20% 구간 → 폭발 가능성)
        if bb_w is not None and len(data) >= 20:
            recent_bw = data["BB_Width"].dropna().tail(20)
            if len(recent_bw) >= 5 and bb_w <= float(recent_bw.quantile(0.2)):
                reasons.append("볼린저 밴드 Squeeze (폭발적 변동성 임박)")

    # ══ 8. ROC 모멘텀 확인 — ±0.5 ════════════════════════════════════════════
    roc = _f(last, "ROC")
    if roc is not None:
        if roc > 15:
            score += 0.5; reasons.append(f"ROC 강한 상승 모멘텀 ({roc:.1f}%)")
        elif roc > 5:
            score += 0.2
        elif roc < -15:
            score -= 0.5; reasons.append(f"ROC 강한 하락 모멘텀 ({roc:.1f}%)")
        elif roc < -5:
            score -= 0.2

    # ══ 9. 일목균형표 구름 위치 — ±1.8 ═══════════════════════════════════════
    span_a = _f(last, "ICHI_SPAN_A")
    span_b = _f(last, "ICHI_SPAN_B")
    tenkan = _f(last, "ICHI_TENKAN")
    kijun  = _f(last, "ICHI_KIJUN")
    if span_a and span_b:
        cloud_top = max(span_a, span_b)
        cloud_bot = min(span_a, span_b)
        if price > cloud_top:
            score += 1.5
            reasons.append(f"일목 구름 위 ({price:,.0f} > {cloud_top:,.0f}) → 중기 강세 추세")
        elif price > cloud_bot:
            score += 0.3
            reasons.append(f"일목 구름 내부 → 방향 탐색 중 (중립)")
        else:
            score -= 1.5
            reasons.append(f"일목 구름 아래 ({price:,.0f} < {cloud_bot:,.0f}) → 중기 약세 추세")
    if tenkan is not None and kijun is not None:
        if tenkan > kijun:
            score += 0.3
            reasons.append(f"일목 전환선 > 기준선 ({tenkan:,.0f} > {kijun:,.0f}) → 단기 강세")
        else:
            score -= 0.3
            reasons.append(f"일목 전환선 < 기준선 ({tenkan:,.0f} < {kijun:,.0f}) → 단기 약세")

    # ══ 10. Z-Score 통계적 위치 — ±1.0 ══════════════════════════════════════
    zscore = _f(last, "Z_SCORE")
    if zscore is not None:
        if zscore > 2.5:
            score -= 1.0; reasons.append(f"Z-Score +{zscore:.2f}σ → 통계적 과매수 (평균 회귀 위험)")
        elif zscore > 1.5:
            score -= 0.4
        elif zscore < -2.5:
            score += 1.0; reasons.append(f"Z-Score {zscore:.2f}σ → 통계적 과매도 (평균 회귀 기대)")
        elif zscore < -1.5:
            score += 0.4

    # ══ 11. 다이버전스 (RSI·MACD 히스토그램) — ±1.5 ═════════════════════════
    div = detect_divergence(data)
    if div.get("bearish_rsi") and div.get("bearish_macd"):
        score -= 1.5; reasons.append("RSI·MACD 이중 하락 다이버전스 → 강한 추세 약화 경고")
    elif div.get("bearish_rsi") or div.get("bearish_macd"):
        score -= 0.8; reasons.append("하락 다이버전스 포착 → 추세 약화 경고")
    elif div.get("bullish_rsi") and div.get("bullish_macd"):
        score += 1.5; reasons.append("RSI·MACD 이중 상승 다이버전스 → 강한 추세 전환 기대")
    elif div.get("bullish_rsi") or div.get("bullish_macd"):
        score += 0.8; reasons.append("상승 다이버전스 포착 → 추세 전환 기대")

    # ══ 12. VWAP 멀티 타임프레임 (Shannon) — ±2.0 ════════════════════════════
    # 주간(5봉) / 월간(20봉) / 분기(60봉) VWAP 대비 현재가 위치
    vwap_w = _f(last, "VWAP_W")
    vwap_m = _f(last, "VWAP_M")
    vwap_q = _f(last, "VWAP_Q")

    _vwap_delta = 0.0
    if vwap_w:
        if price > vwap_w:
            _vwap_delta += 0.6
        else:
            _vwap_delta -= 0.6
    if vwap_m:
        if price > vwap_m:
            _vwap_delta += 0.7
        else:
            _vwap_delta -= 0.7
    if vwap_q:
        if price > vwap_q:
            _vwap_delta += 0.7
        else:
            _vwap_delta -= 0.7

    # VWAP 스택 정렬 보너스 (Shannon의 핵심: 단기>중기>장기 = 추세 방향 일치)
    if vwap_w and vwap_m and vwap_q:
        if vwap_w > vwap_m > vwap_q:
            _vwap_delta = min(_vwap_delta + 0.3, 2.0)
            reasons.append(
                f"VWAP 상승 스택 (주{vwap_w:,.0f} > 월{vwap_m:,.0f} > 분기{vwap_q:,.0f}) → 다중 타임프레임 강세 정렬"
            )
        elif vwap_w < vwap_m < vwap_q:
            _vwap_delta = max(_vwap_delta - 0.3, -2.0)
            reasons.append(
                f"VWAP 하락 스택 (주{vwap_w:,.0f} < 월{vwap_m:,.0f} < 분기{vwap_q:,.0f}) → 다중 타임프레임 약세 정렬"
            )

    # 월간 VWAP 돌파/이탈 크로스 (Shannon의 핵심 진입 신호)
    if vwap_m:
        _prev_vwap_m = _f(prev, "VWAP_M")
        if _prev_vwap_m:
            _prev_price = float(data["Close"].iloc[-2])
            if _prev_price < _prev_vwap_m and price > vwap_m:
                _vwap_delta = min(_vwap_delta + 0.3, 2.0)
                reasons.append(f"월간 VWAP 상향 돌파 ({vwap_m:,.0f}) → 수급 균형 회복, 매수 신호")
            elif _prev_price > _prev_vwap_m and price < vwap_m:
                _vwap_delta = max(_vwap_delta - 0.3, -2.0)
                reasons.append(f"월간 VWAP 하향 이탈 ({vwap_m:,.0f}) → 수급 우위 상실, 매도 신호")

    score += max(-2.0, min(2.0, _vwap_delta))

    # ══ 최종 판정 ════════════════════════════════════════════════════════════
    score = max(-10.0, min(10.0, score))  # cap ±10
    s = int(round(score))
    if   s >= 6:  label, badge = "강력 매수", "🟢🟢"
    elif s >= 4:  label, badge = "매수",      "🟢"
    elif s >= 2:  label, badge = "약한 매수", "🔵"
    elif s >= -1: label, badge = "중립/관망", "⚪"
    elif s >= -3: label, badge = "약한 매도", "🟡"
    elif s >= -5: label, badge = "매도",      "🔴"
    else:         label, badge = "강력 매도", "🔴🔴"

    return {
        "score":   round(score, 1),
        "score_int": s,
        "label":   label,
        "badge":   badge,
        "reasons": reasons,
    }


# ─── 거래량 이상 감지 ─────────────────────────────────────────────────────────

def check_volume_anomaly(data: pd.DataFrame, low_ratio: float = 0.05) -> dict:
    """
    최근 1~2 거래일 거래량이 0이거나 20일 평균 대비 low_ratio 미만이면
    거래 정지/주의 신호를 반환한다.

    Returns:
        is_halted (bool): True → 거래 정지/주의 상태
        reason   (str):  표시할 경고 메시지
        recent_vol (int): 최근 2일 평균 거래량
        avg_vol  (float): 20일 평균 거래량
        ratio    (float): recent_vol / avg_vol
    """
    if data.empty or len(data) < 5:
        return {"is_halted": False, "ratio": 1.0, "recent_vol": 0, "avg_vol": 0.0}

    volume = data["Volume"].fillna(0)
    recent_vols = volume.iloc[-2:].tolist()
    recent_avg  = sum(recent_vols) / max(len(recent_vols), 1)

    hist   = volume.iloc[:-2]
    window = min(20, len(hist))
    avg_vol = float(hist.iloc[-window:].mean()) if window > 0 else 0.0

    if avg_vol <= 0:
        return {"is_halted": False, "ratio": 1.0,
                "recent_vol": int(recent_avg), "avg_vol": avg_vol}

    ratio = recent_avg / avg_vol

    if recent_avg == 0:
        return {
            "is_halted": True,
            "reason": "최근 1~2일 거래량 0 — 거래 정지 의심",
            "recent_vol": 0,
            "avg_vol": avg_vol,
            "ratio": 0.0,
        }
    if ratio < low_ratio:
        return {
            "is_halted": True,
            "reason": (
                f"최근 거래량이 20일 평균 대비 {ratio * 100:.1f}% 수준"
                f" ({int(recent_avg):,} vs 평균 {int(avg_vol):,}) — 거래 정지/주의 필요"
            ),
            "recent_vol": int(recent_avg),
            "avg_vol": avg_vol,
            "ratio": ratio,
        }
    return {"is_halted": False, "ratio": ratio,
            "recent_vol": int(recent_avg), "avg_vol": avg_vol}


# ─── 기회비용 지수: Dead Time Check ──────────────────────────────────────────

def check_dead_time(ticker: str, days: int = 14) -> dict:
    """
    최근 N일간 변동성·거래량으로 '지지부진 구간(Dead Time)' 여부 판단.

    조건:
      is_dead  (AND):
        • 14일 일간 수익률 표준편차 < 5%
        • 14일 평균 거래량 < 이전 60일 평균 거래량의 70%
      buy_hold (OR, 독립):
        • 14일 평균 거래량 ≤ 이전 60일 평균 거래량의 80%
        → 매수 권장 유보 플래그 (유동성 부족 시 체결 슬리피지 위험)
    """
    try:
        raw = yf.download(ticker, period="3mo", auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(1)
    except Exception:
        return {
            "is_dead": False, "buy_hold": False,
            "volatility_14d": 0.0, "vol_ratio": 1.0, "message": "",
        }

    if raw.empty or len(raw) < days + 5:
        return {
            "is_dead": False, "buy_hold": False,
            "volatility_14d": 0.0, "vol_ratio": 1.0, "message": "",
        }

    close  = raw["Close"]
    volume = raw["Volume"]

    returns_14d    = close.pct_change().iloc[-days:].dropna()
    volatility_14d = float(returns_14d.std() * 100)   # 일 단위 %

    avg_vol_14d = float(volume.iloc[-days:].mean())
    ref_len     = min(60, len(volume) - days)
    avg_vol_ref = (
        float(volume.iloc[-(days + ref_len):-days].mean())
        if ref_len > 0 else avg_vol_14d
    )
    vol_ratio = avg_vol_14d / max(avg_vol_ref, 1)

    is_dead  = (volatility_14d < 5.0) and (vol_ratio < 0.70)
    buy_hold = vol_ratio <= 0.80   # 매수 권장 유보: 거래량이 평균의 80% 이하

    if is_dead:
        msg = (
            f"⏳ 지지부진 구간: 자금 고착 주의 "
            f"(14일 변동성 {volatility_14d:.1f}% / 거래량 {vol_ratio*100:.0f}% 수준)"
        )
    elif buy_hold:
        msg = (
            f"⚠️ 거래량 부족 — 매수 유보 권장 "
            f"(최근 {days}일 거래량이 기준 대비 {vol_ratio*100:.0f}% 수준, 80% 미달)"
        )
    elif volatility_14d < 5.0:
        msg = f"📉 저변동성 구간 (변동성 {volatility_14d:.1f}%) — 거래량 보통 수준"
    else:
        msg = f"✅ 정상 거래 구간 (변동성 {volatility_14d:.1f}% / 거래량 {vol_ratio*100:.0f}%)"

    return {
        "is_dead":        is_dead,
        "buy_hold":       buy_hold,
        "volatility_14d": round(volatility_14d, 2),
        "avg_vol_14d":    int(avg_vol_14d),
        "avg_vol_ref":    int(avg_vol_ref),
        "vol_ratio":      round(vol_ratio, 3),
        "message":        msg,
    }


# ─── 상승 임계치(Breakout Point) 필터 ────────────────────────────────────────

def check_breakout_signal(data: pd.DataFrame) -> dict:
    """
    기술적 돌파 조건 판단 — 조건 미충족 시 '관망(Wait)' 유지.

    조건 A: 20일 MA 상향 돌파 (전일 종가 < SMA20, 당일 종가 ≥ SMA20)
    조건 B: 당일 거래량 > 전일 거래량 × 200%
    """
    if data.empty or len(data) < 22:
        return {"status": "wait", "detail": "데이터 부족 — 관망"}

    close  = data["Close"]
    volume = data["Volume"]
    sma20  = data["SMA_20"] if "SMA_20" in data.columns else close.rolling(20).mean()

    curr_c = float(close.iloc[-1])
    prev_c = float(close.iloc[-2])
    curr_s = float(sma20.iloc[-1])
    prev_s = float(sma20.iloc[-2])
    curr_v = float(volume.iloc[-1])
    prev_v = float(volume.iloc[-2])

    ma_break  = (prev_c < prev_s) and (curr_c >= curr_s)
    vol_break = (prev_v > 0) and (curr_v >= prev_v * 2.0)

    if ma_break and vol_break:
        return {
            "status": "breakout_both",
            "detail": (
                f"20일 MA 상향 돌파({curr_s:,.0f}) + "
                f"거래량 폭증({curr_v/prev_v:.1f}배) → 강력 진입 신호"
            ),
        }
    if ma_break:
        return {
            "status": "breakout_ma",
            "detail": f"20일 MA 상향 돌파({curr_s:,.0f}) — 거래량 확인 후 진입 권장",
        }
    if vol_break:
        return {
            "status": "breakout_vol",
            "detail": f"거래량 폭증({curr_v/prev_v:.1f}배) — MA 돌파 미확인, 추가 확인 필요",
        }
    return {
        "status": "wait",
        "detail": f"돌파 조건 미충족 (현가 {curr_c:,.0f} / SMA20 {curr_s:,.0f}) → 관망 유지",
    }


# ─── 리스크 관리: 보수적 조정 ────────────────────────────────────────────────

def adjust_risk_conservative(expected: dict) -> dict:
    """
    승률 50% 미만일 때 목표 수익률·손절 라인을 보수적으로 조정.

    규칙:
      - 목표 수익률(M) → M × 0.5 로 하향
      - 손절 라인     → 현재가 대비 −5.0% 로 타이트하게 설정
    """
    if not expected:
        return {}
    win_prob = expected.get("win_prob", 50.0)
    if win_prob >= 50.0:
        return {**expected, "conservative_applied": False}

    M_adj = round(expected.get("expected_return_pct", 0.0) * 0.5, 2)
    return {
        **expected,
        "conservative_applied":  True,
        "conservative_target":   M_adj,
        "conservative_stoploss": -5.0,
        "conservative_note": (
            f"승률 {win_prob:.0f}% (<50%) — 목표 {M_adj:+.1f}% / 손절 −5.0%로 보수 조정"
        ),
    }


# ─── 예상 수익률 & 리스크 분석 ────────────────────────────────────────────────

def calculate_expected_return(
    data: pd.DataFrame,
    signals: dict,
    horizon_days: int = 20,
    ticker: str = "",
    benchmark_returns: "pd.Series | None" = None,
) -> dict:
    """
    기술 신호 + ATR 구간 + 베타 + 매물대 저항 + 켈리 공식 기반 예상 수익률 추정.

    수익률 구간: A(최저) = M - V×1.5 / M(중간) / B(최고) = M + V×1.5
      V = ATR% × √horizon  (시장 위험 계수 1.5 적용)
    베타: benchmark_returns(S&P500 or KOSPI) 대비 상관관계 → M에 반영
    매물대 저항: 목표가 부근 거래량 집중 시 B 하향 조정
    켈리: 승률·RR비로 추천 투자 비중(%) 산출 (하프-켈리)
    """
    if data.empty or not signals:
        return {}

    close   = data["Close"]
    returns = close.pct_change().dropna()
    if len(returns) < 10:
        return {}

    current_price   = float(close.iloc[-1])
    hist_vol_annual = float(returns.std() * np.sqrt(252) * 100)
    daily_drift     = float(returns.mean() * 100)
    momentum_20d    = float((close.iloc[-1] / close.iloc[-21] - 1) * 100) if len(close) >= 21 else 0.0
    score           = signals.get("score", 0)

    # ── ATR (변동폭 V 기반) ───────────────────────────────────────────────────
    if "High" in data.columns and "Low" in data.columns:
        hl  = data["High"] - data["Low"]
        hpc = (data["High"] - close.shift(1)).abs()
        lpc = (data["Low"]  - close.shift(1)).abs()
        tr  = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
        atr_14 = float(tr.rolling(14).mean().iloc[-1])
        if np.isnan(atr_14):
            atr_14 = float(tr.mean())
    else:
        atr_14 = current_price * hist_vol_annual / 100 / np.sqrt(252)
    atr_pct = atr_14 / current_price * 100

    # V = ATR% × √horizon
    V = atr_pct * np.sqrt(horizon_days)

    # ── Beta + 시장 가중치 ─────────────────────────────────────────────────────
    beta, market_weight = 1.0, 1.0
    if benchmark_returns is not None and not benchmark_returns.empty:
        common = returns.index.intersection(benchmark_returns.index)
        if len(common) >= 30:
            rs = returns.loc[common].values.astype(float)
            rb = benchmark_returns.loc[common].values.astype(float)
            cov = np.cov(rs, rb)
            var_b = cov[1, 1]
            beta = float(cov[0, 1] / var_b) if var_b > 0 else 1.0
            beta = max(-2.0, min(3.0, beta))
            recent_bench = float(benchmark_returns.iloc[-20:].add(1).prod() - 1)
            market_weight = max(0.6, min(1.4, 1.0 + recent_bench * 0.5))

    # ── 중간값 M = 기존 수식 × beta_factor × market_weight ────────────────────
    beta_factor         = max(0.5, beta)
    signal_contribution = score * (hist_vol_annual / 252) * horizon_days * 0.35
    momentum_adj        = momentum_20d * 0.15
    base_return         = signal_contribution + momentum_adj + daily_drift * horizon_days
    expected_return     = base_return * beta_factor * market_weight

    # ── 구간: A = M - V×1.5 / B = M + V×1.5 ────────────────────────────────
    return_low      = expected_return - V * 1.5
    return_high_raw = expected_return + V * 1.5

    # ── 매물대 저항 보정 (인라인 간이 VPVR) ─────────────────────────────────
    vpvr_resistance = False
    return_high     = return_high_raw
    if "Volume" in data.columns and len(data) >= 20 and return_high_raw > 0:
        target_price = current_price * (1 + return_high_raw / 100)
        recent_n     = data.tail(min(120, len(data)))
        p_min = float(recent_n["Close"].min())
        p_max = float(recent_n["Close"].max())
        if p_max > p_min and p_min <= target_price <= p_max * 1.05:
            n_bins   = 20
            bin_size = (p_max - p_min) / n_bins
            vol_bins = np.zeros(n_bins)
            for _i in range(len(recent_n)):
                _p = float(recent_n["Close"].iloc[_i])
                _v = float(recent_n["Volume"].iloc[_i])
                _b = min(int((_p - p_min) / bin_size), n_bins - 1)
                vol_bins[_b] += _v
            total_vol = vol_bins.sum()
            if total_vol > 0:
                t_bin = max(0, min(n_bins - 1, int((target_price - p_min) / bin_size)))
                if vol_bins[t_bin] / total_vol > 0.10:
                    vpvr_resistance = True
                    return_high = return_high_raw - V * 0.5

    # ── 켈리 공식 (하프-켈리) ────────────────────────────────────────────────
    win_prob  = max(0.15, min(0.85, 0.5 + score / 25.0))
    loss_side = max(abs(return_low), 0.1)
    rr_ratio  = abs(return_high) / loss_side if return_low < 0 else max(2.0, abs(return_high))
    kelly_full = win_prob - (1.0 - win_prob) / max(rr_ratio, 0.01)
    kelly_pct  = max(0.0, min(50.0, kelly_full * 50.0))  # 하프-켈리, 최대 50%

    # ── 기타 ─────────────────────────────────────────────────────────────────
    max_drawdown  = float(((close / close.cummax()) - 1).min() * 100)
    risk_free     = 3.5 / 252 * 100
    sharpe_approx = ((daily_drift - risk_free) / (returns.std() * 100) * np.sqrt(252)) if returns.std() > 0 else 0.0

    return {
        "expected_return_pct": round(expected_return, 2),
        "return_low":          round(return_low, 2),
        "return_high":         round(return_high, 2),
        "hist_volatility":     round(hist_vol_annual, 2),
        "atr_pct":             round(atr_pct, 2),
        "momentum_20d":        round(momentum_20d, 2),
        "max_drawdown":        round(max_drawdown, 2),
        "sharpe":              round(sharpe_approx, 2),
        "daily_drift":         round(daily_drift, 3),
        "beta":                round(beta, 2),
        "market_weight":       round(market_weight, 2),
        "vpvr_resistance":     vpvr_resistance,
        "kelly_pct":           round(kelly_pct, 1),
        "win_prob":            round(win_prob * 100, 1),
    }


# ─── 시장 급등/급락 종목 ──────────────────────────────────────────────────────

def get_market_movers(tickers_dict: dict) -> pd.DataFrame:
    """여러 종목의 전일 대비 등락률 계산"""
    rows = []
    for name, ticker in tickers_dict.items():
        try:
            d = yf.download(ticker, period="5d", auto_adjust=True, progress=False)
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.droplevel(1)
            if len(d) < 2:
                continue
            price      = float(d["Close"].iloc[-1])
            prev_price = float(d["Close"].iloc[-2])
            chg        = (price - prev_price) / prev_price * 100
            volume     = int(d["Volume"].iloc[-1])
            rows.append({
                "종목명": name, "티커": ticker,
                "현재가": price, "등락률(%)": round(chg, 2), "거래량": volume
            })
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("등락률(%)", ascending=False).reset_index(drop=True)


def _yf_movers_fallback(top_n: int = 10) -> tuple:
    """FDR 실패 시 yfinance로 하드코딩 종목 등락률 조회"""
    all_stocks = {**KOSPI_STOCKS, **KOSDAQ_STOCKS}
    tickers    = list(all_stocks.values())
    name_map   = {v: k for k, v in all_stocks.items()}
    mkt_map    = {v: ("KOSPI" if v.endswith(".KS") else "KOSDAQ") for v in tickers}

    try:
        raw = yf.download(
            tickers, period="2d", auto_adjust=True, progress=False, group_by="ticker"
        )
    except Exception:
        return pd.DataFrame(), pd.DataFrame()

    rows = []
    single = len(tickers) == 1
    for ticker in tickers:
        try:
            d = raw if single else raw[ticker]
            if isinstance(d.columns, pd.MultiIndex):
                d = d.droplevel(1, axis=1)
            d = d.dropna(subset=["Close"])
            if len(d) < 2:
                continue
            price = float(d["Close"].iloc[-1])
            prev  = float(d["Close"].iloc[-2])
            chg   = (price - prev) / prev * 100
            vol   = int(d["Volume"].iloc[-1]) if "Volume" in d.columns else 0
            rows.append({
                "종목명":   name_map.get(ticker, ticker),
                "티커":     ticker,
                "현재가":   price,
                "등락률(%)": round(chg, 2),
                "거래량":   vol,
                "시장":     mkt_map.get(ticker, ""),
            })
        except Exception:
            continue

    if not rows:
        return pd.DataFrame(), pd.DataFrame()

    df      = pd.DataFrame(rows)
    gainers = df.nlargest(top_n, "등락률(%)").reset_index(drop=True)
    losers  = df.nsmallest(top_n, "등락률(%)").reset_index(drop=True)
    return gainers, losers


def get_full_market_movers(top_n: int = 10) -> tuple:
    """
    FinanceDataReader StockListing으로 KOSPI+KOSDAQ 전체 종목 등락률 조회.
    FDR 실패 시 yfinance 기반 하드코딩 폴백 사용.
    반환: (gainers_df, losers_df) — 각 top_n개, 컬럼: 종목명·티커·현재가·등락률(%)·거래량·시장
    """
    rows = []
    try:
        import FinanceDataReader as fdr
        for market, suffix in [("KOSPI", "KS"), ("KOSDAQ", "KQ")]:
            try:
                listing = fdr.StockListing(market)
                if listing is None or listing.empty:
                    continue
                for _, row in listing.iterrows():
                    try:
                        code  = str(row.get("Code", "")).strip().zfill(6)
                        name  = str(row.get("Name", "")).strip()
                        chg   = float(row.get("ChagesRatio", 0) or 0)
                        price = float(row.get("Close", 0) or 0)
                        vol   = int(row.get("Volume", 0) or 0)
                        if not code or not name or price == 0:
                            continue
                        rows.append({
                            "종목명": name,
                            "티커": f"{code}.{suffix}",
                            "현재가": price,
                            "등락률(%)": round(chg, 2),
                            "거래량": vol,
                            "시장": market,
                        })
                    except Exception:
                        continue
            except Exception:
                continue
    except ImportError:
        pass

    if rows:
        df      = pd.DataFrame(rows)
        gainers = df.nlargest(top_n, "등락률(%)").reset_index(drop=True)
        losers  = df.nsmallest(top_n, "등락률(%)").reset_index(drop=True)
        return gainers, losers

    return _yf_movers_fallback(top_n)


# ─── 환율 정보 ────────────────────────────────────────────────────────────────

def get_exchange_rates() -> dict:
    """주요 환율 및 전일 대비 변동 반환"""
    result = {}
    for name, sym in EXCHANGE_PAIRS.items():
        try:
            d = yf.Ticker(sym).history(period="5d")
            if len(d) >= 2:
                rate   = float(d["Close"].iloc[-1])
                prev   = float(d["Close"].iloc[-2])
                change = (rate - prev) / prev * 100
                result[name] = {"rate": round(rate, 2), "change": round(change, 3)}
        except Exception:
            result[name] = {"rate": 0.0, "change": 0.0}
    return result


def get_investor_trading_naver(ticker: str) -> dict:
    """
    Naver Finance frgn.naver 페이지에서 최근 영업일 투자자별 매매 동향 조회.
    (pykrx KRX 로그인 불필요)
    반환: {date, 외국인, 기관, 개인} — 단위: 주(株)
    """
    code = ticker.split(".")[0]
    if not code.isdigit():
        return {}
    if not HAS_BS4:
        return {}

    import requests
    from bs4 import BeautifulSoup

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Referer": f"https://finance.naver.com/item/main.naver?code={code}",
    }

    def _parse_int(text: str):
        t = text.strip().replace(",", "").replace("+", "").replace("\xa0", "").replace(" ", "")
        if not t or t in ("-", ""):
            return None
        try:
            return int(t)
        except ValueError:
            return None

    # 외국인/기관 순매수 — frgn.naver 테이블
    # 컬럼: [날짜, 종가, 전일비, 등락률, 거래량, 외국인순매수, 기관순매수, 보유주수, 보유율]
    frgn_url = f"https://finance.naver.com/item/frgn.naver?code={code}"
    frgn_foreign, frgn_inst = None, None
    frgn_date = ""
    try:
        resp = requests.get(frgn_url, headers=headers, timeout=10)
        resp.encoding = "euc-kr"
        soup = BeautifulSoup(resp.text, "html.parser")
        # type2 테이블이 2개 — 두 번째(데이터 행이 많은) 테이블 사용
        tables = soup.find_all("table", {"class": "type2"})
        table = tables[1] if len(tables) >= 2 else (tables[0] if tables else None)
        if table:
            for tr in table.find_all("tr"):
                tds = tr.find_all("td")
                if len(tds) >= 7:
                    date_txt = tds[0].text.strip()
                    if len(date_txt) == 10 and date_txt[4] == ".":
                        frgn_date    = date_txt.replace(".", "")[:8]
                        frgn_foreign = _parse_int(tds[5].text)
                        frgn_inst    = _parse_int(tds[6].text)
                        break
    except Exception:
        pass

    if frgn_foreign is None and frgn_inst is None:
        return {}

    result = {"date": frgn_date, "외국인": frgn_foreign, "기관합계": frgn_inst}
    if frgn_foreign is not None and frgn_inst is not None:
        result["개인"] = -(frgn_foreign + frgn_inst)
    return result


# ─── 추천 종목 종합 분석 ──────────────────────────────────────────────────────

def _composite_score(tech: float, ret_pct: float, sharpe: float) -> float:
    """
    기술·수익률·샤프 종합 점수 계산 (범위 약 ±7).

    가중치:
      기술점수   50 %  — generate_signals 결과 (-10 ~ +10)
      수익률점수 30 %  — 예상수익률(%)을 ±5 스케일로 변환
                         +10% → +5 / -10% → -5 (clamp)
      샤프점수   20 %  — 샤프지수를 ±3 스케일로 변환
                         sharpe 2.0 → +3 / sharpe -2.0 → -3 (clamp)

    수익률이 마이너스면 기술 점수가 아무리 높아도 종합 점수는 반드시 하락.
    """
    ret_score    = max(-5.0, min(5.0,  ret_pct / 2.0))
    sharpe_score = max(-3.0, min(3.0,  sharpe  * 1.5))
    composite    = tech * 0.50 + ret_score * 0.30 + sharpe_score * 0.20
    return round(composite, 2)


def _composite_label(score: float) -> tuple[str, str]:
    """종합점수 → (레이블, 배지)"""
    s = score
    if   s >= 4.5: return "강력 추천", "🟢🟢"
    elif s >= 3.0: return "추천",     "🟢"
    elif s >= 1.5: return "관심",     "🔵"
    elif s >= 0.0: return "중립",     "⚪"
    elif s >= -2.0: return "주의",    "🟡"
    elif s >= -4.0: return "비추천",  "🔴"
    else:           return "강력비추", "🔴🔴"


def get_recommendations(tickers_dict: dict) -> pd.DataFrame:
    """
    여러 종목을 분석하여 종합 점수 순으로 추천 목록 생성.
    종합점수 = get_hybrid_signal 동일 공식 — 기술점수(70%) + 뉴스감성(30%)
    범위: -10 ~ +10
    """
    rows = []
    for name, ticker in tickers_dict.items():
        try:
            data = get_stock_data(ticker, "3mo")
            if data.empty:
                continue
            sig  = generate_signals(data)
            exp  = calculate_expected_return(data, sig)
            close = data["Close"]

            price   = float(close.iloc[-1])
            chg_1d  = float((close.iloc[-1] / close.iloc[-2] - 1) * 100) if len(close) >= 2 else 0.0
            tech    = float(sig.get("score", 0))
            ret_pct = float(exp.get("expected_return_pct", 0))
            sharpe  = float(exp.get("sharpe", 0))

            # 뉴스 감성 키워드 분석 (yfinance 뉴스 활용, 실패 시 중립 0.0)
            try:
                raw_news = yf.Ticker(ticker).news or []
                news_items = []
                for it in raw_news[:10]:
                    c = it.get("content", it)
                    news_items.append({
                        "title":     c.get("title", it.get("title", "")),
                        "summary":   c.get("description", ""),
                        "pub_date":  "",
                        "publisher": (c.get("provider", {}).get("displayName") or it.get("publisher", "")),
                    })
                news_result = analyze_news_sentiment_keywords(news_items, ticker, name)
            except Exception:
                news_result = {"score": 0.0}
            news_score = float(news_result.get("score", 0.0))

            hybrid     = get_hybrid_signal(tech, news_score)
            comp       = hybrid["hybrid_score"]
            comp_label = hybrid["label"]
            comp_badge = hybrid["badge"]

            # ── 샤프 가드: 예상수익률 ≥50% 이지만 샤프지수 < 0.5 → '주의' 강제 ──
            # 고수익 기대치가 위험 조정 후 실질 매력이 없는 종목을 걸러냄
            sharpe_warn = False
            if ret_pct >= 50.0 and sharpe < 0.5:
                comp_label = "주의"
                comp_badge = "🟡"
                sharpe_warn = True

            # ── Dead-time 거래량 체크 (매수 유보) ────────────────────────────
            dt = check_dead_time(ticker)
            buy_hold_flag = dt.get("buy_hold", False)
            vol_ratio_val = dt.get("vol_ratio", 1.0)

            rows.append({
                "종목명":         name,
                "티커":           ticker,
                "현재가":         price,
                "등락률(1일)%":   round(chg_1d, 2),
                "종합추천":       f"{comp_badge} {comp_label}",
                "종합점수":       comp,
                "기술점수":       tech,
                "뉴스점수":       round(news_score, 2),
                "예상수익률(%)":  ret_pct,
                "변동성(%)":      round(float(exp.get("hist_volatility", 0)), 1),
                "모멘텀(20일)%":  round(float(exp.get("momentum_20d", 0)), 2),
                "샤프지수":       round(sharpe, 2),
                "샤프경고":       sharpe_warn,
                "거래량비율(%)":  round(vol_ratio_val * 100, 1),
                "매수유보":       buy_hold_flag,
            })
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values("종합점수", ascending=False)
        .reset_index(drop=True)
    )


# ─── 유틸 ─────────────────────────────────────────────────────────────────────

def _f(row, col):
    """Series 행에서 float 값을 안전하게 추출"""
    if col not in row.index:
        return None
    val = row[col]
    if pd.isna(val):
        return None
    return float(val)


def _find_local_peaks(series: pd.Series, order: int = 5) -> list:
    """scipy 없이 로컬 고점 인덱스 목록 반환"""
    vals = series.values.astype(float)
    n    = len(vals)
    peaks = []
    for i in range(order, n - order):
        w_max = max(vals[max(0, i - order):i + order + 1])
        if not np.isnan(vals[i]) and vals[i] == w_max:
            peaks.append(i)
    return peaks


def _find_local_troughs(series: pd.Series, order: int = 5) -> list:
    """scipy 없이 로컬 저점 인덱스 목록 반환"""
    vals = series.values.astype(float)
    n    = len(vals)
    troughs = []
    for i in range(order, n - order):
        w_min = min(vals[max(0, i - order):i + order + 1])
        if not np.isnan(vals[i]) and vals[i] == w_min:
            troughs.append(i)
    return troughs


def detect_divergence(data: pd.DataFrame, lookback: int = 40, order: int = 5) -> dict:
    """
    주가 vs RSI / MACD 히스토그램 다이버전스 감지.
    Bearish: 가격 고점 상승 + 지표 고점 하락 → 추세 약화 경고
    Bullish: 가격 저점 하락 + 지표 저점 상승 → 반등 기대 신호
    """
    result = {
        "bearish_rsi": False, "bullish_rsi": False,
        "bearish_macd": False, "bullish_macd": False,
        "descriptions": [],
    }
    if len(data) < lookback + order + 5:
        return result

    recent = data.tail(lookback).reset_index(drop=True)
    close  = recent["Close"].astype(float)

    for ind_col, label in [("RSI", "RSI"), ("MACD_Hist", "MACD 히스토그램")]:
        if ind_col not in recent.columns:
            continue
        ind = recent[ind_col].astype(float)

        price_peaks   = _find_local_peaks(close, order=order)
        price_troughs = _find_local_troughs(close, order=order)

        # Bearish: 가격 고점↑ + 지표 고점↓
        if len(price_peaks) >= 2:
            p1, p2 = price_peaks[-2], price_peaks[-1]
            if (p1 < len(ind) and p2 < len(ind) and
                    float(close.iloc[p2]) > float(close.iloc[p1]) and
                    float(ind.iloc[p2])   < float(ind.iloc[p1])):
                if ind_col == "RSI":
                    result["bearish_rsi"]  = True
                else:
                    result["bearish_macd"] = True
                result["descriptions"].append(
                    f"⚠️ 하락 다이버전스({label}): 가격 고점↑ vs {label} 고점↓"
                )

        # Bullish: 가격 저점↓ + 지표 저점↑
        if len(price_troughs) >= 2:
            t1, t2 = price_troughs[-2], price_troughs[-1]
            if (t1 < len(ind) and t2 < len(ind) and
                    float(close.iloc[t2]) < float(close.iloc[t1]) and
                    float(ind.iloc[t2])   > float(ind.iloc[t1])):
                if ind_col == "RSI":
                    result["bullish_rsi"]  = True
                else:
                    result["bullish_macd"] = True
                result["descriptions"].append(
                    f"✅ 상승 다이버전스({label}): 가격 저점↓ vs {label} 저점↑"
                )

    return result


def calculate_vpvr(data: pd.DataFrame, n_bins: int = 20) -> dict:
    """
    가격대별 거래량 프로파일 (Volume Profile Visible Range).
    POC(거래량 집중가), 현재가 위 최대 매물벽(저항), 아래 최대 매물대(지지) 반환.
    """
    if data.empty or len(data) < 10:
        return {}
    close  = data["Close"].astype(float)
    volume = data["Volume"].astype(float)

    p_min, p_max = float(close.min()), float(close.max())
    if p_max <= p_min:
        return {}

    bins        = np.linspace(p_min, p_max, n_bins + 1)
    vol_profile = np.zeros(n_bins)
    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (close >= bins[i]) & (close < bins[i + 1])
        else:
            mask = (close >= bins[i])
        vol_profile[i] = float(volume[mask].sum())

    poc_idx   = int(np.argmax(vol_profile))
    poc_price = float((bins[poc_idx] + bins[poc_idx + 1]) / 2)
    current   = float(close.iloc[-1])

    res_level, res_vol = None, 0.0
    sup_level, sup_vol = None, 0.0
    for i in range(n_bins):
        center = float((bins[i] + bins[i + 1]) / 2)
        if center > current and vol_profile[i] > res_vol:
            res_level, res_vol = center, float(vol_profile[i])
        elif center < current and vol_profile[i] > sup_vol:
            sup_level, sup_vol = center, float(vol_profile[i])

    return {
        "poc_price":        round(poc_price, 2),
        "poc_volume":       round(float(vol_profile[poc_idx]), 0),
        "bins":             bins.tolist(),
        "volumes":          vol_profile.tolist(),
        "current_price":    current,
        "resistance_level": round(res_level, 2) if res_level else None,
        "support_level":    round(sup_level, 2) if sup_level else None,
    }


def get_advanced_analysis(data: pd.DataFrame) -> dict:
    """
    고급 기술 분석 종합.
      추세 점수  (0–100): EMA 배열 + ADX + 일목균형표
      탄력 점수  (0–100): MACD + RSI + ROC + CCI
      에너지 점수(0–100): OBV + MFI
    + 다이버전스 · Z-Score · VPVR · 일목균형표 구름 해석
    """
    _empty = {
        "trend_score": 50.0, "momentum_score": 50.0, "volume_score": 50.0,
        "divergence": {}, "zscore": None, "vpvr": {}, "ichimoku": {}, "summary_items": [],
    }
    if data.empty or len(data) < 30:
        return _empty

    last  = data.iloc[-1]
    close = data["Close"].astype(float)
    price = float(close.iloc[-1])

    # ── 추세 점수 (±6 → 0–100) ────────────────────────────────────────────
    t = 0.0
    ema20  = _f(last, "EMA_20")  or price
    ema50  = _f(last, "EMA_50")  or price
    ema200 = _f(last, "EMA_200")
    if ema20 > ema50:   t += 1.5
    else:               t -= 1.5
    if ema200:
        if price > ema200: t += 1.5
        else:              t -= 1.5

    adx     = _f(last, "ADX")
    adx_pos = _f(last, "ADX_POS")
    adx_neg = _f(last, "ADX_NEG")
    if adx:
        if adx > 30:   t += 1.0
        elif adx > 20: t += 0.5
        else:          t -= 0.5
    if adx_pos is not None and adx_neg is not None:
        t += 0.5 if adx_pos > adx_neg else -0.5

    span_a = _f(last, "ICHI_SPAN_A")
    span_b = _f(last, "ICHI_SPAN_B")
    tenkan = _f(last, "ICHI_TENKAN")
    kijun  = _f(last, "ICHI_KIJUN")
    if span_a and span_b:
        cloud_top = max(span_a, span_b)
        cloud_bot = min(span_a, span_b)
        if price > cloud_top:   t += 1.5
        elif price > cloud_bot: t += 0.3
        else:                   t -= 1.5
    if tenkan and kijun:
        t += 0.5 if tenkan > kijun else -0.5

    trend_score = max(0.0, min(100.0, 50.0 + t / 6.0 * 50.0))

    # ── 탄력 점수 (±5 → 0–100) ────────────────────────────────────────────
    m = 0.0
    macd_val = _f(last, "MACD")
    macd_sig = _f(last, "MACD_Signal")
    rsi      = _f(last, "RSI")
    roc      = _f(last, "ROC")
    cci      = _f(last, "CCI")

    if macd_val is not None and macd_sig is not None:
        m += 1.5 if macd_val > macd_sig else -1.5
    if rsi is not None:
        if rsi < 30:    m += 2.0
        elif rsi < 45:  m += 0.8
        elif rsi > 70:  m -= 2.0
        elif rsi > 55:  m -= 0.8
    if roc is not None:
        if roc > 10:    m += 0.8
        elif roc > 0:   m += 0.3
        elif roc < -10: m -= 0.8
        else:           m -= 0.3
    if cci is not None:
        if cci < -100:  m += 0.7
        elif cci > 100: m -= 0.7

    momentum_score = max(0.0, min(100.0, 50.0 + m / 5.0 * 50.0))

    # ── 에너지 점수 (±2 → 0–100) ──────────────────────────────────────────
    v = 0.0
    obv    = _f(last, "OBV")
    obv_ma = _f(last, "OBV_MA20")
    mfi    = _f(last, "MFI")

    if obv is not None and obv_ma is not None:
        v += 1.0 if obv > obv_ma else -1.0
    if mfi is not None:
        if mfi < 20:   v += 1.0
        elif mfi < 40: v += 0.4
        elif mfi > 80: v -= 1.0
        elif mfi > 60: v -= 0.4

    volume_score = max(0.0, min(100.0, 50.0 + v / 2.0 * 50.0))

    # ── 보조 지표 ──────────────────────────────────────────────────────────
    zscore     = _f(last, "Z_SCORE")
    divergence = detect_divergence(data)
    vpvr       = calculate_vpvr(data)

    # ── 일목균형표 해석 ────────────────────────────────────────────────────
    ichimoku = {}
    if span_a and span_b:
        cloud_top = max(span_a, span_b)
        cloud_bot = min(span_a, span_b)
        thick_pct = abs(span_a - span_b) / max(span_a, span_b) * 100
        if price > cloud_top:
            ichi_signal, ichi_color = "☁️ 구름 위 (강세)", "#a5d6a7"
        elif price > cloud_bot:
            ichi_signal, ichi_color = "☁️ 구름 내 (중립)", "#fff176"
        else:
            ichi_signal, ichi_color = "☁️ 구름 아래 (약세)", "#ef9a9a"
        ichimoku = {
            "signal": ichi_signal, "color": ichi_color,
            "cloud_top": cloud_top, "cloud_bot": cloud_bot,
            "thickness_pct": round(thick_pct, 1),
            "bullish_cloud": bool(span_a > span_b),
            "tenkan": tenkan, "kijun": kijun,
        }

    # ── 스코어보드 요약 항목 ───────────────────────────────────────────────
    items = []
    div = divergence
    if div.get("bearish_rsi") or div.get("bearish_macd"):
        items.append({"항목": "다이버전스", "상태": "⚠️ 하락 다이버전스 포착", "색상": "#ffcc80"})
    elif div.get("bullish_rsi") or div.get("bullish_macd"):
        items.append({"항목": "다이버전스", "상태": "✅ 상승 다이버전스 포착", "색상": "#a5d6a7"})
    else:
        items.append({"항목": "다이버전스", "상태": "—  감지 없음", "색상": "#757575"})

    if vpvr.get("resistance_level"):
        r      = vpvr["resistance_level"]
        r_dist = abs(r - price) / price * 100
        items.append({
            "항목": "매물대 저항",
            "상태": f"🛑 {r:,.0f}원  (현재가 +{r_dist:.1f}%)",
            "색상": "#ef9a9a",
        })
    if vpvr.get("poc_price"):
        poc     = vpvr["poc_price"]
        poc_d   = (poc - price) / price * 100
        poc_dir = "위" if poc > price else "아래"
        items.append({
            "항목": "POC (핵심매물)",
            "상태": f"🎯 {poc:,.0f}원  ({abs(poc_d):.1f}% {poc_dir})",
            "색상": "#80cbc4",
        })

    if zscore is not None:
        if zscore > 2.5:
            items.append({"항목": "Z-Score", "상태": f"📉 +{zscore:.2f}σ  통계적 과매수", "색상": "#ef9a9a"})
        elif zscore < -2.5:
            items.append({"항목": "Z-Score", "상태": f"📈 {zscore:.2f}σ  통계적 과매도", "색상": "#a5d6a7"})
        else:
            items.append({"항목": "Z-Score", "상태": f"〰️ {zscore:.2f}σ  정상 범위", "색상": "#757575"})

    atr_val = _f(last, "ATR")
    if atr_val:
        atr_stop = price - 2.0 * atr_val
        items.append({
            "항목": "변동성 (ATR)",
            "상태": f"⚡ ATR {atr_val:,.2f}  |  ATR×2 손절 {atr_stop:,.0f}",
            "색상": "#fff176",
        })

    if ichimoku.get("signal"):
        thick_txt = f"  |  구름 두께 {ichimoku['thickness_pct']:.1f}%" if ichimoku.get("thickness_pct") else ""
        items.append({
            "항목": "일목균형표",
            "상태": ichimoku["signal"] + thick_txt,
            "색상": ichimoku["color"],
        })

    return {
        "trend_score":    round(trend_score, 1),
        "momentum_score": round(momentum_score, 1),
        "volume_score":   round(volume_score, 1),
        "divergence":     divergence,
        "zscore":         zscore,
        "vpvr":           vpvr,
        "ichimoku":       ichimoku,
        "summary_items":  items,
    }


# ─── 상위 종목 리스트 ─────────────────────────────────────────────────────────

def get_top_kospi_stocks(n: int = 500) -> dict:
    """
    시가총액 상위 KOSPI 종목 반환
    반환 형태: {"삼성전자": "005930.KS", ...}
    """
    try:
        import FinanceDataReader as fdr
        df = fdr.StockListing("KOSPI")
        df = df.dropna(subset=["Name", "Code", "Marcap"])
        df = df[df["Marcap"] > 0].sort_values("Marcap", ascending=False).head(n)
        result = {}
        for _, row in df.iterrows():
            code = str(row["Code"]).strip().zfill(6)
            name = str(row["Name"]).strip()
            if name and code:
                result[name] = f"{code}.KS"
        return result if result else KOSPI_STOCKS
    except Exception:
        return KOSPI_STOCKS


def get_top_kosdaq_stocks(n: int = 500) -> dict:
    """
    시가총액 상위 KOSDAQ 종목 반환
    반환 형태: {"에코프로비엠": "247540.KQ", ...}
    """
    try:
        import FinanceDataReader as fdr
        df = fdr.StockListing("KOSDAQ")
        df = df.dropna(subset=["Name", "Code", "Marcap"])
        df = df[df["Marcap"] > 0].sort_values("Marcap", ascending=False).head(n)
        result = {}
        for _, row in df.iterrows():
            code = str(row["Code"]).strip().zfill(6)
            name = str(row["Name"]).strip()
            if name and code:
                result[name] = f"{code}.KQ"
        return result if result else {}
    except Exception:
        return {}


def get_top_us_stocks(n: int = 503) -> dict:
    """
    S&P 500 구성 종목 반환 (시장 대표 500개)
    반환 형태: {"Apple": "AAPL", ...}
    """
    try:
        import FinanceDataReader as fdr
        df = fdr.StockListing("S&P500")
        df = df.dropna(subset=["Name", "Symbol"]).head(n)
        result = {}
        for _, row in df.iterrows():
            sym  = str(row["Symbol"]).strip()
            name = str(row["Name"]).strip()
            if sym and name:
                result[name] = sym
        return result if result else US_STOCKS
    except Exception:
        return US_STOCKS


def get_top_nasdaq_stocks(n: int = 500) -> dict:
    """
    나스닥 상장 종목 반환 (시가총액 상위 n개)
    반환 형태: {"Apple": "AAPL", ...}
    """
    try:
        import FinanceDataReader as fdr
        df = fdr.StockListing("NASDAQ")
        df = df.dropna(subset=["Name", "Symbol"])
        # 시가총액 컬럼이 있으면 정렬, 없으면 그냥 상위 n개
        if "Marcap" in df.columns:
            df = df[df["Marcap"] > 0].sort_values("Marcap", ascending=False)
        result = {}
        for _, row in df.head(n).iterrows():
            sym  = str(row["Symbol"]).strip()
            name = str(row["Name"]).strip()
            if sym and name and sym.isalpha():   # 정상 티커만
                result[name] = sym
        return result if result else US_STOCKS
    except Exception:
        return US_STOCKS


def get_us_stock_list() -> dict:
    """
    S&P500 + 나스닥 통합 종목 이름→티커 매핑 (이름 검색용)
    반환 형태: {"Apple (AAPL) [S&P500]": "AAPL", ...}
    """
    try:
        import FinanceDataReader as fdr
        combined: dict[str, str] = {}

        for market_tag, listing_key in [("S&P500", "S&P500"), ("NASDAQ", "NASDAQ")]:
            try:
                df = fdr.StockListing(listing_key).dropna(subset=["Name", "Symbol"])
                df["Symbol"] = df["Symbol"].astype(str).str.strip()
                df["Name"]   = df["Name"].astype(str).str.strip()
                mask = (
                    df["Symbol"].ne("") & df["Name"].ne("") &
                    df["Symbol"].str.replace(".", "", regex=False).str.isalpha()
                )
                df = df[mask]
                df["_display"] = df["Name"] + " (" + df["Symbol"] + ") [" + market_tag + "]"
                combined.update(dict(zip(df["_display"], df["Symbol"])))
            except Exception:
                continue

        return combined if combined else {
            f"{n} ({s}) [기본]": s for n, s in US_STOCKS.items()
        }
    except Exception:
        return {f"{n} ({s}) [기본]": s for n, s in US_STOCKS.items()}


# ─── KRX 전체 종목 리스트 (이름 검색용) ──────────────────────────────────────

def _krx_stock_fallback() -> dict:
    """FDR 실패 시 KOSPI+KOSDAQ 하드코딩 폴백"""
    result = {}
    for name, ticker in {**KOSPI_STOCKS, **KOSDAQ_STOCKS}.items():
        code = ticker.split(".")[0]
        result[f"{name} ({code})"] = ticker
    return result


def get_krx_stock_list() -> dict:
    """
    KOSPI + KOSDAQ 전체 종목 이름 → 티커 매핑 반환
    반환 형태: {"삼성전자 (005930)": "005930.KS", ...}
    데이터 소스: FinanceDataReader (KRX 실시간 종목 목록)
    FDR 실패 시 주요 종목 하드코딩 폴백 반환.
    """
    try:
        import FinanceDataReader as fdr
        result = {}
        for market, suffix in [("KOSPI", "KS"), ("KOSDAQ", "KQ")]:
            try:
                df = fdr.StockListing(market)
            except Exception:
                continue
            if df is None or df.empty:
                continue
            df["Code"] = df["Code"].astype(str).str.strip()
            df["Name"] = df["Name"].astype(str).str.strip()
            mask = df["Name"].ne("") & df["Code"].str.len().eq(6) & df["Code"].str.isdigit()
            df = df[mask]
            df["_display"] = df["Name"] + " (" + df["Code"] + ")"
            df["_ticker"]  = df["Code"] + "." + suffix
            result.update(dict(zip(df["_display"], df["_ticker"])))
        return result if result else _krx_stock_fallback()
    except Exception:
        return _krx_stock_fallback()


# ─── KRX ETF 리스트 ──────────────────────────────────────────────────────────

def get_krx_etf_list() -> dict:
    """
    국내 ETF 전체 목록 반환 (FinanceDataReader 기반)
    반환 형태: {"KODEX 200 (069500)": "069500.KS", ...}
    """
    try:
        import FinanceDataReader as fdr
        df = fdr.StockListing("ETF/KR")
        if df.empty:
            raise ValueError("empty")
        code_col = "Symbol" if "Symbol" in df.columns else "Code"
        df[code_col] = df[code_col].astype(str).str.strip().str.zfill(6)
        df["Name"]   = df["Name"].astype(str).str.strip()
        mask = df["Name"].ne("") & df[code_col].str.len().eq(6) & df[code_col].str.isdigit()
        df = df[mask]
        result = {}
        for _, row in df.iterrows():
            code    = row[code_col]
            name    = row["Name"]
            display = f"{name} ({code})"
            result[display] = f"{code}.KS"
        return result if result else _etf_fallback_list()
    except Exception:
        return _etf_fallback_list()


def _etf_fallback_list() -> dict:
    """FDR 실패 시 주요 ETF 하드코딩 폴백"""
    return {f"{v['name']} ({k})": f"{k}.KS" for k, v in _ETF_PORTFOLIO_MAP.items()}


def is_etf_ticker(ticker: str) -> bool:
    """
    ETF 여부 판별 (FDR 네트워크 호출 없이 빠르게 판별).
    _ETF_PORTFOLIO_MAP에 등록된 주요 ETF만 True 반환.
    전체 ETF 목록 대조가 필요할 때는 get_krx_etf_list()를 별도로 사용.
    """
    code = ticker.replace(".KS", "").replace(".KQ", "").strip().zfill(6)
    return code in _ETF_PORTFOLIO_MAP


# ─── 펀더멘털 데이터 ──────────────────────────────────────────────────────────

def _is_krx_ticker(ticker: str) -> bool:
    return ticker.endswith(".KS") or ticker.endswith(".KQ")


def get_naver_fundamentals(code: str, price: float = None) -> dict:
    """
    Naver Finance main.naver per_table에서 PER / PBR / 배당수익률 스크래핑.
    EPS = price / PER,  BPS = price / PBR 로 역산.
    반환: {per, pbr, eps_ttm, bps, div} or {}
    """
    if not code.isdigit() or not HAS_BS4:
        return {}
    import requests
    from bs4 import BeautifulSoup

    url = f"https://finance.naver.com/item/main.naver?code={code}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Referer": "https://finance.naver.com/",
    }

    def _parse_float(text: str):
        import re
        m = re.search(r'-?[\d,]+\.?\d*', text.strip())
        if not m:
            return None
        try:
            return float(m.group().replace(",", ""))
        except ValueError:
            return None

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.encoding = "euc-kr"
        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table", {"class": "per_table"})
        if not table:
            return {}

        rows_data = []
        row_values = []   # 두 번째 값 (EPS/BPS 원화 금액)
        for tr in table.find_all("tr"):
            td = tr.find("td")
            if td:
                parts = [p.strip() for p in td.text.split("\n") if p.strip() and p.strip() != "l"]
                rows_data.append(_parse_float(parts[0]) if parts else None)
                # 마지막 부분이 원화 금액 (EPS/BPS 역산보다 직접값 우선)
                row_values.append(_parse_float(parts[-1]) if len(parts) > 1 else None)

        # rows_data: [PER, 동종업체PER, PBR, 배당수익률(%)]
        # row_values: [EPS(원), 동종업체EPS(원), BPS(원), None]
        per = rows_data[0] if len(rows_data) > 0 else None
        pbr = rows_data[2] if len(rows_data) > 2 else None
        div = rows_data[3] if len(rows_data) > 3 else None   # %

        # EPS/BPS: td 내 직접값 우선, 없으면 price/ratio 역산
        eps_ttm = row_values[0] if (row_values and row_values[0]) else (
            round(price / per, 0) if price and per and per > 0 else None
        )
        bps = row_values[2] if (len(row_values) > 2 and row_values[2]) else (
            round(price / pbr, 0) if price and pbr and pbr > 0 else None
        )

        result = {}
        if per    is not None: result["per"]    = per
        if pbr    is not None: result["pbr"]    = pbr
        if eps_ttm is not None: result["eps_ttm"] = eps_ttm
        if bps    is not None: result["bps"]    = bps
        if div    is not None: result["div"]    = div
        return result
    except Exception:
        return {}


def _krx_fundamental_to_dict(row: dict, ticker: str) -> dict:
    """fundamental_db row → get_fundamental_data 반환 형식으로 변환"""
    return {
        "per":               row.get("per"),
        "pbr":               row.get("pbr"),
        "roe":               None,
        "debt_equity":       None,
        "revenue_growth":    None,
        "earnings_growth":   None,
        "operating_margins": None,
        "w52_high":          None,
        "w52_low":           None,
        "market_cap":        row.get("market_cap"),
        "free_cashflow":     None,
        "eps_ttm":           row.get("eps"),
        "forward_pe":        None,
        "div":               row.get("div"),
        "bps":               row.get("bps"),
        "dps":               row.get("dps"),
        "sector":            "N/A",
        "industry":          "N/A",
        "short_name":        row.get("name", ticker),
        "last_updated":      row.get("last_updated"),
        "source":            "pykrx",
    }


def get_fundamental_data(ticker: str) -> dict:
    """
    KRX 종목(.KS/.KQ): yfinance 조회 후 DB에 PER/PBR/EPS 저장; DB 기존값과 병합
    해외 종목: yfinance 직접 조회
    """
    _db_row = None
    if _is_krx_ticker(ticker):
        try:
            from fundamental_db import get_ticker_fundamental
            _db_row = get_ticker_fundamental(ticker)
        except Exception:
            pass

    try:
        t    = yf.Ticker(ticker)
        info = t.info

        # ── 가격 (info → previousClose → fast_info 순서로 fallback) ─────────────
        price = (info.get("currentPrice")
                 or info.get("regularMarketPrice")
                 or info.get("previousClose"))
        if price is None:
            try:
                price = t.fast_info.last_price
            except Exception:
                pass

        # ── 시가총액 (info → fast_info) ──────────────────────────────────────────
        market_cap = info.get("marketCap")
        if market_cap is None:
            try:
                market_cap = t.fast_info.market_cap
            except Exception:
                pass

        # ── 52주 고/저 (info → fast_info) ───────────────────────────────────────
        w52_high = info.get("fiftyTwoWeekHigh")
        w52_low  = info.get("fiftyTwoWeekLow")
        if w52_high is None:
            try:
                w52_high = t.fast_info.year_high
            except Exception:
                pass
        if w52_low is None:
            try:
                w52_low = t.fast_info.year_low
            except Exception:
                pass

        eps_ttm   = info.get("trailingEps")
        book_val  = info.get("bookValue")
        total_rev = info.get("totalRevenue")

        # PER fallback: price / trailingEps
        per = info.get("trailingPE")
        if per is None and price and eps_ttm and eps_ttm > 0:
            per = round(price / eps_ttm, 2)

        # PBR fallback: price / bookValue
        pbr = info.get("priceToBook")
        if pbr is None and price and book_val and book_val > 0:
            pbr = round(price / book_val, 2)

        # PSR: marketCap / totalRevenue
        psr = None
        if market_cap and total_rev and total_rev > 0:
            psr = round(market_cap / total_rev, 2)

        # ── 재무제표: 영업이익·순이익·OCF·자본 (멀티연도) ────────────────────────
        operating_income = None
        net_income       = None
        ocf              = info.get("operatingCashflow")   # yfinance info 직접값 (최신년)
        roe_history: list[float] = []                       # ROE 연도별 리스트
        # 재무제표 fallback 변수 (info에 없을 때 계산)
        roe_from_fin         = None
        revenue_growth_calc  = None
        earnings_growth_calc = None
        op_margins_calc      = None
        fcf_from_cf          = None

        try:
            fin = t.financials
            bs  = t.balance_sheet
            cf  = t.cashflow

            if fin is not None and not fin.empty:
                for label in ["Operating Income", "EBIT"]:
                    if label in fin.index:
                        operating_income = float(fin.loc[label].iloc[0])
                        break
                for label in ["Net Income", "Net Income Common Stockholders",
                               "Net Income From Continuing Operation Net Minority Interest"]:
                    if label in fin.index:
                        net_income = float(fin.loc[label].iloc[0])
                        break

                # OCF (cashflow 우선, info fallback)
                if cf is not None and not cf.empty:
                    for cf_label in ["Operating Cash Flow", "Total Cash From Operating Activities"]:
                        if cf_label in cf.index:
                            ocf = float(cf.loc[cf_label].iloc[0])
                            break

                # total_rev fallback: info에 없으면 financials에서 추출
                if total_rev is None:
                    for label in ["Total Revenue", "Operating Revenue"]:
                        if label in fin.index:
                            _v = fin.loc[label].iloc[0]
                            if pd.notna(_v):
                                total_rev = float(_v)
                            break

                # revenue_growth fallback: YoY 매출 변화율
                for label in ["Total Revenue", "Operating Revenue"]:
                    if label in fin.index:
                        _rev_row = fin.loc[label].dropna()
                        if len(_rev_row) >= 2:
                            _r0 = float(_rev_row.iloc[0])
                            _r1 = float(_rev_row.iloc[1])
                            if _r1 != 0 and not pd.isna(_r0) and not pd.isna(_r1):
                                revenue_growth_calc = round((_r0 - _r1) / abs(_r1), 6)
                        break

                # earnings_growth fallback: YoY 순이익 변화율
                for label in ["Net Income", "Net Income Common Stockholders",
                               "Net Income From Continuing Operation Net Minority Interest"]:
                    if label in fin.index:
                        _ni_row2 = fin.loc[label].dropna()
                        if len(_ni_row2) >= 2:
                            _n0 = float(_ni_row2.iloc[0])
                            _n1 = float(_ni_row2.iloc[1])
                            if _n1 != 0 and not pd.isna(_n0) and not pd.isna(_n1):
                                earnings_growth_calc = round((_n0 - _n1) / abs(_n1), 6)
                        break

                # operating_margins fallback: 영업이익 / 매출
                if operating_income is not None and total_rev and total_rev > 0:
                    op_margins_calc = round(operating_income / total_rev, 6)

                # FCF fallback: cashflow에서 직접 추출
                if cf is not None and not cf.empty and "Free Cash Flow" in cf.index:
                    _fcf_v = cf.loc["Free Cash Flow"].iloc[0]
                    if pd.notna(_fcf_v):
                        fcf_from_cf = float(_fcf_v)

                # ROE fallback: 순이익 / 자기자본 (단년도)
                if net_income is not None and bs is not None and not bs.empty:
                    for label in ["Common Stock Equity", "Stockholders Equity",
                                   "Total Equity Gross Minority Interest"]:
                        if label in bs.index:
                            _eq = bs.loc[label].iloc[0]
                            if pd.notna(_eq) and float(_eq) > 0:
                                roe_from_fin = round(net_income / float(_eq), 6)
                            break

                # ROE 연도별 계산 (최대 4년)
                if bs is not None and not bs.empty:
                    ni_vals   = None
                    eq_vals   = None
                    for label in ["Net Income", "Net Income Common Stockholders",
                                   "Net Income From Continuing Operation Net Minority Interest"]:
                        if label in fin.index:
                            ni_vals = fin.loc[label]
                            break
                    for label in ["Common Stock Equity", "Stockholders Equity",
                                   "Total Equity Gross Minority Interest"]:
                        if label in bs.index:
                            eq_vals = bs.loc[label]
                            break
                    if ni_vals is not None and eq_vals is not None:
                        common_cols = [c for c in ni_vals.index if c in eq_vals.index]
                        for col in common_cols[:4]:
                            ni_v  = ni_vals.get(col)
                            eq_v  = eq_vals.get(col)
                            if ni_v and eq_v and float(eq_v) > 0 and not pd.isna(ni_v) and not pd.isna(eq_v):
                                roe_history.append(round(float(ni_v) / float(eq_v) * 100, 2))

            # ── EPS 연도별 리스트 (린치 3년 CAGR 직접 계산용) ─────────────────
            # yfinance financials 열 순서: 최신→과거, 역순하여 과거→현재로 저장
            eps_history: list[float] = []
            for _eps_label in ["Diluted EPS", "Basic EPS"]:
                if _eps_label in fin.index:
                    _eps_row = fin.loc[_eps_label].dropna()
                    eps_history = [float(v) for v in reversed(_eps_row.values[:4])]
                    break

            # fallback: Net Income ÷ Diluted Average Shares
            if not eps_history:
                _ni_row     = None
                _shares_row = None
                for _lbl in ["Net Income", "Net Income Common Stockholders",
                              "Net Income From Continuing Operation Net Minority Interest"]:
                    if _lbl in fin.index:
                        _ni_row = fin.loc[_lbl]
                        break
                for _lbl in ["Diluted Average Shares", "Ordinary Shares Number", "Share Issued"]:
                    if _lbl in fin.index:
                        _shares_row = fin.loc[_lbl]
                        break
                    if bs is not None and _lbl in bs.index:
                        _shares_row = bs.loc[_lbl]
                        break
                if _ni_row is not None and _shares_row is not None:
                    _common = [c for c in _ni_row.index if c in _shares_row.index]
                    _raw = []
                    for col in _common[:4]:
                        _ni_v  = _ni_row.get(col)
                        _sh_v  = _shares_row.get(col)
                        if (_ni_v is not None and _sh_v is not None
                                and float(_sh_v) > 0
                                and not pd.isna(_ni_v) and not pd.isna(_sh_v)):
                            _raw.append(float(_ni_v) / float(_sh_v))
                    eps_history = list(reversed(_raw))  # 과거→현재 순
        except Exception:
            eps_history = []

        # ── 자사주 매입 + 배당금 → 주주환원율 ────────────────────────────────────
        buyback_amount = None
        div_paid_amount = None
        shareholder_yield = None
        try:
            cf = t.cashflow
            if cf is not None and not cf.empty:
                for label in ["Repurchase Of Capital Stock", "Common Stock Payments",
                               "Purchase Of Business", "Common Stock Repurchased"]:
                    if label in cf.index:
                        v = cf.loc[label].iloc[0]
                        if pd.notna(v) and float(v) < 0:
                            buyback_amount = abs(float(v))
                            break
                for label in ["Cash Dividends Paid", "Payment Of Dividends"]:
                    if label in cf.index:
                        v = cf.loc[label].iloc[0]
                        if pd.notna(v) and float(v) < 0:
                            div_paid_amount = abs(float(v))
                            break
            if market_cap and market_cap > 0:
                total_return = (buyback_amount or 0) + (div_paid_amount or 0)
                if total_return > 0:
                    shareholder_yield = round(total_return / market_cap * 100, 2)
        except Exception:
            pass

        def _coalesce(*vals):
            for v in vals:
                if v is not None:
                    return v
            return None

        # PSR 재계산: total_rev가 fallback으로 채워졌을 수 있음
        if psr is None and market_cap and total_rev and total_rev > 0:
            psr = round(market_cap / total_rev, 2)

        result = {
            "per":              per,
            "pbr":              pbr,
            "psr":              psr,
            "roe":              _coalesce(info.get("returnOnEquity"), roe_from_fin),
            "roe_history":      roe_history,           # 연도별 ROE(%) 리스트
            "eps_history":      eps_history,           # 연도별 EPS 리스트 (과거→현재, 린치 CAGR용)
            "debt_equity":      info.get("debtToEquity"),
            "revenue_growth":   _coalesce(info.get("revenueGrowth"),  revenue_growth_calc),
            "earnings_growth":  _coalesce(info.get("earningsGrowth"), earnings_growth_calc),
            "operating_margins":_coalesce(info.get("operatingMargins"), op_margins_calc),
            "w52_high":         w52_high,
            "w52_low":          w52_low,
            "market_cap":       market_cap,
            "total_revenue":    total_rev,
            "operating_income": operating_income,
            "net_income":       net_income,
            "free_cashflow":    _coalesce(info.get("freeCashflow"), fcf_from_cf),
            "ocf":              ocf,                   # 영업활동현금흐름
            "buyback_amount":   buyback_amount,        # 자사주 매입액
            "div_paid_amount":  div_paid_amount,       # 배당금 지급액
            "shareholder_yield": shareholder_yield,    # 주주환원율(%)
            "eps_ttm":          eps_ttm,
            "forward_pe":       info.get("forwardPE"),
            "div_yield":        info.get("dividendYield"),
            "sector":           info.get("sector", "N/A"),
            "industry":         info.get("industry", "N/A"),
            "short_name":       info.get("shortName", ticker),
            "source":           "yfinance",
        }

        # KRX 종목: Naver Finance에서 PER/PBR/EPS 보완 후 DB 저장
        if _is_krx_ticker(ticker):
            code = ticker.split(".")[0]
            current_price = result.get("market_cap") and price  # price는 위에서 구함
            naver = get_naver_fundamentals(code, price)
            for k in ("per", "pbr", "eps_ttm", "bps", "div"):
                if result.get(k) is None and naver.get(k) is not None:
                    result[k] = naver[k]
            try:
                from fundamental_db import save_yfinance_fundamental, get_ticker_fundamental
                # Naver 값을 info dict에 주입해 DB 저장 함수 재사용
                merged_info = dict(info)
                if naver.get("per")    is not None: merged_info["trailingPE"]   = naver["per"]
                if naver.get("pbr")    is not None: merged_info["priceToBook"]  = naver["pbr"]
                if naver.get("eps_ttm") is not None: merged_info["trailingEps"] = naver["eps_ttm"]
                if naver.get("bps")    is not None: merged_info["bookValue"]    = naver["bps"]
                if naver.get("div")    is not None: merged_info["dividendYield"] = naver["div"] / 100
                save_yfinance_fundamental(ticker, merged_info)
                # DB에만 있는 값(dps 등) 병합
                db = _db_row or get_ticker_fundamental(ticker)
                if db:
                    if result.get("dps") is None and db.get("dps") is not None:
                        result["dps"] = db["dps"]
            except Exception:
                pass

        return result
    except Exception:
        # yfinance 실패 시 DB 값만으로라도 반환
        if _db_row:
            return _krx_fundamental_to_dict(_db_row, ticker)
        return {}


def calculate_fundamental_score(info: dict, close_price: float = None) -> dict:
    """
    버핏·그레이엄·린치·오닐 투자 법칙 기반 펀더멘털 점수 (가중 4분류 체계)

    가중치 구조:
      성장성   (40%): PEG, 매출성장 — 린치 핵심
      수익성   (30%): ROE 지속성, FCF Yield, OCF/NI 품질 — 버핏 핵심
      안정성   (20%): 그레이엄 공식, 부채비율
      모멘텀   (10%): 52주 고가 근접, 주주환원율 — 오닐·Value-up

    최종 점수: ±8 (각 분류 ±1 정규화 후 가중합 × 8)
    """
    reasons: list[str] = []

    per             = info.get("per")
    pbr             = info.get("pbr")
    roe             = info.get("roe")             # 최신년 ROE (소수, e.g. 0.15 = 15%)
    roe_history     = info.get("roe_history", []) # 연도별 ROE(%) 리스트
    debt_equity     = info.get("debt_equity")
    revenue_growth  = info.get("revenue_growth")
    earnings_growth = info.get("earnings_growth")
    eps_history     = info.get("eps_history", [])   # 연도별 EPS 리스트 (과거→현재)
    w52_high        = info.get("w52_high")
    w52_low         = info.get("w52_low")
    fcf             = info.get("free_cashflow")
    ocf             = info.get("ocf")
    net_income      = info.get("net_income")
    market_cap      = info.get("market_cap")
    sh_yield        = info.get("shareholder_yield")   # 주주환원율(%)
    buyback         = info.get("buyback_amount")
    div_paid        = info.get("div_paid_amount")

    # ── 린치 EPS 3년 CAGR 직접 계산 ────────────────────────────────────────────
    # (금년 EPS / 3년 전 EPS)^(1/3) - 1
    # eps_history: [eps_n-3, eps_n-2, eps_n-1, eps_n] (과거→현재 순)
    eps_cagr_3yr: float | None = None
    eps_cagr_note: str = ""
    if len(eps_history) >= 4:
        eps_old = eps_history[-4]   # 3년 전
        eps_new = eps_history[-1]   # 최신년
        if eps_old > 0 and eps_new > 0:
            eps_cagr_3yr = (eps_new / eps_old) ** (1.0 / 3.0) - 1.0
            eps_cagr_note = (
                f"EPS 3년 CAGR {eps_cagr_3yr*100:.1f}%"
                f"  ({eps_old:.2f} → {eps_new:.2f})"
            )
        elif eps_old <= 0 or eps_new <= 0:
            eps_cagr_note = f"EPS 음수 포함 (CAGR 계산 불가: {eps_old:.2f}→{eps_new:.2f})"
    elif len(eps_history) >= 2:
        # 2~3년치만 있을 때: 가용 기간으로 연환산
        eps_old = eps_history[0]
        eps_new = eps_history[-1]
        n = len(eps_history) - 1
        if eps_old > 0 and eps_new > 0:
            eps_cagr_3yr = (eps_new / eps_old) ** (1.0 / n) - 1.0
            eps_cagr_note = (
                f"EPS {n}년 CAGR {eps_cagr_3yr*100:.1f}%"
                f"  ({eps_old:.2f} → {eps_new:.2f}, {n}년 데이터)"
            )

    # EPS CAGR 우선, 없으면 yfinance earningsGrowth fallback
    effective_growth = eps_cagr_3yr if eps_cagr_3yr is not None else earnings_growth

    g = 0.0   # 성장성 raw score  (cap ±3.5)
    p = 0.0   # 수익성 raw score  (cap ±6.0)
    s = 0.0   # 안정성 raw score  (cap ±3.5)
    m = 0.0   # 모멘텀 raw score  (cap ±2.5)

    # ════════════════════════════════════════════════════════════════
    # 성장성 (Growth) — 린치 핵심, 가중치 40%
    # ════════════════════════════════════════════════════════════════

    # PEG (±2.0) — EPS 3년 CAGR 직접 계산값 우선 사용
    peg = None
    _per_ok    = per is not None and not (isinstance(per, float) and per != per) and per > 0
    _growth_ok = effective_growth is not None and effective_growth > 0
    if _per_ok and _growth_ok:
        peg = per / (effective_growth * 100)
        growth_src = eps_cagr_note if eps_cagr_3yr is not None else f"성장률={earnings_growth*100:.1f}%(yf)"
        if peg < 0.5:
            g += 2.0; reasons.append(f"[린치] PEG={peg:.2f} < 0.5 → 강력 성장 저평가  ({growth_src})")
        elif peg < 1.0:
            g += 1.0; reasons.append(f"[린치] PEG={peg:.2f} < 1.0 → 성장 대비 저평가  ({growth_src})")
        elif peg < 1.5:
            g += 0.3; reasons.append(f"[린치] PEG={peg:.2f}  ({growth_src})")
        elif peg > 2.0:
            g -= 1.0; reasons.append(f"[린치] PEG={peg:.2f} > 2.0 → 성장 대비 고평가  ({growth_src})")
        else:
            reasons.append(f"[린치] PEG={peg:.2f}  ({growth_src})")
    elif eps_cagr_note:
        if not _per_ok:
            reasons.append(f"[린치] {eps_cagr_note} (PER 없어 PEG 계산 불가)")
        else:
            reasons.append(f"[린치] {eps_cagr_note} (EPS 감소세 → PEG 계산 불가)")

    # 매출 성장 (±1.5)
    if revenue_growth is not None:
        rev_pct = revenue_growth * 100
        if rev_pct >= 25:
            g += 1.5; reasons.append(f"[린치] 매출 {rev_pct:.1f}% 급성장 → 텐배거 후보")
        elif rev_pct >= 10:
            g += 0.5; reasons.append(f"[린치] 매출 {rev_pct:.1f}% 성장 → 양호")
        elif rev_pct < -10:
            g -= 1.0; reasons.append(f"[린치] 매출 {rev_pct:.1f}% 감소 → 펀더멘털 악화")

    # ════════════════════════════════════════════════════════════════
    # 수익성/지속성 (Profitability) — 버핏 핵심, 가중치 30%
    # ════════════════════════════════════════════════════════════════

    # ROE 지속성 (±2.0) — 단발성 vs 꾸준함
    roe_mean_val = None
    roe_std_val  = None
    if len(roe_history) >= 2:
        roe_mean_val = float(np.mean(roe_history))
        roe_std_val  = float(np.std(roe_history, ddof=min(1, len(roe_history)-1)))
        if roe_mean_val >= 15 and roe_std_val <= 5:
            p += 2.0
            reasons.append(
                f"[버핏] ROE 지속성 우수: 평균 {roe_mean_val:.1f}%·표준편차 {roe_std_val:.1f}%p "
                f"→ 꾸준한 수익 창출 우량 기업"
            )
        elif roe_mean_val >= 15 and roe_std_val <= 10:
            p += 1.0
            reasons.append(
                f"[버핏] ROE 평균 {roe_mean_val:.1f}% 충족, 편차 {roe_std_val:.1f}%p "
                f"→ 다소 들쭉날쭉하나 수익성 양호"
            )
        elif roe_mean_val >= 15:
            reasons.append(
                f"[버핏] ROE 평균 {roe_mean_val:.1f}%이나 편차 {roe_std_val:.1f}%p 과대 "
                f"→ 단발성 고ROE, 지속성 미흡"
            )
        elif roe_mean_val < 8:
            p -= 1.0
            reasons.append(f"[버핏] ROE 평균 {roe_mean_val:.1f}% → 수익성 부진")
    elif roe is not None:
        # 단년도 데이터만 있을 때
        roe_pct = roe * 100
        if roe_pct >= 20:
            p += 1.5; reasons.append(f"[버핏] ROE {roe_pct:.1f}% ≥ 20% → 우량 기업")
        elif roe_pct >= 15:
            p += 0.8; reasons.append(f"[버핏] ROE {roe_pct:.1f}% ≥ 15% → 버핏 기준 충족")
        elif roe_pct < 5:
            p -= 1.0; reasons.append(f"[버핏] ROE {roe_pct:.1f}% < 5% → 수익성 부진")

    # FCF Yield (±1.5)
    fcf_yield = None
    if fcf and market_cap and market_cap > 0:
        fcf_yield = fcf / market_cap * 100
        if fcf_yield > 8:
            p += 1.5; reasons.append(f"[버핏] FCF Yield {fcf_yield:.1f}% > 8% → 현금창출 탁월")
        elif fcf_yield > 5:
            p += 0.5; reasons.append(f"[버핏] FCF Yield {fcf_yield:.1f}% > 5% → 양호")
        elif fcf_yield < 0:
            p -= 1.0; reasons.append(f"[버핏] FCF Yield {fcf_yield:.1f}% < 0 → 현금소진 경고")

    # OCF / 당기순이익 (±1.5) — 현금흐름 품질 (Accrual 분석)
    ocf_ni_ratio = None
    if ocf and net_income and net_income > 0 and not pd.isna(ocf) and not pd.isna(net_income):
        ocf_ni_ratio = ocf / net_income
        if ocf_ni_ratio > 1.5:
            p += 1.5
            reasons.append(
                f"[버핏/린치] OCF/NI={ocf_ni_ratio:.2f} > 1.5 → 현금흐름 품질 탁월 (이익의 질 높음)"
            )
        elif ocf_ni_ratio > 1.0:
            p += 0.8
            reasons.append(
                f"[버핏/린치] OCF/NI={ocf_ni_ratio:.2f} > 1.0 → 현금흐름 양호 (이익 신뢰성 높음)"
            )
        elif ocf_ni_ratio < 0.5:
            p -= 0.8
            reasons.append(
                f"[주의] OCF/NI={ocf_ni_ratio:.2f} < 0.5 → 이익 대비 현금 부족 (분식·채권 회수 이슈 점검)"
            )
        elif ocf_ni_ratio < 0:
            p -= 1.5
            reasons.append(
                f"[경고] OCF/NI={ocf_ni_ratio:.2f} < 0 → 이익 발생 중 현금 유출 경고 (재무 위험)"
            )

    # ════════════════════════════════════════════════════════════════
    # 안정성 (Stability) — 그레이엄·버핏, 가중치 20%
    # ════════════════════════════════════════════════════════════════

    # 그레이엄: PBR × PER < 22.5 (±2.0)
    gnum = None
    if per and pbr and per > 0 and pbr > 0:
        gnum = per * pbr
        if gnum < 15:
            s += 2.0; reasons.append(f"[그레이엄] PBR×PER={gnum:.1f} < 15 → 강한 저평가")
        elif gnum < 22.5:
            s += 1.0; reasons.append(f"[그레이엄] PBR×PER={gnum:.1f} < 22.5 → 적정 평가")
        elif gnum > 45:
            s -= 2.0; reasons.append(f"[그레이엄] PBR×PER={gnum:.1f} > 45 → 고평가 경고")
        elif gnum > 30:
            s -= 1.0; reasons.append(f"[그레이엄] PBR×PER={gnum:.1f} > 30 → 다소 고평가")

    # 버핏: 부채비율 (±1.5)
    if debt_equity is not None and debt_equity >= 0:
        if debt_equity < 50:
            s += 1.5; reasons.append(f"[버핏] 부채비율 {debt_equity:.0f}% < 50% → 재무 우량")
        elif debt_equity < 100:
            s += 0.5
        elif debt_equity > 200:
            s -= 1.5; reasons.append(f"[버핏] 부채비율 {debt_equity:.0f}% > 200% → 재무 위험")
        elif debt_equity > 100:
            s -= 0.5; reasons.append(f"[버핏] 부채비율 {debt_equity:.0f}% > 100% → 주의 필요")

    # ════════════════════════════════════════════════════════════════
    # 모멘텀/주주환원 (Momentum) — 오닐·Value-up, 가중치 10%
    # ════════════════════════════════════════════════════════════════

    # 오닐 CANSLIM: 52주 고가 근접 (±1.5)
    oneil_ratio = None
    if close_price and w52_high and w52_low and w52_high > w52_low:
        oneil_ratio = close_price / w52_high
        pos = (close_price - w52_low) / (w52_high - w52_low)
        if oneil_ratio >= 0.95:
            m += 1.5; reasons.append(f"[오닐] 52주 고가 {oneil_ratio*100:.1f}% 근접 → 신고가 돌파 모멘텀")
        elif oneil_ratio >= 0.85:
            m += 0.5
        elif pos <= 0.15:
            m += 0.3; reasons.append(f"[오닐] 52주 저가 근접 ({pos*100:.0f}%) → 바닥 반등 기대")

    # 주주환원율 (±1.0) — 배당 + 자사주 소각
    if sh_yield is not None and sh_yield > 0:
        if sh_yield >= 5:
            m += 1.0; reasons.append(f"[주주환원] 배당+자사주 {sh_yield:.1f}% → 주주 가치 우수")
        elif sh_yield >= 3:
            m += 0.5; reasons.append(f"[주주환원] 배당+자사주 {sh_yield:.1f}% → 주주 환원 양호")
        elif sh_yield >= 1:
            m += 0.2

    # ════════════════════════════════════════════════════════════════
    # 가중 합산: 40·30·20·10 정규화 후 ±8 스케일
    # ════════════════════════════════════════════════════════════════
    def _norm(val, cap):
        return max(-1.0, min(1.0, val / cap)) if cap > 0 else 0.0

    score = round(
        (_norm(g, 3.5) * 0.40 + _norm(p, 6.0) * 0.30 +
         _norm(s, 3.5) * 0.20 + _norm(m, 2.5) * 0.10) * 8,
        1
    )

    # ── 세부 서브 점수 (0~100 스케일, UI 표시용) ─────────────────────────────
    growth_sub   = round(50 + _norm(g, 3.5) * 50, 1)
    profit_sub   = round(50 + _norm(p, 6.0) * 50, 1)
    stable_sub   = round(50 + _norm(s, 3.5) * 50, 1)
    moment_sub   = round(50 + _norm(m, 2.5) * 50, 1)

    # ── 거장의 한 줄 평 ───────────────────────────────────────────────────────
    master_verdicts: dict[str, dict] = {}

    # 그레이엄
    if gnum is not None:
        if gnum < 22.5:
            master_verdicts["그레이엄"] = {
                "icon": "✅", "판정": "통과",
                "comment": f"안전마진 확보. PBR×PER={gnum:.1f}으로 그레이엄 기준({22.5}) 이내입니다."
            }
        elif gnum < 35:
            master_verdicts["그레이엄"] = {
                "icon": "⚠️", "판정": "주의",
                "comment": f"안전마진이 부족합니다. PBR×PER={gnum:.1f}이 기준치({22.5})를 초과했습니다."
            }
        else:
            master_verdicts["그레이엄"] = {
                "icon": "🚫", "판정": "경고",
                "comment": f"안전마진 없음. PBR×PER={gnum:.1f}은 현명한 투자자 기준을 크게 초과합니다."
            }
    else:
        master_verdicts["그레이엄"] = {"icon": "—", "판정": "데이터 부족", "comment": "PER·PBR 정보가 없어 평가 불가"}

    # 버핏
    _buffett_ok = []
    _buffett_ng = []
    roe_pct_display = (roe_history[0] if roe_history else ((roe or 0) * 100))
    if roe_mean_val is not None and roe_mean_val >= 15 and (roe_std_val or 99) <= 10:
        _buffett_ok.append(f"ROE 지속성 {roe_mean_val:.0f}%")
    elif roe_pct_display >= 15:
        _buffett_ok.append(f"ROE {roe_pct_display:.0f}%")
    else:
        _buffett_ng.append(f"ROE {roe_pct_display:.0f}%")
    if debt_equity is not None:
        if debt_equity < 100:
            _buffett_ok.append(f"부채비율 {debt_equity:.0f}%")
        else:
            _buffett_ng.append(f"부채비율 {debt_equity:.0f}%")
    if fcf_yield is not None:
        if fcf_yield > 5:
            _buffett_ok.append(f"FCF Yield {fcf_yield:.1f}%")
        elif fcf_yield < 0:
            _buffett_ng.append(f"FCF 음수")
    if ocf_ni_ratio is not None:
        if ocf_ni_ratio > 1.0:
            _buffett_ok.append(f"OCF/NI={ocf_ni_ratio:.1f}")
        elif ocf_ni_ratio < 0.5:
            _buffett_ng.append(f"OCF/NI 낮음({ocf_ni_ratio:.1f})")

    if len(_buffett_ok) >= 3 and not _buffett_ng:
        master_verdicts["버핏"] = {
            "icon": "✅", "판정": "통과",
            "comment": f"{', '.join(_buffett_ok)}로 경제적 해자가 튼튼한 기업입니다."
        }
    elif len(_buffett_ng) >= 2:
        master_verdicts["버핏"] = {
            "icon": "🚫", "판정": "미달",
            "comment": f"{', '.join(_buffett_ng)} 등 버핏 기준 미달. 장기 보유 신중 검토 필요."
        }
    else:
        master_verdicts["버핏"] = {
            "icon": "⚠️", "판정": "부분 충족",
            "comment": f"긍정 {', '.join(_buffett_ok) or '없음'} / 우려 {', '.join(_buffett_ng) or '없음'}"
        }

    # 린치
    _growth_basis = eps_cagr_note if eps_cagr_note else (
        f"성장률={earnings_growth*100:.1f}%(yf)" if earnings_growth else "성장률 데이터 없음"
    )
    if peg is not None:
        if peg < 0.5:
            master_verdicts["린치"] = {
                "icon": "🚀", "판정": "강력추천",
                "comment": (
                    f"성장성 대비 주가가 매우 쌉니다 (PEG={peg:.2f}). 전형적인 성장주 패턴입니다. "
                    f"({_growth_basis})"
                )
            }
        elif peg < 1.0:
            master_verdicts["린치"] = {
                "icon": "✅", "판정": "추천",
                "comment": f"PEG={peg:.2f}로 성장 대비 저평가. 린치 기준 매수권입니다. ({_growth_basis})"
            }
        elif peg > 2.0:
            master_verdicts["린치"] = {
                "icon": "⚠️", "판정": "과열",
                "comment": f"PEG={peg:.2f} > 2.0. 성장률 대비 주가가 앞서 나갔습니다. ({_growth_basis})"
            }
        else:
            master_verdicts["린치"] = {
                "icon": "⚪", "판정": "중립",
                "comment": f"PEG={peg:.2f}. 성장성 대비 적정 밸류에이션 구간입니다. ({_growth_basis})"
            }
    elif eps_cagr_note:
        if not _per_ok:
            _lynch_comment = f"PER 없어 PEG 계산 불가. {eps_cagr_note}"
        else:
            _lynch_comment = f"EPS 감소세로 PEG 계산 불가. {eps_cagr_note}"
        master_verdicts["린치"] = {
            "icon": "—", "판정": "데이터 부족",
            "comment": _lynch_comment
        }
    else:
        master_verdicts["린치"] = {
            "icon": "—", "판정": "데이터 부족",
            "comment": "PEG 계산 불가 (EPS 이력 및 성장률 데이터 없음)"
        }

    # 오닐
    if oneil_ratio is not None:
        if oneil_ratio >= 0.95:
            master_verdicts["오닐"] = {
                "icon": "🔥", "판정": "추세확인",
                "comment": f"신고가 {oneil_ratio*100:.0f}% 수준. 거래량 동반 여부를 확인하고 진입하세요."
            }
        elif oneil_ratio >= 0.80:
            master_verdicts["오닐"] = {
                "icon": "👀", "판정": "관망",
                "comment": f"52주 고가 {oneil_ratio*100:.0f}%. 신고가 돌파 시 진입 전략을 준비하세요."
            }
        else:
            master_verdicts["오닐"] = {
                "icon": "⚠️", "판정": "조심",
                "comment": f"52주 고가 대비 {oneil_ratio*100:.0f}%. CANSLIM 기준 아직 상승 모멘텀 부재."
            }
    else:
        master_verdicts["오닐"] = {"icon": "—", "판정": "데이터 부족", "comment": "52주 고가/저가 데이터 없음"}

    label = ("펀더멘털 강함" if score >= 3 else
             "펀더멘털 약함" if score <= -2 else
             "펀더멘털 보통")

    return {
        "fund_score":      score,
        "fund_label":      label,
        "fund_reasons":    reasons,
        "master_verdicts": master_verdicts,
        # 서브 점수 (0~100)
        "sub_growth":   growth_sub,
        "sub_profit":   profit_sub,
        "sub_stable":   stable_sub,
        "sub_moment":   moment_sub,
        # 계산된 보조 지표 (UI 표시용)
        "roe_mean":      round(roe_mean_val, 1) if roe_mean_val is not None else None,
        "roe_std":       round(roe_std_val,  1) if roe_std_val  is not None else None,
        "ocf_ni_ratio":  round(ocf_ni_ratio, 2) if ocf_ni_ratio is not None else None,
        "shareholder_yield": info.get("shareholder_yield"),
        "fcf_yield":     round(fcf_yield, 1) if fcf_yield is not None else None,
        "peg":           round(peg, 2) if peg is not None else None,
        "eps_history":   eps_history,
        "eps_cagr_3yr":  round(eps_cagr_3yr * 100, 1) if eps_cagr_3yr is not None else None,
    }


# ─── 매수 적정가 계산 ────────────────────────────────────────────────────────

def get_buy_target_price(data: pd.DataFrame, mode: str = "classic") -> dict:
    """
    매수 적정가 산출 — 4가지 모드 지원

    mode="classic"  : 종합 — 지지선 가중 평균 (BB하단 50% + SMA20 30% + 5일저 20%)
    mode="sale"     : 세일 모드 — BB 하단 대기 (현재가 대비 -5~10%)
    mode="breakout" : 추격 모드 — 저항선 돌파 확인 후 진입 (현재가 대비 +2~3%)
    mode="vwap"     : 정석 모드 — 시장 참여자 평균가(VWAP) 기준 매수
    """
    if data.empty or len(data) < 20:
        return {}

    def _f(val):
        """pandas 스칼라 → Python float 변환 (pd.NA / NaN / TypeError 안전 처리)."""
        try:
            f = float(val)
            return None if pd.isna(f) else f
        except Exception:
            return None

    close   = data["Close"]
    current = _f(close.iloc[-1]) or 0.0

    bb_lower = _f(data["BB_Lower"].iloc[-1]) if "BB_Lower" in data.columns else None
    sma20    = _f(data["SMA_20"].iloc[-1])   if "SMA_20"   in data.columns else None
    atr      = _f(data["ATR"].iloc[-1])      if "ATR"      in data.columns else None

    # 5일 저점 (당일 포함 최근 5봉의 Low)
    try:
        low5 = _f(data["Low"].iloc[-5:].min()) if "Low" in data.columns and len(data) >= 5 else current
    except Exception:
        low5 = current
    if low5 is None:
        low5 = current

    # BB Lower 폴백
    if bb_lower is None:
        try:
            _mid = float(close.rolling(20).mean().iloc[-1])
            _std = float(close.rolling(20).std().iloc[-1])
            bb_lower = _mid - 2 * _std if not pd.isna(_mid) and not pd.isna(_std) else current * 0.95
        except Exception:
            bb_lower = current * 0.95

    # SMA20 폴백
    if sma20 is None:
        try:
            sma20 = float(close.rolling(20).mean().iloc[-1])
            if pd.isna(sma20):
                sma20 = current
        except Exception:
            sma20 = current

    # ── 모드별 계산 ──────────────────────────────────────────────────────────

    if mode == "sale":
        # 세일 모드: BB 하단 — 안전하게 밑에서 기다리기
        buy_target = round(bb_lower, 2)
        gap_pct    = (current - buy_target) / buy_target * 100  # 양수 = 현재가가 target 위

        if current <= bb_lower * 1.005:
            timing       = "⚡ BB하단 도달 — 즉시 매수 타이밍"
            timing_color = "#69f0ae"
        elif gap_pct <= 3.0:
            timing       = "🟡 BB하단 근접 — 분할매수 준비"
            timing_color = "#fff176"
        else:
            timing       = f"⏳ BB하단까지 -{gap_pct:.1f}% 하락 대기"
            timing_color = "#aaa"

        mode_label  = "🏷️ 세일 모드 (LBB)"
        mode_desc   = "안전하게 밑에서 기다릴게요"
        mode_color  = "#82b1ff"
        detail_line = f"BB하단 {round(bb_lower, 2)} · 현재가 대비 -{gap_pct:.1f}%"

    elif mode == "breakout":
        # 추격 모드: 저항선 돌파 확인 후 진입
        vpvr       = calculate_vpvr(data)
        resistance = vpvr.get("resistance_level")
        bb_upper   = float(data["BB_Upper"].iloc[-1]) if "BB_Upper" in data.columns else None

        # 가장 가까운 저항선 선택 (현재가 위)
        candidates = [r for r in [resistance, bb_upper] if r and r > current]
        ref_resistance = min(candidates) if candidates else current * 1.03

        buy_target = round(ref_resistance * 1.025, 2)  # 저항선 2.5% 돌파 확인 진입
        gap_pct    = (buy_target - current) / current * 100  # 양수 = 목표가가 현재가 위

        if current >= ref_resistance * 0.99:
            timing       = "⚡ 저항선 돌파 직전 — 매수 준비"
            timing_color = "#69f0ae"
        elif gap_pct <= 5.0:
            timing       = "🟡 저항선 접근 중 — 알림 설정 권고"
            timing_color = "#fff176"
        else:
            timing       = f"⏳ 저항선까지 +{gap_pct:.1f}% — 돌파 대기"
            timing_color = "#aaa"

        mode_label  = "⚡ 추격 모드 (Breakout)"
        mode_desc   = "저항선을 뚫으면 바로 탈게요"
        mode_color  = "#ff8a65"
        detail_line = f"저항선 {round(ref_resistance, 2)} · 돌파확인가 {buy_target}"

    elif mode == "vwap":
        # 정석 모드: VWAP 기준 평균 매수가
        vwap_w = float(data["VWAP_W"].iloc[-1]) if "VWAP_W" in data.columns and pd.notna(data["VWAP_W"].iloc[-1]) else None
        vwap_m = float(data["VWAP_M"].iloc[-1]) if "VWAP_M" in data.columns and pd.notna(data["VWAP_M"].iloc[-1]) else None

        if vwap_w:
            buy_target  = round(vwap_w, 2)
            vwap_label  = "VWAP(5일)"
            extra       = f" · VWAP(20일) {round(vwap_m, 2)}" if vwap_m else ""
        elif vwap_m:
            buy_target  = round(vwap_m, 2)
            vwap_label  = "VWAP(20일)"
            extra       = ""
        else:
            buy_target  = round(sma20, 2)
            vwap_label  = "SMA20 (VWAP 대체)"
            extra       = ""

        gap_pct = (current - buy_target) / buy_target * 100

        if current <= buy_target * 1.005:
            timing       = "⚡ VWAP 하단 — 평균 이하 진입 기회"
            timing_color = "#69f0ae"
        elif current <= buy_target * 1.02:
            timing       = "🔵 VWAP 근접 — 분할매수 적합"
            timing_color = "#82b1ff"
        elif gap_pct <= 5.0:
            timing       = "🟡 VWAP 소폭 상회 — 눌림목 대기"
            timing_color = "#fff176"
        else:
            timing       = f"⏳ VWAP 대비 +{gap_pct:.1f}% 고평가 — 대기"
            timing_color = "#aaa"

        mode_label  = "📊 정석 모드 (VWAP)"
        mode_desc   = "시장 참여자 평균가에 살게요"
        mode_color  = "#a5d6a7"
        detail_line = f"{vwap_label} {buy_target}{extra}"

    else:
        # classic (기본): 지지선 가중 평균
        buy_target = round(bb_lower * 0.50 + sma20 * 0.30 + low5 * 0.20, 2)
        gap_pct    = (current - buy_target) / buy_target * 100

        if bb_lower and current <= bb_lower * 1.005:
            timing       = "⚡ 과매도 — 지금이 매수 타이밍"
            timing_color = "#69f0ae"
        elif sma20 and current <= sma20 * 1.01:
            timing       = "🔵 SMA20 지지 — 분할매수 적합"
            timing_color = "#82b1ff"
        elif gap_pct <= 3.0:
            timing       = "🟡 적정가 근접 — 모니터링 권고"
            timing_color = "#fff176"
        else:
            timing       = f"⏳ 대기 중 ({gap_pct:+.1f}% 고평가)"
            timing_color = "#aaa"

        mode_label  = "🎯 종합 모드"
        mode_desc   = "지지선 가중 평균 기준"
        mode_color  = "#e0e0e0"
        detail_line = f"BB하단 {round(bb_lower, 2)} · SMA20 {round(sma20, 2)} · 5일저 {round(low5, 2)}"

    return {
        "buy_target":   buy_target,
        "bb_lower":     round(bb_lower, 2),
        "sma20":        round(sma20, 2),
        "low5":         round(low5, 2),
        "gap_pct":      round(gap_pct, 2),
        "timing":       timing,
        "timing_color": timing_color,
        "current":      round(current, 2),
        "atr":          round(atr, 2) if atr else None,
        "mode":         mode,
        "mode_label":   mode_label,
        "mode_desc":    mode_desc,
        "mode_color":   mode_color,
        "detail_line":  detail_line,
    }


# ─── 손절·익절 레벨 계산 ─────────────────────────────────────────────────────

def get_stop_loss_targets(data: pd.DataFrame, entry_price: float = None) -> dict:
    """
    오닐 7~8% 손절 원칙 + ATR 기반 손절 + 3:1 리스크/리워드 목표가
    참고: 《최고의 주식 최적의 타이밍》(윌리엄 오닐)
    """
    if data.empty:
        return {}

    close   = data["Close"]
    current = float(close.iloc[-1])
    entry   = entry_price or current

    # ATR — get_stock_data에서 미리 계산된 값 우선, 없으면 직접 계산
    if "ATR" in data.columns and pd.notna(data["ATR"].iloc[-1]):
        atr_val = float(data["ATR"].iloc[-1])
    else:
        high   = data["High"]
        low    = data["Low"]
        prev_c = close.shift(1)
        tr     = pd.concat([high - low, (high - prev_c).abs(), (low - prev_c).abs()], axis=1).max(axis=1)
        atr_val = float(tr.rolling(14).mean().iloc[-1])

    # 손절선
    stop_8pct = round(entry * 0.92, 2)                              # 오닐 8% 손절
    stop_atr  = round(max(entry - atr_val * 2.5, entry * 0.85), 2) # ATR×2.5 (최대 15%)

    # 목표가 (3:1 / 2:1 리스크/리워드)
    risk       = entry - stop_8pct
    target_2r  = round(entry + risk * 2, 2)
    target_3r  = round(entry + risk * 3, 2)

    # 52주 고가
    high_52w = round(float(close.rolling(min(252, len(close))).max().iloc[-1]), 2)

    # 볼린저밴드 상단
    bb_upper = round(float(data["BB_Upper"].iloc[-1]), 2) if "BB_Upper" in data.columns else None

    return {
        "current":     round(current, 2),
        "entry":       round(entry, 2),
        "stop_8pct":   stop_8pct,
        "stop_atr":    stop_atr,
        "target_2r":   target_2r,
        "target_3r":   target_3r,
        "high_52w":    high_52w,
        "bb_upper":    bb_upper,
        "atr":         round(atr_val, 2),
        "atr_ratio":   round(atr_val / current * 100, 2),
    }


# ─── 매도 적정가 계산 ────────────────────────────────────────────────────────

def get_sell_target_price(data: pd.DataFrame) -> dict:
    """
    3가지 방법론으로 매도 적정가 산출:
      1. 볼린저밴드 상단 (2σ) — 통계적 과매수 확률 5% 미만
      2. 매물대 저항 (VPVR POC 위 최대 거래량 구간) — 본전 매물 압력
      3. 피보나치 확장 1.618× — 신고가 영역 심리적 저항
    보수적 목표가: BB상단 vs 매물대 저항 중 낮은 값 (50% 분할매도 기준)
    공격적 목표가: 피보나치 1.618 확장선 (전량매도 / 트레일링 스탑 기준)
    """
    if data.empty or len(data) < 20:
        return {}

    close   = data["Close"].astype(float)
    current = float(close.iloc[-1])

    # 1. 볼린저밴드 상단
    if "BB_Upper" in data.columns and pd.notna(data["BB_Upper"].iloc[-1]):
        bb_upper = float(data["BB_Upper"].iloc[-1])
    else:
        _mid = close.rolling(20).mean().iloc[-1]
        _std = close.rolling(20).std().iloc[-1]
        bb_upper = float(_mid + 2 * _std)

    # 2. 매물대 저항 (현재가 위 거래량 최대 구간)
    vpvr = calculate_vpvr(data)
    resistance_level = vpvr.get("resistance_level")  # 현재가 위 최대 매물벽
    poc_price        = vpvr.get("poc_price")          # 거래량 집중가

    # 3. 피보나치 확장 (최근 1년 스윙 저점 → 고점 기준)
    lookback   = min(len(close), 252)
    swing_low  = float(close.iloc[-lookback:].min())
    swing_high = float(close.iloc[-lookback:].max())
    fib_range  = swing_high - swing_low

    fib_1618 = round(swing_low + fib_range * 1.618, 2) if fib_range > 0 else None
    fib_2618 = round(swing_low + fib_range * 2.618, 2) if fib_range > 0 else None

    # 52주 신고가 여부 (현재가가 52주 고점의 98% 이상)
    high_52w    = float(close.rolling(min(252, len(close))).max().iloc[-1])
    is_new_high = current >= high_52w * 0.98

    # ── 보수적 목표가: BB상단 vs 현재가 위 매물대 저항 중 낮은 값 ─────────
    candidates_cons = [bb_upper]
    if resistance_level and resistance_level > current:
        candidates_cons.append(resistance_level)
    conservative_target = round(min(candidates_cons), 2)

    # ── 공격적 목표가: 신고가 영역이면 피보 1.618, 아니면 최대 저항 × 1.05 ──
    if is_new_high and fib_1618 and fib_1618 > current * 1.01:
        aggressive_target = fib_1618
    elif resistance_level and resistance_level > current:
        aggressive_target = round(max(bb_upper, resistance_level) * 1.05, 2)
    else:
        aggressive_target = round(bb_upper * 1.05, 2)

    # 현재가 대비 목표가 괴리율
    cons_gap = (conservative_target - current) / current * 100
    aggr_gap = (aggressive_target - current) / current * 100

    # 매도 타이밍 신호
    if current >= bb_upper * 0.995:
        sell_timing = "🔴 BB상단 돌파 — 즉시 분할매도 고려"
        sell_color  = "#ef9a9a"
    elif resistance_level and current >= resistance_level * 0.98:
        sell_timing = "🟠 매물대 저항 근접 — 부분 익절 고려"
        sell_color  = "#ffab91"
    elif cons_gap <= 3.0:
        sell_timing = "🟡 보수적 목표가 근접 — 매도 준비"
        sell_color  = "#fff176"
    else:
        sell_timing = f"⏳ 목표가까지 {cons_gap:.1f}% 여유"
        sell_color  = "#aaa"

    return {
        "current":              round(current, 2),
        "conservative_target":  conservative_target,
        "aggressive_target":    round(aggressive_target, 2),
        "bb_upper":             round(bb_upper, 2),
        "resistance_level":     round(resistance_level, 2) if resistance_level else None,
        "poc_price":            round(poc_price, 2) if poc_price else None,
        "fib_1618":             fib_1618,
        "fib_2618":             fib_2618,
        "swing_low":            round(swing_low, 2),
        "swing_high":           round(swing_high, 2),
        "high_52w":             round(high_52w, 2),
        "is_new_high":          is_new_high,
        "cons_gap":             round(cons_gap, 2),
        "aggr_gap":             round(aggr_gap, 2),
        "sell_timing":          sell_timing,
        "sell_color":           sell_color,
    }


# ─── SEC EDGAR 내부자 거래 (미국 주식 전용) ───────────────────────────────────

def _get_sec_cik(ticker: str) -> str | None:
    """SEC company_tickers.json에서 CIK 조회"""
    try:
        resp = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers={"User-Agent": "AutoStockAnalyzer contact@example.com"},
            timeout=10,
        )
        if resp.status_code != 200:
            return None
        for entry in resp.json().values():
            if entry.get("ticker", "").upper() == ticker.upper():
                return str(entry["cik_str"]).zfill(10)
    except Exception:
        pass
    return None


def get_insider_trades_sec(ticker: str, days: int = 90) -> pd.DataFrame:
    """
    SEC EDGAR Form 4 내부자 거래 공시 조회 (미국 주식 전용)
    데이터 소스: edgar.sec.gov (완전 무료, 공식 API)
    """
    if "." in ticker:   # KS, KQ 등 해외 티커 제외
        return pd.DataFrame()

    try:
        headers = {"User-Agent": "AutoStockAnalyzer contact@example.com"}

        cik = _get_sec_cik(ticker)
        if not cik:
            return pd.DataFrame()

        # Atom 피드로 Form 4 목록 조회
        atom_url = (
            f"https://www.sec.gov/cgi-bin/browse-edgar"
            f"?action=getcompany&CIK={cik}&type=4&dateb=&owner=include&count=20&output=atom"
        )
        resp = requests.get(atom_url, headers=headers, timeout=15)
        if resp.status_code != 200:
            return pd.DataFrame()

        ns   = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(resp.content)
        cutoff = datetime.now() - timedelta(days=days)

        rows = []
        for entry in root.findall("atom:entry", ns):
            title   = entry.findtext("atom:title",   "", ns)
            updated = entry.findtext("atom:updated", "", ns)[:10]
            link_el = entry.find("atom:link", ns)
            href    = link_el.get("href", "") if link_el is not None else ""

            try:
                if datetime.strptime(updated, "%Y-%m-%d") < cutoff:
                    continue
            except ValueError:
                pass

            rows.append({"공시일": updated, "내용": title, "링크": href})

        return pd.DataFrame(rows) if rows else pd.DataFrame()

    except Exception:
        return pd.DataFrame()


# ─── 뉴스 감성 분석 ───────────────────────────────────────────────────────────

# ── 선 필터링용 노이즈 회사명 (분석 대상이 아닌 증권사/금융사) ────────────────────
_NOISE_FIRMS: list[str] = [
    "NH투자", "미래에셋", "키움증권", "삼성증권", "한국투자증권", "신한투자",
    "KB증권", "하나증권", "메리츠증권", "이베스트투자", "대신증권", "유안타증권",
    "한화투자증권", "교보증권", "현대차증권", "DB금융투자", "흥국증권", "IBK투자",
    "Goldman Sachs", "Morgan Stanley", "JPMorgan", "Deutsche Bank",
    "Citigroup", "UBS", "Barclays", "Nomura", "HSBC",
]

# ── 뉴스 결과 인메모리 캐시 (제목 해시 기반, 1시간 TTL, 최대 300건) ─────────────
_news_result_cache: dict[str, tuple[dict, float]] = {}
_NEWS_CACHE_TTL = 3600  # seconds


def _news_cache_key(news_items: list[dict], ticker: str) -> str:
    """뉴스 제목 집합 + ticker 해시 → 캐시 키."""
    titles = sorted(item.get("title", "") for item in news_items)
    raw = ticker + "|" + "|".join(titles)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _get_cached_news_result(key: str) -> dict | None:
    entry = _news_result_cache.get(key)
    if entry and (time.time() - entry[1]) < _NEWS_CACHE_TTL:
        return entry[0]
    if entry:
        del _news_result_cache[key]
    return None


def _set_cached_news_result(key: str, result: dict) -> None:
    if len(_news_result_cache) >= 300:
        # 오래된 항목 50개 제거
        oldest = sorted(_news_result_cache.items(), key=lambda x: x[1][1])[:50]
        for k, _ in oldest:
            del _news_result_cache[k]
    _news_result_cache[key] = (result, time.time())


# ── A급 (가중치 2.0): 실적·수주·판매량·M&A — 종목 직접 영향 ─────────────────────
_KW_A_POS = [
    "어닝서프라이즈", "어닝 서프라이즈", "어닝비트", "실적 서프라이즈",
    "역대 최대", "사상 최대", "역대 최고", "분기 최대", "최대 실적",
    "판매량", "판매 호조", "출하량", "점유율", "시장점유율",
    "대규모 수주", "수주 확보", "수주 잔고", "계약 체결", "신규 계약",
    "인수합병", "M&A", "기업 인수",
    "배당 확대", "특별배당", "자사주 소각",
    "흑자전환", "턴어라운드",
    "FDA 승인", "양산 시작", "상업화",
]
_KW_A_NEG = [
    "어닝쇼크", "어닝 쇼크", "어닝미스", "실적 쇼크",
    "적자전환", "대규모 적자", "순손실 전환",
    "파산", "상장폐지", "영업정지",
    "검찰 수사", "금감원 제재", "분식회계", "회계 부정",
    "집단소송", "대규모 리콜",
    "배당 삭감", "배당 중단",
]

# ── B급 (가중치 1.0): 업황·섹터·목표가·ETF — 간접 영향 ──────────────────────────
_KW_B_POS = [
    "수주", "돌파", "상승", "이익", "증가", "추천", "급등", "호재", "성장", "흑자",
    "신고가", "매수", "목표가 상향", "배당", "실적 개선", "확대",
    "강세", "반등", "회복", "신규", "수혜", "기대", "긍정", "상향", "계약", "수출",
    "ETF 자금", "ETF 유입", "ETF 순유입", "머니무브",
    "반도체 업황", "DRAM 가격", "HBM", "메모리 수요", "AI 반도체",
    "목표가 올려", "투자의견 상향",
    "수요 회복", "공급 부족", "업종 강세",
]
_KW_B_NEG = [
    "하락", "감소", "적자", "조사", "우려", "매도", "급락", "악재", "손실",
    "신저가", "목표가 하향", "부진", "위기", "하향", "규제", "소송", "제재",
    "약세", "불안", "취소", "철수", "실망", "하락세", "폭락",
    "DRAM 가격 하락", "반도체 업황 악화",
    "투자의견 하향",
    "수요 감소", "공급 과잉",
]

# ── C급 (가중치 0.5): 거시경제·시황 — 배경 영향 ─────────────────────────────────
_KW_C_POS = [
    "코스피 상승", "코스닥 상승", "증시 반등",
    "외국인 순매수", "기관 순매수",
    "달러 약세", "원화 강세", "금리 인하", "경기 회복",
]
_KW_C_NEG = [
    "코스피 하락", "코스닥 하락", "증시 급락",
    "외국인 순매도", "기관 순매도",
    "달러 강세", "원화 약세", "금리 인상", "경기 침체",
    "중동 위기", "지정학 리스크", "전쟁",
]

# ── 이벤트성 뉴스 키워드 (합병·분할·거래정지) — 기술 분석 점수에서 제외 ──────────
_KW_EVENT_HALT = [
    "거래 정지", "거래정지", "매매 정지", "매매정지",
    "상장폐지", "상장 폐지", "관리종목", "관리 종목",
    "투자주의", "투자 주의", "투자경고", "투자 경고",
    "투자위험", "투자 위험", "불성실공시", "불성실 공시",
    "trading halt", "trading suspension",
]
_KW_EVENT_CORP = [
    "주식 병합", "주식병합", "역주식분할", "역 주식분할",
    "주식 분할", "주식분할", "액면분할", "액면 분할",
    "액면병합", "액면 병합",
    "합병", "흡수합병", "분할합병", "인적분할", "물적분할",
    "기업분할", "기업 분할",
    "stock split", "reverse split", "stock merger",
]
_KW_EVENT_ALL = _KW_EVENT_HALT + _KW_EVENT_CORP

# 하위호환 (get_advanced_sentiment 등 기존 코드에서 참조)
_POS_KEYWORDS = _KW_B_POS
_NEG_KEYWORDS = _KW_B_NEG

# ── ETF 포트폴리오 맵 (코드 → 섹터 + 대표 구성종목) ─────────────────────────
_ETF_PORTFOLIO_MAP: dict[str, dict] = {
    "069500": {"sector": "코스피200",  "name": "KODEX 200",
               "holdings": ["005930.KS", "000660.KS", "005380.KS", "373220.KS", "207940.KS"]},
    "091160": {"sector": "반도체",     "name": "KODEX 반도체",
               "holdings": ["005930.KS", "000660.KS", "042700.KS", "000990.KS", "066970.KQ"]},
    "091170": {"sector": "자동차",     "name": "KODEX 자동차",
               "holdings": ["005380.KS", "000270.KS", "012330.KS", "241560.KS"]},
    "305720": {"sector": "배터리",     "name": "KODEX 2차전지산업",
               "holdings": ["373220.KS", "006400.KS", "051910.KS", "247540.KQ"]},
    "091220": {"sector": "금융",       "name": "TIGER 은행",
               "holdings": ["105560.KS", "055550.KS", "086790.KS", "138930.KS", "316140.KS"]},
    "143860": {"sector": "바이오",     "name": "TIGER 헬스케어",
               "holdings": ["207940.KS", "068270.KS", "128940.KS", "326030.KS"]},
    "266370": {"sector": "IT플랫폼",   "name": "KODEX IT",
               "holdings": ["005930.KS", "000660.KS", "035420.KS", "035720.KS"]},
    "228790": {"sector": "화학소재",   "name": "TIGER 화학",
               "holdings": ["051910.KS", "011170.KS", "006400.KS", "010950.KS"]},
    "232080": {"sector": "코스닥150",  "name": "TIGER KOSDAQ150",
               "holdings": ["247540.KQ", "086900.KQ", "035900.KQ", "041510.KQ"]},
    "114800": {"sector": "인버스",     "name": "KODEX 인버스",     "holdings": []},
    "122630": {"sector": "레버리지",   "name": "KODEX 레버리지",
               "holdings": ["005930.KS", "000660.KS", "005380.KS"]},
    "360750": {"sector": "미국S&P500", "name": "TIGER 미국S&P500",
               "holdings": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]},
    "133690": {"sector": "미국나스닥", "name": "TIGER 미국나스닥100",
               "holdings": ["AAPL", "MSFT", "NVDA", "AMZN", "META"]},
    "395160": {"sector": "미국배당",   "name": "TIGER 미국배당다우존스",
               "holdings": ["JNJ", "KO", "PG", "MMM", "VZ"]},
}

# ── 섹터별 업황 지표 키워드 ────────────────────────────────────────────────────
_TICKER_SECTOR: dict[str, str] = {
    # 국내 개별 종목
    "005930.KS": "반도체", "000660.KS": "반도체", "042700.KS": "반도체",
    "005380.KS": "자동차", "000270.KS": "자동차", "012330.KS": "자동차",
    "373220.KS": "배터리", "006400.KS": "배터리", "051910.KS": "배터리",
    "035420.KS": "플랫폼", "035720.KS": "플랫폼",
    "207940.KS": "바이오",  "068270.KS": "바이오",  "128940.KS": "바이오",
    "105560.KS": "금융", "055550.KS": "금융", "086790.KS": "금융",
    "138930.KS": "금융", "316140.KS": "금융",
    # 국내 ETF
    "069500.KS": "코스피200", "091160.KS": "반도체", "091170.KS": "자동차",
    "305720.KS": "배터리",    "091220.KS": "금융",   "143860.KS": "바이오",
    "266370.KS": "IT플랫폼",  "228790.KS": "화학소재","232080.KS": "코스닥150",
    "114800.KS": "인버스",    "122630.KS": "레버리지","360750.KS": "미국S&P500",
    "133690.KS": "미국나스닥","395160.KS": "미국배당",
    # 미국 개별 종목
    "NVDA": "semiconductor", "AMD": "semiconductor",
    "INTC": "semiconductor", "AVGO": "semiconductor", "QCOM": "semiconductor",
    "AAPL": "bigtech", "MSFT": "bigtech", "GOOGL": "bigtech",
    "META": "bigtech", "AMZN": "bigtech",
    "TSLA": "ev", "RIVN": "ev", "NIO": "ev",
    "GM": "auto", "F": "auto",
    "JPM": "finance", "BAC": "finance", "GS": "finance", "MS": "finance",
    "WFC": "finance", "C": "finance",
    # 국내 추가
    "017670.KS": "통신",   "030200.KS": "통신",   "032640.KS": "통신",
    "329180.KS": "조선",   "010140.KS": "조선",   "042660.KS": "조선",   "009540.KS": "조선",
    "012450.KS": "방산",   "079550.KS": "방산",   "047810.KS": "방산",   "272210.KS": "방산",
    "000720.KS": "건설",   "006360.KS": "건설",   "047040.KS": "건설",   "028050.KS": "건설",
    "096770.KS": "에너지", "010950.KS": "에너지",
    "011170.KS": "화학",   "009830.KS": "화학",
    "259960.KS": "게임",   "251270.KS": "게임",   "293490.KQ": "게임",   "263750.KQ": "게임",
    "352820.KS": "엔터",   "041510.KQ": "엔터",   "035900.KQ": "엔터",   "122870.KQ": "엔터",
    "032830.KS": "보험",   "088350.KS": "보험",   "005830.KS": "보험",   "000060.KS": "보험",
    "006800.KS": "증권",   "016360.KS": "증권",   "005940.KS": "증권",   "071050.KS": "증권",
    "138930.KS": "금융",   "316140.KS": "금융",
    "010130.KS": "철강",   "005490.KS": "철강",   "004020.KS": "철강",
    "000100.KS": "바이오", "012330.KS": "자동차",  "241560.KS": "자동차",
    "000990.KS": "반도체", "066970.KQ": "반도체",
    "247540.KQ": "배터리",
    # 미국 추가
    "QCOM": "semiconductor", "MU": "semiconductor",  "TXN": "semiconductor",
    "LRCX": "semiconductor", "AMAT": "semiconductor", "KLAC": "semiconductor",
    "ARM":  "semiconductor", "MRVL": "semiconductor",
    "JNJ":  "healthcare",    "PFE":  "healthcare",    "ABBV": "healthcare",
    "MRK":  "healthcare",    "LLY":  "healthcare",    "UNH":  "healthcare",
    "XOM":  "energy",        "CVX":  "energy",        "COP":  "energy",
    "OXY":  "energy",        "SLB":  "energy",
    "WMT":  "retail",        "COST": "retail",        "TGT":  "retail",
    "NFLX": "media",         "DIS":  "media",         "SPOT": "media",
    "V":    "payments",      "MA":   "payments",      "PYPL": "payments",
    "CRM":  "saas",          "SNOW": "saas",           "ORCL": "saas",   "NOW": "saas",
    "VZ":   "telecom",       "T":    "telecom",        "TMUS": "telecom",
}

_SECTOR_KEYWORDS: dict[str, dict[str, list[str]]] = {
    "반도체": {
        "pos": [
            "DRAM 가격 상승", "NAND 가격 상승", "HBM 수요", "AI 반도체 수요",
            "반도체 업황 개선", "필라델피아 반도체", "메모리 수요 증가",
            "서버 수요 증가", "DDR5", "CoWoS", "반도체 수출 증가",
        ],
        "neg": [
            "DRAM 가격 하락", "메모리 가격 하락", "반도체 업황 악화",
            "반도체 재고 급증", "공급 과잉", "메모리 수요 감소",
        ],
    },
    "semiconductor": {
        "pos": [
            "DRAM price rise", "HBM demand", "AI chip demand",
            "Philadelphia Semiconductor", "server demand surge", "chip shortage",
        ],
        "neg": [
            "DRAM price fall", "chip oversupply", "inventory buildup",
            "semiconductor slowdown", "memory demand weakness",
        ],
    },
    "자동차": {
        "pos": [
            "전기차 판매 증가", "친환경차 수요", "자동차 수출 증가",
            "완성차 판매 호조", "자동차 수주 확대",
        ],
        "neg": [
            "완성차 판매 감소", "전기차 캐즘", "자동차 파업",
            "자동차 관세", "판매 부진",
        ],
    },
    "배터리": {
        "pos": [
            "배터리 수주 증가", "전고체 배터리", "에너지 밀도 향상",
            "배터리 수요 급증", "생산 캐파 확대",
        ],
        "neg": [
            "배터리 화재", "배터리 리콜", "전기차 캐즘",
            "배터리 수요 부진", "중국 배터리 경쟁",
        ],
    },
    "플랫폼": {
        "pos": [
            "MAU 증가", "광고 매출 증가", "구독자 급증", "AI 서비스 확대",
        ],
        "neg": [
            "이용자 감소", "광고 규제", "플랫폼 규제",
            "반독점 조사",
        ],
    },
    "바이오": {
        "pos": [
            "임상 성공", "FDA 승인", "글로벌 라이선스 계약",
            "기술 수출", "신약 허가", "블록버스터",
        ],
        "neg": [
            "임상 실패", "FDA 거부", "심각한 부작용",
            "임상 중단",
        ],
    },
    "bigtech": {
        "pos": [
            "cloud revenue record", "AI revenue growth", "subscription growth",
            "ad revenue beat", "data center expansion",
        ],
        "neg": [
            "antitrust lawsuit", "regulatory fine", "user growth stall",
            "ad revenue miss", "layoff",
        ],
    },
    "ev": {
        "pos": [
            "delivery record", "EV demand surge", "charging network expansion",
            "autonomous driving approval",
        ],
        "neg": [
            "delivery miss", "EV slowdown", "price cut margin hit",
            "vehicle recall",
        ],
    },
    "auto": {
        "pos": ["sales record", "export growth", "EV transition", "strong delivery"],
        "neg": ["recall", "strike", "tariff impact", "sales decline"],
    },
    "finance": {
        "pos": ["interest income growth", "loan growth", "trading revenue beat", "rate benefit"],
        "neg": ["credit loss surge", "regulatory fine", "loan default rise", "NIM compression"],
    },
    "금융": {
        "pos": ["금리 인하", "NIM 개선", "이자 수익 증가", "대출 성장", "배당 확대",
                "은행 실적 호조", "건전성 개선", "연체율 하락"],
        "neg": ["부실 채권", "충당금 증가", "금융 규제", "연체율 상승",
                "금리 역전", "예대마진 축소", "부동산 PF 위기"],
    },
    "IT플랫폼": {
        "pos": ["MAU 증가", "광고 매출 증가", "클라우드 성장", "AI 수익화",
                "구독자 확대", "검색 점유율 상승"],
        "neg": ["이용자 감소", "플랫폼 규제", "광고 감소", "반독점 조사",
                "개인정보 규제", "수익성 악화"],
    },
    "코스피200": {
        "pos": ["외국인 순매수", "지수 상승", "증시 반등", "경기 회복",
                "원화 강세", "반도체 업황 개선", "수출 증가"],
        "neg": ["외국인 순매도", "지수 하락", "경기 침체 우려", "원화 약세",
                "환율 급등", "글로벌 증시 급락"],
    },
    "코스닥150": {
        "pos": ["바이오 임상 성공", "2차전지 수주", "플랫폼 성장", "기술주 반등",
                "외국인 코스닥 매수"],
        "neg": ["바이오 임상 실패", "코스닥 급락", "성장주 매도", "금리 인상"],
    },
    "화학소재": {
        "pos": ["석유화학 마진 개선", "소재 수요 증가", "배터리 소재 수주",
                "원료비 하락", "스프레드 확대"],
        "neg": ["석유화학 공급 과잉", "원료비 급등", "중국 경쟁 심화",
                "마진 압박", "수요 부진"],
    },
    "미국S&P500": {
        "pos": ["S&P500 상승", "미국 증시 반등", "Fed 금리 인하",
                "미국 경기 호조", "빅테크 실적 서프라이즈"],
        "neg": ["S&P500 하락", "미국 경기 침체", "Fed 금리 인상",
                "인플레이션 재부상", "미국 부채 위기"],
    },
    "미국나스닥": {
        "pos": ["나스닥 상승", "AI 수혜", "기술주 반등", "빅테크 실적 호조",
                "반도체 AI 수요"],
        "neg": ["나스닥 급락", "기술주 밸류에이션 부담", "AI 거품 우려",
                "금리 인상 압박", "반독점 규제"],
    },
    "미국배당": {
        "pos": ["배당 인상", "안정 수익", "금리 인하 수혜", "배당주 선호",
                "방어주 매수"],
        "neg": ["배당 삭감", "금리 인상으로 배당주 매도", "실적 부진",
                "인플레이션 실질 수익률 훼손"],
    },
    "인버스": {
        "pos": ["지수 하락", "증시 급락", "경기 침체", "리스크 오프"],
        "neg": ["지수 상승", "증시 반등", "리스크 온"],
    },
    "레버리지": {
        "pos": ["지수 상승", "증시 급등", "외국인 대규모 매수", "리스크 온"],
        "neg": ["지수 하락", "증시 급락", "변동성 확대"],
    },
}


def _news_time_decay(pub_date_str: str) -> float:
    """뉴스 발행 시간 경과에 따른 감쇠 계수.
    1h 이내: 1.0 / 4h 이내: 0.7 / 12h 이내: 0.5 / 이후: 0.3
    """
    if not pub_date_str:
        return 0.5
    fmts = [
        "%Y.%m.%d %H:%M", "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ",
        "%Y.%m.%d", "%Y-%m-%d",
    ]
    for fmt in fmts:
        try:
            dt = datetime.strptime(pub_date_str.strip()[:19], fmt)
            hours_ago = (datetime.now() - dt).total_seconds() / 3600
            if hours_ago <= 1:    return 1.0
            elif hours_ago <= 4:  return 0.7
            elif hours_ago <= 12: return 0.5
            else:                 return 0.3
        except ValueError:
            continue
    return 0.5


def _check_news_relevance(text: str, company_name: str) -> tuple[bool, str]:
    """뉴스가 해당 종목에 직접 관련됐는지 판단.
    다른 회사가 주인공이고 company_name이 비교군으로만 언급된 경우 False 반환.
    """
    if not company_name or company_name not in text:
        return True, ""
    comparison_kws = [
        " vs ", " VS ", " 대비 ", " 보다 ", "에 비해", "와 비교", "과 비교",
        "와 달리", "과 달리", "보다 더 ", "보다 덜 ",
        " versus ", " compared to ", " relative to ",
    ]
    has_comparison = any(kw in text for kw in comparison_kws)
    if not has_comparison:
        return True, ""
    idx = text.find(company_name)
    if idx <= 8:
        return True, ""
    if has_comparison:
        return False, "타사 주체·비교군 언급 추정"
    return True, ""


def _classify_news_tier(text: str) -> tuple[float, str, list[str]]:
    """뉴스 텍스트의 등급 분류 및 기본 점수 반환.

    Returns: (raw_score[-1~+1], tier, matched_keywords)
    EVENT 먼저 → A급 → B급 → C급
    """
    # EVENT: 합병·분할·거래정지 — 이벤트 대기 처리, 점수 제외
    event_kw = [kw for kw in _KW_EVENT_ALL if kw in text]
    if event_kw:
        return 0.0, "EVENT", event_kw

    # A급: 직접 실적·판매량·M&A (가중치 2.0)
    a_pos = [kw for kw in _KW_A_POS if kw in text]
    a_neg = [kw for kw in _KW_A_NEG if kw in text]
    if a_pos or a_neg:
        raw = (len(a_pos) - len(a_neg)) * 2.0
        return max(-1.0, min(1.0, raw)), "A", a_pos + a_neg

    # B급: 업황·섹터·ETF·목표가 (가중치 1.0)
    b_pos = [kw for kw in _KW_B_POS if kw in text]
    b_neg = [kw for kw in _KW_B_NEG if kw in text]
    if b_pos or b_neg:
        raw = (len(b_pos) - len(b_neg)) * 0.5
        return max(-1.0, min(1.0, raw)), "B", b_pos + b_neg

    # C급: 거시·시황 (가중치 0.5)
    c_pos = [kw for kw in _KW_C_POS if kw in text]
    c_neg = [kw for kw in _KW_C_NEG if kw in text]
    raw = (len(c_pos) - len(c_neg)) * 0.5
    return max(-1.0, min(1.0, raw)), "C", c_pos + c_neg


def _select_top_news(news_items: list[dict], n: int = 10) -> list[dict]:
    """A/B 등급 우선, C급은 A/B가 부족할 때만 보충. 최대 n개."""
    _TIER_ORDER = {"A": 0, "B": 1, "C": 2}

    scored = []
    for item in news_items:
        text = item.get("title", "") + " " + item.get("summary", "")
        _, tier, _ = _classify_news_tier(text)
        decay = _news_time_decay(item.get("pub_date", ""))
        scored.append((tier, decay, item))

    ab = sorted(
        [(t, d, it) for t, d, it in scored if t in ("A", "B", "EVENT")],
        key=lambda x: (_TIER_ORDER.get(x[0], 1), -x[1]),
    )
    c = sorted(
        [(t, d, it) for t, d, it in scored if t == "C"],
        key=lambda x: -x[1],
    )
    combined = ab + (c if len(ab) < n else [])
    return [it for _, _, it in combined[:n]]


def _deduplicate_news(news_items: list[dict]) -> list[dict]:
    """중복 뉴스 제거: 제목 앞 15자(공백 제거) 일치 또는 자카드 유사도 50% 이상."""
    seen_prefixes: set[str] = set()
    seen_words:    list[set] = []
    unique:        list[dict] = []

    for item in news_items:
        title  = item.get("title", "")
        prefix = title.replace(" ", "")[:15]

        # 1. 제목 앞 15자 중복
        if prefix and prefix in seen_prefixes:
            continue

        # 2. 자카드 유사도 50% 이상 중복
        words = set(title.split())
        if words and any(
            len(words & s) / max(len(words | s), 1) >= 0.5
            for s in seen_words
        ):
            continue

        if prefix:
            seen_prefixes.add(prefix)
        if words:
            seen_words.append(words)
        unique.append(item)

    return unique


def _extract_news_snippet(item: dict) -> str:
    """제목 + 요약 핵심 문장 3개 추출 (요약 없으면 제목만)."""
    title = item.get("title", "")
    summary = item.get("summary", "").strip()
    if not summary:
        return title
    sentences = [s.strip() for s in _re_global.split(r"[.!?。]\s*", summary) if s.strip()]
    snippet = " ".join(sentences[:3])
    return f"{title} / {snippet}" if snippet else title


def _extract_sentences_near(text: str, keyword: str, max_sentences: int = 3) -> str:
    """keyword가 포함된 문장 ± 인접 1문장을 최대 max_sentences개 추출."""
    sentences = [s.strip() for s in _re_global.split(r"[.!?。]\s*", text) if s.strip()]
    if not sentences:
        return ""
    result: list[str] = []
    for i, sent in enumerate(sentences):
        if keyword in sent:
            for s in sentences[max(0, i - 1):min(len(sentences), i + 2)]:
                if s not in result:
                    result.append(s)
    return ". ".join(result[:max_sentences])


def _prefilter_news(
    news_items: list[dict],
    company_name: str,
    noise_ratio_threshold: float = 3.0,
) -> list[dict]:
    """
    AI 전달 전 로컬 선 필터링 4단계:
    1. 제목+요약 모두에 종목명 없으면 즉시 폐기 (하드 필터)
    2. 제목에 없고 요약에만 있으면 하순위 (펀드·간접 언급)
    3. 노이즈 회사명이 종목명보다 noise_ratio_threshold배 초과하면 하순위
    4. summary 없는 항목은 종목명 주변 문장을 snippet으로 보강
    """
    if not company_name or not news_items:
        return news_items

    high: list[dict] = []
    low: list[dict] = []

    for item in news_items:
        title   = item.get("title", "")
        summary = item.get("summary", "")
        text    = title + " " + summary

        title_hit   = company_name in title
        summary_hit = company_name in summary

        # 1. 하드 필터: 제목·요약 모두에 종목명 없음 → 즉시 폐기
        if not title_hit and not summary_hit:
            continue

        # 2. 제목에 없고 요약에만 있음 → 하순위 (펀드·간접 언급)
        if not title_hit:
            low.append(item)
            continue

        # 3. 노이즈 비율
        company_count = max(text.count(company_name), 1)
        noise_count   = sum(text.count(firm) for firm in _NOISE_FIRMS)
        if noise_count > company_count * noise_ratio_threshold:
            low.append(item)
            continue

        # 4. summary 보강
        if not summary:
            snippet = _extract_sentences_near(text, company_name, max_sentences=3)
            if snippet:
                item = dict(item)
                item["summary"] = snippet

        high.append(item)

    return high + low


def _calc_sector_score(ticker: str, news_items: list[dict]) -> float:
    """섹터 업황 키워드 기반 섹터 점수 계산. 매핑 없으면 0.0 반환."""
    sector = _TICKER_SECTOR.get(ticker, "")
    if not sector:
        return 0.0
    kw_def = _SECTOR_KEYWORDS.get(sector, {})
    pos_kws = kw_def.get("pos", [])
    neg_kws = kw_def.get("neg", [])
    if not pos_kws and not neg_kws:
        return 0.0

    weighted_sum = 0.0
    count = 0
    for item in news_items:
        text = item.get("title", "") + " " + item.get("summary", "")
        decay = _news_time_decay(item.get("pub_date", ""))
        pos_hits = sum(1 for kw in pos_kws if kw in text)
        neg_hits = sum(1 for kw in neg_kws if kw in text)
        if pos_hits or neg_hits:
            raw = max(-1.0, min(1.0, (pos_hits - neg_hits) * 0.5))
            weighted_sum += raw * decay
            count += 1

    if count == 0:
        return 0.0
    return round(max(-5.0, min(5.0, (weighted_sum / count) * 5.0)), 2)


def get_naver_news(ticker_code: str, max_items: int = 10) -> list[dict]:
    """
    네이버 금융에서 특정 종목 뉴스 스크래핑 (국내 주식 전용)
    ticker_code: "005930.KS" 또는 "005930" 형태 모두 허용
    반환: [{"title": str, "link": str, "publisher": str, "pub_date": str}, ...]
    """
    if not HAS_BS4:
        return []

    code = ticker_code.split(".")[0].strip()
    if not code.isdigit():
        return []

    url = f"https://finance.naver.com/item/news_news.naver?code={code}&page=1"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Referer": "https://finance.naver.com/",
        "Accept-Language": "ko-KR,ko;q=0.9",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            return []
        resp.encoding = "euc-kr"
        soup = BeautifulSoup(resp.text, "html.parser")

        items = []
        for row in soup.select("table.type5 tr"):
            title_el = row.select_one("td.title a")
            info_el  = row.select_one("td.info")
            date_el  = row.select_one("td.date")
            if not title_el:
                continue
            href = title_el.get("href", "")
            if href and not href.startswith("http"):
                href = "https://finance.naver.com" + href
            items.append({
                "title":     title_el.get_text(strip=True),
                "link":      href,
                "publisher": info_el.get_text(strip=True) if info_el else "",
                "pub_date":  date_el.get_text(strip=True) if date_el else "",
            })
            if len(items) >= max_items:
                break
        return items
    except Exception:
        return []


def analyze_news_sentiment_keywords(
    news_items: list[dict],
    ticker: str = "",
    company_name: str = "",
) -> dict:
    """
    A/B/C 등급 가중치 + 시간 감쇠 + 섹터 블렌딩 키워드 감성 분석.

    특수 처리:
      EVENT — 합병·분할·거래정지 뉴스는 점수 제외, event_flags 목록에 기록
      SKIP  — company_name이 비교군으로만 언급된 뉴스는 점수 제외

    등급별 집계 가중치:
      A급 (실적·판매량·M&A):  2.0
      B급 (업황·ETF·목표가):  1.0
      C급 (거시·시황):        0.5

    반환: {
      "score": float(-5~+5),
      "individual_score": float,
      "sector_score": float,
      "label": str,
      "detail": list[dict],
      "event_flags": list[dict],   # 이벤트 대기 항목
    }
    """
    if not news_items:
        return {
            "score": 0.0, "individual_score": 0.0,
            "sector_score": 0.0, "label": "중립",
            "detail": [], "event_flags": [],
        }

    _TIER_AGG_WEIGHT = {"A": 2.0, "B": 1.0, "C": 0.5}

    detail        = []
    event_flags   = []
    weighted_sum  = 0.0
    total_weight  = 0.0

    for item in news_items:
        title = item.get("title", "")
        text  = title + " " + item.get("summary", "")
        decay = _news_time_decay(item.get("pub_date", ""))

        base_score, tier, matched_kw = _classify_news_tier(text)

        base_detail = {
            "title":      title,
            "link":       item.get("link", "#"),
            "publisher":  item.get("publisher", ""),
            "pub_date":   item.get("pub_date", ""),
            "matched_kw": matched_kw[:4],
            "decay":      round(decay, 2),
        }

        # ── EVENT: 이벤트 대기 처리 (점수 합산 제외) ───────────────────────────
        if tier == "EVENT":
            event_flags.append({"title": title, "event_kw": matched_kw})
            detail.append({
                **base_detail,
                "score":  0.0,
                "tier":   "EVENT",
                "skipped": True,
                "reason": f"[이벤트 대기] {'·'.join(matched_kw[:3])} — 기술 분석 제외",
            })
            continue

        # ── SKIP: 비교군 언급 — 직접 관련 뉴스 아님 ────────────────────────────
        if company_name:
            is_relevant, skip_reason = _check_news_relevance(text, company_name)
            if not is_relevant:
                detail.append({
                    **base_detail,
                    "score":  0.0,
                    "tier":   "SKIP",
                    "skipped": True,
                    "reason": f"[제외] {skip_reason} — 직접 관련 뉴스 아님",
                })
                continue

        item_score = round(base_score * decay, 2)
        agg_weight = _TIER_AGG_WEIGHT.get(tier, 1.0)

        weighted_sum += item_score * agg_weight
        total_weight += agg_weight

        detail.append({
            **base_detail,
            "score":   item_score,
            "tier":    tier,
            "skipped": False,
            "hidden":  tier == "C" and not matched_kw,
            "reason":  (
                f"[{tier}급] {'·'.join(matched_kw[:3]) or '키워드 없음'}"
                f"  시간감쇠×{decay:.1f}"
            ),
        })

    avg              = weighted_sum / max(total_weight, 1.0)
    individual_score = round(max(-5.0, min(5.0, avg * 5.0)), 2)

    sector_score = _calc_sector_score(ticker, news_items) if ticker else 0.0

    if sector_score != 0.0:
        score = round(max(-5.0, min(5.0, 0.7 * individual_score + 0.3 * sector_score)), 2)
    else:
        score = individual_score

    if   score >= 3:  label = "매우 긍정"
    elif score >= 1:  label = "긍정"
    elif score >= -1: label = "중립"
    elif score >= -3: label = "부정"
    else:             label = "매우 부정"

    return {
        "score":            score,
        "individual_score": individual_score,
        "sector_score":     sector_score,
        "label":            label,
        "detail":           detail,
        "event_flags":      event_flags,
    }


def analyze_news_sentiment_llm(
    news_items: list[dict],
    ticker: str,
    api_key: str,
    groq_api_key: str = "",
    company_name: str = "",
) -> dict:
    """
    뉴스 감성 분석: 1순위 Gemini → 2순위 Groq(쿼터 초과 시) → 3순위 키워드 분석.
    - [선 필터링] 종목명 미포함·노이즈 과다 뉴스 사전 제거
    - [캐싱] 제목 해시 기반 1시간 인메모리 캐시 (중복 API 호출 제거)
    - 중복 제거 → 상위 10개 배치 처리 → API 1회 호출로 0~100 투자 심리 지수 도출
    - 이벤트 뉴스(합병·분할·거래정지) 및 비관련 뉴스(비교군 언급) 자동 제외
    - 반환: {"score": float(-5~+5), "label": str, "detail": list[dict],
              "summary": str, "individual_score": float, "sector_score": float,
              "event_flags": list[dict]}
    """
    if not api_key or not news_items:
        return analyze_news_sentiment_keywords(news_items, ticker, company_name)

    # ── 캐시 확인 ────────────────────────────────────────────────────────────
    _cache_key = _news_cache_key(news_items, ticker)
    _cached = _get_cached_news_result(_cache_key)
    if _cached is not None:
        return _cached

    # ── 선 필터링 (로컬, 즉시) ───────────────────────────────────────────────
    filtered_items = _prefilter_news(news_items, company_name)
    # 필터링 후 뉴스가 없으면 원본으로 폴백 (company_name 미제공 등)
    effective_items = filtered_items if filtered_items else news_items

    kw_result    = analyze_news_sentiment_keywords(effective_items, ticker, company_name)
    sector_score = kw_result.get("sector_score", 0.0)

    try:
        import json as _json

        deduped   = _deduplicate_news(effective_items)
        top_news  = _select_top_news(deduped, n=10)
        ticker_display = ticker.replace(".KS", "").replace(".KQ", "")
        display_name   = company_name or ticker_display
        headlines = "\n".join(
            f"{i+1}. {_extract_news_snippet(it)} ({it.get('pub_date', '날짜미상')})"
            for i, it in enumerate(top_news)
        )

        prompt = f"""당신은 주식 애널리스트입니다. 아래 {len(top_news)}개 뉴스가 종목 '{display_name}'의 향후 7일 주가에 미칠 투자 심리를 종합 평가하세요.

[뉴스 목록]
{headlines}

[분석 지침]
- 뉴스 전체를 종합한 투자 심리 지수를 0~100 사이 정수로 평가하세요. (0=매우 부정, 50=중립, 100=매우 긍정)
- 실적/수주/판매량 등 직접 재무 영향 뉴스는 가중치를 높게 부여하세요.
- 업황/ETF/시황 뉴스는 간접 영향이므로 보수적으로 반영하세요.
- 주식 병합·분할·거래 정지·관리종목·상장폐지 등 이벤트성 뉴스가 있다면 "event_type" 필드에 기재하고 sentiment_index 산정에서 제외하세요.
- 각 뉴스가 '{display_name}'에 직접적인 소식인지 판단하세요. 다른 회사가 주인공이고 '{display_name}'은 단순 비교군으로만 언급된 경우 해당 뉴스는 제외하고 "skipped_irrelevant" 필드에 제외 수를 기재하세요.

[출력 형식] JSON만 출력하고 다른 설명은 절대 포함하지 마세요:
{{
  "sentiment_index": <0~100 정수>,
  "summary": "<종목에 미치는 핵심 영향 2~3문장 한국어>",
  "event_type": <이벤트 유형 문자열(합병/분할/거래정지 등) 또는 null>,
  "skipped_irrelevant": <비관련 뉴스로 제외한 수(기본값 0)>
}}"""

        # ── 1순위: Gemini ──────────────────────────────────────────────────────
        raw = None
        _provider = "Gemini"
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain_core.messages import HumanMessage
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=api_key,
                temperature=0.0,
            )
            raw = llm.invoke([HumanMessage(content=prompt)]).content.strip()
        except Exception as _gemini_exc:
            _msg = str(_gemini_exc)
            _is_quota = (
                "429" in _msg
                or "quota" in _msg.lower()
                or "ResourceExhausted" in type(_gemini_exc).__name__
            )
            # ── 2순위: Groq (쿼터 초과이고 Groq 키가 있을 때만) ────────────
            if _is_quota and groq_api_key:
                try:
                    from groq import Groq as _Groq
                    _client = _Groq(api_key=groq_api_key)
                    _resp = _client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=512,
                    )
                    raw = _resp.choices[0].message.content.strip()
                    _provider = "Groq"
                except Exception:
                    raise _gemini_exc  # Groq도 실패하면 원래 Gemini 에러로 폴백
            else:
                raise

        # ── JSON 파싱 (Gemini/Groq 공통) ──────────────────────────────────────
        json_match = _re_global.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```|(\{[\s\S]*\})', raw)
        if not json_match:
            raise ValueError("JSON not found in response")
        json_str = json_match.group(1) or json_match.group(2)
        parsed, _ = _json.JSONDecoder().raw_decode(json_str.strip())

        if "sentiment_index" not in parsed:
            raise ValueError("Missing required fields in response")

        sentiment_index     = max(0.0, min(100.0, float(parsed["sentiment_index"])))
        overall_score       = round((sentiment_index - 50.0) / 10.0, 2)
        summary             = parsed.get("summary", "")
        llm_event_type      = parsed.get("event_type") or None
        llm_skipped_cnt     = int(parsed.get("skipped_irrelevant", 0) or 0)
        if _provider != "Gemini":
            summary = f"[{_provider}] {summary}"

        # LLM 이벤트 감지 → summary 에 이벤트 메시지 추가
        if llm_event_type:
            summary = f"[이벤트 대기: {llm_event_type}] {summary}".strip()
        if llm_skipped_cnt > 0:
            summary = f"{summary}  (비관련 뉴스 {llm_skipped_cnt}건 제외)".strip()

        # 키워드 분석의 event_flags + detail 병합
        event_flags     = kw_result.get("event_flags", [])
        detail_by_title = {d["title"]: d for d in kw_result["detail"]}

        # LLM이 감지한 이벤트가 있고 키워드 분석에서 못 잡은 경우 summary로만 표시
        if llm_event_type and not event_flags:
            event_flags = [{"title": llm_event_type, "event_kw": [llm_event_type]}]

        if sector_score != 0.0:
            blended = round(max(-5.0, min(5.0, 0.7 * overall_score + 0.3 * sector_score)), 2)
        else:
            blended = round(overall_score, 2)

        if   blended >= 3:  label = "매우 긍정"
        elif blended >= 1:  label = "긍정"
        elif blended >= -1: label = "중립"
        elif blended >= -3: label = "부정"
        else:               label = "매우 부정"

        _result = {
            "score":            blended,
            "individual_score": round(overall_score, 2),
            "sector_score":     sector_score,
            "label":            label,
            "detail":           list(detail_by_title.values()),
            "summary":          summary,
            "event_flags":      event_flags,
        }
        _set_cached_news_result(_cache_key, _result)
        return _result

    except Exception as _exc:
        result = kw_result.copy()
        _msg = str(_exc)
        if "429" in _msg or "quota" in _msg.lower() or "ResourceExhausted" in type(_exc).__name__:
            if groq_api_key:
                result["summary"] = "(Gemini 쿼터 초과 + Groq 실패 — 키워드+섹터 분석으로 대체)"
            else:
                result["summary"] = "(Gemini API 일일 쿼터 초과 — 키워드+섹터 분석으로 대체)"
        elif "rate" in _msg.lower() or "limit" in _msg.lower():
            result["summary"] = "(Gemini API 요청 한도 초과 — 키워드+섹터 분석으로 대체)"
        else:
            result["summary"] = f"(AI 분석 실패: {type(_exc).__name__}: {_msg[:120]} — 키워드+섹터 분석으로 대체)"
        return result


def fetch_article_content(url: str) -> str:
    """
    기사 URL에서 본문 텍스트 스크래핑 (최대 2000자).
    네이버 뉴스 / 일반 HTML 페이지 지원.
    """
    if not HAS_BS4 or not url or url == "#":
        return ""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "ko-KR,ko;q=0.9",
        "Referer": "https://finance.naver.com/",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.encoding = resp.apparent_encoding or "utf-8"
        soup = BeautifulSoup(resp.text, "html.parser")

        # 네이버 뉴스 본문 선택자 우선 시도
        for sel in [
            "#dic_area", "#newsct_article", "#articeBody",
            ".article-body", ".news_end", "article",
        ]:
            el = soup.select_one(sel)
            if el:
                return el.get_text(separator="\n", strip=True)[:2000]

        # 폴백: 모든 <p> 태그 합산
        paras = [p.get_text(strip=True) for p in soup.find_all("p") if p.get_text(strip=True)]
        return "\n".join(paras[:30])[:2000]
    except Exception:
        return ""


def analyze_news_batch(
    ticker_configs: list[dict],
    api_key: str,
    groq_api_key: str = "",
    max_workers: int = 5,
) -> dict[str, dict]:
    """
    여러 종목 뉴스를 ThreadPoolExecutor로 병렬 분석.

    ticker_configs: [
        {"ticker": str, "news": list[dict], "company_name": str},
        ...
    ]
    반환: {ticker: analyze_news_sentiment_llm 결과, ...}

    - 각 종목은 독립적으로 비동기 처리 (I/O 대기 동시 실행)
    - 캐시 히트 시 즉시 반환, 미스 시에만 API 호출
    - API 호출 실패한 종목은 키워드 분석 결과로 자동 대체
    """
    if not ticker_configs:
        return {}

    def _analyze_one(cfg: dict) -> tuple[str, dict]:
        t = cfg["ticker"]
        news = cfg.get("news", [])
        cname = cfg.get("company_name", "")
        try:
            result = analyze_news_sentiment_llm(news, t, api_key, groq_api_key, cname)
        except Exception:
            result = analyze_news_sentiment_keywords(news, t, cname)
        return t, result

    results: dict[str, dict] = {}
    n = min(max_workers, len(ticker_configs))
    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = {pool.submit(_analyze_one, cfg): cfg["ticker"] for cfg in ticker_configs}
        for future in as_completed(futures):
            try:
                ticker, result = future.result()
                results[ticker] = result
            except Exception:
                t = futures[future]
                results[t] = {"score": 0.0, "label": "중립", "detail": [], "event_flags": []}
    return results


def summarize_article_llm(
    title: str,
    link: str,
    ticker: str,
    api_key: str,
) -> dict:
    """
    단일 기사를 Gemini AI로 요약.
    반환: {
        "summary": str,          # 핵심 내용 3~5문장
        "sentiment": str,        # 긍정/중립/부정
        "score": float,          # -1.0 ~ 1.0
        "key_points": list[str], # 핵심 포인트
        "investment_implication": str,  # 투자 시사점
        "used_content": bool,    # 본문 스크래핑 성공 여부
    }
    """
    if not api_key:
        return {
            "summary": "AI 요약을 사용하려면 Gemini API 키를 입력하세요.",
            "sentiment": "N/A", "score": 0.0,
            "key_points": [], "investment_implication": "",
            "used_content": False,
        }

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage

        # 기사 본문 스크래핑 시도
        body = fetch_article_content(link)
        used_content = bool(body)

        if used_content:
            article_text = f"기사 제목: {title}\n\n기사 본문:\n{body}"
        else:
            article_text = f"기사 제목: {title}\n\n(본문 스크래핑 불가 — 제목만으로 분석합니다)"

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=api_key,
            temperature=0.1,
        )

        prompt = f"""다음 뉴스 기사를 분석해 주세요. 분석 대상 종목: {ticker}

{article_text}

아래 JSON 형식으로만 답하세요 (설명 없이 JSON만):
{{
  "summary": "<기사 핵심 내용 3~5문장 요약>",
  "sentiment": "<긍정/중립/부정 중 하나>",
  "score": <-1.0 ~ 1.0 사이 소수 감성 점수>,
  "key_points": ["<핵심 포인트 1>", "<핵심 포인트 2>", "<핵심 포인트 3>"],
  "investment_implication": "<투자자 관점 시사점 1~2문장>"
}}"""

        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()

        import json as _json, re as _re
        match = _re.search(r'\{[\s\S]*\}', raw)
        if not match:
            raise ValueError("JSON not found in response")
        parsed, _ = _json.JSONDecoder().raw_decode(match.group().strip())

        return {
            "summary":                parsed.get("summary", "요약 없음"),
            "sentiment":              parsed.get("sentiment", "중립"),
            "score":                  float(parsed.get("score", 0.0)),
            "key_points":             parsed.get("key_points", []),
            "investment_implication": parsed.get("investment_implication", ""),
            "used_content":           used_content,
        }

    except Exception as exc:
        return {
            "summary": f"AI 요약 실패: {str(exc)[:120]}",
            "sentiment": "N/A", "score": 0.0,
            "key_points": [], "investment_implication": "",
            "used_content": False,
        }


def get_hybrid_signal(technical_score: float, news_score: float) -> dict:
    """
    기술적 신호(70%) + 뉴스 감성(30%) 하이브리드 종합 점수
    technical_score: -10 ~ +10 (generate_signals 반환값)
    news_score:      -5  ~ +5  (analyze_news_sentiment_* 반환값)
    반환: {"hybrid_score": float, "label": str, "badge": str}
    """
    # 두 점수를 동일 스케일(-10~+10)로 정규화 후 가중 합산
    news_norm  = news_score * 2.0      # -5~+5  → -10~+10
    hybrid_raw = technical_score * 0.7 + news_norm * 0.3
    hybrid     = round(max(-10.0, min(10.0, hybrid_raw)), 2)

    s = int(round(hybrid))
    if   s >= 5:  label, badge = "강력 매수", "🟢🟢"
    elif s >= 3:  label, badge = "매수",      "🟢"
    elif s >= 1:  label, badge = "약한 매수", "🔵"
    elif s == 0:  label, badge = "중립/관망", "⚪"
    elif s >= -2: label, badge = "약한 매도", "🟡"
    elif s >= -4: label, badge = "매도",      "🔴"
    else:         label, badge = "강력 매도", "🔴🔴"

    return {"hybrid_score": hybrid, "label": label, "badge": badge}


def get_investment_recommendation(
    current_price: float,
    avg_price: float,
    indicators: dict,
    tech_score: float,
    news_score: float,
    fund_score: float,
    dead_time: dict | None = None,
) -> dict:
    """
    평단가 기반 개인화 매매 추천.

    우선순위: 손절 > 익절 > 매수유보(dead_time) > 추가매수 > 보유 > 관망

    dead_time (check_dead_time 결과):
      buy_hold=True  → 매수 계열 액션을 '매수 유보'로 전환
                       (거래량이 기준 대비 80% 이하 — 슬리피지·유동성 위험)

    반환: {
      "action":     str,   # "손절검토" | "익절" | "매수유보" | "추가매수" | "보유" | "관망"
      "badge":      str,
      "title":      str,
      "color_bg":   str,
      "color_fg":   str,
      "profit_rate": float,
      "reason":     str,
      "details":    list[str],
    }
    """
    if avg_price <= 0 or current_price <= 0:
        return {
            "action": "오류", "badge": "—", "title": "입력 오류",
            "color_bg": "#1e2130", "color_fg": "#bdbdbd",
            "profit_rate": 0.0, "reason": "평단가 또는 현재가가 0입니다.",
            "details": [],
        }

    profit_rate = (current_price - avg_price) / avg_price * 100
    rsi       = float(indicators.get("RSI",       50.0) or 50.0)
    macd_hist = float(indicators.get("MACD_Hist",  0.0) or 0.0)
    bb_lower  = indicators.get("BB_Lower")
    bb_upper  = indicators.get("BB_Upper")
    is_profit = profit_rate > 0

    # ── 손절 검토 ─────────────────────────────────────────────────────────────
    if profit_rate <= -7.0 and tech_score <= -1 and macd_hist < 0:
        return {
            "action": "손절검토", "badge": "🛑", "title": "손절 검토",
            "color_bg": "#3a1212", "color_fg": "#ef9a9a",
            "profit_rate": profit_rate,
            "reason": f"손실 {profit_rate:.1f}% — 오닐 -7% 기준 초과 + 하락 지표 확인",
            "details": [
                f"현재 손실 {profit_rate:.1f}% (오닐 손절 기준 -7% 초과)",
                f"기술 신호 {tech_score:+.1f}점 (하락 추세 강화)",
                f"MACD 히스토그램 음수 ({macd_hist:+.4f})",
                f"뉴스 감성 {news_score:+.1f}점",
                "✂️ 추가 손실 방지를 위한 리스크 관리 권장",
            ],
        }

    if profit_rate <= -5.0 and tech_score <= -2:
        return {
            "action": "손절검토", "badge": "⚠️", "title": "손절 주의",
            "color_bg": "#2d1818", "color_fg": "#ffab91",
            "profit_rate": profit_rate,
            "reason": f"손실 {profit_rate:.1f}% — 손절선 -7% 접근, 기술 지표 약세",
            "details": [
                f"현재 손실 {profit_rate:.1f}% (손절 기준 -7% 접근)",
                f"기술 신호 {tech_score:+.1f}점 (약세)",
                f"RSI {rsi:.0f}",
                "⚠️ 추이 모니터링 — 이탈 시 손절 실행 고려",
            ],
        }

    # ── 익절 추천 ─────────────────────────────────────────────────────────────
    if profit_rate >= 10.0 and rsi >= 70:
        return {
            "action": "익절", "badge": "💰", "title": "익절 추천",
            "color_bg": "#0d2b1a", "color_fg": "#c8e6c9",
            "profit_rate": profit_rate,
            "reason": f"수익 {profit_rate:.1f}% + RSI {rsi:.0f} 과매수 — 차익 실현 시점",
            "details": [
                f"수익 {profit_rate:.1f}% 확보",
                f"RSI {rsi:.0f} (70 이상 = 과매수 구간)",
                f"기술 신호 {tech_score:+.1f}점",
                f"뉴스 감성 {news_score:+.1f}점",
                "💡 일부 또는 전량 익절 고려 / 트레일링 스탑 활용 가능",
            ],
        }

    if profit_rate >= 15.0 and rsi >= 65 and macd_hist < 0:
        return {
            "action": "익절", "badge": "💰", "title": "익절 고려",
            "color_bg": "#0d2b1a", "color_fg": "#a5d6a7",
            "profit_rate": profit_rate,
            "reason": f"수익 {profit_rate:.1f}% + MACD 약화 — 추세 전환 징후",
            "details": [
                f"수익 {profit_rate:.1f}% 확보",
                f"MACD 히스토그램 음전환 ({macd_hist:+.4f})",
                f"RSI {rsi:.0f}",
                "💡 부분 익절 또는 손절선 상향 조정 고려",
            ],
        }

    # ── 매수 유보 가드 (dead_time: 거래량 80% 이하) ──────────────────────────
    # 손절·익절 이후 단계에서만 적용 — 이미 보유 중인 포지션 청산은 차단하지 않음
    _dt        = dead_time or {}
    _buy_hold  = _dt.get("buy_hold", False)
    _vol_ratio = _dt.get("vol_ratio", 1.0)
    if _buy_hold:
        return {
            "action":     "매수유보",
            "badge":      "⏸️",
            "title":      "매수 권장 유보",
            "color_bg":   "#1e1e12",
            "color_fg":   "#fff176",
            "profit_rate": profit_rate,
            "reason": (
                f"최근 거래량이 기준 대비 {_vol_ratio*100:.0f}% 수준 (80% 미달) — "
                "유동성 부족으로 매수 시 슬리피지 위험 높음"
            ),
            "details": [
                f"거래량 비율: 최근 14일 평균이 기준(60일) 대비 {_vol_ratio*100:.0f}%",
                "거래량이 회복될 때까지 신규 매수는 유보 권장",
                "기존 보유 포지션은 그대로 유지",
                f"기술 신호 {tech_score:+.1f}점 / RSI {rsi:.0f} (참고용)",
                "💡 거래량이 기준 대비 80% 이상 회복 후 재진입 검토",
            ],
        }

    # ── 추가 매수 ─────────────────────────────────────────────────────────────
    if (not is_profit and fund_score >= 5 and rsi < 40
            and tech_score >= 0 and macd_hist > 0):
        return {
            "action": "추가매수", "badge": "✨", "title": "추가 매수 유효",
            "color_bg": "#0a1e30", "color_fg": "#90caf9",
            "profit_rate": profit_rate,
            "reason": f"손실 {profit_rate:.1f}% — 우량주 저점, 평단가 낮추기 기회",
            "details": [
                f"현재 손실 {profit_rate:.1f}% (평단가 대비 저렴)",
                f"펀더멘털 점수 {fund_score:+.1f}점 (우량 기업)",
                f"RSI {rsi:.0f} (40 미만 = 저평가 구간)",
                f"MACD 히스토그램 양전환 ({macd_hist:+.4f})",
                "💡 기존 비중의 30~50% 이내 분할 추가 매수 권장",
            ],
        }

    if not is_profit and fund_score >= 3 and rsi < 40:
        return {
            "action": "추가매수", "badge": "📉➕", "title": "추가 매수 검토",
            "color_bg": "#101828", "color_fg": "#80cbc4",
            "profit_rate": profit_rate,
            "reason": f"손실 {profit_rate:.1f}% — 펀더멘털 양호, 기술 반전 대기",
            "details": [
                f"현재 손실 {profit_rate:.1f}%",
                f"펀더멘털 점수 {fund_score:+.1f}점 (양호)",
                f"RSI {rsi:.0f} (저평가 구간)",
                f"MACD {macd_hist:+.4f} (양전환 대기 중)",
                "⚠️ MACD 양전환 확인 후 매수 권장",
            ],
        }

    # ── 보유 ─────────────────────────────────────────────────────────────────
    if is_profit and rsi < 70 and macd_hist >= 0 and tech_score >= 0:
        return {
            "action": "보유", "badge": "📈", "title": "보유 (추세 유지)",
            "color_bg": "#121e14", "color_fg": "#a5d6a7",
            "profit_rate": profit_rate,
            "reason": f"수익 {profit_rate:.1f}% — 상승 모멘텀 지속, 과열 없음",
            "details": [
                f"수익 {profit_rate:.1f}% 진행 중",
                f"RSI {rsi:.0f} (과매수 아님)",
                f"MACD 히스토그램 양수 ({macd_hist:+.4f})",
                f"기술 신호 {tech_score:+.1f}점",
                "💡 트레일링 스탑 설정으로 수익 보호 권장",
            ],
        }

    if not is_profit and bb_lower is not None and current_price > bb_lower:
        return {
            "action": "보유", "badge": "🛡️", "title": "보유 (지지선 확인)",
            "color_bg": "#1e1e12", "color_fg": "#fff176",
            "profit_rate": profit_rate,
            "reason": f"손실 {profit_rate:.1f}% — BB 하단 지지선 위, 반등 대기",
            "details": [
                f"현재 손실 {profit_rate:.1f}%",
                f"BB 하단({bb_lower:,.2f}) 위에서 지지 중",
                f"RSI {rsi:.0f}",
                "⚠️ BB 하단 이탈 시 손절 재검토 필요",
            ],
        }

    # ── 관망 (기본) ───────────────────────────────────────────────────────────
    direction = "수익" if is_profit else "손실"
    return {
        "action": "관망", "badge": "😐", "title": "관망 / 보유",
        "color_bg": "#1e2130", "color_fg": "#bdbdbd",
        "profit_rate": profit_rate,
        "reason": f"{direction} {abs(profit_rate):.1f}% — 특이 신호 없음, 방향 확인 필요",
        "details": [
            f"수익률 {profit_rate:+.1f}%",
            f"기술 신호 {tech_score:+.1f}점",
            f"뉴스 감성 {news_score:+.1f}점",
            f"RSI {rsi:.0f}",
            "😐 뚜렷한 매매 신호 없음 — 추가 지표 확인 후 판단",
        ],
    }


# ─── 확장 감성 분석 ───────────────────────────────────────────────────────────

# 4단계 키워드 사전 (가중치: 초강세 +2 / 강세 +1 / 약세 -1 / 초악재 -2)
_ADV_KEYWORDS: dict[str, tuple[int, list[str]]] = {
    "초강세": (2, [
        "상한가", "독점", "역대 최대", "역대 최고", "공급계약", "M&A", "인수합병",
        "어닝서프라이즈", "FDA 승인", "상업화", "양산", "세계 최초", "국내 최초",
        "대규모 수주", "턴어라운드",
    ]),
    "강세": (1, [
        "수주", "특허", "흑자전환", "신기술", "양산 개시", "최초", "목표가 상향",
        "신고가", "급등", "성장", "배당 확대", "자사주 매입", "실적 개선",
        "수출 증가", "신제품", "파트너십", "투자 유치", "상승", "흑자",
    ]),
    "약세": (-1, [
        "유상증자", "소송", "조사", "적자", "하향", "과징금", "공매도",
        "실적 부진", "매출 감소", "목표가 하향", "악재", "하락", "규제",
        "리콜", "납품 중단", "계약 해지",
    ]),
    "초악재": (-2, [
        "횡령", "부도", "상장폐지", "검찰 수사", "부적정", "디폴트",
        "분식회계", "대표이사 구속", "영업 정지", "파산", "최대주주 매도",
    ]),
}


def get_advanced_sentiment(news_list: list[dict]) -> dict:
    """
    4단계 키워드 기반 확장 감성 분석.
    반환:
      score       float   -5 ~ +5 정규화 점수
      summary     str     주요 키워드 포착 문구
      hits        dict    {"초강세": [...], "강세": [...], "약세": [...], "초악재": [...]}
      all_hits    list    중복 제거된 전체 히트 키워드 (최대 10개)
      detail      list    기사별 {"title", "link", "score", "hits": [...]}
    """
    if not news_list:
        return {"score": 0.0, "summary": "분석 데이터 없음",
                "hits": {}, "all_hits": [], "detail": []}

    total_score = 0.0
    hits_by_cat: dict[str, set] = {c: set() for c in _ADV_KEYWORDS}
    detail = []

    for item in news_list[:10]:
        title = item.get("title", "")
        art_score = 0.0
        art_hits: list[str] = []

        for category, (weight, words) in _ADV_KEYWORDS.items():
            for word in words:
                if word in title:
                    art_score += weight
                    art_hits.append(word)
                    hits_by_cat[category].add(word)

        # 기사 1건 점수 클리핑: -2 ~ +2
        art_score = max(-2.0, min(2.0, art_score))
        total_score += art_score
        detail.append({
            "title":  title,
            "link":   item.get("link", "#"),
            "publisher": item.get("publisher", ""),
            "pub_date":  item.get("pub_date", ""),
            "score":  round(art_score, 1),
            "hits":   art_hits,
        })

    # 전체 점수: 평균을 5배 스케일 (-5~+5)
    avg   = total_score / len(news_list)
    score = round(max(-5.0, min(5.0, avg * 2.5)), 2)

    # 요약 문구
    all_hits_ordered: list[str] = []
    for cat in ["초강세", "강세", "약세", "초악재"]:
        all_hits_ordered.extend(list(hits_by_cat[cat]))
    top3 = all_hits_ordered[:3]
    summary = f"'{', '.join(top3)}' 등 키워드 포착" if top3 else "특이 키워드 없음"

    hits_serializable = {c: list(v) for c, v in hits_by_cat.items()}

    return {
        "score":     score,
        "summary":   summary,
        "hits":      hits_serializable,
        "all_hits":  all_hits_ordered[:10],
        "detail":    detail,
    }


# ─── 섹터 동조화 분석 ─────────────────────────────────────────────────────────

# 종목별 연관 섹터 종목 매핑 (대표 경쟁/동반 종목 2~4개)
_SECTOR_MAP: dict[str, list[str]] = {
    # 국내 반도체
    "005930.KS": ["000660.KS", "000990.KS", "042700.KS"],
    "000660.KS": ["005930.KS", "000990.KS", "042700.KS"],
    "042700.KS": ["005930.KS", "000660.KS"],
    # 국내 플랫폼
    "035420.KS": ["035720.KS", "259960.KS"],
    "035720.KS": ["035420.KS", "293490.KQ"],
    # 국내 자동차
    "005380.KS": ["000270.KS", "012330.KS"],
    "000270.KS": ["005380.KS", "012330.KS"],
    # 국내 배터리
    "373220.KS": ["006400.KS", "051910.KS"],
    "006400.KS": ["373220.KS", "051910.KS"],
    "051910.KS": ["373220.KS", "006400.KS"],
    # 국내 바이오/제약
    "207940.KS": ["068270.KS", "128940.KS", "000100.KS"],
    "068270.KS": ["207940.KS", "128940.KS"],
    "128940.KS": ["068270.KS", "207940.KS"],
    # 국내 철강/소재
    "005490.KS": ["004020.KS", "010130.KS"],
    "004020.KS": ["005490.KS", "010130.KS"],
    # 미국 빅테크
    "AAPL":  ["MSFT", "GOOGL", "META", "AMZN"],
    "MSFT":  ["AAPL", "GOOGL", "AMZN", "CRM"],
    "GOOGL": ["META", "MSFT", "SNAP", "TTD"],
    "META":  ["GOOGL", "SNAP", "PINS", "RDDT"],
    "AMZN":  ["MSFT", "GOOGL", "WMT", "SHOP"],
    # 미국 반도체
    "NVDA":  ["AMD", "INTC", "AVGO", "QCOM"],
    "AMD":   ["NVDA", "INTC", "AVGO"],
    "INTC":  ["NVDA", "AMD",  "AVGO"],
    "AVGO":  ["NVDA", "AMD",  "QCOM"],
    # 미국 전기차/자동차
    "TSLA":  ["GM", "F", "RIVN", "NIO"],
    "GM":    ["TSLA", "F", "STLA"],
    "F":     ["TSLA", "GM", "STLA"],
    # 미국 금융
    "JPM":   ["BAC", "GS", "MS", "WFC"],
    "BAC":   ["JPM", "GS", "MS", "C"],
    "GS":    ["JPM", "MS", "BAC"],
    # 국내 ETF → 구성종목 (섹터 대표성)
    "069500.KS": ["005930.KS", "000660.KS", "005380.KS", "373220.KS"],
    "091160.KS": ["005930.KS", "000660.KS", "042700.KS", "000990.KS"],
    "091170.KS": ["005380.KS", "000270.KS", "012330.KS"],
    "305720.KS": ["373220.KS", "006400.KS", "051910.KS"],
    "091220.KS": ["105560.KS", "055550.KS", "086790.KS", "138930.KS"],
    "143860.KS": ["207940.KS", "068270.KS", "128940.KS"],
    "266370.KS": ["005930.KS", "035420.KS", "035720.KS"],
    "228790.KS": ["051910.KS", "011170.KS", "006400.KS"],
    "232080.KS": ["247540.KQ", "086900.KQ", "035900.KQ"],
    "122630.KS": ["005930.KS", "000660.KS", "005380.KS"],
    "360750.KS": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"],
    "133690.KS": ["AAPL", "MSFT", "NVDA", "AMZN", "META"],
    # 국내 금융 (은행)
    "105560.KS": ["055550.KS", "086790.KS", "138930.KS", "316140.KS"],
    "055550.KS": ["105560.KS", "086790.KS", "138930.KS"],
    "086790.KS": ["105560.KS", "055550.KS", "316140.KS"],
    "138930.KS": ["105560.KS", "055550.KS", "086790.KS"],
    "316140.KS": ["105560.KS", "086790.KS"],
    # 국내 통신
    "017670.KS": ["030200.KS", "032640.KS"],
    "030200.KS": ["017670.KS", "032640.KS"],
    "032640.KS": ["017670.KS", "030200.KS"],
    # 국내 조선
    "329180.KS": ["010140.KS", "042660.KS", "009540.KS"],
    "010140.KS": ["329180.KS", "042660.KS", "009540.KS"],
    "042660.KS": ["329180.KS", "010140.KS", "009540.KS"],
    "009540.KS": ["329180.KS", "010140.KS", "042660.KS"],
    # 국내 방산
    "012450.KS": ["079550.KS", "047810.KS", "272210.KS"],
    "079550.KS": ["012450.KS", "047810.KS", "272210.KS"],
    "047810.KS": ["012450.KS", "079550.KS", "272210.KS"],
    "272210.KS": ["012450.KS", "079550.KS", "047810.KS"],
    # 국내 건설
    "000720.KS": ["006360.KS", "047040.KS", "028050.KS"],
    "006360.KS": ["000720.KS", "047040.KS", "028050.KS"],
    "047040.KS": ["000720.KS", "006360.KS", "028050.KS"],
    "028050.KS": ["000720.KS", "006360.KS"],
    # 국내 정유/에너지
    "096770.KS": ["010950.KS", "051910.KS"],
    "010950.KS": ["096770.KS", "051910.KS"],
    # 국내 화학
    "011170.KS": ["009830.KS", "096770.KS"],
    "009830.KS": ["011170.KS", "051910.KS"],
    # 국내 게임
    "259960.KS": ["251270.KS", "293490.KQ", "263750.KQ"],
    "251270.KS": ["259960.KS", "293490.KQ", "263750.KQ"],
    "293490.KQ": ["251270.KS", "259960.KS", "263750.KQ"],
    "263750.KQ": ["251270.KS", "293490.KQ"],
    # 국내 엔터
    "352820.KS": ["041510.KQ", "035900.KQ", "122870.KQ"],
    "041510.KQ": ["352820.KS", "035900.KQ", "122870.KQ"],
    "035900.KQ": ["352820.KS", "041510.KQ", "122870.KQ"],
    "122870.KQ": ["352820.KS", "041510.KQ", "035900.KQ"],
    # 국내 보험
    "032830.KS": ["088350.KS", "005830.KS", "000060.KS"],
    "088350.KS": ["032830.KS", "005830.KS"],
    "005830.KS": ["032830.KS", "088350.KS", "000060.KS"],
    "000060.KS": ["032830.KS", "005830.KS"],
    # 국내 증권
    "006800.KS": ["016360.KS", "005940.KS", "071050.KS"],
    "016360.KS": ["006800.KS", "005940.KS"],
    "005940.KS": ["006800.KS", "016360.KS"],
    "071050.KS": ["006800.KS", "016360.KS"],
    # 국내 철강 추가
    "010130.KS": ["005490.KS", "004020.KS"],
    # 국내 바이오 추가
    "000100.KS": ["207940.KS", "068270.KS", "128940.KS"],
    # 국내 자동차 추가
    "012330.KS": ["005380.KS", "000270.KS"],
    "241560.KS": ["005380.KS", "000270.KS"],
    # 국내 반도체 추가
    "000990.KS": ["005930.KS", "000660.KS", "042700.KS"],
    "066970.KQ": ["005930.KS", "000660.KS", "042700.KS"],
    # 국내 배터리 추가
    "247540.KQ": ["373220.KS", "006400.KS", "051910.KS"],
    # 미국 반도체 추가
    "QCOM":  ["NVDA", "AVGO", "TXN", "ARM"],
    "MU":    ["NVDA", "WDC",  "LRCX", "AMAT"],
    "TXN":   ["AVGO", "ADI",  "MCHP", "QCOM"],
    "LRCX":  ["AMAT", "KLAC", "ASML", "NVDA"],
    "AMAT":  ["LRCX", "KLAC", "NVDA"],
    "KLAC":  ["LRCX", "AMAT", "NVDA"],
    "ARM":   ["NVDA", "AMD",  "QCOM", "AVGO"],
    "MRVL":  ["NVDA", "AVGO", "AMD",  "QCOM"],
    # 미국 금융 추가
    "MS":    ["GS",   "JPM",  "BAC"],
    "WFC":   ["JPM",  "BAC",  "C"],
    "C":     ["JPM",  "BAC",  "WFC"],
    # 미국 헬스케어
    "JNJ":   ["PFE",  "ABBV", "MRK",  "LLY"],
    "PFE":   ["JNJ",  "ABBV", "MRK"],
    "ABBV":  ["JNJ",  "PFE",  "MRK",  "LLY"],
    "MRK":   ["JNJ",  "PFE",  "ABBV", "LLY"],
    "LLY":   ["ABBV", "MRK",  "JNJ"],
    "UNH":   ["CVS",  "CI",   "HUM"],
    # 미국 에너지
    "XOM":   ["CVX",  "COP",  "OXY",  "SLB"],
    "CVX":   ["XOM",  "COP",  "OXY"],
    "COP":   ["XOM",  "CVX",  "OXY"],
    "OXY":   ["XOM",  "CVX",  "COP"],
    # 미국 소비재/유통
    "WMT":   ["COST", "TGT",  "AMZN"],
    "COST":  ["WMT",  "TGT"],
    "TGT":   ["WMT",  "COST"],
    # 미국 미디어/스트리밍
    "NFLX":  ["DIS",  "SPOT", "PARA", "WBD"],
    "DIS":   ["NFLX", "PARA", "CMCSA"],
    "SPOT":  ["NFLX", "AAPL", "AMZN"],
    # 미국 결제
    "V":     ["MA",   "PYPL", "AXP"],
    "MA":    ["V",    "PYPL", "AXP"],
    "PYPL":  ["V",    "MA",   "SQ"],
    # 미국 클라우드/SaaS
    "CRM":   ["MSFT", "NOW",  "SNOW", "ORCL"],
    "SNOW":  ["CRM",  "MSFT", "DDOG"],
    "ORCL":  ["CRM",  "MSFT"],
    "NOW":   ["CRM",  "MSFT", "SNOW"],
    # 미국 통신
    "VZ":    ["T",    "TMUS"],
    "T":     ["VZ",   "TMUS"],
    "TMUS":  ["T",    "VZ"],
}


def get_related_sector_performance(ticker: str) -> dict:
    """
    동일 섹터 연관 종목들의 당일 평균 등락률 계산.
    ETF 티커: ETF 구성종목이 섹터 피어.
    일반 주식: 직접 섹터맵 → ETF 포트폴리오 역조회 순으로 피어 탐색.
    반환:
      avg_chg   float    연관 종목 평균 등락률(%)
      tickers   list     [{"name": str, "ticker": str, "chg": float}, ...]
      has_data  bool
      sector    str      섹터 레이블
    """
    code = ticker.replace(".KS", "").replace(".KQ", "")
    sector_label = _TICKER_SECTOR.get(ticker, "")

    # ETF인 경우 → 포트폴리오 구성종목을 피어로 사용
    if code in _ETF_PORTFOLIO_MAP:
        portfolio = _ETF_PORTFOLIO_MAP[code]
        related = portfolio.get("holdings", [])
        sector_label = portfolio.get("sector", sector_label)
    else:
        # 일반 주식: 직접 섹터맵 조회
        related = _SECTOR_MAP.get(ticker, [])
        # 없으면 ETF 포트폴리오 역조회 (해당 종목을 담고 있는 ETF의 다른 구성종목)
        if not related:
            for etf_code, etf_info in _ETF_PORTFOLIO_MAP.items():
                if ticker in etf_info.get("holdings", []):
                    related = [t for t in etf_info["holdings"] if t != ticker]
                    if not sector_label:
                        sector_label = etf_info.get("sector", "")
                    break

    if not related:
        return {"avg_chg": 0.0, "tickers": [], "has_data": False, "sector": sector_label}

    def _fetch_peer(sym: str) -> dict | None:
        try:
            d = yf.download(sym, period="2d", auto_adjust=True, progress=False)
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.droplevel(1)
            if len(d) < 2:
                return None
            chg = float((d["Close"].iloc[-1] - d["Close"].iloc[-2]) / d["Close"].iloc[-2] * 100)
            return {"name": sym, "ticker": sym, "chg": round(chg, 2)}
        except Exception:
            return None

    # 섹터 피어 yfinance 호출을 병렬 실행 (순차 → 동시)
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=min(len(related), 6)) as pool:
        for result in pool.map(_fetch_peer, related):
            if result is not None:
                rows.append(result)

    if not rows:
        return {"avg_chg": 0.0, "tickers": [], "has_data": False, "sector": sector_label}

    avg = round(sum(r["chg"] for r in rows) / len(rows), 2)
    return {"avg_chg": avg, "tickers": rows, "has_data": True, "sector": sector_label}


# ─── ETF 펀더멘털 분석 ────────────────────────────────────────────────────────

def get_etf_fundamental_data(ticker: str) -> dict:
    """
    ETF 핵심 지표 수집 — KRX 공공 데이터 API 비동기 기반.
    (NAV, 괴리율, 추적오차, AUM, 구성종목, 운용보수)
    """
    from src.etf_async import get_etf_fundamental_data as _get_etf
    code           = ticker.replace(".KS", "").replace(".KQ", "").strip().zfill(6)
    portfolio_info = _ETF_PORTFOLIO_MAP.get(code, {})
    return _get_etf(ticker, portfolio_info)


def calculate_etf_score(etf_data: dict) -> dict:
    """
    ETF 투자 점수 산출 — 그레이엄 수식 대체.

    평가 항목:
      괴리율  (NAV Premium/Discount): 할인 → 매수 기회, 프리미엄 → 고평가
      운용보수 (Expense Ratio)       : 낮을수록 장기 수익률 유리
      추적오차 (Tracking Error)      : 낮을수록 지수 추종 정밀
      배당수익률                     : 보너스 가점

    반환: {
      etf_score     float,   # ±7 범위
      etf_label     str,     # 강력매수/긍정적/중립/주의/회피
      etf_reasons   list,    # 근거 목록
      score_breakdown dict,  # 항목별 세부 점수
    }
    """
    score    = 0.0
    reasons: list[str] = []
    breakdown: dict[str, float] = {}

    # ── 괴리율 점수 (±3) ─────────────────────────────────────────────────────
    premium = etf_data.get("nav_premium")
    if premium is not None:
        if premium < -2.0:
            ps = 3.0;  reasons.append(f"NAV 대비 {premium:.2f}% 할인 — 저평가 매수 기회 (할인 ETF)")
        elif premium < -0.5:
            ps = 2.0;  reasons.append(f"NAV 대비 {premium:.2f}% 소폭 할인 — 양호한 진입 시점")
        elif premium <= 0.5:
            ps = 1.0;  reasons.append(f"NAV 괴리율 {premium:.2f}% — 공정 가치 수준")
        elif premium <= 2.0:
            ps = -0.5; reasons.append(f"NAV 대비 {premium:.2f}% 프리미엄 — 다소 고평가")
        else:
            ps = -2.0; reasons.append(f"NAV 대비 {premium:.2f}% 고프리미엄 — 시장가 과열 주의")
        breakdown["괴리율"] = ps
        score += ps
    else:
        breakdown["괴리율"] = 0.0
        reasons.append("NAV 괴리율 데이터 없음 (KRX API 일시 장애 또는 장외 시간)")

    # ── 운용보수 점수 (±2) ────────────────────────────────────────────────────
    er = etf_data.get("expense_ratio")
    if er is not None:
        if er < 0.1:
            es = 2.0;  reasons.append(f"운용보수 {er:.3f}% — 업계 최저 수준, 장기 보유 유리")
        elif er < 0.3:
            es = 1.0;  reasons.append(f"운용보수 {er:.3f}% — 합리적 수준")
        elif er < 0.5:
            es = 0.0;  reasons.append(f"운용보수 {er:.3f}% — 업계 평균")
        elif er < 0.8:
            es = -1.0; reasons.append(f"운용보수 {er:.3f}% — 다소 높음, 장기 수익률 저하 요인")
        else:
            es = -2.0; reasons.append(f"운용보수 {er:.3f}% — 높은 수준, 복리 수익률 잠식 위험")
        breakdown["운용보수"] = es
        score += es
    else:
        breakdown["운용보수"] = 0.0
        reasons.append("운용보수 데이터 없음 (미등록 ETF)")

    # ── 추적오차 점수 (±1.5) ─────────────────────────────────────────────────
    te = etf_data.get("tracking_error")
    if te is not None:
        te_abs = abs(te)
        if te_abs < 0.3:
            ts = 1.5;  reasons.append(f"추적오차 {te_abs:.2f}% — 기초지수 정밀 추종")
        elif te_abs < 1.0:
            ts = 0.5;  reasons.append(f"추적오차 {te_abs:.2f}% — 양호한 추종 품질")
        elif te_abs < 2.0:
            ts = -0.5; reasons.append(f"추적오차 {te_abs:.2f}% — 추종 오차 다소 높음")
        else:
            ts = -1.5; reasons.append(f"추적오차 {te_abs:.2f}% — 지수 추종 품질 불량")
        breakdown["추적오차"] = ts
        score += ts
    else:
        breakdown["추적오차"] = 0.0

    # ── 배당수익률 가산점 (±0.5) ─────────────────────────────────────────────
    div = etf_data.get("dividend_yield")
    if div is not None:
        if div >= 3.0:
            ds = 0.5;  reasons.append(f"배당수익률 {div:.1f}% — 배당형 ETF, 현금흐름 수익 기대")
        elif div >= 1.0:
            ds = 0.0
        else:
            ds = -0.3; reasons.append(f"배당수익률 {div:.1f}% — 저배당 (성장형 ETF)")
        breakdown["배당수익률"] = ds
        score += ds

    # ── 판정 레이블 ──────────────────────────────────────────────────────────
    score = round(score, 2)
    if score >= 3.5:
        label = "강력 매수"
    elif score >= 1.5:
        label = "긍정적"
    elif score >= -0.5:
        label = "중립"
    elif score >= -2.0:
        label = "주의"
    else:
        label = "회피"

    return {
        "etf_score":      score,
        "etf_label":      label,
        "etf_reasons":    reasons,
        "score_breakdown": breakdown,
    }


def get_etf_news_with_holdings(ticker: str, etf_data: dict, max_items: int = 15) -> list[dict]:
    """
    ETF 뉴스 수집 (섹터 뉴스 + 상위 구성종목 뉴스 통합).
    - ETF 자체 뉴스: 종목명 필터 느슨하게 적용
    - 상위 구성종목 최대 3개의 뉴스도 포함해 정확도 향상
    - ETF + 구성종목 뉴스를 ThreadPoolExecutor로 동시 수집
    """
    holdings = etf_data.get("top_holdings", [])
    valid_holdings = [h for h in holdings[:3] if h.get("ticker")]

    # (ticker, source_type, holding_name, max_items) 수집 작업 목록
    tasks: list[tuple[str, str, str, int]] = [
        (ticker, "etf_direct", "", 10),
    ] + [
        (h["ticker"], "holding", h.get("name", h["ticker"]), 6)
        for h in valid_holdings
    ]

    def _fetch(task: tuple[str, str, str, int]) -> list[dict]:
        t, source_type, holding_name, n = task
        try:
            items = get_naver_news(t, max_items=n)
            for item in items:
                item["_source_type"]  = source_type
                item["_holding_name"] = holding_name
            return items
        except Exception:
            return []

    # 동시 수집 (ETF 1 + 구성종목 최대 3 → 최대 4개 HTTP 요청 병렬화)
    all_news:   list[dict] = []
    seen_titles: set[str]  = set()

    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        for batch in pool.map(_fetch, tasks):
            limit = 4 if batch and batch[0].get("_source_type") == "holding" else len(batch)
            for item in batch[:limit]:
                t = item.get("title", "").strip()
                if t and t not in seen_titles:
                    all_news.append(item)
                    seen_titles.add(t)

    return all_news[:max_items]


def analyze_etf_news_sentiment(ticker: str, etf_data: dict, news_items: list[dict]) -> dict:
    """
    ETF 뉴스 감성 분석.
    - 섹터 키워드를 중심으로 스코어링
    - 종목명 일치 필터를 느슨하게 적용 (ETF는 섹터 뉴스가 핵심)
    """
    sector = etf_data.get("sector", "")
    etf_name = etf_data.get("etf_name", "")

    # 섹터 키워드 가져오기
    sector_kws = _SECTOR_KEYWORDS.get(sector, {})
    pos_kws = sector_kws.get("pos", [])
    neg_kws = sector_kws.get("neg", [])

    total_score  = 0.0
    scored_count = 0
    detail: list[dict] = []

    for item in news_items:
        title    = item.get("title", "")
        pub_date = item.get("pub_date", "")
        decay    = _news_time_decay(pub_date)
        text     = title

        # 섹터 키워드 스코어
        pos_hits = sum(1 for kw in pos_kws if kw in text)
        neg_hits = sum(1 for kw in neg_kws if kw in text)
        kw_score = (pos_hits - neg_hits) * 0.6

        # ETF명 / 섹터 직접 언급 보너스
        if etf_name and etf_name[:4] in text:
            kw_score += 0.3
        if sector and sector in text:
            kw_score += 0.2

        art_score = max(-2.0, min(2.0, kw_score)) * decay
        total_score += art_score
        scored_count += 1
        detail.append({
            "title":     title,
            "link":      item.get("link", "#"),
            "publisher": item.get("publisher", ""),
            "pub_date":  pub_date,
            "score":     round(art_score, 2),
            "reason":    f"섹터({sector}) 키워드: +{pos_hits}/-{neg_hits}",
            "tier":      "B",
            "decay":     decay,
            "skipped":   False,
            "source_type": item.get("_source_type", ""),
            "holding_name": item.get("_holding_name", ""),
        })

    if scored_count == 0:
        final_score = 0.0
    else:
        avg   = total_score / scored_count
        final_score = round(max(-5.0, min(5.0, avg * 2.5)), 2)

    if final_score >= 3:   label = "매우 긍정"
    elif final_score >= 1: label = "긍정"
    elif final_score <= -3: label = "매우 부정"
    elif final_score <= -1: label = "부정"
    else:                  label = "중립"

    top3 = [d["title"][:20] for d in sorted(detail, key=lambda x: abs(x["score"]), reverse=True)[:3]]
    summary = f"섹터({sector}) 뉴스 중심 분석: {', '.join(top3[:2])}" if top3 else f"{sector} 섹터 뉴스 분석 중"

    return {
        "score":         final_score,
        "label":         label,
        "summary":       summary,
        "individual_score": final_score,
        "sector_score":  0.0,
        "detail":        detail,
        "event_flags":   [],
        "is_etf":        True,
    }
