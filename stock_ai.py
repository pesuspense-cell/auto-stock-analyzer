"""
stock_ai.py - 주식 분석 핵심 알고리즘 엔진
매수/매도 신호, 추천 종목, 예상 수익률 분석
"""
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
import ta as ta_lib
import warnings
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
    주식 데이터 로드 및 기술적 지표 계산.
    지표 목록:
      이동평균 : SMA 5/20/60, EMA 20/50/200
      모멘텀   : RSI(14), 스토캐스틱(14,3), CCI(20), Williams %R(14), ROC(12)
      추세     : MACD(12/26/9), ADX±DI(14)
      변동성   : 볼린저밴드(20,2), ATR(14)
      거래량   : Volume MA20, OBV, OBV MA20, MFI(14)
    """
    data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if data.empty or len(data) < 20:
        return data

    # 멀티레벨 컬럼 처리 (yfinance 0.2+)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)

    close  = data["Close"]
    high   = data["High"]
    low    = data["Low"]
    volume = data["Volume"]

    # ── 이동평균선 (SMA / EMA) ────────────────────────────────────────────────
    data["SMA_5"]   = ta_lib.trend.SMAIndicator(close, window=5).sma_indicator()
    data["SMA_20"]  = ta_lib.trend.SMAIndicator(close, window=20).sma_indicator()
    data["SMA_60"]  = ta_lib.trend.SMAIndicator(close, window=60).sma_indicator()
    data["EMA_20"]  = ta_lib.trend.EMAIndicator(close, window=20).ema_indicator()
    data["EMA_50"]  = ta_lib.trend.EMAIndicator(close, window=50).ema_indicator()
    # EMA 200은 데이터가 충분할 때만 (1y/2y 조회 시 활성화)
    if len(data) >= 200:
        data["EMA_200"] = ta_lib.trend.EMAIndicator(close, window=200).ema_indicator()

    # ── RSI (14) ──────────────────────────────────────────────────────────────
    data["RSI"] = ta_lib.momentum.RSIIndicator(close, window=14).rsi()

    # ── MACD (12/26/9) ────────────────────────────────────────────────────────
    try:
        macd_ind = ta_lib.trend.MACD(close, window_fast=12, window_slow=26, window_sign=9)
        data["MACD"]        = macd_ind.macd()
        data["MACD_Signal"] = macd_ind.macd_signal()
        data["MACD_Hist"]   = macd_ind.macd_diff()
    except Exception:
        pass

    # ── 볼린저밴드 (20, 2σ) ───────────────────────────────────────────────────
    try:
        bb_ind = ta_lib.volatility.BollingerBands(close, window=20, window_dev=2)
        data["BB_Upper"]  = bb_ind.bollinger_hband()
        data["BB_Middle"] = bb_ind.bollinger_mavg()
        data["BB_Lower"]  = bb_ind.bollinger_lband()
        data["BB_PCT"]    = bb_ind.bollinger_pband()   # %B: 0~1 (밴드 내 위치)
        data["BB_Width"]  = bb_ind.bollinger_wband()   # 밴드 폭 (변동성 squeeze 감지)
    except Exception:
        pass

    # ── 스토캐스틱 (14, 3) ────────────────────────────────────────────────────
    try:
        stoch_ind = ta_lib.momentum.StochasticOscillator(
            high, low, close, window=14, smooth_window=3
        )
        data["STOCH_K"] = stoch_ind.stoch()
        data["STOCH_D"] = stoch_ind.stoch_signal()
    except Exception:
        pass

    # ── ADX + 방향성 지수 (14) ────────────────────────────────────────────────
    try:
        adx_ind = ta_lib.trend.ADXIndicator(high, low, close, window=14)
        data["ADX"]     = adx_ind.adx()
        data["ADX_POS"] = adx_ind.adx_pos()   # +DI (상승 추세력)
        data["ADX_NEG"] = adx_ind.adx_neg()   # -DI (하락 추세력)
    except Exception:
        pass

    # ── CCI (Commodity Channel Index, 20) ────────────────────────────────────
    try:
        data["CCI"] = ta_lib.trend.CCIIndicator(high, low, close, window=20).cci()
    except Exception:
        pass

    # ── Williams %R (14) ──────────────────────────────────────────────────────
    try:
        data["WILLIAMS_R"] = ta_lib.momentum.WilliamsRIndicator(
            high, low, close, lbp=14
        ).williams_r()
    except Exception:
        pass

    # ── ATR (Average True Range, 14) ──────────────────────────────────────────
    try:
        data["ATR"] = ta_lib.volatility.AverageTrueRange(
            high, low, close, window=14
        ).average_true_range()
    except Exception:
        pass

    # ── ROC (Rate of Change, 12) ──────────────────────────────────────────────
    try:
        data["ROC"] = ta_lib.momentum.ROCIndicator(close, window=12).roc()
    except Exception:
        pass

    # ── OBV (On-Balance Volume) + 추세선 ─────────────────────────────────────
    try:
        data["OBV"]      = ta_lib.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
        data["OBV_MA20"] = data["OBV"].rolling(window=20).mean()
    except Exception:
        pass

    # ── MFI (Money Flow Index, 14) ────────────────────────────────────────────
    try:
        data["MFI"] = ta_lib.volume.MFIIndicator(
            high, low, close, volume, window=14
        ).money_flow_index()
    except Exception:
        pass

    # ── 거래량 이평 ───────────────────────────────────────────────────────────
    data["Volume_MA20"] = volume.rolling(window=20).mean()

    return data


# ─── 매매 신호 알고리즘 ───────────────────────────────────────────────────────

def generate_signals(data: pd.DataFrame) -> dict:
    """
    8개 모듈 복합 채점으로 매매 신호 생성.
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
    이론적 최대: ±13 → cap ±10
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


# ─── 예상 수익률 & 리스크 분석 ────────────────────────────────────────────────

def calculate_expected_return(data: pd.DataFrame, signals: dict, horizon_days: int = 20) -> dict:
    """
    기술 신호 + 역사적 변동성 + 모멘텀 기반 예상 수익률 추정
    """
    if data.empty or not signals:
        return {}

    close   = data["Close"]
    returns = close.pct_change().dropna()

    hist_vol_annual = float(returns.std() * np.sqrt(252) * 100)       # 연간 변동성(%)
    daily_drift     = float(returns.mean() * 100)                      # 평균 일간 수익률(%)
    momentum_20d    = float((close.iloc[-1] / close.iloc[-21] - 1) * 100) if len(close) >= 21 else 0.0

    score = signals.get("score", 0)

    # 신호 기반 기대 수익률 = (신호 강도 × 일평균변동폭) + 모멘텀 조정
    signal_contribution = score * (hist_vol_annual / 252) * horizon_days * 0.35
    momentum_adj        = momentum_20d * 0.15
    expected_return     = signal_contribution + momentum_adj + daily_drift * horizon_days

    # 최대낙폭
    max_drawdown = float(((close / close.cummax()) - 1).min() * 100)

    # 샤프 비율 근사 (무위험 수익률 3.5% 가정)
    risk_free = 3.5 / 252 * 100
    sharpe_approx = ((daily_drift - risk_free) / (returns.std() * 100) * np.sqrt(252)) if returns.std() > 0 else 0.0

    return {
        "expected_return_pct": round(expected_return, 2),
        "hist_volatility":     round(hist_vol_annual, 2),
        "momentum_20d":        round(momentum_20d, 2),
        "max_drawdown":        round(max_drawdown, 2),
        "sharpe":              round(sharpe_approx, 2),
        "daily_drift":         round(daily_drift, 3),
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
    종합점수 = 기술점수(50%) + 수익률점수(30%) + 샤프점수(20%)
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

            comp       = _composite_score(tech, ret_pct, sharpe)
            comp_label, comp_badge = _composite_label(comp)

            rows.append({
                "종목명":         name,
                "티커":           ticker,
                "현재가":         price,
                "등락률(1일)%":   round(chg_1d, 2),
                "종합추천":       f"{comp_badge} {comp_label}",
                "종합점수":       comp,
                "기술점수":       tech,
                "예상수익률(%)":  ret_pct,
                "변동성(%)":      round(float(exp.get("hist_volatility", 0)), 1),
                "모멘텀(20일)%":  round(float(exp.get("momentum_20d", 0)), 2),
                "샤프지수":       round(sharpe, 2),
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
                df = fdr.StockListing(listing_key)
                df = df.dropna(subset=["Name", "Symbol"])
                for _, row in df.iterrows():
                    sym  = str(row["Symbol"]).strip()
                    name = str(row["Name"]).strip()
                    if sym and name and sym.replace(".", "").isalpha():
                        display = f"{name} ({sym}) [{market_tag}]"
                        combined[display] = sym
            except Exception:
                continue

        return combined if combined else {
            f"{n} ({s}) [기본]": s for n, s in US_STOCKS.items()
        }
    except Exception:
        return {f"{n} ({s}) [기본]": s for n, s in US_STOCKS.items()}


# ─── KRX 전체 종목 리스트 (이름 검색용) ──────────────────────────────────────

def get_krx_stock_list() -> dict:
    """
    KOSPI + KOSDAQ 전체 종목 이름 → 티커 매핑 반환
    반환 형태: {"삼성전자 (005930)": "005930.KS", ...}
    데이터 소스: FinanceDataReader (KRX 실시간 종목 목록)
    """
    try:
        import FinanceDataReader as fdr
        result = {}
        for market, suffix in [("KOSPI", "KS"), ("KOSDAQ", "KQ")]:
            df = fdr.StockListing(market)
            if df.empty:
                continue
            for _, row in df.iterrows():
                code = str(row.get("Code", "")).strip()
                name = str(row.get("Name", "")).strip()
                if name and code and len(code) == 6 and code.isdigit():
                    display = f"{name} ({code})"
                    result[display] = f"{code}.{suffix}"
        return result
    except Exception:
        return {}


# ─── 펀더멘털 데이터 ──────────────────────────────────────────────────────────

def _is_krx_ticker(ticker: str) -> bool:
    return ticker.endswith(".KS") or ticker.endswith(".KQ")


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
    KRX 종목(.KS/.KQ): SQLite 캐시 우선 조회 → 없으면 pykrx로 단일 fetch 후 저장
    해외 종목: yfinance 직접 조회
    """
    if _is_krx_ticker(ticker):
        try:
            from fundamental_db import get_ticker_fundamental, fetch_and_cache_single
            row = get_ticker_fundamental(ticker)
            if row is None:
                row = fetch_and_cache_single(ticker)
            if row:
                return _krx_fundamental_to_dict(row, ticker)
        except Exception:
            pass

    try:
        t         = yf.Ticker(ticker)
        info      = t.info
        price     = info.get("currentPrice") or info.get("regularMarketPrice")
        eps_ttm   = info.get("trailingEps")
        book_val  = info.get("bookValue")
        market_cap= info.get("marketCap")
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

        # 영업이익·순이익: financials DataFrame에서 추출
        operating_income = None
        net_income       = None
        try:
            fin = t.financials
            if fin is not None and not fin.empty:
                for label in ["Operating Income", "EBIT"]:
                    if label in fin.index:
                        operating_income = float(fin.loc[label].iloc[0])
                        break
                for label in ["Net Income", "Net Income Common Stockholders"]:
                    if label in fin.index:
                        net_income = float(fin.loc[label].iloc[0])
                        break
        except Exception:
            pass

        return {
            "per":              per,
            "pbr":              pbr,
            "psr":              psr,
            "roe":              info.get("returnOnEquity"),
            "debt_equity":      info.get("debtToEquity"),
            "revenue_growth":   info.get("revenueGrowth"),
            "earnings_growth":  info.get("earningsGrowth"),
            "operating_margins":info.get("operatingMargins"),
            "w52_high":         info.get("fiftyTwoWeekHigh"),
            "w52_low":          info.get("fiftyTwoWeekLow"),
            "market_cap":       market_cap,
            "total_revenue":    total_rev,
            "operating_income": operating_income,
            "net_income":       net_income,
            "free_cashflow":    info.get("freeCashflow"),
            "eps_ttm":          eps_ttm,
            "forward_pe":       info.get("forwardPE"),
            "sector":           info.get("sector", "N/A"),
            "industry":         info.get("industry", "N/A"),
            "short_name":       info.get("shortName", ticker),
            "source":           "yfinance",
        }
    except Exception:
        return {}


def calculate_fundamental_score(info: dict, close_price: float = None) -> dict:
    """
    버핏·그레이엄·린치·오닐 투자 법칙 기반 펀더멘털 점수
    참고 문헌: 《현명한 투자자》《전설로 떠나는 월가의 영웅》《최고의 주식 최적의 타이밍》
    점수 범위: -8 ~ +8
    """
    score = 0.0
    reasons = []

    per             = info.get("per")
    pbr             = info.get("pbr")
    roe             = info.get("roe")
    debt_equity     = info.get("debt_equity")
    revenue_growth  = info.get("revenue_growth")
    earnings_growth = info.get("earnings_growth")
    w52_high        = info.get("w52_high")
    w52_low         = info.get("w52_low")
    fcf             = info.get("free_cashflow")
    market_cap      = info.get("market_cap")

    # ── 그레이엄 공식: PBR × PER < 22.5 ────────────────────────────────────
    if per and pbr and per > 0 and pbr > 0:
        gnum = per * pbr
        if gnum < 15:
            score += 2.0; reasons.append(f"[그레이엄] PBR×PER={gnum:.1f} < 15 → 강한 저평가")
        elif gnum < 22.5:
            score += 1.0; reasons.append(f"[그레이엄] PBR×PER={gnum:.1f} < 22.5 → 적정 평가")
        elif gnum > 45:
            score -= 2.0; reasons.append(f"[그레이엄] PBR×PER={gnum:.1f} > 45 → 고평가 경고")
        elif gnum > 30:
            score -= 1.0; reasons.append(f"[그레이엄] PBR×PER={gnum:.1f} > 30 → 다소 고평가")

    # ── 버핏: ROE 15% 이상 ──────────────────────────────────────────────────
    if roe is not None:
        roe_pct = roe * 100
        if roe_pct >= 20:
            score += 2.0; reasons.append(f"[버핏] ROE {roe_pct:.1f}% ≥ 20% → 우량 기업")
        elif roe_pct >= 15:
            score += 1.0; reasons.append(f"[버핏] ROE {roe_pct:.1f}% ≥ 15% → 버핏 기준 충족")
        elif roe_pct < 5:
            score -= 1.0; reasons.append(f"[버핏] ROE {roe_pct:.1f}% < 5% → 수익성 부진")

    # ── 버핏: 부채비율 50% 이하 ─────────────────────────────────────────────
    if debt_equity is not None and debt_equity >= 0:
        if debt_equity < 50:
            score += 1.0; reasons.append(f"[버핏] 부채비율 {debt_equity:.0f}% < 50% → 재무 우량")
        elif debt_equity > 200:
            score -= 1.5; reasons.append(f"[버핏] 부채비율 {debt_equity:.0f}% > 200% → 재무 위험")
        elif debt_equity > 100:
            score -= 0.5; reasons.append(f"[버핏] 부채비율 {debt_equity:.0f}% > 100% → 주의 필요")

    # ── 버핏: FCF Yield > 5% ────────────────────────────────────────────────
    if fcf and market_cap and market_cap > 0:
        fcf_yield = fcf / market_cap * 100
        if fcf_yield > 8:
            score += 1.5; reasons.append(f"[버핏] FCF Yield {fcf_yield:.1f}% > 8% → 현금창출 탁월")
        elif fcf_yield > 5:
            score += 0.5; reasons.append(f"[버핏] FCF Yield {fcf_yield:.1f}% > 5% → 양호")
        elif fcf_yield < 0:
            score -= 1.0; reasons.append(f"[버핏] FCF Yield {fcf_yield:.1f}% < 0 → 현금소진 경고")

    # ── 피터 린치: PEG < 1.0 매수, > 2.0 매도 ──────────────────────────────
    if per and earnings_growth and per > 0 and earnings_growth > 0:
        peg = per / (earnings_growth * 100)
        if peg < 0.5:
            score += 2.0; reasons.append(f"[린치] PEG={peg:.2f} < 0.5 → 강력 성장 저평가")
        elif peg < 1.0:
            score += 1.0; reasons.append(f"[린치] PEG={peg:.2f} < 1.0 → 성장 대비 저평가")
        elif peg > 2.0:
            score -= 1.0; reasons.append(f"[린치] PEG={peg:.2f} > 2.0 → 성장 대비 고평가")

    # ── 피터 린치: 매출 성장 20% 이상 (텐배거) ──────────────────────────────
    if revenue_growth is not None:
        rev_pct = revenue_growth * 100
        if rev_pct >= 25:
            score += 1.5; reasons.append(f"[린치] 매출 {rev_pct:.1f}% 급성장 → 텐배거 후보")
        elif rev_pct >= 10:
            score += 0.5; reasons.append(f"[린치] 매출 {rev_pct:.1f}% 성장 → 양호")
        elif rev_pct < -10:
            score -= 1.0; reasons.append(f"[린치] 매출 {rev_pct:.1f}% 감소 → 펀더멘털 악화")

    # ── 오닐 CANSLIM: 52주 고가 근접 (신고가 모멘텀) ────────────────────────
    if close_price and w52_high and w52_low and w52_high > w52_low:
        ratio = close_price / w52_high
        pos   = (close_price - w52_low) / (w52_high - w52_low)
        if ratio >= 0.95:
            score += 1.5; reasons.append(f"[오닐] 52주 고가 {ratio*100:.1f}% 근접 → 신고가 돌파 모멘텀")
        elif pos <= 0.15:
            score += 0.5; reasons.append(f"[오닐] 52주 저가 근접 ({pos*100:.0f}%) → 바닥 반등 기대")

    label = "펀더멘털 강함" if score >= 3 else ("펀더멘털 약함" if score <= -2 else "펀더멘털 보통")

    return {
        "fund_score":   round(score, 1),
        "fund_label":   label,
        "fund_reasons": reasons,
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

# 긍정/부정 키워드 사전
_POS_KEYWORDS = [
    "수주", "돌파", "상승", "이익", "증가", "추천", "급등", "호재", "성장", "흑자",
    "신고가", "매수", "목표가 상향", "어닝서프라이즈", "배당", "실적 개선", "확대",
    "강세", "반등", "회복", "신규", "수혜", "기대", "긍정", "상향", "계약", "수출",
]
_NEG_KEYWORDS = [
    "하락", "감소", "적자", "조사", "우려", "매도", "급락", "악재", "손실", "적자전환",
    "신저가", "목표가 하향", "어닝쇼크", "부진", "위기", "하향", "규제", "소송", "제재",
    "약세", "불안", "취소", "철수", "파산", "실망", "하락세", "폭락",
]


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


def analyze_news_sentiment_keywords(news_items: list[dict]) -> dict:
    """
    키워드 매칭 기반 뉴스 감성 분석 (API 없이 동작하는 폴백)
    반환: {"score": float(-5~+5), "label": str, "detail": list[dict]}
    """
    if not news_items:
        return {"score": 0.0, "label": "중립", "detail": []}

    detail = []
    total  = 0.0

    for item in news_items:
        text = (item.get("title", "") + " " + item.get("summary", "")).lower()
        pos  = sum(1 for kw in _POS_KEYWORDS if kw in text)
        neg  = sum(1 for kw in _NEG_KEYWORDS if kw in text)
        raw  = pos - neg
        # 기사 1건당 점수를 -1~+1로 클리핑
        item_score = max(-1.0, min(1.0, raw * 0.5))
        total += item_score
        detail.append({
            "title":  item.get("title", ""),
            "link":   item.get("link", "#"),
            "publisher": item.get("publisher", ""),
            "pub_date":  item.get("pub_date", ""),
            "score":  round(item_score, 2),
            "pos_kw": pos,
            "neg_kw": neg,
        })

    # 전체 점수: 평균 × 5 (스케일 -5~+5)
    avg   = total / len(news_items)
    score = round(avg * 5, 2)
    score = max(-5.0, min(5.0, score))

    if   score >= 3:  label = "매우 긍정"
    elif score >= 1:  label = "긍정"
    elif score >= -1: label = "중립"
    elif score >= -3: label = "부정"
    else:             label = "매우 부정"

    return {"score": score, "label": label, "detail": detail}


def analyze_news_sentiment_llm(
    news_items: list[dict],
    ticker: str,
    api_key: str,
) -> dict:
    """
    LangChain + Gemini API를 이용한 뉴스 감성 분석
    반환: {"score": float(-5~+5), "label": str, "detail": list[dict], "summary": str}
    API 키가 없거나 오류 시 키워드 폴백
    """
    if not api_key or not news_items:
        return analyze_news_sentiment_keywords(news_items)

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=api_key,
            temperature=0.1,
        )

        titles = "\n".join(
            f"{i+1}. {it.get('title','')}" for i, it in enumerate(news_items[:10])
        )
        prompt = f"""다음은 주식 종목 '{ticker}'에 관한 최신 뉴스 헤드라인입니다.

{titles}

각 헤드라인을 분석하여 아래 형식으로 JSON을 출력하세요.
- 전체 감성 점수: -5(매우 부정) ~ +5(매우 긍정) 사이의 소수
- 각 뉴스 별 점수: -1 ~ +1

출력 형식 (JSON만, 설명 없이):
{{
  "overall_score": <float>,
  "overall_summary": "<한국어 2~3문장 요약>",
  "items": [
    {{"index": 1, "score": <float>, "reason": "<한국어 한 줄>"}},
    ...
  ]
}}"""

        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()

        # JSON 파싱
        import json, re
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if not json_match:
            raise ValueError("JSON not found in response")
        parsed = json.loads(json_match.group())

        overall_score = float(parsed.get("overall_score", 0))
        overall_score = max(-5.0, min(5.0, overall_score))
        summary       = parsed.get("overall_summary", "")

        # 키워드 분석 결과와 병합 (링크, 출판사 등 메타 보존)
        kw_result = analyze_news_sentiment_keywords(news_items)
        detail    = kw_result["detail"]
        for item_res in parsed.get("items", []):
            idx = item_res.get("index", 0) - 1
            if 0 <= idx < len(detail):
                detail[idx]["score"]  = round(float(item_res.get("score", 0)), 2)
                detail[idx]["reason"] = item_res.get("reason", "")

        if   overall_score >= 3:  label = "매우 긍정"
        elif overall_score >= 1:  label = "긍정"
        elif overall_score >= -1: label = "중립"
        elif overall_score >= -3: label = "부정"
        else:                     label = "매우 부정"

        return {
            "score":   round(overall_score, 2),
            "label":   label,
            "detail":  detail,
            "summary": summary,
        }

    except Exception:
        # API 오류 시 키워드 폴백
        result = analyze_news_sentiment_keywords(news_items)
        result["summary"] = "(AI 분석 실패 — 키워드 분석으로 대체)"
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
        parsed = _json.loads(match.group())

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
    "035720.KS": ["035420.KS", "293490.KS"],
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
}


def get_related_sector_performance(ticker: str) -> dict:
    """
    동일 섹터 연관 종목들의 당일 평균 등락률 계산.
    반환:
      avg_chg   float    연관 종목 평균 등락률(%)
      tickers   list     [{"name": str, "ticker": str, "chg": float}, ...]
      has_data  bool
    """
    related = _SECTOR_MAP.get(ticker, [])
    if not related:
        return {"avg_chg": 0.0, "tickers": [], "has_data": False}

    rows: list[dict] = []
    for sym in related:
        try:
            d = yf.download(sym, period="2d", auto_adjust=True, progress=False)
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.droplevel(1)
            if len(d) < 2:
                continue
            chg = float((d["Close"].iloc[-1] - d["Close"].iloc[-2]) / d["Close"].iloc[-2] * 100)
            # 표시 이름: 섹터맵 역조회하거나 그냥 티커 사용
            rows.append({"name": sym, "ticker": sym, "chg": round(chg, 2)})
        except Exception:
            continue

    if not rows:
        return {"avg_chg": 0.0, "tickers": [], "has_data": False}

    avg = round(sum(r["chg"] for r in rows) / len(rows), 2)
    return {"avg_chg": avg, "tickers": rows, "has_data": True}
