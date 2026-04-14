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
import pandas_ta as ta
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
    """주식 데이터 로드 및 기술적 지표 계산"""
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

    # 이동평균선
    data["SMA_5"]  = ta.sma(close, length=5)
    data["SMA_20"] = ta.sma(close, length=20)
    data["SMA_60"] = ta.sma(close, length=60)

    # RSI
    data["RSI"] = ta.rsi(close, length=14)

    # MACD
    macd = ta.macd(close)
    if macd is not None and not macd.empty:
        data["MACD"]        = macd.iloc[:, 0]   # MACD_12_26_9
        data["MACD_Signal"] = macd.iloc[:, 2]   # MACDs_12_26_9
        data["MACD_Hist"]   = macd.iloc[:, 1]   # MACDh_12_26_9

    # 볼린저밴드
    bb = ta.bbands(close, length=20)
    if bb is not None and not bb.empty:
        data["BB_Upper"]  = bb.iloc[:, 2]  # BBU
        data["BB_Middle"] = bb.iloc[:, 1]  # BBM
        data["BB_Lower"]  = bb.iloc[:, 0]  # BBL

    # 스토캐스틱
    stoch = ta.stoch(high, low, close)
    if stoch is not None and not stoch.empty:
        data["STOCH_K"] = stoch.iloc[:, 0]
        data["STOCH_D"] = stoch.iloc[:, 1]

    # 거래량 이평
    data["Volume_MA20"] = ta.sma(volume, length=20)

    return data


# ─── 매매 신호 알고리즘 ───────────────────────────────────────────────────────

def generate_signals(data: pd.DataFrame) -> dict:
    """
    5가지 기술 지표를 복합 채점하여 매매 신호 생성
    점수 범위: -10 ~ +10  (양수=매수, 음수=매도)
    """
    if data.empty or len(data) < 21:
        return {}

    score   = 0.0
    reasons = []
    last    = data.iloc[-1]
    prev    = data.iloc[-2]
    close   = data["Close"]

    # ── 1. RSI ─────────────────────────────────────────────────────────────
    rsi = _f(last, "RSI")
    if rsi is not None:
        if rsi < 25:
            score += 2.5; reasons.append(f"RSI 극과매도 ({rsi:.1f}) → 강한 반등 기대")
        elif rsi < 35:
            score += 1.5; reasons.append(f"RSI 과매도 ({rsi:.1f}) → 매수 고려")
        elif rsi < 45:
            score += 0.5; reasons.append(f"RSI 매수권 진입 ({rsi:.1f})")
        elif rsi > 75:
            score -= 2.5; reasons.append(f"RSI 극과매수 ({rsi:.1f}) → 강한 매도 신호")
        elif rsi > 65:
            score -= 1.5; reasons.append(f"RSI 과매수 ({rsi:.1f}) → 매도 고려")
        elif rsi > 55:
            score -= 0.5; reasons.append(f"RSI 매도권 진입 ({rsi:.1f})")

    # ── 2. MACD ────────────────────────────────────────────────────────────
    macd, sig  = _f(last, "MACD"),        _f(last, "MACD_Signal")
    pmacd, psg = _f(prev, "MACD"),        _f(prev, "MACD_Signal")
    if all(v is not None for v in [macd, sig, pmacd, psg]):
        cross_up   = pmacd < psg and macd > sig
        cross_down = pmacd > psg and macd < sig
        if cross_up:
            score += 2.0; reasons.append("MACD 골든크로스 → 강한 매수 신호")
        elif cross_down:
            score -= 2.0; reasons.append("MACD 데드크로스 → 강한 매도 신호")
        elif macd > sig:
            score += 0.5; reasons.append("MACD 매수 우위 유지")
        else:
            score -= 0.5; reasons.append("MACD 매도 우위 유지")

    # ── 3. 볼린저밴드 ──────────────────────────────────────────────────────
    price  = float(close.iloc[-1])
    bb_u   = _f(last, "BB_Upper")
    bb_l   = _f(last, "BB_Lower")
    bb_m   = _f(last, "BB_Middle")
    if bb_u and bb_l and bb_m:
        band_width = (bb_u - bb_l) / bb_m
        if price < bb_l:
            score += 1.5; reasons.append(f"볼린저 하단 이탈 ({price:,.0f} < {bb_l:,.0f}) → 반등 기대")
        elif price > bb_u:
            score -= 1.5; reasons.append(f"볼린저 상단 돌파 ({price:,.0f} > {bb_u:,.0f}) → 과열 주의")
        elif price > bb_m:
            score += 0.3; reasons.append("볼린저 중간선 위 → 상승 추세")
        else:
            score -= 0.3

    # ── 4. 이동평균선 크로스 ───────────────────────────────────────────────
    sma5, sma20    = _f(last, "SMA_5"), _f(last, "SMA_20")
    psma5, psma20  = _f(prev, "SMA_5"), _f(prev, "SMA_20")
    if all(v is not None for v in [sma5, sma20, psma5, psma20]):
        if psma5 < psma20 and sma5 > sma20:
            score += 2.0; reasons.append("5일선 골든크로스 → 단기 매수 신호")
        elif psma5 > psma20 and sma5 < sma20:
            score -= 2.0; reasons.append("5일선 데드크로스 → 단기 매도 신호")
        elif sma5 > sma20:
            score += 0.5; reasons.append("단기선 > 중기선 (상승 배열)")
        else:
            score -= 0.5; reasons.append("단기선 < 중기선 (하락 배열)")

    # ── 5. 거래량 ──────────────────────────────────────────────────────────
    vol    = float(data["Volume"].iloc[-1])
    vol_ma = _f(last, "Volume_MA20")
    if vol_ma and vol_ma > 0:
        p_chg = (price - float(close.iloc[-2])) / float(close.iloc[-2])
        ratio = vol / vol_ma
        if ratio > 2.0:
            if p_chg > 0:
                score += 1.0; reasons.append(f"거래량 폭증 (+{ratio:.1f}x) + 상승 → 강한 매수세")
            else:
                score -= 1.0; reasons.append(f"거래량 폭증 (+{ratio:.1f}x) + 하락 → 강한 매도세")
        elif ratio > 1.5:
            if p_chg > 0:
                score += 0.5; reasons.append(f"거래량 증가 ({ratio:.1f}x) + 상승")
            else:
                score -= 0.5

    # ── 최종 판정 ──────────────────────────────────────────────────────────
    s = int(round(score))
    if   s >= 5: label, badge = "강력 매수", "🟢🟢"
    elif s >= 3: label, badge = "매수",      "🟢"
    elif s >= 1: label, badge = "약한 매수", "🔵"
    elif s == 0: label, badge = "중립/관망", "⚪"
    elif s >= -2: label, badge = "약한 매도", "🟡"
    elif s >= -4: label, badge = "매도",      "🔴"
    else:         label, badge = "강력 매도", "🔴🔴"

    return {
        "score":   s,
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

def get_recommendations(tickers_dict: dict) -> pd.DataFrame:
    """여러 종목을 분석하여 점수 순으로 추천 목록 생성"""
    rows = []
    for name, ticker in tickers_dict.items():
        try:
            data = get_stock_data(ticker, "3mo")
            if data.empty:
                continue
            sig = generate_signals(data)
            exp = calculate_expected_return(data, sig)
            close = data["Close"]
            price  = float(close.iloc[-1])
            chg_1d = float((close.iloc[-1] / close.iloc[-2] - 1) * 100) if len(close) >= 2 else 0
            rows.append({
                "종목명":         name,
                "티커":           ticker,
                "현재가":         price,
                "등락률(1일)%":   round(chg_1d, 2),
                "신호":           sig.get("label", "N/A"),
                "신호점수":       sig.get("score", 0),
                "예상수익률(%)":  exp.get("expected_return_pct", 0),
                "변동성(%)":      exp.get("hist_volatility", 0),
                "모멘텀(20일)%":  exp.get("momentum_20d", 0),
                "샤프지수":       exp.get("sharpe", 0),
            })
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("신호점수", ascending=False).reset_index(drop=True)


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

def get_fundamental_data(ticker: str) -> dict:
    """yfinance에서 펀더멘털 지표 조회"""
    try:
        info = yf.Ticker(ticker).info
        return {
            "per":              info.get("trailingPE"),
            "pbr":              info.get("priceToBook"),
            "roe":              info.get("returnOnEquity"),       # decimal (0.15 = 15%)
            "debt_equity":      info.get("debtToEquity"),         # % (50 = 50%)
            "revenue_growth":   info.get("revenueGrowth"),        # decimal
            "earnings_growth":  info.get("earningsGrowth"),       # decimal
            "operating_margins":info.get("operatingMargins"),
            "w52_high":         info.get("fiftyTwoWeekHigh"),
            "w52_low":          info.get("fiftyTwoWeekLow"),
            "market_cap":       info.get("marketCap"),
            "free_cashflow":    info.get("freeCashflow"),
            "eps_ttm":          info.get("trailingEps"),
            "forward_pe":       info.get("forwardPE"),
            "sector":           info.get("sector", "N/A"),
            "industry":         info.get("industry", "N/A"),
            "short_name":       info.get("shortName", ticker),
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

    # ATR (Average True Range, 14일)
    high      = data["High"]
    low       = data["Low"]
    prev_c    = close.shift(1)
    tr        = pd.concat([high - low, (high - prev_c).abs(), (low - prev_c).abs()], axis=1).max(axis=1)
    atr_val   = float(tr.rolling(14).mean().iloc[-1])

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
