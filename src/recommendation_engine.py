"""
recommendation_engine.py — 투자금 기반 AI 종목·비중 추천 엔진

[플로우]
  투자금 입력 → 후보 종목 가격/뉴스 배치 수집
    → 감성 점수 0.6 이상 & RSI 과매도(30 이하) 제외 필터링
    → 섹터 중복 없이 상위 5개 선정
    → 투자 성향별 10~30% 비중 배분
    → 종목별 최적 매수 수량(정수) 산출
    → 추천 이유 한 줄 생성
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger("recommendation_engine")

# ─── 후보 종목 풀 (섹터 대표주 25개) ─────────────────────────────────────────
CANDIDATE_STOCKS: list[tuple[str, str, str]] = [
    # (yfinance ticker, 회사명, 섹터)
    ("005930.KS", "삼성전자",       "반도체"),
    ("000660.KS", "SK하이닉스",     "반도체"),
    ("035420.KS", "NAVER",         "IT/플랫폼"),
    ("035720.KS", "카카오",         "IT/플랫폼"),
    ("207940.KS", "삼성바이오로직스","바이오"),
    ("068270.KS", "셀트리온",       "바이오"),
    ("105560.KS", "KB금융",         "금융"),
    ("055550.KS", "신한지주",       "금융"),
    ("086790.KS", "하나금융지주",   "금융"),
    ("066570.KS", "LG전자",         "가전/전자"),
    ("012330.KS", "현대모비스",     "자동차"),
    ("005380.KS", "현대차",         "자동차"),
    ("000270.KS", "기아",           "자동차"),
    ("051910.KS", "LG화학",         "배터리/화학"),
    ("006400.KS", "삼성SDI",        "배터리/화학"),
    ("017670.KS", "SK텔레콤",       "통신"),
    ("030200.KS", "KT",             "통신"),
    ("034730.KS", "SK이노베이션",   "에너지"),
    ("090430.KS", "아모레퍼시픽",   "소비재"),
    ("051900.KS", "LG생활건강",     "소비재"),
    ("010130.KS", "고려아연",       "소재"),
    ("036570.KS", "엔씨소프트",     "게임"),
    ("259960.KS", "크래프톤",       "게임"),
    ("009830.KS", "한화솔루션",     "신재생에너지"),
    ("018260.KS", "삼성에스디에스", "IT서비스"),
]

# ─── 필터링 기준 ──────────────────────────────────────────────────────────────
SENTIMENT_THRESHOLD = 0.60   # 정규화 감성 점수 하한 (0–1 범위, raw -5~+5 → 0~1 변환)
RSI_OVERSOLD_BOUND  = 30     # RSI 과매도 기준 이하 종목 제외


# ─── 결과 데이터클래스 ────────────────────────────────────────────────────────
@dataclass
class RecommendedStock:
    ticker:          str
    name:            str
    sector:          str
    current_price:   float
    weight:          float    # 0.0–1.0
    quantity:        int      # 매수 수량 (정수)
    invested:        float    # 실제 투자액 = quantity * current_price
    sentiment_score: float    # 정규화 0–1
    rsi:             float
    reason:          str      # 추천 이유 한 줄


# ─── RSI 계산 ─────────────────────────────────────────────────────────────────
def _calc_rsi(prices: pd.Series, period: int = 14) -> float:
    if len(prices) < period + 1:
        return 50.0
    delta = prices.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    last_loss = float(loss.iloc[-1]) if not loss.empty else 0.0
    if last_loss == 0:
        return 100.0
    rs = float(gain.iloc[-1]) / last_loss
    return round(100 - 100 / (1 + rs), 1)


# ─── 추천 이유 생성 ───────────────────────────────────────────────────────────
def _build_reason(sector: str, sentiment: float, rsi: float) -> str:
    sent_desc = (
        "강한 뉴스 호재" if sentiment >= 0.80
        else "긍정적 뉴스 흐름" if sentiment >= 0.65
        else "중립 이상 뉴스 분위기"
    )
    rsi_desc = (
        f"기술적으로 지지선(RSI {rsi:.0f})에 위치해 반등이 기대됩니다"
        if rsi < 50
        else f"RSI {rsi:.0f}으로 상승 모멘텀이 유지되고 있습니다"
    )
    return f"{sector} 섹터의 {sent_desc}가 뚜렷하며, {rsi_desc}."


# ─── 비중 배분 ────────────────────────────────────────────────────────────────
def _allocate_weights(scores: list[float], risk_profile: str = "중립형") -> list[float]:
    """
    감성 점수 비례 배분 후 성향별 상·하한 클램핑.

    보수형: 10–25%  /  중립형: 10–30%  /  공격형: 10–40%
    """
    n = len(scores)
    if n == 0:
        return []

    bounds = {"보수형": (0.10, 0.25), "중립형": (0.10, 0.30), "공격형": (0.10, 0.40)}
    lo, hi = bounds.get(risk_profile, (0.10, 0.30))

    total = sum(scores)
    raw   = [s / total for s in scores] if total > 0 else [1 / n] * n
    clamped = [max(lo, min(hi, w)) for w in raw]

    c_sum = sum(clamped)
    return [round(w / c_sum, 4) for w in clamped]


# ─── 메인 추천 함수 ───────────────────────────────────────────────────────────
def run_recommendation(
    investment_amount: int,
    risk_profile: str = "중립형",
    api_key: str = "",
    groq_api_key: str = "",
) -> dict:
    """
    투자금 기반 AI 종목·비중 추천.

    반환 dict:
      recommendations : list[RecommendedStock]
      remaining_cash  : float   — 매수 후 잔금
      total_invested  : float   — 실제 투자액 합계
      error           : str | None
    """
    import yfinance as yf

    try:
        from src.news_async import (
            fetch_multi_news_fast,
            stage1_title_filter,
            stage2_keyword_filter,
        )
        from stock_ai import analyze_news_sentiment_keywords
    except Exception as exc:
        return {
            "recommendations": [], "remaining_cash": investment_amount,
            "total_invested": 0, "error": f"모듈 로드 실패: {exc}",
        }

    tickers    = [c[0] for c in CANDIDATE_STOCKS]
    name_map   = {c[0]: c[1] for c in CANDIDATE_STOCKS}
    sector_map = {c[0]: c[2] for c in CANDIDATE_STOCKS}

    # ── Step 1: 배치 가격 데이터 (3개월 일봉) ─────────────────────────────────
    prices_map: dict[str, float] = {}
    rsi_map:    dict[str, float] = {}
    try:
        raw = yf.download(
            tickers, period="3mo", auto_adjust=True,
            progress=False, threads=True,
        )
        # MultiIndex (metric, ticker) 형태
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw

        for t in tickers:
            try:
                s = close[t].dropna() if t in close.columns else pd.Series(dtype=float)
                if not s.empty:
                    prices_map[t] = float(s.iloc[-1])
                    rsi_map[t]    = _calc_rsi(s)
            except Exception:
                pass
    except Exception as exc:
        logger.warning("가격 다운로드 실패: %s", exc)
        return {
            "recommendations": [], "remaining_cash": investment_amount,
            "total_invested": 0, "error": f"가격 데이터 조회 실패: {exc}",
        }

    # ── Step 2: 뉴스 감성 분석 (키워드 고속 경로) ─────────────────────────────
    news_map: dict[str, list] = {}
    try:
        news_map = fetch_multi_news_fast(tickers, max_items=8)
    except Exception as exc:
        logger.warning("뉴스 수집 실패 (중립 처리): %s", exc)

    sentiment_map: dict[str, float] = {}
    for t in tickers:
        items = news_map.get(t, [])
        if not items:
            sentiment_map[t] = 0.50
            continue
        nm        = name_map.get(t, "")
        filtered  = stage1_title_filter(items, nm)
        deep, pre = stage2_keyword_filter(filtered)
        result    = analyze_news_sentiment_keywords(deep + pre, t, nm)
        raw_score = result.get("score", 0.0)
        # -5 ~ +5 → 0 ~ 1 정규화
        sentiment_map[t] = round((raw_score + 5) / 10, 4)

    # ── Step 3: 필터링 (감성 + RSI + 가격 상한) ───────────────────────────────
    candidates: list[tuple[str, float, float, float]] = []
    for t in tickers:
        price = prices_map.get(t, 0)
        if price <= 0 or price > investment_amount:
            continue
        sent = sentiment_map.get(t, 0.50)
        if sent < SENTIMENT_THRESHOLD:
            continue
        rsi = rsi_map.get(t, 50.0)
        if rsi < RSI_OVERSOLD_BOUND:
            continue
        candidates.append((t, sent, rsi, price))

    if not candidates:
        return {
            "recommendations": [],
            "remaining_cash":  investment_amount,
            "total_invested":  0,
            "error": (
                "현재 조건(뉴스 감성 0.6 이상, RSI 과매도 제외)을 충족하는 종목이 없습니다. "
                "투자금을 높이거나 잠시 후 다시 시도해보세요."
            ),
        }

    # ── Step 4: 감성 점수 내림차순 정렬 후 섹터 중복 제거, 상위 5개 선정 ────
    candidates.sort(key=lambda x: x[1], reverse=True)
    selected: list[tuple[str, float, float, float]] = []
    used_sectors: set[str] = set()

    for t, sent, rsi, price in candidates:
        sec = sector_map[t]
        if sec in used_sectors:
            continue
        used_sectors.add(sec)
        selected.append((t, sent, rsi, price))
        if len(selected) >= 5:
            break

    # 섹터 제한으로 5개 미달이면 같은 섹터 추가 허용 (투자금 활용)
    if len(selected) < 5:
        for t, sent, rsi, price in candidates:
            if any(s[0] == t for s in selected):
                continue
            selected.append((t, sent, rsi, price))
            if len(selected) >= 5:
                break

    if not selected:
        return {
            "recommendations": [],
            "remaining_cash":  investment_amount,
            "total_invested":  0,
            "error": "섹터 다각화 후 선정 가능한 종목이 없습니다.",
        }

    # ── Step 5: 비중 배분 및 수량 산출 ───────────────────────────────────────
    scores  = [s[1] for s in selected]
    weights = _allocate_weights(scores, risk_profile)

    results: list[RecommendedStock] = []
    total_invested = 0.0

    for (t, sent, rsi, price), w in zip(selected, weights):
        alloc    = investment_amount * w
        qty      = int(alloc // price)
        qty      = max(qty, 1)          # 최소 1주 보장
        invested = qty * price
        # 잔금 부족 시 수량 조정
        if total_invested + invested > investment_amount:
            qty      = max(int((investment_amount - total_invested) // price), 0)
            invested = qty * price
        if qty == 0:
            continue
        total_invested += invested
        results.append(RecommendedStock(
            ticker          = t,
            name            = name_map[t],
            sector          = sector_map[t],
            current_price   = price,
            weight          = w,
            quantity        = qty,
            invested        = invested,
            sentiment_score = sent,
            rsi             = rsi,
            reason          = _build_reason(sector_map[t], sent, rsi),
        ))

    remaining = max(investment_amount - total_invested, 0.0)

    return {
        "recommendations": results,
        "remaining_cash":  remaining,
        "total_invested":  total_invested,
        "error":           None,
    }


def recommendation_to_dict(r: RecommendedStock) -> dict:
    """RecommendedStock → JSON 직렬화 가능 dict."""
    return {
        "ticker":          r.ticker,
        "name":            r.name,
        "sector":          r.sector,
        "current_price":   r.current_price,
        "weight":          r.weight,
        "quantity":        r.quantity,
        "invested":        r.invested,
        "sentiment_score": r.sentiment_score,
        "rsi":             r.rsi,
        "reason":          r.reason,
    }
