"""
recommendation_engine.py — 투자금 기반 AI 종목·비중 추천 엔진 (다시장 동적 버전)

[플로우]
  투자금 입력
  → KOSPI + KOSDAQ + NASDAQ 시가총액 상위 500개씩 동적 로드 (FinanceDataReader)
  → L1: 시장별 상위 150개 가격 배치 다운로드 → RSI 30-80 + 20일 이평 모멘텀 필터 → 전체 상위 60개
  → L2: 뉴스 키워드 감성 분석 → 0.6 이상 통과
  → 선정: 감성 점수 내림차순 + 시장별 최대 2개 제한으로 5개 선정
  → 투자 성향별 10~40% 비중 배분
  → 종목별 최적 매수 수량(정수) 산출 (USD 종목은 환율 적용)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger("recommendation_engine")

# ─── 후보 규모 설정 ────────────────────────────────────────────────────────────
CANDIDATES_PER_MARKET  = 500   # FDR에서 시장별 로드할 최대 종목 수 (시총 기준)
L1_PER_MARKET          = 150   # 가격 다운로드 대상 (시총 상위 N개 × 3시장 = 최대 450개)
L2_MAX_TOTAL           = 60    # 뉴스 분석 대상 최대 종목 수
MAX_PER_MARKET         = 2     # 최종 추천 시장별 최대 편입 수 (2×3 = 6 → 상위 5개)
PRICE_DL_BATCH         = 100   # yfinance 배치 크기 (타임아웃 방지)

# ─── 필터링 기준 ──────────────────────────────────────────────────────────────
SENTIMENT_THRESHOLD  = 0.60
RSI_OVERSOLD_BOUND   = 30
RSI_OVERBOUGHT_BOUND = 80

# ─── 하위 호환 상수 (XAI 섹션이 참조할 수 있으므로 유지) ─────────────────────
CANDIDATE_STOCKS: list[tuple[str, str, str]] = []   # 동적 로드 후 채워지지 않음 (레거시)


# ─── 결과 데이터클래스 ────────────────────────────────────────────────────────
@dataclass
class RecommendedStock:
    ticker:            str
    name:              str
    sector:            str
    market:            str       # "KOSPI" | "KOSDAQ" | "NASDAQ"
    currency:          str       # "KRW" | "USD"
    current_price:     float     # 원래 통화 가격 (KRW 또는 USD)
    current_price_krw: float     # KRW 환산가 (수량 계산·비교용)
    weight:            float     # 0.0–1.0
    quantity:          int       # 매수 수량 (정수)
    invested:          float     # KRW 기준 실제 투자액
    sentiment_score:   float     # 정규화 0–1
    rsi:               float
    reason:            str


# ─── 동적 후보 로드 ───────────────────────────────────────────────────────────
def _load_candidates() -> list[tuple[str, str, str, str]]:
    """KOSPI + KOSDAQ + NASDAQ 시가총액 상위 종목 로드.

    반환: [(ticker, name, sector, market), ...]
    시장별 CANDIDATES_PER_MARKET개 로드 후 L1_PER_MARKET개만 실제 가격 다운로드 대상.
    """
    try:
        import FinanceDataReader as fdr
    except ImportError:
        logger.error("FinanceDataReader가 설치되어 있지 않습니다.")
        return []

    results: list[tuple[str, str, str, str]] = []

    for listing_key, suffix, market_label in [
        ("KOSPI",  ".KS", "KOSPI"),
        ("KOSDAQ", ".KQ", "KOSDAQ"),
        ("NASDAQ", "",    "NASDAQ"),
    ]:
        try:
            df = fdr.StockListing(listing_key)
            df = df.dropna(subset=["Name"])
            code_col = "Code" if "Code" in df.columns else "Symbol"
            df = df.dropna(subset=[code_col])

            if "Marcap" in df.columns:
                df = df[df["Marcap"] > 0].sort_values("Marcap", ascending=False)

            count = 0
            for _, row in df.iterrows():
                if count >= L1_PER_MARKET:
                    break
                raw_code = str(row[code_col]).strip()
                name     = str(row["Name"]).strip()
                if not raw_code or not name:
                    continue

                if suffix:  # KRX
                    ticker = f"{raw_code.zfill(6)}{suffix}"
                else:       # NASDAQ — 알파벳 티커만 허용
                    if not raw_code.replace("-", "").isalpha():
                        continue
                    ticker = raw_code

                # 섹터: FDR 제공 시 활용, 없으면 시장명 사용
                sector = (
                    str(row.get("Sector", row.get("Industry", ""))).strip()
                    or market_label
                )
                results.append((ticker, name, sector, market_label))
                count += 1

        except Exception as exc:
            logger.warning("[%s] 목록 로드 실패: %s", listing_key, exc)

    return results


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
def _build_reason(sector: str, market: str, sentiment: float, rsi: float) -> str:
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
    market_label = {"KOSPI": "코스피", "KOSDAQ": "코스닥", "NASDAQ": "나스닥"}.get(market, market)
    return f"{market_label} {sector} 섹터의 {sent_desc}가 뚜렷하며, {rsi_desc}."


# ─── 비중 배분 ────────────────────────────────────────────────────────────────
def _allocate_weights(scores: list[float], risk_profile: str = "중립형") -> list[float]:
    """감성 점수 비례 배분 후 성향별 상·하한 클램핑.

    보수형: 10–25%  /  중립형: 10–30%  /  공격형: 10–40%
    """
    n = len(scores)
    if n == 0:
        return []
    bounds = {"보수형": (0.10, 0.25), "중립형": (0.10, 0.30), "공격형": (0.10, 0.40)}
    lo, hi = bounds.get(risk_profile, (0.10, 0.30))
    total  = sum(scores)
    raw    = [s / total for s in scores] if total > 0 else [1 / n] * n
    clamped = [max(lo, min(hi, w)) for w in raw]
    c_sum   = sum(clamped)
    return [round(w / c_sum, 4) for w in clamped]


# ─── 메인 추천 함수 ───────────────────────────────────────────────────────────
def run_recommendation(
    investment_amount: int,
    risk_profile: str = "중립형",
    api_key: str = "",
    groq_api_key: str = "",
) -> dict:
    """KOSPI + KOSDAQ + NASDAQ 통합 모멘텀 주도주 추천.

    반환 dict:
      recommendations : list[RecommendedStock]
      remaining_cash  : float   — 매수 후 잔금 (KRW)
      total_invested  : float   — 실제 투자액 합계 (KRW)
      pool_size       : int     — L1 후보 풀 크기
      l1_pass         : int     — RSI·모멘텀 통과 수
      l2_pass         : int     — 뉴스 감성 통과 수
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
            "total_invested": 0, "pool_size": 0, "l1_pass": 0, "l2_pass": 0,
            "error": f"모듈 로드 실패: {exc}",
        }

    # ── Step 0: 후보 로드 ────────────────────────────────────────────────────
    all_candidates = _load_candidates()
    if not all_candidates:
        return {
            "recommendations": [], "remaining_cash": investment_amount,
            "total_invested": 0, "pool_size": 0, "l1_pass": 0, "l2_pass": 0,
            "error": "후보 종목 로드 실패 (FinanceDataReader 오류)",
        }

    tickers    = [c[0] for c in all_candidates]
    name_map   = {c[0]: c[1] for c in all_candidates}
    sector_map = {c[0]: c[2] for c in all_candidates}
    market_map = {c[0]: c[3] for c in all_candidates}
    pool_size  = len(tickers)

    # ── Step 0.5: USD/KRW 환율 ───────────────────────────────────────────────
    usd_krw = 1300.0
    try:
        _fx = yf.download("USDKRW=X", period="2d", auto_adjust=True, progress=False)
        if not _fx.empty:
            _fx_s = (_fx["Close"] if "Close" in _fx.columns else _fx.iloc[:, 0]).dropna()
            if not _fx_s.empty:
                usd_krw = float(_fx_s.iloc[-1])
    except Exception:
        pass

    def _to_krw(ticker: str, price: float) -> float:
        """USD 종목이면 KRW 환산, KRW 종목은 그대로."""
        return price if ticker.upper().endswith((".KS", ".KQ")) else price * usd_krw

    def _currency(ticker: str) -> str:
        return "KRW" if ticker.upper().endswith((".KS", ".KQ")) else "USD"

    # ── Step 1: 배치 가격 다운로드 (PRICE_DL_BATCH개씩) ─────────────────────
    prices_map: dict[str, float] = {}
    rsi_map:    dict[str, float] = {}
    ma20_map:   dict[str, float] = {}

    for i in range(0, len(tickers), PRICE_DL_BATCH):
        batch = tickers[i: i + PRICE_DL_BATCH]
        try:
            raw = yf.download(
                batch, period="3mo", auto_adjust=True,
                progress=False, threads=True,
            )
            if raw.empty:
                continue
            close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw

            for t in batch:
                try:
                    s = (
                        close[t].dropna()
                        if isinstance(close, pd.DataFrame) and t in close.columns
                        else (close.dropna() if not isinstance(close, pd.DataFrame) else pd.Series(dtype=float))
                    )
                    if not s.empty and len(s) >= 5:
                        prices_map[t] = float(s.iloc[-1])
                        rsi_map[t]    = _calc_rsi(s)
                        ma20_map[t]   = float(s.tail(20).mean())
                except Exception:
                    pass
        except Exception as exc:
            logger.warning("배치 [%d:%d] 다운로드 실패: %s", i, i + PRICE_DL_BATCH, exc)

    # ── Step 2: L1 필터 — RSI 범위 + 20일 이평 모멘텀 ────────────────────────
    l1_pass_list: list[tuple[str, float, float, float]] = []
    # (ticker, rsi, native_price, momentum_pct)

    for t in tickers:
        native_px = prices_map.get(t, 0.0)
        if native_px <= 0:
            continue
        krw_px = _to_krw(t, native_px)
        if krw_px > investment_amount:   # 1주도 못 사는 경우 제외
            continue
        rsi  = rsi_map.get(t, 50.0)
        if rsi < RSI_OVERSOLD_BOUND or rsi > RSI_OVERBOUGHT_BOUND:
            continue
        ma20_native = ma20_map.get(t, native_px)
        momentum    = (native_px / ma20_native - 1) * 100 if ma20_native > 0 else 0.0
        l1_pass_list.append((t, rsi, native_px, momentum))

    # 모멘텀 내림차순 정렬 후 L2_MAX_TOTAL개 선택
    l1_pass_list.sort(key=lambda x: x[3], reverse=True)
    l2_pool   = l1_pass_list[:L2_MAX_TOTAL]
    l2_tickers = [c[0] for c in l2_pool]

    if not l2_pool:
        return {
            "recommendations": [], "remaining_cash": investment_amount,
            "total_invested": 0, "pool_size": pool_size,
            "l1_pass": 0, "l2_pass": 0,
            "error": (
                f"L1 필터(RSI {RSI_OVERSOLD_BOUND}~{RSI_OVERBOUGHT_BOUND} + "
                "20일 이평 모멘텀) 통과 종목이 없습니다. 잠시 후 다시 시도해보세요."
            ),
        }

    # ── Step 3: L2 뉴스 감성 분석 ────────────────────────────────────────────
    news_map: dict[str, list] = {}
    try:
        news_map = fetch_multi_news_fast(l2_tickers, max_items=8)
    except Exception as exc:
        logger.warning("뉴스 수집 실패 (중립 처리): %s", exc)

    sentiment_map: dict[str, float] = {}
    for t in l2_tickers:
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

    # ── Step 4: 감성 필터 ─────────────────────────────────────────────────────
    l3_pool = [
        (t, sentiment_map.get(t, 0.50), rsi, native_px, mom)
        for t, rsi, native_px, mom in l2_pool
        if sentiment_map.get(t, 0.50) >= SENTIMENT_THRESHOLD
    ]

    if not l3_pool:
        return {
            "recommendations": [], "remaining_cash": investment_amount,
            "total_invested": 0, "pool_size": pool_size,
            "l1_pass": len(l1_pass_list), "l2_pass": 0,
            "error": (
                f"뉴스 모멘텀 조건(감성 점수 {SENTIMENT_THRESHOLD:.0%} 이상)을 충족하는 "
                "종목이 없습니다. 잠시 후 다시 시도해 보세요."
            ),
        }

    # ── Step 5: 선정 — 감성 내림차순 + 시장별 MAX_PER_MARKET 제한 ────────────
    l3_pool.sort(key=lambda x: x[1], reverse=True)
    selected: list[tuple[str, float, float, float]] = []
    market_count: dict[str, int] = {}

    for t, sent, rsi, native_px, _ in l3_pool:
        mkt = market_map.get(t, "기타")
        if market_count.get(mkt, 0) >= MAX_PER_MARKET:
            continue
        market_count[mkt] = market_count.get(mkt, 0) + 1
        selected.append((t, sent, rsi, native_px))
        if len(selected) >= 5:
            break

    # 시장 편중 방지로 5개 미달이면 점수 순으로 잔여 추가
    if len(selected) < 5:
        selected_set = {s[0] for s in selected}
        for t, sent, rsi, native_px, _ in l3_pool:
            if t in selected_set:
                continue
            selected.append((t, sent, rsi, native_px))
            selected_set.add(t)
            if len(selected) >= 5:
                break

    if not selected:
        return {
            "recommendations": [], "remaining_cash": investment_amount,
            "total_invested": 0, "pool_size": pool_size,
            "l1_pass": len(l1_pass_list), "l2_pass": len(l3_pool),
            "error": "선정 가능한 종목이 없습니다.",
        }

    # ── Step 6: 비중 배분 + 수량 산출 ─────────────────────────────────────────
    scores  = [s[1] for s in selected]
    weights = _allocate_weights(scores, risk_profile)

    results: list[RecommendedStock] = []
    total_invested = 0.0

    for (t, sent, rsi, native_px), w in zip(selected, weights):
        krw_px   = _to_krw(t, native_px)
        alloc    = investment_amount * w
        qty      = int(alloc // krw_px)
        qty      = max(qty, 1)
        invested = qty * krw_px
        # 잔금 부족 시 수량 조정
        if total_invested + invested > investment_amount:
            qty      = max(int((investment_amount - total_invested) // krw_px), 0)
            invested = qty * krw_px
        if qty == 0:
            continue
        total_invested += invested
        results.append(RecommendedStock(
            ticker            = t,
            name              = name_map.get(t, t.split(".")[0]),
            sector            = sector_map.get(t, "기타"),
            market            = market_map.get(t, "기타"),
            currency          = _currency(t),
            current_price     = native_px,
            current_price_krw = krw_px,
            weight            = w,
            quantity          = qty,
            invested          = invested,
            sentiment_score   = sent,
            rsi               = rsi,
            reason            = _build_reason(
                sector_map.get(t, "기타"), market_map.get(t, "기타"), sent, rsi
            ),
        ))

    remaining = max(investment_amount - total_invested, 0.0)

    return {
        "recommendations": results,
        "remaining_cash":  remaining,
        "total_invested":  total_invested,
        "pool_size":       pool_size,
        "l1_pass":         len(l1_pass_list),
        "l2_pass":         len(l3_pool),
        "error":           None,
    }


def recommendation_to_dict(r: RecommendedStock) -> dict:
    """RecommendedStock → JSON 직렬화 가능 dict."""
    return {
        "ticker":            r.ticker,
        "name":              r.name,
        "sector":            r.sector,
        "market":            r.market,
        "currency":          r.currency,
        "current_price":     r.current_price,
        "current_price_krw": r.current_price_krw,
        "weight":            r.weight,
        "quantity":          r.quantity,
        "invested":          r.invested,
        "sentiment_score":   r.sentiment_score,
        "rsi":               r.rsi,
        "reason":            r.reason,
    }
