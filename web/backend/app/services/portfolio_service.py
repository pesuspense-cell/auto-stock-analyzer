"""portfolio_service.py — 보유 종목 종합 진단 + 매매 대응 지침.

각 보유 종목을 차트(기술) 신호 + 모멘텀 + 손절선 점검으로 종합 평가해 종목별 상세
리포트(대응 액션)를 만들고, 전체 포트폴리오 종합 평가(수익률·집중도·시장상태·액션 분포)를
함께 제공한다. 무거운 연산이라 워커(interactive 레인)에서 실행된다.
"""
from __future__ import annotations

import concurrent.futures
import logging

from app import bootstrap  # noqa: F401
from app.core.cache import ttl_cache
from app.services import analysis_service

from src.indicators import generate_signals, calculate_expected_return
from src.portfolio_optimizer import (
    classify_sectors,
    scan_market_momentum,
    build_rebalancing_guide,
)

logger = logging.getLogger(__name__)

ENTRY_STOP_PCT = -8.0   # 진입가 대비 손절선(%)
ATR_STOP_MULT = 2.0     # 현재가 기준 ATR 손절 배수


@ttl_cache(ttl=600)
def market_momentum() -> dict:
    """KOSPI/KOSDAQ 추세 + 섹터 ETF 모멘텀 (10분 캐시 — 사용자 무관)."""
    try:
        return scan_market_momentum() or {}
    except Exception as e:
        logger.warning("[portfolio] 시장 모멘텀 스캔 실패: %s", e)
        return {"market_status": "데이터 없음", "sector_scores": []}


def _decide_action(pnl_pct: float, tech_score: float, momentum: float,
                   win_prob: float, stop_breached: bool) -> tuple[str, str, list[str]]:
    """종합 신호 → 대응 액션(label, level, reasons). level: danger/warn/good/neutral."""
    reasons: list[str] = []

    if stop_breached:
        return ("손절 검토", "danger",
                [f"진입가 대비 {ENTRY_STOP_PCT:.0f}% 이상 하락 — 손실 확대 방지 위해 손절선 점검"])

    if tech_score <= -4:
        reasons.append(f"기술적 약세 신호(점수 {tech_score:+.0f})")
        if pnl_pct > 0:
            reasons.append("수익 구간에서 약세 전환 — 일부 정리 고려")
            return ("비중 축소", "warn", reasons)
        return ("매도 검토", "warn", reasons)

    if pnl_pct >= 20 and tech_score < 2:
        return ("일부 익절", "good",
                [f"수익 +{pnl_pct:.0f}% 도달 + 모멘텀 둔화 — 분할 익절로 수익 확정"])

    if tech_score >= 4 and momentum > 0:
        return ("보유 / 추가매수", "good",
                [f"강세 추세 지속(점수 {tech_score:+.0f}, 20일 모멘텀 {momentum:+.1f}%)"])

    if tech_score >= 1:
        return ("보유", "neutral", ["추세 유지 — 손절선 관리하며 보유"])

    return ("관망", "neutral", ["신호 혼조 — 방향 확인까지 관망, 추격 매수 자제"])


def _evaluate_holding(item: dict, current_price: float, name: str) -> dict:
    """단일 보유 종목 종합 평가 리포트."""
    ticker = item["ticker"]
    avg = float(item["avg_price"]) or 0.0
    qty = float(item["quantity"])
    price = float(current_price) if current_price else avg
    pnl_pct = (price / avg - 1) * 100 if avg else 0.0

    base = {
        "ticker": ticker, "name": name, "quantity": qty,
        "avg_price": avg, "current_price": price, "pnl_pct": round(pnl_pct, 2),
        "eval_value": round(price * qty, 0),
    }

    try:
        data = analysis_service.stock_data(ticker, "6mo")
        if data is None or data.empty or "Close" not in data.columns:
            raise ValueError("데이터 없음")

        signals = generate_signals(data) or {}
        tech_score = float(signals.get("score", 0))
        expected = calculate_expected_return(
            data, signals, ticker=ticker, benchmark_returns=analysis_service.bench_returns(ticker)
        ) or {}

        atr_pct = float(expected.get("atr_pct", 0) or 0)
        momentum = float(expected.get("momentum_20d", 0) or 0)
        win_prob = expected.get("win_prob")
        exp_ret = expected.get("expected_return_pct")

        # 손절선: 진입가 -8% 와 현재가 -2ATR 중 높은 쪽(수익 보호 트레일링)
        entry_stop = avg * (1 + ENTRY_STOP_PCT / 100) if avg else 0.0
        atr_stop = price * (1 - ATR_STOP_MULT * atr_pct / 100) if atr_pct else entry_stop
        stop_loss_price = max(entry_stop, atr_stop)
        stop_breached = avg > 0 and price <= entry_stop
        stop_distance_pct = (stop_loss_price / price - 1) * 100 if price else 0.0

        action, level, reasons = _decide_action(pnl_pct, tech_score, momentum, win_prob or 50.0, stop_breached)

        return {
            **base,
            "ok": True,
            "tech_score": round(tech_score, 1),
            "signal_label": signals.get("label", ""),
            "signal_badge": signals.get("badge", ""),
            "expected_return_pct": exp_ret,
            "win_prob": win_prob,
            "momentum_20d": round(momentum, 2),
            "atr_pct": round(atr_pct, 2),
            "stop_loss_price": round(stop_loss_price, 2),
            "stop_distance_pct": round(stop_distance_pct, 2),
            "stop_breached": stop_breached,
            "action": action,
            "action_level": level,
            "reasons": reasons,
        }
    except Exception as e:
        logger.warning("[portfolio] %s 평가 실패: %s", ticker, e)
        return {**base, "ok": False, "action": "데이터 없음", "action_level": "neutral",
                "reasons": ["시세/지표 데이터를 불러올 수 없습니다."]}


def _overall(holdings: list[dict], total_value: float, total_cost: float,
             guide: dict, market_status: str) -> dict:
    """전체 종합 평가."""
    counts = {"danger": 0, "warn": 0, "good": 0, "neutral": 0}
    for h in holdings:
        counts[h.get("action_level", "neutral")] = counts.get(h.get("action_level", "neutral"), 0) + 1
    total_pnl_pct = (total_value / total_cost - 1) * 100 if total_cost else 0.0

    if counts["danger"] > 0:
        verdict, vlevel = f"⚠️ 방어적 대응 필요 — 손절 점검 대상 {counts['danger']}종목", "danger"
    elif counts["warn"] >= max(1, len(holdings) // 2):
        verdict, vlevel = "비중 조정 권고 — 약세 종목 비중이 높습니다", "warn"
    elif counts["good"] >= max(1, len(holdings) // 2):
        verdict, vlevel = "양호 — 강세·익절 관리 구간", "good"
    else:
        verdict, vlevel = "중립 — 추세 확인하며 관리", "neutral"

    return {
        "total_value": round(total_value, 0),
        "total_cost": round(total_cost, 0),
        "total_pnl_pct": round(total_pnl_pct, 2),
        "holdings_count": len(holdings),
        "action_counts": counts,
        "verdict": verdict,
        "verdict_level": vlevel,
        "hhi": guide.get("hhi"),
        "is_concentrated": guide.get("is_concentrated", False),
        "market_status": market_status,
    }


def analyze(items: list[dict], prices: dict[str, float], name_map: dict[str, str]) -> dict:
    """보유 종목 종합 진단.

    items    : [{"ticker", "avg_price", "quantity"}, ...]
    prices   : {ticker: current_price}
    name_map : {ticker: 표시명}
    """
    if not items:
        return {"empty": True}

    # 종목별 평가 (병렬 — 대부분 외부 I/O 대기)
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(items))) as pool:
        futs = {
            pool.submit(
                _evaluate_holding, it,
                prices.get(it["ticker"]) or it["avg_price"],
                name_map.get(it["ticker"], it["ticker"]),
            ): it["ticker"]
            for it in items
        }
        holdings = [f.result() for f in futs]

    # 액션 위험도 → 정렬(손절 > 경고 > 양호 > 중립)
    order = {"danger": 0, "warn": 1, "good": 2, "neutral": 3}
    holdings.sort(key=lambda h: (order.get(h.get("action_level", "neutral"), 3), -abs(h.get("pnl_pct", 0))))

    total_value = sum(h["eval_value"] for h in holdings)
    total_cost = sum(h["avg_price"] * h["quantity"] for h in holdings)

    # 섹터 집중도 + 시장/섹터 모멘텀 (보조 — 전체 평가용)
    sector_data = classify_sectors(items, prices)
    momentum = market_momentum()
    guide = build_rebalancing_guide(sector_data, momentum, name_map)
    market_status = momentum.get("market_status", "")

    overall = _overall(holdings, total_value, total_cost, guide, market_status)

    return {
        "empty": False,
        "overall": overall,
        "holdings": holdings,
        "guide": guide,
    }
