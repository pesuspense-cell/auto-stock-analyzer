"""
portfolio_optimizer.py - 포트폴리오 섹터 분석 및 리밸런싱 가이드

[기능]
  1. classify_sectors     - 종목별 섹터 분류 + 비중 계산
  2. scan_market_momentum - 시장 지수 추세 + 섹터 ETF 모멘텀 스코어
  3. build_rebalancing_guide - HHI + 조건 매트릭스 기반 리밸런싱 가이드
"""
from __future__ import annotations

import logging

logger = logging.getLogger("portfolio_optimizer")

_EXCLUDE_SECTORS = {"인버스", "레버리지", "코스피200", "코스닥150"}


# ─── 1. 섹터 분류 ──────────────────────────────────────────────────────────────
def classify_sectors(
    items: list[dict],
    prices: dict[str, float],
) -> dict:
    """
    포트폴리오 종목들을 섹터별로 분류하여 비중(%) 계산.

    items   : [{"ticker": str, "avg_price": float, "quantity": float}, ...]
    prices  : {ticker: current_price}

    반환:
      sectors          : dict[sector → {weight, tickers, value}]
      total_value      : float
      item_values      : list[{ticker, sector, value, avg_price, current_price, pnl_pct}]
      unknown_tickers  : list[str]
    """
    from stock_ai import _TICKER_SECTOR  # 지연 import

    total_value = 0.0
    item_values: list[dict] = []
    for it in items:
        ticker = it["ticker"]
        price  = prices.get(ticker) or it["avg_price"]
        value  = price * it["quantity"]
        total_value += value
        pnl_pct = (price / it["avg_price"] - 1) * 100 if it["avg_price"] else 0.0
        sector = _TICKER_SECTOR.get(ticker, "")
        item_values.append({
            "ticker":        ticker,
            "sector":        sector,
            "value":         value,
            "avg_price":     it["avg_price"],
            "current_price": price,
            "pnl_pct":       pnl_pct,
            "quantity":      it["quantity"],
        })

    sectors: dict[str, dict] = {}
    unknown: list[str] = []
    for iv in item_values:
        if not iv["sector"]:
            unknown.append(iv["ticker"])
            continue
        s = iv["sector"]
        if s not in sectors:
            sectors[s] = {"weight": 0.0, "tickers": [], "value": 0.0}
        sectors[s]["value"] += iv["value"]
        sectors[s]["tickers"].append(iv["ticker"])

    if total_value > 0:
        for s in sectors:
            sectors[s]["weight"] = round(sectors[s]["value"] / total_value * 100, 1)

    return {
        "sectors":         sectors,
        "total_value":     total_value,
        "item_values":     item_values,
        "unknown_tickers": unknown,
    }


# ─── 2. 시장 지수 추세 + 섹터 ETF 모멘텀 스코어 ─────────────────────────────────
def scan_market_momentum() -> dict:
    """
    1. KOSPI/KOSDAQ 20일 이평선 대비 현재가 체크 (시장 위험도)
    2. 섹터 ETF별 모멘텀 스코어 계산
       score = (5일 수익률 × 0.5) + (20일 수익률 × 0.3) + (거래량 증가율 × 0.2)
    3. TOP3 / BOTTOM3 분류

    반환:
      market_status    : "상승장" | "하락장"
      kospi_above_ma   : bool
      kosdaq_above_ma  : bool
      sector_scores    : list[{sector, name, score, return_5d, return_20d, vol_growth, rank}]
    """
    import pandas as pd
    import yfinance as yf
    from stock_ai import _ETF_PORTFOLIO_MAP

    # ── KOSPI/KOSDAQ 20일 MA 체크 ──────────────────────────────────────────
    kospi_above_ma  = True
    kosdaq_above_ma = True
    try:
        idx_raw = yf.download(
            ["^KS11", "^KQ11"],
            period="60d",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        if isinstance(idx_raw.columns, pd.MultiIndex):
            ks_c = idx_raw["Close"]["^KS11"].dropna()
            kq_c = idx_raw["Close"]["^KQ11"].dropna()
        else:
            ks_c = idx_raw["Close"].dropna()
            kq_c = ks_c
        if len(ks_c) >= 20:
            kospi_above_ma  = bool(ks_c.iloc[-1] > ks_c.tail(20).mean())
        if len(kq_c) >= 20:
            kosdaq_above_ma = bool(kq_c.iloc[-1] > kq_c.tail(20).mean())
    except Exception as exc:
        logger.warning("지수 MA 조회 실패: %s", exc)

    # ── 섹터 ETF 모멘텀 스코어 ─────────────────────────────────────────────
    ticker_info = {
        f"{code}.KS": info
        for code, info in _ETF_PORTFOLIO_MAP.items()
        if info.get("sector") not in _EXCLUDE_SECTORS
    }
    sector_scores: list[dict] = []

    if ticker_info:
        try:
            raw = yf.download(
                list(ticker_info.keys()),
                period="60d",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            for t_ks, info in ticker_info.items():
                try:
                    if isinstance(raw.columns, pd.MultiIndex):
                        closes  = raw["Close"][t_ks].dropna()
                        volumes = raw["Volume"][t_ks].dropna()
                    else:
                        closes  = raw["Close"].dropna()
                        volumes = raw["Volume"].dropna()

                    if len(closes) < 6:
                        continue

                    ret_5d  = float((closes.iloc[-1] / closes.iloc[-6]  - 1) * 100)
                    ret_20d = float((closes.iloc[-1] / closes.iloc[-21] - 1) * 100) if len(closes) >= 21 else ret_5d

                    if len(volumes) >= 20:
                        vol_recent = float(volumes.tail(5).mean())
                        vol_prev   = float(volumes.iloc[-20:-5].mean())
                        vol_growth = float((vol_recent / vol_prev - 1) * 100) if vol_prev > 0 else 0.0
                    else:
                        vol_growth = 0.0

                    score = (ret_5d * 0.5) + (ret_20d * 0.3) + (vol_growth * 0.2)
                    sector_scores.append({
                        "ticker":     t_ks,
                        "code":       t_ks.replace(".KS", ""),
                        "name":       info["name"],
                        "sector":     info["sector"],
                        "return_5d":  round(ret_5d,    2),
                        "return_20d": round(ret_20d,   2),
                        "vol_growth": round(vol_growth, 1),
                        "score":      round(score,      2),
                    })
                except Exception:
                    continue
        except Exception as exc:
            logger.warning("섹터 ETF 스캔 실패: %s", exc)

    sector_scores.sort(key=lambda x: x["score"], reverse=True)
    n        = len(sector_scores)
    top_n    = min(3, n)
    bottom_n = min(3, n)
    for i, s in enumerate(sector_scores):
        if   i < top_n:         s["rank"] = "TOP"
        elif i >= n - bottom_n: s["rank"] = "BOTTOM"
        else:                   s["rank"] = "NORMAL"

    return {
        "market_status":   "상승장" if kospi_above_ma else "하락장",
        "kospi_above_ma":  kospi_above_ma,
        "kosdaq_above_ma": kosdaq_above_ma,
        "sector_scores":   sector_scores,
    }


# ─── 3. 리밸런싱 가이드 생성 (HHI + 조건 매트릭스) ────────────────────────────
def build_rebalancing_guide(
    sector_data:   dict,
    momentum_data: dict,
    pf_name_map:   dict[str, str],
) -> dict:
    """
    HHI 지수 + 섹터 조건 매트릭스 기반 포트폴리오 진단.

    반환:
      hhi              : float   (허핀달-허쉬만 지수, 섹터비율%² 합산)
      is_concentrated  : bool    (HHI > 2500)
      market_status    : str
      recommendations  : list[{type, icon, sector, weight, tickers, message}]
      missing_top      : list[{sector, name, score, return_5d}]
      profit_take      : list[{ticker, name, pnl_pct, reason}]
      sector_scores    : list    (UI 전달용 모멘텀 랭킹)
    """
    sectors       = sector_data.get("sectors", {})
    item_values   = sector_data.get("item_values", [])
    sector_scores = momentum_data.get("sector_scores", [])
    market_status = momentum_data.get("market_status", "상승장")

    # ── HHI (섹터 비율% 제곱 합) ─────────────────────────────────────────
    hhi = round(sum(v["weight"] ** 2 for v in sectors.values()), 0)
    is_concentrated = hhi > 2500

    # ── 섹터 → 시장 랭크 맵 ─────────────────────────────────────────────
    rank_map: dict[str, str] = {s["sector"]: s["rank"] for s in sector_scores}

    # ── 조건 매트릭스 ────────────────────────────────────────────────────
    # 비중 ≥40% + BOTTOM → 🚨 비중 축소
    # 비중 ≥40% + TOP    → 📈 유지·익절 준비
    # 비중 ≥40% + NORMAL → ⚠️  집중 경고
    # 비중 ≤10% + TOP    → ✨ 비중 확대 권고
    # 비중 ≤10% + BOTTOM → 💤 관망 (표시 생략)
    recommendations: list[dict] = []
    for sector, sv in sorted(sectors.items(), key=lambda x: x[1]["weight"], reverse=True):
        w    = sv["weight"]
        rank = rank_map.get(sector, "NORMAL")
        tks  = ", ".join(pf_name_map.get(t, t) or t for t in sv["tickers"])

        if w >= 40 and rank == "BOTTOM":
            recommendations.append({
                "type": "reduce", "icon": "🚨", "sector": sector, "weight": w, "tickers": tks,
                "message": "시장 자금이 이탈하는 섹터에 자산이 과도하게 묶여 있습니다. 리스크 관리를 위해 일부 비중을 줄여 현금을 확보하세요.",
            })
        elif w >= 40 and rank == "TOP":
            recommendations.append({
                "type": "hold", "icon": "📈", "sector": sector, "weight": w, "tickers": tks,
                "message": "시장을 주도하는 섹터를 잘 선점하셨습니다. 편중도가 높으니 추세가 꺾이기 전까지 유지하되(트레일링 스톱), 신규 매수는 자제하세요.",
            })
        elif w >= 40:
            recommendations.append({
                "type": "watch", "icon": "⚠️", "sector": sector, "weight": w, "tickers": tks,
                "message": "섹터 집중도가 높습니다. 시장 흐름 변화 시 리스크가 커질 수 있으니 분산을 고려하세요.",
            })
        elif w <= 10 and rank == "TOP":
            recommendations.append({
                "type": "add", "icon": "✨", "sector": sector, "weight": w, "tickers": tks,
                "message": f"시장 주도 섹터이나 비중이 {w:.1f}%로 낮습니다. 10~15% 수준으로 비중 확대를 고려해보세요.",
            })

    # ── 미보유 TOP 섹터 탐색 ────────────────────────────────────────────
    held_sectors = set(sectors.keys())
    missing_top: list[dict] = [
        {
            "sector":    s["sector"],
            "name":      s["name"],
            "score":     s["score"],
            "return_5d": s["return_5d"],
        }
        for s in sector_scores
        if s["rank"] == "TOP" and s["sector"] not in held_sectors
    ]

    # ── 수익 확정 권고 (+15% 이상) ───────────────────────────────────────
    profit_take = sorted(
        [
            {
                "ticker":  iv["ticker"],
                "name":    pf_name_map.get(iv["ticker"], iv["ticker"]),
                "pnl_pct": round(iv["pnl_pct"], 1),
                "reason":  f"수익률 +{iv['pnl_pct']:.1f}% 도달 — 분할 수익 확정 권고",
            }
            for iv in item_values
            if iv["pnl_pct"] >= 15
        ],
        key=lambda x: x["pnl_pct"],
        reverse=True,
    )

    return {
        "hhi":             hhi,
        "is_concentrated": is_concentrated,
        "market_status":   market_status,
        "recommendations": recommendations,
        "missing_top":     missing_top,
        "profit_take":     profit_take,
        "sector_scores":   sector_scores,
    }
