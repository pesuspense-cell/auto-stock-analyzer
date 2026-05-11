"""
portfolio_optimizer.py - 포트폴리오 섹터 분석 및 리밸런싱 가이드

[기능]
  1. classify_sectors   - 종목별 섹터 분류 + 비중 계산
  2. scan_sector_etfs   - 섹터 ETF 5일 수익률 스캔 (시장 주도주 탐지)
  3. build_rebalancing_guide - 리밸런싱 제안 생성
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("portfolio_optimizer")

# 인버스·레버리지·광범위 지수 제외 (섹터 순수 분석 대상만)
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


# ─── 2. 섹터 ETF 5일 수익률 스캔 ──────────────────────────────────────────────
def scan_sector_etfs() -> list[dict]:
    """
    _ETF_PORTFOLIO_MAP 등록 섹터 ETF의 최근 5거래일 수익률 계산.
    인버스·레버리지·광범위 지수는 제외.

    반환: [{ticker, code, name, sector, return_5d}, ...] — return_5d 내림차순
    """
    import pandas as pd
    import yfinance as yf
    from stock_ai import _ETF_PORTFOLIO_MAP  # 지연 import

    ticker_info = {
        f"{code}.KS": info
        for code, info in _ETF_PORTFOLIO_MAP.items()
        if info.get("sector") not in _EXCLUDE_SECTORS
    }
    if not ticker_info:
        return []

    try:
        raw = yf.download(
            list(ticker_info.keys()),
            period="10d",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        results: list[dict] = []
        for t_ks, info in ticker_info.items():
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    closes = raw["Close"][t_ks].dropna()
                else:
                    closes = raw["Close"].dropna()
                if len(closes) < 2:
                    continue
                ret5 = float((closes.iloc[-1] / closes.iloc[0] - 1) * 100)
                results.append({
                    "ticker":    t_ks,
                    "code":      t_ks.replace(".KS", ""),
                    "name":      info["name"],
                    "sector":    info["sector"],
                    "return_5d": round(ret5, 2),
                })
            except Exception:
                continue
        return sorted(results, key=lambda x: x["return_5d"], reverse=True)
    except Exception as exc:
        logger.warning("섹터 ETF 스캔 실패: %s", exc)
        return []


# ─── 3. 리밸런싱 가이드 생성 ──────────────────────────────────────────────────
def build_rebalancing_guide(
    sector_data: dict,
    etf_scan: list[dict],
    pf_name_map: dict[str, str],
) -> dict:
    """
    섹터 분석 + ETF 스캔 결과를 바탕으로 4가지 리밸런싱 제안 생성.

    반환:
      concentration_warnings : list[{sector, weight, tickers}]   30% 초과
      new_candidates         : list[{name, sector, return_5d, reason}]
      profit_take            : list[{ticker, name, pnl_pct, reason}]  +15% 이상
      add_buy                : list[{ticker, name, sector, reason}]   비중 낮은 강세 섹터
    """
    sectors     = sector_data.get("sectors", {})
    item_values = sector_data.get("item_values", [])

    # ── 섹터 집중 경고 (>30%) ──────────────────────────────────────────────
    concentration_warnings = sorted(
        [
            {"sector": s, "weight": v["weight"], "tickers": v["tickers"]}
            for s, v in sectors.items()
            if v["weight"] > 30
        ],
        key=lambda x: x["weight"],
        reverse=True,
    )

    # ── 신규 편입 후보 (미보유 섹터 강세 ETF 상위 2개) ─────────────────────
    held_sectors  = set(sectors.keys())
    new_candidates: list[dict] = []
    for etf in etf_scan:
        if etf["return_5d"] <= 0:
            continue
        if etf["sector"] not in held_sectors:
            new_candidates.append({
                "name":      etf["name"],
                "sector":    etf["sector"],
                "return_5d": etf["return_5d"],
                "reason":    f"5일 수익률 +{etf['return_5d']:.1f}% — 미보유 섹터 편입 검토",
            })
        if len(new_candidates) >= 2:
            break
    # 미보유 섹터가 없으면 수익률 상위 보유 섹터로 보완
    if len(new_candidates) < 2:
        for etf in etf_scan[:3]:
            if etf["return_5d"] > 1.0 and not any(
                c["sector"] == etf["sector"] for c in new_candidates
            ):
                new_candidates.append({
                    "name":      etf["name"],
                    "sector":    etf["sector"],
                    "return_5d": etf["return_5d"],
                    "reason":    f"5일 수익률 +{etf['return_5d']:.1f}% 섹터 강세 — 비중 확대 검토",
                })
            if len(new_candidates) >= 2:
                break

    # ── 수익 확정 권고 (+15% 이상) ─────────────────────────────────────────
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

    # ── 추가 매수 권고 (강세 섹터 & 포트폴리오 비중 <10%) ────────────────
    add_buy: list[dict] = []
    for etf in etf_scan[:4]:
        sector = etf["sector"]
        weight = sectors.get(sector, {}).get("weight", 0.0)
        if etf["return_5d"] > 1.0 and weight < 10:
            wlabel = "미보유 섹터" if weight == 0 else f"비중 {weight:.1f}%"
            add_buy.append({
                "ticker": etf["ticker"],
                "name":   etf["name"],
                "sector": sector,
                "reason": f"섹터 강세 +{etf['return_5d']:.1f}% / {wlabel} — 비중 확대 검토",
            })
        if len(add_buy) >= 2:
            break

    return {
        "concentration_warnings": concentration_warnings,
        "new_candidates":         new_candidates,
        "profit_take":            profit_take,
        "add_buy":                add_buy,
    }
