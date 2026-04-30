"""
src/providers/yahoo_provider.py — yfinance 기반 펀더멘털 수집 Provider

fetch(ticker) 반환 필드:
  per, pbr, psr, roe, roe_history, eps_history, debt_equity,
  revenue_growth, earnings_growth, operating_margins,
  w52_high, w52_low, market_cap, total_revenue, operating_income,
  net_income, free_cashflow, ocf, buyback_amount, div_paid_amount,
  shareholder_yield, eps_ttm, forward_pe, div_yield, sector, industry, short_name
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

import pandas as pd

from .base import BaseProvider

logger = logging.getLogger("provider.YahooProvider")


def _coalesce(*vals):
    for v in vals:
        if v is not None:
            return v
    return None


class YahooProvider(BaseProvider):
    """yfinance 기반 펀더멘털 데이터 수집 Provider."""

    # ── 동기 수집 — executor 오프로드용 ──────────────────────────────────────

    @staticmethod
    def _fetch_sync(ticker: str) -> dict:
        """yfinance로 전체 펀더멘털을 동기 수집. 실패 시 빈 dict."""
        try:
            import yfinance as yf
        except ImportError:
            logger.warning("yfinance 미설치")
            return {}

        try:
            t    = yf.Ticker(ticker)
            info = t.info
        except Exception as e:
            logger.warning("yfinance Ticker 생성 실패 [%s]: %s", ticker, e)
            return {}

        # ── 현재가 ───────────────────────────────────────────────────────────
        price = _coalesce(
            info.get("currentPrice"),
            info.get("regularMarketPrice"),
            info.get("previousClose"),
        )
        if price is None:
            try:
                price = t.fast_info.last_price
            except Exception:
                pass

        # ── 시가총액 ─────────────────────────────────────────────────────────
        market_cap = info.get("marketCap")
        if market_cap is None:
            try:
                market_cap = t.fast_info.market_cap
            except Exception:
                pass

        # ── 52주 고저 ────────────────────────────────────────────────────────
        w52_high = info.get("fiftyTwoWeekHigh")
        w52_low  = info.get("fiftyTwoWeekLow")
        if w52_high is None:
            try: w52_high = t.fast_info.year_high
            except Exception: pass
        if w52_low is None:
            try: w52_low = t.fast_info.year_low
            except Exception: pass

        eps_ttm   = info.get("trailingEps")
        book_val  = info.get("bookValue")
        total_rev = info.get("totalRevenue")

        per = info.get("trailingPE")
        if per is None and price and eps_ttm and eps_ttm > 0:
            per = round(price / eps_ttm, 2)

        pbr = info.get("priceToBook")
        if pbr is None and price and book_val and book_val > 0:
            pbr = round(price / book_val, 2)

        psr = None
        if market_cap and total_rev and total_rev > 0:
            psr = round(market_cap / total_rev, 2)

        # ── 재무제표 상세 ────────────────────────────────────────────────────
        operating_income     = None
        net_income           = None
        ocf                  = info.get("operatingCashflow")
        roe_history: list    = []
        roe_from_fin         = None
        revenue_growth_calc  = None
        earnings_growth_calc = None
        op_margins_calc      = None
        fcf_from_cf          = None
        eps_history: list    = []

        try:
            fin = t.financials
            bs  = t.balance_sheet
            cf  = t.cashflow

            if fin is not None and not fin.empty:
                for lbl in ["Operating Income", "EBIT"]:
                    if lbl in fin.index:
                        operating_income = float(fin.loc[lbl].iloc[0]); break

                for lbl in ["Net Income", "Net Income Common Stockholders",
                            "Net Income From Continuing Operation Net Minority Interest"]:
                    if lbl in fin.index:
                        net_income = float(fin.loc[lbl].iloc[0]); break

                if cf is not None and not cf.empty:
                    for lbl in ["Operating Cash Flow", "Total Cash From Operating Activities"]:
                        if lbl in cf.index:
                            ocf = float(cf.loc[lbl].iloc[0]); break

                if total_rev is None:
                    for lbl in ["Total Revenue", "Operating Revenue"]:
                        if lbl in fin.index:
                            v = fin.loc[lbl].iloc[0]
                            if pd.notna(v):
                                total_rev = float(v); break

                # 매출 성장률 (YoY)
                for lbl in ["Total Revenue", "Operating Revenue"]:
                    if lbl in fin.index:
                        rv = fin.loc[lbl].dropna()
                        if len(rv) >= 2:
                            r0, r1 = float(rv.iloc[0]), float(rv.iloc[1])
                            if r1 != 0 and pd.notna(r0) and pd.notna(r1):
                                revenue_growth_calc = round((r0 - r1) / abs(r1), 6)
                        break

                # 순이익 성장률 (YoY)
                for lbl in ["Net Income", "Net Income Common Stockholders",
                            "Net Income From Continuing Operation Net Minority Interest"]:
                    if lbl in fin.index:
                        nv = fin.loc[lbl].dropna()
                        if len(nv) >= 2:
                            n0, n1 = float(nv.iloc[0]), float(nv.iloc[1])
                            if n1 != 0 and pd.notna(n0) and pd.notna(n1):
                                earnings_growth_calc = round((n0 - n1) / abs(n1), 6)
                        break

                if operating_income is not None and total_rev and total_rev > 0:
                    op_margins_calc = round(operating_income / total_rev, 6)

                if cf is not None and not cf.empty and "Free Cash Flow" in cf.index:
                    v = cf.loc["Free Cash Flow"].iloc[0]
                    if pd.notna(v):
                        fcf_from_cf = float(v)

                # 단년 ROE 역산
                if net_income is not None and bs is not None and not bs.empty:
                    for lbl in ["Common Stock Equity", "Stockholders Equity",
                                "Total Equity Gross Minority Interest"]:
                        if lbl in bs.index:
                            eq = bs.loc[lbl].iloc[0]
                            if pd.notna(eq) and float(eq) > 0:
                                roe_from_fin = round(net_income / float(eq), 6)
                            break

                # 다년 ROE (최대 4년)
                if bs is not None and not bs.empty:
                    ni_vals = eq_vals = None
                    for lbl in ["Net Income", "Net Income Common Stockholders",
                                "Net Income From Continuing Operation Net Minority Interest"]:
                        if lbl in fin.index:
                            ni_vals = fin.loc[lbl]; break
                    for lbl in ["Common Stock Equity", "Stockholders Equity",
                                "Total Equity Gross Minority Interest"]:
                        if lbl in bs.index:
                            eq_vals = bs.loc[lbl]; break
                    if ni_vals is not None and eq_vals is not None:
                        for col in [c for c in ni_vals.index if c in eq_vals.index][:4]:
                            ni_v, eq_v = ni_vals.get(col), eq_vals.get(col)
                            if (ni_v and eq_v and float(eq_v) > 0
                                    and pd.notna(ni_v) and pd.notna(eq_v)):
                                roe_history.append(round(float(ni_v) / float(eq_v) * 100, 2))

            # EPS 이력 (린치 CAGR용, 과거→현재 순)
            for lbl in ["Diluted EPS", "Basic EPS"]:
                if fin is not None and lbl in fin.index:
                    ep = fin.loc[lbl].dropna()
                    eps_history = [float(v) for v in reversed(ep.values[:4])]; break

            if not eps_history and fin is not None and not fin.empty:
                ni_row = sh_row = None
                for lbl in ["Net Income", "Net Income Common Stockholders",
                            "Net Income From Continuing Operation Net Minority Interest"]:
                    if lbl in fin.index:
                        ni_row = fin.loc[lbl]; break
                for lbl in ["Diluted Average Shares", "Ordinary Shares Number", "Share Issued"]:
                    if fin is not None and lbl in fin.index:
                        sh_row = fin.loc[lbl]; break
                    if bs is not None and lbl in bs.index:
                        sh_row = bs.loc[lbl]; break
                if ni_row is not None and sh_row is not None:
                    raw = []
                    for col in [c for c in ni_row.index if c in sh_row.index][:4]:
                        niv, shv = ni_row.get(col), sh_row.get(col)
                        if (niv is not None and shv is not None and float(shv) > 0
                                and pd.notna(niv) and pd.notna(shv)):
                            raw.append(float(niv) / float(shv))
                    eps_history = list(reversed(raw))

        except Exception as e:
            logger.debug("yfinance 재무제표 파싱 오류 [%s]: %s", ticker, e)

        # ── 자사주·배당 → 주주환원율 ─────────────────────────────────────────
        buyback_amount  = None
        div_paid_amount = None
        shareholder_yield = None
        try:
            cf2 = t.cashflow
            if cf2 is not None and not cf2.empty:
                for lbl in ["Repurchase Of Capital Stock", "Common Stock Payments",
                            "Common Stock Repurchased"]:
                    if lbl in cf2.index:
                        v = cf2.loc[lbl].iloc[0]
                        if pd.notna(v) and float(v) < 0:
                            buyback_amount = abs(float(v)); break
                for lbl in ["Cash Dividends Paid", "Payment Of Dividends"]:
                    if lbl in cf2.index:
                        v = cf2.loc[lbl].iloc[0]
                        if pd.notna(v) and float(v) < 0:
                            div_paid_amount = abs(float(v)); break
            if market_cap and market_cap > 0:
                total_return = (buyback_amount or 0) + (div_paid_amount or 0)
                if total_return > 0:
                    shareholder_yield = round(total_return / market_cap * 100, 2)
        except Exception:
            pass

        if psr is None and market_cap and total_rev and total_rev > 0:
            psr = round(market_cap / total_rev, 2)

        return {
            "per":               per,
            "pbr":               pbr,
            "psr":               psr,
            "roe":               _coalesce(info.get("returnOnEquity"), roe_from_fin),
            "roe_history":       roe_history,
            "eps_history":       eps_history,
            "debt_equity":       info.get("debtToEquity"),
            "revenue_growth":    _coalesce(info.get("revenueGrowth"),  revenue_growth_calc),
            "earnings_growth":   _coalesce(info.get("earningsGrowth"), earnings_growth_calc),
            "operating_margins": _coalesce(info.get("operatingMargins"), op_margins_calc),
            "w52_high":          w52_high,
            "w52_low":           w52_low,
            "market_cap":        market_cap,
            "total_revenue":     total_rev,
            "operating_income":  operating_income,
            "net_income":        net_income,
            "free_cashflow":     _coalesce(info.get("freeCashflow"), fcf_from_cf),
            "ocf":               ocf,
            "buyback_amount":    buyback_amount,
            "div_paid_amount":   div_paid_amount,
            "shareholder_yield": shareholder_yield,
            "eps_ttm":           eps_ttm,
            "forward_pe":        info.get("forwardPE"),
            "div_yield":         info.get("dividendYield"),
            "sector":            info.get("sector", "N/A"),
            "industry":          info.get("industry", "N/A"),
            "short_name":        info.get("shortName", ticker),
        }

    # ── BaseProvider.fetch 구현 ───────────────────────────────────────────────

    async def fetch(self, ticker: str) -> Optional[dict]:
        """yfinance 비동기 래퍼. 타임아웃·예외 발생 시 None 반환."""
        loop = asyncio.get_event_loop()
        try:
            data = await asyncio.wait_for(
                loop.run_in_executor(None, self._fetch_sync, ticker),
                timeout=15.0,
            )
            if data and any(data.get(k) is not None
                            for k in ("per", "pbr", "market_cap", "roe", "net_income")):
                self.logger.info("yfinance 수집 성공: %s", ticker)
                return data
            self.logger.warning("yfinance 유효 데이터 없음: %s", ticker)
            return None
        except asyncio.TimeoutError:
            self.logger.warning("yfinance 타임아웃 (15s): %s", ticker)
        except Exception as e:
            self.logger.warning("yfinance 실패 [%s]: %s", ticker, e)
        return None
