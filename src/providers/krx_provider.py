"""
src/providers/krx_provider.py — KRX Open API / Naver Finance / FDR 기반 Provider

fetch(ticker) : KRX 상장 종목 전용. 비상장(미국 등) → None 반환.
  반환 필드: per, pbr, eps_ttm, bps, div_yield, market_cap,
             short_name, sector, w52_high, w52_low

유틸리티 (BaseProvider.fetch 와 별개):
  get_stock_list(market)  → KRX 전체 종목 이름→티커 매핑
  get_top_stocks(market)  → 시가총액 상위 n개
  get_etf_list()          → 국내 ETF 전체 목록
"""
from __future__ import annotations

import asyncio
import re
from datetime import datetime, timedelta
from typing import Any, Optional

from .base import BaseProvider

# ── KRX 엔드포인트 & 헤더 ─────────────────────────────────────────────────────
_KRX_BASE = "https://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"
_KRX_HDR  = {
    "Content-Type":     "application/x-www-form-urlencoded; charset=UTF-8",
    "Referer":          "https://data.krx.co.kr/",
    "User-Agent":       (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept":           "application/json, text/javascript, */*; q=0.01",
    "X-Requested-With": "XMLHttpRequest",
}
_NAVER_HDR = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Referer": "https://finance.naver.com/",
}


def _to_float(val: Any, default: Optional[float] = None) -> Optional[float]:
    if val is None:
        return default
    try:
        return float(str(val).replace(",", "").strip())
    except (ValueError, TypeError):
        return default


class KRXProvider(BaseProvider):
    """
    KRX Open API (data.krx.co.kr) + Naver Finance + FinanceDataReader
    기반 펀더멘털 Provider.

    KRX 비상장(해외 주식 등)에는 None 반환.
    httpx 미설치 시 None 반환 (graceful degradation).
    """

    @staticmethod
    def _is_krx(ticker: str) -> bool:
        return ticker.endswith(".KS") or ticker.endswith(".KQ")

    @staticmethod
    def _code(ticker: str) -> str:
        return ticker.split(".")[0].zfill(6)

    # ── KRX Open API — PER/PBR/EPS/BPS ──────────────────────────────────────

    async def _fetch_per_pbr(self, code6: str, client) -> Optional[dict]:
        """MDCSTAT03501 — 전 종목 PER/PBR/EPS/BPS. 최근 5거래일 시도."""
        for delta in range(5):
            d = (datetime.now() - timedelta(days=delta)).strftime("%Y%m%d")
            for mkt in ("STK", "KSQ", "ALL"):
                try:
                    resp = await client.post(
                        _KRX_BASE,
                        data={
                            "bld":         "dbms/MDC/STAT/standard/MDCSTAT03501",
                            "locale":      "ko_KR",
                            "mktId":       mkt,
                            "trdDd":       d,
                            "money":       "1",
                            "csvxls_isNo": "false",
                        },
                        headers=_KRX_HDR,
                        timeout=10.0,
                    )
                    resp.raise_for_status()
                    for r in resp.json().get("output", []):
                        if str(r.get("ISU_SRT_CD", "")).strip() == code6:
                            per = _to_float(r.get("PER"))
                            pbr = _to_float(r.get("PBR"))
                            if per or pbr:
                                result: dict = {}
                                if per: result["per"]     = per
                                if pbr: result["pbr"]     = pbr
                                eps = _to_float(str(r.get("EPS", "")).replace(",", ""))
                                bps = _to_float(str(r.get("BPS", "")).replace(",", ""))
                                if eps: result["eps_ttm"] = eps
                                if bps: result["bps"]     = bps
                                self.logger.info(
                                    "KRX PER/PBR 수집 성공: %s (날짜=%s mkt=%s)", code6, d, mkt
                                )
                                return result
                except Exception as e:
                    self.logger.debug(
                        "KRX PER/PBR 오류 (code=%s delta=%d mkt=%s): %s", code6, delta, mkt, e
                    )
                    break  # 해당 날짜 실패 → 다음 날짜로
        self.logger.warning("KRX PER/PBR 최종 실패: %s", code6)
        return None

    async def _fetch_market_cap(self, code6: str, client) -> Optional[dict]:
        """MDCSTAT01901 — 시가총액·상장주식수·회사명·섹터."""
        today = datetime.now().strftime("%Y%m%d")
        try:
            resp = await client.post(
                _KRX_BASE,
                data={
                    "bld":         "dbms/MDC/STAT/standard/MDCSTAT01901",
                    "locale":      "ko_KR",
                    "mktId":       "ALL",
                    "trdDd":       today,
                    "money":       "1",
                    "csvxls_isNo": "false",
                },
                headers=_KRX_HDR,
                timeout=10.0,
            )
            resp.raise_for_status()
            for r in resp.json().get("output", []):
                if str(r.get("ISU_SRT_CD", "")).strip() == code6:
                    mcap = _to_float(str(r.get("MKTCAP", "")).replace(",", ""))
                    self.logger.info("KRX 시가총액 수집 성공: %s", code6)
                    return {
                        "market_cap": mcap,
                        "short_name": r.get("ISU_ABBRV", ""),
                        "sector":     r.get("IDX_IND_NM", ""),
                    }
        except Exception as e:
            self.logger.debug("KRX 시가총액 오류 [%s]: %s", code6, e)
        return None

    # ── Naver Finance — PER/PBR/EPS/BPS/배당 ────────────────────────────────

    async def _fetch_naver(self, code6: str, client) -> dict:
        """per_table 스크래핑. BeautifulSoup 미설치 시 빈 dict."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            return {}

        url = f"https://finance.naver.com/item/main.naver?code={code6}"
        try:
            resp = await client.get(url, headers=_NAVER_HDR, timeout=8.0)
            resp.raise_for_status()
            html = resp.content.decode("euc-kr", errors="replace")
            soup = BeautifulSoup(html, "html.parser")
            table = soup.find("table", {"class": "per_table"})
            if not table:
                return {}

            def _pf(text: str) -> Optional[float]:
                m = re.search(r"-?[\d,]+\.?\d*", text.strip())
                return float(m.group().replace(",", "")) if m else None

            rows_data, row_values = [], []
            for tr in table.find_all("tr"):
                td = tr.find("td")
                if td:
                    parts = [
                        p.strip() for p in td.text.split("\n")
                        if p.strip() and p.strip() != "l"
                    ]
                    rows_data.append(_pf(parts[0]) if parts else None)
                    row_values.append(_pf(parts[-1]) if len(parts) > 1 else None)

            per     = rows_data[0] if len(rows_data) > 0 else None
            pbr     = rows_data[2] if len(rows_data) > 2 else None
            div     = rows_data[3] if len(rows_data) > 3 else None
            eps_ttm = row_values[0] if (row_values and row_values[0]) else None
            bps     = row_values[2] if (len(row_values) > 2 and row_values[2]) else None

            result: dict = {}
            if per:     result["per"]      = per
            if pbr:     result["pbr"]      = pbr
            if eps_ttm: result["eps_ttm"]  = eps_ttm
            if bps:     result["bps"]      = bps
            if div:     result["div_yield"] = div / 100

            if result:
                self.logger.info("Naver Finance 수집 성공: %s", code6)
            return result
        except Exception as e:
            self.logger.warning("Naver Finance 실패 [%s]: %s", code6, e)
            return {}

    # ── FinanceDataReader — 52주 고저 보완 ───────────────────────────────────

    async def _fetch_fdr(self, ticker: str) -> dict:
        """FDR 52주 고저. FinanceDataReader 미설치 시 빈 dict."""
        loop = asyncio.get_event_loop()
        code = ticker.split(".")[0]
        try:
            import FinanceDataReader as fdr  # noqa: F401
            start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
            end   = datetime.now().strftime("%Y-%m-%d")
            df = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: fdr.DataReader(code, start, end)),
                timeout=10.0,
            )
            if df is None or df.empty:
                return {}
            result: dict = {}
            if "High" in df.columns:
                result["w52_high"] = float(df["High"].max())
            if "Low" in df.columns:
                result["w52_low"] = float(df["Low"].min())
            self.logger.info("FDR 52주 고저 수집 성공: %s", code)
            return result
        except asyncio.TimeoutError:
            self.logger.warning("FDR 타임아웃: %s", code)
        except ImportError:
            pass
        except Exception as e:
            self.logger.debug("FDR 실패 [%s]: %s", code, e)
        return {}

    # ── BaseProvider.fetch 구현 ──────────────────────────────────────────────

    async def fetch(self, ticker: str) -> Optional[dict]:
        """
        KRX 종목 전용. 해외 종목은 None 반환.
        KRX API + Naver Finance + FDR 병렬 호출 → 우선순위 병합.
        """
        if not self._is_krx(ticker):
            return None  # 해외 종목 — 조용히 스킵
        code6 = self._code(ticker)

        try:
            import httpx
        except ImportError:
            self.logger.warning("httpx 미설치 — KRXProvider 비활성화")
            return None

        async with httpx.AsyncClient(follow_redirects=True) as client:
            tasks = {
                "krx":   asyncio.create_task(self._fetch_per_pbr(code6, client)),
                "mcap":  asyncio.create_task(self._fetch_market_cap(code6, client)),
                "naver": asyncio.create_task(self._fetch_naver(code6, client)),
                "fdr":   asyncio.create_task(self._fetch_fdr(ticker)),
            }
            done, _ = await asyncio.wait(tasks.values(), timeout=12.0)

            results: dict[str, dict] = {}
            for name, task in tasks.items():
                try:
                    results[name] = task.result() or {}
                except Exception as e:
                    self.logger.debug("태스크 예외 [%s/%s]: %s", code6, name, e)
                    results[name] = {}
                    task.cancel()

        merged: dict = {}

        # PER/PBR/EPS/BPS: KRX API → Naver 순으로 보완
        for src in ("krx", "naver"):
            for k in ("per", "pbr", "eps_ttm", "bps", "div_yield"):
                if merged.get(k) is None and results[src].get(k) is not None:
                    merged[k] = results[src][k]

        # 시가총액·회사명·섹터
        for k in ("market_cap", "short_name", "sector"):
            v = results["mcap"].get(k)
            if v is not None:
                merged[k] = v

        # FDR 52주 고저 (yfinance 미수집 보완용)
        for k in ("w52_high", "w52_low"):
            if results["fdr"].get(k) is not None:
                merged[k] = results["fdr"][k]

        if not any(merged.get(k) for k in ("per", "pbr", "market_cap")):
            self.logger.warning("KRXProvider 유효 데이터 없음: %s", ticker)
            return None

        merged["source_krx"] = True
        return merged

    # ── 종목 목록 유틸리티 ───────────────────────────────────────────────────

    @staticmethod
    def get_stock_list(market: str) -> dict:
        """
        KOSPI / KOSDAQ 전체 종목 이름→티커 매핑.
        반환 형태: {"삼성전자 (005930)": "005930.KS", ...}
        실패 시 빈 dict (예외 미발생).
        """
        suffix = "KS" if market == "KOSPI" else "KQ"
        try:
            import FinanceDataReader as fdr
            df = fdr.StockListing(market)
            if df is None or df.empty:
                return {}
            df["Code"] = df["Code"].astype(str).str.strip()
            df["Name"] = df["Name"].astype(str).str.strip()
            mask = (
                df["Name"].ne("") &
                df["Code"].str.len().eq(6) &
                df["Code"].str.isdigit()
            )
            df = df[mask]
            return dict(zip(
                df["Name"] + " (" + df["Code"] + ")",
                df["Code"] + "." + suffix,
            ))
        except Exception:
            return {}

    @staticmethod
    def get_top_stocks(market: str, n: int = 500) -> dict:
        """
        시가총액 상위 n개 종목 이름→티커 매핑.
        반환 형태: {"삼성전자": "005930.KS", ...}
        """
        suffix = "KS" if market == "KOSPI" else "KQ"
        try:
            import FinanceDataReader as fdr
            df = fdr.StockListing(market)
            df = df.dropna(subset=["Name", "Code", "Marcap"])
            df = df[df["Marcap"] > 0].sort_values("Marcap", ascending=False).head(n)
            result: dict = {}
            for _, row in df.iterrows():
                code = str(row["Code"]).strip().zfill(6)
                name = str(row["Name"]).strip()
                if name and code:
                    result[name] = f"{code}.{suffix}"
            return result
        except Exception:
            return {}

    @staticmethod
    def get_etf_list() -> dict:
        """
        국내 ETF 전체 목록 이름→티커 매핑.
        반환 형태: {"KODEX 200 (069500)": "069500.KS", ...}
        """
        try:
            import FinanceDataReader as fdr
            df = fdr.StockListing("ETF/KR")
            if df is None or df.empty:
                return {}
            code_col = "Symbol" if "Symbol" in df.columns else "Code"
            df[code_col] = df[code_col].astype(str).str.strip().str.zfill(6)
            df["Name"]   = df["Name"].astype(str).str.strip()
            mask = (
                df["Name"].ne("") &
                df[code_col].str.len().eq(6) &
                df[code_col].str.isdigit()
            )
            df = df[mask]
            return {
                f"{row['Name']} ({row[code_col]})": f"{row[code_col]}.KS"
                for _, row in df.iterrows()
            }
        except Exception:
            return {}

    @staticmethod
    def get_us_stock_list() -> dict:
        """
        S&P500 + NASDAQ 통합 이름→티커 매핑 (이름 검색용).
        반환 형태: {"Apple (AAPL) [S&P500]": "AAPL", ...}
        """
        try:
            import FinanceDataReader as fdr
            combined: dict[str, str] = {}
            for market_tag, key in [("S&P500", "S&P500"), ("NASDAQ", "NASDAQ")]:
                try:
                    df = fdr.StockListing(key).dropna(subset=["Name", "Symbol"])
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
            return combined
        except Exception:
            return {}
