"""
src/providers/dart_provider.py — DART OpenAPI + SEC EDGAR Provider

fetch(ticker) 반환 필드 (KRX 전용):
  revenue, operating_income, net_income, order_backlog, year

fetch_insider_trades(ticker, days) 반환: pd.DataFrame (미국 주식 전용)
  컬럼: 공시일, 내용, 링크
"""
from __future__ import annotations

import asyncio
import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from .base import BaseProvider

logger = logging.getLogger("provider.DARTProvider")


class DARTProvider(BaseProvider):
    """
    DART OpenAPI 재무 + SEC EDGAR Form 4 내부자 거래 Provider.

    API 키 탐색 순서:
      1. 환경변수 DART_API_KEY
      2. st.secrets["DART_API_KEY"]
      3. st.session_state["dart_api_key"]
    """

    _SEC_HEADERS = {"User-Agent": "AutoStockAnalyzer contact@example.com"}

    def __init__(self) -> None:
        super().__init__()
        self._key = self._resolve_key()

    # ── API 키 탐색 ─────────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_key() -> Optional[str]:
        import os
        key = os.environ.get("DART_API_KEY")
        if key:
            return key
        try:
            import streamlit as st
            key = st.secrets.get("DART_API_KEY", "") or st.session_state.get(
                "dart_api_key", ""
            )
        except Exception:
            pass
        return key or None

    @property
    def available(self) -> bool:
        return bool(self._key)

    # ── DART 동기 수집 — executor 오프로드용 ────────────────────────────────────

    def _fetch_sync(self, stock_code: str) -> dict:
        """OpenDartReader 동기 조회. 실패 시 빈 dict."""
        try:
            from fundamental_db import get_dart_financials
            return get_dart_financials(f"{stock_code}.KS", self._key)
        except Exception as e:
            logger.debug("DART OpenDartReader 조회 실패 [%s]: %s", stock_code, e)
        return {}

    # ── BaseProvider.fetch 구현 ──────────────────────────────────────────────────

    async def fetch(self, ticker: str) -> Optional[dict]:
        """
        DART 연간 재무 비동기 조회 (KRX 전용).
        - KRX 티커(.KS/.KQ)가 아니면 None 즉시 반환.
        - DART_API_KEY 미설정이면 None 반환.
        """
        is_krx = ticker.endswith(".KS") or ticker.endswith(".KQ")
        if not is_krx:
            return None

        if not self.available:
            self.logger.debug("DART API 키 미설정, 건너뜀: %s", ticker)
            return None

        code6 = ticker.split(".")[0]
        loop = asyncio.get_event_loop()
        try:
            data = await asyncio.wait_for(
                loop.run_in_executor(None, self._fetch_sync, code6),
                timeout=12.0,
            )
            if data:
                self.logger.info("DART 재무 수집 성공: %s", ticker)
                return data
            self.logger.debug("DART 유효 데이터 없음: %s", ticker)
            return None
        except asyncio.TimeoutError:
            self.logger.warning("DART 조회 타임아웃 (12s): %s", ticker)
        except Exception as e:
            self.logger.warning("DART 조회 실패 [%s]: %s", ticker, e)
        return None

    # ── SEC EDGAR 내부자 거래 ────────────────────────────────────────────────────

    @staticmethod
    def _get_sec_cik(ticker: str) -> Optional[str]:
        """SEC company_tickers.json에서 CIK 번호 조회."""
        try:
            import requests
            resp = requests.get(
                "https://www.sec.gov/files/company_tickers.json",
                headers=DARTProvider._SEC_HEADERS,
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

    def fetch_insider_trades(self, ticker: str, days: int = 90) -> pd.DataFrame:
        """
        SEC EDGAR Form 4 내부자 거래 공시 조회 (미국 주식 전용).
        KRX 티커(.KS/.KQ 포함 '.'이 있는 경우)는 빈 DataFrame 반환.
        """
        if "." in ticker:
            return pd.DataFrame()

        try:
            import requests
            headers = self._SEC_HEADERS

            cik = self._get_sec_cik(ticker)
            if not cik:
                self.logger.debug("SEC CIK 조회 실패: %s", ticker)
                return pd.DataFrame()

            atom_url = (
                f"https://www.sec.gov/cgi-bin/browse-edgar"
                f"?action=getcompany&CIK={cik}&type=4&dateb=&owner=include&count=20&output=atom"
            )
            resp = requests.get(atom_url, headers=headers, timeout=15)
            if resp.status_code != 200:
                return pd.DataFrame()

            ns     = {"atom": "http://www.w3.org/2005/Atom"}
            root   = ET.fromstring(resp.content)
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

        except Exception as e:
            self.logger.warning("SEC EDGAR 조회 실패 [%s]: %s", ticker, e)
            return pd.DataFrame()
