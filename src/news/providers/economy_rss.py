"""
src/news/providers/economy_rss.py — 매일경제·한국경제 RSS 공급자

source_weight = 1.0 (경제 전문지)

RSS 파싱: feedparser 우선, 미설치 시 xml.etree.ElementTree 폴백
피드 장애 시 해당 피드만 건너뜀 (시스템 중단 없음)
"""
from __future__ import annotations

import asyncio
import xml.etree.ElementTree as ET
from typing import Optional

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

try:
    import feedparser  # type: ignore
    HAS_FEEDPARSER = True
except ImportError:
    HAS_FEEDPARSER = False

from .base import BaseNewsProvider

# 매일경제·한국경제 RSS 피드 목록 (url, 출처명)
_DEFAULT_FEEDS: list[tuple[str, str]] = [
    ("https://www.mk.co.kr/rss/40300001/", "매일경제"),
    ("https://rss.hankyung.com/economy.xml",  "한국경제"),
]

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; EconomyRSSBot/1.0)",
    "Accept":     "application/rss+xml, application/xml, text/xml",
}

# Atom 네임스페이스
_NS_ATOM = "http://www.w3.org/2005/Atom"


class EconomyRSSProvider(BaseNewsProvider):
    """
    매일경제·한국경제 RSS 피드 수집기.

    - asyncio.gather()로 모든 피드를 병렬 수집
    - company_name이 있으면 제목·요약 기준 키워드 필터 적용
    - 피드 파싱 실패 시 해당 피드만 건너뜀
    """

    source_type   = "economy_rss"
    source_weight = 1.0

    def __init__(
        self,
        feeds: list[tuple[str, str]] | None = None,
    ) -> None:
        super().__init__()
        self._feeds = feeds or _DEFAULT_FEEDS

    async def fetch(
        self,
        ticker: str,
        company_name: str = "",
        max_items: int = 12,
        **kwargs,
    ) -> list[dict]:
        if not HAS_HTTPX:
            self.logger.warning("httpx 미설치 — EconomyRSS 소스 건너뜀")
            return []

        try:
            async with httpx.AsyncClient(
                headers=_HEADERS,
                follow_redirects=True,
                timeout=10.0,
            ) as client:
                tasks = [
                    self._fetch_feed(client, url, name)
                    for url, name in self._feeds
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as exc:
            self.logger.warning("EconomyRSSProvider 네트워크 오류: %s", exc)
            return []

        all_items: list[dict] = []
        for result in results:
            if isinstance(result, Exception):
                self.logger.warning("RSS 피드 수집 실패: %s", result)
                continue
            all_items.extend(result)

        # 종목명 관련성 필터 (company_name 있을 때만)
        if company_name:
            filtered = [
                it for it in all_items
                if (
                    company_name in it.get("title", "")
                    or company_name in it.get("content", "")
                )
            ]
            # 필터 결과 없으면 전체 반환 (종목명 변형·약칭 가능성)
            if filtered:
                all_items = filtered

        return all_items[:max_items]

    async def _fetch_feed(
        self,
        client: "httpx.AsyncClient",
        url: str,
        publisher: str,
    ) -> list[dict]:
        try:
            resp = await client.get(url)
        except Exception as exc:
            self.logger.warning("RSS 요청 실패 [%s]: %s", publisher, exc)
            return []

        if resp.status_code != 200:
            self.logger.warning(
                "RSS 응답 %d [%s]", resp.status_code, publisher
            )
            return []

        return self._parse(resp.text, publisher)

    # ── 파싱 ─────────────────────────────────────────────────────────────────

    def _parse(self, xml_text: str, publisher: str) -> list[dict]:
        """feedparser 우선, 실패 시 ElementTree 폴백."""
        if HAS_FEEDPARSER:
            return self._parse_feedparser(xml_text, publisher)
        return self._parse_etree(xml_text, publisher)

    def _parse_feedparser(self, xml_text: str, publisher: str) -> list[dict]:
        try:
            feed = feedparser.parse(xml_text)
            items: list[dict] = []
            for entry in feed.entries:
                title   = getattr(entry, "title",   "").strip()
                link    = getattr(entry, "link",    "")
                pub     = getattr(entry, "published", "") or getattr(entry, "updated", "")
                summary = getattr(entry, "summary", "")
                if not title:
                    continue
                items.append(
                    self._make_item(
                        title=title,
                        link=link,
                        publisher=publisher,
                        pub_date=pub,
                        content=summary[:500],
                    )
                )
            return items
        except Exception as exc:
            self.logger.warning(
                "feedparser 파싱 실패 [%s] — ETtree 폴백: %s", publisher, exc
            )
            return self._parse_etree(xml_text, publisher)

    def _parse_etree(self, xml_text: str, publisher: str) -> list[dict]:
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            self.logger.warning("XML 파싱 오류 [%s]: %s", publisher, exc)
            return []

        items: list[dict] = []

        # RSS 2.0
        for item_el in root.iter("item"):
            title    = (item_el.findtext("title")       or "").strip()
            link     = (item_el.findtext("link")        or "").strip()
            pub_date = (item_el.findtext("pubDate")     or "").strip()
            desc     = (item_el.findtext("description") or "").strip()
            if not title:
                continue
            items.append(
                self._make_item(
                    title=title,
                    link=link,
                    publisher=publisher,
                    pub_date=pub_date,
                    content=desc[:500],
                )
            )

        # Atom 피드 폴백
        if not items:
            for entry in root.findall(f"{{{_NS_ATOM}}}entry"):
                title    = (entry.findtext(f"{{{_NS_ATOM}}}title")   or "").strip()
                link_el  = entry.find(f"{{{_NS_ATOM}}}link")
                link     = link_el.get("href", "") if link_el is not None else ""
                updated  = (entry.findtext(f"{{{_NS_ATOM}}}updated") or "").strip()
                summary  = (entry.findtext(f"{{{_NS_ATOM}}}summary") or "").strip()
                if not title:
                    continue
                items.append(
                    self._make_item(
                        title=title,
                        link=link,
                        publisher=publisher,
                        pub_date=updated,
                        content=summary[:500],
                    )
                )

        return items
