"""
src/news/providers/naver.py — 네이버 금융 뉴스 공급자

기존 news_async.py의 _fetch_one() 로직을 Provider 패턴으로 이식.
source_weight = 0.8 (일반 뉴스 포털)
"""
from __future__ import annotations

import asyncio

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

from .base import BaseNewsProvider

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Referer":         "https://finance.naver.com/",
    "Accept-Language": "ko-KR,ko;q=0.9",
}


class NaverNewsProvider(BaseNewsProvider):
    """
    네이버 금융 뉴스 비동기 크롤러.

    - 403/타임아웃 시 최대 2회 재시도
    - httpx 또는 bs4 미설치 시 빈 리스트 반환 (예외 없음)
    """

    source_type   = "naver"
    source_weight = 0.8

    async def fetch(
        self,
        ticker: str,
        company_name: str = "",
        max_items: int = 12,
        **kwargs,
    ) -> list[dict]:
        if not HAS_HTTPX or not HAS_BS4:
            self.logger.warning("httpx 또는 bs4 미설치 — Naver 소스 건너뜀")
            return []

        code = ticker.split(".")[0].strip()
        if not code.isdigit():
            return []

        url = (
            f"https://finance.naver.com/item/news_news.naver"
            f"?code={code}&page=1"
        )

        try:
            async with httpx.AsyncClient(
                headers=_HEADERS,
                follow_redirects=True,
                limits=httpx.Limits(
                    max_connections=5, max_keepalive_connections=3
                ),
            ) as client:
                for attempt in range(3):
                    try:
                        resp = await client.get(url, timeout=8.0)
                    except Exception as exc:
                        if "Timeout" in type(exc).__name__ and attempt < 2:
                            self.logger.warning(
                                "타임아웃 [%s] — %d회 재시도", code, attempt + 1
                            )
                            await asyncio.sleep(1)
                            continue
                        self.logger.warning("크롤링 실패 [%s]: %s", code, exc)
                        return []

                    if resp.status_code == 403 and attempt < 2:
                        self.logger.warning("403 차단 [%s] — 재시도", code)
                        await asyncio.sleep(1)
                        continue
                    if resp.status_code != 200:
                        return []

                    html = resp.content.decode("euc-kr", errors="replace")
                    soup = BeautifulSoup(html, "html.parser")

                    items: list[dict] = []
                    for row in soup.select("table.type5 tr"):
                        title_el = row.select_one("td.title a")
                        info_el  = row.select_one("td.info")
                        date_el  = row.select_one("td.date")
                        if not title_el:
                            continue
                        href = title_el.get("href", "")
                        if href and not href.startswith("http"):
                            href = "https://finance.naver.com" + href
                        items.append(
                            self._make_item(
                                title=title_el.get_text(strip=True),
                                link=href,
                                publisher=(
                                    info_el.get_text(strip=True) if info_el else ""
                                ),
                                pub_date=(
                                    date_el.get_text(strip=True) if date_el else ""
                                ),
                            )
                        )
                        if len(items) >= max_items:
                            break
                    return items
        except Exception as exc:
            self.logger.warning("NaverNewsProvider 예외 [%s]: %s", ticker, exc)

        return []
