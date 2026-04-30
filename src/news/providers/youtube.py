"""
src/news/providers/youtube.py — YouTube 자막 기반 뉴스 공급자

대상: SBS Biz 모닝벨 등 증권 전문 채널
source_weight = 1.5 (전문가 해설, 최고 신뢰도)

의존성:
  pip install youtube-transcript-api

동작 방식:
  1. YouTube 채널 공개 Atom RSS로 최신 영상 목록 수집 (API 키 불필요)
  2. title_filter('모닝벨')에 매칭되는 영상만 선택
  3. youtube-transcript-api로 자막 추출 (한국어 우선, 영어 폴백)
  4. 자막 길이 > TRANSCRIPT_MAX_CHARS → _needs_summary=True 플래그
     (news_async.py가 LLM 사전 요약 단계를 별도 실행)

예외 처리:
  - youtube-transcript-api 미설치 → 빈 리스트 반환
  - 채널 RSS 접근 실패 → 빈 리스트 반환
  - 개별 영상 자막 없음 → 해당 영상만 건너뜀
"""
from __future__ import annotations

import asyncio
import xml.etree.ElementTree as ET

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

from .base import BaseNewsProvider

# ── 상수 ─────────────────────────────────────────────────────────────────────

# SBS Biz 채널 ID (YouTube 채널 관리 > 기본 정보에서 확인 가능)
# 변경 시: https://www.youtube.com/@sbsbiz → 채널 소스 내 "channel_id" 검색
_SBS_BIZ_CHANNEL_ID = "UCXzpKExCVQv5YjXfqOqNEYQ"

_YT_RSS_TMPL      = "https://www.youtube.com/feeds/videos.xml?channel_id={}"
_TRANSCRIPT_MAX   = 4_000   # 이상이면 요약 플래그
_MAX_VIDEOS       = 3       # 채널당 최대 처리 영상 수

_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "yt":   "http://www.youtube.com/xml/schemas/2015",
    "media":"http://search.yahoo.com/mrss/",
}

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; YTRSSBot/1.0)",
}


class YouTubeTranscriptProvider(BaseNewsProvider):
    """
    YouTube 채널 자막 기반 뉴스 공급자.

    Parameters
    ----------
    channel_ids : list[str] | None
        수집 대상 YouTube 채널 ID 목록. 기본값은 SBS Biz.
    title_filter : str
        영상 제목 필터 키워드. 기본값 "모닝벨".
    max_videos : int
        채널당 최대 처리 영상 수. 기본값 3.
    api_key : str
        Gemini API 키 (사전 요약 단계에서 사용, 선택).
    """

    source_type   = "youtube_transcript"
    source_weight = 1.5

    def __init__(
        self,
        channel_ids: list[str] | None = None,
        title_filter: str = "모닝벨",
        max_videos: int = _MAX_VIDEOS,
        api_key: str = "",
    ) -> None:
        super().__init__()
        self._channel_ids  = channel_ids or [_SBS_BIZ_CHANNEL_ID]
        self._title_filter = title_filter
        self._max_videos   = max_videos
        self._api_key      = api_key

    # ── 공개 인터페이스 ───────────────────────────────────────────────────────

    async def fetch(
        self,
        ticker: str,
        company_name: str = "",
        max_items: int = 5,
        **kwargs,
    ) -> list[dict]:
        try:
            from youtube_transcript_api import YouTubeTranscriptApi  # type: ignore
        except ImportError:
            self.logger.warning(
                "youtube-transcript-api 미설치 — YouTube 소스 건너뜀\n"
                "  설치: pip install youtube-transcript-api"
            )
            return []

        if not HAS_HTTPX:
            self.logger.warning("httpx 미설치 — YouTube 소스 건너뜀")
            return []

        # 채널별 최신 영상 목록을 병렬 수집
        try:
            async with httpx.AsyncClient(
                headers=_HEADERS, follow_redirects=True, timeout=10.0
            ) as client:
                rss_tasks = [
                    self._fetch_channel_videos(client, ch_id)
                    for ch_id in self._channel_ids
                ]
                channel_results = await asyncio.gather(
                    *rss_tasks, return_exceptions=True
                )
        except Exception as exc:
            self.logger.warning("YouTube RSS 수집 중 예외: %s", exc)
            return []

        # 전체 채널 영상 목록 합산
        video_pool: list[tuple[str, str]] = []  # (video_id, title)
        for res in channel_results:
            if isinstance(res, Exception):
                self.logger.warning("채널 RSS 수집 실패: %s", res)
                continue
            video_pool.extend(res)

        # 제목 필터 (모닝벨 등)
        if self._title_filter:
            video_pool = [
                (vid, title)
                for vid, title in video_pool
                if self._title_filter in title
            ]

        video_pool = video_pool[: self._max_videos]
        if not video_pool:
            self.logger.info(
                "유효 영상 없음 (title_filter=%r)", self._title_filter
            )
            return []

        # 자막 추출 (blocking → run_in_executor)
        loop = asyncio.get_event_loop()
        items: list[dict] = []
        for video_id, title in video_pool:
            text = await loop.run_in_executor(
                None,
                lambda vid=video_id: self._get_transcript(
                    vid, YouTubeTranscriptApi
                ),
            )
            if not text:
                continue

            item = self._make_item(
                title=title,
                link=f"https://www.youtube.com/watch?v={video_id}",
                publisher="SBS Biz",
                content=text,
            )
            item["_needs_summary"] = len(text) > _TRANSCRIPT_MAX
            item["_video_id"]      = video_id
            items.append(item)
            self.logger.info(
                "자막 수집 완료 [%s] %d자 (요약=%s)",
                video_id, len(text), item["_needs_summary"],
            )

        return items[:max_items]

    # ── 내부 헬퍼 ────────────────────────────────────────────────────────────

    async def _fetch_channel_videos(
        self,
        client: "httpx.AsyncClient",
        channel_id: str,
    ) -> list[tuple[str, str]]:
        url = _YT_RSS_TMPL.format(channel_id)
        try:
            resp = await client.get(url)
        except Exception as exc:
            self.logger.warning(
                "YouTube RSS 요청 실패 [%s]: %s", channel_id, exc
            )
            return []

        if resp.status_code != 200:
            self.logger.warning(
                "YouTube RSS 응답 %d [%s]", resp.status_code, channel_id
            )
            return []

        return self._parse_yt_rss(resp.text, channel_id)

    def _parse_yt_rss(
        self, xml_text: str, channel_id: str
    ) -> list[tuple[str, str]]:
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            self.logger.warning(
                "YouTube RSS 파싱 오류 [%s]: %s", channel_id, exc
            )
            return []

        results: list[tuple[str, str]] = []
        for entry in root.findall("atom:entry", _NS):
            video_id = entry.findtext("yt:videoId", namespaces=_NS) or ""
            title    = (
                entry.findtext("atom:title", namespaces=_NS) or ""
            ).strip()
            if video_id and title:
                results.append((video_id, title))
        return results

    def _get_transcript(self, video_id: str, api_cls) -> str:
        """동기 함수 — asyncio.run_in_executor()에서 실행."""
        try:
            segments = api_cls.get_transcript(
                video_id, languages=["ko", "en"]
            )
            return " ".join(seg["text"] for seg in segments)
        except Exception as exc:
            self.logger.warning("자막 추출 실패 [%s]: %s", video_id, exc)
            return ""
