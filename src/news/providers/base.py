"""
src/news/providers/base.py — 뉴스 공급자 추상 기반 클래스

계약:
  - fetch()는 절대 예외를 발생시키지 않습니다. 내부 처리 후 빈 리스트 반환.
  - 반환 아이템은 반드시 _make_item() 스키마를 따릅니다.
    (source_type, source_weight, relevance_score 포함)
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod


class BaseNewsProvider(ABC):
    """
    뉴스 소스 공급자 공통 추상 기반 클래스.

    서브클래스 필수 속성:
      source_type   — 소스 식별자 문자열 ("naver" | "economy_rss" | "youtube_transcript")
      source_weight — 신뢰도 가중치 float (기본 1.0)

    서브클래스 필수 구현:
      fetch(ticker, company_name, max_items, **kwargs) -> list[dict]
    """

    source_type: str = "unknown"
    source_weight: float = 1.0

    def __init__(self) -> None:
        self.logger = logging.getLogger(f"news.provider.{self.source_type}")
        if not self.logger.handlers:
            _h = logging.StreamHandler()
            _h.setFormatter(
                logging.Formatter(
                    f"%(asctime)s [%(levelname)s] {self.__class__.__name__}: %(message)s",
                    datefmt="%H:%M:%S",
                )
            )
            self.logger.addHandler(_h)
        self.logger.setLevel(logging.INFO)

    @abstractmethod
    async def fetch(
        self,
        ticker: str,
        company_name: str = "",
        max_items: int = 12,
        **kwargs,
    ) -> list[dict]:
        """
        뉴스 수집. 실패 시 빈 리스트 반환 (예외 발생 금지).
        반환 dict는 _make_item() 스키마를 따라야 합니다.
        """

    def _make_item(
        self,
        title: str,
        link: str = "",
        publisher: str = "",
        pub_date: str = "",
        content: str = "",
    ) -> dict:
        """표준 뉴스 아이템 dict — source_type, source_weight 자동 주입."""
        return {
            "title":         title,
            "link":          link,
            "publisher":     publisher,
            "pub_date":      pub_date,
            "content":       content,
            "source_type":   self.source_type,
            "source_weight": self.source_weight,
            "relevance_score": 0.0,
            "score":         0.0,
        }
