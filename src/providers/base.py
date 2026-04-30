"""
src/providers/base.py — Provider 추상 기반 클래스

Provider 패턴 인터페이스:
  fetch(ticker)               : 데이터 수집. 실패 시 None. 예외 금지.
  fetch_with_fallback(ticker) : fetch + 인메모리 캐시 폴백 (Silent Fallback).
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional


class BaseProvider(ABC):
    """
    데이터 수집 Provider 공통 추상 기반 클래스.

    서브클래스는 fetch()를 구현해야 하며,
    네트워크·파싱 예외를 내부에서 처리하고 실패 시 None을 반환해야 합니다.
    """

    def __init__(self) -> None:
        name = self.__class__.__name__
        self.logger = logging.getLogger(f"provider.{name}")
        if not self.logger.handlers:
            _h = logging.StreamHandler()
            _h.setFormatter(
                logging.Formatter(
                    f"%(asctime)s [%(levelname)s] {name}: %(message)s",
                    datefmt="%H:%M:%S",
                )
            )
            self.logger.addHandler(_h)
        self.logger.setLevel(logging.INFO)
        # 인메모리 폴백 캐시 (프로세스 생존 동안 유지)
        self._mem_cache: dict[str, dict] = {}

    @abstractmethod
    async def fetch(self, ticker: str) -> Optional[dict]:
        """
        ticker 의 데이터를 수집합니다.
        성공: 비어있지 않은 dict 반환.
        실패: None 반환 (예외 발생 금지 — 호출자 보호).
        """

    async def fetch_with_fallback(self, ticker: str) -> dict:
        """
        fetch() 를 시도하고, 실패 시 마지막 성공 인메모리 캐시를
        {"_stale": True} 플래그와 함께 반환합니다.
        캐시도 없으면 빈 dict 반환 (절대 예외 미발생).
        """
        try:
            result = await self.fetch(ticker)
            if result:
                self._mem_cache[ticker] = result
                return result
        except Exception as e:
            self.logger.error("[%s] fetch 예외 — 폴백 시도: %s", ticker, e)

        cached = self._mem_cache.get(ticker)
        if cached:
            self.logger.warning("[%s] 인메모리 폴백 캐시 사용 (_stale=True)", ticker)
            return {**cached, "_stale": True}

        self.logger.warning("[%s] 데이터 없음 — 빈 dict 반환", ticker)
        return {}
