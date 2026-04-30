"""
src/providers — 데이터 수집 Provider 패키지

각 Provider는 BaseProvider를 상속하며 fetch(ticker) → dict 인터페이스를 구현합니다.
실패 시 None 반환, 예외 미발생, 인메모리 캐시 폴백 내장.
"""
from .base import BaseProvider
from .yahoo_provider import YahooProvider
from .krx_provider import KRXProvider
from .dart_provider import DARTProvider

__all__ = ["BaseProvider", "YahooProvider", "KRXProvider", "DARTProvider"]
