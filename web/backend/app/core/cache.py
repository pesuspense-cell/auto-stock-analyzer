"""cache.py — Streamlit @st.cache_data 대체 (cachetools.TTLCache 기반).

app.py 의 `@st.cache_data(ttl=N)` 래퍼들을 1:1로 옮기기 위한 데코레이터.
스레드 안전하며, 인자(해시 가능)별로 TTL 캐시를 유지한다.
"""
from __future__ import annotations

import functools
import threading
from typing import Callable

from cachetools import TTLCache
from cachetools.keys import hashkey


def ttl_cache(ttl: int = 300, maxsize: int = 256) -> Callable:
    """`@ttl_cache(ttl=300)` 형태로 사용. 동기 함수 전용.

    @st.cache_data(ttl=...) 와 동일한 의미(인자 키별 TTL 메모이즈).
    """

    def decorator(func: Callable) -> Callable:
        cache: TTLCache = TTLCache(maxsize=maxsize, ttl=ttl)
        lock = threading.Lock()

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = hashkey(*args, **kwargs)
            with lock:
                if key in cache:
                    return cache[key]
            result = func(*args, **kwargs)
            with lock:
                cache[key] = result
            return result

        wrapper.cache_clear = cache.clear  # type: ignore[attr-defined]
        return wrapper

    return decorator
