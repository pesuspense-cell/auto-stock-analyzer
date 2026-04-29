"""
news_async.py - 비동기 뉴스 수집 및 3단계 계층형 분석 파이프라인

[병목 해소 전략]
  1. httpx + asyncio.gather()  → 네이버 뉴스 동시 크롤링 (순차 → 병렬)
  2. 3단계 계층형 필터          → LLM 호출을 상위 5건으로 제한
  3. Streamlit 호환 run_async() → tornado 이벤트 루프 충돌 없이 실행

[성능 목표]
  Before: 뉴스 수집·분석 30초+
  After:  목표 5~10초 이내
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import time
from typing import Any

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

# ─── 로거 ─────────────────────────────────────────────────────────────────────
logger = logging.getLogger("news_async")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] news_async: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

# ─── 상수 ─────────────────────────────────────────────────────────────────────
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Referer": "https://finance.naver.com/",
    "Accept-Language": "ko-KR,ko;q=0.9",
}

# 2단계 즉시 처리용 키워드 (LLM 호출 없이 점수 확정)
_NEG_FAST_KW: list[str] = [
    "부진", "하락", "급락", "소송", "제재", "손실", "적자", "리콜",
    "하향", "매도", "위기", "파산", "감소", "악화", "폭락", "경고",
    "취소", "철수", "연기", "거절",
]
_POS_FAST_KW: list[str] = [
    "실적", "흑자", "수주", "급등", "상향", "매수", "신고가", "성장",
    "개선", "호재", "돌파", "선정", "계약", "출시", "강세", "최대",
    "증가", "신규", "목표가 상향",
]


# ─── Streamlit 호환 비동기 실행기 ─────────────────────────────────────────────
def run_async(coro) -> Any:
    """
    Streamlit 동기 환경에서 코루틴을 안전하게 실행.

    Streamlit은 Tornado 기반 이벤트 루프 위에서 동작하므로
    asyncio.run()을 직접 호출하면 'This event loop is already running' 에러 발생.
    전용 스레드에서 새 이벤트 루프를 생성해 충돌을 방지한다.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


# ─── 비동기 네이버 뉴스 크롤링 ───────────────────────────────────────────────
async def _fetch_one(
    client: "httpx.AsyncClient",
    code: str,
    max_items: int,
) -> list[dict]:
    """단일 종목 코드 → 네이버 금융 뉴스 비동기 스크래핑."""
    url = f"https://finance.naver.com/item/news_news.naver?code={code}&page=1"
    try:
        resp = await client.get(url, timeout=8.0)
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
            items.append({
                "title":     title_el.get_text(strip=True),
                "link":      href,
                "publisher": info_el.get_text(strip=True) if info_el else "",
                "pub_date":  date_el.get_text(strip=True) if date_el else "",
            })
            if len(items) >= max_items:
                break
        return items
    except Exception as exc:
        logger.warning("크롤링 실패 [%s]: %s", code, exc)
        return []


async def async_fetch_multi_news(
    tickers: list[str],
    max_items: int = 12,
) -> dict[str, list[dict]]:
    """
    여러 티커의 뉴스를 asyncio.gather()로 동시 수집.

    반환: {ticker: [{"title", "link", "publisher", "pub_date"}, ...]}
    httpx 미설치 시 빈 dict 반환 (기존 requests 경로로 폴백 가능).
    """
    if not HAS_HTTPX or not HAS_BS4:
        logger.warning("httpx 또는 bs4 미설치 — 비동기 수집 불가")
        return {t: [] for t in tickers}

    t0 = time.perf_counter()

    # 국내 종목 코드만 추출 (숫자 6자리)
    code_map: dict[str, str] = {}
    for ticker in tickers:
        code = ticker.split(".")[0].strip()
        if code.isdigit():
            code_map[ticker] = code

    if not code_map:
        return {t: [] for t in tickers}

    async with httpx.AsyncClient(headers=_HEADERS, follow_redirects=True) as client:
        aws = [_fetch_one(client, code, max_items) for code in code_map.values()]
        raw_results = await asyncio.gather(*aws, return_exceptions=True)

    mapping: dict[str, list[dict]] = {}
    for ticker, result in zip(code_map.keys(), raw_results):
        mapping[ticker] = result if isinstance(result, list) else []

    elapsed = time.perf_counter() - t0
    logger.info(
        "동시 뉴스 수집 완료 — %d종목, %.2fs (평균 %.2fs/종목)",
        len(tickers), elapsed, elapsed / max(len(tickers), 1),
    )
    return mapping


async def async_fetch_naver_news(ticker: str, max_items: int = 12) -> list[dict]:
    """단일 티커 비동기 뉴스 수집 — async_fetch_multi_news() 래퍼."""
    result = await async_fetch_multi_news([ticker], max_items=max_items)
    return result.get(ticker, [])


def fetch_naver_news_fast(ticker: str, max_items: int = 12) -> list[dict]:
    """Streamlit 동기 컨텍스트에서 호출 가능한 단일 종목 빠른 뉴스 수집."""
    return run_async(async_fetch_naver_news(ticker, max_items))


def fetch_multi_news_fast(
    tickers: list[str],
    max_items: int = 12,
) -> dict[str, list[dict]]:
    """Streamlit 동기 컨텍스트에서 호출 가능한 다중 종목 동시 뉴스 수집."""
    return run_async(async_fetch_multi_news(tickers, max_items))


# ─── 3단계 계층형 필터 ────────────────────────────────────────────────────────
def stage1_title_filter(
    news_items: list[dict],
    company_name: str,
) -> list[dict]:
    """
    1단계: 제목에 종목명 없는 뉴스 즉시 제외.

    company_name이 없거나 필터 후 결과가 0건이면 원본 반환(폴백).
    """
    if not company_name:
        return news_items
    passed = [it for it in news_items if company_name in it.get("title", "")]
    logger.debug("Stage1 — %d→%d건 (기준: '%s')", len(news_items), len(passed), company_name)
    return passed if passed else news_items


def stage2_keyword_filter(
    news_items: list[dict],
) -> tuple[list[dict], list[dict]]:
    """
    2단계: 단일 방향 키워드(순부정/순긍정) 뉴스 즉시 점수 확정.

    반환:
      deep_candidates  — 혼재/키워드 없음 → 3단계 정밀 분석 대상
      pre_scored       — 이미 점수 확정된 항목 (_fast_score, _fast_reason 필드 추가)

    부정만 있으면 즉시 감점, 긍정만 있으면 즉시 가산 → LLM 호출 횟수 감소.
    """
    deep_candidates: list[dict] = []
    pre_scored: list[dict] = []

    for item in news_items:
        title    = item.get("title", "")
        neg_hits = sum(1 for kw in _NEG_FAST_KW if kw in title)
        pos_hits = sum(1 for kw in _POS_FAST_KW if kw in title)

        if neg_hits > 0 and pos_hits == 0:
            scored = dict(item)
            scored["_fast_score"]  = round(-neg_hits * 0.4, 2)
            scored["_fast_reason"] = f"[2단계 즉시 감점] 부정 키워드 {neg_hits}개"
            pre_scored.append(scored)
        elif pos_hits > 0 and neg_hits == 0:
            scored = dict(item)
            scored["_fast_score"]  = round(pos_hits * 0.4, 2)
            scored["_fast_reason"] = f"[2단계 즉시 가산] 긍정 키워드 {pos_hits}개"
            pre_scored.append(scored)
        else:
            # 혼재(상충) 또는 키워드 없음 → 정밀 판단 필요
            deep_candidates.append(item)

    logger.debug(
        "Stage2 — deep=%d, pre_scored=%d",
        len(deep_candidates), len(pre_scored),
    )
    return deep_candidates, pre_scored


def stage3_select_for_deep(candidates: list[dict], n: int = 5) -> list[dict]:
    """
    3단계: LLM 정밀 분석에 보낼 상위 N개 선택.
    현재는 최신순(원본 순서) 상위 N개 — 추후 A/B 등급 우선 정렬 가능.
    """
    return candidates[:n]


# ─── 전체 최적화 뉴스 분석 파이프라인 ────────────────────────────────────────
def analyze_news_fast(
    ticker: str,
    company_name: str = "",
    api_key: str = "",
    groq_api_key: str = "",
    news_items: list[dict] | None = None,
    max_news: int = 12,
    deep_n: int = 5,
) -> dict:
    """
    최적화된 3단계 뉴스 분석 파이프라인.

    [단계별 소요 시간 목표]
      Step 1  뉴스 수집    (async httpx)     : 1~2s
      Step 2  1단계 필터  (제목 종목명)      : <0.01s
      Step 3  2단계 필터  (키워드 즉시 처리) : <0.01s
      Step 4  3단계 정밀  (LLM, 상위 5건만) : 2~4s
      Step 5  결과 병합                      : <0.01s
      ─────────────────────────────────────────────
      합계                                   : 3~7s  (기존 30s+ → 80% 단축)

    반환 스키마: analyze_news_sentiment_llm()과 동일
      {"score", "label", "detail", "event_flags", "summary",
       "individual_score", "sector_score"}
    """
    # 지연 import — 순환 참조 방지
    from stock_ai import (
        analyze_news_sentiment_keywords,
        analyze_news_sentiment_llm,
    )

    _t_total = time.perf_counter()
    _empty = {
        "score": 0.0, "label": "중립", "detail": [],
        "event_flags": [], "summary": "뉴스 없음",
        "individual_score": 0.0, "sector_score": 0.0,
    }

    # ── Step 1: 뉴스 수집 ────────────────────────────────────────────────────
    if news_items is None:
        t0 = time.perf_counter()
        news_items = fetch_naver_news_fast(ticker, max_items=max_news)
        logger.info("Step1 뉴스수집: %.2fs (%d건)", time.perf_counter() - t0, len(news_items))

    if not news_items:
        logger.info("뉴스 없음 — 분석 종료 (%.2fs)", time.perf_counter() - _t_total)
        return _empty

    # ── Step 2: 1단계 필터 (제목 종목명 없으면 제외) ─────────────────────────
    t0 = time.perf_counter()
    n_before = len(news_items)
    filtered = stage1_title_filter(news_items, company_name)
    logger.info(
        "Step2 제목필터: %.3fs (%d→%d건)",
        time.perf_counter() - t0, n_before, len(filtered),
    )

    # ── Step 3: 2단계 필터 (키워드 즉시 점수 확정) ───────────────────────────
    t0 = time.perf_counter()
    deep_candidates, pre_scored = stage2_keyword_filter(filtered)
    logger.info(
        "Step3 키워드필터: %.3fs — LLM대상=%d건, 즉시처리=%d건",
        time.perf_counter() - t0, len(deep_candidates), len(pre_scored),
    )

    # ── Step 4: 3단계 정밀 분석 (상위 deep_n건만 LLM 호출) ──────────────────
    t0 = time.perf_counter()
    top_for_deep = stage3_select_for_deep(deep_candidates, n=deep_n)

    if api_key and top_for_deep:
        deep_result = analyze_news_sentiment_llm(
            top_for_deep, ticker, api_key, groq_api_key, company_name
        )
    elif top_for_deep:
        deep_result = analyze_news_sentiment_keywords(top_for_deep, ticker, company_name)
    else:
        deep_result = dict(_empty)
        deep_result["summary"] = ""

    logger.info("Step4 정밀분석: %.2fs", time.perf_counter() - t0)

    # ── Step 5: pre_scored 결과 병합 ─────────────────────────────────────────
    if pre_scored:
        t0 = time.perf_counter()
        extra = analyze_news_sentiment_keywords(pre_scored, ticker, company_name)

        d_score = deep_result.get("score", 0.0)
        e_score = extra.get("score", 0.0)
        # deep(LLM): 70%, keyword(pre_scored): 30% 가중 블렌딩
        blended = round(max(-5.0, min(5.0, 0.7 * d_score + 0.3 * e_score)), 2)

        if   blended >= 3:  label = "매우 긍정"
        elif blended >= 1:  label = "긍정"
        elif blended >= -1: label = "중립"
        elif blended >= -3: label = "부정"
        else:               label = "매우 부정"

        deep_result = dict(deep_result)
        deep_result["score"]        = blended
        deep_result["label"]        = label
        deep_result["detail"]       = deep_result.get("detail", []) + extra.get("detail", [])
        deep_result["event_flags"]  = deep_result.get("event_flags", []) + extra.get("event_flags", [])
        logger.info("Step5 결과병합: %.3fs", time.perf_counter() - t0)

    total_elapsed = time.perf_counter() - _t_total
    logger.info(
        "=== analyze_news_fast 완료 — %.2fs | score=%.2f | label=%s ===",
        total_elapsed, deep_result.get("score", 0.0), deep_result.get("label", ""),
    )
    return deep_result
