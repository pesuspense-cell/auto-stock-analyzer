"""
news_async.py - 다중 소스 통합 뉴스 에이전트 파이프라인

[아키텍처 개요]
  ┌────────────────────────────────────────────────────────┐
  │              analyze_news_fast()                       │
  │                                                        │
  │  Step 1  ┌─────────────┐ asyncio.gather               │
  │  병렬수집 │ NaverNews   │ source_weight=0.8            │
  │          │ EconomyRSS  │ source_weight=1.0            │
  │          │ YouTubeTrns │ source_weight=1.5            │
  │          └──────┬──────┘                              │
  │                 │ merge                               │
  │  Step 1.5  source_weight 주입                         │
  │  Step 1.6  LLM 임베딩 코사인 유사도 → 관련성 필터      │
  │  Step 1.7  YouTube 장문 대본 LLM 사전 요약             │
  │                 │                                     │
  │  Step 2  Stage1 제목 가중치 필터                       │
  │  Step 3  Stage2 키워드·부정문맥 즉시처리               │
  │  Step 3.5 DART 공시 교차검증                          │
  │  Step 4  Stage3 LLM 정밀분석 (상위 5건)               │
  │  Step 5  결과 병합                                     │
  │  Step 6  뉴스-수급-가격 삼각검증                       │
  └────────────────────────────────────────────────────────┘

[반환 스키마] (기존 필드 유지 + 신규 추가)
  score, label, detail, event_flags, summary,
  individual_score, sector_score,          ← 기존 (하위 호환)
  source_type, relevance_score,            ← 신규: 소스 타입 목록 / 평균 관련성
  source_breakdown                         ← 신규: 소스별 수집 건수
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import re
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

# ─── 소스별 신뢰도 가중치 ────────────────────────────────────────────────────
SOURCE_WEIGHTS: dict[str, float] = {
    "youtube_transcript": 1.5,   # 전문가 해설 (SBS Biz 모닝벨 등)
    "economy_rss":        1.0,   # 경제 전문지 (매경·한경)
    "naver":              0.8,   # 일반 뉴스 포털
    "unknown":            0.8,
}

_RELEVANCE_THRESHOLD = 0.30   # 코사인 유사도 하한선

# ─── 기존 상수 (하위 호환 유지) ───────────────────────────────────────────────
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Referer":         "https://finance.naver.com/",
    "Accept-Language": "ko-KR,ko;q=0.9",
}

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

_RELATED_STOCK_RE = re.compile(r"관련주|테마주|수혜주|동반\s*(?:상승|하락)|함께\s*주목")
_CORE_KW_ALL: list[str] = _NEG_FAST_KW + _POS_FAST_KW

_NEG_CONTEXT_PHRASES: list[str] = [
    "실패", "우려", "제한", "불발", "무산", "둔화", "위축",
    "악화", "실망", "하락", "약세", "부담", "어렵", "않",
]
_NEG_COMPOUND_RE = re.compile(
    r"(?:" + "|".join(re.escape(kw) for kw in _POS_FAST_KW) + r")"
    r"\s{0,3}"
    r"(?:" + "|".join(re.escape(p) for p in _NEG_CONTEXT_PHRASES) + r")"
)
_CONTRACT_KW: list[str] = ["수주", "계약"]


# ─── DART 공시 교차 검증 ──────────────────────────────────────────────────────
def _check_dart_contract_disclosure(ticker: str, dart_api_key: str) -> bool:
    """최근 30일 DART 공시에서 수주/계약 관련 공시가 있으면 True."""
    if not dart_api_key:
        return False
    try:
        import OpenDartReader  # type: ignore
        import datetime as _dt

        code = ticker.split(".")[0]
        dart = OpenDartReader.OpenDartReader(dart_api_key)

        end_dt   = time.strftime("%Y%m%d")
        start_dt = (
            _dt.date.today() - _dt.timedelta(days=30)
        ).strftime("%Y%m%d")

        corp_df = dart.corp_codes
        rows = corp_df[corp_df["stock_code"] == code]
        if rows.empty:
            return False
        corp_code = rows.iloc[0]["corp_code"]

        disc = dart.list(corp_code, start=start_dt, end=end_dt)
        if disc is None or (hasattr(disc, "empty") and disc.empty):
            return False

        title_col = "report_nm" if "report_nm" in disc.columns else "title"
        return any(
            any(kw in str(t) for kw in _CONTRACT_KW)
            for t in disc[title_col]
        )
    except Exception as exc:
        logger.debug("DART 공시 조회 실패 [%s]: %s", ticker, exc)
        return False


# ─── Streamlit 호환 비동기 실행기 ─────────────────────────────────────────────
def run_async(coro) -> Any:
    """Streamlit(Tornado) 동기 컨텍스트에서 코루틴을 안전하게 실행."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


# ─── 소스 가중치 유틸리티 ─────────────────────────────────────────────────────
def apply_source_weights(news_items: list[dict]) -> list[dict]:
    """
    source_type에 따른 신뢰도 가중치를 source_weight 필드로 주입.
    Provider가 이미 설정한 경우 덮어쓰지 않음.
    """
    result: list[dict] = []
    for item in news_items:
        it = dict(item)
        if "source_weight" not in it:
            src = it.get("source_type", "unknown")
            it["source_weight"] = SOURCE_WEIGHTS.get(src, 0.8)
        result.append(it)
    return result


# ─── LLM 임베딩 기반 관련성 점수 ─────────────────────────────────────────────
def compute_relevance_scores(
    news_items: list[dict],
    company_name: str,
    api_key: str = "",
) -> list[dict]:
    """
    뉴스 아이템에 종목 프로필과의 코사인 유사도를 relevance_score로 주입.

    - api_key(Gemini) 있음 → LLM 임베딩 (GoogleGenerativeAIEmbeddings)
    - api_key 없음 → 키워드 기반 폴백
    - 임베딩 실패 → 키워드 기반 폴백 (시스템 중단 없음)
    - company_name 없음 → 전체 1.0 (필터링 방지)
    """
    if not news_items:
        return news_items
    if not company_name:
        return [{**it, "relevance_score": 1.0} for it in news_items]

    needs = [(i, it) for i, it in enumerate(news_items)
             if it.get("relevance_score", 0.0) == 0.0]
    if not needs:
        return news_items

    if api_key:
        return _relevance_embedding(news_items, needs, company_name, api_key)
    return _relevance_keyword(news_items, needs, company_name)


def _relevance_embedding(
    news_items: list[dict],
    needs: list[tuple[int, dict]],
    company_name: str,
    api_key: str,
) -> list[dict]:
    """Google Generative AI embedding-001 → 코사인 유사도."""
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings  # type: ignore
        import numpy as np  # type: ignore

        embedder = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key,
        )
        profile = f"{company_name} 주식 투자 관련 핵심 뉴스 및 사업 동향"
        texts   = [
            f"{it['title']} {it.get('content', '')}"[:500]
            for _, it in needs
        ]
        vectors = embedder.embed_documents([profile] + texts)
        pv      = np.array(vectors[0])

        result = list(news_items)
        for (idx, _), vec in zip(needs, vectors[1:]):
            nv   = np.array(vec)
            dnom = float(np.linalg.norm(pv) * np.linalg.norm(nv))
            sim  = float(np.dot(pv, nv) / dnom) if dnom else 0.0
            result[idx] = {**result[idx], "relevance_score": round(sim, 4)}
        return result

    except Exception as exc:
        logger.warning("임베딩 계산 실패 — 키워드 폴백: %s", exc)
        return _relevance_keyword(news_items, needs, company_name)


def _relevance_keyword(
    news_items: list[dict],
    needs: list[tuple[int, dict]],
    company_name: str,
) -> list[dict]:
    """종목명 키워드 매칭 기반 관련성 점수 (임베딩 폴백)."""
    result = list(news_items)
    for idx, it in needs:
        in_title   = int(company_name in it.get("title",   ""))
        in_content = int(company_name in it.get("content", ""))
        score = min(in_title * 0.6 + in_content * 0.3 + 0.1, 1.0)
        result[idx] = {**it, "relevance_score": round(score, 4)}
    return result


def filter_by_relevance(
    news_items: list[dict],
    threshold: float = _RELEVANCE_THRESHOLD,
) -> list[dict]:
    """
    relevance_score < threshold 아이템 제거.
    전체 제거 시 상위 3건 보존 (안전 폴백).
    """
    passed = [it for it in news_items
              if it.get("relevance_score", 1.0) >= threshold]
    if not passed and news_items:
        passed = sorted(
            news_items,
            key=lambda x: x.get("relevance_score", 0.0),
            reverse=True,
        )[:3]
        logger.warning("관련성 필터 — 전체 필터링됨, 상위 3건 보존")
    return passed


# ─── YouTube 대본 LLM 사전 요약 ───────────────────────────────────────────────
async def _summarize_transcript(
    text: str,
    api_key: str = "",
    groq_api_key: str = "",
) -> str:
    """
    증권 방송 대본을 '주요 투자 아이디어' 위주로 300자 이내 요약.

    Gemini 우선 → Groq 폴백 → 앞 1,000자 단순 절삭
    """
    PROMPT = (
        "다음 증권 방송 대본에서 주요 투자 아이디어, 종목 언급, 시장 전망만 "
        "핵심적으로 추출하여 300자 이내로 요약해주세요.\n\n"
        "대본:\n{text}\n\n요약:"
    )
    loop = asyncio.get_event_loop()

    if api_key:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
            from langchain_core.messages import HumanMessage           # type: ignore

            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=api_key,
                temperature=0.1,
            )
            prompt   = PROMPT.format(text=text[:6_000])
            response = await loop.run_in_executor(
                None, lambda: llm.invoke([HumanMessage(content=prompt)])
            )
            return response.content.strip()
        except Exception as exc:
            logger.warning("Gemini 요약 실패: %s", exc)

    if groq_api_key:
        try:
            from groq import Groq  # type: ignore

            client = Groq(api_key=groq_api_key)
            prompt = PROMPT.format(text=text[:4_000])
            response = await loop.run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                ),
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            logger.warning("Groq 요약 실패: %s", exc)

    return text[:1_000]


async def _summarize_youtube_items(
    items: list[dict],
    api_key: str,
    groq_api_key: str,
) -> list[dict]:
    """_needs_summary=True인 YouTube 아이템을 병렬로 요약."""
    async def _process(item: dict) -> dict:
        if not item.get("_needs_summary"):
            return item
        summarized = await _summarize_transcript(
            item["content"], api_key, groq_api_key
        )
        return {**item, "content": summarized, "_needs_summary": False}

    tasks   = [_process(it) for it in items]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    out: list[dict] = []
    for orig, res in zip(items, results):
        if isinstance(res, Exception):
            logger.warning("YouTube 요약 실패 — 원본 유지: %s", res)
            out.append(orig)
        else:
            out.append(res)
    return out


# ─── 다중 소스 병렬 수집 ──────────────────────────────────────────────────────
async def _collect_all_sources(
    ticker: str,
    company_name: str,
    max_news: int,
    providers: list,
) -> list[dict]:
    """모든 Provider를 asyncio.gather()로 병렬 실행하여 결과 합산."""
    tasks = [
        p.fetch(ticker, company_name, max_items=max_news)
        for p in providers
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_items: list[dict] = []
    for provider, result in zip(providers, results):
        src = getattr(provider, "source_type", "unknown")
        if isinstance(result, Exception):
            logger.warning("Provider [%s] 수집 실패: %s", src, result)
            continue
        logger.info("Provider [%s] %d건 수집", src, len(result))
        all_items.extend(result)

    return all_items


def _build_providers(api_key: str = "") -> list:
    """기본 Provider 인스턴스 목록 생성. 임포트 실패 시 해당 소스 건너뜀."""
    providers = []
    try:
        from .news.providers import NaverNewsProvider
        providers.append(NaverNewsProvider())
    except Exception as exc:
        logger.warning("NaverNewsProvider 로드 실패: %s", exc)

    try:
        from .news.providers import EconomyRSSProvider
        providers.append(EconomyRSSProvider())
    except Exception as exc:
        logger.warning("EconomyRSSProvider 로드 실패: %s", exc)

    try:
        from .news.providers import YouTubeTranscriptProvider
        providers.append(YouTubeTranscriptProvider(api_key=api_key))
    except Exception as exc:
        logger.warning("YouTubeTranscriptProvider 로드 실패: %s", exc)

    return providers


def _source_breakdown(items: list[dict]) -> dict[str, int]:
    """소스별 수집 건수 집계."""
    counts: dict[str, int] = {}
    for it in items:
        src = it.get("source_type", "unknown")
        counts[src] = counts.get(src, 0) + 1
    return counts


# ─── 기존 네이버 단독 수집 함수 (하위 호환 유지) ─────────────────────────────
async def _fetch_one(
    client: "httpx.AsyncClient",
    code: str,
    max_items: int,
) -> list[dict]:
    """단일 종목 코드 → 네이버 금융 뉴스 비동기 스크래핑 (레거시 경로)."""
    url = f"https://finance.naver.com/item/news_news.naver?code={code}&page=1"
    for attempt in range(3):
        try:
            resp = await client.get(url, timeout=8.0)
        except Exception as exc:
            if "Timeout" in type(exc).__name__ and attempt < 2:
                await asyncio.sleep(1)
                continue
            logger.warning("크롤링 실패 [%s]: %s", code, exc)
            return []

        if resp.status_code == 403 and attempt < 2:
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
            items.append({
                "title":         title_el.get_text(strip=True),
                "link":          href,
                "publisher":     info_el.get_text(strip=True) if info_el else "",
                "pub_date":      date_el.get_text(strip=True) if date_el else "",
                "source_type":   "naver",
                "source_weight": 0.8,
                "relevance_score": 0.0,
                "score":         0.0,
            })
            if len(items) >= max_items:
                break
        return items

    return []


async def async_fetch_multi_news(
    tickers: list[str],
    max_items: int = 12,
) -> dict[str, list[dict]]:
    """여러 티커의 뉴스를 asyncio.gather()로 동시 수집 (네이버 전용 레거시)."""
    if not HAS_HTTPX or not HAS_BS4:
        return {t: [] for t in tickers}

    t0 = time.perf_counter()
    code_map: dict[str, str] = {}
    for ticker in tickers:
        code = ticker.split(".")[0].strip()
        if code.isdigit():
            code_map[ticker] = code

    if not code_map:
        return {t: [] for t in tickers}

    async with httpx.AsyncClient(
        headers=_HEADERS,
        follow_redirects=True,
        limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
    ) as client:
        aws         = [_fetch_one(client, code, max_items) for code in code_map.values()]
        raw_results = await asyncio.gather(*aws, return_exceptions=True)

    mapping: dict[str, list[dict]] = {}
    for ticker, result in zip(code_map.keys(), raw_results):
        mapping[ticker] = result if isinstance(result, list) else []

    elapsed = time.perf_counter() - t0
    logger.info(
        "동시 뉴스 수집 완료 — %d종목, %.2fs",
        len(tickers), elapsed,
    )
    return mapping


async def async_fetch_naver_news(ticker: str, max_items: int = 12) -> list[dict]:
    result = await async_fetch_multi_news([ticker], max_items=max_items)
    return result.get(ticker, [])


def fetch_naver_news_fast(ticker: str, max_items: int = 12) -> list[dict]:
    """Streamlit 동기 컨텍스트 — 단일 종목 네이버 뉴스 수집 (레거시)."""
    return run_async(async_fetch_naver_news(ticker, max_items))


def fetch_multi_news_fast(
    tickers: list[str],
    max_items: int = 12,
) -> dict[str, list[dict]]:
    """Streamlit 동기 컨텍스트 — 다중 종목 네이버 뉴스 수집 (레거시)."""
    return run_async(async_fetch_multi_news(tickers, max_items))


# ─── 3단계 계층형 필터 (기존 로직 그대로 유지) ───────────────────────────────
def stage1_title_filter(
    news_items: list[dict],
    company_name: str,
) -> list[dict]:
    """
    1단계: 제목 가중치 부여 + 관련주 단순 나열 제외.

    _title_weight:
      2.0 — 종목명 + 핵심KW 동시 포함
      1.0 — 종목명만 포함
      0.5 — 종목명 없음 (본문 언급 추정)
      제외 — 관련주/테마주 패턴
    """
    results: list[dict] = []
    for item in news_items:
        title = item.get("title", "")
        if _RELATED_STOCK_RE.search(title):
            logger.debug("Stage1 관련주 패턴 제외: %.40s", title)
            continue

        scored = dict(item)
        if not company_name:
            scored["_title_weight"] = 1.0
        elif company_name in title:
            scored["_title_weight"] = (
                2.0 if any(kw in title for kw in _CORE_KW_ALL) else 1.0
            )
        else:
            scored["_title_weight"] = 0.5
        results.append(scored)

    if not results:
        results = [{**it, "_title_weight": 1.0} for it in news_items]

    logger.debug(
        "Stage1 — %d→%d건",
        len(news_items), len(results),
    )
    return results


def _has_negative_context(title: str) -> bool:
    if _NEG_COMPOUND_RE.search(title):
        return True
    words = title.split()
    for i, word in enumerate(words):
        if not any(kw in word for kw in _POS_FAST_KW):
            continue
        lo, hi = max(0, i - 5), min(len(words), i + 6)
        window = " ".join(words[lo:hi])
        if any(phrase in window for phrase in _NEG_CONTEXT_PHRASES):
            return True
    return False


def stage2_keyword_filter(
    news_items: list[dict],
) -> tuple[list[dict], list[dict]]:
    """
    2단계: 단일 방향 키워드 즉시 점수 확정 + 부정 문맥 감지.

    반환:
      deep_candidates — LLM 정밀 분석 대상
      pre_scored      — 즉시 점수 확정 (_fast_score, _fast_reason 추가)
    """
    deep_candidates: list[dict] = []
    pre_scored: list[dict] = []

    for item in news_items:
        title  = item.get("title", "")
        weight = item.get("_title_weight", 1.0)
        # source_weight도 반영
        src_w  = item.get("source_weight", 1.0)

        neg_hits = sum(1 for kw in _NEG_FAST_KW if kw in title)
        pos_hits = sum(1 for kw in _POS_FAST_KW if kw in title)

        if neg_hits > 0 and pos_hits == 0:
            scored = dict(item)
            scored["_fast_score"]  = round(-neg_hits * 0.4 * weight * src_w, 2)
            scored["_fast_reason"] = (
                f"[2단계 즉시 감점] 부정KW {neg_hits}개 "
                f"(title_w={weight}, src_w={src_w})"
            )
            pre_scored.append(scored)

        elif pos_hits > 0 and neg_hits == 0:
            if _has_negative_context(title):
                item_copy = dict(item)
                item_copy["_neg_context_flag"] = True
                deep_candidates.append(item_copy)
            else:
                scored = dict(item)
                scored["_fast_score"]  = round(pos_hits * 0.4 * weight * src_w, 2)
                scored["_fast_reason"] = (
                    f"[2단계 즉시 가산] 긍정KW {pos_hits}개 "
                    f"(title_w={weight}, src_w={src_w})"
                )
                pre_scored.append(scored)
        else:
            deep_candidates.append(item)

    logger.debug(
        "Stage2 — deep=%d, pre_scored=%d",
        len(deep_candidates), len(pre_scored),
    )
    return deep_candidates, pre_scored


def stage3_select_for_deep(candidates: list[dict], n: int = 5) -> list[dict]:
    """3단계: _title_weight 높은 순으로 상위 N건 선택 (LLM 비용 최적화)."""
    return sorted(
        candidates,
        key=lambda x: x.get("_title_weight", 1.0),
        reverse=True,
    )[:n]


# ─── 통합 분석 파이프라인 ─────────────────────────────────────────────────────
def analyze_news_fast(
    ticker: str,
    company_name: str = "",
    api_key: str = "",
    groq_api_key: str = "",
    dart_api_key: str = "",
    news_items: list[dict] | None = None,
    max_news: int = 12,
    deep_n: int = 5,
    price_change_pct: float | None = None,
    net_foreign_buy: float | None = None,
    net_institution_buy: float | None = None,
    providers: list | None = None,
) -> dict:
    """
    다중 소스 통합 뉴스 분석 파이프라인.

    [기존 파라미터 — 하위 호환 유지]
      ticker, company_name, api_key, groq_api_key, dart_api_key,
      news_items, max_news, deep_n,
      price_change_pct, net_foreign_buy, net_institution_buy

    [신규 파라미터]
      providers : list[BaseNewsProvider] | None
          None 이면 NaverNewsProvider + EconomyRSSProvider +
          YouTubeTranscriptProvider 기본 세트 사용.
          news_items 가 직접 전달된 경우 무시됨.

    [반환 스키마] (기존 필드 + 신규 필드)
      score, label, detail, event_flags, summary,
      individual_score, sector_score,    ← 기존 (변경 없음)
      source_type       : str            ← 수집된 소스 목록 (콤마 구분)
      relevance_score   : float          ← 전체 아이템 평균 관련성 점수
      source_breakdown  : dict           ← 소스별 수집 건수
    """
    from stock_ai import (  # 지연 import — 순환 참조 방지
        analyze_news_sentiment_keywords,
        analyze_news_sentiment_llm,
    )

    _t_total = time.perf_counter()
    _empty = {
        "score": 0.0, "label": "중립", "detail": [],
        "event_flags": [], "summary": "뉴스 없음",
        "individual_score": 0.0, "sector_score": 0.0,
        # 신규
        "source_type":      "",
        "relevance_score":  0.0,
        "source_breakdown": {},
    }

    # ── Step 1: 다중 소스 병렬 수집 ──────────────────────────────────────────
    if news_items is None:
        t0    = time.perf_counter()
        _prvs = providers if providers is not None else _build_providers(api_key)
        news_items = run_async(
            _collect_all_sources(ticker, company_name, max_news, _prvs)
        )
        logger.info(
            "Step1 다중소스수집: %.2fs (%d건 합산) — 소스: %s",
            time.perf_counter() - t0,
            len(news_items),
            ", ".join(sorted({it.get("source_type","?") for it in news_items})),
        )
    else:
        # 외부에서 직접 전달된 경우 source_type/weight 보정
        news_items = [
            {
                "source_type":   it.get("source_type", "naver"),
                "source_weight": it.get("source_weight", 0.8),
                "relevance_score": it.get("relevance_score", 0.0),
                "score": it.get("score", 0.0),
                **it,
            }
            for it in news_items
        ]

    breakdown = _source_breakdown(news_items)

    if not news_items:
        logger.info("뉴스 없음 — 분석 종료 (%.2fs)", time.perf_counter() - _t_total)
        return _empty

    # ── Step 1.5: source_weight 주입 ─────────────────────────────────────────
    news_items = apply_source_weights(news_items)

    # ── Step 1.6: LLM 임베딩 기반 관련성 점수 → 필터링 ───────────────────────
    t0 = time.perf_counter()
    news_items = compute_relevance_scores(news_items, company_name, api_key)
    n_before   = len(news_items)
    news_items = filter_by_relevance(news_items, _RELEVANCE_THRESHOLD)
    logger.info(
        "Step1.6 관련성필터: %.2fs (%d→%d건, threshold=%.2f)",
        time.perf_counter() - t0, n_before, len(news_items), _RELEVANCE_THRESHOLD,
    )

    # ── Step 1.7: YouTube 대본 사전 요약 ─────────────────────────────────────
    youtube_items = [it for it in news_items if it.get("_needs_summary")]
    if youtube_items:
        t0 = time.perf_counter()
        news_items = run_async(
            _summarize_youtube_items(news_items, api_key, groq_api_key)
        )
        logger.info(
            "Step1.7 YouTube요약: %.2fs (%d건 요약 처리)",
            time.perf_counter() - t0, len(youtube_items),
        )

    # 평균 관련성 점수
    avg_relevance = round(
        sum(it.get("relevance_score", 0.0) for it in news_items)
        / max(len(news_items), 1),
        4,
    )
    source_types_str = ",".join(sorted({it.get("source_type","") for it in news_items}))

    # ── Step 2: Stage1 제목 필터 ──────────────────────────────────────────────
    t0 = time.perf_counter()
    n_before = len(news_items)
    filtered = stage1_title_filter(news_items, company_name)
    logger.info(
        "Step2 제목필터: %.3fs (%d→%d건)",
        time.perf_counter() - t0, n_before, len(filtered),
    )

    # ── Step 3: Stage2 키워드 필터 ───────────────────────────────────────────
    t0 = time.perf_counter()
    deep_candidates, pre_scored = stage2_keyword_filter(filtered)
    logger.info(
        "Step3 키워드필터: %.3fs — LLM대상=%d건, 즉시처리=%d건",
        time.perf_counter() - t0, len(deep_candidates), len(pre_scored),
    )

    # ── Step 3.5: DART 공시 교차 검증 ────────────────────────────────────────
    if dart_api_key and pre_scored and any(
        any(kw in item.get("title", "") for kw in _CONTRACT_KW)
        for item in pre_scored
    ):
        t0 = time.perf_counter()
        dart_confirmed = _check_dart_contract_disclosure(ticker, dart_api_key)
        logger.info(
            "Step3.5 DART교차검증: %.2fs — %s",
            time.perf_counter() - t0,
            "공시 확인됨 (+1.0)" if dart_confirmed else "공시 없음",
        )
        if dart_confirmed:
            boosted: list[dict] = []
            for item in pre_scored:
                if any(kw in item.get("title", "") for kw in _CONTRACT_KW):
                    item = dict(item)
                    item["_fast_score"]  = round(item.get("_fast_score", 0.0) + 1.0, 2)
                    item["_fast_reason"] = (
                        item.get("_fast_reason", "") + " +DART공시확인(+1.0)"
                    )
                boosted.append(item)
            pre_scored = boosted

    # ── Step 4: Stage3 LLM 정밀분석 (상위 deep_n건) ──────────────────────────
    t0 = time.perf_counter()
    top_for_deep = stage3_select_for_deep(deep_candidates, n=deep_n)

    if api_key and top_for_deep:
        deep_result = analyze_news_sentiment_llm(
            top_for_deep, ticker, api_key, groq_api_key, company_name
        )
    elif top_for_deep:
        deep_result = analyze_news_sentiment_keywords(
            top_for_deep, ticker, company_name
        )
    else:
        deep_result = dict(_empty)
        deep_result["summary"] = ""

    logger.info("Step4 정밀분석: %.2fs", time.perf_counter() - t0)

    # ── Step 5: pre_scored 결과 병합 ─────────────────────────────────────────
    if pre_scored:
        t0    = time.perf_counter()
        extra = analyze_news_sentiment_keywords(pre_scored, ticker, company_name)

        d_score  = deep_result.get("score", 0.0)
        e_score  = extra.get("score", 0.0)
        blended  = round(max(-5.0, min(5.0, 0.7 * d_score + 0.3 * e_score)), 2)

        if   blended >= 3:  label = "매우 긍정"
        elif blended >= 1:  label = "긍정"
        elif blended >= -1: label = "중립"
        elif blended >= -3: label = "부정"
        else:               label = "매우 부정"

        deep_result = dict(deep_result)
        deep_result["score"]       = blended
        deep_result["label"]       = label
        deep_result["detail"]      = (
            deep_result.get("detail", []) + extra.get("detail", [])
        )
        deep_result["event_flags"] = (
            deep_result.get("event_flags", []) + extra.get("event_flags", [])
        )
        logger.info("Step5 결과병합: %.3fs", time.perf_counter() - t0)

    # ── Step 6: 뉴스-수급-가격 삼각 검증 ─────────────────────────────────────
    final_score    = deep_result.get("score", 0.0)
    has_supply_data = (
        price_change_pct is not None
        or net_foreign_buy is not None
        or net_institution_buy is not None
    )
    if final_score >= 1.0 and has_supply_data:
        price_down  = price_change_pct is not None and price_change_pct < 0
        supply_sell = (
            net_foreign_buy is not None and net_foreign_buy < 0
            and net_institution_buy is not None and net_institution_buy < 0
        )
        if price_down or supply_sell:
            deep_result = dict(deep_result)
            flags = list(deep_result.get("event_flags", []))
            if "수급_미동반" not in flags:
                flags.append("수급_미동반")
            deep_result["event_flags"] = flags

            warn_parts: list[str] = []
            if price_down:
                warn_parts.append(f"주가 {price_change_pct:+.1f}% 하락")
            if supply_sell:
                warn_parts.append("외국인·기관 동반 순매도")
            warn_str = " / ".join(warn_parts)

            existing = deep_result.get("summary", "")
            warning_line = (
                f"⚠️ 수급 동반 없는 뉴스 ({warn_str}) — 루머 가능성 주의"
            )
            deep_result["summary"] = (
                f"{warning_line}\n{existing}" if existing else warning_line
            )
            logger.info(
                "Step6 삼각검증: 신뢰도 경고 — score=%.2f, %s",
                final_score, warn_str,
            )

    # ── 신규 필드 주입 ────────────────────────────────────────────────────────
    deep_result = dict(deep_result)
    deep_result.setdefault("individual_score", 0.0)
    deep_result.setdefault("sector_score",     0.0)
    deep_result["source_type"]      = source_types_str
    deep_result["relevance_score"]  = avg_relevance
    deep_result["source_breakdown"] = breakdown

    total_elapsed = time.perf_counter() - _t_total
    logger.info(
        "=== analyze_news_fast 완료 — %.2fs | score=%.2f | label=%s | 소스=%s ===",
        total_elapsed,
        deep_result.get("score", 0.0),
        deep_result.get("label", ""),
        source_types_str,
    )
    return deep_result
