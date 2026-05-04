# 🔍 Multi-Source News Analysis Agent - 아키텍처 & 구현 가이드

**최종 구현 완료**: 2026-05-04  
**아키텍처**: 다중 소스 통합 뉴스 에이전트 with 벡터 기반 관련성 필터링

---

## 📋 목차
1. [아키텍처 개요](#아키텍처-개요)
2. [핵심 컴포넌트](#핵심-컴포넌트)
3. [파이프라인 상세](#파이프라인-상세)
4. [사용 예제](#사용-예제)
5. [성능 목표](#성능-목표)
6. [에러 핸들링](#에러-핸들링)

---

## 🏗️ 아키텍처 개요

### 기본 구조

```
┌─────────────────────────────────────────────┐
│   Multi-Source Integrated News Agent       │
├─────────────────────────────────────────────┤
│                                             │
│  ⚡ Step 1: asyncio.gather() 병렬 수집    │
│  ├─ NaverNewsProvider (weight=0.8)        │
│  ├─ EconomyRSSProvider (weight=1.0)       │
│  └─ YouTubeTranscriptProvider (weight=1.5)│
│                                             │
│  ↓                                          │
│  🎯 Step 1.5-1.7: 정규화 & 필터링         │
│  ├─ source_weight 주입                     │
│  ├─ LLM 임베딩 관련성 필터링                 │
│  └─ YouTube 자막 사전 요약                  │
│                                             │
│  ↓                                          │
│  📊 Stage 1-3: 3단계 계층형 필터           │
│  ├─ 제목 가중치 필터 (관련주 제외)         │
│  ├─ 키워드 즉시처리 + 부정문맥 감지        │
│  └─ LLM 정밀분석 (상위 5건)               │
│                                             │
│  ↓                                          │
│  ✅ Final Result                           │
│  {score, label, source_type,               │
│   relevance_score, source_breakdown, ...}  │
│                                             │
└─────────────────────────────────────────────┘
```

### 신뢰도 가중치

| 소스 | 가중치 | 근거 |
|------|-------|------|
| YouTube Transcript (전문가) | **1.5x** | SBS Biz 모닝벨 등 전문가 해설 |
| 경제지 RSS | **1.0x** | 매일경제, 한국경제 등 신뢰도 높음 |
| 네이버 뉴스 | **0.8x** | 일반 포탈, 혼재 기사 많음 |

---

## 🔧 핵심 컴포넌트

### 1. BaseNewsProvider (추상 기본 클래스)

**위치**: `src/news/providers/base.py`

```python
class BaseNewsProvider(ABC):
    """
    모든 뉴스 제공자가 상속해야 할 공통 인터페이스.
    
    필수 속성:
    - source_type: str          (고유 식별자)
    - source_weight: float      (신뢰도 가중치)
    
    필수 메서드:
    - async fetch(ticker, company_name, max_items, **kwargs)
      → list[dict] 반환 (절대 예외 발생 금지)
    """
    
    def _make_item(self, title, link, publisher, pub_date, content):
        """표준 아이템 생성 (source_type, source_weight 자동 주입)"""
        return {
            "title": title,
            "link": link,
            "publisher": publisher,
            "pub_date": pub_date,
            "content": content,
            "source_type": self.source_type,
            "source_weight": self.source_weight,
            "relevance_score": 0.0,
            "score": 0.0,
        }
```

### 2. NaverNewsProvider

**위치**: `src/news/providers/naver.py`

- 네이버 금융 뉴스 비동기 크롤링
- 403/타임아웃 시 자동 재시도 (최대 2회)
- **source_weight = 0.8**

```python
async def fetch(self, ticker, company_name="", max_items=12):
    # httpx + BeautifulSoup으로 HTML 파싱
    # → [{"title", "link", "publisher", "pub_date", "source_type", ...}]
```

### 3. EconomyRSSProvider

**위치**: `src/news/providers/economy_rss.py`

- 매일경제, 한국경제 등 RSS 피드 수집
- feedparser 우선, ElementTree 폴백
- company_name으로 자동 필터링
- **source_weight = 1.0**

```python
async def fetch(self, ticker, company_name="", max_items=12):
    # asyncio.gather()로 모든 피드 병렬 수집
    # company_name 키워드 매칭
    # → [{"title", "link", "publisher", "pub_date", "content", ...}]
```

### 4. YouTubeTranscriptProvider

**위치**: `src/news/providers/youtube.py`

- YouTube 채널 RSS → 영상 목록 수집 (API 키 불필요)
- `youtube-transcript-api`로 자막 추출 (한국어 우선)
- **자막 길이 > 4,000자 → `_needs_summary=True` 플래그** ← 중요!
- **source_weight = 1.5**

```python
async def fetch(self, ticker, company_name="", max_items=5):
    # 1. YouTube 채널 Atom RSS 수집
    # 2. "모닝벨" 등 제목 필터
    # 3. youtube-transcript-api로 자막 추출
    # 4. 장문 자막에 _needs_summary=True 플래그 설정
    # → [{"title", "link", "content": transcript, "_needs_summary": True/False, ...}]
```

---

## 📊 파이프라인 상세

### Step 1: 다중 소스 병렬 수집 (~1-2초)

```python
async def _collect_all_sources(ticker, company_name, max_news, providers):
    """asyncio.gather()로 모든 Provider 병렬 실행"""
    tasks = [p.fetch(ticker, company_name, max_items=max_news) for p in providers]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    # 예외 처리: 개별 Provider 실패 시에도 시스템 계속 진행
    return merged_results  # 전체 합산
```

**산출물**: `all_items` (총 n건, 소스 혼합)

---

### Step 1.5: 소스 가중치 주입 (<0.01초)

```python
def apply_source_weights(news_items: list[dict]) -> list[dict]:
    """source_type에 따른 가중치를 source_weight 필드로 주입"""
    # SOURCE_WEIGHTS = {
    #     "youtube_transcript": 1.5,
    #     "economy_rss": 1.0,
    #     "naver": 0.8,
    #     "unknown": 0.8,
    # }
    return [
        {**item, "source_weight": SOURCE_WEIGHTS.get(item["source_type"], 0.8)}
        for item in news_items
    ]
```

**산출물**: `news_items` (source_weight 필드 추가)

---

### Step 1.6: LLM 임베딩 기반 관련성 필터링 (~2-3초)

```python
def compute_relevance_scores(
    news_items: list[dict],
    company_name: str,
    api_key: str = "",
) -> list[dict]:
    """
    종목 프로필과 각 뉴스의 코사인 유사도 계산.
    
    - API Key 있음: Google Generative AI embedding-001 사용
    - API Key 없음: 키워드 기반 폴백 (company_name 매칭)
    - 실패 시: 키워드 폴백 (시스템 중단 없음)
    """
    # 프로필: "삼성전자 주식 투자 관련 핵심 뉴스 및 사업 동향"
    # 각 뉴스: title + content 합산
    # → 코사인 유사도 계산
    # → relevance_score (0~1) 주입
```

**필터링**:

```python
def filter_by_relevance(
    news_items: list[dict],
    threshold: float = 0.30,
) -> list[dict]:
    """relevance_score < 0.30인 항목 제거"""
    passed = [it for it in news_items if it.get("relevance_score", 1.0) >= threshold]
    # 전체 필터링 시 상위 3건 보존 (안전 폴백)
    return passed
```

**산출물**: `news_items` (relevance_score 필드 추가, 무관 기사 제거)

---

### Step 1.7: YouTube 자막 사전 요약 (~2-4초, _needs_summary 항목만)

```python
async def _summarize_youtube_items(
    items: list[dict],
    api_key: str,
    groq_api_key: str,
) -> list[dict]:
    """
    _needs_summary=True인 YouTube 자막을 LLM으로 요약.
    
    프롬프트: "주요 투자 아이디어, 종목 언급, 시장 전망만 핵심적으로 추출하여 
               300자 이내로 요약"
    
    LLM 선택: Gemini (우선) → Groq llama3-8b (폴백) → 앞 1,000자 절삭
    """
    # 병렬 처리 (asyncio.gather)
    # 실패 시 원본 유지 (시스템 중단 없음)
    return summarized_items
```

**산출물**: `news_items` (content 필드 300자로 정규화)

---

### Step 2-3: 기존 3단계 필터 파이프라인 (~0.1초)

#### Stage 1: 제목 가중치 필터

```python
def stage1_title_filter(news_items: list[dict], company_name: str):
    """
    _title_weight 부여 + 관련주/테마주 단순 나열 제외.
    
    _title_weight:
    - 2.0: 제목에 회사명 + 핵심 키워드(실적, 수주 등) 동시 포함
    - 1.0: 제목에 회사명만 포함
    - 0.5: 제목에 회사명 없음 (본문 언급 추정)
    - 제외: "관련주", "테마주" 등 패턴
    """
```

#### Stage 2: 키워드 즉시처리 + 부정문맥 감지

```python
def stage2_keyword_filter(news_items: list[dict]):
    """
    긍정/부정 키워드 단일 방향만 있으면 즉시 점수 확정.
    부정 키워드 주변 5어절 내 부정 어구 감지 시 LLM 정밀 판단으로 격하.
    """
    return deep_candidates, pre_scored
    # deep_candidates: LLM 정밀분석 대상
    # pre_scored: 즉시 점수 확정 (_fast_score, _fast_reason)
```

#### Stage 3: LLM 정밀분석 (상위 5건)

```python
def stage3_select_for_deep(candidates, n=5):
    """LLM 호출 비용 절감을 위해 _title_weight 상위 n건만 선택"""
    return candidates[:n]
```

---

### Step 3.5: DART 공시 교차검증 (옵션)

수주/계약 키워드 뉴스에 대해 최근 30일 DART 공시 확인.

---

### Step 4-6: 결과 병합 & 삼각검증 (~2-4초)

최종 점수 = 0.7 × LLM점수 + 0.3 × 키워드점수

뉴스-수급-가격 불일치 시 "수급_미동반" 플래그 추가.

---

## 💡 사용 예제

### 기본 사용법

```python
from src.news_logic import analyze_news_fast

# 간단한 호출
result = analyze_news_fast(
    ticker="005930",
    company_name="삼성전자",
    api_key="YOUR_GEMINI_API_KEY",  # 임베딩용
    groq_api_key="YOUR_GROQ_API_KEY",  # YouTube 요약용
)

print(result)
# {
#     "score": 2.3,
#     "label": "긍정",
#     "detail": [...],
#     "event_flags": [],
#     "summary": "...",
#     "source_type": "economy_rss,naver,youtube_transcript",
#     "relevance_score": 0.62,  # 평균 관련성
#     "source_breakdown": {
#         "economy_rss": 5,
#         "naver": 4,
#         "youtube_transcript": 2
#     }
# }
```

### 커스텀 Provider 사용

```python
from src.news.providers import (
    NaverNewsProvider,
    EconomyRSSProvider,
    YouTubeTranscriptProvider,
)

providers = [
    NaverNewsProvider(),
    EconomyRSSProvider(),
    YouTubeTranscriptProvider(
        channel_ids=["YOUR_CHANNEL_ID"],
        title_filter="모닝벨",
        max_videos=3,
    ),
]

result = analyze_news_fast(
    ticker="005930",
    company_name="삼성전자",
    api_key="YOUR_GEMINI_API_KEY",
    providers=providers,  # 커스텀 Provider 세트
)
```

### Streamlit 통합

```python
import streamlit as st
from src.news_logic import analyze_news_fast

ticker = st.text_input("종목 코드")
company_name = st.text_input("회사명")

if st.button("뉴스 분석"):
    # analyze_news_fast는 내부적으로 run_async()를 사용하여
    # Streamlit의 Tornado 이벤트 루프 충돌 방지
    result = analyze_news_fast(
        ticker=ticker,
        company_name=company_name,
        api_key=st.secrets["gemini_api_key"],
        groq_api_key=st.secrets["groq_api_key"],
    )
    st.metric("뉴스 감성", result["label"])
    st.write(f"Score: {result['score']:.2f}")
    st.write(f"Relevance: {result['relevance_score']:.2%}")
    st.json(result["source_breakdown"])
```

---

## ⚡ 성능 목표

| 단계 | 예상 시간 | 병목 |
|------|----------|------|
| Step 1 (병렬 수집) | 1-2s | 네이버 크롤링 |
| Step 1.5-1.6 (임베딩 + 필터) | 2-3s | Gemini API |
| Step 1.7 (YouTube 요약) | 2-4s (선택) | Gemini/Groq LLM |
| Step 2-3 (필터) | <0.1s | 로컬 계산 |
| Step 4-6 (LLM 분석) | 2-4s | Gemini/Groq LLM |
| **총합** | **5-15s** | - |

### 기존 대비 개선

| 항목 | 기존 (단독 네이버) | 신규 (다중 소스) |
|------|-----------------|-----------------|
| 뉴스 건수 | 5-10건 | 15-25건 |
| 신뢰도 | 단일 소스 편향 | 가중 평균 |
| 관련성 | 키워드만 | 벡터 기반 |
| 소요 시간 | ~30s | ~5-15s |

---

## 🛡️ 에러 핸들링

### 설계 원칙

**"개별 Provider 실패 시에도 시스템은 멈추지 않는다"**

```python
# 예시 1: 유튜브 자막 없음
try:
    items = await YouTubeTranscriptProvider().fetch(...)
except Exception:
    logger.warning("YouTube 소스 건너뜀")
    items = []  # 빈 리스트 반환 → 다른 소스로 계속 진행

# 예시 2: RSS 파싱 실패
try:
    parsed = feedparser.parse(xml_text)
except Exception:
    logger.warning("feedparser 폴백 → ElementTree")
    parsed = self._parse_etree(xml_text)  # 폴백

# 예시 3: 임베딩 API 오류
try:
    relevance_scores = compute_relevance_scores(items, company_name, api_key)
except Exception:
    logger.warning("임베딩 실패 → 키워드 폴백")
    relevance_scores = _relevance_keyword(items, needs, company_name)
```

### 외부 의존성 검증

```python
if not HAS_HTTPX:
    logger.warning("httpx 미설치 — 비동기 수집 불가")
    return []

if not HAS_FEEDPARSER:
    logger.warning("feedparser 미설치 — ElementTree 폴백")
    return self._parse_etree(xml_text)

try:
    from youtube_transcript_api import YouTubeTranscriptApi
except ImportError:
    logger.warning("youtube-transcript-api 미설치")
    return []
```

---

## 📦 설치 & 의존성

### requirements.txt

```
# 기존
httpx>=0.27.0
beautifulsoup4>=4.12.0
feedparser>=6.0.0

# 신규
youtube-transcript-api>=0.6.0
langchain-core>=0.3.0
langchain-google-genai>=2.0.0
groq>=0.9.0
```

### 설치

```bash
pip install -r requirements.txt
```

---

## 🔑 API 키 설정

### Streamlit (권장)

```bash
# .streamlit/secrets.toml
gemini_api_key = "YOUR_GEMINI_API_KEY"
groq_api_key = "YOUR_GROQ_API_KEY"
dart_api_key = "YOUR_DART_API_KEY"  # 선택
```

### 환경 변수

```bash
export GEMINI_API_KEY="..."
export GROQ_API_KEY="..."
```

---

## ✅ 검증 체크리스트

- [x] BaseNewsProvider 추상 클래스
- [x] NaverNewsProvider 구현 (weight=0.8)
- [x] EconomyRSSProvider 구현 (weight=1.0)
- [x] YouTubeTranscriptProvider 구현 (weight=1.5)
- [x] asyncio.gather() 병렬 수집
- [x] LLM 임베딩 기반 관련성 필터 (코사인 유사도)
- [x] YouTube 자막 LLM 사전 요약
- [x] 소스 가중치 통합
- [x] 하위 호환성 유지 (analyze_news_sentiment 인터페이스)
- [x] 예외 처리 (개별 소스 실패 시 계속 진행)
- [x] 모든 분석 데이터 source_type + relevance_score 포함
- [x] requirements.txt 업데이트

---

## 📞 문의 & 개선사항

### 향후 가능한 개선사항

1. **다중 언어 지원**: 영문, 일문, 중문 뉴스 소스 추가
2. **실시간 스트리밍**: WebSocket 기반 실시간 뉴스 수집
3. **주제별 분류**: 뉴스를 "실적", "상품", "규제" 등으로 자동 분류
4. **감정 분석 고도화**: 감정 강도(매우 강함~약함) 세분화
5. **캐싱 레이어**: Redis 기반 최근 분석 캐시

---

**End of Documentation**
