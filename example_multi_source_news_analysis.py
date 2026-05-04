"""
example_multi_source_news_analysis.py

다중 소스 통합 뉴스 에이전트 사용 예제.

실행: python example_multi_source_news_analysis.py
"""
import asyncio
import logging
from src.news_logic import (
    analyze_news_fast,
    SOURCE_WEIGHTS,
    apply_source_weights,
    compute_relevance_scores,
    filter_by_relevance,
)
from src.news.providers import (
    NaverNewsProvider,
    EconomyRSSProvider,
    YouTubeTranscriptProvider,
)

# ── 로깅 설정 ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 예제 1: 기본 사용법 (기본 Provider 자동 사용)
# ─────────────────────────────────────────────────────────────────────────────
def example_basic():
    """
    가장 간단한 사용법.
    내부적으로 NaverNewsProvider + EconomyRSSProvider + YouTubeTranscriptProvider
    를 자동으로 생성하여 사용.
    """
    logger.info("=" * 80)
    logger.info("예제 1: 기본 사용법")
    logger.info("=" * 80)

    result = analyze_news_fast(
        ticker="005930",
        company_name="삼성전자",
        api_key="YOUR_GEMINI_API_KEY",  # 선택: 임베딩용
        groq_api_key="YOUR_GROQ_API_KEY",  # 선택: YouTube 요약용
        max_news=12,
    )

    logger.info(f"Score: {result['score']:.2f}")
    logger.info(f"Label: {result['label']}")
    logger.info(f"Sources: {result['source_type']}")
    logger.info(f"Avg Relevance: {result['relevance_score']:.2%}")
    logger.info(f"Source Breakdown: {result['source_breakdown']}")
    logger.info(f"\nSummary:\n{result['summary']}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 예제 2: 커스텀 Provider 세트
# ─────────────────────────────────────────────────────────────────────────────
def example_custom_providers():
    """
    특정 소스만 사용하거나, Provider 설정을 커스터마이즈하는 경우.
    """
    logger.info("=" * 80)
    logger.info("예제 2: 커스텀 Provider 세트")
    logger.info("=" * 80)

    # YouTube 채널을 다르게 설정
    providers = [
        NaverNewsProvider(),
        EconomyRSSProvider(),
        YouTubeTranscriptProvider(
            channel_ids=["UCXzpKExCVQv5YjXfqOqNEYQ"],  # SBS Biz
            title_filter="모닝벨",
            max_videos=3,
        ),
    ]

    result = analyze_news_fast(
        ticker="035720",
        company_name="카카오",
        api_key="YOUR_GEMINI_API_KEY",
        providers=providers,
    )

    logger.info(f"Score: {result['score']:.2f}")
    logger.info(f"Event Flags: {result['event_flags']}")


# ─────────────────────────────────────────────────────────────────────────────
# 예제 3: 직접 뉴스 아이템 전달
# ─────────────────────────────────────────────────────────────────────────────
def example_direct_news_items():
    """
    외부에서 수집한 뉴스 아이템을 직접 분석하는 경우.
    기존 소스와의 하위 호환성 유지.
    """
    logger.info("=" * 80)
    logger.info("예제 3: 직접 뉴스 아이템 전달")
    logger.info("=" * 80)

    news_items = [
        {
            "title": "삼성전자 3분기 영업이익 29조원 돌파",
            "link": "https://example.com/news1",
            "publisher": "연합뉴스",
            "pub_date": "2026-05-04",
            "content": "삼성전자가 3분기 영업이익 기대치를 상회했다...",
            "source_type": "naver",
        },
        {
            "title": "반도체 수출 사상 최고 기록",
            "link": "https://example.com/news2",
            "publisher": "매일경제",
            "pub_date": "2026-05-04",
            "content": "올해 반도체 수출이 사상 최고를 기록했다...",
            "source_type": "rss_economy",
        },
    ]

    result = analyze_news_fast(
        ticker="005930",
        company_name="삼성전자",
        api_key="YOUR_GEMINI_API_KEY",
        news_items=news_items,  # 직접 전달
    )

    logger.info(f"Score: {result['score']:.2f}")
    logger.info(f"Label: {result['label']}")


# ─────────────────────────────────────────────────────────────────────────────
# 예제 4: 뉴스-수급-가격 삼각검증
# ─────────────────────────────────────────────────────────────────────────────
def example_triple_verification():
    """
    뉴스 감성이 긍정이지만 주가가 하락하거나 외국인·기관이 순매도하는
    불일치 상황을 감지하는 경우.
    """
    logger.info("=" * 80)
    logger.info("예제 4: 뉴스-수급-가격 삼각검증")
    logger.info("=" * 80)

    result = analyze_news_fast(
        ticker="005930",
        company_name="삼성전자",
        api_key="YOUR_GEMINI_API_KEY",
        # 뉴스 분석 + 수급 데이터로 삼각검증
        price_change_pct=-2.5,  # 당일 -2.5% 하락
        net_foreign_buy=-10_000_000_000,  # 외국인 순매도
        net_institution_buy=-5_000_000_000,  # 기관 순매도
    )

    # 결과: score가 양수(긍정) but event_flags에 "수급_미동반" 추가
    logger.info(f"Score: {result['score']:.2f}")
    logger.info(f"Event Flags: {result['event_flags']}")
    if "수급_미동반" in result["event_flags"]:
        logger.warning(f"⚠️ 경고: 뉴스와 수급/가격 불일치 감지")
        logger.warning(f"Summary:\n{result['summary']}")


# ─────────────────────────────────────────────────────────────────────────────
# 예제 5: 소스별 신뢰도 가중치 확인
# ─────────────────────────────────────────────────────────────────────────────
def example_source_weights():
    """
    각 소스별 신뢰도 가중치 확인 및 수동 조정.
    """
    logger.info("=" * 80)
    logger.info("예제 5: 소스별 신뢰도 가중치")
    logger.info("=" * 80)

    logger.info("기본 SOURCE_WEIGHTS:")
    for source, weight in SOURCE_WEIGHTS.items():
        logger.info(f"  {source:30s}: {weight}x")

    # 수동으로 가중치 조정
    custom_items = [
        {
            "title": "삼성전자 신제품 출시",
            "link": "...",
            "source_type": "youtube_transcript",
            "source_weight": 1.5,  # 전문가 의견
        },
        {
            "title": "삼성 경영진 인터뷰",
            "link": "...",
            "source_type": "rss_economy",
            "source_weight": 1.2,  # 수동 증강
        },
    ]

    logger.info("\nCustom weights applied:")
    for item in custom_items:
        logger.info(
            f"  {item['source_type']:30s}: {item['source_weight']}x"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 예제 6: 단계별 파이프라인 가시화
# ─────────────────────────────────────────────────────────────────────────────
def example_pipeline_visibility():
    """
    analyze_news_fast의 각 단계 로그를 통해 파이프라인 상황을 확인.
    """
    logger.info("=" * 80)
    logger.info("예제 6: 파이프라인 단계별 처리 상황")
    logger.info("=" * 80)

    # 로깅 레벨을 DEBUG로 상향하여 상세 정보 확인
    logging.getLogger("news_async").setLevel(logging.DEBUG)
    logging.getLogger("news.provider").setLevel(logging.DEBUG)

    result = analyze_news_fast(
        ticker="005930",
        company_name="삼성전자",
        api_key="YOUR_GEMINI_API_KEY",
        max_news=12,
        deep_n=5,
    )

    logger.info("\n[최종 결과]")
    logger.info(f"Total Time: 실행 로그 참고")
    logger.info(f"Score: {result['score']:.2f}")
    logger.info(f"Source Breakdown: {result['source_breakdown']}")


# ─────────────────────────────────────────────────────────────────────────────
# 예제 7: 에러 핸들링
# ─────────────────────────────────────────────────────────────────────────────
def example_error_handling():
    """
    외부 API 실패, 뉴스 없음 등 다양한 상황 처리.
    """
    logger.info("=" * 80)
    logger.info("예제 7: 에러 핸들링")
    logger.info("=" * 80)

    # 케이스 1: API 키 없음 → 키워드 기반 분석으로 폴백
    result = analyze_news_fast(
        ticker="005930",
        company_name="삼성전자",
        # api_key는 전달하지 않음 → 임베딩 없이 키워드 기반 필터링
    )
    logger.info("✓ API 키 없음 상황: 키워드 기반으로 정상 분석 완료")
    logger.info(f"  Score: {result['score']:.2f}")

    # 케이스 2: YouTube 설치 없음 → 다른 소스로 계속 진행
    logger.info("\n✓ youtube-transcript-api 미설치 상황:")
    logger.info("  → NaverNewsProvider + EconomyRSSProvider만 사용 계속")

    # 케이스 3: 뉴스 없음 → 빈 결과 반환
    result = analyze_news_fast(
        ticker="000999",  # 존재하지 않는 종목
        company_name="가상회사",
    )
    logger.info(f"\n✓ 뉴스 없음 상황: {result['label']} (score={result['score']})")


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("\n" + "=" * 80)
    logger.info("🔍 Multi-Source News Analysis Agent - 사용 예제")
    logger.info("=" * 80 + "\n")

    # 각 예제 실행 (주석 처리하여 필요한 것만 실행)
    # example_basic()
    # example_custom_providers()
    # example_direct_news_items()
    # example_triple_verification()
    example_source_weights()
    # example_pipeline_visibility()
    # example_error_handling()

    logger.info("\n" + "=" * 80)
    logger.info("✅ 모든 예제 완료")
    logger.info("=" * 80)
