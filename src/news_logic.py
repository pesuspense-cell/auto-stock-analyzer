"""
news_logic.py - 뉴스 수집 및 감성 분석 모듈
네이버 뉴스 수집, 선 필터링, 키워드/LLM 기반 감성 분석, 배치 처리 및 캐싱

[성능 최적화 추가]
  - analyze_news_fast  : 3단계 필터 + 비동기 수집 통합 파이프라인
  - fetch_naver_news_fast : httpx 기반 단일 종목 빠른 수집
"""
from stock_ai import (
    get_naver_news,
    analyze_news_sentiment_keywords,
    analyze_news_sentiment_llm,
    analyze_news_batch,
    summarize_article_llm,
    fetch_article_content,
    get_advanced_sentiment,
    get_related_sector_performance,
    get_etf_news_with_holdings,
    analyze_etf_news_sentiment,
)
from src.news_async import (
    analyze_news_fast,
    fetch_naver_news_fast,
    fetch_multi_news_fast,
    run_async,
    stage1_title_filter,
    stage2_keyword_filter,
    stage3_select_for_deep,
)

__all__ = [
    # 기존 함수 (호환성 유지)
    "get_naver_news",
    "analyze_news_sentiment_keywords",
    "analyze_news_sentiment_llm",
    "analyze_news_batch",
    "summarize_article_llm",
    "fetch_article_content",
    "get_advanced_sentiment",
    "get_related_sector_performance",
    "get_etf_news_with_holdings",
    "analyze_etf_news_sentiment",
    # 성능 최적화 신규 함수
    "analyze_news_fast",
    "fetch_naver_news_fast",
    "fetch_multi_news_fast",
    "run_async",
    "stage1_title_filter",
    "stage2_keyword_filter",
    "stage3_select_for_deep",
]
