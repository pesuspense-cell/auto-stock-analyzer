"""
news_logic.py - 뉴스 수집 및 감성 분석 통합 퍼사드

[구성]
  - stock_ai 레거시 함수 재익스포트 (하위 호환)
  - news_async 최적화 파이프라인 재익스포트
  - 다중 소스 신규 API (SOURCE_WEIGHTS, apply_source_weights,
    compute_relevance_scores, filter_by_relevance)
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
    # 파이프라인
    analyze_news_fast,
    # 수집
    fetch_naver_news_fast,
    fetch_multi_news_fast,
    run_async,
    # 3단계 필터
    stage1_title_filter,
    stage2_keyword_filter,
    stage3_select_for_deep,
    # 다중 소스 가중치·관련성 API (신규)
    SOURCE_WEIGHTS,
    apply_source_weights,
    compute_relevance_scores,
    filter_by_relevance,
)

__all__ = [
    # ── 레거시 (stock_ai) ─────────────────────────────────────────────────────
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
    # ── 비동기 파이프라인 ─────────────────────────────────────────────────────
    "analyze_news_fast",
    "fetch_naver_news_fast",
    "fetch_multi_news_fast",
    "run_async",
    "stage1_title_filter",
    "stage2_keyword_filter",
    "stage3_select_for_deep",
    # ── 다중 소스 신규 API ────────────────────────────────────────────────────
    "SOURCE_WEIGHTS",
    "apply_source_weights",
    "compute_relevance_scores",
    "filter_by_relevance",
]
