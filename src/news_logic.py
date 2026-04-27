"""
news_logic.py - 뉴스 수집 및 감성 분석 모듈
네이버 뉴스 수집, 선 필터링, 키워드/LLM 기반 감성 분석, 배치 처리 및 캐싱
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

__all__ = [
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
]
