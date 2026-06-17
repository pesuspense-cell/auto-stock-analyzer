"""news.py — 뉴스 탭 스키마."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class Article(BaseModel):
    title: str = ""
    link: str = "#"
    publisher: str = ""
    pub_date: str = ""


class SentimentDetail(BaseModel):
    title: str = ""
    sentiment: str = "중립"
    score: float = 0.0


class Sentiment(BaseModel):
    sentiment: str = "N/A"
    score: float = 0.0
    summary: str = ""
    detail: list[dict[str, Any]] = []


class EtfMeta(BaseModel):
    sector: str = ""
    holdings: list[str] = []


class NewsResponse(BaseModel):
    ticker: str
    is_etf: bool = False
    articles: list[Article] = []
    sentiment: Sentiment = Sentiment()
    sector_performance: dict[str, Any] = {}
    etf_meta: EtfMeta | None = None


class SummarizeRequest(BaseModel):
    title: str
    link: str
    ticker: str


class SummarizeResponse(BaseModel):
    summary: str = ""
    sentiment: str = "중립"
    score: float = 0.0
    key_points: list[str] = []
    investment_implication: str = ""
