"""fundamental_service.py — 펀더멘털 & 기관 탭 데이터 + AI 심층 리포트.

render_fund_tab 의 데이터 수집부와 src/ai_report.generate_financial_report 를 이식.
"""
from __future__ import annotations

import concurrent.futures
import logging

import pandas as pd

from app import bootstrap  # noqa: F401
from app.core.cache import ttl_cache
from app.services import analysis_service, market_service, stock_lists

from src.fundamental import (
    get_fundamental_data,
    calculate_fundamental_score,
    get_etf_fundamental_data,
    calculate_etf_score,
    get_insider_trades_sec,
)
from src.utils import get_investor_trading_naver, get_investor_trading_naver_history
from src.ai_report import generate_financial_report, compute_quick_assessment

logger = logging.getLogger(__name__)


@ttl_cache(ttl=300)
def investors(ticker: str) -> dict:
    try:
        return get_investor_trading_naver(ticker) or {}
    except Exception:
        return {}


@ttl_cache(ttl=300)
def investor_history(ticker: str) -> list[dict]:
    try:
        return get_investor_trading_naver_history(ticker, days=10) or []
    except Exception:
        return []


@ttl_cache(ttl=3600)
def insiders(ticker: str) -> list[dict]:
    try:
        df = get_insider_trades_sec(ticker)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df.head(30).astype(object).where(df.notna(), None).to_dict("records")
    except Exception:
        pass
    return []


def fundamental(ticker: str) -> dict:
    """펀더멘털 종합 — 일반주/ETF 분기.

    독립적인 외부 호출(시세·재무·수급·내부자)을 ThreadPoolExecutor 로 병렬 수행해
    첫 조회 지연을 줄인다(각 호출은 외부 I/O 대기라 GIL 영향이 작다).
    """
    is_etf = stock_lists.is_etf(ticker)
    is_kr = ticker.endswith((".KS", ".KQ"))

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        f_price = pool.submit(lambda: analysis_service.realtime_price(ticker).get("price", 0.0))
        f_inv = pool.submit(investors, ticker)
        f_hist = pool.submit(investor_history, ticker)

        if is_etf:
            f_etf = pool.submit(get_etf_fundamental_data, ticker)
            etf_data = f_etf.result() or {}
            etf_score = calculate_etf_score(etf_data) or {}
            return {
                "ticker": ticker, "is_etf": True,
                "fund_info": {}, "fund_score_data": {},
                "etf_data": etf_data, "etf_score": etf_score,
                "investors": f_inv.result(),
                "investor_history": f_hist.result(),
                "insiders": [],
            }

        f_fi = pool.submit(analysis_service.fundamental, ticker)
        f_ins = pool.submit(insiders, ticker) if not is_kr else None

        last = f_price.result()
        fi = f_fi.result()
        fsd = calculate_fundamental_score(fi, last) if fi else {}
        return {
            "ticker": ticker, "is_etf": False,
            "fund_info": fi or {}, "fund_score_data": fsd or {},
            "etf_data": {}, "etf_score": {},
            "investors": f_inv.result(),
            "investor_history": f_hist.result(),
            "insiders": f_ins.result() if f_ins else [],
        }


def ai_report(ticker: str, gemini: str, groq: str, use_llm: bool, sname: str = "") -> dict:
    """AI 심층 재무분석 리포트 + 규칙 기반 결론 요약.

    분석 파이프라인(analyze)을 한 번 돌려 hybrid/news/fund 를 수집한 뒤
    generate_financial_report 에 주입한다.
    """
    a = analysis_service.analyze(ticker, "6mo", use_llm, gemini, groq, sname)
    quick = compute_quick_assessment(
        fund_info=a["fund_info"], fund_score_data=a["fund_score_data"],
        hybrid=a["hybrid"], news_result=a["news_result"],
    )

    report = {"ok": False, "report": "", "provider": "", "error": ""}
    if gemini or groq:
        try:
            rates = {r["pair"]: {"rate": r["rate"]} for r in market_service.exchange_rates()}
        except Exception:
            rates = None
        report = generate_financial_report(
            ticker=ticker, company_name=sname or ticker,
            current_price=a["realtime"].get("price"),
            fund_info=a["fund_info"], fund_score_data=a["fund_score_data"],
            signals=a["signals"], hybrid=a["hybrid"], news_result=a["news_result"],
            rates=rates, inv_data=investors(ticker),
            gemini_api_key=gemini, groq_api_key=groq,
        )
    else:
        report["error"] = "Gemini 또는 Groq API 키가 필요합니다."

    return {**report, "quick_assessment": quick}
