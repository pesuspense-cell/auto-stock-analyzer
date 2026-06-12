"""
src/ai_report.py — AI 심층 재무분석 리포트 생성기

앱이 이미 수집한 데이터(펀더멘털·기술 신호·뉴스 감성·수급·환율)를
재무분석 전문가 프롬프트에 주입해 LLM(Gemini → Groq 폴백)으로
구조화된 한국어 투자 리포트를 생성한다.

설계 원칙:
- LLM에게 제공된 수치 외 새로운 수치를 창작하지 못하도록 강제한다.
- 데이터가 없는 항목은 '데이터 없음'으로 명시하게 한다 (추측 금지).
- 결론(투자 점수·판정)을 맨 앞에 배치한다.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

_KST = timezone(timedelta(hours=9))

_GEMINI_MODEL = "gemini-2.5-flash"
_GROQ_MODEL   = "llama-3.3-70b-versatile"   # 장문 리포트용 — 8b-instant보다 추론 품질 우선


# ─── 데이터 블록 빌더 ─────────────────────────────────────────────────────────

def _fmt(v, kind: str = "num") -> str:
    """수치 포맷. None/NaN → '데이터 없음'."""
    try:
        import pandas as pd
        if v is None or (hasattr(pd, "isna") and pd.isna(v)):
            return "데이터 없음"
    except Exception:
        if v is None:
            return "데이터 없음"
    try:
        if kind == "money":
            v = float(v)
            if abs(v) >= 1e12:
                return f"{v/1e12:,.2f}조"
            if abs(v) >= 1e8:
                return f"{v/1e8:,.0f}억"
            return f"{v:,.0f}"
        if kind == "pct":            # 0.15 → 15.0%
            return f"{float(v)*100:.1f}%"
        if kind == "pct_raw":        # 15.0 → 15.0%
            return f"{float(v):.1f}%"
        if kind == "x":
            return f"{float(v):.2f}x"
        return f"{float(v):,.2f}"
    except Exception:
        return str(v)


def _build_data_block(
    *,
    ticker: str,
    company_name: str,
    current_price: float | None,
    fund_info: dict,
    fund_score_data: dict,
    signals: dict,
    hybrid: dict,
    news_result: dict,
    rates: dict | None,
    inv_data: dict | None,
) -> str:
    """LLM에 주입할 데이터 블록을 사람이 읽을 수 있는 형태로 직렬화."""
    fi  = fund_info or {}
    fsd = fund_score_data or {}
    is_krw = ticker.upper().endswith((".KS", ".KQ"))
    unit   = "원(KRW)" if is_krw else "달러(USD)"

    lines = [
        f"종목: {company_name} ({ticker})",
        f"기준 통화 단위: {unit}",
        f"현재가: {_fmt(current_price)}" if current_price else "현재가: 데이터 없음",
        "",
        "[밸류에이션·수익성 지표]",
        f"- 시가총액: {_fmt(fi.get('market_cap'), 'money')}",
        f"- PER: {_fmt(fi.get('per'), 'x')}  |  Forward PER: {_fmt(fi.get('forward_pe'), 'x')}",
        f"- PBR: {_fmt(fi.get('pbr'), 'x')}  |  PSR: {_fmt(fi.get('psr'), 'x')}",
        f"- EPS(TTM): {_fmt(fi.get('eps_ttm'))}",
        f"- ROE: {_fmt(fi.get('roe'), 'pct')}",
        f"- 영업이익률: {_fmt(fi.get('operating_margins'), 'pct')}",
        f"- 부채비율(D/E): {_fmt(fi.get('debt_equity'), 'pct_raw')}",
        f"- 매출 성장률(YoY): {_fmt(fi.get('revenue_growth'), 'pct')}",
        f"- 순이익 성장률(YoY): {_fmt(fi.get('earnings_growth'), 'pct')}",
        f"- 잉여현금흐름(FCF): {_fmt(fi.get('free_cashflow'), 'money')}",
        f"- 배당수익률: {_fmt(fi.get('dividend_yield'), 'pct')}",
    ]

    if fsd.get("roe_mean") is not None:
        lines.append(
            f"- ROE 다년 지속성: 평균 {_fmt(fsd.get('roe_mean'), 'pct_raw')}"
            + (f" / 표준편차 {_fmt(fsd.get('roe_std'), 'pct_raw')}p" if fsd.get("roe_std") is not None else "")
        )
    if fsd.get("ocf_ni_ratio") is not None:
        lines.append(f"- OCF/순이익 (이익의 현금 질): {_fmt(fsd.get('ocf_ni_ratio'), 'x')}")
    if fsd.get("shareholder_yield") is not None:
        lines.append(f"- 주주환원율: {_fmt(fsd.get('shareholder_yield'), 'pct_raw')}")

    lines += [
        "",
        "[앱 자체 펀더멘털 스코어링 (참고용)]",
        f"- 장투 점수: {fsd.get('fund_score', 0):+.1f} ({fsd.get('fund_label', 'N/A')})",
        f"- 성장성 {fsd.get('sub_growth', 50):.0f} / 수익성 {fsd.get('sub_profit', 50):.0f}"
        f" / 안정성 {fsd.get('sub_stable', 50):.0f} / 모멘텀 {fsd.get('sub_moment', 50):.0f} (각 100점 만점)",
    ]
    fund_reasons = fsd.get("fund_reasons") or []
    if fund_reasons:
        lines.append("- 스코어링 근거: " + " / ".join(str(r) for r in fund_reasons[:8]))

    sig = signals or {}
    hyb = hybrid or {}
    lines += [
        "",
        "[기술적 신호 (단기 관점, 참고용)]",
        f"- 기술 신호: {sig.get('label', 'N/A')} (점수 {sig.get('score', 0):+.1f})",
        f"- 하이브리드 종합: {hyb.get('label', 'N/A')} (점수 {hyb.get('hybrid_score', 0.0):+.1f})",
    ]

    nr = news_result if isinstance(news_result, dict) else {}
    if nr:
        lines += [
            "",
            "[뉴스 감성 분석]",
            f"- 감성 점수: {nr.get('score', 0.0):+.1f} ({nr.get('sentiment', '중립')})",
        ]
        if nr.get("summary"):
            lines.append(f"- 뉴스 요약: {nr['summary']}")

    if inv_data:
        _frn = inv_data.get("foreign_net")
        _ins = inv_data.get("institution_net")
        if _frn is not None or _ins is not None:
            lines += ["", "[수급 (최근일)]"]
            if _frn is not None:
                lines.append(f"- 외국인 순매수: {_frn:+,.0f}주")
            if _ins is not None:
                lines.append(f"- 기관 순매수: {_ins:+,.0f}주")

    if rates:
        _usd = rates.get("USD/KRW") or rates.get("usdkrw")
        if isinstance(_usd, dict):
            _usd = _usd.get("rate") or _usd.get("value")
        if _usd:
            lines += ["", "[거시 지표]", f"- 원/달러 환율: {_usd}"]

    return "\n".join(lines)


# ─── 규칙 기반 퀵 평가 (리포트 '결론 요약' 로직의 LLM-free 재현) ──────────────

def compute_quick_assessment(
    *,
    fund_info: dict | None = None,
    fund_score_data: dict | None = None,
    hybrid: dict | None = None,
    news_result: dict | None = None,
) -> dict:
    """
    AI 심층 재무분석 리포트의 '## 1. 결론 요약' 판정 로직을 LLM 호출 없이 재현한다.

    리포트와 동일한 분석 틀을 사용한다:
      - 핵심 재무 분석 (그레이엄·버핏·린치 벤치마크 기반 장투 점수 ±8) → 50%
      - 기술적 신호 (하이브리드 종합 ±10, 단기 관점)                  → 30%
      - 뉴스 감성 (±5)                                               → 20%
    재무 데이터가 없으면 기술 60% / 뉴스 40%로 재가중하고 분석 제한을 명시한다.

    Returns:
        {
            "score10":  float,   # 투자 점수 0~10
            "verdict":  str,     # "매수" | "중립" | "매도"
            "reasons":  list,    # 핵심 근거 (최대 3개)
            "summary":  str,     # 한 줄 요약평
            "has_fund": bool,    # 재무 데이터 기반 여부
        }
    """
    fsd = fund_score_data or {}
    hyb = hybrid or {}
    nr  = news_result if isinstance(news_result, dict) else {}

    def _clamp(v: float) -> float:
        return max(-1.0, min(1.0, v))

    fund_score = fsd.get("fund_score")
    has_fund   = isinstance(fund_score, (int, float)) and bool(
        fsd.get("fund_reasons") or fsd.get("master_verdicts")
    )

    h_score    = float(hyb.get("hybrid_score", 0.0) or 0.0)
    news_score = float(nr.get("score", 0.0) or 0.0)
    tech_norm  = _clamp(h_score / 10.0)
    news_norm  = _clamp(news_score / 5.0)

    if has_fund:
        fund_norm = _clamp(float(fund_score) / 8.0)
        composite = fund_norm * 0.5 + tech_norm * 0.3 + news_norm * 0.2
    else:
        composite = tech_norm * 0.6 + news_norm * 0.4

    score10 = round(max(0.0, min(10.0, 5.0 + composite * 5.0)), 1)
    verdict = "매수" if score10 >= 6.5 else ("매도" if score10 <= 3.5 else "중립")

    # ── 핵심 근거 수집 (리포트의 '핵심 근거 3줄' 대응) ───────────────────────
    cand_pos: list[str] = []
    cand_neg: list[str] = []

    mv = fsd.get("master_verdicts") or {}
    for name in ("버핏", "그레이엄", "린치", "오닐"):
        v = mv.get(name)
        if not isinstance(v, dict):
            continue
        icon = v.get("icon", "")
        txt  = f"{name} 기준 {v.get('판정', '')}"
        if icon in ("✅", "🚀", "🔥"):
            cand_pos.append(txt)
        elif icon in ("🚫", "⚠️"):
            cand_neg.append(txt)

    if h_score >= 2.0:
        cand_pos.append(f"기술 신호 강세 ({h_score:+.1f}점)")
    elif h_score <= -2.0:
        cand_neg.append(f"기술 신호 약세 ({h_score:+.1f}점)")

    if news_score >= 1.5:
        cand_pos.append(f"뉴스 감성 긍정 ({news_score:+.1f}점)")
    elif news_score <= -1.5:
        cand_neg.append(f"뉴스 감성 부정 ({news_score:+.1f}점)")

    if verdict == "매수":
        reasons = (cand_pos + cand_neg)[:3]
    elif verdict == "매도":
        reasons = (cand_neg + cand_pos)[:3]
    else:
        reasons = [r for pair in zip(cand_pos, cand_neg) for r in pair]
        reasons = (reasons or cand_pos or cand_neg)[:3]
    if not reasons:
        reasons = ["지표 중립 — 뚜렷한 우위 신호 없음"]

    # ── 한 줄 요약평 ─────────────────────────────────────────────────────────
    if has_fund:
        fund_label = fsd.get("fund_label", "펀더멘털 보통")
        summary = (
            f"그레이엄·버핏·린치 벤치마크 기준 {fund_label}({float(fund_score):+.1f}점), "
            f"기술 신호 {hyb.get('label', '중립')}({h_score:+.1f}점), "
            f"뉴스 감성 {nr.get('sentiment', '중립')}({news_score:+.1f}점)을 종합한 판정입니다."
        )
    else:
        summary = (
            f"재무 데이터 없음 — 기술 신호 {hyb.get('label', '중립')}({h_score:+.1f}점)과 "
            f"뉴스 감성({news_score:+.1f}점)만으로 판정했습니다 (분석 제한)."
        )

    return {
        "score10":  score10,
        "verdict":  verdict,
        "reasons":  reasons,
        "summary":  summary,
        "has_fund": has_fund,
    }


# ─── 프롬프트 ─────────────────────────────────────────────────────────────────

def _build_prompt(data_block: str, company_name: str, as_of: str) -> str:
    return f"""너는 경제·금융 분야 최고 수준의 재무분석 전문가다. 아래 [제공 데이터]만을 근거로 '{company_name}'의 투자 가치를 분석하라. 거시경제 상황(금리·환율·원자재 흐름)에 대한 일반적 지식은 방향성 판단에 활용하되, 구체적 수치를 지어내지 마라.

[절대 원칙]
1. [제공 데이터]에 없는 재무 수치를 절대 창작하거나 추측하지 마라. 없는 항목은 "데이터 없음"으로 명시하고, 해당 분석이 제한됨을 밝혀라.
2. 제공된 수치를 임의로 반올림하거나 단위를 바꾸지 마라. 보고서 전체에서 기준 통화 단위를 일관되게 유지하고, 숫자는 천 단위 구분 기호를 사용하라.
3. 단순 수치 나열이 아니라 숫자 뒤에 숨은 맥락과 의미까지 해석하라.
4. 불필요한 서론 없이 결론부터 시작하라.
5. DCF 등 미래 현금흐름 추정이 필요한 절대가치 평가는 제공 데이터로 불가능하므로 시도하지 마라. PER·PBR·PSR 등 상대가치 지표와 부채비율·현금흐름 기반 안정성 진단에 집중하라.

[보고서 구조 — 순서를 바꾸지 마라. 마크다운으로 작성하라.]

## 1. 결론 요약
- **투자 점수: N/10** 과 함께 **매수 / 중립 / 매도** 중 하나로 명확히 판정하라. 애매한 표현 금지.
- 핵심 근거를 3줄 이내로 요약하라.

## 2. 거시경제·산업 환경
- 현재 금리·환율·산업 사이클이 이 기업에 미치는 영향을 '유리한 요인'과 '불리한 요인'으로 나눠 정리하라. 방향성(유리/불리)을 명확히 판단하라.

## 3. 핵심 재무 분석
- 핵심 수치·비율을 표로 정리하라. 각 지표마다 일반적 기준치(예: 그레이엄 PBR<1, 버핏 ROE≥15% 등) 대비 평가를 함께 써라.
- 전년 대비 변화(성장률 지표)가 의미하는 바를 해석하라.
- 부채비율·현금흐름 기반으로 재무 안정성(부실 위험)을 진단하라.

## 4. 매수 이유 3가지 vs 반대 이유 3가지
- 균형 있게 제시하라. 각 이유마다 [제공 데이터]의 구체적 수치를 반드시 인용하라. 감이 아니라 숫자로 말하라.

## 5. 위험 시나리오와 대응 전략
- 가장 현실적인 위험 시나리오 2~3가지와 각각의 대응 전략을 제시하라. 위험만 나열하고 끝내지 마라.

## 6. 적합 투자자 유형
- 투자 성향·투자 기간·리스크 허용 범위 기준으로 구분하라 (단기 트레이더 / 장기 가치투자자 / 배당 투자자 적합 여부).

## 7. 시장이 틀릴 가능성
- 시장이 이 종목을 과소평가 또는 과대평가하고 있을 가능성이 가장 높은 포인트 1가지를 짚고, 그 판단이 맞을 경우 주가 영향의 방향과 대략적 폭을 추정하라 (추정임을 명시).

## 8. 핵심 질문
- 사용자의 투자 논리를 흔들 수 있는 날카로운 질문 1개를 던져라. 단순 확인 질문 금지.

---
보고서 하단에 다음을 명시하라: "데이터 기준일: {as_of} | 출처: Yahoo Finance·KRX·DART·네이버금융 (앱 수집 데이터)"

[제공 데이터]
{data_block}
"""


# ─── LLM 호출 ─────────────────────────────────────────────────────────────────

def _call_gemini(prompt: str, api_key: str) -> str:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage
    llm = ChatGoogleGenerativeAI(
        model=_GEMINI_MODEL,
        google_api_key=api_key,
        temperature=0.2,
        max_output_tokens=4096,
    )
    return llm.invoke([HumanMessage(content=prompt)]).content.strip()


def _call_groq(prompt: str, api_key: str) -> str:
    from groq import Groq
    client = Groq(api_key=api_key)
    resp = client.chat.completions.create(
        model=_GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4096,
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


# ─── 공개 API ─────────────────────────────────────────────────────────────────

def generate_financial_report(
    *,
    ticker: str,
    company_name: str = "",
    current_price: float | None = None,
    fund_info: dict | None = None,
    fund_score_data: dict | None = None,
    signals: dict | None = None,
    hybrid: dict | None = None,
    news_result: dict | None = None,
    rates: dict | None = None,
    inv_data: dict | None = None,
    gemini_api_key: str = "",
    groq_api_key: str = "",
) -> dict:
    """
    AI 심층 재무분석 리포트를 생성한다.

    Returns:
        {"ok": bool, "report": str(markdown), "provider": str, "error": str}
    """
    if not (gemini_api_key or groq_api_key):
        return {"ok": False, "report": "", "provider": "",
                "error": "Gemini 또는 Groq API 키가 필요합니다. 사이드바에서 설정하세요."}

    as_of = datetime.now(_KST).strftime("%Y-%m-%d %H:%M KST")
    data_block = _build_data_block(
        ticker=ticker,
        company_name=company_name or ticker,
        current_price=current_price,
        fund_info=fund_info or {},
        fund_score_data=fund_score_data or {},
        signals=signals or {},
        hybrid=hybrid or {},
        news_result=news_result or {},
        rates=rates,
        inv_data=inv_data,
    )
    prompt = _build_prompt(data_block, company_name or ticker, as_of)

    gemini_err: Exception | None = None
    if gemini_api_key:
        try:
            report = _call_gemini(prompt, gemini_api_key)
            if report:
                return {"ok": True, "report": report, "provider": "Gemini", "error": ""}
        except Exception as exc:
            gemini_err = exc
            logger.warning("AI 리포트 Gemini 실패: %s", exc)

    if groq_api_key:
        try:
            report = _call_groq(prompt, groq_api_key)
            if report:
                provider = "Groq" + (" (Gemini 폴백)" if gemini_err else "")
                return {"ok": True, "report": report, "provider": provider, "error": ""}
        except Exception as exc:
            logger.warning("AI 리포트 Groq 실패: %s", exc)
            _err = f"Gemini: {gemini_err} / Groq: {exc}" if gemini_err else str(exc)
            return {"ok": False, "report": "", "provider": "", "error": _err}

    return {"ok": False, "report": "", "provider": "",
            "error": str(gemini_err) if gemini_err else "LLM 호출 실패"}
