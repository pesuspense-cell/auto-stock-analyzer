"""
ui/layouts.py — 섹션·탭별 렌더링 함수 모음

각 함수는 명시적 파라미터를 통해 필요한 데이터를 받아 Streamlit UI를 구성합니다.
비즈니스 로직은 src/ 모듈에서 처리하고 결과만 받아 표시합니다.
"""
from __future__ import annotations

import queue
import time
import concurrent.futures
import os
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf

from ui.components import (
    loading_card_html,
    rate_card_html,
    header_metric_card_html,
    placeholder_card_html,
    watchlist_item_html,
    sentiment_badge_html,
    halted_banner_html,
    signal_report_html,
    stock_badge_html,
    SVG_WALLET, SVG_BAR_CHART, SVG_TREND,
)
from ui.styles import COLORS, RADIUS

# src 비즈니스 로직 — 차트·신호 계산에 직접 사용
from src.indicators import (
    generate_signals, get_stop_loss_targets,
    get_buy_target_price, get_sell_target_price,
    get_advanced_analysis, calculate_vpvr,
    check_volume_anomaly, check_dead_time,
    check_breakout_signal, adjust_risk_conservative,
)
from src.fundamental import (
    get_investment_recommendation,
    calculate_etf_score,
)
from src.news_logic import (
    analyze_news_sentiment_keywords,
    analyze_news_sentiment_llm,
    summarize_article_llm,
    get_etf_news_with_holdings,
    analyze_etf_news_sentiment,
    analyze_portfolio_news,
)
from src.utils import _flatten_columns


# ═════════════════════════════════════════════════════════════════════════════
# 다이얼로그 (모달)
# ═════════════════════════════════════════════════════════════════════════════

@st.dialog("📰 기사 AI 요약", width="large")
def article_dialog(
    title: str,
    link: str,
    ticker_sym: str,
    api_key: str,
    groq_key: str = "",
) -> None:
    """뉴스 기사 클릭 시 팝업되는 AI 요약 모달."""
    st.markdown(f"### {title}")
    if link and link != "#":
        st.markdown(f"[🔗 원문 기사 열기]({link})")
    st.divider()

    if not api_key:
        st.warning(
            "AI 요약을 사용하려면 사이드바에서 **Gemini API 키**를 입력하세요.  \n"
            "Google AI Studio(aistudio.google.com)에서 무료로 발급받을 수 있습니다."
        )
        return

    with st.spinner("AI가 기사를 분석 중입니다..."):
        result = summarize_article_llm(title, link, ticker_sym, api_key, groq_key)

    senti = result.get("sentiment", "N/A")
    score = result.get("score", 0.0)
    st.markdown(sentiment_badge_html(senti, score), unsafe_allow_html=True)
    st.markdown("")

    summary = result.get("summary", "")
    if summary:
        st.markdown("**📋 핵심 요약**")
        st.info(summary)

    key_points = result.get("key_points", [])
    if key_points:
        st.markdown("**🔑 핵심 포인트**")
        for pt in key_points:
            st.markdown(f"- {pt}")

    implication = result.get("investment_implication", "")
    if implication:
        st.markdown("**💡 투자 시사점**")
        st.success(implication)

    if not result.get("used_content"):
        st.caption("⚠️ 기사 본문 스크래핑 불가 — 제목 기반으로 분석했습니다.")


@st.dialog("📊 차트 분석", width="large")
def chart_dialog(fig: go.Figure, ticker_sym: str) -> None:
    """차트를 모달 창에서 렌더링한다."""
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "scrollZoom": False,
            "displayModeBar": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
            "modeBarButtonsToAdd":    ["pan2d", "zoomIn2d", "zoomOut2d"],
            "toImageButtonOptions": {
                "format": "png",
                "filename": f"{ticker_sym}_chart",
                "scale": 2,
            },
        },
    )


# ═════════════════════════════════════════════════════════════════════════════
# 매매신호 종합 판정 (비즈니스 로직 + 결과 메시지 생성)
# ═════════════════════════════════════════════════════════════════════════════

def generate_signal(
    data: pd.DataFrame,
    advanced: dict,
    hybrid: dict,
    news_result: dict,
    expected,
    signals: dict,
) -> tuple[str, str, list]:
    """신호등 3단 판정: BUY / WAIT / SELL + 액션 메시지 + 근거 리스트 반환."""
    if data.empty or len(data) < 2:
        return "WAIT", "데이터가 부족합니다. 관망을 권장합니다.", ["데이터 부족"]

    _close_raw = data["Close"]
    close = _close_raw.iloc[:, 0] if isinstance(_close_raw, pd.DataFrame) else _close_raw
    try:
        current = float(close.iloc[-1])
    except (IndexError, TypeError, ValueError):
        return "WAIT", "데이터가 부족합니다. 관망을 권장합니다.", ["데이터 부족"]

    trend_score  = advanced.get("trend_score",  50.0)
    volume_score = advanced.get("volume_score", 50.0)
    news_score   = news_result.get("score", 0.0) if isinstance(news_result, dict) else 0.0
    h_score      = hybrid.get("hybrid_score", 0.0)

    ema_downtrend = False
    if all(c in data.columns for c in ["EMA_20", "EMA_50", "EMA_200"]):
        _row = data.iloc[-1]
        _e20  = float(_row["EMA_20"])  if pd.notna(_row.get("EMA_20"))  else None
        _e50  = float(_row["EMA_50"])  if pd.notna(_row.get("EMA_50"))  else None
        _e200 = float(_row["EMA_200"]) if pd.notna(_row.get("EMA_200")) else None
        if all(v is not None for v in [_e20, _e50, _e200]):
            ema_downtrend = _e20 < _e50 < _e200

    stop_loss_breached = False
    _sl = get_stop_loss_targets(data)
    if _sl:
        stop_loss_breached = current < _sl["stop_8pct"]

    vwap_above = False
    if "VWAP_M" in data.columns:
        _vm = data["VWAP_M"].iloc[-1]
        if pd.notna(_vm):
            vwap_above = current > float(_vm)

    vol_ratio = 1.0
    if "Volume_MA20" in data.columns and len(data) >= 1:
        _vma = float(data["Volume_MA20"].iloc[-1])
        if _vma > 0 and pd.notna(_vma):
            vol_ratio = float(data["Volume"].iloc[-1]) / _vma

    reasons = []

    # ── SELL 조건 ─────────────────────────────────────────────────────────
    if stop_loss_breached:
        reasons.append("현재가가 손절 기준선(매수가 -8%) 아래 — 즉시 손절 원칙")
    if h_score <= -2.5:
        reasons.append(f"하이브리드 신호 약세 ({h_score:+.1f}점)")
    if news_score <= -2.0:
        reasons.append(f"뉴스 감성 부정 ({news_score:+.1f}점)")
    if trend_score <= 35:
        reasons.append(f"추세 약세 ({trend_score:.0f}점)")
    if ema_downtrend:
        reasons.append("EMA 역배열 (단기<중기<장기) 하락 추세 확인")

    sell_weight = (
        (2 if stop_loss_breached else 0)
        + (1 if h_score <= -2.5 else 0)
        + (1 if news_score <= -2.0 else 0)
        + (1 if trend_score <= 35 else 0)
        + (1 if ema_downtrend else 0)
    )

    # ── BUY 조건 ──────────────────────────────────────────────────────────
    buy_reasons = []
    if h_score >= 2.0:
        buy_reasons.append(f"하이브리드 매수 신호 강함 ({h_score:+.1f}점)")
    if news_score >= 1.5:
        buy_reasons.append(f"뉴스 감성 긍정 ({news_score:+.1f}점)")
    if trend_score >= 60:
        buy_reasons.append(f"추세 강세 ({trend_score:.0f}점)")
    if vwap_above:
        buy_reasons.append("현재가 VWAP(월간) 상단 — 수급 강세")
    if vol_ratio >= 1.5:
        buy_reasons.append(f"거래량 급증 ({vol_ratio:.1f}배) — 세력 개입 가능")
    if volume_score >= 65:
        buy_reasons.append(f"에너지(거래량) 지표 강세 ({volume_score:.0f}점)")

    buy_weight = len(buy_reasons)

    # ── 판정 ──────────────────────────────────────────────────────────────
    if sell_weight >= 3 or stop_loss_breached:
        final = "SELL"
        action = (
            "복수의 약세 신호가 동시 발생했습니다. "
            "손절 원칙에 따라 포지션 청산 또는 축소를 검토하세요. "
            "반등이 와도 추가 매수보다 리스크 관리를 우선하세요."
        )
        final_reasons = reasons[:3]
    elif buy_weight >= 3 and sell_weight == 0:
        final = "BUY"
        action = (
            "다수의 강세 신호가 동시 확인됩니다. "
            "리스크 대비 수익 비율을 확인 후 분할 매수를 고려하세요. "
            "손절선을 미리 설정하고 포지션 크기를 조절하세요."
        )
        final_reasons = buy_reasons[:3]
    elif buy_weight >= 2 and sell_weight == 0:
        final = "BUY"
        action = (
            "부분적인 매수 신호가 확인됩니다. "
            "전체 매수보다 소량 선취매 후 추가 신호를 확인하는 전략을 권장합니다."
        )
        final_reasons = buy_reasons[:3]
    else:
        final = "WAIT"
        action = (
            "명확한 매수/매도 신호가 없습니다. "
            "추가 데이터 확인 후 방향을 결정하세요. "
            "변동성이 큰 구간에서는 관망이 최선의 전략입니다."
        )
        all_reasons = (buy_reasons + reasons)
        final_reasons = all_reasons[:3] if all_reasons else ["지표 중립 — 방향 불확실"]

    return final, action, final_reasons


# ═════════════════════════════════════════════════════════════════════════════
# 사이드바
# ═════════════════════════════════════════════════════════════════════════════

def render_sidebar(
    *,
    all_stocks: dict,
    rates: dict,
    indices: dict,
    get_index_data,
    watchlist: list,
    save_watchlist_fn,
    get_wl_alerts_fn,
    get_wl_price_fn,
    saved_settings: dict,
    load_settings_fn,
    save_settings_fn,
    db_upsert_portfolio,
    krx_stocks_fn,
    etf_stocks_fn,
    us_stocks_fn,
    now_kst_fn,
) -> dict:
    """
    사이드바 전체를 렌더링한다.

    Returns:
        dict with keys: ticker, sname, period, gemini_api_key, groq_api_key,
                        dart_api_key, use_llm, krx_ok
    """
    result: dict = {}

    def _clear_analysis():
        st.session_state.pop("analyzed_ticker", None)
        st.session_state.pop("analyzed_sname",  None)
        st.session_state.pop("analyzed_period", None)

    with st.sidebar:
        st.markdown("## ⚙️ 종목 설정")

        if all_stocks:
            _unified_q = st.text_input(
                "종목 검색",
                key="_unified_q",
                placeholder="종목명, 코드, 티커 입력 (예: 삼성전자, AAPL, KODEX 200)",
            )
            _unified_opts = (
                {k: v for k, v in all_stocks.items() if _unified_q.lower() in k.lower()}
                if _unified_q else all_stocks
            )
            if not _unified_opts:
                st.caption("검색 결과 없음 — 전체 목록 표시")
                _unified_opts = all_stocks
            selected = st.selectbox(
                f"종목 선택 ({len(_unified_opts):,}개)",
                list(_unified_opts.keys()),
                key="_unified_selected",
                on_change=_clear_analysis,
            )
            ticker = _unified_opts[selected]
            sname  = selected.split(" (")[0]
        else:
            st.warning("종목 목록 로드 실패 — 티커를 직접 입력하세요")
            ticker = st.text_input(
                "티커 직접 입력",
                value=st.session_state.get("_direct_ticker_input", "005930.KS"),
                key="_direct_ticker_input",
                on_change=_clear_analysis,
                help="예) 005930.KS (KOSPI), 247540.KQ (KOSDAQ), AAPL (미국)",
            )
            sname = ticker

        period = st.selectbox(
            "분석 기간",
            ["1mo", "3mo", "6mo", "1y", "2y"],
            index=1,
            key="_period_sel",
            on_change=_clear_analysis,
        )

        st.divider()
        if st.button("🔍 종목 분석 시작", type="primary", use_container_width=True):
            st.session_state.pop("analyzed_ticker", None)
            st.session_state["_pending_ticker"] = ticker
            st.session_state["_pending_sname"]  = sname
            st.session_state["_pending_period"] = period
            st.rerun()

        # ── API 키 ──────────────────────────────────────────────────────────
        st.divider()
        try:
            _gem_secret = st.secrets.get("GEMINI_API_KEY", "")
        except Exception:
            _gem_secret = ""
        gemini_api_key = _gem_secret or st.session_state.get("gemini_api_key", "")

        try:
            _groq_secret = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            _groq_secret = ""
        groq_api_key = _groq_secret or st.session_state.get("groq_api_key", "")

        try:
            _dart_secret = st.secrets.get("DART_API_KEY", "")
        except Exception:
            _dart_secret = ""
        dart_api_key = _dart_secret or st.session_state.get("dart_api_key", "")

        _krx_ok = bool(os.environ.get("KRX_ID") and os.environ.get("KRX_PW"))
        use_llm = bool(gemini_api_key or groq_api_key)

        _cur_gemini = st.session_state.get("gemini_api_key", "")
        _cur_groq   = st.session_state.get("groq_api_key", "")
        _cur_dart   = st.session_state.get("dart_api_key", "")
        _cur_krx_id = saved_settings.get("krx_id", "")
        _cur_krx_pw = saved_settings.get("krx_pw", "")

        def _key_hint(v: str) -> str:
            if not v:
                return ""
            return f"{v[:4]}{'*' * (len(v) - 4)}" if len(v) > 4 else "****"

        _any_missing = not (gemini_api_key and groq_api_key and dart_api_key and _krx_ok)
        with st.expander("🔧 API 키 설정", expanded=_any_missing):
            with st.form("api_key_form"):
                st.caption("저장된 키는 자동 로드됩니다. 변경할 키만 입력하세요.")
                _new_gemini = st.text_input(
                    "🤖 Gemini API 키" + (" ✅" if _cur_gemini else " ⚠️"),
                    placeholder="저장됨 — 변경하려면 입력" if _cur_gemini else "AIza...",
                    type="password",
                    help=f"현재: {_key_hint(_cur_gemini)}" if _cur_gemini else "미설정",
                )
                _new_groq = st.text_input(
                    "🦙 Groq API 키" + (" ✅" if _cur_groq else " ⚠️"),
                    placeholder="저장됨 — 변경하려면 입력" if _cur_groq else "gsk_...",
                    type="password",
                    help=f"현재: {_key_hint(_cur_groq)}" if _cur_groq else "미설정",
                )
                _new_dart = st.text_input(
                    "📑 DART API 키" + (" ✅" if _cur_dart else " ⚠️"),
                    placeholder="저장됨 — 변경하려면 입력" if _cur_dart else "DART OpenAPI 키",
                    type="password",
                    help=f"현재: {_key_hint(_cur_dart)}" if _cur_dart else "미설정",
                )
                _new_krx_id = st.text_input(
                    "📊 KRX 아이디" + (" ✅" if _cur_krx_id else " ⚠️"),
                    placeholder="저장됨 — 변경하려면 입력" if _cur_krx_id else "KRX 로그인 ID",
                    help=f"현재: {_key_hint(_cur_krx_id)}" if _cur_krx_id else "미설정",
                )
                _new_krx_pw = st.text_input(
                    "📊 KRX 비밀번호" + (" ✅" if _cur_krx_pw else " ⚠️"),
                    placeholder="저장됨 — 변경하려면 입력" if _cur_krx_pw else "KRX 로그인 PW",
                    type="password",
                    help=f"현재: {_key_hint(_cur_krx_pw)}" if _cur_krx_pw else "미설정",
                )
                if st.form_submit_button("💾 저장", use_container_width=True, type="primary"):
                    _to_save = {**load_settings_fn()}
                    _to_save["gemini_api_key"] = _new_gemini or _cur_gemini
                    _to_save["groq_api_key"]   = _new_groq   or _cur_groq
                    _to_save["dart_api_key"]   = _new_dart   or _cur_dart
                    _to_save["krx_id"]         = _new_krx_id or _cur_krx_id
                    _to_save["krx_pw"]         = _new_krx_pw or _cur_krx_pw
                    save_settings_fn(_to_save)
                    st.session_state["gemini_api_key"] = _to_save["gemini_api_key"]
                    st.session_state["groq_api_key"]   = _to_save["groq_api_key"]
                    st.session_state["dart_api_key"]   = _to_save["dart_api_key"]
                    if _to_save["krx_id"]:
                        os.environ["KRX_ID"] = _to_save["krx_id"]
                    if _to_save["krx_pw"]:
                        os.environ["KRX_PW"] = _to_save["krx_pw"]
                    st.success("저장되었습니다!")
                    st.rerun()

        # ── 환율 ────────────────────────────────────────────────────────────
        st.divider()
        st.markdown("### 💱 실시간 환율")
        for pair, info in rates.items():
            st.metric(
                pair,
                f"{info['rate']:,.2f}",
                f"{info['change']:+.3f}%",
                help=f"{pair} 환율. 전일 대비 변동률 표시.",
            )

        # ── 주요 지수 ────────────────────────────────────────────────────────
        st.divider()
        st.markdown("### 📊 주요 지수")
        for idx_name, idx_sym in indices.items():
            try:
                d = get_index_data(idx_sym)
                if len(d) >= 2:
                    p   = float(d["Close"].iloc[-1])
                    chg = (p - float(d["Close"].iloc[-2])) / float(d["Close"].iloc[-2]) * 100
                    st.metric(idx_name, f"{p:,.2f}", f"{chg:+.2f}%",
                              help=f"{idx_name} 지수. 전일 종가 대비 등락률.")
            except Exception:
                pass

        # ── 관심종목 ──────────────────────────────────────────────────────────
        st.divider()
        st.markdown("### ⭐ 관심 종목")
        wl = watchlist
        if not wl:
            st.caption("아직 추가된 관심종목이 없습니다.")
        else:
            wl_alerts = get_wl_alerts_fn()
            buy_alerts  = [a for a in wl_alerts if a["score"] > 0]
            sell_alerts = [a for a in wl_alerts if a["score"] < 0]
            if buy_alerts:
                names = ", ".join(a["name"] for a in buy_alerts)
                st.success(f"🔔 매수 신호: **{names}**", icon="🔺")
            if sell_alerts:
                names = ", ".join(a["name"] for a in sell_alerts)
                st.error(f"🔔 매도 신호: **{names}**", icon="🔻")

            for i, item in enumerate(wl):
                col_a, col_b = st.columns([4, 1])
                with col_a:
                    try:
                        p2, chg2 = get_wl_price_fn(item["ticker"])
                        if chg2 is not None:
                            st.markdown(
                                watchlist_item_html(item["name"], chg2),
                                unsafe_allow_html=True,
                            )
                        else:
                            st.caption(item["name"])
                    except Exception:
                        st.caption(item["name"])
                with col_b:
                    if st.button("✕", key=f"rm_{i}", help="관심종목 삭제"):
                        st.session_state.watchlist.pop(i)
                        save_watchlist_fn(st.session_state.watchlist)
                        st.rerun()

        # ── 포트폴리오 종목 추가 (로그인 시만) ──────────────────────────────
        _sb_uid = st.session_state.get("auth_user_id")
        _sb_tok = st.session_state.get("auth_token")
        if _sb_tok and _sb_uid:
            st.divider()
            st.markdown("""
<div style="font-size:.85rem;font-weight:700;color:#8B5CF6;letter-spacing:.5px;margin-bottom:4px">
  ➕ 포트폴리오 종목 추가
</div>
""", unsafe_allow_html=True)
            with st.expander("종목 추가", expanded=False):
                _sb_ticker_val = ""
                _sb_all: dict = {}
                _sb_all.update(krx_stocks_fn() or {})
                _sb_all.update(etf_stocks_fn() or {})
                _sb_all.update(us_stocks_fn() or {})
                if _sb_all:
                    _sbq = st.text_input(
                        "종목 검색",
                        key="sb_add_q",
                        placeholder="종목명, 코드, 티커 입력 (예: 삼성전자, AAPL, KODEX 200)",
                    )
                    _sb_opts = (
                        {k: v for k, v in _sb_all.items() if _sbq.lower() in k.lower()}
                        if _sbq else _sb_all
                    ) or _sb_all
                    _sb_sel = st.selectbox(
                        f"선택 ({len(_sb_opts):,}개)",
                        list(_sb_opts.keys()),
                        key="sb_add_unified",
                    )
                    _sb_ticker_val = _sb_opts[_sb_sel]

                if _sb_ticker_val:
                    _sb_is_krw = _sb_ticker_val.upper().endswith((".KS", ".KQ"))
                    _sb_c1, _sb_c2 = st.columns([2, 1])
                    _sb_price = _sb_c1.number_input(
                        "평단가", min_value=0.01, value=1.0,
                        step=100.0 if _sb_is_krw else 0.01,
                        format="%.0f" if _sb_is_krw else "%.2f",
                        key="sb_add_price",
                    )
                    _sb_qty = _sb_c2.number_input(
                        "수량", min_value=0.01, value=1.0, step=1.0,
                        format="%.2f", key="sb_add_qty",
                    )
                    if st.button("포트폴리오에 추가", use_container_width=True,
                                 type="primary", key="sb_add_btn"):
                        _sb_r = db_upsert_portfolio(_sb_uid, _sb_ticker_val, _sb_price, _sb_qty)
                        if _sb_r.get("merged"):
                            st.toast("추가 매수 완료 — 평단가 자동 합산")
                        else:
                            st.toast(f"{_sb_ticker_val} 포트폴리오에 추가됐습니다.")
                        st.rerun()

        # ── 시스템 연동 상태 (비활성 항목만) ─────────────────────────────────
        _inactive = []
        if not gemini_api_key:
            _inactive.append(("🤖", "Gemini API 미설정", "키워드 분석 모드로 동작"))
        if not groq_api_key:
            _inactive.append(("🦙", "Groq API 미설정", "Gemini 폴백 불가"))
        if not dart_api_key:
            _inactive.append(("📑", "DART API 미설정", "yfinance로 대체"))
        if not _krx_ok:
            _inactive.append(("📊", "KRX 인증 미설정", "기관 매매 데이터 제한"))

        if _inactive:
            st.divider()
            st.markdown(
                '<div style="font-size:.78rem;color:#F59E0B;font-weight:600;'
                'letter-spacing:.3px;margin-bottom:6px">⚠️ 미연동 항목</div>',
                unsafe_allow_html=True,
            )
            for _ic, _it, _ih in _inactive:
                st.markdown(
                    f'<div style="background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.2);'
                    f'border-radius:8px;padding:6px 10px;margin:3px 0;font-size:.75rem">'
                    f'<span style="color:#FCD34D">{_ic} {_it}</span>'
                    f'<div style="color:#78716C;margin-top:1px">{_ih}</div></div>',
                    unsafe_allow_html=True,
                )

    result.update(
        ticker=ticker,
        sname=sname,
        period=period,
        gemini_api_key=gemini_api_key,
        groq_api_key=groq_api_key,
        dart_api_key=dart_api_key,
        use_llm=use_llm,
        krx_ok=_krx_ok,
    )
    return result


# ═════════════════════════════════════════════════════════════════════════════
# 헤더 + 포트폴리오 요약 카드
# ═════════════════════════════════════════════════════════════════════════════

def render_header(portfolio_summary: dict) -> None:
    """메인 헤더 타이틀 + 포트폴리오 3카드를 렌더링한다."""
    st.markdown("""
<div style="display:flex;align-items:center;gap:14px;margin-bottom:18px;margin-top:4px">
  <span class="gradient-text" style="font-size:1.75rem;font-weight:900">AI 주식 분석 대시보드</span>
  <span style="font-size:.72rem;color:#8B5CF6;padding:3px 12px;
               background:rgba(139,92,246,0.12);border:1px solid rgba(139,92,246,0.35);
               border-radius:20px;font-weight:700;letter-spacing:1px">PREMIUM</span>
</div>
""", unsafe_allow_html=True)

    h_val  = portfolio_summary.get("total_val")
    h_pnl  = portfolio_summary.get("total_pnl", 0.0)
    h_pnlp = portfolio_summary.get("total_pnl_pct", 0.0)
    h_ovr  = portfolio_summary.get("overall_pct")

    hc1, hc2, hc3 = st.columns(3)
    if h_val is not None:
        from ui.styles import gain_or_loss_color, gain_or_loss_rgb
        pnl_c   = gain_or_loss_color(h_pnl)
        ovr_c   = gain_or_loss_color(h_ovr or 0)
        pnl_rgb = gain_or_loss_rgb(h_pnl)
        ovr_rgb = gain_or_loss_rgb(h_ovr or 0)

        hc1.markdown(
            header_metric_card_html(
                SVG_WALLET, "총 평가금",
                f"₩{h_val:,.0f}", "포트폴리오 현재가치",
                COLORS["accent"], COLORS["accent_rgb"], COLORS["accent_rgb"],
            ),
            unsafe_allow_html=True,
        )
        hc2.markdown(
            header_metric_card_html(
                SVG_BAR_CHART, "미실현 손익",
                f'<span style="color:{pnl_c}">₩{h_pnl:+,.0f}</span>',
                f'<span style="color:{pnl_c}">{h_pnlp:+.2f}% 수익률</span>',
                pnl_c, pnl_rgb, pnl_rgb,
            ),
            unsafe_allow_html=True,
        )
        hc3.markdown(
            header_metric_card_html(
                SVG_TREND, "누적 수익률",
                f'<span style="color:{ovr_c}">{f"{h_ovr:+.2f}%" if h_ovr is not None else "—"}</span>',
                f'<span style="color:{ovr_c}">매도 이력 포함 전체 기간</span>',
                ovr_c, ovr_rgb, ovr_rgb,
            ),
            unsafe_allow_html=True,
        )
    else:
        ph = placeholder_card_html()
        hc1.markdown(ph, unsafe_allow_html=True)
        hc2.markdown(ph, unsafe_allow_html=True)
        hc3.markdown(ph, unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom:6px'></div>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# 시장 현황 탭
# ═════════════════════════════════════════════════════════════════════════════

def render_market_tab(
    tab,
    *,
    full_movers_fn,
    movers_fn,
    rates: dict,
    usdkrw_fn,
    sector_etf_prices_fn,
) -> None:
    """시장 현황 탭 전체를 렌더링한다."""
    with tab:
        st.subheader("🌐 시장 현황")

        st.markdown("### 🏆 전체 시장 급등·급락 TOP 10 (KOSPI+KOSDAQ)")
        _fm_gainers, _fm_losers = full_movers_fn()

        if not _fm_gainers.empty or not _fm_losers.empty:
            _fmc1, _fmc2 = st.columns(2)
            _cols_show = ["종목명", "티커", "현재가", "등락률(%)", "시장"]
            with _fmc1:
                st.markdown("#### 🚀 급등 상위 10")
                if not _fm_gainers.empty:
                    st.dataframe(
                        _fm_gainers[_cols_show].style
                        .format({"현재가": "{:,.0f}", "등락률(%)": "{:+.2f}%"})
                        .map(lambda v: "color:#ef5350;font-weight:bold"
                             if isinstance(v, float) and v > 0 else "",
                             subset=["등락률(%)"]),
                        use_container_width=True, hide_index=True,
                    )
            with _fmc2:
                st.markdown("#### 📉 급락 하위 10")
                if not _fm_losers.empty:
                    st.dataframe(
                        _fm_losers[_cols_show].style
                        .format({"현재가": "{:,.0f}", "등락률(%)": "{:+.2f}%"})
                        .map(lambda v: "color:#42a5f5;font-weight:bold"
                             if isinstance(v, float) and v < 0 else "",
                             subset=["등락률(%)"]),
                        use_container_width=True, hide_index=True,
                    )
        else:
            st.warning("시장 데이터를 불러올 수 없습니다. (KRX 서버 응답 없음 또는 네트워크 오류)", icon="⚠️")

        st.divider()
        col_left, col_right = st.columns([2, 1])

        with col_left:
            mover_n = st.select_slider(
                "분석 종목 수 (시가총액 상위)", list(range(10, 110, 10)), value=50,
                help="KOSPI 시가총액 상위 N개 종목의 등락률을 분석합니다.",
            )
            movers = movers_fn(mover_n)

            if not movers.empty:
                top_n   = min(10, len(movers) // 2)
                gainers = movers.head(top_n)
                losers  = movers.tail(top_n).sort_values("등락률(%)")

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("### 🚀 급등 상위")
                    st.dataframe(
                        gainers[["종목명", "현재가", "등락률(%)"]].style
                        .format({"현재가": "{:,.0f}", "등락률(%)": "{:+.2f}%"})
                        .map(lambda v: "color:#ef5350; font-weight:bold"
                             if isinstance(v, float) and v > 0 else "",
                             subset=["등락률(%)"]),
                        use_container_width=True, hide_index=True,
                    )
                with c2:
                    st.markdown("### 📉 급락 하위")
                    st.dataframe(
                        losers[["종목명", "현재가", "등락률(%)"]].style
                        .format({"현재가": "{:,.0f}", "등락률(%)": "{:+.2f}%"})
                        .map(lambda v: "color:#42a5f5; font-weight:bold"
                             if isinstance(v, float) and v < 0 else "",
                             subset=["등락률(%)"]),
                        use_container_width=True, hide_index=True,
                    )

                chart_n   = min(15, len(movers) // 2)
                chart_top = pd.concat([movers.head(chart_n), movers.tail(chart_n).sort_values("등락률(%)")])
                st.markdown(f"### 📊 급등·급락 TOP {chart_n} 등락률")
                fig_bar = go.Figure(go.Bar(
                    x=chart_top["종목명"],
                    y=chart_top["등락률(%)"],
                    marker_color=["#ef5350" if val >= 0 else "#42a5f5" for val in chart_top["등락률(%)"]],
                    text=[f"{val:+.2f}%" for val in chart_top["등락률(%)"]],
                    textposition="outside",
                ))
                fig_bar.update_layout(
                    height=380, template="plotly_dark",
                    margin=dict(t=10, b=70), xaxis_tickangle=-40,
                    yaxis_title="등락률 (%)",
                )
                st.plotly_chart(fig_bar, use_container_width=True)

                with st.expander(f"📋 전체 {len(movers)}개 종목 등락률 표"):
                    st.dataframe(
                        movers[["종목명", "티커", "현재가", "등락률(%)", "거래량"]].style
                        .format({"현재가": "{:,.0f}", "등락률(%)": "{:+.2f}%", "거래량": "{:,}"})
                        .map(lambda v: "color:#ef5350" if isinstance(v, float) and v > 0
                             else ("color:#42a5f5" if isinstance(v, float) and v < 0 else ""),
                             subset=["등락률(%)"]),
                        use_container_width=True, hide_index=True,
                    )

        with col_right:
            st.markdown("### 💱 환율 상세")
            for pair, info in rates.items():
                st.markdown(rate_card_html(pair, info["rate"], info["change"]), unsafe_allow_html=True)

            st.markdown("### 📈 USD/KRW 추이 (3개월)")
            try:
                fx = usdkrw_fn()
                if not fx.empty:
                    fig_fx = go.Figure(go.Scatter(
                        x=fx.index, y=fx["Close"],
                        fill="tozeroy", fillcolor="rgba(31,119,180,0.12)",
                        line=dict(color="#42a5f5", width=2),
                    ))
                    fig_fx.update_layout(
                        height=220, template="plotly_dark",
                        margin=dict(t=10, b=10), showlegend=False,
                        xaxis_rangeslider_visible=False,
                    )
                    st.plotly_chart(fig_fx, use_container_width=True)
            except Exception:
                pass

        # ── 섹터 ETF 실시간 등락표 ────────────────────────────────────────────
        st.divider()
        st.markdown("### 🗺️ 주요 섹터 ETF 실시간 등락표")
        st.caption("미국 ETF 15개 + 국내 섹터별 대표 ETF 20개 · 전일 대비 등락률 · 10분 캐시")
        sector_etf_prices_fn()


# ═════════════════════════════════════════════════════════════════════════════
# 추천 종목 탭 — 로딩바 + 다단계 분석 포함
# ═════════════════════════════════════════════════════════════════════════════

def render_rec_tab(
    tab,
    *,
    get_full_stocks_fn,
    get_recommendations_fn,
    news_sentiment_llm_fn,
    news_sentiment_kw_fn,
    ticker_name_map_fn,
    db_save_recommendation,
    db_get_rec_history,
    auth_user_id,
) -> None:
    """추천 종목 탭 전체를 렌더링한다."""
    with tab:
        st.subheader("⭐ AI 추천 종목 분석")

        col_a, col_b = st.columns([3, 1])
        with col_a:
            rec_market = st.radio(
                "분석 시장",
                ["KOSPI", "KOSDAQ", "미국 주식 (S&P500)", "미국 주식 (나스닥)"],
                horizontal=True,
                key="rec_market_radio",
            )
        with col_b:
            run_btn = st.button("🔄 전수 분석 실행", type="primary", use_container_width=True)

        st.caption(
            "💡 **전수 조사 모드** — 시장 전체 종목(최대 500개)을 L1→L2→L3 깔때기 방식으로 분석합니다.  "
            "L1 차트 스크리닝으로 하위 70%를 즉시 탈락시키고, 상위 후보만 정밀 분석·뉴스 감성 처리합니다.  "
            "약 2~5분 소요됩니다."
        )

        if run_btn:
            st.session_state["_rec_run_requested"] = True

        _rec_ph = st.empty()

        if st.session_state.get("_rec_run_requested"):
            st.session_state["_rec_run_requested"] = False

            stocks       = get_full_stocks_fn(rec_market)
            total_stocks = len(stocks)
            _rec_q       = queue.Queue()
            _rec_start   = time.time()

            _stage_icons = {0: "🔍", 1: "📡", 2: "📊", 3: "🎯", 4: "✅"}
            _stage_state = {
                "stage": 0, "icon": "🔍",
                "title": "전 종목 AI 분석 준비 중",
                "msg":   f"시장 전체 {total_stocks}개 종목 데이터 수집 준비 중...",
            }

            def _render_rec_bar(state: dict, elapsed: int, done: bool = False) -> None:
                body = state["msg"]
                _rec_ph.markdown(
                    loading_card_html(
                        state["icon"], state["title"], body, elapsed, done=done,
                    ),
                    unsafe_allow_html=True,
                )

            def _run_rec():
                return get_recommendations_fn(
                    stocks,
                    news_fn=news_sentiment_kw_fn,
                    progress_q=_rec_q,
                )

            _render_rec_bar(_stage_state, 0)
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _pool:
                _f_rec = _pool.submit(_run_rec)
                while not _f_rec.done():
                    _elapsed = int(time.time() - _rec_start)
                    while not _rec_q.empty():
                        _msg = _rec_q.get_nowait()
                        if isinstance(_msg, dict):
                            _stage_state.update(_msg)
                    _render_rec_bar(_stage_state, _elapsed)
                    time.sleep(1)
                recs = _f_rec.result()

            _render_rec_bar(
                {"icon": "✅", "title": "분석 완료", "msg": f"총 {total_stocks}개 종목 전수 분석 완료"},
                int(time.time() - _rec_start), done=True,
            )
            _rec_ph.empty()
            st.session_state["_rec_results"] = recs
            if auth_user_id and recs:
                db_save_recommendation(auth_user_id, recs)

        # ── 결과 표시 ─────────────────────────────────────────────────────────
        recs = st.session_state.get("_rec_results", [])
        if recs:
            st.success(f"✅ {len(recs)}개 추천 종목 발견")
            _name_map = ticker_name_map_fn()
            for r in recs[:20]:
                _tk   = r.get("ticker", "")
                _nm   = _name_map.get(_tk, _tk)
                _sc   = r.get("score", 0)
                _lbl  = r.get("label", "")
                _pct  = r.get("change_pct", 0.0)
                _clr  = COLORS["gain"] if _pct >= 0 else COLORS["loss"]
                st.markdown(
                    f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};'
                    f'border-radius:10px;padding:12px 16px;margin:6px 0;'
                    f'display:flex;justify-content:space-between;align-items:center;">'
                    f'<div><b style="color:{COLORS["text"]}">{_nm}</b>'
                    f'<span style="color:{COLORS["text_2"]};font-size:.8rem;margin-left:8px">({_tk})</span>'
                    f'<div style="font-size:.78rem;color:{COLORS["text_2"]};margin-top:3px">{_lbl}</div></div>'
                    f'<div style="text-align:right">'
                    f'<div style="font-size:1.1rem;font-weight:700;color:{_clr}">{_pct:+.2f}%</div>'
                    f'<div style="font-size:.72rem;color:{COLORS["text_2"]}">점수 {_sc:+.1f}</div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )
        elif not st.session_state.get("_rec_run_requested"):
            st.info("👈 **전수 분석 실행** 버튼을 눌러 추천 종목을 분석하세요.", icon="💡")

        # ── 추천 이력 ──────────────────────────────────────────────────────────
        if auth_user_id:
            with st.expander("📋 추천 이력", expanded=False):
                _hist = db_get_rec_history(auth_user_id)
                if not _hist:
                    st.caption("아직 추천 이력이 없습니다.")
                else:
                    for _h in _hist:
                        _h_dt   = _h.get("created_at", "")[:16].replace("T", " ")
                        _h_amt  = _h.get("investment_amt", 0)
                        _h_prof = _h.get("risk_profile", "중립형")
                        _h_recs = _h.get("recommendations", [])
                        _h_names = ", ".join(r.get("name", r.get("ticker", "")) for r in _h_recs)
                        st.markdown(
                            f'<div style="background:#1e2130;border-radius:8px;'
                            f'padding:10px 14px;margin:4px 0;font-size:.85rem">'
                            f'<span style="color:#9e9e9e">{_h_dt}</span>'
                            f'&nbsp;|&nbsp;'
                            f'<b style="color:#ddd">₩{_h_amt:,}</b>'
                            f'&nbsp;·&nbsp;'
                            f'<span style="color:#4fc3f7">{_h_prof}</span>'
                            f'<div style="color:#aaa;margin-top:4px">{_h_names}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )


# ═════════════════════════════════════════════════════════════════════════════
# 뉴스 & 관련 종목 탭
# ═════════════════════════════════════════════════════════════════════════════

def render_news_tab(
    tab,
    *,
    state: dict,
    api_keys: dict,
    etf_fundamental_fn,
    naver_news_fn,
    sector_perf_fn,
    check_is_etf_fn,
) -> None:
    """뉴스 & 관련 종목 탭 전체를 렌더링한다."""
    with tab:
        ticker      = state["ticker"]
        sname       = state["sname"]
        data_ready  = state["data_ready"]
        asname      = state.get("asname", sname)
        news_result = state.get("news_result", {})
        gemini_key  = api_keys.get("gemini", "")
        groq_key    = api_keys.get("groq", "")

        _news_is_etf = data_ready and check_is_etf_fn(ticker)

        if _news_is_etf:
            st.subheader(f"📊 {asname or sname} 섹터 뉴스 — 돈이 몰리는 섹터 파악")
            st.caption("ETF는 종목 필터를 느슨하게 적용하고 상위 구성종목 뉴스를 함께 수집합니다.")
        else:
            st.subheader(f"📰 {asname or sname} 뉴스 & 관련 정보")

        is_kr_stock = ticker.endswith(".KS") or ticker.endswith(".KQ")

        raw_news: list = []
        _etf_fund_data: dict = {}
        if _news_is_etf:
            with st.spinner("ETF 섹터 뉴스 수집 중 (ETF + 구성종목)..."):
                _etf_fund_data = etf_fundamental_fn(ticker)
                raw_news = get_etf_news_with_holdings(ticker, _etf_fund_data, max_items=15)
            _etf_sector = _etf_fund_data.get("sector", "")
            _etf_holdings_display = [
                h.get("name", h.get("ticker", "")) for h in _etf_fund_data.get("top_holdings", [])[:5]
            ]
            if _etf_sector or _etf_holdings_display:
                st.markdown(
                    f'<div style="background:#1a2f3a;border-radius:8px;padding:10px 16px;margin-bottom:10px;">'
                    f'<span style="color:#80cbc4;font-weight:bold;">섹터: {_etf_sector or "N/A"}</span>'
                    + (f'&nbsp;&nbsp;|&nbsp;&nbsp;<span style="color:#aaa;font-size:0.85rem;">'
                       f'구성종목 뉴스 포함: {", ".join(_etf_holdings_display)}</span>'
                       if _etf_holdings_display else "") +
                    f'</div>',
                    unsafe_allow_html=True,
                )
        elif is_kr_stock:
            with st.spinner("네이버 금융에서 뉴스 수집 중..."):
                raw_news = naver_news_fn(ticker)
        else:
            try:
                for it in (yf.Ticker(ticker).news or [])[:10]:
                    c = it.get("content", it)
                    ts_raw = c.get("pubDate", it.get("providerPublishTime", ""))
                    raw_news.append({
                        "title":     c.get("title",    it.get("title", "제목 없음")),
                        "link":      (c.get("canonicalUrl", {}).get("url") or it.get("link", "#")),
                        "publisher": (c.get("provider", {}).get("displayName") or it.get("publisher", "")),
                        "pub_date":  (
                            datetime.fromtimestamp(ts_raw).strftime("%Y-%m-%d %H:%M")
                            if isinstance(ts_raw, (int, float)) else str(ts_raw)
                        ),
                    })
            except Exception as e:
                st.warning(f"뉴스 로드 실패: {e}")

        _cname = sname if sname != ticker else ""
        sent: dict = {}
        if raw_news:
            with st.spinner("AI 감성 분석 중..."):
                if _news_is_etf:
                    sent = analyze_etf_news_sentiment(raw_news, ticker, _etf_fund_data)
                elif gemini_key or groq_key:
                    sent = analyze_news_sentiment_llm(raw_news, ticker, gemini_key, groq_key, _cname)
                else:
                    sent = analyze_news_sentiment_keywords(raw_news, ticker, _cname)

        news_col, sent_col = st.columns([3, 2])

        with news_col:
            st.markdown(f"#### 📰 최신 뉴스 ({len(raw_news)}건)")
            if not raw_news:
                st.info("수집된 뉴스가 없습니다.")
            for _ni, n in enumerate(raw_news):
                title      = n.get("title", "")
                link       = n.get("link", "#")
                publisher  = n.get("publisher", "")
                pub_date   = n.get("pub_date", "")
                _open_btn_col, _ai_btn_col = st.columns([4, 1])
                with _open_btn_col:
                    st.markdown(
                        f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};'
                        f'border-radius:10px;padding:10px 14px;margin:4px 0;">'
                        f'<a href="{link}" target="_blank" style="color:{COLORS["text"]};'
                        f'text-decoration:none;font-size:.9rem;font-weight:500;line-height:1.5;">{title}</a>'
                        f'<div style="font-size:.73rem;color:{COLORS["text_2"]};margin-top:5px;">'
                        f'{publisher}{"  ·  " + pub_date if pub_date else ""}'
                        f'</div></div>',
                        unsafe_allow_html=True,
                    )
                with _ai_btn_col:
                    if st.button("AI", key=f"ai_{ticker}_{_ni}", help="AI 기사 요약", use_container_width=True):
                        article_dialog(title, link, ticker, gemini_key, groq_key)

        with sent_col:
            st.markdown("#### 🧠 AI 감성 분석")
            if sent:
                _s = sent.get("sentiment", "N/A")
                _sc = sent.get("score", 0.0)
                st.markdown(sentiment_badge_html(_s, _sc), unsafe_allow_html=True)
                st.markdown("")

                _summary = sent.get("summary", "")
                if _summary:
                    st.info(_summary)

                _detail = sent.get("detail", [])
                if _detail:
                    st.markdown("**분석 근거**")
                    for d in _detail[:5]:
                        _d_sent = d.get("sentiment", "중립")
                        _d_icon = "🟢" if _d_sent == "긍정" else ("🔴" if _d_sent == "부정" else "⚪")
                        _d_ttl  = d.get("title", "")[:60]
                        _d_sc   = d.get("score", 0.0)
                        st.markdown(
                            f'<div style="background:#1a1d2e;border-radius:6px;padding:6px 10px;'
                            f'margin:3px 0;font-size:.8rem;color:#aaa;display:flex;gap:6px;">'
                            f'<span>{_d_icon}</span>'
                            f'<span style="flex:1">{_d_ttl}</span>'
                            f'<span style="color:{"#4caf50" if _d_sc >= 0 else "#ef5350"};'
                            f'font-weight:700;white-space:nowrap">{_d_sc:+.1f}</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
            else:
                st.info("감성 분석 결과가 없습니다.")

            # ── 관련 섹터 성과 ─────────────────────────────────────────────
            if data_ready:
                st.markdown("---")
                st.markdown("#### 📊 관련 섹터 성과")
                try:
                    _sec_perf = sector_perf_fn(ticker)
                    if _sec_perf:
                        for _sec, _pct in list(_sec_perf.items())[:5]:
                            _sc = COLORS["gain"] if _pct >= 0 else COLORS["loss"]
                            st.markdown(
                                f'<div style="display:flex;justify-content:space-between;'
                                f'padding:5px 0;border-bottom:1px solid {COLORS["border_md"]};">'
                                f'<span style="color:{COLORS["text_2"]};font-size:.85rem">{_sec}</span>'
                                f'<span style="color:{_sc};font-weight:700;font-size:.85rem">{_pct:+.2f}%</span>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                    else:
                        st.caption("섹터 데이터 없음")
                except Exception:
                    pass


# ═════════════════════════════════════════════════════════════════════════════
# 차트 분석 탭 (메인 분석 탭)
# ═════════════════════════════════════════════════════════════════════════════

def render_chart_tab(
    tab,
    *,
    state: dict,
    api_keys: dict,
    inv_data_fn,
    insider_trades_fn,
    save_watchlist_fn,
) -> None:
    """차트 분석 탭 전체를 렌더링한다."""
    with tab:
        ticker      = state["ticker"]
        sname       = state["sname"]
        data_ready  = state["data_ready"]
        data        = state["data"]
        close       = state["close"]
        signals     = state["signals"]
        advanced    = state["advanced"]
        hybrid      = state["hybrid"]
        expected    = state["expected"]
        fund_info   = state["fund_info"]
        fund_score_data = state["fund_score_data"]
        news_result = state["news_result"]
        dead_time   = state["dead_time"]
        breakout    = state["breakout"]
        risk_adj    = state["risk_adj"]
        vol_anomaly = state["vol_anomaly"]
        rt_price    = state["rt_price"]
        rt_ts       = state["rt_ts"]
        rt_realtime = state["rt_realtime"]
        rt_stale    = state["rt_stale"]
        rt_stale_msg = state["rt_stale_msg"]
        aperiod     = state.get("aperiod", "3mo")
        gemini_key  = api_keys.get("gemini", "")
        groq_key    = api_keys.get("groq", "")

        if not data_ready:
            st.info("👈 사이드바에서 종목을 선택하고 **분석 시작** 버튼을 눌러주세요.", icon="📊")

        # 관심종목 버튼
        is_in_wl = any(w["ticker"] == ticker for w in st.session_state.watchlist)
        wl_col1, wl_col2 = st.columns([6, 1])
        with wl_col2:
            if is_in_wl:
                if st.button("★ 관심 해제", use_container_width=True):
                    st.session_state.watchlist = [
                        w for w in st.session_state.watchlist if w["ticker"] != ticker
                    ]
                    save_watchlist_fn(st.session_state.watchlist)
                    st.rerun()
            else:
                if st.button("☆ 관심 추가", use_container_width=True, type="primary"):
                    st.session_state.watchlist.append({"name": sname, "ticker": ticker})
                    save_watchlist_fn(st.session_state.watchlist)
                    st.rerun()

        if data.empty:
            return

        # 거래정지 경고
        if vol_anomaly.get("is_halted"):
            st.markdown(
                halted_banner_html(
                    vol_anomaly.get("reason", ""),
                    vol_anomaly.get("recent_vol", 0),
                    vol_anomaly.get("avg_vol", 0),
                    vol_anomaly.get("ratio", 0),
                ),
                unsafe_allow_html=True,
            )

        # AI 종합 리포트 배너
        if data_ready:
            _rpt_signal, _rpt_action, _rpt_reasons = generate_signal(
                data=data, advanced=advanced, hybrid=hybrid,
                news_result=news_result, expected=expected, signals=signals,
            )
            _sl_rpt  = get_stop_loss_targets(data) if not data.empty else None
            _has_rpt = not close.empty
            _rpt_cur = float(close.iloc[-1]) if _has_rpt else 0
            _is_krw  = _rpt_cur > 500
            _fmt     = "{:,.0f}" if _is_krw else "{:,.2f}"
            _rpt_sl  = _fmt.format(_sl_rpt["stop_8pct"]) if _sl_rpt else "—"
            _rpt_tgt = _fmt.format(_sl_rpt["target_2r"])  if _sl_rpt else "—"

            st.markdown(
                signal_report_html(
                    signal=_rpt_signal, action=_rpt_action, reasons=_rpt_reasons,
                    h_label=hybrid.get("label", "중립/관망"),
                    h_badge=hybrid.get("badge", "⚪"),
                    h_score=hybrid.get("hybrid_score", 0.0),
                    news_score=news_result.get("score", 0.0) if isinstance(news_result, dict) else 0.0,
                    fund_score=fund_score_data.get("fund_score", 0),
                    fund_label=fund_score_data.get("fund_label", "N/A"),
                    cur_price=_rpt_cur,
                    sl_price=_rpt_sl,
                    tgt_price=_rpt_tgt,
                    is_krw=_is_krw,
                ),
                unsafe_allow_html=True,
            )

        # ── 차트 | 신호 2컬럼 ─────────────────────────────────────────────
        col_chart, col_sig = st.columns([1, 1])

        with col_chart:
            title_label = f"{sname} ({ticker})" if sname != ticker else ticker
            _badge_is_krw = ticker.upper().endswith((".KS", ".KQ"))
            _badge_price_str = (
                ("₩" if _badge_is_krw else "$") +
                ("{:,.0f}" if _badge_is_krw else "{:,.2f}").format(rt_price)
            ) if (rt_price > 0 and data_ready) else "—"
            st.markdown(
                stock_badge_html(title_label, _badge_price_str, rt_realtime and data_ready and rt_price > 0),
                unsafe_allow_html=True,
            )

            # KOSPI 데이터 사전 수집
            try:
                _kospi_raw = yf.download("^KS11", period=aperiod, auto_adjust=True, progress=False)
                _kospi_raw = _flatten_columns(_kospi_raw)
                _kospi_df  = _kospi_raw[["Open", "High", "Low", "Close"]].dropna()
            except Exception:
                _kospi_df = pd.DataFrame()

            data = _flatten_columns(data)

            fig = _build_plotly_chart(data, _kospi_df, ticker, aperiod)

            st.markdown("""
<div style="margin-bottom:6px;font-size:0.8rem;color:#94A3B8;">
  캔들·EMA·볼린저밴드·RSI·MACD·ADX·KOSPI 6개 패널 포함
</div>""", unsafe_allow_html=True)
            if st.button("📊 차트 보기 (새 창)", type="primary", use_container_width=True,
                         key="show_chart_btn", help="클릭하면 차트 분석 창이 열립니다."):
                chart_dialog(fig, ticker)

        # ── 신호 패널 ──────────────────────────────────────────────────────
        with col_sig:
            _render_signal_panel(
                state=state,
                inv_data_fn=inv_data_fn,
                gemini_key=gemini_key,
                groq_key=groq_key,
            )

        # ── 차트 하단 추가 분석 섹션들 ────────────────────────────────────
        if data_ready:
            _render_chart_bottom_sections(
                state=state,
                inv_data_fn=inv_data_fn,
                insider_trades_fn=insider_trades_fn,
            )


def _build_plotly_chart(
    data: pd.DataFrame,
    kospi_df: pd.DataFrame,
    ticker: str,
    aperiod: str,
) -> go.Figure:
    """6패널 Plotly 차트를 생성하여 반환한다."""
    fig = make_subplots(
        rows=6, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(
            "가격 (캔들·EMA·볼린저밴드)",
            "거래량 + OBV 추세",
            "RSI (14) + MFI (14)",
            "MACD (12·26·9)",
            "ADX (14) + ±DI",
            "KOSPI 지수",
        ),
        row_heights=[0.28, 0.084, 0.112, 0.112, 0.112, 0.30],
    )

    if not all(col in data.columns for col in ("Open", "High", "Low", "Close", "Volume")):
        return fig

    o = data.get("Open")
    h = data.get("High")
    lo = data.get("Low")
    c = data.get("Close")
    v = data.get("Volume")

    # 캔들스틱
    fig.add_trace(go.Candlestick(
        x=data.index, open=o, high=h, low=lo, close=c,
        name="가격", increasing_line_color="#ef5350", decreasing_line_color="#42a5f5",
    ), row=1, col=1)

    # 이동평균선
    for col_name, color, lbl, width, dash in [
        ("SMA_5",   "#ffa726", "SMA5",   1.4, "solid"),
        ("EMA_20",  "#ab47bc", "EMA20",  1.4, "solid"),
        ("EMA_50",  "#26c6da", "EMA50",  1.4, "solid"),
        ("EMA_200", "#ffeb3b", "EMA200", 2.0, "dash"),
    ]:
        if col_name in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index, y=data[col_name],
                name=lbl, line=dict(color=color, width=width, dash=dash),
            ), row=1, col=1)

    # 볼린저밴드
    if "BB_Upper" in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index, y=data["BB_Upper"], name="BB상단",
            line=dict(color="rgba(158,158,158,0.6)", width=1, dash="dot"), showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=data.index, y=data["BB_Lower"], name="BB하단",
            line=dict(color="rgba(158,158,158,0.6)", width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(158,158,158,0.07)", showlegend=False,
        ), row=1, col=1)

    # 일목균형표
    if "ICHI_SPAN_A" in data.columns and "ICHI_SPAN_B" in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index, y=data["ICHI_SPAN_A"], name="선행스팬A",
            line=dict(color="rgba(102,187,106,0.7)", width=1), fill=None,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=data.index, y=data["ICHI_SPAN_B"], name="선행스팬B",
            line=dict(color="rgba(239,83,80,0.7)", width=1),
            fill="tonexty", fillcolor="rgba(120,120,120,0.12)",
        ), row=1, col=1)
    for col_name, color, lbl, dash in [
        ("ICHI_TENKAN", "#ef5350", "전환선", "dot"),
        ("ICHI_KIJUN",  "#42a5f5", "기준선", "dot"),
    ]:
        if col_name in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index, y=data[col_name], name=lbl,
                line=dict(color=color, width=1.1, dash=dash),
            ), row=1, col=1)

    # VWAP 멀티 타임프레임
    for _vc, _color, _lbl, _w, _dash in [
        ("VWAP_W", "#ff8f00", "VWAP 주간(5일)",  1.4, "solid"),
        ("VWAP_M", "#ce93d8", "VWAP 월간(20일)", 1.6, "solid"),
        ("VWAP_Q", "#80deea", "VWAP 분기(60일)", 2.0, "dash"),
    ]:
        if _vc in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index, y=data[_vc],
                name=_lbl, line=dict(color=_color, width=_w, dash=_dash),
            ), row=1, col=1)

    # 거래량 (Row 2)
    vol_colors = ["#ef5350" if float(c.iloc[i]) >= float(o.iloc[i]) else "#42a5f5"
                  for i in range(len(data))]
    fig.add_trace(go.Bar(
        x=data.index, y=v, name="거래량",
        marker_color=vol_colors, showlegend=False, opacity=0.7,
    ), row=2, col=1)
    if "Volume_MA20" in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index, y=data["Volume_MA20"], name="Vol MA20",
            line=dict(color="#ffa726", width=1.2),
        ), row=2, col=1)

    # OBV
    if "OBV" in data.columns:
        obv_s = data["OBV"]
        o_min, o_max = float(obv_s.min()), float(obv_s.max())
        v_max = float(data["Volume"].max())
        if o_max > o_min and v_max > 0:
            obv_scaled = (obv_s - o_min) / (o_max - o_min) * v_max
        else:
            obv_scaled = obv_s * 0
        fig.add_trace(go.Scatter(
            x=data.index, y=obv_scaled, name="OBV(스케일)",
            line=dict(color="#80cbc4", width=1.5, dash="dot"),
        ), row=2, col=1)

    # RSI + MFI (Row 3)
    if "RSI" in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index, y=data["RSI"], name="RSI",
            line=dict(color="#ce93d8", width=2),
        ), row=3, col=1)
        for level, clr in [(70, "rgba(239,83,80,0.45)"), (30, "rgba(66,165,245,0.45)")]:
            fig.add_hline(y=level, line_color=clr, line_dash="dash", row=3, col=1)
        fig.add_hrect(y0=70, y1=100, fillcolor="rgba(239,83,80,0.05)", row=3, col=1)
        fig.add_hrect(y0=0,  y1=30,  fillcolor="rgba(66,165,245,0.05)", row=3, col=1)
    if "MFI" in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index, y=data["MFI"], name="MFI",
            line=dict(color="#4dd0e1", width=1.5, dash="dot"),
        ), row=3, col=1)

    # MACD (Row 4)
    if "MACD" in data.columns:
        hist_vals   = data["MACD_Hist"]
        hist_colors = ["#ef5350" if val >= 0 else "#42a5f5" for val in hist_vals]
        fig.add_trace(go.Bar(
            x=data.index, y=hist_vals, name="MACD Hist",
            marker_color=hist_colors, showlegend=False,
        ), row=4, col=1)
        fig.add_trace(go.Scatter(
            x=data.index, y=data["MACD"], name="MACD",
            line=dict(color="#42a5f5", width=1.5),
        ), row=4, col=1)
        fig.add_trace(go.Scatter(
            x=data.index, y=data["MACD_Signal"], name="Signal",
            line=dict(color="#ffa726", width=1.5),
        ), row=4, col=1)

    # ADX + ±DI (Row 5)
    if "ADX" in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index, y=data["ADX"], name="ADX",
            line=dict(color="#fff176", width=2),
        ), row=5, col=1)
        fig.add_hline(y=25, line_color="rgba(255,255,255,0.3)", line_dash="dash", row=5, col=1)
        fig.add_hline(y=35, line_color="rgba(255,235,59,0.3)",  line_dash="dot",  row=5, col=1)
    for col_name, color, lbl in [
        ("ADX_POS", "#66bb6a", "+DI"),
        ("ADX_NEG", "#ef5350", "-DI"),
    ]:
        if col_name in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index, y=data[col_name], name=lbl,
                line=dict(color=color, width=1.3),
            ), row=5, col=1)

    # KOSPI (Row 6)
    if not kospi_df.empty:
        fig.add_trace(go.Scatter(
            x=kospi_df.index, y=kospi_df["Close"], name="KOSPI",
            line=dict(color="#80cbc4", width=1.8),
            fill="tozeroy", fillcolor="rgba(128,203,196,0.10)",
        ), row=6, col=1)

    # 공통 레이아웃
    _rangeselector = dict(
        buttons=[
            dict(count=1,  label="1M", step="month", stepmode="backward"),
            dict(count=3,  label="3M", step="month", stepmode="backward"),
            dict(count=6,  label="6M", step="month", stepmode="backward"),
            dict(count=1,  label="1Y", step="year",  stepmode="backward"),
            dict(step="all", label="ALL"),
        ],
        bgcolor="#263238", activecolor="#1565c0",
        font=dict(color="#cfd8dc", size=10),
        bordercolor="#455a64", borderwidth=1,
    )
    fig.update_xaxes(
        showspikes=True, spikemode="across", spikesnap="cursor",
        spikethickness=1, spikecolor="rgba(150,150,150,0.6)", spikedash="solid",
    )
    fig.update_yaxes(
        showspikes=True, spikethickness=1,
        spikecolor="rgba(150,150,150,0.4)", spikedash="dot",
        fixedrange=False, autorange=True,
    )
    fig.update_xaxes(rangeselector=_rangeselector, row=1, col=1)
    fig.update_annotations(font_size=10, font_color="#9e9e9e")
    fig.update_layout(
        height=1150, template="plotly_dark",
        dragmode=False, xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
                    font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=30, b=10, l=10, r=10),
        uirevision=ticker, hovermode="x unified",
    )
    return fig


def _render_signal_panel(*, state: dict, inv_data_fn, gemini_key: str, groq_key: str) -> None:
    """차트 탭 우측 신호 패널을 렌더링한다."""
    ticker      = state["ticker"]
    data        = state["data"]
    close       = state["close"]
    data_ready  = state["data_ready"]
    signals     = state["signals"]
    advanced    = state["advanced"]
    hybrid      = state["hybrid"]
    expected    = state["expected"]
    fund_score_data = state["fund_score_data"]
    news_result = state["news_result"]
    dead_time   = state["dead_time"]
    breakout    = state["breakout"]
    risk_adj    = state["risk_adj"]
    rt_price    = state["rt_price"]
    rt_ts       = state["rt_ts"]
    rt_realtime = state["rt_realtime"]
    rt_stale    = state["rt_stale"]
    rt_stale_msg = state["rt_stale_msg"]

    st.subheader("🎯 AI 매매 신호")

    with st.expander("🔍 진단 기준 보기", expanded=False):
        st.markdown(
            "**이 패널은 '역추세 기술적 진단' 관점으로 분석합니다.**\n\n"
            "현재 종목의 RSI·MACD·볼린저밴드·ADX·일목균형표 등 **10개 이상의 기술적 지표**를 "
            "종합하여 단기 변곡점(과매도·과매수)과 매수/매도 신호를 포착합니다.\n\n"
            "| 특성 | 내용 |\n"
            "|------|------|\n"
            "| 분석 방식 | 역추세 (RSI 30↓ = 강한 매수 신호) |\n"
            "| 뉴스 처리 | Naver · RSS · YouTube 멀티소스 · LLM 정밀 분석 |\n"
            "| 대상 | 사용자가 사이드바에서 직접 선택한 단일 종목 |\n"
            "| 목적 | 지금 이 종목의 매매 타이밍 포착 |\n\n"
            "> ⚠️ **포트폴리오 탭 'AI 모멘텀 추천'과 결과가 다를 수 있습니다.**"
        )

    h_score = hybrid.get("hybrid_score", 0.0)
    h_label = hybrid.get("label", "중립/관망")
    h_badge = hybrid.get("badge", "⚪")
    fs      = fund_score_data.get("fund_score", 0)
    f_label = fund_score_data.get("fund_label", "N/A")

    _has_price = not close.empty and len(close) >= 2
    if _has_price:
        _daily_last = float(close.iloc[-1])
        prev_price  = float(close.iloc[-2])
        last_price  = rt_price if (rt_price > 0 and data_ready) else _daily_last
        daily_chg   = (last_price - prev_price) / prev_price * 100
        _is_krw     = last_price > 500
        _fmt        = "{:,.0f}" if _is_krw else "{:,.2f}"
    else:
        last_price = prev_price = daily_chg = 0.0
        _is_krw, _fmt = True, "{:,.0f}"

    # 실시간 시세 배지
    if data_ready and rt_price > 0:
        if rt_stale:
            st.warning(rt_stale_msg or "장이 열리지 않은 상태입니다. 가장 최근 종가를 표시합니다.", icon="⏸️")
        _rt_label = "실시간 시세 (KST)" if rt_realtime else "장마감 종가 (KST)"
        _rt_color = COLORS["gain"] if rt_realtime else COLORS["text_2"]
        st.markdown(
            f'<div style="font-size:0.75rem;color:{_rt_color};margin-bottom:8px;'
            f'display:flex;align-items:center;gap:6px;">'
            f'<span style="width:6px;height:6px;border-radius:50%;background:{_rt_color};'
            f'display:inline-block;"></span>{_rt_label} &nbsp;·&nbsp; '
            f'<span style="color:{COLORS["text_2"]};">기준 시각: '
            f'<b style="color:{COLORS["text"]};">{rt_ts}</b></span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # 신호등 카드
    _tl_signal, _tl_action, _tl_reasons = generate_signal(
        data=data, advanced=advanced, hybrid=hybrid,
        news_result=news_result, expected=expected, signals=signals,
    )
    if _tl_signal == "BUY":
        _l1_border, _l1_fc = "#22c55e", "#4ade80"
    elif _tl_signal == "SELL":
        _l1_border, _l1_fc = "#ef4444", "#f87171"
    else:
        _l1_border, _l1_fc = "#eab308", "#facc15"
    _l1_emoji = "🟢" if _tl_signal == "BUY" else ("🔴" if _tl_signal == "SELL" else "🟡")

    st.markdown(f"""
<div style="background:{COLORS['surface']};border:1px solid {_l1_border}88;border-radius:12px;
            padding:22px 20px;text-align:center;margin-bottom:12px;
            box-shadow:0 0 28px {_l1_border}33,0 4px 20px rgba(0,0,0,0.35);">
  <div style="font-size:0.62rem;color:{COLORS['text_3']};letter-spacing:3px;text-transform:uppercase;margin-bottom:10px;">
    AI TRADING SIGNAL
  </div>
  <div style="font-size:2rem;font-weight:900;color:{_l1_fc};letter-spacing:2px;line-height:1.05;margin-bottom:10px;">
    {_l1_emoji} {_tl_signal}
  </div>
  <div style="font-size:0.87rem;color:{COLORS['text_2']};line-height:1.65;word-break:keep-all;">
    {_tl_action}
  </div>
</div>""", unsafe_allow_html=True)

    # Key Metrics
    _sl_data = get_stop_loss_targets(data) if _has_price else None
    _bt_data = get_buy_target_price(data, mode="classic") if _has_price else None
    _m1, _m2, _m3, _m4 = st.columns(4)
    _cur_label = "현재가(실시간)" if (rt_realtime and rt_price > 0 and data_ready) else "현재가"
    _chg_color = "#10B981" if daily_chg >= 0 else "#3B82F6"

    for _col, _lbl, _val_str, _icon_color, _icon_svg in [
        (_m1, _cur_label,  _fmt.format(last_price) if _has_price else "—",
         "#8B5CF6",
         '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>'),
        (_m2, "추천 매수가", _fmt.format(_bt_data["buy_target"]) if _bt_data else "—",
         "#10B981",
         '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>'),
        (_m3, "손절가",     _fmt.format(_sl_data["stop_8pct"]) if _sl_data else "—",
         "#F59E0B",
         '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>'
         '<line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>'),
        (_m4, "등락률",     f"{daily_chg:+.2f}%" if _has_price else "—",
         _chg_color,
         '<polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/>'),
    ]:
        _col.markdown(f"""
<div class="ma-metric-card">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px">
    <span style="font-size:.7rem;color:#94A3B8;font-weight:500">{_lbl}</span>
    <span style="color:{_icon_color}"><svg xmlns="http://www.w3.org/2000/svg" width="14" height="14"
      viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
      stroke-linecap="round" stroke-linejoin="round">{_icon_svg}</svg></span>
  </div>
  <div style="font-size:1.2rem;font-weight:700;color:#E2E8F0">{_val_str}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("---")

    # 신호 근거 박스
    _reason_html = "".join(
        f'<div style="padding:8px 0;border-bottom:1px solid {COLORS["border_md"]};font-size:0.88rem;'
        f'color:{COLORS["text"]};display:flex;align-items:flex-start;gap:8px;line-height:1.6;">'
        f'<span style="color:{_l1_fc};flex-shrink:0;">{_l1_emoji}</span>'
        f'<span style="word-break:keep-all;">{r}</span></div>'
        for r in _tl_reasons[:3]
    ) or f'<div style="color:{COLORS["text_3"]};font-size:0.88rem;">분석 데이터 수집 중...</div>'

    st.markdown(f"""
<div class="card-sm" style="border-color:{_l1_border}44;margin-bottom:12px;">
  <div style="font-size:0.65rem;color:{COLORS['text_2']};letter-spacing:1px;text-transform:uppercase;margin-bottom:10px;">
    📋 신호 근거 (TOP 3)
  </div>
  {_reason_html}
</div>""", unsafe_allow_html=True)

    # Progress bars
    _ts = advanced.get("trend_score",    50.0)
    _ms = advanced.get("momentum_score", 50.0)
    _vs = advanced.get("volume_score",   50.0)

    for _label, _score, _icon in [
        ("추세 (Trend)",    _ts, "📈"),
        ("탄력 (Momentum)", _ms, "⚡"),
        ("에너지 (Volume)", _vs, "🔋"),
    ]:
        _sc = "#26a69a" if _score >= 65 else ("#ef5350" if _score <= 35 else "#eab308")
        _bg = "rgba(38,166,154,0.12)" if _score >= 65 else ("rgba(239,83,80,0.12)" if _score <= 35 else "rgba(234,179,8,0.12)")
        st.markdown(
            f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};'
            f'border-radius:10px;padding:10px 14px;margin-bottom:8px;">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">'
            f'<span style="color:{COLORS["text_2"]};font-size:0.85rem;display:flex;align-items:center;gap:6px;">'
            f'<span>{_icon}</span><span>{_label}</span></span>'
            f'<span style="background:{_bg};color:{_sc};border-radius:20px;'
            f'padding:2px 10px;font-size:0.78rem;font-weight:700;">{_score:.0f}점</span></div>'
            f'<div style="background:{COLORS["border_md"]};border-radius:4px;height:6px;">'
            f'<div style="background:{_sc};width:{int(_score)}%;height:6px;border-radius:4px;'
            f'transition:width 0.4s ease;"></div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    # 수급 동향 (KRX 종목만)
    if data_ready and (ticker.endswith(".KS") or ticker.endswith(".KQ")):
        try:
            _inv_s = inv_data_fn(ticker)
            if _inv_s:
                _for_n  = _inv_s.get("외국인",  0) or 0
                _inst_n = _inv_s.get("기관합계", 0) or 0
                if _for_n > 0 and _inst_n > 0:
                    _inv_txt, _inv_brd = f"외국인·기관 쌍끌이 매수 중 (외국인 {_for_n:+,} / 기관 {_inst_n:+,}주)", "#22c55e"
                elif _for_n < 0 and _inst_n < 0:
                    _inv_txt, _inv_brd = f"외국인·기관 동반 매도 중 (외국인 {_for_n:+,} / 기관 {_inst_n:+,}주)", "#ef4444"
                elif _for_n > 0:
                    _inv_txt, _inv_brd = f"외국인 단독 순매수 {_for_n:+,}주 (기관 {_inst_n:+,}주)", "#4b9cf5"
                elif _inst_n > 0:
                    _inv_txt, _inv_brd = f"기관 단독 순매수 {_inst_n:+,}주 (외국인 {_for_n:+,}주)", "#4b9cf5"
                else:
                    _inv_txt, _inv_brd = f"외국인·기관 중립 (외국인 {_for_n:+,} / 기관 {_inst_n:+,}주)", "#555"
                st.markdown(
                    f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};'
                    f'border-left:3px solid {_inv_brd};border-radius:10px;'
                    f'padding:10px 14px;margin-bottom:8px;">'
                    f'<div style="font-size:0.65rem;color:{COLORS["text_2"]};letter-spacing:0.8px;'
                    f'text-transform:uppercase;margin-bottom:5px;">📊 수급 동향</div>'
                    f'<div style="font-size:0.88rem;color:{COLORS["text"]};line-height:1.5;word-break:keep-all;">'
                    f'{_inv_txt}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        except Exception:
            pass

    st.markdown("---")

    # Dead time / Breakout
    if dead_time.get("message"):
        _dt_dead = dead_time.get("is_dead", False)
        _dt_vol  = dead_time.get("volatility_14d", 0.0)
        _dt_vr   = dead_time.get("vol_ratio", 1.0)
        if _dt_dead:
            st.warning(dead_time["message"], icon="⏳")
            st.markdown(
                f'<div style="background:#1a1d2e;border-radius:7px;padding:8px 12px;'
                f'border-left:3px solid #f39c12;font-size:0.86rem;color:#aaa;margin-top:4px;">'
                f'14일 변동성 <b style="color:#fff176;">{_dt_vol:.1f}%</b> · '
                f'거래량 비율 <b style="color:#fff176;">{_dt_vr*100:.0f}%</b><br>'
                f'<span style="color:#888;">돌파 신호 전까지 신규 진입을 자제하세요.</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info(dead_time["message"], icon="📊")

    _bk_status = breakout.get("status", "wait")
    _bk_detail = breakout.get("detail", "")
    if _bk_status == "breakout_both":
        st.success(f"**돌파(Breakout) 조건 충족** — {_bk_detail}", icon="🚀")
    elif _bk_status in ("breakout_ma", "breakout_vol"):
        st.info(f"**부분 돌파 감지** — {_bk_detail}", icon="📈")
    else:
        st.info(f"**관망(Wait) 유지** — {_bk_detail}  \n*20일 MA 상향 돌파 또는 전일 거래량 200% 초과 시 진입 고려*", icon="⏸️")

    # 종합 스코어보드
    with st.expander("📊 종합 판단 스코어보드", expanded=False):
        ts = advanced.get("trend_score",    50.0)
        ms = advanced.get("momentum_score", 50.0)
        vs = advanced.get("volume_score",   50.0)

        def _sc_color(s):
            return "#a5d6a7" if s >= 65 else ("#ef9a9a" if s <= 35 else "#fff176")

        for _lbl, _sc_val, _wt, _desc in [
            ("추세 (Trend)",    ts, "40%", "EMA 배열 · ADX · 일목균형표"),
            ("탄력 (Momentum)", ms, "30%", "MACD · RSI · ROC · CCI"),
            ("에너지 (Volume)", vs, "30%", "OBV · MFI"),
        ]:
            _clr = _sc_color(_sc_val)
            st.markdown(
                f'<div style="margin-bottom:8px;">'
                f'<div style="display:flex;justify-content:space-between;font-size:0.92rem;">'
                f'<span style="color:#ccc;">{_lbl} <span style="color:#666;">({_wt})</span></span>'
                f'<b style="color:{_clr};">{_sc_val:.0f}점</b></div>'
                f'<div style="background:#2a2d3e;border-radius:4px;height:7px;margin-top:3px;">'
                f'<div style="background:{_clr};width:{int(_sc_val)}%;height:7px;border-radius:4px;"></div>'
                f'</div>'
                f'<div style="font-size:0.85rem;color:#666;margin-top:2px;">{_desc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        composite = round(ts * 0.4 + ms * 0.3 + vs * 0.3, 1)
        if composite >= 65:
            adv_txt, adv_clr = "강세 우위 — 에너지와 추세 모두 양호", "#a5d6a7"
        elif composite <= 35:
            adv_txt, adv_clr = "약세 우위 — 지표 전반 약화 중", "#ef9a9a"
        else:
            adv_txt, adv_clr = "중립 — 방향 확인 필요", "#fff176"
        st.markdown(
            f'<div style="background:#1e2130;border-radius:6px;padding:8px 12px;margin-top:6px;">'
            f'<span style="font-size:0.88rem;color:#888;">가중 종합: </span>'
            f'<b style="color:{adv_clr};">{composite:.0f}점 — {adv_txt}</b>'
            f'</div>',
            unsafe_allow_html=True,
        )

        items = advanced.get("summary_items", [])
        if items:
            st.markdown('<div style="margin-top:10px;font-size:0.75rem;color:#888;">추가 분석 항목</div>',
                        unsafe_allow_html=True)
            for it in items:
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;'
                    f'background:#12141f;border-radius:5px;padding:6px 10px;margin-top:4px;'
                    f'border-left:3px solid {it["색상"]};">'
                    f'<span style="color:#9e9e9e;font-size:0.9rem;">{it["항목"]}</span>'
                    f'<span style="color:{it["색상"]};font-size:0.9rem;font-weight:bold;">{it["상태"]}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        div_descs = advanced.get("divergence", {}).get("descriptions", [])
        for d in div_descs:
            if "하락" in d:
                st.warning(d, icon="⚠️")
            else:
                st.success(d, icon="✅")


def _render_chart_bottom_sections(*, state: dict, inv_data_fn, insider_trades_fn) -> None:
    """차트 탭 하단의 VWAP·손익·내부자 섹션을 렌더링한다."""
    ticker   = state["ticker"]
    data     = state["data"]
    close    = state["close"]
    expected = state["expected"]
    risk_adj = state["risk_adj"]
    advanced = state["advanced"]

    st.divider()
    _tech_left, _tech_right = st.columns([1, 1])

    with _tech_left:
        if not data.empty and len(data) >= 2:
            _cur      = float(data["Close"].iloc[-1])
            _is_krw_v = _cur > 500
            _pf       = "{:,.0f}" if _is_krw_v else "{:,.2f}"
            _vw_row   = data.iloc[-1]

            def _vwap_row_html(label, col, color):
                if col not in data.columns or pd.isna(_vw_row[col]):
                    return ""
                _v    = float(_vw_row[col])
                _diff = (_cur - _v) / _v * 100
                _arrow = "▲" if _diff >= 0 else "▼"
                _dc   = "#69f0ae" if _diff >= 0 else "#ef9a9a"
                return (
                    f'<div style="display:flex;justify-content:space-between;'
                    f'align-items:center;padding:5px 0;border-bottom:1px solid #2a2d3e;gap:6px;">'
                    f'<span style="font-size:0.85rem;color:{color};white-space:nowrap;flex-shrink:0;">● {label}</span>'
                    f'<span style="font-size:0.88rem;color:#ddd;white-space:nowrap;">{_pf.format(_v)}</span>'
                    f'<span style="font-size:0.85rem;color:{_dc};white-space:nowrap;flex-shrink:0;">{_arrow}{abs(_diff):.1f}%</span>'
                    f'</div>'
                )

            vwap_rows = (
                _vwap_row_html("VWAP 주간(5일)",   "VWAP_W", "#ff8f00") +
                _vwap_row_html("VWAP 월간(20일)",  "VWAP_M", "#ce93d8") +
                _vwap_row_html("VWAP 분기(60일)",  "VWAP_Q", "#80deea")
            )
            if vwap_rows:
                st.markdown("#### 📍 VWAP 위치")
                st.markdown(
                    f'<div style="background:#1a1d2e;border-radius:8px;padding:10px 14px;">'
                    f'{vwap_rows}</div>',
                    unsafe_allow_html=True,
                )

    with _tech_right:
        if expected:
            st.markdown("#### 📊 예상 수익·리스크")
            _exp_ret  = expected.get("expected_return_pct", 0.0)
            _sharpe   = expected.get("sharpe", 0.0)
            _max_dd   = expected.get("max_drawdown_pct", 0.0)
            _win_prob = expected.get("win_prob", 0.5) * 100
            _exp_color = COLORS["gain"] if _exp_ret >= 0 else COLORS["loss"]
            st.markdown(
                f'<div style="background:#1a1d2e;border-radius:8px;padding:12px 16px;">'
                f'<div style="display:flex;justify-content:space-between;margin-bottom:6px;">'
                f'<span style="color:#9e9e9e;font-size:.85rem">예상 수익률</span>'
                f'<b style="color:{_exp_color}">{_exp_ret:+.1f}%</b></div>'
                f'<div style="display:flex;justify-content:space-between;margin-bottom:6px;">'
                f'<span style="color:#9e9e9e;font-size:.85rem">샤프지수</span>'
                f'<b style="color:#E6EDF3">{_sharpe:.2f}</b></div>'
                f'<div style="display:flex;justify-content:space-between;margin-bottom:6px;">'
                f'<span style="color:#9e9e9e;font-size:.85rem">최대 낙폭</span>'
                f'<b style="color:{COLORS["loss"]}">{_max_dd:.1f}%</b></div>'
                f'<div style="display:flex;justify-content:space-between;">'
                f'<span style="color:#9e9e9e;font-size:.85rem">승률 추정</span>'
                f'<b style="color:#E6EDF3">{_win_prob:.0f}%</b></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            if risk_adj and risk_adj.get("conservative_target"):
                _ct  = risk_adj.get("conservative_target", 0.0)
                _csl = risk_adj.get("conservative_stoploss", 0.0)
                _cwp = risk_adj.get("win_prob", 0.0)
                if _cwp < 50:
                    st.warning(
                        f"**⚠️ 저승률 보수 조정** — 승률 {_cwp:.0f}%로 50% 미만  \n"
                        f"목표 수익률 → **{_ct:+.1f}%**  |  손절 라인 → **{_csl:+.1f}%**",
                        icon="⚠️",
                    )

    # 손절/목표가 상세
    st.divider()
    _sl = get_stop_loss_targets(data) if not close.empty else None
    if _sl:
        with st.expander("🛡️ 손절가·목표가 상세", expanded=False):
            _is_krw_sl = float(close.iloc[-1]) > 500 if not close.empty else True
            _sfmt = "{:,.0f}" if _is_krw_sl else "{:,.2f}"
            for _label, _key, _color in [
                ("손절가 (−8%)",        "stop_8pct",  COLORS["loss"]),
                ("1차 목표 (2R)",        "target_2r",  COLORS["gain"]),
                ("2차 목표 (3R)",        "target_3r",  COLORS["gain"]),
                ("52주 고가",            "high_52w",   COLORS["text_2"]),
                ("볼린저밴드 상단",      "bb_upper",   "#ce93d8"),
            ]:
                _v = _sl.get(_key)
                if _v:
                    st.markdown(
                        f'<div style="display:flex;justify-content:space-between;'
                        f'padding:6px 0;border-bottom:1px solid {COLORS["border_md"]};">'
                        f'<span style="color:{COLORS["text_2"]};font-size:.88rem">{_label}</span>'
                        f'<b style="color:{_color}">{_sfmt.format(_v)}</b></div>',
                        unsafe_allow_html=True,
                    )

    # SEC 내부자 거래 (미국 주식)
    st.markdown("---")
    is_us = "." not in ticker
    if is_us:
        st.markdown("### 🏦 SEC 내부자 거래 (Form 4, 최근 90일)")
        st.caption("데이터 소스: SEC EDGAR (edgar.sec.gov)")
        with st.spinner("SEC EDGAR에서 내부자 거래 조회 중..."):
            insider_df = insider_trades_fn(ticker)
        if insider_df is not None and not insider_df.empty:
            st.dataframe(
                insider_df,
                use_container_width=True,
                hide_index=True,
                column_config={"링크": st.column_config.LinkColumn("SEC 공시 링크", display_text="바로가기")},
            )
            st.caption(f"총 {len(insider_df)}건의 Form 4 공시 발견")
        else:
            st.info("최근 90일 내 Form 4 공시가 없거나 데이터를 불러올 수 없습니다.")

        st.markdown("""
#### 📌 내부자 거래 해석 가이드
| 신호 | 의미 | 투자 판단 |
|------|------|-----------|
| 임원 다수 동시 매수 | 내부자 확신 → 강한 매수 신호 | ★★★ |
| 임원 대량 매도 ($1M+) | 주가 고점 가능성 | 주의 |
| 소량 분산 매도 | 세금·다각화 목적, 무의미 | 참고만 |
""")
    else:
        st.markdown("### 🏦 기관/내부자 데이터")
        st.info(
            "SEC EDGAR Form 4 내부자 거래는 **미국 주식 전용** 기능입니다.  \n"
            "한국 주식의 경우 [DART 전자공시](https://dart.fss.or.kr)에서 조회 가능합니다."
        )

    # 투자 법칙 레퍼런스
    st.markdown("---")
    with st.expander("📚 적용 투자 법칙 레퍼런스"):
        st.markdown("""
| 투자자 | 법칙 | 기준값 | 적용 |
|--------|------|--------|------|
| 벤저민 그레이엄 | PBR×PER | < 22.5 | 저평가 판단 |
| 워렌 버핏 | ROE 지속성 | ≥ 15% 연속 | 우량 기업 선별 |
| 워렌 버핏 | 부채비율 | < 50% | 재무 안정성 |
| 피터 린치 | PEG 비율 | < 1.0 매수, > 2.0 매도 | 성장 대비 가격 |
| 윌리엄 오닐 | CANSLIM-N | 52주 신고가 근접 | 모멘텀 돌파 |
| 윌리엄 오닐 | 손절 원칙 | 매수가 -7~8% | 손실 제한 |
""")


# ═════════════════════════════════════════════════════════════════════════════
# 펀더멘털 탭
# ═════════════════════════════════════════════════════════════════════════════

def render_fund_tab(
    tab,
    *,
    state: dict,
    api_keys: dict,
    etf_fundamental_fn,
    check_is_etf_fn,
    inv_data_fn,
    insider_trades_fn,
) -> None:
    """펀더멘털 & ETF 분석 탭 전체를 렌더링한다."""
    with tab:
        ticker        = state["ticker"]
        sname         = state["sname"]
        data_ready    = state["data_ready"]
        asname        = state.get("asname", sname)
        data          = state.get("data", pd.DataFrame())
        signals       = state.get("signals", {})
        fund_info     = state.get("fund_info", {})
        fund_score_data = state.get("fund_score_data", {})
        dart_api_key  = api_keys.get("dart", "")

        _is_etf = data_ready and check_is_etf_fn(ticker)

        if _is_etf:
            st.subheader(f"📊 {asname or sname} ETF 분석")
            st.caption("ETF는 괴리율·운용보수·추적오차 중심으로 평가합니다.")

            with st.spinner("ETF 지표 로딩 중... (KRX API)"):
                _etf_data  = etf_fundamental_fn(ticker)
                _etf_score = calculate_etf_score(_etf_data)

            _data_status = _etf_data.get("data_status", "ok")
            if _etf_data.get("cache_used"):
                st.info(f"⏳ {_data_status}", icon="🔄")
            elif _data_status not in ("ok", "krx_api+fdr"):
                st.warning(f"KRX API 상태: {_data_status}", icon="⚠️")

            _em1, _em2, _em3, _em4, _em5 = st.columns(5)
            _price   = _etf_data.get("price")
            _nav     = _etf_data.get("nav")
            _premium = _etf_data.get("nav_premium")
            _er      = _etf_data.get("expense_ratio")
            _te      = _etf_data.get("tracking_error")
            _div     = _etf_data.get("dividend_yield")
            _aum     = _etf_data.get("aum")
            _sector  = _etf_data.get("sector", "")

            _em1.metric("현재가",       f"{_price:,.0f}원" if _price else "N/A")
            _em2.metric("NAV",          f"{_nav:,.0f}원"   if _nav   else "N/A",
                        delta=f"괴리율 {_premium:+.2f}%" if _premium is not None else None)
            _em3.metric("운용보수(ER)", f"{_er:.3f}%"      if _er   is not None else "N/A",
                        help="연간 운용보수. 낮을수록 장기 복리 수익 유리")
            _em4.metric("추적오차",     f"{_te:.2f}%"      if _te   is not None else "N/A",
                        help="기초지수 대비 추종 오차. 낮을수록 지수를 정밀하게 추종")
            _em5.metric("배당수익률",   f"{_div:.2f}%"     if _div  is not None else "N/A")

            _src_label = {"krx_api": "KRX 공공 API", "krx_api+fdr": "KRX API + FDR", "error": "오류"}.get(
                _etf_data.get("source", ""), _etf_data.get("source", "")
            )
            _aum_txt = f"순자산(AUM): **{_aum:,.0f}억원**  |  " if _aum else ""
            st.caption(f"{_aum_txt}섹터: **{_sector}**  |  데이터: {_src_label}")
            st.markdown("---")

            _es_col, _er_col = st.columns([1, 2])
            with _es_col:
                _etf_s   = _etf_score.get("etf_score", 0.0)
                _etf_lbl = _etf_score.get("etf_label", "N/A")
                if _etf_s >= 1.5:
                    _ebg, _efc = "#1a2f3a", "#80cbc4"
                elif _etf_s <= -0.5:
                    _ebg, _efc = "#3a2a1a", "#ffcc80"
                else:
                    _ebg, _efc = "#1e2130", "#bdbdbd"
                st.markdown(f"""
<div class="signal-box" style="background:{_ebg};">
  <div style="font-size:0.7rem;color:#888;margin-bottom:3px;letter-spacing:1px;">📊 ETF 투자 판정</div>
  <div style="font-size:1.3rem;font-weight:bold;color:{_efc};">{_etf_lbl}</div>
  <div style="font-size:0.9rem;color:#aaa;margin-top:4px;">점수: <b style="color:{_efc};">{_etf_s:+.1f}</b> / ±6.5</div>
</div>
""", unsafe_allow_html=True)

                st.markdown("**항목별 점수**")
                _bd = _etf_score.get("score_breakdown", {})
                for _lbl, _rng, _clr in [
                    ("괴리율",    "±3",   "#80cbc4"),
                    ("운용보수",  "±2",   "#a5d6a7"),
                    ("추적오차",  "±1.5", "#ce93d8"),
                    ("배당수익률","±0.5", "#ffcc80"),
                ]:
                    _sv  = _bd.get(_lbl, 0.0)
                    _max = float(_rng.replace("±", ""))
                    _pct = int((_sv + _max) / (_max * 2) * 100)
                    st.markdown(
                        f'<div style="margin-bottom:5px;">'
                        f'<div style="display:flex;justify-content:space-between;font-size:0.75rem;">'
                        f'<span style="color:#ccc;">{_lbl} <span style="color:#555;">({_rng})</span></span>'
                        f'<b style="color:{_clr};">{_sv:+.1f}</b></div>'
                        f'<div style="background:#2a2d3e;border-radius:3px;height:5px;">'
                        f'<div style="background:{_clr};width:{_pct}%;height:5px;border-radius:3px;"></div>'
                        f'</div></div>',
                        unsafe_allow_html=True,
                    )

            with _er_col:
                st.markdown("**📋 ETF 투자 근거**")
                _pos_kw = ["매수", "유리", "양호", "낮음", "정밀", "할인", "최저"]
                _neg_kw = ["주의", "높음", "과열", "프리미엄", "불량", "잠식", "위험"]
                for _r in _etf_score.get("etf_reasons", []):
                    if any(k in _r for k in _pos_kw):
                        st.success(_r, icon="✅")
                    elif any(k in _r for k in _neg_kw):
                        st.warning(_r, icon="⚠️")
                    else:
                        st.info(_r, icon="ℹ️")
                if not _etf_score.get("etf_reasons"):
                    st.info("ETF 지표 데이터를 불러올 수 없습니다.")

            st.markdown("---")

            if _premium is not None:
                _prem_color = "#ef5350" if _premium > 1 else ("#26a69a" if _premium < -0.5 else "#8B949E")
                st.markdown(
                    f'<div style="background:#161B22;border:1px solid #30363D;border-radius:12px;padding:16px 18px;">'
                    f'<div style="font-size:0.72rem;color:#8B949E;letter-spacing:0.6px;margin-bottom:8px;">📐 NAV 괴리율이란?</div>'
                    f'<div style="font-size:0.95rem;color:#E6EDF3;margin-bottom:6px;">'
                    f'현재가(<b>{_price:,.0f}원</b>) vs NAV(<b>{_nav:,.0f}원</b>)</div>'
                    f'<div style="font-size:1.2rem;font-weight:700;color:{_prem_color};">'
                    f'{"🔴 프리미엄" if _premium > 0 else "🟢 할인"} {_premium:+.2f}%</div>'
                    f'<div style="font-size:0.78rem;color:#8B949E;margin-top:8px;line-height:1.6;word-break:keep-all;">'
                    f'{"시장가가 NAV보다 높음 → 고평가. 매수 시 주의 필요" if _premium > 0.5 else "시장가가 NAV보다 낮음 → 저평가. 잠재적 매수 기회"}'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

            st.markdown("---")
            st.markdown("### 🗂️ 상위 구성종목 (PDF)")
            _holdings = _etf_data.get("top_holdings", [])
            if _holdings:
                _h_rows = [
                    {
                        "종목 티커": h.get("ticker", ""),
                        "종목명":    h.get("name", ""),
                        "비중(%)":   f'{h["weight"]:.2f}' if h.get("weight") is not None else "N/A",
                    }
                    for h in _holdings[:10]
                ]
                st.dataframe(pd.DataFrame(_h_rows), use_container_width=True, hide_index=True)
                st.caption("상위 구성종목이 해당 섹터의 시장 흐름을 직접 반영합니다.")
            else:
                st.info("구성종목 데이터를 불러올 수 없습니다.")

            if data_ready:
                st.markdown("### 📈 기술적 분석 (ETF 동일 적용)")
                _sig_score = signals.get("score", 0)
                _sig_label = signals.get("label", "N/A")
                _sig_clr   = "#a5d6a7" if _sig_score >= 3 else ("#ef9a9a" if _sig_score <= -3 else "#bdbdbd")
                st.markdown(
                    f'<div style="background:#161B22;border:1px solid #30363D;border-radius:12px;'
                    f'padding:14px 18px;display:flex;align-items:center;justify-content:space-between;">'
                    f'<span style="font-size:0.85rem;color:#8B949E;">기술적 신호</span>'
                    f'<b style="color:{_sig_clr};font-size:1rem;">{_sig_label} ({_sig_score:+.1f}점)</b>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                for _reason in (signals.get("reasons") or [])[:5]:
                    st.caption(f"• {_reason}")

        if not _is_etf:
            _is_krx = ticker.endswith(".KS") or ticker.endswith(".KQ")
            if _is_krx:
                _fund_db_ok = False
                try:
                    from fundamental_db import get_last_update, needs_update, update_market
                    _fund_db_ok = True
                except Exception:
                    pass
                if _fund_db_ok:
                    _krx_market = "KOSPI" if ticker.endswith(".KS") else "KOSDAQ"
                    _last_upd   = get_last_update(_krx_market)
                    _needs      = needs_update(_krx_market)
                    _db_col1, _db_col2 = st.columns([3, 1])
                    with _db_col1:
                        if _last_upd:
                            _upd_dt = _last_upd[:10]
                            if _needs:
                                st.warning(f"📅 펀더멘털 DB 마지막 갱신: {_upd_dt} — 분기 업데이트가 필요합니다.")
                            else:
                                st.success(f"✅ 펀더멘털 DB 마지막 갱신: {_upd_dt} (pykrx 기준)")
                        else:
                            st.warning("⚠️ 펀더멘털 DB가 비어 있습니다. 전체 업데이트를 실행하세요.")
                    with _db_col2:
                        if st.button("🔄 전체 업데이트", key="btn_fund_update"):
                            with st.spinner(f"{_krx_market} 전체 종목 업데이트 중 (수 분 소요)..."):
                                _cnt = update_market(_krx_market)
                            st.success(f"{_cnt}개 종목 업데이트 완료!")
                            st.rerun()

            col_f1, col_f2 = st.columns([1, 1])

            with col_f1:
                st.markdown("### 🏛️ 장투 신호 근거 (투자법칙)")
                fs   = fund_score_data.get("fund_score", 0)
                flbl = fund_score_data.get("fund_label", "N/A")
                if fs >= 3:
                    fbg, ffc = "#1a2f3a", "#80cbc4"
                elif fs <= -2:
                    fbg, ffc = "#3a2a1a", "#ffcc80"
                else:
                    fbg, ffc = "#1e2130", "#bdbdbd"
                st.markdown(f"""
<div class="signal-box" style="background:{fbg};">
  <div style="font-size:0.7rem;color:#888;margin-bottom:3px;letter-spacing:1px;">🏛️ 장투 신호</div>
  <div style="font-size:1.25rem;font-weight:bold;color:{ffc};">🏛️ {flbl}</div>
  <div style="font-size:0.8rem;color:#aaa;margin-top:3px;">장투 점수: <b style="color:{ffc};">{fs:+.1f}</b></div>
</div>
""", unsafe_allow_html=True)

                def _fund_bar(label, score, weight):
                    c = "#a5d6a7" if score >= 65 else ("#ef9a9a" if score <= 35 else "#fff176")
                    st.markdown(
                        f'<div style="margin-bottom:5px;">'
                        f'<div style="display:flex;justify-content:space-between;font-size:0.75rem;">'
                        f'<span style="color:#ccc;">{label} <span style="color:#555;">({weight})</span></span>'
                        f'<b style="color:{c};">{score:.0f}</b></div>'
                        f'<div style="background:#2a2d3e;border-radius:3px;height:5px;">'
                        f'<div style="background:{c};width:{int(score)}%;height:5px;border-radius:3px;"></div>'
                        f'</div></div>',
                        unsafe_allow_html=True,
                    )
                _fund_bar("성장성 (PEG·매출)",       fund_score_data.get("sub_growth", 50), "40%")
                _fund_bar("수익성 (ROE·FCF·OCF)",    fund_score_data.get("sub_profit", 50), "30%")
                _fund_bar("안정성 (그레이엄·부채)",  fund_score_data.get("sub_stable", 50), "20%")
                _fund_bar("모멘텀 (52주·주주환원)",  fund_score_data.get("sub_moment", 50), "10%")

                st.markdown("**📋 투자법칙 신호 근거**")
                _pos_kw = ["저평가", "우량", "충족", "탁월", "양호", "성장", "모멘텀", "지속성", "환원", "높음", "품질"]
                _neg_kw = ["경고", "위험", "부진", "고평가", "감소", "소진", "낮음", "이슈", "미흡", "단발성"]
                for r in fund_score_data.get("fund_reasons", []):
                    if any(k in r for k in _pos_kw):
                        st.success(r, icon="🔺")
                    elif any(k in r for k in _neg_kw):
                        st.error(r, icon="🔻")
                    else:
                        st.info(r, icon="🔷")
                if not fund_score_data.get("fund_reasons"):
                    st.info("펀더멘털 데이터를 불러올 수 없습니다.")

            with col_f2:
                st.markdown("### 📈 시장 가치 지표")

                def _fmt_money(v):
                    if abs(v) >= 1e12: return f"{v/1e12:.2f}조"
                    if abs(v) >= 1e8:  return f"{v/1e8:.0f}억"
                    return f"{v:,.0f}"

                def _fund_row(label, key, fmt_fn, law_ref=""):
                    val = fund_info.get(key)
                    if val is not None and pd.notna(val):
                        try:
                            return {"지표": label, "값": fmt_fn(val), "참고 법칙": law_ref}
                        except Exception:
                            pass
                    return {"지표": label, "값": "N/A", "참고 법칙": law_ref}

                mc  = fund_info.get("market_cap")
                fcf = fund_info.get("free_cashflow")
                mkt_rows = [
                    _fund_row("시가총액",            "market_cap",        _fmt_money, ""),
                    _fund_row("PER (주가수익비율)",  "per",               lambda v: f"{v:.2f}x", "그레이엄: PBR×PER < 22.5"),
                    _fund_row("PBR (주가순자산비율)","pbr",               lambda v: f"{v:.2f}x", "그레이엄: PBR < 1.0 선호"),
                    _fund_row("PSR (주가매출비율)",  "psr",               lambda v: f"{v:.2f}x", "1.0 이하 저평가 기준"),
                    _fund_row("Forward PER",         "forward_pe",        lambda v: f"{v:.2f}x", "성장 기대치 반영"),
                    _fund_row("EPS (TTM)",           "eps_ttm",           lambda v: f"{v:,.2f}", ""),
                    _fund_row("ROE (자기자본수익률)","roe",               lambda v: f"{v*100:.1f}%", "버핏: 15% 이상"),
                    _fund_row("영업이익률",          "operating_margins", lambda v: f"{v*100:.1f}%", ""),
                    _fund_row("부채비율",            "debt_equity",       lambda v: f"{v:.0f}%", "버핏: 50% 이하"),
                    _fund_row("매출 성장률 (YoY)",   "revenue_growth",    lambda v: f"{v*100:+.1f}%", "린치: 20%+"),
                    _fund_row("순이익 성장률 (YoY)", "earnings_growth",   lambda v: f"{v*100:+.1f}%", "CANSLIM: 25%+"),
                ]
                if fcf and mc and mc > 0:
                    mkt_rows.insert(4, {"지표": "FCF Yield", "값": f"{fcf/mc*100:.1f}%", "참고 법칙": "버핏: 5% 이상"})
                _roe_mean = fund_score_data.get("roe_mean")
                _roe_std  = fund_score_data.get("roe_std")
                if _roe_mean is not None:
                    _roe_consist = f"평균 {_roe_mean:.1f}% / 편차 {_roe_std:.1f}%p" if _roe_std is not None else f"평균 {_roe_mean:.1f}%"
                    mkt_rows.append({"지표": "ROE 지속성 (다년)", "값": _roe_consist, "참고 법칙": "버핏: 평균≥15% & 편차≤5%"})
                _ocf_ni = fund_score_data.get("ocf_ni_ratio")
                if _ocf_ni is not None:
                    mkt_rows.append({"지표": "OCF/순이익 (현금질)", "값": f"{_ocf_ni:.2f}x", "참고 법칙": "> 1.0 = 이익 신뢰성 높음"})
                _sh = fund_score_data.get("shareholder_yield")
                if _sh is not None:
                    mkt_rows.append({"지표": "주주환원율", "값": f"{_sh:.1f}%", "참고 법칙": "배당+자사주 / 시총 (≥3% 양호)"})
                st.dataframe(pd.DataFrame(mkt_rows), use_container_width=True, hide_index=True)

                st.markdown("### 💰 재무 핵심 지표")
                _is_krx_f = ticker.endswith(".KS") or ticker.endswith(".KQ")
                dart_fin  = {}
                if _is_krx_f and dart_api_key:
                    with st.spinner("DART 재무제표 조회 중..."):
                        try:
                            from fundamental_db import get_dart_financials
                            dart_fin = get_dart_financials(ticker, dart_api_key)
                        except Exception:
                            pass

                def _fin_val(dart_key, yf_key):
                    if dart_fin.get(dart_key) is not None:
                        return f"{dart_fin[dart_key]:,.0f} 억원"
                    v = fund_info.get(yf_key)
                    if v is not None and pd.notna(v):
                        return _fmt_money(v)
                    return "N/A"

                fin_src  = f"DART {dart_fin.get('year','')}" if dart_fin else "yfinance"
                fin_rows = [
                    {"지표": "매출액",          "값": _fin_val("revenue",         "total_revenue")},
                    {"지표": "영업이익",         "값": _fin_val("operating_income", "operating_income")},
                    {"지표": "당기순이익",       "값": _fin_val("net_income",       "net_income")},
                    {"지표": "잉여현금흐름(FCF)", "값": _fmt_money(fcf) if fcf else "N/A"},
                ]
                st.dataframe(pd.DataFrame(fin_rows), use_container_width=True, hide_index=True)
                _src = fund_info.get("source", "yfinance")
                st.caption(f"시장 지표: **{_src}** | 재무 지표: **{fin_src}**")

            st.markdown("---")
            st.markdown("### 🎖️ 투자 거장의 한 줄 평")
            _verdicts   = fund_score_data.get("master_verdicts", {})
            _master_meta = [
                ("그레이엄", "📖 벤저민 그레이엄", "#80cbc4", "안전마진·가치투자의 아버지"),
                ("버핏",    "🏛️ 워렌 버핏",       "#a5d6a7", "ROE 지속성·경제적 해자"),
                ("린치",    "🚀 피터 린치",        "#ce93d8", "PEG·성장주 발굴"),
                ("오닐",    "🔥 윌리엄 오닐",      "#ffcc80", "신고가·CANSLIM"),
            ]
            _vcols = st.columns(4)
            for _vcol, (_key, _name, _clr, _sub) in zip(_vcols, _master_meta):
                _v       = _verdicts.get(_key, {})
                _icon    = _v.get("icon", "—")
                _verdict = _v.get("판정", "N/A")
                _comment = _v.get("comment", "데이터 부족")
                _vcol.markdown(
                    f'<div style="background:#161B22;border-radius:12px;padding:14px;'
                    f'border:1px solid #30363D;border-top:3px solid {_clr};min-height:130px;">'
                    f'<div style="font-size:0.65rem;color:#8B949E;letter-spacing:0.6px;margin-bottom:5px;">{_sub}</div>'
                    f'<div style="font-size:0.85rem;font-weight:700;color:{_clr};margin-bottom:6px;">{_name}</div>'
                    f'<div style="font-size:1rem;font-weight:700;color:#E6EDF3;margin-bottom:6px;">{_icon} {_verdict}</div>'
                    f'<div style="font-size:0.75rem;color:#8B949E;line-height:1.6;word-break:keep-all;">{_comment}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            _is_krx_f = ticker.endswith(".KS") or ticker.endswith(".KQ")
            if _is_krx_f:
                st.markdown("---")
                st.markdown("### 📊 전날 투자자별 매매 동향")
                with st.spinner("투자자 데이터 조회 중..."):
                    _inv_f = inv_data_fn(ticker)
                if _inv_f:
                    _inv_date_f = _inv_f.get("date", "")
                    if len(_inv_date_f) == 8:
                        _inv_date_f = f"{_inv_date_f[:4]}.{_inv_date_f[4:6]}.{_inv_date_f[6:]}"
                    st.caption(f"기준일: {_inv_date_f}  |  단위: 주(株)  |  출처: Naver Finance")
                    _inv_f_cols = st.columns(3)
                    for col, key, label in zip(
                        _inv_f_cols,
                        ["외국인", "기관합계", "개인"],
                        ["🌐 외국인", "🏦 기관", "👤 개인"],
                    ):
                        val = _inv_f.get(key)
                        if val is not None:
                            _sign  = "+" if val > 0 else ""
                            _color = "#26a69a" if val > 0 else ("#ef5350" if val < 0 else "#8B949E")
                            _inv_bg = "rgba(38,166,154,0.08)" if val > 0 else ("rgba(239,83,80,0.08)" if val < 0 else "#161B22")
                            col.markdown(
                                f'<div style="background:{_inv_bg};border:1px solid #30363D;border-radius:10px;'
                                f'padding:14px;text-align:center;">'
                                f'<div style="font-size:0.72rem;color:#8B949E;letter-spacing:0.5px;margin-bottom:6px;">{label}</div>'
                                f'<div style="font-size:1.1rem;font-weight:700;color:{_color};">{_sign}{val:,}</div>'
                                f'<div style="font-size:0.68rem;color:#484F58;margin-top:3px;">주(株)</div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                        else:
                            col.markdown(
                                f'<div style="background:#161B22;border:1px solid #30363D;border-radius:10px;'
                                f'padding:14px;text-align:center;">'
                                f'<div style="font-size:0.72rem;color:#8B949E;margin-bottom:6px;">{label}</div>'
                                f'<div style="font-size:1.1rem;color:#484F58;">N/A</div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                else:
                    st.caption("투자자 매매 동향 데이터를 불러올 수 없습니다.")

            st.markdown("---")
            st.markdown("### 🛡️ 손절·익절 레벨 상세 (오닐·ATR 기반)")
            sl = get_stop_loss_targets(data) if not data.empty else None
            if sl:
                s1, s2, s3, s4, s5 = st.columns(5)
                s1.metric("현재가",         f"{sl['current']:,.2f}")
                s2.metric("손절 (오닐 8%)", f"{sl['stop_8pct']:,.2f}", f"{(sl['stop_8pct']/sl['current']-1)*100:.1f}%")
                s3.metric("손절 (ATR×2.5)", f"{sl['stop_atr']:,.2f}",  f"{(sl['stop_atr']/sl['current']-1)*100:.1f}%",
                          help=f"ATR(14일) × 2.5. 현재 ATR: {sl['atr']:,.2f} ({sl['atr_ratio']:.2f}%)")
                s4.metric("목표 2R",        f"{sl['target_2r']:,.2f}", f"{(sl['target_2r']/sl['current']-1)*100:+.1f}%")
                s5.metric("목표 3R",        f"{sl['target_3r']:,.2f}", f"{(sl['target_3r']/sl['current']-1)*100:+.1f}%")
                st.markdown(
                    f"> **ATR (14일):** {sl['atr']:,.2f} ({sl['atr_ratio']:.2f}%)  "
                    f"> **52주 고가:** {sl['high_52w']:,.2f}\n"
                    f"> 📌 오닐 원칙: 매수가 대비 -7~8% 도달 시 무조건 손절 | 목표 3R = 리스크의 3배 수익"
                )

            st.markdown("---")
            is_us = "." not in ticker
            if is_us:
                st.markdown("### 🏦 SEC 내부자 거래 (Form 4, 최근 90일)")
                st.caption("데이터 소스: SEC EDGAR (edgar.sec.gov) — 완전 무료 공식 API")
                with st.spinner("SEC EDGAR에서 내부자 거래 조회 중..."):
                    insider_df = insider_trades_fn(ticker)
                if insider_df is not None and not insider_df.empty:
                    st.dataframe(
                        insider_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={"링크": st.column_config.LinkColumn("SEC 공시 링크", display_text="바로가기")},
                    )
                    st.caption(f"총 {len(insider_df)}건의 Form 4 공시 발견")
                else:
                    st.info("최근 90일 내 Form 4 공시가 없거나 데이터를 불러올 수 없습니다.")
                st.markdown("""
#### 📌 내부자 거래 해석 가이드
| 신호 | 의미 | 투자 판단 |
|------|------|-----------|
| 임원 다수 동시 매수 | 내부자 확신 → 강한 매수 신호 | ★★★ |
| 임원 대량 매도 ($1M+) | 주가 고점 가능성 | 주의 |
| 소량 분산 매도 | 세금·다각화 목적, 무의미 | 참고만 |
""")
            else:
                st.markdown("### 🏦 기관/내부자 데이터")
                st.info(
                    "SEC EDGAR Form 4 내부자 거래는 **미국 주식 전용** 기능입니다.  \n"
                    "한국 주식의 경우 [DART 전자공시](https://dart.fss.or.kr)에서 조회 가능합니다."
                )

            st.markdown("---")
            with st.expander("📚 적용 투자 법칙 레퍼런스"):
                st.markdown("""
| 투자자 | 법칙 | 기준값 | 적용 |
|--------|------|--------|------|
| 벤저민 그레이엄 | PBR×PER | < 22.5 | 저평가 판단 |
| 워렌 버핏 | ROE 지속성 | ≥ 15% 연속 | 우량 기업 선별 |
| 워렌 버핏 | 부채비율 | < 50% | 재무 안정성 |
| 워렌 버핏 | FCF Yield | > 5% | 현금창출 능력 |
| 피터 린치 | PEG 비율 | < 1.0 매수, > 2.0 매도 | 성장 대비 가격 |
| 피터 린치 | 매출 성장 | YoY ≥ 20% | 텐배거 후보 |
| 윌리엄 오닐 | CANSLIM-N | 52주 신고가 근접 | 모멘텀 돌파 |
| 윌리엄 오닐 | 손절 원칙 | 매수가 -7~8% | 손실 제한 |
| 윌리엄 오닐 | 3:1 R/R | 목표 = 리스크×3 | 익절 목표 |

> 📘 참고: 《현명한 투자자》(그레이엄) · 《전설로 떠나는 월가의 영웅》(린치) · 《최고의 주식 최적의 타이밍》(오닐)
""")


# ═════════════════════════════════════════════════════════════════════════════
# 포트폴리오 탭
# ═════════════════════════════════════════════════════════════════════════════

@st.fragment
def _render_pf_body(
    *,
    db_logout,
    db_get_user,
    db_get_portfolio,
    db_get_trade_history,
    db_sell_item,
    db_delete_portfolio,
    db_save_recommendation,
    db_get_rec_history,
    db_clear_trade_history,
    ticker_name_map_fn,
    realtime_price_fn,
    get_stock_data_fn,
    now_kst_fn,
    cookie_mgr,
    has_cookie_mgr,
) -> None:
    """로그인된 포트폴리오 본문 — @st.fragment로 매도·삭제·초기화 시 탭만 재렌더링."""
    _tok  = st.session_state.get("auth_token")
    _uid  = st.session_state.get("auth_user_id")
    _mail = st.session_state.get("auth_email")

    _hdr_col, _out_col = st.columns([5, 1])
    _hdr_col.markdown(f"**💼 내 포트폴리오** — `{_mail}`")
    with _out_col:
        if st.button("로그아웃", use_container_width=True, key="pf_logout"):
            db_logout(_tok)
            st.session_state["auth_token"]   = None
            st.session_state["auth_user_id"] = None
            st.session_state["auth_email"]   = None
            if has_cookie_mgr and cookie_mgr:
                cookie_mgr.delete("auth_token")
            st.rerun()

    st.divider()

    if not _tok or not db_get_user(_tok):
        st.session_state["auth_token"]   = None
        st.session_state["auth_user_id"] = None
        st.session_state["auth_email"]   = None
        if has_cookie_mgr and cookie_mgr:
            cookie_mgr.delete("auth_token")
        st.warning("세션이 만료되었습니다. 다시 로그인해 주세요.")
        st.rerun()
        return

    _items = db_get_portfolio(_uid)
    _pf_nm: dict[str, str] = ticker_name_map_fn() if _items else {}
    try:
        _trade_history: list[dict] = db_get_trade_history(_uid)
    except Exception:
        _trade_history = []

    _pf_tickers = list({it["ticker"] for it in _items}) if _items else []
    _pf_prices: dict[str, float] = {}
    if _pf_tickers:
        try:
            _pf_raw = yf.download(
                _pf_tickers, period="1d", interval="1m", auto_adjust=True,
                progress=False, threads=True,
            )
            for _t in _pf_tickers:
                try:
                    _s = (_pf_raw["Close"][_t] if isinstance(_pf_raw.columns, pd.MultiIndex)
                          else _pf_raw["Close"]).dropna()
                    if not _s.empty:
                        _pf_prices[_t] = float(_s.iloc[-1])
                except Exception:
                    pass
        except Exception:
            pass

        for _t in [t for t in _pf_tickers if not _pf_prices.get(t)]:
            try:
                _lp = float(yf.Ticker(_t).fast_info.last_price)
                if _lp > 0:
                    _pf_prices[_t] = _lp
            except Exception:
                pass

        _still_missing = [t for t in _pf_tickers if not _pf_prices.get(t)]
        if _still_missing:
            try:
                _pf_raw2 = yf.download(_still_missing, period="2d", auto_adjust=True, progress=False, threads=True)
                for _t in _still_missing:
                    try:
                        _s = (_pf_raw2["Close"][_t] if isinstance(_pf_raw2.columns, pd.MultiIndex)
                              else _pf_raw2["Close"]).dropna()
                        if not _s.empty:
                            _pf_prices[_t] = float(_s.iloc[-1])
                    except Exception:
                        pass
            except Exception:
                pass

    _usd_krw = 1300.0
    try:
        _fx_raw = yf.download("USDKRW=X", period="2d", auto_adjust=True, progress=False)
        if not _fx_raw.empty:
            _fx_s = (_fx_raw["Close"] if "Close" in _fx_raw.columns else _fx_raw.iloc[:, 0]).dropna()
            if not _fx_s.empty:
                _usd_krw = float(_fx_s.iloc[-1])
    except Exception:
        pass

    def _krw(price: float, ticker: str) -> float:
        return price if ticker.upper().endswith((".KS", ".KQ")) else price * _usd_krw

    _cum_sell_krw   = sum(_krw(t["sell_price"] * t["quantity"], t["ticker"]) for t in _trade_history)
    _cum_profit_krw = sum(_krw(t["net_profit"],                 t["ticker"]) for t in _trade_history)
    _cum_buy_krw    = sum(_krw(t["buy_price"]  * t["quantity"], t["ticker"]) for t in _trade_history)

    st.markdown("""
<div style="background:rgba(139,92,246,0.08);border:1px solid rgba(139,92,246,0.25);
        border-radius:12px;padding:12px 16px;margin-bottom:8px;
        display:flex;align-items:center;gap:10px">
  <span style="font-size:1.2rem">➕</span>
  <div>
    <div style="font-size:.85rem;font-weight:600;color:#C4B5FD">종목 추가는 사이드바(좌측)에서</div>
    <div style="font-size:.75rem;color:#94A3B8;margin-top:2px">로그인 상태에서 사이드바 하단 '포트폴리오 종목 추가' 섹션 이용</div>
  </div>
</div>
""", unsafe_allow_html=True)

    if not _items:
        st.info("보유 종목이 없습니다. 차트 분석 탭에서 평단가를 입력한 뒤 '포트폴리오에 추가' 버튼을 눌러보세요.", icon="💡")
    else:
        _total_cost = sum(_krw(it["avg_price"], it["ticker"]) * it["quantity"] for it in _items)
        _total_val  = sum(_krw(_pf_prices.get(it["ticker"], it["avg_price"]), it["ticker"]) * it["quantity"] for it in _items)
        _total_pnl  = _total_val - _total_cost
        _total_pnl_pct = (_total_pnl / _total_cost * 100) if _total_cost else 0.0

        _hdr_total_in = _total_cost + _cum_buy_krw
        st.session_state["_pf_header_summary"] = {
            "total_val":     _total_val,
            "total_pnl":     _total_pnl,
            "total_pnl_pct": _total_pnl_pct,
            "overall_pct":   ((_total_val + _cum_sell_krw) / _hdr_total_in * 100 - 100) if _hdr_total_in > 0 else None,
        }

        _pnl_color    = "#10B981" if _total_pnl >= 0 else "#3B82F6"
        _profit_color = "#10B981" if _cum_profit_krw >= 0 else "#3B82F6"
        _pnl_pct_str  = f"{_total_pnl_pct:+.2f}%"
        _m1, _m2, _m3, _m4 = st.columns(4)
        _m1.markdown(f"""
<div class="ma-metric-card">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px">
    <span style="font-size:.75rem;color:#94A3B8;font-weight:500;letter-spacing:.3px">총 매수 금액</span>
  </div>
  <div style="font-size:1.35rem;font-weight:700;color:#E2E8F0;line-height:1.2">₩{_total_cost:,.0f}</div>
  <div style="font-size:.72rem;color:#64748B;margin-top:6px">평단가 × 수량 합계</div>
</div>
""", unsafe_allow_html=True)
        _m2.markdown(f"""
<div class="ma-metric-card">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px">
    <span style="font-size:.75rem;color:#94A3B8;font-weight:500;letter-spacing:.3px">현재 평가금</span>
  </div>
  <div style="font-size:1.35rem;font-weight:700;color:#E2E8F0;line-height:1.2">₩{_total_val:,.0f}</div>
  <div style="font-size:.72rem;color:{_pnl_color};margin-top:6px;font-weight:600">{_pnl_pct_str} &nbsp;·&nbsp; USD/KRW≈{_usd_krw:,.0f}</div>
</div>
""", unsafe_allow_html=True)
        _m3.markdown(f"""
<div class="ma-metric-card">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px">
    <span style="font-size:.75rem;color:#94A3B8;font-weight:500;letter-spacing:.3px">누적 매도금</span>
  </div>
  <div style="font-size:1.35rem;font-weight:700;color:#E2E8F0;line-height:1.2">₩{_cum_sell_krw:,.0f}</div>
  <div style="font-size:.72rem;color:#64748B;margin-top:6px">매도 회수 총액 (원금+수익)</div>
</div>
""", unsafe_allow_html=True)
        _m4.markdown(f"""
<div class="ma-metric-card">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px">
    <span style="font-size:.75rem;color:#94A3B8;font-weight:500;letter-spacing:.3px">누적 실현 손익</span>
  </div>
  <div style="font-size:1.35rem;font-weight:700;color:{_profit_color};line-height:1.2">₩{_cum_profit_krw:+,.0f}</div>
  <div style="font-size:.72rem;color:#64748B;margin-top:6px">매도 완료 종목 손익 합계</div>
</div>
""", unsafe_allow_html=True)

        _total_in = _total_cost + _cum_buy_krw
        if _total_in > 0:
            _overall_pct = (_total_val + _cum_sell_krw) / _total_in * 100 - 100
            _ov_clr = "#4caf50" if _overall_pct >= 0 else "#ef4444"
            st.markdown(
                f'<div style="text-align:right;font-size:.82rem;color:#888;margin-top:-6px">'
                f'전체 기간 수익률 &nbsp;'
                f'<b style="color:{_ov_clr};font-size:.95rem">{_overall_pct:+.2f}%</b>'
                f'&nbsp;&nbsp;|&nbsp;&nbsp;'
                f'(현재 평가금 ₩{_total_val:,.0f} + 누적 매도금 ₩{_cum_sell_krw:,.0f})'
                f' / 전체 투자금 ₩{_total_in:,.0f}</div>',
                unsafe_allow_html=True,
            )

        st.divider()

        _ew_h, _ew_btn_col = st.columns([5, 1])
        _ew_h.markdown("#### 🎯 Event Watch")
        _pf_news_result: dict = st.session_state.get("pf_news_result", {})
        _pf_news_rt_ts: str   = st.session_state.get("pf_news_rt_ts", "")
        if _ew_btn_col.button("뉴스 분석", key="pf_run_news", type="primary", use_container_width=True):
            _rt_prices_news: dict[str, float] = {}
            _rt_pct_news:    dict[str, float] = {}
            with st.spinner("실시간 현재가 조회 중..."):
                for _it in _items:
                    _rt_d = realtime_price_fn(_it["ticker"])
                    if _rt_d["price"] > 0:
                        _rt_prices_news[_it["ticker"]] = _rt_d["price"]
                        _rt_pct_news[_it["ticker"]] = (
                            (_rt_d["price"] / _it["avg_price"] - 1) * 100 if _it["avg_price"] > 0 else 0.0
                        )
            _pf_prices.update(_rt_prices_news)
            _rt_now = now_kst_fn().strftime("%H:%M:%S")
            _pf_holdings = [
                {
                    "ticker":          _it["ticker"],
                    "company_name":    _it["ticker"].split(".")[0],
                    "price_change_pct": _rt_pct_news.get(_it["ticker"], 0.0),
                }
                for _it in _items
            ]
            with st.spinner("포트폴리오 뉴스 분석 중..."):
                _pf_news_result = analyze_portfolio_news(
                    _pf_holdings,
                    api_key=st.session_state.get("gemini_api_key", ""),
                    groq_api_key=st.session_state.get("groq_api_key", ""),
                    dart_api_key=st.session_state.get("dart_api_key", ""),
                )
            _pf_news_result["_rt_ts"] = _rt_now
            st.session_state["pf_news_result"] = _pf_news_result
            st.session_state["pf_news_rt_ts"]  = _rt_now

        if _pf_news_result:
            _news_rt_ts = _pf_news_result.get("_rt_ts", _pf_news_rt_ts)
            if _news_rt_ts:
                st.markdown(
                    f'<div style="font-size:.75rem;color:#22c55e;margin-bottom:6px">'
                    f'✅ 실시간 시세 반영 완료 &nbsp;|&nbsp; 기준 시각: <b>{_news_rt_ts}</b>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            _avg_s = _pf_news_result.get("portfolio_sentiment_avg", 0.0)
            _avg_l = _pf_news_result.get("portfolio_sentiment_label", "중립")
            _s_clr = "#4caf50" if _avg_s >= 0.5 else ("#ef4444" if _avg_s <= -0.5 else "#888")
            _ew_c1, _ew_c2 = st.columns([1, 3])
            _ew_c1.markdown(
                f'<div style="background:#1e2130;border-radius:10px;padding:18px;text-align:center">'
                f'<div style="font-size:.75rem;color:#999;margin-bottom:6px">포트폴리오 심리 지수</div>'
                f'<div style="font-size:2.2rem;font-weight:700;color:{_s_clr}">{_avg_s:+.2f}</div>'
                f'<div style="font-size:.85rem;color:{_s_clr};margin-top:4px">{_avg_l}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            _per_news = _pf_news_result.get("per_ticker", {})
            _bdg = '<div style="display:flex;flex-wrap:wrap;gap:6px;padding:8px 0">'
            for _bt, _br in _per_news.items():
                _bs   = _br.get("score", 0.0)
                _bl   = _br.get("label", "중립")
                _bc   = "#4caf50" if _bs >= 0.5 else ("#ef4444" if _bs <= -0.5 else "#888")
                _bnm  = _pf_nm.get(_bt, "")
                _blbl = (f"{_bnm}<br><span style='font-size:.72rem;color:#777'>{_bt}</span>" if _bnm else _bt)
                _cur_p  = _pf_prices.get(_bt, 0.0)
                _avg_p  = next((i["avg_price"] for i in _items if i["ticker"] == _bt), 0.0)
                _pnl_pc = (_cur_p / _avg_p - 1) * 100 if (_cur_p and _avg_p) else None
                _pnl_html = ""
                if _pnl_pc is not None:
                    _pnl_c  = "#4caf50" if _pnl_pc >= 0 else "#ef4444"
                    _impact = (
                        "호재 반영↑" if (_bs >= 0.5 and _pnl_pc >= 0)
                        else "악재 하락↓" if (_bs <= -0.5 and _pnl_pc < 0)
                        else "뉴스↑ 가격↓" if (_bs >= 0.5 and _pnl_pc < 0)
                        else "뉴스↓ 가격↑" if (_bs <= -0.5 and _pnl_pc >= 0)
                        else "중립"
                    )
                    _pnl_html = f"<span style='font-size:.7rem;color:{_pnl_c};margin-left:4px'>({_pnl_pc:+.1f}% {_impact})</span>"
                _bdg += (
                    f'<span style="background:#252836;border-radius:8px;padding:6px 12px;font-size:.85rem;line-height:1.5">'
                    f'<b style="color:#ddd">{_blbl}</b> '
                    f'<span style="color:{_bc}">{_bl} ({_bs:+.1f})</span>'
                    f'{_pnl_html}</span>'
                )
            _bdg += '</div>'
            _ew_c2.markdown(_bdg, unsafe_allow_html=True)

            _alerts = _pf_news_result.get("important_alerts", [])
            if _alerts:
                st.markdown("**🔔 중요 알림**")
                for _al in _alerts:
                    _akw   = _al["keyword"]
                    _ak_c  = "#ff6b6b" if _akw in ("증자", "상장폐지", "소송", "제재") else "#ffd93d"
                    _aname = _al.get("company_name") or _al["ticker"]
                    st.markdown(
                        f'<div style="background:#1a1d2e;border-left:4px solid {_ak_c};'
                        f'padding:10px 16px;border-radius:0 8px 8px 0;margin:3px 0">'
                        f'<span style="background:{_ak_c};color:#000;font-size:.72rem;'
                        f'border-radius:4px;padding:2px 6px;margin-right:8px;font-weight:700">[{_akw}]</span>'
                        f'<b style="color:#e0e0e0">{_aname}</b> — '
                        f'<span style="color:#ccc">{_al["title"]}</span></div>',
                        unsafe_allow_html=True,
                    )
        else:
            st.caption("'뉴스 분석' 버튼을 눌러 포트폴리오 종목의 뉴스 감성을 분석하세요.")

        st.divider()

        _pf_per_news: dict = st.session_state.get("pf_news_result", {}).get("per_ticker", {})
        _tbl_css = """
<style>
.pf-tbl{width:100%;border-collapse:collapse;font-size:.9rem}
.pf-tbl th{background:#1e2130;color:#9e9e9e;padding:10px 14px;text-align:right;
       font-weight:500;border-bottom:2px solid #333;white-space:nowrap}
.pf-tbl th:first-child,.pf-tbl th:last-child{text-align:left}
.pf-tbl td{padding:10px 14px;border-bottom:1px solid #252836;text-align:right;color:#ccc}
.pf-tbl td:first-child{text-align:left;color:#e0e0e0;font-weight:600}
.pf-tbl tr:last-child td{border-bottom:none}
.pf-tbl tr:hover td{background:#1a1d2e}
.pp{color:#10B981;font-weight:700} .pn{color:#3B82F6;font-weight:700} .pz{color:#888}
.ai-op{text-align:left!important;font-size:.8rem;max-width:200px;word-break:keep-all;line-height:1.4}
</style>"""
        _tbl_head = (
            '<table class="pf-tbl"><thead><tr>'
            '<th>종목</th><th>수량</th><th>평단가</th>'
            '<th>현재가</th><th>수익률(%)</th><th>평가손익</th><th>AI 의견</th>'
            '</tr></thead><tbody>'
        )
        _tbl_rows_html = []
        for _it in _items:
            _t   = _it["ticker"]
            _avg = _it["avg_price"]
            _qty = _it["quantity"]
            _cur = _pf_prices.get(_t)
            _is_krw_item = _t.upper().endswith((".KS", ".KQ"))
            _fp  = (lambda v, k=_is_krw_item: f"₩{v:,.0f}" if k else f"${v:,.2f}")
            _nm  = _pf_nm.get(_t, "")
            if _cur is not None:
                _pnl_val = (_cur - _avg) * _qty
                _pnl_pct = (_cur / _avg - 1) * 100 if _avg else 0.0
                _cls     = "pp" if _pnl_pct > 0 else ("pn" if _pnl_pct < 0 else "pz")
                _cur_str = _fp(_cur)
                _pct_str = f"{_pnl_pct:+.2f}%"
                _pnl_str = (f"+{_fp(abs(_pnl_val))}" if _pnl_val >= 0 else f"-{_fp(abs(_pnl_val))}")
            else:
                _cls = "pz"; _cur_str = "-"; _pct_str = "-"; _pnl_str = "-"
            _nr   = _pf_per_news.get(_t)
            _nsc  = _nr.get("score", 0.0) if _nr else None
            _nlb  = _nr.get("label", "중립") if _nr else None
            _pnlp = (_cur / _avg - 1) * 100 if (_cur and _avg) else None
            if _nsc is None:
                _ai_txt, _ai_c = "분석 대기", "#666"
            elif _nsc >= 2 and _pnlp is not None and _pnlp >= 0:
                _ai_txt, _ai_c = "호재 발생 + 수익 중 — 홀딩 권장", "#81c784"
            elif _nsc >= 1:
                _ai_txt, _ai_c = f"뉴스 {_nlb} — 홀딩 유지", "#a5d6a7"
            elif _nsc <= -2 and _pnlp is not None and _pnlp < -5:
                _ai_txt, _ai_c = "부정 신호 + 손실 — 손절 검토", "#ef9a9a"
            elif _nsc <= -1:
                _ai_txt, _ai_c = f"뉴스 {_nlb} — 비중 축소 검토", "#ffab91"
            elif _pnlp is not None and _pnlp < -10:
                _ai_txt, _ai_c = "큰 손실 중 — 손절라인 점검", "#ff8a65"
            else:
                _ai_txt, _ai_c = "중립 — 관망", "#aaa"
            _name_cell = (
                f"<div style='font-weight:600;color:#e0e0e0'>{_nm}</div>"
                f"<div style='font-size:.75rem;color:#666'>{_t}</div>"
            ) if _nm else _t
            _tbl_rows_html.append(
                f"<tr><td style='text-align:left'>{_name_cell}</td>"
                f"<td>{_qty:g}</td><td>{_fp(_avg)}</td><td>{_cur_str}</td>"
                f"<td class='{_cls}'>{_pct_str}</td><td class='{_cls}'>{_pnl_str}</td>"
                f"<td class='ai-op' style='color:{_ai_c}'>{_ai_txt}</td></tr>"
            )
        st.markdown(_tbl_css + _tbl_head + "".join(_tbl_rows_html) + "</tbody></table>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        _trailing: dict = st.session_state.setdefault("pf_trailing_max", {})
        for _it in _items:
            _c = _pf_prices.get(_it["ticker"])
            if _c:
                _trailing[_it["ticker"]] = max(_trailing.get(_it["ticker"], 0.0), _c)

        with st.expander("🎯 매도 가이드 (Exit Strategy)", expanded=False):
            _eg_h, _eg_btn = st.columns([5, 1])
            _eg_h.markdown(
                "<small style='color:#999'>실시간가 기준 손절/익절 가이드 · 트레일링 스탑 · 목표가 근접 알림</small>",
                unsafe_allow_html=True,
            )
            _exit_result: dict = st.session_state.get("pf_exit_result", {})
            _exit_rt_ts: str   = st.session_state.get("pf_exit_rt_ts", "")
            if _eg_btn.button("매도 가이드 분석", key="pf_exit_calc", type="primary", use_container_width=True):
                _exit_result = {}
                _exit_now    = now_kst_fn().strftime("%H:%M:%S")
                with st.spinner("실시간 현재가 조회 및 차트 분석 중..."):
                    for _it in _items:
                        _t = _it["ticker"]
                        try:
                            _rt_exit = realtime_price_fn(_t)
                            if _rt_exit["price"] > 0:
                                _pf_prices[_t] = _rt_exit["price"]
                            if _rt_exit.get("stale") and _rt_exit.get("stale_msg"):
                                st.caption(f"⏸️ {_t}: {_rt_exit['stale_msg']}")
                            _cdata = get_stock_data_fn(_t, period="3mo")
                            _sell_targets = get_sell_target_price(_cdata) if (_cdata is not None and not _cdata.empty) else {}
                            _rt_p   = _pf_prices.get(_t, _it["avg_price"])
                            _avg_p  = _it["avg_price"]
                            _pnl_pc = (_rt_p / _avg_p - 1) * 100 if _avg_p else 0.0
                            _stop8  = _avg_p * 0.92
                            _stop5  = _avg_p * 0.95
                            _tp1    = _avg_p * 1.10
                            _tp2    = _avg_p * 1.20
                            if _pnl_pc >= 20:
                                _guide, _guide_clr = "익절 구간 진입 — 분할 매도 권장 (1/2 이상 청산 고려)", "#ffd93d"
                            elif _pnl_pc >= 10:
                                _guide, _guide_clr = "수익 중 — 1차 목표가 도달. 일부 수익 확정 고려", "#81c784"
                            elif _pnl_pc > 0:
                                _guide, _guide_clr = "소폭 수익 — 홀딩 유지 권장", "#a5d6a7"
                            elif _pnl_pc > -5:
                                _guide, _guide_clr = "소폭 손실 — 추세 모니터링", "#fff176"
                            elif _pnl_pc > -8:
                                _guide, _guide_clr = "손절 접근 — 5% 스탑라인 이탈 시 즉시 손절 고려", "#ffab91"
                            else:
                                _guide, _guide_clr = "손절 구간 — 추가 손실 방지를 위한 즉시 손절 권장", "#ef9a9a"
                            _exit_result[_t] = {
                                **_sell_targets,
                                "rt_price": _rt_p, "avg_price": _avg_p, "pnl_pct": _pnl_pc,
                                "stop_loss_8": _stop8, "stop_loss_5": _stop5,
                                "take_profit_1": _tp1, "take_profit_2": _tp2,
                                "guide": _guide, "guide_clr": _guide_clr,
                                "is_rt": _rt_exit.get("is_realtime", False), "rt_ts": _exit_now,
                            }
                        except Exception:
                            pass
                st.session_state["pf_exit_result"] = _exit_result
                st.session_state["pf_exit_rt_ts"]  = _exit_now

            if not _exit_result:
                st.caption("'매도 가이드 분석' 버튼으로 실시간가 기준 손절/익절 가이드를 확인하세요.")
            else:
                _shown_rt = next((v.get("rt_ts") for v in _exit_result.values() if v.get("rt_ts")), _exit_rt_ts)
                if _shown_rt:
                    st.markdown(
                        f'<div style="font-size:.75rem;color:#22c55e;margin-bottom:8px">'
                        f'✅ 실시간 시세 반영 완료 &nbsp;|&nbsp; 기준 시각: <b>{_shown_rt}</b>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            for _it in _items:
                _t   = _it["ticker"]
                _avg = _it["avg_price"]
                _cur = _pf_prices.get(_t)
                _is_krw_ex = _t.upper().endswith((".KS", ".KQ"))
                _efmt = (lambda v, k=_is_krw_ex: f"₩{v:,.0f}" if k else f"${v:,.2f}")
                _enm  = _pf_nm.get(_t, "")
                _stp  = _exit_result.get(_t, {})
                _cons = _stp.get("conservative_target")
                _aggr = _stp.get("aggressive_target")
                _t_max     = _trailing.get(_t, 0.0)
                _guide_txt = _stp.get("guide", "")
                _guide_clr = _stp.get("guide_clr", "#888")
                _stop8_p   = _stp.get("stop_loss_8",    _avg * 0.92 if _avg else 0)
                _stop5_p   = _stp.get("stop_loss_5",    _avg * 0.95 if _avg else 0)
                _tp1_p     = _stp.get("take_profit_1",  _avg * 1.10 if _avg else 0)
                _tp2_p     = _stp.get("take_profit_2",  _avg * 1.20 if _avg else 0)
                _cur_label = "실시간가" if _stp.get("is_rt") else "현재가"

                _exit_alerts: list[tuple[str, str]] = []
                if _cur and _t_max > 0 and _cur < _t_max * 0.95:
                    _drop = (_cur / _t_max - 1) * 100
                    _exit_alerts.append((f"⚠️ 트레일링 스탑: 최고가 {_efmt(_t_max)} 대비 {_drop:.1f}% — 수익 보존을 위한 매도 권장", "#ff8a65"))
                if _cur and _cons and _cur >= _cons * 0.95:
                    _exit_alerts.append((f"🎯 목표가 근접 ({_cur/_cons*100:.0f}%): 보수적 목표가 {_efmt(_cons)} — 분할 매도 고려", "#ffd93d"))
                if _cur and _avg and _cur <= _avg * 0.92:
                    _exit_alerts.append((f"🔴 손절 구간 돌입: 실시간가 {_efmt(_cur)} — 8% 손절라인 이탈, 즉시 매도 권장", "#ef4444"))
                elif _cur and _avg and _cur <= _avg * 0.95:
                    _exit_alerts.append((f"🟠 손절 주의: 실시간가 {_efmt(_cur)} — 5% 스탑라인 접근", "#ff8a65"))

                _cp: list[str] = []
                if _cur:
                    _pr  = (_cur / _avg - 1) * 100 if _avg else 0
                    _prc = "#4caf50" if _pr >= 0 else "#ef4444"
                    _cp.append(f'{_cur_label} <b style="color:#ddd">{_efmt(_cur)}</b> <span style="color:{_prc}">({_pr:+.1f}%)</span>')
                if _avg:    _cp.append(f'평단가 <b style="color:#aaa">{_efmt(_avg)}</b>')
                if _stop5_p: _cp.append(f'손절(5%) <b style="color:#ff8a65">{_efmt(_stop5_p)}</b>')
                if _stop8_p: _cp.append(f'손절(8%) <b style="color:#ef4444">{_efmt(_stop8_p)}</b>')
                if _tp1_p:   _cp.append(f'1차 익절 <b style="color:#81c784">{_efmt(_tp1_p)}</b>')
                if _tp2_p:   _cp.append(f'2차 익절 <b style="color:#4fc3f7">{_efmt(_tp2_p)}</b>')
                if _cons:
                    _cg = (_cons / _avg - 1) * 100 if _avg else 0
                    _cp.append(f'보수적 목표가 <b style="color:#81c784">{_efmt(_cons)}</b> <span style="color:#81c784">({_cg:+.1f}%)</span>')
                if _aggr:
                    _ag = (_aggr / _avg - 1) * 100 if _avg else 0
                    _cp.append(f'공격적 목표가 <b style="color:#4fc3f7">{_efmt(_aggr)}</b> <span style="color:#4fc3f7">({_ag:+.1f}%)</span>')
                if _t_max and _cur and _t_max > _cur:
                    _cp.append(f'추적 최고가 <b style="color:#ffb74d">{_efmt(_t_max)}</b> | 스탑라인 <b style="color:#ff8a65">{_efmt(_t_max*0.95)}</b>')

                _card_body  = " &nbsp;|&nbsp; ".join(_cp) if _cp else "-"
                _guide_html = (
                    f'<div style="margin-top:6px;padding:6px 10px;background:#252836;'
                    f'border-left:3px solid {_guide_clr};border-radius:0 6px 6px 0;'
                    f'font-size:.85rem;color:{_guide_clr};font-weight:600">📋 {_guide_txt}</div>'
                ) if _guide_txt else ""
                _alert_html = "".join(
                    f'<div style="margin-top:6px;padding:6px 10px;background:#252836;'
                    f'border-left:3px solid {_ac};border-radius:0 6px 6px 0;font-size:.82rem;color:{_ac}">{_amsg}</div>'
                    for _amsg, _ac in _exit_alerts
                )
                st.markdown(
                    f'<div style="background:#1e2130;border-radius:10px;padding:12px 16px;margin:6px 0">'
                    f'<div style="margin-bottom:6px">'
                    + (f'<span style="font-size:.95rem;font-weight:700;color:#e0e0e0">{_enm}</span> ' if _enm else "")
                    + f'<span style="font-size:.78rem;color:#666">{_t}</span></div>'
                    f'<div style="font-size:.85rem;line-height:1.9">{_card_body}</div>'
                    f'{_guide_html}{_alert_html}</div>',
                    unsafe_allow_html=True,
                )

                if _stp and _stp.get("rt_price"):
                    _rt_sell_px = _stp["rt_price"]
                    _exit_qty_col, _sell_btn_col, _ = st.columns([2, 2, 3])
                    _exit_sell_qty = _exit_qty_col.number_input(
                        "매도수량", min_value=0.01, max_value=float(_it["quantity"]),
                        value=float(_it["quantity"]), step=1.0, format="%.2f",
                        key=f"pf_exit_qty_{_it['id']}", help=f"최대 {_it['quantity']:g}주",
                    )
                    if _sell_btn_col.button(f"✅ 매도 확정  {_efmt(_rt_sell_px)}", key=f"pf_sell_exit_{_it['id']}"):
                        _sell_r = db_sell_item(_uid, _it["id"], _rt_sell_px, _exit_sell_qty)
                        if _sell_r["ok"]:
                            _pnl_d = (f"₩{_sell_r['net_profit']:+,.0f}" if _is_krw_ex else f"${_sell_r['net_profit']:+,.2f}")
                            st.success(f"매도 완료! 실현 손익: {_pnl_d} ({_sell_r['return_rate']:+.2f}%)")
                            st.rerun(scope="fragment")
                        else:
                            st.error(_sell_r.get("error", "매도 실패"))

        st.markdown("<br>", unsafe_allow_html=True)

        with st.expander("💸 매도 기록 / 종목 삭제", expanded=False):
            st.caption(
                "**매도** 버튼: 입력한 매도가·수량으로 trade_history에 기록 후 포트폴리오에서 차감  "
                "│  **삭제** 버튼: 기록 없이 포트폴리오에서만 제거"
            )
            for _it in _items:
                _is_krw_del = _it["ticker"].upper().endswith((".KS", ".KQ"))
                _price_str  = f"₩{_it['avg_price']:,.0f}" if _is_krw_del else f"${_it['avg_price']:,.2f}"
                _del_nm     = _pf_nm.get(_it["ticker"], "")
                _del_label  = f"**{_del_nm}** `{_it['ticker']}`" if _del_nm else f"`{_it['ticker']}`"
                _dc1, _dc2, _dc3, _dc4, _dc5 = st.columns([3, 2, 1.5, 1, 1])
                _dc1.markdown(f"{_del_label}  \n{_it['quantity']:g}주 @ {_price_str}")
                _default_px = float(_pf_prices.get(_it["ticker"], _it["avg_price"]))
                _sell_px_input = _dc2.number_input(
                    "매도가", min_value=0.01, value=_default_px,
                    step=100.0 if _is_krw_del else 0.01,
                    format="%.0f" if _is_krw_del else "%.2f",
                    key=f"pf_sell_px_{_it['id']}", label_visibility="collapsed",
                )
                _sell_qty_input = _dc3.number_input(
                    "매도수량", min_value=0.01, max_value=float(_it["quantity"]),
                    value=float(_it["quantity"]), step=1.0, format="%.2f",
                    key=f"pf_sell_qty_{_it['id']}", label_visibility="collapsed",
                    help=f"최대 {_it['quantity']:g}주",
                )
                if _dc4.button("매도", key=f"pf_sell_{_it['id']}", type="primary", use_container_width=True):
                    _sell_r = db_sell_item(_uid, _it["id"], _sell_px_input, _sell_qty_input)
                    if _sell_r["ok"]:
                        _pnl_msg = (f"₩{_sell_r['net_profit']:+,.0f}" if _is_krw_del else f"${_sell_r['net_profit']:+,.2f}")
                        st.toast(f"매도 완료! 손익: {_pnl_msg} ({_sell_r['return_rate']:+.2f}%)")
                        st.rerun(scope="fragment")
                    else:
                        st.error(_sell_r.get("error", "매도 실패"))
                if _dc5.button("삭제", key=f"pf_del_{_it['id']}", use_container_width=True):
                    _r3 = db_delete_portfolio(_it["id"], _uid)
                    if _r3["ok"]:
                        st.rerun(scope="fragment")
                    else:
                        st.error(_r3.get("error", "삭제 실패"))

        st.divider()

        _ai_h, _ai_btn = st.columns([5, 1])
        _ai_h.markdown("#### 🤖 AI의 이번 주 제안")
        _opt_result: dict = st.session_state.get("pf_opt_result", {})
        if _ai_btn.button("섹터 분석", key="pf_opt_run", type="primary", use_container_width=True):
            from src.portfolio_optimizer import classify_sectors, scan_sector_etfs, build_rebalancing_guide
            with st.spinner("섹터 분석 및 시장 주도주 스캔 중..."):
                _sd  = classify_sectors(_items, _pf_prices)
                _es  = scan_sector_etfs()
                _opt_result = {
                    "sector_data": _sd, "etf_scan": _es,
                    "guide": build_rebalancing_guide(_sd, _es, _pf_nm),
                }
            st.session_state["pf_opt_result"] = _opt_result

        if not _opt_result:
            st.caption("'섹터 분석' 버튼으로 포트폴리오 섹터 편중도와 리밸런싱 제안을 확인하세요.")
        else:
            _sd_r  = _opt_result["sector_data"]
            _guide = _opt_result["guide"]
            _sctrs = _sd_r.get("sectors", {})
            if _sctrs:
                st.markdown("**📊 섹터 비중**")
                _bar_sorted = sorted(_sctrs.items(), key=lambda x: x[1]["weight"], reverse=True)
                _bar_max    = max(v["weight"] for _, v in _bar_sorted) or 1
                _bar_html   = '<div style="display:grid;gap:5px;margin-bottom:12px">'
                for _sn, _sv in _bar_sorted:
                    _sw   = _sv["weight"]
                    _bc   = "#ef4444" if _sw > 30 else ("#ffd93d" if _sw > 20 else "#4fc3f7")
                    _bpct = _sw / _bar_max * 100
                    _tks  = ", ".join(_pf_nm.get(t, t) or t for t in _sv["tickers"])
                    _bar_html += (
                        f'<div style="display:flex;align-items:center;gap:8px">'
                        f'<div style="width:80px;font-size:.8rem;color:#ccc;text-align:right">{_sn}</div>'
                        f'<div style="flex:1;background:#252836;border-radius:4px;height:18px;overflow:hidden">'
                        f'<div style="width:{_bpct:.0f}%;height:100%;background:{_bc};border-radius:4px"></div></div>'
                        f'<div style="width:44px;font-size:.8rem;font-weight:700;color:{_bc}">{_sw:.1f}%</div>'
                        f'<div style="font-size:.75rem;color:#666">{_tks}</div>'
                        f'</div>'
                    )
                _bar_html += "</div>"
                st.markdown(_bar_html, unsafe_allow_html=True)

            st.markdown("""<style>
.opt-card{background:#1e2130;border-radius:12px;padding:16px;margin-bottom:8px;min-height:100px}
.opt-card-title{font-size:.8rem;font-weight:700;color:#9e9e9e;margin-bottom:10px;letter-spacing:.5px}
.opt-item{font-size:.85rem;line-height:1.7;padding:6px 10px;background:#252836;border-radius:8px;margin:4px 0;word-break:keep-all}
.opt-empty{color:#555;font-size:.82rem;font-style:italic}
</style>""", unsafe_allow_html=True)

            _gc1, _gc2 = st.columns(2)

            def _opt_card(col, title, items_html):
                col.markdown(f'<div class="opt-card"><div class="opt-card-title">{title}</div>{items_html}</div>', unsafe_allow_html=True)

            _warn = _guide.get("concentration_warnings", [])
            _warn_body = "".join(
                f'<div class="opt-item" style="border-left:3px solid {"#ef4444" if w["weight"]>40 else "#ff8a65"}">'
                f'<b style="color:{"#ef4444" if w["weight"]>40 else "#ff8a65"}">{w["sector"]} {w["weight"]:.1f}%</b> 집중 — '
                f'<span style="color:#aaa">{", ".join(_pf_nm.get(t,t) or t for t in w["tickers"])}</span>'
                f'<br><span style="font-size:.75rem;color:#888">30% 초과: 비중 분산 권고</span></div>'
                for w in _warn
            ) or '<span class="opt-empty">집중 위험 없음 — 양호한 분산</span>'
            _opt_card(_gc1, "⚠️ 섹터 집중 위험", _warn_body)

            _cands = _guide.get("new_candidates", [])
            _cand_body = "".join(
                f'<div class="opt-item" style="border-left:3px solid #4fc3f7">'
                f'<b style="color:#4fc3f7">{cd["name"]}</b> <span style="color:#81c784">+{cd["return_5d"]:.1f}%</span>'
                f'<br><span style="font-size:.75rem;color:#888">{cd["reason"]}</span></div>'
                for cd in _cands
            ) or '<span class="opt-empty">현재 추천 섹터 없음</span>'
            _opt_card(_gc2, "🎯 신규 편입 후보", _cand_body)

            _pt = _guide.get("profit_take", [])
            _pt_body = "".join(
                f'<div class="opt-item" style="border-left:3px solid #ffd93d">'
                f'<b style="color:#e0e0e0">{p["name"]}</b> <span style="color:#4caf50;font-weight:700">+{p["pnl_pct"]:.1f}%</span>'
                f'<br><span style="font-size:.75rem;color:#888">{p["reason"]}</span></div>'
                for p in _pt
            ) or '<span class="opt-empty">수익 확정 기준(+15%) 도달 종목 없음</span>'
            _opt_card(_gc1, "💰 수익 확정 권고", _pt_body)

            _ab = _guide.get("add_buy", [])
            _ab_body = "".join(
                f'<div class="opt-item" style="border-left:3px solid #81c784">'
                f'<b style="color:#e0e0e0">{a["name"]}</b>'
                f'<br><span style="font-size:.75rem;color:#888">{a["reason"]}</span></div>'
                for a in _ab
            ) or '<span class="opt-empty">추가 매수 후보 없음</span>'
            _opt_card(_gc2, "📈 추가 매수 권고", _ab_body)

        st.divider()

        st.markdown("#### 📈 AI 전략 자산 배분 — 모멘텀 주도주 포트폴리오")
        with st.expander("💡 분석 가이드 — 차트 정밀 진단과 무엇이 다른가요?", expanded=False):
            st.markdown(
                "**이 추천은 '추세추종형(모멘텀)' 관점으로 종목을 선정합니다.**\n\n"
                "시장 주도력과 뉴스 심리가 **현재 양호한** 종목을 골라 투자금을 최적 배분합니다.\n\n"
                "| 구분 | AI 모멘텀 추천 | 차트 정밀 진단 |\n"
                "|------|----------------|----------------|\n"
                "| 전략 | 추세추종 — 이미 오르는 종목 편승 | 역추세 — 과매도 반등 포착 |\n"
                "| RSI 기준 | **RSI < 30 제외** | **RSI < 30 = 강력 매수 신호** |\n"
                "| 후보 풀 | **코스피+코스닥+나스닥 각 상위 500개** | 사용자 선택 종목 |\n"
            )

        _rec_c1, _rec_c2, _rec_c3 = st.columns([3, 1, 1])
        _inv_amt = _rec_c1.number_input(
            "투자 예정 금액 (원)", min_value=100_000, max_value=1_000_000_000,
            value=5_000_000, step=500_000, format="%d", key="rec_investment_amount",
        )
        _risk_profile = _rec_c2.selectbox("투자 성향", ["중립형", "보수형", "공격형"], key="rec_risk_profile")
        _rec_c3.write("")
        _rec_run = _rec_c3.button("모멘텀 추천 실행", type="primary", key="rec_run", use_container_width=True)

        if _rec_run:
            from src.recommendation_engine import run_recommendation, recommendation_to_dict
            with st.spinner("KOSPI·KOSDAQ·NASDAQ 각 상위 500개 후보 로드 → L1 필터 → L2 뉴스 감성 분석 중... (1~2분 소요)"):
                _rec_result = run_recommendation(
                    investment_amount=int(_inv_amt),
                    risk_profile=_risk_profile,
                    api_key=st.session_state.get("gemini_api_key", ""),
                    groq_api_key=st.session_state.get("groq_api_key", ""),
                )
            st.session_state["pf_rec_result"] = _rec_result
            if not _rec_result.get("error") and _rec_result.get("recommendations"):
                try:
                    db_save_recommendation(
                        _uid, int(_inv_amt), _risk_profile,
                        [recommendation_to_dict(r) for r in _rec_result["recommendations"]],
                    )
                except Exception:
                    pass

        _rec_result: dict = st.session_state.get("pf_rec_result", {})
        if _rec_result.get("error"):
            st.warning(_rec_result["error"], icon="⚠️")
        elif _rec_result.get("recommendations"):
            _recs      = _rec_result["recommendations"]
            _total_inv = _rec_result.get("total_invested", 0.0)
            _remaining = _rec_result.get("remaining_cash", 0.0)
            _pool_sz   = _rec_result.get("pool_size", 0)
            _l1_cnt    = _rec_result.get("l1_pass", 0)
            _l2_cnt    = _rec_result.get("l2_pass", 0)
            _rm1, _rm2, _rm3, _rm4 = st.columns(4)
            _rm1.metric("투자 예정 금액", f"₩{int(_inv_amt):,}")
            _rm2.metric("실제 투자액",   f"₩{_total_inv:,.0f}")
            _rm3.metric("매수 후 잔금",  f"₩{_remaining:,.0f}", delta=f"-₩{_total_inv:,.0f}", delta_color="inverse")
            _rm4.metric("분석 깔때기",   f"{len(_recs)}개 선정",
                        help=f"후보 풀 {_pool_sz}개 → L1 {_l1_cnt}개 → L2 {_l2_cnt}개 → 최종 {len(_recs)}개")
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""<style>
.rec-card{background:#1e2130;border-radius:12px;padding:16px 18px;margin-bottom:8px;border-top:3px solid #4fc3f7}
.rec-card-name{font-size:1rem;font-weight:700;color:#e0e0e0}
.rec-card-sector{font-size:.72rem;background:#252836;color:#9e9e9e;padding:2px 8px;border-radius:10px;margin-left:6px}
.rec-card-reason{font-size:.8rem;color:#7ecfff;font-style:italic;margin-top:8px;line-height:1.5;
             border-left:2px solid #4fc3f7;padding-left:8px}
.rec-bar-bg{background:#252836;border-radius:4px;height:8px;margin:6px 0}
.rec-bar-fg{height:8px;border-radius:4px;background:linear-gradient(90deg,#4fc3f7,#81c784)}
</style>""", unsafe_allow_html=True)

            def _render_rec_card(r, col):
                _sent_pct  = round(r.sentiment_score * 100, 1)
                _wt_pct    = round(r.weight * 100, 1)
                _rsi_clr   = "#4caf50" if r.rsi < 50 else ("#ffd93d" if r.rsi < 70 else "#ef4444")
                _cur_str   = getattr(r, "currency", "KRW")
                _mkt       = getattr(r, "market", "KOSPI")
                _native_px = getattr(r, "current_price", 0)
                _price_str = f"₩{_native_px:,.0f}" if _cur_str == "KRW" else f"${_native_px:,.2f}"
                _mkt_clr   = "#4fc3f7" if _mkt == "KOSPI" else ("#81c784" if _mkt == "KOSDAQ" else "#ffb74d")
                col.markdown(
                    f'<div class="rec-card">'
                    f'<div style="margin-bottom:6px">'
                    f'<span class="rec-card-name">{r.name}</span>'
                    f'<span class="rec-card-sector">{r.sector}</span>'
                    f'<span style="font-size:.68rem;background:{_mkt_clr}22;color:{_mkt_clr};padding:1px 7px;border-radius:8px;margin-left:4px;font-weight:600">{_mkt}</span>'
                    f'</div>'
                    f'<div style="font-size:.85rem;color:#aaa;margin-top:4px">현재가 <b style="color:#ddd">{_price_str}</b>'
                    + (f' <span style="font-size:.75rem;color:#777">(₩{getattr(r,"current_price_krw",_native_px):,.0f})</span>' if _cur_str == "USD" else "")
                    + f'</div>'
                    f'<div style="margin-top:10px;display:flex;gap:10px;flex-wrap:wrap">'
                    f'<span style="font-size:.82rem;color:#9e9e9e">비중</span> <span style="font-size:.9rem;font-weight:700;color:#4fc3f7">{_wt_pct:.1f}%</span>'
                    f'&nbsp;·&nbsp;<span style="font-size:.82rem;color:#9e9e9e">수량</span> <span style="font-size:.9rem;font-weight:700;color:#ffd93d">{r.quantity:,}주</span>'
                    f'&nbsp;·&nbsp;<span style="font-size:.82rem;color:#9e9e9e">투자액</span> <span style="font-size:.9rem;font-weight:700;color:#81c784">₩{r.invested:,.0f}</span>'
                    f'</div>'
                    f'<div style="margin-top:8px;display:flex;gap:16px">'
                    f'<span style="font-size:.78rem;color:#999">뉴스감성 <b style="color:#4fc3f7">{_sent_pct:.0f}</b>/100</span>'
                    f'<span style="font-size:.78rem;color:#999">RSI <b style="color:{_rsi_clr}">{r.rsi:.0f}</b></span>'
                    f'</div>'
                    f'<div class="rec-bar-bg"><div class="rec-bar-fg" style="width:{_wt_pct/40*100:.0f}%"></div></div>'
                    f'<div class="rec-card-reason">{r.reason}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            _row1, _row2 = _recs[:3], _recs[3:]
            if _row1:
                _cols1 = st.columns(min(len(_row1), 3))
                for _r, _c in zip(_row1, _cols1):
                    _render_rec_card(_r, _c)
            if _row2:
                _pad   = (3 - len(_row2)) // 2
                _cols2 = st.columns([1] * _pad + [1] * len(_row2) + [1] * (3 - len(_row2) - _pad))
                for _r, _c in zip(_row2, _cols2[_pad: _pad + len(_row2)]):
                    _render_rec_card(_r, _c)

            try:
                from src.recommendation_engine import SENTIMENT_THRESHOLD, RSI_OVERSOLD_BOUND, RSI_OVERBOUGHT_BOUND
                _rec_set = {r.ticker for r in _recs}
                _missed  = {it["ticker"] for it in _items} - _rec_set
                if _missed:
                    with st.expander(f"🔎 내 포트폴리오 {len(_missed)}개 종목이 이번 추천에 없는 이유", expanded=False):
                        for _mt in _missed:
                            _mn = _pf_nm.get(_mt, "") or _mt.split(".")[0]
                            st.markdown(
                                f'<div style="background:#1e2130;border-radius:8px;padding:10px 14px;margin:5px 0;border-left:3px solid #555">'
                                f'<span style="font-size:.9rem;font-weight:600;color:#e0e0e0">🟡 {_mn}</span>'
                                f'<span style="font-size:.75rem;color:#666;margin-left:8px">{_mt}</span>'
                                f'<div style="font-size:.82rem;color:#9e9e9e;margin-top:5px;line-height:1.5">'
                                f'선정 기준: RSI {RSI_OVERSOLD_BOUND}–{RSI_OVERBOUGHT_BOUND} + 20일 MA 돌파 + 뉴스 점수 {SENTIMENT_THRESHOLD*100:.0f}점 이상</div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
            except Exception:
                pass

        elif not _rec_result:
            st.caption("'모멘텀 추천 실행' 버튼을 누르면 뉴스 감성·RSI 기반으로 투자금에 맞는 모멘텀 주도주 5개와 매수 수량을 제안합니다.")

        with st.expander("🕐 지난 추천 이력", expanded=False):
            try:
                _hist = db_get_rec_history(_uid, limit=5)
            except Exception:
                _hist = []
            if not _hist:
                st.caption("저장된 추천 이력이 없습니다.")
            else:
                for _h in _hist:
                    _h_dt    = _h.get("created_at", "")[:16].replace("T", " ")
                    _h_names = ", ".join(r.get("name", r.get("ticker", "")) for r in _h.get("recommendations", []))
                    st.markdown(
                        f'<div style="background:#1e2130;border-radius:8px;padding:10px 14px;margin:4px 0;font-size:.85rem">'
                        f'<span style="color:#9e9e9e">{_h_dt}</span>&nbsp;|&nbsp;'
                        f'<b style="color:#ddd">₩{_h.get("investment_amt",0):,}</b>&nbsp;·&nbsp;'
                        f'<span style="color:#4fc3f7">{_h.get("risk_profile","중립형")}</span>'
                        f'<div style="color:#aaa;margin-top:4px">{_h_names}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        with st.expander("📋 매도 이력", expanded=False):
            if not _trade_history:
                st.caption("매도 이력이 없습니다. '매도 기록 / 종목 삭제' 또는 '매도 가이드'의 매도 확정 버튼으로 기록이 남습니다.")
            else:
                _th_rows_data = []
                for _th in _trade_history:
                    _th_t      = _th["ticker"]
                    _th_nm     = _pf_nm.get(_th_t, "") or _th_t
                    _th_is_krw = _th_t.upper().endswith((".KS", ".KQ"))
                    _th_fp     = (lambda v, k=_th_is_krw: f"₩{v:,.0f}" if k else f"${v:,.2f}")
                    _th_dt     = _th.get("traded_at", "")[:16].replace("T", " ")
                    _th_rows_data.append({
                        "종목":      f"{_th_nm} ({_th_t})" if _th_nm != _th_t else _th_t,
                        "매수가":    _th_fp(_th["buy_price"]),
                        "매도가":    _th_fp(_th["sell_price"]),
                        "수량":      f"{_th['quantity']:g}주",
                        "실현 손익": _th_fp(_th["net_profit"]),
                        "수익률":    f"{_th['return_rate']:+.2f}%",
                        "매도일시":  _th_dt,
                    })
                _th_total_cnt  = len(_trade_history)
                _th_profit_cnt = sum(1 for t in _trade_history if t["net_profit"] >= 0)
                _th_hdr_col, _th_clr_col = st.columns([7, 1])
                _th_hdr_col.markdown(
                    f'<div style="display:flex;gap:24px;padding-top:6px;font-size:.85rem">'
                    f'<span style="color:#9e9e9e">총 {_th_total_cnt}건</span>'
                    f'<span style="color:#4caf50">수익 {_th_profit_cnt}건</span>'
                    f'<span style="color:#ef4444">손실 {_th_total_cnt - _th_profit_cnt}건</span>'
                    f'<span style="color:#aaa">누적 실현 손익 '
                    f'<b style="color:{"#4caf50" if _cum_profit_krw >= 0 else "#ef4444"}">'
                    f'₩{_cum_profit_krw:+,.0f}</b></span></div>',
                    unsafe_allow_html=True,
                )
                if _th_clr_col.button("🗑️ 초기화", key="th_clear_btn", help="매도 이력 전체 삭제 (복구 불가)"):
                    st.session_state["_th_confirm_clear"] = True
                if st.session_state.get("_th_confirm_clear"):
                    st.warning("매도 이력을 전부 삭제합니다. 누적 매도금·실현 손익이 0으로 초기화됩니다.")
                    _cf_ok, _cf_no = st.columns(2)
                    if _cf_ok.button("삭제 확인", key="th_clear_confirm", type="primary", use_container_width=True):
                        db_clear_trade_history(_uid)
                        st.session_state.pop("_th_confirm_clear", None)
                        st.toast("매도 이력이 초기화됐습니다.")
                        st.rerun(scope="fragment")
                    if _cf_no.button("취소", key="th_clear_cancel", use_container_width=True):
                        st.session_state.pop("_th_confirm_clear", None)
                        st.rerun(scope="fragment")
                st.dataframe(_th_rows_data, use_container_width=True, hide_index=True)


# ─── 포트폴리오 탭 진입점 ─────────────────────────────────────────────────────
def render_portfolio_tab(
    tab,
    *,
    db_login,
    db_logout,
    db_register,
    db_get_user,
    db_get_portfolio,
    db_get_trade_history,
    db_sell_item,
    db_delete_portfolio,
    db_save_recommendation,
    db_get_rec_history,
    db_clear_trade_history,
    ticker_name_map_fn,
    realtime_price_fn,
    get_stock_data_fn,
    now_kst_fn,
    cookie_mgr=None,
    has_cookie_mgr: bool = False,
) -> None:
    """포트폴리오 탭: 로그인 폼은 여기서, 본문은 @st.fragment인 _render_pf_body에 위임."""
    with tab:
        _tok = st.session_state.get("auth_token")

        if not _tok:
            st.markdown("**💼 내 포트폴리오** — 로그인이 필요합니다")
            st.divider()
            _auth_mode = st.radio(
                "모드 선택", ["로그인", "회원가입"],
                horizontal=True, label_visibility="collapsed",
            )
            with st.form("pf_auth_form"):
                _pf_email  = st.text_input("이메일", placeholder="you@example.com")
                _pf_pw     = st.text_input("비밀번호", type="password", placeholder="6자 이상")
                _pf_submit = st.form_submit_button(_auth_mode, use_container_width=True, type="primary")

            if _pf_submit:
                if _auth_mode == "회원가입":
                    if len(_pf_pw) < 6:
                        st.error("비밀번호는 6자 이상이어야 합니다.")
                    else:
                        _r = db_register(_pf_email, _pf_pw)
                        if _r["ok"]:
                            st.success("회원가입 완료! 로그인해 주세요.")
                        else:
                            st.error(_r["error"])
                else:
                    _r = db_login(_pf_email, _pf_pw)
                    if _r["ok"]:
                        st.session_state["auth_token"]   = _r["token"]
                        st.session_state["auth_user_id"] = _r["user_id"]
                        st.session_state["auth_email"]   = _r["email"]
                        if has_cookie_mgr and cookie_mgr:
                            from datetime import datetime as _dt
                            cookie_mgr.set("auth_token", _r["token"], expires_at=_dt(2099, 1, 1))
                        st.rerun()
                    else:
                        st.error(_r["error"])
            return

        # 로그인 상태: fragment에 위임 — 매도·삭제·초기화 시 탭만 재렌더링
        _render_pf_body(
            db_logout=db_logout,
            db_get_user=db_get_user,
            db_get_portfolio=db_get_portfolio,
            db_get_trade_history=db_get_trade_history,
            db_sell_item=db_sell_item,
            db_delete_portfolio=db_delete_portfolio,
            db_save_recommendation=db_save_recommendation,
            db_get_rec_history=db_get_rec_history,
            db_clear_trade_history=db_clear_trade_history,
            ticker_name_map_fn=ticker_name_map_fn,
            realtime_price_fn=realtime_price_fn,
            get_stock_data_fn=get_stock_data_fn,
            now_kst_fn=now_kst_fn,
            cookie_mgr=cookie_mgr,
            has_cookie_mgr=has_cookie_mgr,
        )
