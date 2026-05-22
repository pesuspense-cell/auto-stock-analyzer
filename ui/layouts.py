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
    get_buy_target_price, get_sell_target_price, calc_atr_trailing_guide,
    get_advanced_analysis, calculate_vpvr,
    check_volume_anomaly, check_dead_time,
    check_breakout_signal, adjust_risk_conservative,
    analyze_investor_trend,
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

    # 분석 시작 버튼 클릭 후 사이드바 자동 닫기
    if st.session_state.pop("_collapse_sidebar", False):
        import streamlit.components.v1 as _cmp
        _cmp.html(
            """<script>
            setTimeout(function() {
                var btn = window.parent.document.querySelector(
                    '[data-testid="stSidebarCollapseButton"]'
                );
                if (btn) { btn.click(); return; }
                // 모바일: 오버레이 닫기 버튼
                var mBtn = window.parent.document.querySelector(
                    '[data-testid="stSidebar"] button'
                );
                if (mBtn) mBtn.click();
            }, 200);
            </script>""",
            height=0,
            scrolling=False,
        )

    def _clear_analysis():
        st.session_state.pop("analyzed_ticker", None)
        st.session_state.pop("analyzed_sname",  None)
        st.session_state.pop("analyzed_period", None)

    with st.sidebar:
        st.markdown("## ⚙️ 종목 설정")

        if all_stocks:
            selected = st.selectbox(
                f"종목 선택 ({len(all_stocks):,}개)",
                list(all_stocks.keys()),
                key="_unified_selected",
                on_change=_clear_analysis,
            )
            ticker = all_stocks[selected]
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
            st.session_state["_collapse_sidebar"] = True
            st.rerun()

        # API 키 값 로드 (UI 표시 전 필요)
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
                        _wl_cur: list = st.session_state.get("watchlist", [])
                        if i < len(_wl_cur):
                            _wl_cur.pop(i)
                        st.session_state["watchlist"] = _wl_cur
                        save_watchlist_fn(_wl_cur)
                        st.rerun()

        # ── 포트폴리오 종목 추가 (로그인 시만) ──────────────────────────────
        _sb_uid = st.session_state.get("auth_user_id")
        _sb_tok = st.session_state.get("auth_token")
        if _sb_tok and _sb_uid:
            st.divider()
            st.markdown("""
<div style="font-size:.85rem;font-weight:600;color:#0066cc;margin-bottom:4px">
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
                    _sb_sel = st.selectbox(
                        f"선택 ({len(_sb_all):,}개)",
                        list(_sb_all.keys()),
                        key="sb_add_unified",
                    )
                    _sb_ticker_val = _sb_all[_sb_sel]

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

        # ── 환율 ────────────────────────────────────────────────────────────
        st.divider()
        st.markdown('<p style="font-size:.8rem;font-weight:600;color:#7a7a7a;margin:0 0 4px 0;letter-spacing:0;">💱 실시간 환율</p>', unsafe_allow_html=True)

        def _sb_row(label, value, change, chg_color):
            return (
                f'<div style="display:flex;justify-content:space-between;align-items:center;'
                f'padding:5px 0;border-bottom:1px solid #e0e0e0;">'
                f'<span style="font-size:.78rem;color:#7a7a7a;">{label}</span>'
                f'<div style="text-align:right">'
                f'<div style="font-size:.88rem;font-weight:600;color:#1d1d1f;font-variant-numeric:tabular-nums;">{value}</div>'
                f'<div style="font-size:.72rem;color:{chg_color};font-variant-numeric:tabular-nums;">{change}</div>'
                f'</div></div>'
            )

        rate_html = ""
        for pair, info in rates.items():
            chg_val = info["change"]
            chg_color = "#34c759" if chg_val >= 0 else "#ff3b30"
            chg_sign = "+" if chg_val >= 0 else ""
            rate_html += _sb_row(pair, f"{info['rate']:,.2f}", f"{chg_sign}{chg_val:.3f}%", chg_color)
        st.markdown(rate_html, unsafe_allow_html=True)

        # ── 주요 지수 ────────────────────────────────────────────────────────
        st.divider()
        st.markdown('<p style="font-size:.8rem;font-weight:600;color:#7a7a7a;margin:0 0 4px 0;letter-spacing:0;">📊 주요 지수</p>', unsafe_allow_html=True)

        idx_html = ""
        for idx_name, idx_sym in indices.items():
            try:
                d = get_index_data(idx_sym)
                if len(d) >= 2:
                    p   = float(d["Close"].iloc[-1])
                    chg = (p - float(d["Close"].iloc[-2])) / float(d["Close"].iloc[-2]) * 100
                    chg_color = "#34c759" if chg >= 0 else "#ff3b30"
                    chg_sign = "+" if chg >= 0 else ""
                    idx_html += _sb_row(idx_name, f"{p:,.2f}", f"{chg_sign}{chg:.2f}%", chg_color)
            except Exception:
                pass
        if idx_html:
            st.markdown(idx_html, unsafe_allow_html=True)

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
                '<div style="font-size:.78rem;color:#e07000;font-weight:600;'
                'letter-spacing:0;margin-bottom:6px">⚠️ 미연동 항목</div>',
                unsafe_allow_html=True,
            )
            for _ic, _it, _ih in _inactive:
                st.markdown(
                    f'<div style="background:rgba(255,149,0,0.06);border:1px solid rgba(255,149,0,0.2);'
                    f'border-radius:8px;padding:6px 10px;margin:3px 0;font-size:.75rem">'
                    f'<span style="color:#e07000">{_ic} {_it}</span>'
                    f'<div style="color:#7a7a7a;margin-top:1px">{_ih}</div></div>',
                    unsafe_allow_html=True,
                )

        # ── API 키 ──────────────────────────────────────────────────────────
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
        st.divider()
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
  <span style="font-size:1.6rem;font-weight:600;color:#1d1d1f;font-family:'SF Pro Display',system-ui,-apple-system,sans-serif;letter-spacing:-0.374px;">AI 주식 분석 대시보드</span>
  <span style="font-size:.72rem;color:#0066cc;padding:3px 12px;
               background:rgba(0,102,204,0.08);border:1px solid rgba(0,102,204,0.2);
               border-radius:9999px;font-weight:600;letter-spacing:0">Pro</span>
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
                f'<span style="color:{COLORS["accent"]}">₩{h_val:,.0f}</span>',
                f'<span style="color:{COLORS["accent"]}">포트폴리오 현재가치</span>',
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
    st.markdown("<div style='margin-bottom:6px'></div>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# 시장 현황 탭
# ═════════════════════════════════════════════════════════════════════════════

def _render_toss_mover_list(df: pd.DataFrame, show_market: bool = True) -> None:
    """토스 스타일 종목 리스트 HTML 렌더링."""
    if df is None or df.empty:
        st.caption("데이터 없음")
        return
    html = '<div class="toss-list">'
    for _, row in df.iterrows():
        chg = row.get("등락률(%)", None)
        try:
            chg = float(chg)
            color   = "#ff3b30" if chg >= 0 else "#0066cc"
            arrow   = "▲" if chg >= 0 else "▼"
            chg_str = f"{arrow} {abs(chg):.2f}%"
        except (TypeError, ValueError):
            color, chg_str = "#b0b0b0", "—"
        price = row.get("현재가", 0)
        try:
            price_str = f"₩{float(price):,.0f}"
        except (TypeError, ValueError):
            price_str = "—"
        name   = row.get("종목명", "")
        ticker = row.get("티커", "")
        market = row.get("시장", "")
        sub    = f"{ticker} · {market}" if (show_market and market) else ticker
        html += (
            f'<div class="toss-item">'
            f'<div style="overflow:hidden;flex:1;margin-right:10px">'
            f'<div class="toss-item-name">{name}</div>'
            f'<div class="toss-item-sub">{sub}</div>'
            f'</div>'
            f'<div style="text-align:right;flex-shrink:0">'
            f'<div class="toss-item-price">{price_str}</div>'
            f'<div style="font-size:.75rem;font-weight:600;color:{color};margin-top:2px">{chg_str}</div>'
            f'</div></div>'
        )
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


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

        # ── 환율 수평 스크롤 카드 ─────────────────────────────────────
        _rates_html = '<div class="toss-rate-row">'
        for pair, info in rates.items():
            _r  = info["rate"]
            _c  = info["change"]
            _arrow = "▲" if _c > 0 else "▼"
            _cc    = "#ff3b30" if _c > 0 else "#0066cc"
            _rfmt  = f"{_r:,.0f}" if _r >= 100 else f"{_r:,.2f}"
            _rates_html += (
                f'<div class="toss-rate-card">'
                f'<div style="font-size:.65rem;color:#7a7a7a;margin-bottom:4px;white-space:nowrap">{pair}</div>'
                f'<div style="font-size:1rem;font-weight:600;color:#1d1d1f;font-variant-numeric:tabular-nums;">{_rfmt}</div>'
                f'<div style="font-size:.7rem;color:{_cc};font-weight:600;margin-top:2px;font-variant-numeric:tabular-nums;">'
                f'{_arrow} {abs(_c):.2f}%</div>'
                f'</div>'
            )
        _rates_html += '</div>'
        st.markdown(_rates_html, unsafe_allow_html=True)

        with st.expander("📈 USD/KRW 환율 추이 (3개월)"):
            try:
                fx = usdkrw_fn()
                if not fx.empty:
                    fig_fx = go.Figure(go.Scatter(
                        x=fx.index, y=fx["Close"],
                        fill="tozeroy", fillcolor="rgba(0,102,204,0.08)",
                        line=dict(color="#0066cc", width=2),
                    ))
                    fig_fx.update_layout(
                        height=200, template="plotly_white",
                        margin=dict(t=10, b=10), showlegend=False,
                        xaxis_rangeslider_visible=False,
                    )
                    st.plotly_chart(fig_fx, use_container_width=True)
            except Exception:
                pass

        # ── 섹터 ETF 등락표 ──────────────────────────────────────────
        st.divider()
        st.markdown("### 🗺️ 주요 섹터 ETF")
        st.caption("미국 ETF 15개 + 국내 섹터별 대표 ETF 20개 · 전일 대비 등락률 · 10분 캐시")
        sector_etf_prices_fn()

        # ── 시장 급등·급락 TOP10 + 시가총액 상위 (좌우 2칼럼) ────────
        st.divider()
        _col_top10, _col_cap = st.columns(2)

        with _col_top10:
            st.markdown("### 🏆 시장 급등·급락 TOP 10")
            _fm_gainers, _fm_losers = full_movers_fn()
            if not _fm_gainers.empty or not _fm_losers.empty:
                _t_gain, _t_loss = st.tabs(["🚀 급등 TOP 10", "📉 급락 TOP 10"])
                with _t_gain:
                    _render_toss_mover_list(_fm_gainers, show_market=True)
                with _t_loss:
                    _render_toss_mover_list(_fm_losers, show_market=True)
            else:
                st.warning("시장 데이터를 불러올 수 없습니다. (KRX 서버 응답 없음)", icon="⚠️")

        with _col_cap:
            st.markdown("### 📊 시가총액 상위 분석")
            mover_n = st.select_slider(
                "분석 종목 수 (시가총액 상위)", list(range(10, 110, 10)), value=50,
                help="KOSPI 시가총액 상위 N개 종목의 등락률을 분석합니다.",
            )
            movers = movers_fn(mover_n)

            if not movers.empty:
                top_n   = min(10, len(movers) // 2)
                gainers = movers.head(top_n)
                losers  = movers.tail(top_n).sort_values("등락률(%)")

                _m1, _m2 = st.tabs(["🚀 급등 상위", "📉 급락 하위"])
                with _m1:
                    _render_toss_mover_list(gainers, show_market=False)
                with _m2:
                    _render_toss_mover_list(losers, show_market=False)

                chart_n   = min(15, len(movers) // 2)
                chart_top = pd.concat([movers.head(chart_n), movers.tail(chart_n).sort_values("등락률(%)")])
                with st.expander(f"📊 급등·급락 차트 (TOP {chart_n})"):
                    fig_bar = go.Figure(go.Bar(
                        x=chart_top["종목명"],
                        y=chart_top["등락률(%)"],
                        marker_color=["#ff3b30" if val >= 0 else "#0066cc" for val in chart_top["등락률(%)"]],
                        text=[f"{val:+.2f}%" for val in chart_top["등락률(%)"]],
                        textposition="outside",
                    ))
                    fig_bar.update_layout(
                        height=360, template="plotly_white",
                        margin=dict(t=10, b=70), xaxis_tickangle=-40,
                        yaxis_title="등락률 (%)",
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

                with st.expander(f"📋 전체 {len(movers)}개 종목 등락률 표"):
                    st.dataframe(
                        movers[["종목명", "티커", "현재가", "등락률(%)", "거래량"]].style
                        .format({"현재가": "{:,.0f}", "등락률(%)": "{:+.2f}%", "거래량": "{:,}"})
                        .map(lambda v: "color:#ff3b30" if isinstance(v, float) and v > 0
                             else ("color:#0066cc" if isinstance(v, float) and v < 0 else ""),
                             subset=["등락률(%)"]),
                        use_container_width=True, hide_index=True,
                    )



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

            # stage 번호 → (아이콘, 제목, 메시지 생성 함수) 매핑
            _STAGE_META = {
                1: ("📡", "OHLCV 데이터 수집 중",
                    lambda m: f"{m.get('fetched', 0)}/{m.get('total', 0)}개 종목 수집 완료"),
                2: ("📊", "L1 차트 스크리닝 중",
                    lambda m: f"L1 필터 통과: {m.get('l1_count', 0)}개 종목"),
                3: ("🎯", "L2·L3 심층 분석 중",
                    lambda m: f"심층 분석 대상: {m.get('l2_count', 0)}개 종목"),
                4: ("✅", "분석 완료",
                    lambda m: f"추천 종목 확정: {m.get('final_count', 0)}개"),
            }

            def _run_rec():
                try:
                    # 수정: news_fn 파라미터 제거, _progress_q 이름 정정
                    return get_recommendations_fn(
                        stocks,
                        _progress_q=_rec_q,
                    )
                except Exception as _thread_exc:
                    _rec_q.put_nowait({
                        "stage": 0, "icon": "🚨",
                        "title": "분석 오류",
                        "msg": f"분석 중 오류: {_thread_exc}",
                    })
                    raise

            _render_rec_bar(_stage_state, 0)
            recs = []
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _pool:
                    _f_rec = _pool.submit(_run_rec)
                    while not _f_rec.done():
                        _elapsed = int(time.time() - _rec_start)
                        while not _rec_q.empty():
                            _msg = _rec_q.get_nowait()
                            if isinstance(_msg, dict):
                                _s = _msg.get("stage", _stage_state.get("stage", 0))
                                if _s in _STAGE_META:
                                    _ico, _ttl, _msg_fn = _STAGE_META[_s]
                                    _stage_state.update({
                                        "stage": _s, "icon": _ico,
                                        "title": _ttl, "msg": _msg_fn(_msg),
                                    })
                                else:
                                    _stage_state.update(_msg)
                        _render_rec_bar(_stage_state, _elapsed)
                        time.sleep(1)
                    _raw = _f_rec.result()
                    # DataFrame → list[dict] 변환 (display 코드가 dict.get()으로 접근)
                    if hasattr(_raw, "iterrows") and not _raw.empty:
                        recs = [
                            {
                                "ticker":     str(row.get("티커", "") or ""),
                                "score":      float(row.get("종합점수", 0) or 0),
                                "label":      str(row.get("종합추천", "") or ""),
                                "change_pct": float(row.get("등락률(1일)%", 0.0) or 0.0),
                            }
                            for _, row in _raw.iterrows()
                        ]
                    elif isinstance(_raw, list):
                        recs = _raw
            except Exception as _exc:
                import traceback as _tb
                st.error(
                    f"🚨 **추천 분석 실패**: {_exc}\n\n"
                    f"```\n{_tb.format_exc()}\n```"
                )
                recs = []

            _render_rec_bar(
                {"icon": "✅", "title": "분석 완료", "msg": f"총 {total_stocks}개 종목 전수 분석 완료"},
                int(time.time() - _rec_start), done=True,
            )
            _rec_ph.empty()
            st.session_state["_rec_results"] = recs
            if auth_user_id and recs:
                try:
                    # save_recommendation(user_id, investment_amt, risk_profile, recommendations)
                    db_save_recommendation(auth_user_id, 0, "전수분석", recs)
                except Exception as _db_exc:
                    st.warning(f"추천 결과 저장 실패 (분석 결과는 유효): {_db_exc}")

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
                    f'<div style="background:#ffffff;border:1px solid #e0e0e0;'
                    f'border-radius:12px;padding:14px 16px;margin:6px 0;'
                    f'display:flex;justify-content:space-between;align-items:center;'
                    f'box-shadow:0 1px 3px rgba(0,0,0,0.06);">'
                    f'<div><b style="color:#1d1d1f">{_nm}</b>'
                    f'<span style="color:#7a7a7a;font-size:.8rem;margin-left:8px">({_tk})</span>'
                    f'<div style="font-size:.78rem;color:#7a7a7a;margin-top:3px">{_lbl}</div></div>'
                    f'<div style="text-align:right">'
                    f'<div style="font-size:1.1rem;font-weight:600;color:{_clr};font-variant-numeric:tabular-nums;">{_pct:+.2f}%</div>'
                    f'<div style="font-size:.72rem;color:#7a7a7a;font-variant-numeric:tabular-nums;">점수 {_sc:+.1f}</div>'
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
                            f'<div style="background:#f5f5f7;border:1px solid #e0e0e0;border-radius:8px;'
                            f'padding:10px 14px;margin:4px 0;font-size:.85rem">'
                            f'<span style="color:#7a7a7a">{_h_dt}</span>'
                            f'&nbsp;|&nbsp;'
                            f'<b style="color:#1d1d1f">₩{_h_amt:,}</b>'
                            f'&nbsp;·&nbsp;'
                            f'<span style="color:#0066cc">{_h_prof}</span>'
                            f'<div style="color:#7a7a7a;margin-top:4px">{_h_names}</div>'
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
        gemini_key  = api_keys.get("gemini", "")
        groq_key    = api_keys.get("groq", "")
        _cname      = sname if sname != ticker else ""

        _news_is_etf = data_ready and check_is_etf_fn(ticker)
        is_kr_stock  = ticker.endswith(".KS") or ticker.endswith(".KQ")

        if _news_is_etf:
            st.subheader(f"📊 {asname or sname} 섹터 뉴스 — 돈이 몰리는 섹터 파악")
            st.caption("ETF는 종목 필터를 느슨하게 적용하고 상위 구성종목 뉴스를 함께 수집합니다.")
        else:
            st.subheader(f"📰 {asname or sname} 뉴스 & 관련 정보")

        raw_news: list       = []
        _etf_fund_data: dict = {}
        sent: dict           = {}
        _sec_perf: dict      = {}
        _etf_meta: tuple     = ("", [])

        with st.status("📡 뉴스 & 데이터 수집 중...", expanded=True) as _status:
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as _pool:

                if _news_is_etf:
                    # ETF 펀더멘탈과 섹터 성과를 병렬로 시작
                    _f_fund   = _pool.submit(etf_fundamental_fn, ticker)
                    _f_sector = _pool.submit(sector_perf_fn, ticker) if data_ready else None

                    st.write("🔄 ETF 펀더멘탈 & 섹터 데이터 수집 중...")
                    _etf_fund_data = _f_fund.result()
                    raw_news = get_etf_news_with_holdings(ticker, _etf_fund_data, max_items=15)
                    st.write(f"✅ ETF 뉴스 {len(raw_news)}건 수집 완료")

                    # 뉴스 수집 완료 즉시 감성 분석 제출
                    _f_sent = _pool.submit(
                        analyze_etf_news_sentiment, ticker, _etf_fund_data, raw_news
                    ) if raw_news else None
                    st.write("🔄 AI 감성 분석 중...")

                    _sec_perf = _f_sector.result() if _f_sector else {}
                    if _sec_perf:
                        st.write("✅ 섹터 데이터 수집 완료")

                    sent = _f_sent.result() if _f_sent else {}
                    if sent:
                        st.write("✅ 감성 분석 완료")

                    _etf_meta = (
                        _etf_fund_data.get("sector", ""),
                        [h.get("name", h.get("ticker", "")) for h in _etf_fund_data.get("top_holdings", [])[:5]],
                    )

                elif is_kr_stock:
                    # 네이버 뉴스와 섹터 성과를 병렬로 시작
                    _f_news   = _pool.submit(naver_news_fn, ticker)
                    _f_sector = _pool.submit(sector_perf_fn, ticker) if data_ready else None

                    st.write("🔄 네이버 뉴스 & 섹터 데이터 수집 중...")
                    raw_news = _f_news.result()
                    st.write(f"✅ 뉴스 {len(raw_news)}건 수집 완료")

                    # 뉴스 도착 즉시 감성 분석 제출 (섹터 대기 없이)
                    if raw_news:
                        def _do_sent(_rn=raw_news):
                            if gemini_key or groq_key:
                                return analyze_news_sentiment_llm(_rn, ticker, gemini_key, groq_key, _cname)
                            return analyze_news_sentiment_keywords(_rn, ticker, _cname)
                        _f_sent = _pool.submit(_do_sent)
                        st.write("🔄 AI 감성 분석 중...")
                    else:
                        _f_sent = None

                    _sec_perf = _f_sector.result() if _f_sector else {}
                    if _sec_perf:
                        st.write("✅ 섹터 데이터 수집 완료")

                    sent = _f_sent.result() if _f_sent else {}
                    if sent:
                        st.write("✅ 감성 분석 완료")

                else:
                    # 미국 주식: yfinance(빠름) + 섹터 성과 병렬
                    _f_sector = _pool.submit(sector_perf_fn, ticker) if data_ready else None

                    st.write("🔄 뉴스 & 섹터 데이터 수집 중...")
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

                    st.write(f"✅ 뉴스 {len(raw_news)}건 수집 완료")

                    if raw_news:
                        def _do_sent_us(_rn=raw_news):
                            if gemini_key or groq_key:
                                return analyze_news_sentiment_llm(_rn, ticker, gemini_key, groq_key, _cname)
                            return analyze_news_sentiment_keywords(_rn, ticker, _cname)
                        _f_sent = _pool.submit(_do_sent_us)
                        st.write("🔄 AI 감성 분석 중...")
                    else:
                        _f_sent = None

                    _sec_perf = _f_sector.result() if _f_sector else {}
                    if _sec_perf:
                        st.write("✅ 섹터 데이터 수집 완료")

                    sent = _f_sent.result() if _f_sent else {}
                    if sent:
                        st.write("✅ 감성 분석 완료")

            _status.update(label="✅ 분석 완료", state="complete", expanded=False)

        # ── ETF 메타데이터 배지 ───────────────────────────────────────────────
        if _news_is_etf:
            _etf_sector, _etf_holdings_display = _etf_meta
            if _etf_sector or _etf_holdings_display:
                st.markdown(
                    f'<div style="background:#f5f5f7;border:1px solid #e0e0e0;border-radius:8px;padding:10px 16px;margin-bottom:10px;">'
                    f'<span style="color:#0066cc;font-weight:600;">섹터: {_etf_sector or "N/A"}</span>'
                    + (f'&nbsp;&nbsp;|&nbsp;&nbsp;<span style="color:#7a7a7a;font-size:0.85rem;">'
                       f'구성종목 뉴스 포함: {", ".join(_etf_holdings_display)}</span>'
                       if _etf_holdings_display else "") +
                    f'</div>',
                    unsafe_allow_html=True,
                )

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
                            f'<div style="background:#f5f5f7;border:1px solid #e0e0e0;border-radius:6px;padding:6px 10px;'
                            f'margin:3px 0;font-size:.8rem;color:#7a7a7a;display:flex;gap:6px;">'
                            f'<span>{_d_icon}</span>'
                            f'<span style="flex:1">{_d_ttl}</span>'
                            f'<span style="color:{"#34c759" if _d_sc >= 0 else "#ff3b30"};'
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
                if _sec_perf:
                    for _sec, _pct in list(_sec_perf.items())[:5]:
                        try:
                            _pct = float(_pct)
                        except (TypeError, ValueError):
                            continue
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


# ═════════════════════════════════════════════════════════════════════════════
# 차트 분석 탭 (메인 분석 탭)
# ═════════════════════════════════════════════════════════════════════════════

def render_chart_tab(
    tab,
    *,
    state: dict,
    api_keys: dict,
    inv_data_fn,
    inv_history_fn=None,
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
        _wl: list = st.session_state.get("watchlist", [])
        is_in_wl = any(w["ticker"] == ticker for w in _wl)
        wl_col1, wl_col2 = st.columns([6, 1])
        with wl_col2:
            if is_in_wl:
                if st.button("★ 관심 해제", use_container_width=True):
                    st.session_state["watchlist"] = [
                        w for w in _wl if w["ticker"] != ticker
                    ]
                    save_watchlist_fn(st.session_state["watchlist"])
                    st.rerun()
            else:
                if st.button("☆ 관심 추가", use_container_width=True, type="primary"):
                    _wl.append({"name": sname, "ticker": ticker})
                    st.session_state["watchlist"] = _wl
                    save_watchlist_fn(_wl)
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

        # ── 차트 생성 (dialog용) ───────────────────────────────────────────
        try:
            _kospi_raw = yf.download("^KS11", period=aperiod, auto_adjust=True, progress=False)
            _kospi_raw = _flatten_columns(_kospi_raw)
            _kospi_df  = _kospi_raw[["Open", "High", "Low", "Close"]].dropna()
        except Exception:
            _kospi_df = pd.DataFrame()

        data = _flatten_columns(data)
        fig  = _build_plotly_chart(data, _kospi_df, ticker, aperiod)

        # ── 종목 배지 + 차트 보기 버튼 + 진단기준 (한 줄) ───────────────
        title_label = f"{sname} ({ticker})" if sname != ticker else ticker
        _badge_is_krw = ticker.upper().endswith((".KS", ".KQ"))
        _badge_price_str = (
            ("₩" if _badge_is_krw else "$") +
            ("{:,.0f}" if _badge_is_krw else "{:,.2f}").format(rt_price)
        ) if (rt_price > 0 and data_ready) else "—"

        # 등락률 계산
        _has_close2 = not close.empty and len(close) >= 2
        if _has_close2:
            _c_last = float(close.iloc[-1])
            _c_prev = float(close.iloc[-2])
            _c_cur  = rt_price if (rt_price > 0 and data_ready) else _c_last
            _badge_chg = (_c_cur - _c_prev) / _c_prev * 100
        else:
            _badge_chg = None

        _badge_col, _btn_chart_col, _btn_diag_col = st.columns([6, 1, 1])
        with _badge_col:
            st.markdown(
                stock_badge_html(title_label, _badge_price_str,
                                 rt_realtime and data_ready and rt_price > 0,
                                 chg_pct=_badge_chg,
                                 rt_ts=rt_ts if (data_ready and rt_price > 0) else ""),
                unsafe_allow_html=True,
            )
        with _btn_chart_col:
            if st.button("📊 차트", key="show_chart_btn",
                         help="캔들·EMA·볼린저밴드·RSI·MACD·ADX·KOSPI 6패널",
                         use_container_width=True):
                chart_dialog(fig, ticker)
        with _btn_diag_col:
            with st.expander("🔍 진단기준"):
                st.caption(
                    "역추세 기술적 진단 — RSI·MACD·볼린저밴드·ADX·일목균형표 등 10개+ 지표 종합.  \n"
                    "Naver·RSS·YouTube 멀티소스 뉴스 LLM 감성 포함. "
                    "포트폴리오 탭 AI 모멘텀 추천과 결과가 다를 수 있습니다."
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

        # ── AI 매매 신호 패널 (풀 너비) ──────────────────────────────────
        _render_signal_panel(
            state=state,
            inv_data_fn=inv_data_fn,
            inv_history_fn=inv_history_fn,
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
        name="가격", increasing_line_color="#ff3b30", decreasing_line_color="#0066cc",
    ), row=1, col=1)

    # 이동평균선
    for col_name, color, lbl, width, dash in [
        ("SMA_5",   "#ff9500", "SMA5",   1.4, "solid"),
        ("EMA_20",  "#0066cc", "EMA20",  1.4, "solid"),
        ("EMA_50",  "#34c759", "EMA50",  1.4, "solid"),
        ("EMA_200", "#ff3b30", "EMA200", 2.0, "dash"),
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
            line=dict(color="rgba(255,59,48,0.7)", width=1),
            fill="tonexty", fillcolor="rgba(120,120,120,0.12)",
        ), row=1, col=1)
    for col_name, color, lbl, dash in [
        ("ICHI_TENKAN", "#ff3b30", "전환선", "dot"),
        ("ICHI_KIJUN",  "#0066cc", "기준선", "dot"),
    ]:
        if col_name in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index, y=data[col_name], name=lbl,
                line=dict(color=color, width=1.1, dash=dash),
            ), row=1, col=1)

    # VWAP 멀티 타임프레임
    for _vc, _color, _lbl, _w, _dash in [
        ("VWAP_W", "#ff9500", "VWAP 주간(5일)",  1.4, "solid"),
        ("VWAP_M", "#0066cc", "VWAP 월간(20일)", 1.6, "solid"),
        ("VWAP_Q", "#34c759", "VWAP 분기(60일)", 2.0, "dash"),
    ]:
        if _vc in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index, y=data[_vc],
                name=_lbl, line=dict(color=_color, width=_w, dash=_dash),
            ), row=1, col=1)

    # 거래량 (Row 2)
    vol_colors = ["#ff3b30" if float(c.iloc[i]) >= float(o.iloc[i]) else "#0066cc"
                  for i in range(len(data))]
    fig.add_trace(go.Bar(
        x=data.index, y=v, name="거래량",
        marker_color=vol_colors, showlegend=False, opacity=0.7,
    ), row=2, col=1)
    if "Volume_MA20" in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index, y=data["Volume_MA20"], name="Vol MA20",
            line=dict(color="#ff9500", width=1.2),
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
            line=dict(color="#0066cc", width=1.5, dash="dot"),
        ), row=2, col=1)

    # RSI + MFI (Row 3)
    if "RSI" in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index, y=data["RSI"], name="RSI",
            line=dict(color="#0066cc", width=2),
        ), row=3, col=1)
        for level, clr in [(70, "rgba(239,83,80,0.45)"), (30, "rgba(66,165,245,0.45)")]:
            fig.add_hline(y=level, line_color=clr, line_dash="dash", row=3, col=1)
        fig.add_hrect(y0=70, y1=100, fillcolor="rgba(239,83,80,0.05)", row=3, col=1)
        fig.add_hrect(y0=0,  y1=30,  fillcolor="rgba(66,165,245,0.05)", row=3, col=1)
    if "MFI" in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index, y=data["MFI"], name="MFI",
            line=dict(color="#0066cc", width=1.5, dash="dot"),
        ), row=3, col=1)

    # MACD (Row 4)
    if "MACD" in data.columns:
        hist_vals   = data["MACD_Hist"]
        hist_colors = ["#ff3b30" if val >= 0 else "#0066cc" for val in hist_vals]
        fig.add_trace(go.Bar(
            x=data.index, y=hist_vals, name="MACD Hist",
            marker_color=hist_colors, showlegend=False,
        ), row=4, col=1)
        fig.add_trace(go.Scatter(
            x=data.index, y=data["MACD"], name="MACD",
            line=dict(color="#0066cc", width=1.5),
        ), row=4, col=1)
        fig.add_trace(go.Scatter(
            x=data.index, y=data["MACD_Signal"], name="Signal",
            line=dict(color="#ff9500", width=1.5),
        ), row=4, col=1)

    # ADX + ±DI (Row 5)
    if "ADX" in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index, y=data["ADX"], name="ADX",
            line=dict(color="#1d1d1f", width=2),
        ), row=5, col=1)
        fig.add_hline(y=25, line_color="rgba(0,0,0,0.2)", line_dash="dash", row=5, col=1)
        fig.add_hline(y=35, line_color="rgba(255,149,0,0.3)",  line_dash="dot",  row=5, col=1)
    for col_name, color, lbl in [
        ("ADX_POS", "#34c759", "+DI"),
        ("ADX_NEG", "#ff3b30", "-DI"),
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
            line=dict(color="#0066cc", width=1.8),
            fill="tozeroy", fillcolor="rgba(0,102,204,0.08)",
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
    fig.update_annotations(font_size=10, font_color="#7a7a7a")
    fig.update_layout(
        height=1150, template="plotly_white",
        dragmode=False, xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
                    font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=30, b=10, l=10, r=10),
        uirevision=ticker, hovermode="x unified",
    )
    return fig


def _render_signal_panel(*, state: dict, inv_data_fn, inv_history_fn=None, gemini_key: str, groq_key: str) -> None:
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

    # ── 추천 매수가 4모드 (왼쪽) + 추세·탄력·에너지 (오른쪽) ──────────
    st.markdown(
        f'<div style="font-size:0.72rem;font-weight:600;color:{COLORS["text_2"]};'
        f'letter-spacing:0.8px;text-transform:uppercase;margin-bottom:8px;">'
        f'추천 매수가 (4가지 모드)</div>',
        unsafe_allow_html=True,
    )
    _buy_col, _score_col = st.columns([1, 1])

    with _buy_col:
        if _has_price:
            for _mode, _mode_short in [
                ("classic",  "🎯 종합"),
                ("sale",     "🏷️ 세일"),
                ("breakout", "⚡ 추격"),
                ("vwap",     "📊 VWAP"),
            ]:
                _bt = get_buy_target_price(data, mode=_mode)
                if not _bt:
                    continue
                _bt_val   = _bt.get("buy_target")
                _bt_color = _bt.get("mode_color", "#1d1d1f")
                _timing   = _bt.get("timing", "")
                _t_color  = _bt.get("timing_color", "#7a7a7a")
                _bt_str   = _fmt.format(_bt_val) if _bt_val else "—"
                st.markdown(
                    f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};'
                    f'border-left:3px solid {_bt_color};border-radius:10px;padding:8px 12px;margin-bottom:6px;">'
                    f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                    f'<span style="font-size:0.72rem;color:{_bt_color};font-weight:600;">{_mode_short}</span>'
                    f'<span style="font-size:0.95rem;font-weight:600;color:#1d1d1f;font-variant-numeric:tabular-nums;">{_bt_str}</span>'
                    f'</div>'
                    f'<div style="font-size:0.72rem;color:{_t_color};margin-top:3px;">{_timing}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("데이터 없음")

    with _score_col:
        _ts = advanced.get("trend_score",    50.0)
        _ms = advanced.get("momentum_score", 50.0)
        _vs = advanced.get("volume_score",   50.0)
        for _label, _score, _icon in [
            ("추세",  _ts, "📈"),
            ("탄력",  _ms, "⚡"),
            ("에너지", _vs, "🔋"),
        ]:
            _sc = "#34c759" if _score >= 65 else ("#ff3b30" if _score <= 35 else "#ff9500")
            _bg = "rgba(52,199,89,0.10)" if _score >= 65 else ("rgba(255,59,48,0.10)" if _score <= 35 else "rgba(255,149,0,0.10)")
            st.markdown(
                f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};'
                f'border-radius:10px;padding:9px 12px;margin-bottom:6px;">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">'
                f'<span style="color:{COLORS["text_2"]};font-size:0.78rem;">{_icon} {_label}</span>'
                f'<span style="background:{_bg};color:{_sc};border-radius:20px;'
                f'padding:1px 8px;font-size:0.73rem;font-weight:700;">{_score:.0f}</span></div>'
                f'<div style="background:{COLORS["border_md"]};border-radius:4px;height:5px;">'
                f'<div style="background:{_sc};width:{int(_score)}%;height:5px;border-radius:4px;"></div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

    # 수급 모멘텀 (KRX 종목만)
    if data_ready and (ticker.endswith(".KS") or ticker.endswith(".KQ")):
        try:
            _inv_history = inv_history_fn(ticker) if inv_history_fn else []
            _trend = analyze_investor_trend(_inv_history)
            _t_score = _trend["score"]

            if _t_score >= 3:    _inv_brd = "#34c759"
            elif _t_score >= 1:  _inv_brd = "#0066cc"
            elif _t_score <= -2: _inv_brd = "#ff3b30"
            else:                _inv_brd = "#b0b0b0"

            # 가장 중요한 한 줄 — 경고 우선, 없으면 첫 번째 근거
            _key_line = (
                _trend["warnings"][0] if _trend["warnings"] else
                _trend["reasons"][0]  if _trend["reasons"]  else ""
            )
            if len(_key_line) > 85:
                _key_line = _key_line[:82] + "…"

            st.markdown(
                f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};'
                f'border-left:3px solid {_inv_brd};border-radius:10px;'
                f'padding:10px 14px;margin-bottom:8px;">'
                f'<div style="font-size:0.65rem;color:{COLORS["text_2"]};letter-spacing:0.8px;'
                f'text-transform:uppercase;margin-bottom:5px;">📊 수급 모멘텀 (5일)</div>'
                f'<div style="font-size:0.83rem;color:{COLORS["text"]};font-weight:600;margin-bottom:4px;">'
                f'{_trend["status"]}</div>'
                f'<div style="font-size:0.72rem;color:{COLORS["text_2"]};line-height:1.4;word-break:keep-all;">'
                f'{_key_line}</div>'
                f'<div style="font-size:0.65rem;color:#b0b0b0;margin-top:5px;">{_trend["summary_text"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        except Exception:
            pass

    # ── Dead time / Breakout — 컴팩트 인라인 카드 ──────────────────────
    _dt_dead = dead_time.get("is_dead", False)
    _dt_msg  = dead_time.get("message", "")
    _bk_status = breakout.get("status", "wait")
    _bk_detail = breakout.get("detail", "")

    if _dt_msg or _bk_detail:
        _dt_border = "#ff9500" if _dt_dead else "#0066cc"
        _dt_icon   = "⏳" if _dt_dead else "📊"
        _bk_border = "#34c759" if _bk_status == "breakout_both" else ("#0066cc" if "breakout" in _bk_status else "#b0b0b0")
        _bk_icon   = "🚀" if _bk_status == "breakout_both" else ("📈" if "breakout" in _bk_status else "⏸️")
        _bk_label  = "돌파 충족" if _bk_status == "breakout_both" else ("부분 돌파" if "breakout" in _bk_status else "관망")

        _left_col, _right_col = st.columns(2)
        if _dt_msg:
            _left_col.markdown(
                f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};'
                f'border-left:3px solid {_dt_border};border-radius:10px;padding:9px 12px;">'
                f'<div style="font-size:0.65rem;color:{COLORS["text_2"]};letter-spacing:.6px;margin-bottom:4px;">'
                f'{_dt_icon} 거래 에너지</div>'
                f'<div style="font-size:0.82rem;color:{COLORS["text"]};line-height:1.45;word-break:keep-all;">'
                f'{_dt_msg}</div></div>',
                unsafe_allow_html=True,
            )
        if _bk_detail:
            _right_col.markdown(
                f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};'
                f'border-left:3px solid {_bk_border};border-radius:10px;padding:9px 12px;">'
                f'<div style="font-size:0.65rem;color:{COLORS["text_2"]};letter-spacing:.6px;margin-bottom:4px;">'
                f'{_bk_icon} 돌파 — <b style="color:{_bk_border}">{_bk_label}</b></div>'
                f'<div style="font-size:0.82rem;color:{COLORS["text"]};line-height:1.45;word-break:keep-all;">'
                f'{_bk_detail}</div></div>',
                unsafe_allow_html=True,
            )

    # ── 신호 근거 + 리스크 경고 (Enhanced Hybrid Signal) ─────────────────
    _h_strategy = hybrid.get("strategy", "")
    _h_reasons  = hybrid.get("reasons", [])
    _h_warnings = hybrid.get("warnings", [])

    # 전략 배지 — 적용된 투자 전략을 한눈에 표시
    if _h_strategy:
        _s_color = (
            "#ff9500" if "초단기" in _h_strategy else
            "#0066cc" if "단기"   in _h_strategy else
            "#34c759"
        )
        st.markdown(
            f'<div style="display:inline-flex;align-items:center;gap:6px;'
            f'background:rgba(0,0,0,0.04);'
            f'border:1px solid {_s_color}55;border-radius:9999px;'
            f'padding:3px 14px;margin-bottom:10px;">'
            f'<span style="color:{_s_color};font-size:0.72rem;font-weight:600;">'
            f'📊 {_h_strategy}</span></div>',
            unsafe_allow_html=True,
        )

    # 양쪽 모두 있을 때만 2컬럼, 한쪽만 있으면 전체 폭 사용
    _has_both = bool(_h_reasons) and bool(_h_warnings)
    if _has_both:
        _r_col, _w_col = st.columns([1, 1])
        _reason_ctx    = _r_col
        _warn_ctx      = _w_col
    else:
        _reason_ctx = st
        _warn_ctx   = st

    # 확정 매매 근거 (초록 왼쪽 테두리)
    if _h_reasons:
        _reasons_rows = "".join(
            f'<div style="display:flex;align-items:flex-start;gap:8px;padding:6px 0;'
            f'border-bottom:1px solid {COLORS["border"]};">'
            f'<span style="font-size:0.84rem;line-height:1.5;color:{COLORS["text"]};'
            f'word-break:keep-all;">{r}</span>'
            f'</div>'
            for r in _h_reasons
        )
        _reason_ctx.markdown(
            f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};'
            f'border-left:3px solid #34c759;border-radius:10px;'
            f'padding:10px 14px;margin-bottom:8px;">'
            f'<div style="font-size:0.65rem;color:#34c759;letter-spacing:.8px;'
            f'text-transform:uppercase;font-weight:600;margin-bottom:6px;">📋 확정 매매 근거</div>'
            f'{_reasons_rows}</div>',
            unsafe_allow_html=True,
        )

    # 주의 신호 (황색 배경 + 강조 테두리로 시각적 분리)
    if _h_warnings:
        _warn_rows = "".join(
            f'<div style="display:flex;align-items:flex-start;gap:8px;padding:6px 0;'
            f'border-bottom:1px solid rgba(245,158,11,0.2);">'
            f'<span style="font-size:0.84rem;line-height:1.5;color:#fbbf24;'
            f'word-break:keep-all;">{w}</span>'
            f'</div>'
            for w in _h_warnings
        )
        _warn_ctx.markdown(
            f'<div style="background:rgba(245,158,11,0.07);'
            f'border:1px solid rgba(245,158,11,0.35);'
            f'border-left:3px solid #f59e0b;border-radius:10px;'
            f'padding:10px 14px;margin-bottom:8px;">'
            f'<div style="font-size:0.65rem;color:#f59e0b;letter-spacing:.8px;'
            f'text-transform:uppercase;font-weight:600;margin-bottom:6px;">⚠️ 주의 신호</div>'
            f'{_warn_rows}</div>',
            unsafe_allow_html=True,
        )


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
                _dc   = "#34c759" if _diff >= 0 else "#ff3b30"
                return (
                    f'<div style="display:flex;justify-content:space-between;'
                    f'align-items:center;padding:5px 0;border-bottom:1px solid #e0e0e0;gap:6px;">'
                    f'<span style="font-size:0.85rem;color:{color};white-space:nowrap;flex-shrink:0;">● {label}</span>'
                    f'<span style="font-size:0.88rem;color:#1d1d1f;white-space:nowrap;font-variant-numeric:tabular-nums;">{_pf.format(_v)}</span>'
                    f'<span style="font-size:0.85rem;color:{_dc};white-space:nowrap;flex-shrink:0;">{_arrow}{abs(_diff):.1f}%</span>'
                    f'</div>'
                )

            vwap_rows = (
                _vwap_row_html("VWAP 주간(5일)",   "VWAP_W", "#ff9500") +
                _vwap_row_html("VWAP 월간(20일)",  "VWAP_M", "#0066cc") +
                _vwap_row_html("VWAP 분기(60일)",  "VWAP_Q", "#34c759")
            )
            if vwap_rows:
                st.markdown("#### 📍 VWAP 위치")
                st.markdown(
                    f'<div style="background:#f5f5f7;border:1px solid #e0e0e0;border-radius:8px;padding:10px 14px;">'
                    f'{vwap_rows}</div>',
                    unsafe_allow_html=True,
                )

    with _tech_right:
        if expected:
            st.markdown("#### 📊 예상 수익·리스크")
            _exp_ret  = expected.get("expected_return_pct", 0.0)
            _sharpe   = expected.get("sharpe", 0.0)
            _max_dd   = expected.get("max_drawdown_pct", 0.0)
            _win_prob = expected.get("win_prob", 50.0)
            _exp_color = COLORS["gain"] if _exp_ret >= 0 else COLORS["loss"]
            st.markdown(
                f'<div style="background:#f5f5f7;border:1px solid #e0e0e0;border-radius:8px;padding:12px 16px;">'
                f'<div style="display:flex;justify-content:space-between;margin-bottom:6px;">'
                f'<span style="color:#7a7a7a;font-size:.85rem">예상 수익률</span>'
                f'<b style="color:{_exp_color};font-variant-numeric:tabular-nums;">{_exp_ret:+.1f}%</b></div>'
                f'<div style="display:flex;justify-content:space-between;margin-bottom:6px;">'
                f'<span style="color:#7a7a7a;font-size:.85rem">샤프지수</span>'
                f'<b style="color:#1d1d1f;font-variant-numeric:tabular-nums;">{_sharpe:.2f}</b></div>'
                f'<div style="display:flex;justify-content:space-between;margin-bottom:6px;">'
                f'<span style="color:#7a7a7a;font-size:.85rem">최대 낙폭</span>'
                f'<b style="color:{COLORS["loss"]};font-variant-numeric:tabular-nums;">{_max_dd:.1f}%</b></div>'
                f'<div style="display:flex;justify-content:space-between;">'
                f'<span style="color:#7a7a7a;font-size:.85rem">승률 추정</span>'
                f'<b style="color:#1d1d1f;font-variant-numeric:tabular-nums;">{_win_prob:.0f}%</b></div>'
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
                ("볼린저밴드 상단",      "bb_upper",   "#0066cc"),
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
    inv_history_fn=None,
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

            _src_label = {
                "krx_api":     "KRX 공공 API",
                "krx_api+fdr": "KRX API + FDR",
                "naver":       "네이버 금융",
                "static_map":  "정적 데이터",
                "error":       "오류",
            }.get(_etf_data.get("source", ""), _etf_data.get("source", ""))
            _aum_txt = f"순자산(AUM): **{_aum:,.0f}억원**  |  " if _aum else ""
            st.caption(f"{_aum_txt}섹터: **{_sector}**  |  데이터: {_src_label}")
            st.markdown("---")

            _es_col, _er_col = st.columns([1, 2])
            with _es_col:
                _etf_s   = _etf_score.get("etf_score", 0.0)
                _etf_lbl = _etf_score.get("etf_label", "N/A")
                if _etf_s >= 1.5:
                    _ebg, _efc = "rgba(52,199,89,0.06)", "#1a7a35"
                elif _etf_s <= -0.5:
                    _ebg, _efc = "rgba(255,59,48,0.06)", "#cc2200"
                else:
                    _ebg, _efc = "#f5f5f7", "#7a7a7a"
                st.markdown(f"""
<div class="signal-box" style="background:{_ebg};">
  <div style="font-size:0.7rem;color:#7a7a7a;margin-bottom:3px;">📊 ETF 투자 판정</div>
  <div style="font-size:1.3rem;font-weight:600;color:{_efc};">{_etf_lbl}</div>
  <div style="font-size:0.9rem;color:#7a7a7a;margin-top:4px;font-variant-numeric:tabular-nums;">점수: <b style="color:{_efc};">{_etf_s:+.1f}</b> / ±6.5</div>
</div>
""", unsafe_allow_html=True)

                st.markdown("**항목별 점수**")
                _bd = _etf_score.get("score_breakdown", {})
                for _lbl, _rng, _clr in [
                    ("괴리율",    "±3",   "#0066cc"),
                    ("운용보수",  "±2",   "#34c759"),
                    ("추적오차",  "±1.5", "#ff9500"),
                    ("배당수익률","±0.5", "#ff9500"),
                ]:
                    _sv  = _bd.get(_lbl, 0.0)
                    _max = float(_rng.replace("±", ""))
                    _pct = int((_sv + _max) / (_max * 2) * 100)
                    st.markdown(
                        f'<div style="margin-bottom:5px;">'
                        f'<div style="display:flex;justify-content:space-between;font-size:0.75rem;">'
                        f'<span style="color:#1d1d1f;">{_lbl} <span style="color:#b0b0b0;">({_rng})</span></span>'
                        f'<b style="color:{_clr};font-variant-numeric:tabular-nums;">{_sv:+.1f}</b></div>'
                        f'<div style="background:#e0e0e0;border-radius:3px;height:4px;">'
                        f'<div style="background:{_clr};width:{_pct}%;height:4px;border-radius:3px;"></div>'
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
                _prem_color = "#ff3b30" if _premium > 1 else ("#34c759" if _premium < -0.5 else "#7a7a7a")
                st.markdown(
                    f'<div style="background:#ffffff;border:1px solid #e0e0e0;border-radius:12px;padding:16px 18px;box-shadow:0 1px 3px rgba(0,0,0,0.06);">'
                    f'<div style="font-size:0.72rem;color:#7a7a7a;margin-bottom:8px;">📐 NAV 괴리율이란?</div>'
                    f'<div style="font-size:0.95rem;color:#1d1d1f;margin-bottom:6px;font-variant-numeric:tabular-nums;">'
                    f'현재가(<b>{_price:,.0f}원</b>) vs NAV(<b>{_nav:,.0f}원</b>)</div>'
                    f'<div style="font-size:1.2rem;font-weight:700;color:{_prem_color};">'
                    f'{"🔴 프리미엄" if _premium > 0 else "🟢 할인"} {_premium:+.2f}%</div>'
                    f'<div style="font-size:0.78rem;color:#7a7a7a;margin-top:8px;line-height:1.6;word-break:keep-all;">'
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
                _sig_clr   = "#34c759" if _sig_score >= 3 else ("#ff3b30" if _sig_score <= -3 else "#b0b0b0")
                st.markdown(
                    f'<div style="background:#ffffff;border:1px solid #e0e0e0;border-radius:12px;'
                    f'padding:14px 18px;display:flex;align-items:center;justify-content:space-between;box-shadow:0 1px 2px rgba(0,0,0,0.06);">'
                    f'<span style="font-size:0.85rem;color:#7a7a7a;">기술적 신호</span>'
                    f'<b style="color:{_sig_clr};font-size:1rem;font-variant-numeric:tabular-nums;">{_sig_label} ({_sig_score:+.1f}점)</b>'
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
                    fbg, ffc = "rgba(52,199,89,0.06)", "#1a7a35"
                elif fs <= -2:
                    fbg, ffc = "rgba(255,59,48,0.06)", "#cc2200"
                else:
                    fbg, ffc = "#f5f5f7", "#7a7a7a"
                st.markdown(f"""
<div class="signal-box" style="background:{fbg};">
  <div style="font-size:0.7rem;color:#7a7a7a;margin-bottom:3px;">🏛️ 장투 신호</div>
  <div style="font-size:1.25rem;font-weight:600;color:{ffc};">🏛️ {flbl}</div>
  <div style="font-size:0.8rem;color:#7a7a7a;margin-top:3px;font-variant-numeric:tabular-nums;">장투 점수: <b style="color:{ffc};">{fs:+.1f}</b></div>
</div>
""", unsafe_allow_html=True)

                def _fund_bar(label, score, weight):
                    c = "#34c759" if score >= 65 else ("#ff3b30" if score <= 35 else "#ff9500")
                    st.markdown(
                        f'<div style="margin-bottom:5px;">'
                        f'<div style="display:flex;justify-content:space-between;font-size:0.75rem;">'
                        f'<span style="color:#1d1d1f;">{label} <span style="color:#b0b0b0;">({weight})</span></span>'
                        f'<b style="color:{c};font-variant-numeric:tabular-nums;">{score:.0f}</b></div>'
                        f'<div style="background:#e0e0e0;border-radius:3px;height:4px;">'
                        f'<div style="background:{c};width:{int(score)}%;height:4px;border-radius:3px;"></div>'
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
                ("그레이엄", "📖 벤저민 그레이엄", "#0066cc", "안전마진·가치투자의 아버지"),
                ("버핏",    "🏛️ 워렌 버핏",       "#34c759", "ROE 지속성·경제적 해자"),
                ("린치",    "🚀 피터 린치",        "#ff9500", "PEG·성장주 발굴"),
                ("오닐",    "🔥 윌리엄 오닐",      "#ff3b30", "신고가·CANSLIM"),
            ]
            _vcols = st.columns(4)
            for _vcol, (_key, _name, _clr, _sub) in zip(_vcols, _master_meta):
                _v       = _verdicts.get(_key, {})
                _icon    = _v.get("icon", "—")
                _verdict = _v.get("판정", "N/A")
                _comment = _v.get("comment", "데이터 부족")
                _vcol.markdown(
                    f'<div style="background:#ffffff;border-radius:12px;padding:14px;'
                    f'border:1px solid #e0e0e0;border-top:3px solid {_clr};min-height:130px;'
                    f'box-shadow:0 1px 3px rgba(0,0,0,0.06);">'
                    f'<div style="font-size:0.65rem;color:#7a7a7a;margin-bottom:5px;">{_sub}</div>'
                    f'<div style="font-size:0.85rem;font-weight:600;color:{_clr};margin-bottom:6px;">{_name}</div>'
                    f'<div style="font-size:1rem;font-weight:600;color:#1d1d1f;margin-bottom:6px;">{_icon} {_verdict}</div>'
                    f'<div style="font-size:0.75rem;color:#7a7a7a;line-height:1.6;word-break:keep-all;">{_comment}</div>'
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
                            _color = "#34c759" if val > 0 else ("#ff3b30" if val < 0 else "#7a7a7a")
                            _inv_bg = "rgba(52,199,89,0.06)" if val > 0 else ("rgba(255,59,48,0.06)" if val < 0 else "#f5f5f7")
                            col.markdown(
                                f'<div style="background:{_inv_bg};border:1px solid #e0e0e0;border-radius:10px;'
                                f'padding:14px;text-align:center;">'
                                f'<div style="font-size:0.72rem;color:#7a7a7a;margin-bottom:6px;">{label}</div>'
                                f'<div style="font-size:1.1rem;font-weight:600;color:{_color};font-variant-numeric:tabular-nums;">{_sign}{val:,}</div>'
                                f'<div style="font-size:0.68rem;color:#b0b0b0;margin-top:3px;">주(株)</div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                        else:
                            col.markdown(
                                f'<div style="background:#f5f5f7;border:1px solid #e0e0e0;border-radius:10px;'
                                f'padding:14px;text-align:center;">'
                                f'<div style="font-size:0.72rem;color:#7a7a7a;margin-bottom:6px;">{label}</div>'
                                f'<div style="font-size:1.1rem;color:#b0b0b0;">N/A</div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                else:
                    st.caption("투자자 매매 동향 데이터를 불러올 수 없습니다.")

                # 수급 모멘텀 트렌드 분석 (5거래일)
                if inv_history_fn:
                    with st.spinner("수급 트렌드 분석 중..."):
                        _hist = inv_history_fn(ticker)
                    _trend = analyze_investor_trend(_hist)
                    _tr_score = _trend["score"]

                    if _tr_score >= 3:    _tr_color = "#34c759"
                    elif _tr_score >= 1:  _tr_color = "#0066cc"
                    elif _tr_score <= -2: _tr_color = "#ff3b30"
                    else:                 _tr_color = "#7a7a7a"

                    st.markdown("---")
                    st.markdown(
                        f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};'
                        f'border-left:4px solid {_tr_color};border-radius:12px;padding:14px 18px;margin-bottom:14px;">'
                        f'<div style="font-size:0.68rem;color:{COLORS["text_2"]};letter-spacing:0.8px;'
                        f'text-transform:uppercase;margin-bottom:6px;">📈 수급 모멘텀 트렌드 분석 (최근 5거래일)</div>'
                        f'<div style="font-size:1.05rem;font-weight:700;color:{_tr_color};">{_trend["status"]}</div>'
                        f'<div style="font-size:0.75rem;color:{COLORS["text_2"]};margin-top:4px;">{_trend["summary_text"]}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    _tr_col1, _tr_col2 = st.columns(2)
                    with _tr_col1:
                        st.markdown(
                            '<div style="font-size:0.75rem;font-weight:600;color:#34c759;'
                            'margin-bottom:8px;letter-spacing:0.5px;">🎯 수급 긍정 근거</div>',
                            unsafe_allow_html=True,
                        )
                        for _r in _trend["reasons"]:
                            st.markdown(
                                f'<div style="background:rgba(52,199,89,0.07);border-left:3px solid #34c759;'
                                f'border-radius:8px;padding:8px 12px;margin-bottom:6px;'
                                f'font-size:0.78rem;color:{COLORS["text"]};line-height:1.5;word-break:keep-all;">'
                                f'{_r}</div>',
                                unsafe_allow_html=True,
                            )
                    with _tr_col2:
                        st.markdown(
                            '<div style="font-size:0.75rem;font-weight:600;color:#ff3b30;'
                            'margin-bottom:8px;letter-spacing:0.5px;">⚠️ 수급 리스크 경고</div>',
                            unsafe_allow_html=True,
                        )
                        if _trend["warnings"]:
                            for _w in _trend["warnings"]:
                                st.markdown(
                                    f'<div style="background:rgba(255,59,48,0.08);border-left:3px solid #ff3b30;'
                                    f'border-radius:8px;padding:8px 12px;margin-bottom:6px;'
                                    f'font-size:0.78rem;color:{COLORS["text"]};line-height:1.5;word-break:keep-all;">'
                                    f'{_w}</div>',
                                    unsafe_allow_html=True,
                                )
                        else:
                            st.markdown(
                                f'<div style="background:rgba(52,199,89,0.07);border-left:3px solid #34c759;'
                                f'border-radius:8px;padding:8px 12px;'
                                f'font-size:0.78rem;color:#34c759;line-height:1.5;">'
                                f'✅ 메이저 수급 이탈 징후나 개인에게 물량을 떠넘기는 불리한 패턴이 발견되지 않았습니다.'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

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
    usd_krw: float = 1300.0,
    set_cookie_fn=None,
    delete_cookie_fn=None,
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
            if delete_cookie_fn:
                delete_cookie_fn("auth_token")
            st.rerun()

    st.divider()

    if not _tok or not db_get_user(_tok):
        st.session_state["auth_token"]   = None
        st.session_state["auth_user_id"] = None
        st.session_state["auth_email"]   = None
        if delete_cookie_fn:
            delete_cookie_fn("auth_token")
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

    # 환율: render_portfolio_tab에서 계산된 값 사용 (비정상 시 yfinance 재시도)
    _usd_krw = usd_krw if usd_krw > 100 else 1300.0
    if _usd_krw == 1300.0:
        try:
            _fx_raw = yf.download("USDKRW=X", period="2d", auto_adjust=True, progress=False)
            if not _fx_raw.empty:
                _fx_s = (_fx_raw["Close"] if "Close" in _fx_raw.columns else _fx_raw.iloc[:, 0]).dropna()
                if not _fx_s.empty:
                    _v = float(_fx_s.iloc[-1])
                    if _v > 100:
                        _usd_krw = _v
        except Exception:
            pass

    def _krw(price: float, ticker: str) -> float:
        return price if ticker.upper().endswith((".KS", ".KQ")) else price * _usd_krw

    _cum_sell_krw   = sum(_krw(t["sell_price"] * t["quantity"], t["ticker"]) for t in _trade_history)
    _cum_profit_krw = sum(_krw(t["net_profit"],                 t["ticker"]) for t in _trade_history)
    _cum_buy_krw    = sum(_krw(t["buy_price"]  * t["quantity"], t["ticker"]) for t in _trade_history)

    st.markdown("""
<div style="background:rgba(0,102,204,0.06);border:1px solid rgba(0,102,204,0.18);
        border-radius:12px;padding:12px 16px;margin-bottom:8px;
        display:flex;align-items:center;gap:10px">
  <span style="font-size:1.2rem">➕</span>
  <div>
    <div style="font-size:.85rem;font-weight:600;color:#0066cc">종목 추가는 사이드바(좌측)에서</div>
    <div style="font-size:.75rem;color:#7a7a7a;margin-top:2px">로그인 상태에서 사이드바 하단 '포트폴리오 종목 추가' 섹션 이용</div>
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

        _pnl_color    = COLORS["gain"] if _total_pnl >= 0 else COLORS["loss"]
        _profit_color = COLORS["gain"] if _cum_profit_krw >= 0 else COLORS["loss"]
        _pnl_pct_str  = f"{_total_pnl_pct:+.2f}%"
        _m1, _m2, _m3, _m4 = st.columns(4)
        _m1.markdown(f"""
<div class="ma-metric-card">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px">
    <span style="font-size:.75rem;color:#7a7a7a;font-weight:500;">총 매수 금액</span>
  </div>
  <div style="font-size:1.35rem;font-weight:600;color:#1d1d1f;line-height:1.14;font-variant-numeric:tabular-nums;">₩{_total_cost:,.0f}</div>
  <div style="font-size:.72rem;color:#7a7a7a;margin-top:6px">평단가 × 수량 합계</div>
</div>
""", unsafe_allow_html=True)
        _m2.markdown(f"""
<div class="ma-metric-card">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px">
    <span style="font-size:.75rem;color:#7a7a7a;font-weight:500;">현재 평가금</span>
  </div>
  <div style="font-size:1.35rem;font-weight:600;color:#1d1d1f;line-height:1.14;font-variant-numeric:tabular-nums;">₩{_total_val:,.0f}</div>
  <div style="font-size:.72rem;color:{_pnl_color};margin-top:6px;font-weight:600;font-variant-numeric:tabular-nums;">{_pnl_pct_str} &nbsp;·&nbsp; USD/KRW≈{_usd_krw:,.0f}</div>
</div>
""", unsafe_allow_html=True)
        _m3.markdown(f"""
<div class="ma-metric-card">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px">
    <span style="font-size:.75rem;color:#7a7a7a;font-weight:500;">누적 매도금</span>
  </div>
  <div style="font-size:1.35rem;font-weight:600;color:#1d1d1f;line-height:1.14;font-variant-numeric:tabular-nums;">₩{_cum_sell_krw:,.0f}</div>
  <div style="font-size:.72rem;color:#7a7a7a;margin-top:6px">매도 회수 총액 (원금+수익)</div>
</div>
""", unsafe_allow_html=True)
        _m4.markdown(f"""
<div class="ma-metric-card">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px">
    <span style="font-size:.75rem;color:#7a7a7a;font-weight:500;">누적 실현 손익</span>
  </div>
  <div style="font-size:1.35rem;font-weight:600;color:{_profit_color};line-height:1.14;font-variant-numeric:tabular-nums;">₩{_cum_profit_krw:+,.0f}</div>
  <div style="font-size:.72rem;color:#7a7a7a;margin-top:6px">매도 완료 종목 손익 합계</div>
</div>
""", unsafe_allow_html=True)

        _total_in = _total_cost + _cum_buy_krw
        if _total_in > 0:
            _overall_pct = (_total_val + _cum_sell_krw) / _total_in * 100 - 100
            _ov_clr = "#34c759" if _overall_pct >= 0 else "#ff3b30"
            st.markdown(
                f'<div style="text-align:right;font-size:.82rem;color:#7a7a7a;margin-top:-6px">'
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
                    f'<div style="font-size:.75rem;color:#34c759;margin-bottom:6px">'
                    f'✅ 실시간 시세 반영 완료 &nbsp;|&nbsp; 기준 시각: <b>{_news_rt_ts}</b>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            _avg_s = _pf_news_result.get("portfolio_sentiment_avg", 0.0)
            _avg_l = _pf_news_result.get("portfolio_sentiment_label", "중립")
            _s_clr = "#34c759" if _avg_s >= 0.5 else ("#ff3b30" if _avg_s <= -0.5 else "#7a7a7a")
            _ew_c1, _ew_c2 = st.columns([1, 3])
            _ew_c1.markdown(
                f'<div style="background:#ffffff;border:1px solid #e0e0e0;border-radius:10px;padding:18px;text-align:center;box-shadow:0 1px 2px rgba(0,0,0,0.06)">'
                f'<div style="font-size:.75rem;color:#7a7a7a;margin-bottom:6px">포트폴리오 심리 지수</div>'
                f'<div style="font-size:2.2rem;font-weight:600;color:{_s_clr};font-variant-numeric:tabular-nums;">{_avg_s:+.2f}</div>'
                f'<div style="font-size:.85rem;color:{_s_clr};margin-top:4px">{_avg_l}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            _per_news = _pf_news_result.get("per_ticker", {})
            _bdg = '<div style="display:flex;flex-wrap:wrap;gap:6px;padding:8px 0">'
            for _bt, _br in _per_news.items():
                _bs   = _br.get("score", 0.0)
                _bl   = _br.get("label", "중립")
                _bc   = "#34c759" if _bs >= 0.5 else ("#ff3b30" if _bs <= -0.5 else "#7a7a7a")
                _bnm  = _pf_nm.get(_bt, "")
                _blbl = (f"{_bnm}<br><span style='font-size:.72rem;color:#7a7a7a'>{_bt}</span>" if _bnm else _bt)
                _cur_p  = _pf_prices.get(_bt, 0.0)
                _avg_p  = next((i["avg_price"] for i in _items if i["ticker"] == _bt), 0.0)
                _pnl_pc = (_cur_p / _avg_p - 1) * 100 if (_cur_p and _avg_p) else None
                _pnl_html = ""
                if _pnl_pc is not None:
                    _pnl_c  = "#34c759" if _pnl_pc >= 0 else "#ff3b30"
                    _impact = (
                        "호재 반영↑" if (_bs >= 0.5 and _pnl_pc >= 0)
                        else "악재 하락↓" if (_bs <= -0.5 and _pnl_pc < 0)
                        else "뉴스↑ 가격↓" if (_bs >= 0.5 and _pnl_pc < 0)
                        else "뉴스↓ 가격↑" if (_bs <= -0.5 and _pnl_pc >= 0)
                        else "중립"
                    )
                    _pnl_html = f"<span style='font-size:.7rem;color:{_pnl_c};margin-left:4px;font-variant-numeric:tabular-nums;'>({_pnl_pc:+.1f}% {_impact})</span>"
                _bdg += (
                    f'<span style="background:#f5f5f7;border:1px solid #e0e0e0;border-radius:8px;padding:6px 12px;font-size:.85rem;line-height:1.5">'
                    f'<b style="color:#1d1d1f">{_blbl}</b> '
                    f'<span style="color:{_bc};font-variant-numeric:tabular-nums;">{_bl} ({_bs:+.1f})</span>'
                    f'{_pnl_html}</span>'
                )
            _bdg += '</div>'
            _ew_c2.markdown(_bdg, unsafe_allow_html=True)

            _alerts = _pf_news_result.get("important_alerts", [])
            if _alerts:
                st.markdown("**🔔 중요 알림**")
                for _al in _alerts:
                    _akw   = _al["keyword"]
                    _ak_c  = "#ff3b30" if _akw in ("증자", "상장폐지", "소송", "제재") else "#ff9500"
                    _aname = _al.get("company_name") or _al["ticker"]
                    st.markdown(
                        f'<div style="background:#f5f5f7;border-left:4px solid {_ak_c};border:1px solid #e0e0e0;'
                        f'padding:10px 16px;border-radius:0 8px 8px 0;margin:3px 0">'
                        f'<span style="background:{_ak_c};color:#000;font-size:.72rem;'
                        f'border-radius:4px;padding:2px 6px;margin-right:8px;font-weight:700">[{_akw}]</span>'
                        f'<b style="color:#1d1d1f">{_aname}</b> — '
                        f'<span style="color:#7a7a7a">{_al["title"]}</span></div>',
                        unsafe_allow_html=True,
                    )
        else:
            st.caption("'뉴스 분석' 버튼을 눌러 포트폴리오 종목의 뉴스 감성을 분석하세요.")

        st.divider()

        _pf_per_news: dict = st.session_state.get("pf_news_result", {}).get("per_ticker", {})
        _tbl_css = """
<style>
.pf-tbl{width:100%;border-collapse:collapse;font-size:.9rem;font-variant-numeric:tabular-nums;
        font-family:"SF Pro Text",system-ui,-apple-system,sans-serif}
.pf-tbl th{background:#f5f5f7;color:#7a7a7a;padding:10px 14px;text-align:right;
       font-weight:600;border-bottom:1px solid #e0e0e0;white-space:nowrap;letter-spacing:-0.12px}
.pf-tbl th:first-child,.pf-tbl th:last-child{text-align:left}
.pf-tbl td{padding:10px 14px;border-bottom:1px solid #e0e0e0;text-align:right;color:#1d1d1f}
.pf-tbl td:first-child{text-align:left;color:#1d1d1f;font-weight:600}
.pf-tbl tr:last-child td{border-bottom:none}
.pf-tbl tr:hover td{background:rgba(0,102,204,0.04)}
.pp{color:#34c759;font-weight:600} .pn{color:#ff3b30;font-weight:600} .pz{color:#b0b0b0}
.ai-op{text-align:left!important;font-size:.8rem;max-width:200px;word-break:keep-all;line-height:1.4;color:#7a7a7a}
</style>"""
        _tbl_head = (
            '<table class="pf-tbl"><thead><tr>'
            '<th>종목</th><th>수량</th><th>평단가</th>'
            '<th>현재가</th><th>수익률(%)</th><th>평가손익</th><th>평가금(₩)</th><th>AI 의견</th>'
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
                _pnl_val   = (_cur - _avg) * _qty
                _pnl_pct   = (_cur / _avg - 1) * 100 if _avg else 0.0
                _cls       = "pp" if _pnl_pct > 0 else ("pn" if _pnl_pct < 0 else "pz")
                _cur_str   = _fp(_cur)
                _pct_str   = f"{_pnl_pct:+.2f}%"
                _pnl_str   = (f"+{_fp(abs(_pnl_val))}" if _pnl_val >= 0 else f"-{_fp(abs(_pnl_val))}")
                # 현재평가금 원화 환산 — USD 종목은 실시간 환율 적용
                _val_krw   = _krw(_cur, _t) * _qty
                _val_str   = f"₩{_val_krw:,.0f}"
                if not _is_krw_item:
                    _val_str += f"<div style='font-size:.68rem;color:#7a7a7a'>@{_usd_krw:,.0f}</div>"
            else:
                _cls = "pz"; _cur_str = "-"; _pct_str = "-"; _pnl_str = "-"; _val_str = "-"
            _nr   = _pf_per_news.get(_t)
            _nsc  = _nr.get("score", 0.0) if _nr else None
            _nlb  = _nr.get("label", "중립") if _nr else None
            _pnlp = (_cur / _avg - 1) * 100 if (_cur and _avg) else None
            if _nsc is None:
                _ai_txt, _ai_c = "분석 대기", "#7a7a7a"
            elif _nsc >= 2 and _pnlp is not None and _pnlp >= 0:
                _ai_txt, _ai_c = "호재 발생 + 수익 중 — 홀딩 권장", "#34c759"
            elif _nsc >= 1:
                _ai_txt, _ai_c = f"뉴스 {_nlb} — 홀딩 유지", "#34c759"
            elif _nsc <= -2 and _pnlp is not None and _pnlp < -5:
                _ai_txt, _ai_c = "부정 신호 + 손실 — 손절 검토", "#ff3b30"
            elif _nsc <= -1:
                _ai_txt, _ai_c = f"뉴스 {_nlb} — 비중 축소 검토", "#ff9500"
            elif _pnlp is not None and _pnlp < -10:
                _ai_txt, _ai_c = "큰 손실 중 — 손절라인 점검", "#ff3b30"
            else:
                _ai_txt, _ai_c = "중립 — 관망", "#7a7a7a"
            _name_cell = (
                f"<div style='font-weight:600;color:#e0e0e0'>{_nm}</div>"
                f"<div style='font-size:.75rem;color:#7a7a7a'>{_t}</div>"
            ) if _nm else _t
            _tbl_rows_html.append(
                f"<tr><td style='text-align:left'>{_name_cell}</td>"
                f"<td>{_qty:g}</td><td>{_fp(_avg)}</td><td>{_cur_str}</td>"
                f"<td class='{_cls}'>{_pct_str}</td><td class='{_cls}'>{_pnl_str}</td>"
                f"<td>{_val_str}</td>"
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
                "<small style='color:#7a7a7a'>실시간가 기준 손절/익절 가이드 · 트레일링 스탑 · 목표가 근접 알림</small>",
                unsafe_allow_html=True,
            )
            _exit_result: dict = st.session_state.get("pf_exit_result", {})
            _exit_rt_ts: str   = st.session_state.get("pf_exit_rt_ts", "")
            if _eg_btn.button("매도 가이드 분석", key="pf_exit_calc", type="primary", use_container_width=True):
                _exit_result = {}
                _exit_now    = now_kst_fn().strftime("%H:%M:%S")
                with st.spinner("실시간 현재가 조회 및 ATR 변동성 분석 중..."):
                    for _it in _items:
                        _t = _it["ticker"]
                        try:
                            _rt_exit = realtime_price_fn(_t)
                            if _rt_exit["price"] > 0:
                                _pf_prices[_t] = _rt_exit["price"]
                            if _rt_exit.get("stale") and _rt_exit.get("stale_msg"):
                                st.caption(f"⏸️ {_t}: {_rt_exit['stale_msg']}")
                            _cdata  = get_stock_data_fn(_t, period="3mo")
                            _rt_p   = _pf_prices.get(_t, _it["avg_price"])
                            _avg_p  = _it["avg_price"]
                            _is_krw = _t.upper().endswith((".KS", ".KQ"))

                            # ATR-14 추출 (차트 데이터 우선, 없으면 가격의 2% 추정)
                            _atr14 = 0.0
                            if _cdata is not None and not _cdata.empty and "ATR" in _cdata.columns:
                                _atr_val = _cdata["ATR"].dropna()
                                if not _atr_val.empty:
                                    _atr14 = float(_atr_val.iloc[-1])
                            if _atr14 <= 0:
                                _atr14 = _avg_p * 0.02

                            # max_price_since_buy: 기존 세션 최고가 활용
                            _prev_max = _trailing.get(_t, 0.0)
                            _atr_guide = calc_atr_trailing_guide(
                                buying_price=_avg_p,
                                current_price=_rt_p,
                                atr_14=_atr14,
                                max_price_since_buy=_prev_max,
                                is_krw=_is_krw,
                            )
                            # 갱신된 최고가 세션에 반영
                            _trailing[_t] = _atr_guide["max_price_since_buy"]

                            _exit_result[_t] = {
                                **_atr_guide,
                                "rt_price":  _rt_p,
                                "avg_price": _avg_p,
                                "pnl_pct":   (_rt_p / _avg_p - 1) * 100 if _avg_p else 0.0,
                                "is_rt":     _rt_exit.get("is_realtime", False),
                                "rt_ts":     _exit_now,
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
                        f'<div style="font-size:.75rem;color:#34c759;margin-bottom:8px">'
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
                _cur_label = "실시간가" if _stp.get("is_rt") else "현재가"

                if _stp:
                    # ── ATR 가이드 결과 렌더링 ──────────────────────────────────
                    _status    = _stp.get("status", "")
                    _guide_clr = _stp.get("guide_clr", "#b0b0b0")
                    _message   = _stp.get("message", "")
                    _trigger   = _stp.get("final_trigger_line", 0.0)
                    _atr_v     = _stp.get("atr_14", 0.0)
                    _atr_stop  = _stp.get("atr_stop", 0.0)
                    _trail     = _stp.get("trail_stop", 0.0)
                    _t_max     = _stp.get("max_price_since_buy", _trailing.get(_t, 0.0))
                    _pnl_pc    = _stp.get("pnl_pct", 0.0)
                    _pnl_clr   = "#34c759" if _pnl_pc >= 0 else "#ff3b30"

                    # 가격 정보 행
                    _info_items = []
                    if _cur:
                        _info_items.append(
                            f'{_cur_label} <b style="color:#1d1d1f">{_efmt(_cur)}</b> '
                            f'<span style="color:{_pnl_clr}">({_pnl_pc:+.1f}%)</span>'
                        )
                    if _avg:
                        _info_items.append(f'평단가 <b style="color:#7a7a7a">{_efmt(_avg)}</b>')
                    if _t_max and _avg and _t_max > _avg:
                        _info_items.append(
                            f'추적 최고가 <b style="color:#ff9500">{_efmt(_t_max)}</b>'
                        )

                    # ATR 분석 상세 행
                    _atr_items = []
                    if _atr_v:
                        _atr_items.append(f'ATR(14) <b style="color:#0066cc">{_efmt(_atr_v)}</b>')
                    if _atr_stop:
                        _atr_items.append(
                            f'ATR 손절선 <b style="color:#ff3b30">{_efmt(_atr_stop)}</b>'
                            f'<span style="color:#7a7a7a;font-size:.75rem"> (평단 − ATR×2)</span>'
                        )
                    if _trail:
                        _atr_items.append(
                            f'트레일링 스톱 <b style="color:#ff3b30">{_efmt(_trail)}</b>'
                            f'<span style="color:#7a7a7a;font-size:.75rem"> (최고가 − ATR×2.5)</span>'
                        )

                    _info_html = " &nbsp;|&nbsp; ".join(_info_items) if _info_items else ""
                    _atr_html  = " &nbsp;|&nbsp; ".join(_atr_items)  if _atr_items  else ""
                    _trigger_html = (
                        f'<div style="margin-top:8px;padding:7px 12px;background:#f5f5f7;'
                        f'border:1px solid {_guide_clr};border-radius:8px;'
                        f'display:inline-flex;align-items:center;gap:10px">'
                        f'<span style="font-size:.78rem;color:#7a7a7a">📌 최종 기준선 (매도 트리거)</span>'
                        f'<span style="font-size:1.05rem;font-weight:700;color:{_guide_clr}">'
                        f'{_efmt(_trigger)}</span></div>'
                    )
                    _status_html = (
                        f'<div style="margin-top:6px;padding:8px 12px;background:rgba(0,0,0,0.03);'
                        f'border-left:3px solid {_guide_clr};border-radius:0 8px 8px 0">'
                        f'<div style="font-size:.95rem;font-weight:600;color:{_guide_clr};margin-bottom:3px">'
                        f'{_status}</div>'
                        f'<div style="font-size:.82rem;color:#7a7a7a">{_message}</div></div>'
                    )

                    st.markdown(
                        f'<div style="background:#ffffff;border:1px solid #e0e0e0;border-radius:10px;padding:12px 16px;margin:6px 0;box-shadow:0 1px 2px rgba(0,0,0,0.05)">'
                        f'<div style="margin-bottom:6px">'
                        + (f'<span style="font-size:.95rem;font-weight:600;color:#1d1d1f">{_enm}</span> ' if _enm else "")
                        + f'<span style="font-size:.78rem;color:#7a7a7a">{_t}</span></div>'
                        f'<div style="font-size:.84rem;line-height:1.9;color:#1d1d1f">{_info_html}</div>'
                        f'<div style="font-size:.80rem;line-height:1.9;margin-top:2px;color:#7a7a7a">{_atr_html}</div>'
                        f'{_trigger_html}{_status_html}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    # 분석 전 기본 카드 (분석 버튼 클릭 전)
                    _pnl_pc  = (_cur / _avg - 1) * 100 if (_cur and _avg) else 0.0
                    _pnl_clr = "#34c759" if _pnl_pc >= 0 else "#ff3b30"
                    st.markdown(
                        f'<div style="background:#ffffff;border:1px solid #e0e0e0;border-radius:10px;padding:12px 16px;margin:6px 0">'
                        f'<div style="margin-bottom:4px">'
                        + (f'<span style="font-size:.95rem;font-weight:600;color:#1d1d1f">{_enm}</span> ' if _enm else "")
                        + f'<span style="font-size:.78rem;color:#7a7a7a">{_t}</span></div>'
                        f'<div style="font-size:.84rem;color:#7a7a7a;font-variant-numeric:tabular-nums;">'
                        f'{_cur_label} <b style="color:#1d1d1f">{_efmt(_cur)}</b> '
                        f'<span style="color:{_pnl_clr}">({_pnl_pc:+.1f}%)</span>'
                        f' &nbsp;|&nbsp; 평단가 <b style="color:#7a7a7a">{_efmt(_avg)}</b>'
                        f'</div></div>',
                        unsafe_allow_html=True,
                    )


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
        _raw_result = st.session_state.get("pf_opt_result")
        _opt_result: dict = _raw_result if isinstance(_raw_result, dict) else {}
        # 구버전 캐시 무효화 (momentum_data 키 미존재 시)
        if _opt_result and "momentum_data" not in _opt_result:
            _opt_result = {}
            st.session_state.pop("pf_opt_result", None)
        if _ai_btn.button("섹터 분석", key="pf_opt_run", type="primary", use_container_width=True):
            from src.portfolio_optimizer import classify_sectors, scan_market_momentum, build_rebalancing_guide
            with st.spinner("섹터 분석 및 시장 모멘텀 스캔 중..."):
                _sd  = classify_sectors(_items, _pf_prices)
                _mm  = scan_market_momentum()
                _opt_result = {
                    "sector_data":   _sd,
                    "momentum_data": _mm,
                    "guide": build_rebalancing_guide(_sd, _mm, _pf_nm),
                }
            st.session_state["pf_opt_result"] = _opt_result

        if not _opt_result:
            st.caption("'섹터 분석' 버튼으로 포트폴리오 섹터 편중도와 리밸런싱 제안을 확인하세요.")
        else:
            _sd_r  = _opt_result["sector_data"]
            _guide = _opt_result["guide"]
            _mm_r  = _opt_result.get("momentum_data", {})
            _sctrs = _sd_r.get("sectors", {})

            # ── HHI 지수 + 시장 추세 ─────────────────────────────────────
            _hhi        = _guide.get("hhi", 0)
            _is_conc    = _guide.get("is_concentrated", False)
            _kospi_up   = _mm_r.get("kospi_above_ma", True)
            _kosdaq_up  = _mm_r.get("kosdaq_above_ma", True)
            _mkt_status = _guide.get("market_status", "상승장")
            _hhi_color  = "#ff3b30" if _hhi > 2500 else ("#ff9500" if _hhi > 1500 else "#34c759")
            _hhi_label  = "과집중" if _hhi > 2500 else ("중간" if _hhi > 1500 else "양호")
            _hhi_sub    = "섹터 집중 과도 — 분산 권고" if _is_conc else "균형 잡힌 분산 상태"
            _mkt_color  = "#34c759" if _kospi_up else "#ff3b30"
            _mkt_icon   = "📈" if _kospi_up else "📉"
            st.markdown(
                f'<div style="display:flex;gap:12px;margin-bottom:14px;flex-wrap:wrap">'
                f'<div style="background:#ffffff;border:1px solid #e0e0e0;border-radius:10px;padding:12px 18px;flex:1;min-width:160px;box-shadow:0 1px 2px rgba(0,0,0,0.06)">'
                f'<div style="font-size:.75rem;color:#7a7a7a;margin-bottom:4px">HHI 편중도 지수</div>'
                f'<div style="font-size:1.5rem;font-weight:600;color:{_hhi_color};font-variant-numeric:tabular-nums;">{_hhi:,.0f}</div>'
                f'<div style="font-size:.76rem;color:{_hhi_color}">{_hhi_label}</div>'
                f'<div style="font-size:.72rem;color:#b0b0b0;margin-top:2px">{_hhi_sub}</div>'
                f'</div>'
                f'<div style="background:#ffffff;border:1px solid #e0e0e0;border-radius:10px;padding:12px 18px;flex:1;min-width:160px;box-shadow:0 1px 2px rgba(0,0,0,0.06)">'
                f'<div style="font-size:.75rem;color:#7a7a7a;margin-bottom:4px">시장 추세 (20일 MA)</div>'
                f'<div style="font-size:1.2rem;font-weight:600;color:{_mkt_color}">{_mkt_icon} {_mkt_status}</div>'
                f'<div style="font-size:.76rem;color:#7a7a7a;margin-top:4px">'
                f'KOSPI {"▲ MA 위" if _kospi_up else "▼ MA 아래"} &nbsp;·&nbsp; KOSDAQ {"▲ MA 위" if _kosdaq_up else "▼ MA 아래"}'
                f'</div></div></div>',
                unsafe_allow_html=True,
            )

            # ── 섹터 비중 바 차트 (TOP/BOTTOM 뱃지 포함) ─────────────────
            if _sctrs:
                st.markdown("**📊 포트폴리오 섹터 비중**")
                _sc_rank_map = {s["sector"]: s["rank"] for s in _guide.get("sector_scores", [])}
                _bar_sorted  = sorted(_sctrs.items(), key=lambda x: x[1]["weight"], reverse=True)
                _bar_max     = max(v["weight"] for _, v in _bar_sorted) or 1
                _bar_html    = '<div style="display:grid;gap:5px;margin-bottom:12px">'
                for _sn, _sv in _bar_sorted:
                    _sw    = _sv["weight"]
                    _rank  = _sc_rank_map.get(_sn, "")
                    _bc    = "#ff3b30" if _sw > 40 else ("#ff9500" if _sw > 30 else ("#ff9500" if _sw > 20 else "#0066cc"))
                    _bpct  = _sw / _bar_max * 100
                    _tks   = ", ".join(_pf_nm.get(t, t) or t for t in _sv["tickers"])
                    _snc   = "#34c759" if _rank == "TOP" else ("#ff3b30" if _rank == "BOTTOM" else "#7a7a7a")
                    _snlbl = f"▲{_sn}" if _rank == "TOP" else (f"▼{_sn}" if _rank == "BOTTOM" else _sn)
                    _bar_html += (
                        f'<div style="display:flex;align-items:center;gap:8px">'
                        f'<div style="width:90px;font-size:.8rem;color:{_snc};text-align:right;white-space:nowrap">{_snlbl}</div>'
                        f'<div style="flex:1;background:#e0e0e0;border-radius:4px;height:14px;overflow:hidden">'
                        f'<div style="width:{_bpct:.0f}%;height:100%;background:{_bc};border-radius:4px"></div></div>'
                        f'<div style="width:44px;font-size:.8rem;font-weight:600;color:{_bc};font-variant-numeric:tabular-nums;">{_sw:.1f}%</div>'
                        f'<div style="font-size:.75rem;color:#7a7a7a">{_tks}</div>'
                        f'</div>'
                    )
                _bar_html += "</div>"
                st.markdown(_bar_html, unsafe_allow_html=True)

            # ── 섹터 모멘텀 랭킹 (TOP3 / BOTTOM3) ───────────────────────
            _sc_list = _guide.get("sector_scores", [])
            if _sc_list:
                st.markdown("**⚡ 섹터 모멘텀 랭킹**")
                _top_secs = [s for s in _sc_list if s["rank"] == "TOP"]
                _btm_secs = [s for s in _sc_list if s["rank"] == "BOTTOM"]
                _rnk_html = '<div style="display:flex;gap:10px;margin-bottom:14px;flex-wrap:wrap">'
                # 주도 섹터
                _rnk_html += (
                    '<div style="flex:1;min-width:180px;background:#ffffff;border:1px solid #e0e0e0;border-radius:10px;padding:10px 14px">'
                    '<div style="font-size:.75rem;color:#34c759;font-weight:600;margin-bottom:6px">▲ 주도 섹터 TOP 3</div>'
                )
                for _s in _top_secs[:3]:
                    _5d_str = f'+{_s["return_5d"]:.1f}%' if _s["return_5d"] >= 0 else f'{_s["return_5d"]:.1f}%'
                    _rnk_html += (
                        f'<div style="font-size:.82rem;padding:4px 0;border-bottom:1px solid #e0e0e0;font-variant-numeric:tabular-nums;">'
                        f'<b style="color:#1d1d1f">{_s["sector"]}</b>'
                        f'<span style="color:#34c759;margin-left:6px">{_5d_str}</span>'
                        f'<span style="color:#b0b0b0;font-size:.7rem;margin-left:6px">· 20d {_s["return_20d"]:+.1f}% · 거래량 {_s["vol_growth"]:+.0f}%</span>'
                        f'<br><span style="font-size:.7rem;color:#7a7a7a">{_s["name"]} · 점수 {_s["score"]:.1f}</span>'
                        f'</div>'
                    )
                _rnk_html += '</div>'
                # 소외 섹터
                _rnk_html += (
                    '<div style="flex:1;min-width:180px;background:#ffffff;border:1px solid #e0e0e0;border-radius:10px;padding:10px 14px">'
                    '<div style="font-size:.75rem;color:#ff3b30;font-weight:600;margin-bottom:6px">▼ 소외 섹터 BTM 3</div>'
                )
                for _s in _btm_secs:
                    _5d_str = f'+{_s["return_5d"]:.1f}%' if _s["return_5d"] >= 0 else f'{_s["return_5d"]:.1f}%'
                    _clr    = "#ff3b30" if _s["return_5d"] < 0 else "#7a7a7a"
                    _rnk_html += (
                        f'<div style="font-size:.82rem;padding:4px 0;border-bottom:1px solid #e0e0e0;font-variant-numeric:tabular-nums;">'
                        f'<b style="color:#1d1d1f">{_s["sector"]}</b>'
                        f'<span style="color:{_clr};margin-left:6px">{_5d_str}</span>'
                        f'<span style="color:#b0b0b0;font-size:.7rem;margin-left:6px">· 20d {_s["return_20d"]:+.1f}% · 거래량 {_s["vol_growth"]:+.0f}%</span>'
                        f'<br><span style="font-size:.7rem;color:#7a7a7a">{_s["name"]} · 점수 {_s["score"]:.1f}</span>'
                        f'</div>'
                    )
                _rnk_html += '</div></div>'
                st.markdown(_rnk_html, unsafe_allow_html=True)

            # ── 조건 매트릭스 진단 카드 ──────────────────────────────────
            st.markdown("""<style>
.opt-card{background:#ffffff;border:1px solid #e0e0e0;border-radius:12px;padding:16px;margin-bottom:8px;box-shadow:0 1px 3px rgba(0,0,0,0.06)}
.opt-card-title{font-size:.8rem;font-weight:600;color:#7a7a7a;margin-bottom:10px;letter-spacing:0}
.opt-item{font-size:.85rem;line-height:1.7;padding:6px 10px;background:#f5f5f7;border-radius:8px;margin:4px 0;word-break:keep-all;color:#1d1d1f}
.opt-empty{color:#b0b0b0;font-size:.82rem;font-style:italic}
</style>""", unsafe_allow_html=True)

            _border_map = {"reduce": "#ff3b30", "hold": "#34c759", "watch": "#ff9500", "add": "#0066cc"}
            _recs       = _guide.get("recommendations", [])
            _miss       = _guide.get("missing_top", [])
            _pt         = _guide.get("profit_take", [])
            _gc1, _gc2  = st.columns(2)

            # 왼쪽: 조건 매트릭스 + 미보유 TOP 섹터
            _diag_html = ""
            for _r in _recs:
                _bc2 = _border_map.get(_r["type"], "#b0b0b0")
                _diag_html += (
                    f'<div class="opt-item" style="border-left:3px solid {_bc2}">'
                    f'<b style="color:#1d1d1f">{_r["icon"]} {_r["sector"]} {_r["weight"]:.1f}%</b>'
                    f'<br><span style="font-size:.78rem;color:#7a7a7a">{_r["tickers"]}</span>'
                    f'<br><span style="font-size:.74rem;color:#b0b0b0">{_r["message"]}</span>'
                    f'</div>'
                )
            for _m in _miss:
                _diag_html += (
                    f'<div class="opt-item" style="border-left:3px solid #0066cc">'
                    f'<b style="color:#1d1d1f">🔍 {_m["sector"]} 미보유</b>'
                    f'<br><span style="font-size:.78rem;color:#0066cc">{_m["name"]} · 5일 {_m["return_5d"]:+.1f}%</span>'
                    f'<br><span style="font-size:.74rem;color:#b0b0b0">포트폴리오에 없는 시장 주도 섹터 — 다변화 편입 검토</span>'
                    f'</div>'
                )
            if not _diag_html:
                _diag_html = '<span class="opt-empty">💤 현재 별도 조치 불필요 — 포트폴리오 안정적</span>'
            _gc1.markdown(
                f'<div class="opt-card"><div class="opt-card-title">🎯 포트폴리오 진단</div>{_diag_html}</div>',
                unsafe_allow_html=True,
            )

            # 오른쪽: 수익 확정 권고
            _pt_html = "".join(
                f'<div class="opt-item" style="border-left:3px solid #ff9500">'
                f'<b style="color:#1d1d1f">{p["name"]}</b>'
                f' <span style="color:#34c759;font-weight:600">+{p["pnl_pct"]:.1f}%</span>'
                f'<br><span style="font-size:.74rem;color:#7a7a7a">{p["reason"]}</span></div>'
                for p in _pt
            ) or '<span class="opt-empty">수익 확정 기준(+15%) 도달 종목 없음</span>'
            _gc2.markdown(
                f'<div class="opt-card"><div class="opt-card-title">💰 수익 확정 권고</div>{_pt_html}</div>',
                unsafe_allow_html=True,
            )

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
            "투자 예정 금액 (원)", min_value=0, max_value=1_000_000_000,
            value=0, step=500_000, format="%d", key="rec_investment_amount",
        )
        if _inv_amt > 0:
            _rec_c1.caption(f"₩ {int(_inv_amt):,}")
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
.rec-card{background:#ffffff;border:1px solid #e0e0e0;border-radius:12px;padding:16px 18px;margin-bottom:8px;border-top:3px solid #0066cc;box-shadow:0 1px 3px rgba(0,0,0,0.06)}
.rec-card-name{font-size:1rem;font-weight:600;color:#1d1d1f}
.rec-card-sector{font-size:.72rem;background:#f5f5f7;color:#7a7a7a;padding:2px 8px;border-radius:9999px;margin-left:6px;border:1px solid #e0e0e0}
.rec-card-reason{font-size:.8rem;color:#0066cc;font-style:italic;margin-top:8px;line-height:1.5;
             border-left:2px solid #0066cc;padding-left:8px}
.rec-bar-bg{background:#e0e0e0;border-radius:4px;height:6px;margin:6px 0}
.rec-bar-fg{height:6px;border-radius:4px;background:#0066cc}
</style>""", unsafe_allow_html=True)

            def _render_rec_card(r, col):
                _sent_pct  = round(r.sentiment_score * 100, 1)
                _wt_pct    = round(r.weight * 100, 1)
                _rsi_clr   = "#34c759" if r.rsi < 50 else ("#ff9500" if r.rsi < 70 else "#ff3b30")
                _cur_str   = getattr(r, "currency", "KRW")
                _mkt       = getattr(r, "market", "KOSPI")
                _native_px = getattr(r, "current_price", 0)
                _price_str = f"₩{_native_px:,.0f}" if _cur_str == "KRW" else f"${_native_px:,.2f}"
                _mkt_clr   = "#0066cc" if _mkt == "KOSPI" else ("#34c759" if _mkt == "KOSDAQ" else "#ff9500")
                col.markdown(
                    f'<div class="rec-card">'
                    f'<div style="margin-bottom:6px">'
                    f'<span class="rec-card-name">{r.name}</span>'
                    f'<span class="rec-card-sector">{r.sector}</span>'
                    f'<span style="font-size:.68rem;background:{_mkt_clr}22;color:{_mkt_clr};padding:1px 7px;border-radius:8px;margin-left:4px;font-weight:600">{_mkt}</span>'
                    f'</div>'
                    f'<div style="font-size:.85rem;color:#7a7a7a;margin-top:4px">현재가 <b style="color:#1d1d1f">{_price_str}</b>'
                    + (f' <span style="font-size:.75rem;color:#777">(₩{getattr(r,"current_price_krw",_native_px):,.0f})</span>' if _cur_str == "USD" else "")
                    + f'</div>'
                    f'<div style="margin-top:10px;display:flex;gap:10px;flex-wrap:wrap">'
                    f'<span style="font-size:.82rem;color:#7a7a7a">비중</span> <span style="font-size:.9rem;font-weight:600;color:#0066cc;font-variant-numeric:tabular-nums;">{_wt_pct:.1f}%</span>'
                    f'&nbsp;·&nbsp;<span style="font-size:.82rem;color:#7a7a7a">수량</span> <span style="font-size:.9rem;font-weight:600;color:#1d1d1f;font-variant-numeric:tabular-nums;">{r.quantity:,}주</span>'
                    f'&nbsp;·&nbsp;<span style="font-size:.82rem;color:#7a7a7a">투자액</span> <span style="font-size:.9rem;font-weight:600;color:#34c759;font-variant-numeric:tabular-nums;">₩{r.invested:,.0f}</span>'
                    f'</div>'
                    f'<div style="margin-top:8px;display:flex;gap:16px">'
                    f'<span style="font-size:.78rem;color:#7a7a7a">뉴스감성 <b style="color:#0066cc">{_sent_pct:.0f}</b>/100</span>'
                    f'<span style="font-size:.78rem;color:#7a7a7a">RSI <b style="color:{_rsi_clr}">{r.rsi:.0f}</b></span>'
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
                                f'<div style="background:#ffffff;border:1px solid #e0e0e0;border-radius:8px;padding:10px 14px;margin:5px 0;border-left:3px solid #ff9500">'
                                f'<span style="font-size:.9rem;font-weight:600;color:#1d1d1f">🟡 {_mn}</span>'
                                f'<span style="font-size:.75rem;color:#7a7a7a;margin-left:8px">{_mt}</span>'
                                f'<div style="font-size:.82rem;color:#7a7a7a;margin-top:5px;line-height:1.5">'
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
                        f'<div style="background:#f5f5f7;border:1px solid #e0e0e0;border-radius:8px;padding:10px 14px;margin:4px 0;font-size:.85rem">'
                        f'<span style="color:#7a7a7a">{_h_dt}</span>&nbsp;|&nbsp;'
                        f'<b style="color:#1d1d1f">₩{_h.get("investment_amt",0):,}</b>&nbsp;·&nbsp;'
                        f'<span style="color:#0066cc">{_h.get("risk_profile","중립형")}</span>'
                        f'<div style="color:#7a7a7a;margin-top:4px">{_h_names}</div>'
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
                    f'<span style="color:#7a7a7a">총 {_th_total_cnt}건</span>'
                    f'<span style="color:#34c759">수익 {_th_profit_cnt}건</span>'
                    f'<span style="color:#ff3b30">손실 {_th_total_cnt - _th_profit_cnt}건</span>'
                    f'<span style="color:#7a7a7a">누적 실현 손익 '
                    f'<b style="color:{"#34c759" if _cum_profit_krw >= 0 else "#ff3b30"}">'
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
    get_exchange_rates_fn=None,
    set_cookie_fn=None,
    delete_cookie_fn=None,
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
                        if set_cookie_fn:
                            set_cookie_fn("auth_token", _r["token"])
                        st.rerun()
                    else:
                        st.error(_r["error"])
            return

        # 환율 사전 계산 — @st.cache_data 함수를 fragment에 직접 넘기면 해시 실패
        _pf_usd_krw = 1300.0
        if get_exchange_rates_fn is not None:
            try:
                _ex = get_exchange_rates_fn()
                for _pk, _pi in _ex.items():
                    if "USD" in _pk and "KRW" in _pk:
                        _r = float(_pi.get("rate", 0.0))
                        if _r > 100:
                            _pf_usd_krw = _r
                        break
            except Exception:
                pass

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
            usd_krw=_pf_usd_krw,
            set_cookie_fn=set_cookie_fn,
            delete_cookie_fn=delete_cookie_fn,
        )
