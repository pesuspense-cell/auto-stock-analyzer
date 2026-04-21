"""
app.py - AI 주식 분석 대시보드 v2.0
실시간 차트 · 매매 신호 · 시장 현황 · 추천 종목 · 펀더멘털 · 관심종목
"""
import json
import os
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False

from stock_ai import (
    get_stock_data, generate_signals, calculate_expected_return,
    get_market_movers, get_exchange_rates, get_recommendations,
    get_fundamental_data, calculate_fundamental_score,
    get_stop_loss_targets, get_insider_trades_sec,
    get_krx_stock_list, get_us_stock_list,
    get_top_kospi_stocks, get_top_kosdaq_stocks, get_top_us_stocks,
    get_top_nasdaq_stocks,
    get_naver_news, analyze_news_sentiment_keywords,
    analyze_news_sentiment_llm, summarize_article_llm,
    get_hybrid_signal,
    get_advanced_sentiment, get_related_sector_performance,
    KOSPI_STOCKS, US_STOCKS, INDICES,
)

# ─── 페이지 설정 ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI 주식 분석 터미널",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 1rem; }
    div[data-testid="metric-container"] { background:#1e2130; border-radius:8px; padding:8px 12px; }
    .signal-box { padding:12px 8px; border-radius:12px; text-align:center; margin-bottom:12px; word-break:keep-all; }
    .wl-item { background:#1e2130; border-radius:8px; padding:8px 12px; margin:4px 0;
               display:flex; justify-content:space-between; align-items:center; }
</style>
""", unsafe_allow_html=True)

# ─── 비밀번호 게이트 ──────────────────────────────────────────────────────────
_APP_PASSWORD = "qnwkehlwk"

if not st.session_state.get("app_authenticated"):
    st.markdown("<br>" * 4, unsafe_allow_html=True)
    _, col, _ = st.columns([1.5, 1, 1.5])
    with col:
        st.markdown("## 📈 AI 주식 분석 터미널")
        st.markdown("---")
        pw_input = st.text_input(
            "비밀번호",
            type="password",
            placeholder="비밀번호를 입력하세요",
            label_visibility="collapsed",
        )
        if st.button("입장하기", use_container_width=True, type="primary"):
            if pw_input == _APP_PASSWORD:
                st.session_state["app_authenticated"] = True
                st.rerun()
            else:
                st.error("비밀번호가 틀렸습니다.")
    st.stop()

# ─── 관심종목 관리 ────────────────────────────────────────────────────────────
WATCHLIST_FILE = os.path.join(os.path.dirname(__file__), "watchlist.json")
SETTINGS_FILE  = os.path.join(os.path.dirname(__file__), "settings.json")

def load_settings() -> dict:
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def save_settings(data: dict) -> None:
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def load_watchlist() -> list:
    try:
        if os.path.exists(WATCHLIST_FILE):
            with open(WATCHLIST_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return []

def save_watchlist(wl: list) -> None:
    try:
        with open(WATCHLIST_FILE, "w", encoding="utf-8") as f:
            json.dump(wl, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

if "watchlist" not in st.session_state:
    st.session_state.watchlist = load_watchlist()

_saved_settings = load_settings()
if "dart_api_key" not in st.session_state:
    st.session_state["dart_api_key"] = _saved_settings.get("dart_api_key", "")

# KRX 로그인 자격증명을 환경변수로 자동 설정 (pykrx 인증용)
if _saved_settings.get("krx_id") and not os.environ.get("KRX_ID"):
    os.environ["KRX_ID"] = _saved_settings["krx_id"]
if _saved_settings.get("krx_pw") and not os.environ.get("KRX_PW"):
    os.environ["KRX_PW"] = _saved_settings["krx_pw"]

# ─── 캐시 래퍼 ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def _stock_data(ticker, period):
    return get_stock_data(ticker, period)

@st.cache_data(ttl=300)
def _movers(n: int = 100):
    """시가총액 상위 n개 KOSPI 종목 등락률"""
    stocks = _top_kospi(n)
    return get_market_movers(stocks)

@st.cache_data(ttl=300)
def _rates():
    return get_exchange_rates()

@st.cache_data(ttl=600)
def _recs(market: str, n: int = 100):
    """시장별 상위 n개 종목 추천 분석"""
    if market == "KOSPI":
        stocks = _top_kospi(n)
    elif market == "KOSDAQ":
        stocks = _top_kosdaq(n)
    elif market == "미국 주식 (나스닥)":
        stocks = _top_nasdaq(n)
    else:
        stocks = _top_us(n)
    return get_recommendations(stocks)

@st.cache_data(ttl=300)
def _index_data(sym):
    d = yf.download(sym, period="2d", auto_adjust=True, progress=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.droplevel(1)
    return d

@st.cache_data(ttl=600)
def _fundamental(ticker):
    info = get_fundamental_data(ticker)
    return info

@st.cache_data(ttl=3600)
def _insider_trades(ticker):
    return get_insider_trades_sec(ticker)

@st.cache_data(ttl=3600)
def _naver_news(ticker: str) -> list:
    return get_naver_news(ticker, max_items=10)

@st.cache_data(ttl=300)
def _sector_perf(ticker: str) -> dict:
    return get_related_sector_performance(ticker)

@st.cache_data(ttl=3600)
def _news_sentiment_kw(ticker: str) -> dict:
    news = _naver_news(ticker)
    if not news:
        # US 주식이거나 네이버 실패 시 yfinance 뉴스로 키워드 분석
        try:
            raw = yf.Ticker(ticker).news or []
            items = []
            for it in raw[:10]:
                c = it.get("content", it)
                items.append({
                    "title": c.get("title", it.get("title", "")),
                    "link":  (c.get("canonicalUrl", {}).get("url") or it.get("link", "#")),
                    "publisher": (c.get("provider", {}).get("displayName") or it.get("publisher", "")),
                    "pub_date": "",
                })
            news = items
        except Exception:
            news = []
    return analyze_news_sentiment_keywords(news)

def _news_sentiment_llm_cached(ticker: str, api_key: str) -> dict:
    """LLM 분석은 API 키가 세션마다 다를 수 있어 session_state 캐시 사용"""
    cache_key = f"llm_news|{ticker}|{api_key[:8] if api_key else ''}"
    cached = st.session_state.get(cache_key)
    if cached and (datetime.now() - cached["ts"]).seconds < 3600:
        return cached["data"]
    news = _naver_news(ticker)
    if not news:
        try:
            raw = yf.Ticker(ticker).news or []
            items = []
            for it in raw[:10]:
                c = it.get("content", it)
                items.append({
                    "title": c.get("title", it.get("title", "")),
                    "link":  (c.get("canonicalUrl", {}).get("url") or it.get("link", "#")),
                    "publisher": (c.get("provider", {}).get("displayName") or it.get("publisher", "")),
                    "pub_date": "",
                })
            news = items
        except Exception:
            news = []
    result = analyze_news_sentiment_llm(news, ticker, api_key)
    st.session_state[cache_key] = {"data": result, "ts": datetime.now()}
    return result

@st.cache_data(ttl=86400)
def _krx_stocks():
    return get_krx_stock_list()

@st.cache_data(ttl=86400)
def _top_kospi(n: int = 500):
    return get_top_kospi_stocks(n)

@st.cache_data(ttl=86400)
def _top_kosdaq(n: int = 500):
    return get_top_kosdaq_stocks(n)

@st.cache_data(ttl=86400)
def _top_us(n: int = 503):
    return get_top_us_stocks(n)

@st.cache_data(ttl=86400)
def _top_nasdaq(n: int = 500):
    return get_top_nasdaq_stocks(n)

@st.cache_data(ttl=86400)
def _us_stocks():
    return get_us_stock_list()

# ─── 관심종목 알림 체크 ───────────────────────────────────────────────────────
def _get_wl_alerts() -> list:
    """
    관심종목 매매 신호 체크.
    결과는 session_state에 5분 캐시 (yfinance 부하 최소화).
    """
    wl = st.session_state.get("watchlist", [])
    if not wl:
        return []

    cache_key = "wl_alerts|" + "|".join(sorted(w["ticker"] for w in wl))
    cached = st.session_state.get(cache_key)
    # 5분 이내 캐시 재사용
    if cached and (datetime.now() - cached["ts"]).seconds < 300:
        return cached["data"]

    alerts = []
    for item in wl:
        try:
            data = _stock_data(item["ticker"], "3mo")
            if data.empty:
                continue
            sig   = generate_signals(data)
            score = sig.get("score", 0)
            if abs(score) >= 3:
                price = float(data["Close"].iloc[-1])
                prev  = float(data["Close"].iloc[-2])
                alerts.append({
                    "name":   item["name"],
                    "ticker": item["ticker"],
                    "score":  score,
                    "label":  sig.get("label", ""),
                    "badge":  sig.get("badge", ""),
                    "price":  price,
                    "chg":    (price - prev) / prev * 100,
                })
        except Exception:
            continue

    st.session_state[cache_key] = {"data": alerts, "ts": datetime.now()}
    return alerts

# ─── 사이드바 ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ 종목 설정")
    market_sel = st.selectbox("시장", ["국내 주식 (검색)", "미국 주식 (검색)", "직접 입력"])

    if market_sel == "국내 주식 (검색)":
        with st.spinner("종목 목록 로딩 중..."):
            krx = _krx_stocks()

        if krx:
            options = list(krx.keys())
            selected = st.selectbox(
                "종목 검색 (이름·코드 입력)",
                options,
                help="회사 이름이나 종목 코드(6자리)를 입력하면 자동으로 필터링됩니다.",
            )
            ticker = krx[selected]
            sname  = selected.split(" (")[0]
        else:
            st.warning("종목 목록 로드 실패 — 기본 목록 사용")
            sname  = st.selectbox("종목", list(KOSPI_STOCKS.keys()))
            ticker = KOSPI_STOCKS[sname]

    elif market_sel == "미국 주식 (검색)":
        with st.spinner("미국 종목 목록 로딩 중... (S&P500 + 나스닥)"):
            us_list = _us_stocks()

        if us_list:
            us_options = list(us_list.keys())
            us_selected = st.selectbox(
                "종목 검색 (이름·티커 입력)",
                us_options,
                help="S&P500 + 나스닥 전체 종목. 회사명 또는 티커(예: AAPL)를 입력하면 자동 필터링됩니다.",
            )
            ticker = us_list[us_selected]
            sname  = us_selected.split(" (")[0]
        else:
            st.warning("미국 종목 목록 로드 실패 — 기본 목록 사용")
            sname  = st.selectbox("종목", list(US_STOCKS.keys()))
            ticker = US_STOCKS[sname]

    else:
        ticker = st.text_input("티커 직접 입력", value="005930.KS",
                               help="예) 005930.KS (KOSPI), 247540.KQ (KOSDAQ), AAPL (미국)")
        sname  = ticker

    period = st.selectbox("분석 기간", ["1mo", "3mo", "6mo", "1y", "2y"], index=1)

    # ── 분석 시작 버튼 ─────────────────────────────────────────────────────
    st.divider()
    if st.button("🔍 종목 분석 시작", type="primary", use_container_width=True):
        st.session_state["analyzed_ticker"] = ticker
        st.session_state["analyzed_sname"]  = sname
        st.session_state["analyzed_period"] = period

    # ── Gemini API 키 ──────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 🤖 AI 뉴스 분석 (Gemini)")

    # Streamlit Cloud secrets → 세션 → 직접 입력 순서로 우선순위 적용
    try:
        _secret_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        _secret_key = ""
    _default_key = _secret_key or st.session_state.get("gemini_api_key", "")

    if _secret_key:
        # secrets.toml / Cloud 대시보드에 키가 있으면 UI 노출 없이 자동 적용
        gemini_api_key = _secret_key
        st.caption("🟢 AI 분석 활성 (secrets 자동 적용)")
    else:
        gemini_api_key = st.text_input(
            "Gemini API Key",
            value=_default_key,
            type="password",
            placeholder="AIza...",
            help=(
                "Google AI Studio (aistudio.google.com)에서 발급받은 Gemini API 키.\n\n"
                "• 입력 시: LangChain + Gemini로 뉴스 감성 AI 분석\n"
                "• 미입력 시: 키워드 기반 감성 분석(무료)으로 자동 전환\n"
                "• 배포 시에는 .streamlit/secrets.toml 또는 Streamlit Cloud\n"
                "  대시보드 Secrets에 GEMINI_API_KEY 를 등록하면 UI 없이 자동 적용됩니다."
            ),
        )
        if gemini_api_key:
            st.session_state["gemini_api_key"] = gemini_api_key
        st.caption("🟢 AI 분석 활성" if gemini_api_key else "⚪ 키워드 분석 모드 (API 키 없음)")

    use_llm = bool(gemini_api_key)

    # ── DART API 키 ────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 📑 DART 재무정보 (선택)")
    try:
        _dart_secret = st.secrets.get("DART_API_KEY", "")
    except Exception:
        _dart_secret = ""
    if _dart_secret:
        dart_api_key = _dart_secret
        st.caption("🟢 DART 연동 활성 (secrets 자동 적용)")
    else:
        dart_api_key = st.text_input(
            "DART API Key",
            value=st.session_state.get("dart_api_key", ""),
            type="password",
            placeholder="발급키 입력...",
            help="opendart.fss.or.kr 에서 무료 발급. 입력 시 매출액·영업이익·순이익 등 KRX 재무제표 조회.",
        )
        if dart_api_key and dart_api_key != st.session_state.get("dart_api_key", ""):
            st.session_state["dart_api_key"] = dart_api_key
            save_settings({**load_settings(), "dart_api_key": dart_api_key})
        st.caption("🟢 DART 활성" if dart_api_key else "⚪ DART 미연동 (yfinance 데이터 사용)")

    # ── KRX API 키 ─────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 📊 KRX 데이터")
    _krx_ok = bool(os.environ.get("KRX_ID") and os.environ.get("KRX_PW"))
    st.caption("🟢 KRX 자동 로그인 활성" if _krx_ok else "⚪ KRX 인증 미설정")

    # ── 자동 새로고침 ──────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 🔄 실시간 새로고침")
    auto_refresh = st.toggle("자동 새로고침 활성화", value=False)
    if auto_refresh:
        interval_opt = st.selectbox("새로고침 주기", ["30초", "1분", "5분"], index=1)
        interval_ms  = {"30초": 30_000, "1분": 60_000, "5분": 300_000}[interval_opt]
        if HAS_AUTOREFRESH:
            refresh_count = st_autorefresh(interval=interval_ms, key="auto_refresh")
            st.caption(f"자동 새로고침 활성 | 주기: {interval_opt}")
        else:
            st.warning("streamlit-autorefresh 미설치")
    else:
        if st.button("🔄 수동 새로고침"):
            st.cache_data.clear()
            st.rerun()

    # ── 환율 ───────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 💱 실시간 환율")
    with st.spinner("환율 로딩 중..."):
        rates = _rates()
    for pair, info in rates.items():
        st.metric(pair, f"{info['rate']:,.2f}", f"{info['change']:+.3f}%",
                  help=f"{pair} 환율. 전일 대비 변동률 표시. 양수(🔺)면 원화 대비 해당 통화 강세(원화 약세)를 의미합니다.")

    # ── 주요 지수 ──────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 📊 주요 지수")
    for idx_name, idx_sym in INDICES.items():
        try:
            d = _index_data(idx_sym)
            if len(d) >= 2:
                p   = float(d["Close"].iloc[-1])
                chg = (p - float(d["Close"].iloc[-2])) / float(d["Close"].iloc[-2]) * 100
                st.metric(idx_name, f"{p:,.2f}", f"{chg:+.2f}%",
                          help=f"{idx_name} 지수. 전일 종가 대비 등락률. 🔺 상승 / 🔻 하락.")
        except Exception:
            pass

    # ── 관심종목 ───────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### ⭐ 관심 종목")
    wl = st.session_state.watchlist
    if not wl:
        st.caption("아직 추가된 관심종목이 없습니다.")
    else:
        # 알림 배너 (신호 강도 ≥ 3인 종목)
        wl_alerts = _get_wl_alerts()
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
                    d2 = yf.download(item["ticker"], period="2d", auto_adjust=True, progress=False)
                    if isinstance(d2.columns, pd.MultiIndex):
                        d2.columns = d2.columns.droplevel(1)
                    if len(d2) >= 2:
                        p2  = float(d2["Close"].iloc[-1])
                        chg2= (p2 - float(d2["Close"].iloc[-2])) / float(d2["Close"].iloc[-2]) * 100
                        arrow = "🔺" if chg2 >= 0 else "🔻"
                        color = "#ef5350" if chg2 >= 0 else "#42a5f5"
                        st.markdown(
                            f"**{item['name']}**<br>"
                            f"<span style='color:{color}'>{arrow} {chg2:+.2f}%</span>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.caption(item["name"])
                except Exception:
                    st.caption(item["name"])
            with col_b:
                if st.button("✕", key=f"rm_{i}", help="관심종목 삭제"):
                    st.session_state.watchlist.pop(i)
                    save_watchlist(st.session_state.watchlist)
                    st.rerun()

# ─── 헤더 ────────────────────────────────────────────────────────────────────
st.markdown("# 📈 AI 주식 분석 대시보드")

# ─── 데이터 로드 (분석 시작 버튼 클릭 후에만 실행) ────────────────────────────
_aticker = st.session_state.get("analyzed_ticker")
_asname  = st.session_state.get("analyzed_sname", "")
_aperiod = st.session_state.get("analyzed_period", period)
_data_ready = bool(_aticker)

if _data_ready:
    st.caption(f"업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  분석 종목: **{_asname}** (`{_aticker}`)")

    with st.spinner(f"{_aticker} 데이터 분석 중..."):
        data = _stock_data(_aticker, _aperiod)

    if data.empty:
        st.error(f"'{_aticker}' 데이터를 불러올 수 없습니다. 티커를 확인해주세요.")
        st.session_state.pop("analyzed_ticker", None)
        st.stop()

    signals  = generate_signals(data)
    expected = calculate_expected_return(data, signals)
    close    = data["Close"]

    with st.spinner("펀더멘털 데이터 로딩 중..."):
        fund_info       = _fundamental(_aticker)
        fund_score_data = calculate_fundamental_score(fund_info, float(close.iloc[-1]))
else:
    st.caption(f"업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  사이드바에서 종목을 선택하고 분석을 시작하세요.")
    # 탭 렌더링 시 크래시 방지용 stub 변수
    data            = pd.DataFrame()
    signals         = {"score": 0, "label": "분석 대기", "badge": "—"}
    expected        = None
    close           = pd.Series(dtype=float, name="Close")
    fund_info       = {}
    fund_score_data = {"fund_score": 0, "fund_label": "분석 대기", "fund_reasons": []}

# ─── 관심종목 Toast 알림 (우측 하단 팝업) ────────────────────────────────────
if st.session_state.watchlist:
    _alerts = _get_wl_alerts()
    # 이번 세션에서 이미 띄운 알림은 제외 (중복 방지)
    shown_key = "toast_shown|" + "|".join(sorted(w["ticker"] for w in st.session_state.watchlist))
    already_shown = st.session_state.get(shown_key, set())
    new_shown = set(already_shown)
    for a in _alerts:
        uid = f"{a['ticker']}_{a['score']}"
        if uid not in already_shown:
            if a["score"] >= 4:
                st.toast(
                    f"🟢 **{a['name']}** 강력 매수!\n"
                    f"신호: {a['label']} ({a['score']:+.1f}점)  |  {a['price']:,.0f}  {a['chg']:+.2f}%",
                    icon="🔺",
                )
            elif a["score"] <= -4:
                st.toast(
                    f"🔴 **{a['name']}** 강력 매도!\n"
                    f"신호: {a['label']} ({a['score']:+.1f}점)  |  {a['price']:,.0f}  {a['chg']:+.2f}%",
                    icon="🔻",
                )
            elif a["score"] >= 3:
                st.toast(
                    f"🟡 **{a['name']}** 매수 신호\n"
                    f"신호: {a['label']} ({a['score']:+.1f}점)  |  {a['price']:,.0f}",
                    icon="🔔",
                )
            elif a["score"] <= -3:
                st.toast(
                    f"🟠 **{a['name']}** 매도 신호\n"
                    f"신호: {a['label']} ({a['score']:+.1f}점)  |  {a['price']:,.0f}",
                    icon="🔔",
                )
            new_shown.add(uid)
    st.session_state[shown_key] = new_shown

# ─── 기사 AI 요약 모달 ───────────────────────────────────────────────────────
@st.dialog("📰 기사 AI 요약", width="large")
def _article_dialog(title: str, link: str, ticker_sym: str, api_key: str) -> None:
    """뉴스 기사 클릭 시 팝업되는 AI 요약 모달"""
    # 제목 + 원문 링크
    st.markdown(f"### {title}")
    if link and link != "#":
        st.markdown(f"[🔗 원문 기사 열기]({link})", unsafe_allow_html=False)
    st.divider()

    if not api_key:
        st.warning(
            "AI 요약을 사용하려면 사이드바에서 **Gemini API 키**를 입력하세요.  \n"
            "Google AI Studio(aistudio.google.com)에서 무료로 발급받을 수 있습니다."
        )
        return

    with st.spinner("AI가 기사를 분석 중입니다..."):
        result = summarize_article_llm(title, link, ticker_sym, api_key)

    # 감성 배지
    senti = result.get("sentiment", "N/A")
    score = result.get("score", 0.0)
    if senti == "긍정":
        badge_color, badge_bg = "#a5d6a7", "#1b5e20"
    elif senti == "부정":
        badge_color, badge_bg = "#ef9a9a", "#b71c1c"
    else:
        badge_color, badge_bg = "#bdbdbd", "#212121"

    st.markdown(
        f'<span style="background:{badge_bg};color:{badge_color};'
        f'border-radius:6px;padding:4px 14px;font-size:0.9rem;font-weight:bold;">'
        f'{senti} &nbsp; {score:+.2f}</span>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    # 핵심 요약
    summary = result.get("summary", "")
    if summary:
        st.markdown("**📋 핵심 요약**")
        st.info(summary)

    # 핵심 포인트
    key_points = result.get("key_points", [])
    if key_points:
        st.markdown("**🔑 핵심 포인트**")
        for pt in key_points:
            st.markdown(f"- {pt}")

    # 투자 시사점
    implication = result.get("investment_implication", "")
    if implication:
        st.markdown("**💡 투자 시사점**")
        st.success(implication)

    # 본문 미사용 안내
    if not result.get("used_content"):
        st.caption("⚠️ 기사 본문 스크래핑 불가 — 제목 기반으로 분석했습니다.")


# ─── 탭 레이아웃 ──────────────────────────────────────────────────────────────
tab_market, tab_chart, tab_rec, tab_news, tab_fund = st.tabs([
    "🌐 시장 현황",
    "📊 차트 분석",
    "⭐ 추천 종목",
    "📰 뉴스 & 관련 종목",
    "🏛️ 펀더멘털 & 기관",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1  차트 분석
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# TAB 2  시장 현황
# ══════════════════════════════════════════════════════════════════════════════
with tab_market:
    st.subheader("🌐 시장 현황")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        mover_n = st.select_slider(
            "분석 종목 수 (시가총액 상위)", list(range(10, 110, 10)), value=50,
            help="KOSPI 시가총액 상위 N개 종목의 등락률을 분석합니다. 많을수록 로딩이 느려집니다."
        )
        with st.spinner(f"KOSPI 상위 {mover_n}개 종목 로딩 중..."):
            movers = _movers(mover_n)

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
                    .map(lambda v: "color:#ef5350; font-weight:bold" if isinstance(v, float) and v > 0 else "", subset=["등락률(%)"]),
                    use_container_width=True, hide_index=True,
                )
            with c2:
                st.markdown("### 📉 급락 하위")
                st.dataframe(
                    losers[["종목명", "현재가", "등락률(%)"]].style
                    .format({"현재가": "{:,.0f}", "등락률(%)": "{:+.2f}%"})
                    .map(lambda v: "color:#42a5f5; font-weight:bold" if isinstance(v, float) and v < 0 else "", subset=["등락률(%)"]),
                    use_container_width=True, hide_index=True,
                )

            # 차트는 상위 급등 15 + 급락 15만 표시 (너무 많으면 가독성 저하)
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

            # 전체 테이블 (접기)
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
            arrow = "🔺" if info["change"] > 0 else "🔻"
            color = "#ef5350" if info["change"] > 0 else "#42a5f5"
            st.markdown(f"""
            <div style="background:#1e2130;padding:10px 14px;border-radius:8px;margin:6px 0;">
                <b>{pair}</b><br>
                <span style="font-size:1.4rem;font-weight:bold;">{info['rate']:,.2f}</span>
                <span style="color:{color};"> {arrow} {abs(info['change']):.3f}%</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### 📈 USD/KRW 추이 (3개월)")
        try:
            fx = yf.download("USDKRW=X", period="3mo", auto_adjust=True, progress=False)
            if isinstance(fx.columns, pd.MultiIndex):
                fx.columns = fx.columns.droplevel(1)
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

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3  추천 종목
# ══════════════════════════════════════════════════════════════════════════════
with tab_rec:
    st.subheader("⭐ AI 추천 종목 분석")

    col_a, col_b, col_c = st.columns([2, 2, 1])
    with col_a:
        rec_market = st.radio(
            "분석 시장",
            ["KOSPI", "KOSDAQ", "미국 주식 (S&P500)", "미국 주식 (나스닥)"],
            horizontal=True,
        )
    with col_b:
        rec_n = st.select_slider(
            "분석 종목 수",
            options=list(range(10, 510, 10)),
            value=50,
            help=(
                "시가총액 상위 N개 종목을 분석합니다.\n\n"
                "• 20~50개: 약 30~60초\n"
                "• 100개: 약 2~3분\n"
                "• 200개+: 5분 이상 소요될 수 있습니다."
            ),
        )
    with col_c:
        run_btn = st.button("🔄 분석 실행", type="primary", use_container_width=True)

    if run_btn:
        spinner_msg  = f"AI가 {rec_market} 상위 {rec_n}개 종목 분석 중... (종목 수에 따라 수 분 소요)"
        with st.spinner(spinner_msg):
            _recs.clear()
            rec_df = _recs(rec_market, rec_n)
            st.session_state["rec_df"]     = rec_df
            st.session_state["rec_market"] = rec_market
            st.session_state["rec_n"]      = rec_n

    rec_df = st.session_state.get("rec_df", None)

    if rec_df is not None and not rec_df.empty:
        # ── 요약 메트릭 (종합점수 기준) ─────────────────────────────────────
        rec_n     = int((rec_df["종합점수"] >= 1.5).sum())
        neutral_n = int(((rec_df["종합점수"] > 0) & (rec_df["종합점수"] < 1.5)).sum())
        caution_n = int((rec_df["종합점수"] <= 0).sum())
        pos_ret_n = int((rec_df["예상수익률(%)"] > 0).sum())

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("분석 종목", f"{len(rec_df)}개",
                  help="이번 분석에서 데이터를 불러온 종목 수입니다.")
        m2.metric("🟢 추천 (종합 +1.5↑)", f"{rec_n}개",
                  help="종합점수(기술50%+수익률30%+샤프20%) ≥ 1.5 종목 수.")
        m3.metric("⚪ 중립 (0~1.5)", f"{neutral_n}개",
                  help="종합점수 0 이상 1.5 미만 — 관망 구간.")
        m4.metric("📈 수익률 양수 종목", f"{pos_ret_n}개",
                  help="예상수익률(20일)이 플러스인 종목 수.")

        # ── 종합점수 산정 방식 안내 ──────────────────────────────────────────
        with st.expander("ℹ️ 종합점수 산정 방식", expanded=False):
            st.markdown("""
            **종합점수 = 기술점수 × 0.5 + 수익률점수 × 0.3 + 샤프점수 × 0.2**

            | 항목 | 범위 | 역할 |
            |------|------|------|
            | 기술점수 | -10 ~ +10 | RSI·MACD·EMA·ADX 등 8개 지표 복합 신호 |
            | 수익률점수 | -5 ~ +5 | 예상수익률(%)÷2 — **마이너스면 패널티** |
            | 샤프점수 | -3 ~ +3 | 샤프지수×1.5 — 리스크 대비 수익 효율 |

            > 예상수익률이 -10%이면 수익률점수 -5, 종합점수에서 -1.5점이 차감됩니다.
            > 기술점수가 높아도 수익률·샤프가 나쁘면 순위가 하락합니다.
            """)

        st.divider()
        st.markdown("### 📋 종목별 종합 분석표")

        def _row_style(row):
            s = row["종합점수"]
            if s >= 4.5: return ["background-color:#1b5e20"] * len(row)
            if s >= 3.0: return ["background-color:#2e7d32"] * len(row)
            if s >= 1.5: return ["background-color:#1a3a2a"] * len(row)
            if s <= -4.0: return ["background-color:#b71c1c"] * len(row)
            if s <= -2.0: return ["background-color:#c62828"] * len(row)
            return [""] * len(row)

        disp = ["종목명", "현재가", "등락률(1일)%", "종합추천", "종합점수",
                "기술점수", "예상수익률(%)", "변동성(%)", "모멘텀(20일)%", "샤프지수"]
        styled = (
            rec_df[disp].style
            .apply(_row_style, axis=1)
            .format({
                "현재가":         "{:,.0f}",
                "등락률(1일)%":   "{:+.2f}",
                "종합점수":       "{:+.2f}",
                "기술점수":       "{:+.1f}",
                "예상수익률(%)":  "{:+.2f}",
                "변동성(%)":      "{:.1f}",
                "모멘텀(20일)%":  "{:+.2f}",
                "샤프지수":       "{:.2f}",
            })
            .map(
                lambda v: "color:#ef9a9a" if isinstance(v, float) and v < 0 else
                          ("color:#a5d6a7" if isinstance(v, float) and v > 0 else ""),
                subset=["예상수익률(%)"],
            )
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # ── 리스크-리턴 산점도 (마커 색 = 종합점수) ──────────────────────────
        st.markdown("### 📊 리스크 - 리턴 매트릭스")
        st.caption("마커 색상: 🟢 종합점수 ≥ 3  /  🟡 0 ~ 3  /  🔴 < 0")
        fig_sc = go.Figure()
        for _, row in rec_df.iterrows():
            cs = row["종합점수"]
            if cs >= 3.0:
                color, size = "#66bb6a", 16
            elif cs >= 0:
                color, size = "#ffa726", 13
            else:
                color, size = "#ef5350", 13
            fig_sc.add_trace(go.Scatter(
                x=[row["변동성(%)"]],
                y=[row["예상수익률(%)"]],
                mode="markers+text",
                text=[row["종목명"]],
                textposition="top center",
                marker=dict(size=size, color=color, line=dict(width=1, color="#fff")),
                customdata=[[row["종합점수"], row["기술점수"], row["샤프지수"]]],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "변동성: %{x:.1f}%<br>"
                    "예상수익률: %{y:+.2f}%<br>"
                    "종합점수: %{customdata[0]:+.2f}<br>"
                    "기술점수: %{customdata[1]:+.1f}<br>"
                    "샤프지수: %{customdata[2]:.2f}<extra></extra>"
                ),
                showlegend=False,
            ))
        fig_sc.add_hline(y=0, line_color="rgba(255,255,255,0.25)", line_dash="dash")
        fig_sc.update_layout(
            height=420, template="plotly_dark",
            xaxis_title="변동성 (%)", yaxis_title="예상 수익률 (%)",
            margin=dict(t=10),
        )
        st.plotly_chart(fig_sc, use_container_width=True)

        # ── 종합점수 vs 기술점수 비교 막대 ──────────────────────────────────
        st.markdown("### 🏆 종합점수 순위 (기술점수 비교)")
        st.caption("회색 = 기술점수 / 색상 = 종합점수 — 두 점수 차이가 클수록 수익률·샤프 영향이 큰 종목")
        fig_rank = go.Figure()
        fig_rank.add_trace(go.Bar(
            name="기술점수",
            x=rec_df["종목명"],
            y=rec_df["기술점수"],
            marker_color="rgba(120,120,120,0.45)",
            text=[f"{v:+.1f}" for v in rec_df["기술점수"]],
            textposition="outside",
        ))
        fig_rank.add_trace(go.Bar(
            name="종합점수",
            x=rec_df["종목명"],
            y=rec_df["종합점수"],
            marker_color=["#66bb6a" if v >= 0 else "#ef5350" for v in rec_df["종합점수"]],
            text=[f"{v:+.2f}" for v in rec_df["종합점수"]],
            textposition="outside",
        ))
        fig_rank.update_layout(
            height=360, template="plotly_dark",
            barmode="overlay",
            yaxis_title="점수", margin=dict(t=10, b=60),
            xaxis_tickangle=-30,
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        )
        st.plotly_chart(fig_rank, use_container_width=True)

    else:
        st.info("'분석 실행' 버튼을 클릭하면 AI가 주요 종목을 분석합니다.")
        st.markdown("""
        #### 🔍 종합점수 구성
        **종합점수 = 기술점수(50%) + 수익률점수(30%) + 샤프점수(20%)**

        | 항목 | 기준값 | 비고 |
        |------|--------|------|
        | 기술점수 | RSI·MACD·EMA·ADX·OBV 등 8개 지표 | 범위 -10 ~ +10 |
        | 예상수익률 | +10% → +5점 / -10% → -5점 | 마이너스 시 패널티 |
        | 샤프지수 | 2.0 이상 → +3점 / 음수 → 감점 | 리스크 보정 |

        #### 🔍 기술 지표 법칙
        | 법칙 | 출처 | 매수 조건 | 매도 조건 |
        |------|------|-----------|-----------|
        | RSI  | 기술적 분석 | < 30 과매도 | > 70 과매수 |
        | MACD | 기술적 분석 | 골든크로스 | 데드크로스 |
        | EMA200 | 장기 추세 | 가격 상단 | 가격 하단 |
        | ADX  | 추세 강도 | > 25 추세 신호 신뢰 | < 15 횡보 약화 |
        | OBV + MFI | 수급 | 매집 추세 | 분산 추세 |
        """)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4  뉴스 & 관련 종목
# ══════════════════════════════════════════════════════════════════════════════
with tab_news:
    st.subheader(f"📰 {_asname or sname} 뉴스 & 관련 정보")

    is_kr_stock = ticker.endswith(".KS") or ticker.endswith(".KQ")

    # ── 뉴스 수집 (컬럼 바깥 — 심층 분석에서도 재사용) ──────────────────────
    raw_news: list = []
    if is_kr_stock:
        with st.spinner("네이버 금융에서 뉴스 수집 중..."):
            raw_news = _naver_news(ticker)
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

    # ── AI 감성 분석 (컬럼 바깥 — 두 섹션 공유) ──────────────────────────────
    sent: dict = {}
    if raw_news:
        with st.spinner("AI 감성 분석 중..."):
            if use_llm:
                sent = _news_sentiment_llm_cached(ticker, gemini_api_key)
            else:
                sent = _news_sentiment_kw(ticker)

    col_news, col_rel = st.columns([3, 2])

    with col_news:
        st.markdown(
            "### 최신 뉴스 (네이버 금융)" if is_kr_stock else "### 최신 뉴스"
        )

        if not raw_news:
            st.info("뉴스 데이터가 없습니다.")
        else:
            # 전체 감성 요약 배너
            s_score   = sent.get("score", 0.0)
            s_label   = sent.get("label", "중립")
            s_summary = sent.get("summary", "")
            if s_score >= 1:
                banner_color, text_color = "#1b5e20", "#a5d6a7"
            elif s_score <= -1:
                banner_color, text_color = "#b71c1c", "#ef9a9a"
            else:
                banner_color, text_color = "#1e2130", "#bdbdbd"

            st.markdown(f"""
            <div style="background:{banner_color};border-radius:10px;padding:12px 18px;margin-bottom:14px;">
                <div style="font-size:1.05rem;font-weight:bold;color:{text_color};">
                    뉴스 감성: {s_label} &nbsp;
                    <span style="font-size:0.95rem;font-weight:normal;">({s_score:+.1f}점 / ±5)</span>
                    {"&nbsp;|&nbsp; 🤖 Gemini AI 분석" if use_llm else "&nbsp;|&nbsp; 🔑 키워드 분석"}
                </div>
                {f'<div style="font-size:0.88rem;color:#ccc;margin-top:6px;">{s_summary}</div>' if s_summary else ""}
            </div>
            """, unsafe_allow_html=True)

            # 기사별 표시
            detail_map = {d["title"]: d for d in sent.get("detail", [])}
            for art_idx, item in enumerate(raw_news):
                title      = item.get("title", "제목 없음")
                link       = item.get("link", "#")
                publisher  = item.get("publisher", "")
                pub_date   = item.get("pub_date", "")
                d          = detail_map.get(title, {})
                art_score  = d.get("score", 0.0)
                art_reason = d.get("reason", "")

                if art_score >= 0.3:
                    dot_color, dot = "#66bb6a", "🟢"
                elif art_score <= -0.3:
                    dot_color, dot = "#ef5350", "🔴"
                else:
                    dot_color, dot = "#9e9e9e", "⚪"

                col_article, col_btn = st.columns([8, 1])

                with col_article:
                    st.markdown(
                        f'<div style="margin-bottom:2px;">'
                        f'<span style="color:{dot_color};font-size:0.8rem;">{dot} {art_score:+.1f}</span>'
                        f' &nbsp;<b><a href="{link}" target="_blank"'
                        f' style="color:#90caf9;text-decoration:none;">{title}</a></b>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    meta_parts = []
                    if publisher: meta_parts.append(f"📌 {publisher}")
                    if pub_date:  meta_parts.append(f"🕐 {pub_date}")
                    if meta_parts:
                        st.caption("  ·  ".join(meta_parts))
                    if art_reason:
                        st.caption(f"💡 {art_reason}")

                with col_btn:
                    if st.button(
                        "🤖 요약",
                        key=f"art_sum_{art_idx}",
                        help="AI가 이 기사를 요약합니다",
                        use_container_width=True,
                    ):
                        _article_dialog(title, link, ticker, gemini_api_key)

                st.divider()

    with col_rel:
        st.markdown("### 🔗 관련 종목 비교")
        pool = KOSPI_STOCKS if (ticker.endswith(".KS") or ticker.endswith(".KQ")) else US_STOCKS
        for name, sym in pool.items():
            try:
                d = yf.download(sym, period="2d", auto_adjust=True, progress=False)
                if isinstance(d.columns, pd.MultiIndex):
                    d.columns = d.columns.droplevel(1)
                if len(d) < 2:
                    continue
                p   = float(d["Close"].iloc[-1])
                chg = (p - float(d["Close"].iloc[-2])) / float(d["Close"].iloc[-2]) * 100
                arrow = "🔺" if chg >= 0 else "🔻"
                color = "#ef5350" if chg >= 0 else "#42a5f5"
                badge = " ◀ 현재 선택" if sym == ticker else ""
                bg    = "#1a3a2a" if sym == ticker else "#1e2130"
                st.markdown(f"""
                <div style="background:{bg};padding:8px 14px;border-radius:8px;margin:5px 0;">
                    <b>{name}</b><span style="color:#aaa;font-size:0.8rem;">{badge}</span><br>
                    <span style="font-size:1.15rem;font-weight:bold;">{p:,.0f}</span>
                    <span style="color:{color}"> {arrow} {abs(chg):.2f}%</span>
                </div>
                """, unsafe_allow_html=True)
            except Exception:
                pass

        st.markdown("### 🏢 기업 정보")
        try:
            info = yf.Ticker(ticker).info
            fields = [
                ("시가총액",  "marketCap",        lambda v: f"{v:,.0f}"),
                ("52주 최고", "fiftyTwoWeekHigh",  lambda v: f"{v:,.2f}"),
                ("52주 최저", "fiftyTwoWeekLow",   lambda v: f"{v:,.2f}"),
                ("PER",       "trailingPE",         lambda v: f"{v:.2f}"),
                ("PBR",       "priceToBook",        lambda v: f"{v:.2f}"),
                ("EPS",       "trailingEps",        lambda v: f"{v:,.2f}"),
                ("배당수익률","dividendYield",      lambda v: f"{v*100:.2f}%"),
            ]
            for lbl, key, fmt in fields:
                val = info.get(key)
                if val is not None and pd.notna(val):
                    try:
                        st.caption(f"**{lbl}:** {fmt(val)}")
                    except Exception:
                        pass
        except Exception:
            st.caption("기업 정보를 불러올 수 없습니다.")

    # ══ 심층 분석 섹션 (전체 너비) ══════════════════════════════════════════════
    if raw_news:
        st.markdown("---")
        st.markdown("### 🔍 심층 분석 — 키워드 강도 · 섹터 동조화")

        # 확장 키워드 감성
        adv = get_advanced_sentiment(raw_news)
        adv_score = adv.get("score", 0.0)

        # 섹터 동조화
        with st.spinner("섹터 동조화 분석 중..."):
            sec = _sector_perf(ticker)
        sector_avg = sec.get("avg_chg", 0.0)
        _daily_chg = (float(close.iloc[-1]) - float(close.iloc[-2])) / float(close.iloc[-2]) * 100 if len(close) >= 2 else 0.0
        stock_diff = _daily_chg - sector_avg

        # ── 3개 메트릭 ────────────────────────────────────────────────────────
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "📰 뉴스 키워드 강도",
            f"{adv_score:+.1f}점",
            adv.get("summary", ""),
            help=(
                "초강세(+2)/강세(+1)/약세(-1)/초악재(-2) 4단계 키워드 가중 합산 후 -5~+5 정규화.\n\n"
                "• +3 이상: 강한 호재 뉴스 다수\n"
                "• -3 이하: 강한 악재 뉴스 다수\n"
                "• 0 근처: 중립 또는 뉴스 없음"
            ),
        )
        c2.metric(
            "🏭 섹터 평균 등락",
            f"{sector_avg:+.2f}%",
            f"{'섹터 데이터 있음' if sec['has_data'] else '섹터 매핑 없음'}",
            help=(
                "동일 업종 연관 종목들의 당일 평균 등락률.\n\n"
                "• 섹터 전체가 오르는 날 이 종목도 오르면 동조화\n"
                "• 섹터는 빠지는데 이 종목만 오르면 개별 주도주"
            ),
        )
        diff_color = "normal" if abs(stock_diff) < 1 else ("inverse" if stock_diff > 0 else "off")
        c3.metric(
            "⚡ 섹터 대비 강도",
            f"{stock_diff:+.2f}%p",
            delta_color=diff_color,
            help=(
                "현재 종목 등락률 − 섹터 평균 등락률.\n\n"
                "• +양수: 섹터보다 강한 상승 → 주도주 신호\n"
                "• −음수: 섹터보다 약한 흐름 → 소외주 가능성\n"
                "• ±1%p 이내: 섹터와 동조화 중"
            ),
        )

        # ── 심층 판독 메시지 ──────────────────────────────────────────────────
        if adv_score > 0 and stock_diff > 1:
            st.success(
                f"🌟 **주도주 확인**: 호재 뉴스({adv_score:+.1f}점)와 함께 "
                f"섹터 평균보다 **{stock_diff:+.2f}%p** 강한 상승세입니다. "
                f"업종 대장주 역할을 하고 있습니다.",
                icon="🔺",
            )
        elif adv_score > 0 and stock_diff < -1:
            st.info(
                f"🤔 **재료 선반영 혹은 소외**: 호재 뉴스는 있으나 "
                f"섹터 평균({sector_avg:+.2f}%)보다 {abs(stock_diff):.2f}%p 약합니다. "
                f"추가 매수세 유입을 확인 후 진입하세요.",
                icon="🔷",
            )
        elif adv_score < 0 and stock_diff < -1:
            st.error(
                f"⚠️ **이중 위험 신호**: 악재 키워드({adv_score:+.1f}점)가 감지되었고 "
                f"섹터 대비 **{abs(stock_diff):.2f}%p** 더 가파르게 하락 중입니다. "
                f"손절 라인을 재확인하세요.",
                icon="🔻",
            )
        elif adv_score < 0 and stock_diff > 1:
            st.warning(
                f"🛡️ **악재 방어**: 악재 뉴스({adv_score:+.1f}점)에도 불구하고 "
                f"섹터보다 {stock_diff:+.2f}%p 강하게 버티고 있습니다. "
                f"수급·기관 매수 여부를 추가 확인하세요.",
                icon="🔔",
            )
        else:
            st.info(
                f"📊 **시장 동조화**: 뉴스 영향({adv_score:+.1f}점)이 제한적이며 "
                f"섹터 흐름({sector_avg:+.2f}%)과 비슷하게 움직이고 있습니다.",
                icon="🔷",
            )

        # ── 4단계 키워드 태그 ─────────────────────────────────────────────────
        hits = adv.get("hits", {})
        _CAT_STYLE = {
            "초강세": ("🚀", "#1b5e20", "#a5d6a7"),
            "강세":   ("📈", "#1a3a2a", "#66bb6a"),
            "약세":   ("📉", "#3e1616", "#ef9a9a"),
            "초악재": ("💀", "#7f0000", "#ff8a80"),
        }
        any_hits = any(hits.get(c) for c in _CAT_STYLE)
        if any_hits:
            st.markdown("**📌 포착된 키워드**")
            for cat, (icon, bg, fg) in _CAT_STYLE.items():
                words = hits.get(cat, [])
                if not words:
                    continue
                tags_html = "".join(
                    f'<span style="background:{bg};color:{fg};border-radius:6px;'
                    f'padding:3px 10px;margin:2px 4px;display:inline-block;font-size:0.85rem;">'
                    f'{icon} {w}</span>'
                    for w in words
                )
                st.markdown(
                    f'<div style="margin:4px 0;">'
                    f'<span style="color:#aaa;font-size:0.8rem;margin-right:6px;">{cat}</span>'
                    f'{tags_html}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("이번 뉴스에서 특이 키워드가 감지되지 않았습니다.")

        # ── 섹터 개별 종목 등락 ───────────────────────────────────────────────
        if sec["has_data"]:
            st.markdown("**🏭 섹터 구성 종목 등락**")
            sec_cols = st.columns(min(len(sec["tickers"]), 4))
            for i, t in enumerate(sec["tickers"]):
                arrow = "🔺" if t["chg"] >= 0 else "🔻"
                clr   = "#ef5350" if t["chg"] >= 0 else "#42a5f5"
                sec_cols[i % 4].markdown(
                    f'<div style="background:#1e2130;border-radius:8px;padding:8px 12px;text-align:center;">'
                    f'<div style="font-size:0.82rem;color:#aaa;">{t["ticker"]}</div>'
                    f'<div style="font-size:1rem;font-weight:bold;color:{clr};">{arrow} {t["chg"]:+.2f}%</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5  펀더멘털 & 기관
# ══════════════════════════════════════════════════════════════════════════════
with tab_fund:
    st.subheader(f"🏛️ {_asname or sname} 펀더멘털 & 기관 분석")

    # ── KRX DB 갱신 상태 표시 ─────────────────────────────────────────────────
    _is_krx = ticker.endswith(".KS") or ticker.endswith(".KQ")
    if _is_krx:
        try:
            from fundamental_db import get_last_update, needs_update, update_market
            _krx_market = "KOSPI" if ticker.endswith(".KS") else "KOSDAQ"
            _last_upd = get_last_update(_krx_market)
            _needs = needs_update(_krx_market)

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
        except ImportError:
            st.info("pykrx 미설치 — `pip install pykrx` 후 KRX 펀더멘털 DB를 사용할 수 있습니다.")

    # fund_info / fund_score_data 는 상단에서 이미 계산됨
    col_f1, col_f2 = st.columns([1, 1])

    # ── 장투 신호 근거 ────────────────────────────────────────────────────────
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

        st.markdown("**📋 투자법칙 신호 근거**")
        for r in fund_score_data.get("fund_reasons", []):
            if any(k in r for k in ["저평가", "우량", "충족", "탁월", "양호", "성장", "모멘텀"]):
                st.success(r, icon="🔺")
            elif any(k in r for k in ["경고", "위험", "부진", "고평가", "감소", "소진"]):
                st.error(r, icon="🔻")
            else:
                st.info(r, icon="🔷")

        if not fund_score_data.get("fund_reasons"):
            st.info("펀더멘털 데이터를 불러올 수 없습니다. (한국 주식은 일부 지표 미지원)")

    # ── 펀더멘털 지표 테이블 ──────────────────────────────────────────────────
    with col_f2:
        st.markdown("### 📈 시장 가치 지표")

        def _fund_row(label, key, fmt_fn, law_ref=""):
            val = fund_info.get(key)
            if val is not None and pd.notna(val):
                try:
                    return {"지표": label, "값": fmt_fn(val), "참고 법칙": law_ref}
                except Exception:
                    pass
            return {"지표": label, "값": "N/A", "참고 법칙": law_ref}

        def _fmt_money(v):
            if abs(v) >= 1e12: return f"{v/1e12:.2f}조"
            if abs(v) >= 1e8:  return f"{v/1e8:.0f}억"
            return f"{v:,.0f}"

        mc  = fund_info.get("market_cap")
        fcf = fund_info.get("free_cashflow")

        mkt_rows = [
            _fund_row("시가총액",           "market_cap",   _fmt_money, ""),
            _fund_row("PER (주가수익비율)", "per",          lambda v: f"{v:.2f}x", "그레이엄: PBR×PER < 22.5"),
            _fund_row("PBR (주가순자산비율)","pbr",         lambda v: f"{v:.2f}x", "그레이엄: PBR < 1.0 선호"),
            _fund_row("PSR (주가매출비율)", "psr",          lambda v: f"{v:.2f}x", "1.0 이하 저평가 기준"),
            _fund_row("Forward PER",        "forward_pe",   lambda v: f"{v:.2f}x", "성장 기대치 반영"),
            _fund_row("EPS (TTM)",          "eps_ttm",      lambda v: f"{v:,.2f}", ""),
            _fund_row("ROE (자기자본수익률)","roe",         lambda v: f"{v*100:.1f}%", "버핏: 15% 이상"),
            _fund_row("영업이익률",         "operating_margins", lambda v: f"{v*100:.1f}%", ""),
            _fund_row("부채비율",           "debt_equity",  lambda v: f"{v:.0f}%", "버핏: 50% 이하"),
            _fund_row("매출 성장률 (YoY)",  "revenue_growth",  lambda v: f"{v*100:+.1f}%", "린치: 20%+"),
            _fund_row("순이익 성장률 (YoY)","earnings_growth", lambda v: f"{v*100:+.1f}%", "CANSLIM: 25%+"),
        ]
        if fcf and mc and mc > 0:
            mkt_rows.insert(4, {"지표": "FCF Yield", "값": f"{fcf/mc*100:.1f}%", "참고 법칙": "버핏: 5% 이상"})

        st.dataframe(pd.DataFrame(mkt_rows), use_container_width=True, hide_index=True)

        # ── 재무제표 (DART 우선 → yfinance fallback) ──────────────────────
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

        def _fin_val(dart_key, yf_key, unit_label="억원"):
            # DART 우선
            if dart_fin.get(dart_key) is not None:
                v = dart_fin[dart_key]
                return f"{v:,.0f} {unit_label}"
            # yfinance fallback
            v = fund_info.get(yf_key)
            if v is not None and pd.notna(v):
                return _fmt_money(v)
            return "N/A"

        fin_src = f"DART {dart_fin.get('year','')}" if dart_fin else "yfinance"
        fin_rows = [
            {"지표": "매출액",    "값": _fin_val("revenue",          "total_revenue")},
            {"지표": "영업이익",  "값": _fin_val("operating_income",  "operating_income")},
            {"지표": "당기순이익","값": _fin_val("net_income",        "net_income")},
            {"지표": "잉여현금흐름(FCF)", "값": _fmt_money(fcf) if fcf else "N/A"},
        ]
        st.dataframe(pd.DataFrame(fin_rows), use_container_width=True, hide_index=True)

        _src = fund_info.get("source", "yfinance")
        st.caption(f"시장 지표: **{_src}** | 재무 지표: **{fin_src}**")

    # ── 전날 매매 동향 (pykrx) ────────────────────────────────────────────────
    if _is_krx_f:
        st.markdown("---")
        st.markdown("### 📊 전날 투자자별 매매 동향")
        with st.spinner("매매 동향 조회 중..."):
            try:
                from fundamental_db import get_trading_trend
                trend = get_trading_trend(ticker)
            except Exception:
                trend = {}

        if trend:
            t_date = trend.get("date", "")
            st.caption(f"기준일: {t_date[:4]}-{t_date[4:6]}-{t_date[6:]} (단위: 억원)")
            t_cols = st.columns(5)
            _labels = ["개인", "외국인", "기관합계", "금융투자", "연기금"]
            for col, lbl in zip(t_cols, _labels):
                val = trend.get(lbl)
                if val is not None:
                    color = "normal" if val == 0 else ("inverse" if val > 0 else "normal")
                    col.metric(lbl, f"{val:+,.1f}억", delta_color="normal" if val >= 0 else "inverse")
                else:
                    col.metric(lbl, "N/A")
        else:
            st.info("매매 동향 데이터를 불러올 수 없습니다. (pykrx 미설치 또는 데이터 없음)")

    # ── 손절·익절 상세 ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🛡️ 손절·익절 레벨 상세 (오닐·ATR 기반)")

    sl = get_stop_loss_targets(data)
    if sl:
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("현재가", f"{sl['current']:,.2f}",
                  help="최근 거래일 종가 기준 현재가입니다.")
        s2.metric("손절 (오닐 8%)", f"{sl['stop_8pct']:,.2f}",
                  f"{(sl['stop_8pct']/sl['current']-1)*100:.1f}%",
                  help=(
                      "윌리엄 오닐 CANSLIM 원칙: 매수가 대비 -8% 도달 시 무조건 손절.\n\n"
                      "작은 손실을 빠르게 잘라내어 큰 손실을 방지하는 핵심 원칙입니다.\n"
                      "출처: 《최고의 주식 최적의 타이밍》(윌리엄 오닐)"
                  ))
        s3.metric("손절 (ATR×2.5)", f"{sl['stop_atr']:,.2f}",
                  f"{(sl['stop_atr']/sl['current']-1)*100:.1f}%",
                  help=(
                      f"ATR(평균실질변동폭, 14일) × 2.5를 현재가에서 뺀 손절선.\n\n"
                      f"• 현재 ATR: {sl['atr']:,.2f} (일일 변동폭 {sl['atr_ratio']:.2f}%)\n"
                      f"• 종목의 정상 변동 범위를 벗어날 때 손절하는 방식\n"
                      f"• 변동성이 클수록 손절선이 넓어집니다."
                  ))
        s4.metric("목표 2R", f"{sl['target_2r']:,.2f}",
                  f"{(sl['target_2r']/sl['current']-1)*100:+.1f}%",
                  help=(
                      "리스크(현재가 - 손절가)의 2배를 현재가에 더한 1차 목표가.\n\n"
                      "• 손실 1에 이익 2를 추구하는 비율 (2:1 R/R)\n"
                      "• 최소 권장 비율. 이 가격에서 일부 분할 익절 고려."
                  ))
        s5.metric("목표 3R", f"{sl['target_3r']:,.2f}",
                  f"{(sl['target_3r']/sl['current']-1)*100:+.1f}%",
                  help=(
                      "리스크(현재가 - 손절가)의 3배를 더한 최종 목표가.\n\n"
                      "• 손실 1에 이익 3을 추구 (3:1 R/R)\n"
                      "• 트레이딩 표준 권장 비율. 오닐·린치 등 대부분의 투자법칙이 권장.\n"
                      "• 이 비율이면 승률 25%만 돼도 장기적으로 수익 가능."
                  ))

        st.markdown(f"""
        > **ATR (14일):** {sl['atr']:,.2f} ({sl['atr_ratio']:.2f}%)
        > **52주 고가:** {sl['high_52w']:,.2f}
        > **BB 상단:** {f"{sl['bb_upper']:,.2f}" if sl['bb_upper'] else 'N/A'}
        >
        > 📌 오닐 원칙: 매수가 대비 -7~8% 도달 시 무조건 손절 | 목표 3R = 리스크의 3배 수익
        """)

    # ── SEC 내부자 거래 (미국 주식 전용) ──────────────────────────────────────
    st.markdown("---")
    is_us = "." not in ticker
    if is_us:
        st.markdown("### 🏦 SEC 내부자 거래 (Form 4, 최근 90일)")
        st.caption("데이터 소스: SEC EDGAR (edgar.sec.gov) — 완전 무료 공식 API")

        with st.spinner("SEC EDGAR에서 내부자 거래 조회 중..."):
            insider_df = _insider_trades(ticker)

        if insider_df is not None and not insider_df.empty:
            st.dataframe(
                insider_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "링크": st.column_config.LinkColumn("SEC 공시 링크", display_text="바로가기"),
                },
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

        > 출처: Form 4 (거래 후 2영업일 이내 의무 공시, SEC EDGAR)
        """)
    else:
        st.markdown("### 🏦 기관/내부자 데이터")
        st.info(
            "SEC EDGAR Form 4 내부자 거래는 **미국 주식 전용** 기능입니다.  \n"
            "한국 주식의 경우 [DART 전자공시](https://dart.fss.or.kr)에서 조회 가능합니다."
        )

    # ── 투자 법칙 요약 ────────────────────────────────────────────────────────
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
with tab_chart:
    if not _data_ready:
        st.info("👈 사이드바에서 종목을 선택하고 **분석 시작** 버튼을 눌러주세요.", icon="📊")

    # 관심종목 추가/제거 버튼
    is_in_wl = any(w["ticker"] == ticker for w in st.session_state.watchlist)
    wl_col1, wl_col2 = st.columns([6, 1])
    with wl_col2:
        if is_in_wl:
            if st.button("★ 관심 해제", use_container_width=True):
                st.session_state.watchlist = [w for w in st.session_state.watchlist if w["ticker"] != ticker]
                save_watchlist(st.session_state.watchlist)
                st.rerun()
        else:
            if st.button("☆ 관심 추가", use_container_width=True, type="primary"):
                st.session_state.watchlist.append({"name": sname, "ticker": ticker})
                save_watchlist(st.session_state.watchlist)
                st.rerun()

    if data.empty:
        st.stop()  # 시장 현황 탭은 이미 렌더링됨, 이후 탭은 독립 컨테이너

    col_chart, col_sig = st.columns([3, 1])

    with col_chart:
        title_label = f"{sname} ({ticker})" if sname != ticker else ticker
        st.subheader(f"📈 {title_label} 기술적 분석")

        fig = make_subplots(
            rows=5, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            subplot_titles=(
                "가격 (캔들·EMA·볼린저밴드)",
                "거래량 + OBV 추세",
                "RSI (14) + MFI (14)",
                "MACD (12·26·9)",
                "ADX (14) + ±DI",
            ),
            row_heights=[0.40, 0.12, 0.16, 0.16, 0.16],
        )

        o = data["Open"]; h = data["High"]; lo = data["Low"]; c = data["Close"]; v = data["Volume"]

        # ── 캔들스틱 ────────────────────────────────────────────────────────
        fig.add_trace(go.Candlestick(
            x=data.index, open=o, high=h, low=lo, close=c,
            name="가격", increasing_line_color="#ef5350", decreasing_line_color="#42a5f5",
        ), row=1, col=1)

        # ── 이동평균선 (SMA5 + EMA20/50/200) ────────────────────────────────
        ma_lines = [
            ("SMA_5",   "#ffa726", "SMA5",   1.4, "solid"),
            ("EMA_20",  "#ab47bc", "EMA20",  1.4, "solid"),
            ("EMA_50",  "#26c6da", "EMA50",  1.4, "solid"),
            ("EMA_200", "#ffeb3b", "EMA200", 2.0, "dash"),
        ]
        for col_name, color, lbl, width, dash in ma_lines:
            if col_name in data.columns:
                fig.add_trace(go.Scatter(
                    x=data.index, y=data[col_name],
                    name=lbl, line=dict(color=color, width=width, dash=dash),
                ), row=1, col=1)

        # ── 볼린저밴드 ───────────────────────────────────────────────────────
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

        # ── 거래량 (Row 2) ───────────────────────────────────────────────────
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

        # OBV 추세 — 거래량과 같은 축에 스케일 맞춰 오버레이
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

        # ── RSI + MFI (Row 3) ────────────────────────────────────────────────
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

        # ── MACD (Row 4) ─────────────────────────────────────────────────────
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

        # ── ADX + ±DI (Row 5) ────────────────────────────────────────────────
        if "ADX" in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index, y=data["ADX"], name="ADX",
                line=dict(color="#fff176", width=2),
            ), row=5, col=1)
            # 추세 강도 기준선 (ADX=25 = 추세 시작)
            fig.add_hline(y=25, line_color="rgba(255,255,255,0.3)",
                          line_dash="dash", row=5, col=1)
            fig.add_hline(y=35, line_color="rgba(255,235,59,0.3)",
                          line_dash="dot", row=5, col=1)
        if "ADX_POS" in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index, y=data["ADX_POS"], name="+DI",
                line=dict(color="#66bb6a", width=1.3),
            ), row=5, col=1)
        if "ADX_NEG" in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index, y=data["ADX_NEG"], name="-DI",
                line=dict(color="#ef5350", width=1.3),
            ), row=5, col=1)

        # ── 레이아웃 ─────────────────────────────────────────────────────────
        fig.update_annotations(font_size=10, font_color="#9e9e9e")

        # 기간 선택 버튼 (가격 패널 x축에 표시)
        _rangeselector = dict(
            buttons=[
                dict(count=1,  label="1M", step="month", stepmode="backward"),
                dict(count=3,  label="3M", step="month", stepmode="backward"),
                dict(count=6,  label="6M", step="month", stepmode="backward"),
                dict(count=1,  label="1Y", step="year",  stepmode="backward"),
                dict(step="all", label="ALL"),
            ],
            bgcolor="#263238",
            activecolor="#1565c0",
            font=dict(color="#cfd8dc", size=10),
            bordercolor="#455a64",
            borderwidth=1,
        )

        # 모든 x축: 크로스헤어 스파이크 공통 적용
        fig.update_xaxes(
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikethickness=1,
            spikecolor="rgba(150,150,150,0.6)",
            spikedash="solid",
        )

        # 모든 y축: 크로스헤어 + autorange 보장
        fig.update_yaxes(
            showspikes=True,
            spikethickness=1,
            spikecolor="rgba(150,150,150,0.4)",
            spikedash="dot",
            fixedrange=False,
            autorange=True,
        )

        # 가격 패널(row1) x축에만 기간 선택 버튼 추가
        fig.update_xaxes(rangeselector=_rangeselector, row=1, col=1)

        fig.update_layout(
            height=900,
            template="plotly_dark",
            dragmode="pan",                     # 드래그 = 좌우 이동
            xaxis_rangeslider_visible=False,    # 기본 rangeslider 숨김
            legend=dict(
                orientation="h",
                yanchor="bottom", y=1.01,
                xanchor="right",  x=1,
                font=dict(size=11),
                bgcolor="rgba(0,0,0,0)",
            ),
            margin=dict(t=30, b=10, l=10, r=10),
            uirevision=ticker,     # 종목 바뀔 때만 뷰 초기화
            hovermode="x unified", # 모든 패널 동시 툴팁
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "scrollZoom": True,               # 마우스 휠로 x축 확대/축소
                "displayModeBar": True,
                "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
                "modeBarButtonsToAdd":    ["pan2d", "zoomIn2d", "zoomOut2d"],
                "toImageButtonOptions": {
                    "format": "png",
                    "filename": f"{ticker}_chart",
                    "scale": 2,
                },
            },
        )

    # ── 신호 패널 ─────────────────────────────────────────────────────────────
    with col_sig:
        st.subheader("🎯 AI 매매 신호")

        tech_score = signals.get("score", 0)

        # 뉴스 감성 점수 계산
        with st.spinner("뉴스 감성 분석 중..."):
            if use_llm:
                news_result = _news_sentiment_llm_cached(ticker, gemini_api_key)
            else:
                news_result = _news_sentiment_kw(ticker)
        news_score = news_result.get("score", 0.0)

        # ── 단타 신호 (기술 + 뉴스) ────────────────────────────────────────────
        hybrid  = get_hybrid_signal(tech_score, news_score)
        h_score = hybrid["hybrid_score"]
        h_label = hybrid["label"]
        h_badge = hybrid["badge"]

        if h_score >= 2:
            st_bg, st_fc = "#1b3a28", "#a5d6a7"
        elif h_score <= -2:
            st_bg, st_fc = "#3a1a1a", "#ef9a9a"
        else:
            st_bg, st_fc = "#1e2130", "#bdbdbd"

        # ── 장투 신호 (펀더멘털) ───────────────────────────────────────────────
        fs      = fund_score_data.get("fund_score", 0)
        f_label = fund_score_data.get("fund_label", "N/A")

        if fs >= 3:
            lt_bg, lt_fc = "#1a2f3a", "#80cbc4"
        elif fs <= -2:
            lt_bg, lt_fc = "#3a2a1a", "#ffcc80"
        else:
            lt_bg, lt_fc = "#1e2130", "#bdbdbd"

        f_badge = "🏛️"

        # ── 2개 박스 나란히 ────────────────────────────────────────────────────
        sig_col1, sig_col2 = st.columns(2)
        with sig_col1:
            st.markdown(f"""
            <div class="signal-box" style="background:{st_bg};">
                <div style="font-size:0.7rem;color:#888;margin-bottom:3px;letter-spacing:1px;">⚡ 단타 신호</div>
                <div style="font-size:1.25rem;font-weight:bold;color:{st_fc};">{h_badge} {h_label}</div>
                <div style="font-size:0.8rem;color:#aaa;margin-top:3px;">점수: <b style="color:{st_fc};">{h_score:+.1f}</b></div>
            </div>
            """, unsafe_allow_html=True)
        with sig_col2:
            st.markdown(f"""
            <div class="signal-box" style="background:{lt_bg};">
                <div style="font-size:0.7rem;color:#888;margin-bottom:3px;letter-spacing:1px;">🏛️ 장투 신호</div>
                <div style="font-size:1.25rem;font-weight:bold;color:{lt_fc};">{f_badge} {f_label}</div>
                <div style="font-size:0.8rem;color:#aaa;margin-top:3px;">점수: <b style="color:{lt_fc};">{fs:+.1f}</b></div>
            </div>
            """, unsafe_allow_html=True)

        # ── 점수 구성 미니 테이블 ──────────────────────────────────────────────
        tc = "#a5d6a7" if tech_score >= 0 else "#ef9a9a"
        nc = "#a5d6a7" if news_score >= 0 else "#ef9a9a"
        fc_color = lt_fc
        st.markdown(f"""
        <div style="background:#1e2130;border-radius:8px;padding:10px 14px;margin-top:6px;font-size:0.85rem;">
            <div style="color:#777;font-size:0.75rem;margin-bottom:6px;letter-spacing:0.5px;">⚡ 단타 구성</div>
            <div style="display:flex;justify-content:space-between;">
                <span style="color:#aaa;">📊 기술 분석 (70%)</span>
                <b style="color:{tc};">{tech_score:+.1f}</b>
            </div>
            <div style="display:flex;justify-content:space-between;margin-top:3px;">
                <span style="color:#aaa;">📰 뉴스 감성 (30%)</span>
                <b style="color:{nc};">{news_score:+.1f} <span style="font-weight:normal;color:#888;">({news_result.get('label','N/A')})</span></b>
            </div>
            <div style="border-top:1px solid #333;margin:8px 0 6px;"></div>
            <div style="color:#777;font-size:0.75rem;margin-bottom:6px;letter-spacing:0.5px;">🏛️ 장투 구성</div>
            <div style="display:flex;justify-content:space-between;">
                <span style="color:#aaa;">PER / PBR / ROE</span>
                <b style="color:{fc_color};">{fs:+.1f}</b>
            </div>
            <div style="display:flex;justify-content:space-between;margin-top:3px;">
                <span style="color:#aaa;">FCF / 성장률 / 부채</span>
                <span style="color:#888;font-size:0.8rem;">펀더멘털 탭 상세 ▶</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if not close.empty and len(close) >= 2:
            last_price = float(close.iloc[-1])
            prev_price = float(close.iloc[-2])
            daily_chg  = (last_price - prev_price) / prev_price * 100
            st.metric("현재가", f"{last_price:,.0f}", f"{daily_chg:+.2f}%",
                      help="최근 거래일 종가. 아래 숫자(화살표)는 전일 대비 등락률입니다.")

        if expected:
            st.metric("예상 수익률 (20일)", f"{expected['expected_return_pct']:+.2f}%",
                      help=(
                          "지금 매수 시 20거래일(약 1개월) 후의 예상 수익률 추정치.\n\n"
                          "계산 방식: 기술 신호 강도 × 역사적 변동성 + 20일 모멘텀 + 평균 일간 드리프트\n\n"
                          "⚠️ 보장 수익이 아닌 통계적 추정치입니다. 참고용으로만 활용하세요."
                      ))
            st.metric("연간 변동성", f"{expected['hist_volatility']:.1f}%",
                      help=(
                          "과거 일간 수익률의 표준편차 × √252 (연환산).\n\n"
                          "• 낮을수록 가격이 안정적 (예: 10~15% = 안정형)\n"
                          "• 높을수록 등락 폭이 큼 (예: 40%+ = 고위험)\n"
                          "• 리스크 척도로 활용합니다."
                      ))
            st.metric("20일 모멘텀", f"{expected['momentum_20d']:+.2f}%",
                      help=(
                          "20거래일(약 1개월) 전 대비 현재가 등락률.\n\n"
                          "• 양수(+): 최근 한 달 상승 추세 (매수 모멘텀)\n"
                          "• 음수(-): 최근 한 달 하락 추세 (매도 모멘텀)\n"
                          "• 과거 성과로 미래를 보장하지 않습니다."
                      ))
            st.metric("최대 낙폭", f"{expected['max_drawdown']:.2f}%",
                      help=(
                          "조회 기간 내 고점 대비 최대 하락폭 (MDD, Maximum Drawdown).\n\n"
                          "• 예: -15%이면 최고점에서 15%까지 하락한 적 있음\n"
                          "• 클수록(음수가 클수록) 과거에 큰 손실 구간이 있었음\n"
                          "• 리스크 및 손실 내성 파악에 활용합니다."
                      ))
            st.metric("샤프 지수", f"{expected['sharpe']:.2f}",
                      help=(
                          "무위험 수익률(3.5%) 초과 수익 ÷ 변동성 (연환산).\n\n"
                          "• 1.0 이상: 리스크 대비 양호한 수익\n"
                          "• 2.0 이상: 우수한 리스크/리워드 비율\n"
                          "• 0 미만: 리스크 대비 손실 구간\n"
                          "• 높을수록 효율적인 투자 대상입니다."
                      ))

        st.divider()
        st.markdown("**📋 기술 신호 근거**")
        _BUY_KEYS  = ["매수", "반등", "상승", "골든", "과매도", "매집", "긍정", "유입", "강세"]
        _SELL_KEYS = ["매도", "과열", "하락", "데드", "과매수", "분산", "약세", "이탈 주의"]
        for r in signals.get("reasons", []):
            if any(k in r for k in _BUY_KEYS):
                st.success(r, icon="🔺")
            elif any(k in r for k in _SELL_KEYS):
                st.error(r, icon="🔻")
            else:
                st.info(r, icon="🔷")

        st.divider()
        st.markdown("**📊 지표값**")
        indicator_map = [
            ("RSI",        "RSI",        ".1f"),
            ("MACD",       "MACD",       ".4f"),
            ("ADX",        "ADX",        ".1f"),
            ("+DI",        "ADX_POS",    ".1f"),
            ("-DI",        "ADX_NEG",    ".1f"),
            ("CCI",        "CCI",        ".1f"),
            ("Williams%R", "WILLIAMS_R", ".1f"),
            ("MFI",        "MFI",        ".1f"),
            ("ROC",        "ROC",        ".2f"),
            ("BB상단",     "BB_Upper",   ",.0f"),
            ("BB중단",     "BB_Middle",  ",.0f"),
            ("BB하단",     "BB_Lower",   ",.0f"),
            ("EMA20",      "EMA_20",     ",.0f"),
            ("EMA50",      "EMA_50",     ",.0f"),
            ("EMA200",     "EMA_200",    ",.0f"),
        ]
        last_row = data.iloc[-1]
        for lbl, col, fmt in indicator_map:
            if col in data.columns:
                val = last_row[col]
                if pd.notna(val):
                    st.caption(f"**{lbl}:** {float(val):{fmt}}")

        # ── 손절·익절 레벨 ────────────────────────────────────────────────
        st.divider()
        st.markdown("**🛡️ 손절·익절 가이드** (오닐 법칙)")
        sl = get_stop_loss_targets(data)
        if sl:
            st.caption(f"🔴 손절 (8%): **{sl['stop_8pct']:,.2f}**")
            st.caption(f"🟠 손절 (ATR): **{sl['stop_atr']:,.2f}**")
            st.caption(f"🟡 목표 2R: **{sl['target_2r']:,.2f}**")
            st.caption(f"🟢 목표 3R: **{sl['target_3r']:,.2f}**")
            st.caption(f"📏 ATR({sl['atr_ratio']:.2f}%): {sl['atr']:,.2f}")

