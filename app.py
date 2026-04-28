"""
app.py - AI 주식 분석 대시보드 v2.0
실시간 차트 · 매매 신호 · 시장 현황 · 추천 종목 · 펀더멘털 · 관심종목
"""
import json
import os
import concurrent.futures
import streamlit as st
from fundamental_db import load_settings_db, save_settings_db
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

try:
    from src.indicators import (
        get_stock_data, generate_signals, calculate_expected_return,
        get_stop_loss_targets, get_buy_target_price, get_sell_target_price,
        get_advanced_analysis, calculate_vpvr, detect_divergence,
        get_hybrid_signal, check_volume_anomaly,
    )
    from src.fundamental import (
        get_fundamental_data, calculate_fundamental_score,
        get_investment_recommendation, get_insider_trades_sec,
        get_etf_fundamental_data, calculate_etf_score,
    )
    from src.news_logic import (
        get_naver_news, analyze_news_sentiment_keywords,
        analyze_news_sentiment_llm, analyze_news_batch, summarize_article_llm,
        get_advanced_sentiment, get_related_sector_performance,
        get_etf_news_with_holdings, analyze_etf_news_sentiment,
    )
    from src.utils import (
        KOSPI_STOCKS, US_STOCKS, INDICES,
        get_market_movers, get_full_market_movers, get_exchange_rates,
        get_investor_trading_naver, get_recommendations,
        get_krx_stock_list, get_krx_etf_list, get_us_stock_list,
        get_top_kospi_stocks, get_top_kosdaq_stocks,
        get_top_us_stocks, get_top_nasdaq_stocks,
        is_etf_ticker, _ETF_PORTFOLIO_MAP,
    )
except Exception as _import_err:
    import traceback as _tb
    st.error(
        f"**모듈 로딩 오류** — 아래 전체 트레이스백을 확인하세요:\n\n"
        f"```\n{_tb.format_exc()}\n```"
    )
    st.stop()

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
    @keyframes loading-sweep {
        0%   { transform: translateX(-100%); }
        50%  { transform: translateX(0%); }
        100% { transform: translateX(100%); }
    }
    @keyframes loading-pulse {
        0%, 100% { opacity: 1; }
        50%       { opacity: 0.5; }
    }
    .loading-bar-track {
        background: #1e2444; border-radius: 8px; height: 6px;
        overflow: hidden; margin-top: 16px;
    }
    .loading-bar-fill {
        background: linear-gradient(90deg, #3b82f6, #60a5fa, #3b82f6);
        height: 100%; width: 100%;
        animation: loading-sweep 1.8s ease-in-out infinite;
    }
    .loading-icon { animation: loading-pulse 1.4s ease-in-out infinite; display: inline-block; }

    /* 모바일: 차트분석탭 — 신호 패널을 위로, 차트를 아래로 */
    @media (max-width: 768px) {
        div[data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) {
            flex-direction: column-reverse !important;
        }
        div[data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) > div[data-testid="column"] {
            width: 100% !important;
            flex: 0 0 100% !important;
            max-width: 100% !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# ─── 설정 저장소 (fundamentals.db → settings 테이블) ────────────────────────────
WATCHLIST_FILE = os.path.join(os.path.dirname(__file__), "watchlist.json")

def load_settings() -> dict:
    return load_settings_db()

def save_settings(data: dict) -> None:
    save_settings_db(data)
    _persist_to_secrets(data)

_SECRETS_PATH = os.path.join(os.path.dirname(__file__), ".streamlit", "secrets.toml")
_SECRETS_KEY_MAP = {
    "gemini_api_key": "GEMINI_API_KEY",
    "groq_api_key":   "GROQ_API_KEY",
    "dart_api_key":   "DART_API_KEY",
}

def _persist_to_secrets(data: dict) -> None:
    """API 키를 .streamlit/secrets.toml 에도 저장 — 재시작 후 st.secrets 로 자동 로드됨"""
    try:
        existing: dict[str, str] = {}
        try:
            with open(_SECRETS_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if "=" in line and not line.startswith("#"):
                        k, _, v = line.partition("=")
                        existing[k.strip()] = v.strip().strip('"').strip("'")
        except FileNotFoundError:
            pass
        for db_key, secret_key in _SECRETS_KEY_MAP.items():
            if data.get(db_key):
                existing[secret_key] = data[db_key]
        os.makedirs(os.path.dirname(_SECRETS_PATH), exist_ok=True)
        with open(_SECRETS_PATH, "w", encoding="utf-8") as f:
            for k, v in existing.items():
                f.write(f'{k} = "{v}"\n')
    except Exception:
        pass

# 기존 settings.json → DB 1회 마이그레이션
_SETTINGS_JSON = os.path.join(os.path.dirname(__file__), "settings.json")
if os.path.exists(_SETTINGS_JSON):
    try:
        with open(_SETTINGS_JSON, "r", encoding="utf-8") as _f:
            _old = json.load(_f)
        if _old:
            save_settings(_old)
        os.rename(_SETTINGS_JSON, _SETTINGS_JSON + ".migrated")
    except Exception:
        pass

# ─── 비밀번호 게이트 ──────────────────────────────────────────────────────────
_APP_PASSWORD = "qnwkehlwk"

# settings.json에 저장된 인증 상태 복원 (세션 시작 시 1회만)
if not st.session_state.get("app_authenticated"):
    _boot_settings = load_settings()
    if _boot_settings.get("app_authenticated"):
        st.session_state["app_authenticated"] = True

if not st.session_state.get("app_authenticated"):
    st.markdown("<br>" * 4, unsafe_allow_html=True)
    _, col, _ = st.columns([1.5, 1, 1.5])
    with col:
        st.markdown("## 📈 AI 주식 분석 터미널")
        st.markdown("---")
        with st.form("login_form"):
            pw_input = st.text_input(
                "비밀번호",
                type="password",
                placeholder="비밀번호를 입력하세요",
                label_visibility="collapsed",
            )
            submitted = st.form_submit_button("입장하기", use_container_width=True, type="primary")
        if submitted:
            if pw_input == _APP_PASSWORD:
                st.session_state["app_authenticated"] = True
                save_settings({**load_settings(), "app_authenticated": True})
                st.rerun()
            else:
                st.error("비밀번호가 틀렸습니다.")
    st.stop()

# ─── 관심종목 관리 ────────────────────────────────────────────────────────────
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

# st.secrets → DB 역방향 동기화 (재시작 후 DB가 비어 있어도 secrets에서 복원)
_secrets_loaded: dict[str, str] = {}
try:
    for _db_key, _sec_key in _SECRETS_KEY_MAP.items():
        _v = st.secrets.get(_sec_key, "")
        if _v:
            _secrets_loaded[_db_key] = _v
except Exception:
    pass
if _secrets_loaded and not any(_saved_settings.get(k) for k in _SECRETS_KEY_MAP):
    save_settings({**_saved_settings, **_secrets_loaded})
    _saved_settings = load_settings()

if "gemini_api_key" not in st.session_state:
    st.session_state["gemini_api_key"] = (
        _secrets_loaded.get("gemini_api_key") or _saved_settings.get("gemini_api_key", "")
    )
if "groq_api_key" not in st.session_state:
    st.session_state["groq_api_key"] = (
        _secrets_loaded.get("groq_api_key") or _saved_settings.get("groq_api_key", "")
    )
if "dart_api_key" not in st.session_state:
    st.session_state["dart_api_key"] = (
        _secrets_loaded.get("dart_api_key") or _saved_settings.get("dart_api_key", "")
    )

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
def _full_movers():
    """KOSPI+KOSDAQ 전체 종목 급등/급락 상위 10개"""
    return get_full_market_movers(top_n=10)

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

@st.cache_data(ttl=3600)
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
def _bench_returns(ticker: str) -> "pd.Series":
    """S&P500 또는 KOSPI 6개월 일간 수익률 (베타 계산용)"""
    sym = "^KS11" if (ticker.endswith(".KS") or ticker.endswith(".KQ")) else "^GSPC"
    try:
        d = yf.download(sym, period="6mo", auto_adjust=True, progress=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.droplevel(1)
        return d["Close"].pct_change().dropna()
    except Exception:
        return pd.Series(dtype=float)

@st.cache_data(ttl=3600)
def _news_sentiment_kw(ticker: str, company_name: str = "") -> dict:
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
    return analyze_news_sentiment_keywords(news, ticker, company_name)

def _news_sentiment_llm_cached(
    ticker: str, api_key: str, groq_api_key: str = "", company_name: str = ""
) -> dict:
    """LLM 분석은 API 키가 세션마다 다를 수 있어 session_state 캐시 사용"""
    cache_key = (
        f"llm_news|{ticker}|{api_key[:8] if api_key else ''}"
        f"|{groq_api_key[:8] if groq_api_key else ''}|{company_name}"
    )
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
    result = analyze_news_sentiment_llm(news, ticker, api_key, groq_api_key, company_name)
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

@st.cache_data(ttl=86400)
def _etf_stocks():
    return get_krx_etf_list()

@st.cache_data(ttl=86400)
def _check_is_etf(ticker: str) -> bool:
    """ETF 여부 — 포트폴리오 맵 우선, FDR 전체 목록은 24h 캐시로 한 번만 호출."""
    if is_etf_ticker(ticker):  # 포트폴리오 맵 빠른 확인 (네트워크 없음)
        return True
    # 포트폴리오 맵에 없는 ETF(소형 ETF 등): ETF 전체 목록에서 확인 (24h 캐시)
    if not (ticker.endswith(".KS") or ticker.endswith(".KQ")):
        return False
    try:
        etf_list = get_krx_etf_list()
        code = ticker.replace(".KS", "").replace(".KQ", "").strip().zfill(6)
        return f"{code}.KS" in etf_list.values() or f"{code}.KQ" in etf_list.values()
    except Exception:
        return False

@st.cache_data(ttl=300)
def _etf_fundamental(ticker: str) -> dict:
    return get_etf_fundamental_data(ticker)

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

def _clear_analysis():
    """종목 선택이 바뀌면 기존 분석 결과를 초기화해 자동 재분석을 막는다."""
    st.session_state.pop("analyzed_ticker", None)
    st.session_state.pop("analyzed_sname", None)
    st.session_state.pop("analyzed_period", None)

def _trigger_analysis_from_input():
    """직접 입력 text_input에서 Enter 시 분석을 즉시 시작한다."""
    val = st.session_state.get("_direct_ticker_input", "").strip()
    if val:
        st.session_state["analyzed_ticker"] = val
        st.session_state["analyzed_sname"]  = val
        st.session_state["analyzed_period"] = st.session_state.get("_period_sel", "3mo")

with st.sidebar:
    st.markdown("## ⚙️ 종목 설정")
    market_sel = st.selectbox(
        "시장",
        ["국내 주식 (검색)", "국내 ETF (검색)", "미국 주식 (검색)", "직접 입력"],
        key="_market_sel",
        on_change=_clear_analysis,
    )

    if market_sel == "국내 주식 (검색)":
        with st.spinner("종목 목록 로딩 중..."):
            krx = _krx_stocks()

        if krx:
            options = list(krx.keys())
            selected = st.selectbox(
                "종목 검색 (이름·코드 입력)",
                options,
                key="_krx_selected",
                help="회사 이름이나 종목 코드(6자리)를 입력하면 자동으로 필터링됩니다.",
            )
            ticker = krx[selected]
            sname  = selected.split(" (")[0]
        else:
            st.warning("종목 목록 로드 실패 — 기본 목록 사용")
            sname  = st.selectbox("종목", list(KOSPI_STOCKS.keys()),
                                  key="_krx_fallback")
            ticker = KOSPI_STOCKS[sname]

    elif market_sel == "국내 ETF (검색)":
        with st.spinner("ETF 목록 로딩 중..."):
            etf_list = _etf_stocks()

        if etf_list:
            etf_options = list(etf_list.keys())
            etf_selected = st.selectbox(
                "ETF 검색 (이름·코드 입력)",
                etf_options,
                key="_etf_selected",
                help="ETF 이름이나 6자리 코드를 입력하면 자동 필터링됩니다. 예) KODEX 200, 069500",
            )
            ticker = etf_list[etf_selected]
            sname  = etf_selected.split(" (")[0]
        else:
            st.warning("ETF 목록 로드 실패 — 기본 목록 사용")
            _etf_fb = {f"{v['name']} ({k})": f"{k}.KS" for k, v in _ETF_PORTFOLIO_MAP.items()}
            sname   = st.selectbox("ETF", list(_etf_fb.keys()), key="_etf_fallback")
            ticker  = _etf_fb[sname]
            sname   = sname.split(" (")[0]

        st.info("ETF는 기술적 분석 + ETF 전용 지표(괴리율·운용보수)로 분석됩니다.", icon="📊")

    elif market_sel == "미국 주식 (검색)":
        with st.spinner("미국 종목 목록 로딩 중... (S&P500 + 나스닥)"):
            us_list = _us_stocks()

        if us_list:
            us_options = list(us_list.keys())
            us_selected = st.selectbox(
                "종목 검색 (이름·티커 입력)",
                us_options,
                key="_us_selected",
                help="S&P500 + 나스닥 전체 종목. 회사명 또는 티커(예: AAPL)를 입력하면 자동 필터링됩니다.",
            )
            ticker = us_list[us_selected]
            sname  = us_selected.split(" (")[0]
        else:
            st.warning("미국 종목 목록 로드 실패 — 기본 목록 사용")
            sname  = st.selectbox("종목", list(US_STOCKS.keys()),
                                  key="_us_fallback")
            ticker = US_STOCKS[sname]

    else:
        ticker = st.text_input(
            "티커 직접 입력",
            value=st.session_state.get("_direct_ticker_input", "005930.KS"),
            key="_direct_ticker_input",
            on_change=_trigger_analysis_from_input,
            help="예) 005930.KS (KOSPI), 247540.KQ (KOSDAQ), AAPL (미국) — 입력 후 Enter 또는 아래 버튼 클릭",
        )
        sname  = ticker

    period = st.selectbox("분석 기간", ["1mo", "3mo", "6mo", "1y", "2y"],
                          index=1, key="_period_sel")

    # ── 분석 시작 버튼 ─────────────────────────────────────────────────────
    st.divider()
    if st.button("🔍 종목 분석 시작", type="primary", use_container_width=True):
        st.session_state.pop("analyzed_ticker", None)
        st.session_state["_pending_ticker"]  = ticker
        st.session_state["_pending_sname"]   = sname
        st.session_state["_pending_period"]  = period
        st.rerun()

    # ── API 키 상태 표시 ────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 🔑 API 연동 상태")

    try:
        _gem_secret = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        _gem_secret = ""
    gemini_api_key = _gem_secret or st.session_state.get("gemini_api_key", "")
    st.caption("🤖 Gemini: " + ("🟢 활성" if gemini_api_key else "⚪ 미설정 (키워드 분석)"))

    try:
        _groq_secret = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        _groq_secret = ""
    groq_api_key = _groq_secret or st.session_state.get("groq_api_key", "")
    st.caption("🦙 Groq:   " + ("🟢 활성 (Gemini 폴백)" if groq_api_key else "⚪ 미설정"))

    try:
        _dart_secret = st.secrets.get("DART_API_KEY", "")
    except Exception:
        _dart_secret = ""
    dart_api_key = _dart_secret or st.session_state.get("dart_api_key", "")
    st.caption("📑 DART:  " + ("🟢 활성" if dart_api_key else "⚪ 미설정 (yfinance 사용)"))

    _krx_ok = bool(os.environ.get("KRX_ID") and os.environ.get("KRX_PW"))
    st.caption("📊 KRX:   " + ("🟢 자동 로그인 활성" if _krx_ok else "⚪ 인증 미설정"))

    use_llm = bool(gemini_api_key)

    # ── API 키 설정 폼 (미설정 키 있으면 자동 펼침) ────────────────────────
    _cur_gemini = st.session_state.get("gemini_api_key", "")
    _cur_groq   = st.session_state.get("groq_api_key", "")
    _cur_dart   = st.session_state.get("dart_api_key", "")
    _cur_krx_id = _saved_settings.get("krx_id", "")
    _cur_krx_pw = _saved_settings.get("krx_pw", "")

    def _key_hint(v: str) -> str:
        """저장된 키의 앞 4자 + 마스킹 힌트 반환"""
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
                _to_save = {**load_settings()}
                # 빈칸이면 기존 값 유지, 입력이 있으면 새 값으로 덮어쓰기
                _to_save["gemini_api_key"] = _new_gemini or _cur_gemini
                _to_save["groq_api_key"]   = _new_groq   or _cur_groq
                _to_save["dart_api_key"]   = _new_dart   or _cur_dart
                _to_save["krx_id"]         = _new_krx_id or _cur_krx_id
                _to_save["krx_pw"]         = _new_krx_pw or _cur_krx_pw
                save_settings(_to_save)
                st.session_state["gemini_api_key"] = _to_save["gemini_api_key"]
                st.session_state["groq_api_key"]   = _to_save["groq_api_key"]
                st.session_state["dart_api_key"]   = _to_save["dart_api_key"]
                if _to_save["krx_id"]:
                    os.environ["KRX_ID"] = _to_save["krx_id"]
                if _to_save["krx_pw"]:
                    os.environ["KRX_PW"] = _to_save["krx_pw"]
                st.success("저장되었습니다!")
                st.rerun()

    # ── KRX 데이터 ─────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 📊 KRX 데이터")
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

# 탭 위에 항상 표시되는 로딩 배너 플레이스홀더
_loading_ph = st.empty()

# ─── 탭 레이아웃 (데이터 로딩 전 정의 — 탭 내 로딩 상태 표시용) ─────────────────
tab_market, tab_chart, tab_rec, tab_news, tab_fund = st.tabs([
    "🌐 시장 현황",
    "📊 차트 분석",
    "⭐ 추천 종목",
    "📰 뉴스 & 관련 종목",
    "🏛️ 펀더멘털 & 기관",
])

# 분석 완료 직후 차트 탭으로 자동 이동
if st.session_state.pop("_switch_to_chart", False):
    import streamlit.components.v1 as _components
    _components.html(
        """<script>
        setTimeout(function() {
            var p = window.parent.document;
            var tabs = p.querySelectorAll('[data-testid="stTab"]');
            if (tabs.length > 1) { tabs[1].click(); return; }
            tabs = p.querySelectorAll('button[role="tab"]');
            if (tabs.length > 1) tabs[1].click();
        }, 400);
        </script>""",
        height=0,
    )

# ─── 데이터 로드 (분석 시작 버튼 클릭 후에만 실행) ────────────────────────────
_aticker = st.session_state.get("analyzed_ticker")
_asname  = st.session_state.get("analyzed_sname", "")
_aperiod = st.session_state.get("analyzed_period", period)
_pending = st.session_state.get("_pending_ticker")
_data_ready = bool(_aticker)

# ── Rerun B: pending 처리 → 로딩 UI 표시 후 analyzed로 전환 ──────────────────
if _pending and not _aticker:
    _pname = st.session_state.get('_pending_sname', _pending)
    st.caption(f"업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  분석 중: **{_pname}** (`{_pending}`)")
    _loading_ph.markdown(f"""
<div style="background:linear-gradient(135deg,#1a1f3a 0%,#242b4d 100%);
            border:2px solid #3b82f6; border-radius:16px;
            padding:28px 24px; text-align:center; margin:8px 0 16px;
            box-shadow:0 4px 20px rgba(59,130,246,0.25);">
  <div class="loading-icon" style="font-size:40px;">⏳</div>
  <h3 style="color:#60a5fa; margin:12px 0 8px;">AI 분석 준비 중</h3>
  <p style="color:#94a3b8; margin:0; line-height:1.7;">
    <span style="color:#e2e8f0; font-weight:bold;">{_pname}</span>
    데이터를 불러오는 중입니다.<br>잠시만 기다려 주세요.
  </p>
  <div class="loading-bar-track"><div class="loading-bar-fill"></div></div>
</div>
""", unsafe_allow_html=True)
    # pending → analyzed 로 전환 후 rerun (탭 이동은 분석 완료 후)
    st.session_state["analyzed_ticker"]  = st.session_state.pop("_pending_ticker")
    st.session_state["analyzed_sname"]   = st.session_state.pop("_pending_sname", _pending)
    st.session_state["analyzed_period"]  = st.session_state.pop("_pending_period", period)
    st.session_state["_switch_to_chart"] = True
    st.rerun()

if _data_ready:
    # 분석 섹션 전체에서 사이드바 현재 선택값이 아닌 분석된 종목 기준으로 고정
    ticker = _aticker
    sname  = _asname

    st.caption(f"업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  분석 종목: **{_asname}** (`{_aticker}`)")

    # ── 주가 데이터 + 펀더멘털 병렬 로딩 ────────────────────────────────────
    _loading_ph.markdown(f"""
<div style="background:linear-gradient(135deg,#1a1f3a 0%,#242b4d 100%);
            border:2px solid #3b82f6; border-radius:16px;
            padding:28px 24px; text-align:center; margin:8px 0 16px;
            box-shadow:0 4px 20px rgba(59,130,246,0.25);">
  <div class="loading-icon" style="font-size:40px;">📊</div>
  <h3 style="color:#60a5fa; margin:12px 0 8px;">AI 분석 중</h3>
  <p style="color:#94a3b8; margin:0; line-height:1.7;">
    <span style="color:#e2e8f0; font-weight:bold;">{_asname}</span>
    주가·재무 데이터를 분석하고 있습니다.<br>잠시만 기다려 주세요.
  </p>
  <div class="loading-bar-track"><div class="loading-bar-fill"></div></div>
</div>
""", unsafe_allow_html=True)
    with st.spinner("📊 주가·재무 데이터 병렬 분석 중..."):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as _pool:
            _f_data = _pool.submit(_stock_data, _aticker, _aperiod)
            _f_fund = _pool.submit(_fundamental, _aticker)
            data      = _f_data.result()
            fund_info = _f_fund.result()
    _loading_ph.markdown(f"""
<div style="background:linear-gradient(135deg,#1a1f3a 0%,#242b4d 100%);
            border:2px solid #3b82f6; border-radius:16px;
            padding:28px 24px; text-align:center; margin:8px 0 16px;
            box-shadow:0 4px 20px rgba(59,130,246,0.25);">
  <div class="loading-icon" style="font-size:40px;">🎯</div>
  <h3 style="color:#60a5fa; margin:12px 0 8px;">AI 매매신호 분석 중</h3>
  <p style="color:#94a3b8; margin:0; line-height:1.7;">
    <span style="color:#e2e8f0; font-weight:bold;">{_asname}</span>
    뉴스 감성 · 기술적 지표를 종합하고 있습니다.<br>잠시만 기다려 주세요.
  </p>
  <div class="loading-bar-track"><div class="loading-bar-fill"></div></div>
</div>
""", unsafe_allow_html=True)

    if data.empty:
        st.session_state.pop("analyzed_ticker", None)
        st.error(f"'{_aticker}' 데이터를 불러올 수 없습니다. 티커를 확인해주세요.")
        st.stop()

    close       = data["Close"]
    vol_anomaly = check_volume_anomaly(data)
    signals     = generate_signals(data)

    # ── 거래 정지/주의: 거래량 이상 감지 시 점수 합산 중단 ─────────────────────
    if vol_anomaly.get("is_halted"):
        signals = {
            "score":   0,
            "label":   "거래 정지/주의",
            "badge":   "⛔",
            "reasons": [vol_anomaly.get("reason", "")],
            "_halted": True,
        }

    advanced = get_advanced_analysis(data)
    expected = calculate_expected_return(
        data, signals,
        ticker=_aticker,
        benchmark_returns=_bench_returns(_aticker),
    )
    fund_score_data = calculate_fundamental_score(fund_info, float(close.iloc[-1]))

else:
    st.caption(f"업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  사이드바에서 종목을 선택하고 분석을 시작하세요.")
    data            = pd.DataFrame()
    vol_anomaly     = {"is_halted": False}
    signals         = {"score": 0, "label": "분석 대기", "badge": "—"}
    expected        = None
    advanced        = {"trend_score": 50.0, "momentum_score": 50.0, "volume_score": 50.0,
                       "divergence": {}, "zscore": None, "vpvr": {}, "ichimoku": {}, "summary_items": []}
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


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1  차트 분석
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# TAB 2  시장 현황
# ══════════════════════════════════════════════════════════════════════════════
with tab_market:
    st.subheader("🌐 시장 현황")

    # ── 전체 시장 급등/급락 TOP 10 ──────────────────────────────────────────
    st.markdown("### 🏆 전체 시장 급등·급락 TOP 10 (KOSPI+KOSDAQ)")
    with st.spinner("전체 시장 데이터 로딩 중..."):
        _fm_gainers, _fm_losers = _full_movers()

    if not _fm_gainers.empty or not _fm_losers.empty:
        _fmc1, _fmc2 = st.columns(2)
        _cols_show = ["종목명", "티커", "현재가", "등락률(%)", "시장"]
        with _fmc1:
            st.markdown("#### 🚀 급등 상위 10")
            if not _fm_gainers.empty:
                st.dataframe(
                    _fm_gainers[_cols_show].style
                    .format({"현재가": "{:,.0f}", "등락률(%)": "{:+.2f}%"})
                    .map(lambda v: "color:#ef5350;font-weight:bold" if isinstance(v, float) and v > 0 else "", subset=["등락률(%)"]),
                    use_container_width=True, hide_index=True,
                )
        with _fmc2:
            st.markdown("#### 📉 급락 하위 10")
            if not _fm_losers.empty:
                st.dataframe(
                    _fm_losers[_cols_show].style
                    .format({"현재가": "{:,.0f}", "등락률(%)": "{:+.2f}%"})
                    .map(lambda v: "color:#42a5f5;font-weight:bold" if isinstance(v, float) and v < 0 else "", subset=["등락률(%)"]),
                    use_container_width=True, hide_index=True,
                )
    else:
        st.info("pykrx 또는 FinanceDataReader 미설치 시 전체 시장 데이터를 불러올 수 없습니다.")

    st.divider()

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
            | 기술점수 | -10 ~ +10 | RSI·MACD·EMA·ADX·일목·Z-Score·다이버전스 등 11개 지표 복합 신호 |
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
        | 기술점수 | RSI·MACD·EMA·ADX·OBV·일목·Z-Score·다이버전스 등 11개 지표 | 범위 -10 ~ +10 |
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
        | 일목균형표 | 중기 추세 | 구름 위 강세 | 구름 아래 약세 |
        | Z-Score | 통계적 위치 | < -2.5σ 과매도 | > +2.5σ 과매수 |
        | 다이버전스 | 추세 전환 | 상승 다이버전스 | 하락 다이버전스 |
        """)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4  뉴스 & 관련 종목
# ══════════════════════════════════════════════════════════════════════════════
with tab_news:
    _news_is_etf = _data_ready and _check_is_etf(ticker)

    if _news_is_etf:
        st.subheader(f"📊 {_asname or sname} 섹터 뉴스 — 돈이 몰리는 섹터 파악")
        st.caption("ETF는 종목 필터를 느슨하게 적용하고 상위 구성종목 뉴스를 함께 수집합니다.")
    else:
        st.subheader(f"📰 {_asname or sname} 뉴스 & 관련 정보")

    is_kr_stock = ticker.endswith(".KS") or ticker.endswith(".KQ")

    # ── 뉴스 수집 ────────────────────────────────────────────────────────────
    raw_news: list = []
    if _news_is_etf:
        # ETF: 자체 뉴스 + 상위 구성종목 뉴스 통합 수집
        with st.spinner("ETF 섹터 뉴스 수집 중 (ETF + 구성종목)..."):
            _etf_fund_data = _etf_fundamental(ticker)
            raw_news = get_etf_news_with_holdings(ticker, _etf_fund_data, max_items=15)
        # ETF 섹터 배지 표시
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
    _cname = sname if sname != ticker else ""
    sent: dict = {}
    if raw_news:
        with st.spinner("AI 감성 분석 중..."):
            if _news_is_etf:
                # ETF 전용 섹터 키워드 기반 감성 분석 (_etf_fund_data는 위 수집 단계에서 이미 로드됨)
                sent = analyze_etf_news_sentiment(ticker, _etf_fund_data, raw_news)
            elif use_llm:
                sent = _news_sentiment_llm_cached(ticker, gemini_api_key, groq_api_key, _cname)
            else:
                sent = _news_sentiment_kw(ticker, _cname)

    col_news, col_rel = st.columns([3, 2])

    with col_news:
        if _news_is_etf:
            st.markdown("### 📊 ETF 섹터 뉴스 (ETF 자체 + 구성종목)")
        elif is_kr_stock:
            st.markdown("### 최신 뉴스 (네이버 금융)")
        else:
            st.markdown("### 최신 뉴스")

        if not raw_news:
            st.info("뉴스 데이터가 없습니다.")
        else:
            # ── 이벤트 대기 배너 ─────────────────────────────────────────────
            event_flags = sent.get("event_flags", [])
            if event_flags:
                _ev_labels = "·".join(
                    "·".join(e.get("event_kw", [])[:2]) for e in event_flags[:3]
                )
                st.markdown(
                    f'<div style="background:#4a148c;border:1px solid #ce93d8;border-radius:8px;'
                    f'padding:10px 16px;margin-bottom:12px;">'
                    f'<span style="font-size:1rem;font-weight:bold;color:#e040fb;">🔔 이벤트 대기</span>'
                    f'<span style="color:#ce93d8;margin-left:10px;font-size:0.9rem;">{_ev_labels}</span>'
                    f'<div style="color:#ba68c8;font-size:0.82rem;margin-top:4px;">'
                    f'해당 이벤트 뉴스는 기술적 분석 점수에 반영되지 않습니다. 이벤트 결과 확인 후 재분석하세요.'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # 전체 감성 요약 배너
            s_score        = sent.get("score", 0.0)
            s_label        = sent.get("label", "중립")
            s_summary      = sent.get("summary", "")
            s_indiv        = sent.get("individual_score")
            s_sector       = sent.get("sector_score", 0.0)
            if s_score >= 1:
                banner_color, text_color = "#1b5e20", "#a5d6a7"
            elif s_score <= -1:
                banner_color, text_color = "#b71c1c", "#ef9a9a"
            else:
                banner_color, text_color = "#1e2130", "#bdbdbd"

            sector_html = ""
            if s_sector != 0.0:
                sector_clr = "#a5d6a7" if s_sector > 0 else "#ef9a9a"
                sector_html = (
                    f'<span style="font-size:0.82rem;color:{sector_clr};margin-left:8px;">'
                    f'섹터 {s_sector:+.1f}</span>'
                )
            indiv_html = ""
            if s_indiv is not None and s_sector != 0.0:
                indiv_html = (
                    f'<span style="font-size:0.82rem;color:#bdbdbd;margin-left:4px;">'
                    f'(종목 {s_indiv:+.1f})</span>'
                )
            llm_badge = "&nbsp;|&nbsp; 🤖 Gemini AI 분석" if use_llm else "&nbsp;|&nbsp; 🔑 키워드 분석"
            summary_html = f'<div style="font-size:0.88rem;color:#ccc;margin-top:6px;">{s_summary}</div>' if s_summary else ""

            st.markdown(
                f'<div style="background:{banner_color};border-radius:10px;padding:12px 18px;margin-bottom:14px;">'
                f'<div style="font-size:1.05rem;font-weight:bold;color:{text_color};">'
                f'뉴스 감성: {s_label} &nbsp;'
                f'<span style="font-size:0.95rem;font-weight:normal;">({s_score:+.1f}점 / ±5)</span>'
                f'{sector_html}{indiv_html}'
                f'{llm_badge}'
                f'</div>'
                f'{summary_html}'
                f'</div>',
                unsafe_allow_html=True
            )

            # 기사별 표시
            def _render_article(item, art_idx, d, *, faded=False):
                title      = item.get("title", "제목 없음")
                link       = item.get("link", "#")
                publisher  = item.get("publisher", "")
                pub_date   = item.get("pub_date", "")
                art_score  = d.get("score", 0.0)
                art_reason = d.get("reason", "")
                art_tier   = d.get("tier", "")
                art_decay  = d.get("decay", 1.0)
                is_skipped = d.get("skipped", False)

                if is_skipped:
                    dot_color, dot = "#9e9e9e", "⚫"
                elif art_score >= 0.3:
                    dot_color, dot = "#66bb6a", "🟢"
                elif art_score <= -0.3:
                    dot_color, dot = "#ef5350", "🔴"
                else:
                    dot_color, dot = "#9e9e9e", "⚪"

                _TIER_BADGES = {
                    "A":     '<span style="background:#e65100;color:#fff;font-size:0.7rem;padding:1px 5px;border-radius:4px;margin-left:4px;">A급</span>',
                    "B":     '<span style="background:#1565c0;color:#fff;font-size:0.7rem;padding:1px 5px;border-radius:4px;margin-left:4px;">B급</span>',
                    "C":     '<span style="background:#37474f;color:#ccc;font-size:0.7rem;padding:1px 5px;border-radius:4px;margin-left:4px;">C급</span>',
                    "EVENT": '<span style="background:#6a1b9a;color:#fff;font-size:0.7rem;padding:1px 5px;border-radius:4px;margin-left:4px;">이벤트</span>',
                    "SKIP":  '<span style="background:#424242;color:#bdbdbd;font-size:0.7rem;padding:1px 5px;border-radius:4px;margin-left:4px;">제외</span>',
                }
                tier_badge = _TIER_BADGES.get(art_tier, "")
                title_color = "#607d8b" if faded else "#90caf9"

                col_article, col_btn = st.columns([8, 1])
                with col_article:
                    st.markdown(
                        f'<div style="margin-bottom:2px;">'
                        f'<span style="color:{dot_color};font-size:0.8rem;">{dot} {art_score:+.1f}</span>'
                        f'{tier_badge}'
                        f' &nbsp;<b><a href="{link}" target="_blank"'
                        f' style="color:{title_color};text-decoration:none;">{title}</a></b>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    meta_parts = []
                    if publisher: meta_parts.append(f"📌 {publisher}")
                    if pub_date:  meta_parts.append(f"🕐 {pub_date}")
                    if art_decay < 1.0: meta_parts.append(f"감쇠×{art_decay:.1f}")
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

            detail_map  = {d["title"]: d for d in sent.get("detail", [])}
            main_items  = []
            other_items = []
            for item in raw_news:
                d = detail_map.get(item.get("title", "제목 없음"), {})
                if d.get("hidden") or d.get("skipped"):
                    other_items.append((item, d))
                else:
                    main_items.append((item, d))

            for art_idx, (item, d) in enumerate(main_items):
                _render_article(item, art_idx, d)

            if other_items:
                with st.expander(f"기타 뉴스 ({len(other_items)}건) — C급·제외 항목", expanded=False):
                    for art_idx, (item, d) in enumerate(other_items):
                        _render_article(item, len(main_items) + art_idx, d, faded=True)

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
        if _news_is_etf:
            st.markdown("### 🔍 섹터 자금 흐름 분석 — ETF 구성종목 동조화")
        else:
            st.markdown("### 🔍 심층 분석 — 키워드 강도 · 섹터 동조화")

        # 확장 키워드 감성
        adv = get_advanced_sentiment(raw_news)
        adv_score = adv.get("score", 0.0)

        # 섹터 동조화 (ETF: 구성종목 기준, 일반: 섹터맵 + ETF 역조회)
        with st.spinner("섹터 동조화 분석 중..."):
            sec = _sector_perf(ticker)
        sector_avg = sec.get("avg_chg", 0.0)
        _sec_label = sec.get("sector", "")
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
        _sec_label_str = f"[{_sec_label}] " if _sec_label else ""
        c2.metric(
            "🏭 섹터 평균 등락" if not _news_is_etf else "📦 구성종목 평균 등락",
            f"{sector_avg:+.2f}%",
            f"{_sec_label_str}{'데이터 있음' if sec['has_data'] else '매핑 없음'}",
            help=(
                "동일 업종 연관 종목들의 당일 평균 등락률.\n\n"
                + ("ETF: 구성종목 기준 섹터 평균입니다.\n\n" if _news_is_etf else "")
                + "• 섹터 전체가 오르는 날 이 종목도 오르면 동조화\n"
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
    _is_etf = _data_ready and _check_is_etf(ticker)

    if _is_etf:
        # ══ ETF 전용 분석 UI ═════════════════════════════════════════════════
        st.subheader(f"📊 {_asname or sname} ETF 분석")
        st.caption("ETF는 괴리율·운용보수·추적오차 중심으로 평가합니다. '현재 어느 섹터에 돈이 몰리는가'를 파악하는 용도로 활용하세요.")

        with st.spinner("ETF 지표 로딩 중..."):
            _etf_data   = _etf_fundamental(ticker)
            _etf_score  = calculate_etf_score(_etf_data)

        # ── ETF 핵심 지표 메트릭 행 ──────────────────────────────────────────
        _em1, _em2, _em3, _em4, _em5 = st.columns(5)
        _price     = _etf_data.get("price")
        _nav       = _etf_data.get("nav")
        _premium   = _etf_data.get("nav_premium")
        _er        = _etf_data.get("expense_ratio")
        _te        = _etf_data.get("tracking_error")
        _div       = _etf_data.get("dividend_yield")
        _aum       = _etf_data.get("aum")
        _sector    = _etf_data.get("sector", "")

        _em1.metric("현재가", f"{_price:,.0f}원" if _price else "N/A")
        _em2.metric("NAV",    f"{_nav:,.0f}원"  if _nav   else "N/A",
                    delta=f"괴리율 {_premium:+.2f}%" if _premium is not None else None)
        _em3.metric("운용보수(ER)", f"{_er:.3f}%" if _er is not None else "N/A",
                    help="연간 운용보수. 낮을수록 장기 복리 수익 유리")
        _em4.metric("추적오차",    f"{_te:.2f}%" if _te is not None else "N/A",
                    help="기초지수 대비 추종 오차. 낮을수록 지수를 정밀하게 추종")
        _em5.metric("배당수익률",  f"{_div:.2f}%" if _div is not None else "N/A")

        if _aum:
            st.caption(f"순자산(AUM): **{_aum:,.0f}억원**  |  섹터: **{_sector}**  |  출처: {_etf_data.get('source','')}")

        st.markdown("---")

        # ── ETF 점수 & 판정 ──────────────────────────────────────────────────
        _es_col, _er_col = st.columns([1, 2])
        with _es_col:
            _etf_s    = _etf_score.get("etf_score", 0.0)
            _etf_lbl  = _etf_score.get("etf_label", "N/A")
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

            # 항목별 점수 바
            st.markdown("**항목별 점수**")
            _bd = _etf_score.get("score_breakdown", {})
            _score_meta = [
                ("괴리율",   "±3", "#80cbc4"),
                ("운용보수", "±2", "#a5d6a7"),
                ("추적오차", "±1.5", "#ce93d8"),
                ("배당수익률","±0.5","#ffcc80"),
            ]
            for _lbl, _rng, _clr in _score_meta:
                _sv = _bd.get(_lbl, 0.0)
                # 0~100 변환: 3 → 100%, -3 → 0%
                _max = float(_rng.replace("±",""))
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
            _pos_etf_kw = ["매수", "유리", "양호", "낮음", "정밀", "할인", "최저"]
            _neg_etf_kw = ["주의", "높음", "과열", "프리미엄", "불량", "잠식", "위험"]
            for _r in _etf_score.get("etf_reasons", []):
                if any(k in _r for k in _pos_etf_kw):
                    st.success(_r, icon="✅")
                elif any(k in _r for k in _neg_etf_kw):
                    st.warning(_r, icon="⚠️")
                else:
                    st.info(_r, icon="ℹ️")

            if not _etf_score.get("etf_reasons"):
                st.info("ETF 지표 데이터를 불러올 수 없습니다. (pykrx / Naver Finance 접근 필요)")

        st.markdown("---")

        # ── NAV 괴리율 설명 ──────────────────────────────────────────────────
        if _premium is not None:
            _prem_color = "#ef5350" if _premium > 1 else ("#42a5f5" if _premium < -0.5 else "#bdbdbd")
            st.markdown(
                f'<div style="background:#12141f;border-radius:10px;padding:14px 18px;">'
                f'<div style="font-size:0.85rem;color:#aaa;">📐 NAV 괴리율이란?</div>'
                f'<div style="font-size:1rem;margin-top:4px;">현재가(<b>{_price:,.0f}원</b>) vs NAV(<b>{_nav:,.0f}원</b>)</div>'
                f'<div style="font-size:1.1rem;font-weight:bold;color:{_prem_color};margin-top:4px;">'
                f'{"🔴 프리미엄" if _premium > 0 else "🔵 할인"} {_premium:+.2f}%</div>'
                f'<div style="font-size:0.8rem;color:#888;margin-top:6px;">'
                f'{"시장가가 NAV보다 높음 → 고평가. 매수 시 주의 필요" if _premium > 0.5 else "시장가가 NAV보다 낮음 → 저평가. 잠재적 매수 기회"}'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # ── 구성종목 (PDF) 상위 보유 현황 ────────────────────────────────────
        st.markdown("### 🗂️ 상위 구성종목 (PDF)")
        _holdings = _etf_data.get("top_holdings", [])
        if _holdings:
            _h_rows = []
            for _h in _holdings[:10]:
                _h_rows.append({
                    "종목 티커":  _h.get("ticker", ""),
                    "종목명":     _h.get("name", ""),
                    "비중(%)":    f'{_h["weight"]:.2f}' if _h.get("weight") is not None else "N/A",
                })
            st.dataframe(pd.DataFrame(_h_rows), use_container_width=True, hide_index=True)
            st.caption("상위 구성종목이 해당 섹터의 시장 흐름을 직접 반영합니다.")
        else:
            st.info("구성종목 데이터를 불러올 수 없습니다. (pykrx PDF 미수집)")

        # ── 기술적 분석 점수 요약 (ETF도 동일 적용) ─────────────────────────
        if _data_ready:
            st.markdown("### 📈 기술적 분석 (ETF 동일 적용)")
            _etf_sig = signals
            _sig_score = _etf_sig.get("score", 0)
            _sig_label = _etf_sig.get("label", "N/A")
            _sig_clr = "#a5d6a7" if _sig_score >= 3 else ("#ef9a9a" if _sig_score <= -3 else "#bdbdbd")
            st.markdown(
                f'<div style="background:#1e2130;border-radius:10px;padding:12px 18px;">'
                f'<span style="font-size:0.9rem;color:#aaa;">기술적 신호: </span>'
                f'<b style="color:{_sig_clr};font-size:1.1rem;">{_sig_label} ({_sig_score:+.1f}점)</b>'
                f'</div>',
                unsafe_allow_html=True,
            )
            for _reason in (_etf_sig.get("reasons") or [])[:5]:
                st.caption(f"• {_reason}")

    else:
        pass

    if not _is_etf:
        # ── KRX DB 갱신 상태 표시 ─────────────────────────────────────────────────
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

            # ── 4분류 서브 점수 바 ────────────────────────────────────────────────
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
            _fund_bar("성장성 (PEG·매출)",        fund_score_data.get("sub_growth", 50), "40%")
            _fund_bar("수익성 (ROE·FCF·OCF)",     fund_score_data.get("sub_profit", 50), "30%")
            _fund_bar("안정성 (그레이엄·부채)",   fund_score_data.get("sub_stable", 50), "20%")
            _fund_bar("모멘텀 (52주·주주환원)",   fund_score_data.get("sub_moment", 50), "10%")

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

            # 새 보조 지표 행 추가
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

        # ── 거장의 한 줄 평 ──────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🎖️ 투자 거장의 한 줄 평")

        _verdicts = fund_score_data.get("master_verdicts", {})
        _master_meta = [
            ("그레이엄", "📖 벤저민 그레이엄", "#80cbc4",  "안전마진·가치투자의 아버지"),
            ("버핏",    "🏛️ 워렌 버핏",       "#a5d6a7",  "ROE 지속성·경제적 해자"),
            ("린치",    "🚀 피터 린치",        "#ce93d8",  "PEG·성장주 발굴"),
            ("오닐",    "🔥 윌리엄 오닐",      "#ffcc80",  "신고가·CANSLIM"),
        ]
        _vcols = st.columns(4)
        for _vcol, (_key, _name, _clr, _sub) in zip(_vcols, _master_meta):
            _v = _verdicts.get(_key, {})
            _icon    = _v.get("icon", "—")
            _verdict = _v.get("판정", "N/A")
            _comment = _v.get("comment", "데이터 부족")
            _vcol.markdown(
                f'<div style="background:#12141f;border-radius:10px;padding:12px;'
                f'border-left:3px solid {_clr};min-height:140px;">'
                f'<div style="font-size:0.72rem;color:#888;margin-bottom:4px;">{_sub}</div>'
                f'<div style="font-size:0.9rem;font-weight:bold;color:{_clr};margin-bottom:4px;">'
                f'{_name}</div>'
                f'<div style="font-size:1.1rem;margin-bottom:6px;">'
                f'{_icon} <b style="color:#e0e0e0;">{_verdict}</b></div>'
                f'<div style="font-size:0.75rem;color:#aaa;line-height:1.5;">{_comment}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # ── 전날 매매 동향 (Naver Finance) ───────────────────────────────────────
        if _is_krx_f:
            st.markdown("---")
            st.markdown("### 📊 전날 투자자별 매매 동향")
            with st.spinner("투자자 데이터 조회 중..."):
                _inv_f = get_investor_trading_naver(ticker)
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
                        _sign = "+" if val > 0 else ""
                        _color = "#ef5350" if val > 0 else ("#42a5f5" if val < 0 else "#bdbdbd")
                        col.markdown(
                            f'<div style="background:#1e2130;padding:14px;border-radius:10px;text-align:center;">'
                            f'<div style="font-size:0.75rem;color:#888;margin-bottom:4px;">{label}</div>'
                            f'<div style="font-size:1.1rem;font-weight:bold;color:{_color};">{_sign}{val:,}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        col.markdown(
                            f'<div style="background:#1e2130;padding:14px;border-radius:10px;text-align:center;">'
                            f'<div style="font-size:0.75rem;color:#888;margin-bottom:4px;">{label}</div>'
                            f'<div style="font-size:1.1rem;color:#555;">N/A</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
            else:
                st.caption("투자자 매매 동향 데이터를 불러올 수 없습니다.")

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

    # ── 거래 정지/주의 경고 배너 ─────────────────────────────────────────────
    if vol_anomaly.get("is_halted"):
        st.markdown(
            f'<div style="background:#4a1010;border:2px solid #ef5350;border-radius:10px;'
            f'padding:14px 20px;margin-bottom:16px;">'
            f'<span style="font-size:1.2rem;font-weight:bold;color:#ef5350;">⛔ 거래 정지/주의</span>'
            f'<div style="color:#ffcdd2;margin-top:6px;">{vol_anomaly.get("reason", "")}</div>'
            f'<div style="color:#ef9a9a;font-size:0.85rem;margin-top:4px;">'
            f'최근 거래량: {vol_anomaly.get("recent_vol", 0):,} &nbsp;|&nbsp; '
            f'20일 평균: {int(vol_anomaly.get("avg_vol", 0)):,} &nbsp;|&nbsp; '
            f'비율: {vol_anomaly.get("ratio", 0) * 100:.1f}%'
            f'</div>'
            f'<div style="color:#ff8a80;font-size:0.82rem;margin-top:4px;">'
            f'기술적 분석 점수 합산이 중단되었습니다. 거래 재개 후 분석을 다시 실행하세요.'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    col_chart, col_sig = st.columns([3, 7])

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

        # ── 일목균형표 (Ichimoku Cloud) ──────────────────────────────────────
        if "ICHI_SPAN_A" in data.columns and "ICHI_SPAN_B" in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index, y=data["ICHI_SPAN_A"], name="선행스팬A",
                line=dict(color="rgba(102,187,106,0.7)", width=1),
                fill=None, showlegend=True,
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=data.index, y=data["ICHI_SPAN_B"], name="선행스팬B",
                line=dict(color="rgba(239,83,80,0.7)", width=1),
                fill="tonexty", fillcolor="rgba(120,120,120,0.12)", showlegend=True,
            ), row=1, col=1)
        if "ICHI_TENKAN" in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index, y=data["ICHI_TENKAN"], name="전환선",
                line=dict(color="#ef5350", width=1.1, dash="dot"),
            ), row=1, col=1)
        if "ICHI_KIJUN" in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index, y=data["ICHI_KIJUN"], name="기준선",
                line=dict(color="#42a5f5", width=1.1, dash="dot"),
            ), row=1, col=1)

        # ── VWAP 멀티 타임프레임 (Shannon) ──────────────────────────────────
        vwap_lines = [
            ("VWAP_W", "#ff8f00", "VWAP 주간(5일)",   1.4, "solid"),
            ("VWAP_M", "#ce93d8", "VWAP 월간(20일)",  1.6, "solid"),
            ("VWAP_Q", "#80deea", "VWAP 분기(60일)",  2.0, "dash"),
        ]
        for _vc, _color, _lbl, _w, _dash in vwap_lines:
            if _vc in data.columns:
                fig.add_trace(go.Scatter(
                    x=data.index, y=data[_vc],
                    name=_lbl, line=dict(color=_color, width=_w, dash=_dash),
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
            dragmode=False,                     # 터치 드래그 비활성화
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
                "scrollZoom": False,              # 터치 스크롤 줌 비활성화
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
                news_result = _news_sentiment_llm_cached(ticker, gemini_api_key, groq_api_key)
            else:
                news_result = _news_sentiment_kw(ticker)
        news_score = news_result.get("score", 0.0)

        # ── 단타 신호 (기술 + 뉴스) ────────────────────────────────────────────
        hybrid  = get_hybrid_signal(tech_score, news_score)
        _loading_ph.empty()
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

        # ── 단타 신호 전체 단계 안내 ──────────────────────────────────────────
        with st.expander("⚡ 단타 신호 판정 기준 보기", expanded=False):
            st.markdown("""
<div style="font-size:0.82rem;line-height:1.9;">

| 신호 | 점수 범위 | 의미 |
|------|-----------|------|
| 🟢🟢 **강력 매수** | +5 이상 | 강한 상승 신호 — 적극 매수 고려 |
| 🟢 **매수** | +3 ~ +4 | 상승 신호 — 매수 고려 |
| 🔵 **약한 매수** | +1 ~ +2 | 약한 상승 신호 — 소량 매수 가능 |
| ⚪ **중립/관망** | 0 | 방향성 불명확 — 관망 권고 |
| 🟡 **약한 매도** | -1 ~ -2 | 약한 하락 신호 — 비중 축소 고려 |
| 🔴 **매도** | -3 ~ -4 | 하락 신호 — 매도 고려 |
| 🔴🔴 **강력 매도** | -5 이하 | 강한 하락 신호 — 적극 매도 고려 |

</div>
<div style="font-size:0.75rem;color:#888;margin-top:4px;">
※ 점수 = 기술 분석(70%) × 기술점수 + 뉴스 감성(30%) × 뉴스점수 의 가중 합산
</div>
            """, unsafe_allow_html=True)

        # ── 점수 구성 미니 테이블 ──────────────────────────────────────────────
        tc = "#a5d6a7" if tech_score >= 0 else "#ef9a9a"
        nc = "#a5d6a7" if news_score >= 0 else "#ef9a9a"
        fc_color = lt_fc
        st.markdown(f"""
<div style="background:#1e2130;border-radius:8px;padding:6px 10px;margin-top:4px;">
  <div style="display:grid;grid-template-columns:1fr auto 1fr;gap:4px;align-items:start;">
    <div>
      <div style="color:#666;font-size:0.62rem;margin-bottom:3px;">⚡ 단타 구성</div>
      <div style="display:flex;justify-content:space-between;font-size:0.75rem;"><span style="color:#aaa;">기술(70%)</span><b style="color:{tc};">{tech_score:+.1f}</b></div>
      <div style="display:flex;justify-content:space-between;font-size:0.75rem;margin-top:2px;"><span style="color:#aaa;">뉴스(30%)</span><b style="color:{nc};">{news_score:+.1f}</b></div>
    </div>
    <div style="width:1px;background:#2a2d3e;align-self:stretch;margin:2px 6px;"></div>
    <div>
      <div style="color:#666;font-size:0.62rem;margin-bottom:3px;">🏛️ 장투 구성</div>
      <div style="display:flex;justify-content:space-between;font-size:0.75rem;"><span style="color:#aaa;">펀더멘털</span><b style="color:{fc_color};">{fs:+.1f}</b></div>
      <div style="font-size:0.62rem;color:#555;margin-top:3px;">PER·PBR·ROE·FCF ▶</div>
    </div>
  </div>
</div>
        """, unsafe_allow_html=True)

        if not close.empty and len(close) >= 2:
            last_price = float(close.iloc[-1])
            prev_price = float(close.iloc[-2])
            daily_chg  = (last_price - prev_price) / prev_price * 100
            _is_krw  = last_price > 500
            _fmt     = "{:,.0f}" if _is_krw else "{:,.2f}"
            _chg_clr = "#a5d6a7" if daily_chg >= 0 else "#ef9a9a"
            _chg_sym = "▲" if daily_chg >= 0 else "▼"
            st.markdown(f"""
<div style="display:flex;justify-content:space-between;align-items:center;
            background:#1e2130;border-radius:8px;padding:5px 12px;margin-top:4px;">
  <span style="color:#888;font-size:0.72rem;">💰 현재가</span>
  <span style="font-size:1.05rem;font-weight:bold;color:#e0e0e0;">{_fmt.format(last_price)}</span>
  <span style="font-size:0.78rem;font-weight:bold;color:{_chg_clr};">{_chg_sym} {abs(daily_chg):.2f}%</span>
</div>
            """, unsafe_allow_html=True)

            # ── 매수 / 매도 적정가 (2-column) ──────────────────────────────
            _bt = get_buy_target_price(data)
            _st_data = get_sell_target_price(data)
            _buy_card = ""
            _sell_card = ""

            if _bt:
                _t_price = _bt["buy_target"]
                _gap     = _bt["gap_pct"]
                _timing  = _bt["timing"]
                _tc_clr  = _bt["timing_color"]
                _bb_l    = _bt["bb_lower"]
                _s20     = _bt["sma20"]
                _l5      = _bt["low5"]
                _gap_clr = "#ef9a9a" if _gap > 5 else ("#fff176" if _gap > 0 else "#69f0ae")
                _buy_card = f"""
<div style="background:#12161f;border:1px solid #2a2d3e;border-radius:8px;padding:8px 10px;">
  <div style="font-size:0.62rem;color:#888;margin-bottom:4px;">🎯 매수 적정가</div>
  <div style="font-size:1.05rem;font-weight:bold;color:#e0e0e0;">{_fmt.format(_t_price)}</div>
  <div style="font-size:0.7rem;color:{_gap_clr};margin:2px 0;">현재가 {_gap:+.1f}%</div>
  <div style="font-size:0.68rem;font-weight:bold;color:{_tc_clr};">{_timing}</div>
  <div style="border-top:1px solid #1e2130;padding-top:4px;margin-top:5px;font-size:0.6rem;color:#555;line-height:1.6;">
    BB하단 {_fmt.format(_bb_l)}<br>SMA20 {_fmt.format(_s20)} · 5일저 {_fmt.format(_l5)}
  </div>
</div>"""

            if _st_data:
                _cons     = _st_data["conservative_target"]
                _aggr     = _st_data["aggressive_target"]
                _cons_gap = _st_data["cons_gap"]
                _aggr_gap = _st_data["aggr_gap"]
                _s_timing = _st_data["sell_timing"]
                _s_color  = _st_data["sell_color"]
                _bb_u     = _st_data["bb_upper"]
                _res      = _st_data["resistance_level"]
                _fib_val  = _st_data["fib_1618"]
                _is_nh    = _st_data["is_new_high"]
                _cons_clr = "#a5d6a7" if _cons_gap > 5 else ("#fff176" if _cons_gap > 0 else "#ef9a9a")
                _aggr_clr = "#a5d6a7" if _aggr_gap > 10 else ("#fff176" if _aggr_gap > 0 else "#ef9a9a")
                _nh_badge = '<span style="background:#1a237e;color:#82b1ff;font-size:0.58rem;padding:1px 5px;border-radius:8px;margin-left:4px;">신고가</span>' if _is_nh else ""
                _ref_line = f"BB상단 {_fmt.format(_bb_u)}" + (f"<br>매물대 {_fmt.format(_res)}" if _res else "") + (f" · 피보 {_fmt.format(_fib_val)}" if _fib_val else "")
                _sell_card = f"""
<div style="background:#1a0f0a;border:1px solid #3d1f0f;border-radius:8px;padding:8px 10px;">
  <div style="font-size:0.62rem;color:#888;margin-bottom:4px;">📤 매도 적정가{_nh_badge}</div>
  <div style="display:flex;justify-content:space-between;gap:4px;">
    <div>
      <div style="font-size:0.58rem;color:#777;">보수적</div>
      <div style="font-size:0.88rem;font-weight:bold;color:#ffccbc;">{_fmt.format(_cons)}</div>
      <div style="font-size:0.68rem;color:{_cons_clr};">{_cons_gap:+.1f}%</div>
    </div>
    <div style="text-align:right;">
      <div style="font-size:0.58rem;color:#777;">공격적</div>
      <div style="font-size:0.88rem;font-weight:bold;color:#ff8a65;">{_fmt.format(_aggr)}</div>
      <div style="font-size:0.68rem;color:{_aggr_clr};">{_aggr_gap:+.1f}%</div>
    </div>
  </div>
  <div style="font-size:0.65rem;font-weight:bold;color:{_s_color};margin-top:4px;">{_s_timing}</div>
  <div style="border-top:1px solid #3d1f0f;padding-top:4px;margin-top:5px;font-size:0.6rem;color:#555;line-height:1.6;">
    {_ref_line}
  </div>
</div>"""

            if _buy_card or _sell_card:
                st.markdown(f"""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:5px;margin-top:4px;">
  {_buy_card}
  {_sell_card}
</div>
                """, unsafe_allow_html=True)

        if expected:
            _M    = expected["expected_return_pct"]
            _A    = expected.get("return_low",  _M - 5.0)
            _B    = expected.get("return_high", _M + 5.0)
            _beta = expected.get("beta", 1.0)
            _atr  = expected.get("atr_pct", 0.0)
            _kelly = expected.get("kelly_pct", 0.0)
            _winp  = expected.get("win_prob", 50.0)
            _vpvr  = expected.get("vpvr_resistance", False)

            _m_clr = "#a5d6a7" if _M >= 0 else "#ef9a9a"
            _a_clr = "#ef9a9a" if _A < 0 else "#fff176"
            _b_clr = "#a5d6a7" if _B >= 0 else "#ef9a9a"
            _rng = _B - _A
            _m_pos  = max(2, min(98, int((_M - _A) / _rng * 100))) if _rng > 0 else 50
            _z_pos  = max(0, min(100, int((0  - _A) / _rng * 100))) if _rng > 0 else 50
            _beta_clr = "#a5d6a7" if _beta >= 1.0 else "#fff176"
            _kelly_bar = min(int(_kelly * 2), 100)
            _rr_str   = f"{abs(_B) / max(abs(_A), 0.1):.1f}:1" if _A < 0 else "—"
            _vpvr_note = '<div style="font-size:0.62rem;color:#ffab91;margin-top:2px;">⚠️ 목표가 부근 매물대 저항 — B 하향 보정</div>' if _vpvr else ""

            st.markdown(f"""
<div style="background:#1a1d2e;border-radius:8px;padding:8px 12px;margin-top:4px;border:1px solid #2a2d3e;">
  <div style="font-size:0.62rem;color:#888;margin-bottom:5px;">📈 예상 수익률 구간 (20거래일)</div>
  <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
    <div style="text-align:center;">
      <div style="font-size:0.58rem;color:#777;">최저 A</div>
      <div style="font-size:0.82rem;font-weight:bold;color:{_a_clr};">{_A:+.1f}%</div>
    </div>
    <div style="text-align:center;">
      <div style="font-size:0.58rem;color:#777;">중간값 M</div>
      <div style="font-size:0.92rem;font-weight:bold;color:{_m_clr};">{_M:+.1f}%</div>
    </div>
    <div style="text-align:center;">
      <div style="font-size:0.58rem;color:#777;">최고 B{'⚠️' if _vpvr else ''}</div>
      <div style="font-size:0.82rem;font-weight:bold;color:{_b_clr};">{_B:+.1f}%</div>
    </div>
  </div>
  <div style="position:relative;background:#2a2d3e;border-radius:3px;height:5px;margin-bottom:2px;">
    <div style="position:absolute;left:{_z_pos}%;top:0;width:1px;height:100%;background:rgba(255,255,255,0.25);"></div>
    <div style="position:absolute;left:{_m_pos}%;top:-2px;width:3px;height:9px;background:{_m_clr};border-radius:2px;transform:translateX(-50%);"></div>
  </div>
  {_vpvr_note}
  <div style="border-top:1px solid #2a2d3e;padding-top:5px;margin-top:5px;
              display:grid;grid-template-columns:repeat(4,1fr);gap:2px;text-align:center;">
    <div>
      <div style="font-size:0.58rem;color:#666;">켈리 비중</div>
      <b style="font-size:0.78rem;color:#90caf9;">{_kelly:.1f}%</b>
      <div style="background:#2a2d3e;border-radius:2px;height:3px;margin-top:2px;">
        <div style="background:#90caf9;width:{_kelly_bar}%;height:3px;border-radius:2px;"></div>
      </div>
    </div>
    <div><div style="font-size:0.58rem;color:#666;">승률 추정</div><b style="font-size:0.78rem;color:#aaa;">{_winp:.0f}%</b></div>
    <div><div style="font-size:0.58rem;color:#666;">W/L</div><b style="font-size:0.78rem;color:#aaa;">{_rr_str}</b></div>
    <div><div style="font-size:0.58rem;color:#666;">β 베타</div><b style="font-size:0.78rem;color:{_beta_clr};">{_beta:+.2f}</b></div>
  </div>
</div>
            """, unsafe_allow_html=True)

            with st.expander("📐 상세 통계 지표", expanded=False):
                st.metric("연간 변동성", f"{expected['hist_volatility']:.1f}%",
                          help="과거 일간 수익률의 표준편차 × √252 (연환산)")
                st.metric("20일 모멘텀", f"{expected['momentum_20d']:+.2f}%",
                          help="20거래일 전 대비 현재가 등락률")
                st.metric("최대 낙폭", f"{expected['max_drawdown']:.2f}%",
                          help="조회 기간 내 고점 대비 최대 하락폭 (MDD)")
                st.metric("샤프 지수", f"{expected['sharpe']:.2f}",
                          help="무위험 수익률(3.5%) 초과 수익 ÷ 변동성 (연환산)")

        # ── VWAP 멀티 타임프레임 카드 (Shannon) ─────────────────────────────
        if not data.empty and len(data) >= 2:
            _cur = float(data["Close"].iloc[-1])
            _is_krw = _cur > 500
            _pf = "{:,.0f}" if _is_krw else "{:,.2f}"
            _vw_row  = data.iloc[-1]

            def _vwap_row_html(label, col, color):
                if col not in data.columns or pd.isna(_vw_row[col]):
                    return ""
                _v = float(_vw_row[col])
                _diff = (_cur - _v) / _v * 100
                _arrow = "▲" if _diff >= 0 else "▼"
                _dc = "#69f0ae" if _diff >= 0 else "#ef9a9a"
                return (
                    f'<div style="display:flex;justify-content:space-between;'
                    f'align-items:center;padding:4px 0;border-bottom:1px solid #2a2d3e;">'
                    f'<span style="font-size:0.72rem;color:{color};">● {label}</span>'
                    f'<span style="font-size:0.75rem;color:#ddd;">{_pf.format(_v)}</span>'
                    f'<span style="font-size:0.72rem;color:{_dc};">{_arrow}{abs(_diff):.1f}%</span>'
                    f'</div>'
                )

            # VWAP 스택 방향 판별
            _vw = float(_vw_row["VWAP_W"]) if "VWAP_W" in data.columns and pd.notna(_vw_row["VWAP_W"]) else None
            _vm = float(_vw_row["VWAP_M"]) if "VWAP_M" in data.columns and pd.notna(_vw_row["VWAP_M"]) else None
            _vq = float(_vw_row["VWAP_Q"]) if "VWAP_Q" in data.columns and pd.notna(_vw_row["VWAP_Q"]) else None

            if _vw and _vm and _vq:
                if _vw > _vm > _vq:
                    _stack_txt = "📶 상승 스택 — 단·중·장기 강세 정렬"
                    _stack_clr = "#69f0ae"
                elif _vw < _vm < _vq:
                    _stack_txt = "📉 하락 스택 — 단·중·장기 약세 정렬"
                    _stack_clr = "#ef9a9a"
                else:
                    _stack_txt = "↔ 혼조 — 타임프레임 간 방향 불일치"
                    _stack_clr = "#fff176"
            else:
                _stack_txt, _stack_clr = "데이터 부족", "#888"

            _vwap_rows = (
                _vwap_row_html("주간 VWAP (5봉)",  "VWAP_W", "#ff8f00") +
                _vwap_row_html("월간 VWAP (20봉)", "VWAP_M", "#ce93d8") +
                _vwap_row_html("분기 VWAP (60봉)", "VWAP_Q", "#80deea")
            )
            st.markdown(f"""
<div style="background:#12161f;border:1px solid #2a2d3e;border-radius:10px;
            padding:11px 14px;margin-bottom:6px;">
  <div style="font-size:0.68rem;color:#888;letter-spacing:0.5px;margin-bottom:6px;">
    📊 VWAP 다중 타임프레임
    <span style="float:right;font-size:0.6rem;color:#555;">Shannon, 2008</span>
  </div>
  {_vwap_rows}
  <div style="margin-top:7px;font-size:0.74rem;font-weight:bold;color:{_stack_clr};">
    {_stack_txt}
  </div>
  <div style="font-size:0.65rem;color:#555;margin-top:3px;">
    수식: Σ(Typical Price × Volume) / Σ(Volume) — 롤링 누적합
  </div>
</div>
            """, unsafe_allow_html=True)

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
            ("VWAP 주간",  "VWAP_W",    ",.0f"),
            ("VWAP 월간",  "VWAP_M",    ",.0f"),
            ("VWAP 분기",  "VWAP_Q",    ",.0f"),
        ]
        last_row = data.iloc[-1]
        for lbl, col, fmt in indicator_map:
            if col in data.columns:
                val = last_row[col]
                if pd.notna(val):
                    st.caption(f"**{lbl}:** {float(val):{fmt}}")

        # ── 종합 판단 스코어보드 ──────────────────────────────────────────────
        st.divider()
        with st.expander("📊 종합 판단 스코어보드", expanded=True):
            ts = advanced.get("trend_score",    50.0)
            ms = advanced.get("momentum_score", 50.0)
            vs = advanced.get("volume_score",   50.0)

            def _score_color(s):
                if s >= 65: return "#a5d6a7"
                if s <= 35: return "#ef9a9a"
                return "#fff176"

            def _score_bar(label, score, weight, desc):
                color = _score_color(score)
                pct   = int(score)
                st.markdown(
                    f'<div style="margin-bottom:8px;">'
                    f'<div style="display:flex;justify-content:space-between;font-size:0.78rem;">'
                    f'<span style="color:#ccc;">{label} <span style="color:#666;">({weight})</span></span>'
                    f'<b style="color:{color};">{score:.0f}점</b></div>'
                    f'<div style="background:#2a2d3e;border-radius:4px;height:7px;margin-top:3px;">'
                    f'<div style="background:{color};width:{pct}%;height:7px;border-radius:4px;"></div>'
                    f'</div>'
                    f'<div style="font-size:0.72rem;color:#666;margin-top:2px;">{desc}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            _score_bar("추세 (Trend)",     ts, "40%", "EMA 배열 · ADX · 일목균형표")
            _score_bar("탄력 (Momentum)",  ms, "30%", "MACD · RSI · ROC · CCI")
            _score_bar("에너지 (Volume)",  vs, "30%", "OBV · MFI")

            composite_adv = round(ts * 0.4 + ms * 0.3 + vs * 0.3, 1)
            if composite_adv >= 65:
                adv_txt, adv_clr = "강세 우위 — 에너지와 추세 모두 양호", "#a5d6a7"
            elif composite_adv <= 35:
                adv_txt, adv_clr = "약세 우위 — 지표 전반 약화 중", "#ef9a9a"
            else:
                adv_txt, adv_clr = "중립 — 방향 확인 필요", "#fff176"
            st.markdown(
                f'<div style="background:#1e2130;border-radius:6px;padding:8px 12px;margin-top:6px;">'
                f'<span style="font-size:0.75rem;color:#888;">가중 종합: </span>'
                f'<b style="color:{adv_clr};">{composite_adv:.0f}점 — {adv_txt}</b>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # 추가 분석 항목 테이블
            items = advanced.get("summary_items", [])
            if items:
                st.markdown('<div style="margin-top:10px;font-size:0.75rem;color:#888;">추가 분석 항목</div>',
                            unsafe_allow_html=True)
                for it in items:
                    st.markdown(
                        f'<div style="display:flex;justify-content:space-between;'
                        f'background:#12141f;border-radius:5px;padding:5px 10px;margin-top:4px;'
                        f'border-left:3px solid {it["색상"]};">'
                        f'<span style="color:#9e9e9e;font-size:0.77rem;">{it["항목"]}</span>'
                        f'<span style="color:{it["색상"]};font-size:0.77rem;font-weight:bold;">{it["상태"]}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            # 다이버전스 상세 설명
            div_descs = advanced.get("divergence", {}).get("descriptions", [])
            if div_descs:
                for d in div_descs:
                    if "하락" in d:
                        st.warning(d, icon="⚠️")
                    else:
                        st.success(d, icon="✅")

        # ── 손절·익절 레벨 ────────────────────────────────────────────────
        st.divider()
        sl = get_stop_loss_targets(data)
        if sl:
            _is_krw_sl = sl["current"] > 500
            _sfmt = "{:,.0f}" if _is_krw_sl else "{:,.2f}"
            st.markdown(f"""
<div style="background:#0d1117;border:1px solid #2a2d3e;border-radius:10px;
            padding:11px 14px;margin-bottom:4px;">
  <div style="font-size:0.68rem;color:#888;letter-spacing:0.5px;margin-bottom:8px;">
    🛡️ 손절·익절 가이드
    <span style="float:right;font-size:0.62rem;color:#555;">오닐 원칙 + ATR</span>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;">
    <div style="background:#1a0808;border-radius:6px;padding:6px 9px;">
      <div style="font-size:0.61rem;color:#888;">🔴 손절 (8% 룰)</div>
      <div style="font-size:0.95rem;font-weight:bold;color:#ef9a9a;">{_sfmt.format(sl['stop_8pct'])}</div>
    </div>
    <div style="background:#1a0f08;border-radius:6px;padding:6px 9px;">
      <div style="font-size:0.61rem;color:#888;">🟠 손절 (ATR×2.5)</div>
      <div style="font-size:0.95rem;font-weight:bold;color:#ffab91;">{_sfmt.format(sl['stop_atr'])}</div>
    </div>
    <div style="background:#0f1a08;border-radius:6px;padding:6px 9px;">
      <div style="font-size:0.61rem;color:#888;">🟡 1차 목표 (2R)</div>
      <div style="font-size:0.95rem;font-weight:bold;color:#fff176;">{_sfmt.format(sl['target_2r'])}</div>
    </div>
    <div style="background:#091a08;border-radius:6px;padding:6px 9px;">
      <div style="font-size:0.61rem;color:#888;">🟢 2차 목표 (3R)</div>
      <div style="font-size:0.95rem;font-weight:bold;color:#a5d6a7;">{_sfmt.format(sl['target_3r'])}</div>
    </div>
  </div>
  <div style="font-size:0.67rem;color:#555;margin-top:7px;text-align:right;">
    ATR {sl['atr_ratio']:.2f}% · {_sfmt.format(sl['atr'])}
  </div>
</div>
            """, unsafe_allow_html=True)

        # ── 내 주식 분석 ────────────────────────────────────────────────────────
        st.divider()
        st.markdown("**🧑‍💼 내 주식 분석**")

        _avg_key   = f"avg_price_{ticker}"
        _saved_avg = st.session_state.get(_avg_key, 0.0)
        _step      = 100.0 if ticker.endswith((".KS", ".KQ")) else 0.01

        avg_price_input = st.number_input(
            "평단가",
            min_value=0.0,
            value=float(_saved_avg),
            step=_step,
            format="%.2f" if _step < 1 else "%.0f",
            placeholder="평단가 입력",
            label_visibility="collapsed",
            key=f"avg_input_{ticker}",
        )

        if st.button("📊 내 주식 분석", use_container_width=True,
                     type="primary", key=f"my_analyze_{ticker}"):
            if avg_price_input <= 0:
                st.warning("평단가를 입력해주세요.")
            elif close.empty:
                st.warning("주가 데이터가 없습니다.")
            else:
                st.session_state[_avg_key] = avg_price_input
                _last_price = float(close.iloc[-1])
                _last_row   = data.iloc[-1]

                def _safe(col):
                    v = _last_row.get(col) if col in data.columns else None
                    return float(v) if v is not None and pd.notna(v) else None

                _indicators = {
                    "RSI":       _safe("RSI")       or 50.0,
                    "MACD_Hist": _safe("MACD_Hist") or 0.0,
                    "ADX":       _safe("ADX")        or 20.0,
                    "BB_Lower":  _safe("BB_Lower"),
                    "BB_Upper":  _safe("BB_Upper"),
                }

                rec = get_investment_recommendation(
                    current_price = _last_price,
                    avg_price     = avg_price_input,
                    indicators    = _indicators,
                    tech_score    = tech_score,
                    news_score    = news_score,
                    fund_score    = fs,
                )
                st.session_state[f"rec_{ticker}"] = rec

        # 분석 결과 표시 (버튼 클릭 후 세션에 저장된 결과 렌더링)
        _rec = st.session_state.get(f"rec_{ticker}")
        if _rec:
            _pr   = _rec["profit_rate"]
            _pbg  = "#1b5e20" if _pr > 0 else ("#b71c1c" if _pr < 0 else "#1e2130")
            _pfc  = "#a5d6a7" if _pr > 0 else ("#ef9a9a" if _pr < 0 else "#bdbdbd")
            _psgn = "+" if _pr > 0 else ""

            st.markdown(f"""
            <div style="background:{_rec['color_bg']};border-radius:12px;
                        padding:14px 16px;margin-top:8px;
                        border:1px solid {_rec['color_fg']}44;">
                <div style="font-size:1.3rem;font-weight:bold;
                            color:{_rec['color_fg']};margin-bottom:4px;">
                    {_rec['badge']} {_rec['title']}
                </div>
                <div style="display:inline-block;background:{_pbg};
                            border-radius:6px;padding:2px 10px;
                            font-size:0.85rem;color:{_pfc};margin-bottom:8px;">
                    수익률 {_psgn}{_pr:.2f}%
                </div>
                <div style="font-size:0.83rem;color:#ccc;line-height:1.6;">
                    {_rec['reason']}
                </div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("📋 상세 분석 근거", expanded=True):
                for _d in _rec["details"]:
                    if any(k in _d for k in ["손실", "⚠️", "✂️", "음수", "하락"]):
                        st.error(_d, icon="🔻")
                    elif any(k in _d for k in ["수익", "💡", "✅", "양수", "상승", "우량"]):
                        st.success(_d, icon="🔺")
                    else:
                        st.info(_d, icon="🔷")

    # ── 투자자별 매매 동향 (Naver Finance) ──────────────────────────────────────
    _is_krx_chart = _aticker and (_aticker.endswith(".KS") or _aticker.endswith(".KQ"))
    if _data_ready and _is_krx_chart:
        st.markdown("---")
        st.markdown("### 👥 전날 투자자별 매매 동향")
        with st.spinner("투자자 데이터 조회 중..."):
            _inv = get_investor_trading_naver(_aticker)
        if _inv:
            _inv_date = _inv.get("date", "")
            if len(_inv_date) == 8:
                _inv_date = f"{_inv_date[:4]}.{_inv_date[4:6]}.{_inv_date[6:]}"
            st.caption(f"기준일: {_inv_date}  |  단위: 주(株)  |  출처: Naver Finance")
            _inv_cols = st.columns(3)
            for col, key, label in zip(
                _inv_cols,
                ["외국인", "기관합계", "개인"],
                ["🌐 외국인", "🏦 기관", "👤 개인"],
            ):
                val = _inv.get(key)
                if val is not None:
                    _sign = "+" if val > 0 else ""
                    _color = "#ef5350" if val > 0 else ("#42a5f5" if val < 0 else "#bdbdbd")
                    col.markdown(
                        f'<div style="background:#1e2130;padding:14px;border-radius:10px;text-align:center;">'
                        f'<div style="font-size:0.75rem;color:#888;margin-bottom:4px;">{label}</div>'
                        f'<div style="font-size:1.1rem;font-weight:bold;color:{_color};">{_sign}{val:,}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    col.markdown(
                        f'<div style="background:#1e2130;padding:14px;border-radius:10px;text-align:center;">'
                        f'<div style="font-size:0.75rem;color:#888;margin-bottom:4px;">{label}</div>'
                        f'<div style="font-size:1.1rem;color:#555;">N/A</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
        else:
            st.caption("투자자 매매 동향 데이터를 불러올 수 없습니다.")

