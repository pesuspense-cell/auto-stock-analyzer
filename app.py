"""
app.py - AI 주식 분석 대시보드 v2.0
실시간 차트 · 매매 신호 · 시장 현황 · 추천 종목 · 펀더멘털 · 관심종목
"""
import json
import os
import queue
import concurrent.futures
import time
import streamlit as st
from fundamental_db import load_settings_db, save_settings_db
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timezone, timedelta
import warnings
warnings.filterwarnings("ignore")

# 한국 표준시 (UTC+9) — pytz 불필요
_KST = timezone(timedelta(hours=9))

def _now_kst() -> datetime:
    """현재 KST 시각 반환."""
    return datetime.now(_KST)

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
        check_dead_time, check_breakout_signal, adjust_risk_conservative,
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
        analyze_news_fast, fetch_naver_news_fast, analyze_portfolio_news,
    )
    from src.utils import (
        KOSPI_STOCKS, US_STOCKS, INDICES,
        get_market_movers, get_full_market_movers, get_exchange_rates,
        get_investor_trading_naver, get_recommendations,
        get_krx_stock_list, get_krx_etf_list, get_us_stock_list,
        get_top_kospi_stocks, get_top_kosdaq_stocks,
        get_top_us_stocks, get_top_nasdaq_stocks,
        is_etf_ticker, _ETF_PORTFOLIO_MAP,
        _flatten_columns,
    )
except Exception as _import_err:
    import traceback as _tb
    st.error(
        f"**모듈 로딩 오류** — 아래 전체 트레이스백을 확인하세요:\n\n"
        f"```\n{_tb.format_exc()}\n```"
    )
    st.stop()

# ─── 포트폴리오 DB 직접 연동 ─────────────────────────────────────────────────
from src.database import (
    init_db as _db_init,
    register_user as _db_register,
    login_user as _db_login,
    logout_user as _db_logout,
    get_user_by_token as _db_get_user,
    add_portfolio as _db_add_portfolio,
    upsert_portfolio as _db_upsert_portfolio,
    get_portfolio as _db_get_portfolio,
    delete_portfolio_item as _db_delete_portfolio,
    save_recommendation as _db_save_recommendation,
    get_recommendation_history as _db_get_rec_history,
    sell_item as _db_sell_item,
    get_trade_history as _db_get_trade_history,
    get_trade_summary as _db_get_trade_summary,
    clear_trade_history as _db_clear_trade_history,
)
try:
    _db_init()
    pass  # DB 연결 성공 — 사이드바 최하단에서 비활성 항목만 표시
except Exception as _db_err:
    st.error(f"Supabase 연결 실패: {_db_err}\n\nStreamlit Secrets에 SUPABASE_DB_URL을 설정하세요.", icon="🚨")
    st.stop()


# ─── 페이지 설정 ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI 주식 분석 터미널",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── 쿠키 매니저 (브라우저별 독립 세션 저장소) ──────────────────────────────
try:
    import extra_streamlit_components as _stx
    _cookie_mgr = _stx.CookieManager(key="_auth_cm")
    _HAS_COOKIE_MGR = True
except Exception:
    _cookie_mgr = None
    _HAS_COOKIE_MGR = False

def _inject_midnight_aurora_css() -> None:
    """Midnight Aurora 전역 테마 CSS — static/midnight_aurora.css에서 로드."""
    _css_path = os.path.join(os.path.dirname(__file__), "static", "midnight_aurora.css")
    try:
        with open(_css_path, "r", encoding="utf-8") as _f:
            st.markdown(f"<style>{_f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass  # CSS 파일 없으면 기본 스타일로 동작

_inject_midnight_aurora_css()

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
    "krx_id":         "KRX_ID",
    "krx_pw":         "KRX_PW",
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

# 포트폴리오 인증 세션
for _sk in ("auth_token", "auth_user_id", "auth_email"):
    if _sk not in st.session_state:
        st.session_state[_sk] = None

_saved_settings = load_settings()

# 재시작 후 브라우저 쿠키로 로그인 세션 복원 (기기별 독립 세션)
if not st.session_state.get("auth_token"):
    try:
        _boot_auth_token = st.context.cookies.get("auth_token", "")
    except Exception:
        _boot_auth_token = ""
    if _boot_auth_token:
        _boot_user = _db_get_user(_boot_auth_token)
        if _boot_user:
            st.session_state["auth_token"]   = _boot_auth_token
            st.session_state["auth_user_id"] = _boot_user["id"]
            st.session_state["auth_email"]   = _boot_user["email"]

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

def _get_full_stocks(market: str) -> dict:
    """시장 전체 종목 사전 반환 — 인위적 상한 없음."""
    if market == "KOSPI":
        return _top_kospi(500)
    elif market == "KOSDAQ":
        return _top_kosdaq(500)
    elif market == "미국 주식 (나스닥)":
        return _top_nasdaq(500)
    else:
        return _top_us(503)

@st.cache_data(ttl=300)
def _index_data(sym):
    d = yf.download(sym, period="2d", auto_adjust=True, progress=False)
    d = _flatten_columns(d)
    return d

@st.cache_data(ttl=3600)
def _fundamental(ticker):
    info = get_fundamental_data(ticker)
    return info

@st.cache_data(ttl=3600)
def _insider_trades(ticker):
    return get_insider_trades_sec(ticker)

@st.cache_data(ttl=600)
def _naver_news(ticker: str) -> list:
    # httpx 비동기 수집 우선 시도, 실패 시 requests 폴백
    try:
        items = fetch_naver_news_fast(ticker, max_items=12)
        if items:
            return items
    except Exception:
        pass
    return get_naver_news(ticker, max_items=12)

@st.cache_data(ttl=300)
def _dead_time(ticker: str) -> dict:
    return check_dead_time(ticker)

@st.cache_data(ttl=300)
def _sector_perf(ticker: str) -> dict:
    return get_related_sector_performance(ticker)

@st.cache_data(ttl=3600)
def _bench_returns(ticker: str) -> "pd.Series":
    """S&P500 또는 KOSPI 6개월 일간 수익률 (베타 계산용)"""
    sym = "^KS11" if (ticker.endswith(".KS") or ticker.endswith(".KQ")) else "^GSPC"
    try:
        d = yf.download(sym, period="6mo", auto_adjust=True, progress=False)
        d = _flatten_columns(d)
        return d["Close"].pct_change().dropna()
    except Exception:
        return pd.Series(dtype=float)

@st.cache_data(ttl=600)
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
    ticker: str,
    api_key: str,
    groq_api_key: str = "",
    company_name: str = "",
    price_change_pct: float | None = None,
    net_foreign_buy: float | None = None,
    net_institution_buy: float | None = None,
) -> dict:
    """
    LLM 뉴스 분석 — API 키가 세션마다 다를 수 있어 session_state 캐시 사용.
    analyze_news_fast()를 통해 3단계 필터 + 비동기 수집 파이프라인 사용.
    캐시 TTL: 10분 (기존 1시간에서 단축)
    """
    cache_key = (
        f"llm_news|{ticker}|{api_key[:8] if api_key else ''}"
        f"|{groq_api_key[:8] if groq_api_key else ''}|{company_name}"
    )
    cached = st.session_state.get(cache_key)
    if cached and (datetime.now() - cached["ts"]).seconds < 600:
        return cached["data"]

    # analyze_news_fast: 비동기 수집 + 3단계 필터 + LLM 상위 5건만 처리
    result = analyze_news_fast(
        ticker=ticker,
        company_name=company_name,
        api_key=api_key,
        groq_api_key=groq_api_key,
        max_news=12,
        deep_n=5,
        price_change_pct=price_change_pct,
        net_foreign_buy=net_foreign_buy,
        net_institution_buy=net_institution_buy,
    )

    # analyze_news_fast가 뉴스를 못 가져온 경우 yfinance 뉴스로 폴백
    if not result.get("detail"):
        try:
            raw = yf.Ticker(ticker).news or []
            items = []
            for it in raw[:10]:
                c = it.get("content", it)
                items.append({
                    "title":     c.get("title", it.get("title", "")),
                    "link":      (c.get("canonicalUrl", {}).get("url") or it.get("link", "#")),
                    "publisher": (c.get("provider", {}).get("displayName") or it.get("publisher", "")),
                    "pub_date":  "",
                })
            if items:
                result = analyze_news_sentiment_llm(items, ticker, api_key, groq_api_key, company_name)
        except Exception:
            pass

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

# ─── 종목 목록 백그라운드 사전 로딩 ──────────────────────────────────────────────
if "stock_lists_preloaded" not in st.session_state:
    st.session_state["stock_lists_preloaded"] = True
    _preload_ex = concurrent.futures.ThreadPoolExecutor(max_workers=3, thread_name_prefix="preload")
    _preload_ex.submit(_krx_stocks)
    _preload_ex.submit(_etf_stocks)
    _preload_ex.submit(_us_stocks)
    _preload_ex.shutdown(wait=False)

@st.cache_data(ttl=86400)
def _ticker_name_map() -> dict:
    """티커 → 종목명 역방향 맵 (KRX 주식 + ETF + 미국 주식 통합, 일 1회 캐시)."""
    result: dict[str, str] = {}
    # KRX 주식: "삼성전자 (005930)" → "005930.KS"
    for display, ticker in (_krx_stocks() or {}).items():
        result[ticker] = display.split(" (")[0].strip()
    # ETF: "KODEX 200 (069500)" → "069500.KS"
    for display, ticker in (_etf_stocks() or {}).items():
        result[ticker] = display.split(" (")[0].strip()
    # 미국: "애플 / Apple Inc (AAPL) [S&P500]" → "AAPL"
    for display, ticker in (_us_stocks() or {}).items():
        parts = display.split(" / ")
        result[ticker] = parts[0].strip() if len(parts) > 1 else display.split(" (")[0].strip()
    return result

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

@st.cache_data(ttl=60)
def _etf_fundamental(ticker: str) -> dict:
    return get_etf_fundamental_data(ticker)

@st.cache_data(ttl=300)
def _inv_data(ticker: str) -> dict:
    return get_investor_trading_naver(ticker) or {}

@st.cache_data(ttl=60)
def _realtime_price_1m(ticker: str) -> dict:
    """
    현재가 조회 — fast_info → 1분봉 → 일봉 순으로 폴백 (TTL=60초).

    Returns: {"price": float, "ts": str, "is_realtime": bool, "stale": bool, "stale_msg": str}
    """
    _now_k   = _now_kst()
    _ts      = _now_k.strftime("%H:%M:%S")
    _is_kr   = ticker.endswith(".KS") or ticker.endswith(".KQ")
    _after_close = _is_kr and (_now_k.hour * 60 + _now_k.minute) >= (15 * 60 + 30)

    # ── 1순위: fast_info.last_price (Quote API — 가장 빠름) ──────────────────
    try:
        _fi = yf.Ticker(ticker).fast_info
        _p  = float(_fi.last_price)
        if _p > 0:
            _stale = _after_close
            return {
                "price":       _p,
                "ts":          _ts,
                "is_realtime": not _stale,
                "stale":       _stale,
                "stale_msg":   "장 마감 후 종가입니다." if _stale else "",
            }
    except Exception:
        pass

    # ── 2순위: 1분봉 OHLCV ───────────────────────────────────────────────────
    try:
        _df = yf.download(ticker, period="1d", interval="1m",
                          auto_adjust=True, progress=False)
        _df = _flatten_columns(_df)
        if not _df.empty and "Close" in _df.columns:
            if _df.index.tz is None:
                _df.index = _df.index.tz_localize("UTC").tz_convert("Asia/Seoul")
            else:
                _df.index = _df.index.tz_convert("Asia/Seoul")
            _close_s = _df["Close"].dropna()
            if not _close_s.empty:
                _p = float(_close_s.iloc[-1])
                if _p > 0:
                    _last_dt   = _close_s.index[-1]
                    _now_aware = _now_k.replace(tzinfo=_KST) if _now_k.tzinfo is None else _now_k
                    _age_h     = (_now_aware - _last_dt).total_seconds() / 3600
                    _stale     = _age_h > 15
                    _stale_msg = (
                        f"장이 열리지 않은 상태입니다 (마지막 데이터: {_last_dt.strftime('%m/%d %H:%M')} KST). "
                        "가장 최근 종가를 표시합니다."
                    ) if _stale else ""
                    return {
                        "price":       _p,
                        "ts":          _ts,
                        "is_realtime": not (_after_close or _stale),
                        "stale":       _stale,
                        "stale_msg":   _stale_msg,
                    }
    except Exception:
        pass

    # ── 3순위: 일봉 종가 ─────────────────────────────────────────────────────
    try:
        _d = yf.download(ticker, period="2d", auto_adjust=True, progress=False)
        _d = _flatten_columns(_d)
        if not _d.empty and "Close" in _d.columns:
            _p = float(_d["Close"].dropna().iloc[-1])
            if _p > 0:
                return {"price": _p, "ts": _ts, "is_realtime": False,
                        "stale": True, "stale_msg": "장이 열리지 않은 상태입니다. 가장 최근 종가를 표시합니다."}
    except Exception:
        pass

    return {"price": 0.0, "ts": _ts, "is_realtime": False, "stale": False, "stale_msg": ""}

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
            if data.empty or len(data) < 2 or "Close" not in data.columns:
                continue
            sig   = generate_signals(data)
            score = sig.get("score", 0)
            if abs(score) >= 3:
                price = float(data["Close"].iloc[-1])
                prev  = float(data["Close"].iloc[-2])
                if prev == 0:
                    continue
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
            _krx_q = st.text_input(
                "종목 검색",
                key="_krx_q",
                placeholder="종목명 또는 코드 입력 (예: 삼성전자, 005930)",
            )
            _krx_opts = (
                {k: v for k, v in krx.items() if _krx_q.lower() in k.lower()}
                if _krx_q else krx
            )
            if not _krx_opts:
                st.caption("검색 결과 없음 — 전체 목록 표시")
                _krx_opts = krx
            selected = st.selectbox(
                f"종목 선택 ({len(_krx_opts):,}개)",
                list(_krx_opts.keys()),
                key="_krx_selected",
                on_change=_clear_analysis,
            )
            ticker = _krx_opts[selected]
            sname  = selected.split(" (")[0]
        else:
            st.warning("종목 목록 로드 실패 — 기본 목록 사용")
            sname  = st.selectbox("종목", list(KOSPI_STOCKS.keys()),
                                  key="_krx_fallback", on_change=_clear_analysis)
            ticker = KOSPI_STOCKS[sname]

    elif market_sel == "국내 ETF (검색)":
        with st.spinner("ETF 목록 준비 중..."):
            etf_list = _etf_stocks()

        if etf_list:
            _etf_q = st.text_input(
                "ETF 검색",
                key="_etf_q",
                placeholder="ETF명 또는 코드 입력 (예: KODEX 200, 069500)",
            )
            _etf_opts = (
                {k: v for k, v in etf_list.items() if _etf_q.lower() in k.lower()}
                if _etf_q else etf_list
            )
            if not _etf_opts:
                st.caption("검색 결과 없음 — 전체 목록 표시")
                _etf_opts = etf_list
            etf_selected = st.selectbox(
                f"ETF 선택 ({len(_etf_opts):,}개)",
                list(_etf_opts.keys()),
                key="_etf_selected",
                on_change=_clear_analysis,
            )
            ticker = _etf_opts[etf_selected]
            sname  = etf_selected.split(" (")[0]
        else:
            st.warning("ETF 목록 로드 실패 — 기본 목록 사용")
            _etf_fb = {f"{v['name']} ({k})": f"{k}.KS" for k, v in _ETF_PORTFOLIO_MAP.items()}
            sname   = st.selectbox("ETF", list(_etf_fb.keys()), key="_etf_fallback",
                                   on_change=_clear_analysis)
            ticker  = _etf_fb[sname]
            sname   = sname.split(" (")[0]

        st.info("ETF는 기술적 분석 + ETF 전용 지표(괴리율·운용보수)로 분석됩니다.", icon="📊")

    elif market_sel == "미국 주식 (검색)":
        with st.spinner("미국 종목 목록 준비 중... (S&P500 + 나스닥)"):
            us_list = _us_stocks()

        if us_list:
            _us_q = st.text_input(
                "종목 검색",
                key="_us_q",
                placeholder="회사명 또는 티커 입력 (예: Apple, AAPL, 애플)",
            )
            _us_opts = (
                {k: v for k, v in us_list.items() if _us_q.lower() in k.lower()}
                if _us_q else us_list
            )
            if not _us_opts:
                st.caption("검색 결과 없음 — 전체 목록 표시")
                _us_opts = us_list
            us_selected = st.selectbox(
                f"종목 선택 ({len(_us_opts):,}개)",
                list(_us_opts.keys()),
                key="_us_selected",
                on_change=_clear_analysis,
            )
            ticker = _us_opts[us_selected]
            sname  = us_selected.split(" (")[0]
        else:
            st.warning("미국 종목 목록 로드 실패 — 기본 목록 사용")
            sname  = st.selectbox("종목", list(US_STOCKS.keys()),
                                  key="_us_fallback", on_change=_clear_analysis)
            ticker = US_STOCKS[sname]

    else:
        ticker = st.text_input(
            "티커 직접 입력",
            value=st.session_state.get("_direct_ticker_input", "005930.KS"),
            key="_direct_ticker_input",
            on_change=_clear_analysis,
            help="예) 005930.KS (KOSPI), 247540.KQ (KOSDAQ), AAPL (미국) — 아래 버튼 클릭",
        )
        sname  = ticker

    period = st.selectbox("분석 기간", ["1mo", "3mo", "6mo", "1y", "2y"],
                          index=1, key="_period_sel", on_change=_clear_analysis)

    # ── 분석 시작 버튼 ─────────────────────────────────────────────────────
    st.divider()
    if st.button("🔍 종목 분석 시작", type="primary", use_container_width=True):
        st.session_state.pop("analyzed_ticker", None)
        st.session_state["_pending_ticker"]  = ticker
        st.session_state["_pending_sname"]   = sname
        st.session_state["_pending_period"]  = period
        st.rerun()

    # ── API 키 변수 로드 (상태 알림은 사이드바 최하단에서 비활성 시만 표시) ──────
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
                    d2 = _flatten_columns(d2)
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

    # ── 포트폴리오 종목 추가 (사이드바 하단 — 로그인 시만 표시) ──────────────────
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
            _sb_market = st.radio(
                "시장", ["국내 주식", "국내 ETF", "미국 주식"],
                horizontal=True, label_visibility="collapsed", key="sb_add_market",
            )
            _sb_ticker_val = ""
            if _sb_market == "국내 주식":
                _sb_list = _krx_stocks() or {}
                if _sb_list:
                    _sbq = st.text_input("종목 검색", key="sb_add_krx_q",
                                         placeholder="삼성전자, 005930 ...")
                    _sb_opts = ({k: v for k, v in _sb_list.items() if _sbq.lower() in k.lower()}
                                if _sbq else _sb_list) or _sb_list
                    _sb_sel = st.selectbox(f"선택 ({len(_sb_opts):,}개)",
                                           list(_sb_opts.keys()), key="sb_add_krx")
                    _sb_ticker_val = _sb_opts[_sb_sel]
            elif _sb_market == "국내 ETF":
                _sb_list = _etf_stocks() or {}
                if _sb_list:
                    _sbq = st.text_input("ETF 검색", key="sb_add_etf_q",
                                         placeholder="KODEX 200, 069500 ...")
                    _sb_opts = ({k: v for k, v in _sb_list.items() if _sbq.lower() in k.lower()}
                                if _sbq else _sb_list) or _sb_list
                    _sb_sel = st.selectbox(f"선택 ({len(_sb_opts):,}개)",
                                           list(_sb_opts.keys()), key="sb_add_etf")
                    _sb_ticker_val = _sb_opts[_sb_sel]
            else:
                _sb_list = _us_stocks() or {}
                if _sb_list:
                    _sbq = st.text_input("종목 검색", key="sb_add_us_q",
                                         placeholder="Apple, AAPL, 애플 ...")
                    _sb_opts = ({k: v for k, v in _sb_list.items() if _sbq.lower() in k.lower()}
                                if _sbq else _sb_list) or _sb_list
                    _sb_sel = st.selectbox(f"선택 ({len(_sb_opts):,}개)",
                                           list(_sb_opts.keys()), key="sb_add_us")
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
                    _sb_r = _db_upsert_portfolio(_sb_uid, _sb_ticker_val, _sb_price, _sb_qty)
                    if _sb_r.get("merged"):
                        st.toast(f"추가 매수 완료 — 평단가 자동 합산")
                    else:
                        st.toast(f"{_sb_ticker_val} 포트폴리오에 추가됐습니다.")
                    st.rerun()

    # ── 시스템 연동 상태 (비활성 항목만 표시) ──────────────────────────────────
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

# ─── 헤더 + 포트폴리오 대시보드 요약 카드 ────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;gap:14px;margin-bottom:18px;margin-top:4px">
  <span class="gradient-text" style="font-size:1.75rem;font-weight:900">AI 주식 분석 대시보드</span>
  <span style="font-size:.72rem;color:#8B5CF6;padding:3px 12px;
               background:rgba(139,92,246,0.12);border:1px solid rgba(139,92,246,0.35);
               border-radius:20px;font-weight:700;letter-spacing:1px">PREMIUM</span>
</div>
""", unsafe_allow_html=True)

# ── 포트폴리오 요약 3카드 (포트폴리오 탭 방문 후 갱신) ──────────────────────────
_pf_hdr  = st.session_state.get("_pf_header_summary", {})
_h_val   = _pf_hdr.get("total_val")
_h_pnl   = _pf_hdr.get("total_pnl", 0.0)
_h_pnlp  = _pf_hdr.get("total_pnl_pct", 0.0)
_h_ovr   = _pf_hdr.get("overall_pct")

_SVG_WALLET = ('<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24"'
               ' fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"'
               ' stroke-linejoin="round"><path d="M21 12V7H5a2 2 0 0 1 0-4h14v4"/>'
               '<path d="M3 5v14a2 2 0 0 0 2 2h16v-5"/>'
               '<path d="M18 12a2 2 0 0 0 0 4h4v-4Z"/></svg>')
_SVG_CHART  = ('<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24"'
               ' fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"'
               ' stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/>'
               '<line x1="12" y1="20" x2="12" y2="4"/>'
               '<line x1="6" y1="20" x2="6" y2="14"/></svg>')
_SVG_TREND  = ('<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24"'
               ' fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"'
               ' stroke-linejoin="round"><polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/>'
               '<polyline points="16 7 22 7 22 13"/></svg>')

_hc1, _hc2, _hc3 = st.columns(3)
if _h_val is not None:
    _pnl_c  = "#10B981" if _h_pnl  >= 0 else "#3B82F6"
    _ovr_c  = "#10B981" if (_h_ovr or 0) >= 0 else "#3B82F6"
    _pnl_rgb = "16,185,129" if _h_pnl >= 0 else "59,130,246"
    _ovr_rgb = "16,185,129" if (_h_ovr or 0) >= 0 else "59,130,246"
    _hc1.markdown(f"""
<div class="ma-header-card" style="border:1px solid rgba(139,92,246,0.25)">
  <div style="position:absolute;top:-15px;right:-15px;width:70px;height:70px;
              background:radial-gradient(circle,rgba(139,92,246,.18) 0%,transparent 70%);border-radius:50%"></div>
  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:14px">
    <span style="font-size:.73rem;color:#94A3B8;font-weight:500;letter-spacing:.8px;text-transform:uppercase">총 평가금</span>
    <span style="color:#8B5CF6">{_SVG_WALLET}</span>
  </div>
  <div style="font-size:1.6rem;font-weight:800;color:#E2E8F0;line-height:1.1">₩{_h_val:,.0f}</div>
  <div style="font-size:.74rem;color:#8B5CF6;font-weight:600;margin-top:10px">포트폴리오 현재가치</div>
</div>
""", unsafe_allow_html=True)
    _hc2.markdown(f"""
<div class="ma-header-card" style="border:1px solid rgba({_pnl_rgb},.25)">
  <div style="position:absolute;top:-15px;right:-15px;width:70px;height:70px;
              background:radial-gradient(circle,rgba({_pnl_rgb},.15) 0%,transparent 70%);border-radius:50%"></div>
  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:14px">
    <span style="font-size:.73rem;color:#94A3B8;font-weight:500;letter-spacing:.8px;text-transform:uppercase">미실현 손익</span>
    <span style="color:{_pnl_c}">{_SVG_CHART}</span>
  </div>
  <div style="font-size:1.6rem;font-weight:800;color:{_pnl_c};line-height:1.1">₩{_h_pnl:+,.0f}</div>
  <div style="font-size:.74rem;color:{_pnl_c};font-weight:600;margin-top:10px">{_h_pnlp:+.2f}% 수익률</div>
</div>
""", unsafe_allow_html=True)
    _hc3.markdown(f"""
<div class="ma-header-card" style="border:1px solid rgba({_ovr_rgb},.25)">
  <div style="position:absolute;top:-15px;right:-15px;width:70px;height:70px;
              background:radial-gradient(circle,rgba({_ovr_rgb},.15) 0%,transparent 70%);border-radius:50%"></div>
  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:14px">
    <span style="font-size:.73rem;color:#94A3B8;font-weight:500;letter-spacing:.8px;text-transform:uppercase">누적 수익률</span>
    <span style="color:{_ovr_c}">{_SVG_TREND}</span>
  </div>
  <div style="font-size:1.6rem;font-weight:800;color:{_ovr_c};line-height:1.1">{f"{_h_ovr:+.2f}%" if _h_ovr is not None else "—"}</div>
  <div style="font-size:.74rem;color:{_ovr_c};font-weight:600;margin-top:10px">매도 이력 포함 전체 기간</div>
</div>
""", unsafe_allow_html=True)
else:
    _placeholder_html = (
        '<div class="ma-header-card" style="border:1px solid rgba(255,255,255,0.06);text-align:center">'
        '<div style="font-size:1.5rem;font-weight:700;color:#1E293B;margin-bottom:8px">—</div>'
        '<div style="font-size:.72rem;color:#475569">💼 포트폴리오 탭 접속 후 갱신</div></div>'
    )
    _hc1.markdown(_placeholder_html, unsafe_allow_html=True)
    _hc2.markdown(_placeholder_html, unsafe_allow_html=True)
    _hc3.markdown(_placeholder_html, unsafe_allow_html=True)

st.markdown("<div style='margin-bottom:6px'></div>", unsafe_allow_html=True)

# 탭 위에 항상 표시되는 로딩 배너 플레이스홀더
_loading_ph = st.empty()

# ─── 탭 레이아웃 (데이터 로딩 전 정의 — 탭 내 로딩 상태 표시용) ─────────────────
tab_market, tab_chart, tab_rec, tab_news, tab_fund, tab_portfolio = st.tabs([
    "🌐 시장 현황",
    "📊 차트 분석",
    "⭐ 추천 종목",
    "📰 뉴스 & 관련 종목",
    "🏛️ 펀더멘털 & 기관",
    "💼 내 포트폴리오",
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
    st.caption(f"업데이트: {_now_kst().strftime('%Y-%m-%d %H:%M:%S')}  |  분석 중: **{_pname}** (`{_pending}`)")
    _load_start = time.time()
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
  <p style="color:#93c5fd; margin:8px 0 0; font-size:13px;">⏱ <b>0s</b> 경과</p>
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

    st.caption(f"업데이트: {_now_kst().strftime('%Y-%m-%d %H:%M:%S')}  |  분석 종목: **{_asname}** (`{_aticker}`)")

    # ── 주가 데이터 + 펀더멘털 병렬 로딩 ────────────────────────────────────
    _load_start = time.time()
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
  <p style="color:#93c5fd; margin:8px 0 0; font-size:13px;">⏱ <b>0s</b> 경과</p>
</div>
""", unsafe_allow_html=True)
    # 병렬 로딩 시작 (spinner 제거하고 실시간 업데이트)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as _pool:
        _f_data = _pool.submit(_stock_data, _aticker, _aperiod)
        _f_fund = _pool.submit(_fundamental, _aticker)
        # 작업 완료까지 매초마다 경과 시간 업데이트
        while not (_f_data.done() and _f_fund.done()):
            _elapsed = int(time.time() - _load_start)
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
  <p style="color:#93c5fd; margin:8px 0 0; font-size:13px;">⏱ <b>{_elapsed}s</b> 경과</p>
</div>
""", unsafe_allow_html=True)
            time.sleep(1)
        # 작업 결과 수집
        data      = _f_data.result()
        fund_info = _f_fund.result()
    _load_elapsed = int(time.time() - _load_start)
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
  <p style="color:#93c5fd; margin:8px 0 0; font-size:13px;">⏱ <b>{_load_elapsed}s</b> 경과</p>
</div>
""", unsafe_allow_html=True)

    if data.empty or "Close" not in data.columns:
        st.session_state.pop("analyzed_ticker", None)
        st.error(f"'{_aticker}' 데이터를 불러올 수 없습니다. 티커를 확인해주세요.")
        st.stop()

    # data["Close"] 가 중복 컬럼으로 DataFrame 반환될 수 있으므로 Series로 보장
    _close_raw = data["Close"]
    close = _close_raw.iloc[:, 0] if isinstance(_close_raw, pd.DataFrame) else _close_raw

    def _compute_signals_and_fund():
        try:
            _va  = check_volume_anomaly(data)
            _sig = generate_signals(data)
            if _va.get("is_halted"):
                _sig = {
                    "score":   0,
                    "label":   "거래 정지/주의",
                    "badge":   "⛔",
                    "reasons": [_va.get("reason", "")],
                    "_halted": True,
                }
            _adv = get_advanced_analysis(data)
            _exp = calculate_expected_return(
                data, _sig,
                ticker=_aticker,
                benchmark_returns=_bench_returns(_aticker),
            )
            _last = float(close.iloc[-1]) if not close.empty else 0.0
            _fsd = calculate_fundamental_score(fund_info, _last)
            return _va, _sig, _adv, _exp, _fsd
        except Exception as _e:
            import logging
            logging.getLogger(__name__).warning(f"[compute] {_aticker}: {type(_e).__name__}: {_e}")
            return {}, {"score": 0, "label": "분석 오류", "badge": "⚠️", "reasons": []}, {}, {}, {}

    # ── 신호 연산 + 뉴스 감성을 병렬 실행, 완료까지 타이머 갱신 ─────────────────
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as _pool2:
        _f_comp = _pool2.submit(_compute_signals_and_fund)
        _f_news = _pool2.submit(
            _news_sentiment_llm_cached if use_llm else _news_sentiment_kw,
            ticker,
            *(gemini_api_key, groq_api_key) if use_llm else ()
        )
        while not (_f_comp.done() and _f_news.done()):
            _elapsed = int(time.time() - _load_start)
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
  <p style="color:#93c5fd; margin:8px 0 0; font-size:13px;">⏱ <b>{_elapsed}s</b> 경과</p>
</div>
""", unsafe_allow_html=True)
            time.sleep(1)
        try:
            vol_anomaly, signals, advanced, expected, fund_score_data = _f_comp.result()
        except Exception as _ce:
            import logging
            logging.getLogger(__name__).warning(f"[comp.result] {_aticker}: {_ce}")
            vol_anomaly, signals, advanced, expected, fund_score_data = (
                {}, {"score": 0, "label": "분석 오류", "badge": "⚠️", "reasons": []}, {}, {}, {}
            )
        try:
            news_result = _f_news.result()
        except Exception as _ne:
            import logging
            logging.getLogger(__name__).warning(f"[news.result] {_aticker}: {_ne}")
            news_result = {}

    news_score = news_result.get("score", 0.0) if isinstance(news_result, dict) else 0.0
    tech_score = signals.get("score", 0)       if isinstance(signals, dict)     else 0
    hybrid     = get_hybrid_signal(tech_score, news_score)
    dead_time  = _dead_time(_aticker)

    # ── 샤프 가드: 예상수익률 ≥50% 이지만 샤프지수 < 0.5 → 신호 레이블 '주의' 강제 ──
    if expected:
        _exp_ret = expected.get("expected_return_pct", 0.0)
        _sharpe  = expected.get("sharpe", 1.0)
        if _exp_ret >= 50.0 and _sharpe < 0.5:
            hybrid = {**hybrid, "label": "주의", "badge": "🟡"}
    breakout   = check_breakout_signal(data)
    risk_adj   = adjust_risk_conservative(expected) if expected else {}
    _total_elapsed = int(time.time() - _load_start)
    _loading_ph.markdown(f"""
<div style="background:linear-gradient(135deg,#1a1f3a 0%,#242b4d 100%);
            border:2px solid #3b82f6; border-radius:16px;
            padding:28px 24px; text-align:center; margin:8px 0 16px;
            box-shadow:0 4px 20px rgba(59,130,246,0.25);">
  <div class="loading-icon" style="font-size:40px;">✅</div>
  <h3 style="color:#60a5fa; margin:12px 0 8px;">분석 완료</h3>
  <p style="color:#94a3b8; margin:0; line-height:1.7;">
    <span style="color:#e2e8f0; font-weight:bold;">{_asname}</span>
    데이터 분석이 완료되었습니다.
  </p>
  <div class="loading-bar-track"><div class="loading-bar-fill" style="animation:none;background:#60a5fa;"></div></div>
  <p style="color:#93c5fd; margin:8px 0 0; font-size:13px;">⏱ 총 소요 시간: <b>{_total_elapsed}s</b></p>
</div>
""", unsafe_allow_html=True)
    _loading_ph.empty()

    # ── 실시간 현재가 (1분봉) — 버튼 클릭 직후만 호출, 60초 캐시 ────────────
    _rt = _realtime_price_1m(_aticker)
    _rt_price     = _rt["price"]
    _rt_ts        = _rt["ts"]
    _rt_realtime  = _rt["is_realtime"]
    _rt_stale     = _rt.get("stale", False)
    _rt_stale_msg = _rt.get("stale_msg", "")

else:
    st.caption(f"업데이트: {_now_kst().strftime('%Y-%m-%d %H:%M:%S')}  |  사이드바에서 종목을 선택하고 분석을 시작하세요.")
    _rt_price     = 0.0
    _rt_ts        = ""
    _rt_realtime  = False
    _rt_stale     = False
    _rt_stale_msg = ""
    data            = pd.DataFrame()
    vol_anomaly     = {"is_halted": False}
    signals         = {"score": 0, "label": "분석 대기", "badge": "—"}
    expected        = None
    advanced        = {"trend_score": 50.0, "momentum_score": 50.0, "volume_score": 50.0,
                       "divergence": {}, "zscore": None, "vpvr": {}, "ichimoku": {}, "summary_items": []}
    close           = pd.Series(dtype=float, name="Close")
    fund_info       = {}
    fund_score_data = {"fund_score": 0, "fund_label": "분석 대기", "fund_reasons": []}
    dead_time       = {"is_dead": False, "message": ""}
    breakout        = {"status": "wait", "detail": ""}
    risk_adj        = {}
    hybrid          = {"hybrid_score": 0.0, "label": "중립/관망", "badge": "⚪"}
    news_result     = {}
    tech_score      = 0
    news_score      = 0.0

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
def _article_dialog(title: str, link: str, ticker_sym: str, api_key: str, groq_key: str = "") -> None:
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
        result = summarize_article_llm(title, link, ticker_sym, api_key, groq_key)

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
        st.warning("시장 데이터를 불러올 수 없습니다. (KRX 서버 응답 없음 또는 네트워크 오류)", icon="⚠️")

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
            fx = _flatten_columns(fx)
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

    # ── 주요 섹터 ETF 실시간 등락표 (35종목) ────────────────────────────────
    st.divider()
    st.markdown("### 🗺️ 주요 섹터 ETF 실시간 등락표")
    st.caption("미국 ETF 15개 + 국내 섹터별 대표 ETF 20개 · 전일 대비 등락률 · 10분 캐시")

    # (ticker, ETF명, 국가, 태그)  ← .KS/.KQ 접미사를 리스트에서 명시적으로 지정
    _SECTOR_ETF_LIST = (
        # ── 미국 ETF 15개 ─────────────────────────────────────────────────────
        ("SPY",  "S&P 500",         "미국", "📊 지수"),
        ("QQQ",  "나스닥 100",       "미국", "📊 지수"),
        ("DIA",  "다우존스",         "미국", "📊 지수"),
        ("SCHD", "배당성장",         "미국", "💰 배당"),
        ("SOXX", "반도체 (SOX)",     "미국", "💻 반도체"),
        ("VGT",  "기술주",           "미국", "🤖 테크"),
        ("BOTZ", "AI/로봇",          "미국", "🤖 AI/로봇"),
        ("XLV",  "헬스케어",         "미국", "🏥 헬스케어"),
        ("XLF",  "금융",             "미국", "🏦 금융"),
        ("XLE",  "에너지",           "미국", "⚡ 에너지"),
        ("GDX",  "금광주",           "미국", "⛏ 원자재"),
        ("LIT",  "2차전지/리튬",     "미국", "🔋 2차전지"),
        ("TLT",  "미국채 20년",      "미국", "📋 채권"),
        ("BIL",  "미국채 단기",      "미국", "📋 채권"),
        ("VNQ",  "리츠",             "미국", "🏢 리츠"),
        # ── 국내 섹터별 대표 ETF 20개 ─────────────────────────────────────────
        ("069500.KS", "KODEX 200",                    "국내", "📊 코스피"),
        ("229200.KQ", "KODEX 코스닥150",              "국내", "📊 코스닥"),    # .KQ 주의
        ("396500.KS", "TIGER Fn반도체TOP10",          "국내", "💻 반도체"),
        ("464930.KS", "SOL AI반도체소부장",           "국내", "💻 반도체소부장"),
        ("464520.KS", "KoAct 바이오헬스케어액티브",  "국내", "🧬 바이오"),
        ("305710.KS", "KODEX 2차전지산업",            "국내", "🔋 2차전지"),
        ("091180.KS", "KODEX 자동차",                 "국내", "🚗 자동차"),
        ("466920.KS", "SOL 조선TOP3플러스",           "국내", "🚢 조선"),
        ("449450.KS", "PLUS K방산",                   "국내", "🛡 방산"),
        ("484310.KS", "KODEX AI전력핵심설비",         "국내", "⚡ 전력"),
        ("438100.KS", "TIGER 코리아원자력",           "국내", "⚛ 원전"),
        ("440790.KS", "KODEX 로봇액티브",             "국내", "🤖 로봇"),
        ("414270.KS", "PLUS 우주항공&UAM",            "국내", "🚀 우주"),
        ("408220.KS", "PLUS 태양광&ESS",              "국내", "☀ 태양광"),
        ("385530.KS", "KODEX 신재생에너지액티브",     "국내", "🌱 재생에너지"),
        ("228790.KS", "TIGER 화장품",                 "국내", "💄 화장품"),
        ("475580.KS", "ACE KPOP포커스",               "국내", "🎵 엔터"),
        ("300640.KS", "TIGER 지주회사",               "국내", "🏢 지주사"),
        ("466940.KS", "TIGER 은행고배당플러스TOP10",  "국내", "🏦 은행"),
        ("102960.KS", "KODEX 증권",                   "국내", "📈 증권"),
    )

    @st.cache_data(ttl=600, show_spinner=False)
    def _fetch_sector_etfs(etf_list: tuple) -> pd.DataFrame:
        tickers = [row[0] for row in etf_list]
        # .KS/.KQ 누락 방지 — 숫자 6자리만 있으면 .KS 자동 보완
        def _fix_ticker(t: str) -> str:
            if t.isdigit() and len(t) == 6:
                return t + ".KS"
            return t
        tickers = [_fix_ticker(t) for t in tickers]

        try:
            raw = yf.download(tickers, period="2d", auto_adjust=True, progress=False)
        except Exception:
            return pd.DataFrame()
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                close = raw["Close"]
            elif "Close" in raw.columns:
                close = raw[["Close"]]
                close.columns = tickers[:1]
            else:
                return pd.DataFrame()
            if len(close) < 2:
                return pd.DataFrame()
            rows = []
            for orig_ticker, name, market, tag in etf_list:
                ticker = _fix_ticker(orig_ticker)
                try:
                    if ticker not in close.columns:
                        continue
                    series = close[ticker].dropna()
                    if len(series) < 2:
                        continue
                    prev = float(series.iloc[-2])
                    curr = float(series.iloc[-1])
                    if prev == 0:
                        continue
                    chg = round((curr - prev) / prev * 100, 2)
                    is_kr = ticker.endswith(".KS") or ticker.endswith(".KQ")
                    price_str = (f"{curr:,.0f}₩" if is_kr else f"${curr:,.2f}")
                    rows.append({
                        "국가":      market,
                        "태그":      tag,
                        "ETF명":     name,
                        "티커":      ticker,
                        "현재가":    price_str,
                        "방향":      "🔺" if chg >= 0 else "🔻",
                        "등락률(%)": chg,
                    })
                except Exception:
                    continue
            return pd.DataFrame(rows)
        except Exception:
            return pd.DataFrame()

    def _render_etf_table(view: pd.DataFrame) -> None:
        """등락률 행 색상(빨강/파랑) + ProgressColumn 막대 렌더링"""
        cols = ["국가", "태그", "ETF명", "티커", "현재가", "방향", "등락률(%)"]

        def _row_style(row):
            chg = row["등락률(%)"]
            if isinstance(chg, (int, float)) and chg > 0:
                bg = "rgba(239,83,80,0.13)"
            elif isinstance(chg, (int, float)) and chg < 0:
                bg = "rgba(66,165,245,0.13)"
            else:
                bg = "transparent"
            return [f"background-color:{bg}"] * len(row)

        styled = view[cols].style.apply(_row_style, axis=1)
        st.dataframe(
            styled,
            use_container_width=True,
            hide_index=True,
            column_config={
                "국가":   st.column_config.TextColumn("국가",   width="small"),
                "태그":   st.column_config.TextColumn("테마",   width="medium"),
                "ETF명":  st.column_config.TextColumn("ETF명",  width="large"),
                "티커":   st.column_config.TextColumn("티커",   width="small"),
                "현재가": st.column_config.TextColumn("현재가", width="small"),
                "방향":   st.column_config.TextColumn("↕",      width="small"),
                "등락률(%)": st.column_config.ProgressColumn(
                    "등락률 (%)",
                    format="%+.2f%%",
                    min_value=-15.0,
                    max_value=15.0,
                    help="전일 대비 등락률 · 막대 중앙(50%) = 0% 기준",
                ),
            },
        )

    with st.spinner("섹터 ETF 데이터 로딩 중 (35종목)..."):
        _etf_df = _fetch_sector_etfs(_SECTOR_ETF_LIST)

    if not _etf_df.empty:
        # 요약 메트릭 (5열)
        _etf_up  = int((_etf_df["등락률(%)"] > 0).sum())
        _etf_dn  = int((_etf_df["등락률(%)"] < 0).sum())
        _etf_avg = float(_etf_df["등락률(%)"].mean())
        _us_df   = _etf_df[_etf_df["국가"] == "미국"]["등락률(%)"]
        _kr_df   = _etf_df[_etf_df["국가"] == "국내"]["등락률(%)"]
        _us_avg  = float(_us_df.mean()) if not _us_df.empty else 0.0
        _kr_avg  = float(_kr_df.mean()) if not _kr_df.empty else 0.0
        _em1, _em2, _em3, _em4, _em5 = st.columns(5)
        _em1.metric("🔺 상승",       f"{_etf_up}개")
        _em2.metric("🔻 하락",       f"{_etf_dn}개")
        _em3.metric("전체 평균",     f"{_etf_avg:+.2f}%")
        _em4.metric("🇺🇸 미국 평균", f"{_us_avg:+.2f}%")
        _em5.metric("🇰🇷 국내 평균", f"{_kr_avg:+.2f}%")

        def _sort_index_first(df: pd.DataFrame, *cols: str) -> pd.DataFrame:
            """📊 지수 태그를 맨 위로, 나머지는 cols 순 정렬."""
            tmp = df.assign(_pri=(~df["태그"].str.startswith("📊")).astype(int))
            return tmp.sort_values(["_pri", *cols]).drop(columns=["_pri"])

        _etf_t1, _etf_t2, _etf_t3 = st.tabs(
            ["전체 (35종목)", "🇺🇸 미국 ETF (15종목)", "🇰🇷 국내 ETF (20종목)"]
        )
        with _etf_t1:
            _render_etf_table(_sort_index_first(_etf_df, "국가", "태그", "ETF명"))
        with _etf_t2:
            _us = _etf_df[_etf_df["국가"] == "미국"]
            _render_etf_table(_sort_index_first(_us, "태그")) if not _us.empty else st.caption("미국 ETF 데이터 없음")
        with _etf_t3:
            _kr = _etf_df[_etf_df["국가"] == "국내"]
            _render_etf_table(_sort_index_first(_kr, "태그")) if not _kr.empty else st.caption("국내 ETF 데이터 없음")
    else:
        st.warning("ETF 데이터를 불러올 수 없습니다. (네트워크 오류 또는 장 마감 시간)", icon="⚠️")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3  추천 종목
# ══════════════════════════════════════════════════════════════════════════════
with tab_rec:
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

    # 라디오 변경으로 인한 rerun에서 분석이 실행되지 않도록 버튼 클릭 시에만 플래그를 세운다.
    if run_btn:
        st.session_state["_rec_run_requested"] = True

    # ── 다단계 로딩바 플레이스홀더 (탭 내부 전용) ─────────────────────────────
    _rec_ph = st.empty()

    if st.session_state.get("_rec_run_requested"):
        st.session_state["_rec_run_requested"] = False

        stocks = _get_full_stocks(rec_market)
        total_stocks = len(stocks)

        _rec_q     = queue.Queue()
        _rec_start = time.time()

        # 단계별 아이콘/메시지 매핑
        _stage_icons = {0: "🔍", 1: "📡", 2: "📊", 3: "🎯", 4: "✅"}
        _stage_state = {
            "stage": 0,
            "icon":  "🔍",
            "title": "전 종목 AI 분석 준비 중",
            "msg":   f"시장 전체 {total_stocks}개 종목 데이터 수집 준비 중...",
        }

        def _render_rec_bar(state: dict, elapsed: int, done: bool = False):
            bar_style = "animation:none;background:#60a5fa;" if done else ""
            _rec_ph.markdown(f"""
<div style="background:linear-gradient(135deg,#1a1f3a 0%,#242b4d 100%);
            border:2px solid #3b82f6; border-radius:16px;
            padding:28px 24px; text-align:center; margin:8px 0 16px;
            box-shadow:0 4px 20px rgba(59,130,246,0.25);">
  <div class="loading-icon" style="font-size:40px;">{state["icon"]}</div>
  <h3 style="color:#60a5fa; margin:12px 0 8px;">{state["title"]}</h3>
  <p style="color:#94a3b8; margin:0; line-height:1.7;">{state["msg"]}</p>
  <div class="loading-bar-track">
    <div class="loading-bar-fill" style="{bar_style}"></div>
  </div>
  <p style="color:#93c5fd; margin:8px 0 0; font-size:13px;">
    ⏱ <b>{elapsed}s</b> 경과 &nbsp;|&nbsp; 전체 {total_stocks}개 종목 전수 분석
  </p>
</div>""", unsafe_allow_html=True)

        _render_rec_bar(_stage_state, 0)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _rec_pool:
            _rec_future = _rec_pool.submit(get_recommendations, stocks, _rec_q)

            while not _rec_future.done():
                # 큐에서 단계 업데이트 수신
                try:
                    while True:
                        upd = _rec_q.get_nowait()
                        s = upd.get("stage", 0)
                        if s == 1:
                            _stage_state.update({
                                "stage": 1, "icon": "📡", "title": "단계 1 · 데이터 수집 완료",
                                "msg": (
                                    f"✅ {upd.get('fetched','?')}/{upd.get('total','?')}개 OHLCV 수집 완료 → "
                                    "L1 차트 스크리닝 진행 중..."
                                ),
                            })
                        elif s == 2:
                            _stage_state.update({
                                "stage": 2, "icon": "📊", "title": "단계 2 · 정밀 차트 분석 중",
                                "msg": (
                                    f"L1 통과 <b>{upd.get('l1_count','?')}개</b> 종목 — "
                                    "RSI·MACD·ADX·일목 등 12개 지표 정밀 채점 중..."
                                ),
                            })
                        elif s == 3:
                            _stage_state.update({
                                "stage": 3, "icon": "🎯", "title": "단계 3 · 뉴스·재무 심층 분석 중",
                                "msg": (
                                    f"차트 상위 <b>{upd.get('l2_count','?')}개</b> 종목 — "
                                    "뉴스 감성·Dead-time·기대수익률 병렬 분석 중..."
                                ),
                            })
                except queue.Empty:
                    pass

                _elapsed = int(time.time() - _rec_start)
                _render_rec_bar(_stage_state, _elapsed)
                time.sleep(1)

            rec_df = _rec_future.result()

        _total_elapsed = int(time.time() - _rec_start)
        _stage_state.update({
            "stage": 4, "icon": "✅", "title": "분석 완료",
            "msg": f"최종 추천 종목 <b>{len(rec_df) if rec_df is not None else 0}개</b> 선정 완료.",
        })
        _render_rec_bar(_stage_state, _total_elapsed, done=True)
        time.sleep(0.8)
        _rec_ph.empty()

        st.session_state["rec_df"]     = rec_df
        st.session_state["rec_market"] = rec_market

    rec_df = st.session_state.get("rec_df", None)

    if rec_df is not None and not rec_df.empty:
        # 구버전 세션 캐시 호환 — 새로 추가된 컬럼이 없으면 기본값으로 채움
        for _col, _default in [("뉴스점수", 0.0), ("기술점수", 0.0), ("chart_precision_score", 0.0)]:
            if _col not in rec_df.columns:
                rec_df[_col] = _default

        # ── 요약 메트릭 (종합점수 기준) ─────────────────────────────────────
        rec_n     = int((rec_df["종합점수"] >= 1).sum())
        neutral_n = int(((rec_df["종합점수"] > 0) & (rec_df["종합점수"] < 1)).sum())
        caution_n = int((rec_df["종합점수"] <= 0).sum())
        pos_ret_n = int((rec_df["예상수익률(%)"] > 0).sum())

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("분석 종목", f"{len(rec_df)}개",
                  help="이번 분석에서 데이터를 불러온 종목 수입니다.")
        m2.metric("🟢 매수 신호 (종합 +1↑)", f"{rec_n}개",
                  help="종합점수(기술70%+뉴스30%) ≥ 1 — 약한 매수 이상 종목 수.")
        m3.metric("⚪ 중립 (0~1)", f"{neutral_n}개",
                  help="종합점수 0 이상 1 미만 — 관망 구간.")
        m4.metric("📈 수익률 양수 종목", f"{pos_ret_n}개",
                  help="예상수익률(20일)이 플러스인 종목 수.")

        # ── 종합점수 산정 방식 안내 ──────────────────────────────────────────
        with st.expander("ℹ️ 종합점수 산정 방식", expanded=False):
            st.markdown("""
            **종합점수 = 기술점수 × 0.7 + 뉴스감성 × 0.3** (차트분석 탭과 동일 공식)

            | 항목 | 범위 | 역할 |
            |------|------|------|
            | 기술점수 | -10 ~ +10 | RSI·MACD·EMA·ADX·일목·Z-Score·다이버전스 등 11개 지표 복합 신호 |
            | 뉴스감성 | -5 ~ +5 | yfinance 뉴스 키워드 감성 분석 (A/B/C 등급 가중 합산) |

            | 종합점수 | 신호 |
            |----------|------|
            | ≥ +5 | 🟢🟢 강력 매수 |
            | +3 ~ +5 | 🟢 매수 |
            | +1 ~ +3 | 🔵 약한 매수 |
            | 0 | ⚪ 중립/관망 |
            | -2 ~ 0 | 🟡 약한 매도 |
            | -4 ~ -2 | 🔴 매도 |
            | < -4 | 🔴🔴 강력 매도 |

            > 뉴스 수집 실패 시 뉴스감성 0점(중립)으로 처리됩니다.
            """)

        st.divider()
        st.markdown("### 📋 종목별 종합 분석표")

        def _row_style(row):
            s = row["종합점수"]
            if s >= 5.0: return ["background-color:#1b5e20"] * len(row)
            if s >= 3.0: return ["background-color:#2e7d32"] * len(row)
            if s >= 1.0: return ["background-color:#1a3a2a"] * len(row)
            if s <= -5.0: return ["background-color:#b71c1c"] * len(row)
            if s <= -3.0: return ["background-color:#c62828"] * len(row)
            return [""] * len(row)

        disp = ["종목명", "현재가", "등락률(1일)%", "종합추천", "종합점수",
                "기술점수", "뉴스점수", "예상수익률(%)", "변동성(%)", "모멘텀(20일)%", "샤프지수"]
        styled = (
            rec_df[disp].style
            .apply(_row_style, axis=1)
            .format({
                "현재가":         "{:,.0f}",
                "등락률(1일)%":   "{:+.2f}",
                "종합점수":       "{:+.2f}",
                "기술점수":       "{:+.1f}",
                "뉴스점수":       "{:+.2f}",
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
        **종합점수 = 기술점수(70%) + 뉴스감성(30%)** — 차트분석 탭과 동일 공식

        | 항목 | 기준값 | 비고 |
        |------|--------|------|
        | 기술점수 | RSI·MACD·EMA·ADX·OBV·일목·Z-Score·다이버전스 등 11개 지표 | 범위 -10 ~ +10 |
        | 뉴스감성 | yfinance 뉴스 키워드 A/B/C 등급 가중 합산 | 범위 -5 ~ +5 |

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
                # 등락률: data가 로드된 상태면 추가 비용 없이 계산
                _pct: float | None = None
                try:
                    if _data_ready and len(data["Close"]) >= 2:
                        _pct = float(
                            (data["Close"].iloc[-1] / data["Close"].iloc[-2] - 1) * 100
                        )
                except Exception:
                    pass
                sent = _news_sentiment_llm_cached(
                    ticker, gemini_api_key, groq_api_key, _cname,
                    price_change_pct=_pct,
                )
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
            # dict 형식 이벤트만 배너에 표시 (문자열 플래그 "수급_미동반" 등 제외)
            _dict_flags = [e for e in event_flags if isinstance(e, dict)]
            if _dict_flags:
                _ev_labels = "·".join(
                    "·".join(e.get("event_kw", [])[:2]) for e in _dict_flags[:3]
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
            if gemini_api_key:
                llm_badge = "&nbsp;|&nbsp; 🤖 Gemini AI 분석"
            elif groq_api_key:
                llm_badge = "&nbsp;|&nbsp; 🦙 Groq AI 분석"
            else:
                llm_badge = "&nbsp;|&nbsp; 🔑 키워드 분석"
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
                        _article_dialog(title, link, ticker, gemini_api_key, groq_api_key)
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
                d = _flatten_columns(d)
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

        with st.spinner("ETF 지표 로딩 중... (KRX API)"):
            _etf_data   = _etf_fundamental(ticker)
            _etf_score  = calculate_etf_score(_etf_data)

        # ── 데이터 상태 배너 ─────────────────────────────────────────────────
        _data_status = _etf_data.get("data_status", "ok")
        if _etf_data.get("cache_used"):
            st.info(f"⏳ {_data_status}", icon="🔄")
        elif _data_status not in ("ok", "krx_api+fdr"):
            st.warning(f"KRX API 상태: {_data_status}", icon="⚠️")

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

        _src_label = {"krx_api": "KRX 공공 API", "krx_api+fdr": "KRX API + FDR", "error": "오류"}.get(
            _etf_data.get("source", ""), _etf_data.get("source", "")
        )
        _aum_txt = f"순자산(AUM): **{_aum:,.0f}억원**  |  " if _aum else ""
        st.caption(f"{_aum_txt}섹터: **{_sector}**  |  데이터: {_src_label}")

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
                st.info("ETF 지표 데이터를 불러올 수 없습니다. (KRX API 장애 또는 지원되지 않는 종목)")

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
            st.info("구성종목 데이터를 불러올 수 없습니다. (KRX PDF API 미수집 또는 비상장 ETF)")

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
def generate_signal(data, advanced, hybrid, news_result, expected, signals):
    """신호등 3단 판정: BUY / WAIT / SELL + 액션 메시지 + 근거 리스트 반환"""
    if data.empty or len(data) < 2:
        return "WAIT", "데이터가 부족합니다. 관망을 권장합니다.", ["데이터 부족"]

    _close_raw = data["Close"]
    close   = _close_raw.iloc[:, 0] if isinstance(_close_raw, pd.DataFrame) else _close_raw
    try:
        current = float(close.iloc[-1])
    except (IndexError, TypeError, ValueError):
        return "WAIT", "데이터가 부족합니다. 관망을 권장합니다.", ["데이터 부족"]

    trend_score  = advanced.get("trend_score",  50.0)
    volume_score = advanced.get("volume_score", 50.0)
    news_score   = news_result.get("score", 0.0)
    h_score      = hybrid.get("hybrid_score", 0.0)

    # EMA 역배열 (단기 < 중기 < 장기 = 하락 배열)
    ema_downtrend = False
    if all(c in data.columns for c in ["EMA_20", "EMA_50", "EMA_200"]):
        _row = data.iloc[-1]
        _e20  = float(_row["EMA_20"])  if pd.notna(_row.get("EMA_20"))  else None
        _e50  = float(_row["EMA_50"])  if pd.notna(_row.get("EMA_50"))  else None
        _e200 = float(_row["EMA_200"]) if pd.notna(_row.get("EMA_200")) else None
        if all(v is not None for v in [_e20, _e50, _e200]):
            ema_downtrend = _e20 < _e50 < _e200

    # 손절가 이탈 여부
    stop_loss_breached = False
    _sl = get_stop_loss_targets(data)
    if _sl:
        stop_loss_breached = current < _sl["stop_8pct"]

    # VWAP(월간) 위치
    vwap_above = False
    if "VWAP_M" in data.columns:
        _vm = data["VWAP_M"].iloc[-1]
        if pd.notna(_vm):
            vwap_above = current > float(_vm)

    # 거래량 비율 (최근 1봉 / 20일 평균)
    vol_ratio = 1.0
    if "Volume_MA20" in data.columns and len(data) >= 1:
        _vma = float(data["Volume_MA20"].iloc[-1])
        if _vma > 0 and pd.notna(_vma):
            vol_ratio = float(data["Volume"].iloc[-1]) / _vma

    reasons = []

    # ── SELL 조건 ────────────────────────────────────────────────────────────
    if stop_loss_breached:
        reasons.append(f"손절가({_sl['stop_8pct']:,.0f}) 이탈 — 즉시 손절 고려")
        return "SELL", "손절가를 이탈했습니다. 포지션 정리를 검토하세요.", reasons

    if ema_downtrend and news_score <= -1.5:
        reasons.append("EMA 역배열 — 단·중·장기 하락 추세 배열")
        reasons.append(f"부정적 뉴스 감성 ({news_score:+.1f}점) 확인")
        if h_score <= -1:
            reasons.append(f"종합 신호 약세 ({h_score:+.1f}점)")
        return "SELL", "추세 하락과 악재 뉴스가 겹쳤습니다. 비중 축소를 검토하세요.", reasons

    if ema_downtrend and h_score <= -3:
        reasons.append("EMA 역배열 — 하락 추세 지속")
        reasons.append(f"강한 하락 신호 ({h_score:+.1f}점)")
        return "SELL", "강한 하락 추세가 지속 중입니다. 신중한 접근이 필요합니다.", reasons

    # ── BUY 조건 ─────────────────────────────────────────────────────────────
    if trend_score > 70 and vol_ratio > 1.0 and vwap_above:
        reasons.append(f"추세 점수 {trend_score:.0f}점 — {'강력한 ' if trend_score > 80 else ''}상승 추세 형성")
        reasons.append(f"거래량 평균 대비 {vol_ratio:.1f}배 — 충분한 매수 에너지")
        reasons.append("현재가 월간 VWAP 상단 — 시장 평균가 이상")
        if news_score >= 1.0:
            reasons.append(f"긍정적 뉴스 감성 뒷받침 ({news_score:+.1f}점)")
        action = (
            f"거래량 평균 {vol_ratio:.1f}배 — 매수 조건이 충족되었습니다."
            if vol_ratio > 1.5 else "매수 조건이 충족되었습니다. 진입을 검토하세요."
        )
        return "BUY", action, reasons

    # ── WAIT 조건 (기본값) ────────────────────────────────────────────────────
    if trend_score <= 70:
        reasons.append(f"추세 점수 {trend_score:.0f}점 — 상승 추세 미확인 (기준: 70점)")
    else:
        reasons.append("추세 점수 양호하나 추가 조건 미충족")
    if not vwap_above:
        reasons.append("현재가 월간 VWAP 하단 — 시장 평균가 미달")
    if vol_ratio <= 1.0:
        reasons.append(f"거래량 비율 {vol_ratio:.1f}x — 관심 부족 구간 (기준: 1.0배)")

    if h_score >= 2:
        action = f"기술 신호 양호({h_score:+.1f}점)하나 매수 조건 미충족 — 돌파 대기 중입니다."
    elif h_score <= -2:
        action = "하락 신호 감지. 손절 라인을 재확인하고 신중히 접근하세요."
    else:
        action = "지금은 관망 구간입니다. 돌파 신호 확인 후 진입을 검토하세요."

    return "WAIT", action, reasons


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
        _data_ready = False  # 데이터 없음 — 이후 차트 분석은 _data_ready 가드로 건너뜀

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

    col_chart, col_sig = st.columns([1, 1])

    with col_chart:
        title_label = f"{sname} ({ticker})" if sname != ticker else ticker
        _badge_is_krw = ticker.upper().endswith((".KS", ".KQ"))
        _badge_price_str = (
            ("₩" if _badge_is_krw else "$") +
            ("{:,.0f}" if _badge_is_krw else "{:,.2f}").format(_rt_price)
        ) if (_rt_price > 0 and _data_ready) else "—"
        st.markdown(f"""
<div class="ma-stock-badge">
  <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px">
    <div>
      <div class="gradient-text" style="font-size:1.15rem;font-weight:800">{title_label}</div>
      <div style="font-size:.72rem;color:#94A3B8;margin-top:3px">
        <svg xmlns="http://www.w3.org/2000/svg" width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;margin-right:3px"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
        기술적 분석 · AI 매매 신호
      </div>
    </div>
    <div style="text-align:right">
      <div style="font-size:1.3rem;font-weight:700;color:#E2E8F0">{_badge_price_str}</div>
      <div style="font-size:.68rem;color:#8B5CF6;margin-top:2px">{"● 실시간" if (_rt_realtime and _data_ready and _rt_price > 0) else "○ 장마감"}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

        # KOSPI 데이터 사전 수집
        try:
            _kospi_raw = yf.download("^KS11", period=_aperiod, auto_adjust=True, progress=False)
            _kospi_raw = _flatten_columns(_kospi_raw)
            _kospi_df = _kospi_raw[["Open", "High", "Low", "Close"]].dropna()
        except Exception:
            _kospi_df = pd.DataFrame()

        # yfinance MultiIndex 잔존 가능성 방어 — 차트 진입 직전 재보정
        data = _flatten_columns(data)

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
            st.warning("차트 데이터 컬럼을 읽을 수 없습니다. 티커를 다시 선택해 주세요.")
            _data_ready = False
        else:
            pass  # 아래 차트 렌더링 진행

        o = data.get("Open"); h = data.get("High"); lo = data.get("Low"); c = data.get("Close"); v = data.get("Volume")

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

        # ── KOSPI 지수 (Row 6) ───────────────────────────────────────────────
        if not _kospi_df.empty:
            fig.add_trace(go.Scatter(
                x=_kospi_df.index,
                y=_kospi_df["Close"],
                name="KOSPI",
                line=dict(color="#80cbc4", width=1.8),
                fill="tozeroy",
                fillcolor="rgba(128,203,196,0.10)",
            ), row=6, col=1)
        else:
            fig.add_annotation(
                text="KOSPI 데이터를 불러올 수 없습니다.",
                xref="x6", yref="y6", x=0.5, y=0.5, showarrow=False,
                font=dict(color="#9e9e9e", size=12),
            )

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
            height=1150,
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
                "> ⚠️ **포트폴리오 탭 'AI 모멘텀 추천'과 결과가 다를 수 있습니다.**  \n"
                "> 추천은 '추세추종(모멘텀)' 기반이라 RSI 저점 구간을 오히려 제외합니다."
            )

        # 사전 계산 (아래 섹션에서 재사용)
        h_score = hybrid.get("hybrid_score", 0.0)
        h_label = hybrid.get("label", "중립/관망")
        h_badge = hybrid.get("badge", "⚪")
        fs      = fund_score_data.get("fund_score", 0)
        f_label = fund_score_data.get("fund_label", "N/A")

        _has_price = not close.empty and len(close) >= 2
        if _has_price:
            _daily_last = float(close.iloc[-1])
            prev_price  = float(close.iloc[-2])
            # 실시간 가격 우선 — 없으면 일봉 마지막 종가
            last_price  = _rt_price if (_rt_price > 0 and _data_ready) else _daily_last
            daily_chg   = (last_price - prev_price) / prev_price * 100
            _is_krw     = last_price > 500
            _fmt        = "{:,.0f}" if _is_krw else "{:,.2f}"
        else:
            last_price, prev_price, daily_chg = 0.0, 0.0, 0.0
            _is_krw, _fmt = True, "{:,.0f}"

        # 실시간 시세 배지 (버튼 클릭 후 분석 완료 시만 표시)
        if _data_ready and _rt_price > 0:
            if _rt_stale:
                st.warning(_rt_stale_msg or "장이 열리지 않은 상태입니다. 가장 최근 종가를 표시합니다.", icon="⏸️")
            _rt_label = "실시간 시세 (KST)" if _rt_realtime else "장마감 종가 (KST)"
            _rt_color = "#22c55e" if _rt_realtime else "#94a3b8"
            st.markdown(
                f'<div style="font-size:0.78rem;color:{_rt_color};margin-bottom:6px;">'
                f'✅ {_rt_label} 반영 완료 &nbsp;|&nbsp; 기준 시각: <b>{_rt_ts}</b>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # ═══════════════════════════════════════════════════════════════════
        # LAYER 1 — 결론 레이어 (신호등 카드)
        # ═══════════════════════════════════════════════════════════════════
        _tl_signal, _tl_action, _tl_reasons = generate_signal(
            data=data, advanced=advanced, hybrid=hybrid,
            news_result=news_result, expected=expected, signals=signals,
        )
        if _tl_signal == "BUY":
            _l1_bg, _l1_border, _l1_fc = "#0d2318", "#22c55e", "#4ade80"
        elif _tl_signal == "SELL":
            _l1_bg, _l1_border, _l1_fc = "#1f0d0d", "#ef4444", "#f87171"
        else:
            _l1_bg, _l1_border, _l1_fc = "#1c1a0a", "#eab308", "#facc15"
        _l1_emoji = "🟢" if _tl_signal == "BUY" else ("🔴" if _tl_signal == "SELL" else "🟡")

        st.markdown(f"""
<div style="background:{_l1_bg};border:2px solid {_l1_border};border-radius:16px;
            padding:20px 18px;text-align:center;margin-bottom:10px;
            box-shadow:0 0 24px {_l1_border}44;">
  <div style="font-size:0.7rem;color:#888;letter-spacing:2px;margin-bottom:8px;">AI TRADING SIGNAL</div>
  <div style="font-size:2.8rem;font-weight:900;color:{_l1_fc};letter-spacing:4px;">{_l1_emoji} {_tl_signal}</div>
  <div style="font-size:0.88rem;color:{_l1_fc}cc;margin-top:10px;line-height:1.5;">{_tl_action}</div>
</div>
        """, unsafe_allow_html=True)

        # Key Metrics — 현재가·목표가·손절가·등락률
        _sl_data = get_stop_loss_targets(data) if _has_price else None
        _bt_data = get_buy_target_price(data, mode="classic") if _has_price else None
        _m1, _m2, _m3, _m4 = st.columns(4)
        _cur_label = "현재가(실시간)" if (_rt_realtime and _rt_price > 0 and _data_ready) else "현재가"
        _chg_color = "#10B981" if daily_chg >= 0 else "#3B82F6"
        _m1.markdown(f"""
<div class="ma-metric-card">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px">
    <span style="font-size:.7rem;color:#94A3B8;font-weight:500">{_cur_label}</span>
    <span style="color:#8B5CF6"><svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg></span>
  </div>
  <div style="font-size:1.2rem;font-weight:700;color:#E2E8F0">{_fmt.format(last_price) if _has_price else "—"}</div>
</div>
""", unsafe_allow_html=True)
        _m2.markdown(f"""
<div class="ma-metric-card">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px">
    <span style="font-size:.7rem;color:#94A3B8;font-weight:500">추천 매수가</span>
    <span style="color:#10B981"><svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg></span>
  </div>
  <div style="font-size:1.2rem;font-weight:700;color:#E2E8F0">{_fmt.format(_bt_data["buy_target"]) if _bt_data else "—"}</div>
</div>
""", unsafe_allow_html=True)
        _m3.markdown(f"""
<div class="ma-metric-card">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px">
    <span style="font-size:.7rem;color:#94A3B8;font-weight:500">손절가</span>
    <span style="color:#F59E0B"><svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg></span>
  </div>
  <div style="font-size:1.2rem;font-weight:700;color:#E2E8F0">{_fmt.format(_sl_data["stop_8pct"]) if _sl_data else "—"}</div>
</div>
""", unsafe_allow_html=True)
        _m4.markdown(f"""
<div class="ma-metric-card">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px">
    <span style="font-size:.7rem;color:#94A3B8;font-weight:500">등락률</span>
    <span style="color:{_chg_color}"><svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/></svg></span>
  </div>
  <div style="font-size:1.2rem;font-weight:700;color:{_chg_color}">{f"{daily_chg:+.2f}%" if _has_price else "—"}</div>
</div>
""", unsafe_allow_html=True)

        # ═══════════════════════════════════════════════════════════════════
        # LAYER 2 — 근거 레이어
        # ═══════════════════════════════════════════════════════════════════
        st.markdown("---")

        # Reasoning Box — 신호 근거 TOP 3
        _sig_icon = _l1_emoji
        _reason_html = "".join(
            f'<div style="padding:6px 0;border-bottom:1px solid #1e2130;font-size:0.9rem;color:#ddd;">'
            f'{_sig_icon} {r}</div>'
            for r in _tl_reasons[:3]
        ) or '<div style="color:#555;font-size:0.88rem;">분석 데이터 수집 중...</div>'
        st.markdown(f"""
<div style="background:#12161f;border:1px solid {_l1_border}44;border-radius:12px;
            padding:14px 16px;margin-bottom:14px;">
  <div style="font-size:0.72rem;color:#888;letter-spacing:1px;margin-bottom:8px;">📋 신호 근거 (TOP 3)</div>
  {_reason_html}
</div>
        """, unsafe_allow_html=True)

        # Scorecard — 추세/탄력/에너지 progress bars
        _ts = advanced.get("trend_score",    50.0)
        _ms = advanced.get("momentum_score", 50.0)
        _vs = advanced.get("volume_score",   50.0)

        def _signal_prog_bar(label, score, icon):
            _sc = "#4ade80" if score >= 65 else ("#f87171" if score <= 35 else "#facc15")
            st.markdown(
                f'<div style="margin-bottom:10px;">'
                f'<div style="display:flex;justify-content:space-between;font-size:0.88rem;">'
                f'<span style="color:#aaa;">{icon} {label}</span>'
                f'<b style="color:{_sc};">{score:.0f}점</b></div>'
                f'<div style="background:#1e2130;border-radius:4px;height:8px;margin-top:4px;">'
                f'<div style="background:{_sc};width:{int(score)}%;height:8px;border-radius:4px;"></div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        _signal_prog_bar("추세 (Trend)",    _ts, "📈")
        _signal_prog_bar("탄력 (Momentum)", _ms, "⚡")
        _signal_prog_bar("에너지 (Volume)", _vs, "🔋")

        # 수급 요약 (KRX 종목만)
        _is_kr_sig = _data_ready and (ticker.endswith(".KS") or ticker.endswith(".KQ"))
        if _is_kr_sig:
            try:
                _inv_s = _inv_data(ticker)
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
                        f'<div style="background:#1a1d2e;border-radius:8px;padding:10px 14px;'
                        f'border-left:3px solid {_inv_brd};margin-bottom:6px;">'
                        f'<div style="font-size:0.72rem;color:#888;margin-bottom:3px;">📊 수급 동향</div>'
                        f'<div style="font-size:0.9rem;color:#e0e0e0;">{_inv_txt}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            except Exception:
                pass

        st.markdown("---")

        # ═══════════════════════════════════════════════════════════════════
        # LAYER 3 — 검증 레이어 (expander)
        # ═══════════════════════════════════════════════════════════════════
        # ── 단타/장투 신호 세부 (expander) ──────────────────────────────────
        with st.expander("⚡ 단타/장투 신호 세부 보기", expanded=False):
            if h_score >= 2:
                st_bg, st_fc = "#1b3a28", "#a5d6a7"
            elif h_score <= -2:
                st_bg, st_fc = "#3a1a1a", "#ef9a9a"
            else:
                st_bg, st_fc = "#1e2130", "#bdbdbd"

            if fs >= 3:
                lt_bg, lt_fc = "#1a2f3a", "#80cbc4"
            elif fs <= -2:
                lt_bg, lt_fc = "#3a2a1a", "#ffcc80"
            else:
                lt_bg, lt_fc = "#1e2130", "#bdbdbd"

            _chg_clr = "#a5d6a7" if daily_chg >= 0 else "#ef9a9a"
            _chg_sym = "▲" if daily_chg >= 0 else "▼"
            _price_inner = (
                f'<div style="font-size:1.35rem;font-weight:bold;color:#e0e0e0;">{_fmt.format(last_price)}</div>'
                f'<div style="font-size:1.02rem;font-weight:bold;color:{_chg_clr};margin-top:4px;">{_chg_sym} {abs(daily_chg):.2f}%</div>'
            ) if _has_price else '<div style="color:#555;font-size:0.9rem;">가격 데이터 없음</div>'

            st.markdown(f"""
<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;align-items:stretch;">
  <div class="signal-box" style="background:{st_bg};margin:0;">
    <div style="font-size:0.92rem;color:#888;margin-bottom:5px;letter-spacing:1px;">⚡ 단타 신호</div>
    <div style="font-size:1.6rem;font-weight:bold;color:{st_fc};">{h_badge} {h_label}</div>
    <div style="font-size:1.02rem;color:#aaa;margin-top:5px;">점수: <b style="color:{st_fc};">{h_score:+.1f}</b></div>
  </div>
  <div class="signal-box" style="background:{lt_bg};margin:0;">
    <div style="font-size:0.92rem;color:#888;margin-bottom:5px;letter-spacing:1px;">🏛️ 장투 신호</div>
    <div style="font-size:1.6rem;font-weight:bold;color:{lt_fc};">🏛️ {f_label}</div>
    <div style="font-size:1.02rem;color:#aaa;margin-top:5px;">점수: <b style="color:{lt_fc};">{fs:+.1f}</b></div>
  </div>
  <div style="background:{st_bg};border-radius:8px;padding:10px 12px;border:1px solid #2a2d3e;">
    <div style="font-size:0.92rem;color:#888;margin-bottom:5px;letter-spacing:1px;">💰 현재가</div>
    {_price_inner}
  </div>
</div>
            """, unsafe_allow_html=True)

            st.markdown("""
<div style="font-size:0.82rem;line-height:1.9;margin-top:12px;">

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

            tc = "#a5d6a7" if tech_score >= 0 else "#ef9a9a"
            nc = "#a5d6a7" if news_score >= 0 else "#ef9a9a"
            fc_color = lt_fc
            st.markdown(f"""
<div style="background:#1e2130;border-radius:8px;padding:8px 14px;margin-top:10px;">
  <div style="display:grid;grid-template-columns:1fr auto 1fr;gap:4px;align-items:start;">
    <div>
      <div style="color:#666;font-size:0.85rem;margin-bottom:4px;">⚡ 단타 구성</div>
      <div style="display:flex;justify-content:space-between;font-size:0.98rem;"><span style="color:#aaa;">기술(70%)</span><b style="color:{tc};">{tech_score:+.1f}</b></div>
      <div style="display:flex;justify-content:space-between;font-size:0.98rem;margin-top:3px;"><span style="color:#aaa;">뉴스(30%)</span><b style="color:{nc};">{news_score:+.1f}</b></div>
    </div>
    <div style="width:1px;background:#2a2d3e;align-self:stretch;margin:2px 10px;"></div>
    <div>
      <div style="color:#666;font-size:0.85rem;margin-bottom:4px;">🏛️ 장투 구성</div>
      <div style="display:flex;justify-content:space-between;font-size:0.98rem;"><span style="color:#aaa;">펀더멘털</span><b style="color:{fc_color};">{fs:+.1f}</b></div>
      <div style="font-size:0.82rem;color:#555;margin-top:4px;">PER·PBR·ROE·FCF ▶</div>
    </div>
  </div>
</div>
            """, unsafe_allow_html=True)

        # ── 매수 적정가 모드 선택 ────────────────────────────────────────────────
        _mode_map = {
            "🏷️ 세일 (LBB)":       "sale",
            "⚡ 추격 (Breakout)":   "breakout",
            "📊 정석 (VWAP)":       "vwap",
            "🎯 종합 (가중평균)":    "classic",
        }
        _mode_descs = {
            "🏷️ 세일 (LBB)":       "안전하게 밑에서 기다릴게요  ·  현재가 -5~10%",
            "⚡ 추격 (Breakout)":   "저항선을 뚫으면 바로 탈게요  ·  현재가 +2~3% 돌파 시",
            "📊 정석 (VWAP)":       "시장 참여자 평균가에 살게요  ·  당일 VWAP 기준",
            "🎯 종합 (가중평균)":    "BB하단·SMA20·5일저 가중 평균  ·  50% + 30% + 20%",
        }
        st.markdown(
            '<div style="font-size:0.8rem;color:#888;margin-bottom:4px;margin-top:8px;">💡 매수 적정가 모드 선택</div>',
            unsafe_allow_html=True,
        )
        _mode_label = st.radio(
            "매수 적정가 모드",
            list(_mode_map.keys()),
            index=0,
            horizontal=True,
            key="buy_mode_selector",
            label_visibility="collapsed",
        )
        _buy_mode = _mode_map.get(_mode_label, "classic")
        st.markdown(
            f'<div style="font-size:0.78rem;color:#555;margin-bottom:6px;padding-left:2px;">{_mode_descs[_mode_label]}</div>',
            unsafe_allow_html=True,
        )

        # ── 매수/매도 적정가 (세로) + 손절익절 가이드 (우측) — 등높이 HTML 그리드
        _buy_card_html  = ""
        _sell_card_html = ""
        _sl_card_html   = ""

        if _has_price:
            _bt      = get_buy_target_price(data, mode=_buy_mode)
            _st_data = get_sell_target_price(data)

            if _bt:
                _t_price    = _bt["buy_target"]
                _gap        = _bt["gap_pct"]
                _timing     = _bt["timing"]
                _tc_clr     = _bt["timing_color"]
                _m_label    = _bt["mode_label"]
                _m_desc     = _bt["mode_desc"]
                _m_color    = _bt["mode_color"]
                _detail     = _bt["detail_line"]
                _bt_comment = _bt.get("comment", "")

                # 세일/종합/vwap: 양수 gap = 현재가가 목표보다 높음(비쌈) → 빨강
                # 추격 모드: 양수 gap = 목표가가 현재가보다 높음(올라야 진입) → 노랑
                if _buy_mode == "breakout":
                    _gap_clr = "#fff176" if _gap > 0 else "#69f0ae"
                else:
                    _gap_clr = "#ef9a9a" if _gap > 5 else ("#fff176" if _gap > 0 else "#69f0ae")

                _gap_label = f"목표가 +{_gap:.1f}%" if _buy_mode == "breakout" else f"현재가 {_gap:+.1f}%"

                _buy_card_html = f"""
<div style="background:#12161f;border:1px solid #2a2d3e;border-radius:8px;padding:10px 12px;">
  <div style="display:flex;align-items:center;gap:6px;margin-bottom:6px;">
    <span style="font-size:0.75rem;font-weight:bold;color:{_m_color};background:rgba(255,255,255,0.05);border:1px solid {_m_color}44;border-radius:10px;padding:1px 8px;">{_m_label}</span>
  </div>
  <div style="font-size:0.72rem;color:#666;margin-bottom:4px;">{_m_desc}</div>
  <div style="font-size:1.18rem;font-weight:bold;color:#e0e0e0;">{_fmt.format(_t_price)}</div>
  <div style="font-size:0.88rem;color:{_gap_clr};margin:3px 0;">{_gap_label}</div>
  <div style="font-size:0.85rem;font-weight:bold;color:{_tc_clr};">{_timing}</div>
  <div style="border-top:1px solid #1e2130;padding-top:5px;margin-top:6px;font-size:0.75rem;color:#555;line-height:1.7;">
    {_detail}
  </div>
  {'<div style="margin-top:5px;font-size:0.75rem;color:#ffb74d;">⚠️ ' + _bt_comment + '</div>' if _bt_comment else ''}
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
                _nh_badge = '<span style="background:#1a237e;color:#82b1ff;font-size:0.72rem;padding:1px 6px;border-radius:8px;margin-left:4px;">신고가</span>' if _is_nh else ""
                _ref_line = f"BB상단 {_fmt.format(_bb_u)}" + (f"<br>매물대 {_fmt.format(_res)}" if _res else "") + (f" · 피보 {_fmt.format(_fib_val)}" if _fib_val else "")
                _sell_card_html = f"""
<div style="background:#1a0f0a;border:1px solid #3d1f0f;border-radius:8px;padding:10px 12px;margin-top:6px;">
  <div style="font-size:0.78rem;color:#888;margin-bottom:5px;">📤 매도 적정가{_nh_badge}</div>
  <div style="display:flex;justify-content:space-between;gap:4px;">
    <div>
      <div style="font-size:0.75rem;color:#777;">보수적</div>
      <div style="font-size:1.05rem;font-weight:bold;color:#ffccbc;">{_fmt.format(_cons)}</div>
      <div style="font-size:0.85rem;color:{_cons_clr};">{_cons_gap:+.1f}%</div>
    </div>
    <div style="text-align:right;">
      <div style="font-size:0.75rem;color:#777;">공격적</div>
      <div style="font-size:1.05rem;font-weight:bold;color:#ff8a65;">{_fmt.format(_aggr)}</div>
      <div style="font-size:0.85rem;color:{_aggr_clr};">{_aggr_gap:+.1f}%</div>
    </div>
  </div>
  <div style="font-size:0.82rem;font-weight:bold;color:{_s_color};margin-top:5px;">{_s_timing}</div>
  <div style="border-top:1px solid #3d1f0f;padding-top:5px;margin-top:6px;font-size:0.75rem;color:#555;line-height:1.7;">
    {_ref_line}
  </div>
</div>"""

        sl = get_stop_loss_targets(data)
        if sl:
            _is_krw_sl = sl["current"] > 500
            _sfmt = "{:,.0f}" if _is_krw_sl else "{:,.2f}"
            _sl_card_html = f"""
<div style="background:#0d1117;border:1px solid #2a2d3e;border-radius:10px;padding:12px 14px;height:100%;box-sizing:border-box;">
  <div style="font-size:0.85rem;color:#888;letter-spacing:0.5px;margin-bottom:10px;">
    🛡️ 손절·익절 가이드
    <span style="float:right;font-size:0.75rem;color:#555;">오닐 원칙 + ATR</span>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;">
    <div style="background:#1a0808;border-radius:6px;padding:7px 10px;">
      <div style="font-size:0.75rem;color:#888;">🔴 손절 (8% 룰)</div>
      <div style="font-size:1.08rem;font-weight:bold;color:#ef9a9a;">{_sfmt.format(sl['stop_8pct'])}</div>
    </div>
    <div style="background:#1a0f08;border-radius:6px;padding:7px 10px;">
      <div style="font-size:0.75rem;color:#888;">🟠 손절 (ATR×2.5)</div>
      <div style="font-size:1.08rem;font-weight:bold;color:#ffab91;">{_sfmt.format(sl['stop_atr'])}</div>
    </div>
    <div style="background:#0f1a08;border-radius:6px;padding:7px 10px;">
      <div style="font-size:0.75rem;color:#888;">🟡 1차 목표 (2R)</div>
      <div style="font-size:1.08rem;font-weight:bold;color:#fff176;">{_sfmt.format(sl['target_2r'])}</div>
    </div>
    <div style="background:#091a08;border-radius:6px;padding:7px 10px;">
      <div style="font-size:0.75rem;color:#888;">🟢 2차 목표 (3R)</div>
      <div style="font-size:1.08rem;font-weight:bold;color:#a5d6a7;">{_sfmt.format(sl['target_3r'])}</div>
    </div>
  </div>
  <div style="font-size:0.82rem;color:#555;margin-top:8px;text-align:right;">
    ATR {sl['atr_ratio']:.2f}% · {_sfmt.format(sl['atr'])}
  </div>
</div>"""

        if _buy_card_html or _sell_card_html or _sl_card_html:
            st.markdown(f"""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;align-items:stretch;margin-top:6px;">
  <div>{_buy_card_html}{_sell_card_html}</div>
  <div>{_sl_card_html}</div>
</div>
            """, unsafe_allow_html=True)

        # ── 예상 수익률 구간 ───────────────────────────────────────────────────
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
            _vpvr_note = '<div style="font-size:0.78rem;color:#ffab91;margin-top:3px;">⚠️ 목표가 부근 매물대 저항 — B 하향 보정</div>' if _vpvr else ""

            st.markdown(f"""
<div style="background:#1a1d2e;border-radius:8px;padding:10px 14px;margin-top:4px;border:1px solid #2a2d3e;">
  <div style="font-size:0.78rem;color:#888;margin-bottom:6px;">📈 예상 수익률 구간 (20거래일)</div>
  <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
    <div style="text-align:center;">
      <div style="font-size:0.75rem;color:#777;">최저 A</div>
      <div style="font-size:1.0rem;font-weight:bold;color:{_a_clr};">{_A:+.1f}%</div>
    </div>
    <div style="text-align:center;">
      <div style="font-size:0.75rem;color:#777;">중간값 M</div>
      <div style="font-size:1.12rem;font-weight:bold;color:{_m_clr};">{_M:+.1f}%</div>
    </div>
    <div style="text-align:center;">
      <div style="font-size:0.75rem;color:#777;">최고 B{'⚠️' if _vpvr else ''}</div>
      <div style="font-size:1.0rem;font-weight:bold;color:{_b_clr};">{_B:+.1f}%</div>
    </div>
  </div>
  <div style="position:relative;background:#2a2d3e;border-radius:3px;height:5px;margin-bottom:3px;">
    <div style="position:absolute;left:{_z_pos}%;top:0;width:1px;height:100%;background:rgba(255,255,255,0.25);"></div>
    <div style="position:absolute;left:{_m_pos}%;top:-2px;width:3px;height:9px;background:{_m_clr};border-radius:2px;transform:translateX(-50%);"></div>
  </div>
  {_vpvr_note}
  <div style="border-top:1px solid #2a2d3e;padding-top:6px;margin-top:6px;
              display:grid;grid-template-columns:repeat(4,1fr);gap:2px;text-align:center;">
    <div>
      <div style="font-size:0.75rem;color:#666;">켈리 비중</div>
      <b style="font-size:0.95rem;color:#90caf9;">{_kelly:.1f}%</b>
      <div style="background:#2a2d3e;border-radius:2px;height:3px;margin-top:3px;">
        <div style="background:#90caf9;width:{_kelly_bar}%;height:3px;border-radius:2px;"></div>
      </div>
    </div>
    <div><div style="font-size:0.75rem;color:#666;">승률 추정</div><b style="font-size:0.95rem;color:#aaa;">{_winp:.0f}%</b></div>
    <div><div style="font-size:0.75rem;color:#666;">W/L</div><b style="font-size:0.95rem;color:#aaa;">{_rr_str}</b></div>
    <div><div style="font-size:0.75rem;color:#666;">β 베타</div><b style="font-size:0.95rem;color:{_beta_clr};">{_beta:+.2f}</b></div>
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

        # ── 보수적 리스크 조정 (승률 < 50%) ──────────────────────────────────
        if risk_adj.get("conservative_applied"):
            _ct  = risk_adj["conservative_target"]
            _csl = risk_adj["conservative_stoploss"]
            _cwp = risk_adj.get("win_prob", 0.0)
            st.warning(
                f"**⚠️ 저승률 보수 조정** — 승률 {_cwp:.0f}%로 50% 미만  \n"
                f"목표 수익률 → **{_ct:+.1f}%** (원래 M의 50%)  \n"
                f"손절 라인 → **{_csl:+.1f}%** (현재가 대비 타이트 설정)  \n"
                f"*분석 결과가 틀릴 확률이 더 높으므로 포지션 크기를 줄이고 손절을 철저히 지키세요.*",
                icon="⚠️",
            )

        # ── Dead Time Check (기회비용 지수) ──────────────────────────────────
        if dead_time.get("message"):
            _dt_vol  = dead_time.get("volatility_14d", 0.0)
            _dt_vr   = dead_time.get("vol_ratio", 1.0)
            _dt_dead = dead_time.get("is_dead", False)
            if _dt_dead:
                st.warning(dead_time["message"], icon="⏳")
                st.markdown(
                    f'<div style="background:#1a1d2e;border-radius:7px;padding:8px 12px;'
                    f'border-left:3px solid #f39c12;font-size:0.86rem;color:#aaa;margin-top:4px;">'
                    f'14일 변동성 <b style="color:#fff176;">{_dt_vol:.1f}%</b> (기준 5% 미만) · '
                    f'거래량 비율 <b style="color:#fff176;">{_dt_vr*100:.0f}%</b> (기준 70% 미만)<br>'
                    f'<span style="color:#888;">자금이 묶인 채 방향을 잡지 못하는 구간입니다. '
                    f'돌파 신호(Breakout) 전까지 신규 진입을 자제하세요.</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.info(dead_time["message"], icon="📊")

        # ── 상승 임계치(Breakout) 필터 ────────────────────────────────────────
        _bk_status = breakout.get("status", "wait")
        _bk_detail = breakout.get("detail", "")
        if _bk_status == "breakout_both":
            st.success(f"**돌파(Breakout) 조건 충족** — {_bk_detail}", icon="🚀")
        elif _bk_status in ("breakout_ma", "breakout_vol"):
            st.info(f"**부분 돌파 감지** — {_bk_detail}", icon="📈")
        else:
            st.info(
                f"**관망(Wait) 유지** — {_bk_detail}  \n"
                f"*20일 MA 상향 돌파 또는 전일 거래량 200% 초과 시 진입 고려*",
                icon="⏸️",
            )

        # ── 종합 판단 스코어보드 ─────────────────────────────────────────────
        with st.expander("📊 종합 판단 스코어보드", expanded=False):
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
                    f'<div style="display:flex;justify-content:space-between;font-size:0.92rem;">'
                    f'<span style="color:#ccc;">{label} <span style="color:#666;">({weight})</span></span>'
                    f'<b style="color:{color};">{score:.0f}점</b></div>'
                    f'<div style="background:#2a2d3e;border-radius:4px;height:7px;margin-top:3px;">'
                    f'<div style="background:{color};width:{pct}%;height:7px;border-radius:4px;"></div>'
                    f'</div>'
                    f'<div style="font-size:0.85rem;color:#666;margin-top:2px;">{desc}</div>'
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
                f'<span style="font-size:0.88rem;color:#888;">가중 종합: </span>'
                f'<b style="color:{adv_clr};">{composite_adv:.0f}점 — {adv_txt}</b>'
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
            if div_descs:
                for d in div_descs:
                    if "하락" in d:
                        st.warning(d, icon="⚠️")
                    else:
                        st.success(d, icon="✅")

        st.divider()

        # ── VWAP + 지표값 (좌) │ 기술 신호 근거 (우) ───────────────────────────
        _tech_left, _tech_right = st.columns([1, 1])

        with _tech_left:
            if not data.empty and len(data) >= 2:
                _cur    = float(data["Close"].iloc[-1])
                _is_krw_v = _cur > 500
                _pf     = "{:,.0f}" if _is_krw_v else "{:,.2f}"
                _vw_row = data.iloc[-1]

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
            padding:12px 15px;margin-bottom:6px;">
  <div style="font-size:0.82rem;color:#888;letter-spacing:0.5px;margin-bottom:7px;">
    📊 VWAP 다중 타임프레임
    <span style="float:right;font-size:0.75rem;color:#555;">Shannon, 2008</span>
  </div>
  {_vwap_rows}
  <div style="margin-top:8px;font-size:0.88rem;font-weight:bold;color:{_stack_clr};">
    {_stack_txt}
  </div>
  <div style="font-size:0.78rem;color:#555;margin-top:4px;">
    수식: Σ(Typical Price × Volume) / Σ(Volume) — 롤링 누적합
  </div>
</div>
                """, unsafe_allow_html=True)

            with st.expander("📊 지표값", expanded=False):
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
                if data.empty:
                    st.caption("분석 데이터가 없습니다.")
                    last_row = None
                else:
                    last_row = data.iloc[-1]
                if last_row is not None:
                    for lbl, col, fmt in indicator_map:
                        if col in data.columns:
                            val = last_row[col]
                            if pd.notna(val):
                                st.caption(f"**{lbl}:** {float(val):{fmt}}")

        with _tech_right:
            with st.expander("📋 기술 신호 근거", expanded=False):
                _BUY_KEYS  = ["매수", "반등", "상승", "골든", "과매도", "매집", "긍정", "유입", "강세"]
                _SELL_KEYS = ["매도", "과열", "하락", "데드", "과매수", "분산", "약세", "이탈 주의"]
                _reason_items = []
                for r in signals.get("reasons", []):
                    if any(k in r for k in _BUY_KEYS):
                        _reason_items.append(
                            f'<div style="background:#1b3a28;border-radius:5px;padding:5px 7px;'
                            f'font-size:0.88rem;color:#a5d6a7;line-height:1.4;">🔺 {r}</div>')
                    elif any(k in r for k in _SELL_KEYS):
                        _reason_items.append(
                            f'<div style="background:#3a1a1a;border-radius:5px;padding:5px 7px;'
                            f'font-size:0.88rem;color:#ef9a9a;line-height:1.4;">🔻 {r}</div>')
                    else:
                        _reason_items.append(
                            f'<div style="background:#1a1d2e;border-radius:5px;padding:5px 7px;'
                            f'font-size:0.88rem;color:#90caf9;line-height:1.4;">🔷 {r}</div>')
                if _reason_items:
                    st.markdown(
                        f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:5px;">'
                        + "".join(_reason_items)
                        + "</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.caption("신호 근거 없음")

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
                    dead_time     = dead_time,
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

        # ── 포트폴리오 추가 버튼 ───────────────────────────────────────────────
        if avg_price_input > 0:
            _pf_token = st.session_state.get("auth_token")
            if _pf_token:
                _qty_col, _add_col = st.columns([2, 3])
                with _qty_col:
                    _qty_input = st.number_input(
                        "수량",
                        min_value=0.01, value=1.0, step=0.01,
                        format="%.2f",
                        label_visibility="collapsed",
                        key=f"qty_input_{ticker}",
                    )
                with _add_col:
                    if st.button("💼 포트폴리오에 추가", use_container_width=True,
                                 key=f"pf_add_{ticker}"):
                        _up_r = _db_upsert_portfolio(
                            st.session_state["auth_user_id"],
                            ticker, avg_price_input, _qty_input,
                        )
                        if _up_r.get("merged"):
                            st.success(f"{ticker} 추가 매수 완료 — 평단가 자동 합산")
                        else:
                            st.success(f"{ticker} 포트폴리오에 추가되었습니다.")
            else:
                st.caption("💼 포트폴리오에 추가하려면 **내 포트폴리오** 탭에서 로그인하세요.")

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


# ─── 내 포트폴리오 탭 ─────────────────────────────────────────────────────────
@st.fragment
def _render_portfolio_tab():
    _tok  = st.session_state.get("auth_token")
    _uid  = st.session_state.get("auth_user_id")
    _mail = st.session_state.get("auth_email")

    # ── 로그인 상태 헤더 ──────────────────────────────────────────────────────
    if _tok:
        _hdr_col, _out_col = st.columns([5, 1])
        _hdr_col.markdown(f"**💼 내 포트폴리오** — `{_mail}`")
        with _out_col:
            if st.button("로그아웃", use_container_width=True, key="pf_logout"):
                _db_logout(_tok)
                st.session_state["auth_token"]   = None
                st.session_state["auth_user_id"] = None
                st.session_state["auth_email"]   = None
                if _HAS_COOKIE_MGR and _cookie_mgr:
                    _cookie_mgr.delete("auth_token")
                st.rerun(scope="fragment")
    else:
        st.markdown("**💼 내 포트폴리오** — 로그인이 필요합니다")

    st.divider()

    # ── 미로그인: 로그인 / 회원가입 폼 ──────────────────────────────────────
    if not _tok:
        _auth_mode = st.radio(
            "모드 선택",
            ["로그인", "회원가입"],
            horizontal=True,
            label_visibility="collapsed",
        )

        with st.form("pf_auth_form"):
            _pf_email = st.text_input("이메일", placeholder="you@example.com")
            _pf_pw    = st.text_input("비밀번호", type="password",
                                      placeholder="6자 이상")
            _pf_submit = st.form_submit_button(
                _auth_mode, use_container_width=True, type="primary"
            )

        if _pf_submit:
            if _auth_mode == "회원가입":
                if len(_pf_pw) < 6:
                    st.error("비밀번호는 6자 이상이어야 합니다.")
                else:
                    _r = _db_register(_pf_email, _pf_pw)
                    if _r["ok"]:
                        st.success("회원가입 완료! 로그인해 주세요.")
                    else:
                        st.error(_r["error"])
            else:
                _r = _db_login(_pf_email, _pf_pw)
                if _r["ok"]:
                    st.session_state["auth_token"]   = _r["token"]
                    st.session_state["auth_user_id"] = _r["user_id"]
                    st.session_state["auth_email"]   = _r["email"]
                    if _HAS_COOKIE_MGR and _cookie_mgr:
                        from datetime import datetime as _dt
                        _cookie_mgr.set(
                            "auth_token", _r["token"],
                            expires_at=_dt(2099, 1, 1),
                        )
                    st.rerun(scope="fragment")
                else:
                    st.error(_r["error"])
        return

    # ── 토큰 만료 확인 ────────────────────────────────────────────────────────
    if not _db_get_user(_tok):
        st.session_state["auth_token"]   = None
        st.session_state["auth_user_id"] = None
        st.session_state["auth_email"]   = None
        if _HAS_COOKIE_MGR and _cookie_mgr:
            _cookie_mgr.delete("auth_token")
        st.warning("세션이 만료되었습니다. 다시 로그인해 주세요.")
        st.rerun(scope="fragment")
        return

    _items = _db_get_portfolio(_uid)
    _pf_nm: dict[str, str] = _ticker_name_map() if _items else {}
    try:
        _trade_history: list[dict] = _db_get_trade_history(_uid)
    except Exception:
        _trade_history = []

    # ── 수익률 계산을 위한 현재가 일괄 조회 ──────────────────────────────────
    _pf_tickers = list({it["ticker"] for it in _items}) if _items else []
    _pf_prices: dict[str, float] = {}
    if _pf_tickers:
        # 1순위: 1분봉 배치 다운로드 (장중 실시간 가격)
        try:
            _pf_raw = yf.download(
                _pf_tickers, period="1d", interval="1m", auto_adjust=True,
                progress=False, threads=True,
            )
            for _t in _pf_tickers:
                try:
                    if isinstance(_pf_raw.columns, pd.MultiIndex):
                        _s = _pf_raw["Close"][_t].dropna()
                    else:
                        _s = _pf_raw["Close"].dropna()
                    if not _s.empty:
                        _pf_prices[_t] = float(_s.iloc[-1])
                except Exception:
                    pass
        except Exception:
            pass

        # 2순위: fast_info.last_price — 1분봉에서 누락된 종목 개별 보완
        _pf_missing = [t for t in _pf_tickers if not _pf_prices.get(t)]
        for _t in _pf_missing:
            try:
                _lp = float(yf.Ticker(_t).fast_info.last_price)
                if _lp > 0:
                    _pf_prices[_t] = _lp
            except Exception:
                pass

        # 3순위: 일봉 — 그래도 누락된 종목 최종 보완
        _pf_still_missing = [t for t in _pf_tickers if not _pf_prices.get(t)]
        if _pf_still_missing:
            try:
                _pf_raw2 = yf.download(
                    _pf_still_missing, period="2d", auto_adjust=True,
                    progress=False, threads=True,
                )
                for _t in _pf_still_missing:
                    try:
                        if isinstance(_pf_raw2.columns, pd.MultiIndex):
                            _s = _pf_raw2["Close"][_t].dropna()
                        else:
                            _s = _pf_raw2["Close"].dropna()
                        if not _s.empty:
                            _pf_prices[_t] = float(_s.iloc[-1])
                    except Exception:
                        pass
            except Exception:
                pass

    # USD/KRW 환율 — 미국 주식 원화 환산용
    _usd_krw = 1300.0
    try:
        _fx_raw = yf.download("USDKRW=X", period="2d", auto_adjust=True, progress=False)
        if not _fx_raw.empty:
            _fx_s = (
                _fx_raw["Close"] if "Close" in _fx_raw.columns
                else _fx_raw.iloc[:, 0]
            ).dropna()
            if not _fx_s.empty:
                _usd_krw = float(_fx_s.iloc[-1])
    except Exception:
        pass

    def _krw(price: float, ticker: str) -> float:
        """USD 종목이면 원화 환산, KRW 종목은 그대로."""
        return price if ticker.upper().endswith((".KS", ".KQ")) else price * _usd_krw

    # 매도 이력 KRW 환산 집계 (통화 혼재 → _krw 함수로 통일)
    _cum_sell_krw   = sum(_krw(t["sell_price"] * t["quantity"], t["ticker"]) for t in _trade_history)
    _cum_profit_krw = sum(_krw(t["net_profit"],                 t["ticker"]) for t in _trade_history)
    _cum_buy_krw    = sum(_krw(t["buy_price"]  * t["quantity"], t["ticker"]) for t in _trade_history)

    # ── 종목 추가 안내 (사이드바에서 추가) ──────────────────────────────────────
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

    # ── 보유 종목 테이블 ──────────────────────────────────────────────────────
    if not _items:
        st.info("보유 종목이 없습니다. 차트 분석 탭에서 평단가를 입력한 뒤 '포트폴리오에 추가' 버튼을 눌러보세요.", icon="💡")
    else:
        # 요약 메트릭 — 미국 주식은 USD/KRW 환율 적용해 원화 환산 합산
        _total_cost = sum(
            _krw(it["avg_price"], it["ticker"]) * it["quantity"] for it in _items
        )
        _total_val  = sum(
            _krw(_pf_prices.get(it["ticker"], it["avg_price"]), it["ticker"]) * it["quantity"]
            for it in _items
        )
        _total_pnl  = _total_val - _total_cost
        _total_pnl_pct = (_total_pnl / _total_cost * 100) if _total_cost else 0.0

        # 헤더 3카드용 — 매 포트폴리오 탭 렌더 시 갱신
        _hdr_total_in = _total_cost + _cum_buy_krw
        st.session_state["_pf_header_summary"] = {
            "total_val":     _total_val,
            "total_pnl":     _total_pnl,
            "total_pnl_pct": _total_pnl_pct,
            "overall_pct":   ((_total_val + _cum_sell_krw) / _hdr_total_in * 100 - 100)
                             if _hdr_total_in > 0 else None,
        }

        _pnl_color    = "#10B981" if _total_pnl >= 0 else "#3B82F6"
        _profit_color = "#10B981" if _cum_profit_krw >= 0 else "#3B82F6"
        _pnl_pct_str  = f"{_total_pnl_pct:+.2f}%"
        _m1, _m2, _m3, _m4 = st.columns(4)
        _m1.markdown(f"""
<div class="ma-metric-card">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px">
    <span style="font-size:.75rem;color:#94A3B8;font-weight:500;letter-spacing:.3px">총 매수 금액</span>
    <span style="color:#8B5CF6"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12V7H5a2 2 0 0 1 0-4h14v4"/><path d="M3 5v14a2 2 0 0 0 2 2h16v-5"/><path d="M18 12a2 2 0 0 0 0 4h4v-4Z"/></svg></span>
  </div>
  <div style="font-size:1.35rem;font-weight:700;color:#E2E8F0;line-height:1.2">₩{_total_cost:,.0f}</div>
  <div style="font-size:.72rem;color:#64748B;margin-top:6px">평단가 × 수량 합계</div>
</div>
""", unsafe_allow_html=True)
        _m2.markdown(f"""
<div class="ma-metric-card">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px">
    <span style="font-size:.75rem;color:#94A3B8;font-weight:500;letter-spacing:.3px">현재 평가금</span>
    <span style="color:#3B82F6"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg></span>
  </div>
  <div style="font-size:1.35rem;font-weight:700;color:#E2E8F0;line-height:1.2">₩{_total_val:,.0f}</div>
  <div style="font-size:.72rem;color:{_pnl_color};margin-top:6px;font-weight:600">{_pnl_pct_str} &nbsp;·&nbsp; USD/KRW≈{_usd_krw:,.0f}</div>
</div>
""", unsafe_allow_html=True)
        _m3.markdown(f"""
<div class="ma-metric-card">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px">
    <span style="font-size:.75rem;color:#94A3B8;font-weight:500;letter-spacing:.3px">누적 매도금</span>
    <span style="color:#94A3B8"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="1" x2="12" y2="23"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg></span>
  </div>
  <div style="font-size:1.35rem;font-weight:700;color:#E2E8F0;line-height:1.2">₩{_cum_sell_krw:,.0f}</div>
  <div style="font-size:.72rem;color:#64748B;margin-top:6px">매도 회수 총액 (원금+수익)</div>
</div>
""", unsafe_allow_html=True)
        _m4.markdown(f"""
<div class="ma-metric-card">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px">
    <span style="font-size:.75rem;color:#94A3B8;font-weight:500;letter-spacing:.3px">누적 실현 손익</span>
    <span style="color:{_profit_color}"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/></svg></span>
  </div>
  <div style="font-size:1.35rem;font-weight:700;color:{_profit_color};line-height:1.2">₩{_cum_profit_krw:+,.0f}</div>
  <div style="font-size:.72rem;color:#64748B;margin-top:6px">매도 완료 종목 손익 합계</div>
</div>
""", unsafe_allow_html=True)

        # 전체 기간 수익률: (현재 평가금 + 누적 매도금) / (보유 원가 + 매도 원가)
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

        # ── Event Watch — AI 뉴스 감성 분석 ──────────────────────────────────
        _ew_h, _ew_btn_col = st.columns([5, 1])
        _ew_h.markdown("#### 🎯 Event Watch")
        _pf_news_result: dict = st.session_state.get("pf_news_result", {})
        _pf_news_rt_ts: str   = st.session_state.get("pf_news_rt_ts", "")
        if _ew_btn_col.button("뉴스 분석", key="pf_run_news", type="primary",
                              use_container_width=True):
            # 실시간 현재가 갱신 (1분봉) — 뉴스 분석 시 등락률 계산 기반
            _rt_prices_news: dict[str, float] = {}
            _rt_pct_news:    dict[str, float] = {}
            with st.spinner("실시간 현재가 조회 중..."):
                for _it in _items:
                    _rt_d = _realtime_price_1m(_it["ticker"])
                    if _rt_d["price"] > 0:
                        _rt_prices_news[_it["ticker"]] = _rt_d["price"]
                        if _it["avg_price"] > 0:
                            _rt_pct_news[_it["ticker"]] = (
                                _rt_d["price"] / _it["avg_price"] - 1
                            ) * 100
                        else:
                            _rt_pct_news[_it["ticker"]] = 0.0
            # 조회한 실시간 가격으로 _pf_prices 갱신
            _pf_prices.update(_rt_prices_news)
            _rt_now = _now_kst().strftime("%H:%M:%S")

            _pf_holdings = [
                {
                    "ticker":       _it["ticker"],
                    "company_name": _it["ticker"].split(".")[0],
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
            _avg_s  = _pf_news_result.get("portfolio_sentiment_avg", 0.0)
            _avg_l  = _pf_news_result.get("portfolio_sentiment_label", "중립")
            _s_clr  = "#4caf50" if _avg_s >= 0.5 else ("#ef4444" if _avg_s <= -0.5 else "#888")
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
                _blbl = f"{_bnm}<br><span style='font-size:.72rem;color:#777'>{_bt}</span>" if _bnm else _bt
                # 실시간 가격 대비 뉴스 영향력 표시
                _cur_p  = _pf_prices.get(_bt, 0.0)
                _avg_p  = next((i["avg_price"] for i in _items if i["ticker"] == _bt), 0.0)
                _pnl_pc = (_cur_p / _avg_p - 1) * 100 if (_cur_p and _avg_p) else None
                _pnl_html = ""
                if _pnl_pc is not None:
                    _pnl_c   = "#4caf50" if _pnl_pc >= 0 else "#ef4444"
                    _impact  = (
                        "호재 반영↑" if (_bs >= 0.5 and _pnl_pc >= 0)
                        else "악재 하락↓" if (_bs <= -0.5 and _pnl_pc < 0)
                        else "뉴스↑ 가격↓" if (_bs >= 0.5 and _pnl_pc < 0)
                        else "뉴스↓ 가격↑" if (_bs <= -0.5 and _pnl_pc >= 0)
                        else "중립"
                    )
                    _pnl_html = (
                        f"<span style='font-size:.7rem;color:{_pnl_c};margin-left:4px'>"
                        f"({_pnl_pc:+.1f}% {_impact})</span>"
                    )
                _bdg += (
                    f'<span style="background:#252836;border-radius:8px;'
                    f'padding:6px 12px;font-size:.85rem;line-height:1.5">'
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
                        f'border-radius:4px;padding:2px 6px;margin-right:8px;'
                        f'font-weight:700">[{_akw}]</span>'
                        f'<b style="color:#e0e0e0">{_aname}</b> — '
                        f'<span style="color:#ccc">{_al["title"]}</span></div>',
                        unsafe_allow_html=True,
                    )
        else:
            st.caption("'뉴스 분석' 버튼을 눌러 포트폴리오 종목의 뉴스 감성을 분석하세요.")

        st.divider()

        # 종목별 수익률 표 (AI 의견 컬럼 포함)
        _pf_per_news: dict = (
            st.session_state.get("pf_news_result", {}).get("per_ticker", {})
        )

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
.pp{color:#10B981;font-weight:700}
.pn{color:#3B82F6;font-weight:700}
.pz{color:#888}
.ai-op{text-align:left!important;font-size:.8rem;max-width:200px;word-break:keep-all;line-height:1.4}
</style>"""

        _tbl_head = (
            '<table class="pf-tbl"><thead><tr>'
            '<th>종목</th><th>수량</th><th>평단가</th>'
            '<th>현재가</th><th>수익률(%)</th><th>평가손익</th>'
            '<th>AI 의견</th>'
            '</tr></thead><tbody>'
        )

        _tbl_rows_html = []
        for _it in _items:
            _t   = _it["ticker"]
            _avg = _it["avg_price"]
            _qty = _it["quantity"]
            _cur = _pf_prices.get(_t)
            _krw = _t.upper().endswith((".KS", ".KQ"))
            _fp  = (lambda v, k=_krw: f"₩{v:,.0f}" if k else f"${v:,.2f}")
            _nm  = _pf_nm.get(_t, "")

            if _cur is not None:
                _pnl     = (_cur - _avg) * _qty
                _pnl_pct = (_cur / _avg - 1) * 100 if _avg else 0.0
                _cls     = "pp" if _pnl_pct > 0 else ("pn" if _pnl_pct < 0 else "pz")
                _cur_str = _fp(_cur)
                _pct_str = f"{_pnl_pct:+.2f}%"
                _pnl_str = (f"+{_fp(abs(_pnl))}" if _pnl >= 0 else f"-{_fp(abs(_pnl))}")
            else:
                _cls     = "pz"
                _cur_str = "-"
                _pct_str = "-"
                _pnl_str = "-"

            # AI 의견 생성
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

            # 종목 셀: 이름(굵게) + 티커(작은 회색)
            _name_cell = (
                f"<div style='font-weight:600;color:#e0e0e0'>{_nm}</div>"
                f"<div style='font-size:.75rem;color:#666'>{_t}</div>"
            ) if _nm else _t

            _tbl_rows_html.append(
                f"<tr>"
                f"<td style='text-align:left'>{_name_cell}</td>"
                f"<td>{_qty:g}</td>"
                f"<td>{_fp(_avg)}</td>"
                f"<td>{_cur_str}</td>"
                f"<td class='{_cls}'>{_pct_str}</td>"
                f"<td class='{_cls}'>{_pnl_str}</td>"
                f"<td class='ai-op' style='color:{_ai_c}'>{_ai_txt}</td>"
                f"</tr>"
            )

        st.markdown(
            _tbl_css + _tbl_head + "".join(_tbl_rows_html) + "</tbody></table>",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # ── 트레일링 스탑 추적 (매 렌더마다 최고가 갱신) ────────────────────
        _trailing: dict = st.session_state.setdefault("pf_trailing_max", {})
        for _it in _items:
            _c = _pf_prices.get(_it["ticker"])
            if _c:
                _t_key = _it["ticker"]
                _trailing[_t_key] = max(_trailing.get(_t_key, 0.0), _c)

        # ── 매도 가이드 (Exit Strategy) ──────────────────────────────────────
        with st.expander("🎯 매도 가이드 (Exit Strategy)", expanded=False):
            _eg_h, _eg_btn = st.columns([5, 1])
            _eg_h.markdown(
                "<small style='color:#999'>실시간가 기준 손절/익절 가이드 · 트레일링 스탑 · 목표가 근접 알림</small>",
                unsafe_allow_html=True,
            )
            _exit_result: dict = st.session_state.get("pf_exit_result", {})
            _exit_rt_ts: str   = st.session_state.get("pf_exit_rt_ts", "")
            if _eg_btn.button("매도 가이드 분석", key="pf_exit_calc", type="primary",
                              use_container_width=True):
                _exit_result = {}
                _exit_now = _now_kst().strftime("%H:%M:%S")

                with st.spinner("실시간 현재가 조회 및 차트 분석 중..."):
                    for _it in _items:
                        _t = _it["ticker"]
                        try:
                            # 실시간 1분봉 현재가 갱신 (KST 기준)
                            _rt_exit = _realtime_price_1m(_t)
                            if _rt_exit["price"] > 0:
                                _pf_prices[_t] = _rt_exit["price"]
                            if _rt_exit.get("stale") and _rt_exit.get("stale_msg"):
                                st.caption(f"⏸️ {_t}: {_rt_exit['stale_msg']}")

                            _cdata = get_stock_data(_t, period="3mo")
                            if _cdata is not None and not _cdata.empty:
                                _sell_targets = get_sell_target_price(_cdata)
                            else:
                                _sell_targets = {}

                            _rt_p   = _pf_prices.get(_t, _it["avg_price"])
                            _avg_p  = _it["avg_price"]
                            _pnl_pc = (_rt_p / _avg_p - 1) * 100 if _avg_p else 0.0

                            # 손절/익절 가이드 생성
                            _stop8    = _avg_p * 0.92           # 기본 8% 손절
                            _stop5    = _avg_p * 0.95           # 타이트 5% 손절
                            _tp1      = _avg_p * 1.10           # 1차 익절 +10%
                            _tp2      = _avg_p * 1.20           # 2차 익절 +20%

                            if _pnl_pc >= 20:
                                _guide = "익절 구간 진입 — 분할 매도 권장 (1/2 이상 청산 고려)"
                                _guide_clr = "#ffd93d"
                            elif _pnl_pc >= 10:
                                _guide = "수익 중 — 1차 목표가 도달. 일부 수익 확정 고려"
                                _guide_clr = "#81c784"
                            elif _pnl_pc > 0:
                                _guide = "소폭 수익 — 홀딩 유지 권장"
                                _guide_clr = "#a5d6a7"
                            elif _pnl_pc > -5:
                                _guide = "소폭 손실 — 추세 모니터링"
                                _guide_clr = "#fff176"
                            elif _pnl_pc > -8:
                                _guide = "손절 접근 — 5% 스탑라인 이탈 시 즉시 손절 고려"
                                _guide_clr = "#ffab91"
                            else:
                                _guide = "손절 구간 — 추가 손실 방지를 위한 즉시 손절 권장"
                                _guide_clr = "#ef9a9a"

                            _exit_result[_t] = {
                                **_sell_targets,
                                "rt_price":  _rt_p,
                                "avg_price": _avg_p,
                                "pnl_pct":   _pnl_pc,
                                "stop_loss_8": _stop8,
                                "stop_loss_5": _stop5,
                                "take_profit_1": _tp1,
                                "take_profit_2": _tp2,
                                "guide":     _guide,
                                "guide_clr": _guide_clr,
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
                _shown_rt = next(
                    (v.get("rt_ts") for v in _exit_result.values() if v.get("rt_ts")), _exit_rt_ts
                )
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
                _krw = _t.upper().endswith((".KS", ".KQ"))
                _efmt = (lambda v, k=_krw: f"₩{v:,.0f}" if k else f"${v:,.2f}")
                _enm  = _pf_nm.get(_t, "")

                _stp   = _exit_result.get(_t, {})
                _cons  = _stp.get("conservative_target")
                _aggr  = _stp.get("aggressive_target")
                _t_max = _trailing.get(_t, 0.0)
                # 매도 가이드 분석 결과 우선 사용
                _guide_txt = _stp.get("guide", "")
                _guide_clr = _stp.get("guide_clr", "#888")
                _stop8_p   = _stp.get("stop_loss_8", _avg * 0.92 if _avg else 0)
                _stop5_p   = _stp.get("stop_loss_5", _avg * 0.95 if _avg else 0)
                _tp1_p     = _stp.get("take_profit_1", _avg * 1.10 if _avg else 0)
                _tp2_p     = _stp.get("take_profit_2", _avg * 1.20 if _avg else 0)
                _pnl_pc    = _stp.get("pnl_pct")
                _cur_label = "실시간가" if _stp.get("is_rt") else "현재가"

                # 알림 계산
                _exit_alerts: list[tuple[str, str]] = []
                if _cur and _t_max > 0 and _cur < _t_max * 0.95:
                    _drop = (_cur / _t_max - 1) * 100
                    _exit_alerts.append((
                        f"⚠️ 트레일링 스탑: 최고가 {_efmt(_t_max)} 대비 {_drop:.1f}% — "
                        "수익 보존을 위한 매도 권장",
                        "#ff8a65",
                    ))
                if _cur and _cons and _cur >= _cons * 0.95:
                    _prox = _cur / _cons * 100
                    _exit_alerts.append((
                        f"🎯 목표가 근접 ({_prox:.0f}%): 보수적 목표가 {_efmt(_cons)} — "
                        "분할 매도 고려",
                        "#ffd93d",
                    ))
                # 손절 경고
                if _cur and _avg and _cur <= _avg * 0.92:
                    _exit_alerts.append((
                        f"🔴 손절 구간 돌입: 실시간가 {_efmt(_cur)} — 8% 손절라인 이탈, 즉시 매도 권장",
                        "#ef4444",
                    ))
                elif _cur and _avg and _cur <= _avg * 0.95:
                    _exit_alerts.append((
                        f"🟠 손절 주의: 실시간가 {_efmt(_cur)} — 5% 스탑라인 접근",
                        "#ff8a65",
                    ))

                # 카드 본문
                _cp: list[str] = []
                if _cur:
                    _pr  = (_cur / _avg - 1) * 100 if _avg else 0
                    _prc = "#4caf50" if _pr >= 0 else "#ef4444"
                    _cp.append(
                        f'{_cur_label} <b style="color:#ddd">{_efmt(_cur)}</b> '
                        f'<span style="color:{_prc}">({_pr:+.1f}%)</span>'
                    )
                if _avg:
                    _cp.append(f'평단가 <b style="color:#aaa">{_efmt(_avg)}</b>')
                if _stop5_p:
                    _cp.append(
                        f'손절(5%) <b style="color:#ff8a65">{_efmt(_stop5_p)}</b>'
                    )
                if _stop8_p:
                    _cp.append(
                        f'손절(8%) <b style="color:#ef4444">{_efmt(_stop8_p)}</b>'
                    )
                if _tp1_p:
                    _cp.append(
                        f'1차 익절 <b style="color:#81c784">{_efmt(_tp1_p)}</b>'
                    )
                if _tp2_p:
                    _cp.append(
                        f'2차 익절 <b style="color:#4fc3f7">{_efmt(_tp2_p)}</b>'
                    )
                if _cons:
                    _cg = (_cons / _avg - 1) * 100 if _avg else 0
                    _cp.append(
                        f'보수적 목표가 <b style="color:#81c784">{_efmt(_cons)}</b> '
                        f'<span style="color:#81c784">({_cg:+.1f}%)</span>'
                    )
                if _aggr:
                    _ag = (_aggr / _avg - 1) * 100 if _avg else 0
                    _cp.append(
                        f'공격적 목표가 <b style="color:#4fc3f7">{_efmt(_aggr)}</b> '
                        f'<span style="color:#4fc3f7">({_ag:+.1f}%)</span>'
                    )
                if _t_max and _cur and _t_max > _cur:
                    _cp.append(
                        f'추적 최고가 <b style="color:#ffb74d">{_efmt(_t_max)}</b> '
                        f'| 스탑라인 <b style="color:#ff8a65">{_efmt(_t_max * 0.95)}</b>'
                    )

                _card_body = " &nbsp;|&nbsp; ".join(_cp) if _cp else "-"
                _guide_html = (
                    f'<div style="margin-top:6px;padding:6px 10px;background:#252836;'
                    f'border-left:3px solid {_guide_clr};border-radius:0 6px 6px 0;'
                    f'font-size:.85rem;color:{_guide_clr};font-weight:600">'
                    f'📋 {_guide_txt}</div>'
                ) if _guide_txt else ""
                _alert_html = "".join(
                    f'<div style="margin-top:6px;padding:6px 10px;background:#252836;'
                    f'border-left:3px solid {_ac};border-radius:0 6px 6px 0;'
                    f'font-size:.82rem;color:{_ac}">{_amsg}</div>'
                    for _amsg, _ac in _exit_alerts
                )
                st.markdown(
                    f'<div style="background:#1e2130;border-radius:10px;'
                    f'padding:12px 16px;margin:6px 0">'
                    f'<div style="margin-bottom:6px">'
                    + (f'<span style="font-size:.95rem;font-weight:700;color:#e0e0e0">{_enm}</span> ' if _enm else "")
                    + f'<span style="font-size:.78rem;color:#666">{_t}</span></div>'
                    f'<div style="font-size:.85rem;line-height:1.9">{_card_body}</div>'
                    f'{_guide_html}'
                    f'{_alert_html}</div>',
                    unsafe_allow_html=True,
                )

                # 매도 확정 버튼 — 분석 결과가 있을 때만 표시
                if _stp and _stp.get("rt_price"):
                    _rt_sell_px = _stp["rt_price"]
                    _exit_qty_col, _sell_btn_col, _ = st.columns([2, 2, 3])
                    _exit_sell_qty = _exit_qty_col.number_input(
                        "매도수량",
                        min_value=0.01,
                        max_value=float(_it["quantity"]),
                        value=float(_it["quantity"]),
                        step=1.0,
                        format="%.2f",
                        key=f"pf_exit_qty_{_it['id']}",
                        help=f"최대 {_it['quantity']:g}주",
                    )
                    if _sell_btn_col.button(
                        f"✅ 매도 확정  {_efmt(_rt_sell_px)}",
                        key=f"pf_sell_exit_{_it['id']}",
                        help=(
                            f"실시간가 {_efmt(_rt_sell_px)} 기준으로 "
                            f"{_exit_sell_qty:g}주 매도 기록"
                        ),
                    ):
                        _sell_r = _db_sell_item(_uid, _it["id"], _rt_sell_px, _exit_sell_qty)
                        if _sell_r["ok"]:
                            _pnl_d = (
                                f"₩{_sell_r['net_profit']:+,.0f}"
                                if _krw else
                                f"${_sell_r['net_profit']:+,.2f}"
                            )
                            st.success(
                                f"매도 완료! 실현 손익: {_pnl_d}"
                                f" ({_sell_r['return_rate']:+.2f}%)"
                            )
                            st.rerun(scope="fragment")
                        else:
                            st.error(_sell_r.get("error", "매도 실패"))

        st.markdown("<br>", unsafe_allow_html=True)

        # 매도 기록 / 종목 삭제
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
                    "매도가",
                    min_value=0.01,
                    value=_default_px,
                    step=100.0 if _is_krw_del else 0.01,
                    format="%.0f" if _is_krw_del else "%.2f",
                    key=f"pf_sell_px_{_it['id']}",
                    label_visibility="collapsed",
                )
                _sell_qty_input = _dc3.number_input(
                    "매도수량",
                    min_value=0.01,
                    max_value=float(_it["quantity"]),
                    value=float(_it["quantity"]),
                    step=1.0,
                    format="%.2f",
                    key=f"pf_sell_qty_{_it['id']}",
                    label_visibility="collapsed",
                    help=f"최대 {_it['quantity']:g}주",
                )
                if _dc4.button("매도", key=f"pf_sell_{_it['id']}", type="primary",
                               use_container_width=True):
                    _sell_r = _db_sell_item(_uid, _it["id"], _sell_px_input, _sell_qty_input)
                    if _sell_r["ok"]:
                        _pnl_msg = (
                            f"₩{_sell_r['net_profit']:+,.0f}"
                            if _is_krw_del else
                            f"${_sell_r['net_profit']:+,.2f}"
                        )
                        st.toast(
                            f"매도 완료! 손익: {_pnl_msg}"
                            f" ({_sell_r['return_rate']:+.2f}%)"
                        )
                        st.rerun(scope="fragment")
                    else:
                        st.error(_sell_r.get("error", "매도 실패"))

                if _dc5.button("삭제", key=f"pf_del_{_it['id']}", use_container_width=True):
                    _r3 = _db_delete_portfolio(_it["id"], _uid)
                    if _r3["ok"]:
                        st.rerun(scope="fragment")
                    else:
                        st.error(_r3.get("error", "삭제 실패"))

        st.divider()

        # ── AI의 이번 주 제안 ─────────────────────────────────────────────────
        _ai_h, _ai_btn = st.columns([5, 1])
        _ai_h.markdown("#### 🤖 AI의 이번 주 제안")
        _opt_result: dict = st.session_state.get("pf_opt_result", {})
        if _ai_btn.button("섹터 분석", key="pf_opt_run", type="primary",
                          use_container_width=True):
            from src.portfolio_optimizer import (
                classify_sectors, scan_sector_etfs, build_rebalancing_guide,
            )
            with st.spinner("섹터 분석 및 시장 주도주 스캔 중..."):
                _sd  = classify_sectors(_items, _pf_prices)
                _es  = scan_sector_etfs()
                _opt_result = {
                    "sector_data": _sd,
                    "etf_scan":    _es,
                    "guide":       build_rebalancing_guide(_sd, _es, _pf_nm),
                }
            st.session_state["pf_opt_result"] = _opt_result

        if not _opt_result:
            st.caption("'섹터 분석' 버튼으로 포트폴리오 섹터 편중도와 리밸런싱 제안을 확인하세요.")
        else:
            _sd_r   = _opt_result["sector_data"]
            _es_r   = _opt_result["etf_scan"]
            _guide  = _opt_result["guide"]
            _sctrs  = _sd_r.get("sectors", {})

            # ── 섹터 비중 바 차트 ──────────────────────────────────────────
            if _sctrs:
                st.markdown("**📊 섹터 비중**")
                _bar_sorted = sorted(_sctrs.items(), key=lambda x: x[1]["weight"], reverse=True)
                _bar_max    = max(v["weight"] for _, v in _bar_sorted) or 1
                _bar_html   = '<div style="display:grid;gap:5px;margin-bottom:12px">'
                for _sn, _sv in _bar_sorted:
                    _sw   = _sv["weight"]
                    _bc   = "#ef4444" if _sw > 30 else ("#ffd93d" if _sw > 20 else "#4fc3f7")
                    _bpct = _sw / _bar_max * 100
                    _tks  = ", ".join(
                        _pf_nm.get(t, t) or t for t in _sv["tickers"]
                    )
                    _bar_html += (
                        f'<div style="display:flex;align-items:center;gap:8px">'
                        f'<div style="width:80px;font-size:.8rem;color:#ccc;text-align:right">{_sn}</div>'
                        f'<div style="flex:1;background:#252836;border-radius:4px;height:18px;overflow:hidden">'
                        f'<div style="width:{_bpct:.0f}%;height:100%;background:{_bc};'
                        f'border-radius:4px;transition:.3s"></div></div>'
                        f'<div style="width:44px;font-size:.8rem;font-weight:700;color:{_bc}">'
                        f'{_sw:.1f}%</div>'
                        f'<div style="font-size:.75rem;color:#666">{_tks}</div>'
                        f'</div>'
                    )
                _bar_html += "</div>"
                st.markdown(_bar_html, unsafe_allow_html=True)

            # ── 4개 카드 그리드 ────────────────────────────────────────────
            _gc1, _gc2 = st.columns(2)

            # 카드 공통 CSS
            _card_css = """<style>
.opt-card{background:#1e2130;border-radius:12px;padding:16px;margin-bottom:8px;min-height:100px}
.opt-card-title{font-size:.8rem;font-weight:700;color:#9e9e9e;margin-bottom:10px;letter-spacing:.5px}
.opt-item{font-size:.85rem;line-height:1.7;padding:6px 10px;background:#252836;
          border-radius:8px;margin:4px 0;word-break:keep-all}
.opt-empty{color:#555;font-size:.82rem;font-style:italic}
</style>"""
            st.markdown(_card_css, unsafe_allow_html=True)

            # 카드 1 — 섹터 집중 위험
            _warn = _guide.get("concentration_warnings", [])
            _warn_body = ""
            for _w in _warn:
                _wc  = "#ef4444" if _w["weight"] > 40 else "#ff8a65"
                _wtk = ", ".join(_pf_nm.get(t, t) or t for t in _w["tickers"])
                _warn_body += (
                    f'<div class="opt-item" style="border-left:3px solid {_wc}">'
                    f'<b style="color:{_wc}">{_w["sector"]} {_w["weight"]:.1f}%</b> 집중 — '
                    f'<span style="color:#aaa">{_wtk}</span>'
                    f'<br><span style="font-size:.75rem;color:#888">30% 초과: 비중 분산 권고</span>'
                    f'</div>'
                )
            if not _warn_body:
                _warn_body = '<span class="opt-empty">집중 위험 없음 — 양호한 분산</span>'
            _gc1.markdown(
                f'<div class="opt-card">'
                f'<div class="opt-card-title">⚠️ 섹터 집중 위험</div>'
                f'{_warn_body}</div>',
                unsafe_allow_html=True,
            )

            # 카드 2 — 신규 편입 후보
            _cands = _guide.get("new_candidates", [])
            _cand_body = ""
            for _cd in _cands:
                _cand_body += (
                    f'<div class="opt-item" style="border-left:3px solid #4fc3f7">'
                    f'<b style="color:#4fc3f7">{_cd["name"]}</b> '
                    f'<span style="color:#81c784">+{_cd["return_5d"]:.1f}%</span>'
                    f'<br><span style="font-size:.75rem;color:#888">{_cd["reason"]}</span>'
                    f'</div>'
                )
            if not _cand_body:
                _cand_body = '<span class="opt-empty">현재 추천 섹터 없음</span>'
            _gc2.markdown(
                f'<div class="opt-card">'
                f'<div class="opt-card-title">🎯 신규 편입 후보</div>'
                f'{_cand_body}</div>',
                unsafe_allow_html=True,
            )

            # 카드 3 — 수익 확정 권고
            _pt = _guide.get("profit_take", [])
            _pt_body = ""
            for _p in _pt:
                _pt_body += (
                    f'<div class="opt-item" style="border-left:3px solid #ffd93d">'
                    f'<b style="color:#e0e0e0">{_p["name"]}</b> '
                    f'<span style="color:#4caf50;font-weight:700">+{_p["pnl_pct"]:.1f}%</span>'
                    f'<br><span style="font-size:.75rem;color:#888">{_p["reason"]}</span>'
                    f'</div>'
                )
            if not _pt_body:
                _pt_body = '<span class="opt-empty">수익 확정 기준(+15%) 도달 종목 없음</span>'
            _gc1.markdown(
                f'<div class="opt-card">'
                f'<div class="opt-card-title">💰 수익 확정 권고</div>'
                f'{_pt_body}</div>',
                unsafe_allow_html=True,
            )

            # 카드 4 — 추가 매수 권고
            _ab = _guide.get("add_buy", [])
            _ab_body = ""
            for _a in _ab:
                _ab_body += (
                    f'<div class="opt-item" style="border-left:3px solid #81c784">'
                    f'<b style="color:#e0e0e0">{_a["name"]}</b>'
                    f'<br><span style="font-size:.75rem;color:#888">{_a["reason"]}</span>'
                    f'</div>'
                )
            if not _ab_body:
                _ab_body = '<span class="opt-empty">추가 매수 후보 없음</span>'
            _gc2.markdown(
                f'<div class="opt-card">'
                f'<div class="opt-card-title">📈 추가 매수 권고</div>'
                f'{_ab_body}</div>',
                unsafe_allow_html=True,
            )

        st.divider()

        # ── AI 모멘텀 주도주 포트폴리오 ──────────────────────────────────────
        st.markdown("#### 📈 AI 전략 자산 배분 — 모멘텀 주도주 포트폴리오")

        with st.expander("💡 분석 가이드 — 차트 정밀 진단과 무엇이 다른가요?", expanded=False):
            st.markdown(
                "**이 추천은 '추세추종형(모멘텀)' 관점으로 종목을 선정합니다.**\n\n"
                "시장 주도력과 뉴스 심리가 **현재 양호한** 종목을 골라 투자금을 최적 배분합니다. "
                "따라서 과매도 구간(RSI 저점)에서 반등을 노리는 "
                "**차트 정밀 진단** 결과와 추천 종목이 다를 수 있습니다.\n\n"
                "| 구분 | AI 모멘텀 추천 (이 섹션) | 차트 정밀 진단 (차트 분석 탭) |\n"
                "|------|--------------------------|--------------------------------|\n"
                "| 전략 | 추세추종 — 이미 오르는 종목 편승 | 역추세 — 과매도 반등 포착 |\n"
                "| RSI 기준 | **RSI < 30 제외** (불안정 회피) | **RSI < 30 = 강력 매수 신호** |\n"
                "| 뉴스 처리 | 키워드 고속 분석 (0~1 정규화) | LLM 정밀 분석 포함 (멀티소스) |\n"
                "| 후보 풀 | **코스피+코스닥+나스닥 각 상위 500개** (동적) | 사용자 선택 종목 (제한 없음) |\n"
                "| 출력 | 매수 수량 + 섹터별 비중 배분 | 매수/매도 신호 강도 |\n\n"
                "> ℹ️ **뉴스 점수 차이**: 이 추천의 뉴스 감성은 키워드 기반 고속 처리로 계산되어 "
                "차트 탭의 LLM 정밀 분석 점수와 수치가 다를 수 있습니다. "
                "두 점수는 동일 종목이라도 ±0.3~1.0 수준의 차이가 발생할 수 있습니다."
            )

        _rec_c1, _rec_c2, _rec_c3 = st.columns([3, 1, 1])
        _inv_amt = _rec_c1.number_input(
            "투자 예정 금액 (원)",
            min_value=100_000,
            max_value=1_000_000_000,
            value=5_000_000,
            step=500_000,
            format="%d",
            key="rec_investment_amount",
            help="실제 매수에 사용할 원화 금액을 입력하세요.",
        )
        _risk_profile = _rec_c2.selectbox(
            "투자 성향",
            ["중립형", "보수형", "공격형"],
            key="rec_risk_profile",
        )
        _rec_c3.write("")   # 레이블 높이 보정
        _rec_run = _rec_c3.button(
            "모멘텀 추천 실행", type="primary", key="rec_run",
            use_container_width=True,
        )

        if _rec_run:
            from src.recommendation_engine import (
                run_recommendation, recommendation_to_dict,
            )
            with st.spinner(
                "KOSPI·KOSDAQ·NASDAQ 각 상위 500개 후보 로드 → "
                "L1 RSI·모멘텀 필터 → L2 뉴스 감성 분석 중... (1~2분 소요)"
            ):
                _rec_result = run_recommendation(
                    investment_amount=int(_inv_amt),
                    risk_profile=_risk_profile,
                    api_key=st.session_state.get("gemini_api_key", ""),
                    groq_api_key=st.session_state.get("groq_api_key", ""),
                )
            st.session_state["pf_rec_result"] = _rec_result

            # 결과 DB 저장
            if not _rec_result.get("error") and _rec_result.get("recommendations"):
                _to_save = [
                    recommendation_to_dict(r)
                    for r in _rec_result["recommendations"]
                ]
                try:
                    _db_save_recommendation(_uid, int(_inv_amt), _risk_profile, _to_save)
                except Exception:
                    pass

        _rec_result: dict = st.session_state.get("pf_rec_result", {})

        if _rec_result.get("error"):
            st.warning(_rec_result["error"], icon="⚠️")

        elif _rec_result.get("recommendations"):
            _recs        = _rec_result["recommendations"]
            _total_inv   = _rec_result.get("total_invested", 0.0)
            _remaining   = _rec_result.get("remaining_cash", 0.0)

            # 요약 메트릭 바
            _pool_sz  = _rec_result.get("pool_size", 0)
            _l1_cnt   = _rec_result.get("l1_pass", 0)
            _l2_cnt   = _rec_result.get("l2_pass", 0)
            _rm1, _rm2, _rm3, _rm4 = st.columns(4)
            _rm1.metric("투자 예정 금액", f"₩{int(_inv_amt):,}")
            _rm2.metric("실제 투자액",   f"₩{_total_inv:,.0f}")
            _rm3.metric("매수 후 잔금",  f"₩{_remaining:,.0f}",
                        delta=f"-₩{_total_inv:,.0f}", delta_color="inverse")
            _rm4.metric(
                "분석 깔때기",
                f"{len(_recs)}개 선정",
                help=(
                    f"후보 풀 {_pool_sz}개 → "
                    f"L1(RSI·모멘텀) {_l1_cnt}개 → "
                    f"L2(뉴스감성) {_l2_cnt}개 → "
                    f"최종 {len(_recs)}개"
                ),
            )

            st.markdown("<br>", unsafe_allow_html=True)

            # 종목 카드 (5개 / 열당 최대 3개)
            _card_css_rec = """<style>
.rec-card{background:#1e2130;border-radius:12px;padding:16px 18px;margin-bottom:8px;
          border-top:3px solid #4fc3f7}
.rec-card-name{font-size:1rem;font-weight:700;color:#e0e0e0}
.rec-card-sector{font-size:.72rem;background:#252836;color:#9e9e9e;
                 padding:2px 8px;border-radius:10px;margin-left:6px}
.rec-card-price{font-size:.85rem;color:#aaa;margin-top:4px}
.rec-card-reason{font-size:.8rem;color:#7ecfff;font-style:italic;
                 margin-top:8px;line-height:1.5;border-left:2px solid #4fc3f7;
                 padding-left:8px}
.rec-bar-bg{background:#252836;border-radius:4px;height:8px;margin:6px 0}
.rec-bar-fg{height:8px;border-radius:4px;background:linear-gradient(90deg,#4fc3f7,#81c784)}
</style>"""
            st.markdown(_card_css_rec, unsafe_allow_html=True)

            # 3열 + 2열 레이아웃
            _row1 = _recs[:3]
            _row2 = _recs[3:]

            def _render_rec_card(r, col):
                _sent_pct   = round(r.sentiment_score * 100, 1)
                _weight_pct = round(r.weight * 100, 1)
                _rsi_clr    = (
                    "#4caf50" if r.rsi < 50
                    else "#ffd93d" if r.rsi < 70
                    else "#ef4444"
                )
                # 통화별 가격 포맷
                _cur        = getattr(r, "currency", "KRW")
                _mkt        = getattr(r, "market", "KOSPI")
                _native_px  = getattr(r, "current_price", r.current_price if hasattr(r, "current_price") else 0)
                _price_str  = (
                    f"₩{_native_px:,.0f}" if _cur == "KRW"
                    else f"${_native_px:,.2f}"
                )
                _mkt_badge_clr = (
                    "#4fc3f7" if _mkt == "KOSPI"
                    else "#81c784" if _mkt == "KOSDAQ"
                    else "#ffb74d"
                )
                col.markdown(
                    f'<div class="rec-card">'
                    f'<div style="margin-bottom:6px">'
                    f'<span class="rec-card-name">{r.name}</span>'
                    f'<span class="rec-card-sector">{r.sector}</span>'
                    f'<span style="font-size:.68rem;background:{_mkt_badge_clr}22;color:{_mkt_badge_clr};'
                    f'padding:1px 7px;border-radius:8px;margin-left:4px;font-weight:600">{_mkt}</span>'
                    f'</div>'
                    f'<div class="rec-card-price">'
                    f'현재가 <b style="color:#ddd">{_price_str}</b>'
                    + (f' <span style="font-size:.75rem;color:#777">(₩{getattr(r, "current_price_krw", _native_px):,.0f})</span>'
                       if _cur == "USD" else "")
                    + f'</div>'
                    f'<div style="margin-top:10px;display:flex;gap:10px;flex-wrap:wrap">'
                    f'<span style="font-size:.82rem;color:#9e9e9e">비중</span>'
                    f'<span style="font-size:.9rem;font-weight:700;color:#4fc3f7">{_weight_pct:.1f}%</span>'
                    f'&nbsp;·&nbsp;'
                    f'<span style="font-size:.82rem;color:#9e9e9e">수량</span>'
                    f'<span style="font-size:.9rem;font-weight:700;color:#ffd93d">{r.quantity:,}주</span>'
                    f'&nbsp;·&nbsp;'
                    f'<span style="font-size:.82rem;color:#9e9e9e">투자액</span>'
                    f'<span style="font-size:.9rem;font-weight:700;color:#81c784">₩{r.invested:,.0f}</span>'
                    f'</div>'
                    f'<div style="margin-top:8px;display:flex;gap:16px">'
                    f'<span style="font-size:.78rem;color:#999">뉴스감성 '
                    f'<b style="color:#4fc3f7">{_sent_pct:.0f}</b>/100</span>'
                    f'<span style="font-size:.78rem;color:#999">RSI '
                    f'<b style="color:{_rsi_clr}">{r.rsi:.0f}</b></span>'
                    f'</div>'
                    f'<div class="rec-bar-bg">'
                    f'<div class="rec-bar-fg" style="width:{_weight_pct / 40 * 100:.0f}%"></div>'
                    f'</div>'
                    f'<div class="rec-card-reason">{r.reason}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            if _row1:
                _cols1 = st.columns(min(len(_row1), 3))
                for _r, _c in zip(_row1, _cols1):
                    _render_rec_card(_r, _c)

            if _row2:
                _pad  = (3 - len(_row2)) // 2
                _cols2 = st.columns([1] * _pad + [1] * len(_row2) + [1] * (3 - len(_row2) - _pad))
                for _r, _c in zip(_row2, _cols2[_pad: _pad + len(_row2)]):
                    _render_rec_card(_r, _c)

            # ── XAI: 내 포트폴리오 종목이 추천에 없는 이유 ────────────────
            try:
                from src.recommendation_engine import (
                    SENTIMENT_THRESHOLD, RSI_OVERSOLD_BOUND, RSI_OVERBOUGHT_BOUND,
                )
                _rec_tickers_set = {r.ticker for r in _recs}
                _pf_tickers_set  = {it["ticker"] for it in _items}
                _missed_tickers  = _pf_tickers_set - _rec_tickers_set

                if _missed_tickers:
                    with st.expander(
                        f"🔎 내 포트폴리오 {len(_missed_tickers)}개 종목이 이번 추천에 없는 이유",
                        expanded=False,
                    ):
                        st.caption(
                            "추세추종(모멘텀) 기준으로 선정하므로, "
                            "기술적으로 매력적인 종목이라도 모멘텀 조건 미충족 시 제외될 수 있습니다."
                        )
                        for _mt in _missed_tickers:
                            _mn = _pf_nm.get(_mt, "") or _mt.split(".")[0]
                            _xai_reason = (
                                f"KOSPI·KOSDAQ·NASDAQ 상위 500개 후보 중 이번 분석 기준을 충족하지 못했습니다. "
                                f"선정 기준: RSI {RSI_OVERSOLD_BOUND}–{RSI_OVERBOUGHT_BOUND} 구간 + "
                                f"20일 이동평균선 돌파 + 뉴스 모멘텀 점수 {SENTIMENT_THRESHOLD*100:.0f}점 이상. "
                                "차트 분석 탭에서는 역추세 기술적 지표로 독립 평가되므로 결과가 다를 수 있습니다."
                            )
                            st.markdown(
                                f'<div style="background:#1e2130;border-radius:8px;'
                                f'padding:10px 14px;margin:5px 0;border-left:3px solid #555">'
                                f'<span style="font-size:.9rem;font-weight:600;color:#e0e0e0">'
                                f'🟡 {_mn}</span>'
                                f'<span style="font-size:.75rem;color:#666;margin-left:8px">{_mt}</span>'
                                f'<div style="font-size:.82rem;color:#9e9e9e;margin-top:5px;line-height:1.5">'
                                f'{_xai_reason}</div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
            except Exception:
                pass

        elif not _rec_result:
            st.caption(
                "'모멘텀 추천 실행' 버튼을 누르면 뉴스 감성·RSI 기반으로 "
                "투자금에 맞는 모멘텀 주도주 5개와 매수 수량을 제안합니다."
            )

        # ── 추천 이력 조회 ────────────────────────────────────────────────────
        with st.expander("🕐 지난 추천 이력", expanded=False):
            try:
                _hist = _db_get_rec_history(_uid, limit=5)
            except Exception:
                _hist = []

            if not _hist:
                st.caption("저장된 추천 이력이 없습니다.")
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

        # ── 매도 이력 ────────────────────────────────────────────────────────
        with st.expander("📋 매도 이력", expanded=False):
            if not _trade_history:
                st.caption(
                    "매도 이력이 없습니다. '매도 기록 / 종목 삭제' 또는 "
                    "'매도 가이드'의 매도 확정 버튼으로 기록이 남습니다."
                )
            else:
                _th_rows_data = []
                for _th in _trade_history:
                    _th_t       = _th["ticker"]
                    _th_nm      = _pf_nm.get(_th_t, "") or _th_t
                    _th_is_krw  = _th_t.upper().endswith((".KS", ".KQ"))
                    _th_fp      = (lambda v, k=_th_is_krw: f"₩{v:,.0f}" if k else f"${v:,.2f}")
                    _th_dt      = _th.get("traded_at", "")[:16].replace("T", " ")
                    _th_rows_data.append({
                        "종목":     f"{_th_nm} ({_th_t})" if _th_nm != _th_t else _th_t,
                        "매수가":   _th_fp(_th["buy_price"]),
                        "매도가":   _th_fp(_th["sell_price"]),
                        "수량":     f"{_th['quantity']:g}주",
                        "실현 손익": _th_fp(_th["net_profit"]),
                        "수익률":   f"{_th['return_rate']:+.2f}%",
                        "매도일시": _th_dt,
                    })

                # 요약 헤더 + 초기화 버튼
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
                    f'₩{_cum_profit_krw:+,.0f}</b></span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if _th_clr_col.button("🗑️ 초기화", key="th_clear_btn",
                                      help="매도 이력 전체 삭제 (복구 불가)"):
                    st.session_state["_th_confirm_clear"] = True

                if st.session_state.get("_th_confirm_clear"):
                    st.warning("매도 이력을 전부 삭제합니다. 누적 매도금·실현 손익이 0으로 초기화됩니다.")
                    _cf_ok, _cf_no = st.columns(2)
                    if _cf_ok.button("삭제 확인", key="th_clear_confirm", type="primary",
                                     use_container_width=True):
                        _db_clear_trade_history(_uid)
                        st.session_state.pop("_th_confirm_clear", None)
                        st.toast("매도 이력이 초기화됐습니다.")
                        st.rerun(scope="fragment")
                    if _cf_no.button("취소", key="th_clear_cancel",
                                     use_container_width=True):
                        st.session_state.pop("_th_confirm_clear", None)
                        st.rerun(scope="fragment")

                st.dataframe(
                    _th_rows_data,
                    use_container_width=True,
                    hide_index=True,
                )


with tab_portfolio:
    _render_portfolio_tab()
