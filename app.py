"""
app.py — AI 주식 분석 대시보드 v3.0 (모듈화)
ui/ 모듈을 조립하는 최소 메인 파일.
비즈니스 로직은 src/, 렌더링은 ui/ 에 위임.
"""
import json
import os
import concurrent.futures
import time
import warnings

import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone, timedelta

warnings.filterwarnings("ignore")

# ─── KST ──────────────────────────────────────────────────────────────────────
_KST = timezone(timedelta(hours=9))

def _now_kst() -> datetime:
    return datetime.now(_KST)

# ─── 선택적 의존 ──────────────────────────────────────────────────────────────
try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False

# ─── src/ 비즈니스 로직 ───────────────────────────────────────────────────────
try:
    from src.indicators import (
        get_stock_data, generate_signals, calculate_expected_return,
        get_stop_loss_targets, get_buy_target_price, get_sell_target_price,
        get_advanced_analysis, calculate_vpvr, detect_divergence,
        get_hybrid_signal, get_enhanced_hybrid_signal, check_volume_anomaly,
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
        get_investor_trading_naver, get_investor_trading_naver_history, get_recommendations,
        get_krx_stock_list, get_krx_etf_list, get_us_stock_list,
        get_top_kospi_stocks, get_top_kosdaq_stocks,
        get_top_us_stocks, get_top_nasdaq_stocks,
        is_etf_ticker,
        _flatten_columns,
    )
except Exception as _import_err:
    import traceback as _tb
    st.error(
        f"**모듈 로딩 오류**\n\n```\n{_tb.format_exc()}\n```"
    )
    st.stop()

# ─── DB ───────────────────────────────────────────────────────────────────────
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
from fundamental_db import load_settings_db, save_settings_db

try:
    _db_init()
except Exception as _db_err:
    st.error(f"Supabase 연결 실패: {_db_err}", icon="🚨")
    st.stop()

# ─── ui/ 렌더링 모듈 ──────────────────────────────────────────────────────────
from ui.styles import inject_css
from ui.layouts import (
    render_sidebar,
    render_header,
    render_market_tab,
    render_rec_tab,
    render_news_tab,
    render_fund_tab,
    render_chart_tab,
    render_portfolio_tab,
)
from ui.backtest_tab import render_backtest_tab

# ─── 페이지 설정 ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI 주식 분석 터미널",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

# ─── 쿠키 헬퍼 (JS window.parent 방식 — 외부 컴포넌트 없음) ──────────────────
def _set_cookie(name: str, value: str, days: int = 3650) -> None:
    """로그인 토큰을 브라우저 쿠키에 기록한다."""
    import streamlit.components.v1 as _cv1
    from datetime import datetime as _dt, timedelta as _td, timezone as _tz
    expires = (_dt.now(_tz.utc) + _td(days=days)).strftime("%a, %d %b %Y %H:%M:%S GMT")
    _cv1.html(
        f"<script>try{{window.parent.document.cookie="
        f"'{name}={value};path=/;expires={expires};SameSite=Lax'}}catch(e){{}}</script>",
        height=0,
    )

def _delete_cookie(name: str) -> None:
    """브라우저 쿠키를 만료시킨다."""
    import streamlit.components.v1 as _cv1
    _cv1.html(
        f"<script>try{{window.parent.document.cookie="
        f"'{name}=;path=/;expires=Thu, 01 Jan 1970 00:00:00 GMT'}}catch(e){{}}</script>",
        height=0,
    )

# ─── 설정 관리 ────────────────────────────────────────────────────────────────
WATCHLIST_FILE   = os.path.join(os.path.dirname(__file__), "watchlist.json")
_SECRETS_PATH    = os.path.join(os.path.dirname(__file__), ".streamlit", "secrets.toml")
_SECRETS_KEY_MAP = {
    "gemini_api_key": "GEMINI_API_KEY",
    "groq_api_key":   "GROQ_API_KEY",
    "dart_api_key":   "DART_API_KEY",
    "krx_id":         "KRX_ID",
    "krx_pw":         "KRX_PW",
}

def load_settings() -> dict:
    return load_settings_db()

def save_settings(data: dict) -> None:
    save_settings_db(data)
    _persist_to_secrets(data)

def _persist_to_secrets(data: dict) -> None:
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

# settings.json → DB 1회 마이그레이션
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

# ─── 앱 비밀번호 게이트 ───────────────────────────────────────────────────────
_APP_PASSWORD = "qnwkehlwk"

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
            pw_input  = st.text_input("비밀번호", type="password",
                                      placeholder="비밀번호를 입력하세요",
                                      label_visibility="collapsed")
            submitted = st.form_submit_button("입장하기", use_container_width=True, type="primary")
        if submitted:
            if pw_input == _APP_PASSWORD:
                st.session_state["app_authenticated"] = True
                save_settings({**load_settings(), "app_authenticated": True})
                st.rerun()
            else:
                st.error("비밀번호가 틀렸습니다.")
    st.stop()

# ─── 관심종목 ─────────────────────────────────────────────────────────────────
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

for _sk in ("auth_token", "auth_user_id", "auth_email"):
    if _sk not in st.session_state:
        st.session_state[_sk] = None

_saved_settings = load_settings()

# 쿠키 세션 복원
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

# st.secrets → DB 역방향 동기화
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

for _key in ("gemini_api_key", "groq_api_key", "dart_api_key"):
    if _key not in st.session_state:
        st.session_state[_key] = _secrets_loaded.get(_key) or _saved_settings.get(_key, "")

if _saved_settings.get("krx_id") and not os.environ.get("KRX_ID"):
    os.environ["KRX_ID"] = _saved_settings["krx_id"]
if _saved_settings.get("krx_pw") and not os.environ.get("KRX_PW"):
    os.environ["KRX_PW"] = _saved_settings["krx_pw"]

# ─── 캐시 래퍼 ────────────────────────────────────────────────────────────────

def _market_bucket() -> str:
    """장중(평일 09:00~15:30 KST) → 1분 단위 버킷, 장외 → 1시간 단위 버킷.
    @st.cache_data의 TTL을 런타임에 바꿀 수 없으므로 bucket 인자로 사실상 TTL을 제어한다.
    """
    now = _now_kst()
    m = now.hour * 60 + now.minute
    if now.weekday() < 5 and 9 * 60 <= m < 15 * 60 + 30:
        return now.strftime("%Y%m%d-%H%M")
    return now.strftime("%Y%m%d-%H")

@st.cache_data(ttl=3600)
def _stock_data_inner(ticker: str, period: str, bucket: str) -> "pd.DataFrame":
    _ = bucket  # 캐시 키 분리용 — 함수 본문에서는 사용하지 않음
    return get_stock_data(ticker, period)

def _stock_data(ticker: str, period: str) -> "pd.DataFrame":
    return _stock_data_inner(ticker, period, _market_bucket())

@st.cache_data(ttl=300)
def _movers(n: int = 100):
    stocks = _top_kospi(n)
    return get_market_movers(stocks)

@st.cache_data(ttl=300)
def _full_movers():
    return get_full_market_movers(top_n=10)

@st.cache_data(ttl=300)
def _rates():
    return get_exchange_rates()

@st.cache_data(ttl=600)
def _usdkrw_history():
    d = yf.download("USDKRW=X", period="3mo", auto_adjust=True, progress=False)
    return _flatten_columns(d)

def _get_full_stocks(market: str) -> dict:
    if market == "KOSPI":              return _top_kospi(500)
    elif market == "KOSDAQ":           return _top_kosdaq(500)
    elif market == "KOSPI + KOSDAQ":   return {**_top_kospi(500), **_top_kosdaq(500)}
    elif market == "미국 주식 (나스닥)": return _top_nasdaq(500)
    else:                               return _top_us(503)

@st.cache_data(ttl=300)
def _index_data(sym):
    d = yf.download(sym, period="2d", auto_adjust=True, progress=False)
    return _flatten_columns(d)

@st.cache_data(ttl=3600)
def _fundamental(ticker):
    return get_fundamental_data(ticker)

@st.cache_data(ttl=3600)
def _insider_trades(ticker):
    return get_insider_trades_sec(ticker)

@st.cache_data(ttl=600)
def _naver_news(ticker: str) -> list:
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
def _bench_returns(ticker: str):
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
            raw   = yf.Ticker(ticker).news or []
            items = []
            for it in raw[:10]:
                c = it.get("content", it)
                items.append({
                    "title":     c.get("title", it.get("title", "")),
                    "link":      (c.get("canonicalUrl", {}).get("url") or it.get("link", "#")),
                    "publisher": (c.get("provider", {}).get("displayName") or it.get("publisher", "")),
                    "pub_date":  "",
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
    price_change_pct=None,
    net_foreign_buy=None,
    net_institution_buy=None,
) -> dict:
    cache_key = (
        f"llm_news|{ticker}|{api_key[:8] if api_key else ''}"
        f"|{groq_api_key[:8] if groq_api_key else ''}|{company_name}"
    )
    cached = st.session_state.get(cache_key)
    if cached and (datetime.now() - cached["ts"]).seconds < 600:
        return cached["data"]

    result = analyze_news_fast(
        ticker=ticker, company_name=company_name,
        api_key=api_key, groq_api_key=groq_api_key,
        max_news=12, deep_n=5,
        price_change_pct=price_change_pct,
        net_foreign_buy=net_foreign_buy,
        net_institution_buy=net_institution_buy,
    )
    if not result.get("detail"):
        try:
            raw   = yf.Ticker(ticker).news or []
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

@st.cache_data(ttl=86400, show_spinner="주식 목록을 불러오는 중")
def _krx_stocks():  return get_krx_stock_list()

@st.cache_data(ttl=86400, show_spinner="주식 목록을 불러오는 중")
def _top_kospi(n: int = 500):   return get_top_kospi_stocks(n)

@st.cache_data(ttl=86400, show_spinner="주식 목록을 불러오는 중")
def _top_kosdaq(n: int = 500):  return get_top_kosdaq_stocks(n)

@st.cache_data(ttl=86400, show_spinner="주식 목록을 불러오는 중")
def _top_us(n: int = 503):      return get_top_us_stocks(n)

@st.cache_data(ttl=86400, show_spinner="주식 목록을 불러오는 중")
def _top_nasdaq(n: int = 500):  return get_top_nasdaq_stocks(n)

@st.cache_data(ttl=86400, show_spinner="주식 목록을 불러오는 중")
def _us_stocks():   return get_us_stock_list()

@st.cache_data(ttl=3600, show_spinner="주식 목록을 불러오는 중")
def _etf_stocks_inner(bucket: str) -> dict:
    _ = bucket  # 캐시 키 분리용
    return get_krx_etf_list()

def _etf_stocks() -> dict:
    return _etf_stocks_inner(_market_bucket())

@st.cache_data(ttl=86400, show_spinner="주식 목록을 불러오는 중")
def _all_stocks_merged() -> dict:
    """세 종목 사전을 합산한 결과를 하루 단위로 캐싱."""
    result: dict = {}
    result.update(_krx_stocks() or {})
    result.update(_etf_stocks() or {})
    result.update(_us_stocks() or {})
    return result

# 종목 목록 백그라운드 사전 로딩
if "stock_lists_preloaded" not in st.session_state:
    st.session_state["stock_lists_preloaded"] = True
    _ex = concurrent.futures.ThreadPoolExecutor(max_workers=3, thread_name_prefix="preload")
    _ex.submit(_krx_stocks)
    _ex.submit(_etf_stocks)
    _ex.submit(_us_stocks)
    _ex.shutdown(wait=False)

@st.cache_data(ttl=86400)
def _ticker_name_map() -> dict:
    result: dict[str, str] = {}
    for display, ticker in (_krx_stocks() or {}).items():
        result[ticker] = display.split(" (")[0].strip()
    for display, ticker in (_etf_stocks() or {}).items():
        result[ticker] = display.split(" (")[0].strip()
    for display, ticker in (_us_stocks() or {}).items():
        parts = display.split(" / ")
        result[ticker] = parts[0].strip() if len(parts) > 1 else display.split(" (")[0].strip()
    return result

@st.cache_data(ttl=86400)
def _check_is_etf(ticker: str) -> bool:
    if is_etf_ticker(ticker):
        return True
    if not (ticker.endswith(".KS") or ticker.endswith(".KQ")):
        return False
    try:
        etf_list = _etf_stocks()
        code = ticker.replace(".KS", "").replace(".KQ", "").strip().zfill(6)
        return f"{code}.KS" in etf_list.values() or f"{code}.KQ" in etf_list.values()
    except Exception:
        return False

@st.cache_data(ttl=60)
def _etf_fundamental(ticker: str) -> dict:
    return get_etf_fundamental_data(ticker) or {}

@st.cache_data(ttl=300)
def _inv_data(ticker: str) -> dict:
    return get_investor_trading_naver(ticker) or {}

@st.cache_data(ttl=300)
def _inv_data_history(ticker: str) -> list:
    return get_investor_trading_naver_history(ticker, days=10) or []

@st.cache_data(ttl=60)
def _realtime_price_1m(ticker: str) -> dict:
    _now_k       = _now_kst()
    _ts          = _now_k.strftime("%H:%M:%S")
    _is_kr       = ticker.endswith(".KS") or ticker.endswith(".KQ")
    _after_close = _is_kr and (_now_k.hour * 60 + _now_k.minute) >= (15 * 60 + 30)

    try:
        _fi = yf.Ticker(ticker).fast_info
        _p  = float(_fi.last_price)
        if _p > 0:
            _stale = _after_close
            return {"price": _p, "ts": _ts, "is_realtime": not _stale,
                    "stale": _stale, "stale_msg": "장 마감 후 종가입니다." if _stale else ""}
    except Exception:
        pass

    try:
        _df = yf.download(ticker, period="1d", interval="1m", auto_adjust=True, progress=False)
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
                    return {"price": _p, "ts": _ts, "is_realtime": not (_after_close or _stale),
                            "stale": _stale, "stale_msg": _stale_msg}
    except Exception:
        pass

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

@st.cache_data(ttl=60)
def _wl_price(ticker: str):
    d = yf.download(ticker, period="2d", auto_adjust=True, progress=False)
    d = _flatten_columns(d)
    if len(d) >= 2:
        p   = float(d["Close"].iloc[-1])
        chg = (p - float(d["Close"].iloc[-2])) / float(d["Close"].iloc[-2]) * 100
        return p, chg
    return None, None

def _get_wl_alerts() -> list:
    wl = st.session_state.get("watchlist", [])
    if not wl:
        return []
    cache_key = "wl_alerts|" + "|".join(sorted(w["ticker"] for w in wl))
    cached    = st.session_state.get(cache_key)
    if cached and (datetime.now() - cached["ts"]).seconds < 300:
        return cached["data"]
    alerts = []
    for item in wl:
        try:
            data  = _stock_data(item["ticker"], "3mo")
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

# ─── 섹터 ETF 렌더링 함수 (render_market_tab에 주입) ──────────────────────────
_SECTOR_ETF_LIST = (
    ("SPY",       "S&P 500",                       "미국", "📊 지수"),
    ("QQQ",       "나스닥 100",                    "미국", "📊 지수"),
    ("DIA",       "다우존스",                      "미국", "📊 지수"),
    ("SCHD",      "배당성장",                      "미국", "💰 배당"),
    ("SOXX",      "반도체 (SOX)",                  "미국", "💻 반도체"),
    ("VGT",       "기술주",                        "미국", "🤖 테크"),
    ("BOTZ",      "AI/로봇",                       "미국", "🤖 AI/로봇"),
    ("XLV",       "헬스케어",                      "미국", "🏥 헬스케어"),
    ("XLF",       "금융",                          "미국", "🏦 금융"),
    ("XLE",       "에너지",                        "미국", "⚡ 에너지"),
    ("GDX",       "금광주",                        "미국", "⛏ 원자재"),
    ("LIT",       "2차전지/리튬",                  "미국", "🔋 2차전지"),
    ("TLT",       "미국채 20년",                   "미국", "📋 채권"),
    ("BIL",       "미국채 단기",                   "미국", "📋 채권"),
    ("VNQ",       "리츠",                          "미국", "🏢 리츠"),
    ("069500.KS", "KODEX 200",                     "국내", "📊 코스피"),
    ("229200.KQ", "KODEX 코스닥150",               "국내", "📊 코스닥"),
    ("396500.KS", "TIGER Fn반도체TOP10",           "국내", "💻 반도체"),
    ("464930.KS", "SOL AI반도체소부장",            "국내", "💻 반도체소부장"),
    ("464520.KS", "KoAct 바이오헬스케어액티브",   "국내", "🧬 바이오"),
    ("305710.KS", "KODEX 2차전지산업",             "국내", "🔋 2차전지"),
    ("091180.KS", "KODEX 자동차",                  "국내", "🚗 자동차"),
    ("466920.KS", "SOL 조선TOP3플러스",            "국내", "🚢 조선"),
    ("449450.KS", "PLUS K방산",                    "국내", "🛡 방산"),
    ("484310.KS", "KODEX AI전력핵심설비",          "국내", "⚡ 전력"),
    ("438100.KS", "TIGER 코리아원자력",            "국내", "⚛ 원전"),
    ("440790.KS", "KODEX 로봇액티브",              "국내", "🤖 로봇"),
    ("414270.KS", "PLUS 우주항공&UAM",             "국내", "🚀 우주"),
    ("408220.KS", "PLUS 태양광&ESS",               "국내", "☀ 태양광"),
    ("385530.KS", "KODEX 신재생에너지액티브",      "국내", "🌱 재생에너지"),
    ("228790.KS", "TIGER 화장품",                  "국내", "💄 화장품"),
    ("475580.KS", "ACE KPOP포커스",                "국내", "🎵 엔터"),
    ("300640.KS", "TIGER 지주회사",                "국내", "🏢 지주사"),
    ("466940.KS", "TIGER 은행고배당플러스TOP10",   "국내", "🏦 은행"),
    ("102960.KS", "KODEX 증권",                    "국내", "📈 증권"),
)

@st.cache_data(ttl=600, show_spinner=False)
def _fetch_sector_etfs(etf_list: tuple) -> pd.DataFrame:
    def _fix(t):
        return t + ".KS" if (t.isdigit() and len(t) == 6) else t
    tickers = [_fix(row[0]) for row in etf_list]
    try:
        raw = yf.download(tickers, period="2d", auto_adjust=True, progress=False)
    except Exception:
        return pd.DataFrame()
    try:
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]].rename(columns={"Close": tickers[0]})
        if len(close) < 2:
            return pd.DataFrame()
        rows = []
        for orig, name, market, tag in etf_list:
            t = _fix(orig)
            if t not in close.columns:
                continue
            s = close[t].dropna()
            if len(s) < 2:
                continue
            prev, curr = float(s.iloc[-2]), float(s.iloc[-1])
            if prev == 0:
                continue
            chg = round((curr - prev) / prev * 100, 2)
            rows.append({
                "국가": market, "태그": tag, "ETF명": name, "티커": t,
                "현재가": f"{curr:,.0f}₩" if t.endswith((".KS", ".KQ")) else f"${curr:,.2f}",
                "방향": "🔺" if chg >= 0 else "🔻",
                "등락률(%)": chg,
            })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()

def _render_sector_etf_prices():
    _etf_df = _fetch_sector_etfs(_SECTOR_ETF_LIST)
    if _etf_df.empty:
        st.warning("ETF 데이터를 불러올 수 없습니다.", icon="⚠️")
        return

    _up  = int((_etf_df["등락률(%)"] > 0).sum())
    _dn  = int((_etf_df["등락률(%)"] < 0).sum())
    _avg = float(_etf_df["등락률(%)"].mean())
    _us_avg = float(_etf_df[_etf_df["국가"] == "미국"]["등락률(%)"].mean()) if "미국" in _etf_df["국가"].values else 0.0
    _kr_avg = float(_etf_df[_etf_df["국가"] == "국내"]["등락률(%)"].mean()) if "국내" in _etf_df["국가"].values else 0.0
    _up_c  = "#ef5350" if _up  > _dn else "#8B949E"
    _dn_c  = "#42a5f5" if _dn  > _up else "#8B949E"
    _av_c  = "#ef5350" if _avg > 0 else ("#42a5f5" if _avg < 0 else "#8B949E")
    _us_c  = "#ef5350" if _us_avg > 0 else ("#42a5f5" if _us_avg < 0 else "#8B949E")
    _kr_c  = "#ef5350" if _kr_avg > 0 else ("#42a5f5" if _kr_avg < 0 else "#8B949E")

    def _stat_chip(label: str, value: str, color: str) -> str:
        return (
            f'<div style="flex:1;min-width:0;background:#161B22;border:1px solid #30363D;'
            f'border-radius:10px;padding:7px 6px;text-align:center;">'
            f'<div style="font-size:.6rem;color:#8B949E;white-space:nowrap;overflow:hidden;'
            f'text-overflow:ellipsis;margin-bottom:3px">{label}</div>'
            f'<div style="font-size:.85rem;font-weight:700;color:{color};white-space:nowrap">{value}</div>'
            f'</div>'
        )

    st.markdown(
        '<div style="display:flex;gap:6px;margin-bottom:8px">'
        + _stat_chip("🔺 상승",   f"{_up}개",          _up_c)
        + _stat_chip("🔻 하락",   f"{_dn}개",          _dn_c)
        + _stat_chip("전체 평균", f"{_avg:+.2f}%",      _av_c)
        + _stat_chip("🇺🇸 미국",  f"{_us_avg:+.2f}%",  _us_c)
        + _stat_chip("🇰🇷 국내",  f"{_kr_avg:+.2f}%",  _kr_c)
        + '</div>',
        unsafe_allow_html=True,
    )

    # ETF명 + 티커 합치기 (티커에서 .KS/.KQ 제거)
    _etf_df["ETF"] = (
        _etf_df["ETF명"] + "  ·  " + _etf_df["티커"].str.split(".").str[0]
    )

    def _sort_idx_first(df):
        _country_order = {"국내": 0, "미국": 1}
        tmp = df.assign(
            _p=(~df["태그"].str.startswith("📊")).astype(int),
            _c=df["국가"].map(_country_order).fillna(2),
        )
        return tmp.sort_values(["_c", "_p", "태그"]).drop(columns=["_p", "_c"])

    def _etf_table_html(df: "pd.DataFrame", benchmark_chg: float = None, benchmark_ticker: str = None) -> str:
        _MAX = 15.0
        hdr = (
            '<table style="width:100%;border-collapse:collapse;font-size:0.82rem">'
            '<thead><tr style="border-bottom:1px solid #30363D;color:#8B949E">'
            '<th style="padding:5px 4px;text-align:left;width:36px">국가</th>'
            '<th style="padding:5px 4px;text-align:left;width:90px">테마</th>'
            '<th style="padding:5px 4px;text-align:left">ETF · 티커</th>'
            '<th style="padding:5px 8px;text-align:center;width:68px">현재가</th>'
            '<th style="padding:5px 8px;text-align:right;width:160px">등락률</th>'
            '</tr></thead><tbody>'
        )
        rows = []
        for _, r in df.iterrows():
            chg = float(r["등락률(%)"])
            up  = chg >= 0
            bc  = "#ef5350" if up else "#42a5f5"
            bg  = "rgba(239,83,80,0.10)" if up else "rgba(66,165,245,0.10)"
            bw  = min(abs(chg) / _MAX * 100, 100)
            sgn = "+" if up else ""
            rel = ""
            if benchmark_chg is not None and r["티커"] != benchmark_ticker:
                if chg > benchmark_chg:
                    rel = '<span style="color:#ef5350;font-size:0.68rem;font-weight:600;margin-left:3px">(강)</span>'
                elif chg < benchmark_chg:
                    rel = '<span style="color:#42a5f5;font-size:0.68rem;font-weight:600;margin-left:3px">(약)</span>'
            rows.append(
                f'<tr style="border-bottom:1px solid #21262D;background:{bg}">'
                f'<td style="padding:5px 4px">{r["국가"]}</td>'
                f'<td style="padding:5px 4px;white-space:nowrap">{r["태그"]}</td>'
                f'<td style="padding:5px 4px">{r["ETF"]}</td>'
                f'<td style="padding:5px 8px;text-align:center;white-space:nowrap;font-size:0.78rem">{r["현재가"]}</td>'
                f'<td style="padding:5px 8px">'
                f'<div style="display:flex;align-items:center;gap:5px;justify-content:flex-end">'
                f'<div style="width:52px;background:#21262D;border-radius:3px;height:6px;flex-shrink:0">'
                f'<div style="width:{bw:.1f}%;background:{bc};border-radius:3px;height:100%"></div>'
                f'</div>'
                f'<div style="display:flex;align-items:baseline;justify-content:flex-end;min-width:80px">'
                f'<span style="color:{bc};font-weight:700;font-size:1rem">{sgn}{chg:.2f}%</span>'
                f'{rel}'
                f'</div>'
                f'</div></td>'
                f'</tr>'
            )
        return hdr + "".join(rows) + "</tbody></table>"

    _kr = _etf_df[_etf_df["국가"] == "국내"]
    _us = _etf_df[_etf_df["국가"] == "미국"]

    _kodex200_vals = _etf_df.loc[_etf_df["티커"] == "069500.KS", "등락률(%)"].values
    _kodex200_chg  = float(_kodex200_vals[0]) if len(_kodex200_vals) > 0 else None
    _qqq_vals      = _etf_df.loc[_etf_df["티커"] == "QQQ", "등락률(%)"].values
    _qqq_chg       = float(_qqq_vals[0]) if len(_qqq_vals) > 0 else None

    t1, t2, t3 = st.tabs(["전체", "🇺🇸 미국 ETF", "🇰🇷 국내 ETF"])
    with t1:
        _c_kr, _c_us = st.columns(2)
        with _c_kr:
            st.caption("🇰🇷 국내 ETF — 기준: KODEX 200")
            if not _kr.empty:
                st.markdown(_etf_table_html(_sort_idx_first(_kr), _kodex200_chg, "069500.KS"), unsafe_allow_html=True)
            else:
                st.caption("국내 ETF 데이터 없음")
        with _c_us:
            st.caption("🇺🇸 미국 ETF — 기준: 나스닥 100")
            if not _us.empty:
                st.markdown(_etf_table_html(_sort_idx_first(_us), _qqq_chg, "QQQ"), unsafe_allow_html=True)
            else:
                st.caption("미국 ETF 데이터 없음")
    with t2:
        if not _us.empty:
            st.markdown(_etf_table_html(_sort_idx_first(_us), _qqq_chg, "QQQ"), unsafe_allow_html=True)
        else:
            st.caption("미국 ETF 데이터 없음")
    with t3:
        if not _kr.empty:
            st.markdown(_etf_table_html(_sort_idx_first(_kr), _kodex200_chg, "069500.KS"), unsafe_allow_html=True)
        else:
            st.caption("국내 ETF 데이터 없음")

# ═════════════════════════════════════════════════════════════════════════════
# 사이드바
# ═════════════════════════════════════════════════════════════════════════════
_all_stocks: dict = _all_stocks_merged()

_sidebar_result = render_sidebar(
    all_stocks        = _all_stocks,
    rates             = _rates(),
    indices           = INDICES,
    get_index_data    = _index_data,
    watchlist         = st.session_state.watchlist,
    save_watchlist_fn = save_watchlist,
    get_wl_alerts_fn  = _get_wl_alerts,
    get_wl_price_fn   = _wl_price,
    saved_settings    = _saved_settings,
    load_settings_fn  = load_settings,
    save_settings_fn  = save_settings,
    db_upsert_portfolio = _db_upsert_portfolio,
    krx_stocks_fn     = _krx_stocks,
    etf_stocks_fn     = _etf_stocks,
    us_stocks_fn      = _us_stocks,
    now_kst_fn        = _now_kst,
)

ticker         = _sidebar_result["ticker"]
sname          = _sidebar_result["sname"]
period         = _sidebar_result["period"]
gemini_api_key = _sidebar_result["gemini_api_key"]
groq_api_key   = _sidebar_result["groq_api_key"]
dart_api_key   = _sidebar_result["dart_api_key"]
use_llm        = _sidebar_result["use_llm"]

# ═════════════════════════════════════════════════════════════════════════════
# 헤더
# ═════════════════════════════════════════════════════════════════════════════
render_header(st.session_state.get("_pf_header_summary", {}))

# 로딩 배너 플레이스홀더
_loading_ph = st.empty()

# ═════════════════════════════════════════════════════════════════════════════
# 탭 정의
# ═════════════════════════════════════════════════════════════════════════════
tab_market, tab_chart, tab_rec, tab_news, tab_fund, tab_portfolio, tab_backtest = st.tabs([
    "🌐 시장 현황",
    "📊 차트 분석",
    "⭐ 추천 종목",
    "📰 뉴스 & 관련 종목",
    "🏛️ 펀더멘털 & 기관",
    "💼 내 포트폴리오",
    "🔬 백테스트",
])

# 분석 완료 직후 차트 탭 자동 이동
if st.session_state.pop("_switch_to_chart", False):
    import streamlit.components.v1 as _components
    _components.html(
        """<script>setTimeout(function(){
        var p=window.parent.document;
        var tabs=p.querySelectorAll('[data-testid="stTab"]');
        if(tabs.length>1){tabs[1].click();return;}
        tabs=p.querySelectorAll('button[role="tab"]');
        if(tabs.length>1)tabs[1].click();},400);</script>""",
        height=0,
    )

# ═════════════════════════════════════════════════════════════════════════════
# 데이터 로딩 (분석 시작 버튼 클릭 후에만 실행)
# ═════════════════════════════════════════════════════════════════════════════
_aticker    = st.session_state.get("analyzed_ticker")
_asname     = st.session_state.get("analyzed_sname", "")
_aperiod    = st.session_state.get("analyzed_period", period)
_pending    = st.session_state.get("_pending_ticker")
_data_ready = bool(_aticker)

_LOADING_BASE = (
    '<div style="background:#161B22;border:1px solid #30363D;border-radius:12px;'
    'padding:28px;text-align:center;margin:8px 0 16px;box-shadow:0 4px 24px rgba(0,0,0,0.4);">'
    '<div class="loading-icon" style="font-size:36px;margin-bottom:12px;">{icon}</div>'
    '<div style="font-size:1.05rem;font-weight:700;color:#E6EDF3;margin-bottom:8px;">{title}</div>'
    '<div style="color:#8B949E;font-size:0.88rem;line-height:1.7;">'
    '<span style="color:#E6EDF3;font-weight:600;">{name}</span> {msg}</div>'
    '<div class="loading-bar-track"><div class="loading-bar-fill"></div></div>'
    '<div style="color:#8B949E;margin:10px 0 0;font-size:0.78rem;">⏱ <b style="color:#E6EDF3;">{elapsed}s</b> 경과</div>'
    '</div>'
)

if _pending and not _aticker:
    _pname = st.session_state.get("_pending_sname", _pending)
    st.caption(f"업데이트: {_now_kst().strftime('%Y-%m-%d %H:%M:%S')}  |  분석 중: **{_pname}** (`{_pending}`)")
    _loading_ph.markdown(
        _LOADING_BASE.format(icon="⏳", title="AI 분석 준비 중",
                             name=_pname, msg="데이터를 불러오는 중입니다. 잠시만 기다려 주세요.", elapsed=0),
        unsafe_allow_html=True,
    )
    st.session_state["analyzed_ticker"]  = st.session_state.pop("_pending_ticker")
    st.session_state["analyzed_sname"]   = st.session_state.pop("_pending_sname", _pending)
    st.session_state["analyzed_period"]  = st.session_state.pop("_pending_period", period)
    st.session_state["_switch_to_chart"] = True
    st.rerun()

if _data_ready:
    ticker = _aticker
    sname  = _asname
    st.caption(f"업데이트: {_now_kst().strftime('%Y-%m-%d %H:%M:%S')}  |  분석 종목: **{_asname}** (`{_aticker}`)")

    _load_start = time.time()
    _loading_ph.markdown(
        _LOADING_BASE.format(icon="📊", title="AI 분석 중",
                             name=_asname, msg="주가·재무 데이터를 분석하고 있습니다.", elapsed=0),
        unsafe_allow_html=True,
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as _pool:
        _f_data = _pool.submit(_stock_data, _aticker, _aperiod)
        _f_fund = _pool.submit(_fundamental, _aticker)
        while not (_f_data.done() and _f_fund.done()):
            _elapsed = int(time.time() - _load_start)
            _loading_ph.markdown(
                _LOADING_BASE.format(icon="📊", title="AI 분석 중",
                                     name=_asname, msg="주가·재무 데이터를 분석하고 있습니다.", elapsed=_elapsed),
                unsafe_allow_html=True,
            )
            time.sleep(1)
        data      = _f_data.result()
        fund_info = _f_fund.result()

    _load_elapsed = int(time.time() - _load_start)
    _loading_ph.markdown(
        _LOADING_BASE.format(icon="🎯", title="AI 매매신호 분석 중",
                             name=_asname, msg="뉴스 감성 · 기술적 지표를 종합하고 있습니다.", elapsed=_load_elapsed),
        unsafe_allow_html=True,
    )

    if data.empty or "Close" not in data.columns:
        st.session_state.pop("analyzed_ticker", None)
        st.error(f"'{_aticker}' 데이터를 불러올 수 없습니다. 티커를 확인해주세요.")
        st.stop()

    _close_raw = data["Close"]
    close = _close_raw.iloc[:, 0] if isinstance(_close_raw, pd.DataFrame) else _close_raw

    def _compute_signals_and_fund():
        try:
            _va  = check_volume_anomaly(data)
            _sig = generate_signals(data)
            if _va.get("is_halted"):
                _sig = {"score": 0, "label": "거래 정지/주의", "badge": "⛔",
                        "reasons": [_va.get("reason", "")], "_halted": True}
            _adv = get_advanced_analysis(data)
            _exp = calculate_expected_return(data, _sig, ticker=_aticker,
                                             benchmark_returns=_bench_returns(_aticker))
            _last = float(close.iloc[-1]) if not close.empty else 0.0
            _fsd  = calculate_fundamental_score(fund_info, _last)
            return _va, _sig, _adv, _exp, _fsd
        except Exception as _e:
            import logging
            logging.getLogger(__name__).warning(f"[compute] {_aticker}: {type(_e).__name__}: {_e}")
            return {}, {"score": 0, "label": "분석 오류", "badge": "⚠️", "reasons": []}, {}, {}, {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as _pool2:
        _f_comp = _pool2.submit(_compute_signals_and_fund)
        _f_news = _pool2.submit(
            _news_sentiment_llm_cached if use_llm else _news_sentiment_kw,
            ticker,
            *(gemini_api_key, groq_api_key) if use_llm else ()
        )
        while not (_f_comp.done() and _f_news.done()):
            _elapsed = int(time.time() - _load_start)
            _loading_ph.markdown(
                _LOADING_BASE.format(icon="🎯", title="AI 매매신호 분석 중",
                                     name=_asname, msg="뉴스 감성 · 기술적 지표를 종합하고 있습니다.", elapsed=_elapsed),
                unsafe_allow_html=True,
            )
            time.sleep(1)
        try:
            vol_anomaly, signals, advanced, expected, fund_score_data = _f_comp.result()
        except Exception:
            vol_anomaly, signals, advanced, expected, fund_score_data = (
                {}, {"score": 0, "label": "분석 오류", "badge": "⚠️", "reasons": []}, {}, {}, {}
            )
        try:
            news_result = _f_news.result()
        except Exception:
            news_result = {}

    news_score = news_result.get("score", 0.0) if isinstance(news_result, dict) else 0.0
    tech_score = signals.get("score", 0)       if isinstance(signals, dict)     else 0
    dead_time  = _dead_time(_aticker)
    breakout   = check_breakout_signal(data)

    # 각 점수를 get_enhanced_hybrid_signal 파라미터 범위로 정규화
    _tech5    = max(-5, min(5, round(tech_score / 2)))            # -10~+10 → -5~+5
    _news1    = max(-1.0, min(1.0, news_score / 5.0))             # -5~+5  → -1.0~+1.0
    _raw_fund = fund_score_data.get("fund_score", 0)              # ±8
    _fund_100 = max(0, min(100, int((_raw_fund + 8) / 16 * 100))) # ±8 → 0~100
    _rsi      = (
        float(data["RSI"].iloc[-1])
        if not data.empty and "RSI" in data.columns and not data["RSI"].isna().iloc[-1]
        else 0.0
    )

    hybrid = get_enhanced_hybrid_signal(
        tech_score  = _tech5,
        news_score  = _news1,
        fund_score  = _fund_100,
        vol_anomaly = vol_anomaly,
        dead_time   = dead_time,
        breakout    = breakout,
        advanced    = advanced,
        period      = _aperiod,
        rsi         = _rsi,
    )

    if expected:
        _exp_ret = expected.get("expected_return_pct", 0.0)
        _sharpe  = expected.get("sharpe", 1.0)
        if _exp_ret >= 50.0 and _sharpe < 0.5:
            hybrid["warnings"].append(
                "⚠️ 샤프지수 주의: 예상 수익률이 높지만 샤프지수가 낮아 리스크 대비 수익이 불안정합니다."
            )
            if hybrid["label"] in ("강력 매수", "매수 추천"):
                hybrid["label"] = "주의"
                hybrid["badge"] = "🟡"

    risk_adj = adjust_risk_conservative(expected) if expected else {}
    _total_elapsed = int(time.time() - _load_start)
    _loading_ph.markdown(
        f'<div style="background:#161B22;border:1px solid #26a69a55;border-radius:12px;'
        f'padding:28px;text-align:center;margin:8px 0 16px;box-shadow:0 4px 24px rgba(38,166,154,0.15);">'
        f'<div style="font-size:36px;margin-bottom:12px;">✅</div>'
        f'<div style="font-size:1.05rem;font-weight:700;color:#E6EDF3;margin-bottom:8px;">분석 완료</div>'
        f'<div style="color:#8B949E;font-size:0.88rem;line-height:1.7;">'
        f'<span style="color:#26a69a;font-weight:600;">{_asname}</span> 데이터 분석이 완료되었습니다.</div>'
        f'<div class="loading-bar-track"><div style="background:#26a69a;width:100%;height:4px;border-radius:4px;"></div></div>'
        f'<div style="color:#8B949E;margin:10px 0 0;font-size:0.78rem;">⏱ 총 소요 시간: <b style="color:#E6EDF3;">{_total_elapsed}s</b></div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    _loading_ph.empty()

    _rt           = _realtime_price_1m(_aticker)
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
    hybrid          = {"hybrid_score": 0.0, "combined_score": 50.0, "label": "중립/관망", "badge": "⚪", "reasons": [], "warnings": []}
    news_result     = {}
    tech_score      = 0
    news_score      = 0.0

# ─── 관심종목 Toast 알림 ─────────────────────────────────────────────────────
if st.session_state.watchlist:
    _alerts      = _get_wl_alerts()
    shown_key    = "toast_shown|" + "|".join(sorted(w["ticker"] for w in st.session_state.watchlist))
    already_shown = st.session_state.get(shown_key, set())
    new_shown    = set(already_shown)
    for a in _alerts:
        uid = f"{a['ticker']}_{a['score']}"
        if uid not in already_shown:
            if a["score"] >= 4:
                st.toast(f"🟢 **{a['name']}** 강력 매수!\n신호: {a['label']} ({a['score']:+.1f}점)  |  {a['price']:,.0f}  {a['chg']:+.2f}%", icon="🔺")
            elif a["score"] <= -4:
                st.toast(f"🔴 **{a['name']}** 강력 매도!\n신호: {a['label']} ({a['score']:+.1f}점)  |  {a['price']:,.0f}  {a['chg']:+.2f}%", icon="🔻")
            elif a["score"] >= 3:
                st.toast(f"🟡 **{a['name']}** 매수 신호\n신호: {a['label']} ({a['score']:+.1f}점)  |  {a['price']:,.0f}", icon="🔔")
            elif a["score"] <= -3:
                st.toast(f"🟠 **{a['name']}** 매도 신호\n신호: {a['label']} ({a['score']:+.1f}점)  |  {a['price']:,.0f}", icon="🔔")
            new_shown.add(uid)
    st.session_state[shown_key] = new_shown

# ═════════════════════════════════════════════════════════════════════════════
# AnalysisState — 탭 렌더 함수에 주입
# ═════════════════════════════════════════════════════════════════════════════
_state = {
    "ticker":        ticker,
    "sname":         sname,
    "asname":        _asname,
    "aperiod":       _aperiod,
    "data_ready":    _data_ready,
    "data":          data,
    "close":         close,
    "signals":       signals,
    "advanced":      advanced,
    "hybrid":        hybrid,
    "expected":      expected,
    "fund_info":     fund_info,
    "fund_score_data": fund_score_data,
    "news_result":   news_result,
    "dead_time":     dead_time,
    "breakout":      breakout,
    "risk_adj":      risk_adj,
    "vol_anomaly":   vol_anomaly,
    "rt_price":      _rt_price,
    "rt_ts":         _rt_ts,
    "rt_realtime":   _rt_realtime,
    "rt_stale":      _rt_stale,
    "rt_stale_msg":  _rt_stale_msg,
}
_api_keys = {
    "gemini": gemini_api_key,
    "groq":   groq_api_key,
    "dart":   dart_api_key,
}

# ═════════════════════════════════════════════════════════════════════════════
# 탭 렌더링
# ═════════════════════════════════════════════════════════════════════════════
render_market_tab(
    tab_market,
    full_movers_fn      = _full_movers,
    movers_fn           = _movers,
    rates               = _rates(),
    usdkrw_fn           = _usdkrw_history,
    sector_etf_prices_fn = _render_sector_etf_prices,
)

render_chart_tab(
    tab_chart,
    state             = _state,
    api_keys          = _api_keys,
    inv_data_fn       = _inv_data,
    inv_history_fn    = _inv_data_history,
    insider_trades_fn = _insider_trades,
    save_watchlist_fn = save_watchlist,
)

render_rec_tab(
    tab_rec,
    get_full_stocks_fn      = _get_full_stocks,
    get_recommendations_fn  = get_recommendations,
    news_sentiment_llm_fn   = _news_sentiment_llm_cached,
    news_sentiment_kw_fn    = _news_sentiment_kw,
    ticker_name_map_fn      = _ticker_name_map,
    db_save_recommendation  = _db_save_recommendation,
    db_get_rec_history      = _db_get_rec_history,
    auth_user_id            = st.session_state.get("auth_user_id"),
)

render_news_tab(
    tab_news,
    state             = _state,
    api_keys          = _api_keys,
    etf_fundamental_fn = _etf_fundamental,
    naver_news_fn     = _naver_news,
    sector_perf_fn    = _sector_perf,
    check_is_etf_fn   = _check_is_etf,
)

render_fund_tab(
    tab_fund,
    state             = _state,
    api_keys          = _api_keys,
    etf_fundamental_fn = _etf_fundamental,
    check_is_etf_fn   = _check_is_etf,
    inv_data_fn       = _inv_data,
    inv_history_fn    = _inv_data_history,
    insider_trades_fn = _insider_trades,
)

render_portfolio_tab(
    tab_portfolio,
    db_login              = _db_login,
    db_logout             = _db_logout,
    db_register           = _db_register,
    db_get_user           = _db_get_user,
    db_get_portfolio      = _db_get_portfolio,
    db_get_trade_history  = _db_get_trade_history,
    db_sell_item          = _db_sell_item,
    db_delete_portfolio   = _db_delete_portfolio,
    db_save_recommendation = _db_save_recommendation,
    db_get_rec_history    = _db_get_rec_history,
    db_clear_trade_history = _db_clear_trade_history,
    ticker_name_map_fn    = _ticker_name_map,
    realtime_price_fn     = _realtime_price_1m,
    get_stock_data_fn     = _stock_data,
    now_kst_fn            = _now_kst,
    get_exchange_rates_fn = _rates,
    set_cookie_fn         = _set_cookie,
    delete_cookie_fn      = _delete_cookie,
)

render_backtest_tab(tab_backtest)
