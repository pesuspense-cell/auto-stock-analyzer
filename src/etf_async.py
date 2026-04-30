"""
etf_async.py - KRX 공공 데이터 API 기반 ETF 지표 비동기 수집

데이터 우선순위:
  1. KRX 공공 데이터 API (data.krx.co.kr) → NAV, 괴리율, 추적오차, AUM, 구성종목
  2. FinanceDataReader → 현재가 보완
  3. 정적 운용보수 맵 → 오프라인 안정 보장 (외부 API 불필요)
  4. SQLite 캐시 (fundamentals.db) → API 장애 시 최근 데이터 서빙

환경 변수:
  KRX_API_KEY : KRX Open API 인증키 (선택)
                설정 시 openapi.krx.co.kr 사용, 미설정 시 data.krx.co.kr 사용 (인증 불필요)
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

logger = logging.getLogger("etf_async")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] etf_async: %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

# ─── KRX 공공 데이터 API 상수 ─────────────────────────────────────────────────
_KRX_URL = "https://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"
_KRX_HEADERS = {
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    "Referer": "https://data.krx.co.kr/",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "X-Requested-With": "XMLHttpRequest",
}

# ─── 운용보수 정적 맵 (출처: 각 운용사 공시 자료 2025년 기준) ─────────────────
_EXPENSE_RATIO_MAP: dict[str, float] = {
    # ── KODEX (삼성자산운용) ────────────────────────────────────────────────────
    "069500": 0.150,  # KODEX 200
    "091160": 0.450,  # KODEX 반도체
    "091170": 0.450,  # KODEX 자동차
    "305720": 0.450,  # KODEX 2차전지산업
    "266370": 0.450,  # KODEX IT
    "122630": 0.070,  # KODEX 레버리지
    "114800": 0.150,  # KODEX 인버스
    "069660": 0.300,  # KODEX 코스닥150레버리지
    "251340": 0.070,  # KODEX 코스닥150
    "278530": 0.070,  # KODEX MSCI Korea TR
    # ── TIGER (미래에셋자산운용) ────────────────────────────────────────────────
    "102110": 0.150,  # TIGER 200
    "091220": 0.150,  # TIGER 은행
    "143860": 0.450,  # TIGER 헬스케어
    "228790": 0.450,  # TIGER 화학
    "232080": 0.100,  # TIGER KOSDAQ150
    "360750": 0.070,  # TIGER 미국S&P500
    "133690": 0.070,  # TIGER 미국나스닥100
    "395160": 0.250,  # TIGER 미국배당다우존스
    # ── ACE (한국투자신탁운용) ──────────────────────────────────────────────────
    "273130": 0.150,  # ACE 코스피200
    "411060": 0.070,  # ACE 미국S&P500
    # ── KBSTAR (KB자산운용) ─────────────────────────────────────────────────────
    "148070": 0.070,  # KBSTAR 200
    "261220": 0.250,  # KBSTAR 미국S&P500
}

# ─── SQLite 캐시 ──────────────────────────────────────────────────────────────
_DB_PATH = Path(__file__).parent.parent / "fundamentals.db"
_CACHE_TTL_MARKET  = 10 * 60    # 장중: 10분
_CACHE_TTL_OFFHOUR = 6 * 3600   # 장외: 6시간


def _init_cache() -> None:
    with sqlite3.connect(_DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS etf_cache (
                code       TEXT PRIMARY KEY,
                data_json  TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn.commit()


def _cache_ttl() -> int:
    t = datetime.now().hour * 60 + datetime.now().minute
    return _CACHE_TTL_MARKET if 9 * 60 <= t <= 15 * 60 + 30 else _CACHE_TTL_OFFHOUR


def _load_cache(code: str) -> dict | None:
    try:
        with sqlite3.connect(_DB_PATH) as conn:
            row = conn.execute(
                "SELECT data_json, updated_at FROM etf_cache WHERE code = ?", (code,)
            ).fetchone()
        if not row:
            return None
        if (datetime.now() - datetime.fromisoformat(row[1])).total_seconds() < _cache_ttl():
            return json.loads(row[0])
        return None
    except Exception:
        return None


def _load_stale_cache(code: str) -> dict | None:
    """만료 무시 — API 장애 시 최신 캐시 서빙"""
    try:
        with sqlite3.connect(_DB_PATH) as conn:
            row = conn.execute(
                "SELECT data_json FROM etf_cache WHERE code = ?", (code,)
            ).fetchone()
        return json.loads(row[0]) if row else None
    except Exception:
        return None


def _save_cache(code: str, data: dict) -> None:
    try:
        with sqlite3.connect(_DB_PATH) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO etf_cache (code, data_json, updated_at) VALUES (?,?,?)",
                (code, json.dumps(data, ensure_ascii=False, default=str), datetime.now().isoformat()),
            )
            conn.commit()
    except Exception:
        pass


# ─── 파싱 헬퍼 ───────────────────────────────────────────────────────────────
def _to_float(val: Any, default: float | None = None) -> float | None:
    if val is None:
        return default
    try:
        return float(str(val).replace(",", "").strip())
    except (ValueError, TypeError):
        return default


def _pick(row: dict, *keys: str) -> Any:
    """여러 가능한 키 이름 중 의미 있는 첫 번째 값 반환"""
    for k in keys:
        v = row.get(k)
        if v is not None and str(v).strip() not in ("", "-", "N/A", "0"):
            return v
    return None


def _to_aum(raw: Any) -> float | None:
    v = _to_float(raw)
    return round(v / 1e8, 0) if v and v > 0 else None


# ─── KRX 공공 API 비동기 수집 ─────────────────────────────────────────────────
async def _krx_fetch_nav(
    code: str, client: "httpx.AsyncClient", date_str: str
) -> dict | None:
    """
    KRX data.krx.co.kr → ETF NAV / 괴리율 / 추적오차 / AUM / 현재가.
    최근 6거래일 중 데이터가 있는 가장 최근 날짜 사용.
    """
    for delta in range(6):
        d = (datetime.strptime(date_str, "%Y%m%d") - timedelta(days=delta)).strftime("%Y%m%d")
        try:
            resp = await client.post(
                _KRX_URL,
                data={
                    "bld":         "dbms/MDC/STAT/standard/MDCSTAT04301",
                    "locale":      "ko_KR",
                    "isuCd":       code,
                    "isuCd2":      code,
                    "strtDd":      d,
                    "endDd":       d,
                    "csvxls_isNo": "false",
                },
                headers=_KRX_HEADERS,
                timeout=12.0,
            )
            resp.raise_for_status()
            rows = resp.json().get("output", [])
            if not rows:
                continue
            row = rows[0]

            nav_raw = _to_float(_pick(row, "NAV", "ETF_NAV", "nav"))
            if not nav_raw:
                continue

            return {
                "nav":            nav_raw,
                "nav_premium":    _to_float(_pick(row, "괴리율", "DSRT_RT", "ETF_DSRT_RT")),
                "tracking_error": _to_float(_pick(row, "추적오차율", "TCKR_ERSS_RT", "ETF_TCKR_ERSS_RT")),
                "aum":            _to_aum(_pick(row, "순자산총액", "NETASST_TOTAMT", "ETF_NASS_TOT_AMT")),
                "price":          _to_float(_pick(row, "기준가격", "TDD_CLSPRC", "기준가", "CLSPRC", "ETF_CLSPRC")),
                "trade_date":     d,
            }
        except Exception as e:
            logger.debug("KRX NAV fetch error (delta=%d): %s", delta, e)
    return None


async def _krx_fetch_holdings(
    code: str, client: "httpx.AsyncClient", date_str: str
) -> list[dict]:
    """KRX PDF 구성종목 상위 10개 수집"""
    for delta in range(6):
        d = (datetime.strptime(date_str, "%Y%m%d") - timedelta(days=delta)).strftime("%Y%m%d")
        try:
            resp = await client.post(
                _KRX_URL,
                data={
                    "bld":         "dbms/MDC/STAT/standard/MDCSTAT04601",
                    "locale":      "ko_KR",
                    "isuCd":       code,
                    "trdDd":       d,
                    "csvxls_isNo": "false",
                },
                headers=_KRX_HEADERS,
                timeout=12.0,
            )
            resp.raise_for_status()
            rows = resp.json().get("output", [])
            if not rows:
                continue

            holdings = []
            for r in rows[:10]:
                t_code = str(_pick(r, "ISU_CD", "종목코드") or "").strip()
                t_name = str(_pick(r, "ISU_ABBRV", "종목명", "ISU_NM") or "").strip()
                t_wgt  = _to_float(_pick(r, "COMPST_RT", "비중", "WGT_RT", "WGFT"))
                if t_code and t_name:
                    holdings.append({
                        "ticker": f"{t_code.zfill(6)}.KS",
                        "name":   t_name,
                        "weight": t_wgt,
                    })
            if holdings:
                return holdings
        except Exception as e:
            logger.debug("KRX holdings fetch error (delta=%d): %s", delta, e)
    return []


async def _fdr_fetch_price(ticker: str) -> float | None:
    """FinanceDataReader로 현재가 보완 (동기 라이브러리 → executor 오프로드)"""
    try:
        import FinanceDataReader as fdr  # type: ignore
        code = ticker.replace(".KS", "").replace(".KQ", "")
        loop = asyncio.get_event_loop()
        today = datetime.now().strftime("%Y-%m-%d")
        df = await loop.run_in_executor(None, lambda: fdr.DataReader(code, today))
        if df is not None and not df.empty and "Close" in df.columns:
            return float(df["Close"].iloc[-1])
    except Exception:
        pass
    return None


# ─── 통합 비동기 수집 ─────────────────────────────────────────────────────────
async def _fetch_etf_all(ticker: str, portfolio_info: dict) -> dict:
    code     = ticker.replace(".KS", "").replace(".KQ", "").strip().zfill(6)
    date_str = datetime.now().strftime("%Y%m%d")

    result: dict = {
        "code":           code,
        "ticker":         ticker,
        "expense_ratio":  _EXPENSE_RATIO_MAP.get(code),
        "tracking_error": None,
        "nav":            None,
        "price":          None,
        "nav_premium":    None,
        "dividend_yield": None,
        "aum":            None,
        "top_holdings":   [],
        "sector":         portfolio_info.get("sector", ""),
        "etf_name":       portfolio_info.get("name", ""),
        "source":         "krx_api",
        "cache_used":     False,
        "data_status":    "ok",
    }

    if not HAS_HTTPX:
        result["data_status"] = "httpx 미설치 (pip install httpx)"
        return result

    async with httpx.AsyncClient(follow_redirects=True) as client:
        nav_task, holdings_task = (
            asyncio.create_task(_krx_fetch_nav(code, client, date_str)),
            asyncio.create_task(_krx_fetch_holdings(code, client, date_str)),
        )
        nav_data, holdings = await asyncio.gather(
            nav_task, holdings_task, return_exceptions=True
        )

    # NAV / 괴리율 / 추적오차 / AUM / 기준가격
    if isinstance(nav_data, dict) and nav_data:
        result.update({
            "nav":            nav_data.get("nav"),
            "nav_premium":    nav_data.get("nav_premium"),
            "tracking_error": nav_data.get("tracking_error"),
            "aum":            nav_data.get("aum"),
        })
        if nav_data.get("price"):
            result["price"] = nav_data["price"]
    else:
        result["data_status"] = "krx_api_일시_장애"
        logger.warning("KRX NAV fetch returned no data for %s", code)

    # 구성종목
    if isinstance(holdings, list) and holdings:
        result["top_holdings"] = holdings
    elif not isinstance(holdings, list):
        logger.debug("Holdings task raised: %s", holdings)

    # 현재가 보완 (KRX API 미수집 시)
    if result["price"] is None:
        fdr_price = await _fdr_fetch_price(ticker)
        if fdr_price:
            result["price"] = fdr_price
            result["source"] = "krx_api+fdr"

    # NAV 괴리율 직접 계산 (API 미수집 시)
    if result["nav_premium"] is None and result["price"] and result["nav"]:
        result["nav_premium"] = round(
            (result["price"] - result["nav"]) / result["nav"] * 100, 2
        )

    # 포트폴리오 맵 폴백 (구성종목 미수집 시)
    if not result["top_holdings"] and portfolio_info.get("holdings"):
        result["top_holdings"] = [
            {"ticker": t, "name": t.split(".")[0], "weight": None}
            for t in portfolio_info["holdings"]
        ]

    return result


# ─── Streamlit 동기 래퍼 ──────────────────────────────────────────────────────
def _run_async(coro) -> Any:
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def get_etf_fundamental_data(ticker: str, portfolio_info: dict | None = None) -> dict:
    """
    ETF 핵심 지표 수집 — KRX 공공 API 비동기 기반.
    Streamlit 동기 환경에서 안전하게 호출 가능.

    캐시 전략:
      장중(09:00-15:30 KST): 10분 TTL
      장외: 6시간 TTL
      API 완전 장애 시: 마지막 캐시 + data_status = "데이터 업데이트 중 (캐시 사용)"
    """
    _init_cache()
    code = ticker.replace(".KS", "").replace(".KQ", "").strip().zfill(6)
    if portfolio_info is None:
        portfolio_info = {}

    cached = _load_cache(code)
    if cached:
        return cached

    try:
        data = _run_async(_fetch_etf_all(ticker, portfolio_info))
    except Exception as e:
        logger.error("ETF fetch failed for %s: %s", ticker, e)
        stale = _load_stale_cache(code)
        if stale:
            stale["cache_used"]  = True
            stale["data_status"] = "데이터 업데이트 중 (캐시 사용)"
            return stale
        return {
            "code": code, "ticker": ticker,
            "source": "error", "cache_used": True,
            "data_status": "데이터를 불러올 수 없습니다",
            "expense_ratio":  _EXPENSE_RATIO_MAP.get(code),
            "tracking_error": None, "nav": None, "price": None,
            "nav_premium": None, "dividend_yield": None, "aum": None,
            "top_holdings": [],
            "sector":   portfolio_info.get("sector", ""),
            "etf_name": portfolio_info.get("name", ""),
        }

    _save_cache(code, data)
    return data
