"""
etf_async.py - KRX 공공 데이터 API 기반 ETF 지표 비동기 수집

데이터 우선순위:
  1. KRX data.krx.co.kr (로그인 세션) → NAV, 괴리율, 추적오차, AUM, 구성종목
  2. 네이버 금융 스크래핑 → NAV, 괴리율, AUM, 현재가 (KRX 인증 실패 시 폴백)
  3. FinanceDataReader → 현재가 보완
  4. 정적 운용보수 맵 → expense_ratio 오프라인 보장
  5. SQLite 캐시 (fundamentals.db) → 전체 실패 시 최근 데이터 서빙

인증:
  KRX_ID / KRX_PW 환경변수 또는 Streamlit secrets 설정 시 KRX 로그인 자동 수행.
  미설정 시 네이버 금융 폴백 사용.
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
    "Content-Type":     "application/x-www-form-urlencoded; charset=UTF-8",
    "Origin":           "https://data.krx.co.kr",
    "Referer":          "https://data.krx.co.kr/contents/MDC/MDI/mdistats/MDCSTAT04301.cmd",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept":           "application/json, text/javascript, */*; q=0.01",
    "Accept-Language":  "ko-KR,ko;q=0.9,en-US;q=0.8",
    "X-Requested-With": "XMLHttpRequest",
}


def _extract_rows(body: dict) -> list[dict]:
    """KRX API 응답 바디에서 row 리스트 추출 (엔드포인트마다 키가 다름)."""
    for key in ("OutBlock_1", "output", "block1", "etfList", "Output1", "data"):
        val = body.get(key)
        if isinstance(val, list) and val:
            return val
    return []


# ─── KRX 인증 세션 캐시 ───────────────────────────────────────────────────────
_KRX_AUTH: dict[str, Any] = {"cookie": "", "expires": 0.0}


def _get_krx_cookie_str() -> str:
    """
    KRX 인증 쿠키 문자열 반환 (50분 캐시).
    KRX_ID / KRX_PW를 환경변수 → Streamlit secrets 순서로 탐색.
    로그인 실패 또는 미설정 시 빈 문자열 반환 (Naver 폴백 사용).
    """
    if _KRX_AUTH["cookie"] and time.time() < _KRX_AUTH["expires"]:
        return _KRX_AUTH["cookie"]

    krx_id = os.getenv("KRX_ID", "")
    krx_pw = os.getenv("KRX_PW", "")

    if not (krx_id and krx_pw):
        try:
            import streamlit as st
            krx_id = st.secrets.get("KRX_ID", "")
            krx_pw = st.secrets.get("KRX_PW", "")
        except Exception:
            pass

    if not (krx_id and krx_pw):
        return ""

    try:
        from pykrx.website.comm.auth import build_krx_session
        sess = build_krx_session(krx_id, krx_pw)
        if sess and sess.is_authenticated:
            cookie_str = "; ".join(
                f"{k}={v['value']}" for k, v in sess.cookies.items()
            )
            _KRX_AUTH["cookie"]   = cookie_str
            _KRX_AUTH["expires"]  = time.time() + 3000  # 50분
            logger.info("KRX 로그인 성공 (세션 50분 캐시)")
            return cookie_str
    except Exception as e:
        logger.warning("KRX 로그인 실패: %s", e)
    return ""

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
    code: str, client: "httpx.AsyncClient", date_str: str,
    cookie_str: str = "",
) -> dict | None:
    """
    KRX data.krx.co.kr → ETF NAV / 괴리율 / 추적오차 / AUM / 현재가.
    MDCSTAT04301은 전체 ETF 일괄 조회 엔드포인트이므로 trdDd 파라미터로 호출 후
    code(ISU_SRT_CD)로 필터링. _isu_cd(전체 ISU 코드)도 함께 반환 (구성종목 조회용).
    """
    headers = {**_KRX_HEADERS, **({"Cookie": cookie_str} if cookie_str else {})}
    for delta in range(6):
        d = (datetime.strptime(date_str, "%Y%m%d") - timedelta(days=delta)).strftime("%Y%m%d")
        try:
            resp = await client.post(
                _KRX_URL,
                data={
                    "bld":         "dbms/MDC/STAT/standard/MDCSTAT04301",
                    "locale":      "ko_KR",
                    "trdDd":       d,
                    "share":       "1",
                    "money":       "1",
                    "csvxls_isNo": "false",
                },
                headers=headers,
                timeout=20.0,
            )
            resp.raise_for_status()
            rows = _extract_rows(resp.json())
            if not rows:
                continue

            # 전체 ETF 목록에서 해당 코드 필터링
            row = None
            for r in rows:
                short = str(r.get("ISU_SRT_CD", r.get("isuSrtCd", ""))).strip().zfill(6)
                if short == code:
                    row = r
                    break
            if row is None:
                continue

            nav_raw = _to_float(_pick(row, "NAV", "ETF_NAV", "NAV_PRC", "navPrc", "nav"))
            if not nav_raw:
                continue

            # AUM: INVSTASST_NETASST_TOTAMT 또는 MKTCAP (원 단위 → 억원)
            aum_raw = _to_float(_pick(row, "INVSTASST_NETASST_TOTAMT", "NETASST_TOTAMT",
                                      "netasstTotamt", "순자산총액", "NET_ASST"))
            if not aum_raw or aum_raw == 0:
                aum_raw = _to_float(row.get("MKTCAP"))
            aum = round(aum_raw / 1e8, 0) if aum_raw and aum_raw > 0 else None

            return {
                "nav":            nav_raw,
                "nav_premium":    _to_float(_pick(
                    row, "DVDNDRTO", "dvdndrto", "NAV_DISRATE", "navDisrate",
                    "DISRATE", "disrate", "괴리율", "DSRT_RT", "ETF_DSRT_RT",
                )),
                "tracking_error": _to_float(_pick(
                    row, "TRKNG_ERR_RT", "trkngErrRt", "TRACE_ERR_RT", "traceErrRt",
                    "추적오차율", "TCKR_ERSS_RT", "ETF_TCKR_ERSS_RT",
                )),
                "aum":            aum,
                "price":          _to_float(_pick(
                    row, "TDD_CLSPRC", "기준가격", "기준가", "CLSPRC", "ETF_CLSPRC",
                )),
                "_isu_cd":        str(row.get("ISU_CD", row.get("isuCd", ""))).strip(),
                "trade_date":     d,
            }
        except Exception as e:
            logger.debug("KRX NAV fetch error (delta=%d): %s", delta, e)
    return None


async def _krx_fetch_holdings(
    isu_cd: str, client: "httpx.AsyncClient", date_str: str,
    cookie_str: str = "",
) -> list[dict]:
    """KRX PDF(구성종목) 상위 10개 수집. isu_cd는 전체 ISU 코드(예: KR7069500007)."""
    if not isu_cd:
        return []
    headers = {**_KRX_HEADERS, **({"Cookie": cookie_str} if cookie_str else {})}
    for delta in range(6):
        d = (datetime.strptime(date_str, "%Y%m%d") - timedelta(days=delta)).strftime("%Y%m%d")
        try:
            resp = await client.post(
                _KRX_URL,
                data={
                    "bld":         "dbms/MDC/STAT/standard/MDCSTAT04302",
                    "locale":      "ko_KR",
                    "isuCd":       isu_cd,
                    "trdDd":       d,
                    "csvxls_isNo": "false",
                },
                headers=headers,
                timeout=15.0,
            )
            resp.raise_for_status()
            rows = _extract_rows(resp.json())
            if not rows:
                continue

            holdings = []
            for r in rows[:10]:
                t_code = str(_pick(r, "ISU_SRT_CD", "isuSrtCd", "ISU_CD", "종목코드") or "").strip()
                t_name = str(_pick(r, "ISU_ABBRV", "isuAbbrv", "종목명", "ISU_NM") or "").strip()
                t_wgt  = _to_float(_pick(r, "COMPST_RT", "compstRt", "비중", "WGHT", "wght"))
                if t_code and t_name:
                    mkt = str(r.get("MKT_ID", r.get("mktId", ""))).upper()
                    suffix = ".KQ" if mkt in ("KOSDAQ", "KQ") else ".KS"
                    holdings.append({
                        "ticker": f"{t_code.zfill(6)}{suffix}",
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


async def _naver_fetch_etf(code: str, client: "httpx.AsyncClient") -> dict:
    """
    Naver Finance ETF 종목 페이지 스크래핑.
    KRX 포털이 인증을 요구할 때의 폴백 소스.
    - 현재가: .no_today .blind
    - NAV·괴리율: 날짜/가격/NAV/괴리율 테이블 최신 행
    - AUM: 순자산총액 행 (백만원 단위 → 억원 환산)
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return {}

    url = f"https://finance.naver.com/item/main.naver?code={code}"
    try:
        resp = await client.get(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Referer":    "https://finance.naver.com/",
                "Accept-Language": "ko-KR,ko;q=0.9",
            },
            timeout=12.0,
        )
        # httpx가 Content-Type charset(euc-kr)을 자동 감지해 Unicode str로 반환
        soup = BeautifulSoup(resp.text, "lxml")
    except Exception:
        return {}

    out: dict = {}

    # ── 현재가 ──────────────────────────────────────────────────────────────
    for sel in (".no_today .blind", "#_nowVal", ".today .num"):
        tag = soup.select_one(sel)
        if tag:
            try:
                out["price"] = float(tag.get_text(strip=True).replace(",", ""))
                break
            except ValueError:
                pass

    tables = soup.find_all("table")

    # ── NAV·괴리율 테이블 (header: 날짜/가격/NAV/괴리율) ───────────────────
    for tbl in tables:
        ths = [th.get_text(strip=True) for th in tbl.find_all("th")]
        if "NAV" in ths and any(k in ths for k in ("괴리율", "愿대━���")):
            for tr in tbl.find_all("tr"):
                tds = [td.get_text(strip=True).replace(",", "").replace("%", "") for td in tr.find_all("td")]
                if len(tds) >= 3:
                    try:
                        out["nav"]         = float(tds[2])
                        out["nav_premium"] = float(tds[3]) if len(tds) > 3 else None
                    except (ValueError, IndexError):
                        pass
                    break
            break

    # ── AUM (시가총액 또는 순자산 기반 추정) ────────────────────────────────
    import re as _re
    shares: float | None = None
    for tbl in tables:
        for tr in tbl.find_all("tr"):
            tds = tr.find_all(["th", "td"])
            if len(tds) >= 2:
                label = tds[0].get_text(strip=True)
                val_txt = tds[1].get_text(strip=True)
                # 시가총액: "26조\n...4,023억원" 형태
                if "시가총액" in label or "순자산" in label:
                    # 조 + 억 파싱: "26조4,023억원"
                    trillion = _re.search(r"([\d,]+)조", val_txt)
                    billion  = _re.search(r"([\d,]+)억", val_txt)
                    try:
                        t_val = float(trillion.group(1).replace(",", "")) * 10000 if trillion else 0.0
                        b_val = float(billion.group(1).replace(",", ""))           if billion  else 0.0
                        if t_val + b_val > 0:
                            out["aum"] = round(t_val + b_val, 0)
                    except (ValueError, AttributeError):
                        pass
                # 상장주식수: 숫자만 있는 경우
                elif "상장주식수" in label or "좌수" in label:
                    try:
                        shares = float(val_txt.replace(",", ""))
                    except ValueError:
                        pass

    # 시가총액 파싱 실패 시 주식수 × 현재가로 추정
    if out.get("aum") is None and shares and out.get("price"):
        out["aum"] = round(shares * out["price"] / 1e8, 0)

    return out


def _pykrx_extras(code: str, date_str: str) -> dict:
    """
    pykrx로 추적오차·괴리율·구성종목 추가 수집 (동기, thread executor용).
    KRX 인증 성공 시에만 유효 데이터를 반환.
    """
    out: dict = {"tracking_error": None, "nav_premium_krx": None, "top_holdings": []}
    try:
        import pykrx.stock as _stock  # type: ignore
        end_d   = date_str
        start_d = (datetime.strptime(date_str, "%Y%m%d") - timedelta(days=10)).strftime("%Y%m%d")

        # 추적오차
        try:
            te_df = _stock.get_etf_tracking_error(start_d, end_d, code)
            if te_df is not None and not te_df.empty and "추적오차율" in te_df.columns:
                # 오늘 값이 0이면 전일 값 사용
                for row in reversed(te_df["추적오차율"].tolist()):
                    if row and row != 0:
                        out["tracking_error"] = round(float(row), 2)
                        break
        except Exception as e:
            logger.debug("pykrx tracking_error failed: %s", e)

        # 괴리율 (KRX MDCSTAT04301은 오늘 값이 intraday라 부정확할 수 있음)
        try:
            dev_df = _stock.get_etf_price_deviation(start_d, end_d, code)
            if dev_df is not None and not dev_df.empty and "괴리율" in dev_df.columns:
                vals = dev_df["괴리율"].dropna().tolist()
                if vals:
                    out["nav_premium_krx"] = round(float(vals[-1]), 2)
        except Exception as e:
            logger.debug("pykrx price_deviation failed: %s", e)

        # 구성종목 PDF (당일 → 전일 → ... 순으로 시도)
        for delta in range(5):
            d = (datetime.strptime(date_str, "%Y%m%d") - timedelta(days=delta)).strftime("%Y%m%d")
            try:
                pdf = _stock.get_etf_portfolio_deposit_file(d, code)
                if pdf is not None and not pdf.empty:
                    holdings = []
                    for _, row in pdf.head(10).iterrows():
                        t_code = str(row.get("종목코드", row.name) or "").strip().zfill(6)
                        t_name = str(row.get("종목명", row.get("ISU_ABBRV", "")) or "").strip()
                        t_wgt  = None
                        for wk in ("비중", "구성비중", "COMPST_RT"):
                            if wk in row.index and row[wk] not in (None, ""):
                                try:
                                    t_wgt = float(row[wk])
                                    break
                                except (ValueError, TypeError):
                                    pass
                        if t_code and t_name:
                            holdings.append({"ticker": f"{t_code}.KS", "name": t_name, "weight": t_wgt})
                    if holdings:
                        out["top_holdings"] = holdings
                        break
            except Exception as e:
                logger.debug("pykrx portfolio_deposit_file (delta=%d) failed: %s", delta, e)
    except Exception as e:
        logger.debug("pykrx_extras overall failed: %s", e)
    return out


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

    # KRX 인증 쿠키 (동기 호출 — 캐시 히트 시 즉시 반환, 미스 시 로그인 수행)
    _cookie = _get_krx_cookie_str()

    async with httpx.AsyncClient(follow_redirects=True, verify=False) as client:
        # 1) KRX NAV 먼저 (ISU_CD 획득 후 구성종목 조회)
        nav_data = await _krx_fetch_nav(code, client, date_str, cookie_str=_cookie)

        isu_cd     = ""
        trade_date = date_str
        if isinstance(nav_data, dict) and nav_data:
            isu_cd     = nav_data.pop("_isu_cd", "")
            trade_date = nav_data.get("trade_date", date_str)

        # 2) Naver·Holdings(KRX)·FDR·pykrx extras 병렬 수집
        loop = asyncio.get_event_loop()
        naver_task    = asyncio.create_task(_naver_fetch_etf(code, client))
        holdings_task = asyncio.create_task(_krx_fetch_holdings(isu_cd, client, trade_date, cookie_str=_cookie))
        fdr_task      = asyncio.create_task(_fdr_fetch_price(ticker))
        # pykrx extras (추적오차·구성종목) — KRX 인증 시에만 유효, 동기라 executor 사용
        if _cookie:
            pykrx_task = loop.run_in_executor(None, _pykrx_extras, code, date_str)
        else:
            async def _no_extras():
                return {}
            pykrx_task = asyncio.create_task(_no_extras())
        naver_raw, holdings, fdr_price_result, pykrx_raw = await asyncio.gather(
            naver_task, holdings_task, fdr_task, pykrx_task, return_exceptions=True
        )

    naver_data: dict  = naver_raw  if isinstance(naver_raw,  dict) else {}
    pykrx_data: dict  = pykrx_raw  if isinstance(pykrx_raw,  dict) else {}

    # ── NAV / 괴리율 / 추적오차 / AUM / 기준가격 ─────────────────────────────
    if isinstance(nav_data, dict) and nav_data:
        # KRX MDCSTAT04301 성공
        result.update({
            "nav":            nav_data.get("nav"),
            "nav_premium":    nav_data.get("nav_premium"),
            "tracking_error": nav_data.get("tracking_error"),
            "aum":            nav_data.get("aum"),
        })
        if nav_data.get("price"):
            result["price"] = nav_data["price"]
    else:
        # KRX 실패 → Naver Finance 폴백
        logger.info("KRX NAV 미수집 (%s) → 네이버 금융 폴백", code)
        if naver_data.get("nav"):
            result.update({
                "nav":         naver_data.get("nav"),
                "nav_premium": naver_data.get("nav_premium"),
                "aum":         naver_data.get("aum"),
            })
            result["source"]      = "naver"
            result["data_status"] = "ok"
        else:
            hint = "KRX_ID/KRX_PW 미설정" if not _cookie else "KRX API 일시 장애"
            result["data_status"] = f"데이터 수집 실패 ({hint})"
            logger.warning("ETF 데이터 수집 실패 [%s]: %s", code, hint)

    # ── pykrx extras 보완 (추적오차·괴리율 KRX 정식 집계값) ─────────────────
    if pykrx_data.get("tracking_error") is not None:
        result["tracking_error"] = pykrx_data["tracking_error"]
    # pykrx 괴리율은 종가 기준이라 더 정확 — nav_premium이 없거나 오늘 intraday라면 교체
    if pykrx_data.get("nav_premium_krx") is not None and result["nav_premium"] is None:
        result["nav_premium"] = pykrx_data["nav_premium_krx"]

    # ── 구성종목 (httpx KRX → pykrx 순) ─────────────────────────────────────
    if isinstance(holdings, list) and holdings:
        result["top_holdings"] = holdings
    elif pykrx_data.get("top_holdings"):
        result["top_holdings"] = pykrx_data["top_holdings"]
    elif not isinstance(holdings, list):
        logger.debug("Holdings task raised: %s", holdings)

    _fdr_price = fdr_price_result if isinstance(fdr_price_result, float) else None

    # ── 현재가 보완 (KRX 미수집 시 Naver → FDR 순) ─────────────────────────
    if result["price"] is None:
        if naver_data.get("price"):
            result["price"] = naver_data["price"]
        elif _fdr_price:
            result["price"] = _fdr_price
            if result["source"] == "krx_api":
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
