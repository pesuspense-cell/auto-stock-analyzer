"""
etf_krx.py — ETF 데이터 수집 (pykrx 대체)

우선순위:
  1) KRX 데이터포털 직접 호출 (httpx, 브라우저 헤더)  ← NAV·괴리율·추적오차·순자산
  2) Naver Finance 페이지 스크래핑                    ← 운용보수·배당·현재가·NAV 보완
  3) 정적 운용보수 맵                                 ← 주요 ETF 운용보수 폴백
  4) JSON 파일 캐시                                   ← API 실패 시 최근 값 반환

환경변수:
  KRX_API_KEY — 현재 data.krx.co.kr 직접 호출에는 불필요.
                추후 공공데이터포털 공식 API 전환 시 사용.
"""
from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import httpx

try:
    from bs4 import BeautifulSoup
    _HAS_BS4 = True
except ImportError:
    _HAS_BS4 = False

# ── 캐시 경로 ────────────────────────────────────────────────────────────────

_CACHE_DIR  = Path("data")
_CACHE_FILE = _CACHE_DIR / "etf_fundamental_cache.json"
_CACHE_TTL  = 86_400  # 24시간 (초)

# ── KRX 데이터포털 공통 설정 ──────────────────────────────────────────────────

_KRX_BASE = "https://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"

# 브라우저를 흉내내는 헤더 (pykrx 차단 우회 핵심)
_KRX_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer":          "https://data.krx.co.kr/contents/MDC/MDI/mdistats/MDCSTAT04301.cmd",
    "Origin":           "https://data.krx.co.kr",
    "Content-Type":     "application/x-www-form-urlencoded; charset=UTF-8",
    "Accept":           "application/json, text/javascript, */*; q=0.01",
    "Accept-Language":  "ko-KR,ko;q=0.9,en-US;q=0.8",
    "X-Requested-With": "XMLHttpRequest",
}

# ── 주요 ETF 운용보수 정적 맵 ────────────────────────────────────────────────
# 출처: 각 운용사 공시 (연 1회 이하 변경)
_EXPENSE_RATIO_MAP: dict[str, float] = {
    # ── KODEX (삼성자산운용) ──────────────────────────────────────────────────
    "069500": 0.150,  # KODEX 200
    "091160": 0.450,  # KODEX 반도체
    "091170": 0.450,  # KODEX 자동차
    "305720": 0.450,  # KODEX 2차전지산업
    "266370": 0.450,  # KODEX IT
    "114800": 0.640,  # KODEX 인버스
    "122630": 0.640,  # KODEX 레버리지
    "252670": 0.640,  # KODEX 200선물인버스2X
    "278530": 0.050,  # KODEX MSCI Korea TR
    "278540": 0.050,  # KODEX 200 TR
    "379800": 0.010,  # KODEX 미국S&P500(H)
    "219480": 0.070,  # KODEX 미국S&P500TR
    "367380": 0.050,  # KODEX 미국나스닥100TR
    "143460": 0.070,  # KODEX 미국나스닥100(H)
    "292150": 0.450,  # KODEX 글로벌배터리산업
    "385510": 0.450,  # KODEX K-메타버스액티브
    "381180": 0.450,  # KODEX 미국반도체MV
    "411060": 0.010,  # KODEX 미국S&P500(H) 환헤지
    # ── TIGER (미래에셋자산운용) ──────────────────────────────────────────────
    "102110": 0.090,  # TIGER 200
    "091220": 0.400,  # TIGER 은행
    "143860": 0.400,  # TIGER 헬스케어
    "228790": 0.400,  # TIGER 화학
    "232080": 0.400,  # TIGER KOSDAQ150
    "360750": 0.070,  # TIGER 미국S&P500
    "133690": 0.070,  # TIGER 미국나스닥100
    "395160": 0.080,  # TIGER 미국배당다우존스
    "381170": 0.070,  # TIGER 미국테크TOP10 INDXX
    "192090": 0.350,  # TIGER 차이나CSI300
    "239660": 0.490,  # TIGER 200 IT
    "266160": 0.490,  # TIGER 200 에너지화학
    "244580": 0.490,  # TIGER 200 생활소비재
    "453850": 0.090,  # TIGER 미국배당+3%프리미엄다우존스
    # ── KINDEX (한국투자신탁운용) ─────────────────────────────────────────────
    "152100": 0.170,  # KINDEX 200
    "130680": 0.400,  # KINDEX 반도체
    "310970": 0.090,  # KINDEX 미국S&P500
    # ── ACE (한국투신운용) ───────────────────────────────────────────────────
    "273130": 0.070,  # ACE 미국S&P500
    "426410": 0.050,  # ACE 미국나스닥100
    "441680": 0.100,  # ACE 미국배당다우존스
    # ── SOL (신한자산운용) ───────────────────────────────────────────────────
    "444480": 0.090,  # SOL 미국S&P500
    "448290": 0.090,  # SOL 미국배당다우존스
    # ── HANARO (NH아문디자산운용) ─────────────────────────────────────────────
    "292560": 0.070,  # HANARO 미국S&P500
    "315960": 0.400,  # HANARO 200
}


# ═══════════════════════════════════════════════ 캐시 헬퍼 ═══════════════════

def _load_cache() -> dict:
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if _CACHE_FILE.exists():
            return json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_cache(cache: dict) -> None:
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _CACHE_FILE.write_text(
            json.dumps(cache, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


def _get_cache(code: str) -> Optional[dict]:
    entry = _load_cache().get(code)
    if entry and (time.time() - entry.get("_ts", 0)) < _CACHE_TTL:
        return entry
    return None


def _set_cache(code: str, data: dict) -> None:
    cache = _load_cache()
    data["_ts"] = time.time()
    cache[code] = data
    _save_cache(cache)


# ═══════════════════════════════════════════════ KRX 데이터포털 ═══════════════

def _extract_rows(body: dict) -> list[dict]:
    """KRX API 응답에서 데이터 rows 추출 (endpoint마다 키가 다름)."""
    for key in ("OutBlock_1", "output", "block1", "etfList", "Output1", "data"):
        val = body.get(key)
        if isinstance(val, list) and val:
            return val
    return []


async def _krx_etf_all_fundamentals(date_str: str, client: httpx.AsyncClient) -> list[dict]:
    """
    KRX 전체 ETF 지표 일괄 조회.
    bld=MDCSTAT04301 → 전 종목 NAV·괴리율·추적오차·순자산 반환.
    """
    payload = {
        "bld":          "dbms/MDC/STAT/standard/MDCSTAT04301",
        "locale":       "ko_KR",
        "trdDd":        date_str,
        "share":        "1",
        "money":        "1",
        "csvxls_isNo":  "false",
    }
    resp = await client.post(_KRX_BASE, data=payload, headers=_KRX_HEADERS, timeout=20.0)
    resp.raise_for_status()
    return _extract_rows(resp.json())


async def _krx_etf_holdings(isu_cd: str, date_str: str, client: httpx.AsyncClient) -> list[dict]:
    """
    KRX ETF PDF (구성종목) 조회.
    bld=MDCSTAT04302 → 해당 ETF 구성종목 비중 반환.
    isu_cd: 전체 ISU 코드 (MDCSTAT04301 응답의 ISU_CD 필드)
    """
    payload = {
        "bld":          "dbms/MDC/STAT/standard/MDCSTAT04302",
        "locale":       "ko_KR",
        "trdDd":        date_str,
        "isuCd":        isu_cd,
        "csvxls_isNo":  "false",
    }
    resp = await client.post(_KRX_BASE, data=payload, headers=_KRX_HEADERS, timeout=15.0)
    resp.raise_for_status()
    return _extract_rows(resp.json())


def _parse_krx_fundamental(row: dict) -> dict:
    """
    KRX 응답 row에서 ETF 핵심 지표 추출.
    필드명이 버전마다 다를 수 있으므로 후보 키 목록을 순서대로 시도.
    """
    def fval(*keys: str) -> Optional[float]:
        for k in keys:
            v = row.get(k)
            if v not in (None, "", "-", "N/A"):
                try:
                    return float(str(v).replace(",", ""))
                except (ValueError, TypeError):
                    pass
        return None

    nav     = fval("NAV", "nav", "NAV_PRC", "navPrc", "CLSPRC")
    premium = fval(
        "DVDNDRTO", "dvdndrto",
        "NAV_DISRATE", "navDisrate",
        "DISRATE", "disrate",
        "괴리율",
    )
    te = fval(
        "TRKNG_ERR_RT", "trkngErrRt",
        "TRACE_ERR_RT", "traceErrRt",
        "추적오차율",
    )
    aum_raw = fval(
        "NETASST_TOTAMT", "netasstTotamt",
        "순자산총액", "NET_ASST", "netAsst",
    )
    isu_cd = row.get("ISU_CD", row.get("isuCd", ""))

    return {
        "nav":            nav if nav and nav > 0 else None,
        "nav_premium":    premium,
        "tracking_error": te,
        "aum":            round(aum_raw / 1e8, 0) if aum_raw and aum_raw > 0 else None,
        "_isu_cd":        isu_cd,
    }


def _parse_krx_holdings(rows: list[dict]) -> list[dict]:
    """KRX PDF 응답 rows에서 구성종목 리스트 추출."""
    result = []
    for r in rows[:10]:
        code = str(
            r.get("ISU_SRT_CD", r.get("isuSrtCd", r.get("티커", r.get("STND_ISU_CD", ""))))
        ).strip()
        name = str(
            r.get("ISU_ABBRV", r.get("isuAbbrv", r.get("종목명", r.get("ISU_NM", ""))))
        ).strip()
        weight = None
        for wk in ("COMPST_RT", "compstRt", "비중", "WGHT", "wght"):
            v = r.get(wk)
            if v is not None:
                try:
                    weight = float(str(v).replace(",", ""))
                    break
                except (ValueError, TypeError):
                    pass
        if code and name:
            mkt = r.get("MKT_ID", r.get("mktId", "")).upper()
            suffix = ".KQ" if mkt in ("KOSDAQ", "KQ") else ".KS"
            result.append({"ticker": f"{code}{suffix}", "name": name, "weight": weight})
    return result


# ═══════════════════════════════════════════════ Naver Finance ═══════════════

async def _naver_etf_page(code: str, client: httpx.AsyncClient) -> dict:
    """
    Naver Finance ETF 종목 페이지 스크래핑.
    NAV·괴리율·추적오차·운용보수·배당수익률·현재가 수집.
    """
    if not _HAS_BS4:
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
            timeout=10.0,
        )
        soup = BeautifulSoup(resp.text, "lxml")
    except Exception:
        return {}

    out: dict = {}

    # 현재가
    for selector in (".no_today .blind", "#_nowVal", ".today .num"):
        tag = soup.select_one(selector)
        if tag:
            try:
                out["price"] = float(tag.get_text(strip=True).replace(",", ""))
                break
            except ValueError:
                pass

    # ETF 상세 정보 dt/dd 파싱
    label_map: list[tuple[tuple[str, ...], str]] = [
        (("NAV", "순자산가치"),           "nav"),
        (("괴리율",),                      "nav_premium"),
        (("추적오차",),                    "tracking_error"),
        (("운용보수", "총보수", "수수료"), "expense_ratio"),
        (("배당수익률",),                  "dividend_yield"),
    ]

    for dt in soup.select("dt"):
        label  = dt.get_text(strip=True)
        dd_tag = dt.find_next_sibling("dd")
        if not dd_tag:
            continue
        raw = (
            dd_tag.get_text(strip=True)
            .replace("%", "")
            .replace(",", "")
            .replace("억원", "")
            .strip()
        )
        try:
            val = float(raw)
        except ValueError:
            continue
        for keywords, field in label_map:
            if any(k in label for k in keywords) and field not in out:
                out[field] = val
                break

    return out


# ═══════════════════════════════════════════════ 비동기 파이프라인 ══════════

async def _fetch_all_async(code: str) -> dict:
    """KRX + Naver Finance 비동기 병렬 수집."""
    async with httpx.AsyncClient(
        verify=False,
        follow_redirects=True,
        timeout=httpx.Timeout(25.0),
    ) as client:

        # ── 1) KRX: 최근 5 영업일 시도 ────────────────────────────────────────
        krx: dict = {}
        isu_cd = ""
        last_date = ""
        for delta in range(5):
            d_str = (datetime.now() - timedelta(days=delta)).strftime("%Y%m%d")
            try:
                rows = await _krx_etf_all_fundamentals(d_str, client)
                for row in rows:
                    short = row.get("ISU_SRT_CD", row.get("isuSrtCd", ""))
                    if str(short).strip() == code:
                        parsed = _parse_krx_fundamental(row)
                        isu_cd = parsed.pop("_isu_cd", "")
                        krx    = parsed
                        last_date = d_str
                        break
                if krx.get("nav"):
                    break
            except Exception:
                continue

        # ── 2) Naver + 구성종목 병렬 수집 ─────────────────────────────────────
        naver_task = asyncio.ensure_future(_naver_etf_page(code, client))

        if isu_cd and last_date:
            hold_task = asyncio.ensure_future(
                _krx_etf_holdings(isu_cd, last_date, client)
            )
        else:
            async def _no_holdings() -> list:
                return []
            hold_task = asyncio.ensure_future(_no_holdings())

        naver_raw, hold_rows = await asyncio.gather(
            naver_task, hold_task, return_exceptions=True
        )

    naver_data: dict      = naver_raw  if isinstance(naver_raw,  dict) else {}
    holdings_rows: list   = hold_rows  if isinstance(hold_rows,  list) else []
    top_holdings          = _parse_krx_holdings(holdings_rows)

    # ── 데이터 병합 (KRX 우선, Naver 보완) ────────────────────────────────────
    def pick(*vals):
        for v in vals:
            if v is not None:
                return v
        return None

    expense = pick(
        naver_data.get("expense_ratio"),
        _EXPENSE_RATIO_MAP.get(code),
    )

    return {
        "nav":            pick(krx.get("nav"),            naver_data.get("nav")),
        "nav_premium":    pick(krx.get("nav_premium"),    naver_data.get("nav_premium")),
        "tracking_error": pick(krx.get("tracking_error"), naver_data.get("tracking_error")),
        "aum":            krx.get("aum"),
        "expense_ratio":  expense,
        "dividend_yield": naver_data.get("dividend_yield"),
        "price":          naver_data.get("price"),
        "top_holdings":   top_holdings,
        "source":         "krx_direct" if krx.get("nav") else (
                          "naver"      if naver_data.get("nav") else "static_map"
                          ),
    }


# ═══════════════════════════════════════════════ 동기 공개 API ═══════════════

def _run_async(coro) -> dict:
    """
    비동기 코루틴을 동기 컨텍스트에서 실행.
    Streamlit/Tornado 이벤트 루프와의 충돌을 방지하기 위해
    이미 루프가 실행 중이면 별도 스레드에서 asyncio.run() 호출.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result(timeout=35)
        else:
            return loop.run_until_complete(coro)
    except Exception:
        try:
            return asyncio.run(coro)
        except Exception:
            return {}


def fetch_etf_data(ticker: str) -> dict:
    """
    ETF 기본 지표 수집 (동기 래퍼, 캐시 포함).

    반환 필드:
      nav, nav_premium, tracking_error, aum,
      expense_ratio, dividend_yield, price,
      top_holdings, source, _stale(캐시 사용 시)
    """
    code = ticker.replace(".KS", "").replace(".KQ", "").strip().zfill(6)

    try:
        fresh = _run_async(_fetch_all_async(code))
    except Exception:
        fresh = {}

    if fresh.get("nav") or fresh.get("expense_ratio"):
        _set_cache(code, fresh)
        return fresh

    # API 실패 → 캐시 반환
    cached = _get_cache(code)
    if cached:
        cached["_stale"] = True
        cached["source"] = f"{cached.get('source','?')} (캐시)"
        return cached

    # 캐시도 없음 → 정적 데이터만
    return {
        "nav":            None,
        "nav_premium":    None,
        "tracking_error": None,
        "aum":            None,
        "expense_ratio":  _EXPENSE_RATIO_MAP.get(code),
        "dividend_yield": None,
        "price":          None,
        "top_holdings":   [],
        "source":         "static_map",
    }


def get_expense_ratio(ticker: str) -> Optional[float]:
    """티커 코드로 운용보수(%) 직접 조회 (네트워크 없음)."""
    code = ticker.replace(".KS", "").replace(".KQ", "").strip().zfill(6)
    return _EXPENSE_RATIO_MAP.get(code)
