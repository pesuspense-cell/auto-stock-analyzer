"""
fundamental_db.py - 코스피/코스닥 펀더멘털 SQLite 캐시
분기(90일)마다 전체 업데이트, 단일 종목은 즉시 조회
"""
import sqlite3
import os
import pandas as pd
from datetime import datetime, timedelta
from contextlib import contextmanager
from typing import Optional

DB_PATH = os.path.join(os.path.dirname(__file__), "fundamentals.db")
QUARTER_DAYS = 90  # 분기 업데이트 주기


# ── DB 초기화 ──────────────────────────────────────────────────────────────────

def _init_db(conn: sqlite3.Connection):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS fundamentals (
            ticker       TEXT PRIMARY KEY,
            name         TEXT,
            market       TEXT,
            per          REAL,
            pbr          REAL,
            div          REAL,
            bps          REAL,
            eps          REAL,
            dps          REAL,
            market_cap   REAL,
            last_updated TEXT
        );
        CREATE TABLE IF NOT EXISTS update_log (
            market           TEXT PRIMARY KEY,
            last_full_update TEXT
        );
    """)
    conn.commit()


@contextmanager
def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    _init_db(conn)
    try:
        yield conn
    finally:
        conn.close()


# ── 업데이트 필요 여부 ─────────────────────────────────────────────────────────

def needs_update(market: str) -> bool:
    """마지막 전체 업데이트가 QUARTER_DAYS 이상 지났으면 True"""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT last_full_update FROM update_log WHERE market = ?", (market,)
        ).fetchone()
    if not row or not row["last_full_update"]:
        return True
    last = datetime.fromisoformat(row["last_full_update"])
    return (datetime.now() - last).days >= QUARTER_DAYS


def get_last_update(market: str) -> Optional[str]:
    """마지막 전체 업데이트 날짜 문자열 반환"""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT last_full_update FROM update_log WHERE market = ?", (market,)
        ).fetchone()
    return row["last_full_update"] if row else None


# ── 전체 시장 업데이트 (pykrx) ────────────────────────────────────────────────

def update_market(market: str, date: str = None) -> int:
    """
    pykrx로 KOSPI 또는 KOSDAQ 전체 펀더멘털을 한 번에 가져와 저장.
    date: 'YYYYMMDD' 형식, 기본값은 오늘 (영업일 아니면 자동으로 최근 영업일 사용)
    반환: 저장된 종목 수
    """
    try:
        from pykrx import stock as pkrx
    except ImportError:
        raise ImportError("pykrx 가 설치되어 있지 않습니다: pip install pykrx")

    if date is None:
        date = datetime.now().strftime("%Y%m%d")

    suffix = "KS" if market == "KOSPI" else "KQ"

    # 펀더멘털 (BPS, PER, PBR, EPS, DIV, DPS)
    fund_df = pkrx.get_market_fundamental(date, market=market)
    if fund_df is None or fund_df.empty:
        return 0

    # 시가총액
    cap_df = pkrx.get_market_cap(date, market=market)

    # 종목명
    tickers = pkrx.get_market_ticker_list(date, market=market)
    name_map = {t: pkrx.get_market_ticker_name(t) for t in tickers}

    now_str = datetime.now().isoformat()
    rows = []
    for ticker in fund_df.index:
        f = fund_df.loc[ticker]
        cap = float(cap_df.loc[ticker, "시가총액"]) if (cap_df is not None and ticker in cap_df.index) else None
        rows.append((
            f"{ticker}.{suffix}",
            name_map.get(ticker, ticker),
            market,
            _safe(f, "PER"),
            _safe(f, "PBR"),
            _safe(f, "DIV"),
            _safe(f, "BPS"),
            _safe(f, "EPS"),
            _safe(f, "DPS"),
            cap,
            now_str,
        ))

    with _get_conn() as conn:
        conn.executemany("""
            INSERT INTO fundamentals
                (ticker, name, market, per, pbr, div, bps, eps, dps, market_cap, last_updated)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(ticker) DO UPDATE SET
                name=excluded.name, market=excluded.market,
                per=excluded.per, pbr=excluded.pbr, div=excluded.div,
                bps=excluded.bps, eps=excluded.eps, dps=excluded.dps,
                market_cap=excluded.market_cap, last_updated=excluded.last_updated
        """, rows)
        conn.execute("""
            INSERT INTO update_log (market, last_full_update)
            VALUES (?, ?)
            ON CONFLICT(market) DO UPDATE SET last_full_update=excluded.last_full_update
        """, (market, now_str))
        conn.commit()

    return len(rows)


def _safe(row, col):
    try:
        v = row[col]
        return float(v) if pd.notna(v) and v != 0 else None
    except Exception:
        return None


# ── 단일 종목 조회 ─────────────────────────────────────────────────────────────

def get_ticker_fundamental(ticker: str) -> Optional[dict]:
    """
    DB에서 ticker 펀더멘털 조회.
    ticker 형식: '005930.KS' 또는 '005930.KQ'
    없으면 None 반환.
    """
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM fundamentals WHERE ticker = ?", (ticker,)
        ).fetchone()
    if not row:
        return None
    return dict(row)


def fetch_and_cache_single(ticker: str) -> Optional[dict]:
    """
    DB에 없는 종목을 pykrx로 단일 조회해 저장.
    ticker: '005930.KS' 형식
    """
    try:
        from pykrx import stock as pkrx
    except ImportError:
        return None

    code = ticker.split(".")[0]
    suffix = ticker.split(".")[-1] if "." in ticker else "KS"
    market = "KOSPI" if suffix == "KS" else "KOSDAQ"

    date = datetime.now().strftime("%Y%m%d")
    try:
        fund_df = pkrx.get_market_fundamental(date, date, code)
        if fund_df is None or fund_df.empty:
            return None
        f = fund_df.iloc[0]

        cap_df = pkrx.get_market_cap(date, date, code)
        cap = float(cap_df.iloc[0]["시가총액"]) if (cap_df is not None and not cap_df.empty) else None

        name = pkrx.get_market_ticker_name(code)
        now_str = datetime.now().isoformat()

        with _get_conn() as conn:
            conn.execute("""
                INSERT INTO fundamentals
                    (ticker, name, market, per, pbr, div, bps, eps, dps, market_cap, last_updated)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(ticker) DO UPDATE SET
                    name=excluded.name, market=excluded.market,
                    per=excluded.per, pbr=excluded.pbr, div=excluded.div,
                    bps=excluded.bps, eps=excluded.eps, dps=excluded.dps,
                    market_cap=excluded.market_cap, last_updated=excluded.last_updated
            """, (
                ticker, name, market,
                _safe(f, "PER"), _safe(f, "PBR"), _safe(f, "DIV"),
                _safe(f, "BPS"), _safe(f, "EPS"), _safe(f, "DPS"),
                cap, now_str,
            ))
            conn.commit()

        return get_ticker_fundamental(ticker)
    except Exception:
        return None


# ── 전체 시장 자동 업데이트 (분기 체크 포함) ──────────────────────────────────

def ensure_updated(market: str) -> bool:
    """분기가 지났으면 update_market 실행. 반환: 업데이트 실행 여부"""
    if needs_update(market):
        update_market(market)
        return True
    return False


def get_all_fundamentals(market: str = None) -> pd.DataFrame:
    """DB 전체 또는 특정 시장 펀더멘털을 DataFrame으로 반환"""
    with _get_conn() as conn:
        if market:
            df = pd.read_sql(
                "SELECT * FROM fundamentals WHERE market = ?", conn, params=(market,)
            )
        else:
            df = pd.read_sql("SELECT * FROM fundamentals", conn)
    return df


# ── DART 재무제표 ──────────────────────────────────────────────────────────────

def get_dart_financials(ticker: str, dart_api_key: str) -> dict:
    """
    DART OpenAPI로 최근 연간 재무제표 조회.
    반환: {revenue, operating_income, net_income, year} (단위: 억원)
    ticker: '005930.KS' 형식
    """
    if not dart_api_key:
        return {}
    try:
        import OpenDartReader
    except ImportError:
        return {}

    code = ticker.split(".")[0]
    dart = OpenDartReader.OpenDartReader(dart_api_key)

    def _parse_amount(val) -> Optional[float]:
        """DART 금액 문자열 → float (억원). △ 음수 처리 포함."""
        if val is None:
            return None
        s = str(val).strip().replace(",", "").replace(" ", "")
        if not s or s == "-":
            return None
        negative = s.startswith("△") or s.startswith("-")
        s = s.lstrip("△").lstrip("-")
        try:
            v = float(s) / 1e8
            return -v if negative else v
        except ValueError:
            return None

    def _extract(fs_is, keywords: list) -> Optional[float]:
        """IS(손익계산서) 행에서 키워드 포함 계정 찾아 당기 금액 반환."""
        for kw in keywords:
            rows = fs_is[fs_is["account_nm"].str.contains(kw, na=False)]
            if not rows.empty:
                val = _parse_amount(rows.iloc[0]["thstrm_amount"])
                if val is not None:
                    return val
        return None

    # 최근 3개년 시도 (사업보고서 기준)
    for year in [datetime.now().year - 1, datetime.now().year - 2, datetime.now().year - 3]:
        try:
            fs = dart.finstate(code, year)
            if fs is None or fs.empty:
                continue

            # 손익계산서 행만 필터
            fs_is = fs[fs["sj_div"] == "IS"] if "sj_div" in fs.columns else fs

            rev  = _extract(fs_is, ["매출액", "수익(매출액)", "영업수익", "매출"])
            oinc = _extract(fs_is, ["영업이익"])
            ninc = _extract(fs_is, ["당기순이익", "분기순이익", "순이익"])

            if rev is not None or oinc is not None:
                return {"revenue": rev, "operating_income": oinc,
                        "net_income": ninc, "year": year}
        except Exception:
            continue
    return {}


# ── pykrx 투자자별 매매 동향 ──────────────────────────────────────────────────

def get_trading_trend(ticker: str) -> dict:
    """
    pykrx로 최근 영업일 투자자별 순매수 금액 조회.
    반환: {date, 개인, 외국인, 기관합계, 금융투자, 연기금}  (단위: 억원)
    """
    try:
        from pykrx import stock as pkrx
    except ImportError:
        return {}

    code = ticker.split(".")[0]
    # 최근 5영업일 중 데이터 있는 날 사용
    for days_ago in range(1, 8):
        date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y%m%d")
        try:
            df = pkrx.get_market_trading_value_by_investor(date, date, code)
            if df is None or df.empty:
                continue
            # 순매수 컬럼 추출
            net = df["순매수"] if "순매수" in df.columns else df.iloc[:, 2]

            def _val(label: str) -> Optional[float]:
                if label in net.index:
                    return round(float(net[label]) / 1e8, 1)
                return None

            return {
                "date":   date,
                "개인":   _val("개인"),
                "외국인": _val("외국인"),
                "기관합계": _val("기관합계"),
                "금융투자": _val("금융투자"),
                "연기금":  _val("연기금등") or _val("연기금"),
            }
        except Exception:
            continue
    return {}
