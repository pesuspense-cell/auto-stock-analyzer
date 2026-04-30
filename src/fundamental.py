"""
fundamental.py — 펀더멘털 분석 모듈 (KRX API + 다중 폴백 + 비동기)

데이터 수집 파이프라인 (우선순위):
  1. SQLite 캐시 (TTL 내): 즉시 반환
  2. yfinance + Naver Finance + KRX Open API + DART (비동기 병렬 수집)
  3. FinanceDataReader (KRX 종목 가격·52주 고저 보완)
  4. SQLite 만료 캐시 (모든 소스 실패 시 → 빈 dict 대신 최신 캐시 반환)

KRX Open API (data.krx.co.kr):
  - 인증 없이 사용 가능 (공공 데이터 포털)
  - 사용 엔드포인트:
      MDCSTAT03501 → 전 종목 PER/PBR/EPS/BPS
      MDCSTAT01901 → 시가총액·상장주식수

DART 연동:
  - opendart.fss.or.kr / OpenDartReader 라이브러리
  - DART_API_KEY 환경변수 또는 st.secrets["DART_API_KEY"]
  - 수주잔고·최신 분기 재무 → calculate_fundamental_score 보너스 반영
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import os
import re
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from .providers import YahooProvider, KRXProvider, DARTProvider

# ── 로깅 설정 ─────────────────────────────────────────────────────────────────
logger = logging.getLogger("fundamental")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] fundamental: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

# ── 상수 ──────────────────────────────────────────────────────────────────────
_DB_PATH          = Path(__file__).parent.parent / "fundamentals.db"
_CACHE_TTL_MARKET = 10 * 60     # 장중(09:00–15:30): 10분
_CACHE_TTL_OFF    = 6 * 3600    # 장외: 6시간


# ─────────────────────────────────────────────────────────────────────────────
# 유틸리티
# ─────────────────────────────────────────────────────────────────────────────

def _is_krx(ticker: str) -> bool:
    return ticker.endswith(".KS") or ticker.endswith(".KQ")


def _code(ticker: str) -> str:
    """'005930.KS' → '005930'"""
    return ticker.split(".")[0].zfill(6)


def _to_float(val: Any, default: Optional[float] = None) -> Optional[float]:
    if val is None:
        return default
    try:
        return float(str(val).replace(",", "").strip())
    except (ValueError, TypeError):
        return default


def _coalesce(*vals):
    for v in vals:
        if v is not None:
            return v
    return None


def _cache_ttl() -> int:
    t = datetime.now().hour * 60 + datetime.now().minute
    return _CACHE_TTL_MARKET if 9 * 60 <= t <= 15 * 60 + 30 else _CACHE_TTL_OFF


# ─────────────────────────────────────────────────────────────────────────────
# SQLite 캐시 (장애 시 만료 캐시 서빙 포함)
# ─────────────────────────────────────────────────────────────────────────────

def _init_cache() -> None:
    with sqlite3.connect(_DB_PATH) as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS fundamental_cache (
                ticker     TEXT PRIMARY KEY,
                data_json  TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        c.commit()


def _load_cache(ticker: str) -> Optional[dict]:
    try:
        with sqlite3.connect(_DB_PATH) as c:
            row = c.execute(
                "SELECT data_json, updated_at FROM fundamental_cache WHERE ticker=?",
                (ticker,),
            ).fetchone()
        if not row:
            return None
        age = (datetime.now() - datetime.fromisoformat(row[1])).total_seconds()
        if age < _cache_ttl():
            logger.info("캐시 히트: %s (경과 %.0f초)", ticker, age)
            return json.loads(row[0])
        return None
    except Exception:
        return None


def _load_stale(ticker: str) -> Optional[dict]:
    """TTL 만료 무시 — API 완전 장애 시 마지막 캐시 반환"""
    try:
        with sqlite3.connect(_DB_PATH) as c:
            row = c.execute(
                "SELECT data_json, updated_at FROM fundamental_cache WHERE ticker=?",
                (ticker,),
            ).fetchone()
        if not row:
            return None
        data = json.loads(row[0])
        data["_stale"]       = True
        data["_stale_since"] = row[1]
        logger.warning("만료 캐시 사용: %s (저장=%s)", ticker, row[1])
        return data
    except Exception:
        return None


def _save_cache(ticker: str, data: dict) -> None:
    try:
        with sqlite3.connect(_DB_PATH) as c:
            c.execute(
                "INSERT OR REPLACE INTO fundamental_cache "
                "(ticker, data_json, updated_at) VALUES (?,?,?)",
                (
                    ticker,
                    json.dumps(data, ensure_ascii=False, default=str),
                    datetime.now().isoformat(),
                ),
            )
            c.commit()
        logger.info("캐시 저장: %s", ticker)
    except Exception as e:
        logger.warning("캐시 저장 실패: %s — %s", ticker, e)


# ─────────────────────────────────────────────────────────────────────────────
# 통합 비동기 수집 파이프라인 (Provider 패턴)
# ─────────────────────────────────────────────────────────────────────────────

# KRXApiClient / DartApiClient — 하위 호환 재노출 (외부 참조 유지)
from .providers.krx_provider  import KRXProvider  as KRXApiClient   # noqa: F401
from .providers.dart_provider import DARTProvider as DartApiClient  # noqa: F401


async def _fetch_all(ticker: str) -> dict:
    """
    Provider 패턴으로 병렬 수집 → 우선순위 병합.
    YahooProvider(기본) > KRXProvider(KRX: per/pbr/52주/시총) > DARTProvider(KRX: 재무)
    """
    yahoo_p = YahooProvider()
    krx_p   = KRXProvider()
    dart_p  = DARTProvider()

    t_yf   = asyncio.create_task(yahoo_p.fetch_with_fallback(ticker))
    t_krx  = asyncio.create_task(krx_p.fetch_with_fallback(ticker))
    t_dart = asyncio.create_task(dart_p.fetch_with_fallback(ticker))

    done, pending = await asyncio.wait(
        {t_yf, t_krx, t_dart}, timeout=20.0
    )
    for t in pending:
        t.cancel()

    def _safe_result(task: asyncio.Task) -> dict:
        try:
            return task.result() if not task.cancelled() else {}
        except Exception:
            return {}

    yf_data   = _safe_result(t_yf)
    krx_data  = _safe_result(t_krx)
    dart_data = _safe_result(t_dart)

    merged: dict = {}

    if yf_data:
        merged.update({k: v for k, v in yf_data.items() if not k.startswith("_")})
    else:
        logger.warning("yfinance 미수집, KRX 폴백: %s", ticker)

    # KRXProvider → per/pbr/eps_ttm/bps/market_cap/w52_high/w52_low/short_name/sector 보완
    for k in ("per", "pbr", "eps_ttm", "bps", "market_cap",
              "w52_high", "w52_low", "short_name", "sector", "div_yield"):
        if merged.get(k) is None and krx_data.get(k) is not None:
            merged[k] = krx_data[k]

    # DART — 별도 키 첨부 + 핵심 재무 보완 (억원 → 원 환산)
    if dart_data:
        merged["dart"] = dart_data
        for k, dk in [
            ("operating_income", "operating_income"),
            ("net_income",       "net_income"),
        ]:
            if merged.get(k) is None and dart_data.get(dk):
                merged[k] = dart_data[dk] * 1e8

    sources = []
    if yf_data:   sources.append("yfinance")
    if krx_data:  sources.append("krx")
    if dart_data: sources.append("dart")
    merged["source"] = "+".join(sources) if sources else "none"

    return merged




def _run_async(coro) -> Any:
    """Streamlit 동기 환경에서 비동기 코루틴 안전 실행"""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


# ─────────────────────────────────────────────────────────────────────────────
# 공개 API
# ─────────────────────────────────────────────────────────────────────────────

def get_fundamental_data(ticker: str) -> dict:
    """
    종목 펀더멘털 수집 — 다중 소스 폴백 + SQLite 캐시.

    반환 보장:
      - TTL 내 캐시 → 즉시 반환
      - 전 소스 실패 → 만료 캐시 반환 (빈 dict 반환 없음)
      - DB에도 없으면 {"source": "error", "data_status": "..."}
    """
    _init_cache()

    cached = _load_cache(ticker)
    if cached:
        return cached

    t0 = time.perf_counter()
    try:
        data = _run_async(_fetch_all(ticker))
        elapsed = time.perf_counter() - t0
        logger.info(
            "펀더멘털 수집 완료 [%s]: %.2f초 (소스=%s)",
            ticker, elapsed, data.get("source", "?"),
        )

        # 유효 데이터 여부 확인 (최소 한 개 핵심 지표)
        has_data = any(
            data.get(k) is not None
            for k in ("per", "pbr", "market_cap", "roe", "net_income")
        )
        if has_data:
            _save_cache(ticker, data)
            return data
        raise ValueError("유효 데이터 없음")

    except Exception as e:
        logger.error("펀더멘털 수집 최종 실패 [%s]: %s", ticker, e)
        stale = _load_stale(ticker)
        if stale:
            stale["data_status"] = "캐시 사용 중 (API 일시 장애)"
            return stale
        return {
            "source":      "error",
            "data_status": "데이터를 불러올 수 없습니다",
        }


# ─────────────────────────────────────────────────────────────────────────────
# 펀더멘털 점수 계산 (버핏·그레이엄·린치·오닐 + DART 수주잔고)
# ─────────────────────────────────────────────────────────────────────────────

def calculate_fundamental_score(
    info: dict,
    close_price: float = None,
    dart_data: dict = None,
) -> dict:
    """
    버핏·그레이엄·린치·오닐 투자 법칙 기반 펀더멘털 점수.

    가중치 구조 (최종 점수 ±8):
      성장성  40%: PEG·매출성장·DART 수주잔고
      수익성  30%: ROE 지속성·FCF Yield·OCF/NI 품질
      안정성  20%: 그레이엄 공식·부채비율
      모멘텀  10%: 52주 고가 근접·주주환원율

    DART 수주잔고 보너스:
      수주잔고/매출 > 3배 → +1.5 (향후 3년 매출 확보)
      수주잔고/매출 > 1.5배 → +0.8
      수주잔고/매출 > 0.5배 → +0.3
    """
    # dart_data는 파라미터 또는 info["dart"] 중 우선
    dart = dart_data or info.get("dart", {})

    reasons: list[str] = []

    per             = info.get("per")
    pbr             = info.get("pbr")
    roe             = info.get("roe")
    roe_history     = info.get("roe_history", [])
    debt_equity     = info.get("debt_equity")
    revenue_growth  = info.get("revenue_growth")
    earnings_growth = info.get("earnings_growth")
    eps_history     = info.get("eps_history", [])
    w52_high        = info.get("w52_high")
    w52_low         = info.get("w52_low")
    fcf             = info.get("free_cashflow")
    ocf             = info.get("ocf")
    net_income      = info.get("net_income")
    market_cap      = info.get("market_cap")
    sh_yield        = info.get("shareholder_yield")

    # ── 린치 EPS 3년 CAGR ────────────────────────────────────────────────────
    eps_cagr_3yr: Optional[float] = None
    eps_cagr_note = ""
    if len(eps_history) >= 4:
        eps_old, eps_new = eps_history[-4], eps_history[-1]
        if eps_old > 0 and eps_new > 0:
            eps_cagr_3yr = (eps_new / eps_old) ** (1.0 / 3.0) - 1.0
            eps_cagr_note = (
                f"EPS 3년 CAGR {eps_cagr_3yr*100:.1f}%"
                f" ({eps_old:.2f}→{eps_new:.2f})"
            )
        elif eps_old <= 0 or eps_new <= 0:
            eps_cagr_note = (
                f"EPS 음수 포함 — CAGR 계산 불가 ({eps_old:.2f}→{eps_new:.2f})"
            )
    elif len(eps_history) >= 2:
        eps_old, eps_new = eps_history[0], eps_history[-1]
        n = len(eps_history) - 1
        if eps_old > 0 and eps_new > 0:
            eps_cagr_3yr = (eps_new / eps_old) ** (1.0 / n) - 1.0
            eps_cagr_note = (
                f"EPS {n}년 CAGR {eps_cagr_3yr*100:.1f}%"
                f" ({eps_old:.2f}→{eps_new:.2f}, {n}년 데이터)"
            )

    effective_growth = eps_cagr_3yr if eps_cagr_3yr is not None else earnings_growth

    g = 0.0   # 성장성  cap ±3.5 (DART 보너스 포함 ±5.0)
    p = 0.0   # 수익성  cap ±6.0
    s = 0.0   # 안정성  cap ±3.5
    m = 0.0   # 모멘텀  cap ±2.5

    # ════════════════════════════════════════════════════════════════
    # 성장성 (Growth) — 린치 핵심, 40%
    # ════════════════════════════════════════════════════════════════

    peg = None
    _per_ok    = per is not None and per > 0
    _growth_ok = effective_growth is not None and effective_growth > 0

    if _per_ok and _growth_ok:
        peg = per / (effective_growth * 100)
        src = eps_cagr_note if eps_cagr_3yr is not None else (
            f"성장률={earnings_growth*100:.1f}%(yf)"
            if earnings_growth else "yf"
        )
        if peg < 0.5:
            g += 2.0
            reasons.append(f"[린치] PEG={peg:.2f} < 0.5 → 강력 성장 저평가 ({src})")
        elif peg < 1.0:
            g += 1.0
            reasons.append(f"[린치] PEG={peg:.2f} < 1.0 → 성장 대비 저평가 ({src})")
        elif peg < 1.5:
            g += 0.3
            reasons.append(f"[린치] PEG={peg:.2f} ({src})")
        elif peg > 2.0:
            g -= 1.0
            reasons.append(f"[린치] PEG={peg:.2f} > 2.0 → 성장 대비 고평가 ({src})")
        else:
            reasons.append(f"[린치] PEG={peg:.2f} ({src})")
    elif eps_cagr_note:
        tag = "PER 없어 PEG 계산 불가" if not _per_ok else "EPS 감소세 → PEG 계산 불가"
        reasons.append(f"[린치] {eps_cagr_note} ({tag})")

    # 매출 성장
    if revenue_growth is not None:
        rp = revenue_growth * 100
        if rp >= 25:
            g += 1.5
            reasons.append(f"[린치] 매출 {rp:.1f}% 급성장 → 텐배거 후보")
        elif rp >= 10:
            g += 0.5
            reasons.append(f"[린치] 매출 {rp:.1f}% 성장 → 양호")
        elif rp < -10:
            g -= 1.0
            reasons.append(f"[린치] 매출 {rp:.1f}% 감소 → 펀더멘털 악화")

    # ── DART 수주잔고 보너스 (수주 기반 기업 전용) ───────────────────────────
    order_backlog = dart.get("order_backlog")
    dart_revenue  = dart.get("revenue")
    base_revenue  = info.get("total_revenue")  # 단위: 원
    # DART 매출은 억원 단위이므로 비교 시 환산
    effective_rev = (
        (dart_revenue * 1e8) if dart_revenue
        else base_revenue
    )
    if order_backlog and effective_rev and effective_rev > 0:
        backlog_in_won = order_backlog * 1e8
        bl_ratio       = backlog_in_won / effective_rev
        if bl_ratio > 3.0:
            g += 1.5
            reasons.append(
                f"[DART] 수주잔고/매출={bl_ratio:.1f}배 "
                f"→ 향후 3년치 매출 가시성 확보 (성장 안정성 높음)"
            )
        elif bl_ratio > 1.5:
            g += 0.8
            reasons.append(
                f"[DART] 수주잔고/매출={bl_ratio:.1f}배 → 중기 매출 안정 확보"
            )
        elif bl_ratio > 0.5:
            g += 0.3
            reasons.append(
                f"[DART] 수주잔고/매출={bl_ratio:.1f}배 → 수주 기반 안정성 보통"
            )
    elif order_backlog and not effective_rev:
        reasons.append(
            f"[DART] 수주잔고 {order_backlog:.0f}억원 확인 (매출 데이터 없어 비율 계산 불가)"
        )

    # ════════════════════════════════════════════════════════════════
    # 수익성 (Profitability) — 버핏 핵심, 30%
    # ════════════════════════════════════════════════════════════════

    roe_mean_val = roe_std_val = None
    if len(roe_history) >= 2:
        roe_mean_val = float(np.mean(roe_history))
        roe_std_val  = float(np.std(roe_history, ddof=min(1, len(roe_history) - 1)))
        if roe_mean_val >= 15 and roe_std_val <= 5:
            p += 2.0
            reasons.append(
                f"[버핏] ROE 지속성 우수: 평균 {roe_mean_val:.1f}%·표준편차 {roe_std_val:.1f}%p"
                f" → 꾸준한 수익 창출"
            )
        elif roe_mean_val >= 15 and roe_std_val <= 10:
            p += 1.0
            reasons.append(
                f"[버핏] ROE 평균 {roe_mean_val:.1f}%, 편차 {roe_std_val:.1f}%p → 양호"
            )
        elif roe_mean_val >= 15:
            reasons.append(
                f"[버핏] ROE 평균 {roe_mean_val:.1f}%이나 편차 {roe_std_val:.1f}%p 과대"
                f" → 단발성 고ROE"
            )
        elif roe_mean_val < 8:
            p -= 1.0
            reasons.append(f"[버핏] ROE 평균 {roe_mean_val:.1f}% → 수익성 부진")
    elif roe is not None:
        roe_pct = roe * 100
        if roe_pct >= 20:
            p += 1.5
            reasons.append(f"[버핏] ROE {roe_pct:.1f}% ≥ 20% → 우량 기업")
        elif roe_pct >= 15:
            p += 0.8
            reasons.append(f"[버핏] ROE {roe_pct:.1f}% ≥ 15% → 버핏 기준 충족")
        elif roe_pct < 5:
            p -= 1.0
            reasons.append(f"[버핏] ROE {roe_pct:.1f}% < 5% → 수익성 부진")

    fcf_yield = None
    if fcf and market_cap and market_cap > 0:
        fcf_yield = fcf / market_cap * 100
        if fcf_yield > 8:
            p += 1.5
            reasons.append(f"[버핏] FCF Yield {fcf_yield:.1f}% > 8% → 현금창출 탁월")
        elif fcf_yield > 5:
            p += 0.5
            reasons.append(f"[버핏] FCF Yield {fcf_yield:.1f}% > 5% → 양호")
        elif fcf_yield < 0:
            p -= 1.0
            reasons.append(f"[버핏] FCF Yield {fcf_yield:.1f}% < 0 → 현금소진 경고")

    ocf_ni_ratio = None
    if (
        ocf and net_income and net_income > 0
        and not pd.isna(ocf) and not pd.isna(net_income)
    ):
        ocf_ni_ratio = ocf / net_income
        if ocf_ni_ratio > 1.5:
            p += 1.5
            reasons.append(
                f"[버핏/린치] OCF/NI={ocf_ni_ratio:.2f} > 1.5 → 이익의 질 탁월"
            )
        elif ocf_ni_ratio > 1.0:
            p += 0.8
            reasons.append(
                f"[버핏/린치] OCF/NI={ocf_ni_ratio:.2f} > 1.0 → 현금흐름 양호"
            )
        elif ocf_ni_ratio < 0.5:
            p -= 0.8
            reasons.append(
                f"[주의] OCF/NI={ocf_ni_ratio:.2f} < 0.5 → 이익 대비 현금 부족"
            )
        elif ocf_ni_ratio < 0:
            p -= 1.5
            reasons.append(
                f"[경고] OCF/NI={ocf_ni_ratio:.2f} < 0 → 이익 중 현금 유출 경고"
            )

    # ════════════════════════════════════════════════════════════════
    # 안정성 (Stability) — 그레이엄·버핏, 20%
    # ════════════════════════════════════════════════════════════════

    gnum = None
    if per and pbr and per > 0 and pbr > 0:
        gnum = per * pbr
        if gnum < 15:
            s += 2.0
            reasons.append(f"[그레이엄] PBR×PER={gnum:.1f} < 15 → 강한 저평가")
        elif gnum < 22.5:
            s += 1.0
            reasons.append(f"[그레이엄] PBR×PER={gnum:.1f} < 22.5 → 적정 평가")
        elif gnum > 45:
            s -= 2.0
            reasons.append(f"[그레이엄] PBR×PER={gnum:.1f} > 45 → 고평가 경고")
        elif gnum > 30:
            s -= 1.0
            reasons.append(f"[그레이엄] PBR×PER={gnum:.1f} > 30 → 다소 고평가")

    if debt_equity is not None and debt_equity >= 0:
        if debt_equity < 50:
            s += 1.5
            reasons.append(f"[버핏] 부채비율 {debt_equity:.0f}% < 50% → 재무 우량")
        elif debt_equity < 100:
            s += 0.5
        elif debt_equity > 200:
            s -= 1.5
            reasons.append(f"[버핏] 부채비율 {debt_equity:.0f}% > 200% → 재무 위험")
        elif debt_equity > 100:
            s -= 0.5
            reasons.append(f"[버핏] 부채비율 {debt_equity:.0f}% > 100% → 주의 필요")

    # ════════════════════════════════════════════════════════════════
    # 모멘텀 (Momentum) — 오닐·Value-up, 10%
    # ════════════════════════════════════════════════════════════════

    oneil_ratio = None
    if close_price and w52_high and w52_low and w52_high > w52_low:
        oneil_ratio = close_price / w52_high
        pos = (close_price - w52_low) / (w52_high - w52_low)
        if oneil_ratio >= 0.95:
            m += 1.5
            reasons.append(
                f"[오닐] 52주 고가 {oneil_ratio*100:.1f}% 근접 → 신고가 돌파 모멘텀"
            )
        elif oneil_ratio >= 0.85:
            m += 0.5
        elif pos <= 0.15:
            m += 0.3
            reasons.append(f"[오닐] 52주 저가 근접 ({pos*100:.0f}%) → 바닥 반등 기대")

    if sh_yield is not None and sh_yield > 0:
        if sh_yield >= 5:
            m += 1.0
            reasons.append(f"[주주환원] 배당+자사주 {sh_yield:.1f}% → 주주 가치 우수")
        elif sh_yield >= 3:
            m += 0.5
            reasons.append(f"[주주환원] 배당+자사주 {sh_yield:.1f}% → 양호")
        elif sh_yield >= 1:
            m += 0.2

    # ════════════════════════════════════════════════════════════════
    # 가중 합산: 40·30·20·10 → ±8 스케일
    # ════════════════════════════════════════════════════════════════

    def _norm(val: float, cap: float) -> float:
        return max(-1.0, min(1.0, val / cap)) if cap > 0 else 0.0

    # 성장성 cap을 DART 보너스 포함 시 확장
    g_cap = 5.0 if order_backlog else 3.5

    score = round(
        (
            _norm(g, g_cap) * 0.40
            + _norm(p, 6.0) * 0.30
            + _norm(s, 3.5) * 0.20
            + _norm(m, 2.5) * 0.10
        )
        * 8,
        1,
    )

    growth_sub = round(50 + _norm(g, g_cap) * 50, 1)
    profit_sub = round(50 + _norm(p, 6.0) * 50, 1)
    stable_sub = round(50 + _norm(s, 3.5) * 50, 1)
    moment_sub = round(50 + _norm(m, 2.5) * 50, 1)

    # ── 거장의 한 줄 평 ────────────────────────────────────────────────────────
    master_verdicts: dict[str, dict] = {}

    # 그레이엄
    if gnum is not None:
        if gnum < 22.5:
            master_verdicts["그레이엄"] = {
                "icon": "✅", "판정": "통과",
                "comment": (
                    f"안전마진 확보. PBR×PER={gnum:.1f}으로 그레이엄 기준(22.5) 이내."
                ),
            }
        elif gnum < 35:
            master_verdicts["그레이엄"] = {
                "icon": "⚠️", "판정": "주의",
                "comment": f"안전마진 부족. PBR×PER={gnum:.1f}이 기준치(22.5) 초과.",
            }
        else:
            master_verdicts["그레이엄"] = {
                "icon": "🚫", "판정": "경고",
                "comment": f"안전마진 없음. PBR×PER={gnum:.1f}은 기준을 크게 초과.",
            }
    else:
        master_verdicts["그레이엄"] = {
            "icon": "—", "판정": "데이터 부족",
            "comment": "PER·PBR 정보 없어 평가 불가",
        }

    # 버핏
    ok_items, ng_items = [], []
    roe_pct_display = (
        roe_history[0] if roe_history else ((roe or 0) * 100)
    )
    if roe_mean_val is not None and roe_mean_val >= 15 and (roe_std_val or 99) <= 10:
        ok_items.append(f"ROE 지속성 {roe_mean_val:.0f}%")
    elif roe_pct_display >= 15:
        ok_items.append(f"ROE {roe_pct_display:.0f}%")
    else:
        ng_items.append(f"ROE {roe_pct_display:.0f}%")

    if debt_equity is not None:
        (ok_items if debt_equity < 100 else ng_items).append(
            f"부채비율 {debt_equity:.0f}%"
        )
    if fcf_yield is not None:
        if fcf_yield > 5:
            ok_items.append(f"FCF Yield {fcf_yield:.1f}%")
        elif fcf_yield < 0:
            ng_items.append("FCF 음수")
    if ocf_ni_ratio is not None:
        if ocf_ni_ratio > 1.0:
            ok_items.append(f"OCF/NI={ocf_ni_ratio:.1f}")
        elif ocf_ni_ratio < 0.5:
            ng_items.append(f"OCF/NI 낮음({ocf_ni_ratio:.1f})")

    if len(ok_items) >= 3 and not ng_items:
        master_verdicts["버핏"] = {
            "icon": "✅", "판정": "통과",
            "comment": f"{', '.join(ok_items)}로 경제적 해자가 튼튼한 기업.",
        }
    elif len(ng_items) >= 2:
        master_verdicts["버핏"] = {
            "icon": "🚫", "판정": "미달",
            "comment": (
                f"{', '.join(ng_items)} 등 버핏 기준 미달. 장기 보유 신중 검토 필요."
            ),
        }
    else:
        master_verdicts["버핏"] = {
            "icon": "⚠️", "판정": "부분 충족",
            "comment": (
                f"긍정 {', '.join(ok_items) or '없음'} / "
                f"우려 {', '.join(ng_items) or '없음'}"
            ),
        }

    # 린치
    growth_basis = eps_cagr_note if eps_cagr_note else (
        f"성장률={earnings_growth*100:.1f}%(yf)" if earnings_growth
        else "성장률 데이터 없음"
    )
    if peg is not None:
        if peg < 0.5:
            master_verdicts["린치"] = {
                "icon": "🚀", "판정": "강력추천",
                "comment": (
                    f"성장 대비 주가가 매우 쌉니다 (PEG={peg:.2f}). "
                    f"전형적 성장주 패턴. ({growth_basis})"
                ),
            }
        elif peg < 1.0:
            master_verdicts["린치"] = {
                "icon": "✅", "판정": "추천",
                "comment": (
                    f"PEG={peg:.2f}로 성장 대비 저평가. "
                    f"린치 기준 매수권. ({growth_basis})"
                ),
            }
        elif peg > 2.0:
            master_verdicts["린치"] = {
                "icon": "⚠️", "판정": "과열",
                "comment": f"PEG={peg:.2f} > 2.0. 성장률 대비 주가 앞섬. ({growth_basis})",
            }
        else:
            master_verdicts["린치"] = {
                "icon": "⚪", "판정": "중립",
                "comment": f"PEG={peg:.2f}. 적정 밸류에이션 구간. ({growth_basis})",
            }
    elif eps_cagr_note:
        master_verdicts["린치"] = {
            "icon": "—", "판정": "데이터 부족",
            "comment": (
                f"PEG 계산 불가 ({'PER 없음' if not _per_ok else 'EPS 감소세'}). "
                f"{eps_cagr_note}"
            ),
        }
    else:
        master_verdicts["린치"] = {
            "icon": "—", "판정": "데이터 부족",
            "comment": "PEG 계산 불가 (EPS·성장률 데이터 없음)",
        }

    # 오닐
    if oneil_ratio is not None:
        if oneil_ratio >= 0.95:
            master_verdicts["오닐"] = {
                "icon": "🔥", "판정": "추세확인",
                "comment": (
                    f"신고가 {oneil_ratio*100:.0f}% 수준. "
                    "거래량 동반 여부 확인 후 진입."
                ),
            }
        elif oneil_ratio >= 0.80:
            master_verdicts["오닐"] = {
                "icon": "👀", "판정": "관망",
                "comment": (
                    f"52주 고가 {oneil_ratio*100:.0f}%. "
                    "신고가 돌파 시 진입 전략 준비."
                ),
            }
        else:
            master_verdicts["오닐"] = {
                "icon": "⚠️", "판정": "조심",
                "comment": (
                    f"52주 고가 대비 {oneil_ratio*100:.0f}%. "
                    "CANSLIM 기준 상승 모멘텀 부재."
                ),
            }
    else:
        master_verdicts["오닐"] = {
            "icon": "—", "판정": "데이터 부족",
            "comment": "52주 고가/저가 데이터 없음",
        }

    label = (
        "펀더멘털 강함" if score >= 3
        else "펀더멘털 약함" if score <= -2
        else "펀더멘털 보통"
    )

    # 수주잔고 표시용 (UI)
    bl_ratio_display = None
    if order_backlog and effective_rev and effective_rev > 0:
        bl_ratio_display = round((order_backlog * 1e8) / effective_rev, 2)

    return {
        "fund_score":      score,
        "fund_label":      label,
        "fund_reasons":    reasons,
        "master_verdicts": master_verdicts,
        "sub_growth":      growth_sub,
        "sub_profit":      profit_sub,
        "sub_stable":      stable_sub,
        "sub_moment":      moment_sub,
        "roe_mean":        round(roe_mean_val, 1) if roe_mean_val is not None else None,
        "roe_std":         round(roe_std_val,  1) if roe_std_val  is not None else None,
        "ocf_ni_ratio":    round(ocf_ni_ratio, 2) if ocf_ni_ratio is not None else None,
        "shareholder_yield": info.get("shareholder_yield"),
        "fcf_yield":       round(fcf_yield, 1) if fcf_yield is not None else None,
        "peg":             round(peg, 2) if peg is not None else None,
        "eps_history":     eps_history,
        "eps_cagr_3yr":    round(eps_cagr_3yr * 100, 1) if eps_cagr_3yr is not None else None,
        "dart_order_backlog": order_backlog,
        "dart_backlog_ratio": bl_ratio_display,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 패스-스루 (stock_ai.py 위임)
# ─────────────────────────────────────────────────────────────────────────────

def get_investment_recommendation(
    current_price: float,
    avg_price: float,
    indicators: dict,
    tech_score: float = 0.0,
    news_score: float = 0.0,
    fund_score: float = 0.0,
    dead_time: Optional[dict] = None,
) -> dict:
    """매수·매도 추천 — stock_ai.py 위임 (dead_time 매수유보 포함)"""
    from stock_ai import get_investment_recommendation as _fn
    return _fn(
        current_price, avg_price, indicators,
        tech_score, news_score, fund_score,
        dead_time,
    )


def get_insider_trades_sec(ticker: str, days: int = 90) -> "pd.DataFrame":
    """SEC EDGAR Form 4 내부자 거래 (미국 주식 전용)."""
    return DARTProvider().fetch_insider_trades(ticker, days)


def get_etf_fundamental_data(ticker: str) -> dict:
    """ETF 핵심 지표 수집 — etf_async.py 위임 (KRX API 기반)"""
    from src.etf_async import get_etf_fundamental_data as _fn
    from stock_ai import _ETF_PORTFOLIO_MAP
    code = ticker.replace(".KS", "").replace(".KQ", "").strip().zfill(6)
    portfolio_info = _ETF_PORTFOLIO_MAP.get(code, {})
    return _fn(ticker, portfolio_info)


def calculate_etf_score(etf_data: dict) -> dict:
    """ETF 투자 점수 산출 — stock_ai.py 위임"""
    from stock_ai import calculate_etf_score as _fn
    return _fn(etf_data)


__all__ = [
    "get_fundamental_data",
    "calculate_fundamental_score",
    "get_investment_recommendation",
    "get_insider_trades_sec",
    "get_etf_fundamental_data",
    "calculate_etf_score",
    "KRXApiClient",
    "DartApiClient",
]
