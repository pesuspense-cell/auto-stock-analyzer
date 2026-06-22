"use client";

import useSWR from "swr";

import { signClass } from "@/lib/format";
import { pollingIntervalMs, marketState, marketStateLabel } from "@/lib/market-hours";
import type { IndicatorsResponse, IndicatorSnapshot, StockHit } from "@/lib/api-types";
import { PriceChart } from "./PriceChart";
import { AnalysisPanel } from "./AnalysisPanel";

const jfetch = async (url: string) => {
  const res = await fetch(url, { credentials: "same-origin" });
  if (!res.ok) throw new Error((await res.json().catch(() => ({})))?.error ?? res.statusText);
  return res.json();
};

const fmt = (v: number | null, d = 1) =>
  v == null ? "—" : v.toLocaleString("ko-KR", { minimumFractionDigits: d, maximumFractionDigits: d });

/**
 * 차트 분석 탭 — 가격 차트 + 기술지표 + AI 심층 분석을 하나로 통합.
 * 매매 우선순위 순서: ① 가격 차트 ② 종합 신호 ③ 핵심 지표 ④ (접힘) 보조 지표·근거 ⑤ AI 심층 분석.
 */
export function ChartAnalysisTab({ picked }: { picked: StockHit | null }) {
  const { data, error, isLoading } = useSWR<IndicatorsResponse>(
    picked ? `/api/v1/indicators/${encodeURIComponent(picked.ticker)}` : null,
    jfetch,
    {
      refreshInterval: () => (picked ? pollingIntervalMs(picked.ticker) : 0),
      revalidateOnFocus: false,
      keepPreviousData: true,
    }
  );

  if (!picked) return <Placeholder>좌측에서 종목을 선택하면 차트·기술지표·AI 분석이 표시됩니다.</Placeholder>;

  const ind = data?.indicators;
  const sig = data?.signal;
  const state = marketState(picked.ticker);

  return (
    <div className="space-y-4">
      {/* 헤더 */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h2 className="text-lg font-bold text-ink">
            {picked.nameKr || picked.name} <span className="tnum text-sm text-ink-2">{picked.ticker}</span>
          </h2>
          {data && <p className="text-xs text-ink-2">기준일 {data.asOf} · {data.cached ? "캐시" : "실시간 계산"}</p>}
        </div>
        <span className="rounded-full bg-canvas px-3 py-1 text-xs font-medium text-ink-2">
          {marketStateLabel(state)}
        </span>
      </div>

      {isLoading && !data && <Placeholder>📐 차트·지표 계산 중…</Placeholder>}
      {error && !data && <Placeholder>지표를 불러오지 못했습니다.</Placeholder>}

      {/* ① 가격 차트 */}
      {data?.chart && data.chart.candles.length > 0 && <PriceChart chart={data.chart} />}

      {/* ② AI 심층 분석 — 종목 선택 시 항상 자동 가동(주가·기술·뉴스·펀더멘털 결합) */}
      <AnalysisPanel picked={picked} />

      {/* ③ 핵심 지표 (매매 판단에 직접 쓰는 지표) */}
      {ind && <CoreIndicators ind={ind} />}

      {/* ④ 보조 지표 + 신호 근거 (접힘) */}
      {ind && <SecondaryIndicators ind={ind} />}
      {sig && sig.reasons.length > 0 && (
        <details className="rounded-card border border-hairline bg-surface p-4 shadow-card">
          <summary className="cursor-pointer text-sm font-semibold text-ink">🔎 기술 신호 근거 ({sig.reasons.length})</summary>
          <ul className="mt-2 space-y-1 text-[0.82rem] text-ink-2">
            {sig.reasons.map((r, i) => <li key={i}>• {r}</li>)}
          </ul>
        </details>
      )}
    </div>
  );
}

/* ── 핵심 지표 ─────────────────────────────────────────────────────── */
function CoreIndicators({ ind }: { ind: IndicatorSnapshot }) {
  return (
    <section className="space-y-2">
      <h3 className="text-sm font-semibold text-ink">📌 핵심 지표</h3>
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-4">
        <Metric
          label="현재가"
          value={fmt(ind.price, 2)}
          tone={signClass(ind.changePct)}
          sub={`${ind.changePct >= 0 ? "+" : ""}${ind.changePct.toFixed(2)}%`}
        />
        <Metric
          label="RSI(14)"
          value={fmt(ind.rsi)}
          tone={ind.rsi == null ? "" : ind.rsi < 30 ? "text-gain" : ind.rsi > 70 ? "text-loss" : ""}
          sub={ind.rsi == null ? undefined : ind.rsi < 30 ? "과매도" : ind.rsi > 70 ? "과매수" : "중립"}
        />
        <Metric
          label="MACD Hist"
          value={fmt(ind.macdHist, 3)}
          tone={ind.macdHist == null ? "" : signClass(ind.macdHist)}
          sub={ind.macdHist == null ? undefined : ind.macdHist >= 0 ? "상승 모멘텀" : "하락 모멘텀"}
        />
        <Metric
          label="ADX(14)"
          value={fmt(ind.adx)}
          tone={ind.adx != null && ind.adx > 25 ? "text-accent" : ""}
          sub={ind.adx == null ? undefined : ind.adx > 25 ? "추세 강함" : "추세 약함"}
        />
        <Metric
          label="볼린저 %B"
          value={ind.bbPct == null ? "—" : `${(ind.bbPct * 100).toFixed(0)}%`}
          tone={ind.bbPct == null ? "" : ind.bbPct > 0.8 ? "text-loss" : ind.bbPct < 0.2 ? "text-gain" : ""}
          sub={ind.bbPct == null ? undefined : ind.bbPct > 0.8 ? "상단 과열" : ind.bbPct < 0.2 ? "하단 과매도" : "밴드 내"}
        />
        <Metric
          label="스토캐스틱 %K/%D"
          value={`${fmt(ind.stochK)} / ${fmt(ind.stochD)}`}
          tone={ind.stochK == null ? "" : ind.stochK < 20 ? "text-gain" : ind.stochK > 80 ? "text-loss" : ""}
        />
        <Metric
          label="MFI(14)"
          value={fmt(ind.mfi)}
          tone={ind.mfi == null ? "" : ind.mfi < 20 ? "text-gain" : ind.mfi > 80 ? "text-loss" : ""}
          sub={ind.mfi == null ? undefined : ind.mfi < 20 ? "유입 약" : ind.mfi > 80 ? "유입 과열" : undefined}
        />
        <TrendMetric ind={ind} />
      </div>
    </section>
  );
}

/** 이동평균 배열(정/역배열) — 추세 한눈에. */
function TrendMetric({ ind }: { ind: IndicatorSnapshot }) {
  const { price, ema20, ema50, ema200 } = ind;
  let label = "혼조";
  let tone = "";
  if (ema20 != null && ema50 != null && ema200 != null) {
    if (price > ema20 && ema20 > ema50 && ema50 > ema200) {
      label = "정배열 ▲";
      tone = "text-gain";
    } else if (price < ema20 && ema20 < ema50 && ema50 < ema200) {
      label = "역배열 ▼";
      tone = "text-loss";
    } else {
      label = price > ema50 ? "상승 우위" : "하락 우위";
      tone = price > ema50 ? "text-gain" : "text-loss";
    }
  }
  const aboveCount = [ema20, ema50, ema200].filter((e) => e != null && price > e).length;
  return <Metric label="이평선 추세" value={label} tone={tone} sub={`현재가 > 이평 ${aboveCount}/3`} />;
}

/* ── 보조 지표 (접힘) ─────────────────────────────────────────────── */
function SecondaryIndicators({ ind }: { ind: IndicatorSnapshot }) {
  return (
    <details className="rounded-card border border-hairline bg-surface p-4 shadow-card">
      <summary className="cursor-pointer text-sm font-semibold text-ink">🔧 보조 · 참고 지표</summary>
      <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-4">
        <Metric label="EMA 20" value={fmt(ind.ema20, 2)} />
        <Metric label="EMA 50" value={fmt(ind.ema50, 2)} />
        <Metric label="EMA 200" value={fmt(ind.ema200, 2)} />
        <Metric label="SMA 20" value={fmt(ind.sma20, 2)} />
        <Metric label="CCI(20)" value={fmt(ind.cci, 0)} tone={ind.cci == null ? "" : ind.cci < -100 ? "text-gain" : ind.cci > 100 ? "text-loss" : ""} />
        <Metric label="Williams %R" value={fmt(ind.williamsR)} />
        <Metric label="ROC(12)" value={ind.roc == null ? "—" : `${ind.roc.toFixed(1)}%`} tone={ind.roc == null ? "" : signClass(ind.roc)} />
        <Metric label="Z-Score" value={fmt(ind.zscore, 2)} tone={ind.zscore == null ? "" : ind.zscore < -1.5 ? "text-gain" : ind.zscore > 1.5 ? "text-loss" : ""} />
        <Metric label="ATR(14)" value={fmt(ind.atr, 2)} />
        <Metric label="월 VWAP" value={fmt(ind.vwapM, 2)} />
        <Metric label="OBV" value={fmt(ind.obv, 0)} />
        <Metric label="MACD / Signal" value={`${fmt(ind.macd, 2)} / ${fmt(ind.macdSignal, 2)}`} />
      </div>
    </details>
  );
}

/* ── 공용 프리미티브 ───────────────────────────────────────────────── */
function Metric({ label, value, tone = "", sub }: { label: string; value: string; tone?: string; sub?: string }) {
  return (
    <div className="rounded-lg border border-hairline bg-surface p-3 shadow-card">
      <div className="text-[0.68rem] text-ink-2">{label}</div>
      <div className={`tnum text-[0.95rem] font-semibold ${tone || "text-ink"}`}>{value}</div>
      {sub && <div className={`text-[0.66rem] ${tone || "text-ink-2"}`}>{sub}</div>}
    </div>
  );
}

function Placeholder({ children }: { children: React.ReactNode }) {
  return <div className="rounded-card border border-dashed border-hairline-md p-12 text-center text-ink-2">{children}</div>;
}
