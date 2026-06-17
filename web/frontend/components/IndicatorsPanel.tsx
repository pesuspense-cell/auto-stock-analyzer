"use client";

import useSWR from "swr";

import { signClass } from "@/lib/format";
import { pollingIntervalMs, marketState, marketStateLabel } from "@/lib/market-hours";
import type { IndicatorsResponse, StockHit } from "@/lib/api-types";

const jfetch = async (url: string) => {
  const res = await fetch(url, { credentials: "same-origin" });
  if (!res.ok) throw new Error((await res.json().catch(() => ({})))?.error ?? res.statusText);
  return res.json();
};

const fmt = (v: number | null, d = 1) => (v == null ? "—" : v.toLocaleString("ko-KR", { minimumFractionDigits: d, maximumFractionDigits: d }));

/**
 * 기술지표 대시보드 — stock_ai 의 12모듈 신호 + 지표를 TS 포팅으로 계산.
 * KST 장 마감/주말 기준 Polling 자동 연장·정지(market-hours)를 SWR refreshInterval 에 적용.
 */
export function IndicatorsPanel({ picked }: { picked: StockHit | null }) {
  const { data, error, isLoading } = useSWR<IndicatorsResponse>(
    picked ? `/api/v1/indicators/${encodeURIComponent(picked.ticker)}` : null,
    jfetch,
    {
      // ⬇️ 트래픽 방어: 장중 15초 / 장외 5분 / 주말 0(정지). 함수형이라 매 틱마다 재평가됨.
      refreshInterval: () => (picked ? pollingIntervalMs(picked.ticker) : 0),
      revalidateOnFocus: false,
      keepPreviousData: true,
    }
  );

  if (!picked) return <Placeholder>종목을 선택하면 기술지표 분석이 표시됩니다.</Placeholder>;
  if (isLoading && !data) return <Placeholder>📐 지표 계산 중…</Placeholder>;
  if (error || !data) return <Placeholder>지표를 불러오지 못했습니다.</Placeholder>;

  const ind = data.indicators;
  const sig = data.signal;
  const state = marketState(picked.ticker);

  return (
    <div className="space-y-4">
      {/* 헤더 */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h2 className="text-lg font-bold text-ink">{picked.name} <span className="tnum text-sm text-ink-2">{picked.ticker}</span></h2>
          <p className="text-xs text-ink-2">기준일 {data.asOf} · {data.cached ? "캐시" : "실시간 계산"}</p>
        </div>
        <span className="rounded-full bg-canvas px-3 py-1 text-xs font-medium text-ink-2">
          {marketStateLabel(state)}
        </span>
      </div>

      {/* 종합 신호 카드 */}
      <SignalGauge sig={sig} />

      {/* 지표 그리드 — 모바일 2열 / 태블릿 3열 / 데스크톱 4열 */}
      <section className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-4">
        <Metric label="RSI(14)" value={fmt(ind.rsi)} tone={ind.rsi == null ? "" : ind.rsi < 30 ? "text-gain" : ind.rsi > 70 ? "text-loss" : ""} />
        <Metric label="MACD Hist" value={fmt(ind.macdHist, 3)} tone={ind.macdHist == null ? "" : signClass(ind.macdHist)} />
        <Metric label="Stoch %K / %D" value={`${fmt(ind.stochK)} / ${fmt(ind.stochD)}`} />
        <Metric label="ADX(14)" value={fmt(ind.adx)} tone={ind.adx != null && ind.adx > 25 ? "text-accent" : ""} />
        <Metric label="EMA 20" value={fmt(ind.ema20, 2)} />
        <Metric label="EMA 50" value={fmt(ind.ema50, 2)} />
        <Metric label="EMA 200" value={fmt(ind.ema200, 2)} />
        <Metric label="볼린저 %B" value={ind.bbPct == null ? "—" : `${(ind.bbPct * 100).toFixed(0)}%`} tone={ind.bbPct == null ? "" : ind.bbPct > 0.7 ? "text-loss" : ind.bbPct < 0.3 ? "text-gain" : ""} />
        <Metric label="CCI(20)" value={fmt(ind.cci, 0)} tone={ind.cci == null ? "" : ind.cci < -100 ? "text-gain" : ind.cci > 100 ? "text-loss" : ""} />
        <Metric label="Williams %R" value={fmt(ind.williamsR)} />
        <Metric label="ROC(12)" value={ind.roc == null ? "—" : `${ind.roc.toFixed(1)}%`} tone={ind.roc == null ? "" : signClass(ind.roc)} />
        <Metric label="MFI(14)" value={fmt(ind.mfi)} tone={ind.mfi == null ? "" : ind.mfi < 20 ? "text-gain" : ind.mfi > 80 ? "text-loss" : ""} />
        <Metric label="Z-Score" value={fmt(ind.zscore, 2)} tone={ind.zscore == null ? "" : ind.zscore < -1.5 ? "text-gain" : ind.zscore > 1.5 ? "text-loss" : ""} />
        <Metric label="ATR(14)" value={fmt(ind.atr, 2)} />
        <Metric label="월 VWAP" value={fmt(ind.vwapM, 2)} />
        <Metric label="현재가" value={fmt(ind.price, 2)} tone={signClass(ind.changePct)} sub={`${ind.changePct >= 0 ? "+" : ""}${ind.changePct.toFixed(2)}%`} />
      </section>

      {/* 신호 근거 */}
      {sig.reasons.length > 0 && (
        <details className="rounded-card border border-hairline bg-surface p-4 shadow-card" open>
          <summary className="cursor-pointer text-sm font-semibold text-ink">🔎 신호 근거 ({sig.reasons.length})</summary>
          <ul className="mt-2 space-y-1 text-[0.82rem] text-ink-2">
            {sig.reasons.map((r, i) => <li key={i}>• {r}</li>)}
          </ul>
        </details>
      )}
    </div>
  );
}

function SignalGauge({ sig }: { sig: IndicatorsResponse["signal"] }) {
  // -10..+10 → 0..100%
  const pct = ((sig.score + 10) / 20) * 100;
  const pos = sig.score >= 0;
  return (
    <section className="rounded-card border border-term-border bg-term-1 p-5 shadow-elevated">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="text-3xl">{sig.badge}</span>
          <div>
            <div className="text-lg font-bold text-term-ink">{sig.label}</div>
            <div className="text-xs text-term-muted">12모듈 종합 점수</div>
          </div>
        </div>
        <div className={`tnum text-3xl font-bold ${pos ? "text-gain" : "text-loss"}`}>
          {sig.score >= 0 ? "+" : ""}{sig.score.toFixed(1)}
        </div>
      </div>
      {/* 게이지 (-10 ··· 0 ··· +10) */}
      <div className="relative mt-4 h-2 w-full rounded-full bg-term-border">
        <div className="absolute left-1/2 top-[-3px] h-[14px] w-px bg-term-muted/60" />
        <div
          className={`absolute top-0 h-2 rounded-full ${pos ? "bg-gain" : "bg-loss"}`}
          style={pos ? { left: "50%", width: `${pct - 50}%` } : { left: `${pct}%`, width: `${50 - pct}%` }}
        />
      </div>
      <div className="mt-1 flex justify-between text-[0.62rem] text-term-muted">
        <span>강력 매도 -10</span><span>중립 0</span><span>+10 강력 매수</span>
      </div>
    </section>
  );
}

function Metric({ label, value, tone = "", sub }: { label: string; value: string; tone?: string; sub?: string }) {
  return (
    <div className="rounded-lg border border-hairline bg-surface p-3 shadow-card">
      <div className="text-[0.68rem] text-ink-2">{label}</div>
      <div className={`tnum text-[0.95rem] font-semibold ${tone || "text-ink"}`}>{value}</div>
      {sub && <div className={`tnum text-[0.68rem] ${tone}`}>{sub}</div>}
    </div>
  );
}

function Placeholder({ children }: { children: React.ReactNode }) {
  return <div className="rounded-card border border-dashed border-hairline-md p-12 text-center text-ink-2">{children}</div>;
}
