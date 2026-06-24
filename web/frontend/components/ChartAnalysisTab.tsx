"use client";

import { useEffect, useState } from "react";
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

  // 뉴스·펀더멘털을 백그라운드로 미리 수집(prefetch) → 워커가 market_cache 를 채워두면
  // 사용자가 뉴스/펀더멘털 탭으로 이동하기 전에 결과가 준비되어 즉시 표시된다.
  // 결과는 사용하지 않는다(캐시 워밍 목적). 캐시가 신선하면 라우트가 큐 없이 즉시응답하므로
  // 뉴스(1일)·펀더멘털(분기) 캐시가 살아있는 동안 prefetch 는 워커 부하를 거의 만들지 않는다.
  useEffect(() => {
    if (!picked) return;
    const opts = {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "same-origin" as const,
      body: JSON.stringify({ ticker: picked.ticker }),
    };
    // 화면용 AI 분석(AnalysisPanel)이 먼저 큐에 들어가 워커를 잡도록 prefetch 는 1.2s 지연.
    // 워커 클레임 우선순위도 analysis 를 앞세우지만, 동시 요청 버스트를 줄이는 이중 안전장치.
    const id = setTimeout(() => {
      fetch("/api/v1/news/run", opts).catch(() => {});
      fetch("/api/v1/fundamental/run", opts).catch(() => {});
    }, 1200);
    return () => clearTimeout(id);
  }, [picked?.ticker]);

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
          help="현재 거래되는 주가와 전일 종가 대비 등락률입니다. 모든 기술지표 계산의 기준이 되는 값이에요."
        />
        <Metric
          label="RSI(14)"
          value={fmt(ind.rsi)}
          tone={ind.rsi == null ? "" : ind.rsi < 30 ? "text-gain" : ind.rsi > 70 ? "text-loss" : ""}
          sub={ind.rsi == null ? undefined : ind.rsi < 30 ? "과매도" : ind.rsi > 70 ? "과매수" : "중립"}
          help="상대강도지수(0~100). 최근 14일간 상승·하락 폭을 비교해 과열/침체를 봅니다. 70 이상이면 과매수(단기 고점 부담), 30 이하면 과매도(단기 반등 기대) 구간으로 봅니다."
        />
        <Metric
          label="MACD Hist"
          value={fmt(ind.macdHist, 3)}
          tone={ind.macdHist == null ? "" : signClass(ind.macdHist)}
          sub={ind.macdHist == null ? undefined : ind.macdHist >= 0 ? "상승 모멘텀" : "하락 모멘텀"}
          help="MACD선과 시그널선의 차이(히스토그램)입니다. 0보다 크고 커질수록 상승 모멘텀이 강해지고, 0보다 작아질수록 하락 모멘텀이 강해집니다. 부호가 바뀌는 지점이 추세 전환 신호로 쓰입니다."
        />
        <Metric
          label="ADX(14)"
          value={fmt(ind.adx)}
          tone={ind.adx != null && ind.adx > 25 ? "text-accent" : ""}
          sub={ind.adx == null ? undefined : ind.adx > 25 ? "추세 강함" : "추세 약함"}
          help="추세의 '강도'를 나타내는 지표(0~100). 방향(상승·하락)이 아니라 추세가 얼마나 뚜렷한지를 봅니다. 보통 25 이상이면 추세가 강해 추세추종 전략이 유리하고, 그 이하면 횡보 구간으로 봅니다."
        />
        <Metric
          label="볼린저 %B"
          value={ind.bbPct == null ? "—" : `${(ind.bbPct * 100).toFixed(0)}%`}
          tone={ind.bbPct == null ? "" : ind.bbPct > 0.8 ? "text-loss" : ind.bbPct < 0.2 ? "text-gain" : ""}
          sub={ind.bbPct == null ? undefined : ind.bbPct > 0.8 ? "상단 과열" : ind.bbPct < 0.2 ? "하단 과매도" : "밴드 내"}
          help="현재가가 볼린저밴드 내 어디에 있는지를 0~100%로 표시합니다. 100%는 상단 밴드(과열), 0%는 하단 밴드(과매도), 50%는 중심선(20일 이동평균)입니다. 밴드를 벗어나면 변동성 확대 신호로 봅니다."
        />
        <Metric
          label="스토캐스틱 %K/%D"
          value={`${fmt(ind.stochK)} / ${fmt(ind.stochD)}`}
          tone={ind.stochK == null ? "" : ind.stochK < 20 ? "text-gain" : ind.stochK > 80 ? "text-loss" : ""}
          help="최근 가격이 일정 기간의 고저 범위 중 어디에 있는지를 0~100으로 봅니다. %K는 빠른 선, %D는 %K를 평활한 느린 선입니다. 80 이상은 과매수, 20 이하는 과매도이며, %K가 %D를 상향 돌파하면 매수 신호로 해석합니다."
        />
        <Metric
          label="MFI(14)"
          value={fmt(ind.mfi)}
          tone={ind.mfi == null ? "" : ind.mfi < 20 ? "text-gain" : ind.mfi > 80 ? "text-loss" : ""}
          sub={ind.mfi == null ? undefined : ind.mfi < 20 ? "유입 약" : ind.mfi > 80 ? "유입 과열" : undefined}
          help="자금흐름지수(0~100). RSI에 '거래량'을 더한 지표로, 돈이 실제로 들어오고 나가는지를 봅니다. 80 이상은 자금 유입 과열, 20 이하는 유입이 약해진 과매도 구간으로 해석합니다."
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
  return (
    <Metric
      label="이평선 추세"
      value={label}
      tone={tone}
      sub={`현재가 > 이평 ${aboveCount}/3`}
      help="단기(20)·중기(50)·장기(200) 이동평균의 배열 상태입니다. 현재가>20>50>200 순이면 '정배열(상승추세)', 그 반대면 '역배열(하락추세)'입니다. 현재가가 위에 있는 이평선이 많을수록 추세가 견고합니다."
    />
  );
}

/* ── 보조 지표 (접힘) ─────────────────────────────────────────────── */
function SecondaryIndicators({ ind }: { ind: IndicatorSnapshot }) {
  return (
    <details className="rounded-card border border-hairline bg-surface p-4 shadow-card">
      <summary className="cursor-pointer text-sm font-semibold text-ink">🔧 보조 · 참고 지표</summary>
      <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-4">
        <Metric label="EMA 20" value={fmt(ind.ema20, 2)} help="최근 20일 지수이동평균. 최근 가격에 더 큰 가중치를 둬 단기 추세를 빠르게 반영합니다. 현재가가 이 선 위에 있으면 단기 상승 흐름입니다." />
        <Metric label="EMA 50" value={fmt(ind.ema50, 2)} help="최근 50일 지수이동평균. 중기 추세의 기준선으로, 현재가·단기 이평선이 이 선 위/아래에 있는지로 추세 방향을 판단합니다." />
        <Metric label="EMA 200" value={fmt(ind.ema200, 2)} help="최근 200일 지수이동평균. 장기 추세의 핵심 기준선입니다. 현재가가 이 선 위면 장기 상승장, 아래면 장기 하락장으로 봅니다." />
        <Metric label="SMA 20" value={fmt(ind.sma20, 2)} help="최근 20일 단순이동평균(20일 종가의 평균). 볼린저밴드의 중심선으로도 쓰이며, 기간 내 모든 가격을 동일 가중치로 평균냅니다." />
        <Metric label="CCI(20)" value={fmt(ind.cci, 0)} tone={ind.cci == null ? "" : ind.cci < -100 ? "text-gain" : ind.cci > 100 ? "text-loss" : ""} help="상품채널지수. 가격이 평균에서 얼마나 벗어났는지를 봅니다. +100 이상은 과열, -100 이하는 과매도 구간으로 해석합니다." />
        <Metric label="Williams %R" value={fmt(ind.williamsR)} help="윌리엄스 %R(-100~0). 스토캐스틱과 비슷하게 과매수·과매도를 봅니다. -20 이상은 과매수, -80 이하는 과매도 구간입니다." />
        <Metric label="ROC(12)" value={ind.roc == null ? "—" : `${ind.roc.toFixed(1)}%`} tone={ind.roc == null ? "" : signClass(ind.roc)} help="변화율(Rate of Change). 현재가가 12일 전 대비 몇 % 올랐는지를 나타내는 모멘텀 지표입니다. 0보다 크면 상승 모멘텀입니다." />
        <Metric label="Z-Score" value={fmt(ind.zscore, 2)} tone={ind.zscore == null ? "" : ind.zscore < -1.5 ? "text-gain" : ind.zscore > 1.5 ? "text-loss" : ""} help="현재가가 평균에서 표준편차의 몇 배만큼 떨어져 있는지를 봅니다. +1.5 이상은 통계적으로 비싼(과열), -1.5 이하는 싼(과매도) 상태로 평균 회귀를 기대할 수 있습니다." />
        <Metric label="ATR(14)" value={fmt(ind.atr, 2)} help="평균진폭(Average True Range). 최근 14일의 하루 변동폭 평균으로 '변동성' 크기를 나타냅니다. 손절·목표가 폭을 정할 때 기준으로 씁니다(값이 클수록 변동성 큼)." />
        <Metric label="월 VWAP" value={fmt(ind.vwapM, 2)} help="거래량가중평균가격(월간). 한 달간 거래량을 가중치로 둔 평균 매매단가입니다. 현재가가 이 값보다 높으면 평균 매수자들이 이익 구간에 있다는 의미입니다." />
        <Metric label="OBV" value={fmt(ind.obv, 0)} help="누적거래량(On-Balance Volume). 상승일 거래량은 더하고 하락일은 빼서 누적합니다. 가격보다 OBV가 먼저 움직이면 자금 유출입의 선행 신호로 봅니다." />
        <Metric label="MACD / Signal" value={`${fmt(ind.macd, 2)} / ${fmt(ind.macdSignal, 2)}`} help="MACD선(단기·장기 이평선 차이)과 그 신호선(MACD의 9일 이평)입니다. MACD가 시그널을 상향 돌파하면 매수, 하향 돌파하면 매도 신호로 해석합니다." />
      </div>
    </details>
  );
}

/* ── 공용 프리미티브 ───────────────────────────────────────────────── */
function Metric({
  label,
  value,
  tone = "",
  sub,
  help,
}: {
  label: string;
  value: string;
  tone?: string;
  sub?: string;
  help?: string;
}) {
  const [open, setOpen] = useState(false);
  return (
    <div className="relative rounded-lg border border-hairline bg-surface p-3 shadow-card">
      <div className="flex items-start justify-between gap-1">
        <div className="text-[0.68rem] text-ink-2">{label}</div>
        {help && (
          <button
            type="button"
            aria-label={`${label} 설명`}
            aria-expanded={open}
            onClick={() => setOpen((v) => !v)}
            onBlur={() => setOpen(false)}
            className="-mr-1 -mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center rounded-full border border-hairline text-[0.6rem] font-bold leading-none text-ink-2 transition-colors hover:bg-canvas hover:text-ink"
          >
            ?
          </button>
        )}
      </div>
      <div className={`tnum text-[0.95rem] font-semibold ${tone || "text-ink"}`}>{value}</div>
      {sub && <div className={`text-[0.66rem] ${tone || "text-ink-2"}`}>{sub}</div>}
      {help && open && (
        <div className="absolute right-2 top-8 z-20 w-56 rounded-lg border border-hairline bg-surface p-2.5 text-[0.7rem] leading-relaxed text-ink-2 shadow-card">
          {help}
        </div>
      )}
    </div>
  );
}

function Placeholder({ children }: { children: React.ReactNode }) {
  return <div className="rounded-card border border-dashed border-hairline-md p-12 text-center text-ink-2">{children}</div>;
}
