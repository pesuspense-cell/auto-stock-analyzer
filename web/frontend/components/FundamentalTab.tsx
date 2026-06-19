"use client";

import { useEffect } from "react";
import { useJob } from "@/hooks/useJob";
import { fmtNum, signClass } from "@/lib/format";
import type { StockHit } from "@/lib/api-types";

interface FundamentalResponse {
  ticker: string;
  is_etf: boolean;
  fund_info: Record<string, any>;
  fund_score_data: Record<string, any>;
  etf_data: Record<string, any>;
  etf_score: Record<string, any>;
  investors: Record<string, any>;
  investor_history: Record<string, any>[];
  insiders: Record<string, any>[];
}
interface QuickAssessment {
  score10: number;
  verdict: string;
  reasons: string[];
  summary: string;
  has_fund: boolean;
}
interface AiReport {
  ok: boolean;
  report: string;
  provider: string;
  error: string;
  quick_assessment: QuickAssessment;
}

const METRICS: [string, string, (v: any) => string][] = [
  ["시가총액", "market_cap", (v) => (v ? `${(v / 1e8).toLocaleString()}억` : "—")],
  ["PER", "per", (v) => (v ? `${Number(v).toFixed(2)}x` : "—")],
  ["PBR", "pbr", (v) => (v ? `${Number(v).toFixed(2)}x` : "—")],
  ["ROE", "roe", (v) => (v != null ? `${(Number(v) * 100).toFixed(1)}%` : "—")],
  ["영업이익률", "operating_margins", (v) => (v != null ? `${(Number(v) * 100).toFixed(1)}%` : "—")],
  ["배당수익률", "dividend_yield", (v) => (v != null ? `${(Number(v) * 100).toFixed(2)}%` : "—")],
];

/** 펀더멘털 & 기관 탭 — POST /fundamental/run 으로 jobs 큐 적재 후 폴링(워커 interactive 레인). */
export function FundamentalTab({ picked }: { picked: StockHit | null }) {
  const { result: data, busy, status, error, enqueue, reset } = useJob<FundamentalResponse>();

  useEffect(() => {
    if (picked) enqueue("/api/v1/fundamental/run", { ticker: picked.ticker });
    else reset();
  }, [picked?.ticker, enqueue, reset]);

  if (!picked) return <Placeholder>종목을 선택하면 펀더멘털 분석이 표시됩니다.</Placeholder>;
  if (busy) return <Placeholder>{status === "pending" ? "🏛️ 펀더멘털 작업 대기 중…" : "🏛️ 재무·수급 데이터 수집 중…"}</Placeholder>;
  if (error) return <Placeholder>데이터를 불러오지 못했습니다. ({error})</Placeholder>;
  if (!data) return <Placeholder>데이터를 불러오지 못했습니다.</Placeholder>;

  const fsd = data.fund_score_data;
  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
      <div className="space-y-4 lg:col-span-2">
        {!data.is_etf && (
          <section className="rounded-card border border-hairline bg-surface p-5 shadow-card">
            <div className="mb-3 flex items-center justify-between">
              <h3 className="text-sm font-semibold text-ink">📊 핵심 지표</h3>
              {fsd?.fund_label && (
                <span className="rounded-full bg-accent/10 px-3 py-1 text-xs font-bold text-accent">
                  {fsd.fund_label} {fsd.fund_score != null && `(${fsd.fund_score >= 0 ? "+" : ""}${fsd.fund_score})`}
                </span>
              )}
            </div>
            <dl className="grid grid-cols-2 gap-x-6 gap-y-3 md:grid-cols-3">
              {METRICS.map(([label, key, fmt]) => (
                <div key={key} className="flex flex-col">
                  <dt className="text-[0.72rem] text-ink-2">{label}</dt>
                  <dd className="tnum text-[0.95rem] font-semibold text-ink">{fmt(data.fund_info[key])}</dd>
                </div>
              ))}
            </dl>
          </section>
        )}

        {data.is_etf && <EtfCard data={data} />}

        <InvestorCard investors={data.investors} />
      </div>

      <AiReportCard ticker={data.ticker} />
    </div>
  );
}

function EtfCard({ data }: { data: FundamentalResponse }) {
  const e = data.etf_data;
  const s = data.etf_score;
  return (
    <section className="rounded-card border border-hairline bg-surface p-5 shadow-card">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-ink">📊 ETF 지표</h3>
        {s?.etf_label && (
          <span className="rounded-full bg-accent/10 px-3 py-1 text-xs font-bold text-accent">{s.etf_label}</span>
        )}
      </div>
      <dl className="grid grid-cols-2 gap-x-6 gap-y-3 md:grid-cols-3">
        {[
          ["NAV", e.nav],
          ["괴리율", e.premium_discount != null ? `${Number(e.premium_discount).toFixed(2)}%` : "—"],
          ["운용보수", e.expense_ratio != null ? `${Number(e.expense_ratio).toFixed(2)}%` : "—"],
          ["추적오차", e.tracking_error != null ? `${Number(e.tracking_error).toFixed(2)}%` : "—"],
          ["AUM", e.aum],
          ["배당수익률", e.dividend_yield != null ? `${Number(e.dividend_yield).toFixed(2)}%` : "—"],
        ].map(([label, val]) => (
          <div key={String(label)} className="flex flex-col">
            <dt className="text-[0.72rem] text-ink-2">{label}</dt>
            <dd className="tnum text-[0.95rem] font-semibold text-ink">{val == null ? "—" : String(val)}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}

function InvestorCard({ investors }: { investors: Record<string, any> }) {
  const rows: [string, number | null][] = [
    ["외국인", investors["외국인"] ?? null],
    ["기관", investors["기관합계"] ?? null],
    ["개인", investors["개인"] ?? null],
  ];
  const has = rows.some(([, v]) => v != null);
  return (
    <section className="rounded-card border border-hairline bg-surface p-5 shadow-card">
      <h3 className="mb-3 text-sm font-semibold text-ink">
        🏦 투자자별 매매동향 {investors.date && <span className="text-xs text-ink-2">({investors.date})</span>}
      </h3>
      {!has ? (
        <p className="text-sm text-ink-2">수급 데이터 없음 (국내 종목 전용)</p>
      ) : (
        <div className="grid grid-cols-3 gap-3">
          {rows.map(([label, v]) => (
            <div key={label} className="rounded-lg bg-canvas p-3 text-center">
              <div className="text-[0.72rem] text-ink-2">{label}</div>
              <div className={`tnum text-[0.95rem] font-bold ${v != null ? signClass(v) : "text-ink-2"}`}>
                {v != null ? `${v >= 0 ? "+" : ""}${fmtNum(v)}` : "—"}
              </div>
              <div className="text-[0.62rem] text-ink-3">주</div>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

function AiReportCard({ ticker }: { ticker: string }) {
  const { result: data, busy, error, enqueue, reset } = useJob<AiReport>();

  // 종목 변경 시 이전 리포트/폴링을 비운다.
  useEffect(() => {
    reset();
  }, [ticker, reset]);

  function run() {
    enqueue("/api/v1/fundamental/ai-report", { ticker, use_llm: true });
  }

  const q = data?.quick_assessment;
  const errMsg = error ?? data?.error;
  const vColor =
    q?.verdict === "매수" ? "text-gain" : q?.verdict === "매도" ? "text-loss" : "text-ink-2";

  return (
    <aside className="rounded-card border border-term-border bg-term-1 p-5 shadow-elevated">
      <h3 className="mb-3 text-sm font-semibold text-term-ink">🤖 AI 심층 재무분석</h3>
      {q && (
        <div className="mb-4 rounded-lg border border-term-border bg-black/20 p-4">
          <div className="flex items-baseline gap-2">
            <span className={`text-2xl font-bold ${vColor}`}>{q.verdict}</span>
            <span className="tnum text-sm text-term-muted">투자점수 {q.score10}/10</span>
          </div>
          <p className="mt-2 text-[0.82rem] text-term-muted">{q.summary}</p>
          {q.reasons.length > 0 && (
            <ul className="mt-2 space-y-0.5 text-[0.8rem] text-term-muted">
              {q.reasons.map((r, i) => (
                <li key={i}>• {r}</li>
              ))}
            </ul>
          )}
        </div>
      )}

      <button
        onClick={run}
        disabled={busy}
        className="w-full rounded-lg bg-accent px-4 py-2 text-sm font-semibold text-white hover:brightness-110 disabled:opacity-40"
      >
        {busy ? "리포트 생성 중…" : data ? "리포트 다시 생성" : "AI 리포트 생성"}
      </button>

      {errMsg && <p className="mt-3 text-[0.82rem] text-loss">{errMsg}</p>}
      {data?.report && (
        <article className="prose-sm mt-4 max-h-[60vh] overflow-auto whitespace-pre-wrap text-[0.82rem] leading-relaxed text-term-ink">
          {data.report}
          <p className="mt-3 text-[0.7rem] text-term-muted">— {data.provider}</p>
        </article>
      )}
    </aside>
  );
}

function Placeholder({ children }: { children: React.ReactNode }) {
  return (
    <div className="rounded-card border border-dashed border-hairline-md p-12 text-center text-ink-2">
      {children}
    </div>
  );
}
