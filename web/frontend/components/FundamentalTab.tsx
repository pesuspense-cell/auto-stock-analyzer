"use client";

import { useEffect } from "react";
import { useJob } from "@/hooks/useJob";
import { fmtNum, signClass } from "@/lib/format";
import { InfoTip } from "./InfoTip";
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

// ── 값 포맷터 ───────────────────────────────────────────────────────
const N = (v: any) => (v == null || v === "" || Number.isNaN(Number(v)) ? null : Number(v));
const mult = (v: any, d = 2) => { const n = N(v); return n == null ? "—" : `${n.toFixed(d)}x`; };
const num = (v: any, d = 2) => { const n = N(v); return n == null ? "—" : n.toFixed(d); };
const pctFrac = (v: any, d = 1) => { const n = N(v); return n == null ? "—" : `${(n * 100).toFixed(d)}%`; }; // 분수→%
const pctRaw = (v: any, d = 1) => { const n = N(v); return n == null ? "—" : `${n.toFixed(d)}%`; };           // 이미 %
// 배당/주주환원수익률 — yfinance 분수(0.025)와 KRX/신버전 퍼센트(2.5)가 섞여 단위 모호 → 휴리스틱.
const yld = (v: any, d = 2) => { const n = N(v); return n == null ? "—" : `${(Math.abs(n) <= 1 ? n * 100 : n).toFixed(d)}%`; };
const capW = (v: any) => { const n = N(v); return n == null ? "—" : `${(n / 1e8).toLocaleString(undefined, { maximumFractionDigits: 0 })}억`; };

interface Row { label: string; value: string; tone?: string; help: string }

/** fund_info(fi) + fund_score_data(fsd) → 기업가치 판단용 지표 그룹. */
function buildGroups(fi: Record<string, any>, fsd: Record<string, any>): { title: string; rows: Row[] }[] {
  const peg = N(fsd.peg);
  const pegTone = peg == null ? "" : peg < 1 ? "text-gain" : peg > 2 ? "text-loss" : "";
  const growthTone = (v: any) => { const n = N(v); return n == null ? "" : signClass(n); };

  const valuation: Row[] = [
    { label: "PER", value: mult(fi.per), help: "주가수익비율. 주가를 주당순이익(EPS)으로 나눈 값으로, 이익 1원에 시장이 매기는 가격입니다. 낮을수록 이익 대비 저평가입니다." },
    { label: "PBR", value: mult(fi.pbr), help: "주가순자산비율. 주가를 주당순자산(BPS)으로 나눈 값입니다. 1배 미만이면 장부상 순자산보다 싸게 거래됩니다." },
    { label: "PSR", value: mult(fi.psr), help: "주가매출비율. 시가총액을 매출로 나눈 값으로, 적자·성장주 평가에 씁니다. 낮을수록 매출 대비 저평가입니다." },
    { label: "선행 PER", value: mult(fi.forward_pe), help: "향후 12개월 추정이익 기준 PER입니다. 현재 PER보다 낮으면 이익 성장이 기대된다는 뜻입니다." },
    { label: "PEG", value: num(peg), tone: pegTone, help: "PER을 이익성장률로 나눈 값(피터 린치 지표). 1 미만이면 성장 대비 저평가, 2 초과면 고평가로 봅니다." },
    { label: "시가총액", value: capW(fi.market_cap), help: "발행주식 전체의 시장가치(주가×주식수)로 기업 규모를 나타냅니다." },
  ];
  const profit: Row[] = [
    { label: "ROE", value: pctFrac(fi.roe), help: "자기자본이익률(순이익/자기자본). 주주 자본으로 얼마를 벌었는지로, 통상 15% 이상을 우량으로 봅니다." },
    { label: "ROE 3년평균", value: pctRaw(fsd.roe_mean), help: "최근 3년 ROE 평균입니다. 평균이 높고 편차가 작을수록(버핏 기준) 꾸준한 수익 창출력을 의미합니다." },
    { label: "영업이익률", value: pctFrac(fi.operating_margins), help: "매출 대비 영업이익 비율입니다. 본업의 수익성과 가격 결정력(해자)을 나타냅니다." },
    { label: "이익의 질", value: num(fsd.ocf_ni_ratio), help: "영업현금흐름÷순이익입니다. 1 이상이면 장부 이익이 실제 현금으로 잘 들어온다는 뜻으로 이익의 질이 좋습니다." },
  ];
  const growth: Row[] = [
    { label: "매출 성장률", value: pctFrac(fi.revenue_growth), tone: growthTone(fi.revenue_growth), help: "전년 대비 매출 증가율로 외형 성장 속도를 봅니다." },
    { label: "이익 성장률", value: pctFrac(fi.earnings_growth), tone: growthTone(fi.earnings_growth), help: "전년 대비 순이익 증가율입니다." },
    { label: "EPS 3년 CAGR", value: pctRaw(fsd.eps_cagr_3yr), tone: growthTone(fsd.eps_cagr_3yr), help: "주당순이익(EPS)의 최근 3년 연평균 성장률로, 실제 주주 몫 이익의 성장 추세를 봅니다." },
    ...(N(fsd.dart_backlog_ratio) != null
      ? [{ label: "수주잔고/매출", value: mult(fsd.dart_backlog_ratio, 2), help: "DART 공시 수주잔고를 연매출로 나눈 값입니다. 향후 매출로 잡힐 일감이 몇 년치인지(수주 가시성)를 나타냅니다." } as Row]
      : []),
  ];
  const stable: Row[] = [
    { label: "부채비율", value: pctRaw(fi.debt_equity, 0), help: "자기자본 대비 총부채(부채/자기자본)입니다. 낮을수록 재무가 안정적이며 통상 100% 이하를 양호하게 봅니다." },
    { label: "FCF 수익률", value: pctRaw(fsd.fcf_yield), tone: growthTone(fsd.fcf_yield), help: "잉여현금흐름(FCF)÷시가총액입니다. 높을수록 벌어들인 현금 대비 주가가 싸다는 의미입니다." },
    { label: "배당수익률", value: yld(fi.div_yield), help: "주가 대비 연간 배당금 비율입니다." },
    { label: "주주환원수익률", value: yld(fsd.shareholder_yield ?? fi.shareholder_yield), help: "배당+자사주 매입을 합한 주주환원을 시가총액으로 나눈 값으로, 배당만 보는 것보다 폭넓게 봅니다." },
  ];

  return [
    { title: "밸류에이션", rows: valuation },
    { title: "수익성", rows: profit },
    { title: "성장성", rows: growth },
    { title: "재무안정성·주주환원", rows: stable },
  ].filter((g) => g.rows.some((r) => r.value !== "—")); // 전부 결측인 그룹은 숨김
}

// 가치투자 마스터 판정 → 색.
function verdictTone(v: string): string {
  if (/통과|추천|추세확인/.test(v)) return "text-gain";
  if (/경고|미달/.test(v)) return "text-loss";
  if (/주의|과열|관망|조심|부분/.test(v)) return "text-amber-500";
  return "text-ink-2";
}

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
  const groups = data.is_etf ? [] : buildGroups(data.fund_info, fsd ?? {});
  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
      <div className="space-y-4 lg:col-span-2">
        {!data.is_etf && (
          <section className="rounded-card border border-hairline bg-surface p-5 shadow-card">
            <div className="mb-4 flex items-center justify-between">
              <h3 className="text-sm font-semibold text-ink">📊 핵심 지표</h3>
              {fsd?.fund_label && (
                <span className="rounded-full bg-accent/10 px-3 py-1 text-xs font-bold text-accent">
                  {fsd.fund_label} {fsd.fund_score != null && `(${fsd.fund_score >= 0 ? "+" : ""}${fsd.fund_score})`}
                </span>
              )}
            </div>
            <div className="space-y-4">
              {groups.map((g) => (
                <MetricGroup key={g.title} title={g.title} rows={g.rows} />
              ))}
            </div>
          </section>
        )}

        {!data.is_etf && <MasterVerdicts verdicts={fsd?.master_verdicts} />}

        {data.is_etf && <EtfCard data={data} />}

        <InvestorCard investors={data.investors} />
      </div>

      <AiReportCard ticker={data.ticker} />
    </div>
  );
}

/** 한 카테고리(밸류에이션 등)의 지표 타일 묶음. */
function MetricGroup({ title, rows }: { title: string; rows: Row[] }) {
  return (
    <div>
      <h4 className="mb-2 text-[0.72rem] font-semibold text-ink-2">{title}</h4>
      <dl className="grid grid-cols-2 gap-x-6 gap-y-3 md:grid-cols-4">
        {rows.map((r) => (
          <div key={r.label} className="flex flex-col">
            <dt className="flex items-center gap-1 text-[0.72rem] text-ink-2">
              {r.label}
              <InfoTip text={r.help} label={r.label} variant="surface" />
            </dt>
            <dd className={`tnum text-[0.95rem] font-semibold ${r.tone || "text-ink"}`}>{r.value}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

/** 가치투자 마스터 진단(그레이엄·버핏·린치·오닐) — fund_score_data.master_verdicts. */
function MasterVerdicts({ verdicts }: { verdicts?: Record<string, { icon?: string; 판정?: string; comment?: string }> }) {
  const ORDER = ["그레이엄", "버핏", "린치", "오닐"];
  const SUB: Record<string, string> = {
    그레이엄: "안전마진 (PER×PBR)",
    버핏: "수익성·해자 (ROE 지속성)",
    린치: "성장 대비 가격 (PEG)",
    오닐: "추세·신고가 모멘텀",
  };
  const entries = verdicts ? ORDER.filter((k) => verdicts[k]).map((k) => [k, verdicts[k]] as const) : [];
  if (entries.length === 0) return null;
  return (
    <section className="rounded-card border border-hairline bg-surface p-5 shadow-card">
      <h3 className="mb-3 text-sm font-semibold text-ink">🎓 가치투자 마스터 진단</h3>
      <div className="space-y-2">
        {entries.map(([name, v]) => (
          <div key={name} className="flex items-start gap-3 rounded-lg bg-canvas p-3">
            <span className="text-lg leading-none">{v.icon ?? "—"}</span>
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2">
                <span className="text-[0.82rem] font-semibold text-ink">{name}</span>
                <span className="text-[0.64rem] text-ink-3">{SUB[name]}</span>
                {v.판정 && (
                  <span className={`ml-auto shrink-0 text-[0.72rem] font-bold ${verdictTone(v.판정)}`}>{v.판정}</span>
                )}
              </div>
              {v.comment && <p className="mt-1 text-[0.76rem] leading-relaxed text-ink-2">{v.comment}</p>}
            </div>
          </div>
        ))}
      </div>
    </section>
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
