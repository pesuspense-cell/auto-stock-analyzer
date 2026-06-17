"use client";

import { useState } from "react";

import { AuthGate, UserMenu } from "@/components/AuthGate";
import { useJob } from "@/hooks/useJob";
import { JobStatusBanner } from "@/components/JobStatusBanner";
import { fmtNum, signClass } from "@/lib/format";
import type { BacktestJobResult } from "@/lib/api-types";

const MARKETS = ["KOSPI", "KOSDAQ", "S&P500", "NASDAQ"];
const BENCHMARKS: Record<string, string | null> = {
  "^KS11 (KOSPI)": "^KS11",
  "^KQ11 (KOSDAQ)": "^KQ11",
  "^IXIC (NASDAQ)": "^IXIC",
  "^GSPC (S&P500)": "^GSPC",
  "없음": null,
};

export function BacktestTab() {
  return <AuthGate>{() => <BacktestRunner />}</AuthGate>;
}

function BacktestRunner() {
  const [markets, setMarkets] = useState<string[]>(["KOSPI", "KOSDAQ"]);
  const [universeN, setUniverseN] = useState(200);
  const [topN, setTopN] = useState(20);
  const [capital, setCapital] = useState(10_000_000);
  const [startDate, setStartDate] = useState("2020-01-01");
  const [endDate, setEndDate] = useState("2024-12-31");
  const [benchmark, setBenchmark] = useState("^KS11 (KOSPI)");
  const [deposits, setDeposits] = useState("2021-01-04,5000000\n2022-01-03,5000000\n2023-01-02,5000000");
  const [formErr, setFormErr] = useState<string | null>(null);

  const { status, result, error, busy, enqueue } = useJob<BacktestJobResult>();

  function toggleMarket(m: string) {
    setMarkets((cur) => (cur.includes(m) ? cur.filter((x) => x !== m) : [...cur, m]));
  }

  function run() {
    setFormErr(null);
    if (markets.length === 0) { setFormErr("마켓을 하나 이상 선택하세요."); return; }
    if (startDate >= endDate) { setFormErr("종료일이 시작일보다 이후여야 합니다."); return; }
    const deposit_schedule: Record<string, number> = {};
    deposits.split("\n").forEach((line) => {
      const [d, a] = line.split(",");
      if (d?.trim() && a?.trim()) deposit_schedule[d.trim()] = Number(a.trim());
    });
    enqueue("/api/v1/backtest/run", {
      markets, universe_n: universeN, top_n: topN, initial_capital: capital,
      start_date: startDate, end_date: endDate, deposit_schedule,
      benchmark_ticker: BENCHMARKS[benchmark], benchmark_label: benchmark,
    });
  }

  const r = result;
  return (
    <div className="space-y-4">
      <header className="flex items-start justify-between gap-2">
        <div>
          <h2 className="text-lg font-bold text-ink">🔬 백테스트</h2>
          <p className="text-sm text-ink-2">모멘텀·추세·거래량으로 종목을 자동 선정하고 과거 수익률을 검증합니다. (백그라운드 워커 처리)</p>
        </div>
        <UserMenu />
      </header>

      <div className="grid grid-cols-1 gap-4 rounded-card border border-hairline bg-surface p-5 shadow-card md:grid-cols-2">
        <div className="space-y-3">
          <Field label="대상 마켓">
            <div className="flex flex-wrap gap-2">
              {MARKETS.map((m) => (
                <button key={m} onClick={() => toggleMarket(m)}
                  className={`rounded-full px-3 py-1 text-xs font-medium transition ${markets.includes(m) ? "bg-accent text-white" : "bg-canvas text-ink-2"}`}>
                  {m}
                </button>
              ))}
            </div>
          </Field>
          <Slider label={`마켓별 후보 (시총 상위): ${universeN}`} min={50} max={500} step={50} value={universeN} onChange={setUniverseN} />
          <Slider label={`최종 선정 종목 수: ${topN}`} min={5} max={50} step={5} value={topN} onChange={setTopN} />
          <Field label="벤치마크">
            <select value={benchmark} onChange={(e) => setBenchmark(e.target.value)} className="w-full rounded-lg border border-hairline-md bg-surface px-3 py-2 text-sm">
              {Object.keys(BENCHMARKS).map((k) => <option key={k}>{k}</option>)}
            </select>
          </Field>
        </div>
        <div className="space-y-3">
          <div className="grid grid-cols-2 gap-3">
            <Field label="시작일"><input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} className="w-full rounded-lg border border-hairline-md bg-surface px-3 py-2 text-sm" /></Field>
            <Field label="종료일"><input type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} className="w-full rounded-lg border border-hairline-md bg-surface px-3 py-2 text-sm" /></Field>
          </div>
          <Field label="초기 자본금 (원)">
            <input type="number" value={capital} step={1_000_000} onChange={(e) => setCapital(Number(e.target.value))} className="w-full rounded-lg border border-hairline-md bg-surface px-3 py-2 text-sm tabular-nums" />
          </Field>
          <Field label="추가 입금 일정 (YYYY-MM-DD,금액)">
            <textarea value={deposits} onChange={(e) => setDeposits(e.target.value)} rows={3} className="w-full rounded-lg border border-hairline-md bg-surface px-3 py-2 text-xs tabular-nums" />
          </Field>
        </div>
        <div className="md:col-span-2">
          <button onClick={run} disabled={busy} className="w-full rounded-lg bg-accent px-4 py-2.5 text-sm font-semibold text-white hover:brightness-110 disabled:opacity-40">
            {busy ? "처리 중…" : "▶ 스크리닝 후 백테스트 실행"}
          </button>
        </div>
      </div>

      {formErr && <div className="rounded-card border border-loss/40 bg-loss/10 p-4 text-sm text-loss">{formErr}</div>}
      <JobStatusBanner status={status} error={error} processingHint="📊 스크리닝·백테스트 실행 중 — 다운로드 포함 수 분 소요…" />

      {r && (
        <div className="space-y-4">
          <MetricsGrid m={r.metrics} />
          <SelectedTable rows={r.selected_stocks as unknown as SelectedStock[]} />
          {r.log_text && (
            <details className="rounded-card border border-term-border bg-term-1 p-4">
              <summary className="cursor-pointer text-sm font-semibold text-term-ink">📜 실행 로그</summary>
              <pre className="mt-2 max-h-96 overflow-auto text-[0.72rem] leading-relaxed text-term-muted">{r.log_text}</pre>
            </details>
          )}
        </div>
      )}
    </div>
  );
}

interface SelectedStock { ticker: string; name: string; score: number; close: number; volume: number; }

function MetricsGrid({ m }: { m: Record<string, number> }) {
  const cards: [string, string, string?][] = [
    ["최종 자산", `${fmtNum(m.final_asset ?? 0)}원`],
    ["펀드 수익률", `${(m.return_pct ?? 0) >= 0 ? "+" : ""}${(m.return_pct ?? 0).toFixed(2)}%`, signClass(m.return_pct ?? 0)],
    ["CAGR", `${(m.cagr ?? 0) >= 0 ? "+" : ""}${(m.cagr ?? 0).toFixed(2)}%`, signClass(m.cagr ?? 0)],
    ["최대 낙폭(MDD)", `${(m.mdd ?? 0).toFixed(2)}%`, "text-loss"],
    ["승률", `${(m.win_rate ?? 0).toFixed(1)}% (${m.total_sells ?? 0}회)`],
    ["손절(SL)", `${m.sl_count ?? 0}회`],
  ];
  return (
    <div className="grid grid-cols-2 gap-3 md:grid-cols-3 lg:grid-cols-6">
      {cards.map(([label, val, cls]) => (
        <div key={label} className="rounded-card border border-hairline bg-surface p-4 text-center shadow-card">
          <div className="text-[0.72rem] text-ink-2">{label}</div>
          <div className={`tnum mt-1 text-[0.95rem] font-bold ${cls ?? "text-ink"}`}>{val}</div>
        </div>
      ))}
    </div>
  );
}

function SelectedTable({ rows }: { rows: SelectedStock[] }) {
  return (
    <section className="rounded-card border border-hairline bg-surface p-5 shadow-card">
      <h3 className="mb-3 text-sm font-semibold text-ink">📡 선정 종목 ({rows.length})</h3>
      <div className="overflow-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-hairline text-left text-xs text-ink-2">
              <th className="py-2 pr-4">#</th><th className="py-2 pr-4">종목명</th><th className="py-2 pr-4">티커</th>
              <th className="py-2 pr-4 text-right">확신도</th><th className="py-2 pr-4 text-right">현재가</th><th className="py-2 text-right">거래량</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((s, i) => (
              <tr key={s.ticker} className="border-b border-hairline-md/40">
                <td className="py-1.5 pr-4 text-ink-2">{i + 1}</td>
                <td className="py-1.5 pr-4 text-ink">{s.name}</td>
                <td className="py-1.5 pr-4 tnum text-ink-2">{s.ticker}</td>
                <td className="py-1.5 pr-4 text-right tnum font-semibold text-ink">{s.score.toFixed(1)}pt</td>
                <td className="py-1.5 pr-4 text-right tnum text-ink">{fmtNum(s.close)}</td>
                <td className="py-1.5 text-right tnum text-ink-2">{fmtNum(s.volume)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <span className="mb-1 block text-xs font-medium text-ink-2">{label}</span>
      {children}
    </label>
  );
}

function Slider({ label, min, max, step, value, onChange }: {
  label: string; min: number; max: number; step: number; value: number; onChange: (v: number) => void;
}) {
  return (
    <Field label={label}>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} className="w-full accent-accent" />
    </Field>
  );
}
