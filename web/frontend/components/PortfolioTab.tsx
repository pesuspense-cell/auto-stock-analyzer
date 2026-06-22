"use client";

import { useState } from "react";
import useSWR, { mutate } from "swr";

import { AuthGate, UserMenu } from "@/components/AuthGate";
import { useJob } from "@/hooks/useJob";
import { fmtNum, signClass } from "@/lib/format";
import type { PortfolioItem, TradeItem, StockHit, PortfolioAnalysis, PfRecommendation, PfHolding, PfOverall, PfActionLevel } from "@/lib/api-types";
import { StockSearch } from "@/components/StockSearch";

// 쿠키 세션 기반 동일 출처 fetch (Supabase Auth 쿠키 자동 포함)
const jfetch = async <T,>(url: string): Promise<T> => {
  const res = await fetch(url, { credentials: "same-origin" });
  if (!res.ok) throw new Error((await res.json().catch(() => ({})))?.error ?? res.statusText);
  return res.json();
};

export function PortfolioTab() {
  return (
    <AuthGate>
      {(user) => (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-bold text-ink">💼 내 포트폴리오</h2>
            <UserMenu />
          </div>
          <AddForm />
          <Holdings />
          <PortfolioAnalysisCard />
          <Trades />
          <p className="text-xs text-ink-3">로그인: {user.email}</p>
        </div>
      )}
    </AuthGate>
  );
}

function AddForm() {
  const [picked, setPicked] = useState<{ ticker: string; name: string } | null>(null);
  const [price, setPrice] = useState("");
  const [qty, setQty] = useState("1");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function add() {
    if (!picked || !(Number(price) > 0)) { setErr("종목과 평균단가를 입력하세요."); return; }
    setBusy(true); setErr(null);
    try {
      const res = await fetch("/api/v1/portfolio", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "same-origin",
        body: JSON.stringify({ ticker: picked.ticker, avgPrice: Number(price), quantity: Number(qty) }),
      });
      if (!res.ok) throw new Error((await res.json())?.error ?? "추가 실패");
      setPicked(null); setPrice(""); setQty("1");
      mutate("/api/v1/portfolio");
    } catch (e) {
      setErr(e instanceof Error ? e.message : "추가 실패");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="rounded-card border border-hairline bg-surface p-4 shadow-card">
      <h3 className="mb-2 text-sm font-semibold text-ink">➕ 종목 추가 / 추가매수</h3>
      <div className="grid grid-cols-1 gap-2 md:grid-cols-4">
        <div className="md:col-span-2">
          {picked ? (
            <button onClick={() => setPicked(null)} className="w-full rounded-lg border border-hairline-md px-3 py-2 text-left text-sm">
              {picked.name} <span className="tnum text-ink-2">({picked.ticker})</span> · 변경
            </button>
          ) : (
            <StockSearch onPick={(h: StockHit) => setPicked({ ticker: h.ticker, name: h.nameKr || h.name })} />
          )}
        </div>
        <input type="number" value={price} onChange={(e) => setPrice(e.target.value)} placeholder="평균단가"
          className="rounded-lg border border-hairline-md bg-surface px-3 py-2 text-sm tabular-nums" />
        <div className="flex gap-2">
          <input type="number" value={qty} onChange={(e) => setQty(e.target.value)} placeholder="수량"
            className="w-full rounded-lg border border-hairline-md bg-surface px-3 py-2 text-sm tabular-nums" />
          <button onClick={add} disabled={busy} className="shrink-0 rounded-lg bg-accent px-4 py-2 text-sm font-semibold text-white hover:brightness-110 disabled:opacity-40">
            추가
          </button>
        </div>
      </div>
      {err && <p className="mt-2 text-sm text-loss">{err}</p>}
    </section>
  );
}

function Holdings() {
  const { data, isLoading } = useSWR<PortfolioItem[]>("/api/v1/portfolio", jfetch, { refreshInterval: 30_000 });

  async function sell(item: PortfolioItem) {
    const input = prompt(`'${item.name ?? item.ticker}' 매도가를 입력하세요`, String(item.currentPrice ?? item.avgPrice));
    if (!input) return;
    const res = await fetch(`/api/v1/portfolio/${item.id}/sell`, {
      method: "POST", headers: { "Content-Type": "application/json" }, credentials: "same-origin",
      body: JSON.stringify({ sellPrice: Number(input) }),
    });
    if (res.ok) { mutate("/api/v1/portfolio"); mutate("/api/v1/portfolio/trades"); }
  }
  async function remove(item: PortfolioItem) {
    if (!confirm("삭제하시겠습니까?")) return;
    const res = await fetch(`/api/v1/portfolio/${item.id}`, { method: "DELETE", credentials: "same-origin" });
    if (res.ok) mutate("/api/v1/portfolio");
  }

  if (isLoading) return <p className="text-sm text-ink-2">불러오는 중…</p>;
  if (!data || data.length === 0) return <p className="text-sm text-ink-2">보유 종목이 없습니다.</p>;

  return (
    <section className="overflow-auto rounded-card border border-hairline bg-surface p-4 shadow-card">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-hairline text-left text-xs text-ink-2">
            <th className="py-2 pr-4">종목</th><th className="py-2 pr-4 text-right">평균단가</th>
            <th className="py-2 pr-4 text-right">현재가</th><th className="py-2 pr-4 text-right">수량</th>
            <th className="py-2 pr-4 text-right">수익률</th><th className="py-2 text-right">관리</th>
          </tr>
        </thead>
        <tbody>
          {data.map((it) => (
            <tr key={it.id} className="border-b border-hairline-md/40">
              <td className="py-2 pr-4 text-ink">{it.name ?? it.ticker} <span className="tnum text-xs text-ink-2">{it.ticker}</span></td>
              <td className="py-2 pr-4 text-right tnum text-ink">{fmtNum(it.avgPrice, 2)}</td>
              <td className="py-2 pr-4 text-right tnum text-ink">{it.currentPrice != null ? fmtNum(it.currentPrice, 2) : "—"}</td>
              <td className="py-2 pr-4 text-right tnum text-ink">{it.quantity}</td>
              <td className={`py-2 pr-4 text-right tnum font-semibold ${it.returnPct != null ? signClass(it.returnPct) : "text-ink-2"}`}>
                {it.returnPct != null ? `${it.returnPct >= 0 ? "+" : ""}${it.returnPct.toFixed(2)}%` : "—"}
              </td>
              <td className="py-2 text-right">
                <button onClick={() => sell(it)} className="mr-2 text-xs text-accent hover:underline">매도</button>
                <button onClick={() => remove(it)} className="text-xs text-loss hover:underline">삭제</button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

// 액션 위험도별 색
const LEVEL: Record<PfActionLevel, { badge: string; cls: string; chip: string }> = {
  danger: { badge: "bg-loss/20 text-loss", cls: "border-loss/40", chip: "text-loss" },
  warn: { badge: "bg-amber-500/20 text-amber-300", cls: "border-amber-500/40", chip: "text-amber-300" },
  good: { badge: "bg-gain/20 text-gain", cls: "border-gain/40", chip: "text-gain" },
  neutral: { badge: "bg-black/30 text-term-muted", cls: "border-term-border", chip: "text-term-muted" },
};

// 섹터 권고 타입별 라벨/색
const REC_META: Record<PfRecommendation["type"], { label: string; cls: string }> = {
  reduce: { label: "비중 축소 / 매도", cls: "border-loss/40 bg-loss/10" },
  add: { label: "비중 확대 / 매수", cls: "border-gain/40 bg-gain/10" },
  hold: { label: "유지 / 익절 준비", cls: "border-accent/40 bg-accent/10" },
  watch: { label: "집중 경고", cls: "border-amber-500/40 bg-amber-500/10" },
};

function PortfolioAnalysisCard() {
  const { result, busy, status, error, enqueue } = useJob<PortfolioAnalysis>();

  function run() {
    enqueue("/api/v1/portfolio/analyze", {});
  }

  return (
    <section className="rounded-card border border-term-border bg-term-1 p-5 shadow-elevated">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-term-ink">🧭 포트폴리오 종합 진단</h3>
        <button
          onClick={run}
          disabled={busy}
          className="rounded-lg bg-accent px-3 py-1.5 text-xs font-semibold text-white hover:brightness-110 disabled:opacity-40"
        >
          {busy ? "분석 중…" : result ? "다시 분석" : "분석 시작"}
        </button>
      </div>

      {busy && (
        <p className="text-sm text-term-muted">
          {status === "pending" ? "분석 작업 대기 중…" : "종목별 차트·모멘텀·손절선을 진단하고 있습니다…"}
        </p>
      )}
      {error && <p className="text-sm text-loss">분석 실패: {error}</p>}
      {!busy && !error && !result && (
        <p className="text-sm text-term-muted">보유 종목별 차트 신호·모멘텀·손절선을 종합해 대응 액션을 제시하고, 전체 포트폴리오를 평가합니다.</p>
      )}

      {result && !busy && !result.empty && (
        <div className="space-y-4">
          <OverallCard o={result.overall} />

          <div className="space-y-2">
            <h4 className="text-[0.8rem] font-semibold text-term-ink">📋 종목별 대응 리포트</h4>
            {result.holdings.map((h) => <HoldingReport key={h.ticker} h={h} />)}
          </div>

          <SectorGuide guide={result.guide} />
        </div>
      )}
    </section>
  );
}

function OverallCard({ o }: { o: PfOverall }) {
  const lv = LEVEL[o.verdict_level];
  return (
    <div className={`rounded-lg border ${lv.cls} bg-black/20 p-4`}>
      <div className={`text-sm font-bold ${lv.chip}`}>{o.verdict}</div>
      <div className="mt-3 grid grid-cols-3 gap-2">
        <Tile label="총 평가액" value={fmtNum(o.total_value)} tone="text-term-ink" />
        <Tile label="총 수익률" value={`${o.total_pnl_pct >= 0 ? "+" : ""}${o.total_pnl_pct.toFixed(2)}%`} tone={signClass(o.total_pnl_pct)} />
        <Tile
          label="섹터 집중도(HHI)"
          value={o.hhi ? fmtNum(o.hhi) : "—"}
          tone={o.is_concentrated ? "text-loss" : "text-gain"}
          sub={o.is_concentrated ? "과집중" : "분산 양호"}
        />
      </div>
      {/* 액션 분포 */}
      <div className="mt-2 flex flex-wrap gap-1.5 text-[0.66rem]">
        <CountChip n={o.action_counts.danger} label="손절 점검" level="danger" />
        <CountChip n={o.action_counts.warn} label="축소/매도" level="warn" />
        <CountChip n={o.action_counts.good} label="강세/익절" level="good" />
        <CountChip n={o.action_counts.neutral} label="보유/관망" level="neutral" />
        {o.market_status && <span className="ml-auto rounded-full bg-black/30 px-2 py-0.5 text-term-muted">시장: {o.market_status}</span>}
      </div>
    </div>
  );
}

function CountChip({ n, label, level }: { n: number; label: string; level: PfActionLevel }) {
  if (!n) return null;
  return <span className={`rounded-full px-2 py-0.5 font-semibold ${LEVEL[level].badge}`}>{label} {n}</span>;
}

function HoldingReport({ h }: { h: PfHolding }) {
  const lv = LEVEL[h.action_level];
  return (
    <div className={`rounded-lg border ${lv.cls} bg-black/20 p-3`}>
      {/* 헤더: 종목 / 수익률 / 액션 */}
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-semibold text-term-ink">{h.name}</span>
        <span className="tnum text-[0.7rem] text-term-muted">{h.ticker}</span>
        <span className={`tnum text-sm font-bold ${signClass(h.pnl_pct)}`}>{h.pnl_pct >= 0 ? "+" : ""}{h.pnl_pct.toFixed(2)}%</span>
        <span className={`ml-auto rounded-full px-2.5 py-0.5 text-[0.7rem] font-bold ${lv.badge}`}>
          {h.signal_badge ? `${h.signal_badge} ` : ""}{h.action}
        </span>
      </div>

      {/* 지표 줄 */}
      {h.ok !== false && (
        <div className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 text-[0.74rem] sm:grid-cols-4">
          <KV label="기술신호" value={`${h.signal_label ?? "—"}${h.tech_score != null ? ` (${h.tech_score >= 0 ? "+" : ""}${h.tech_score})` : ""}`} tone={h.tech_score != null ? signClass(h.tech_score) : ""} />
          <KV label="예상수익" value={h.expected_return_pct != null ? `${h.expected_return_pct >= 0 ? "+" : ""}${h.expected_return_pct.toFixed(1)}%` : "—"} tone={h.expected_return_pct != null ? signClass(h.expected_return_pct) : ""} />
          <KV label="승률" value={h.win_prob != null ? `${h.win_prob.toFixed(0)}%` : "—"} />
          <KV label="20일 모멘텀" value={h.momentum_20d != null ? `${h.momentum_20d >= 0 ? "+" : ""}${h.momentum_20d.toFixed(1)}%` : "—"} tone={h.momentum_20d != null ? signClass(h.momentum_20d) : ""} />
        </div>
      )}

      {/* 손절선 */}
      {h.stop_loss_price != null && (
        <div className="mt-2 flex items-center gap-2 text-[0.74rem]">
          <span className="text-term-muted">손절선</span>
          <span className="tnum font-semibold text-term-ink">{fmtNum(h.stop_loss_price, 2)}</span>
          {h.stop_distance_pct != null && (
            <span className={`tnum ${h.stop_breached ? "text-loss font-bold" : "text-term-muted"}`}>
              ({h.stop_distance_pct >= 0 ? "+" : ""}{h.stop_distance_pct.toFixed(1)}%)
            </span>
          )}
          {h.stop_breached && <span className="rounded bg-loss/20 px-1.5 text-[0.62rem] font-bold text-loss">손절선 이탈</span>}
          <span className="ml-auto tnum text-[0.7rem] text-term-muted">평단 {fmtNum(h.avg_price, 2)} · 현재 {fmtNum(h.current_price, 2)}</span>
        </div>
      )}

      {/* 사유 */}
      {h.reasons.length > 0 && (
        <ul className="mt-1.5 space-y-0.5 text-[0.74rem] text-term-muted">
          {h.reasons.map((r, i) => <li key={i}>• {r}</li>)}
        </ul>
      )}
    </div>
  );
}

function KV({ label, value, tone = "" }: { label: string; value: string; tone?: string }) {
  return (
    <div className="flex items-center justify-between gap-1 sm:block sm:text-center">
      <span className="text-[0.62rem] text-term-muted">{label}</span>
      <span className={`tnum font-semibold ${tone || "text-term-ink"}`}>{value}</span>
    </div>
  );
}

/** 섹터/시장 리밸런싱 진단 (보조 — 접힘). */
function SectorGuide({ guide: g }: { guide: PortfolioAnalysis["guide"] }) {
  const has = g.recommendations.length || g.missing_top.length || g.sector_scores.length;
  if (!has) return null;
  return (
    <details className="rounded-lg border border-term-border bg-black/20 p-3">
      <summary className="cursor-pointer text-[0.8rem] font-semibold text-term-ink">⚖️ 섹터·시장 리밸런싱 진단</summary>
      <div className="mt-3 space-y-3">
        {g.recommendations.map((r, i) => {
          const m = REC_META[r.type];
          return (
            <div key={i} className={`rounded-lg border px-3 py-2 text-[0.8rem] ${m.cls}`}>
              <div className="flex items-center gap-2">
                <span>{r.icon}</span>
                <span className="font-semibold text-term-ink">{r.sector}</span>
                <span className="tnum text-term-muted">{r.weight}%</span>
                <span className="ml-auto rounded-full bg-black/30 px-2 py-0.5 text-[0.62rem] font-semibold text-term-ink">{m.label}</span>
              </div>
              <p className="mt-1 text-term-muted">{r.message}</p>
            </div>
          );
        })}
        {g.missing_top.length > 0 && (
          <div>
            <p className="mb-1 text-[0.74rem] font-semibold text-term-ink">✨ 미보유 주도 섹터 (매수 후보)</p>
            {g.missing_top.map((m) => (
              <div key={m.sector} className="flex items-center justify-between text-[0.78rem] text-term-muted">
                <span>{m.sector} · {m.name}</span>
                <span className={`tnum ${signClass(m.return_5d)}`}>{m.return_5d >= 0 ? "+" : ""}{m.return_5d}%</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </details>
  );
}

function Tile({ label, value, tone, sub }: { label: string; value: string; tone: string; sub?: string }) {
  return (
    <div className="rounded-lg border border-term-border bg-black/20 px-3 py-2 text-center">
      <div className="text-[0.62rem] text-term-muted">{label}</div>
      <div className={`tnum text-sm font-bold ${tone}`}>{value}</div>
      {sub && <div className="text-[0.6rem] text-term-muted">{sub}</div>}
    </div>
  );
}

function Trades() {
  const { data } = useSWR<TradeItem[]>("/api/v1/portfolio/trades", jfetch);
  if (!data || data.length === 0) return null;
  return (
    <section className="overflow-auto rounded-card border border-hairline bg-surface p-4 shadow-card">
      <h3 className="mb-2 text-sm font-semibold text-ink">📜 매도 이력</h3>
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-hairline text-left text-xs text-ink-2">
            <th className="py-2 pr-4">종목</th><th className="py-2 pr-4 text-right">매수→매도</th>
            <th className="py-2 pr-4 text-right">수량</th><th className="py-2 pr-4 text-right">순손익</th><th className="py-2 text-right">수익률</th>
          </tr>
        </thead>
        <tbody>
          {data.map((t) => (
            <tr key={t.id} className="border-b border-hairline-md/40">
              <td className="py-1.5 pr-4 text-ink">{t.name ?? t.ticker} <span className="tnum text-xs text-ink-2">{t.ticker}</span></td>
              <td className="py-1.5 pr-4 text-right tnum text-ink-2">{fmtNum(t.buyPrice, 2)} → {fmtNum(t.sellPrice, 2)}</td>
              <td className="py-1.5 pr-4 text-right tnum text-ink">{t.quantity}</td>
              <td className={`py-1.5 pr-4 text-right tnum font-semibold ${signClass(t.netProfit)}`}>{t.netProfit >= 0 ? "+" : ""}{fmtNum(t.netProfit, 0)}</td>
              <td className={`py-1.5 text-right tnum ${signClass(t.returnRate)}`}>{t.returnRate >= 0 ? "+" : ""}{t.returnRate.toFixed(2)}%</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
