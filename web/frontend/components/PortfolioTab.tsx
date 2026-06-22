"use client";

import { useState } from "react";
import useSWR, { mutate } from "swr";

import { AuthGate, UserMenu } from "@/components/AuthGate";
import { useJob } from "@/hooks/useJob";
import { fmtNum, signClass } from "@/lib/format";
import type { PortfolioItem, TradeItem, StockHit, PortfolioAnalysis, PfRecommendation } from "@/lib/api-types";
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

// 권고 타입별 라벨/색
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

  const g = result?.guide;

  return (
    <section className="rounded-card border border-term-border bg-term-1 p-5 shadow-elevated">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-term-ink">🧭 포트폴리오 매매 지침 분석</h3>
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
          {status === "pending" ? "분석 작업 대기 중…" : "섹터·시장 모멘텀을 진단하고 있습니다…"}
        </p>
      )}
      {error && <p className="text-sm text-loss">분석 실패: {error}</p>}
      {!busy && !error && !result && (
        <p className="text-sm text-term-muted">보유 종목의 섹터 집중도·시장 모멘텀을 진단해 매수/매도/유지 지침을 제시합니다.</p>
      )}

      {result && !busy && g && (
        <div className="space-y-4">
          {/* 요약 */}
          <div className="grid grid-cols-3 gap-2">
            <SummaryTile label="시장 상태" value={result.market_status || "—"} tone={result.market_status === "상승장" ? "text-gain" : result.market_status === "하락장" ? "text-loss" : "text-term-ink"} />
            <SummaryTile label="섹터 집중도(HHI)" value={g.hhi ? fmtNum(g.hhi) : "—"} tone={g.is_concentrated ? "text-loss" : "text-gain"} sub={g.is_concentrated ? "과집중" : "분산 양호"} />
            <SummaryTile label="총 평가액" value={result.total_value ? `${fmtNum(result.total_value)}` : "—"} tone="text-term-ink" />
          </div>

          {/* 익절 권고 (매도) */}
          {g.profit_take.length > 0 && (
            <RecBlock title="💰 수익 확정 권고 (매도)">
              {g.profit_take.map((p) => (
                <div key={p.ticker} className="rounded-lg border border-gain/40 bg-gain/10 px-3 py-2 text-[0.82rem]">
                  <span className="font-semibold text-term-ink">{p.name}</span>{" "}
                  <span className={`tnum font-bold ${signClass(p.pnl_pct)}`}>{p.pnl_pct >= 0 ? "+" : ""}{p.pnl_pct}%</span>
                  <p className="text-term-muted">{p.reason}</p>
                </div>
              ))}
            </RecBlock>
          )}

          {/* 섹터 리밸런싱 권고 */}
          {g.recommendations.length > 0 && (
            <RecBlock title="⚖️ 섹터 리밸런싱 지침">
              {g.recommendations.map((r, i) => {
                const m = REC_META[r.type];
                return (
                  <div key={i} className={`rounded-lg border px-3 py-2 text-[0.82rem] ${m.cls}`}>
                    <div className="flex items-center gap-2">
                      <span>{r.icon}</span>
                      <span className="font-semibold text-term-ink">{r.sector}</span>
                      <span className="tnum text-term-muted">{r.weight}%</span>
                      <span className="ml-auto rounded-full bg-black/30 px-2 py-0.5 text-[0.62rem] font-semibold text-term-ink">{m.label}</span>
                    </div>
                    <p className="mt-1 text-term-muted">{r.message}</p>
                    {r.tickers && <p className="mt-0.5 text-[0.7rem] text-term-muted">{r.tickers}</p>}
                  </div>
                );
              })}
            </RecBlock>
          )}

          {/* 미보유 주도 섹터 (매수 후보) */}
          {g.missing_top.length > 0 && (
            <RecBlock title="✨ 미보유 주도 섹터 (매수 후보)">
              {g.missing_top.map((m) => (
                <div key={m.sector} className="flex items-center justify-between rounded-lg border border-gain/30 bg-gain/5 px-3 py-1.5 text-[0.82rem]">
                  <span className="text-term-ink">{m.sector} <span className="text-term-muted">· {m.name}</span></span>
                  <span className={`tnum font-bold ${signClass(m.return_5d)}`}>{m.return_5d >= 0 ? "+" : ""}{m.return_5d}% <span className="text-[0.66rem] text-term-muted">5일</span></span>
                </div>
              ))}
            </RecBlock>
          )}

          {/* 섹터 모멘텀 랭킹 */}
          {g.sector_scores.length > 0 && (
            <details className="rounded-lg border border-term-border bg-black/20 p-3">
              <summary className="cursor-pointer text-[0.8rem] font-semibold text-term-ink">📊 섹터 모멘텀 랭킹 ({g.sector_scores.length})</summary>
              <ul className="mt-2 space-y-1">
                {g.sector_scores.map((s) => (
                  <li key={s.sector} className="flex items-center justify-between text-[0.78rem]">
                    <span className="text-term-muted">
                      <RankBadge rank={s.rank} /> {s.sector} <span className="text-term-muted/70">· {s.name}</span>
                    </span>
                    <span className={`tnum ${signClass(s.return_5d)}`}>{s.return_5d >= 0 ? "+" : ""}{s.return_5d}%</span>
                  </li>
                ))}
              </ul>
            </details>
          )}

          {(result.unknown_tickers?.length ?? 0) > 0 && (
            <p className="text-[0.72rem] text-term-muted">※ 섹터 미분류: {result.unknown_tickers!.join(", ")}</p>
          )}
        </div>
      )}
    </section>
  );
}

function SummaryTile({ label, value, tone, sub }: { label: string; value: string; tone: string; sub?: string }) {
  return (
    <div className="rounded-lg border border-term-border bg-black/20 px-3 py-2 text-center">
      <div className="text-[0.62rem] text-term-muted">{label}</div>
      <div className={`tnum text-sm font-bold ${tone}`}>{value}</div>
      {sub && <div className="text-[0.6rem] text-term-muted">{sub}</div>}
    </div>
  );
}

function RecBlock({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="space-y-1.5">
      <h4 className="text-[0.8rem] font-semibold text-term-ink">{title}</h4>
      {children}
    </div>
  );
}

function RankBadge({ rank }: { rank: "TOP" | "NORMAL" | "BOTTOM" }) {
  const m = rank === "TOP" ? "text-gain" : rank === "BOTTOM" ? "text-loss" : "text-term-muted";
  const t = rank === "TOP" ? "▲" : rank === "BOTTOM" ? "▼" : "·";
  return <span className={`${m} font-bold`}>{t}</span>;
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
