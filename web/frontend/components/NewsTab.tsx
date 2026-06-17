"use client";

import { useState } from "react";
import useSWR from "swr";
import { api, fetcher, ApiError } from "@/lib/api";
import { signClass } from "@/lib/format";
import type { StockHit } from "@/lib/api-types";

interface Article {
  title: string;
  link: string;
  publisher: string;
  pub_date: string;
}
interface NewsResponse {
  ticker: string;
  is_etf: boolean;
  articles: Article[];
  sentiment: { sentiment: string; score: number; summary: string; detail: any[] };
  sector_performance: Record<string, number | { avg_chg?: number } | any>;
  etf_meta: { sector: string; holdings: string[] } | null;
}
interface Summary {
  summary: string;
  sentiment: string;
  score: number;
  key_points: string[];
  investment_implication: string;
}

/** 뉴스 & 관련 종목 탭 — 기존 render_news_tab 의 2컬럼 레이아웃을 Tailwind 로 재구축. */
export function NewsTab({ picked }: { picked: StockHit | null }) {
  const { data, isLoading } = useSWR<NewsResponse>(
    picked ? `/news/${encodeURIComponent(picked.ticker)}` : null,
    fetcher
  );

  if (!picked) {
    return <Placeholder>종목을 선택하면 뉴스와 감성 분석이 표시됩니다.</Placeholder>;
  }
  if (isLoading) return <Placeholder>📡 뉴스 & 데이터 수집 중…</Placeholder>;
  if (!data) return <Placeholder>뉴스를 불러오지 못했습니다.</Placeholder>;

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-5">
      {/* 뉴스 목록 (3/5) */}
      <section className="lg:col-span-3">
        {data.is_etf && data.etf_meta && (
          <div className="mb-3 rounded-lg border border-hairline bg-canvas px-4 py-2.5 text-sm">
            <span className="font-semibold text-accent">섹터: {data.etf_meta.sector || "N/A"}</span>
            {data.etf_meta.holdings.length > 0 && (
              <span className="ml-2 text-ink-2">
                구성종목 뉴스 포함: {data.etf_meta.holdings.join(", ")}
              </span>
            )}
          </div>
        )}
        <h3 className="mb-2 text-sm font-semibold text-ink">📰 최신 뉴스 ({data.articles.length}건)</h3>
        <div className="space-y-1.5">
          {data.articles.length === 0 && <p className="text-sm text-ink-2">수집된 뉴스가 없습니다.</p>}
          {data.articles.map((n, i) => (
            <NewsCard key={i} article={n} ticker={data.ticker} />
          ))}
        </div>
      </section>

      {/* 감성 + 섹터 (2/5) */}
      <aside className="space-y-4 lg:col-span-2">
        <SentimentCard s={data.sentiment} />
        <SectorCard perf={data.sector_performance} />
      </aside>
    </div>
  );
}

function NewsCard({ article, ticker }: { article: Article; ticker: string }) {
  const [sum, setSum] = useState<Summary | null>(null);
  const [busy, setBusy] = useState(false);

  async function summarize() {
    setBusy(true);
    try {
      const res = await api.post<Summary>("/news/summarize", {
        title: article.title,
        link: article.link,
        ticker,
      });
      setSum(res);
    } catch (e) {
      setSum({
        summary: e instanceof ApiError ? e.message : "요약 실패",
        sentiment: "중립", score: 0, key_points: [], investment_implication: "",
      });
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="rounded-lg border border-hairline bg-surface p-3">
      <div className="flex items-start gap-2">
        <a
          href={article.link}
          target="_blank"
          rel="noreferrer"
          className="flex-1 text-[0.9rem] font-medium leading-relaxed text-ink hover:text-accent"
        >
          {article.title}
        </a>
        <button
          onClick={summarize}
          disabled={busy}
          className="shrink-0 rounded-md border border-hairline-md px-2 py-1 text-xs text-ink-2 hover:bg-elevated disabled:opacity-40"
        >
          {busy ? "…" : "AI 요약"}
        </button>
      </div>
      <div className="mt-1.5 text-[0.73rem] text-ink-2">
        {article.publisher}
        {article.pub_date && `  ·  ${article.pub_date}`}
      </div>
      {sum && (
        <div className="mt-2 rounded-md bg-canvas p-2.5 text-[0.82rem] text-ink-2">
          <p>{sum.summary}</p>
          {sum.key_points.length > 0 && (
            <ul className="mt-1.5 list-disc space-y-0.5 pl-4">
              {sum.key_points.map((k, i) => (
                <li key={i}>{k}</li>
              ))}
            </ul>
          )}
          {sum.investment_implication && (
            <p className="mt-1.5 font-medium text-ink">💡 {sum.investment_implication}</p>
          )}
        </div>
      )}
    </div>
  );
}

function SentimentCard({ s }: { s: NewsResponse["sentiment"] }) {
  const pos = s.score >= 0;
  return (
    <section className="rounded-card border border-hairline bg-surface p-4 shadow-card">
      <h3 className="mb-2 text-sm font-semibold text-ink">🧠 AI 감성 분석</h3>
      <div
        className={`inline-flex items-center gap-2 rounded-full px-3 py-1 text-sm font-bold ${
          pos ? "bg-gain/10 text-gain" : "bg-loss/10 text-loss"
        }`}
      >
        {s.sentiment} <span className="tnum">{s.score >= 0 ? "+" : ""}{s.score.toFixed(1)}</span>
      </div>
      {s.summary && <p className="mt-3 text-sm text-ink-2">{s.summary}</p>}
      {s.detail?.length > 0 && (
        <div className="mt-3 space-y-1">
          {s.detail.slice(0, 5).map((d: any, i: number) => {
            const sc = d.score ?? 0;
            const icon = d.sentiment === "긍정" ? "🟢" : d.sentiment === "부정" ? "🔴" : "⚪";
            return (
              <div key={i} className="flex items-center gap-2 rounded-md bg-canvas px-2.5 py-1.5 text-[0.82rem]">
                <span>{icon}</span>
                <span className="flex-1 truncate text-ink-2">{d.title}</span>
                <span className={`tnum font-bold ${signClass(sc)}`}>{sc >= 0 ? "+" : ""}{sc.toFixed(1)}</span>
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
}

function SectorCard({ perf }: { perf: NewsResponse["sector_performance"] }) {
  // 백엔드는 get_related_sector_performance 형태(tickers 배열 또는 {섹터:pct})를 통과시킨다.
  const tickers: { name: string; ticker: string; chg: number }[] = Array.isArray((perf as any)?.tickers)
    ? (perf as any).tickers
    : [];
  const rows = tickers.length
    ? tickers.map((t) => [t.name, t.chg] as [string, number])
    : Object.entries(perf).filter(([, v]) => typeof v === "number") as [string, number][];

  return (
    <section className="rounded-card border border-hairline bg-surface p-4 shadow-card">
      <h3 className="mb-2 text-sm font-semibold text-ink">📊 관련 섹터 성과</h3>
      {rows.length === 0 ? (
        <p className="text-sm text-ink-2">섹터 데이터 없음</p>
      ) : (
        <ul className="divide-y divide-hairline-md">
          {rows.slice(0, 6).map(([name, pct], i) => (
            <li key={i} className="flex items-center justify-between py-1.5">
              <span className="text-[0.85rem] text-ink-2">{name}</span>
              <span className={`tnum text-[0.85rem] font-bold ${signClass(pct)}`}>
                {pct >= 0 ? "+" : ""}{pct.toFixed(2)}%
              </span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

function Placeholder({ children }: { children: React.ReactNode }) {
  return (
    <div className="rounded-card border border-dashed border-hairline-md p-12 text-center text-ink-2">
      {children}
    </div>
  );
}
