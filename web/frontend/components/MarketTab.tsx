"use client";

import useSWR from "swr";
import { fetcher } from "@/lib/api";
import { fmtNum, fmtPct, signClass, priceLabel } from "@/lib/format";
import type { MarketOverview, ExchangeRate, MoverItem } from "@/lib/types";

/** 시장 현황 탭 — 5분 간격 자동 부분 갱신 (Streamlit 전체 rerun 제거). */
export function MarketTab() {
  const { data: overview } = useSWR<MarketOverview>("/market/overview", fetcher, {
    refreshInterval: 300_000,
  });
  const { data: rates } = useSWR<ExchangeRate[]>("/market/rates", fetcher, {
    refreshInterval: 300_000,
  });

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
      <MoverCard title="🔺 상승 상위" items={overview?.gainers} />
      <MoverCard title="🔻 하락 상위" items={overview?.losers} />
      <RatesCard rates={rates} />
    </div>
  );
}

function MoverCard({ title, items }: { title: string; items?: MoverItem[] }) {
  return (
    <section className="rounded-card border border-hairline bg-surface p-5 shadow-card">
      <h3 className="mb-3 text-sm font-semibold text-ink">{title}</h3>
      {!items ? (
        <SkeletonRows />
      ) : (
        <ul className="divide-y divide-hairline">
          {items.map((m) => (
            <li key={m.ticker} className="flex items-center justify-between py-2">
              <span className="truncate text-[0.85rem] text-ink">{m.name}</span>
              <span className="flex items-baseline gap-2">
                <span className="tnum text-[0.85rem] text-ink-2">
                  {priceLabel(m.price, m.ticker)}
                </span>
                <span className={`tnum text-[0.85rem] font-semibold ${signClass(m.change_pct)}`}>
                  {fmtPct(m.change_pct)}
                </span>
              </span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

function RatesCard({ rates }: { rates?: ExchangeRate[] }) {
  return (
    <section className="rounded-card border border-hairline bg-surface p-5 shadow-card">
      <h3 className="mb-3 text-sm font-semibold text-ink">💱 환율</h3>
      {!rates ? (
        <SkeletonRows />
      ) : (
        <ul className="divide-y divide-hairline">
          {rates.map((r) => (
            <li key={r.pair} className="flex items-center justify-between py-2">
              <span className="text-[0.85rem] text-ink-2">{r.pair}</span>
              <span className="flex items-baseline gap-2">
                <span className="tnum text-[0.85rem] font-semibold text-ink">{fmtNum(r.rate, 2)}</span>
                {r.change_pct != null && (
                  <span className={`tnum text-[0.72rem] ${signClass(r.change_pct)}`}>
                    {fmtPct(r.change_pct)}
                  </span>
                )}
              </span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

function SkeletonRows() {
  return (
    <div className="space-y-2">
      {Array.from({ length: 6 }).map((_, i) => (
        <div key={i} className="h-6 animate-pulse rounded bg-hairline/60" />
      ))}
    </div>
  );
}
