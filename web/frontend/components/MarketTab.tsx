"use client";

import useSWR from "swr";
import { fetcher } from "@/lib/api";
import { fmtNum, fmtPct, signClass, priceLabel } from "@/lib/format";
import type { MarketOverview, MoverItem } from "@/lib/types";
import type { RatesResponse, RateItem } from "@/lib/api-types";
import { SectorEtfTable } from "./SectorEtfTable";

/** 시장 현황 탭 — 5분 간격 자동 부분 갱신 (Streamlit 전체 rerun 제거). */
export function MarketTab() {
  const { data: overview } = useSWR<MarketOverview>("/market/overview", fetcher, {
    refreshInterval: 300_000,
  });
  // /market/rates 는 { rates: RateItem[] } 객체를 반환한다 (배열 아님).
  const { data: ratesResp } = useSWR<RatesResponse>("/market/rates", fetcher, {
    refreshInterval: 300_000,
  });

  return (
    <div className="space-y-4">
      {/* 한국/미국 주요 ETF 등락률 비교 */}
      <SectorEtfTable />

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <MoverCard title="🔺 상승 상위" items={overview?.gainers} />
        <MoverCard title="🔻 하락 상위" items={overview?.losers} />
        <RatesCard rates={ratesResp?.rates} />
      </div>
    </div>
  );
}

function MoverCard({ title, items }: { title: string; items?: MoverItem[] }) {
  return (
    <section className="rounded-card border border-hairline bg-surface p-5 shadow-card">
      <h3 className="mb-3 text-sm font-semibold text-ink">{title}</h3>
      {!Array.isArray(items) ? (
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

function RatesCard({ rates }: { rates?: RateItem[] }) {
  return (
    <section className="rounded-card border border-hairline bg-surface p-5 shadow-card">
      <h3 className="mb-3 text-sm font-semibold text-ink">💱 환율</h3>
      {!Array.isArray(rates) ? (
        <SkeletonRows />
      ) : (
        <ul className="divide-y divide-hairline">
          {rates.map((r) => (
            <li key={r.pair} className="flex items-center justify-between py-2">
              <span className="text-[0.85rem] text-ink-2">{r.pair}</span>
              <span className="flex items-baseline gap-2">
                <span className="tnum text-[0.85rem] font-semibold text-ink">{fmtNum(r.rate, 2)}</span>
                {r.changePct != null && (
                  <span className={`tnum text-[0.72rem] ${signClass(r.changePct)}`}>
                    {fmtPct(r.changePct)}
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
