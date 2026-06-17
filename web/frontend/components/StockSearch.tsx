"use client";

import { useState } from "react";
import useSWR from "swr";
import { fetcher } from "@/lib/api";
import type { StockHit } from "@/lib/api-types";

/** 사이드바 종목 검색 — 한글/영문/티커 통합 (백엔드 resolve_ticker). */
export function StockSearch({ onPick }: { onPick: (hit: StockHit) => void }) {
  const [q, setQ] = useState("");
  const { data } = useSWR<{ query: string; results: StockHit[] }>(
    q.trim().length >= 1 ? `/stocks/search?q=${encodeURIComponent(q)}&limit=8` : null,
    fetcher,
    { keepPreviousData: true }
  );

  return (
    <div className="relative">
      <input
        value={q}
        onChange={(e) => setQ(e.target.value)}
        placeholder="종목명·티커 검색 (예: 삼성전자, NVDA)"
        className="w-full rounded-lg border border-hairline-md bg-surface px-3 py-2 text-sm text-ink
                   outline-none focus:border-accent focus:ring-2 focus:ring-accent/20"
      />
      {data && data.results.length > 0 && q.trim() && (
        <ul className="absolute z-10 mt-1 max-h-72 w-full overflow-auto rounded-lg border border-hairline bg-surface shadow-elevated">
          {data.results.map((h) => (
            <li key={h.ticker}>
              <button
                onClick={() => {
                  onPick(h);
                  setQ("");
                }}
                className="flex w-full items-center justify-between px-3 py-2 text-left text-sm hover:bg-elevated"
              >
                <span className="text-ink">{h.name}</span>
                <span className="tnum text-xs text-ink-2">{h.ticker}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
