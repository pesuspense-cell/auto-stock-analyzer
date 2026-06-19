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
          {data.results.map((h) => {
            // 한글명 우선 표시, 영문명이 다르면 보조로 함께 노출
            const primary = h.nameKr || h.name;
            const secondary = h.nameKr && h.name && h.name !== h.nameKr ? h.name : null;
            return (
              <li key={h.ticker}>
                <button
                  onClick={() => {
                    onPick(h);
                    setQ("");
                  }}
                  className="flex w-full items-center justify-between gap-2 px-3 py-2 text-left text-sm hover:bg-elevated"
                >
                  <span className="flex min-w-0 items-center gap-1.5">
                    <span className="truncate text-ink">{primary}</span>
                    {secondary && <span className="truncate text-xs text-ink-3">{secondary}</span>}
                    {h.isEtf && (
                      <span className="shrink-0 rounded bg-accent/10 px-1 text-[0.6rem] font-semibold text-accent">
                        ETF
                      </span>
                    )}
                  </span>
                  <span className="tnum shrink-0 text-xs text-ink-2">{h.ticker}</span>
                </button>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
