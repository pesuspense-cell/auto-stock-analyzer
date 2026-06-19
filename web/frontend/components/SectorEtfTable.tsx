"use client";

import useSWR from "swr";
import { fetcher } from "@/lib/api";
import { fmtNum, fmtPct, signClass } from "@/lib/format";
import type { EtfItem, EtfsResponse } from "@/lib/api-types";

const KR_BENCH = "069500.KS"; // KODEX 200
const US_BENCH = "QQQ"; // 나스닥 100
const MAX_BAR = 15; // 등락률 막대 100% 기준(%)

/** 주요 섹터 ETF 등락 비교 — 한국/미국 대표 ETF 일간 등락률 + 벤치마크 상대강도. */
export function SectorEtfTable() {
  const { data } = useSWR<EtfsResponse>("/market/etfs", fetcher, {
    refreshInterval: 600_000, // 10분 (서버 캐시와 동일)
    revalidateOnFocus: false,
  });

  return (
    <section className="rounded-card border border-hairline bg-surface p-5 shadow-card">
      <div className="mb-1 flex items-baseline justify-between">
        <h3 className="text-sm font-semibold text-ink">🗺️ 주요 섹터 ETF 등락 비교</h3>
        <span className="text-[0.7rem] text-ink-2">미국 15 · 국내 20 · 10분 캐시</span>
      </div>

      {!data ? (
        <SkeletonRows n={10} />
      ) : (
        <>
          <SummaryChips s={data.summary} />
          <div className="mt-3 grid grid-cols-1 gap-4 lg:grid-cols-2">
            <EtfColumn
              title="🇰🇷 국내 ETF"
              caption="기준: KODEX 200"
              items={data.etfs.filter((e) => e.country === "KR")}
              benchTicker={KR_BENCH}
              etfs={data.etfs}
            />
            <EtfColumn
              title="🇺🇸 미국 ETF"
              caption="기준: 나스닥 100"
              items={data.etfs.filter((e) => e.country === "US")}
              benchTicker={US_BENCH}
              etfs={data.etfs}
            />
          </div>
        </>
      )}
    </section>
  );
}

function SummaryChips({ s }: { s: EtfsResponse["summary"] }) {
  const upMore = s.up > s.down;
  return (
    <div className="grid grid-cols-3 gap-2 sm:grid-cols-5">
      <Chip label="🔺 상승" value={`${s.up}개`} tone={upMore ? "text-gain" : "text-ink-2"} />
      <Chip label="🔻 하락" value={`${s.down}개`} tone={!upMore ? "text-loss" : "text-ink-2"} />
      <Chip label="전체 평균" value={fmtPct(s.avg)} tone={signClass(s.avg)} />
      <Chip label="🇺🇸 미국" value={fmtPct(s.usAvg)} tone={signClass(s.usAvg)} />
      <Chip label="🇰🇷 국내" value={fmtPct(s.krAvg)} tone={signClass(s.krAvg)} />
    </div>
  );
}

function Chip({ label, value, tone }: { label: string; value: string; tone: string }) {
  return (
    <div className="rounded-lg border border-hairline bg-canvas px-2 py-1.5 text-center">
      <div className="truncate text-[0.6rem] text-ink-2">{label}</div>
      <div className={`tnum text-[0.85rem] font-bold ${tone}`}>{value}</div>
    </div>
  );
}

function EtfColumn({
  title,
  caption,
  items,
  benchTicker,
  etfs,
}: {
  title: string;
  caption: string;
  items: EtfItem[];
  benchTicker: string;
  etfs: EtfItem[];
}) {
  const benchChg = etfs.find((e) => e.ticker === benchTicker)?.changePct ?? null;
  // 지수형 우선 → 등락률 내림차순
  const sorted = [...items].sort(
    (a, b) => Number(b.isIndex) - Number(a.isIndex) || b.changePct - a.changePct
  );

  if (sorted.length === 0) {
    return (
      <div>
        <ColHeader title={title} caption={caption} />
        <p className="py-4 text-center text-xs text-ink-2">데이터 없음</p>
      </div>
    );
  }

  return (
    <div>
      <ColHeader title={title} caption={caption} />
      <ul className="divide-y divide-hairline">
        {sorted.map((e) => (
          <EtfRow key={e.ticker} e={e} benchChg={benchChg} isBench={e.ticker === benchTicker} />
        ))}
      </ul>
    </div>
  );
}

function ColHeader({ title, caption }: { title: string; caption: string }) {
  return (
    <div className="mb-1 flex items-baseline justify-between">
      <span className="text-[0.8rem] font-semibold text-ink">{title}</span>
      <span className="text-[0.65rem] text-ink-2">{caption}</span>
    </div>
  );
}

function EtfRow({ e, benchChg, isBench }: { e: EtfItem; benchChg: number | null; isBench: boolean }) {
  const up = e.changePct >= 0;
  const barW = Math.min((Math.abs(e.changePct) / MAX_BAR) * 100, 100);
  // 벤치마크 대비 상대강도(강/약)
  let rel: { text: string; cls: string } | null = null;
  if (benchChg != null && !isBench) {
    if (e.changePct > benchChg) rel = { text: "강", cls: "text-gain" };
    else if (e.changePct < benchChg) rel = { text: "약", cls: "text-loss" };
  }
  const isKr = e.ticker.endsWith(".KS") || e.ticker.endsWith(".KQ");
  const priceStr = isKr ? `${fmtNum(e.price)}₩` : `$${fmtNum(e.price, 2)}`;

  return (
    <li className="flex items-center gap-2 py-1.5">
      <span className="w-[4.5rem] shrink-0 truncate text-[0.68rem] text-ink-2" title={e.tag}>
        {e.tag}
      </span>
      <span className="min-w-0 flex-1 truncate text-[0.8rem] text-ink" title={e.name}>
        {e.name}
      </span>
      <span className="tnum hidden w-16 shrink-0 text-right text-[0.7rem] text-ink-2 sm:block">
        {priceStr}
      </span>
      {/* 등락률 막대 */}
      <span className="h-1.5 w-10 shrink-0 overflow-hidden rounded-full bg-hairline">
        <span
          className={`block h-full rounded-full ${up ? "bg-gain" : "bg-loss"}`}
          style={{ width: `${barW}%` }}
        />
      </span>
      <span className={`tnum w-14 shrink-0 text-right text-[0.8rem] font-bold ${signClass(e.changePct)}`}>
        {fmtPct(e.changePct)}
      </span>
      <span className="w-4 shrink-0 text-center">
        {rel && <span className={`text-[0.62rem] font-bold ${rel.cls}`}>{rel.text}</span>}
      </span>
    </li>
  );
}

function SkeletonRows({ n }: { n: number }) {
  return (
    <div className="mt-3 space-y-2">
      {Array.from({ length: n }).map((_, i) => (
        <div key={i} className="h-6 animate-pulse rounded bg-hairline/60" />
      ))}
    </div>
  );
}
