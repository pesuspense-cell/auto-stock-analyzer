// GET /api/v1/indicators/[ticker]
// 레거시 stock_ai.generate_signals + _add_indicators 의 TS 포팅을 서버에서 실행.
// OHLCV(Yahoo) → 지표/신호 계산 → market_cache(장중 60초 / 장외 30분 TTL) → 타입 JSON.
import { NextResponse } from "next/server";

import { fetchOhlc } from "@/lib/providers/yahoo";
import { computeIndicators, generateSignals } from "@/lib/indicators/signals";
import { marketState } from "@/lib/market-hours";
import { createServiceClient } from "@/lib/supabase/server";
import type { ApiError, IndicatorsResponse } from "@/lib/api-types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const ttlSeconds = (ticker: string) => (marketState(ticker) === "open" ? 60 : 1800);
const isFresh = (fetchedAt: string, ttl: number) => Date.now() - new Date(fetchedAt).getTime() < ttl * 1000;

export async function GET(
  _req: Request,
  { params }: { params: Promise<{ ticker: string }> }
): Promise<NextResponse<IndicatorsResponse | ApiError>> {
  const { ticker: raw } = await params;
  const ticker = raw.toUpperCase();
  const scope = `indicators:${ticker}`;
  const db = createServiceClient();

  // 1) 캐시 조회 (실패는 무시 → stale-if-error)
  try {
    const { data: row } = await db.from("market_cache").select("payload,fetched_at").eq("scope", scope).maybeSingle();
    if (row && isFresh(row.fetched_at, ttlSeconds(ticker))) {
      const p = row.payload as Omit<IndicatorsResponse, "ticker" | "cached">;
      return NextResponse.json({ ticker, ...p, cached: true });
    }
  } catch {
    /* noop */
  }

  // 2) 외부 OHLC → 지표/신호 계산
  const ohlc = await fetchOhlc(ticker, "2y", "1d");
  if (!ohlc) {
    return NextResponse.json({ error: `'${ticker}' OHLC 데이터를 불러올 수 없습니다.` }, { status: 404 });
  }
  const indicators = computeIndicators(ohlc);
  const signal = generateSignals(ohlc);
  const asOf = ohlc.date[ohlc.date.length - 1];
  const payload = { asOf, indicators, signal };

  // 3) 캐시 저장 (실패 무시)
  try {
    await db.from("market_cache").upsert({ scope, payload, fetched_at: new Date().toISOString() });
  } catch {
    /* noop */
  }

  return NextResponse.json({ ticker, ...payload, cached: false });
}
