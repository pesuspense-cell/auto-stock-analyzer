// GET /api/v1/indicators/[ticker]
// 레거시 stock_ai.generate_signals + _add_indicators 의 TS 포팅을 서버에서 실행.
// OHLCV(Yahoo) → 지표/신호 계산 → market_cache(장중 60초 / 장외 30분 TTL) → 타입 JSON.
import { NextResponse } from "next/server";

import { fetchOhlc, type RawOhlc } from "@/lib/providers/yahoo";
import { computeIndicators, generateSignals } from "@/lib/indicators/signals";
import * as ta from "@/lib/indicators/ta";
import { marketState } from "@/lib/market-hours";
import { createServiceClient } from "@/lib/supabase/server";
import type { ApiError, IndicatorsResponse, ChartSeries } from "@/lib/api-types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const ttlSeconds = (ticker: string) => (marketState(ticker) === "open" ? 60 : 1800);
const isFresh = (fetchedAt: string, ttl: number) => Date.now() - new Date(fetchedAt).getTime() < ttl * 1000;

const CHART_BARS = 160; // 차트에 노출할 최근 봉 수 (≈ 7개월 일봉)

/** OHLC + EMA(20/50/200) → 차트 시리즈(최근 CHART_BARS봉). */
function buildChart(o: RawOhlc, currency: string | null): ChartSeries {
  const ema20 = ta.ewm(o.close, 20);
  const ema50 = ta.ewm(o.close, 50);
  const ema200 = o.close.length >= 200 ? ta.ewm(o.close, 200) : o.close.map(() => Number.NaN);
  const n = o.close.length;
  const start = Math.max(0, n - CHART_BARS);

  const candles: ChartSeries["candles"] = [];
  const e20: ChartSeries["ema20"] = [], e50: ChartSeries["ema50"] = [], e200: ChartSeries["ema200"] = [];
  for (let i = start; i < n; i++) {
    const time = o.date[i];
    candles.push({ time, open: o.open[i], high: o.high[i], low: o.low[i], close: o.close[i], volume: o.volume[i] });
    if (!Number.isNaN(ema20[i])) e20.push({ time, value: ema20[i] });
    if (!Number.isNaN(ema50[i])) e50.push({ time, value: ema50[i] });
    if (!Number.isNaN(ema200[i])) e200.push({ time, value: ema200[i] });
  }
  return { candles, ema20: e20, ema50: e50, ema200: e200, currency };
}

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
  const isKr = ticker.endsWith(".KS") || ticker.endsWith(".KQ");
  const chart = buildChart(ohlc, isKr ? "KRW" : "USD");
  const payload = { asOf, indicators, signal, chart };

  // 3) 캐시 저장 (실패 무시)
  try {
    await db.from("market_cache").upsert({ scope, payload, fetched_at: new Date().toISOString() });
  } catch {
    /* noop */
  }

  return NextResponse.json({ ticker, ...payload, cached: false });
}
