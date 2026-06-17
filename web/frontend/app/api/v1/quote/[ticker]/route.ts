// GET /api/v1/quote/[ticker]
// 기존 _realtime_price_1m 대체 — Supabase 캐시(장중 3초/장외 60초) → 만료 시 Yahoo 갱신.
import { NextResponse } from "next/server";

import { getQuote } from "@/lib/cache";
import type { ApiError, QuoteResponse } from "@/lib/api-types";

export const runtime = "nodejs"; // service_role 키 사용 → Edge 아닌 Node 런타임
export const dynamic = "force-dynamic";

export async function GET(
  _req: Request,
  { params }: { params: Promise<{ ticker: string }> }
): Promise<NextResponse<QuoteResponse | ApiError>> {
  const { ticker } = await params;
  if (!ticker) {
    return NextResponse.json({ error: "ticker가 필요합니다." }, { status: 400 });
  }

  const q = await getQuote(ticker.toUpperCase());
  if (!q) {
    return NextResponse.json(
      { error: `'${ticker}' 시세를 불러올 수 없습니다.` },
      { status: 404 }
    );
  }

  const body: QuoteResponse = {
    ticker: q.ticker,
    price: q.price,
    prevClose: q.prevClose,
    changePct: q.changePct,
    volume: q.volume,
    currency: q.currency,
    isRealtime: q.isRealtime,
    fetchedAt: q.fetchedAt,
    cached: q.cached,
  };
  return NextResponse.json(body);
}
