// GET /api/v1/stocks/search?q=삼성&limit=8
// 기존 resolve_ticker(한글/영문/티커 통합검색) 대체 — stocks 테이블 ilike(pg_trgm).
import { NextResponse } from "next/server";

import { createServiceClient } from "@/lib/supabase/server";
import type { ApiError, StockHit, StockSearchResponse } from "@/lib/api-types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(
  req: Request
): Promise<NextResponse<StockSearchResponse | ApiError>> {
  const url = new URL(req.url);
  const q = (url.searchParams.get("q") ?? "").trim();
  const limit = Math.min(Number(url.searchParams.get("limit") ?? 8), 20);

  if (!q) return NextResponse.json({ query: q, results: [] });

  const db = createServiceClient();
  const pattern = `%${q}%`;

  // 티커 정확/접두 일치 우선, 이후 name·name_kr 부분 일치
  const { data, error } = await db
    .from("stocks")
    .select("ticker,name,name_kr,market,is_etf")
    .or(`ticker.ilike.${pattern},name.ilike.${pattern},name_kr.ilike.${pattern}`)
    .limit(limit);

  if (error) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }

  // 정확 일치를 상단으로 정렬 (간이 랭킹 — resolve_ticker 우선순위 근사)
  const ql = q.toLowerCase();
  const rank = (s: { ticker: string; name: string; name_kr: string | null }) => {
    const t = s.ticker.toLowerCase();
    const n = s.name.toLowerCase();
    const k = (s.name_kr ?? "").toLowerCase();
    if (t === ql || n === ql || k === ql) return 0;
    if (t.startsWith(ql) || n.startsWith(ql) || k.startsWith(ql)) return 1;
    return 2;
  };

  const results: StockHit[] = (data ?? [])
    .sort((a, b) => rank(a) - rank(b))
    .map((s) => ({
      ticker: s.ticker,
      name: s.name,
      nameKr: s.name_kr,
      market: s.market,
      isEtf: s.is_etf,
    }));

  return NextResponse.json({ query: q, results });
}
