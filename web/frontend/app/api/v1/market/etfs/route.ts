// GET /api/v1/market/etfs
// 주요 섹터 ETF(미국 15 + 국내 20)의 일간 등락률 비교 — 10분 캐시(market_cache).
import { NextResponse } from "next/server";

import { getSectorEtfs } from "@/lib/sector-etfs";
import type { EtfsResponse } from "@/lib/api-types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(): Promise<NextResponse<EtfsResponse>> {
  const data = await getSectorEtfs();
  return NextResponse.json(data);
}
