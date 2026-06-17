// GET /api/v1/market/rates
// 기존 get_exchange_rates 대체 — exchange_rate_cache(5분 TTL) 기반.
import { NextResponse } from "next/server";

import { getRates } from "@/lib/cache";
import type { RatesResponse } from "@/lib/api-types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(): Promise<NextResponse<RatesResponse>> {
  const rates = await getRates();
  return NextResponse.json({ rates });
}
