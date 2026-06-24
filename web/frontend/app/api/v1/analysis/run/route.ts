// POST /api/v1/analysis/run — 차트 분석을 jobs 큐에 pending 으로 인서트(비동기).
// 무거운 연산(주가·재무·뉴스·하이브리드)은 Python 워커(interactive 레인)가 수행한다.
import { NextResponse } from "next/server";

import { createServerSupabase } from "@/lib/supabase/server";
import { requireUser, UnauthorizedError, unauthorized } from "@/lib/auth-route";
import { readFreshCache } from "@/lib/job-cache";
import { marketState } from "@/lib/market-hours";
import type { ApiError, JobEnqueued } from "@/lib/api-types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// 분석 캐시 신선도(초) — 장중엔 짧게(가격·신호 변동 반영), 장외엔 길게(자료 불변 → 재분석 불필요).
function analysisCacheTtlSec(ticker: string): number {
  switch (marketState(ticker)) {
    case "open": return 180; // 장중 3분
    case "closed-weekday": return 6 * 3600; // 장외 평일 6시간
    case "closed-weekend": return 24 * 3600; // 주말 24시간
  }
}

export async function POST(req: Request): Promise<NextResponse<JobEnqueued | ApiError>> {
  const supabase = await createServerSupabase();
  try {
    const user = await requireUser(supabase);
    const body = await req.json().catch(() => ({}));
    const ticker = String(body?.ticker ?? "").trim();
    if (!ticker) return NextResponse.json({ error: "ticker가 필요합니다." }, { status: 400 });
    const params = {
      ticker,
      period: String(body?.period ?? "6mo"),
      useLlm: Boolean(body?.use_llm ?? body?.useLlm ?? false),
    };
    const force = Boolean(body?.force ?? false);

    // 지속 캐시(market_cache) 가 신선하면 워커 큐 없이 기존 분석을 즉시 반환(리소스 절감).
    // "다시 분석"(force=true) 은 캐시를 우회해 강제 재분석한다.
    // scope 형식은 워커 _cache_scope(analysis) 와 반드시 일치해야 한다.
    if (!force) {
      const scope = `analysis:${ticker.toUpperCase()}:${params.period}:${params.useLlm ? "llm" : "base"}`;
      const cached = await readFreshCache(supabase, scope, analysisCacheTtlSec(ticker));
      if (cached) return NextResponse.json({ jobId: null, status: "completed", result: cached });
    }

    const { data, error } = await supabase
      .from("jobs")
      .insert({ user_id: user.id, kind: "analysis", status: "pending", params })
      .select("id,status")
      .single();
    if (error || !data) {
      return NextResponse.json({ error: error?.message ?? "큐 등록 실패" }, { status: 500 });
    }
    return NextResponse.json({ jobId: data.id, status: data.status as JobEnqueued["status"] });
  } catch (e) {
    if (e instanceof UnauthorizedError) return unauthorized();
    throw e;
  }
}
