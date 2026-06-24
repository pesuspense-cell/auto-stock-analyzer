// POST /api/v1/news/run — 뉴스·감성·섹터 수집을 jobs 큐에 인서트(비동기).
// 무거운 수집/LLM 연산은 Python 워커(interactive 레인)가 수행한다.
import { NextResponse } from "next/server";

import { createServerSupabase } from "@/lib/supabase/server";
import { requireUser, UnauthorizedError, unauthorized } from "@/lib/auth-route";
import { readFreshCache } from "@/lib/job-cache";
import type { ApiError, JobEnqueued } from "@/lib/api-types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(req: Request): Promise<NextResponse<JobEnqueued | ApiError>> {
  const supabase = await createServerSupabase();
  try {
    const user = await requireUser(supabase);
    const body = await req.json().catch(() => ({}));
    const ticker = String(body?.ticker ?? "").trim();
    if (!ticker) return NextResponse.json({ error: "ticker가 필요합니다." }, { status: 400 });

    // 지속 캐시(market_cache) 신선하면 워커 큐 없이 즉시응답 (뉴스 TTL 1일 — 하루 1회 갱신)
    const cached = await readFreshCache(supabase, `news:${ticker.toUpperCase()}`, 86400);
    if (cached) return NextResponse.json({ jobId: null, status: "completed", result: cached });

    const { data, error } = await supabase
      .from("jobs")
      .insert({ user_id: user.id, kind: "news", status: "pending", params: { ticker } })
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
