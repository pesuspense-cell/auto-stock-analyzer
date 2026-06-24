// POST /api/v1/investors/run — 투자자별 매매동향(수급) 수집을 jobs 큐에 인서트(비동기).
// 분기 캐시되는 fundamental 과 분리 — 수급은 매일 변하므로 일 단위 scope + 짧은 TTL 로 갱신.
import { NextResponse } from "next/server";

import { createServerSupabase } from "@/lib/supabase/server";
import { requireUser, UnauthorizedError, unauthorized } from "@/lib/auth-route";
import { readFreshCache } from "@/lib/job-cache";
import type { ApiError, JobEnqueued } from "@/lib/api-types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// KST 날짜 키(YYYYMMDD). 워커 _cache_scope(investors) 와 형식을 반드시 일치시킬 것.
function dateKey(date = new Date()): string {
  const parts = new Intl.DateTimeFormat("en-CA", {
    timeZone: "Asia/Seoul", year: "numeric", month: "2-digit", day: "2-digit",
  }).formatToParts(date);
  let y = "", m = "", d = "";
  for (const p of parts) {
    if (p.type === "year") y = p.value;
    else if (p.type === "month") m = p.value;
    else if (p.type === "day") d = p.value;
  }
  return `${y}${m}${d}`;
}

export async function POST(req: Request): Promise<NextResponse<JobEnqueued | ApiError>> {
  const supabase = await createServerSupabase();
  try {
    const user = await requireUser(supabase);
    const body = await req.json().catch(() => ({}));
    const ticker = String(body?.ticker ?? "").trim();
    if (!ticker) return NextResponse.json({ error: "ticker가 필요합니다." }, { status: 400 });

    // 일 단위 scope + 1시간 TTL — 장중 몇 차례 갱신되되 대부분 캐시 적중.
    const scope = `investors:${ticker.toUpperCase()}:${dateKey()}`;
    const cached = await readFreshCache(supabase, scope, 3600);
    if (cached) return NextResponse.json({ jobId: null, status: "completed", result: cached });

    const { data, error } = await supabase
      .from("jobs")
      .insert({ user_id: user.id, kind: "investors", status: "pending", params: { ticker } })
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
