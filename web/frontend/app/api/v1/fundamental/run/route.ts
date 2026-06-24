// POST /api/v1/fundamental/run — 펀더멘털·수급·기관 수집을 jobs 큐에 인서트(비동기).
// 무거운 KRX/DART/yfinance 연산은 Python 워커(interactive 레인)가 수행한다.
import { NextResponse } from "next/server";

import { createServerSupabase } from "@/lib/supabase/server";
import { requireUser, UnauthorizedError, unauthorized } from "@/lib/auth-route";
import { readFreshCache } from "@/lib/job-cache";
import type { ApiError, JobEnqueued } from "@/lib/api-types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// 현재 분기 키(KST 기준, 예: 2026Q2). 분기가 바뀌면 scope 가 달라져 캐시가 자동 무효화된다.
// scope 형식은 워커 _cache_scope(fundamental) 와 반드시 일치해야 한다.
function quarterKey(date = new Date()): string {
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone: "Asia/Seoul", year: "numeric", month: "numeric",
  }).formatToParts(date);
  let y = 0, m = 1;
  for (const p of parts) {
    if (p.type === "year") y = parseInt(p.value, 10);
    else if (p.type === "month") m = parseInt(p.value, 10);
  }
  return `${y}Q${Math.floor((m - 1) / 3) + 1}`;
}

export async function POST(req: Request): Promise<NextResponse<JobEnqueued | ApiError>> {
  const supabase = await createServerSupabase();
  try {
    const user = await requireUser(supabase);
    const body = await req.json().catch(() => ({}));
    const ticker = String(body?.ticker ?? "").trim();
    if (!ticker) return NextResponse.json({ error: "ticker가 필요합니다." }, { status: 400 });

    // 펀더멘털은 분기 단위로만 의미있게 변하므로 분기 키를 scope 에 넣어 분기당 1회만 갱신한다.
    // (분기 경계 전까지는 항상 캐시 적중 → 워커 큐·재수집 없이 즉시응답). TTL 은 안전 상한.
    const scope = `fundamental:${ticker.toUpperCase()}:${quarterKey()}`;
    const cached = await readFreshCache(supabase, scope, 100 * 24 * 3600);
    if (cached) return NextResponse.json({ jobId: null, status: "completed", result: cached });

    const { data, error } = await supabase
      .from("jobs")
      .insert({ user_id: user.id, kind: "fundamental", status: "pending", params: { ticker } })
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
