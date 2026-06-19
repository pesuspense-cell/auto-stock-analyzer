// POST /api/v1/analysis/run — 차트 분석을 jobs 큐에 pending 으로 인서트(비동기).
// 무거운 연산(주가·재무·뉴스·하이브리드)은 Python 워커(interactive 레인)가 수행한다.
import { NextResponse } from "next/server";

import { createServerSupabase } from "@/lib/supabase/server";
import { requireUser, UnauthorizedError, unauthorized } from "@/lib/auth-route";
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
    const params = {
      ticker,
      period: String(body?.period ?? "6mo"),
      useLlm: Boolean(body?.use_llm ?? body?.useLlm ?? false),
    };

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
