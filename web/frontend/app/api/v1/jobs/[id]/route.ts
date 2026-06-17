// GET /api/v1/jobs/[id] — job 상태/결과 조회 (RLS: 본인 job 만).
// 프론트가 폴링하여 Loading/Processing/Success 를 추적한다.
import { NextResponse } from "next/server";

import { createServerSupabase } from "@/lib/supabase/server";
import { requireUser, UnauthorizedError, unauthorized } from "@/lib/auth-route";
import type { ApiError, JobView } from "@/lib/api-types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(
  _req: Request,
  { params }: { params: Promise<{ id: string }> }
): Promise<NextResponse<JobView | ApiError>> {
  const supabase = await createServerSupabase();
  try {
    const user = await requireUser(supabase);
    const { id } = await params;
    const { data, error } = await supabase
      .from("jobs")
      .select("id,kind,status,result,error,created_at,updated_at")
      .eq("id", id)
      .eq("user_id", user.id)
      .maybeSingle();
    if (error) return NextResponse.json({ error: error.message }, { status: 500 });
    if (!data) return NextResponse.json({ error: "작업을 찾을 수 없습니다." }, { status: 404 });

    return NextResponse.json({
      id: data.id,
      kind: data.kind as JobView["kind"],
      status: data.status as JobView["status"],
      result: data.result ?? null,
      error: data.error ?? null,
      createdAt: data.created_at,
      updatedAt: data.updated_at,
    });
  } catch (e) {
    if (e instanceof UnauthorizedError) return unauthorized();
    throw e;
  }
}
