// POST /api/v1/portfolio/analyze — 보유 종목 진단/매매 지침을 jobs 큐에 적재(비동기).
// 보유목록 + 현재가 + 종목명을 서버에서 모아 params 로 넘기고, 워커가 섹터/모멘텀/
// 리밸런싱 가이드를 계산한다(interactive 레인).
import { NextResponse } from "next/server";

import { createServerSupabase, createServiceClient } from "@/lib/supabase/server";
import { requireUser, UnauthorizedError, unauthorized } from "@/lib/auth-route";
import { getQuote } from "@/lib/cache";
import { resolveNames } from "@/lib/resolve-names";
import type { ApiError, JobEnqueued } from "@/lib/api-types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(): Promise<NextResponse<JobEnqueued | ApiError>> {
  const supabase = await createServerSupabase();
  try {
    const user = await requireUser(supabase);

    const { data: rows, error } = await supabase
      .from("portfolios")
      .select("ticker,avg_price,quantity")
      .eq("user_id", user.id);
    if (error) return NextResponse.json({ error: error.message }, { status: 500 });
    if (!rows || rows.length === 0) {
      return NextResponse.json({ error: "보유 종목이 없습니다." }, { status: 400 });
    }

    const svc = createServiceClient();
    const nameMap = await resolveNames(svc, rows.map((r) => r.ticker), { yahoo: true });

    // 현재가 병렬 조회 후 분석 입력 구성
    const items = await Promise.all(
      rows.map(async (r) => {
        const q = await getQuote(r.ticker);
        return {
          ticker: r.ticker,
          avgPrice: r.avg_price,
          quantity: r.quantity,
          price: q?.price ?? null,
          name: nameMap.get(r.ticker) ?? r.ticker,
        };
      })
    );

    const { data, error: jErr } = await supabase
      .from("jobs")
      .insert({ user_id: user.id, kind: "portfolio_analysis", status: "pending", params: { items } })
      .select("id,status")
      .single();
    if (jErr || !data) {
      return NextResponse.json({ error: jErr?.message ?? "큐 등록 실패" }, { status: 500 });
    }
    return NextResponse.json({ jobId: data.id, status: data.status as JobEnqueued["status"] });
  } catch (e) {
    if (e instanceof UnauthorizedError) return unauthorized();
    throw e;
  }
}
