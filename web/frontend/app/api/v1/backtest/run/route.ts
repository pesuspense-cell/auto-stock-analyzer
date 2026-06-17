// POST /api/v1/backtest/run — 백테스트 작업을 jobs 큐에 pending 으로 인서트.
import { NextResponse } from "next/server";

import { createServerSupabase } from "@/lib/supabase/server";
import { requireUser, UnauthorizedError, unauthorized } from "@/lib/auth-route";
import type { ApiError, JobEnqueued } from "@/lib/api-types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

interface BacktestParams {
  markets: string[];
  universe_n: number;
  top_n: number;
  initial_capital: number;
  start_date: string;
  end_date: string;
  deposit_schedule: Record<string, number>;
  benchmark_ticker: string | null;
  benchmark_label: string;
}

export async function POST(req: Request): Promise<NextResponse<JobEnqueued | ApiError>> {
  const supabase = await createServerSupabase();
  try {
    const user = await requireUser(supabase);
    const p = (await req.json()) as Partial<BacktestParams>;

    const params: BacktestParams = {
      markets: p.markets ?? ["KOSPI", "KOSDAQ"],
      universe_n: p.universe_n ?? 200,
      top_n: p.top_n ?? 20,
      initial_capital: p.initial_capital ?? 10_000_000,
      start_date: p.start_date ?? "2020-01-01",
      end_date: p.end_date ?? "2024-12-31",
      deposit_schedule: p.deposit_schedule ?? {},
      benchmark_ticker: p.benchmark_ticker ?? "^KS11",
      benchmark_label: p.benchmark_label ?? "^KS11 (KOSPI)",
    };
    if (!params.markets.length) {
      return NextResponse.json({ error: "마켓을 하나 이상 선택하세요." }, { status: 400 });
    }

    const { data, error } = await supabase
      .from("jobs")
      .insert({ user_id: user.id, kind: "backtest", status: "pending", params })
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
