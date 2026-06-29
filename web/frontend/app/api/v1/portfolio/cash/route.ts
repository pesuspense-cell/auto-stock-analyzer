// GET /api/v1/portfolio/cash   예수금(현금 잔고) 조회
// PUT /api/v1/portfolio/cash   예수금 설정/수정 (user_settings.cash_balance)
import { NextResponse } from "next/server";

import { createServerSupabase } from "@/lib/supabase/server";
import { requireUser, UnauthorizedError, unauthorized } from "@/lib/auth-route";
import type { ApiError, CashBalance, CashUpdateRequest } from "@/lib/api-types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(): Promise<NextResponse<CashBalance | ApiError>> {
  const supabase = await createServerSupabase();
  try {
    const user = await requireUser(supabase);
    const { data, error } = await supabase
      .from("user_settings")
      .select("cash_balance")
      .eq("user_id", user.id)
      .maybeSingle();
    if (error) return NextResponse.json({ error: error.message }, { status: 500 });
    return NextResponse.json({ cashBalance: data?.cash_balance ?? 0 });
  } catch (e) {
    if (e instanceof UnauthorizedError) return unauthorized();
    throw e;
  }
}

export async function PUT(req: Request): Promise<NextResponse<CashBalance | ApiError>> {
  const supabase = await createServerSupabase();
  try {
    const user = await requireUser(supabase);
    const body = (await req.json()) as CashUpdateRequest;
    const cash = Number(body.cashBalance);
    if (!Number.isFinite(cash) || cash < 0) {
      return NextResponse.json({ error: "예수금은 0 이상의 숫자여야 합니다." }, { status: 400 });
    }
    // user_settings 행은 가입 트리거(handle_new_user)로 생성되지만, 누락 가능성 방어 위해 upsert
    const { error } = await supabase
      .from("user_settings")
      .upsert(
        { user_id: user.id, cash_balance: cash, updated_at: new Date().toISOString() },
        { onConflict: "user_id" },
      );
    if (error) return NextResponse.json({ error: error.message }, { status: 500 });
    return NextResponse.json({ cashBalance: cash });
  } catch (e) {
    if (e instanceof UnauthorizedError) return unauthorized();
    throw e;
  }
}
