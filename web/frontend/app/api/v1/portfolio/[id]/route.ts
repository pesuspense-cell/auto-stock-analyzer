// DELETE /api/v1/portfolio/[id]   보유 종목 삭제
// PATCH  /api/v1/portfolio/[id]   손절가/익절가 수정 (null=해제 → 봇이 ATR로 산출)
import { NextResponse } from "next/server";

import { createServerSupabase } from "@/lib/supabase/server";
import { requireUser, UnauthorizedError, unauthorized } from "@/lib/auth-route";
import type { ApiError, OkResponse, PortfolioUpdateRequest } from "@/lib/api-types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function PATCH(
  req: Request,
  { params }: { params: Promise<{ id: string }> }
): Promise<NextResponse<OkResponse | ApiError>> {
  const supabase = await createServerSupabase();
  try {
    const user = await requireUser(supabase);
    const { id } = await params;
    const body = (await req.json()) as PortfolioUpdateRequest;

    // 손절/익절가: 양수 → 설정, null/0 → 해제(봇 ATR 산출). 둘 다 미지정이면 변경 없음.
    const patch: { stop_loss?: number | null; take_profit?: number | null } = {};
    if ("stopLoss" in body) {
      const v = body.stopLoss == null ? null : Number(body.stopLoss);
      patch.stop_loss = v != null && v > 0 ? v : null;
    }
    if ("takeProfit" in body) {
      const v = body.takeProfit == null ? null : Number(body.takeProfit);
      patch.take_profit = v != null && v > 0 ? v : null;
    }
    if (Object.keys(patch).length === 0) {
      return NextResponse.json({ error: "변경할 손절가/익절가가 없습니다." }, { status: 400 });
    }

    const { error } = await supabase
      .from("portfolios")
      .update(patch)
      .eq("id", id)
      .eq("user_id", user.id);
    if (error) return NextResponse.json({ error: error.message }, { status: 500 });
    return NextResponse.json({ ok: true });
  } catch (e) {
    if (e instanceof UnauthorizedError) return unauthorized();
    throw e;
  }
}

export async function DELETE(
  _req: Request,
  { params }: { params: Promise<{ id: string }> }
): Promise<NextResponse<OkResponse | ApiError>> {
  const supabase = await createServerSupabase();
  try {
    const user = await requireUser(supabase);
    const { id } = await params;
    const { error } = await supabase
      .from("portfolios")
      .delete()
      .eq("id", id)
      .eq("user_id", user.id);
    if (error) return NextResponse.json({ error: error.message }, { status: 500 });
    return NextResponse.json({ ok: true });
  } catch (e) {
    if (e instanceof UnauthorizedError) return unauthorized();
    throw e;
  }
}
