// POST /api/v1/portfolio/[id]/sell   매도 (database.py sell_item 이식)
//   순손익·수익률 계산 → trade_history 기록 → 수량 차감(또는 삭제)
import { NextResponse } from "next/server";

import { createServerSupabase } from "@/lib/supabase/server";
import { requireUser, UnauthorizedError, unauthorized } from "@/lib/auth-route";
import type { ApiError, SellRequest, SellResponse } from "@/lib/api-types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(
  req: Request,
  { params }: { params: Promise<{ id: string }> }
): Promise<NextResponse<SellResponse | ApiError>> {
  const supabase = await createServerSupabase();
  try {
    const user = await requireUser(supabase);
    const { id } = await params;
    const body = (await req.json()) as SellRequest;
    const sellPrice = Number(body.sellPrice);
    if (!(sellPrice > 0)) {
      return NextResponse.json({ error: "sellPrice 가 유효하지 않습니다." }, { status: 400 });
    }

    const { data: row } = await supabase
      .from("portfolios")
      .select("ticker,avg_price,quantity")
      .eq("id", id)
      .eq("user_id", user.id)
      .maybeSingle();
    if (!row) {
      return NextResponse.json({ ok: false, error: "항목을 찾을 수 없습니다." }, { status: 404 });
    }

    const buyPrice = row.avg_price;
    const totalQty = row.quantity;
    const sellQty = body.quantity != null ? Math.min(Number(body.quantity), totalQty) : totalQty;
    if (!(sellQty > 0)) {
      return NextResponse.json({ ok: false, error: "매도 수량이 0 이하입니다." }, { status: 400 });
    }

    const netProfit = (sellPrice - buyPrice) * sellQty;
    const returnRate = buyPrice ? (sellPrice / buyPrice - 1) * 100 : 0;

    const { error: tradeErr } = await supabase.from("trade_history").insert({
      user_id: user.id, ticker: row.ticker, buy_price: buyPrice, sell_price: sellPrice,
      quantity: sellQty, net_profit: netProfit, return_rate: returnRate,
    });
    if (tradeErr) return NextResponse.json({ error: tradeErr.message }, { status: 500 });

    const remaining = totalQty - sellQty;
    if (remaining <= 0.001) {
      await supabase.from("portfolios").delete().eq("id", id).eq("user_id", user.id);
    } else {
      await supabase.from("portfolios").update({ quantity: remaining }).eq("id", id).eq("user_id", user.id);
    }

    return NextResponse.json({ ok: true, netProfit, returnRate });
  } catch (e) {
    if (e instanceof UnauthorizedError) return unauthorized();
    throw e;
  }
}
