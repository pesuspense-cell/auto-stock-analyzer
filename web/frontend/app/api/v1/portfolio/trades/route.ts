// GET /api/v1/portfolio/trades   매도 이력
import { NextResponse } from "next/server";

import { createServerSupabase } from "@/lib/supabase/server";
import { requireUser, UnauthorizedError, unauthorized } from "@/lib/auth-route";
import type { ApiError, TradeItem } from "@/lib/api-types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(): Promise<NextResponse<TradeItem[] | ApiError>> {
  const supabase = await createServerSupabase();
  try {
    const user = await requireUser(supabase);
    const { data: rows, error } = await supabase
      .from("trade_history")
      .select("id,ticker,buy_price,sell_price,quantity,net_profit,return_rate,traded_at")
      .eq("user_id", user.id)
      .order("traded_at", { ascending: false })
      .limit(100);
    if (error) return NextResponse.json({ error: error.message }, { status: 500 });

    const items: TradeItem[] = (rows ?? []).map((r) => ({
      id: r.id, ticker: r.ticker, buyPrice: r.buy_price, sellPrice: r.sell_price,
      quantity: r.quantity, netProfit: r.net_profit, returnRate: r.return_rate, tradedAt: r.traded_at,
    }));
    return NextResponse.json(items);
  } catch (e) {
    if (e instanceof UnauthorizedError) return unauthorized();
    throw e;
  }
}
