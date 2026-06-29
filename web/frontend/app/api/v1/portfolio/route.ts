// GET  /api/v1/portfolio        보유 종목 조회 (현재가·수익률 보강)
// POST /api/v1/portfolio        추가/추가매수 (가중평균 — database.py upsert_portfolio 이식)
import { NextResponse } from "next/server";

import { createServerSupabase, createServiceClient } from "@/lib/supabase/server";
import { requireUser, UnauthorizedError, unauthorized } from "@/lib/auth-route";
import { getQuote } from "@/lib/cache";
import { resolveNames } from "@/lib/resolve-names";
import type {
  ApiError, OkResponse, PortfolioAddRequest, PortfolioItem,
} from "@/lib/api-types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(): Promise<NextResponse<PortfolioItem[] | ApiError>> {
  const supabase = await createServerSupabase();
  try {
    const user = await requireUser(supabase);
    // RLS로 본인 행만 반환되지만 명시적으로도 필터
    const { data: rows, error } = await supabase
      .from("portfolios")
      .select("id,ticker,avg_price,quantity,stop_loss,take_profit,added_at")
      .eq("user_id", user.id)
      .order("added_at", { ascending: false });
    if (error) return NextResponse.json({ error: error.message }, { status: 500 });

    // 종목명(stocks + 별칭/ETF 사전) + 현재가(캐시) 보강
    const svc = createServiceClient();
    const tickers = (rows ?? []).map((r) => r.ticker);
    const nameMap = await resolveNames(svc, tickers, { yahoo: true });

    const items: PortfolioItem[] = await Promise.all(
      (rows ?? []).map(async (r) => {
        const q = await getQuote(r.ticker);
        const price = q?.price ?? null;
        const avg = r.avg_price;
        return {
          id: r.id,
          ticker: r.ticker,
          name: nameMap.get(r.ticker) ?? null,
          avgPrice: avg,
          quantity: r.quantity,
          stopLoss: r.stop_loss ?? null,
          takeProfit: r.take_profit ?? null,
          addedAt: r.added_at,
          currentPrice: price,
          returnPct: price && avg ? (price / avg - 1) * 100 : null,
          evalAmount: price ? price * r.quantity : null,
        };
      })
    );
    return NextResponse.json(items);
  } catch (e) {
    if (e instanceof UnauthorizedError) return unauthorized();
    throw e;
  }
}

export async function POST(req: Request): Promise<NextResponse<OkResponse | ApiError>> {
  const supabase = await createServerSupabase();
  try {
    const user = await requireUser(supabase);
    const body = (await req.json()) as PortfolioAddRequest;
    const ticker = body.ticker?.toUpperCase().trim();
    const avgPrice = Number(body.avgPrice);
    const quantity = Number(body.quantity ?? 1);
    if (!ticker || !(avgPrice > 0) || !(quantity > 0)) {
      return NextResponse.json({ error: "ticker/avgPrice/quantity 가 유효하지 않습니다." }, { status: 400 });
    }
    // 손절/익절가는 선택 입력 — 양수일 때만 반영
    const stopLoss = body.stopLoss != null && Number(body.stopLoss) > 0 ? Number(body.stopLoss) : null;
    const takeProfit = body.takeProfit != null && Number(body.takeProfit) > 0 ? Number(body.takeProfit) : null;

    // 가중평균 업서트 (upsert_portfolio 이식)
    const { data: existing } = await supabase
      .from("portfolios")
      .select("id,avg_price,quantity")
      .eq("user_id", user.id)
      .eq("ticker", ticker)
      .maybeSingle();

    if (existing) {
      const newQty = existing.quantity + quantity;
      const newAvg = (existing.avg_price * existing.quantity + avgPrice * quantity) / newQty;
      // sl/tp 는 이번에 입력했을 때만 갱신(미입력 시 기존값 유지)
      const patch: { avg_price: number; quantity: number; stop_loss?: number; take_profit?: number } =
        { avg_price: newAvg, quantity: newQty };
      if (stopLoss != null) patch.stop_loss = stopLoss;
      if (takeProfit != null) patch.take_profit = takeProfit;
      const { error } = await supabase.from("portfolios").update(patch).eq("id", existing.id);
      if (error) return NextResponse.json({ error: error.message }, { status: 500 });
    } else {
      const { error } = await supabase
        .from("portfolios")
        .insert({ user_id: user.id, ticker, avg_price: avgPrice, quantity, stop_loss: stopLoss, take_profit: takeProfit });
      if (error) return NextResponse.json({ error: error.message }, { status: 500 });
    }
    return NextResponse.json({ ok: true });
  } catch (e) {
    if (e instanceof UnauthorizedError) return unauthorized();
    throw e;
  }
}
