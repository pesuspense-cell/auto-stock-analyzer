// GET /api/v1/stocks/search?q=삼성&limit=8
// 기존 resolve_ticker(한글/영문/티커 통합검색) 대체 — stocks 테이블 ilike(pg_trgm) +
// 앱 레벨 한글 별칭/ETF 사전(kr-aliases) 병합. DB 의 name_kr 공백·ETF 누락을 보강한다.
import { NextResponse } from "next/server";

import { createServiceClient } from "@/lib/supabase/server";
import { matchAliases, ALIAS_BY_TICKER } from "@/lib/kr-aliases";
import type { ApiError, StockHit, StockSearchResponse } from "@/lib/api-types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(
  req: Request
): Promise<NextResponse<StockSearchResponse | ApiError>> {
  const url = new URL(req.url);
  const q = (url.searchParams.get("q") ?? "").trim();
  const limit = Math.min(Number(url.searchParams.get("limit") ?? 8), 20);

  if (!q) return NextResponse.json({ query: q, results: [] });

  const db = createServiceClient();
  const pattern = `%${q}%`;

  // 티커/영문명/한글명 부분 일치 (DB) — 여유분을 받아 별칭 병합 후 limit 적용
  const { data, error } = await db
    .from("stocks")
    .select("ticker,name,name_kr,market,is_etf")
    .or(`ticker.ilike.${pattern},name.ilike.${pattern},name_kr.ilike.${pattern}`)
    .limit(limit * 2);

  // DB 오류여도 별칭 사전 결과(주요 미국주·ETF)는 계속 제공 — 검색이 완전히 죽지 않게.
  if (error) {
    console.warn("[stocks/search] DB 조회 실패 — 별칭 사전만 사용:", error.message);
  }

  // ── DB 결과 → StockHit, 별칭 사전으로 한글명 오버레이 ───────────────────────
  const byTicker = new Map<string, StockHit>();
  for (const s of data ?? []) {
    const alias = ALIAS_BY_TICKER.get(s.ticker.toUpperCase());
    byTicker.set(s.ticker, {
      ticker: s.ticker,
      name: s.name,
      nameKr: s.name_kr ?? alias?.nameKr ?? null,
      market: s.market,
      isEtf: s.is_etf || Boolean(alias?.isEtf),
    });
  }

  // ── 한글 별칭/ETF 매칭 병합 (DB 에 없는 ETF 는 직접 주입, 있으면 한글명 보강) ──
  for (const a of matchAliases(q)) {
    const existing = byTicker.get(a.ticker);
    if (existing) {
      if (!existing.nameKr) existing.nameKr = a.nameKr;
      if (a.isEtf) existing.isEtf = true;
    } else {
      byTicker.set(a.ticker, {
        ticker: a.ticker,
        name: a.nameEn ?? a.nameKr,
        nameKr: a.nameKr,
        market: a.market ?? null,
        isEtf: Boolean(a.isEtf),
      });
    }
  }

  // ── 랭킹: 정확 일치 > 접두 일치 > 부분 일치 (한글명·영문명·티커 통합) ──────────
  const ql = q.toLowerCase();
  const rank = (s: StockHit) => {
    const fields = [s.ticker, s.name, s.nameKr ?? ""].map((x) => x.toLowerCase());
    if (fields.some((f) => f === ql)) return 0;
    if (fields.some((f) => f.startsWith(ql))) return 1;
    return 2;
  };

  const results = [...byTicker.values()]
    .sort((a, b) => rank(a) - rank(b))
    .slice(0, limit);

  return NextResponse.json({ query: q, results });
}
