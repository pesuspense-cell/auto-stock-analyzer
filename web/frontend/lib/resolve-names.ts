// 티커 → 표시명 일괄 해소 (서버 전용). stocks 테이블 우선, 없으면 한글 별칭/ETF 사전 폴백.
// 포트폴리오·매도이력 등에서 ETF 가 티커로만 보이던 문제를 보강한다.
import type { SupabaseClient } from "@supabase/supabase-js";

import { ALIAS_BY_TICKER } from "@/lib/kr-aliases";

/** 주어진 티커들의 표시명 Map. (stocks.name → 없으면 별칭 nameKr) */
export async function resolveNames(
  svc: SupabaseClient,
  tickers: string[]
): Promise<Map<string, string>> {
  const map = new Map<string, string>();
  const uniq = [...new Set(tickers.filter(Boolean))];
  if (uniq.length === 0) return map;

  try {
    const { data } = await svc.from("stocks").select("ticker,name").in("ticker", uniq);
    (data ?? []).forEach((n) => {
      if (n.name) map.set(n.ticker, n.name);
    });
  } catch {
    /* DB 실패해도 별칭 폴백으로 진행 */
  }

  for (const t of uniq) {
    if (!map.has(t)) {
      const alias = ALIAS_BY_TICKER.get(t.toUpperCase());
      if (alias) map.set(t, alias.nameKr);
    }
  }
  return map;
}
