// 티커 → 표시명 일괄 해소 (서버 전용).
//   1) stocks 테이블  2) 한글 별칭/ETF 사전  3) (옵션) Yahoo longName/shortName 동적 해소
// 포트폴리오·매도이력 등에서 ETF/비주류 종목이 티커로만 보이던 문제를 보강한다.
// Yahoo 해소 결과는 market_cache(scope="name:TICKER")에 장기 캐시해 재호출을 막는다.
import type { SupabaseClient } from "@supabase/supabase-js";

import { ALIAS_BY_TICKER } from "@/lib/kr-aliases";
import { fetchName } from "@/lib/providers/yahoo";

const NAME_TTL = 30 * 24 * 3600; // 30일 — 종목명은 거의 안 바뀜
const isFresh = (fetchedAt: string, ttl: number) =>
  Date.now() - new Date(fetchedAt).getTime() < ttl * 1000;

/**
 * 주어진 티커들의 표시명 Map.
 * @param svc service-role 클라이언트 (stocks 조회 + name 캐시 쓰기)
 * @param yahoo true 면 stocks/별칭에 없는 티커를 Yahoo 로 해소(+캐시). 기본 false.
 */
export async function resolveNames(
  svc: SupabaseClient,
  tickers: string[],
  { yahoo = false }: { yahoo?: boolean } = {}
): Promise<Map<string, string>> {
  const map = new Map<string, string>();
  const uniq = [...new Set(tickers.filter(Boolean))];
  if (uniq.length === 0) return map;

  // 1) stocks 테이블
  try {
    const { data } = await svc.from("stocks").select("ticker,name").in("ticker", uniq);
    (data ?? []).forEach((n) => {
      if (n.name) map.set(n.ticker, n.name);
    });
  } catch {
    /* DB 실패해도 별칭/Yahoo 폴백으로 진행 */
  }

  // 2) 한글 별칭/ETF 사전
  for (const t of uniq) {
    if (!map.has(t)) {
      const alias = ALIAS_BY_TICKER.get(t.toUpperCase());
      if (alias) map.set(t, alias.nameKr);
    }
  }

  // 3) Yahoo 동적 해소 (옵션) — 남은 미해소 티커
  if (yahoo) {
    const missing = uniq.filter((t) => !map.has(t));
    if (missing.length > 0) {
      const scopeOf = (t: string) => `name:${t.toUpperCase()}`;
      // 3-1) name 캐시 조회
      const toFetch: string[] = [];
      try {
        const { data: cached } = await svc
          .from("market_cache")
          .select("scope,payload,fetched_at")
          .in("scope", missing.map(scopeOf));
        const byScope = new Map((cached ?? []).map((r) => [r.scope, r]));
        for (const t of missing) {
          const row = byScope.get(scopeOf(t));
          const name = (row?.payload as { name?: string } | undefined)?.name;
          if (row && name && isFresh(row.fetched_at, NAME_TTL)) map.set(t, name);
          else toFetch.push(t);
        }
      } catch {
        toFetch.push(...missing);
      }

      // 3-2) Yahoo 병렬 조회 → 캐시 upsert
      if (toFetch.length > 0) {
        const fetched = await Promise.all(
          toFetch.map(async (t) => ({ t, name: await fetchName(t) }))
        );
        const now = new Date().toISOString();
        const upserts: { scope: string; payload: { name: string }; fetched_at: string }[] = [];
        for (const { t, name } of fetched) {
          if (name) {
            map.set(t, name);
            upserts.push({ scope: scopeOf(t), payload: { name }, fetched_at: now });
          }
        }
        if (upserts.length > 0) {
          try {
            await svc.from("market_cache").upsert(upserts);
          } catch {
            /* 캐시 저장 실패는 무시 */
          }
        }
      }
    }
  }

  return map;
}
