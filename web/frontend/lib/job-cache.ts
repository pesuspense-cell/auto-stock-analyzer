// 워커가 기록한 지속 캐시(market_cache) 읽기 — enqueue 단계 즉시응답용.
// 워커는 news/fundamental 결과를 scope="news:TICKER" / "fundamental:TICKER" 로 upsert 한다.
import type { SupabaseClient } from "@supabase/supabase-js";

/** scope 의 캐시가 TTL 내면 payload 반환, 아니면 null. (market_cache 는 public read) */
export async function readFreshCache(
  supabase: SupabaseClient,
  scope: string,
  ttlSeconds: number
): Promise<unknown | null> {
  try {
    const { data } = await supabase
      .from("market_cache")
      .select("payload,fetched_at")
      .eq("scope", scope)
      .maybeSingle();
    if (data && Date.now() - new Date(data.fetched_at).getTime() < ttlSeconds * 1000) {
      return data.payload;
    }
  } catch {
    /* 캐시 조회 실패는 무시 → 정상 큐 경로로 폴백 */
  }
  return null;
}
