// Supabase 캐싱 레이어 — "읽기 시점 신선도 판정 → 만료 시 외부 갱신 후 upsert".
// 기존 @st.cache_data(ttl=...) 의 TTL 캐시를 DB 기반으로 대체한다.
//
// 핵심 전략(요청 사항):
//   · 장중(KST 09:00–15:30) 시세는 3초 TTL → 짧은 시간 내 재요청은 DB만 조회
//   · 장외/주말은 60초 TTL
//   · 만료 시에만 Yahoo 호출 → quote_cache upsert
import { createServiceClient } from "@/lib/supabase/server";
import { fetchQuote, fetchRate } from "@/lib/providers/yahoo";

const KST_OFFSET = 9 * 60; // minutes

function isKoreanMarketOpen(d = new Date()): boolean {
  const utcMin = d.getUTCHours() * 60 + d.getUTCMinutes();
  const kstMin = (utcMin + KST_OFFSET) % (24 * 60);
  const kstDay = (d.getUTCDay() + (utcMin + KST_OFFSET >= 24 * 60 ? 1 : 0)) % 7;
  const open = 9 * 60;
  const close = 15 * 60 + 30;
  return kstDay >= 1 && kstDay <= 5 && kstMin >= open && kstMin < close;
}

function quoteTtlSeconds(ticker: string): number {
  const isKr = ticker.endsWith(".KS") || ticker.endsWith(".KQ");
  if (isKr && isKoreanMarketOpen()) return 3; // 장중 3초
  return 60; // 장외/미국장 등 60초
}

function isFresh(fetchedAt: string, ttlSeconds: number): boolean {
  return Date.now() - new Date(fetchedAt).getTime() < ttlSeconds * 1000;
}

export interface CachedQuote {
  ticker: string;
  price: number;
  prevClose: number | null;
  changePct: number | null;
  volume: number | null;
  currency: string | null;
  isRealtime: boolean;
  fetchedAt: string;
  cached: boolean; // true = DB 히트(외부 API 미호출)
}

/** 시세 조회: 캐시 신선하면 DB, 만료면 Yahoo 갱신 후 upsert. */
export async function getQuote(ticker: string): Promise<CachedQuote | null> {
  const db = createServiceClient();
  const ttl = quoteTtlSeconds(ticker);

  const { data: row } = await db
    .from("quote_cache")
    .select("*")
    .eq("ticker", ticker)
    .maybeSingle();

  if (row && isFresh(row.fetched_at, ttl)) {
    return {
      ticker, price: row.price, prevClose: row.prev_close,
      changePct: row.change_pct, volume: row.volume, currency: row.currency,
      isRealtime: row.is_realtime, fetchedAt: row.fetched_at, cached: true,
    };
  }

  const fresh = await fetchQuote(ticker);
  if (!fresh) {
    // 외부 실패 → 만료된 캐시라도 폴백 반환 (stale-if-error)
    if (row) {
      return {
        ticker, price: row.price, prevClose: row.prev_close,
        changePct: row.change_pct, volume: row.volume, currency: row.currency,
        isRealtime: false, fetchedAt: row.fetched_at, cached: true,
      };
    }
    return null;
  }

  const fetchedAt = new Date().toISOString();
  const isRealtime = quoteTtlSeconds(ticker) === 3;
  await db.from("quote_cache").upsert({
    ticker, price: fresh.price, prev_close: fresh.prevClose,
    change_pct: fresh.changePct, volume: fresh.volume, currency: fresh.currency,
    is_realtime: isRealtime, fetched_at: fetchedAt,
  });

  return {
    ticker, price: fresh.price, prevClose: fresh.prevClose,
    changePct: fresh.changePct, volume: fresh.volume, currency: fresh.currency,
    isRealtime, fetchedAt, cached: false,
  };
}

const RATE_TTL = 300; // 5분
const RATE_SYMBOLS: Record<string, string> = {
  "USD/KRW": "USDKRW=X",
  "EUR/KRW": "EURKRW=X",
  "JPY/KRW": "JPYKRW=X",
  "USD/JPY": "USDJPY=X",
};

export interface CachedRate {
  pair: string;
  rate: number;
  changePct: number | null;
}

/** 환율 일괄 조회 (5분 TTL 캐시). */
export async function getRates(): Promise<CachedRate[]> {
  const db = createServiceClient();
  const out: CachedRate[] = [];

  for (const [pair, symbol] of Object.entries(RATE_SYMBOLS)) {
    const { data: row } = await db
      .from("exchange_rate_cache")
      .select("*")
      .eq("pair", pair)
      .maybeSingle();

    if (row && isFresh(row.fetched_at, RATE_TTL)) {
      out.push({ pair, rate: row.rate, changePct: row.change_pct });
      continue;
    }

    const fresh = await fetchRate(symbol);
    if (fresh) {
      await db.from("exchange_rate_cache").upsert({
        pair, rate: fresh.rate, change_pct: fresh.changePct,
        fetched_at: new Date().toISOString(),
      });
      out.push({ pair, rate: fresh.rate, changePct: fresh.changePct });
    } else if (row) {
      out.push({ pair, rate: row.rate, changePct: row.change_pct });
    }
  }
  return out;
}
