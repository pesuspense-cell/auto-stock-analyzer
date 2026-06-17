// Yahoo Finance 차트 API(공개 JSON, 키 불필요) 직접 호출 — 기존 yfinance 의존 대체.
// stock_ai._realtime_price_1m / get_exchange_rates 의 외부 호출부에 대응한다.

export interface RawQuote {
  ticker: string;
  price: number;
  prevClose: number | null;
  changePct: number | null;
  volume: number | null;
  currency: string | null;
}

const CHART = "https://query1.finance.yahoo.com/v8/finance/chart";

/** 단일 종목 시세 — meta.regularMarketPrice / chartPreviousClose 사용. */
export async function fetchQuote(ticker: string): Promise<RawQuote | null> {
  try {
    const res = await fetch(`${CHART}/${encodeURIComponent(ticker)}?range=5d&interval=1d`, {
      headers: { "User-Agent": "Mozilla/5.0" },
      // Next.js 서버 fetch 캐시는 우리가 Supabase로 직접 제어하므로 비활성화
      cache: "no-store",
    });
    if (!res.ok) return null;
    const json = await res.json();
    const meta = json?.chart?.result?.[0]?.meta;
    if (!meta) return null;

    const price = Number(meta.regularMarketPrice);
    const prev = meta.chartPreviousClose ?? meta.previousClose ?? null;
    if (!price || Number.isNaN(price)) return null;

    const prevClose = prev != null ? Number(prev) : null;
    const changePct = prevClose ? ((price - prevClose) / prevClose) * 100 : null;

    return {
      ticker,
      price,
      prevClose,
      changePct: changePct != null ? Math.round(changePct * 100) / 100 : null,
      volume: meta.regularMarketVolume != null ? Number(meta.regularMarketVolume) : null,
      currency: meta.currency ?? null,
    };
  } catch {
    return null;
  }
}

/** 환율 — 동일 차트 엔드포인트 (예: "USDKRW=X"). */
export async function fetchRate(symbol: string): Promise<{ rate: number; changePct: number | null } | null> {
  const q = await fetchQuote(symbol);
  if (!q) return null;
  return { rate: q.price, changePct: q.changePct };
}

export interface RawOhlc {
  date: string[];
  open: number[];
  high: number[];
  low: number[];
  close: number[];
  volume: number[];
}

/**
 * OHLCV 시계열 — 지표 계산용. (예: range="2y", interval="1d")
 * EMA200 등 장기 지표를 위해 기본 2년치를 받는다. null/누락 봉은 제거한다.
 */
export async function fetchOhlc(ticker: string, range = "2y", interval = "1d"): Promise<RawOhlc | null> {
  try {
    const res = await fetch(`${CHART}/${encodeURIComponent(ticker)}?range=${range}&interval=${interval}`, {
      headers: { "User-Agent": "Mozilla/5.0" },
      cache: "no-store",
    });
    if (!res.ok) return null;
    const json = await res.json();
    const result = json?.chart?.result?.[0];
    const ts: number[] = result?.timestamp;
    const q = result?.indicators?.quote?.[0];
    if (!ts || !q) return null;

    const out: RawOhlc = { date: [], open: [], high: [], low: [], close: [], volume: [] };
    for (let i = 0; i < ts.length; i++) {
      const o = q.open?.[i], h = q.high?.[i], l = q.low?.[i], c = q.close?.[i], v = q.volume?.[i];
      // 종가 누락 봉은 제외 (지표 왜곡 방지)
      if (c == null || h == null || l == null) continue;
      out.date.push(new Date(ts[i] * 1000).toISOString().slice(0, 10));
      out.open.push(Number(o ?? c));
      out.high.push(Number(h));
      out.low.push(Number(l));
      out.close.push(Number(c));
      out.volume.push(Number(v ?? 0));
    }
    return out.close.length >= 21 ? out : null;
  } catch {
    return null;
  }
}
