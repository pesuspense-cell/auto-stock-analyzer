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

// query1/query2 두 호스트 + 재시도로 레이트리밋(429)·일시 차단에 견디게 한다.
// 서버리스(Vercel/Render) IP 가 query1 에서 429/401 을 받으면 query2 로 폴백한다.
const CHART_HOSTS = [
  "https://query1.finance.yahoo.com/v8/finance/chart",
  "https://query2.finance.yahoo.com/v8/finance/chart",
];
const CHART_HEADERS = {
  "User-Agent":
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
  Accept: "application/json,text/plain,*/*",
  "Accept-Language": "en-US,en;q=0.9",
};
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/**
 * Yahoo chart JSON 견고 호출 — 2호스트 × 2시도. 429/5xx 는 재시도, 404 는 즉시 포기.
 * 모든 시도 실패 시 null.
 */
async function chartJson(ticker: string, query: string): Promise<unknown | null> {
  for (let attempt = 0; attempt < 2; attempt++) {
    for (const host of CHART_HOSTS) {
      try {
        const res = await fetch(`${host}/${encodeURIComponent(ticker)}?${query}`, {
          headers: CHART_HEADERS,
          cache: "no-store",
        });
        if (res.ok) return await res.json();
        if (res.status === 404) return null; // 존재하지 않는 티커 — 재시도 무의미
      } catch {
        /* 다음 호스트/시도 */
      }
    }
    if (attempt === 0) await sleep(350); // 429 백오프
  }
  return null;
}

/**
 * 종목 표시명 — chart meta 의 longName/shortName. (예: 069500.KS → "KODEX 200")
 * stocks 테이블·별칭사전에 없는 ETF/종목명을 동적으로 해소할 때 쓴다.
 */
export async function fetchName(ticker: string): Promise<string | null> {
  const json = await chartJson(ticker, "range=1d&interval=1d");
  const meta = (json as any)?.chart?.result?.[0]?.meta;
  const name = meta?.longName || meta?.shortName;
  return typeof name === "string" && name.trim() ? name.trim() : null;
}

/** 단일 종목 시세 — meta.regularMarketPrice / chartPreviousClose 사용. */
export async function fetchQuote(ticker: string): Promise<RawQuote | null> {
  try {
    const json = await chartJson(ticker, "range=5d&interval=1d");
    const meta = (json as any)?.chart?.result?.[0]?.meta;
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

/**
 * 일간 등락률 — 마지막 두 일봉 종가로 (curr-prev)/prev 계산.
 * fetchQuote 의 changePct 는 meta.chartPreviousClose(범위 직전 종가) 기반이라
 * range=5d 에선 "전일 대비"가 아닌 수일치 변동이 된다. ETF 등락 비교표처럼
 * 정확한 전일 대비가 필요한 곳은 이 함수를 쓴다.
 */
export async function fetchDailyChange(
  ticker: string
): Promise<{ price: number; changePct: number; currency: string | null } | null> {
  try {
    const json = await chartJson(ticker, "range=5d&interval=1d");
    const result = (json as any)?.chart?.result?.[0];
    const closeRaw: (number | null)[] = result?.indicators?.quote?.[0]?.close;
    if (!Array.isArray(closeRaw)) return null;
    const closes = closeRaw.filter((c): c is number => c != null && !Number.isNaN(c));
    if (closes.length < 2) return null;
    const prev = closes[closes.length - 2];
    const curr = closes[closes.length - 1];
    if (!prev) return null;
    return {
      price: curr,
      changePct: Math.round(((curr - prev) / prev) * 10000) / 100,
      currency: result?.meta?.currency ?? null,
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
    const json = await chartJson(ticker, `range=${range}&interval=${interval}`);
    const result = (json as any)?.chart?.result?.[0];
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
