// 네이버 금융 — 국내(.KS/.KQ) 종목/ETF 의 한글 공식명칭 + 실시간 시세.
// Yahoo 는 KR ETF 를 영문 번역명("Samsung Kodex Ai Electric Power…")으로 주고
// KRX 시세를 ~15분 지연 제공하므로, 국내 티커는 네이버에서 한글명·실시간 시세를 가져온다.

import type { RawQuote } from "@/lib/providers/yahoo";

const AC = "https://ac.stock.naver.com/ac";

// 네이버 실시간 폴링(무인증, delayTime=0). 6자리 코드로 KOSPI/KOSDAQ/ETF 공통.
const POLL = "https://polling.finance.naver.com/api/realtime/domestic/stock";

const NAVER_HEADERS = {
  "User-Agent": "Mozilla/5.0",
  Referer: "https://m.stock.naver.com/",
};

/** "330,500" → 330500, 빈값/비정상 → null. */
function parseNum(s: unknown): number | null {
  if (typeof s !== "string" && typeof s !== "number") return null;
  const n = Number(String(s).replace(/,/g, ""));
  return Number.isFinite(n) ? n : null;
}

// compareToPreviousPrice.code: 1 상한·2 상승 → +, 4 하한·5 하락 → −, 3 보합 → 0
function changeSign(code: unknown): number {
  if (code === "1" || code === "2") return 1;
  if (code === "4" || code === "5") return -1;
  return 0;
}

/**
 * 국내(.KS/.KQ) 실시간 시세 — 네이버 폴링 API. (예: "005930.KS" → 코드 005930)
 * Yahoo 의 지연 시세 대신 쓴다. 등락 부호는 compareToPreviousPrice.code 로 판정.
 */
export async function fetchNaverQuote(ticker: string): Promise<RawQuote | null> {
  const code = ticker.replace(/\.(KS|KQ)$/i, "");
  try {
    const res = await fetch(`${POLL}/${encodeURIComponent(code)}`, {
      headers: NAVER_HEADERS,
      cache: "no-store",
    });
    if (!res.ok) return null;
    const j = await res.json();
    const d = Array.isArray(j?.datas) ? j.datas[0] : null;
    if (!d) return null;

    const price = parseNum(d.closePrice);
    if (price == null || price <= 0) return null;

    const sign = changeSign(d.compareToPreviousPrice?.code);
    const absChg = parseNum(d.compareToPreviousClosePrice) ?? 0;
    const signedChg = sign * absChg;
    const prevClose = Math.round(price - signedChg) || null;
    const absPct = parseNum(d.fluctuationsRatio);
    const changePct = absPct != null ? Math.round(sign * absPct * 100) / 100 : null;

    return {
      ticker,
      price,
      prevClose,
      changePct,
      volume: parseNum(d.accumulatedTradingVolume),
      currency: "KRW",
    };
  } catch {
    return null;
  }
}

/**
 * 6자리 종목코드 → 한글 종목명. (예: 466920 → "SOL 조선TOP3플러스")
 * 자동완성 결과에서 코드가 정확히 일치하는 항목의 name 을 반환한다.
 */
export async function fetchNaverName(code: string): Promise<string | null> {
  try {
    const res = await fetch(`${AC}?q=${encodeURIComponent(code)}&target=stock,index,etf`, {
      headers: { "User-Agent": "Mozilla/5.0" },
      cache: "no-store",
    });
    if (!res.ok) return null;
    const j = await res.json();
    const items: Array<{ code?: string; name?: string }> = Array.isArray(j?.items) ? j.items : [];
    const hit = items.find((i) => i?.code === code) ?? items[0];
    const name = hit?.name;
    return typeof name === "string" && name.trim() ? name.trim() : null;
  } catch {
    return null;
  }
}
