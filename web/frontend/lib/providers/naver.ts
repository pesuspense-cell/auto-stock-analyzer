// 네이버 금융 — 국내(.KS/.KQ) 종목/ETF 의 한글 공식명칭 해소.
// Yahoo 는 KR ETF 를 영문 번역명("Samsung Kodex Ai Electric Power…")으로 주므로,
// 국내 티커는 네이버 자동완성(ac.stock.naver.com)에서 한글 정식명을 가져온다.

const AC = "https://ac.stock.naver.com/ac";

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
