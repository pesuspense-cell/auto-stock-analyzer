// 주요 섹터 ETF 등락 비교 — 레거시 app.py `_render_sector_etf_prices` 의 ETF 목록 +
// 10분 캐시(market_cache scope="sector_etfs") TS 포팅.
//   · 미국 15개 + 국내 21개, 전일 종가 대비 일간 등락률(fetchQuote 의 changePct).
//   · 벤치마크 상대강도(강/약)는 클라이언트(SectorEtfTable)에서 계산한다.
import { createServiceClient } from "@/lib/supabase/server";
import { fetchDailyChange } from "@/lib/providers/yahoo";
import type { EtfItem, EtfsResponse, EtfsSummary } from "@/lib/api-types";

const SCOPE = "sector_etfs";
const TTL_SECONDS = 600; // 10분

/** [티커, ETF명, 국가, 테마, 순자산규모(국내 조원·미국 십억$, 근사치)] — isIndex 는 "📊" 태그로 판정.
 *  aum 은 표시용이 아닌 규모순 정렬 기준이라 정확값이 아닌 상대 크기만 맞으면 된다. */
const SECTOR_ETFS: readonly [string, string, "US" | "KR", string, number][] = [
  // 미국 — AUM 내림차순
  ["SPY", "S&P 500", "US", "📊 지수", 600],
  ["QQQ", "나스닥 100", "US", "📊 지수", 310],
  ["VGT", "기술주", "US", "🤖 테크", 90],
  ["SCHD", "배당성장", "US", "💰 배당", 68],
  ["XLF", "금융", "US", "🏦 금융", 50],
  ["TLT", "미국채 20년", "US", "📋 채권", 48],
  ["XLV", "헬스케어", "US", "🏥 헬스케어", 42],
  ["BIL", "미국채 단기", "US", "📋 채권", 40],
  ["XLE", "에너지", "US", "⚡ 에너지", 37],
  ["DIA", "다우존스", "US", "📊 지수", 37],
  ["VNQ", "리츠", "US", "🏢 리츠", 35],
  ["GDX", "금광주", "US", "⛏ 원자재", 15],
  ["SOXX", "반도체 (SOX)", "US", "💻 반도체", 14],
  ["BOTZ", "AI/로봇", "US", "🤖 AI/로봇", 3],
  ["LIT", "2차전지/리튬", "US", "🔋 2차전지", 1.5],
  // 국내 — AUM 내림차순
  ["069500.KS", "KODEX 200", "KR", "📊 코스피", 6.5],
  ["229200.KQ", "KODEX 코스닥150", "KR", "📊 코스닥", 0.55],
  ["449450.KS", "PLUS K방산", "KR", "🛡 방산", 1.2],
  ["305720.KS", "KODEX 2차전지산업", "KR", "🔋 2차전지", 1.0],
  ["455850.KS", "SOL AI반도체소부장", "KR", "💻 반도체소부장", 0.7],
  ["396500.KS", "TIGER 반도체TOP10", "KR", "💻 반도체", 0.6],
  ["466920.KS", "SOL 조선TOP3플러스", "KR", "🚢 조선", 0.5],
  ["487240.KS", "KODEX AI전력핵심설비", "KR", "⚡ 전력", 0.5],
  ["462900.KS", "KoAct 바이오헬스케어액티브", "KR", "🧬 바이오", 0.4],
  ["091180.KS", "KODEX 자동차", "KR", "🚗 자동차", 0.4],
  ["0091P0.KS", "TIGER 코리아원자력", "KR", "⚛ 원전", 0.3],
  ["445290.KS", "KODEX 로봇액티브", "KR", "🤖 로봇", 0.3],
  ["466940.KS", "TIGER 은행고배당플러스TOP10", "KR", "🏦 은행", 0.2],
  ["102970.KS", "KODEX 증권", "KR", "📈 증권", 0.2],
  ["228790.KS", "TIGER 화장품", "KR", "💄 화장품", 0.2],
  ["421320.KS", "PLUS 우주항공&UAM", "KR", "🚀 우주", 0.2],
  ["475050.KS", "ACE KPOP포커스", "KR", "🎵 엔터", 0.15],
  ["307520.KS", "TIGER 지주회사", "KR", "🏢 지주사", 0.12],
  ["117700.KS", "KODEX 건설", "KR", "🏗 건설", 0.1],
  ["457990.KS", "PLUS 태양광&ESS", "KR", "☀ 태양광", 0.1],
  ["385510.KS", "KODEX 신재생에너지액티브", "KR", "🌱 재생에너지", 0.1],
];

const isFresh = (fetchedAt: string) => Date.now() - new Date(fetchedAt).getTime() < TTL_SECONDS * 1000;

function summarize(etfs: EtfItem[]): EtfsSummary {
  const up = etfs.filter((e) => e.changePct > 0).length;
  const down = etfs.filter((e) => e.changePct < 0).length;
  const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0);
  const round2 = (n: number) => Math.round(n * 100) / 100;
  return {
    up,
    down,
    avg: round2(mean(etfs.map((e) => e.changePct))),
    krAvg: round2(mean(etfs.filter((e) => e.country === "KR").map((e) => e.changePct))),
    usAvg: round2(mean(etfs.filter((e) => e.country === "US").map((e) => e.changePct))),
  };
}

/** 섹터 ETF 등락 일괄 조회 (10분 TTL · market_cache). 일부 실패는 건너뛴다. */
export async function getSectorEtfs(): Promise<EtfsResponse> {
  const db = createServiceClient();

  // 1) 캐시 신선하면 그대로 반환
  try {
    const { data: row } = await db
      .from("market_cache")
      .select("payload,fetched_at")
      .eq("scope", SCOPE)
      .maybeSingle();
    if (row && isFresh(row.fetched_at)) {
      return row.payload as EtfsResponse;
    }
  } catch {
    /* noop */
  }

  // 2) 만료 → Yahoo 병렬 조회
  const results = await Promise.all(
    SECTOR_ETFS.map(async ([ticker, name, country, tag, aum]) => {
      const q = await fetchDailyChange(ticker);
      if (!q) return null;
      const item: EtfItem = {
        ticker,
        name,
        country,
        tag,
        price: q.price,
        changePct: q.changePct,
        isIndex: tag.startsWith("📊"),
        aum,
      };
      return item;
    })
  );
  const etfs = results.filter((e): e is EtfItem => e !== null);

  const payload: EtfsResponse = {
    etfs,
    summary: summarize(etfs),
    asOf: new Date().toISOString(),
  };

  // 3) 일부라도 받았으면 캐시 저장 (전부 실패면 저장 안 함 → 다음 요청에서 재시도)
  if (etfs.length > 0) {
    try {
      await db.from("market_cache").upsert({ scope: SCOPE, payload, fetched_at: payload.asOf });
    } catch {
      /* noop */
    }
  }

  return payload;
}
