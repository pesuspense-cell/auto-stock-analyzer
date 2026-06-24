// data/krx_etf.json (레포 루트, 전체 국내 ETF 목록) → web/frontend/lib/kr-etfs.generated.ts.
// 손으로 추린 kr-aliases 의 KR ETF 22종 대신 KRX 전체 ETF(약 1,100종)를 검색 별칭으로 끌어온다.
//
// 사용법:  cd web/frontend && node scripts/gen-kr-etfs.mjs
import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const src = resolve(__dirname, "../../../data/krx_etf.json");
const out = resolve(__dirname, "../lib/kr-etfs.generated.ts");

const raw = JSON.parse(readFileSync(src, "utf8")); // { "이름 (CODE)": "CODE.KS", ... }

const entries = Object.entries(raw)
  .map(([key, ticker]) => {
    // "TIGER 미국우주테크 (0183J0)" → "TIGER 미국우주테크"
    const nameKr = key.replace(/\s*\([0-9A-Z]{6}\)\s*$/, "").trim();
    return { ticker, nameKr };
  })
  .filter((e) => e.nameKr && /^[0-9A-Z]{6}\.K[SQ]$/.test(e.ticker))
  // 티커 기준 dedupe, 이름순 정렬(결정적 출력)
  .filter((e, i, arr) => arr.findIndex((x) => x.ticker === e.ticker) === i)
  .sort((a, b) => a.nameKr.localeCompare(b.nameKr, "ko"));

const body = entries
  .map((e) => `  { ticker: ${JSON.stringify(e.ticker)}, nameKr: ${JSON.stringify(e.nameKr)} },`)
  .join("\n");

const ts = `// AUTO-GENERATED — 편집 금지. data/krx_etf.json 에서 scripts/gen-kr-etfs.mjs 로 생성.
// KRX 전체 국내 ETF 목록(검색 별칭용). 갱신: cd web/frontend && node scripts/gen-kr-etfs.mjs
export const KR_ETFS: { ticker: string; nameKr: string }[] = [
${body}
];
`;

writeFileSync(out, ts, "utf8");
console.log(`✅ ${entries.length}개 국내 ETF → lib/kr-etfs.generated.ts`);
