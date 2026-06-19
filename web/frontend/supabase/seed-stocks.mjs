// stock_metadata.json (레포 루트, 8,922종목) → Supabase `stocks` 테이블 시드.
//
// 사용법:
//   cd web/frontend
//   node supabase/seed-stocks.mjs
//   (NEXT_PUBLIC_SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY 가 .env.local 에 있어야 함)
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { createClient } from "@supabase/supabase-js";

const __dirname = dirname(fileURLToPath(import.meta.url));

// .env.local 간이 로더 (dotenv 없이)
try {
  const env = readFileSync(resolve(__dirname, "../.env.local"), "utf8");
  for (const line of env.split("\n")) {
    const m = line.match(/^\s*([A-Z0-9_]+)\s*=\s*(.*)\s*$/);
    if (m && !process.env[m[1]]) process.env[m[1]] = m[2].replace(/^["']|["']$/g, "");
  }
} catch {
  /* .env.local 없으면 실제 환경변수 사용 */
}

const URL = process.env.NEXT_PUBLIC_SUPABASE_URL;
const KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;
if (!URL || !KEY) {
  console.error("NEXT_PUBLIC_SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY 가 필요합니다.");
  process.exit(1);
}

const db = createClient(URL, KEY, { auth: { persistSession: false } });

// 레포 루트의 stock_metadata.json 로드 (web/frontend/supabase → ../../../)
const metaPath = resolve(__dirname, "../../../stock_metadata.json");
const meta = JSON.parse(readFileSync(metaPath, "utf8"));
const stocks = meta.stocks ?? {};

// KR ETF 운용사 접두 휴리스틱 (is_etf 추정 — 정밀 분류는 추후 KRX ETF 목록으로 보강).
// ⚠️ 국내(.KS/.KQ) 티커에만 적용 — 과거 "SOL" 접두가 미국주 Solventum/Solstice 등을
//    ETF 로 오분류했던 버그 방지. 미국 ETF 는 메타데이터에 없어 앱 별칭사전(kr-aliases)이 보강.
const ETF_PREFIXES = ["KODEX", "TIGER", "ACE", "SOL", "PLUS", "KBSTAR", "ARIRANG",
  "HANARO", "KOSEF", "TIMEFOLIO", "RISE", "WON", "KIWOOM"];
const isEtf = (ticker, name) => {
  if (!/\.K[SQ]$/i.test(ticker)) return false; // 국내 티커만
  const up = (name ?? "").toUpperCase();
  return ETF_PREFIXES.some((p) => up.startsWith(p + " ") || up === p);
};

const rows = Object.entries(stocks).map(([ticker, v]) => ({
  ticker,
  name: v.name,
  name_kr: v.name_kr ?? null, // stock_metadata.json 의 한글 별칭 사용(기존엔 무시되던 버그)
  market: v.market ?? null,
  is_etf: isEtf(ticker, v.name ?? ""),
}));

console.log(`총 ${rows.length}종목 업서트 시작...`);
const CHUNK = 1000;
let done = 0;
for (let i = 0; i < rows.length; i += CHUNK) {
  const batch = rows.slice(i, i + CHUNK);
  const { error } = await db.from("stocks").upsert(batch, { onConflict: "ticker" });
  if (error) {
    console.error(`배치 ${i} 실패:`, error.message);
    process.exit(1);
  }
  done += batch.length;
  console.log(`  ${done}/${rows.length}`);
}
console.log("✅ 시드 완료");
