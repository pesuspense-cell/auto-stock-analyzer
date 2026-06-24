// 손으로 작성한 KR 티커 목록(sector-etfs.ts, kr-aliases.ts)을 KRX 권위 소스
// (data/krx_etf.json + data/krx_stocks.json)와 대조해 잘못된 코드를 잡는다.
//
// 두 부류의 오류를 검출한다:
//   1) 미존재 코드 — KRX 어디에도 없는 유령 코드 (예: 484310 KODEX AI전력핵심설비)
//   2) 이름 불일치 — 코드는 존재하나 다른 종목 (예: 475580 = 에이럭스, ACE KPOP포커스 아님)
//
// 사용법:  cd web/frontend && node scripts/validate-kr-tickers.mjs
// 종료코드: 오류 발견 시 1 (CI/배포 게이트용). prebuild 훅으로 자동 실행.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = resolve(__dirname, "../../..");

// ── 권위 소스: code(6자리) → 정식 종목명 ─────────────────────────────────────
/** @type {Map<string,string>} */
const authority = new Map();
for (const rel of ["data/krx_etf.json", "data/krx_stocks.json"]) {
  const raw = JSON.parse(readFileSync(resolve(root, rel), "utf8")); // { "이름 (CODE)": "CODE.KS", ... }
  for (const [key, ticker] of Object.entries(raw)) {
    const code = String(ticker).split(".")[0];
    const name = key.replace(/\s*\([0-9A-Z]{6}\)\s*$/, "").trim();
    if (!authority.has(code)) authority.set(code, name);
  }
}

// 이름 비교용 정규화: 소문자 + 공백/&/괄호/"액티브"/"(합성)" 제거
const norm = (s) =>
  s
    .toLowerCase()
    .replace(/\(합성\)/g, "")
    .replace(/액티브/g, "")
    .replace(/[\s&()]/g, "");
const nameMatches = (a, b) => {
  const x = norm(a), y = norm(b);
  return x && y && (x.includes(y) || y.includes(x));
};

// ── 수동 목록에서 (code, nameKr) 추출 ────────────────────────────────────────
/** sector-etfs.ts: ["484310.KS", "KODEX AI전력핵심설비", "KR", "⚡ 전력"] 형태 */
function parseSectorEtfs(txt) {
  const out = [];
  const re = /\["([0-9A-Z]{6})\.K[SQ]",\s*"([^"]+)"/g;
  let m;
  while ((m = re.exec(txt))) out.push({ code: m[1], name: m[2] });
  return out;
}
/** kr-aliases.ts: { ticker: "484310.KS", nameKr: "KODEX AI전력핵심설비", ... } 형태 (KR만) */
function parseAliases(txt) {
  const out = [];
  const re = /ticker:\s*"([0-9A-Z]{6})\.K[SQ]",\s*nameKr:\s*"([^"]+)"/g;
  let m;
  while ((m = re.exec(txt))) out.push({ code: m[1], name: m[2] });
  return out;
}

const targets = [
  { file: "web/frontend/lib/sector-etfs.ts", parse: parseSectorEtfs },
  { file: "web/frontend/lib/kr-aliases.ts", parse: parseAliases },
];

let errors = 0;
for (const { file, parse } of targets) {
  const txt = readFileSync(resolve(root, file), "utf8");
  const entries = parse(txt);
  for (const { code, name } of entries) {
    const authName = authority.get(code);
    if (!authName) {
      console.error(`❌ [${file}] ${code} (${name}) — KRX 권위 소스에 없는 코드`);
      errors++;
    } else if (!nameMatches(name, authName)) {
      console.error(`❌ [${file}] ${code} — 이름 불일치: 목록 "${name}" ≠ KRX "${authName}"`);
      errors++;
    }
  }
}

if (errors) {
  console.error(`\n검증 실패: ${errors}건. data/krx_etf.json 기준으로 코드/이름을 수정하세요.`);
  process.exit(1);
}
console.log("✅ KR 티커 검증 통과 — 모든 수동 코드가 KRX 권위 소스와 일치합니다.");
