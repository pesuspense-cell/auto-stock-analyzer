// 한글 별칭/ETF 사전 — Supabase stocks 테이블의 name_kr 공백을 앱 레벨에서 보강한다.
//
// 배경: stocks 테이블은 name_kr 이 전부 null 이고 주요 ETF(069500.KS·QQQ 등)는 행 자체가
//   없어, 미국/나스닥 종목 한글 검색과 ETF 표시가 불가능했다. 재시드(서비스롤 키 필요) 대신
//   이 사전을 검색 라우트에서 병합한다:
//     · US 종목: ticker → nameKr 오버레이(DB 영문명 위에 한글명 표시 + 한글 검색).
//     · ETF: DB 에 없으면 이 사전 항목을 결과에 직접 주입.
//   레거시 stock_ai.py `_US_KR_ALIASES` 를 이식·확장했다.

import { KR_ETFS } from "./kr-etfs.generated";

export interface AliasEntry {
  ticker: string;
  nameKr: string;
  nameEn?: string;      // ETF 주입 시 보조 표기(영문/원어)
  market?: string;      // ETF 주입 시 표시용
  isEtf?: boolean;
}

// ── 미국/나스닥 주요 종목 한글 별칭 (DB 영문명에 오버레이) ──────────────────────
const US_ALIASES: Record<string, string> = {
  // 반도체·AI
  NVDA: "엔비디아", AMD: "AMD", INTC: "인텔", QCOM: "퀄컴", AVGO: "브로드컴",
  TXN: "텍사스인스트루먼트", MU: "마이크론", AMAT: "어플라이드머티리얼즈",
  LRCX: "램리서치", KLAC: "KLA", ASML: "ASML", ARM: "ARM홀딩스",
  MRVL: "마벨테크놀로지", ON: "온세미컨덕터", SWKS: "스카이웍스", QRVO: "코르보",
  TSM: "TSMC", SMCI: "슈퍼마이크로",
  // 빅테크·소프트웨어
  AAPL: "애플", MSFT: "마이크로소프트", GOOGL: "알파벳(A)", GOOG: "알파벳(C)",
  AMZN: "아마존", META: "메타", NFLX: "넷플릭스", TSLA: "테슬라", CRM: "세일즈포스",
  ORCL: "오라클", IBM: "IBM", DELL: "델테크놀로지스", NOW: "서비스나우",
  SNOW: "스노우플레이크", PLTR: "팔란티어", UBER: "우버", LYFT: "리프트",
  ABNB: "에어비앤비", SPOT: "스포티파이", SHOP: "쇼피파이", ADBE: "어도비",
  INTU: "인튜이트", PANW: "팔로알토네트웍스", CRWD: "크라우드스트라이크",
  ZS: "지스케일러", DDOG: "데이터독", HUBS: "허브스팟", NET: "클라우드플레어",
  MDB: "몽고DB", TEAM: "아틀라시안", WDAY: "워크데이", SNPS: "시놉시스",
  CDNS: "케이던스", ANET: "아리스타네트웍스", MSTR: "마이크로스트래티지",
  // 금융
  JPM: "JP모건", BAC: "뱅크오브아메리카", WFC: "웰스파고", GS: "골드만삭스",
  MS: "모건스탠리", C: "씨티그룹", BLK: "블랙록", V: "비자", MA: "마스터카드",
  AXP: "아메리칸익스프레스", PYPL: "페이팔", SQ: "블록(스퀘어)", COF: "캐피털원",
  USB: "US뱅코프", COIN: "코인베이스", HOOD: "로빈후드",
  // 헬스케어
  LLY: "일라이릴리", JNJ: "존슨앤존슨", PFE: "화이자", MRNA: "모더나", ABBV: "애브비",
  UNH: "유나이티드헬스", CVS: "CVS헬스", AMGN: "암젠", GILD: "길리어드사이언스",
  BIIB: "바이오젠", REGN: "리제네론", VRTX: "버텍스", ISRG: "인튜이티브서지컬",
  // 소비재·리테일·미디어
  WMT: "월마트", COST: "코스트코", TGT: "타깃", NKE: "나이키", SBUX: "스타벅스",
  MCD: "맥도날드", DIS: "월트디즈니", CMCSA: "컴캐스트", PEP: "펩시코", KO: "코카콜라",
  PG: "P&G", MDLZ: "몬델리즈", LULU: "룰루레몬",
  // 에너지·산업
  XOM: "엑슨모빌", CVX: "셰브런", COP: "코노코필립스", OXY: "옥시덴탈", SLB: "슐럼버거",
  CAT: "캐터필러", BA: "보잉", GE: "GE에어로스페이스", HON: "허니웰", DE: "디어",
  // 통신·전기차·모빌리티
  T: "AT&T", VZ: "버라이즌", TMUS: "T모바일", RIVN: "리비안", LCID: "루시드",
  F: "포드", GM: "제너럴모터스",
};

// ── ETF (DB 미존재 → 검색 결과에 직접 주입) ────────────────────────────────────
const ETFS: AliasEntry[] = [
  // 미국 대표·테마 ETF
  { ticker: "SPY", nameKr: "S&P500 ETF (SPY)", nameEn: "SPDR S&P 500", market: "US ETF", isEtf: true },
  { ticker: "QQQ", nameKr: "나스닥100 ETF (QQQ)", nameEn: "Invesco QQQ", market: "US ETF", isEtf: true },
  { ticker: "DIA", nameKr: "다우존스 ETF (DIA)", nameEn: "SPDR Dow Jones", market: "US ETF", isEtf: true },
  { ticker: "IWM", nameKr: "러셀2000 ETF (IWM)", nameEn: "iShares Russell 2000", market: "US ETF", isEtf: true },
  { ticker: "SCHD", nameKr: "미국 배당성장 ETF (SCHD)", nameEn: "Schwab US Dividend", market: "US ETF", isEtf: true },
  { ticker: "SOXX", nameKr: "미국 반도체 ETF (SOXX)", nameEn: "iShares Semiconductor", market: "US ETF", isEtf: true },
  { ticker: "SMH", nameKr: "미국 반도체 ETF (SMH)", nameEn: "VanEck Semiconductor", market: "US ETF", isEtf: true },
  { ticker: "SOXL", nameKr: "반도체 3배 레버리지 ETF (SOXL)", nameEn: "Direxion Semi Bull 3X", market: "US ETF", isEtf: true },
  { ticker: "SOXS", nameKr: "반도체 인버스 ETF (SOXS)", nameEn: "Direxion Semi Bear 3X", market: "US ETF", isEtf: true },
  { ticker: "TQQQ", nameKr: "나스닥100 3배 ETF (TQQQ)", nameEn: "ProShares UltraPro QQQ", market: "US ETF", isEtf: true },
  { ticker: "SQQQ", nameKr: "나스닥100 인버스 3배 ETF (SQQQ)", nameEn: "ProShares UltraPro Short QQQ", market: "US ETF", isEtf: true },
  { ticker: "VGT", nameKr: "미국 기술주 ETF (VGT)", nameEn: "Vanguard Info Tech", market: "US ETF", isEtf: true },
  { ticker: "BOTZ", nameKr: "AI·로봇 ETF (BOTZ)", nameEn: "Global X Robotics & AI", market: "US ETF", isEtf: true },
  { ticker: "ARKK", nameKr: "ARK 혁신 ETF (ARKK)", nameEn: "ARK Innovation", market: "US ETF", isEtf: true },
  { ticker: "XLV", nameKr: "미국 헬스케어 ETF (XLV)", nameEn: "Health Care Select", market: "US ETF", isEtf: true },
  { ticker: "XLF", nameKr: "미국 금융 ETF (XLF)", nameEn: "Financial Select", market: "US ETF", isEtf: true },
  { ticker: "XLE", nameKr: "미국 에너지 ETF (XLE)", nameEn: "Energy Select", market: "US ETF", isEtf: true },
  { ticker: "GDX", nameKr: "금광주 ETF (GDX)", nameEn: "VanEck Gold Miners", market: "US ETF", isEtf: true },
  { ticker: "LIT", nameKr: "2차전지·리튬 ETF (LIT)", nameEn: "Global X Lithium", market: "US ETF", isEtf: true },
  { ticker: "TLT", nameKr: "미국 장기국채 ETF (TLT)", nameEn: "iShares 20Y Treasury", market: "US ETF", isEtf: true },
  { ticker: "BIL", nameKr: "미국 단기국채 ETF (BIL)", nameEn: "SPDR 1-3M T-Bill", market: "US ETF", isEtf: true },
  { ticker: "HYG", nameKr: "하이일드 채권 ETF (HYG)", nameEn: "iShares High Yield", market: "US ETF", isEtf: true },
  { ticker: "VNQ", nameKr: "미국 리츠 ETF (VNQ)", nameEn: "Vanguard Real Estate", market: "US ETF", isEtf: true },
  { ticker: "GLD", nameKr: "금 ETF (GLD)", nameEn: "SPDR Gold", market: "US ETF", isEtf: true },
  { ticker: "SLV", nameKr: "은 ETF (SLV)", nameEn: "iShares Silver", market: "US ETF", isEtf: true },
  // 국내 대표·섹터 ETF
  { ticker: "069500.KS", nameKr: "KODEX 200", market: "KR ETF", isEtf: true },
  { ticker: "229200.KQ", nameKr: "KODEX 코스닥150", market: "KR ETF", isEtf: true },
  { ticker: "069660.KS", nameKr: "KIWOOM 200", market: "KR ETF", isEtf: true },
  { ticker: "102110.KS", nameKr: "TIGER 200", market: "KR ETF", isEtf: true },
  { ticker: "396500.KS", nameKr: "TIGER 반도체TOP10", market: "KR ETF", isEtf: true },
  { ticker: "455850.KS", nameKr: "SOL AI반도체소부장", market: "KR ETF", isEtf: true },
  { ticker: "462900.KS", nameKr: "KoAct 바이오헬스케어액티브", market: "KR ETF", isEtf: true },
  { ticker: "305720.KS", nameKr: "KODEX 2차전지산업", market: "KR ETF", isEtf: true },
  { ticker: "091180.KS", nameKr: "KODEX 자동차", market: "KR ETF", isEtf: true },
  { ticker: "466920.KS", nameKr: "SOL 조선TOP3플러스", market: "KR ETF", isEtf: true },
  { ticker: "449450.KS", nameKr: "PLUS K방산", market: "KR ETF", isEtf: true },
  { ticker: "487240.KS", nameKr: "KODEX AI전력핵심설비", market: "KR ETF", isEtf: true },
  { ticker: "0091P0.KS", nameKr: "TIGER 코리아원자력", market: "KR ETF", isEtf: true },
  { ticker: "445290.KS", nameKr: "KODEX 로봇액티브", market: "KR ETF", isEtf: true },
  { ticker: "421320.KS", nameKr: "PLUS 우주항공&UAM", market: "KR ETF", isEtf: true },
  { ticker: "457990.KS", nameKr: "PLUS 태양광&ESS", market: "KR ETF", isEtf: true },
  { ticker: "385510.KS", nameKr: "KODEX 신재생에너지액티브", market: "KR ETF", isEtf: true },
  { ticker: "228790.KS", nameKr: "TIGER 화장품", market: "KR ETF", isEtf: true },
  { ticker: "475050.KS", nameKr: "ACE KPOP포커스", market: "KR ETF", isEtf: true },
  { ticker: "307520.KS", nameKr: "TIGER 지주회사", market: "KR ETF", isEtf: true },
  { ticker: "466940.KS", nameKr: "TIGER 은행고배당플러스TOP10", market: "KR ETF", isEtf: true },
  { ticker: "102970.KS", nameKr: "KODEX 증권", market: "KR ETF", isEtf: true },
];

// ── KRX 전체 국내 ETF(약 1,100종) — 위 ETFS 의 큐레이션과 중복되는 티커는 제외 ──
//   큐레이션 항목(보기 좋은 market 라벨/표기)을 우선하고, 나머지는 자동 생성분으로 보강.
const CURATED_TICKERS = new Set(ETFS.map((e) => e.ticker.toUpperCase()));
const KR_ETF_ENTRIES: AliasEntry[] = KR_ETFS.filter(
  (e) => !CURATED_TICKERS.has(e.ticker.toUpperCase())
).map((e) => ({ ticker: e.ticker, nameKr: e.nameKr, market: "KR ETF", isEtf: true }));

/** 전체 별칭 항목(US 종목 + 큐레이션 ETF + KRX 전체 국내 ETF). */
export const ALIAS_ENTRIES: AliasEntry[] = [
  ...Object.entries(US_ALIASES).map(([ticker, nameKr]) => ({ ticker, nameKr })),
  ...ETFS,
  ...KR_ETF_ENTRIES,
];

/** ticker → 별칭 항목 (DB 결과에 nameKr 오버레이용). */
export const ALIAS_BY_TICKER: Map<string, AliasEntry> = new Map(
  ALIAS_ENTRIES.map((e) => [e.ticker.toUpperCase(), e])
);

/** 질의(한글/영문/티커)와 부분 일치하는 별칭 항목을 반환. */
export function matchAliases(q: string): AliasEntry[] {
  const ql = q.trim().toLowerCase();
  if (!ql) return [];
  return ALIAS_ENTRIES.filter(
    (e) =>
      e.nameKr.toLowerCase().includes(ql) ||
      (e.nameEn && e.nameEn.toLowerCase().includes(ql)) ||
      e.ticker.toLowerCase().includes(ql)
  );
}
