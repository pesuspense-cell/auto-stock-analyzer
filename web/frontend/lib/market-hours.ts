// market-hours.ts — 장 운영시간 기반 Polling 트래픽 방어.
//
// 설계(요청 사항): "한국 시간(KST) 기준 장 마감 및 주말에는 Polling 주기를 자동 연장/정지".
//   · 장중      → 짧은 주기로 적극 갱신
//   · 장외(평일) → 주기 대폭 연장 (트래픽 절감)
//   · 주말      → Polling 정지 (0)
//
// 타임존/DST 는 Intl.DateTimeFormat 으로 정확히 처리한다(KST 무 DST, 미국장 DST 자동 반영).

interface ZonedParts {
  weekday: number; // 0=일 … 6=토
  minutes: number; // 자정 기준 분
}

const WD: Record<string, number> = { Sun: 0, Mon: 1, Tue: 2, Wed: 3, Thu: 4, Fri: 5, Sat: 6 };

function zonedParts(date: Date, timeZone: string): ZonedParts {
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone, hour12: false, weekday: "short", hour: "2-digit", minute: "2-digit",
  }).formatToParts(date);
  let weekday = 0, hour = 0, minute = 0;
  for (const p of parts) {
    if (p.type === "weekday") weekday = WD[p.value] ?? 0;
    else if (p.type === "hour") hour = parseInt(p.value, 10) % 24;
    else if (p.type === "minute") minute = parseInt(p.value, 10);
  }
  return { weekday, minutes: hour * 60 + minute };
}

const isWeekday = (wd: number) => wd >= 1 && wd <= 5;

/** 한국 정규장 (KST 09:00–15:30, 평일). */
export function isKoreanMarketOpen(date = new Date()): boolean {
  const { weekday, minutes } = zonedParts(date, "Asia/Seoul");
  return isWeekday(weekday) && minutes >= 9 * 60 && minutes < 15 * 60 + 30;
}

/** 미국 정규장 (ET 09:30–16:00, 평일 — DST 자동). */
export function isUSMarketOpen(date = new Date()): boolean {
  const { weekday, minutes } = zonedParts(date, "America/New_York");
  return isWeekday(weekday) && minutes >= 9 * 60 + 30 && minutes < 16 * 60;
}

const isKr = (ticker: string) => ticker.endsWith(".KS") || ticker.endsWith(".KQ");

export type MarketState = "open" | "closed-weekday" | "closed-weekend";

/** 종목 기준 시장 상태 — KST 주말이면 정지 대상. */
export function marketState(ticker: string, date = new Date()): MarketState {
  // 주말 판정은 KST 기준 (요청 사항: KST 주말 정지)
  const { weekday: kstWd } = zonedParts(date, "Asia/Seoul");
  if (!isWeekday(kstWd)) return "closed-weekend";
  const open = isKr(ticker) ? isKoreanMarketOpen(date) : isUSMarketOpen(date);
  return open ? "open" : "closed-weekday";
}

// Polling 주기 (ms)
export const POLL_OPEN = 15_000;        // 장중: 15초
export const POLL_CLOSED_WEEKDAY = 300_000; // 장외 평일: 5분 (연장)
export const POLL_STOPPED = 0;          // 주말: 정지 (SWR refreshInterval 0 = 미갱신)

/**
 * SWR refreshInterval 로 쓸 Polling 주기(ms).
 * 장중 15초 / 장외 평일 5분 / 주말 0(정지).
 */
export function pollingIntervalMs(ticker: string, date = new Date()): number {
  switch (marketState(ticker, date)) {
    case "open": return POLL_OPEN;
    case "closed-weekday": return POLL_CLOSED_WEEKDAY;
    case "closed-weekend": return POLL_STOPPED;
  }
}

/** UI 표시용 라벨. */
export function marketStateLabel(state: MarketState): string {
  switch (state) {
    case "open": return "🟢 장중 (15초 갱신)";
    case "closed-weekday": return "🌙 장외 (5분 갱신)";
    case "closed-weekend": return "💤 주말 (갱신 정지)";
  }
}
