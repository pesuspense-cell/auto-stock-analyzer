// 숫자·등락 포맷 유틸 (기존 f-string 포맷팅 대체)

export const fmtNum = (n: number, digits = 0) =>
  n.toLocaleString("ko-KR", { minimumFractionDigits: digits, maximumFractionDigits: digits });

export const fmtPct = (n: number) => `${n >= 0 ? "+" : ""}${n.toFixed(2)}%`;

/** 양수=gain, 음수/0=loss — Tailwind 텍스트 클래스 반환 */
export const signClass = (n: number) => (n >= 0 ? "text-gain" : "text-loss");

/** 통화 라벨: .KS/.KQ → ₩, 그 외 → $ */
export const priceLabel = (price: number, ticker: string) =>
  ticker.endsWith(".KS") || ticker.endsWith(".KQ")
    ? `${fmtNum(price)}₩`
    : `$${fmtNum(price, 2)}`;
