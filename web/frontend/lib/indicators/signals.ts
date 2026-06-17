// signals.ts — 레거시 stock_ai.py `generate_signals()` 12개 모듈 채점 로직 TS 포팅.
// 점수 범위 -10~+10 (양수=매수). 입력은 OHLCV 시계열, 출력은 타입 계약을 따른다.
//
// ⚠️ 모듈 11(다이버전스)은 별도 detect_divergence() 의존이라 본 포팅에서는 기여 0으로 둔다
//    (divergenceProvider 주입 시 확장 가능). 나머지 11개 모듈은 임계값까지 1:1 재현.

import * as ta from "@/lib/indicators/ta";

export interface Ohlc {
  date: string[];
  open: number[];
  high: number[];
  low: number[];
  close: number[];
  volume: number[];
}

export interface IndicatorSnapshot {
  price: number;
  changePct: number;
  rsi: number | null;
  macd: number | null;
  macdSignal: number | null;
  macdHist: number | null;
  ema20: number | null;
  ema50: number | null;
  ema200: number | null;
  sma20: number | null;
  bbUpper: number | null;
  bbLower: number | null;
  bbMiddle: number | null;
  bbPct: number | null;
  stochK: number | null;
  stochD: number | null;
  cci: number | null;
  williamsR: number | null;
  adx: number | null;
  atr: number | null;
  roc: number | null;
  mfi: number | null;
  obv: number | null;
  zscore: number | null;
  vwapM: number | null;
}

export interface SignalResult {
  score: number; // -10 ~ +10 (소수 1자리)
  scoreInt: number;
  label: string;
  badge: string;
  reasons: string[];
}

const f = (arr: number[], i: number): number | null => {
  const v = arr[i];
  return v != null && !Number.isNaN(v) ? v : null;
};

/** 지표 일괄 계산 + 마지막 행 스냅샷. */
export function computeIndicators(o: Ohlc): IndicatorSnapshot {
  const n = o.close.length;
  const i = n - 1;
  const macd = ta.macd(o.close);
  const bb = ta.bollinger(o.close);
  const stoch = ta.stochastic(o.high, o.low, o.close);
  const adx = ta.adx(o.high, o.low, o.close);
  const ema200 = n >= 200 ? ta.ewm(o.close, 200) : new Array(n).fill(Number.NaN);
  const vwapM = ta.vwapRolling(o.high, o.low, o.close, o.volume, 20);

  return {
    price: o.close[i],
    changePct: i >= 1 ? ta.changePct(o.close[i], o.close[i - 1]) : 0,
    rsi: f(ta.rsi(o.close), i),
    macd: f(macd.macd, i),
    macdSignal: f(macd.signal, i),
    macdHist: f(macd.hist, i),
    ema20: f(ta.ewm(o.close, 20), i),
    ema50: f(ta.ewm(o.close, 50), i),
    ema200: f(ema200, i),
    sma20: f(ta.sma(o.close, 20), i),
    bbUpper: f(bb.upper, i),
    bbLower: f(bb.lower, i),
    bbMiddle: f(bb.middle, i),
    bbPct: f(bb.pct, i),
    stochK: f(stoch.k, i),
    stochD: f(stoch.d, i),
    cci: f(ta.cci(o.high, o.low, o.close), i),
    williamsR: f(ta.williamsR(o.high, o.low, o.close), i),
    adx: f(adx.adx, i),
    atr: f(ta.atr(o.high, o.low, o.close), i),
    roc: f(ta.roc(o.close), i),
    mfi: f(ta.mfi(o.high, o.low, o.close, o.volume), i),
    obv: f(ta.obv(o.close, o.volume), i),
    zscore: f(ta.zscore(o.close), i),
    vwapM: f(vwapM, i),
  };
}

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));

/** generate_signals() 포팅 — 12 모듈 채점. */
export function generateSignals(o: Ohlc): SignalResult {
  const n = o.close.length;
  if (n < 21) return { score: 0, scoreInt: 0, label: "데이터 부족", badge: "—", reasons: [] };

  const i = n - 1;
  const close = o.close;
  const price = close[i];
  const pChg = (price - close[i - 1]) / close[i - 1];

  let score = 0;
  const reasons: string[] = [];

  // 시계열 지표
  const ema20a = ta.ewm(close, 20);
  const ema50a = ta.ewm(close, 50);
  const ema200a = n >= 200 ? ta.ewm(close, 200) : new Array(n).fill(Number.NaN);
  const m = ta.macd(close);
  const rsiA = ta.rsi(close);
  const stoch = ta.stochastic(o.high, o.low, close);
  const cciA = ta.cci(o.high, o.low, close);
  const wrA = ta.williamsR(o.high, o.low, close);
  const adxR = ta.adx(o.high, o.low, close);
  const obvA = ta.obv(close, o.volume);
  const obvMa = ta.rollingMean(obvA, 20);
  const mfiA = ta.mfi(o.high, o.low, close, o.volume);
  const volMa = ta.rollingMean(o.volume, 20);
  const bb = ta.bollinger(close);
  const rocA = ta.roc(close);
  const zA = ta.zscore(close);
  const vwapW = ta.vwapRolling(o.high, o.low, close, o.volume, 5);
  const vwapM = ta.vwapRolling(o.high, o.low, close, o.volume, 20);
  const vwapQ = ta.vwapRolling(o.high, o.low, close, o.volume, 60);

  // ══ 1. EMA200 장기추세 — ±1.5 ══
  const ema200 = f(ema200a, i);
  if (ema200) {
    if (price > ema200 * 1.02) { score += 1.5; reasons.push(`EMA200 상단 +2% → 장기 강세장`); }
    else if (price > ema200) { score += 0.8; reasons.push(`EMA200 상단 → 장기 상승 추세`); }
    else if (price < ema200 * 0.98) { score -= 1.5; reasons.push(`EMA200 하단 -2% → 장기 약세장`); }
    else { score -= 0.8; reasons.push(`EMA200 하단 → 장기 하락 추세`); }
  }

  // ══ 2. MACD — ±2.5 ══
  const macd = f(m.macd, i), sig = f(m.signal, i), pmacd = f(m.macd, i - 1), psig = f(m.signal, i - 1);
  if (macd != null && sig != null && pmacd != null && psig != null) {
    if (pmacd < psig && macd > sig) { score += 2.0; reasons.push("MACD 골든크로스 → 강한 매수 신호"); }
    else if (pmacd > psig && macd < sig) { score -= 2.0; reasons.push("MACD 데드크로스 → 강한 매도 신호"); }
    else if (macd > sig) { score += 0.5; reasons.push("MACD 매수 우위"); }
    else { score -= 0.5; reasons.push("MACD 매도 우위"); }
    const h0 = f(m.hist, i - 2), h1 = f(m.hist, i - 1), h2 = f(m.hist, i);
    if (h0 != null && h1 != null && h2 != null) {
      if (h0 < h1 && h1 < h2) { score += 0.5; reasons.push("MACD 히스토그램 연속 상승 → 모멘텀 강화"); }
      else if (h0 > h1 && h1 > h2) { score -= 0.5; reasons.push("MACD 히스토그램 연속 하락 → 모멘텀 약화"); }
    }
  }

  // ══ 3. RSI(14) — ±2.0 ══
  const rsi = f(rsiA, i);
  if (rsi != null) {
    if (rsi < 20) { score += 2.0; reasons.push(`RSI 극과매도 (${rsi.toFixed(1)}) → 강반등 기대`); }
    else if (rsi < 30) { score += 1.5; reasons.push(`RSI 과매도 (${rsi.toFixed(1)}) → 매수 고려`); }
    else if (rsi < 40) { score += 0.7; reasons.push(`RSI 매수권 (${rsi.toFixed(1)})`); }
    else if (rsi < 50) { score += 0.2; }
    else if (rsi > 80) { score -= 2.0; reasons.push(`RSI 극과매수 (${rsi.toFixed(1)}) → 강한 매도 신호`); }
    else if (rsi > 70) { score -= 1.5; reasons.push(`RSI 과매수 (${rsi.toFixed(1)}) → 매도 고려`); }
    else if (rsi > 60) { score -= 0.7; reasons.push(`RSI 과열 진입 (${rsi.toFixed(1)})`); }
    else if (rsi > 50) { score -= 0.2; }
  }

  // ══ 4. 오실레이터 컨센서스 — ±2.0 ══
  const osc: number[] = [];
  const sk = f(stoch.k, i), sd = f(stoch.d, i), psk = f(stoch.k, i - 1);
  if (sk != null && sd != null) {
    if (sk < 20 && sd < 20) { osc.push(0.8); reasons.push(`스토캐스틱 과매도 (K:${sk.toFixed(1)} D:${sd.toFixed(1)})`); }
    else if (sk > 80 && sd > 80) { osc.push(-0.8); reasons.push(`스토캐스틱 과매수 (K:${sk.toFixed(1)} D:${sd.toFixed(1)})`); }
    else if (sk > sd && (psk == null || psk <= sd)) osc.push(0.4);
    else if (sk < sd && (psk == null || psk >= sd)) osc.push(-0.4);
    else if (sk > sd) osc.push(0.2);
    else osc.push(-0.2);
  }
  const cci = f(cciA, i);
  if (cci != null) {
    if (cci < -200) { osc.push(0.6); reasons.push(`CCI 극과매도 (${cci.toFixed(0)})`); }
    else if (cci < -100) { osc.push(0.4); reasons.push(`CCI 과매도 (${cci.toFixed(0)})`); }
    else if (cci > 200) { osc.push(-0.6); reasons.push(`CCI 극과매수 (${cci.toFixed(0)})`); }
    else if (cci > 100) { osc.push(-0.4); reasons.push(`CCI 과매수 (${cci.toFixed(0)})`); }
    else if (cci > 0 && cci < 100) osc.push(0.1);
    else osc.push(-0.1);
  }
  const wr = f(wrA, i);
  if (wr != null) {
    if (wr < -90) { osc.push(0.6); reasons.push(`Williams %R 극과매도 (${wr.toFixed(1)})`); }
    else if (wr < -80) { osc.push(0.4); reasons.push(`Williams %R 과매도 (${wr.toFixed(1)})`); }
    else if (wr > -10) { osc.push(-0.6); reasons.push(`Williams %R 극과매수 (${wr.toFixed(1)})`); }
    else if (wr > -20) { osc.push(-0.4); reasons.push(`Williams %R 과매수 (${wr.toFixed(1)})`); }
    else if (wr > -50) osc.push(-0.1);
    else osc.push(0.1);
  }
  if (osc.length) {
    let oscSum = osc.reduce((a, b) => a + b, 0);
    if (osc.length === 3) {
      if (osc.every((v) => v > 0)) { oscSum += 0.4; reasons.push("오실레이터 3종 매수 합의 → 신뢰도 상승"); }
      else if (osc.every((v) => v < 0)) { oscSum -= 0.4; reasons.push("오실레이터 3종 매도 합의 → 신뢰도 상승"); }
    }
    score += clamp(oscSum, -2.0, 2.0);
  }

  // ══ 5. 거래량 (OBV + MFI + 원시) — ±1.5 ══
  const vol = o.volume[i];
  const obv = f(obvA, i), obvM = f(obvMa, i);
  if (obv != null && obvM != null) {
    if (obv > obvM) { score += 0.5; reasons.push("OBV > MA20 → 매집 추세 (수급 긍정)"); }
    else { score -= 0.5; reasons.push("OBV < MA20 → 분산 추세 (수급 부정)"); }
  }
  const mfi = f(mfiA, i);
  if (mfi != null) {
    if (mfi < 20) { score += 0.7; reasons.push(`MFI 과매도 (${mfi.toFixed(1)}) → 자금 유입 기대`); }
    else if (mfi < 30) score += 0.4;
    else if (mfi > 80) { score -= 0.7; reasons.push(`MFI 과매수 (${mfi.toFixed(1)}) → 자금 이탈 주의`); }
    else if (mfi > 70) score -= 0.4;
  }
  const vMa = f(volMa, i);
  if (vMa && vMa > 0) {
    const ratio = vol / vMa;
    if (ratio > 2.5) {
      if (pChg > 0) { score += 0.3; reasons.push(`거래량 폭증 (${ratio.toFixed(1)}x) + 상승 → 강한 매수세`); }
      else { score -= 0.3; reasons.push(`거래량 폭증 (${ratio.toFixed(1)}x) + 하락 → 강한 매도세`); }
    } else if (ratio > 1.5) score += pChg > 0 ? 0.2 : -0.2;
  }

  // ══ 6. ADX 필터 EMA 크로스 — ±2.0 ══
  const adxV = f(adxR.adx, i), adxPos = f(adxR.plusDI, i), adxNeg = f(adxR.minusDI, i);
  const ema20 = f(ema20a, i), ema50 = f(ema50a, i), pema20 = f(ema20a, i - 1), pema50 = f(ema50a, i - 1);
  let adxW = 0.5;
  if (adxV != null) adxW = adxV > 35 ? 1.0 : adxV > 25 ? 0.8 : adxV > 20 ? 0.6 : adxV > 15 ? 0.4 : 0.2;
  if (ema20 != null && ema50 != null && pema20 != null && pema50 != null) {
    if (pema20 < pema50 && ema20 >= ema50) { score += Math.round(2.0 * adxW * 100) / 100; reasons.push(`EMA20 골든크로스 → 상승 전환`); }
    else if (pema20 > pema50 && ema20 <= ema50) { score -= Math.round(2.0 * adxW * 100) / 100; reasons.push(`EMA20 데드크로스 → 하락 전환`); }
    else if (ema20 > ema50) score += Math.round(0.8 * adxW * 100) / 100;
    else score -= Math.round(0.8 * adxW * 100) / 100;
  }
  if (adxPos != null && adxNeg != null && adxV != null && adxV > 20) {
    if (adxPos > adxNeg) { score += 0.3; reasons.push(`+DI > -DI → 상승 우세`); }
    else { score -= 0.3; reasons.push(`-DI > +DI → 하락 우세`); }
  }

  // ══ 7. 볼린저밴드 + Squeeze — ±1.0 ══
  const bbU = f(bb.upper, i), bbL = f(bb.lower, i), bbM = f(bb.middle, i), bbW = f(bb.width, i);
  if (bbU && bbL && bbM) {
    if (price < bbL) { score += 1.0; reasons.push(`볼린저 하단 이탈 → 반등 기대`); }
    else if (price > bbU) { score -= 1.0; reasons.push(`볼린저 상단 돌파 → 과열 주의`); }
    else {
      const bbPct = bbU - bbL > 0 ? (price - bbL) / (bbU - bbL) : 0.5;
      if (bbPct > 0.7) { score -= 0.3; reasons.push(`볼린저 상단 70% 이상 (%B:${bbPct.toFixed(2)})`); }
      else if (bbPct < 0.3) { score += 0.3; reasons.push(`볼린저 하단 30% 이하 (%B:${bbPct.toFixed(2)})`); }
    }
    if (bbW != null && n >= 20) {
      const recent = bb.width.slice(-20).filter((x) => !Number.isNaN(x));
      if (recent.length >= 5 && bbW <= ta.quantile(recent, 0.2)) reasons.push("볼린저 밴드 Squeeze (폭발적 변동성 임박)");
    }
  }

  // ══ 8. ROC — ±0.5 ══
  const roc = f(rocA, i);
  if (roc != null) {
    if (roc > 15) { score += 0.5; reasons.push(`ROC 강한 상승 모멘텀 (${roc.toFixed(1)}%)`); }
    else if (roc > 5) score += 0.2;
    else if (roc < -15) { score -= 0.5; reasons.push(`ROC 강한 하락 모멘텀 (${roc.toFixed(1)}%)`); }
    else if (roc < -5) score -= 0.2;
  }

  // ══ 9. 일목균형표 — ±1.8 ══
  const tenkanA = ta.rollingMax(o.high, 9).map((h, idx) => (h + ta.rollingMin(o.low, 9)[idx]) / 2);
  const kijunA = ta.rollingMax(o.high, 26).map((h, idx) => (h + ta.rollingMin(o.low, 26)[idx]) / 2);
  const spanARaw = tenkanA.map((t, idx) => (t + kijunA[idx]) / 2);
  const spanBRaw = ta.rollingMax(o.high, 52).map((h, idx) => (h + ta.rollingMin(o.low, 52)[idx]) / 2);
  const spanA = f(spanARaw, i - 26), spanB = f(spanBRaw, i - 26); // shift(26)
  const tenkan = f(tenkanA, i), kijun = f(kijunA, i);
  if (spanA && spanB) {
    const top = Math.max(spanA, spanB), bot = Math.min(spanA, spanB);
    if (price > top) { score += 1.5; reasons.push(`일목 구름 위 → 중기 강세 추세`); }
    else if (price > bot) { score += 0.3; reasons.push(`일목 구름 내부 → 방향 탐색 중`); }
    else { score -= 1.5; reasons.push(`일목 구름 아래 → 중기 약세 추세`); }
  }
  if (tenkan != null && kijun != null) {
    if (tenkan > kijun) { score += 0.3; reasons.push(`일목 전환선 > 기준선 → 단기 강세`); }
    else { score -= 0.3; reasons.push(`일목 전환선 < 기준선 → 단기 약세`); }
  }

  // ══ 10. Z-Score — ±1.0 ══
  const z = f(zA, i);
  if (z != null) {
    if (z > 2.5) { score -= 1.0; reasons.push(`Z-Score +${z.toFixed(2)}σ → 통계적 과매수`); }
    else if (z > 1.5) score -= 0.4;
    else if (z < -2.5) { score += 1.0; reasons.push(`Z-Score ${z.toFixed(2)}σ → 통계적 과매도`); }
    else if (z < -1.5) score += 0.4;
  }

  // ══ 11. 다이버전스 — (detect_divergence 별도 의존, 본 포팅 기여 0) ══

  // ══ 12. VWAP 멀티 타임프레임 — ±2.0 ══
  const vw = f(vwapW, i), vm = f(vwapM, i), vq = f(vwapQ, i);
  let vDelta = 0;
  if (vw) vDelta += price > vw ? 0.6 : -0.6;
  if (vm) vDelta += price > vm ? 0.7 : -0.7;
  if (vq) vDelta += price > vq ? 0.7 : -0.7;
  if (vw && vm && vq) {
    if (vw > vm && vm > vq) { vDelta = Math.min(vDelta + 0.3, 2.0); reasons.push(`VWAP 상승 스택 → 다중 타임프레임 강세 정렬`); }
    else if (vw < vm && vm < vq) { vDelta = Math.max(vDelta - 0.3, -2.0); reasons.push(`VWAP 하락 스택 → 다중 타임프레임 약세 정렬`); }
  }
  if (vm) {
    const pvm = f(vwapM, i - 1);
    if (pvm) {
      const pPrice = close[i - 1];
      if (pPrice < pvm && price > vm) { vDelta = Math.min(vDelta + 0.3, 2.0); reasons.push(`월간 VWAP 상향 돌파 → 매수 신호`); }
      else if (pPrice > pvm && price < vm) { vDelta = Math.max(vDelta - 0.3, -2.0); reasons.push(`월간 VWAP 하향 이탈 → 매도 신호`); }
    }
  }
  score += clamp(vDelta, -2.0, 2.0);

  // ══ 최종 판정 ══
  score = clamp(score, -10.0, 10.0);
  const s = Math.round(score);
  let label: string, badge: string;
  if (s >= 6) { label = "강력 매수"; badge = "🟢🟢"; }
  else if (s >= 4) { label = "매수"; badge = "🟢"; }
  else if (s >= 2) { label = "약한 매수"; badge = "🔵"; }
  else if (s >= -1) { label = "중립/관망"; badge = "⚪"; }
  else if (s >= -3) { label = "약한 매도"; badge = "🟡"; }
  else if (s >= -5) { label = "매도"; badge = "🔴"; }
  else { label = "강력 매도"; badge = "🔴🔴"; }

  return { score: Math.round(score * 10) / 10, scoreInt: s, label, badge, reasons };
}
