// ta.ts — 기술적 지표 순수 함수 라이브러리.
// 레거시 stock_ai.py `_add_indicators()` 의 pandas 공식을 1:1 재현한다.
// 모든 함수는 number[] (warmup 구간은 NaN) 을 반환하며 부수효과가 없다.
//
// pandas 의미 매핑:
//   rolling(n).mean()  → rollingMean    (window 에 NaN 포함 시 NaN, i<n-1 → NaN)
//   rolling(n).std()   → rollingStd     (표본표준편차, ddof=1)
//   ewm(span, adjust=False).mean() → ewm (재귀, seed = 첫 유효값)
//   series.diff()      → diff           (index 0 = NaN)

export const NaNv = Number.NaN;

const isN = (v: number) => !Number.isNaN(v);

export function diff(v: number[]): number[] {
  return v.map((x, i) => (i === 0 ? NaNv : x - v[i - 1]));
}

export function rollingMean(v: number[], period: number): number[] {
  const out = new Array(v.length).fill(NaNv);
  for (let i = period - 1; i < v.length; i++) {
    let s = 0;
    let ok = true;
    for (let j = i - period + 1; j <= i; j++) {
      if (!isN(v[j])) { ok = false; break; }
      s += v[j];
    }
    if (ok) out[i] = s / period;
  }
  return out;
}

export function rollingSum(v: number[], period: number): number[] {
  const out = new Array(v.length).fill(NaNv);
  for (let i = period - 1; i < v.length; i++) {
    let s = 0;
    let ok = true;
    for (let j = i - period + 1; j <= i; j++) {
      if (!isN(v[j])) { ok = false; break; }
      s += v[j];
    }
    if (ok) out[i] = s;
  }
  return out;
}

/** 표본표준편차(ddof=1) — pandas .std() 기본값과 동일. */
export function rollingStd(v: number[], period: number): number[] {
  const out = new Array(v.length).fill(NaNv);
  for (let i = period - 1; i < v.length; i++) {
    let ok = true;
    let mean = 0;
    for (let j = i - period + 1; j <= i; j++) {
      if (!isN(v[j])) { ok = false; break; }
      mean += v[j];
    }
    if (!ok) continue;
    mean /= period;
    let ss = 0;
    for (let j = i - period + 1; j <= i; j++) ss += (v[j] - mean) ** 2;
    out[i] = Math.sqrt(ss / (period - 1));
  }
  return out;
}

export function rollingMin(v: number[], period: number): number[] {
  const out = new Array(v.length).fill(NaNv);
  for (let i = period - 1; i < v.length; i++) {
    let m = Infinity;
    let ok = true;
    for (let j = i - period + 1; j <= i; j++) {
      if (!isN(v[j])) { ok = false; break; }
      if (v[j] < m) m = v[j];
    }
    if (ok) out[i] = m;
  }
  return out;
}

export function rollingMax(v: number[], period: number): number[] {
  const out = new Array(v.length).fill(NaNv);
  for (let i = period - 1; i < v.length; i++) {
    let m = -Infinity;
    let ok = true;
    for (let j = i - period + 1; j <= i; j++) {
      if (!isN(v[j])) { ok = false; break; }
      if (v[j] > m) m = v[j];
    }
    if (ok) out[i] = m;
  }
  return out;
}

/** 윈도 평균절대편차(MAD) — CCI 용. mean(|x - mean(x)|). */
export function rollingMad(v: number[], period: number): number[] {
  const out = new Array(v.length).fill(NaNv);
  for (let i = period - 1; i < v.length; i++) {
    let ok = true;
    let mean = 0;
    for (let j = i - period + 1; j <= i; j++) {
      if (!isN(v[j])) { ok = false; break; }
      mean += v[j];
    }
    if (!ok) continue;
    mean /= period;
    let mad = 0;
    for (let j = i - period + 1; j <= i; j++) mad += Math.abs(v[j] - mean);
    out[i] = mad / period;
  }
  return out;
}

/** ewm(span=N, adjust=False).mean() — 재귀. NaN 은 건너뛰고 직전값 유지. */
export function ewm(v: number[], span: number): number[] {
  const alpha = 2 / (span + 1);
  const out = new Array(v.length).fill(NaNv);
  let prev = NaNv;
  for (let i = 0; i < v.length; i++) {
    if (!isN(v[i])) { out[i] = prev; continue; }
    prev = isN(prev) ? alpha * v[i] + (1 - alpha) * prev : v[i];
    out[i] = prev;
  }
  return out;
}

export const sma = rollingMean;

// ── 모멘텀 지표 ──────────────────────────────────────────────────────────────

/** RSI(14) — stock_ai 와 동일하게 "단순 이동평균" 변형(Wilder 아님). */
export function rsi(close: number[], period = 14): number[] {
  const d = diff(close);
  const up = d.map((x) => (isN(x) ? Math.max(x, 0) : NaNv));
  const down = d.map((x) => (isN(x) ? Math.max(-x, 0) : NaNv));
  const upMa = rollingMean(up, period);
  const downMa = rollingMean(down, period);
  return upMa.map((u, i) => {
    const dn = downMa[i];
    if (!isN(u) || !isN(dn)) return NaNv;
    if (dn === 0) return 100;
    const rs = u / dn;
    return 100 - 100 / (1 + rs);
  });
}

export interface MacdResult { macd: number[]; signal: number[]; hist: number[]; }

export function macd(close: number[], fast = 12, slow = 26, signalP = 9): MacdResult {
  const emaFast = ewm(close, fast);
  const emaSlow = ewm(close, slow);
  const macdLine = emaFast.map((f, i) => f - emaSlow[i]);
  const signal = ewm(macdLine, signalP);
  const hist = macdLine.map((m, i) => m - signal[i]);
  return { macd: macdLine, signal, hist };
}

export interface StochResult { k: number[]; d: number[]; }

export function stochastic(high: number[], low: number[], close: number[], period = 14, smooth = 3): StochResult {
  const low14 = rollingMin(low, period);
  const high14 = rollingMax(high, period);
  const raw = close.map((c, i) => {
    const rng = high14[i] - low14[i];
    if (!isN(rng) || rng === 0) return NaNv;
    return ((c - low14[i]) / rng) * 100;
  });
  const k = rollingMean(raw, smooth);
  const d = rollingMean(k, smooth);
  return { k, d };
}

/** True Range — pandas concat(...).max(axis=1) 는 skipna=True. */
export function trueRange(high: number[], low: number[], close: number[]): number[] {
  return high.map((h, i) => {
    const cands = [h - low[i]];
    if (i > 0) {
      cands.push(Math.abs(h - close[i - 1]));
      cands.push(Math.abs(low[i] - close[i - 1]));
    }
    return Math.max(...cands.filter(isN));
  });
}

export function atr(high: number[], low: number[], close: number[], period = 14): number[] {
  return rollingMean(trueRange(high, low, close), period);
}

export interface AdxResult { adx: number[]; plusDI: number[]; minusDI: number[]; }

export function adx(high: number[], low: number[], close: number[], period = 14): AdxResult {
  const hDiff = diff(high);
  const lDiff = diff(low).map((x) => (isN(x) ? -x : NaNv)); // (-low).diff()
  const plusDM = hDiff.map((h, i) => {
    const l = lDiff[i];
    if (!isN(h) || !isN(l)) return 0; // np.where: NaN 조건 → False → 0
    return h > l && h > 0 ? h : 0;
  });
  const minusDM = lDiff.map((l, i) => {
    const h = hDiff[i];
    if (!isN(h) || !isN(l)) return 0;
    return l > h && l > 0 ? l : 0;
  });
  const atr14 = rollingMean(trueRange(high, low, close), period);
  const plusDI = rollingMean(plusDM, period).map((v, i) => (atr14[i] ? (100 * v) / atr14[i] : NaNv));
  const minusDI = rollingMean(minusDM, period).map((v, i) => (atr14[i] ? (100 * v) / atr14[i] : NaNv));
  const dx = plusDI.map((p, i) => {
    const m = minusDI[i];
    if (!isN(p) || !isN(m)) return NaNv;
    const sum = p + m;
    return sum === 0 ? NaNv : (Math.abs(p - m) / sum) * 100;
  });
  return { adx: rollingMean(dx, period), plusDI, minusDI };
}

export function cci(high: number[], low: number[], close: number[], period = 20): number[] {
  const tp = high.map((h, i) => (h + low[i] + close[i]) / 3);
  const ma = rollingMean(tp, period);
  const mad = rollingMad(tp, period);
  return tp.map((t, i) => {
    if (!isN(ma[i]) || !isN(mad[i]) || mad[i] === 0) return NaNv;
    return (t - ma[i]) / (0.015 * mad[i]);
  });
}

export function williamsR(high: number[], low: number[], close: number[], period = 14): number[] {
  const hh = rollingMax(high, period);
  const ll = rollingMin(low, period);
  return close.map((c, i) => {
    const rng = hh[i] - ll[i];
    if (!isN(rng) || rng === 0) return NaNv;
    return ((hh[i] - c) / rng) * -100;
  });
}

export function roc(close: number[], period = 12): number[] {
  return close.map((c, i) => (i < period || !isN(close[i - period]) || close[i - period] === 0 ? NaNv : (c / close[i - period] - 1) * 100));
}

export function obv(close: number[], volume: number[]): number[] {
  const out = new Array(close.length).fill(NaNv);
  let acc = 0;
  for (let i = 0; i < close.length; i++) {
    if (i === 0) { acc = 0; } // sign(diff)[0] = 0
    else {
      const d = close[i] - close[i - 1];
      const sign = d > 0 ? 1 : d < 0 ? -1 : 0;
      acc += volume[i] * sign;
    }
    out[i] = acc;
  }
  return out;
}

export function mfi(high: number[], low: number[], close: number[], volume: number[], period = 14): number[] {
  const tp = high.map((h, i) => (h + low[i] + close[i]) / 3);
  const mf = tp.map((t, i) => t * volume[i]);
  const tpDiff = diff(tp);
  const posMf = mf.map((m, i) => (isN(tpDiff[i]) && tpDiff[i] > 0 ? m : 0));
  const negMf = mf.map((m, i) => (isN(tpDiff[i]) && tpDiff[i] < 0 ? m : 0));
  const posSum = rollingSum(posMf, period);
  const negSum = rollingSum(negMf, period);
  return posSum.map((p, i) => {
    const n = negSum[i];
    if (!isN(p) || !isN(n)) return NaNv;
    if (n === 0) return 100;
    return 100 - 100 / (1 + p / n);
  });
}

export function zscore(close: number[], period = 20): number[] {
  const mean = rollingMean(close, period);
  const std = rollingStd(close, period);
  return close.map((c, i) => (!isN(mean[i]) || !isN(std[i]) || std[i] === 0 ? NaNv : (c - mean[i]) / std[i]));
}

/** 멀티 타임프레임 롤링 VWAP — (TP×Vol) 롤링합 / Vol 롤링합. */
export function vwapRolling(high: number[], low: number[], close: number[], volume: number[], window: number): number[] {
  const tp = high.map((h, i) => (h + low[i] + close[i]) / 3);
  const pv = tp.map((t, i) => t * volume[i]);
  const cumPv = rollingSum(pv, window);
  const cumV = rollingSum(volume, window);
  return cumPv.map((p, i) => (!isN(p) || !isN(cumV[i]) || cumV[i] === 0 ? NaNv : p / cumV[i]));
}

export interface BollingerResult { middle: number[]; upper: number[]; lower: number[]; width: number[]; pct: number[]; }

export function bollinger(close: number[], period = 20, mult = 2): BollingerResult {
  const mid = rollingMean(close, period);
  const std = rollingStd(close, period);
  const upper = mid.map((m, i) => m + mult * std[i]);
  const lower = mid.map((m, i) => m - mult * std[i]);
  const width = mid.map((m, i) => (m ? (upper[i] - lower[i]) / m : NaNv));
  const pct = close.map((c, i) => {
    const rng = upper[i] - lower[i];
    return !isN(rng) || rng === 0 ? NaNv : (c - lower[i]) / rng;
  });
  return { middle: mid, upper, lower, width, pct };
}

/** 변동률(전일 대비, %) — 단일 값 헬퍼. */
export function changePct(price: number, prevClose: number): number {
  return prevClose ? ((price - prevClose) / prevClose) * 100 : 0;
}

/** 분위수(선형보간) — pandas Series.quantile 과 동일. */
export function quantile(values: number[], q: number): number {
  const v = values.filter(isN).sort((a, b) => a - b);
  if (v.length === 0) return NaNv;
  const pos = (v.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  if (lo === hi) return v[lo];
  return v[lo] + (pos - lo) * (v[hi] - v[lo]);
}
