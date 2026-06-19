"use client";

import { useEffect, useRef } from "react";
import {
  createChart,
  ColorType,
  CrosshairMode,
  type IChartApi,
  type UTCTimestamp,
} from "lightweight-charts";
import type { ChartSeries } from "@/lib/api-types";

// Apple Finance 토큰(tailwind)과 동일한 색. 상승=green / 하락=red (signClass 일관).
const C = {
  gain: "#34c759",
  loss: "#ff3b30",
  ema20: "#0066cc",
  ema50: "#ff9500",
  ema200: "#8e8e93",
  grid: "#ececec",
  text: "#7a7a7a",
};

const toTime = (d: string): UTCTimestamp => (Date.parse(`${d}T00:00:00Z`) / 1000) as UTCTimestamp;

/** 가격 캔들 차트 — EMA(20/50/200) 오버레이 + 거래량 히스토그램. */
export function PriceChart({ chart }: { chart: ChartSeries }) {
  const ref = useRef<HTMLDivElement>(null);
  const apiRef = useRef<IChartApi | null>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    const c = createChart(el, {
      layout: { background: { type: ColorType.Solid, color: "transparent" }, textColor: C.text, fontSize: 11 },
      grid: { vertLines: { color: C.grid }, horzLines: { color: C.grid } },
      crosshair: { mode: CrosshairMode.Normal },
      rightPriceScale: { borderColor: C.grid },
      timeScale: { borderColor: C.grid, timeVisible: false },
      height: 340,
      autoSize: true,
    });
    apiRef.current = c;

    const candle = c.addCandlestickSeries({
      upColor: C.gain,
      downColor: C.loss,
      borderUpColor: C.gain,
      borderDownColor: C.loss,
      wickUpColor: C.gain,
      wickDownColor: C.loss,
    });
    candle.setData(
      chart.candles.map((k) => ({ time: toTime(k.time), open: k.open, high: k.high, low: k.low, close: k.close }))
    );

    const addEma = (data: ChartSeries["ema20"], color: string) => {
      if (!data.length) return;
      const s = c.addLineSeries({ color, lineWidth: 2, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false });
      s.setData(data.map((p) => ({ time: toTime(p.time), value: p.value })));
    };
    addEma(chart.ema20, C.ema20);
    addEma(chart.ema50, C.ema50);
    addEma(chart.ema200, C.ema200);

    // 거래량 — 하단 20% 영역에 오버레이
    const vol = c.addHistogramSeries({ priceFormat: { type: "volume" }, priceScaleId: "vol" });
    c.priceScale("vol").applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } });
    vol.setData(
      chart.candles.map((k) => ({
        time: toTime(k.time),
        value: k.volume,
        color: k.close >= k.open ? "rgba(52,199,89,0.35)" : "rgba(255,59,48,0.35)",
      }))
    );

    c.timeScale().fitContent();

    return () => {
      c.remove();
      apiRef.current = null;
    };
  }, [chart]);

  return (
    <section className="rounded-card border border-hairline bg-surface p-3 shadow-card">
      <div className="mb-2 flex items-center gap-3 px-1 text-[0.68rem] text-ink-2">
        <Legend color={C.ema20} label="EMA 20" />
        <Legend color={C.ema50} label="EMA 50" />
        <Legend color={C.ema200} label="EMA 200" />
        <span className="ml-auto">최근 {chart.candles.length}봉 · 일봉</span>
      </div>
      <div ref={ref} className="w-full" style={{ height: 340 }} />
    </section>
  );
}

function Legend({ color, label }: { color: string; label: string }) {
  return (
    <span className="flex items-center gap-1">
      <span className="inline-block h-[2px] w-3 rounded" style={{ background: color }} />
      {label}
    </span>
  );
}
