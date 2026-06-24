"use client";

import { useState } from "react";

/**
 * 지표 설명용 "?" 툴팁 — 클릭(또는 포커스)하면 설명을 펼친다.
 * 종합신호·예상수익 등 다크(term) 카드의 각 지표 옆에 붙여 의미를 안내한다.
 * (핵심·보조 지표 타일은 ChartAnalysisTab 의 Metric 이 자체 ? 툴팁을 가짐)
 */
export function InfoTip({ text, label }: { text: string; label?: string }) {
  const [open, setOpen] = useState(false);
  return (
    <span className="relative inline-flex">
      <button
        type="button"
        aria-label={label ? `${label} 설명` : "설명"}
        aria-expanded={open}
        onClick={(e) => { e.stopPropagation(); setOpen((v) => !v); }}
        onBlur={() => setOpen(false)}
        className="flex h-3.5 w-3.5 shrink-0 items-center justify-center rounded-full border border-term-border text-[0.55rem] font-bold leading-none text-term-muted transition-colors hover:bg-black/40 hover:text-term-ink"
      >
        ?
      </button>
      {open && (
        <span className="absolute left-1/2 top-5 z-30 w-56 -translate-x-1/2 rounded-lg border border-term-border bg-term-1 p-2.5 text-left text-[0.7rem] font-normal leading-relaxed text-term-muted shadow-elevated">
          {text}
        </span>
      )}
    </span>
  );
}
