"use client";

import { useState } from "react";

/**
 * 지표 설명용 "?" 툴팁 — 클릭(또는 포커스)하면 설명을 펼친다.
 * 종합신호·예상수익 등 다크(term) 카드의 각 지표 옆에 붙여 의미를 안내한다.
 * (핵심·보조 지표 타일은 ChartAnalysisTab 의 Metric 이 자체 ? 툴팁을 가짐)
 */
type Variant = "term" | "surface";

const STYLES: Record<Variant, { btn: string; box: string }> = {
  // 다크(term) 카드용 — 종합신호·예상수익·AI 카드
  term: {
    btn: "border-term-border text-term-muted hover:bg-black/40 hover:text-term-ink",
    box: "border-term-border bg-term-1 text-term-muted shadow-elevated",
  },
  // 밝은(surface) 카드용 — 펀더멘털 핵심지표 등
  surface: {
    btn: "border-hairline text-ink-2 hover:bg-canvas hover:text-ink",
    box: "border-hairline bg-surface text-ink-2 shadow-card",
  },
};

export function InfoTip({
  text,
  label,
  variant = "term",
}: {
  text: string;
  label?: string;
  variant?: Variant;
}) {
  const [open, setOpen] = useState(false);
  const s = STYLES[variant];
  return (
    <span className="relative inline-flex">
      <button
        type="button"
        aria-label={label ? `${label} 설명` : "설명"}
        aria-expanded={open}
        onClick={(e) => { e.stopPropagation(); setOpen((v) => !v); }}
        onBlur={() => setOpen(false)}
        className={`flex h-3.5 w-3.5 shrink-0 items-center justify-center rounded-full border text-[0.55rem] font-bold leading-none transition-colors ${s.btn}`}
      >
        ?
      </button>
      {open && (
        <span className={`absolute left-1/2 top-5 z-30 w-56 -translate-x-1/2 rounded-lg border p-2.5 text-left text-[0.7rem] font-normal leading-relaxed ${s.box}`}>
          {text}
        </span>
      )}
    </span>
  );
}
