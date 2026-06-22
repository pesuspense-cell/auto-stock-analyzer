"use client";

import { useEffect, useState } from "react";
import { useJob } from "@/hooks/useJob";
import type { AnalysisResponse } from "@/lib/types";
import type { StockHit } from "@/lib/api-types";
import { SignalCard } from "./SignalCard";

/**
 * AI 심층 분석 — 종목 선택 시 자동 가동(POST /analysis/run → 워커 interactive 레인 → 폴링).
 * 주가·기술·뉴스·펀더멘털을 결합한 하이브리드 신호 카드를 항상 표시한다.
 */
export function AnalysisPanel({ picked }: { picked: StockHit | null }) {
  const { result, error, status, busy, enqueue, reset } = useJob<AnalysisResponse>();
  const [useLlm, setUseLlm] = useState(false);

  // 선택 종목이 바뀌면 자동으로 분석을 가동한다(탭 keep-alive 하에서도 동작).
  useEffect(() => {
    if (picked) enqueue("/api/v1/analysis/run", { ticker: picked.ticker, period: "6mo", use_llm: useLlm });
    else reset();
    // useLlm 토글은 아래 onChange 에서 즉시 재가동하므로 deps 에서 제외.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [picked?.ticker]);

  function rerun(llm: boolean) {
    if (picked) enqueue("/api/v1/analysis/run", { ticker: picked.ticker, period: "6mo", use_llm: llm });
  }

  if (!picked) {
    return (
      <section className="rounded-card border border-term-border bg-term-1 p-5 text-center text-sm text-term-muted shadow-elevated">
        종목을 선택하면 AI 심층 분석이 표시됩니다.
      </section>
    );
  }

  if (busy) {
    return (
      <section className="rounded-card border border-term-border bg-term-1 p-8 text-center shadow-elevated">
        <div className="mx-auto mb-3 h-1 w-48 overflow-hidden rounded-full bg-term-border">
          <div className="loading-bar-fill h-full w-1/3 rounded-full bg-accent" />
        </div>
        <p className="text-sm text-term-muted">
          {status === "pending" ? "🧠 AI 심층 분석 대기 중…" : "🧠 주가·재무·뉴스를 분석하고 있습니다…"}
        </p>
      </section>
    );
  }

  if (error) {
    return (
      <section className="rounded-card border border-term-border bg-term-1 p-5 shadow-elevated">
        <p className="text-sm text-loss">AI 심층 분석 실패: {error}</p>
        <button onClick={() => rerun(useLlm)} className="mt-3 rounded-lg bg-accent px-3 py-1.5 text-xs font-semibold text-white hover:brightness-110">
          다시 분석
        </button>
      </section>
    );
  }

  if (!result) return null;

  return (
    <div className="space-y-2">
      <SignalCard a={result} />
      <div className="flex items-center justify-end gap-3 px-1">
        <label className="flex items-center gap-1 text-xs text-ink-2">
          <input
            type="checkbox"
            checked={useLlm}
            onChange={(e) => { setUseLlm(e.target.checked); rerun(e.target.checked); }}
          />
          AI 뉴스 감성 포함
        </label>
        <button onClick={() => rerun(useLlm)} className="text-xs text-accent hover:underline">
          다시 분석
        </button>
      </div>
    </div>
  );
}
