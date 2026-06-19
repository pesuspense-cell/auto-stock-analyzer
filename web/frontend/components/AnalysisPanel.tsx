"use client";

import { useEffect, useState } from "react";
import { useJob } from "@/hooks/useJob";
import type { AnalysisResponse } from "@/lib/types";
import type { StockHit } from "@/lib/api-types";
import { SignalCard } from "./SignalCard";

/**
 * 차트 분석(AI 심층) — POST /analysis/run 으로 jobs 큐에 적재 후 폴링.
 * 무거운 연산은 Python 워커(interactive 레인)가 수행하므로 API 는 경량 유지된다.
 */
export function AnalysisPanel({ picked }: { picked: StockHit | null }) {
  const { result, error, status, busy, enqueue, reset } = useJob<AnalysisResponse>();
  const [useLlm, setUseLlm] = useState(false);

  // 선택 종목이 바뀌면 이전 결과/폴링을 비운다(다른 종목인데 옛 분석이 남는 것 방지).
  useEffect(() => {
    reset();
  }, [picked?.ticker, reset]);

  function run() {
    if (!picked) return;
    enqueue("/api/v1/analysis/run", { ticker: picked.ticker, period: "6mo", use_llm: useLlm });
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <span className="text-sm text-ink-2">
          선택: <b className="text-ink">{picked ? `${picked.name} (${picked.ticker})` : "없음"}</b>
        </span>
        <label className="flex items-center gap-1 text-sm text-ink-2">
          <input type="checkbox" checked={useLlm} onChange={(e) => setUseLlm(e.target.checked)} />
          AI 뉴스 감성
        </label>
        <button
          disabled={!picked || busy}
          onClick={run}
          className="rounded-lg bg-accent px-4 py-2 text-sm font-semibold text-white
                     transition hover:brightness-110 disabled:opacity-40"
        >
          {busy ? "분석 중…" : "분석 시작"}
        </button>
      </div>

      {busy && (
        <div className="rounded-card border border-term-border bg-term-1 p-8 text-center">
          <div className="mx-auto mb-3 h-1 w-48 overflow-hidden rounded-full bg-term-border">
            <div className="loading-bar-fill h-full w-1/3 rounded-full bg-accent" />
          </div>
          <p className="text-sm text-term-muted">
            {status === "pending" ? "분석 작업이 큐에 등록되었습니다…" : "주가·재무·뉴스 데이터를 분석하고 있습니다…"}
          </p>
        </div>
      )}

      {error && (
        <div className="rounded-card border border-loss/40 bg-loss/10 p-4 text-sm text-loss">
          {error}
        </div>
      )}

      {result && !busy && <SignalCard a={result} />}
    </div>
  );
}
