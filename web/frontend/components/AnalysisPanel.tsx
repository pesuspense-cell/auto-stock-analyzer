"use client";

import { useEffect, useState } from "react";
import { api, ApiError } from "@/lib/api";
import type { AnalysisResponse } from "@/lib/types";
import type { StockHit } from "@/lib/api-types";
import { SignalCard } from "./SignalCard";

/**
 * 차트 분석 탭 본문 — 종목 선택 시 POST /analysis 한 번만 호출.
 * 기존 Streamlit 의 전체 rerun + 세션 캐시 파이프라인을 단일 비동기 호출로 대체한다.
 */
export function AnalysisPanel({ picked }: { picked: StockHit | null }) {
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [useLlm, setUseLlm] = useState(false);

  // 선택 종목이 바뀌면 이전 결과를 비운다(다른 종목인데 옛 분석이 남는 것 방지).
  // 탭 전환 시엔 picked 가 그대로라 결과가 유지된다(keep-alive).
  useEffect(() => {
    setResult(null);
    setError(null);
  }, [picked?.ticker]);

  async function run() {
    if (!picked) return;
    setLoading(true);
    setError(null);
    try {
      const res = await api.post<AnalysisResponse>("/analysis", {
        ticker: picked.ticker,
        period: "6mo",
        use_llm: useLlm,
      });
      setResult(res);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "분석 실패");
    } finally {
      setLoading(false);
    }
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
          disabled={!picked || loading}
          onClick={run}
          className="rounded-lg bg-accent px-4 py-2 text-sm font-semibold text-white
                     transition hover:brightness-110 disabled:opacity-40"
        >
          {loading ? "분석 중…" : "분석 시작"}
        </button>
      </div>

      {loading && (
        <div className="rounded-card border border-term-border bg-term-1 p-8 text-center">
          <div className="mx-auto mb-3 h-1 w-48 overflow-hidden rounded-full bg-term-border">
            <div className="loading-bar-fill h-full w-1/3 rounded-full bg-accent" />
          </div>
          <p className="text-sm text-term-muted">주가·재무·뉴스 데이터를 분석하고 있습니다…</p>
        </div>
      )}

      {error && (
        <div className="rounded-card border border-loss/40 bg-loss/10 p-4 text-sm text-loss">
          {error}
        </div>
      )}

      {result && !loading && <SignalCard a={result} />}
    </div>
  );
}
