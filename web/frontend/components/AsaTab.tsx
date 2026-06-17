"use client";

import { useState } from "react";

import { AuthGate, UserMenu } from "@/components/AuthGate";
import { useJob } from "@/hooks/useJob";
import { JobStatusBanner } from "@/components/JobStatusBanner";
import type { AsaResult } from "@/lib/api-types";

/**
 * ASA 추천 탭 — jobs 큐 기반 비동기.
 * POST /api/v1/asa/run 으로 작업을 큐에 넣고, Python 워커가 Supabase 포트폴리오를 참조해
 * 처리한 결과를 폴링으로 받아 출력한다. (로그인 필요 — 본인 포트폴리오 참조)
 */
export function AsaTab() {
  return (
    <AuthGate>
      {() => <AsaRunner />}
    </AuthGate>
  );
}

function AsaRunner() {
  const [cash, setCash] = useState(1_000_000);
  const { status, result, error, busy, enqueue } = useJob<AsaResult>();

  return (
    <div className="space-y-4">
      <header className="flex items-start justify-between gap-2">
        <div>
          <h2 className="text-lg font-bold text-ink">🤖 ASA 추천</h2>
          <p className="text-sm text-ink-2">
            장 마감 데이터 기준 내일의 매매 지침을 자동 산출합니다. 작업은 백그라운드 워커가 처리하며 수 분 소요될 수 있습니다.
          </p>
        </div>
        <UserMenu />
      </header>

      <div className="rounded-card border border-hairline bg-surface p-5 shadow-card">
        <p className="mb-2 text-sm text-ink-2">💼 로그인된 Supabase 포트폴리오가 보유 포지션으로 자동 반영됩니다.</p>
        <label className="mb-1 block text-sm font-medium text-ink">💰 예치금 (가용 현금, 원)</label>
        <div className="flex flex-wrap items-center gap-3">
          <input
            type="number" min={0} step={100_000} value={cash}
            onChange={(e) => setCash(Number(e.target.value))}
            className="w-48 rounded-lg border border-hairline-md bg-surface px-3 py-2 text-sm tabular-nums text-ink outline-none focus:border-accent focus:ring-2 focus:ring-accent/20"
          />
          <button
            onClick={() => enqueue("/api/v1/asa/run", { cash })}
            disabled={busy}
            className="rounded-lg bg-accent px-4 py-2 text-sm font-semibold text-white hover:brightness-110 disabled:opacity-40"
          >
            {busy ? "처리 중…" : "▶ ASA 추천 실행"}
          </button>
          <span className="text-sm text-ink-2">입력: <b className="tnum text-ink">{cash.toLocaleString("ko-KR")}원</b></span>
        </div>
      </div>

      <JobStatusBanner status={status} error={error} processingHint="📊 ASA 분석 실행 중 — 워커가 처리 중입니다 (수 분 소요)…" />

      {result?.output && (
        <pre className="max-h-[60vh] overflow-auto rounded-card border border-term-border bg-term-1 p-4 text-[0.78rem] leading-relaxed text-term-ink">
          {result.output}
        </pre>
      )}
    </div>
  );
}
