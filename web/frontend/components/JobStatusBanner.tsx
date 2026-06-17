"use client";

import type { JobStatus } from "@/lib/api-types";

type RunState = "idle" | JobStatus;

/** 비동기 Job 상태 배너 — pending(큐)/processing(처리)/error 를 시각화. completed 는 호출부가 결과 렌더. */
export function JobStatusBanner({
  status,
  error,
  processingHint = "워커가 작업을 처리 중입니다…",
}: {
  status: RunState;
  error: string | null;
  processingHint?: string;
}) {
  if (status === "idle" || status === "completed") return null;

  if (status === "error") {
    return (
      <div className="rounded-card border border-loss/40 bg-loss/10 p-4 text-sm text-loss">
        ❌ {error ?? "작업 실패"}
      </div>
    );
  }

  const isQueued = status === "pending";
  return (
    <div className="flex items-center gap-3 rounded-card border border-term-border bg-term-1 p-5 text-sm text-term-muted">
      <span className="relative flex h-3 w-3">
        <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-accent/60" />
        <span className="relative inline-flex h-3 w-3 rounded-full bg-accent" />
      </span>
      <div>
        <div className="font-semibold text-term-ink">
          {isQueued ? "⏳ 큐 대기 중 (Pending)" : "⚙️ 처리 중 (Processing)"}
        </div>
        <div className="mt-0.5">
          {isQueued ? "워커가 작업을 가져갈 때까지 대기합니다." : processingHint}
        </div>
      </div>
    </div>
  );
}
