"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { JobEnqueued, JobStatus, JobView } from "@/lib/api-types";

type RunState = "idle" | JobStatus;

/**
 * 비동기 Job 큐 클라이언트 — enqueue 후 jobs/[id] 를 폴링해 상태를 추적한다.
 * pending(큐 대기) → processing(워커 처리 중) → completed(결과) | error.
 */
export function useJob<R = unknown>(pollMs = 2500) {
  const [status, setStatus] = useState<RunState>("idle");
  const [result, setResult] = useState<R | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);

  const stop = () => { if (timer.current) { clearInterval(timer.current); timer.current = null; } };
  useEffect(() => stop, []);

  const poll = useCallback((id: string) => {
    stop();
    timer.current = setInterval(async () => {
      try {
        const res = await fetch(`/api/v1/jobs/${id}`, { credentials: "same-origin" });
        if (!res.ok) return;
        const j = (await res.json()) as JobView<R>;
        setStatus(j.status);
        if (j.status === "completed") { setResult(j.result); stop(); }
        else if (j.status === "error") { setError(j.error ?? "작업 실패"); stop(); }
      } catch {
        /* 폴링 일시 오류 무시 */
      }
    }, pollMs);
  }, [pollMs]);

  const reset = useCallback(() => {
    stop();
    setStatus("idle"); setResult(null); setError(null); setJobId(null);
  }, []);

  const enqueue = useCallback(async (endpoint: string, body: unknown) => {
    stop();
    setStatus("pending"); setResult(null); setError(null); setJobId(null);
    try {
      const res = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "same-origin",
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const e = await res.json().catch(() => ({}));
        setError(e?.error ?? "큐 등록 실패"); setStatus("error"); return;
      }
      const { jobId: id } = (await res.json()) as JobEnqueued;
      setJobId(id);
      poll(id);
    } catch (e) {
      setError(e instanceof Error ? e.message : "요청 실패"); setStatus("error");
    }
  }, [poll]);

  const busy = status === "pending" || status === "processing";
  return { status, result, error, jobId, busy, enqueue, reset };
}
