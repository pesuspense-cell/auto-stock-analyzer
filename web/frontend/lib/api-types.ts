// API Route 요청/응답 계약 — 클라이언트(SWR)와 서버(route.ts)가 공유한다.

export interface QuoteResponse {
  ticker: string;
  price: number;
  prevClose: number | null;
  changePct: number | null;
  volume: number | null;
  currency: string | null;
  isRealtime: boolean;
  fetchedAt: string;
  cached: boolean; // true면 DB 히트(외부 API 미호출) — 디버깅/관측용
}

export interface RateItem {
  pair: string;
  rate: number;
  changePct: number | null;
}
export interface RatesResponse {
  rates: RateItem[];
}

export interface StockHit {
  ticker: string;
  name: string;
  nameKr: string | null;
  market: string | null;
  isEtf: boolean;
}
export interface StockSearchResponse {
  query: string;
  results: StockHit[];
}

export interface ApiError {
  error: string;
}

// ── 포트폴리오 ──────────────────────────────────────────────────────
export interface PortfolioItem {
  id: string;
  ticker: string;
  name: string | null;
  avgPrice: number;
  quantity: number;
  addedAt: string;
  currentPrice: number | null;
  returnPct: number | null;
  evalAmount: number | null;
}
export interface PortfolioAddRequest {
  ticker: string;
  avgPrice: number;
  quantity?: number;
}
export interface SellRequest {
  sellPrice: number;
  quantity?: number | null; // null = 전량
}
export interface SellResponse {
  ok: boolean;
  netProfit?: number;
  returnRate?: number;
  error?: string;
}
export interface TradeItem {
  id: string;
  ticker: string;
  buyPrice: number;
  sellPrice: number;
  quantity: number;
  netProfit: number;
  returnRate: number;
  tradedAt: string;
}
export interface OkResponse {
  ok: boolean;
  error?: string;
}

// ── 비동기 Job 큐 (ASA / 백테스트) ──────────────────────────────────
export type JobKind = "asa" | "backtest";
export type JobStatus = "pending" | "processing" | "completed" | "error";

export interface JobEnqueued {
  jobId: string;
  status: JobStatus;
}

export interface JobView<R = unknown> {
  id: string;
  kind: JobKind;
  status: JobStatus;
  result: R | null;
  error: string | null;
  createdAt: string;
  updatedAt: string;
}

/** ASA 결과 페이로드. */
export interface AsaResult {
  output: string;
}

/** 백테스트 결과 페이로드 (worker 가 jobs.result 에 기록). */
export interface BacktestJobResult {
  metrics: Record<string, number>;
  equity_curve: Record<string, unknown>[];
  trade_log: Record<string, unknown>[];
  selected_stocks: Record<string, unknown>[];
  benchmark: Record<string, unknown>[];
  benchmark_label: string;
  log_text: string;
}

// ── 기술지표 (TS 포팅) ──────────────────────────────────────────────
import type { IndicatorSnapshot, SignalResult } from "@/lib/indicators/signals";

export interface IndicatorsResponse {
  ticker: string;
  asOf: string;          // 최신 봉 날짜 (YYYY-MM-DD)
  indicators: IndicatorSnapshot;
  signal: SignalResult;
  cached: boolean;
}
export type { IndicatorSnapshot, SignalResult };
