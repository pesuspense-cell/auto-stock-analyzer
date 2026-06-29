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

// ── 섹터 ETF 등락 비교 (시장 현황) ─────────────────────────────────
export interface EtfItem {
  ticker: string;
  name: string;
  country: "KR" | "US";
  tag: string;        // 테마 라벨 (이모지 포함)
  price: number;
  changePct: number;
  isIndex: boolean;   // 지수형 ETF(정렬 우선)
}
export interface EtfsSummary {
  up: number;
  down: number;
  avg: number;
  krAvg: number;
  usAvg: number;
}
export interface EtfsResponse {
  etfs: EtfItem[];
  summary: EtfsSummary;
  asOf: string;       // ISO timestamp
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
  stopLoss: number | null;   // 실제 MTS 손절가(원). null=봇이 ATR로 산출
  takeProfit: number | null; // 실제 MTS 익절가(원). null=봇이 ATR로 산출
  addedAt: string;
  currentPrice: number | null;
  returnPct: number | null;
  evalAmount: number | null;
}
export interface PortfolioAddRequest {
  ticker: string;
  avgPrice: number;
  quantity?: number;
  stopLoss?: number | null;
  takeProfit?: number | null;
}
export interface PortfolioUpdateRequest {
  stopLoss?: number | null;
  takeProfit?: number | null;
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
  name: string | null;
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
export interface CashBalance {
  cashBalance: number; // 예수금(현금 잔고), KRW
}
export interface CashUpdateRequest {
  cashBalance: number;
}

// 시그널 봇 알림 on/off (웹 UI ↔ 봇 공유, user_settings.alert_prefs)
export const ALERT_TYPES = ["BUY", "SELL_TP", "SELL_SL", "SELL_TS", "SELL_REBAL"] as const;
export type AlertType = (typeof ALERT_TYPES)[number];
export type AlertPrefs = Partial<Record<AlertType, boolean>>;
export interface AlertPrefsResponse {
  alertPrefs: AlertPrefs;
}

// ── 비동기 Job 큐 (ASA / 백테스트 / 분석·뉴스·펀더멘털) ──────────────
export type JobKind =
  | "asa"
  | "backtest"
  | "analysis"
  | "news"
  | "fundamental"
  | "fundamental_ai"
  | "portfolio_analysis";

// ── 포트폴리오 분석 결과 (워커 portfolio_service.analyze) ────────────
export interface PfRecommendation {
  type: "reduce" | "hold" | "watch" | "add";
  icon: string;
  sector: string;
  weight: number;
  tickers: string;
  message: string;
}
export interface PfProfitTake {
  ticker: string;
  name: string;
  pnl_pct: number;
  reason: string;
}
export interface PfMissingTop {
  sector: string;
  name: string;
  score: number;
  return_5d: number;
}
export interface PfSectorScore {
  sector: string;
  name: string;
  score: number;
  return_5d: number;
  return_20d: number;
  rank: "TOP" | "NORMAL" | "BOTTOM";
}
export type PfActionLevel = "danger" | "warn" | "good" | "neutral";

/** 종목별 종합 평가 리포트 (차트신호+모멘텀+손절선). */
export interface PfHolding {
  ticker: string;
  name: string;
  quantity: number;
  avg_price: number;
  current_price: number;
  pnl_pct: number;
  eval_value: number;
  ok?: boolean;
  tech_score?: number;
  signal_label?: string;
  signal_badge?: string;
  expected_return_pct?: number | null;
  win_prob?: number | null;
  momentum_20d?: number;
  atr_pct?: number;
  stop_loss_price?: number;
  stop_distance_pct?: number;
  stop_breached?: boolean;
  action: string;
  action_level: PfActionLevel;
  reasons: string[];
}

export interface PfOverall {
  total_value: number;
  total_cost: number;
  total_pnl_pct: number;
  holdings_count: number;
  action_counts: Record<PfActionLevel, number>;
  verdict: string;
  verdict_level: PfActionLevel;
  hhi: number | null;
  is_concentrated: boolean;
  market_status: string;
}

export interface PortfolioAnalysis {
  empty: boolean;
  overall: PfOverall;
  holdings: PfHolding[];
  guide: {
    hhi: number;
    is_concentrated: boolean;
    recommendations: PfRecommendation[];
    missing_top: PfMissingTop[];
    profit_take: PfProfitTake[];
    sector_scores: PfSectorScore[];
  };
}
export type JobStatus = "pending" | "processing" | "completed" | "error";

export interface JobEnqueued<R = unknown> {
  jobId: string | null;     // 캐시 즉시응답 시 null
  status: JobStatus;
  result?: R;               // status="completed" 이면 폴링 없이 즉시 사용
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

/** 가격 차트용 캔들 + 이동평균 오버레이 (최근 N봉). */
export interface ChartCandle {
  time: string;   // YYYY-MM-DD
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}
export interface ChartLinePoint {
  time: string;
  value: number;
}
export interface ChartSeries {
  candles: ChartCandle[];
  ema20: ChartLinePoint[];
  ema50: ChartLinePoint[];
  ema200: ChartLinePoint[];
  currency: string | null;
}

export interface IndicatorsResponse {
  ticker: string;
  asOf: string;          // 최신 봉 날짜 (YYYY-MM-DD)
  indicators: IndicatorSnapshot;
  signal: SignalResult;
  chart?: ChartSeries;   // 차트 분석 탭에서 렌더 (구버전 캐시엔 없을 수 있음)
  cached: boolean;
}
export type { IndicatorSnapshot, SignalResult };
