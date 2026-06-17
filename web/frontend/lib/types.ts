// 백엔드 Pydantic 스키마와 1:1 대응하는 프론트 타입.

export interface MoverItem {
  name: string;
  ticker: string;
  price: number;
  change_pct: number;
}

export interface MarketOverview {
  indices: { name: string; symbol: string; price: number; change_pct: number }[];
  gainers: MoverItem[];
  losers: MoverItem[];
}

export interface ExchangeRate {
  pair: string;
  rate: number;
  change_pct: number | null;
}

export interface StockHit {
  display: string;
  name: string;
  ticker: string;
}

export interface Signal {
  score: number;
  label: string;
  badge: string;
  reasons: string[];
}

export interface HybridSignal {
  hybrid_score: number;
  combined_score: number;
  label: string;
  badge: string;
  reasons: string[];
  warnings: string[];
}

export interface RealtimePrice {
  price: number;
  ts: string;
  is_realtime: boolean;
  stale: boolean;
  stale_msg: string;
}

export interface AnalysisResponse {
  ticker: string;
  sname: string;
  period: string;
  data_ready: boolean;
  signals: Signal;
  hybrid: HybridSignal;
  realtime: RealtimePrice;
  advanced: Record<string, unknown>;
  expected: Record<string, unknown> | null;
  risk_adj: Record<string, unknown>;
  fund_score_data: Record<string, unknown>;
  fund_info: Record<string, unknown>;
  news_result: Record<string, unknown>;
  dead_time: Record<string, unknown>;
  breakout: Record<string, unknown>;
  vol_anomaly: Record<string, unknown>;
  tech_score: number;
  news_score: number;
}

export interface OhlcPoint {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface PortfolioItem {
  id: number;
  ticker: string;
  avg_price: number;
  quantity: number;
  added_at: string;
  name: string | null;
  current_price: number | null;
  return_pct: number | null;
  eval_amount: number | null;
}

export interface LoginResponse {
  ok: boolean;
  token: string;
  user_id: number;
  email: string;
}
