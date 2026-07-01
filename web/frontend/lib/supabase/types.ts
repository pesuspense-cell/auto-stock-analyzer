// Supabase 테이블 타입. 운영에선 `supabase gen types typescript` 로 자동생성 권장.
// 여기서는 schema.sql 과 1:1로 손수 정의해 타입 안정성을 확보한다.
//
// NOTE: @supabase/supabase-js v2.108+ 의 GenericTable 은 `Relationships` 필드를
//       요구한다. 누락 시 Schema 가 never 로 붕괴해 모든 쿼리가 never 가 된다.
//       (FK 관계 추론이 필요 없으면 빈 배열 []로 충분)

export interface Database {
  public: {
    Tables: {
      stocks: {
        Row: { ticker: string; name: string; name_kr: string | null; market: string | null; is_etf: boolean; updated_at: string };
        Insert: { ticker: string; name: string; name_kr?: string | null; market?: string | null; is_etf?: boolean };
        Update: Partial<Database["public"]["Tables"]["stocks"]["Insert"]>;
        Relationships: [];
      };
      quote_cache: {
        Row: { ticker: string; price: number; prev_close: number | null; change_pct: number | null; volume: number | null; currency: string | null; is_realtime: boolean; fetched_at: string };
        Insert: { ticker: string; price: number; prev_close?: number | null; change_pct?: number | null; volume?: number | null; currency?: string | null; is_realtime?: boolean; fetched_at?: string };
        Update: Partial<Database["public"]["Tables"]["quote_cache"]["Insert"]>;
        Relationships: [];
      };
      exchange_rate_cache: {
        Row: { pair: string; rate: number; change_pct: number | null; fetched_at: string };
        Insert: { pair: string; rate: number; change_pct?: number | null; fetched_at?: string };
        Update: Partial<Database["public"]["Tables"]["exchange_rate_cache"]["Insert"]>;
        Relationships: [];
      };
      market_cache: {
        Row: { scope: string; payload: unknown; fetched_at: string };
        Insert: { scope: string; payload: unknown; fetched_at?: string };
        Update: Partial<Database["public"]["Tables"]["market_cache"]["Insert"]>;
        Relationships: [];
      };
      user_settings: {
        Row: { user_id: string; gemini_api_key: string | null; groq_api_key: string | null; dart_api_key: string | null; watchlist: unknown; cash_balance: number; alert_prefs: Record<string, boolean>; telegram_chat_id: string | null; telegram_enabled: boolean; telegram_link_token: string | null; risk_profile: "safe" | "aggressive"; updated_at: string };
        Insert: { user_id: string; gemini_api_key?: string | null; groq_api_key?: string | null; dart_api_key?: string | null; watchlist?: unknown; cash_balance?: number; alert_prefs?: Record<string, boolean>; telegram_chat_id?: string | null; telegram_enabled?: boolean; telegram_link_token?: string | null; risk_profile?: "safe" | "aggressive"; updated_at?: string };
        Update: Partial<Database["public"]["Tables"]["user_settings"]["Insert"]>;
        Relationships: [];
      };
      portfolios: {
        Row: { id: string; user_id: string; ticker: string; avg_price: number; quantity: number; stop_loss: number | null; take_profit: number | null; added_at: string };
        Insert: { user_id: string; ticker: string; avg_price: number; quantity?: number; stop_loss?: number | null; take_profit?: number | null };
        Update: Partial<Database["public"]["Tables"]["portfolios"]["Insert"]>;
        Relationships: [];
      };
      trade_history: {
        Row: { id: string; user_id: string; ticker: string; buy_price: number; sell_price: number; quantity: number; net_profit: number; return_rate: number; traded_at: string };
        Insert: { user_id: string; ticker: string; buy_price: number; sell_price: number; quantity: number; net_profit: number; return_rate: number };
        Update: Partial<Database["public"]["Tables"]["trade_history"]["Insert"]>;
        Relationships: [];
      };
      jobs: {
        Row: { id: string; user_id: string | null; kind: string; status: string; params: unknown; result: unknown; error: string | null; created_at: string; updated_at: string };
        Insert: { user_id?: string | null; kind: string; status?: string; params?: unknown; result?: unknown; error?: string | null };
        Update: Partial<Database["public"]["Tables"]["jobs"]["Insert"]>;
        Relationships: [];
      };
    };
    Views: Record<string, never>;
    Functions: Record<string, never>;
    Enums: Record<string, never>;
    CompositeTypes: Record<string, never>;
  };
}
