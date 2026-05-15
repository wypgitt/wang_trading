/**
 * Trade Ideas API types and React Query hook.
 *
 * Shape mirrors docs/api_contracts_v2.md §1.1 (the v2 superset).
 * Every v2-only field is nullable so older payloads from the v1 service
 * still parse without runtime errors.
 */

import { useQuery, type UseQueryResult } from "@tanstack/react-query";
import { apiFetch, type ApiEnvelope, type RegimeSnapshot } from "./client";

export type TradeAction = "BUY" | "SELL" | "WATCH" | "MODEL_REQUIRED" | "NO_DATA" | "ERROR";

export interface TopShapFeature {
  feature: string;
  value: number;
  contribution: number;
  abs_contribution: number;
  percentile: number | null;
}

export interface StageLatencies {
  data_fetch?: number;
  feature_compute?: number;
  signal_generation?: number;
  meta_inference?: number;
  sizing?: number;
  target_generation?: number;
  [stage: string]: number | undefined;
}

export interface TradeIdeaError {
  code: string;
  message: string;
  stage?: string | null;
}

export interface TradeIdea {
  // Identity
  symbol: string;
  action: TradeAction;

  // Sizing (nullable when action is ERROR / NO_DATA / MODEL_REQUIRED)
  target_weight: number | null;
  target_notional: number | null;
  estimated_quantity: number | null;
  latest_price: number | null;
  latest_bar_at: string | null;
  bar_type: string | null;
  bars_loaded: number | null;
  feature_rows: number | null;

  // Signals
  signal_count: number | null;
  top_signal_family: string | null;
  top_signal_side: number | null;
  top_signal_confidence: number | null;
  avg_signal_confidence: number | null;

  // Model
  meta_probability: number | null;
  calibrated_probability: number | null;

  // v2-only
  regime: RegimeSnapshot | null;
  regime_fit_score: number | null;
  bet_size: number | null;
  sizing_constraints_applied: string[] | null;
  strategy: string | null;
  reason: string | null;
  expected_cost_bps: number | null;
  top_shap_feature: TopShapFeature | null;
  track_record_win_rate: number | null;
  track_record_n: number | null;
  stage_latency_seconds: StageLatencies | null;

  errors: TradeIdeaError[];
}

export interface TradeIdeasTotals {
  buy: number;
  sell: number;
  watch: number;
  model_required: number;
  no_data: number;
  error: number;
  gross_target_weight: number;
  net_target_weight: number;
}

export interface TradeIdeasResponse {
  idea_count: number;
  totals: TradeIdeasTotals;
  ideas: TradeIdea[];
}

export const TRADE_IDEAS_QUERY_KEY = ["trade-ideas"] as const;

export function useTradeIdeas(): UseQueryResult<ApiEnvelope<TradeIdeasResponse>, Error> {
  return useQuery<ApiEnvelope<TradeIdeasResponse>, Error>({
    queryKey: TRADE_IDEAS_QUERY_KEY,
    queryFn: () => apiFetch<TradeIdeasResponse>("/api/v1/trade-ideas"),
    staleTime: 30_000,
    // refetchInterval intentionally undefined — opt in per page.
  });
}
