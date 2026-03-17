import { API_CONFIG } from '../config';
import type { MonAlertTrade, TradeViolationsResult, PreferencesResponse } from './types';

const BASE_URL = `${API_CONFIG.baseURL}/violations_monalert`;

export interface AddTradeFlowTrace {
  request_id?: string;
  ticker?: string;
  trade_id?: number;
  force_recompute_ms?: number;
  total_backend_ms?: number;
  result_ready?: boolean;
  direct_compute?: {
    success?: boolean;
    trade_id?: number;
    ticker?: string;
    daily_rows?: number;
    spy_rows?: number;
    bars_5m?: number;
    daily_ms?: number;
    spy_ms?: number;
    bars_5m_ms?: number;
    precompute_ms?: number;
    compute_ms?: number;
    elapsed_ms?: number;
    error?: string;
    reason?: string;
  };
  fetch?: {
    success?: boolean;
    provider?: string;
    created?: boolean;
    force_full?: boolean;
    fetch_profile?: {
      daily_period?: string;
      intraday_period?: string;
    };
    fetch_ms?: number;
    metrics_ms?: number;
    elapsed_ms?: number;
    daily_count?: number;
    m5_count?: number;
    reason?: string;
    error?: string;
  };
}

export interface AddTradeResponse extends MonAlertTrade {
  initial_result?: TradeViolationsResult | null;
  flow_trace?: AddTradeFlowTrace;
}

export interface SmaChartSetting {
  length: number;
  r: number;
  g: number;
  b: number;
  opacity: number;
  thickness: number;
  enabled: boolean;
  source: SmaSource;
}

export type SmaSource = 'close' | 'open' | 'high' | 'low';

export interface ChartSettingsResponse {
  ticker: string;
  sma_settings: SmaChartSetting[];
  daily_sma_settings: SmaChartSetting[];
  weekly_sma_settings: SmaChartSetting[];
  highlight_marker_gap: number;
  open_on_bars: boolean;
}

// ---------------------------------------------------------------------------
// Trades CRUD
// ---------------------------------------------------------------------------

export async function fetchTrades(): Promise<MonAlertTrade[]> {
  const res = await fetch(`${BASE_URL}/trades/`);
  if (!res.ok) throw new Error('Failed to fetch trades');
  return res.json();
}

export async function createTrade(data: {
  ticker: string;
  start_date: string;
  end_date?: string | null;
  use_latest_end_date?: boolean;
}): Promise<AddTradeResponse> {
  const res = await fetch(`${BASE_URL}/trades/`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      ...data,
      is_active: true,
      use_latest_end_date: data.use_latest_end_date ?? true,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || err.error || 'Failed to create trade');
  }
  return res.json();
}

export async function updateTrade(
  id: number,
  data: Partial<MonAlertTrade>
): Promise<MonAlertTrade> {
  const res = await fetch(`${BASE_URL}/trades/${id}/`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error('Failed to update trade');
  return res.json();
}

export async function deleteTrade(id: number): Promise<void> {
  const res = await fetch(`${BASE_URL}/trades/${id}/`, { method: 'DELETE' });
  if (!res.ok) throw new Error('Failed to delete trade');
}

// ---------------------------------------------------------------------------
// Violations computation
// ---------------------------------------------------------------------------

export async function computeViolations(tradeId: number): Promise<TradeViolationsResult> {
  const res = await fetch(`${BASE_URL}/compute/${tradeId}/`);
  if (!res.ok) throw new Error('Failed to compute violations');
  return res.json();
}

export async function computeSessionViolations(
  tradeId: number,
  data: {
    start_date?: string;
    end_date?: string | null;
    use_latest_end_date?: boolean;
  },
  options?: {
    signal?: AbortSignal;
  }
): Promise<TradeViolationsResult> {
  const res = await fetch(`${BASE_URL}/compute-session/${tradeId}/`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
    signal: options?.signal,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    const message = err.detail || err.error || 'Failed to compute chart session violations';
    const apiError = new Error(message) as Error & { status?: number; payload?: unknown };
    apiError.status = res.status;
    apiError.payload = err;
    throw apiError;
  }
  return res.json();
}

export async function computeAllViolations(): Promise<TradeViolationsResult[]> {
  const res = await fetch(`${BASE_URL}/compute-all/`);
  if (!res.ok) throw new Error('Failed to compute all violations');
  return res.json();
}

const GLOBAL_SETTINGS_KEY = '__global__';

export async function fetchChartSettings(): Promise<ChartSettingsResponse> {
  const res = await fetch(`${BASE_URL}/chart-settings/${GLOBAL_SETTINGS_KEY}/`);
  if (!res.ok) throw new Error('Failed to fetch chart settings');
  return res.json();
}

export async function updateChartSettings(
  dailySmaSettings: SmaChartSetting[],
  weeklySmaSettings: SmaChartSetting[],
  highlightMarkerGap: number,
  openOnBars: boolean
): Promise<ChartSettingsResponse> {
  const res = await fetch(`${BASE_URL}/chart-settings/${GLOBAL_SETTINGS_KEY}/`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      daily_sma_settings: dailySmaSettings,
      weekly_sma_settings: weeklySmaSettings,
      highlight_marker_gap: highlightMarkerGap,
      open_on_bars: openOnBars,
    }),
  });
  if (!res.ok) throw new Error('Failed to update chart settings');
  return res.json();
}

/** Force a synchronous recompute — used after date/trade changes for instant results. */
export async function forceComputeAll(): Promise<TradeViolationsResult[]> {
  const res = await fetch(`${BASE_URL}/compute-all/`, { method: 'POST' });
  if (!res.ok) throw new Error('Failed to force compute all violations');
  return res.json();
}

export async function fetchComputeStatus(): Promise<{
  running: boolean;
  cached_trades: number;
  last_compute: string | null;
  is_computing: boolean;
  is_hard_computing: boolean;
  message: string;
}> {
  const res = await fetch(`${BASE_URL}/compute-status/`);
  if (!res.ok) throw new Error('Failed to fetch compute status');
  return res.json();
}

// ---------------------------------------------------------------------------
// Preferences
// ---------------------------------------------------------------------------

export async function fetchPreferences(): Promise<PreferencesResponse> {
  const res = await fetch(`${BASE_URL}/preferences/`);
  if (!res.ok) throw new Error('Failed to fetch preferences');
  return res.json();
}

export async function updatePreferences(
  preferences: Record<string, boolean>
): Promise<{ preferences: Record<string, boolean> }> {
  const res = await fetch(`${BASE_URL}/preferences/`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ preferences }),
  });
  if (!res.ok) throw new Error('Failed to update preferences');
  return res.json();
}

// ---------------------------------------------------------------------------
// Data refresh
// ---------------------------------------------------------------------------

export async function refreshHistoricalData(ticker: string): Promise<{ count: number }> {
  const res = await fetch(`${BASE_URL}/refresh/`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ticker }),
  });
  if (!res.ok) throw new Error('Failed to refresh data');
  return res.json();
}
