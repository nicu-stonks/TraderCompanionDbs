// Shared types extracted from monolithic CustomOrdersPage for reuse across new tab components
export interface OrderConfig {
  ticker: string;
  lower_price: number;
  higher_price: number;
  volume_requirements: string[];
  pivot_adjustment: string;
  day_high_max_percent_off: number;
  time_in_pivot: number;
  time_in_pivot_positions: string;
  data_server: string;
  trade_server: string;
  volume_multipliers: number[];
  max_day_low: number | null;
  min_day_low?: number | null;
  wait_after_open_minutes?: number;
  breakout_lookback_minutes?: number;
  breakout_exclude_minutes?: number;
  start_minutes_before_close?: number | null;
  stop_minutes_before_close?: number | null;
  request_lower_price?: number | null;
  request_higher_price?: number | null;
}

export interface NewTradeStop {
  price?: number;
  position_pct: number; // fraction of entire position to sell at this stop (0..1)
  percent_below_fill?: number;
  __ui_mode?: 'price' | 'percent';
}

export interface NewTrade {
  ticker: string;
  shares: number; // auto-calculated, 2 decimals
  risk_amount: number;
  risk_percent_of_equity?: number;
  lower_price_range: number;
  higher_price_range: number;
  order_type?: 'MKT' | 'IBALGO';
  adaptive_priority?: 'Patient' | 'Normal' | 'Urgent';
  timeout_seconds?: number;
  sell_stops: NewTradeStop[];
  consider_zero_risk?: boolean; // bypass risk limits for this trade
}

export interface ServerStatus {
  success: boolean;
  active_trades: number;
  available_risk: number;
  server_uptime: string;
  last_trade_time: string;
  trades: Array<{
    trade_id: string;
    ticker: string;
    shares: number;
    risk_amount: number;
    lower_price_range: number;
    higher_price_range: number;
    order_type?: 'MKT' | 'IBALGO';
    adaptive_priority?: 'Patient' | 'Normal' | 'Urgent' | null;
    timeout_seconds?: number;
    sell_stops: Array<{ price?: number; shares: number; percent_below_fill?: number }>;
  }>;
  error_count?: number;
  is_processing?: boolean;
}

export interface IbConnectionStatus {
  success: boolean;
  message: string;
  stage?: string;
  sample_symbol?: string;
  sample_conid?: number | string;
  checked_at?: string;
}

export interface TradeData {
  ticker?: string;
  shares?: number;
  risk_amount?: number;
  lower_price_range?: number;
  higher_price_range?: number;
  sell_stops?: Array<{ price?: number; shares: number; percent_below_fill?: number; __ui_mode?: 'price' | 'percent' }>;
  [key: string]: unknown;
}

export interface ErrorLog {
  timestamp: string;
  error_message: string;
  error_type: string;
  ticker: string;
  trade_data: TradeData;
}
