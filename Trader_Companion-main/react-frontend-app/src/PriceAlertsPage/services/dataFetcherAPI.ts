import axios from 'axios';

// Ticker data fetcher is served by Django on port 8000
const api = axios.create({
  baseURL: 'http://localhost:8000/ticker_data/api/ticker_data'
});

export interface DataFetcherStatus {
  running: boolean;
  market_open: boolean;
  market_message: string;
  current_time: string;
  market_hours: string;
  tickers_count: number;
  current_ticker_index: number;
  max_records_per_ticker: number;
  request_interval_seconds: number;
  request_interval_ms: number;
  last_cleanup_date: string | null;
  price_alert_tickers_count: number;
}

export interface RequestInterval {
  interval_seconds: number;
  interval_ms: number;
}

export const dataFetcherAPI = {
  getStatus: () =>
    api.get<DataFetcherStatus>('/status'),

  getRequestInterval: () =>
    api.get<RequestInterval>('/request-interval'),

  setRequestInterval: (intervalMs: number) =>
    api.post<RequestInterval & { message: string }>('/request-interval', { interval_ms: intervalMs }),
};
