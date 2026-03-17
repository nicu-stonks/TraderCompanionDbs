import axios from 'axios';
import { API_CONFIG } from '@/config';

const api = axios.create({
  baseURL: API_CONFIG.tickerDataBaseURL
});

export interface OHLCVBar {
  date: string;   // "YYYY-MM-DD"
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface HistoricalResponse {
  symbol: string;
  count: number;
  data: OHLCVBar[];
}

export type OhlcvTimeframe = 'daily' | 'weekly';

export const ohlcvAPI = {
  getHistoricalData: (symbol: string) =>
    api.get<HistoricalResponse>(`/historical/${symbol}`),
  getHistoricalWeeklyData: (symbol: string) =>
    api.get<HistoricalResponse>(`/historical_weekly/${symbol}`),
};
