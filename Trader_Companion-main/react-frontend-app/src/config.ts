export const API_CONFIG = {
  // For Vite, use import.meta.env
  baseURL: import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000',
  tickerDataBaseURL: import.meta.env.VITE_TICKER_DATA_BASE_URL || 'http://127.0.0.1:8000/ticker_data/api/ticker_data',
};