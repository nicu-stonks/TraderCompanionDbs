export type ScreenerType = 'OBLIGATORY' | 'RANKING' | 'SENTIMENT';

export interface ScreeningOptions {
  min_price_increase: number;
  ranking_method: 'price' | 'screeners'
  fetch_data: boolean;
  top_n: number;
  obligatory_screens: string[];
  ranking_screens: string[];
  skip_obligatory: boolean;
  skip_sentiment: boolean;
  sleep_after: boolean;
}