// types/index.ts
import { Trade } from '@/TradeHistoryPage/types/Trade';

export interface MonthlyStats {
  tradingMonth: string;
  averageGain: number;
  averageLoss: number;
  winningPercentage: number;
  totalTrades: number;
  largestGain: number;
  largestLoss: number;
  avgDaysGains: number;
  avgDaysLoss: number;
  isInTrailingYear: boolean;
  useInYearly: boolean;
  netProfitLoss: number; // Compounded net profit/loss for the month
  equityChangePct: number; // Dollar returns as % of equity at start of month
}

export interface YearlyStats {
  winningPercentage: number;
  averageGain: number;
  averageLoss: number;
  winLossRatio: number;
  expectedValuePerTrade: number;
  expectedReturnOn10Trades_125?: number; // 12.5% position sizing
  expectedReturnOn50Trades_125?: number; // 12.5% position sizing
  expectedReturnOn10Trades_25?: number;  // 25% position sizing
  expectedReturnOn50Trades_25?: number;  // 25% position sizing
  avgLargestGain: number;
  avgLargestLoss: number;
  avgLargestGainLossRatio: number;
  avgDaysGains: number;
  avgDaysLoss: number;
  avgDaysRatio: number;
}

export interface ExtendedFilters extends Partial<Trade> {
  // minEarningsQuality?: number;
  // minFundamentalsQuality?: number;
  maxPriceTightness?: number;
  maxNrBases?: number;
  pctOff52WHigh?: number;
}

// Map of metricId -> array of selected optionIds
export type MetricOptionFilters = Record<number, number[]>;