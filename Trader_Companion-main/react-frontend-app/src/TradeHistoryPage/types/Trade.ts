// types/Trade.ts

export interface Trade {
  ID: number;
  Ticker: string;
  Status: string;
  Entry_Date: string;
  Exit_Date: string | null;
  Entry_Price: number;
  Exit_Price: number | null;
  Return: number | null;
  Pattern: string;
  Price_Tightness_1_Week_Before: number;
  Exit_Reason: string;
  Market_Condition: string;
  Category: string;
  Nr_Bases: number;
  Has_Earnings_Acceleration: boolean;
  IPO_Last_10_Years: boolean;
  Is_BioTech: boolean;
  Under_30M_Shares: boolean;
  Case: string;
  If_You_Could_Only_Make_10_Trades: boolean;
  Pct_Off_52W_High: number;
  C: boolean;
  A: boolean;
  N: boolean;
  S: boolean;
  L: boolean;
  I: boolean;
  M: boolean;
  Pct_Of_Equity: number;
  streakId?: number;
}