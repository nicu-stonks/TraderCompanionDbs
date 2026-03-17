// Types for the Violations Monitor feature

export interface MonAlertTrade {
  id: number;
  ticker: string;
  start_date: string; // YYYY-MM-DD
  end_date: string | null;
  use_latest_end_date: boolean;
  is_active: boolean;
  created_at: string;
}

export interface ViolationItem {
  date: string;
  type: string;
  description: string;
  severity: 'red' | 'green' | 'orange' | 'gray';
}

export interface RiskDistributionBin {
  x: number;
  density_pct: number;
  cdf_pct?: number;
  exceed_pct?: number;
  tail_pct: number;
}

export interface RiskDistribution {
  sample_size: number;
  mean: number;
  median?: number;
  q1?: number;
  q3?: number;
  p10?: number;
  p90?: number;
  split_point?: number;
  std: number;
  min: number;
  max: number;
  bins: RiskDistributionBin[];
}

export interface RiskAnalysis {
  lookback_days: number;
  sample_size: number;
  as_of_date: string | null;
  gap_risk: RiskDistribution;
  daily_change: RiskDistribution;
  daily_spread: RiskDistribution;
}

export interface TradeViolationsResult {
  trade_id: number;
  ticker: string;
  start_date: string;
  end_date: string;
  use_latest_end_date: boolean;
  trend_up_start: string | null;
  violations: ViolationItem[];
  confirmations: ViolationItem[];
  total_violations: number;
  total_confirmations: number;
  risk_analysis?: RiskAnalysis;
  info: {
    days_up: number;
    days_down: number;
    days_up_pct: number;
    days_up_color: 'green' | 'red' | 'gray';
    good_closes: number;
    bad_closes: number;
    good_close_pct: number;
    good_close_color: 'green' | 'red' | 'gray';
    current_price?: number;
    current_change_pct?: number;
    current_volume?: number;
    interpolated_volume?: number;
    volume_50ma?: number;
    vol_vs_time_of_day_pct?: number;
  };
}

export interface CheckInfo {
  name: string;
  type: 'violation' | 'confirmation' | 'info';
  severity: 'red' | 'green' | 'orange' | 'gray';
}

export interface PreferencesResponse {
  preferences: Record<string, boolean>;
  all_checks: Record<string, CheckInfo>;
}

// Glossary entries for the info modal
export const GLOSSARY: Record<string, string> = {
  closes_on_high: "Close price is within 10% of the day's high (from the top of the daily range). Indicates buying pressure.",
  closes_on_low: "Close price is within 10% of the day's low. Indicates selling pressure.",
  down_up_largest_vol: "The largest-volume day since trend-up start. If it closes down, it is flagged as a violation; if it closes up, it is flagged as a confirmation.",
  down_50pct_vol_increase: "A down day where volume is at least 50% above the 50-day moving average volume, AND the down move magnitude is at least this ticker's average down-day size. The signal text shows the actual volume increase percentage for that day.",
  largest_pct_down_high_vol: "A new largest percentage down day on high volume (50%+ above 50-day MA) since trend-up start, but only when that down move is statistically meaningful: at least 1 standard deviation below the ticker's normal signed daily change baseline.",
  largest_down_day_pct: "The current day is the largest percentage decline since the 200MA uptrend started.",
  largest_down_week_high_vol: "Each time a week sets a new worst decline since trend-up start and weekly volume is 50%+ above the 10-week average, it is flagged. It is shown only when that record-setting week falls inside the current start/end view.",
  largest_pct_down_week: "Each time a week sets a new worst percentage decline since trend-up start, it is flagged. It is shown only when that record-setting week falls inside the current start/end view.",
  bearish_engulfing: "Current day's high is above previous day's high AND current close is below previous day's low. Bearish reversal pattern.",
  bullish_engulfing: "Current day's low is below previous day's low AND current close is above previous day's high. Bullish reversal pattern.",
  squat_reversal: "On the buy day, the stock closed in the lower 50% of its daily price range. Weak close on entry.",
  rs_new_high: "RS Line (stock price / S&P 500) is at a new 52-week high. Stock outperforming the market — same as MarketSurge RS New High.",
  rs_new_low: "RS Line (stock price / S&P 500) is at a new 52-week low. Stock underperforming the market — same as MarketSurge RS New Low.",
  up_30pct_vol_increase: "An up day where volume is at least 30% above the 50-day moving-average volume. The signal text shows the actual volume increase percentage for that day.",
  widest_spread: "Each time a day sets a new widest percentage spread (high-low)/low since trend-up start, it is flagged. It is shown only when that record-setting day falls inside the current start/end view.",
  inside_day: "Current high is lower than previous day's high AND current low is higher than previous day's low. Consolidation / tight action.",
  close_below_200ma: "Stock closed below the 200-day simple moving average. Major support broken.",
  close_above_200ma: "Stock closed above the 200-day simple moving average. Trading above major support.",
  largest_gap: "Largest percentage gap (previous close vs current open) since the trend up started.",
  extended_warning: "Method (distribution + percentile): 1) Compute current rally % = (EndClose-StartClose)/StartClose*100 for your selected window. 2) Let N be that window length in trading days. 3) Build the comparison distribution from this ticker's PRIOR N-day windows before the monitored start date (no look-ahead). Baseline rule: use price history since trend-up start; if that gives less than about 1 trading year, use 1 year of history when available. 4) Rank current rally inside that distribution using percentile rank. Trigger Extended Warning when current rally is >= 75th percentile (red signal). Layman terms: we compare this move to similar-length past moves and flag it if it is stronger than about 75% of them.",
  very_extended: "Same distribution method as Extended Warning, but stricter threshold. Baseline rule is the same: use price history since trend-up start; if that gives less than 1 trading year, use about 1 year of history when available. Trigger Very Extended when current rally is >= 90th percentile (red).",
  days_up_down: "UD:N/N (from Up Days) — Days Up / Days Down. The ratio of up days to down days in the monitoring range. Green if ≥70% up, red if ≤30% up, gray otherwise. An up day = close > previous close.",
  weekly_lower_lows: "3 or more consecutive weeks where the weekly low is lower than the previous week's low. Downtrend pattern.",
  weekly_higher_highs: "2 or more consecutive weeks where the weekly high is higher than the previous week's high. Uptrend pattern.",
  daily_megaphone: "2+ consecutive days where high > prev high AND low < prev low (engulfing pattern). Expanding volatility / indecision.",
  daily_lower_lows: "Consecutive daily lower lows. The monitor shows the actual streak length.",
  daily_higher_highs: "Consecutive daily higher highs. The monitor shows the actual streak length.",
  big_down_day: "A day where the stock declined by at least 2x the ticker's average down-day magnitude (minimum 3% floor). Baseline rule: use history since trend-up start, but include about 1 year before selected start when available (no look-ahead).",
  big_up_day: "A day where the stock gained by at least 2x the ticker's average up-day magnitude (minimum 3% floor). Baseline rule: use history since trend-up start, but include about 1 year before selected start when available (no look-ahead).",
  above_20day_20pct: "Stock price crossed above the 20% extension threshold over the 20-day moving average. The signal text shows the actual extension percentage.",
  angle_d_warning: "The 50-day SMA has been in a downtrend or flat over the last 5 trading days. Momentum slowing. Red signal. This signal is only counted if it occurs within the last 2 weeks of the selected END date.",
  gap_alert: "Signed gap % (previous close to current open) is flagged when it is outside 2 standard deviations from the ticker's historical mean gap for the pre-range baseline. Baseline rule: use history since trend-up start, but include about 1 year before selected start when available (no look-ahead). This is ticker-specific, not a hardcoded 3% rule.",
  large_squat_reversal: "Day's trading range is 2x the average daily range, and the stock closed in the lower half. Large reversal.",
  ants: "ANTS = at least 12 of the last 15 days were up days, AND average volume over those 15 days is at least 20% above the 50-day average volume baseline, AND total price gain over the 15-day window is at least 20%.",
  daily_confirmations_count: "D:N — Number of daily confirmations shown in the compact header badges for each trade.",
  weekly_confirmations_count: "W:N — Number of weekly confirmations shown in the compact header badges for each trade.",
  daily_violations_count: "D:N — Number of daily violations shown in the compact header badges for each trade.",
  weekly_violations_count: "W:N — Number of weekly violations shown in the compact header badges for each trade.",
  good_bad_close: "GC:N/N — Good Closes / Bad Closes. A 'good close' means the stock closed in the upper 50% of its daily range; 'bad close' is the lower 50%. Green if ≥70% good, red if ≤30%, gray otherwise.",
  close_below_10ma: "Stock closed below the 10-day simple moving average. Short-term weakness.",
  close_below_20ma: "Stock closed below the 20-day simple moving average. Near-term weakness.",
  close_below_50ma: "Stock closed below the 50-day simple moving average. Medium-term weakness.",
  close_above_50ma: "Stock closed above the 50-day simple moving average. Trading above medium-term support.",
  close_below_50ma_high_vol: "Stock closed below 50MA with volume 50%+ above the 50-day volume average. High-volume breakdown.",
  risk_window: "Risk analysis distributions use up to the most recent 252 trading days (~1 year). They are computed during hard compute refreshes, not every intraday lightweight pass.",
  intraday_volume_differential: "The % change to the right of volume measures the difference between today's live volume so far compared to the trailing 50-day average volume for this exact time of day (using 5-minute intervals). Positive values show unusual intraday high volume, negative shows low volume compared to historical norms.",
};

// Trend-up definition for glossary (array of paragraphs for proper rendering)
export const TREND_UP_DEFINITION = [
  "A 'trend up' is determined by the 200-day Simple Moving Average (SMA). The 200MA is checked at 20-day intervals going backwards from today. At each interval, the more recent 200MA value must be at least 0.5% above the older value. The earliest point where this condition holds continuously is the 'beginning of trend up'. Many checks reference 'since trend up' to compare current behavior against historical norms.",
  "📊 REALTIME NOTE: During market hours, today's bar uses live price data from the server. Volume for today is interpolated (projected to a full 6.5-hour trading day based on elapsed market time). For example, if 3 hours have elapsed and current volume is 1M, projected volume ≈ 2.17M. This allows volume-based checks (50%+ vol increase, high volume breakdowns) to be meaningful intraday."
];
