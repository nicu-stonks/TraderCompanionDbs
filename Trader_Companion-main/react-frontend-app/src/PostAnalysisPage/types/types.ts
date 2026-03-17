export interface MetricOption {
  id: number;
  name: string;
  value: number;
}

export interface Metric {
  id: number;
  name: string;
  description: string;
  options: MetricOption[];
  created_at: string;
  updated_at: string;
}


export interface TradeGrade {
  tradeId: number;
  metricId: string; // kept as string to align with existing API
  selectedOptionId: string; // always present for actual saved grades
}

// For bulk deletion (unchecking a metric)
export interface TradeGradeDeletion {
  tradeId: number;
  metricId: string;
}

export interface MetricOptionRecommendation {
  id: number;
  metric: number;
  option: number;
  recommended_pct: number;
  is_minimum: boolean;
  created_at: string;
  updated_at: string;
}

export interface MetricGradeCheckSetting {
  id: number;
  required_metrics: string;
  exclude_metric: string;
  created_at: string;
  updated_at: string;
}

export interface MetricPercentBaseSetting {
  id: number;
  metric_id: number;
  use_total_trades: boolean;
  created_at: string;
  updated_at: string;
}

export interface APIError {
  message: string;
  details?: any;
}