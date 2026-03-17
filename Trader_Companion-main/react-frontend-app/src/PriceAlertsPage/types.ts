export interface Alert {
  id: number;
  ticker: string;
  alert_price: number;
  is_active: boolean;
  triggered: boolean;
  created_at: string;
  triggered_at: string | null;
  current_price: number | null;
  last_checked: string | null;
  initial_price_above_alert: boolean | null;
  previous_close: number | null;
  percent_change: number | null;
}

export interface AlarmSettings {
  id: number;
  alarm_sound_path: string;
  play_duration: number;
  pause_duration: number;
  cycles: number;
}

export interface CreateAlertData {
  ticker: string;
  alert_price: number;
}


