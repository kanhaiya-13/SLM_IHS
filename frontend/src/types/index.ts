export interface StateInfo {
  id: string;
  name: string;
  best_config: string;
  window_pr_auc: number;
  window_f1: number;
  window_recall: number;
  window_precision: number;
  historical_year_range: string;
  total_recorded_weeks: number;
}

export interface StatesListResponse {
  states: StateInfo[];
  default_state: string;
  total_states: number;
}

export interface WeekRecord {
  year: number;
  week: number;
  date_label: string;
  Cases: number;
  preci: number;
  LAI: number;
  Temp: number;
  outbreak: number;
  is_covered_week: boolean;
}

export interface HistoryResponse {
  state: string;
  requested_weeks: number;
  total_records: number;
  data: WeekRecord[];
  data_snapshot_date: string;
}

export interface CustomWeekInput {
  week_index: number;
  Cases: number;
  preci: number;
  LAI: number;
  Temp: number;
  week_num?: number;
  year?: number;
}

export interface CustomForecastRequest {
  state: string;
  weeks: CustomWeekInput[];
}

export interface ForecastResponse {
  state: string;
  window_level_probability: number;
  per_lead_week: {
    "t+4": number;
    "t+5": number;
    "t+6": number;
  };
  risk_label: "Low" | "Moderate" | "High";
  risk_color: "emerald" | "amber" | "rose";
  risk_description: string;
  threshold_guide: {
    Low: string;
    Moderate: string;
    High: string;
  };
  input_window_summary: WeekRecord[];
  input_date_range: string;
  forecast_lead_range: string;
  model_used: string;
  data_snapshot_note: string;
  is_simulated: boolean;
}

export interface ModelMetricSet {
  precision: number;
  recall: number;
  f1: number;
  roc_auc: number;
  pr_auc: number;
}

export interface BaselineComparison {
  model: string;
  precision: number;
  recall: number;
  f1: number;
  roc_auc: number;
  pr_auc: number;
  is_selected_best: boolean;
}

export interface ModelInfoResponse {
  state: string;
  best_config_name: string;
  architecture_type: string;
  channel_strategy: string;
  weather_features: boolean;
  calendar_features: boolean;
  active_channels: string[];
  plain_english_summary: string;
  test_metrics_window: ModelMetricSet;
  test_metrics_leads: {
    "t+4": ModelMetricSet;
    "t+5": ModelMetricSet;
    "t+6": ModelMetricSet;
  };
  baseline_comparison: BaselineComparison[];
  interpretation: string;
  limitations_and_disclaimer: string;
}
