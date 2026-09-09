"""
schemas.py — Pydantic models for the Outbreak Forecasting Dashboard API.

Defines all request and response structures with strict typing and validation.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator


class StateInfo(BaseModel):
    id: str = Field(..., description="Unique state identifier, e.g. 'Maharashtra'")
    name: str = Field(..., description="State display name")
    best_config: str = Field(..., description="Best performing model configuration, e.g. 'CI_W_NC'")
    window_pr_auc: float = Field(..., description="Test set window-level PR-AUC")
    window_f1: float = Field(..., description="Test set window-level F1 score")
    window_recall: float = Field(..., description="Test set window-level Recall")
    window_precision: float = Field(..., description="Test set window-level Precision")
    historical_year_range: str = Field("2009–2022", description="Data time span")
    total_recorded_weeks: int = Field(..., description="Total weeks in historical grid")


class StatesListResponse(BaseModel):
    states: List[StateInfo]
    default_state: str = "Maharashtra"
    total_states: int = 3


class WeekRecord(BaseModel):
    year: int = Field(..., description="Calendar year")
    week: int = Field(..., description="Week number of year (1-52)")
    date_label: str = Field(..., description="Formatted string e.g. '2022-W52'")
    Cases: float = Field(..., description="Aggregated vector-borne case count")
    preci: float = Field(..., description="Precipitation (mm)")
    LAI: float = Field(..., description="Leaf Area Index (vegetation density)")
    Temp: float = Field(..., description="Surface temperature (°C)")
    outbreak: int = Field(..., description="1 if confirmed outbreak week (Cases > 0), else 0")
    is_covered_week: bool = Field(..., description="True if present in IDSP surveillance reports")


class HistoryResponse(BaseModel):
    state: str
    requested_weeks: int
    total_records: int
    data: List[WeekRecord]
    data_snapshot_date: str = Field(..., description="Date range covered in the response")


class CustomWeekInput(BaseModel):
    week_index: int = Field(..., ge=1, le=12, description="1-indexed position in 12-week window (1..12)")
    Cases: float = Field(..., ge=0, description="Vector-borne case count (>= 0)")
    preci: float = Field(..., ge=0, description="Precipitation in mm (>= 0)")
    LAI: float = Field(..., ge=0, description="Leaf Area Index (>= 0)")
    Temp: float = Field(..., description="Temperature in °C")
    week_num: Optional[int] = Field(None, ge=1, le=53, description="Optional week number 1..52")
    year: Optional[int] = Field(None, ge=2000, le=2050, description="Optional calendar year")


class CustomForecastRequest(BaseModel):
    state: str = Field(..., description="Target state ('Maharashtra', 'Karnataka', 'Tamil Nadu')")
    weeks: List[CustomWeekInput] = Field(..., description="Array of exactly 12 weekly measurements")

    @field_validator("weeks")
    @classmethod
    def validate_12_weeks(cls, v: List[CustomWeekInput]) -> List[CustomWeekInput]:
        if len(v) != 12:
            raise ValueError(f"Custom forecast requires exactly 12 weeks of historical input, got {len(v)}.")
        return sorted(v, key=lambda x: x.week_index)


class ForecastResponse(BaseModel):
    state: str = Field(..., description="State evaluated")
    window_level_probability: float = Field(..., description="Probability of >=1 outbreak week in t+4..t+6")
    per_lead_week: Dict[str, float] = Field(..., description="Breakdown of probabilities for t+4, t+5, t+6")
    risk_label: str = Field(..., description="'Low', 'Moderate', or 'High'")
    risk_color: str = Field(..., description="'emerald', 'amber', or 'rose'")
    risk_description: str = Field(..., description="Actionable recommendation for public health officer")
    threshold_guide: Dict[str, str] = Field(
        default_factory=lambda: {
            "Low": "P < 0.35: Baseline surveillance & routine vector source reduction",
            "Moderate": "0.35 <= P < 0.65: Heightened vigilance, prepare larvicide & fogging reserves",
            "High": "P >= 0.65: High outbreak probability 4-6 weeks out; mobilize district field teams"
        }
    )
    input_window_summary: List[WeekRecord] = Field(..., description="12 weeks of data used for the prediction")
    input_date_range: str = Field(..., description="e.g. '2022-W41 to 2022-W52'")
    forecast_lead_range: str = Field(..., description="e.g. 'Weeks t+4 to t+6 (2023-W04 to 2023-W06)'")
    model_used: str = Field(..., description="Model architecture & ablation tag")
    data_snapshot_note: str = Field(..., description="Clarification on historical snapshot date")
    is_simulated: bool = Field(False, description="True if produced by a custom scenario simulation")


class ModelMetricSet(BaseModel):
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float


class ModelInfoResponse(BaseModel):
    state: str
    best_config_name: str
    architecture_type: str
    channel_strategy: str
    weather_features: bool
    calendar_features: bool
    active_channels: List[str]
    plain_english_summary: str
    test_metrics_window: ModelMetricSet
    test_metrics_leads: Dict[str, ModelMetricSet]
    baseline_comparison: List[Dict[str, Any]]
    interpretation: str
    limitations_and_disclaimer: str
