"""
inference.py — Model loading, feature engineering, and inference engine.

Loads trained checkpoints and state scalers to serve verified predictions for
the Outbreak Forecasting Dashboard.
"""

import math
import sys
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import torch

# Ensure src/ and backend/ are on python path
BACKEND_DIR = Path(__file__).resolve().parent
ROOT_DIR = BACKEND_DIR.parent
for p in [str(BACKEND_DIR), str(ROOT_DIR / "src")]:
    if p not in sys.path:
        sys.path.insert(0, p)

import config as C
from model import build_model
from schemas import (
    StateInfo, WeekRecord, HistoryResponse, ForecastResponse,
    CustomForecastRequest, ModelInfoResponse, ModelMetricSet
)

# ─────────────────────────────────────────────────────────────────────────────
# State Configuration & Model Mapping
# ─────────────────────────────────────────────────────────────────────────────

BEST_CHECKPOINTS = {
    "Maharashtra": {
        "ckpt_file": "Maharashtra_CI_W_NC_best.pt",
        "config_name": "CI_W_NC (Channel-Independent, Weather Only, No Calendar)",
        "channel_indep": True,
        "use_weather": True,
        "use_calendar": False,
        "summary": "Best performing model for Maharashtra (PR-AUC 0.847, Recall 90.6%, Precision 71.6%). Relies on strong weather covariates (precipitation and LAI) as leading vector-breeding indicators.",
        "interpretation": "In Maharashtra, historical weather signals provide strong predictive separation 4–6 weeks prior to outbreak spikes. The channel-independent architecture prevents noisy sensor fluctuations from diluting clean case autocorrelation."
    },
    "Karnataka": {
        "ckpt_file": "Karnataka_CI_W_C_best.pt",
        "config_name": "CI_W_C (Channel-Independent, Weather + Calendar)",
        "channel_indep": True,
        "use_weather": True,
        "use_calendar": True,
        "summary": "Best PR-AUC ranking model for Karnataka (PR-AUC 0.202, ROC-AUC 0.509). Test period (2021–2022) had low positive density (20.7%), representing a challenging regime shift.",
        "interpretation": "Karnataka's low test-set positive rate (20.7%) and post-2020 surveillance reporting shifts make early detection harder. The full model with calendar seasonality provides the highest relative probability ranking across the state."
    },
    "Tamil Nadu": {
        "ckpt_file": "Tamil_Nadu_CM_W_C_best.pt",
        "config_name": "CM_W_C (Channel-Mixing, Weather + Calendar)",
        "channel_indep": False,
        "use_weather": True,
        "use_calendar": True,
        "summary": "Best performing model for Tamil Nadu (PR-AUC 0.475, Recall 94.7%, ROC-AUC 0.734). Channel-mixing captures precipitation × temperature interactions key to Tamil Nadu's northeast monsoon.",
        "interpretation": "Tamil Nadu benefits from channel-mixing, where cross-attention between rainfall and temperature captures the complex multivariate dynamics of the winter/northeast monsoon vector lifecycle."
    }
}


class ForecastingEngine:
    """
    Inference & Historical Data Service.
    Loads models, scalers, and continuous grids on startup.
    """

    def __init__(self, models_dir: Path = C.MODELS_DIR, proc_data_dir: Path = C.DATA_PROC_DIR):
        self.models_dir = models_dir
        self.proc_data_dir = proc_data_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.models: Dict[str, torch.nn.Module] = {}
        self.scalers: Dict[str, Dict[str, Any]] = {}
        self.channel_indices: Dict[str, List[int]] = {}
        self.active_channels: Dict[str, List[str]] = {}
        self.grids: Dict[str, pd.DataFrame] = {}
        self.ablation_df: Optional[pd.DataFrame] = None
        self.baseline_df: Optional[pd.DataFrame] = None

    def initialize(self):
        """Preload all model weights, scalers, and historical datasets into memory."""
        print(f"[InferenceEngine] Loading models and datasets on device: {self.device}")

        # Load metrics tables
        ablation_csv = C.RESULTS_DIR / "ablation_table.csv"
        if ablation_csv.exists():
            self.ablation_df = pd.read_csv(ablation_csv)

        baseline_csv = C.RESULTS_DIR / "baseline_results.csv"
        if baseline_csv.exists():
            self.baseline_df = pd.read_csv(baseline_csv)

        for state in C.STATES:
            safe_name = state.replace(" ", "_")
            pkl_path = self.proc_data_dir / f"{safe_name}_data.pkl"
            if not pkl_path.exists():
                raise FileNotFoundError(f"Processed state data missing: {pkl_path}")

            with open(pkl_path, "rb") as f:
                state_data = pickle.load(f)

            self.scalers[state] = state_data["scalers"]
            self.grids[state] = state_data["grid_full"]
            all_channels = state_data["channels"]

            # Load checkpoint
            meta = BEST_CHECKPOINTS[state]
            ckpt_path = self.models_dir / meta["ckpt_file"]
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            n_channels = ckpt["n_channels"]
            c_indices = ckpt["channel_indices"]
            channel_indep = ckpt.get("channel_independent", meta["channel_indep"])

            model = build_model(
                n_channels=n_channels,
                model_type="transformer",
                channel_independent=channel_indep,
            ).to(self.device)

            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()

            self.models[state] = model
            self.channel_indices[state] = c_indices
            self.active_channels[state] = [all_channels[i] for i in c_indices]
            print(f"[InferenceEngine] Loaded {state}: {meta['config_name']} ({n_channels} channels)")

    def get_supported_states(self) -> List[StateInfo]:
        """Return list of supported states with test-set performance metrics."""
        result = []
        for state in C.STATES:
            grid = self.grids[state]
            meta = BEST_CHECKPOINTS[state]

            # Find metrics in ablation_df
            pr_auc, f1, prec, rec = 0.0, 0.0, 0.0, 0.0
            if self.ablation_df is not None:
                ci_str = str(meta["channel_indep"])
                w_str = str(meta["use_weather"])
                cal_str = str(meta["use_calendar"])
                match = self.ablation_df[
                    (self.ablation_df["state"] == state) &
                    (self.ablation_df["level"] == "window") &
                    (self.ablation_df["CI"].astype(str) == ci_str) &
                    (self.ablation_df["weather"].astype(str) == w_str) &
                    (self.ablation_df["calendar"].astype(str) == cal_str)
                ]
                if not match.empty:
                    row = match.iloc[0]
                    pr_auc = float(row["pr_auc"])
                    f1 = float(row["f1"])
                    prec = float(row["precision"])
                    rec = float(row["recall"])

            result.append(
                StateInfo(
                    id=state,
                    name=state,
                    best_config=meta["config_name"].split(" ")[0],
                    window_pr_auc=round(pr_auc, 3),
                    window_f1=round(f1, 3),
                    window_recall=round(rec, 3),
                    window_precision=round(prec, 3),
                    historical_year_range=f"{int(grid['year'].min())}–{int(grid['year'].max())}",
                    total_recorded_weeks=len(grid),
                )
            )
        return result

    def get_history(self, state: str, weeks: int = 24) -> HistoryResponse:
        """Return the last N weeks of historical data for the state."""
        if state not in self.grids:
            raise ValueError(f"State '{state}' not supported. Choose from {C.STATES}.")

        grid = self.grids[state].sort_values(["year", "week"]).reset_index(drop=True)
        recent_df = grid.tail(weeks).copy()

        records = []
        for _, row in recent_df.iterrows():
            y, w = int(row["year"]), int(row["week"])
            records.append(
                WeekRecord(
                    year=y,
                    week=w,
                    date_label=f"{y}-W{w:02d}",
                    Cases=float(row.get("Cases", 0.0)),
                    preci=float(row.get("preci", 0.0)),
                    LAI=float(row.get("LAI", 0.0)),
                    Temp=float(row.get("Temp", 0.0)),
                    outbreak=int(row.get("outbreak", 0)),
                    is_covered_week=bool(row.get("is_covered_week", False)),
                )
            )

        start_label = records[0].date_label if records else ""
        end_label = records[-1].date_label if records else ""
        date_span = f"{start_label} to {end_label}"

        return HistoryResponse(
            state=state,
            requested_weeks=weeks,
            total_records=len(records),
            data=records,
            data_snapshot_date=date_span,
        )

    def _determine_risk(self, prob: float) -> Tuple[str, str, str]:
        """Convert window outbreak probability to risk label, theme color, and guidance."""
        if prob < 0.35:
            return (
                "Low",
                "emerald",
                "Baseline risk: Probability of vector-borne outbreak in t+4..t+6 is low (<35%). Maintain routine vector source reduction and standard sentinel surveillance."
            )
        elif prob < 0.65:
            return (
                "Moderate",
                "amber",
                "Moderate risk (35%–64%): Elevated vector breeding signals detected. Review municipal larvicide reserves and alert district surveillance officers for targeted inspections."
            )
        else:
            return (
                "High",
                "rose",
                "High risk (>=65%): Significant probability of outbreak in t+4..t+6. Recommend immediate preemptive vector abatement, community awareness alerts, and fever clinic readiness."
            )

    def run_inference_on_rows(
        self, state: str, rows_12: List[Dict[str, Any]], is_simulated: bool = False
    ) -> ForecastResponse:
        """
        Takes exactly 12 weekly measurements, constructs required features,
        normalizes using training scalers, and runs inference.
        """
        if state not in self.models:
            raise ValueError(f"State '{state}' not recognized.")
        if len(rows_12) != 12:
            raise ValueError(f"12-week window required, received {len(rows_12)} weeks.")

        scalers = self.scalers[state]
        meta = BEST_CHECKPOINTS[state]
        model = self.models[state]
        active_channels = self.active_channels[state]

        # Assemble feature dict for 12 weeks
        feat_rows = []
        input_records = []
        for idx, r in enumerate(rows_12):
            cases = float(r.get("Cases", 0.0))
            preci = float(r.get("preci", 0.0))
            lai = float(r.get("LAI", 0.0))
            temp = float(r.get("Temp", 0.0))
            year = int(r.get("year", 2022))
            w = int(r.get("week", (idx + 41) % 52 or 52))

            # Calendar features
            w_sin = math.sin(2 * math.pi * w / 52)
            w_cos = math.cos(2 * math.pi * w / 52)
            month = min(max(math.ceil(w / (52 / 12)), 1), 12)
            is_monsoon = 1.0 if month in C.MONSOON_MONTHS else 0.0
            cases_feat = math.log1p(cases)

            row_feats = {
                "cases_feat": cases_feat,
                "preci": preci,
                "LAI": lai,
                "Temp": temp,
                "week_sin": w_sin,
                "week_cos": w_cos,
                "month": float(month),
                "is_monsoon": is_monsoon,
            }
            feat_rows.append(row_feats)

            input_records.append(
                WeekRecord(
                    year=year,
                    week=w,
                    date_label=f"{year}-W{w:02d}",
                    Cases=cases,
                    preci=preci,
                    LAI=lai,
                    Temp=temp,
                    outbreak=int(cases > 0),
                    is_covered_week=bool(r.get("is_covered_week", True)),
                )
            )

        # Build feature matrix (12, n_active_channels) normalized
        matrix = np.zeros((12, len(active_channels)), dtype=np.float32)
        for t, rf in enumerate(feat_rows):
            for c_idx, ch_name in enumerate(active_channels):
                raw_v = rf[ch_name]
                scaler = scalers[ch_name]
                scaled_v = scaler.transform([[raw_v]])[0, 0]
                matrix[t, c_idx] = scaled_v

        # Replace any residual NaN with 0
        matrix = np.nan_to_num(matrix, nan=0.0)

        # Tensor shape (1, 12, C)
        tensor_x = torch.from_numpy(matrix).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = model(tensor_x)
            prob_window = float(torch.sigmoid(out["logit_window"]).cpu().item())
            lead_probs = torch.sigmoid(out["logit_leads"]).cpu().squeeze(0).tolist()

        per_lead = {
            "t+4": round(float(lead_probs[0]), 4),
            "t+5": round(float(lead_probs[1]), 4),
            "t+6": round(float(lead_probs[2]), 4),
        }
        prob_window = round(prob_window, 4)

        risk_label, risk_color, risk_desc = self._determine_risk(prob_window)

        # Target range calculation
        first_w = input_records[0]
        last_w = input_records[-1]
        input_span = f"{first_w.date_label} to {last_w.date_label}"

        # Lead forecast horizon (weeks t+4 to t+6)
        end_week_num = last_w.week
        end_year = last_w.year
        t4_w = (end_week_num + 4) % 52 or 52
        t4_y = end_year + ((end_week_num + 4 - 1) // 52)
        t6_w = (end_week_num + 6) % 52 or 52
        t6_y = end_year + ((end_week_num + 6 - 1) // 52)
        target_span = f"Weeks t+4..t+6 ({t4_y}-W{t4_w:02d} to {t6_y}-W{t6_w:02d})"

        snapshot_note = (
            "Interactive what-if simulation run."
            if is_simulated else
            f"Forecast computed from the most recent historical IDSP snapshot ({last_w.date_label}). No live streaming feed."
        )

        return ForecastResponse(
            state=state,
            window_level_probability=prob_window,
            per_lead_week=per_lead,
            risk_label=risk_label,
            risk_color=risk_color,
            risk_description=risk_desc,
            input_window_summary=input_records,
            input_date_range=input_span,
            forecast_lead_range=target_span,
            model_used=f"PatchTST ({meta['config_name'].split(' ')[0]})",
            data_snapshot_note=snapshot_note,
            is_simulated=is_simulated,
        )

    def forecast_latest(self, state: str) -> ForecastResponse:
        """Run forecast on the most recent 12-week window in historical data."""
        if state not in self.grids:
            raise ValueError(f"State '{state}' not supported.")

        grid = self.grids[state].sort_values(["year", "week"]).reset_index(drop=True)
        recent_12 = grid.tail(12).to_dict(orient="records")
        return self.run_inference_on_rows(state, recent_12, is_simulated=False)

    def forecast_custom(self, req: CustomForecastRequest) -> ForecastResponse:
        """Run forecast on custom 12-week input for what-if scenario testing."""
        rows = [w.model_dump() for w in req.weeks]
        return self.run_inference_on_rows(req.state, rows, is_simulated=True)

    def get_model_info(self, state: str) -> ModelInfoResponse:
        """Return test set metrics, baseline comparisons, and model architecture details."""
        if state not in self.grids:
            raise ValueError(f"State '{state}' not supported.")

        meta = BEST_CHECKPOINTS[state]
        active_ch = self.active_channels[state]

        # Extract window and lead metrics from ablation_df
        def _get_metrics(level_str: str) -> ModelMetricSet:
            if self.ablation_df is not None:
                ci_str = str(meta["channel_indep"])
                w_str = str(meta["use_weather"])
                cal_str = str(meta["use_calendar"])
                sub = self.ablation_df[
                    (self.ablation_df["state"] == state) &
                    (self.ablation_df["level"] == level_str) &
                    (self.ablation_df["CI"].astype(str) == ci_str) &
                    (self.ablation_df["weather"].astype(str) == w_str) &
                    (self.ablation_df["calendar"].astype(str) == cal_str)
                ]
                if not sub.empty:
                    r = sub.iloc[0]
                    return ModelMetricSet(
                        precision=round(float(r["precision"]), 3),
                        recall=round(float(r["recall"]), 3),
                        f1=round(float(r["f1"]), 3),
                        roc_auc=round(float(r["roc_auc"]), 3),
                        pr_auc=round(float(r["pr_auc"]), 3),
                    )
            return ModelMetricSet(precision=0.0, recall=0.0, f1=0.0, roc_auc=0.0, pr_auc=0.0)

        window_metrics = _get_metrics("window")
        leads_metrics = {
            "t+4": _get_metrics("t+4"),
            "t+5": _get_metrics("t+5"),
            "t+6": _get_metrics("t+6"),
        }

        # Baseline comparison
        baselines = []
        if self.ablation_df is not None:
            state_ablations = self.ablation_df[
                (self.ablation_df["state"] == state) &
                (self.ablation_df["level"] == "window")
            ]
            for _, r in state_ablations.iterrows():
                m_name = str(r["model"])
                baselines.append({
                    "model": m_name,
                    "precision": round(float(r["precision"]), 3),
                    "recall": round(float(r["recall"]), 3),
                    "f1": round(float(r["f1"]), 3),
                    "roc_auc": round(float(r["roc_auc"]), 3),
                    "pr_auc": round(float(r["pr_auc"]), 3),
                    "is_selected_best": (m_name.find(meta["config_name"].split(" ")[0]) != -1)
                })

        plain_summary = (
            f"On held-out test data (2021–2022), this model achieved {window_metrics.recall * 100:.1f}% recall "
            f"(correctly detecting outbreak windows) with {window_metrics.precision * 100:.1f}% precision "
            f"and PR-AUC of {window_metrics.pr_auc:.3f}."
        )

        return ModelInfoResponse(
            state=state,
            best_config_name=meta["config_name"],
            architecture_type="PatchTST Time-Series Transformer (Encoder-only, 3 layers, 8 heads, d_model=128, 4-week patches)",
            channel_strategy="Channel-Independent" if meta["channel_indep"] else "Channel-Mixing",
            weather_features=meta["use_weather"],
            calendar_features=meta["use_calendar"],
            active_channels=active_ch,
            plain_english_summary=plain_summary,
            test_metrics_window=window_metrics,
            test_metrics_leads=leads_metrics,
            baseline_comparison=baselines,
            interpretation=meta["interpretation"],
            limitations_and_disclaimer=(
                "Model trained on historical IDSP reports (2009–2022). IDSP surveillance data is subject "
                "to known underreporting and reporting delays. Output represents probability of confirmed "
                "reports, not absolute epidemiological ground truth."
            ),
        )


# Singleton instance
engine = ForecastingEngine()
