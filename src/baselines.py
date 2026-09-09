"""
baselines.py — Persistence/seasonal-naive and LSTM baselines (§2).

Implements:
  1. Persistence baseline: predicts "outbreak in t+4..t+6" = same as whether
     an outbreak occurred in the equivalent weeks of the prior year.
     Fallback: if no prior-year data, use most-recent observed state.

  2. LSTM classifier baseline: same windowed input as the Transformer, trained
     with the same pipeline and metrics (§4). This gives a learned baseline that
     shares the input format, making model comparison fair.

Why LSTM over ARIMA:
  ARIMA operates on a univariate case-count series and requires thresholding
  post-hoc to produce a binary signal — introducing an additional hyperparameter.
  An LSTM classifier directly outputs a binary probability from the same
  multivariate windowed input as the Transformer, making the comparison cleaner
  (same input format, same loss, same metrics). It is also a more realistic
  competitor for the §4 ablation table.

Usage:
    python src/baselines.py --state Maharashtra
    python src/baselines.py --all-states
"""

import sys
import logging
import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from model import LSTMClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = C.SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute Precision, Recall, F1, ROC-AUC, PR-AUC for binary classification."""
    y_pred = (y_pred_prob >= threshold).astype(int)
    metrics = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_pred_prob)
    except ValueError:
        metrics["roc_auc"] = float("nan")
    try:
        metrics["pr_auc"] = average_precision_score(y_true, y_pred_prob)
    except ValueError:
        metrics["pr_auc"] = float("nan")
    return metrics


def load_state_data(state: str) -> dict:
    safe_name = state.replace(" ", "_")
    pkl_path = C.DATA_PROC_DIR / f"{safe_name}_data.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"Processed data not found: {pkl_path}\n"
            "Run `python src/data_pipeline.py` first."
        )
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Persistence / Seasonal-naive baseline
# ─────────────────────────────────────────────────────────────────────────────

def run_persistence_baseline(state_data: dict, state: str) -> dict:
    """
    Persistence (prior-year) baseline:
      For each test window ending at week t, predict that an outbreak occurs
      in the t+4..t+6 window if an outbreak occurred in those same ISO weeks
      in the prior year.

    If no prior-year window is available, fall back to the most recent observed
    outbreak state (simplest persistence).

    NOTE: This baseline operates directly on the y_leads arrays — it does NOT
    use the feature matrix X, because it is rule-based.

    Implementation:
      - Use the full grid (grid_full) to build a (year, week) → outbreak lookup.
      - For each test window, identify the future weeks (t+4..t+6) and query
        whether those same weeks had an outbreak one year prior.
    """
    grid = state_data["grid_full"]
    grid = grid.set_index(["year", "week"])["outbreak"].to_dict()

    # Pull test y_leads to get the same window indices as training
    y_leads_test = state_data["test"]["y_leads"]      # (N_test, 3)
    y_window_test = state_data["test"]["y_window"]    # (N_test,)

    # We need to recover the actual (year, week) for each test window's future weeks.
    # The grid was built as: each row = one (state, year, week) tuple, sorted by year/week.
    # Test split starts at TEST_START_YEAR. The i-th test window ends at row:
    #   grid_test_start + INPUT_WINDOW - 1 + i
    # and its forecast horizon is t+LEAD_MIN .. t+LEAD_MAX weeks ahead.
    #
    # Reconstruct week indices from grid_full for the test period.
    grid_full = state_data["grid_full"]
    test_grid = grid_full[grid_full["year"] >= C.TEST_START_YEAR].reset_index(drop=True)
    N_test = len(y_window_test)

    prior_year_probs = []
    for i in range(N_test):
        # Window end is at row (INPUT_WINDOW - 1 + i) in test_grid
        # Future weeks are at rows (INPUT_WINDOW + LEAD_MIN - 1 + i) ..
        #                          (INPUT_WINDOW + LEAD_MAX - 1 + i)
        lead_labels_prior = []
        for lead in range(C.LEAD_MIN, C.LEAD_MAX + 1):
            future_idx = C.INPUT_WINDOW + lead - 1 + i
            if future_idx >= len(test_grid):
                lead_labels_prior.append(0)
                continue
            future_year = int(test_grid.iloc[future_idx]["year"])
            future_week = int(test_grid.iloc[future_idx]["week"])
            # Prior-year same week
            prior_key = (future_year - 1, future_week)
            prior_outbreak = grid.get(prior_key, 0)  # fallback: 0 if not found
            lead_labels_prior.append(float(prior_outbreak))

        # Window label: 1 if ANY prior-year lead week had an outbreak
        win_prob = float(max(lead_labels_prior))
        prior_year_probs.append(win_prob)

    prior_year_probs = np.array(prior_year_probs)

    window_metrics = compute_metrics(y_window_test, prior_year_probs)

    log.info("Persistence baseline — %s — window metrics: %s", state, window_metrics)

    # Per-lead-week metrics (use the same prior-year lead value per lead position)
    lead_metrics_list = []
    lead_names = [f"t+{l}" for l in range(C.LEAD_MIN, C.LEAD_MAX + 1)]
    for lead_i, lead_name in enumerate(lead_names):
        # Reconstruct per-lead prior-year probabilities
        lead_probs = []
        for i in range(N_test):
            lead = C.LEAD_MIN + lead_i
            future_idx = C.INPUT_WINDOW + lead - 1 + i
            if future_idx >= len(test_grid):
                lead_probs.append(0.0)
                continue
            future_year = int(test_grid.iloc[future_idx]["year"])
            future_week = int(test_grid.iloc[future_idx]["week"])
            prior_key = (future_year - 1, future_week)
            lead_probs.append(float(grid.get(prior_key, 0)))
        lead_probs = np.array(lead_probs)
        lead_true = y_leads_test[:, lead_i]
        m = compute_metrics(lead_true, lead_probs)
        m["lead"] = lead_name
        lead_metrics_list.append(m)
        log.info("  Persistence %s %s: F1=%.3f, PR-AUC=%.3f", state, lead_name, m["f1"], m["pr_auc"])

    return {
        "model": "persistence",
        "state": state,
        "window": window_metrics,
        "leads": lead_metrics_list,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. LSTM Baseline — training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_lstm_baseline(state_data: dict, state: str, device: torch.device) -> dict:
    """
    Train LSTM classifier as a baseline model (§2).

    Uses the same sliding-window tensors as the Transformer, making the comparison
    architecturally fair (identical input format, same loss function, same metrics).

    Returns a dict with test metrics (same schema as other baselines).
    """
    set_seed(C.SEED)

    n_channels = len(state_data["channels"])
    model = LSTMClassifier(n_channels=n_channels).to(device)

    # Compute positive class weight (handles class imbalance).
    # Cap at 10.0 to prevent extreme values from causing NaN loss at init.
    y_tr = state_data["train"]["y_window"]
    pos_weight_val = min(
        C.POS_WEIGHT_SCALE * (1 - y_tr.mean()) / (y_tr.mean() + 1e-6),
        10.0,
    )
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32, device=device)
    log.info("LSTM pos_weight: %.3f", pos_weight_val)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Use a lower LR for LSTM — it's more sensitive to initial step size than the Transformer
    lstm_lr = min(C.LR, 1e-3)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lstm_lr, weight_decay=C.WEIGHT_DECAY
    )

    def _make_loader(split: str, shuffle: bool) -> DataLoader:
        X_np = state_data[split]["X"]
        # Guard: replace any NaN (e.g. leading weather rows) with 0 before training
        X_np = np.nan_to_num(X_np, nan=0.0)
        X = torch.from_numpy(X_np)
        y = torch.from_numpy(state_data[split]["y_window"]).unsqueeze(1)
        return DataLoader(TensorDataset(X, y), batch_size=C.BATCH_SIZE, shuffle=shuffle)

    train_loader = _make_loader("train", shuffle=True)
    val_loader   = _make_loader("val",   shuffle=False)

    best_pr_auc = -1.0
    patience_counter = 0
    best_state_dict = None

    log.info("Training LSTM baseline for state: %s", state)
    for epoch in range(1, C.LSTM_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            out = model(X_b)
            loss = criterion(out["logit_window"], y_b)
            if torch.isnan(loss):
                log.warning("NaN loss at epoch %d — skipping batch.", epoch)
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(X_b)
        epoch_loss /= len(state_data["train"]["X"])

        # Validation
        model.eval()
        val_probs, val_true = [], []
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b = X_b.to(device)
                out = model(X_b)
                prob = torch.sigmoid(out["logit_window"]).cpu().numpy().flatten()
                val_probs.extend(prob)
                val_true.extend(y_b.numpy().flatten())
        val_probs = np.array(val_probs)
        val_true  = np.array(val_true)
        try:
            val_pr_auc = average_precision_score(val_true, val_probs)
        except ValueError:
            val_pr_auc = 0.0

        if val_pr_auc > best_pr_auc:
            best_pr_auc = val_pr_auc
            patience_counter = 0
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if epoch % 10 == 0 or patience_counter >= C.LSTM_PATIENCE:
            log.info(
                "  LSTM [%s] Epoch %d/%d — loss=%.4f, val_PR-AUC=%.3f (best=%.3f, patience=%d)",
                state, epoch, C.LSTM_EPOCHS, epoch_loss, val_pr_auc, best_pr_auc, patience_counter,
            )

        if patience_counter >= C.LSTM_PATIENCE:
            log.info("  Early stopping at epoch %d.", epoch)
            break

    # Load best weights and evaluate on test
    if best_state_dict is None:
        log.warning(
            "No improving epoch found for LSTM [%s] — using final weights. "
            "Check for data anomalies or extreme class imbalance.",
            state,
        )
        best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state_dict)
    model.eval()

    X_te = torch.from_numpy(np.nan_to_num(state_data["test"]["X"], nan=0.0)).to(device)
    y_te = state_data["test"]["y_window"]
    y_leads_te = state_data["test"]["y_leads"]

    with torch.no_grad():
        out = model(X_te)
        probs_win = torch.sigmoid(out["logit_window"]).cpu().numpy().flatten()
        if "logit_leads" in out:
            probs_leads = torch.sigmoid(out["logit_leads"]).cpu().numpy()  # (N, n_leads)
        else:
            probs_leads = None

    window_metrics = compute_metrics(y_te, probs_win)
    log.info("LSTM baseline — %s — window metrics: %s", state, window_metrics)

    lead_metrics_list = []
    lead_names = [f"t+{l}" for l in range(C.LEAD_MIN, C.LEAD_MAX + 1)]
    for lead_i, lead_name in enumerate(lead_names):
        if probs_leads is not None:
            lp = probs_leads[:, lead_i]
        else:
            lp = probs_win  # fallback: use window prob for all leads
        lt = y_leads_te[:, lead_i]
        m = compute_metrics(lt, lp)
        m["lead"] = lead_name
        lead_metrics_list.append(m)
        log.info("  LSTM %s %s: F1=%.3f, PR-AUC=%.3f", state, lead_name, m["f1"], m["pr_auc"])

    # Save LSTM checkpoint
    safe_name = state.replace(" ", "_")
    ckpt_path = C.MODELS_DIR / f"{safe_name}_lstm.pt"
    torch.save({"model_state_dict": best_state_dict, "n_channels": n_channels}, ckpt_path)
    log.info("LSTM checkpoint saved: %s", ckpt_path)

    return {
        "model": "lstm",
        "state": state,
        "window": window_metrics,
        "leads": lead_metrics_list,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Results saving
# ─────────────────────────────────────────────────────────────────────────────

def save_baseline_results(results: list[dict], out_dir: Path = C.RESULTS_DIR):
    """Save baseline results as a markdown table and CSV."""
    import csv

    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for r in results:
        base = {"model": r["model"], "state": r["state"], "level": "window"}
        base.update(r["window"])
        rows.append(base)
        for lead_m in r["leads"]:
            row = {"model": r["model"], "state": r["state"], "level": lead_m["lead"]}
            row.update({k: v for k, v in lead_m.items() if k != "lead"})
            rows.append(row)

    # CSV
    csv_path = out_dir / "baseline_results.csv"
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    log.info("Baseline results CSV saved: %s", csv_path)

    # Markdown table
    md_path = out_dir / "baseline_results.md"
    with open(md_path, "w") as f:
        f.write("# Baseline Results\n\n")
        headers = list(rows[0].keys()) if rows else []
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            f.write(
                "| " + " | ".join(
                    f"{v:.4f}" if isinstance(v, float) else str(v)
                    for v in row.values()
                ) + " |\n"
            )
    log.info("Baseline results markdown saved: %s", md_path)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run baselines for outbreak forecasting.")
    parser.add_argument("--state", type=str, default=None, help="Single state to run.")
    parser.add_argument("--all-states", action="store_true", help="Run all STATES.")
    parser.add_argument("--no-lstm", action="store_true", help="Skip LSTM baseline (faster).")
    args = parser.parse_args()

    states = C.STATES if args.all_states or args.state is None else [args.state]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    if torch.cuda.is_available():
        log.info("GPU: %s (%.1f GB VRAM)", torch.cuda.get_device_name(0),
                 torch.cuda.get_device_properties(0).total_memory / 1e9)

    all_results = []
    for state in states:
        log.info("=" * 60)
        log.info("Running baselines for: %s", state)
        try:
            state_data = load_state_data(state)
        except FileNotFoundError as e:
            log.error(str(e))
            continue

        # Persistence baseline
        res_pers = run_persistence_baseline(state_data, state)
        all_results.append(res_pers)

        # LSTM baseline
        if not args.no_lstm:
            res_lstm = train_lstm_baseline(state_data, state, device)
            all_results.append(res_lstm)

    save_baseline_results(all_results)
    log.info("Baselines complete.")


if __name__ == "__main__":
    main()
