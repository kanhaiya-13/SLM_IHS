"""
evaluate.py — Test-set evaluation and full ablation study (§4).

Implements:
  - Loads saved checkpoints and evaluates on the held-out test split
  - Computes Precision, Recall, F1, ROC-AUC, PR-AUC (§4)
  - Reports metrics separately for:
      • Window-level target (does at least one outbreak occur in t+4..t+6?)
      • Per-lead-week: t+4, t+5, t+6 individually
  - Runs the full ablation matrix (§4):
      1. Channel-independent vs. channel-mixing
      2. With vs. without calendar/seasonal encoding
      3. With vs. without weather covariates
      4. (Combined-disease vs. single-disease: integrated at pipeline level)
  - Loads baseline results from baselines.py for the comparison table
  - Produces:
      • results/ablation_table.csv  / .md
      • results/plots/loss_curves_{state}.png
      • results/plots/lead_week_degradation.png

Usage:
    python src/evaluate.py --state Maharashtra
    python src/evaluate.py --all-states
    python src/evaluate.py --run-ablations --all-states   # trains + evaluates all ablations
"""

import sys
import json
import logging
import argparse
import pickle
import csv
from pathlib import Path
from itertools import product

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server/headless use
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from model import build_model
from train import train_state, load_state_data, get_channel_indices, make_loaders

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PLOTS_DIR = C.RESULTS_DIR / "plots"


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Precision, Recall, F1, ROC-AUC, PR-AUC for binary classification."""
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


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation of a single checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_checkpoint(
    state: str,
    state_data: dict,
    ckpt_path: Path,
    device: torch.device,
) -> dict:
    """
    Load a saved checkpoint and run inference on the test split.

    Returns a dict with:
      - window-level metrics
      - per-lead-week metrics (t+4, t+5, t+6)
      - config metadata from the checkpoint
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    n_channels       = ckpt["n_channels"]
    channel_indices  = ckpt["channel_indices"]
    channel_indep    = ckpt.get("channel_independent", True)

    model = build_model(
        n_channels=n_channels,
        model_type="transformer",
        channel_independent=channel_indep,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Build test tensors with the same channel subset used during training
    X_full = state_data["test"]["X"]       # (N, T, C_all)
    X = X_full[:, :, channel_indices]      # (N, T, C_active)
    X = np.nan_to_num(X, nan=0.0)
    X_t = torch.from_numpy(X).to(device)
    y_te = state_data["test"]["y_window"]
    y_leads_te = state_data["test"]["y_leads"]

    with torch.no_grad():
        out = model(X_t)
        probs_win = torch.sigmoid(out["logit_window"]).cpu().numpy().flatten()
        if "logit_leads" in out:
            probs_leads = torch.sigmoid(out["logit_leads"]).cpu().numpy()
        else:
            probs_leads = None

    window_metrics = compute_metrics(y_te, probs_win)

    lead_names = [f"t+{l}" for l in range(C.LEAD_MIN, C.LEAD_MAX + 1)]
    lead_metrics_list = []
    for lead_i, lead_name in enumerate(lead_names):
        lp = probs_leads[:, lead_i] if probs_leads is not None else probs_win
        lt = y_leads_te[:, lead_i]
        m = compute_metrics(lt, lp)
        m["lead"] = lead_name
        lead_metrics_list.append(m)

    return {
        "state": state,
        "ckpt": str(ckpt_path),
        "channel_independent": channel_indep,
        "use_weather": ckpt.get("use_weather", True),
        "use_calendar": ckpt.get("use_calendar", True),
        "window": window_metrics,
        "leads": lead_metrics_list,
        "probs_win": probs_win,
        "y_te": y_te,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Ablation matrix runner
# ─────────────────────────────────────────────────────────────────────────────

ABLATION_CONFIGS = [
    # (channel_independent, use_weather, use_calendar, tag)
    (True,  True,  True,  "CI_W_C"),    # full model (default)
    (False, True,  True,  "CM_W_C"),    # channel-mixing (ablation 1)
    (True,  True,  False, "CI_W_NC"),   # no calendar (ablation 2)
    (True,  False, True,  "CI_NW_C"),   # no weather (ablation 3)
    (True,  False, False, "CI_NW_NC"),  # cases only (ablation 2+3 combined)
]


def run_ablations(
    states: list[str],
    device: torch.device,
) -> list[dict]:
    """
    Train and evaluate all ablation configurations for all states.

    Returns a flat list of result dicts (one per state × ablation config).
    """
    all_results = []
    for state in states:
        log.info("=" * 70)
        log.info("Running ablations for: %s", state)
        try:
            state_data = load_state_data(state)
        except FileNotFoundError as e:
            log.error(str(e))
            continue

        for ch_indep, use_w, use_c, tag in ABLATION_CONFIGS:
            safe_name = state.replace(" ", "_")
            ckpt_path = C.MODELS_DIR / f"{safe_name}_{tag}_best.pt"

            log.info("  Ablation: %s | CI=%s, W=%s, Cal=%s", tag, ch_indep, use_w, use_c)

            # Train if checkpoint doesn't exist
            if not ckpt_path.exists():
                train_state(
                    state=state,
                    state_data=state_data,
                    device=device,
                    channel_independent=ch_indep,
                    use_weather=use_w,
                    use_calendar=use_c,
                    run_tag=tag,
                )

            # Evaluate
            result = evaluate_checkpoint(state, state_data, ckpt_path, device)
            result["ablation_tag"] = tag
            result["model"] = f"Transformer ({tag})"
            all_results.append(result)

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Results tables
# ─────────────────────────────────────────────────────────────────────────────

def build_results_table(transformer_results: list[dict], baseline_csv: Path | None) -> list[dict]:
    """
    Merge Transformer ablation results with baseline CSV into one flat table.

    Each row has: model, state, level, precision, recall, f1, roc_auc, pr_auc
    """
    rows = []

    # Transformer results
    for r in transformer_results:
        tag = r.get("ablation_tag", "default")
        base = {
            "model": r.get("model", f"Transformer ({tag})"),
            "state": r["state"],
            "CI": r.get("channel_independent", "?"),
            "weather": r.get("use_weather", "?"),
            "calendar": r.get("use_calendar", "?"),
            "level": "window",
        }
        base.update(r["window"])
        rows.append(base)

        for lm in r["leads"]:
            row = {**base, "level": lm["lead"]}
            row.update({k: v for k, v in lm.items() if k != "lead"})
            rows.append(row)

    # Baseline results
    if baseline_csv and baseline_csv.exists():
        with open(baseline_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row.setdefault("CI", "-")
                row.setdefault("weather", "-")
                row.setdefault("calendar", "-")
                rows.append(row)

    return rows


def save_table(rows: list[dict], out_dir: Path, filename_stem: str):
    """Save results table as CSV and Markdown."""
    out_dir.mkdir(parents=True, exist_ok=True)

    if not rows:
        log.warning("No results rows to save.")
        return

    # Convert float strings for cleaner display
    def _fmt(v):
        try:
            return f"{float(v):.4f}"
        except (ValueError, TypeError):
            return str(v)

    fieldnames = list(rows[0].keys())

    csv_path = out_dir / f"{filename_stem}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log.info("Results CSV: %s", csv_path)

    md_path = out_dir / f"{filename_stem}.md"
    with open(md_path, "w") as f:
        f.write(f"# {filename_stem.replace('_', ' ').title()}\n\n")
        f.write("| " + " | ".join(fieldnames) + " |\n")
        f.write("| " + " | ".join(["---"] * len(fieldnames)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(_fmt(row.get(k, "")) for k in fieldnames) + " |\n")
    log.info("Results MD:  %s", md_path)


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_loss_curves(state: str, tag: str = ""):
    """Plot train loss and val PR-AUC curves for one state."""
    safe = state.replace(" ", "_")
    tag_suffix = f"_{tag}" if tag else ""
    curves_path = C.RESULTS_DIR / "logs" / f"{safe}{tag_suffix}_curves.json"
    if not curves_path.exists():
        log.warning("Loss curve file not found: %s", curves_path)
        return

    with open(curves_path) as f:
        data = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{state} — Training Curves ({tag})", fontsize=13)

    ax1.plot(data["train_loss"], color="#2196F3", lw=1.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("BCE Loss")
    ax1.set_title("Train Loss")
    ax1.grid(True, alpha=0.3)

    ax2.plot(data["val_pr_auc"], color="#4CAF50", lw=1.5)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("PR-AUC")
    ax2.set_title("Val PR-AUC")
    ax2.grid(True, alpha=0.3)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOTS_DIR / f"{safe}{tag_suffix}_loss_curves.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Loss curve plot: %s", out_path)


def plot_lead_week_degradation(all_results: list[dict]):
    """
    Plot per-lead-week F1 and PR-AUC for each model+state combination.

    This shows how performance degrades with forecast horizon (t+4, t+5, t+6).
    """
    lead_labels = [f"t+{l}" for l in range(C.LEAD_MIN, C.LEAD_MAX + 1)]

    # Filter to Transformer full-model (CI_W_C) for clarity
    featured = [r for r in all_results if r.get("ablation_tag") == "CI_W_C"]
    if not featured:
        featured = all_results[:3]  # fallback: use first 3

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Forecast Horizon Degradation (t+4, t+5, t+6)", fontsize=13)

    colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63"]
    markers = ["o", "s", "^", "D"]

    for ax_i, metric_key in enumerate(["f1", "pr_auc"]):
        ax = axes[ax_i]
        for r_i, r in enumerate(featured):
            lead_vals = [lm[metric_key] for lm in r["leads"]]
            label = f"{r['state']} ({r.get('ablation_tag','?')})"
            ax.plot(
                lead_labels, lead_vals,
                color=colors[r_i % len(colors)],
                marker=markers[r_i % len(markers)],
                label=label, lw=1.8, ms=8,
            )
        ax.set_xlabel("Forecast Lead Week")
        ax.set_ylabel(metric_key.upper().replace("_", "-"))
        ax.set_title(metric_key.upper().replace("_", "-") + " by Lead Week")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOTS_DIR / "lead_week_degradation.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Lead-week degradation plot: %s", out_path)


def plot_ablation_heatmap(rows: list[dict]):
    """Heatmap of window-level PR-AUC across ablations × states."""
    import matplotlib.colors as mcolors

    # Filter to window-level Transformer results
    tfm_rows = [
        r for r in rows
        if r.get("level") == "window"
        and "Transformer" in str(r.get("model", ""))
    ]
    if not tfm_rows:
        return

    states_seen = sorted(set(r["state"] for r in tfm_rows))
    tags_seen = sorted(set(r.get("model", "").replace("Transformer (", "").replace(")", "")
                           for r in tfm_rows))

    matrix = np.zeros((len(tags_seen), len(states_seen)))
    for r in tfm_rows:
        tag = r.get("model", "").replace("Transformer (", "").replace(")", "")
        try:
            pr = float(r.get("pr_auc", 0))
        except (ValueError, TypeError):
            pr = 0.0
        if tag in tags_seen and r["state"] in states_seen:
            i = tags_seen.index(tag)
            j = states_seen.index(r["state"])
            matrix[i, j] = pr

    fig, ax = plt.subplots(figsize=(len(states_seen) * 2 + 2, len(tags_seen) * 0.8 + 2))
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(states_seen)))
    ax.set_xticklabels(states_seen, rotation=15, ha="right")
    ax.set_yticks(range(len(tags_seen)))
    ax.set_yticklabels(tags_seen)
    ax.set_title("Window-Level PR-AUC: Ablation × State", fontsize=12)
    plt.colorbar(im, ax=ax, label="PR-AUC")

    for i in range(len(tags_seen)):
        for j in range(len(states_seen)):
            ax.text(j, i, f"{matrix[i,j]:.3f}", ha="center", va="center", fontsize=9,
                    color="black" if matrix[i, j] < 0.7 else "white")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOTS_DIR / "ablation_heatmap.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Ablation heatmap: %s", out_path)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate outbreak forecasting models.")
    parser.add_argument("--state", type=str, default=None)
    parser.add_argument("--all-states", action="store_true")
    parser.add_argument("--run-ablations", action="store_true",
                        help="Train and evaluate all ablation configs (§4).")
    parser.add_argument("--tag", type=str, default="CI_W_C",
                        help="Checkpoint tag to load for single-run evaluation.")
    args = parser.parse_args()

    states = C.STATES if (args.all_states or args.state is None) else [args.state]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    all_transformer_results = []

    if args.run_ablations:
        # Full ablation matrix: train all configs if needed, then evaluate
        all_transformer_results = run_ablations(states, device)
    else:
        # Evaluate single tag checkpoint per state
        for state in states:
            safe = state.replace(" ", "_")
            ckpt_path = C.MODELS_DIR / f"{safe}_{args.tag}_best.pt"
            if not ckpt_path.exists():
                log.warning("Checkpoint not found: %s — run train.py first.", ckpt_path)
                continue
            try:
                state_data = load_state_data(state)
            except FileNotFoundError as e:
                log.error(str(e))
                continue
            result = evaluate_checkpoint(state, state_data, ckpt_path, device)
            result["ablation_tag"] = args.tag
            result["model"] = f"Transformer ({args.tag})"
            all_transformer_results.append(result)

    # Plot loss curves for available runs
    for state in states:
        for _, _, _, tag in ABLATION_CONFIGS:
            plot_loss_curves(state, tag)

    # Lead-week degradation plot
    if all_transformer_results:
        plot_lead_week_degradation(all_transformer_results)

    # Merge with baselines
    baseline_csv = C.RESULTS_DIR / "baseline_results.csv"
    table_rows = build_results_table(all_transformer_results, baseline_csv)

    # Save full ablation table
    save_table(table_rows, C.RESULTS_DIR, "ablation_table")

    # Ablation heatmap
    plot_ablation_heatmap(table_rows)

    log.info("Evaluation complete. Results in: %s", C.RESULTS_DIR)


if __name__ == "__main__":
    main()
