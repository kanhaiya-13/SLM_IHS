"""
train.py — Training loop for PatchTST-style Transformer (and LSTM for ablations).

Implements §4 of the project prompt:
  - AdamW optimizer with cosine LR schedule
  - Early stopping on validation PR-AUC (more informative than ROC-AUC for class imbalance)
  - Positive-class weighting (BCEWithLogitsLoss with pos_weight) for imbalanced labels
  - Logs loss curves and metrics per epoch (both to stdout and to results/logs/)
  - Saves best checkpoint per state to results/checkpoints/
  - Supports all ablation flags via CLI: channel_independent, use_weather, use_calendar

Usage:
    python src/train.py --state Maharashtra
    python src/train.py --all-states
    python src/train.py --state Karnataka --no-channel-independent  # ablation: channel-mixing
    python src/train.py --state Karnataka --no-weather               # ablation: no weather covs
    python src/train.py --state Karnataka --no-calendar              # ablation: no calendar feats
"""

import sys
import json
import logging
import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from model import build_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = C.SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def get_channel_indices(all_channels: list[str], use_weather: bool, use_calendar: bool) -> list[int]:
    """
    Return indices into the full channel list that correspond to the active subset.

    This lets ablation runs (no-weather, no-calendar) slice X without re-running the pipeline.
    The case channel (index 0) is always included.
    """
    selected = []
    for i, ch in enumerate(all_channels):
        if ch == "cases_feat":
            selected.append(i)  # always include
        elif ch in ("preci", "LAI", "Temp"):
            if use_weather:
                selected.append(i)
        elif ch in ("week_sin", "week_cos", "month", "is_monsoon"):
            if use_calendar:
                selected.append(i)
        elif ch == "google_trends_interest":
            # Only include if USE_TRENDS is True in config — never include for now
            if C.USE_TRENDS:
                selected.append(i)
        else:
            # Unknown channel: include by default
            selected.append(i)
    return selected


def make_loaders(
    state_data: dict,
    channel_indices: list[int],
    batch_size: int = C.BATCH_SIZE,
) -> tuple[DataLoader, DataLoader]:
    """Build DataLoaders, selecting only the active channel subset."""
    def _make(split: str, shuffle: bool) -> DataLoader:
        X_full = state_data[split]["X"]               # (N, T, C_all)
        X = X_full[:, :, channel_indices]             # (N, T, C_active)
        y = state_data[split]["y_window"]             # (N,)
        y_leads = state_data[split]["y_leads"]        # (N, n_leads)

        # Replace NaN in X with 0 (guard against missing Google Trends placeholder)
        X = np.nan_to_num(X, nan=0.0)

        X_t = torch.from_numpy(X)
        y_t = torch.from_numpy(y).unsqueeze(1)        # (N, 1)
        y_l = torch.from_numpy(y_leads)               # (N, n_leads)
        return DataLoader(
            TensorDataset(X_t, y_t, y_l),
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False,
        )

    return _make("train", shuffle=True), _make("val", shuffle=False)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_state(
    state: str,
    state_data: dict,
    device: torch.device,
    channel_independent: bool = C.CHANNEL_INDEPENDENT,
    use_weather: bool = C.USE_WEATHER,
    use_calendar: bool = C.USE_CALENDAR,
    run_tag: str = "",
) -> dict:
    """
    Train one PatchTST model for a single state with the given ablation config.

    Returns a dict of training metadata (loss curves, best val metrics, checkpoint path).
    """
    set_seed(C.SEED)

    all_channels = state_data["channels"]
    channel_indices = get_channel_indices(all_channels, use_weather, use_calendar)
    n_channels = len(channel_indices)
    active_channels = [all_channels[i] for i in channel_indices]
    log.info("Active channels (%d): %s", n_channels, active_channels)

    train_loader, val_loader = make_loaders(state_data, channel_indices)

    model = build_model(
        n_channels=n_channels,
        model_type="transformer",
        channel_independent=channel_independent,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model params: %d", total_params)

    # Positive class weight — scale by configurable factor to handle imbalance
    y_tr = state_data["train"]["y_window"]
    pos_ratio = y_tr.mean()
    pos_weight_val = C.POS_WEIGHT_SCALE * (1.0 - pos_ratio) / (pos_ratio + 1e-8)
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32, device=device)
    log.info(
        "Class balance — positive: %.1f%%, pos_weight: %.2f",
        pos_ratio * 100, pos_weight_val,
    )

    criterion_window = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion_lead   = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=C.MAX_EPOCHS, eta_min=C.LR / 100
    )

    best_val_pr_auc = -1.0
    patience_counter = 0
    best_epoch = 0
    best_state_dict = None

    train_losses, val_pr_aucs = [], []

    for epoch in range(1, C.MAX_EPOCHS + 1):
        # ── Training ────────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        for X_b, y_b, y_l_b in train_loader:
            X_b = X_b.to(device)
            y_b = y_b.to(device)
            y_l_b = y_l_b.to(device)

            optimizer.zero_grad()
            out = model(X_b)

            # Primary loss: window-level binary cross-entropy
            loss = criterion_window(out["logit_window"], y_b)

            # Multi-task loss: per-lead-week heads (equal weighting, §3)
            if "logit_leads" in out:
                loss_leads = criterion_lead(out["logit_leads"], y_l_b)
                loss = loss + 0.5 * loss_leads   # secondary, lower weight

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(X_b)

        scheduler.step()
        epoch_loss /= len(state_data["train"]["X"])
        train_losses.append(epoch_loss)

        # ── Validation ──────────────────────────────────────────────────────
        model.eval()
        val_probs, val_true = [], []
        with torch.no_grad():
            for X_b, y_b, _ in val_loader:
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

        val_pr_aucs.append(val_pr_auc)

        if val_pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_pr_auc
            best_epoch = epoch
            patience_counter = 0
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if epoch % 10 == 0 or patience_counter >= C.PATIENCE or epoch == 1:
            log.info(
                "[%s%s] Epoch %3d | loss=%.4f | val_PR-AUC=%.4f (best=%.4f @ep%d, patience=%d)",
                state, f"/{run_tag}" if run_tag else "",
                epoch, epoch_loss, val_pr_auc, best_val_pr_auc, best_epoch, patience_counter,
            )

        if patience_counter >= C.PATIENCE:
            log.info("Early stopping at epoch %d (best epoch: %d).", epoch, best_epoch)
            break

    # ── Save checkpoint ──────────────────────────────────────────────────────
    safe_name = state.replace(" ", "_")
    tag_suffix = f"_{run_tag}" if run_tag else ""
    ckpt_path = C.MODELS_DIR / f"{safe_name}{tag_suffix}_best.pt"
    torch.save(
        {
            "model_state_dict": best_state_dict,
            "epoch": best_epoch,
            "val_pr_auc": best_val_pr_auc,
            "n_channels": n_channels,
            "channel_indices": channel_indices,
            "active_channels": active_channels,
            "channel_independent": channel_independent,
            "use_weather": use_weather,
            "use_calendar": use_calendar,
        },
        ckpt_path,
    )
    log.info("Checkpoint saved: %s", ckpt_path)

    # ── Save loss curves ─────────────────────────────────────────────────────
    curves_dir = C.RESULTS_DIR / "logs"
    curves_dir.mkdir(parents=True, exist_ok=True)
    curves_path = curves_dir / f"{safe_name}{tag_suffix}_curves.json"
    with open(curves_path, "w") as f:
        json.dump({"train_loss": train_losses, "val_pr_auc": val_pr_aucs}, f)
    log.info("Loss curves saved: %s", curves_path)

    return {
        "state": state,
        "run_tag": run_tag,
        "channel_independent": channel_independent,
        "use_weather": use_weather,
        "use_calendar": use_calendar,
        "best_epoch": best_epoch,
        "best_val_pr_auc": best_val_pr_auc,
        "ckpt_path": str(ckpt_path),
        "train_losses": train_losses,
        "val_pr_aucs": val_pr_aucs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train outbreak forecasting Transformer.")
    parser.add_argument("--state", type=str, default=None)
    parser.add_argument("--all-states", action="store_true")
    parser.add_argument("--no-channel-independent", action="store_true",
                        help="Use channel-mixing instead of channel-independent (ablation).")
    parser.add_argument("--no-weather", action="store_true",
                        help="Exclude weather covariates (ablation).")
    parser.add_argument("--no-calendar", action="store_true",
                        help="Exclude calendar features (ablation).")
    parser.add_argument("--tag", type=str, default="",
                        help="Optional run tag appended to checkpoint filename.")
    args = parser.parse_args()

    states = C.STATES if (args.all_states or args.state is None) else [args.state]

    channel_independent = not args.no_channel_independent
    use_weather  = not args.no_weather
    use_calendar = not args.no_calendar

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    if torch.cuda.is_available():
        log.info(
            "GPU: %s (%.1f GB VRAM)",
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )

    # Warn if accidental CPU fallback (§6 hardware note)
    if not torch.cuda.is_available():
        log.warning(
            "CUDA not available — running on CPU. "
            "If a GPU is present, check CUDA installation. "
            "Training will be slower but should still complete."
        )

    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    all_run_meta = []
    for state in states:
        log.info("=" * 60)
        log.info("Training Transformer for: %s", state)
        try:
            state_data = load_state_data(state)
        except FileNotFoundError as e:
            log.error(str(e))
            continue

        tag = args.tag or (
            f"ci{'1' if channel_independent else '0'}"
            f"_w{'1' if use_weather else '0'}"
            f"_c{'1' if use_calendar else '0'}"
        )

        meta = train_state(
            state=state,
            state_data=state_data,
            device=device,
            channel_independent=channel_independent,
            use_weather=use_weather,
            use_calendar=use_calendar,
            run_tag=tag,
        )
        all_run_meta.append(meta)

    # Save run metadata summary
    summary_path = C.RESULTS_DIR / "logs" / "train_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        serialisable = [
            {k: v for k, v in m.items() if k not in ("train_losses", "val_pr_aucs")}
            for m in all_run_meta
        ]
        json.dump(serialisable, f, indent=2)
    log.info("Training summary saved: %s", summary_path)
    log.info("Training complete.")


if __name__ == "__main__":
    main()
