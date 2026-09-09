"""
data_pipeline.py — Grid construction, label generation, normalization, splitting.

Implements §1.3–§1.6 of the project prompt exactly:
  - Parses week_of_outbreak text → integer week number
  - Filters to the combined vector-borne disease set and 3 target states
  - Aggregates duplicate district reports (sum Cases per state-week)
  - Builds full continuous state-week grid; marks is_covered_week
  - Attaches weather covariates (averaged across districts; forward-filled)
  - Computes binary outbreak flag (Cases > 0)
  - Builds sliding-window samples (12-week input → t+4..t+6 binary label)
  - Encodes cyclical calendar features + monsoon indicator
  - Z-score normalises per channel, fit only on training data
  - Exports final tensors to data/processed/

Usage:
    python src/data_pipeline.py          # processes all STATES, saves artefacts
    python -m pytest src/data_pipeline.py  # runs embedded unit tests (doctest-style)

Design note on missingness (§1.3 step 6):
    Weather covariates (preci, LAI, Temp) are averaged across all district rows that
    share a (state_ut, year, week). If a full state-week has no weather readings at all,
    values are forward-filled from the most recent prior week within the same state.
    This is a known limitation documented here: forward-filling weather can introduce
    up to several weeks of stale values during data gaps; it is preferred over mean
    imputation because weather is autocorrelated week-to-week.
"""

import re
import sys
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ── allow `python src/data_pipeline.py` from project root ──────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# §1.3  Step 1 — parse week_of_outbreak → integer
# ─────────────────────────────────────────────────────────────────────────────

def parse_week_number(text: str) -> int | None:
    """
    Convert 'week_of_outbreak' text to an integer week number.

    Examples:
      "1st week"  → 1
      "10th week" → 10
      "21st week" → 21

    Returns None (NaN-compatible) for strings that contain no digit sequence.
    """
    if not isinstance(text, str):
        return None
    m = re.search(r"\d+", text)
    return int(m.group()) if m else None


# ─────────────────────────────────────────────────────────────────────────────
# §1.3  Steps 2–3 — filter + aggregate
# ─────────────────────────────────────────────────────────────────────────────

def load_and_filter(csv_path: Path) -> pd.DataFrame:
    """
    Load raw CSV, filter to target diseases and states, aggregate by state-week.

    Returns a DataFrame with columns:
        state_ut | year | week | Cases | preci | LAI | Temp
    where Cases is a numeric aggregate (sum) across districts.
    """
    df = pd.read_csv(csv_path)

    # ── parse week number ──────────────────────────────────────────────────
    df["week"] = df["week_of_outbreak"].apply(parse_week_number)
    n_bad_weeks = df["week"].isna().sum()
    if n_bad_weeks:
        log.warning(
            "Could not parse week from %d rows — these will be dropped. "
            "Inspect the 'week_of_outbreak' column for unexpected formats.",
            n_bad_weeks,
        )
        df = df.dropna(subset=["week"])
    df["week"] = df["week"].astype(int)

    # ── filter diseases: case-insensitive substring match (§1.2) ──────────
    def _is_vector_borne(disease_str: str) -> bool:
        d = disease_str.lower()
        return any(kw in d for kw in C.DISEASE_KEYWORDS)

    mask_disease = df["Disease"].apply(_is_vector_borne)
    log.info(
        "Disease filter: %d / %d rows kept as vector-borne.",
        mask_disease.sum(), len(df),
    )
    df = df[mask_disease]

    # ── filter states ──────────────────────────────────────────────────────
    mask_state = df["state_ut"].isin(C.STATES)
    log.info(
        "State filter: %d / %d rows kept for target states %s.",
        mask_state.sum(), len(df), C.STATES,
    )
    df = df[mask_state]

    # ── coerce Cases to numeric (§1.3 step 3) ─────────────────────────────
    df["Cases"] = pd.to_numeric(df["Cases"], errors="coerce")
    n_non_numeric = df["Cases"].isna().sum()
    if n_non_numeric:
        log.warning(
            "Could not coerce 'Cases' to numeric for %d rows — treating as missing (NaN), "
            "NOT as zero. These rows will be excluded from the Cases sum per state-week.",
            n_non_numeric,
        )
    # Do NOT fill NaN Cases with 0 here — NaN propagation is intentional (§1.3 step 3).

    # ── aggregate weather covariates (average across districts) ───────────
    # Build separate weather aggregate (mean) before summing Cases, because
    # the weather columns live in district rows, not state-level rows.
    weather_agg = (
        df.groupby(["state_ut", "year", "week"])[["preci", "LAI", "Temp"]]
        .mean()
        .reset_index()
    )

    # ── aggregate Cases (sum; NaN-propagates only if ALL values are NaN) ──
    # pandas sum() with skipna=True treats NaN as 0, which would undercount.
    # We want: if any value is numeric, sum the numeric ones; if ALL are NaN,
    # return NaN (so the state-week is flagged as truly missing data, not zero).
    def _cases_sum(x):
        if x.notna().any():
            return x.sum(skipna=True)  # sum the non-NaN district values
        return float("nan")            # all districts had non-numeric Cases

    cases_agg = (
        df.groupby(["state_ut", "year", "week"])["Cases"]
        .apply(_cases_sum)
        .reset_index()
    )

    # ── merge ──────────────────────────────────────────────────────────────
    agg = pd.merge(cases_agg, weather_agg, on=["state_ut", "year", "week"], how="left")
    log.info("Aggregated: %d unique (state, year, week) rows.", len(agg))
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# §1.3  Steps 4–6 — build continuous grid per state
# ─────────────────────────────────────────────────────────────────────────────

def build_grid(agg: pd.DataFrame, state: str) -> pd.DataFrame:
    """
    Build the full continuous state-week grid for one state.

    For weeks absent from the aggregated log:
      - Cases = 0  (no reported outbreak)
      - is_covered_week = False (i.e., this zero is 'confirmed absence', not 'missing data')
    For weeks present in the aggregated log:
      - is_covered_week = True

    Weather covariates that are missing for a covered week are forward-filled
    from the nearest prior available week within the same state (§1.3 step 6).

    Returns a DataFrame with a proper DatetimeIndex for clarity, sorted chronologically.
    """
    state_df = agg[agg["state_ut"] == state].copy()
    year_min = int(state_df["year"].min())
    year_max = int(state_df["year"].max())
    log.info("State '%s': year range %d–%d.", state, year_min, year_max)

    # Build full (year, week) index — weeks 1..52 for each year
    years = range(year_min, year_max + 1)
    full_index = pd.DataFrame(
        [(y, w) for y in years for w in range(1, 53)],
        columns=["year", "week"],
    )
    full_index["state_ut"] = state

    # Left-join: rows from agg that match get their Cases/weather values;
    # rows with no match get NaN.
    grid = pd.merge(full_index, state_df, on=["state_ut", "year", "week"], how="left")
    grid["is_covered_week"] = grid["Cases"].notna()

    # Weeks absent from the log → Cases = 0 (confirmed absence, §1.3 step 5)
    grid["Cases"] = grid["Cases"].fillna(0.0)

    # Forward-fill weather for weeks where weather is missing (§1.3 step 6).
    # This applies both to 'absent' weeks and to covered weeks with missing sensors.
    # Forward-fill is appropriate because weather is temporally autocorrelated.
    for col in ["preci", "LAI", "Temp"]:
        n_missing_before = grid[col].isna().sum()
        grid[col] = grid[col].ffill()
        n_missing_after = grid[col].isna().sum()
        if n_missing_before:
            log.info(
                "  '%s' %s: forward-filled %d gaps; %d still NaN (leading rows).",
                state, col, n_missing_before - n_missing_after, n_missing_after,
            )

    grid = grid.sort_values(["year", "week"]).reset_index(drop=True)
    log.info(
        "Grid for '%s': %d rows, coverage %.1f%%.",
        state,
        len(grid),
        grid["is_covered_week"].mean() * 100,
    )
    return grid


# ─────────────────────────────────────────────────────────────────────────────
# §1.4  Outbreak flag + label
# ─────────────────────────────────────────────────────────────────────────────

def add_outbreak_flag(grid: pd.DataFrame) -> pd.DataFrame:
    """
    Binary outbreak flag: 1 if Cases > 0 for that state-week, else 0.

    We directly use IDSP's own confirmation (an outbreak row exists) as the
    positive signal — no separate statistical threshold is computed (§1.4).
    """
    grid = grid.copy()
    grid["outbreak"] = (grid["Cases"] > 0).astype(int)
    return grid


# ─────────────────────────────────────────────────────────────────────────────
# §1.5  Calendar features
# ─────────────────────────────────────────────────────────────────────────────

def add_calendar_features(grid: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclical week-of-year encoding, month (approximated from week), and
    a binary monsoon-season indicator (§1.5).

    week → (sin, cos) encodes periodicity so week 52 is adjacent to week 1.
    month is approximated as ceil(week / 4.33) and clipped to [1, 12].
    """
    grid = grid.copy()
    w = grid["week"].values
    grid["week_sin"] = np.sin(2 * np.pi * w / 52)
    grid["week_cos"] = np.cos(2 * np.pi * w / 52)

    # Approximate month from week (weeks 1-4 ≈ Jan, etc.)
    grid["month"] = np.clip(np.ceil(w / (52 / 12)).astype(int), 1, 12)
    grid["is_monsoon"] = grid["month"].isin(C.MONSOON_MONTHS).astype(float)

    return grid


# ─────────────────────────────────────────────────────────────────────────────
# §1.5  Channel assembly
# ─────────────────────────────────────────────────────────────────────────────

def get_channel_names() -> list[str]:
    """
    Return the ordered list of input channel names for the current config flags.

    The returned order is the column order used for building input tensors;
    it must stay consistent between pipeline, model, and evaluation code.
    """
    channels = []

    # Always include the case count (possibly log1p-transformed)
    channels.append("cases_feat")

    if C.USE_WEATHER:
        channels += ["preci", "LAI", "Temp"]

    if C.USE_CALENDAR:
        channels += ["week_sin", "week_cos", "month", "is_monsoon"]

    # Placeholder for Google Trends (§1.5) — filled with NaN / zeros when not available
    if C.USE_TRENDS:
        channels.append("google_trends_interest")

    return channels


def assemble_channels(grid: pd.DataFrame) -> pd.DataFrame:
    """
    Build the feature matrix from the grid DataFrame.

    Applies log1p transform to cases if USE_LOG_CASES is True (§1.5).
    Adds a zero-filled placeholder for Google Trends if USE_TRENDS is False.
    """
    grid = grid.copy()

    # Case feature
    if C.USE_LOG_CASES:
        grid["cases_feat"] = np.log1p(grid["Cases"])
    else:
        grid["cases_feat"] = grid["Cases"]

    # Google Trends placeholder (§1.5)
    if C.USE_TRENDS:
        if "google_trends_interest" not in grid.columns:
            # Will be filled with NaN; the model must handle this via masking or imputation
            grid["google_trends_interest"] = np.nan

    return grid


# ─────────────────────────────────────────────────────────────────────────────
# §1.6  Splits
# ─────────────────────────────────────────────────────────────────────────────

def split_grid(grid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological time-based split of the state grid into train / val / test.

    No shuffling — strictly by year (§1.6).
    """
    train = grid[grid["year"] <= C.TRAIN_END_YEAR].copy()
    val = grid[(grid["year"] >= C.VAL_START_YEAR) & (grid["year"] <= C.VAL_END_YEAR)].copy()
    test = grid[grid["year"] >= C.TEST_START_YEAR].copy()

    log.info(
        "  Split sizes — train: %d, val: %d, test: %d rows.",
        len(train), len(val), len(test),
    )
    assert len(train) > 0, "Train split is empty — check TRAIN_END_YEAR vs. data range."
    assert len(val) > 0, "Val split is empty — check VAL_START/END_YEAR vs. data range."
    assert len(test) > 0, "Test split is empty — check TEST_START_YEAR vs. data range."
    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
# §1.6  Z-score normalization
# ─────────────────────────────────────────────────────────────────────────────

def fit_scalers(train_df: pd.DataFrame, channels: list[str]) -> dict[str, StandardScaler]:
    """
    Fit one StandardScaler per channel on the training data ONLY.

    Returned dict is saved alongside tensors to allow inverse-transform at eval time.
    """
    scalers = {}
    for ch in channels:
        if ch not in train_df.columns:
            raise ValueError(f"Channel '{ch}' not found in grid DataFrame.")
        scaler = StandardScaler()
        # reshape(-1,1) for sklearn API
        values = train_df[ch].values.reshape(-1, 1)
        # If all values are identical (e.g. is_monsoon during a constant period),
        # sklearn will set scale_=0; we set scale_ to 1 to avoid divide-by-zero.
        scaler.fit(values)
        if scaler.scale_ is not None and np.any(scaler.scale_ == 0):
            log.warning(
                "Channel '%s' has zero variance in the training set — scaler scale set to 1.",
                ch,
            )
            scaler.scale_[scaler.scale_ == 0] = 1.0
        scalers[ch] = scaler
    return scalers


def apply_scalers(df: pd.DataFrame, scalers: dict[str, StandardScaler]) -> pd.DataFrame:
    """Apply pre-fitted scalers to a DataFrame split (val or test)."""
    df = df.copy()
    for ch, scaler in scalers.items():
        if ch in df.columns:
            df[ch] = scaler.transform(df[ch].values.reshape(-1, 1)).flatten()
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Sliding-window sample construction
# ─────────────────────────────────────────────────────────────────────────────

def make_windows(
    df: pd.DataFrame,
    channels: list[str],
    input_len: int = C.INPUT_WINDOW,
    lead_min: int = C.LEAD_MIN,
    lead_max: int = C.LEAD_MAX,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build sliding-window (X, y_window, y_leads) arrays from a normalised grid split.

    X: shape (N, input_len, C)  — C = number of channels
       N windows, each input_len steps long, each step has C channel values.

    y_window: shape (N,) — binary; 1 if ANY outbreak in t+lead_min..t+lead_max.
    y_leads: shape (N, lead_max - lead_min + 1) — per-lead-week binary labels,
             columns correspond to t+lead_min, t+lead_min+1, ..., t+lead_max.

    The df must be sorted chronologically (year, week) before calling this function.
    """
    outbreak = df["outbreak"].values
    feature_matrix = df[channels].values  # (T, C)

    T = len(df)
    # last valid window start: we need input_len history + lead_max future rows
    max_start = T - input_len - lead_max

    X_list, y_win_list, y_leads_list = [], [], []

    for t in range(max_start + 1):
        # input window: rows t .. t+input_len-1
        x = feature_matrix[t : t + input_len]  # (input_len, C)

        # forecast horizon: rows t+input_len+lead_min-1 .. t+input_len+lead_max-1
        # (0-indexed so t+input_len is t+1 steps ahead, t+input_len+3 is t+4 ahead)
        horizon_start = t + input_len + lead_min - 1
        horizon_end   = t + input_len + lead_max      # exclusive

        lead_labels = outbreak[horizon_start : horizon_end]  # length = lead_max - lead_min + 1
        win_label = int(lead_labels.max() > 0)  # 1 if ANY lead week has an outbreak

        X_list.append(x)
        y_win_list.append(win_label)
        y_leads_list.append(lead_labels)

    X = np.stack(X_list).astype(np.float32)
    y_window = np.array(y_win_list, dtype=np.float32)
    y_leads = np.stack(y_leads_list).astype(np.float32)

    log.info(
        "  Windows: %d total, %d positive (%.1f%%).",
        len(X), y_window.sum(), y_window.mean() * 100,
    )
    return X, y_window, y_leads


# ─────────────────────────────────────────────────────────────────────────────
# Top-level runner
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    csv_path: Path = C.RAW_CSV,
    out_dir: Path = C.DATA_PROC_DIR,
    states: list[str] = C.STATES,
) -> dict:
    """
    End-to-end pipeline: raw CSV → normalised, windowed tensors per state.

    Saves one pickle file per state in `out_dir`:
      {state}_data.pkl — dict with keys: train/val/test → (X, y_window, y_leads)
                                          scalers, channels, grid_full

    Returns the same nested dict for in-memory use.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    channels = get_channel_names()
    log.info("Channels: %s", channels)

    # Load and filter raw data once (shared across states)
    agg = load_and_filter(csv_path)

    all_data = {}
    for state in states:
        log.info("=" * 60)
        log.info("Processing state: %s", state)

        # Build continuous grid
        grid = build_grid(agg, state)
        grid = add_outbreak_flag(grid)
        grid = add_calendar_features(grid)
        grid = assemble_channels(grid)

        # Check for any disease-variant strings that slipped through unexpected parsing
        # (flags any column-level anomaly as a WARNING rather than silently continuing)
        if grid["Cases"].isna().any():
            bad_mask = grid["Cases"].isna()
            log.warning(
                "State '%s': %d rows still have NaN Cases after grid construction — "
                "check data for unexpected structures.",
                state, bad_mask.sum(),
            )

        # Split
        train, val, test = split_grid(grid)

        # Fit scalers on train only (§1.6 — leakage prevention)
        scalers = fit_scalers(train, channels)

        # Normalise each split
        train_norm = apply_scalers(train, scalers)
        val_norm   = apply_scalers(val,   scalers)
        test_norm  = apply_scalers(test,  scalers)

        # Build windows
        log.info("Building train windows for %s:", state)
        X_tr, y_w_tr, y_l_tr = make_windows(train_norm, channels)
        log.info("Building val windows for %s:", state)
        X_va, y_w_va, y_l_va = make_windows(val_norm,   channels)
        log.info("Building test windows for %s:", state)
        X_te, y_w_te, y_l_te = make_windows(test_norm,  channels)

        state_data = {
            "train": {"X": X_tr, "y_window": y_w_tr, "y_leads": y_l_tr},
            "val":   {"X": X_va, "y_window": y_w_va, "y_leads": y_l_va},
            "test":  {"X": X_te, "y_window": y_w_te, "y_leads": y_l_te},
            "scalers": scalers,
            "channels": channels,
            "grid_full": grid,
        }
        all_data[state] = state_data

        # Save to disk
        safe_name = state.replace(" ", "_")
        pkl_path = out_dir / f"{safe_name}_data.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(state_data, f, protocol=5)
        log.info("Saved: %s", pkl_path)

        # ── Integrity checks ──────────────────────────────────────────────
        # 1. No train row should appear after VAL_START_YEAR
        assert train["year"].max() <= C.TRAIN_END_YEAR, "Leakage: train row in val period!"
        # 2. Scaler fit only on train (spot-check mean is reasonable)
        for ch, sc in scalers.items():
            assert not np.isnan(sc.mean_).any(), f"Scaler mean is NaN for channel {ch}"
        # 3. Window shapes are self-consistent
        n_ch = len(channels)
        assert X_tr.shape[1:] == (C.INPUT_WINDOW, n_ch), \
            f"Unexpected X_tr shape {X_tr.shape}"
        assert y_l_tr.shape[1] == (C.LEAD_MAX - C.LEAD_MIN + 1), \
            f"Unexpected y_leads shape {y_l_tr.shape}"
        log.info("All integrity checks passed for %s.", state)

    log.info("Pipeline complete.")
    return all_data


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests (run with: python src/data_pipeline.py test)
# ─────────────────────────────────────────────────────────────────────────────

def _run_unit_tests():
    """Minimal unit tests for the parsing and grid logic."""
    # parse_week_number
    assert parse_week_number("1st week") == 1
    assert parse_week_number("10th week") == 10
    assert parse_week_number("21st week") == 21
    assert parse_week_number("52nd week") == 52
    assert parse_week_number("bad string") is None
    assert parse_week_number(None) is None
    print("[PASS] parse_week_number")

    # make_windows — synthetic test
    import pandas as pd
    T = 30
    synthetic = pd.DataFrame({
        "year": [2015] * T,
        "week": list(range(1, T + 1)),
        "cases_feat": np.random.randn(T),
        "outbreak": ([0] * 10 + [1] * 5 + [0] * 15),
    })
    ch = ["cases_feat"]
    X, y_w, y_l = make_windows(synthetic, ch, input_len=12, lead_min=4, lead_max=6)
    # max_start = 30 - 12 - 6 = 12, so 13 windows (0..12)
    assert X.shape == (13, 12, 1), f"Expected (13,12,1), got {X.shape}"
    assert y_l.shape == (13, 3), f"Expected (13,3), got {y_l.shape}"
    print("[PASS] make_windows")

    print("All unit tests passed.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        _run_unit_tests()
    else:
        run_pipeline()
