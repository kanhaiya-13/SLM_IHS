# Project Prompt: Vector-Borne Disease Outbreak Forecasting (FYP)

## Context

You are building the forecasting core of a final-year engineering project: an early-warning
system that predicts vector-borne disease outbreaks 4–6 weeks in advance for Indian states,
using a Time-Series Transformer. This prompt covers **data preparation, model implementation,
training, and evaluation, end to end**. Treat this as a research-grade ML pipeline, not a demo
— every design decision below was already debated and settled; implement it as specified rather
than re-deciding architecture choices unless you hit a concrete blocker.

Work in a clean project structure (`data/raw`, `data/processed`, `src/`, `notebooks/` or
`experiments/`, `results/`) and commit reproducible, documented code — this will be read and
defended in a viva.

---

## 1. Data

### 1.1 Source files
- `EpiClim.csv` — primary dataset. Columns include:
  `week_of_outbreak` (text, e.g. "1st week"), `state_ut`, `district`, `Disease`, `Cases`
  (string, needs numeric coercion), `Deaths`, `day`, `mon`, `year`, `Latitude`, `Longitude`,
  `preci`, `LAI`, `Temp`.
- **This file is a sparse outbreak *event log*, not a dense weekly census.** Rows exist only
  for weeks where IDSP officially recorded an outbreak. There are no zero-case rows in the raw
  data — absence of a row means "no outbreak reported," not "reported zero cases." You must
  construct the full grid yourself (see §1.3).
- Google Trends data is **not yet available** — build the pipeline so it can be added later as
  an additional per-state-week channel (see §1.5). Do not block on it.

### 1.2 Scope decisions (already made — implement, don't re-litigate)
- **Disease group**: combine `Dengue`, `Chikungunya`, `Malaria` and their labeling variants
  (e.g. "Suspected Dengue", "Dengue And Chikungunya", "Dengue/Chikungunya", "Malaria (PV)",
  etc. — do a case-insensitive substring match on the three root disease names to catch all
  variants) into a single **combined vector-borne outbreak signal**.
- **Granularity**: state-week, NOT district-week (district-level density was verified
  insufficient — only 3 of 1815 district×disease combinations had ≥52 weeks of data across
  14 years).
- **States**: Maharashtra, Karnataka, Tamil Nadu (highest combined-signal density: 44.1%,
  38.8%, 36.5% week-coverage respectively). Build the pipeline generically over a `STATES` list
  so more can be added later, but train/evaluate on these three initially.
- **Wastewater** is explicitly out of scope for this implementation (excluded because it's
  epidemiologically suited to fecal-orally transmitted disease, not vector-borne — documented
  as future work, not built).

### 1.3 Grid construction
1. Parse `week_of_outbreak` into an integer ISO-style week number (extract digits, e.g.
   "1st week" → 1).
2. Filter to the combined vector-borne disease set and the 3 target states.
3. **Aggregate duplicate reports**: multiple rows can share the same `(state_ut, year, week)`
   (different districts reporting in the same week). Group by `(state_ut, year, week)` and
   **sum** `Cases` (coerce to numeric first, non-numeric → NaN → treat as missing, not zero).
4. Build the **full continuous state-week grid**: every `(state, year, week)` combination for
   each state's actual year range (e.g. 2009–2022), including weeks with no aggregated row.
5. For weeks absent from the aggregated log, set `Cases = 0` (no reported outbreak) — but keep
   a separate boolean `is_covered_week` flag distinguishing "confirmed zero" vs. genuinely
   missing weather data, since these are different kinds of missingness.
6. Attach weather covariates (`preci`, `LAI`, `Temp`) per state-week. Where the raw data has
   multiple weather readings for a state-week (from different district rows), average them.
   Where weather is missing entirely for a covered week, forward-fill from the nearest
   available week within the same state (document this choice in code comments).

### 1.4 Label definition
- **Binary outbreak flag per state-week**: 1 if `Cases > 0` (i.e., ≥1 IDSP-confirmed
  vector-borne outbreak that week), else 0. This inherits IDSP's own S/P/L-confirmed outbreak
  determination — do not compute a separate statistical threshold (mean+2SD, percentile, etc.)
  on top of it.
- **Prediction target (what the model actually outputs)**: given a 12-week input window
  ending at week `t`, predict a single binary label = "does at least one outbreak week occur
  in the window `t+4` to `t+6`?" (a 3-week-ahead lead window, not 6 separate per-week
  predictions). This is the primary target. Also compute and report per-lead-week metrics
  (t+4, t+5, t+6 individually) in evaluation for transparency about degradation with horizon.

### 1.5 Channels (multivariate input per state-week)
Required channels, per state per week:
1. `cases` (or a log1p-transformed version — try both, report which is used)
2. `preci`, `LAI`, `Temp` (weather)
3. Calendar features: week-of-year (cyclically encoded via sin/cos), month, and a
   monsoon-season indicator (binary, roughly June–September) — disease dynamics here are
   strongly seasonal and this must not be omitted.
4. Reserve a placeholder column/interface for a future `google_trends_interest` channel
   (fill with NaN or a config flag `USE_TRENDS=False` for now) — do not hardcode the model
   input dimension in a way that requires rearchitecting to add this later.

### 1.6 Normalization and splits
- Per-channel z-score normalization, **fit only on the training split**, applied to
  val/test — never fit on the full dataset (leakage).
- **Time-based split, not random**: train on earlier years, validate and test on later years,
  per state. Suggested split: train through ~2018, validate 2019–2020, test 2021–2022 (adjust
  if a state's data doesn't support this cleanly, but keep it strictly chronological — no
  shuffling across time).

---

## 2. Baselines (implement and evaluate BEFORE the Transformer)

Do not skip this — the project's justification for using a Transformer depends on having
baseline numbers to compare against, not asserting Transformers are better.

1. **Persistence / seasonal-naive**: predict "outbreak in t+4..t+6" = same as whether an
   outbreak occurred in the equivalent weeks of the prior year (or simplest persistence: same
   as most recent observed state).
2. **Classical baseline**: either an ARIMA model on the case-count series (thresholded post-hoc
   to binary) or a simple LSTM classifier on the same windowed input — pick one, document why.

Report the same metrics (§4) for baselines and the Transformer side by side.

---

## 3. Transformer architecture (PatchTST-style)

Implement or adapt an open-source PatchTST base with the following required properties —
these are the specific decisions already made, not open choices:

- **Patch-based tokenization**: patch length aligned to weekly reporting cadence (e.g., a
  4-week patch), not point-wise per-time-step tokens.
- **Channel-independent** design: each input channel (cases, preci, LAI, Temp, calendar
  features) gets its own attention pathway with shared weights across channels, combined at
  the final layer — not naive channel-mixing/concatenation. This is specifically to prevent
  noisy channels (once Trends is added, it will be the noisiest) from corrupting attention for
  cleaner channels.
- **Encoder-only + classification head**: no autoregressive decoder. Encoder produces a
  representation of the 12-week input window; a linear/MLP head outputs a single sigmoid
  probability for the binary "outbreak in t+4..t+6" target (and optionally three additional
  sigmoid heads for the per-lead-week breakdown — a small multi-task head is fine, but the
  primary loss is the window-level target).
- **Calendar-aware embeddings**: inject week-of-year/seasonality embeddings into the patch
  embeddings (this is not optional — see §1.5).
- Train per-state models first (simpler, matches the current 3-state scope); a shared
  multi-state model with a state embedding is a reasonable stretch goal but not required for
  the first working version. Given the available 24GB-VRAM GPU (see §6), the shared
  multi-state model is realistic to attempt in the same pass rather than deferred purely for
  compute reasons — still keep it a stretch goal, since the 3-state data-density scope decision
  in §1.2 is unrelated to compute.

---

## 4. Evaluation

- **Metrics**: Precision, Recall, F1, ROC-AUC, and PR-AUC (PR-AUC matters more than ROC-AUC
  here given class imbalance — report both, discuss the difference).
- Report metrics **separately for the window-level target and for each individual lead week
  (t+4, t+5, t+6)** — accuracy should degrade with horizon; report that honestly rather than
  hiding it.
- **Required ablations** (train/eval variants, tabulate results):
  1. Channel-independent vs. channel-mixing.
  2. With vs. without calendar/seasonal encoding.
  3. With vs. without weather covariates (cases+calendar only, vs. full).
  4. Combined-disease-group target vs. single-disease (e.g. Dengue-only) target, if time
     permits — to justify the disease-grouping decision empirically, not just narratively.
- Compare all of the above against both baselines from §2 in one summary table.
- Log everything (loss curves, metrics per epoch, final test metrics) to enable direct
  copy-paste into the FYP report's Results section.

---

## 5. Deliverables

1. `src/data_pipeline.py` — grid construction, label generation, normalization, splitting
   (§1.3–§1.6), unit-testable and re-runnable from raw CSV to model-ready tensors.
2. `src/baselines.py` — persistence/seasonal-naive + ARIMA-or-LSTM baseline (§2).
3. `src/model.py` — PatchTST-style architecture per §3.
4. `src/train.py` / `src/evaluate.py` — training loop with early stopping on validation loss,
   and evaluation producing all metrics/ablations in §4.
5. `results/` — saved metrics tables (CSV or markdown), plots (loss curves, per-lead-week
   performance degradation chart), and model checkpoints for the best run per state.
6. A short `RESULTS_SUMMARY.md` at the end, written in plain language, stating: final metrics
   per state, which ablation configuration performed best and why (your interpretation), and
   an explicit list of current limitations (label reflects IDSP reporting/confirmation
   behavior not ground-truth incidence; Google Trends not yet integrated; wastewater out of
   scope; 3-state/3-disease scope, not all-India).

---

## 6. Constraints and working notes

- **Target hardware**: single-machine setup with a 13th Gen Intel i9-13900K, 64GB system RAM,
  and a 24GB-VRAM discrete GPU (single-GPU training — the "multiple GPUs" on this machine
  include a non-CUDA/iGPU display adapter, so don't assume multi-GPU data-parallel training is
  available unless verified with `nvidia-smi`/`torch.cuda.device_count()` first). This is
  meaningfully more headroom than a CPU-only or shared-Colab-GPU setup, but it's still an FYP,
  not a large-scale training run — prioritize a correct, well-evaluated pipeline over scale.
  Concretely, this means:
  - PatchTST's small/base configs (not just the smallest tiny configs) fit comfortably in
    24GB VRAM for this dataset size (3 states × ~14 years × weekly = low tens of thousands of
    windows per state at most) — no need to shrink the architecture below what §3 specifies to
    fit memory.
  - The full ablation matrix in §4 (channel-independent vs. mixing × with/without calendar ×
    with/without weather × combined vs. single-disease) is cheap enough on this GPU to run
    exhaustively rather than sampling a subset — do all of it.
  - 64GB RAM is enough to hold all three states' full grids (§1.3) and the intermediate
    pandas/numpy artifacts in memory at once; no need to stream from disk or process states
    one-at-a-time to conserve memory, which simplifies `src/data_pipeline.py`.
  - The LSTM baseline (§2) and per-state Transformer runs (§3) should each train in well under
    an hour on this GPU; if a single run is taking dramatically longer, treat that as a bug
    (e.g. accidental CPU fallback, unbatched data loading) rather than an expected compute
    ceiling to work around.
  - Still keep checkpoints and configs lightweight/reproducible (§5) — the goal is a clean,
    defensible pipeline, not maxing out the hardware for its own sake.
- If any step in §1.3 hits an ambiguous edge case in the real data (e.g. a state's year range
  differs from assumed, a disease-variant string doesn't match the substring rules), stop and
  flag it with the specific row/value rather than silently guessing — data correctness here is
  more important than pipeline completion speed.
- Do not add data sources, disease scope, or architectural components beyond what's specified
  above without flagging the addition explicitly — the scope in this prompt reflects deliberate
  trade-offs already made against known data-density constraints.
