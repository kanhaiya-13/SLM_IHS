# Vector-Borne Disease Outbreak Forecasting — Implementation Walkthrough

## What Was Built

A complete research-grade ML pipeline per the project prompt, across **6 source files**, all placed in `src/`.

---

## Files Created

| File | Purpose |
|---|---|
| [`src/config.py`](file:///c:/Users/23101A0034/SLM_IHS/src/config.py) | Central config: STATES, splits, model hyperparams, channel flags |
| [`src/data_pipeline.py`](file:///c:/Users/23101A0034/SLM_IHS/src/data_pipeline.py) | §1: grid construction, labels, normalization, windowing |
| [`src/baselines.py`](file:///c:/Users/23101A0034/SLM_IHS/src/baselines.py) | §2: persistence (prior-year) + LSTM classifier baselines |
| [`src/model.py`](file:///c:/Users/23101A0034/SLM_IHS/src/model.py) | §3: PatchTST-style Transformer + LSTMClassifier |
| [`src/train.py`](file:///c:/Users/23101A0034/SLM_IHS/src/train.py) | §4: training loop, early stopping, cosine LR schedule |
| [`src/evaluate.py`](file:///c:/Users/23101A0034/SLM_IHS/src/evaluate.py) | §4: metrics, full ablation matrix, plots |
| [`src/generate_summary.py`](file:///c:/Users/23101A0034/SLM_IHS/src/generate_summary.py) | §5: auto-generates RESULTS_SUMMARY.md from actual metrics |
| [`src/run_all.py`](file:///c:/Users/23101A0034/SLM_IHS/src/run_all.py) | Top-level orchestrator |
| [`README.md`](file:///c:/Users/23101A0034/SLM_IHS/README.md) | Full project documentation |
| [`requirements.txt`](file:///c:/Users/23101A0034/SLM_IHS/requirements.txt) | Python dependencies |
| [`.gitignore`](file:///c:/Users/23101A0034/SLM_IHS/.gitignore) | Git ignore rules |

---

## Data Pipeline Results (Verified ✅)

The data pipeline ran successfully on EpiClim.csv:

| State | Year Range | Grid Rows | Week Coverage | Train Windows | Val Windows | Test Windows |
|---|---|---|---|---|---|---|
| Maharashtra | 2009–2022 | 728 | 43.8% | 503 | 87 | 87 |
| Karnataka | 2010–2022 | 676 | 38.6% | 451 | 87 | 87 |
| Tamil Nadu | 2009–2022 | 728 | 36.5% | 503 | 87 | 87 |

All integrity checks passed: no data leakage, correct chronological splits, correct tensor shapes.

**Disease filter**: 2,977 / 8,985 rows matched vector-borne diseases → 1,523 rows for 3 target states.

**Channels** (8 total): `cases_feat`, `preci`, `LAI`, `Temp`, `week_sin`, `week_cos`, `month`, `is_monsoon`

---

## Model Architecture (Verified ✅)

| Model | Params | Shape in → out |
|---|---|---|
| PatchTST (channel-independent, default) | 1,124,484 | (B,12,8) → logit (B,1) + leads (B,3) |
| PatchTST (channel-mixing, ablation) | 608,388 | (B,12,8) → logit (B,1) |
| LSTM Classifier (baseline) | 52,484 | (B,12,8) → logit (B,1) + leads (B,3) |

Smoke test passed on CPU. All output shapes correct.

---

## Next Steps — Run Training

### Step 1: Set up the virtual environment

The `.venv` was created. Install dependencies:

```powershell
# Activate the venv
.venv\Scripts\Activate.ps1

# Install PyTorch (CPU or GPU — see README for CUDA instructions)
python -m pip install torch --pre

# Verify
python -c "import torch; print(torch.__version__)"
```

### Step 2: Run the full pipeline

```powershell
# Full pipeline: data → baselines → Transformer → all ablations → summary
python src/run_all.py

# OR, step by step:
python src/data_pipeline.py       # already done ✅
python src/baselines.py --all-states
python src/evaluate.py --all-states --run-ablations   # trains + evaluates all ablation configs
python src/generate_summary.py
```

> **Note on GPU**: The pipeline auto-detects CUDA. If `torch.cuda.is_available()` returns True, training uses the GPU. For the 24GB GPU + ~1000 windows per state, each ablation config trains in well under an hour.

---

## Ablation Matrix (§4)

All 5 configs will be trained and evaluated by `evaluate.py --run-ablations`:

| Tag | Channel Mode | Weather | Calendar | Description |
|---|---|---|---|---|
| `CI_W_C` | Independent | ✓ | ✓ | **Default (full model)** |
| `CM_W_C` | Mixing | ✓ | ✓ | Ablation 1: channel-mixing |
| `CI_W_NC` | Independent | ✓ | ✗ | Ablation 2: no calendar |
| `CI_NW_C` | Independent | ✗ | ✓ | Ablation 3: no weather |
| `CI_NW_NC` | Independent | ✗ | ✗ | Ablation 2+3: cases only |

---

## Key Design Decisions (as specified)

- **Labels**: Binary outbreak flag directly from IDSP rows — Cases > 0 per state-week. No separate threshold on top of IDSP's own S/P/L determination.
- **Normalization**: z-score per channel, fit only on train split, applied to val/test.
- **Weather imputation**: forward-fill from nearest prior week within the same state (not mean imputation — weather is autocorrelated week-to-week). Leading NaN rows are left as NaN and handled by `nan_to_num(0)` in the dataloader.
- **Positive class weight**: `BCEWithLogitsLoss(pos_weight=...)` with configurable scale — addresses class imbalance.
- **Google Trends**: Placeholder channel reserved in pipeline and model. Set `USE_TRENDS=True` in `config.py` and populate `google_trends_interest` column when data is available.
- **Early stopping**: On validation PR-AUC (not ROC-AUC — more informative under class imbalance, §4).
