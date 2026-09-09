# Vector-Borne Disease Outbreak Forecasting (FYP)

An early-warning system that predicts vector-borne disease outbreaks (Dengue, Chikungunya, Malaria) **4–6 weeks in advance** for Indian states using a **PatchTST-style Time-Series Transformer**.

---

## Project Structure

```
SLM_IHS/
├── data/
│   ├── raw/EpiClim.csv          ← IDSP outbreak + weather data (2009–2022)
│   └── processed/               ← generated state-week grids + tensors (auto-created)
├── src/
│   ├── config.py                ← central config (states, splits, model hyperparams)
│   ├── data_pipeline.py         ← §1: grid construction, labels, normalization, splitting
│   ├── baselines.py             ← §2: persistence + LSTM classifier baselines
│   ├── model.py                 ← §3: PatchTST-style Transformer + LSTM model definitions
│   ├── train.py                 ← §4: training loop (AdamW, cosine LR, early stopping)
│   ├── evaluate.py              ← §4: metrics, ablation table, plots
│   ├── generate_summary.py      ← generates RESULTS_SUMMARY.md from actual metrics
│   └── run_all.py               ← top-level orchestrator (runs steps 1–5 in order)
├── results/
│   ├── checkpoints/             ← best model per state per ablation config
│   ├── logs/                    ← loss curves (.json), training summaries
│   └── plots/                   ← loss curves, lead-week degradation, ablation heatmap
├── RESULTS_SUMMARY.md           ← auto-generated plain-language results (§5)
└── requirements.txt
```

---

## Quick Start

### 1. Install Dependencies

```bash
# Core packages
python -m pip install pandas numpy scikit-learn matplotlib seaborn statsmodels

# PyTorch (match to your CUDA version — see https://pytorch.org/get-started/locally/)
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 2. Run the Full Pipeline

```bash
# Full pipeline: data → baselines → training + full ablations → evaluation → summary
python src/run_all.py

# Faster: skip full ablation matrix, run only default config
python src/run_all.py --no-ablations
```

### 3. Step-by-Step

```bash
# Step 1: Build processed data grids
python src/data_pipeline.py

# Step 2: Run baselines (persistence + LSTM)
python src/baselines.py --all-states

# Step 3: Train Transformer (default config)
python src/train.py --all-states

# Step 4: Evaluate + generate all plots
python src/evaluate.py --all-states

# Step 4 (alternative): Full ablation matrix (trains missing configs then evaluates)
python src/evaluate.py --all-states --run-ablations

# Step 5: Generate RESULTS_SUMMARY.md
python src/generate_summary.py
```

### 4. Ablation-specific training

```bash
# Ablation: channel-mixing (not channel-independent)
python src/train.py --state Maharashtra --no-channel-independent --tag CM_W_C

# Ablation: no weather covariates
python src/train.py --state Maharashtra --no-weather --tag CI_NW_C

# Ablation: no calendar features
python src/train.py --state Maharashtra --no-calendar --tag CI_W_NC
```

---

## Scope

| Decision | Value |
|---|---|
| Disease group | Dengue + Chikungunya + Malaria (combined, substring-matched) |
| Granularity | State-week (not district-week — density insufficient) |
| States | Maharashtra, Karnataka, Tamil Nadu |
| Train/Val/Test | 2009–2018 / 2019–2020 / 2021–2022 |
| Input window | 12 weeks |
| Forecast horizon | t+4 to t+6 (window label = any outbreak in that 3-week window) |
| Channels | cases (log1p), preci, LAI, Temp, week sin/cos, month, monsoon flag |

---

## Model Architecture

**PatchTST-style encoder-only Transformer:**
- 4-week patches → 3 patches from 12-week window
- Channel-independent design (default): each channel's patch sequence processed through shared Transformer weights separately
- Calendar-aware patch embeddings (learnable positional + sinusoidal week encoding)
- Classification head: sigmoid over mean-pooled representation
- Multi-task head: per-lead-week (t+4, t+5, t+6) probability outputs

---

## Ablation Matrix (§4)

| Config tag | Channel mode | Weather | Calendar |
|---|---|---|---|
| CI_W_C (default) | Independent | ✓ | ✓ |
| CM_W_C | Mixing | ✓ | ✓ |
| CI_W_NC | Independent | ✓ | ✗ |
| CI_NW_C | Independent | ✗ | ✓ |
| CI_NW_NC | Independent | ✗ | ✗ |

---

## Known Limitations

See `RESULTS_SUMMARY.md §3` for a detailed discussion. In brief:
- Label reflects IDSP reporting/confirmation behaviour, not ground-truth incidence
- Google Trends not yet integrated (placeholder reserved in pipeline + model)
- Wastewater out of scope (epidemiologically appropriate — vector-borne, not fecal-oral)
- 3-state / 3-disease scope; pipeline is generic over `STATES` list in `config.py`
