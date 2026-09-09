"""
config.py — Central configuration for the Vector-Borne Disease Outbreak
Forecasting pipeline.

All design decisions referenced in the project prompt are codified here.
Change flags here; do NOT scatter magic numbers through the pipeline code.
"""

from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent          # SLM_IHS/
DATA_RAW_DIR = ROOT / "data" / "raw"
DATA_PROC_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = RESULTS_DIR / "checkpoints"

RAW_CSV = DATA_RAW_DIR / "EpiClim.csv"

# ──────────────────────────────────────────────────────────────────────────────
# Scope decisions (§1.2 of prompt — do not change without updating the prompt)
# ──────────────────────────────────────────────────────────────────────────────

# Target states (highest combined-signal density: 44.1%, 38.8%, 36.5%)
STATES = ["Maharashtra", "Karnataka", "Tamil Nadu"]

# Vector-borne disease root keywords — case-insensitive substring match
# catches: "Dengue", "Suspected Dengue", "Dengue And Chikungunya",
# "Dengue/Chikungunya", "Malaria", "Malaria (PV)", "Chikungunya", etc.
DISEASE_KEYWORDS = ["dengue", "chikungunya", "malaria"]

# ──────────────────────────────────────────────────────────────────────────────
# Time-based splits (§1.6)
# ──────────────────────────────────────────────────────────────────────────────
TRAIN_END_YEAR = 2018      # inclusive — train on 2009..2018
VAL_START_YEAR = 2019
VAL_END_YEAR = 2020        # inclusive
TEST_START_YEAR = 2021
TEST_END_YEAR = 2022       # inclusive

# ──────────────────────────────────────────────────────────────────────────────
# Windowing (§1.4)
# ──────────────────────────────────────────────────────────────────────────────
INPUT_WINDOW = 12          # weeks of history fed to the model
LEAD_MIN = 4               # earliest forecast horizon (t+4)
LEAD_MAX = 6               # latest forecast horizon (t+6)

# ──────────────────────────────────────────────────────────────────────────────
# Channels / features (§1.5)
# ──────────────────────────────────────────────────────────────────────────────
USE_WEATHER = True          # include preci, LAI, Temp
USE_CALENDAR = True         # include week-of-year sin/cos, month, monsoon flag
USE_LOG_CASES = True        # log1p transform on the case-count channel

# Future channel placeholder — set False until Google Trends data is available
USE_TRENDS = False

# Monsoon season: June (month 6) through September (month 9), inclusive
MONSOON_MONTHS = {6, 7, 8, 9}

# ──────────────────────────────────────────────────────────────────────────────
# Model hyperparameters (§3, PatchTST-style)
# ──────────────────────────────────────────────────────────────────────────────
PATCH_LEN = 4              # 4-week patches; 12-week window → 3 patches
D_MODEL = 128              # transformer embedding dimension
N_HEADS = 8
N_LAYERS = 3
D_FF = 512                 # feed-forward hidden dim
DROPOUT = 0.1
CHANNEL_INDEPENDENT = True # default; ablation will also try False

# Classification heads
MULTITASK = True           # also output per-lead-week (t+4, t+5, t+6) heads

# ──────────────────────────────────────────────────────────────────────────────
# Training (§4)
# ──────────────────────────────────────────────────────────────────────────────
BATCH_SIZE = 64
LR = 3e-4
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 100
PATIENCE = 15              # early stopping on val PR-AUC
POS_WEIGHT_SCALE = 2.0     # upweight positive class in BCEWithLogitsLoss

# ──────────────────────────────────────────────────────────────────────────────
# LSTM baseline hyperparameters (§2)
# ──────────────────────────────────────────────────────────────────────────────
LSTM_HIDDEN = 64
LSTM_LAYERS = 2
LSTM_DROPOUT = 0.2
LSTM_EPOCHS = 80
LSTM_PATIENCE = 10

# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────────────
SEED = 42
