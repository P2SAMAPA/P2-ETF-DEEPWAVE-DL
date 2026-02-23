# config.py — single source of truth for all constants
import os
from dotenv import load_dotenv
load_dotenv()

# ── Repos ─────────────────────────────────────────────────────────────────────
HF_DATASET_REPO = "P2SAMAPA/p2-etf-deepwave-dl"
HF_SPACE_REPO   = "P2SAMAPA/P2-ETF-DEEPWAVE-DL"
GITHUB_REPO     = "P2SAMAPA/P2-ETF-DEEPWAVE-DL"

# ── API Keys ───────────────────────────────────────────────────────────────────
HF_TOKEN        = os.getenv("HF_TOKEN", "")
FRED_API_KEY    = os.getenv("FRED_API_KEY", "")
GITHUB_TOKEN    = os.getenv("GITHUB_TOKEN", "")

# ── Universe ───────────────────────────────────────────────────────────────────
ETFS            = ["TLT", "TBT", "VNQ", "GLD", "SLV"]
BENCHMARKS      = ["SPY", "AGG"]
ALL_TICKERS     = ETFS + BENCHMARKS

# ── FRED Macro Series ─────────────────────────────────────────────────────────
MACRO_SERIES = {
    "TNX"        : "DGS10",
    "DXY"        : "DTWEXBGS",
    "CORP_SPREAD": "BAMLC0A0CM",
    "HY_SPREAD"  : "BAMLH0A0HYM2",
    "VIX"        : "VIXCLS",
    "T10Y2Y"     : "T10Y2Y",
    "TBILL_3M"   : "DTB3",
}

# ── Data ───────────────────────────────────────────────────────────────────────
SEED_START      = "2008-01-01"
VOL_WINDOW      = 21

# ── Wavelet (hardcoded — auto-selected by model) ───────────────────────────────
WAVELET         = "db4"
WAVELET_LEVELS  = 3
WAVELET_OPTIONS = ["db4", "db2", "haar", "sym5"]

# ── Model ──────────────────────────────────────────────────────────────────────
LOOKBACKS       = [30, 45, 60]       # auto-selected by lowest val MSE
DEFAULT_LOOKBACK= 30
TRAIN_SPLIT     = 0.80               # hardcoded 80/10/10
VAL_SPLIT       = 0.10
# TEST = remaining 0.10
MAX_EPOCHS      = 80
BATCH_SIZE      = 32
PATIENCE        = 10

# ── Risk Controls (defaults — overridden by UI sliders) ────────────────────────
DEFAULT_TSL_PCT   = 10
DEFAULT_Z_REENTRY = 1.1

# ── Start year range ───────────────────────────────────────────────────────────
START_YEAR_MIN  = 2008
START_YEAR_MAX  = 2024

# ── Paths ──────────────────────────────────────────────────────────────────────
MODELS_DIR      = "models"
DATA_DIR        = "data"
