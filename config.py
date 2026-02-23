# config.py — single source of truth for all constants
import os
from dotenv import load_dotenv
load_dotenv()

# ── Repos ─────────────────────────────────────────────────────────────────────
HF_DATASET_REPO = "P2SAMAPA/p2-etf-deepwave-dl"
HF_SPACE_REPO   = "P2SAMAPA/P2-ETF-DEEPWAVE-DL"

# ── API Keys (from env / GitHub Secrets) ──────────────────────────────────────
HF_TOKEN        = os.getenv("HF_TOKEN", "")
FRED_API_KEY    = os.getenv("FRED_API_KEY", "")

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
VOL_WINDOW      = 21              # rolling days for volatility

# ── Wavelet ────────────────────────────────────────────────────────────────────
WAVELET         = "db4"
WAVELET_LEVELS  = 3

# ── Model ──────────────────────────────────────────────────────────────────────
LOOKBACKS       = [30, 45, 60]
DEFAULT_LOOKBACK= 30
TRAIN_SPLIT     = 0.70
VAL_SPLIT       = 0.15           # remaining 0.15 = test
MAX_EPOCHS      = 80
BATCH_SIZE      = 32
PATIENCE        = 10             # early stopping

# ── Risk Controls (defaults — overridden by UI sliders) ────────────────────────
DEFAULT_TSL_PCT = 10             # trailing stop loss %
DEFAULT_Z_REENTRY = 1.1         # z-score re-entry threshold

# ── Paths ──────────────────────────────────────────────────────────────────────
MODELS_DIR      = "models"
DATA_DIR        = "data"
