# config.py — single source of truth for all constants
import os
from dotenv import load_dotenv
load_dotenv()

# ── Repos ─────────────────────────────────────────────────────────────────────
HF_DATASET_REPO = "P2SAMAPA/p2-etf-deepwave-dl"
HF_SPACE_REPO   = "P2SAMAPA/P2-ETF-DEEPWAVE-DL"
GITHUB_REPO     = "P2SAMAPA/P2-ETF-DEEPWAVE-DL"

# ── API Keys (loaded from env / Streamlit secrets at runtime) ─────────────────
HF_TOKEN     = os.getenv("HF_TOKEN", "")
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
GITHUB_TOKEN = os.getenv("P2SAMAPA_GITHUB_TOKEN", os.getenv("GITHUB_TOKEN", ""))

# ── Universe ───────────────────────────────────────────────────────────────────
FI_ETFS = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]

EQUITY_ETFS = [
    "QQQ",   # NASDAQ 100
    "XLK",   # Technology
    "XLF",   # Financials
    "XLE",   # Energy
    "XLV",   # Health Care
    "XLI",   # Industrials
    "XLY",   # Consumer Disc
    "XLP",   # Consumer Staples
    "XLU",   # Utilities
    "XME",   # Metal and Mining
    "GDX",   # Gold Miners
    "IWM",   # Russell 2000 Small Cap
]

ETFS        = FI_ETFS + EQUITY_ETFS   # all tradeable tickers → etf_price.parquet
BENCHMARKS  = ["SPY", "AGG"]          # never traded, comparison only
ALL_TICKERS = ETFS + BENCHMARKS

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
SEED_START = "2008-01-01"
VOL_WINDOW = 21

# ── Wavelet — auto-selected during training ────────────────────────────────────
# All 4 options are tried per lookback; best validation return is kept.
# The winner is stamped into training_summary.json and shown in the UI.
WAVELET         = "db4"               # fallback if summary not yet available
WAVELET_LEVELS  = 3
WAVELET_OPTIONS = ["db4", "db2", "haar", "sym5"]

# ── Model ──────────────────────────────────────────────────────────────────────
LOOKBACKS        = [30, 45, 60]
DEFAULT_LOOKBACK = 30
TRAIN_SPLIT      = 0.80
VAL_SPLIT        = 0.10
MAX_EPOCHS       = 80                 # hardcoded
BATCH_SIZE       = 32
PATIENCE         = 10

# ── Risk Controls — hardcoded, displayed in sidebar as info only ───────────────
DEFAULT_TSL_PCT   = 12               # trailing stop loss %
DEFAULT_Z_REENTRY = 0.9             # z-score re-entry threshold
FEE_BPS           = 12              # transaction cost in basis points

# ── Start year range ───────────────────────────────────────────────────────────
START_YEAR_MIN = 2008
START_YEAR_MAX = 2024

# ── Paths ──────────────────────────────────────────────────────────────────────
MODELS_DIR = "models"
DATA_DIR   = "data"
