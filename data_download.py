# data_download.py
# Downloads daily price, return, volatility for ETFs + benchmarks from Stooq
# (falls back to yfinance) and macro signals from FRED.
# Usage:
#   python data_download.py --mode seed        # 2008-01-01 → today
#   python data_download.py --mode incremental # last stored date → today

import argparse
import os
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pandas_datareader.data as web
import yfinance as yf
from fredapi import Fred
from tqdm import tqdm

import config

warnings.filterwarnings("ignore")

os.makedirs(config.DATA_DIR, exist_ok=True)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _stooq_pull(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Pull adjusted close from Stooq."""
    try:
        df = web.DataReader(ticker, "stooq", start=start, end=end)
        if df.empty:
            raise ValueError("empty")
        df = df.sort_index()
        return df[["Close"]].rename(columns={"Close": ticker})
    except Exception:
        return pd.DataFrame()


def _yfinance_pull(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fallback: pull adjusted close from yfinance."""
    try:
        df = yf.download(ticker, start=start, end=end,
                         auto_adjust=True, progress=False)
        if df.empty:
            raise ValueError("empty")
        return df[["Close"]].rename(columns={"Close": ticker})
    except Exception:
        return pd.DataFrame()


def fetch_prices(tickers: list, start: str, end: str) -> pd.DataFrame:
    """Fetch daily close prices for all tickers. Stooq → yfinance fallback."""
    frames = []
    for ticker in tqdm(tickers, desc="Fetching prices"):
        df = _stooq_pull(ticker, start, end)
        if df.empty:
            print(f"  Stooq failed for {ticker} — trying yfinance...")
            df = _yfinance_pull(ticker, start, end)
        if df.empty:
            print(f"  WARNING: No data for {ticker}")
        else:
            frames.append(df)
    if not frames:
        raise RuntimeError("No price data fetched.")
    prices = pd.concat(frames, axis=1)
    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "Date"
    return prices


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Log returns: ln(Pt / Pt-1)."""
    return np.log(prices / prices.shift(1)).dropna()


def compute_volatility(returns: pd.DataFrame,
                        window: int = config.VOL_WINDOW) -> pd.DataFrame:
    """Annualised rolling volatility."""
    return (returns.rolling(window).std() * np.sqrt(252)).dropna()


def fetch_macro(start: str, end: str) -> pd.DataFrame:
    """Fetch all macro signals from FRED."""
    fred = Fred(api_key=config.FRED_API_KEY)
    frames = {}
    for name, series_id in tqdm(config.MACRO_SERIES.items(), desc="Fetching macro"):
        try:
            s = fred.get_series(series_id, observation_start=start,
                                observation_end=end)
            s.name = name
            frames[name] = s
        except Exception as e:
            print(f"  WARNING: FRED failed for {name} ({series_id}): {e}")
    macro = pd.DataFrame(frames)
    macro.index = pd.to_datetime(macro.index)
    macro.index.name = "Date"
    # Forward-fill weekends / holidays (FRED publishes on business days)
    macro = macro.ffill().dropna(how="all")
    return macro


# ─── Main build ───────────────────────────────────────────────────────────────

def build_dataset(start: str, end: str) -> dict:
    """
    Returns dict with keys:
      etf_price, etf_ret, etf_vol,
      bench_price, bench_ret, bench_vol,
      macro
    All DataFrames indexed by Date.
    """
    print(f"\n{'='*60}")
    print(f"  Downloading data: {start} → {end}")
    print(f"{'='*60}")

    # ETFs
    etf_price = fetch_prices(config.ETFS, start, end)
    etf_ret   = compute_returns(etf_price)
    etf_vol   = compute_volatility(etf_ret)

    # Benchmarks
    bench_price = fetch_prices(config.BENCHMARKS, start, end)
    bench_ret   = compute_returns(bench_price)
    bench_vol   = compute_volatility(bench_ret)

    # Macro
    macro = fetch_macro(start, end)

    return dict(
        etf_price   = etf_price,
        etf_ret     = etf_ret,
        etf_vol     = etf_vol,
        bench_price = bench_price,
        bench_ret   = bench_ret,
        bench_vol   = bench_vol,
        macro       = macro,
    )


def save_local(data: dict):
    """Save all DataFrames to data/ as parquet."""
    os.makedirs(config.DATA_DIR, exist_ok=True)
    for name, df in data.items():
        path = os.path.join(config.DATA_DIR, f"{name}.parquet")
        df.to_parquet(path)
        print(f"  Saved {path}  ({len(df)} rows)")


def load_local() -> dict:
    """Load all parquet files from data/."""
    data = {}
    for name in ["etf_price","etf_ret","etf_vol",
                 "bench_price","bench_ret","bench_vol","macro"]:
        path = os.path.join(config.DATA_DIR, f"{name}.parquet")
        if os.path.exists(path):
            data[name] = pd.read_parquet(path)
    return data


def incremental_update():
    """Append only new trading days since last stored date."""
    existing = load_local()
    if not existing:
        print("No existing data — running full seed instead.")
        return seed()

    last_date = existing["etf_price"].index.max()
    start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
    end   = datetime.today().strftime("%Y-%m-%d")

    if start >= end:
        print(f"Already up to date through {last_date.date()}.")
        return existing

    print(f"Incremental update: {start} → {end}")
    new_data = build_dataset(start, end)

    merged = {}
    for key in existing:
        if key in new_data and not new_data[key].empty:
            combined = pd.concat([existing[key], new_data[key]])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined.sort_index(inplace=True)
            merged[key] = combined
        else:
            merged[key] = existing[key]

    save_local(merged)
    return merged


def seed():
    """Full seed from SEED_START to today."""
    end  = datetime.today().strftime("%Y-%m-%d")
    data = build_dataset(config.SEED_START, end)
    save_local(data)
    return data


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["seed", "incremental"],
                        default="incremental")
    args = parser.parse_args()

    if args.mode == "seed":
        seed()
    else:
        incremental_update()

    print("\nDone.")
