# data_download.py
# Unified, deterministic dataset builder.
# Seed and incremental modes both:
#   - Update prices
#   - ALWAYS recompute returns & volatility
#   - ALWAYS overwrite all parquet files

import argparse
import os
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
from tqdm import tqdm

import config

warnings.filterwarnings("ignore")
os.makedirs(config.DATA_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# PRICE FETCHING
# ─────────────────────────────────────────────────────────────

def fetch_prices(tickers, start, end):
    """
    Fetch adjusted close prices with rate‑limit avoidance.
    Uses per‑ticker delays, exponential backoff, and no custom session.
    Returns empty DataFrame if no data available for the range.
    """
    print(f"Fetching prices {start} -> {end}")
    import time
    import random

    frames = []
    failed = []

    for ticker in tqdm(tickers, desc="Prices"):
        # Random delay between tickers (3–6 seconds)
        time.sleep(random.uniform(3.0, 6.0))

        success = False
        for attempt in range(3):  # up to 3 attempts
            try:
                tkr = yf.Ticker(ticker)
                df = tkr.history(start=start, end=end, auto_adjust=True)

                if df.empty:
                    # No data for this period (weekend/holiday) – not an error
                    print(f"{ticker}: no data for {start}–{end}")
                    break   # exit retry loop, no need to retry

                # Extract Close column and rename to ticker
                close = df[["Close"]].rename(columns={"Close": ticker})
                frames.append(close)
                success = True
                break

            except Exception as e:
                print(f"{ticker}: attempt {attempt+1} failed: {e}")
                if attempt < 2:
                    # Exponential backoff: 5s, 10s, 20s (+ jitter)
                    wait = (2 ** attempt) * 5 + random.uniform(0, 2)
                    print(f"  Retrying in {wait:.2f}s...")
                    time.sleep(wait)
                # If it's a rate limit, also sleep longer
                if "Rate limit" in str(e) or "Too Many Requests" in str(e):
                    cooldown = random.uniform(30, 60)
                    print(f"  Rate limited – sleeping {cooldown:.0f}s")
                    time.sleep(cooldown)

        if not success:
            failed.append(ticker)

    if failed:
        print(f"Warning: Failed to download tickers: {failed}")

    # If no frames were collected, return empty DataFrame
    if not frames:
        print("No price data fetched for the requested range.")
        return pd.DataFrame()

    prices = pd.concat(frames, axis=1)

    # Clean column names
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = [col[0] if col[0] != '' else col[1] for col in prices.columns]
    prices.columns = [str(c).strip() for c in prices.columns]

    prices.index = pd.to_datetime(prices.index)
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    prices.index.name = "Date"
    return prices.sort_index()


# ─────────────────────────────────────────────────────────────
# DERIVED DATA
# ─────────────────────────────────────────────────────────────

def compute_returns(prices):
    returns = np.log(prices / prices.shift(1)).dropna()
    returns.index.name = "Date"
    return returns


def compute_volatility(returns):
    vol = returns.rolling(config.VOL_WINDOW).std() * np.sqrt(252)
    vol = vol.dropna()
    vol.index.name = "Date"
    return vol


# ─────────────────────────────────────────────────────────────
# MACRO
# ─────────────────────────────────────────────────────────────

def fetch_macro(start, end):
    fred = Fred(api_key=config.FRED_API_KEY)
    frames = {}

    for name, series_id in tqdm(config.MACRO_SERIES.items(), desc="Macro"):
        try:
            s = fred.get_series(series_id, observation_start=start, observation_end=end)
            s.name = name
            frames[name] = s
        except Exception as e:
            print(f"WARNING: FRED {name} failed: {e}")

    macro = pd.DataFrame(frames)
    macro.index = pd.to_datetime(macro.index)
    if macro.index.tz is not None:
        macro.index = macro.index.tz_localize(None)
    macro.index.name = "Date"
    return macro.sort_index().ffill()


# ─────────────────────────────────────────────────────────────
# SAVE / LOAD
# ─────────────────────────────────────────────────────────────

DATASETS = [
    "etf_price",
    "etf_ret",
    "etf_vol",
    "bench_price",
    "bench_ret",
    "bench_vol",
    "macro",
]


def save_all(data):
    for name, df in data.items():
        path = os.path.join(config.DATA_DIR, f"{name}.parquet")

        # Create a copy to avoid modifying the original dataframe in memory
        df_save = df.copy()

        # Flatten MultiIndex columns before saving
        if isinstance(df_save.columns, pd.MultiIndex):
            df_save.columns = [col[0] if col[0] != '' else col[1] for col in df_save.columns]
        df_save.columns = [str(c).strip() for c in df_save.columns]

        # Reset index to make 'Date' a column so Parquet saves it as a column,
        # preventing HF from reading it as an integer index.
        if df_save.index.name == "Date" or df_save.index.name is None:
            df_save = df_save.reset_index()

        if 'Date' in df_save.columns:
            # Ensure datetime, strip timezone
            df_save['Date'] = pd.to_datetime(df_save['Date'])
            if df_save['Date'].dt.tz is not None:
                df_save['Date'] = df_save['Date'].dt.tz_localize(None)
        else:
            print(f"Warning: 'Date' column not found in {name} for saving.")

        # Save with index=False since Date is now a column
        df_save.to_parquet(path, index=False, engine='pyarrow')
        print(f"Saved {name}.parquet ({len(df_save)} rows)")


def _ensure_datetime_index(df):
    """Ensure the DataFrame has a proper DatetimeIndex named 'Date'.
    Handles loading from Parquet files where Date may be a datetime column,
    a date32 column, or (legacy) an int64 Unix timestamp index.
    """

    # 1. Flatten MultiIndex columns if present (common in older files or yfinance artifacts)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(str(c) for c in col if c != '').strip() for col in df.columns.values]

    # 2. Check if 'Date' is in columns (case-insensitive search)
    date_col = None
    for c in df.columns:
        if isinstance(c, str) and c.lower() == 'date':
            date_col = c
            break
        # Handle tuple column names if flattening wasn't perfect
        if isinstance(c, tuple):
            flat_c = '_'.join(str(x) for x in c if x)
            if flat_c.lower() == 'date':
                date_col = c
                break

    if date_col:
        # pd.to_datetime handles datetime64, date32 (Python date objects), and strings
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
        df.index.name = "Date"
    else:
        # 3. If Date is not in columns, check the index
        if df.index.name is None:
            df.index.name = "Date"  # Assume the index is the date

        # Convert Unix timestamp (ms) to datetime if read as int (legacy files)
        if df.index.dtype == 'int64' or str(df.index.dtype).startswith('int'):
            if df.index.max() > 1e12:  # milliseconds
                df.index = pd.to_datetime(df.index, unit='ms')
            else:  # seconds
                df.index = pd.to_datetime(df.index, unit='s')

        # Ensure it is actually datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

    # 4. Clean up timezone info
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    df.index.name = "Date"

    # 5. Drop any residual date-like columns that leaked in (e.g. from double reset_index)
    for col in list(df.columns):
        if isinstance(col, str) and col.lower() in ('date', 'index', 'level_0'):
            df = df.drop(columns=[col])

    return df


def _clean_price_df(df):
    """After loading, ensure all columns are numeric and no date columns remain."""
    # Drop any residual date/index columns
    for col in list(df.columns):
        if isinstance(col, str) and col.lower() in ('date', 'index', 'level_0'):
            df = df.drop(columns=[col])

    # Coerce all remaining columns to numeric, dropping any that fail entirely
    df = df.apply(pd.to_numeric, errors='coerce')

    return df


def load_prices_only():
    data = {}
    for name in ["etf_price", "bench_price"]:
        path = os.path.join(config.DATA_DIR, f"{name}.parquet")
        if os.path.exists(path):
            df = pd.read_parquet(path)
            df = _ensure_datetime_index(df)
            df = _clean_price_df(df)
            print(f"Loaded {name}: {len(df)} rows, last date = {df.index.max()}")
            data[name] = df
    return data


def load_local():
    """
    Loads all available dataset parquet files into a dictionary.
    Used by predict.py to load data for inference.
    Returns None if no data is found.
    """
    data = {}
    if not os.path.exists(config.DATA_DIR):
        return None

    for name in DATASETS:
        path = os.path.join(config.DATA_DIR, f"{name}.parquet")
        if os.path.exists(path):
            try:
                df = pd.read_parquet(path)
                df = _ensure_datetime_index(df)
                # Drop any residual date/index columns
                for col in list(df.columns):
                    if isinstance(col, str) and col.lower() in ('date', 'index', 'level_0'):
                        df = df.drop(columns=[col])
                data[name] = df
            except Exception as e:
                print(f"Warning: Could not load {name}: {e}")

    # If no data was loaded, return None to trigger download logic in predict.py
    if not data:
        return None

    return data


# ─────────────────────────────────────────────────────────────
# FULL REBUILD LOGIC
# ─────────────────────────────────────────────────────────────

def build_full_dataset(start, end):
    print(f"\nBuilding dataset: {start} -> {end}")
    etf_price = fetch_prices(config.ETFS, start, end)
    bench_price = fetch_prices(config.BENCHMARKS, start, end)
    return {
        "etf_price": etf_price,
        "etf_ret": compute_returns(etf_price),
        "etf_vol": compute_volatility(compute_returns(etf_price)),
        "bench_price": bench_price,
        "bench_ret": compute_returns(bench_price),
        "bench_vol": compute_volatility(compute_returns(bench_price)),
        "macro": fetch_macro(start, end),
    }


def incremental_update():
    prices_existing = load_prices_only()

    if not prices_existing:
        print("No local data found -- running seed.")
        return seed()

    # Get last date from existing price data
    last_date = prices_existing["etf_price"].index.max()
    today = datetime.today().normalize()

    # Use business day offset to skip weekends
    from pandas.tseries.offsets import BDay
    start = last_date + BDay(1)

    # If start is beyond today (e.g., weekend), there are no new trading days
    if start > today:
        print(f"No new trading days since last update (last date: {last_date.date()}). Skipping update.")
        # Return existing data as is (derived data already present)
        # Note: We still need to recompute derived data? Actually the function recomputes returns/volatility
        # from the full etf_price and bench_price. So if there's no new data, we can just use the existing data
        # and recompute derived data from it. But since we're not fetching new data, we should just return the
        # existing data (including derived data). However, the caller expects data dict with all keys.
        # We'll load all existing data via load_local() and return that.
        existing_full = load_local()
        if existing_full is None:
            # Fallback: load prices and recompute derived
            etf_price = prices_existing["etf_price"]
            bench_price = prices_existing["bench_price"]
            data = {
                "etf_price": etf_price,
                "etf_ret": compute_returns(etf_price),
                "etf_vol": compute_volatility(compute_returns(etf_price)),
                "bench_price": bench_price,
                "bench_ret": compute_returns(bench_price),
                "bench_vol": compute_volatility(compute_returns(bench_price)),
                "macro": fetch_macro(config.SEED_START, today.strftime("%Y-%m-%d")),
            }
            save_all(data)
            return data
        return existing_full

    end = today.strftime("%Y-%m-%d")
    start_str = start.strftime("%Y-%m-%d")

    print(f"Fetching new prices {start_str} -> {end}")
    new_etf = fetch_prices(config.ETFS, start_str, end)
    new_bench = fetch_prices(config.BENCHMARKS, start_str, end)

    # If either new DataFrame is empty, skip update (no new data)
    if new_etf.empty or new_bench.empty:
        print("No new price data available. Skipping update.")
        # Return existing data as is
        existing_full = load_local()
        if existing_full is None:
            # Fallback: use existing prices and recompute derived
            etf_price = prices_existing["etf_price"]
            bench_price = prices_existing["bench_price"]
            data = {
                "etf_price": etf_price,
                "etf_ret": compute_returns(etf_price),
                "etf_vol": compute_volatility(compute_returns(etf_price)),
                "bench_price": bench_price,
                "bench_ret": compute_returns(bench_price),
                "bench_vol": compute_volatility(compute_returns(bench_price)),
                "macro": fetch_macro(config.SEED_START, today.strftime("%Y-%m-%d")),
            }
            save_all(data)
            return data
        return existing_full

    # Concatenate and drop duplicates
    etf_price = pd.concat([prices_existing["etf_price"], new_etf])
    bench_price = pd.concat([prices_existing["bench_price"], new_bench])

    etf_price = etf_price[~etf_price.index.duplicated(keep="last")]
    bench_price = bench_price[~bench_price.index.duplicated(keep="last")]

    data = {
        "etf_price": etf_price,
        "etf_ret": compute_returns(etf_price),
        "etf_vol": compute_volatility(compute_returns(etf_price)),
        "bench_price": bench_price,
        "bench_ret": compute_returns(bench_price),
        "bench_vol": compute_volatility(compute_returns(bench_price)),
        "macro": fetch_macro(config.SEED_START, end),
    }
    save_all(data)
    return data


def seed():
    end = datetime.today().strftime("%Y-%m-%d")
    data = build_full_dataset(config.SEED_START, end)
    save_all(data)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["seed", "incremental"], default="incremental")
    args = parser.parse_args()

    if args.mode == "seed":
        seed()
    else:
        incremental_update()
    print("\nDataset build complete.")
