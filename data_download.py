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
    print(f"Fetching prices {start} → {end}")
    frames = []

    for ticker in tqdm(tickers, desc="Prices"):
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False
            )

            if df.empty:
                continue

            close = df[["Close"]].rename(columns={"Close": ticker})
            frames.append(close)

        except Exception as e:
            print(f"WARNING: {ticker} failed: {e}")

    if not frames:
        raise RuntimeError("No price data fetched.")

    prices = pd.concat(frames, axis=1)
    prices.index = pd.to_datetime(prices.index)
    prices.index = prices.index.tz_localize(None)
    prices.index.name = "Date"
    prices = prices.sort_index()

    return prices


# ─────────────────────────────────────────────────────────────
# DERIVED DATA
# ─────────────────────────────────────────────────────────────

def compute_returns(prices):
    returns = np.log(prices / prices.shift(1))
    returns = returns.dropna()
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
            s = fred.get_series(
                series_id,
                observation_start=start,
                observation_end=end
            )
            s.name = name
            frames[name] = s
        except Exception as e:
            print(f"WARNING: FRED {name} failed: {e}")

    macro = pd.DataFrame(frames)
    macro.index = pd.to_datetime(macro.index)
    macro.index = macro.index.tz_localize(None)
    macro.index.name = "Date"
    macro = macro.sort_index()
    macro = macro.ffill()

    return macro


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
        df.to_parquet(path)
        print(f"Saved {name}.parquet ({len(df)} rows)")


def load_prices_only():
    data = {}
    for name in ["etf_price", "bench_price"]:
        path = os.path.join(config.DATA_DIR, f"{name}.parquet")
        if os.path.exists(path):
            data[name] = pd.read_parquet(path)
    return data


# ─────────────────────────────────────────────────────────────
# FULL REBUILD LOGIC
# ─────────────────────────────────────────────────────────────

def build_full_dataset(start, end):
    print(f"\nBuilding dataset: {start} → {end}")

    etf_price   = fetch_prices(config.ETFS, start, end)
    bench_price = fetch_prices(config.BENCHMARKS, start, end)

    etf_ret   = compute_returns(etf_price)
    bench_ret = compute_returns(bench_price)

    etf_vol   = compute_volatility(etf_ret)
    bench_vol = compute_volatility(bench_ret)

    macro = fetch_macro(start, end)

    return {
        "etf_price": etf_price,
        "etf_ret": etf_ret,
        "etf_vol": etf_vol,
        "bench_price": bench_price,
        "bench_ret": bench_ret,
        "bench_vol": bench_vol,
        "macro": macro,
    }


def incremental_update():
    prices_existing = load_prices_only()

    if not prices_existing:
        print("No local data found — running seed.")
        return seed()

    last_date = prices_existing["etf_price"].index.max()
    start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
    end = datetime.today().strftime("%Y-%m-%d")

    if start < end:
        print(f"Fetching new prices {start} → {end}")
        new_etf = fetch_prices(config.ETFS, start, end)
        new_bench = fetch_prices(config.BENCHMARKS, start, end)

        etf_price = pd.concat([prices_existing["etf_price"], new_etf])
        bench_price = pd.concat([prices_existing["bench_price"], new_bench])

        etf_price = etf_price[~etf_price.index.duplicated(keep="last")]
        bench_price = bench_price[~bench_price.index.duplicated(keep="last")]
    else:
        print("Already up to date — recomputing derived datasets.")
        etf_price = prices_existing["etf_price"]
        bench_price = prices_existing["bench_price"]

    # 🚨 Always recompute from FULL price history
    etf_ret   = compute_returns(etf_price)
    bench_ret = compute_returns(bench_price)

    etf_vol   = compute_volatility(etf_ret)
    bench_vol = compute_volatility(bench_ret)

    macro = fetch_macro(config.SEED_START, end)

    data = {
        "etf_price": etf_price,
        "etf_ret": etf_ret,
        "etf_vol": etf_vol,
        "bench_price": bench_price,
        "bench_ret": bench_ret,
        "bench_vol": bench_vol,
        "macro": macro,
    }

    save_all(data)
    return data


def seed():
    end = datetime.today().strftime("%Y-%m-%d")
    data = build_full_dataset(config.SEED_START, end)
    save_all(data)
    return data


# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["seed", "incremental"],
                        default="incremental")
    args = parser.parse_args()

    if args.mode == "seed":
        seed()
    else:
        incremental_update()

    print("\nDataset build complete.")
