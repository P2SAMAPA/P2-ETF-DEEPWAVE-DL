# data_utils.py — DEEPWAVE-DL with HURST-style sequential download + Stooq fallback
import io
import json
import logging
import os
import random
import time
import re
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import yfinance as yf
from fredapi import Fred
from pandas.tseries.offsets import BDay

import config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# ─────────────────────────────────────────────────────────────
# PRICE FETCHING — Sequential with exponential backoff + Stooq fallback
# ─────────────────────────────────────────────────────────────

def fetch_prices(tickers, start, end):
    """
    Fetch adjusted close prices using HURST-style sequential download.
    Tries Yahoo Finance first with exponential backoff, falls back to Stooq.
    """
    logger.info(f"Fetching prices {start} -> {end} for {len(tickers)} tickers")
    
    frames = []
    failed = []
    
    for i, ticker in enumerate(tickers):
        logger.info(f"[{i+1}/{len(tickers)}] Fetching {ticker}...")
        
        # Try Yahoo Finance first
        df = _fetch_yf_single_price(ticker, start, end)
        
        # Fallback to Stooq if YF fails
        if df is None:
            logger.warning(f"🔄 YF failed for {ticker}, trying Stooq fallback...")
            df = _fetch_stooq_single_price(ticker, start, end)
        
        if df is not None:
            frames.append(df)
        else:
            failed.append(ticker)
            logger.error(f"❌ All sources failed for {ticker}")
        
        # Polite delay between tickers (1-2.5s like HURST)
        if i < len(tickers) - 1:
            delay = random.uniform(1.0, 2.5)
            time.sleep(delay)
    
    if not frames:
        logger.error("No price data fetched from any source.")
        return pd.DataFrame()
    
    if failed:
        logger.warning(f"⚠️ Failed tickers: {failed} — continuing with {len(frames)} tickers.")
    
    # Combine all tickers
    prices = pd.concat(frames, axis=1)
    prices.index = pd.to_datetime(prices.index)
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    prices.index.name = "Date"
    
    logger.info(f"Prices download complete. Shape: {prices.shape}")
    return prices.sort_index()


def _fetch_yf_single_price(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    """Fetch single ticker Close price from Yahoo Finance with exponential backoff."""
    for attempt in range(6):  # 0-5 attempts
        try:
            raw = yf.download(
                ticker,
                start=start,
                end=end,
                progress=False,
                auto_adjust=True,
                threads=False,
            )
            
            if raw is None or raw.empty:
                raise ValueError(f"Empty response for {ticker}")
            
            # Flatten MultiIndex columns if present
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = [col[0] for col in raw.columns]
            
            if "Close" not in raw.columns:
                raise ValueError(f"No Close column for {ticker}")
            
            close = raw[["Close"]].rename(columns={"Close": ticker})
            close.index = pd.to_datetime(close.index).tz_localize(None)
            
            logger.info(f"✅ {ticker} (YF): {len(close)} rows")
            return close
            
        except Exception as e:
            err_str = str(e).lower()
            is_rate_limit = any(k in err_str for k in ["rate limit", "too many", "429", "ratelimit"])
            
            if is_rate_limit and attempt < 5:
                wait = 30 * (2 ** attempt) + random.randint(5, 15)
                logger.warning(f"⚠️ YF rate limited on {ticker} (attempt {attempt+1}). Waiting {wait}s...")
                time.sleep(wait)
            else:
                logger.warning(f"❌ YF failed for {ticker} after {attempt+1} attempts: {e}")
                return None
    
    return None


def _fetch_stooq_single_price(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    """Fetch single ticker Close price from Stooq as fallback."""
    stooq_symbol = ticker.lower() + ".us"
    url = f"https://stooq.com/q/d/l/?s={stooq_symbol}&i=d"
    
    for attempt in range(3):
        try:
            raw = pd.read_csv(url, parse_dates=["Date"], index_col="Date")
            
            if raw.empty:
                raise ValueError(f"Empty Stooq response for {ticker}")
            
            raw = raw.sort_index()
            
            mask = (raw.index >= start) & (raw.index <= end)
            raw = raw.loc[mask]
            
            if raw.empty:
                raise ValueError(f"No data in range for {ticker} from Stooq")
            
            if "Close" not in raw.columns:
                raise ValueError(f"No Close column in Stooq data for {ticker}")
            
            close = raw[["Close"]].rename(columns={"Close": ticker})
            close.index = pd.to_datetime(close.index).tz_localize(None)
            
            logger.info(f"✅ {ticker} (Stooq): {len(close)} rows")
            return close
            
        except Exception as e:
            if attempt < 2:
                wait = 5 * (2 ** attempt) + random.randint(1, 5)
                logger.warning(f"⚠️ Stooq attempt {attempt+1} failed for {ticker}: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                logger.error(f"❌ Stooq failed for {ticker} after 3 attempts.")
                return None
    
    return None


# ─────────────────────────────────────────────────────────────
# DERIVED DATA
# ─────────────────────────────────────────────────────────────

def compute_returns(prices):
    """Compute log returns from prices."""
    returns = np.log(prices / prices.shift(1)).dropna()
    returns.index.name = "Date"
    logger.info(f"Returns computed. Shape: {returns.shape}")
    return returns


def compute_volatility(returns):
    """Compute annualized rolling volatility from returns."""
    vol = returns.rolling(config.VOL_WINDOW).std() * np.sqrt(252)
    vol = vol.dropna()
    vol.index.name = "Date"
    logger.info(f"Volatility computed. Shape: {vol.shape}")
    return vol


# ─────────────────────────────────────────────────────────────
# MACRO
# ─────────────────────────────────────────────────────────────

def fetch_macro(start, end):
    """Fetch FRED macro series."""
    fred = Fred(api_key=config.FRED_API_KEY)
    frames = {}

    for name, series_id in config.MACRO_SERIES.items():
        try:
            s = fred.get_series(series_id, observation_start=start, observation_end=end)
            s.name = name
            frames[name] = s
            logger.info(f"FRED {series_id} ({name}): {len(s)} observations")
        except Exception as e:
            logger.warning(f"FRED {name} failed: {e}")

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
    """Save all datasets to parquet files."""
    for name, df in data.items():
        path = os.path.join(config.DATA_DIR, f"{name}.parquet")
        os.makedirs(config.DATA_DIR, exist_ok=True)

        df_save = df.copy()

        if isinstance(df_save.columns, pd.MultiIndex):
            df_save.columns = [col[0] if col[0] != '' else col[1] for col in df_save.columns]
            df_save.columns = [str(c).strip() for c in df_save.columns]

        if df_save.index.name == "Date" or df_save.index.name is None:
            df_save = df_save.reset_index()

        if 'Date' in df_save.columns:
            df_save['Date'] = pd.to_datetime(df_save['Date'])
            if df_save['Date'].dt.tz is not None:
                df_save['Date'] = df_save['Date'].dt.tz_localize(None)

        df_save.to_parquet(path, index=False, engine='pyarrow')
        logger.info(f"Saved {name}.parquet ({len(df_save)} rows)")


def _ensure_datetime_index(df):
    """Ensure DataFrame has proper DatetimeIndex named 'Date'."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(str(c) for c in col if c != '').strip() for col in df.columns.values]

    date_col = None
    for c in df.columns:
        if isinstance(c, str) and c.lower() == 'date':
            date_col = c
            break
        if isinstance(c, tuple):
            flat_c = '_'.join(str(x) for x in c if x)
            if flat_c.lower() == 'date':
                date_col = c
                break

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
        df.index.name = "Date"
    else:
        if df.index.name is None:
            df.index.name = "Date"
        if df.index.dtype == 'int64' or str(df.index.dtype).startswith('int'):
            if df.index.max() > 1e12:
                df.index = pd.to_datetime(df.index, unit='ms')
            else:
                df.index = pd.to_datetime(df.index, unit='s')
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.index.name = "Date"

    for col in list(df.columns):
        if isinstance(col, str) and col.lower() in ('date', 'index', 'level_0'):
            df = df.drop(columns=[col])

    return df


def _clean_price_df(df):
    """Ensure all columns are numeric and no date columns remain."""
    for col in list(df.columns):
        if isinstance(col, str) and col.lower() in ('date', 'index', 'level_0'):
            df = df.drop(columns=[col])
    df = df.apply(pd.to_numeric, errors='coerce')
    return df


def load_prices_only():
    """Load only price datasets."""
    data = {}
    for name in ["etf_price", "bench_price"]:
        path = os.path.join(config.DATA_DIR, f"{name}.parquet")
        if os.path.exists(path):
            df = pd.read_parquet(path)
            df = _ensure_datetime_index(df)
            df = _clean_price_df(df)
            logger.info(f"Loaded {name}: {len(df)} rows, last date = {df.index.max()}")
            data[name] = df
    return data


def load_local():
    """Load all available datasets."""
    data = {}
    if not os.path.exists(config.DATA_DIR):
        return None

    for name in DATASETS:
        path = os.path.join(config.DATA_DIR, f"{name}.parquet")
        if os.path.exists(path):
            try:
                df = pd.read_parquet(path)
                df = _ensure_datetime_index(df)
                for col in list(df.columns):
                    if isinstance(col, str) and col.lower() in ('date', 'index', 'level_0'):
                        df = df.drop(columns=[col])
                data[name] = df
            except Exception as e:
                logger.warning(f"Could not load {name}: {e}")

    if not data:
        return None
    return data


# ─────────────────────────────────────────────────────────────
# FULL REBUILD LOGIC
# ─────────────────────────────────────────────────────────────

def build_full_dataset(start, end):
    """Build complete dataset from scratch."""
    logger.info(f"\nBuilding dataset: {start} -> {end}")
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
    """Incrementally update existing dataset."""

    # ── Guard: skip entirely if today is not a NYSE trading day ───────────────
    nyse = mcal.get_calendar("NYSE")
    today = pd.Timestamp.today().normalize()
    today_str = today.strftime("%Y-%m-%d")
    schedule_today = nyse.schedule(start_date=today_str, end_date=today_str)

    if schedule_today.empty:
        logger.info(f"[incremental_update] {today_str} is not a NYSE trading day — skipping fetch.")
        existing_full = load_local()
        return existing_full if existing_full is not None else {}

    # ── Load existing prices ───────────────────────────────────────────────────
    prices_existing = load_prices_only()

    if not prices_existing:
        logger.info("No local data found -- running seed.")
        return seed()

    last_date = prices_existing["etf_price"].index.max()

    # ── Use NYSE calendar to find confirmed new trading days ──────────────────
    search_start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
    schedule_range = nyse.schedule(start_date=search_start, end_date=today_str)
    new_trading_days = mcal.date_range(schedule_range, frequency="1D")

    if len(new_trading_days) == 0:
        logger.info(f"No new trading days since last update (last: {last_date.date()}). Skipping.")
        existing_full = load_local()
        if existing_full is None:
            etf_price = prices_existing["etf_price"]
            bench_price = prices_existing["bench_price"]
            data = {
                "etf_price": etf_price,
                "etf_ret": compute_returns(etf_price),
                "etf_vol": compute_volatility(compute_returns(etf_price)),
                "bench_price": bench_price,
                "bench_ret": compute_returns(bench_price),
                "bench_vol": compute_volatility(compute_returns(bench_price)),
                "macro": fetch_macro(config.SEED_START, today_str),
            }
            save_all(data)
            return data
        return existing_full

    # Fetch only confirmed trading days
    start_str = new_trading_days[0].strftime("%Y-%m-%d")
    end_str = new_trading_days[-1].strftime("%Y-%m-%d")

    logger.info(f"Fetching new prices {start_str} -> {end_str} ({len(new_trading_days)} trading day(s))")
    new_etf = fetch_prices(config.ETFS, start_str, end_str)
    new_bench = fetch_prices(config.BENCHMARKS, start_str, end_str)

    if new_etf.empty or new_bench.empty:
        logger.info("No new price data available. Skipping update.")
        existing_full = load_local()
        if existing_full is None:
            etf_price = prices_existing["etf_price"]
            bench_price = prices_existing["bench_price"]
            data = {
                "etf_price": etf_price,
                "etf_ret": compute_returns(etf_price),
                "etf_vol": compute_volatility(compute_returns(etf_price)),
                "bench_price": bench_price,
                "bench_ret": compute_returns(bench_price),
                "bench_vol": compute_volatility(compute_returns(bench_price)),
                "macro": fetch_macro(config.SEED_START, today_str),
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
        "macro": fetch_macro(config.SEED_START, end_str),
    }
    save_all(data)
    return data


def seed():
    """Full seed from 2008."""
    end = datetime.today().strftime("%Y-%m-%d")
    data = build_full_dataset(config.SEED_START, end)
    save_all(data)
    return data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["seed", "incremental"], default="incremental")
    args = parser.parse_args()

    if args.mode == "seed":
        seed()
    else:
        incremental_update()
    logger.info("\nDataset build complete.")
