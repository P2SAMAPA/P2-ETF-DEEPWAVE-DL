# preprocess.py
import os
import numpy as np
import pandas as pd
import pywt
from sklearn.preprocessing import StandardScaler
import joblib

import config

os.makedirs(config.DATA_DIR, exist_ok=True)
os.makedirs(config.MODELS_DIR, exist_ok=True)


# ─── Flatten + normalize columns ─────────────────────────────────────────────

def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns and strip ticker suffixes from yfinance."""
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance MultiIndex: level 0 = metric, level 1 = ticker
        # We want just the ticker names for price/ret/vol frames
        # e.g. ("Close", "TLT") → "TLT"
        df.columns = [col[1] if col[1] else col[0]
                      for col in df.columns]
    df.columns = [str(c).strip() for c in df.columns]
    return df


def normalize_etf_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure ETF columns match config.ETFS exactly.
    yfinance sometimes appends suffixes like TLT_Close → strip to TLT.
    """
    df = flatten_columns(df)
    rename = {}
    for col in df.columns:
        for etf in config.ETFS + config.BENCHMARKS:
            if col.startswith(etf):
                rename[col] = etf
    if rename:
        df = df.rename(columns=rename)
    # Drop duplicate columns keeping first
    df = df.loc[:, ~df.columns.duplicated()]
    return df


# ─── Wavelet decomposition ────────────────────────────────────────────────────

def wavelet_decompose_1d(series: np.ndarray,
                          wavelet: str = config.WAVELET,
                          level: int   = config.WAVELET_LEVELS) -> np.ndarray:
    """Decompose 1-D series. Returns (T, level+1) array."""
    series = np.array(series).flatten().astype(float)
    coeffs = pywt.wavedec(series, wavelet, level=level)
    reconstructed = []
    for i, c in enumerate(coeffs):
        zeros = [np.zeros_like(cc) for cc in coeffs]
        zeros[i] = c
        rec = pywt.waverec(zeros, wavelet)
        rec = rec[:len(series)]
        if len(rec) < len(series):
            rec = np.pad(rec, (0, len(series) - len(rec)))
        reconstructed.append(rec)
    out = np.stack(reconstructed, axis=1)
    assert out.ndim == 2, f"wavelet output shape error: {out.shape}"
    return out


def apply_wavelet_to_df(df: pd.DataFrame) -> pd.DataFrame:
    """Expand each column into wavelet sub-bands."""
    parts = []
    for col in df.columns:
        series = df[col].values.flatten().astype(float)
        decomp = wavelet_decompose_1d(series)
        labels = [f"{col}_A3", f"{col}_D3", f"{col}_D2", f"{col}_D1"]
        part   = pd.DataFrame(decomp, index=df.index, columns=labels)
        parts.append(part)
    return pd.concat(parts, axis=1)


# ─── Feature / target builders ───────────────────────────────────────────────

def build_features(data: dict) -> pd.DataFrame:
    etf_ret = normalize_etf_columns(data["etf_ret"].copy())
    etf_vol = normalize_etf_columns(data["etf_vol"].copy())
    macro   = flatten_columns(data["macro"].copy())

    # Keep only ETF columns
    etf_cols = [c for c in config.ETFS if c in etf_ret.columns]
    mac_cols  = [c for c in macro.columns]

    print(f"  ETF ret cols found : {etf_cols}")
    print(f"  Macro cols found   : {mac_cols}")

    if not etf_cols:
        print(f"  WARNING: No ETF cols matched! Available: {list(etf_ret.columns)}")

    etf_ret = etf_ret[etf_cols] if etf_cols else etf_ret
    etf_vol = etf_vol[[c for c in config.ETFS if c in etf_vol.columns]]
    etf_vol.columns = [f"{c}_vol" for c in etf_vol.columns]

    combined = pd.concat([etf_ret, etf_vol, macro], axis=1).dropna()
    print(f"  Combined shape before wavelet: {combined.shape}")

    wavelet_features = apply_wavelet_to_df(combined)
    print(f"  Wavelet features shape: {wavelet_features.shape}")
    return wavelet_features


def build_targets(data: dict) -> pd.DataFrame:
    tgt = normalize_etf_columns(data["etf_ret"].copy())
    etf_cols = [c for c in config.ETFS if c in tgt.columns]
    print(f"  Target ETF cols: {etf_cols}")
    if not etf_cols:
        print(f"  WARNING: No target cols! Available: {list(tgt.columns)}")
        # Try to use whatever columns exist
        etf_cols = list(tgt.columns)[:5]
    tgt = tgt[etf_cols]
    return tgt.shift(-1)


# ─── Sequence windowing ───────────────────────────────────────────────────────

def make_sequences(features, targets, lookback):
    common   = features.index.intersection(targets.dropna().index)
    features = features.loc[common]
    targets  = targets.loc[common].dropna()
    common   = features.index.intersection(targets.index)
    features = features.loc[common]
    targets  = targets.loc[common]

    feat_arr = features.values.astype(np.float32)
    tgt_arr  = targets.values.astype(np.float32)
    dates    = features.index

    X, y, d = [], [], []
    for i in range(lookback, len(feat_arr)):
        X.append(feat_arr[i - lookback : i])
        y.append(tgt_arr[i])
        d.append(dates[i])

    return (np.array(X, dtype=np.float32),
            np.array(y, dtype=np.float32),
            np.array(d))


# ─── Train / Val / Test split ─────────────────────────────────────────────────

def split_data(X, y, dates,
               train_frac=config.TRAIN_SPLIT,
               val_frac=config.VAL_SPLIT):
    N     = len(X)
    t_end = int(N * train_frac)
    v_end = int(N * (train_frac + val_frac))
    return (X[:t_end],  y[:t_end],  dates[:t_end],
            X[t_end:v_end], y[t_end:v_end], dates[t_end:v_end],
            X[v_end:],  y[v_end:],  dates[v_end:])


# ─── Scaler ───────────────────────────────────────────────────────────────────

def fit_scaler(X_train):
    N, L, F = X_train.shape
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, F))
    return scaler


def apply_scaler(X, scaler):
    N, L, F = X.shape
    return scaler.transform(X.reshape(-1, F)).reshape(N, L, F)


def save_scaler(scaler, lookback):
    path = os.path.join(config.MODELS_DIR, f"scaler_lb{lookback}.pkl")
    joblib.dump(scaler, path)
    print(f"  Scaler saved → {path}")


def load_scaler(lookback):
    path = os.path.join(config.MODELS_DIR, f"scaler_lb{lookback}.pkl")
    return joblib.load(path)


# ─── Full pipeline ────────────────────────────────────────────────────────────

def run_preprocessing(data: dict, lookback: int) -> dict:
    print(f"\nPreprocessing with lookback={lookback}d ...")

    features = build_features(data)
    targets  = build_targets(data)

    # Validate targets have correct shape
    assert targets.shape[1] > 0, \
        f"Targets have 0 columns! etf_ret cols: {list(data['etf_ret'].columns)}"
    assert targets.shape[1] == len(config.ETFS), \
        f"Expected {len(config.ETFS)} target cols, got {targets.shape[1]}: {list(targets.columns)}"

    X, y, dates = make_sequences(features, targets, lookback)

    (X_tr, y_tr, d_tr,
     X_va, y_va, d_va,
     X_te, y_te, d_te) = split_data(X, y, dates)

    scaler  = fit_scaler(X_tr)
    X_tr_sc = apply_scaler(X_tr, scaler)
    X_va_sc = apply_scaler(X_va, scaler)
    X_te_sc = apply_scaler(X_te, scaler)
    save_scaler(scaler, lookback)

    n_etf_raw      = len(config.ETFS) * 2
    n_etf_features = n_etf_raw * (config.WAVELET_LEVELS + 1)

    print(f"  X shape : {X_tr_sc.shape}  |  y shape: {y_tr.shape}")

    return dict(
        X_tr=X_tr_sc, y_tr=y_tr, d_tr=d_tr,
        X_va=X_va_sc, y_va=y_va, d_va=d_va,
        X_te=X_te_sc, y_te=y_te, d_te=d_te,
        scaler=scaler,
        n_features=X_tr_sc.shape[2],
        n_etf_features=n_etf_features,
        lookback=lookback,
        feature_names=list(features.columns),
    )
