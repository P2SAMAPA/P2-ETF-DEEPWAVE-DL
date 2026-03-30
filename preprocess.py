# preprocess.py — Fixed Income ETFs
# Key change: wavelet is now a parameter to run_preprocessing() and build_features()
# so train.py can sweep all 4 wavelet options and pick the best.
# Scaler is saved with wavelet in filename to avoid collisions between sweeps.

import os
import numpy as np
import pandas as pd
import pywt
from sklearn.preprocessing import StandardScaler
import joblib

import config

os.makedirs(config.DATA_DIR,   exist_ok=True)
os.makedirs(config.MODELS_DIR, exist_ok=True)


# ─── Column normalisation ─────────────────────────────────────────────────────

def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns and clean up stringified tuple column names."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[1] if col[1] else col[0] for col in df.columns]
    cleaned = []
    for c in df.columns:
        c = str(c).strip()
        if c.startswith("('") and c.endswith("')"):
            try:
                parts = c.strip("()").replace("'", "").split(", ")
                c = parts[1] if len(parts) > 1 and parts[1] else parts[0]
            except Exception:
                pass
        cleaned.append(c.strip())
    df.columns = cleaned
    return df


def normalize_etf_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = flatten_columns(df)
    rename = {}
    for col in df.columns:
        for etf in config.FI_ETFS + config.BENCHMARKS:
            if col.startswith(etf):
                rename[col] = etf
    if rename:
        df = df.rename(columns=rename)
    df = df.loc[:, ~df.columns.duplicated()]
    return df


# ─── Wavelet decomposition ────────────────────────────────────────────────────

def wavelet_decompose_1d(series: np.ndarray,
                          wavelet: str = config.WAVELET,
                          level:   int = config.WAVELET_LEVELS) -> np.ndarray:
    series = np.array(series).flatten().astype(float)
    coeffs = pywt.wavedec(series, wavelet, level=level)
    reconstructed = []
    for i, c in enumerate(coeffs):
        zeros    = [np.zeros_like(cc) for cc in coeffs]
        zeros[i] = c
        rec      = pywt.waverec(zeros, wavelet)[:len(series)]
        if len(rec) < len(series):
            rec = np.pad(rec, (0, len(series) - len(rec)))
        reconstructed.append(rec)
    out = np.stack(reconstructed, axis=1)
    assert out.ndim == 2, f"wavelet shape error: {out.shape}"
    return out


def apply_wavelet_to_df(df: pd.DataFrame,
                         wavelet: str = config.WAVELET) -> pd.DataFrame:
    parts = []
    for col in df.columns:
        series = df[col].values.flatten().astype(float)
        decomp = wavelet_decompose_1d(series, wavelet=wavelet)
        labels = [f"{col}_A3", f"{col}_D3", f"{col}_D2", f"{col}_D1"]
        parts.append(pd.DataFrame(decomp, index=df.index, columns=labels))
    return pd.concat(parts, axis=1)


# ─── Feature builder ─────────────────────────────────────────────────────────

def build_features(data: dict,
                   wavelet: str = config.WAVELET) -> pd.DataFrame:
    etf_ret = normalize_etf_columns(data["etf_ret"].copy())
    etf_vol = normalize_etf_columns(data["etf_vol"].copy())
    macro   = flatten_columns(data["macro"].copy())

    # Use FI_ETFS only — equity tickers in the same parquet are ignored here
    etf_cols = [c for c in config.FI_ETFS if c in etf_ret.columns]
    if not etf_cols:
        print(f"  WARNING: No FI ETF cols found. Available: {list(etf_ret.columns)}")

    etf_ret = etf_ret[etf_cols] if etf_cols else etf_ret
    etf_vol = etf_vol[[c for c in config.FI_ETFS if c in etf_vol.columns]]
    etf_vol.columns = [f"{c}_vol" for c in etf_vol.columns]

    combined = pd.concat([etf_ret, etf_vol, macro], axis=1).dropna()
    return apply_wavelet_to_df(combined, wavelet=wavelet)


# ─── TARGET: CLASS LABEL ─────────────────────────────────────────────────────

def build_targets_classification(data: dict) -> pd.Series:
    """
    For each day, the target is the integer index of the FI ETF with the
    highest next-day log-return. Class 0–6 (len(FI_ETFS)-1).
    """
    ret      = normalize_etf_columns(data["etf_ret"].copy())
    etf_cols = [c for c in config.FI_ETFS if c in ret.columns]
    ret      = ret[etf_cols]
    fwd      = ret.shift(-1).dropna()
    labels   = fwd.values.argmax(axis=1)
    return pd.Series(labels, index=fwd.index, dtype=np.int32)


# ─── Sequence windowing ───────────────────────────────────────────────────────

def make_sequences(features: pd.DataFrame,
                   targets:  pd.Series,
                   lookback: int):
    common   = features.index.intersection(targets.index)
    features = features.loc[common]
    targets  = targets.loc[common]

    feat_arr = features.values.astype(np.float32)
    tgt_arr  = targets.values.astype(np.int32)
    dates    = features.index

    X, y, d = [], [], []
    for i in range(lookback, len(feat_arr)):
        X.append(feat_arr[i - lookback: i])
        y.append(tgt_arr[i])
        d.append(dates[i])

    return (np.array(X, dtype=np.float32),
            np.array(y, dtype=np.int32),
            np.array(d))


# ─── Split ────────────────────────────────────────────────────────────────────

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

def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    N, L, F = X_train.shape
    scaler  = StandardScaler()
    scaler.fit(X_train.reshape(-1, F))
    return scaler


def apply_scaler(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    N, L, F = X.shape
    return scaler.transform(X.reshape(-1, F)).reshape(N, L, F)


def save_scaler(scaler, lookback, wavelet):
    path = os.path.join(config.MODELS_DIR,
                        f"scaler_lb{lookback}_{wavelet}.pkl")
    joblib.dump(scaler, path)


def load_scaler(lookback, wavelet=None):
    """Load wavelet-specific scaler, fall back to legacy name."""
    if wavelet:
        path = os.path.join(config.MODELS_DIR,
                            f"scaler_lb{lookback}_{wavelet}.pkl")
        if os.path.exists(path):
            return joblib.load(path)
    # Legacy fallback (models trained before wavelet sweep)
    path = os.path.join(config.MODELS_DIR, f"scaler_lb{lookback}.pkl")
    return joblib.load(path)


# ─── Full pipeline ────────────────────────────────────────────────────────────

def run_preprocessing(data: dict, lookback: int,
                      wavelet: str = config.WAVELET) -> dict:
    print(f"\n[FI] Preprocessing lookback={lookback}d wavelet={wavelet} ...")

    features = build_features(data, wavelet=wavelet)
    targets  = build_targets_classification(data)

    assert len(targets) > 0, "No FI targets generated"
    print(f"  Class distribution: {pd.Series(targets).value_counts().to_dict()}")

    X, y, dates = make_sequences(features, targets, lookback)

    (X_tr, y_tr, d_tr,
     X_va, y_va, d_va,
     X_te, y_te, d_te) = split_data(X, y, dates)

    scaler  = fit_scaler(X_tr)
    X_tr_sc = apply_scaler(X_tr, scaler)
    X_va_sc = apply_scaler(X_va, scaler)
    X_te_sc = apply_scaler(X_te, scaler)
    save_scaler(scaler, lookback, wavelet)

    n_etf_raw      = len(config.FI_ETFS) * 2
    n_etf_features = n_etf_raw * (config.WAVELET_LEVELS + 1)

    print(f"  X={X_tr_sc.shape}  y={y_tr.shape}  "
          f"(classes 0-{len(config.FI_ETFS)-1})")
    print(f"  Train={len(X_tr)}  Val={len(X_va)}  Test={len(X_te)}")

    return dict(
        X_tr=X_tr_sc, y_tr=y_tr, d_tr=d_tr,
        X_va=X_va_sc, y_va=y_va, d_va=d_va,
        X_te=X_te_sc, y_te=y_te, d_te=d_te,
        scaler=scaler,
        n_features     = X_tr_sc.shape[2],
        n_etf_features = n_etf_features,
        lookback       = lookback,
        n_classes      = len(config.FI_ETFS),
        wavelet        = wavelet,
        feature_names  = list(features.columns),
    )
