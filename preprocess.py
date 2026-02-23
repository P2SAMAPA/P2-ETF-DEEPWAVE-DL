# preprocess.py
# Wavelet decomposition (db4, 3-level) + feature engineering + sequence windowing
# Produces X (features) and y (next-day ETF returns) ready for model training.

import os
import numpy as np
import pandas as pd
import pywt
from sklearn.preprocessing import StandardScaler
import joblib

import config

os.makedirs(config.DATA_DIR, exist_ok=True)
os.makedirs(config.MODELS_DIR, exist_ok=True)


# ─── Wavelet decomposition ────────────────────────────────────────────────────

def wavelet_decompose(series: np.ndarray,
                      wavelet: str = config.WAVELET,
                      level: int   = config.WAVELET_LEVELS) -> np.ndarray:
    """
    Decompose 1-D series using DWT.
    Returns array of shape (len(series), level+1):
      col 0  = approximation coefficients (A_level), reconstructed to original length
      col 1..level = detail coefficients (D_level .. D_1), reconstructed
    """
    coeffs = pywt.wavedec(series, wavelet, level=level)
    reconstructed = []
    for i, c in enumerate(coeffs):
        # Reconstruct each sub-band back to original signal length
        zeros = [np.zeros_like(cc) for cc in coeffs]
        zeros[i] = c
        rec = pywt.waverec(zeros, wavelet)
        # Trim / pad to match original length
        rec = rec[:len(series)]
        if len(rec) < len(series):
            rec = np.pad(rec, (0, len(series) - len(rec)))
        reconstructed.append(rec)
    return np.stack(reconstructed, axis=1)   # (T, level+1)


def apply_wavelet_to_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply wavelet decomposition to every column in df.
    Returns expanded DataFrame with columns:
      original_col_A3, original_col_D1, original_col_D2, original_col_D3
    """
    parts = []
    for col in df.columns:
        decomp = wavelet_decompose(df[col].values)   # (T, 4)
        labels = [f"{col}_A3", f"{col}_D3", f"{col}_D2", f"{col}_D1"]
        part = pd.DataFrame(decomp, index=df.index, columns=labels)
        parts.append(part)
    return pd.concat(parts, axis=1)


# ─── Feature builder ─────────────────────────────────────────────────────────

def build_features(data: dict) -> pd.DataFrame:
    """
    Merge ETF returns, ETF vol, macro into one feature DataFrame.
    Apply wavelet decomposition to all columns.
    Returns combined wavelet-expanded feature matrix.
    """
    etf_ret   = data["etf_ret"]
    etf_vol   = data["etf_vol"]
    macro     = data["macro"]

    # Align on common dates
    combined = pd.concat([etf_ret, etf_vol, macro], axis=1).dropna()

    # Apply wavelet to all signals
    wavelet_features = apply_wavelet_to_df(combined)

    return wavelet_features


def build_targets(data: dict) -> pd.DataFrame:
    """
    Target = next-day log return for each ETF (5 outputs).
    Shift by -1 so row t predicts t+1.
    """
    targets = data["etf_ret"][config.ETFS].shift(-1)
    return targets


# ─── Sequence windowing ───────────────────────────────────────────────────────

def make_sequences(features: pd.DataFrame,
                   targets:  pd.DataFrame,
                   lookback: int) -> tuple:
    """
    Build sliding window sequences.
    X shape: (N, lookback, n_features)
    y shape: (N, n_etfs)
    dates:   (N,) — date of prediction target (t+1)
    """
    feat_arr = features.values
    tgt_arr  = targets.values

    # Align indices
    common = features.index.intersection(targets.dropna().index)
    features = features.loc[common]
    targets  = targets.loc[common].dropna()
    common   = features.index.intersection(targets.index)
    features = features.loc[common]
    targets  = targets.loc[common]

    feat_arr = features.values
    tgt_arr  = targets.values
    dates    = features.index

    X, y, d = [], [], []
    for i in range(lookback, len(feat_arr)):
        X.append(feat_arr[i - lookback : i])
        y.append(tgt_arr[i])
        d.append(dates[i])

    return np.array(X, dtype=np.float32), \
           np.array(y, dtype=np.float32), \
           np.array(d)


# ─── Train / Val / Test split ─────────────────────────────────────────────────

def split_data(X, y, dates,
               train_frac: float = config.TRAIN_SPLIT,
               val_frac:   float = config.VAL_SPLIT):
    N     = len(X)
    t_end = int(N * train_frac)
    v_end = int(N * (train_frac + val_frac))

    return (X[:t_end],  y[:t_end],  dates[:t_end],
            X[t_end:v_end], y[t_end:v_end], dates[t_end:v_end],
            X[v_end:],  y[v_end:],  dates[v_end:])


# ─── Scaler ───────────────────────────────────────────────────────────────────

def fit_scaler(X_train: np.ndarray, lookback: int) -> StandardScaler:
    """Fit StandardScaler on flattened training sequences."""
    N, L, F = X_train.shape
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, F))
    return scaler


def apply_scaler(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    N, L, F = X.shape
    return scaler.transform(X.reshape(-1, F)).reshape(N, L, F)


def save_scaler(scaler: StandardScaler, lookback: int):
    path = os.path.join(config.MODELS_DIR, f"scaler_lb{lookback}.pkl")
    joblib.dump(scaler, path)
    print(f"  Scaler saved → {path}")


def load_scaler(lookback: int) -> StandardScaler:
    path = os.path.join(config.MODELS_DIR, f"scaler_lb{lookback}.pkl")
    return joblib.load(path)


# ─── ETF / Macro feature splits (for Model C dual-stream) ─────────────────────

def split_streams(X: np.ndarray, n_etf_features: int) -> tuple:
    """
    Split X into ETF stream and Macro stream for Model C.
    n_etf_features = number of wavelet-expanded ETF columns
    """
    X_etf   = X[:, :, :n_etf_features]
    X_macro = X[:, :, n_etf_features:]
    return X_etf, X_macro


# ─── Full pipeline ─────────────────────────────────────────────────────────────

def run_preprocessing(data: dict, lookback: int) -> dict:
    """
    End-to-end preprocessing for a given lookback window.
    Returns dict with all splits + scaler + stream info.
    """
    print(f"\nPreprocessing with lookback={lookback}d ...")

    features = build_features(data)
    targets  = build_targets(data)

    X, y, dates = make_sequences(features, targets, lookback)

    (X_tr, y_tr, d_tr,
     X_va, y_va, d_va,
     X_te, y_te, d_te) = split_data(X, y, dates)

    scaler  = fit_scaler(X_tr, lookback)
    X_tr_sc = apply_scaler(X_tr, scaler)
    X_va_sc = apply_scaler(X_va, scaler)
    X_te_sc = apply_scaler(X_te, scaler)
    save_scaler(scaler, lookback)

    # Number of wavelet-expanded ETF columns
    # Each ETF signal → (ret + vol) → 2 raw cols × (WAVELET_LEVELS+1) wavelet bands
    n_etf_raw      = len(config.ETFS) * 2                           # ret + vol per ETF
    n_etf_features = n_etf_raw * (config.WAVELET_LEVELS + 1)

    print(f"  X shape : {X_tr_sc.shape}  |  y shape: {y_tr.shape}")
    print(f"  Features: {X_tr_sc.shape[2]} total  "
          f"({n_etf_features} ETF wavelet, "
          f"{X_tr_sc.shape[2]-n_etf_features} macro wavelet)")

    return dict(
        X_tr=X_tr_sc, y_tr=y_tr, d_tr=d_tr,
        X_va=X_va_sc, y_va=y_va, d_va=d_va,
        X_te=X_te_sc, y_te=y_te, d_te=d_te,
        scaler=scaler,
        n_features=X_tr_sc.shape[2],
        n_etf_features=n_etf_features,
        lookback=lookback,
        feature_names=list(build_features(data).columns),
    )


if __name__ == "__main__":
    from data_download import load_local
    data = load_local()
    if not data:
        print("No local data found. Run data_download.py first.")
    else:
        for lb in config.LOOKBACKS:
            result = run_preprocessing(data, lb)
            print(f"  Lookback {lb}d ready. "
                  f"Train={len(result['X_tr'])} "
                  f"Val={len(result['X_va'])} "
                  f"Test={len(result['X_te'])}")
