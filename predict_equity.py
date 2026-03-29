# predict_equity.py
# Generates next-trading-day equity ETF signal from saved equity model weights.
# Mirrors predict.py — uses preprocess_equity, equity weight paths
# (models/model_{a|b|c}_eq/lb{n}/best.keras), writes latest_prediction_equity.json.

import argparse
import json
import os
import shutil
from datetime import datetime, date, timedelta

import numpy as np
import pandas as pd

import config
from data_download import load_local
from preprocess_equity import build_features, apply_scaler, load_scaler, normalize_etf_columns

ETFS           = config.EQUITY_ETFS   # 13
OUTPUT_FILE    = "latest_prediction_equity.json"
EQUITY_N_CLASSES = len(ETFS)

US_HOLIDAYS = {
    date(2025,1,1),  date(2025,1,20),  date(2025,2,17), date(2025,4,18),
    date(2025,5,26), date(2025,6,19),  date(2025,7,4),  date(2025,9,1),
    date(2025,11,27),date(2025,12,25),
    date(2026,1,1),  date(2026,1,19),  date(2026,2,16), date(2026,4,3),
    date(2026,5,25), date(2026,6,19),  date(2026,7,3),  date(2026,9,7),
    date(2026,11,26),date(2026,12,25),
}


def next_trading_day(from_date=None):
    d = from_date or date.today()
    d += timedelta(days=1)
    while d.weekday() >= 5 or d in US_HOLIDAYS:
        d += timedelta(days=1)
    return d


# ─── Download helpers ─────────────────────────────────────────────────────────

def download_equity_weights_from_hf():
    try:
        from huggingface_hub import HfApi, hf_hub_download
        token = config.HF_TOKEN or None
        api   = HfApi(token=token)
        files = api.list_repo_files(repo_id=config.HF_DATASET_REPO,
                                    repo_type="dataset", token=token)
        for f in files:
            # Equity weight directories are named model_a_eq/, model_b_eq/, model_c_eq/
            if ("_eq" in f) and f.startswith("models/") and \
               f.endswith((".keras", ".pkl", ".json")):
                local = f
                os.makedirs(os.path.dirname(local), exist_ok=True)
                try:
                    dl = hf_hub_download(repo_id=config.HF_DATASET_REPO,
                                         filename=f, repo_type="dataset", token=token)
                    shutil.copy(dl, local)
                    print(f"    ✓ {f}")
                except Exception as e:
                    print(f"    ✗ {f}: {e}")
    except Exception as e:
        print(f"  WARNING: Could not download equity weights: {e}")


def download_data_from_hf():
    try:
        from huggingface_hub import hf_hub_download
        token = config.HF_TOKEN or None
        os.makedirs(config.DATA_DIR, exist_ok=True)
        for f in ["etf_price","etf_ret","etf_vol",
                  "bench_price","bench_ret","bench_vol","macro"]:
            try:
                dl = hf_hub_download(repo_id=config.HF_DATASET_REPO,
                                     filename=f"data/{f}.parquet",
                                     repo_type="dataset", token=token)
                shutil.copy(dl, os.path.join(config.DATA_DIR, f"{f}.parquet"))
                print(f"    ✓ data/{f}.parquet")
            except Exception as e:
                print(f"    ✗ {f}: {e}")
    except Exception as e:
        print(f"  WARNING: Could not download data: {e}")


# ─── Softmax + Z-score ───────────────────────────────────────────────────────

def softmax_probs(preds: np.ndarray) -> np.ndarray:
    preds    = np.array(preds)
    row_sums = preds.sum(axis=1)
    if np.allclose(row_sums, 1.0, atol=0.01):
        return np.clip(preds, 0, 1)
    scaled = preds / 0.1
    e      = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def z_score_val(probs: np.ndarray) -> float:
    return float((probs.max() - probs.mean()) / (probs.std() + 1e-8))


# ─── Best lookbacks ───────────────────────────────────────────────────────────

def get_best_lookbacks() -> dict:
    summary_path = os.path.join(config.MODELS_DIR, "training_summary_equity.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            s = json.load(f)
        return {k: s.get(k, {}).get("best_lookback", config.DEFAULT_LOOKBACK)
                for k in ["model_a","model_b","model_c"]}
    return {k: config.DEFAULT_LOOKBACK for k in ["model_a","model_b","model_c"]}


# ─── Load equity model ────────────────────────────────────────────────────────

def load_equity_model(tag: str, lookback: int):
    """Load best.keras from the equity model directory."""
    from tensorflow import keras
    # tag here is model_a / model_b / model_c
    short = tag.replace("model_", "")   # a / b / c
    path  = os.path.join(config.MODELS_DIR, f"model_{short}_eq",
                         f"lb{lookback}", "best.keras")
    return keras.models.load_model(path)


# ─── Single model inference ───────────────────────────────────────────────────

def predict_one(tag: str, data: dict, lookback: int) -> dict:
    is_dual = tag == "model_c"
    try:
        m = load_equity_model(tag, lookback)
    except Exception as e:
        print(f"  [EQ-{tag.upper()}] Could not load model: {e}")
        return {}

    try:
        scaler   = load_scaler(lookback)
        features = build_features(data)
        window   = features.iloc[-lookback:].values
        if len(window) < lookback:
            print(f"  [EQ-{tag.upper()}] Not enough data (need {lookback}, have {len(window)})")
            return {}

        n_features     = window.shape[1]
        n_etf_features = (len(ETFS) * 2) * (config.WAVELET_LEVELS + 1)

        X = apply_scaler(window.reshape(1, lookback, n_features), scaler)

        if is_dual:
            inputs = [X[:, :, :n_etf_features], X[:, :, n_etf_features:]]
        else:
            inputs = X

        preds = m.predict(inputs, verbose=0)
        probs = softmax_probs(preds)[0]
        z     = z_score_val(probs)
        top_i = int(np.argmax(probs))
        etf   = ETFS[top_i]
        conf  = float(probs[top_i])

        prob_dict = {ETFS[i]: round(float(probs[i]), 4) for i in range(len(ETFS))}

        return dict(model=tag, lookback=lookback, signal=etf,
                    confidence=round(conf, 4), z_score=round(z, 3),
                    probabilities=prob_dict)
    except Exception as e:
        print(f"  [EQ-{tag.upper()}] Inference error: {e}")
        return {}


# ─── TSL check ───────────────────────────────────────────────────────────────

def check_tsl_status(data, tsl_pct, z_reentry, current_z):
    ret_df   = data.get("etf_ret", pd.DataFrame())
    if ret_df.empty:
        return dict(two_day_cumul_pct=0, tsl_triggered=False, in_cash=False,
                    current_z=current_z, z_reentry=z_reentry, tsl_pct=tsl_pct)
    ret_df   = normalize_etf_columns(ret_df)
    etf_cols = [c for c in ETFS if c in ret_df.columns]
    if not etf_cols:
        return dict(two_day_cumul_pct=0, tsl_triggered=False, in_cash=False,
                    current_z=current_z, z_reentry=z_reentry, tsl_pct=tsl_pct)
    last2     = ret_df[etf_cols].iloc[-2:]
    held_etf  = last2.iloc[-1].idxmax()
    two_day   = float(last2[held_etf].sum()) * 100
    triggered = two_day <= -tsl_pct
    in_cash   = triggered and (current_z < z_reentry)
    return dict(two_day_cumul_pct=round(two_day, 2), tsl_triggered=triggered,
                in_cash=in_cash, current_z=round(current_z, 3),
                z_reentry=z_reentry, tsl_pct=tsl_pct)


# ─── Main ────────────────────────────────────────────────────────────────────

def run_predict(tsl_pct=config.DEFAULT_TSL_PCT,
                z_reentry=config.DEFAULT_Z_REENTRY) -> dict:

    print(f"\n{'='*60}")
    print(f"  [EQUITY] Predict — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    data = load_local()
    if not data:
        print("\n  No local data — downloading from HF Dataset...")
        download_data_from_hf()
        data = load_local()
    if not data:
        print("  ERROR: No data available.")
        return {}

    summary_path = os.path.join(config.MODELS_DIR, "training_summary_equity.json")
    if not os.path.exists(summary_path):
        print("\n  No equity weights — downloading from HF Dataset...")
        download_equity_weights_from_hf()

    lookbacks = get_best_lookbacks()

    # Signal date
    now_est  = datetime.utcnow() - timedelta(hours=5)
    today    = now_est.date()
    next_td  = today if (today.weekday() < 5 and today not in US_HOLIDAYS
                         and now_est.hour < 16) \
                     else next_trading_day(today)

    # T-bill rate from macro
    tbill_val = 3.6
    try:
        from preprocess_equity import flatten_columns
        macro = flatten_columns(data["macro"].copy())
        if "TBILL_3M" in macro.columns:
            tbill_val = float(macro["TBILL_3M"].iloc[-1])
    except Exception:
        pass

    # Run all 3 models
    predictions = {}
    for tag in ["model_a", "model_b", "model_c"]:
        lb  = lookbacks[tag]
        res = predict_one(tag, data, lb)
        if res:
            predictions[tag] = res
            print(f"  [EQ-{tag.upper()}] {res['signal']} | "
                  f"conf={res['confidence']:.1%} | z={res['z_score']:.2f}σ")
        else:
            print(f"  [EQ-{tag.upper()}] No prediction generated")

    # Winner from equity evaluation results
    winner_model = "model_a"
    eval_path    = "evaluation_results_equity.json"
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            ev = json.load(f)
        winner_model = ev.get("winner", "model_a")

    current_z  = predictions.get(winner_model, {}).get("z_score", 0.0)
    tsl_status = check_tsl_status(data, tsl_pct, z_reentry, current_z)

    if tsl_status["in_cash"]:
        final_signal, final_confidence = "CASH", None
    else:
        wp               = predictions.get(winner_model, {})
        final_signal     = wp.get("signal", "—")
        final_confidence = wp.get("confidence")

    # Training metadata
    trained_from_year = trained_wavelet = trained_at = None
    if os.path.exists(summary_path):
        with open(summary_path) as _f:
            _s = json.load(_f)
        trained_from_year = _s.get("start_year")
        trained_wavelet   = _s.get("wavelet") or "db4"
        trained_at        = _s.get("trained_at")

    output = dict(
        as_of_date        = str(next_td),
        winner_model      = winner_model,
        final_signal      = final_signal,
        final_confidence  = final_confidence,
        tsl_status        = tsl_status,
        tbill_rate        = tbill_val,
        predictions       = predictions,
        trained_from_year = trained_from_year,
        trained_wavelet   = trained_wavelet,
        trained_at        = trained_at,
        universe          = "equity",
        etfs              = ETFS,
    )

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n  Next trading day : {next_td}")
    print(f"  Final signal     : {final_signal}")
    print(f"  Saved → {OUTPUT_FILE}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsl", type=float, default=config.DEFAULT_TSL_PCT)
    parser.add_argument("--z",   type=float, default=config.DEFAULT_Z_REENTRY)
    args = parser.parse_args()
    run_predict(tsl_pct=args.tsl, z_reentry=args.z)
