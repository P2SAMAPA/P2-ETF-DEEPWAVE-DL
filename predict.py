# predict.py
# Generates next-trading-day ETF signal from saved model weights.
# Downloads weights from HF Dataset if not available locally.

import argparse
import json
import os
import shutil
from datetime import datetime, date, timedelta

import numpy as np
import pandas as pd

import config
from data_download import load_local
from preprocess import build_features, apply_scaler, load_scaler, run_preprocessing
import model_a, model_b, model_c

US_HOLIDAYS = {
    date(2025,1,1), date(2025,1,20), date(2025,2,17), date(2025,4,18),
    date(2025,5,26), date(2025,6,19), date(2025,7,4), date(2025,9,1),
    date(2025,11,27), date(2025,12,25),
    date(2026,1,1), date(2026,1,19), date(2026,2,16), date(2026,4,3),
    date(2026,5,25), date(2026,6,19), date(2026,7,3), date(2026,9,7),
    date(2026,11,26), date(2026,12,25),
}

def next_trading_day(from_date=None):
    d = from_date or date.today()
    d += timedelta(days=1)
    while d.weekday() >= 5 or d in US_HOLIDAYS:
        d += timedelta(days=1)
    return d


# ─── Download weights from HF Dataset ────────────────────────────────────────

def download_weights_from_hf():
    """Pull all .keras and .pkl weight files from HF Dataset into local models/."""
    try:
        from huggingface_hub import HfApi, hf_hub_download, list_repo_tree
        token = config.HF_TOKEN or None
        print("  Downloading weights from HF Dataset...")

        # List all files in the dataset repo
        api = HfApi(token=token)
        files = api.list_repo_files(
            repo_id   = config.HF_DATASET_REPO,
            repo_type = "dataset",
            token     = token,
        )

        for f in files:
            if f.endswith(('.keras', '.pkl', '.json')) and \
               (f.startswith('models/') or f == 'models/training_summary.json'):
                local_path = f
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                try:
                    dl = hf_hub_download(
                        repo_id   = config.HF_DATASET_REPO,
                        filename  = f,
                        repo_type = "dataset",
                        token     = token,
                    )
                    shutil.copy(dl, local_path)
                    print(f"    ✓ {f}")
                except Exception as e:
                    print(f"    ✗ {f}: {e}")
        print("  Weights download complete.")
    except Exception as e:
        print(f"  WARNING: Could not download weights from HF: {e}")


def download_data_from_hf():
    """Pull parquet files from HF Dataset into local data/."""
    try:
        from huggingface_hub import hf_hub_download
        token = config.HF_TOKEN or None
        os.makedirs(config.DATA_DIR, exist_ok=True)
        files = ["etf_price","etf_ret","etf_vol",
                 "bench_price","bench_ret","bench_vol","macro"]
        for f in files:
            try:
                dl = hf_hub_download(
                    repo_id   = config.HF_DATASET_REPO,
                    filename  = f"data/{f}.parquet",
                    repo_type = "dataset",
                    token     = token,
                )
                shutil.copy(dl, os.path.join(config.DATA_DIR, f"{f}.parquet"))
                print(f"    ✓ data/{f}.parquet")
            except Exception as e:
                print(f"    ✗ data/{f}: {e}")
    except Exception as e:
        print(f"  WARNING: Could not download data from HF: {e}")


# ─── Softmax + Z-score ───────────────────────────────────────────────────────

def softmax_probs(preds: np.ndarray) -> np.ndarray:
    """Auto-detects if model output is already softmax (classification) or raw (regression)."""
    preds = np.array(preds)
    row_sums = preds.sum(axis=1)
    if np.allclose(row_sums, 1.0, atol=0.01):
        return np.clip(preds, 0, 1)   # already softmax probabilities
    # Legacy regression: apply temperature-scaled softmax
    scaled = preds / 0.1
    e = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def z_score_val(probs: np.ndarray) -> float:
    top   = probs.max()
    mu    = probs.mean()
    sigma = probs.std() + 1e-8
    return float((top - mu) / sigma)


# ─── Best lookbacks ───────────────────────────────────────────────────────────

def get_best_lookbacks() -> dict:
    summary_path = os.path.join(config.MODELS_DIR, "training_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            s = json.load(f)
        return {
            "model_a": s.get("model_a", {}).get("best_lookback", config.DEFAULT_LOOKBACK),
            "model_b": s.get("model_b", {}).get("best_lookback", config.DEFAULT_LOOKBACK),
            "model_c": s.get("model_c", {}).get("best_lookback", config.DEFAULT_LOOKBACK),
        }
    return {k: config.DEFAULT_LOOKBACK for k in ["model_a","model_b","model_c"]}


# ─── Single model inference ───────────────────────────────────────────────────

def predict_one(module, tag: str, data: dict,
                lookback: int, is_dual: bool) -> dict:
    try:
        m = module.load_model(lookback)
    except Exception as e:
        print(f"  [{tag}] Could not load model: {e}")
        return {}

    try:
        scaler   = load_scaler(lookback)
        features = build_features(data)
        window   = features.iloc[-lookback:].values
        if len(window) < lookback:
            print(f"  [{tag}] Not enough data for lookback={lookback}")
            return {}

        N, F = 1, window.shape[1]
        X    = apply_scaler(window.reshape(1, lookback, F), scaler)

        if is_dual:
            n_etf  = (len(config.ETFS) * 2) * (config.WAVELET_LEVELS + 1)
            inputs = [X[:, :, :n_etf], X[:, :, n_etf:]]
        else:
            inputs = X

        preds = m.predict(inputs, verbose=0)       # (1, 5)
        probs = softmax_probs(preds)[0]            # (5,)
        z     = z_score_val(probs)
        top_i = int(np.argmax(probs))
        etf   = config.ETFS[top_i]
        conf  = float(probs[top_i])

        prob_dict = {config.ETFS[i]: round(float(probs[i]), 4)
                     for i in range(len(config.ETFS))}

        return dict(
            model        = tag,
            lookback     = lookback,
            signal       = etf,
            confidence   = round(conf, 4),
            z_score      = round(z, 3),
            probabilities= prob_dict,
        )
    except Exception as e:
        print(f"  [{tag}] Inference error: {e}")
        return {}


# ─── TSL check ───────────────────────────────────────────────────────────────

def check_tsl_status(data, tsl_pct, z_reentry, current_z):
    ret_df  = data["etf_ret"][config.ETFS] if "etf_ret" in data else pd.DataFrame()
    if ret_df.empty:
        return dict(two_day_cumul_pct=0, tsl_triggered=False,
                    in_cash=False, current_z=current_z,
                    z_reentry=z_reentry, tsl_pct=tsl_pct)

    # Normalize columns
    from preprocess import normalize_etf_columns
    ret_df = normalize_etf_columns(ret_df)
    etf_cols = [c for c in config.ETFS if c in ret_df.columns]
    if not etf_cols:
        return dict(two_day_cumul_pct=0, tsl_triggered=False,
                    in_cash=False, current_z=current_z,
                    z_reentry=z_reentry, tsl_pct=tsl_pct)

    last2     = ret_df[etf_cols].iloc[-2:]
    held_etf  = last2.iloc[-1].idxmax()
    two_day   = float(last2[held_etf].sum()) * 100
    triggered = two_day <= -tsl_pct
    in_cash   = triggered and (current_z < z_reentry)

    return dict(
        two_day_cumul_pct = round(two_day, 2),
        tsl_triggered     = triggered,
        in_cash           = in_cash,
        current_z         = round(current_z, 3),
        z_reentry         = z_reentry,
        tsl_pct           = tsl_pct,
    )


# ─── Main ────────────────────────────────────────────────────────────────────

def run_predict(tsl_pct=config.DEFAULT_TSL_PCT,
                z_reentry=config.DEFAULT_Z_REENTRY) -> dict:

    print(f"\n{'='*60}")
    print(f"  Predict — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    # Download data + weights from HF if not present locally
    data = load_local()
    if not data:
        print("\n  No local data — downloading from HF Dataset...")
        download_data_from_hf()
        data = load_local()
    if not data:
        print("  ERROR: No data available.")
        return {}

    # Check if weights exist locally; if not, download
    summary_path = os.path.join(config.MODELS_DIR, "training_summary.json")
    has_weights  = os.path.exists(summary_path)
    if not has_weights:
        print("\n  No local weights — downloading from HF Dataset...")
        download_weights_from_hf()

    lookbacks = get_best_lookbacks()

    # Signal date: today if market not yet closed (< 4pm EST), else next trading day
    now_est  = datetime.utcnow() - timedelta(hours=5)
    today    = now_est.date()
    hour_est = now_est.hour
    if today.weekday() < 5 and today not in US_HOLIDAYS and hour_est < 16:
        next_td = today                    # pre-close: signal is FOR today
    else:
        next_td = next_trading_day(today)  # post-close: signal is for tomorrow

    tbill_val = 3.6
    try:
        from preprocess import normalize_etf_columns, flatten_columns
        macro = flatten_columns(data["macro"].copy())
        if "TBILL_3M" in macro.columns:
            tbill_val = float(macro["TBILL_3M"].iloc[-1])
    except Exception:
        pass

    predictions = {}
    for tag, module, is_dual in [
        ("model_a", model_a, False),
        ("model_b", model_b, False),
        ("model_c", model_c, True),
    ]:
        lb  = lookbacks[tag]
        res = predict_one(module, tag, data, lb, is_dual)
        if res:
            predictions[tag] = res
            print(f"  [{tag.upper()}] Signal={res['signal']}  "
                  f"Conf={res['confidence']:.1%}  Z={res['z_score']:.2f}σ")
        else:
            print(f"  [{tag.upper()}] No prediction generated")

    # Winner from evaluation results
    winner_model = "model_a"
    eval_path    = "evaluation_results.json"
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            ev = json.load(f)
        winner_model = ev.get("winner", "model_a")

    current_z  = predictions.get(winner_model, {}).get("z_score", 0.0)
    tsl_status = check_tsl_status(data, tsl_pct, z_reentry, current_z)

    if tsl_status["in_cash"]:
        final_signal     = "CASH"
        final_confidence = None
    else:
        wp               = predictions.get(winner_model, {})
        final_signal     = wp.get("signal", "—")
        final_confidence = wp.get("confidence")

    output = dict(
        as_of_date       = str(next_td),
        winner_model     = winner_model,
        final_signal     = final_signal,
        final_confidence = final_confidence,
        tsl_status       = tsl_status,
        tbill_rate       = tbill_val,
        predictions      = predictions,
    )

    with open("latest_prediction.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n  Next trading day : {next_td}")
    print(f"  Final signal     : {final_signal}")
    if predictions:
        for tag, p in predictions.items():
            print(f"  [{tag.upper()}] {p['signal']} | "
                  f"conf={p['confidence']:.1%} | z={p['z_score']:.2f}σ")
    print(f"  Saved → latest_prediction.json")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsl", type=float, default=config.DEFAULT_TSL_PCT)
    parser.add_argument("--z",   type=float, default=config.DEFAULT_Z_REENTRY)
    args = parser.parse_args()
    run_predict(tsl_pct=args.tsl, z_reentry=args.z)
