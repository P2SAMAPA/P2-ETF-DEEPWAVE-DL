# predict.py
# Generates next-trading-day FI ETF signal from saved model weights.
# All risk params read from config.py — no CLI args for tsl/z.
# Wavelet read from training_summary.json (best_wavelet auto-selected during training).

import json
import os
import shutil
from datetime import datetime, date, timedelta

import numpy as np
import pandas as pd

import config
from data_utils import load_local
from preprocess import build_features, apply_scaler, load_scaler, \
                       flatten_columns, normalize_etf_columns
import model_a, model_b, model_c

TSL_PCT   = config.DEFAULT_TSL_PCT    # 12
Z_REENTRY = config.DEFAULT_Z_REENTRY  # 0.9

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


# ─── HF download helpers ──────────────────────────────────────────────────────

def download_weights_from_hf():
    try:
        from huggingface_hub import HfApi, hf_hub_download
        token = config.HF_TOKEN or None
        api   = HfApi(token=token)
        files = api.list_repo_files(repo_id=config.HF_DATASET_REPO,
                                    repo_type="dataset", token=token)
        for f in files:
            if f.endswith(('.keras','.pkl','.json')) and f.startswith('models/'):
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
        print(f"  WARNING: Could not download weights: {e}")


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


# ─── Training metadata ────────────────────────────────────────────────────────

def get_training_meta() -> dict:
    """Read per-model best_lookback and best_wavelet from training_summary.json."""
    summary_path = os.path.join(config.MODELS_DIR, "training_summary.json")
    defaults = {k: {"best_lookback": config.DEFAULT_LOOKBACK,
                    "best_wavelet":  config.WAVELET}
                for k in ["model_a","model_b","model_c"]}
    if not os.path.exists(summary_path):
        return defaults, None, None, None
    try:
        with open(summary_path) as f:
            s = json.load(f)
        for k in defaults:
            if k in s:
                defaults[k]["best_lookback"] = s[k].get("best_lookback",
                                                config.DEFAULT_LOOKBACK)
                defaults[k]["best_wavelet"]  = s[k].get("best_wavelet",
                                                config.WAVELET)
        return (defaults,
                s.get("start_year"),
                s.get("best_wavelet", config.WAVELET),
                s.get("trained_at"))
    except Exception as e:
        print(f"  Warning reading training summary: {e}")
        return defaults, None, config.WAVELET, None


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


# ─── Single model inference ───────────────────────────────────────────────────

def predict_one(module, tag: str, data: dict,
                lookback: int, wavelet: str, is_dual: bool) -> dict:
    try:
        m = module.load_model(lookback)
    except Exception as e:
        print(f"  [{tag}] Could not load model: {e}")
        return {}

    try:
        scaler   = load_scaler(lookback, wavelet=wavelet)
        features = build_features(data, wavelet=wavelet)
        window   = features.iloc[-lookback:].values
        if len(window) < lookback:
            print(f"  [{tag}] Not enough data for lookback={lookback}")
            return {}

        F = window.shape[1]
        X = apply_scaler(window.reshape(1, lookback, F), scaler)

        if is_dual:
            n_etf  = (len(config.FI_ETFS) * 2) * (config.WAVELET_LEVELS + 1)
            inputs = [X[:, :, :n_etf], X[:, :, n_etf:]]
        else:
            inputs = X

        preds = m.predict(inputs, verbose=0)
        probs = softmax_probs(preds)[0]
        z     = z_score_val(probs)
        top_i = int(np.argmax(probs))
        etf   = config.FI_ETFS[top_i]
        conf  = float(probs[top_i])

        prob_dict = {config.FI_ETFS[i]: round(float(probs[i]), 4)
                     for i in range(len(config.FI_ETFS))}

        return dict(model=tag, lookback=lookback, wavelet=wavelet,
                    signal=etf, confidence=round(conf, 4),
                    z_score=round(z, 3), probabilities=prob_dict)
    except Exception as e:
        print(f"  [{tag}] Inference error: {e}")
        return {}


# ─── TSL check ───────────────────────────────────────────────────────────────

def check_tsl_status(data, current_z):
    ret_df = data.get("etf_ret", pd.DataFrame())
    if ret_df.empty:
        return dict(two_day_cumul_pct=0, tsl_triggered=False, in_cash=False,
                    current_z=current_z, z_reentry=Z_REENTRY, tsl_pct=TSL_PCT)
    ret_df   = normalize_etf_columns(ret_df)
    etf_cols = [c for c in config.FI_ETFS if c in ret_df.columns]
    if not etf_cols:
        return dict(two_day_cumul_pct=0, tsl_triggered=False, in_cash=False,
                    current_z=current_z, z_reentry=Z_REENTRY, tsl_pct=TSL_PCT)
    last2     = ret_df[etf_cols].iloc[-2:]
    held_etf  = last2.iloc[-1].idxmax()
    two_day   = float(last2[held_etf].sum()) * 100
    triggered = two_day <= -TSL_PCT
    in_cash   = triggered and (current_z < Z_REENTRY)
    return dict(two_day_cumul_pct=round(two_day, 2), tsl_triggered=triggered,
                in_cash=in_cash, current_z=round(current_z, 3),
                z_reentry=Z_REENTRY, tsl_pct=TSL_PCT)


# ─── Main ────────────────────────────────────────────────────────────────────

def run_predict() -> dict:
    print(f"\n{'='*60}")
    print(f"  [FI] Predict — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  TSL={TSL_PCT}%  Z-reentry={Z_REENTRY}σ")
    print(f"{'='*60}")

    data = load_local()
    if not data:
        print("\n  No local data — downloading from HF Dataset...")
        download_data_from_hf()
        data = load_local()
    if not data:
        print("  ERROR: No data available.")
        return {}

    summary_path = os.path.join(config.MODELS_DIR, "training_summary.json")
    if not os.path.exists(summary_path):
        print("\n  No local weights — downloading from HF Dataset...")
        download_weights_from_hf()

    meta, trained_from_year, best_wavelet, trained_at = get_training_meta()

    # Signal date
    now_est  = datetime.utcnow() - timedelta(hours=5)
    today    = now_est.date()
    if today.weekday() < 5 and today not in US_HOLIDAYS and now_est.hour < 16:
        next_td = today
    else:
        next_td = next_trading_day(today)

    tbill_val = 3.6
    try:
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
        lb      = meta[tag]["best_lookback"]
        wavelet = meta[tag]["best_wavelet"]
        res     = predict_one(module, tag, data, lb, wavelet, is_dual)
        if res:
            predictions[tag] = res
            print(f"  [{tag.upper()}] {res['signal']} | "
                  f"conf={res['confidence']:.1%} | z={res['z_score']:.2f}σ | "
                  f"wavelet={wavelet}")
        else:
            print(f"  [{tag.upper()}] No prediction generated")

    # Winner from evaluation results
    winner_model = "model_a"
    if os.path.exists("evaluation_results.json"):
        with open("evaluation_results.json") as f:
            ev = json.load(f)
        winner_model = ev.get("winner", "model_a")

    current_z  = predictions.get(winner_model, {}).get("z_score", 0.0)
    tsl_status = check_tsl_status(data, current_z)

    if tsl_status["in_cash"]:
        final_signal, final_confidence = "CASH", None
    else:
        wp               = predictions.get(winner_model, {})
        final_signal     = wp.get("signal", "—")
        final_confidence = wp.get("confidence")

    output = dict(
        as_of_date        = str(next_td),
        winner_model      = winner_model,
        final_signal      = final_signal,
        final_confidence  = final_confidence,
        tsl_status        = tsl_status,
        tbill_rate        = tbill_val,
        predictions       = predictions,
        trained_from_year = trained_from_year,
        trained_wavelet   = best_wavelet,   # auto-selected during training
        trained_at        = trained_at,
    )

    with open("latest_prediction.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n  Next trading day : {next_td}")
    print(f"  Final signal     : {final_signal}")
    print(f"  Wavelet used     : {best_wavelet} (auto-selected)")
    print(f"  Saved → latest_prediction.json")
    return output


if __name__ == "__main__":
    # No CLI args — all params hardcoded in config.py
    run_predict()
