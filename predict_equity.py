# predict_equity.py
# Generates next-trading-day equity ETF signal from saved equity model weights.
# All risk params read from config.py — no CLI args for tsl/z.
# Wavelet read from training_summary_equity.json (best_wavelet auto-selected).
# Model path: models/model_{a|b|c}_eq/lb{n}_{wavelet}/best.keras

import json
import os
import shutil
from datetime import datetime, date, timedelta

import numpy as np
import pandas as pd

import config
from data_download import load_local
from preprocess_equity import build_features, apply_scaler, load_scaler, \
                              normalize_etf_columns, flatten_columns

ETFS      = config.EQUITY_ETFS
OUTPUT    = "latest_prediction_equity.json"
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


def download_equity_weights_from_hf():
    try:
        from huggingface_hub import HfApi, hf_hub_download
        token = config.HF_TOKEN or None
        api   = HfApi(token=token)
        files = api.list_repo_files(repo_id=config.HF_DATASET_REPO,
                                    repo_type="dataset", token=token)
        for f in files:
            if ("_eq" in f) and f.startswith("models/") and \
               f.endswith((".keras",".pkl",".json")):
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
                print(f"    ✓ {f}.parquet")
            except Exception as e:
                print(f"    ✗ {f}: {e}")
    except Exception as e:
        print(f"  WARNING: Could not download data: {e}")


def get_training_meta() -> tuple:
    summary_path = os.path.join(config.MODELS_DIR, "training_summary_equity.json")
    defaults = {k: {"best_lookback": config.DEFAULT_LOOKBACK,
                    "best_wavelet":  config.WAVELET}
                for k in ["model_a","model_b","model_c"]}
    if not os.path.exists(summary_path):
        return defaults, None, config.WAVELET, None
    try:
        with open(summary_path) as f:
            s = json.load(f)
        for k in defaults:
            if k in s:
                defaults[k]["best_lookback"] = s[k].get("best_lookback", config.DEFAULT_LOOKBACK)
                defaults[k]["best_wavelet"]  = s[k].get("best_wavelet",  config.WAVELET)
        return (defaults, s.get("start_year"),
                s.get("best_wavelet", config.WAVELET), s.get("trained_at"))
    except Exception as e:
        print(f"  Warning reading equity training summary: {e}")
        return defaults, None, config.WAVELET, None


def load_equity_model(tag: str, lookback: int, wavelet: str):
    from tensorflow import keras
    short = tag.replace("model_", "")
    # Wavelet-namespaced path (new)
    path = os.path.join(config.MODELS_DIR, f"model_{short}_eq",
                        f"lb{lookback}_{wavelet}", "best.keras")
    if not os.path.exists(path):
        # Fallback for models trained before wavelet sweep update
        fallback = os.path.join(config.MODELS_DIR, f"model_{short}_eq",
                                f"lb{lookback}", "best.keras")
        if os.path.exists(fallback):
            print(f"  [{tag}] Using legacy path: {fallback}")
            path = fallback
    return keras.models.load_model(path)


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


def predict_one(tag: str, data: dict, lookback: int, wavelet: str) -> dict:
    is_dual = (tag == "model_c")
    try:
        m = load_equity_model(tag, lookback, wavelet)
    except Exception as e:
        print(f"  [EQ-{tag.upper()}] Could not load model: {e}")
        return {}
    try:
        scaler         = load_scaler(lookback, wavelet=wavelet)
        features       = build_features(data, wavelet=wavelet)
        window         = features.iloc[-lookback:].values
        if len(window) < lookback:
            return {}
        n_features     = window.shape[1]
        n_etf_features = (len(ETFS) * 2) * (config.WAVELET_LEVELS + 1)
        X = apply_scaler(window.reshape(1, lookback, n_features), scaler)
        inputs = [X[:, :, :n_etf_features], X[:, :, n_etf_features:]] \
                 if is_dual else X
        preds = m.predict(inputs, verbose=0)
        probs = softmax_probs(preds)[0]
        z     = z_score_val(probs)
        top_i = int(np.argmax(probs))
        return dict(model=tag, lookback=lookback, wavelet=wavelet,
                    signal=ETFS[top_i], confidence=round(float(probs[top_i]), 4),
                    z_score=round(z, 3),
                    probabilities={ETFS[i]: round(float(probs[i]), 4)
                                   for i in range(len(ETFS))})
    except Exception as e:
        print(f"  [EQ-{tag.upper()}] Inference error: {e}")
        return {}


def check_tsl_status(data, current_z):
    ret_df = data.get("etf_ret", pd.DataFrame())
    if ret_df.empty:
        return dict(two_day_cumul_pct=0, tsl_triggered=False, in_cash=False,
                    current_z=current_z, z_reentry=Z_REENTRY, tsl_pct=TSL_PCT)
    ret_df   = normalize_etf_columns(ret_df)
    etf_cols = [c for c in ETFS if c in ret_df.columns]
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


def run_predict() -> dict:
    print(f"\n{'='*60}")
    print(f"  [EQUITY] Predict — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  TSL={TSL_PCT}%  Z-reentry={Z_REENTRY}σ")
    print(f"{'='*60}")

    data = load_local()
    if not data:
        print("\n  No local data — downloading from HF Dataset...")
        download_data_from_hf()
        data = load_local()
    if not data:
        print("  ERROR: No data available."); return {}

    summary_path = os.path.join(config.MODELS_DIR, "training_summary_equity.json")
    if not os.path.exists(summary_path):
        print("\n  No equity weights — downloading from HF Dataset...")
        download_equity_weights_from_hf()

    meta, trained_from_year, best_wavelet, trained_at = get_training_meta()

    now_est = datetime.utcnow() - timedelta(hours=5)
    today   = now_est.date()
    next_td = today if (today.weekday() < 5 and today not in US_HOLIDAYS
                        and now_est.hour < 16) \
                    else next_trading_day(today)

    tbill_val = 3.6
    try:
        macro = flatten_columns(data["macro"].copy())
        if "TBILL_3M" in macro.columns:
            tbill_val = float(macro["TBILL_3M"].iloc[-1])
    except Exception:
        pass

    predictions = {}
    for tag in ["model_a","model_b","model_c"]:
        lb      = meta[tag]["best_lookback"]
        wavelet = meta[tag]["best_wavelet"]
        res     = predict_one(tag, data, lb, wavelet)
        if res:
            predictions[tag] = res
            print(f"  [EQ-{tag.upper()}] {res['signal']} | "
                  f"conf={res['confidence']:.1%} | z={res['z_score']:.2f}σ | "
                  f"wavelet={wavelet}")

    winner_model = "model_a"
    if os.path.exists("evaluation_results_equity.json"):
        with open("evaluation_results_equity.json") as f:
            winner_model = json.load(f).get("winner", "model_a")

    current_z  = predictions.get(winner_model, {}).get("z_score", 0.0)
    tsl_status = check_tsl_status(data, current_z)

    if tsl_status["in_cash"]:
        final_signal, final_confidence = "CASH", None
    else:
        wp = predictions.get(winner_model, {})
        final_signal, final_confidence = wp.get("signal","—"), wp.get("confidence")

    output = dict(
        as_of_date=str(next_td), winner_model=winner_model,
        final_signal=final_signal, final_confidence=final_confidence,
        tsl_status=tsl_status, tbill_rate=tbill_val,
        predictions=predictions, trained_from_year=trained_from_year,
        trained_wavelet=best_wavelet, trained_at=trained_at,
        universe="equity", etfs=ETFS,
    )

    with open(OUTPUT, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n  Signal: {final_signal}  |  Wavelet: {best_wavelet} (auto-selected)")
    print(f"  Saved → {OUTPUT}")
    return output


if __name__ == "__main__":
    # No CLI args — all params hardcoded in config.py
    run_predict()
