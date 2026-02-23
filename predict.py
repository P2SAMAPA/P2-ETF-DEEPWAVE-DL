# predict.py
# Generates next-trading-day ETF signal from saved model weights.
# Loads latest data from HF or local, runs wavelet preprocessing,
# and returns predictions for all 3 models.
# Usage:
#   python predict.py
#   python predict.py --tsl 12 --z 1.2

import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

import config
from data_download import load_local
from preprocess import (build_features, apply_scaler,
                         load_scaler, run_preprocessing)
import model_a, model_b, model_c
from evaluate import softmax_probs, z_score


def load_latest_data() -> dict:
    """Try local first, then download incrementally."""
    data = load_local()
    if not data:
        print("No local data found — running incremental download...")
        from data_download import incremental_update
        data = incremental_update()
    return data


def get_best_lookbacks() -> dict:
    """Read best lookbacks from training summary."""
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


def predict_one(module, tag: str, data: dict,
                lookback: int, is_dual: bool) -> dict:
    """Run inference on the very last window of data."""
    try:
        m = module.load_model(lookback)
    except Exception as e:
        print(f"  [{tag}] Could not load model: {e}")
        return {}

    scaler   = load_scaler(lookback)
    features = build_features(data)

    # Take last `lookback` rows
    window = features.iloc[-lookback:].values          # (lookback, F)
    if len(window) < lookback:
        print(f"  [{tag}] Not enough data for lookback={lookback}")
        return {}

    N, F = 1, window.shape[1]
    X    = apply_scaler(window.reshape(1, lookback, F), scaler)  # (1, lb, F)

    if is_dual:
        n_etf = (len(config.ETFS) * 2) * (config.WAVELET_LEVELS + 1)
        inputs= [X[:, :, :n_etf], X[:, :, n_etf:]]
    else:
        inputs = X

    preds = m.predict(inputs, verbose=0)               # (1, 5)
    probs = softmax_probs(preds)                       # (1, 5)
    z     = float(z_score(probs)[0])
    top_i = int(np.argmax(probs[0]))
    etf   = config.ETFS[top_i]
    conf  = float(probs[0][top_i])

    prob_dict = {config.ETFS[i]: round(float(probs[0][i]), 4)
                 for i in range(len(config.ETFS))}

    return dict(
        model       = tag,
        lookback    = lookback,
        signal      = etf,
        confidence  = round(conf, 4),
        z_score     = round(z, 3),
        probabilities = prob_dict,
    )


def check_tsl_status(data: dict,
                     tsl_pct: float,
                     z_reentry: float,
                     current_z: float) -> dict:
    """
    Check if trailing stop loss is currently triggered.
    Uses last 2 days of ETF returns (most recently held ETF).
    """
    ret_df  = data["etf_ret"][config.ETFS]
    last2   = ret_df.iloc[-2:]
    # Use max-return ETF as proxy for what was held
    held_etf= last2.iloc[-1].idxmax()
    two_day = float(last2[held_etf].sum()) * 100

    triggered = two_day <= -tsl_pct
    can_reenter = current_z >= z_reentry

    return dict(
        two_day_cumul_pct = round(two_day, 2),
        tsl_triggered     = triggered,
        in_cash           = triggered and not can_reenter,
        current_z         = round(current_z, 3),
        z_reentry         = z_reentry,
        tsl_pct           = tsl_pct,
    )


def run_predict(tsl_pct: float   = config.DEFAULT_TSL_PCT,
                z_reentry: float = config.DEFAULT_Z_REENTRY) -> dict:

    print(f"\n{'='*60}")
    print(f"  Predict — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  TSL={tsl_pct}%  Z-reentry={z_reentry}σ")
    print(f"{'='*60}")

    data      = load_latest_data()
    lookbacks = get_best_lookbacks()
    last_date = data["etf_ret"].index.max().date()
    tbill_val = float(data["macro"]["TBILL_3M"].iloc[-1]) \
                if "TBILL_3M" in data["macro"].columns else 3.6

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

    # Determine winner model
    summary_path = os.path.join(config.MODELS_DIR, "training_summary.json")
    winner_model = "model_b"   # default
    if os.path.exists(summary_path):
        eval_path = "evaluation_results.json"
        if os.path.exists(eval_path):
            with open(eval_path) as f:
                ev = json.load(f)
            winner_model = ev.get("winner", "model_b")

    # TSL check using winner model Z
    current_z = predictions.get(winner_model, {}).get("z_score", 1.5)
    tsl_status = check_tsl_status(data, tsl_pct, z_reentry, current_z)

    # Final signal (winner model, unless TSL active → CASH)
    if tsl_status["in_cash"]:
        final_signal     = "CASH"
        final_confidence = None
        final_return     = tbill_val / 252  # daily T-bill accrual
    else:
        wp = predictions.get(winner_model, {})
        final_signal     = wp.get("signal", "—")
        final_confidence = wp.get("confidence")
        final_return     = None

    output = dict(
        as_of_date       = str(last_date),
        winner_model     = winner_model,
        final_signal     = final_signal,
        final_confidence = final_confidence,
        tsl_status       = tsl_status,
        tbill_rate       = tbill_val,
        predictions      = predictions,
    )

    # Save
    with open("latest_prediction.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Final signal: {final_signal}")
    print(f"  Saved → latest_prediction.json")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsl", type=float, default=config.DEFAULT_TSL_PCT)
    parser.add_argument("--z",   type=float, default=config.DEFAULT_Z_REENTRY)
    args = parser.parse_args()
    run_predict(tsl_pct=args.tsl, z_reentry=args.z)
