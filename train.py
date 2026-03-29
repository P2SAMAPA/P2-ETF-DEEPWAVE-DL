# train.py
# Trains Option A, B, C for FIXED INCOME ETFs across all lookback windows.
# Wavelet auto-selection: tries all 4 options × all lookbacks, picks best val_acc.
# Risk params hardcoded: FEE_BPS=12, TSL=12%, Z_REENTRY=0.9, MAX_EPOCHS=80.
#
# Usage:
#   python train.py --model all
#   python train.py --model a --epochs 50

import argparse
import json
import os
from datetime import datetime

import numpy as np

import config
import model_a
import model_b
import model_c
from data_download import load_local
from preprocess import run_preprocessing   # FI preprocessor (uses config.ETFS = FI_ETFS)

os.makedirs(config.MODELS_DIR, exist_ok=True)


# ─── Per-model + wavelet trainer ─────────────────────────────────────────────

def train_one_wavelet(module, tag: str, prep: dict, epochs: int) -> dict:
    """Train one model on one (lookback, wavelet) combination."""
    model, history = module.train(prep, epochs=epochs)
    val_loss = min(history.history["val_loss"])
    val_acc  = max(history.history.get("val_accuracy", [0]))
    wavelet  = prep.get("wavelet", config.WAVELET)
    print(f"  [FI-{tag}] lb={prep['lookback']} wavelet={wavelet}  "
          f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")
    return {"val_loss": val_loss, "val_acc": val_acc,
            "lookback": prep["lookback"], "wavelet": wavelet, "model": tag}


# ─── Sweep for one model ──────────────────────────────────────────────────────

def sweep_model(module, tag: str, data: dict, epochs: int) -> dict:
    label = {"a": "Wavelet-CNN-LSTM",
             "b": "Wavelet-Attention-CNN-LSTM",
             "c": "Wavelet-Parallel-Dual-Stream-CNN-LSTM"}[tag]
    print(f"\n{'─'*56}")
    print(f"  [FI] OPTION {tag.upper()}: {label}")
    print(f"  Sweeping {len(config.LOOKBACKS)} lookbacks × "
          f"{len(config.WAVELET_OPTIONS)} wavelets = "
          f"{len(config.LOOKBACKS)*len(config.WAVELET_OPTIONS)} runs")
    print(f"{'─'*56}")

    all_results = []
    for wavelet in config.WAVELET_OPTIONS:
        for lb in config.LOOKBACKS:
            prep = run_preprocessing(data, lb, wavelet=wavelet)
            res  = train_one_wavelet(module, tag, prep, epochs)
            all_results.append(res)

    best = max(all_results, key=lambda r: r["val_acc"])
    print(f"\n  [FI-{tag.upper()}] Best: lb={best['lookback']}d "
          f"wavelet={best['wavelet']}  val_acc={best['val_acc']:.4f}")
    return {"best": best, "all_results": all_results}


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_training(models_to_train: list, epochs: int, start_year: int = None):
    print(f"\n{'='*60}")
    print(f"  [FI] Training: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Models: {models_to_train}  |  Epochs (max): {epochs}")
    print(f"  ETFs: {config.FI_ETFS}")
    print(f"  Wavelet options: {config.WAVELET_OPTIONS}  (auto-selected)")
    print(f"  Fee: {config.FEE_BPS}bps  TSL: {config.DEFAULT_TSL_PCT}%  "
          f"Z-reentry: {config.DEFAULT_Z_REENTRY}")
    print(f"{'='*60}")

    data = load_local()
    if not data:
        raise RuntimeError("No data found. Run data_download.py first.")

    training_summary = {}

    module_map = {"a": model_a, "b": model_b, "c": model_c}
    for tag in ["a", "b", "c"]:
        if tag not in models_to_train:
            continue
        sweep_result = sweep_model(module_map[tag], tag, data, epochs)
        training_summary[f"model_{tag}"] = {
            "best_lookback": sweep_result["best"]["lookback"],
            "best_wavelet":  sweep_result["best"]["wavelet"],
            "best_val_acc":  sweep_result["best"]["val_acc"],
            "all_results":   sweep_result["all_results"],
        }

    # Determine best overall wavelet
    wavelet_scores: dict = {}
    for key, info in training_summary.items():
        w   = info["best_wavelet"]
        acc = info["best_val_acc"]
        wavelet_scores.setdefault(w, []).append(acc)

    best_overall_wavelet = max(
        wavelet_scores,
        key=lambda w: (len(wavelet_scores[w]), sum(wavelet_scores[w]))
    )

    training_summary.update({
        "trained_at":   datetime.now().isoformat(),
        "epochs":       epochs,
        "start_year":   start_year,
        "best_wavelet": best_overall_wavelet,
        "universe":     "fi",
        "fee_bps":      config.FEE_BPS,
        "tsl_pct":      config.DEFAULT_TSL_PCT,
        "z_reentry":    config.DEFAULT_Z_REENTRY,
    })

    summary_path = os.path.join(config.MODELS_DIR, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(training_summary, f, indent=2)
    print(f"\n  [FI] Best overall wavelet: {best_overall_wavelet}")
    print(f"  [FI] Summary → {summary_path}")
    print(f"{'='*60}\n  [FI] Training complete.\n{'='*60}")
    return training_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="all",
                        help="all | a | b | c | a,b etc.")
    parser.add_argument("--epochs",     type=int, default=config.MAX_EPOCHS)
    parser.add_argument("--start_year", type=int, default=None)
    args = parser.parse_args()
    models = ["a", "b", "c"] if args.model == "all" else \
             [m.strip().lower() for m in args.model.split(",")]
    run_training(models, args.epochs, start_year=args.start_year)
