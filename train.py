# train.py
# Trains Option A, B, C across all lookback windows (30, 45, 60).
# Auto-selects best lookback per model by validation MSE.
# Saves best weights + training metadata to models/
# Usage:
#   python train.py --model all          # train A, B, C
#   python train.py --model a            # train only A
#   python train.py --epochs 50          # override epochs

import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

import config
import model_a
import model_b
import model_c
from data_download import load_local
from preprocess import run_preprocessing

os.makedirs(config.MODELS_DIR, exist_ok=True)


# ─── Per-model trainer ────────────────────────────────────────────────────────

def train_one_model(module, tag: str, prep: dict, epochs: int) -> dict:
    """Train a single model on a single lookback. Returns val_mse."""
    model, history = module.train(prep, epochs=epochs)
    val_mse = min(history.history["val_loss"])
    val_mae = min(history.history["val_mae"])
    print(f"  [{tag}] lb={prep['lookback']}  "
          f"val_mse={val_mse:.6f}  val_mae={val_mae:.6f}")
    return {"val_mse": val_mse, "val_mae": val_mae,
            "lookback": prep["lookback"], "model": tag}


# ─── Lookback selector ────────────────────────────────────────────────────────

def select_best_lookback(results: list) -> int:
    """Return lookback with lowest val_mse."""
    best = min(results, key=lambda r: r["val_mse"])
    print(f"  Best lookback = {best['lookback']}d  "
          f"(val_mse={best['val_mse']:.6f})")
    return best["lookback"]


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_training(models_to_train: list, epochs: int):
    print(f"\n{'='*60}")
    print(f"  Training started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Models: {models_to_train}  |  Max epochs: {epochs}")
    print(f"{'='*60}")

    # Load data
    data = load_local()
    if not data:
        raise RuntimeError("No data found. Run data_download.py first.")

    # Preprocess for all lookbacks
    preps = {}
    for lb in config.LOOKBACKS:
        preps[lb] = run_preprocessing(data, lb)

    training_summary = {}

    # ── Option A ──────────────────────────────────────────────────────────────
    if "a" in models_to_train:
        print("\n" + "─"*50)
        print("  OPTION A: Wavelet-CNN-LSTM")
        print("─"*50)
        results_a = []
        for lb in config.LOOKBACKS:
            r = train_one_model(model_a, "A", preps[lb], epochs)
            results_a.append(r)
        best_lb_a = select_best_lookback(results_a)
        training_summary["model_a"] = {
            "best_lookback": best_lb_a,
            "results": results_a,
        }

    # ── Option B ──────────────────────────────────────────────────────────────
    if "b" in models_to_train:
        print("\n" + "─"*50)
        print("  OPTION B: Wavelet-Attention-CNN-LSTM")
        print("─"*50)
        results_b = []
        for lb in config.LOOKBACKS:
            r = train_one_model(model_b, "B", preps[lb], epochs)
            results_b.append(r)
        best_lb_b = select_best_lookback(results_b)
        training_summary["model_b"] = {
            "best_lookback": best_lb_b,
            "results": results_b,
        }

    # ── Option C ──────────────────────────────────────────────────────────────
    if "c" in models_to_train:
        print("\n" + "─"*50)
        print("  OPTION C: Wavelet-Parallel-Dual-Stream-CNN-LSTM")
        print("─"*50)
        results_c = []
        for lb in config.LOOKBACKS:
            r = train_one_model(model_c, "C", preps[lb], epochs)
            results_c.append(r)
        best_lb_c = select_best_lookback(results_c)
        training_summary["model_c"] = {
            "best_lookback": best_lb_c,
            "results": results_c,
        }

    # ── Save training summary ─────────────────────────────────────────────────
    training_summary["trained_at"] = datetime.now().isoformat()
    training_summary["epochs"]     = epochs
    summary_path = os.path.join(config.MODELS_DIR, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(training_summary, f, indent=2)
    print(f"\n  Training summary → {summary_path}")
    print(f"\n{'='*60}")
    print("  Training complete.")
    print(f"{'='*60}")

    return training_summary


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="all",
                        help="all | a | b | c | a,b | b,c etc.")
    parser.add_argument("--epochs", type=int, default=config.MAX_EPOCHS)
    args = parser.parse_args()

    if args.model == "all":
        models = ["a", "b", "c"]
    else:
        models = [m.strip().lower() for m in args.model.split(",")]

    run_training(models, args.epochs)
