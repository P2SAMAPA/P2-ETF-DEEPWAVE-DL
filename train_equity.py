# train_equity.py
# Trains Option A, B, C across all lookback windows (30, 45, 60) for EQUITY ETFs.
# Mirrors train.py exactly — only imports preprocess_equity and saves to
# equity-namespaced files (model_a_eq_lb{n}.keras, training_summary_equity.json)
# so FI and equity weights never collide.
#
# Usage:
#   python train_equity.py --model all
#   python train_equity.py --model a
#   python train_equity.py --epochs 50

import argparse
import json
import os
from datetime import datetime

import config
import model_a
import model_b
import model_c
from data_download import load_local
from preprocess_equity import run_preprocessing   # equity-specific preprocessor

os.makedirs(config.MODELS_DIR, exist_ok=True)


# ─── Per-model trainer ────────────────────────────────────────────────────────

def train_one_model(module, tag: str, prep: dict, epochs: int) -> dict:
    """Train a single model, save weights under equity-namespaced path."""
    model, history = module.train(prep, epochs=epochs)
    val_loss = min(history.history["val_loss"])
    val_acc  = max(history.history.get("val_accuracy", [0]))
    print(f"  [EQ-{tag}] lb={prep['lookback']}  "
          f"val_loss={val_loss:.6f}  val_acc={val_acc:.4f}")

    # Save weights with _eq suffix — never overwrites FI weights
    lb   = prep["lookback"]
    path = os.path.join(config.MODELS_DIR, f"model_{tag.lower()}_eq_lb{lb}.keras")
    model.save(path)
    print(f"  Saved → {path}")

    return {"val_mse": val_loss, "val_acc": val_acc,
            "lookback": lb, "model": tag}


def select_best_lookback(results: list) -> int:
    best = min(results, key=lambda r: r["val_mse"])
    print(f"  Best lookback = {best['lookback']}d  (val_mse={best['val_mse']:.6f})")
    return best["lookback"]


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_training(models_to_train: list, epochs: int,
                 start_year: int = None, wavelet: str = None):
    print(f"\n{'='*60}")
    print(f"  [EQUITY] Training: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Models: {models_to_train}  |  Epochs: {epochs}")
    print(f"  ETFs ({len(config.EQUITY_ETFS)}): {config.EQUITY_ETFS}")
    print(f"{'='*60}")

    data = load_local()
    if not data:
        raise RuntimeError("No data found. Run data_download.py first.")

    # Preprocess for all lookbacks using equity universe
    preps = {lb: run_preprocessing(data, lb) for lb in config.LOOKBACKS}

    training_summary = {}

    if "a" in models_to_train:
        print("\n" + "─"*50)
        print("  [EQUITY] OPTION A: Wavelet-CNN-LSTM")
        results_a = [train_one_model(model_a, "A", preps[lb], epochs)
                     for lb in config.LOOKBACKS]
        training_summary["model_a"] = {"best_lookback": select_best_lookback(results_a),
                                        "results": results_a}

    if "b" in models_to_train:
        print("\n" + "─"*50)
        print("  [EQUITY] OPTION B: Wavelet-Attention-CNN-LSTM")
        results_b = [train_one_model(model_b, "B", preps[lb], epochs)
                     for lb in config.LOOKBACKS]
        training_summary["model_b"] = {"best_lookback": select_best_lookback(results_b),
                                        "results": results_b}

    if "c" in models_to_train:
        print("\n" + "─"*50)
        print("  [EQUITY] OPTION C: Wavelet-Parallel-Dual-Stream-CNN-LSTM")
        results_c = [train_one_model(model_c, "C", preps[lb], epochs)
                     for lb in config.LOOKBACKS]
        training_summary["model_c"] = {"best_lookback": select_best_lookback(results_c),
                                        "results": results_c}

    training_summary.update({
        "trained_at": datetime.now().isoformat(),
        "epochs":     epochs,
        "start_year": start_year,
        "wavelet":    wavelet,
        "universe":   "equity",
        "etfs":       config.EQUITY_ETFS,
    })

    summary_path = os.path.join(config.MODELS_DIR, "training_summary_equity.json")
    with open(summary_path, "w") as f:
        json.dump(training_summary, f, indent=2)
    print(f"\n  [EQUITY] Summary → {summary_path}")
    print(f"{'='*60}\n  [EQUITY] Training complete.\n{'='*60}")
    return training_summary


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="all",
                        help="all | a | b | c | a,b etc.")
    parser.add_argument("--epochs",     type=int, default=config.MAX_EPOCHS)
    parser.add_argument("--start_year", type=int, default=None)
    parser.add_argument("--wavelet",    default=None)
    args = parser.parse_args()

    models = ["a", "b", "c"] if args.model == "all" else \
             [m.strip().lower() for m in args.model.split(",")]
    run_training(models, args.epochs,
                 start_year=args.start_year, wavelet=args.wavelet)
