# train_equity.py
# Trains Option A, B, C for EQUITY ETFs (12 tickers).
# Wavelet auto-selection: tries all 4 options per lookback, keeps best val accuracy.
# Risk params hardcoded: FEE_BPS=12, TSL=12%, Z_REENTRY=0.9, MAX_EPOCHS=80.
# Saves weights to models/model_{a|b|c}_eq/lb{n}_{wavelet}/best.keras

import argparse
import json
import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils.class_weight import compute_class_weight

import config
from data_utils import load_local
from preprocess_equity import run_preprocessing

os.makedirs(config.MODELS_DIR, exist_ok=True)

EQUITY_N_CLASSES = len(config.EQUITY_ETFS)   # 12


# ─── Model builders ───────────────────────────────────────────────────────────

def build_model_a(lookback: int, n_features: int) -> keras.Model:
    inp = keras.Input(shape=(lookback, n_features))
    x = layers.Conv1D(64, 3, padding="causal", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(64, 3, padding="causal", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(64)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(EQUITY_N_CLASSES, activation="softmax")(x)
    model = keras.Model(inputs=inp, outputs=out, name="model_a_eq")
    model.compile(optimizer=keras.optimizers.Adam(5e-4),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_model_b(lookback: int, n_features: int) -> keras.Model:
    inp  = keras.Input(shape=(lookback, n_features))
    x    = layers.Conv1D(64, 3, padding="causal", activation="relu")(inp)
    x    = layers.BatchNormalization()(x)
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
    x    = layers.Add()([x, attn])
    x    = layers.LayerNormalization()(x)
    x    = layers.Conv1D(64, 3, padding="causal", activation="relu")(x)
    x    = layers.BatchNormalization()(x)
    x    = layers.MaxPooling1D(2)(x)
    x    = layers.Dropout(0.2)(x)
    x    = layers.LSTM(128, return_sequences=True)(x)
    x    = layers.Dropout(0.2)(x)
    x    = layers.LSTM(64)(x)
    x    = layers.Dense(64, activation="relu")(x)
    x    = layers.Dropout(0.3)(x)
    x    = layers.Dense(32, activation="relu")(x)
    out  = layers.Dense(EQUITY_N_CLASSES, activation="softmax")(x)
    model = keras.Model(inputs=inp, outputs=out, name="model_b_eq")
    model.compile(optimizer=keras.optimizers.Adam(5e-4),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_model_c(lookback: int, n_features: int, n_etf_features: int) -> keras.Model:
    n_macro   = n_features - n_etf_features
    inp_etf   = keras.Input(shape=(lookback, n_etf_features), name="etf_stream")
    inp_macro = keras.Input(shape=(lookback, n_macro),         name="macro_stream")

    def cnn_lstm(x, filters=64):
        x = layers.Conv1D(filters, 3, padding="causal", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.LSTM(64)(x)
        return x

    x   = layers.Concatenate()([cnn_lstm(inp_etf, 64), cnn_lstm(inp_macro, 32)])
    x   = layers.Dense(128, activation="relu")(x)
    x   = layers.Dropout(0.3)(x)
    x   = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(EQUITY_N_CLASSES, activation="softmax")(x)
    model = keras.Model(inputs=[inp_etf, inp_macro], outputs=out, name="model_c_eq")
    model.compile(optimizer=keras.optimizers.Adam(5e-4),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


# ─── Paths ────────────────────────────────────────────────────────────────────

def _ckpt_path(tag: str, lookback: int, wavelet: str) -> str:
    path = os.path.join(config.MODELS_DIR,
                        f"model_{tag}_eq", f"lb{lookback}_{wavelet}", "best.keras")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def _final_path(tag: str, lookback: int, wavelet: str) -> str:
    path = os.path.join(config.MODELS_DIR,
                        f"model_{tag}_eq", f"lb{lookback}_{wavelet}", "final.keras")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def load_equity_model(tag: str, lookback: int, wavelet: str) -> keras.Model:
    return keras.models.load_model(_ckpt_path(tag, lookback, wavelet))


def _callbacks(tag: str, lookback: int, wavelet: str) -> list:
    return [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=config.PATIENCE,
                                       restore_best_weights=True, mode="max"),
        keras.callbacks.ModelCheckpoint(_ckpt_path(tag, lookback, wavelet),
                                         monitor="val_accuracy",
                                         save_best_only=True, mode="max"),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                           patience=5, min_lr=1e-6),
    ]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _fix_labels(y):
    if y.ndim == 2 and y.shape[1] > 1:
        return y.argmax(axis=1).astype(np.int32)
    return y.flatten().astype(np.int32)


def _class_weights(y_tr):
    present    = np.unique(y_tr)
    cw         = compute_class_weight("balanced", classes=present, y=y_tr)
    weight_map = dict(zip(present.tolist(), cw.tolist()))
    return {i: float(weight_map.get(i, 1.0)) for i in range(EQUITY_N_CLASSES)}


# ─── Train one model + wavelet combination ────────────────────────────────────

def train_one(tag: str, prep: dict, epochs: int, wavelet: str) -> dict:
    lookback       = prep["lookback"]
    n_features     = prep["n_features"]
    n_etf_features = prep["n_etf_features"]
    y_tr = _fix_labels(prep["y_tr"])
    y_va = _fix_labels(prep["y_va"])
    cw   = _class_weights(y_tr)

    print(f"  [EQ-{tag.upper()}] lb={lookback} wavelet={wavelet} "
          f"features={n_features} classes={EQUITY_N_CLASSES}")

    if tag == "a":
        model  = build_model_a(lookback, n_features)
        X_tr, X_va = prep["X_tr"], prep["X_va"]
    elif tag == "b":
        model  = build_model_b(lookback, n_features)
        X_tr, X_va = prep["X_tr"], prep["X_va"]
    else:
        model  = build_model_c(lookback, n_features, n_etf_features)
        X_tr   = [prep["X_tr"][:, :, :n_etf_features],
                  prep["X_tr"][:, :, n_etf_features:]]
        X_va   = [prep["X_va"][:, :, :n_etf_features],
                  prep["X_va"][:, :, n_etf_features:]]

    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=epochs,
        batch_size=config.BATCH_SIZE,
        callbacks=_callbacks(tag, lookback, wavelet),
        class_weight=cw,
        verbose=0,
    )

    model.save(_final_path(tag, lookback, wavelet))

    val_loss = min(history.history["val_loss"])
    val_acc  = max(history.history.get("val_accuracy", [0]))
    print(f"    val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

    return {"val_loss": val_loss, "val_acc": val_acc,
            "lookback": lookback, "wavelet": wavelet, "model": tag}


# ─── Wavelet + lookback sweep ─────────────────────────────────────────────────

def sweep_model(tag: str, data: dict, epochs: int) -> dict:
    """
    Try every combination of (lookback × wavelet).
    Pick the combination with highest val_acc.
    Returns best result dict.
    """
    label = {"a": "Wavelet-CNN-LSTM",
             "b": "Wavelet-Attention-CNN-LSTM",
             "c": "Wavelet-Parallel-Dual-Stream"}[tag]
    print(f"\n{'─'*56}")
    print(f"  [EQUITY] OPTION {tag.upper()}: {label}")
    print(f"  Sweeping {len(config.LOOKBACKS)} lookbacks × "
          f"{len(config.WAVELET_OPTIONS)} wavelets = "
          f"{len(config.LOOKBACKS)*len(config.WAVELET_OPTIONS)} runs")
    print(f"{'─'*56}")

    all_results = []
    for wavelet in config.WAVELET_OPTIONS:
        for lb in config.LOOKBACKS:
            prep = run_preprocessing(data, lb, wavelet=wavelet)
            res  = train_one(tag, prep, epochs, wavelet)
            all_results.append(res)

    best = max(all_results, key=lambda r: r["val_acc"])
    print(f"\n  [EQ-{tag.upper()}] Best: lookback={best['lookback']}d "
          f"wavelet={best['wavelet']}  val_acc={best['val_acc']:.4f}")
    return {"best": best, "all_results": all_results}


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_training(models_to_train: list, epochs: int, start_year: int = None):
    print(f"\n{'='*60}")
    print(f"  [EQUITY] Training: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Models: {models_to_train}  |  Epochs (max): {epochs}")
    print(f"  ETFs ({EQUITY_N_CLASSES}): {config.EQUITY_ETFS}")
    print(f"  Wavelet options: {config.WAVELET_OPTIONS}  (auto-selected)")
    print(f"  Fee: {config.FEE_BPS}bps  TSL: {config.DEFAULT_TSL_PCT}%  "
          f"Z-reentry: {config.DEFAULT_Z_REENTRY}")
    print(f"{'='*60}")

    data = load_local()
    if not data:
        raise RuntimeError("No data found. Run data_utils.py first.")

    training_summary = {}

    for tag in ["a", "b", "c"]:
        if tag not in models_to_train:
            continue
        sweep_result = sweep_model(tag, data, epochs)
        training_summary[f"model_{tag}"] = {
            "best_lookback": sweep_result["best"]["lookback"],
            "best_wavelet":  sweep_result["best"]["wavelet"],
            "best_val_acc":  sweep_result["best"]["val_acc"],
            "all_results":   sweep_result["all_results"],
        }

    # Determine overall best wavelet across all trained models
    # (most frequent winner, tie-break by highest val_acc)
    wavelet_scores: dict = {}
    for key, info in training_summary.items():
        w   = info["best_wavelet"]
        acc = info["best_val_acc"]
        if w not in wavelet_scores:
            wavelet_scores[w] = []
        wavelet_scores[w].append(acc)

    best_overall_wavelet = max(
        wavelet_scores,
        key=lambda w: (len(wavelet_scores[w]), sum(wavelet_scores[w]))
    )

    training_summary.update({
        "trained_at":     datetime.now().isoformat(),
        "epochs":         epochs,
        "start_year":     start_year,
        "best_wavelet":   best_overall_wavelet,   # stamped for UI display
        "universe":       "equity",
        "n_classes":      EQUITY_N_CLASSES,
        "etfs":           config.EQUITY_ETFS,
        "fee_bps":        config.FEE_BPS,
        "tsl_pct":        config.DEFAULT_TSL_PCT,
        "z_reentry":      config.DEFAULT_Z_REENTRY,
    })

    summary_path = os.path.join(config.MODELS_DIR, "training_summary_equity.json")
    with open(summary_path, "w") as f:
        json.dump(training_summary, f, indent=2)
    print(f"\n  [EQUITY] Best overall wavelet: {best_overall_wavelet}")
    print(f"  [EQUITY] Summary → {summary_path}")
    print(f"{'='*60}\n  [EQUITY] Training complete.\n{'='*60}")
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
