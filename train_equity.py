# train_equity.py
# Trains Option A, B, C for EQUITY ETFs (13 tickers).
#
# Critical fix vs train.py:
#   N_CLASSES = len(config.EQUITY_ETFS) = 13, NOT len(config.ETFS) = 20.
#   model_a/b/c.py hardcode N_CLASSES = len(config.ETFS) so we cannot
#   reuse them directly — we rebuild the identical architectures here
#   with the correct output dimension.
#
# Weight paths: models/model_{a|b|c}_eq/lb{n}/best.keras
#   → never collide with FI weights at models/model_{a|b|c}/lb{n}/best.keras
#
# Usage:
#   python train_equity.py --model all
#   python train_equity.py --model a --epochs 50

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
from data_download import load_local
from preprocess_equity import run_preprocessing

os.makedirs(config.MODELS_DIR, exist_ok=True)

EQUITY_N_CLASSES = len(config.EQUITY_ETFS)   # 13


# ─── Model builders (same architecture as A/B/C, output = 13 classes) ─────────

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
    """Option B: adds multi-head self-attention before the LSTM stack."""
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
    """Option C: parallel dual-stream (ETF features | macro features)."""
    n_macro = n_features - n_etf_features
    inp_etf   = keras.Input(shape=(lookback, n_etf_features),  name="etf_stream")
    inp_macro = keras.Input(shape=(lookback, n_macro),          name="macro_stream")

    def cnn_lstm_stream(x, filters=64):
        x = layers.Conv1D(filters, 3, padding="causal", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.LSTM(64)(x)
        return x

    etf_out   = cnn_lstm_stream(inp_etf,   filters=64)
    macro_out = cnn_lstm_stream(inp_macro,  filters=32)
    x   = layers.Concatenate()([etf_out, macro_out])
    x   = layers.Dense(128, activation="relu")(x)
    x   = layers.Dropout(0.3)(x)
    x   = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(EQUITY_N_CLASSES, activation="softmax")(x)
    model = keras.Model(inputs=[inp_etf, inp_macro], outputs=out, name="model_c_eq")
    model.compile(optimizer=keras.optimizers.Adam(5e-4),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


# ─── Path helpers ─────────────────────────────────────────────────────────────

def _ckpt_path(tag: str, lookback: int) -> str:
    path = os.path.join(config.MODELS_DIR, f"model_{tag}_eq", f"lb{lookback}", "best.keras")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path

def _final_path(tag: str, lookback: int) -> str:
    path = os.path.join(config.MODELS_DIR, f"model_{tag}_eq", f"lb{lookback}", "final.keras")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path

def load_equity_model(tag: str, lookback: int) -> keras.Model:
    """Load best equity checkpoint."""
    path = _ckpt_path(tag, lookback)
    return keras.models.load_model(path)


def _get_callbacks(tag: str, lookback: int) -> list:
    return [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=config.PATIENCE,
                                       restore_best_weights=True, mode="max"),
        keras.callbacks.ModelCheckpoint(_ckpt_path(tag, lookback),
                                         monitor="val_accuracy",
                                         save_best_only=True, mode="max"),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                           patience=5, min_lr=1e-6),
    ]


def _prep_labels(prep: dict):
    def _fix(y):
        if y.ndim == 2 and y.shape[1] > 1:
            print(f"  WARNING: y shape {y.shape} — converting via argmax")
            return y.argmax(axis=1).astype(np.int32)
        return y.flatten().astype(np.int32)
    return _fix(prep["y_tr"]), _fix(prep["y_va"])


def _class_weights(y_tr):
    cw = compute_class_weight("balanced", classes=np.arange(EQUITY_N_CLASSES), y=y_tr)
    return {i: float(w) for i, w in enumerate(cw)}


# ─── Per-model trainer ────────────────────────────────────────────────────────

def train_one(tag: str, prep: dict, epochs: int) -> dict:
    lookback       = prep["lookback"]
    n_features     = prep["n_features"]
    n_etf_features = prep["n_etf_features"]
    y_tr, y_va     = _prep_labels(prep)
    cw             = _class_weights(y_tr)

    print(f"\n[EQ-{tag.upper()}] lookback={lookback}  features={n_features}  "
          f"classes={EQUITY_N_CLASSES}")
    print(f"  Class dist: {dict(zip(*np.unique(y_tr, return_counts=True)))}")

    if tag == "a":
        model = build_model_a(lookback, n_features)
        X_tr, X_va = prep["X_tr"], prep["X_va"]
    elif tag == "b":
        model = build_model_b(lookback, n_features)
        X_tr, X_va = prep["X_tr"], prep["X_va"]
    else:  # c
        model = build_model_c(lookback, n_features, n_etf_features)
        X_tr  = [prep["X_tr"][:, :, :n_etf_features],
                 prep["X_tr"][:, :, n_etf_features:]]
        X_va  = [prep["X_va"][:, :, :n_etf_features],
                 prep["X_va"][:, :, n_etf_features:]]

    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=epochs,
        batch_size=config.BATCH_SIZE,
        callbacks=_get_callbacks(tag, lookback),
        class_weight=cw,
        verbose=1,
    )

    model.save(_final_path(tag, lookback))
    print(f"  Saved → {_final_path(tag, lookback)}")

    val_loss = min(history.history["val_loss"])
    val_acc  = max(history.history.get("val_accuracy", [0]))
    return {"val_mse": val_loss, "val_acc": val_acc, "lookback": lookback, "model": tag}


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
    print(f"  ETFs ({EQUITY_N_CLASSES}): {config.EQUITY_ETFS}")
    print(f"{'='*60}")

    data = load_local()
    if not data:
        raise RuntimeError("No data found. Run data_download.py first.")

    preps = {lb: run_preprocessing(data, lb) for lb in config.LOOKBACKS}
    training_summary = {}

    labels = {"a": "Wavelet-CNN-LSTM",
              "b": "Wavelet-Attention-CNN-LSTM",
              "c": "Wavelet-Parallel-Dual-Stream-CNN-LSTM"}

    for tag in ["a", "b", "c"]:
        if tag not in models_to_train:
            continue
        print(f"\n{'─'*50}")
        print(f"  [EQUITY] OPTION {tag.upper()}: {labels[tag]}")
        results = [train_one(tag, preps[lb], epochs) for lb in config.LOOKBACKS]
        training_summary[f"model_{tag}"] = {
            "best_lookback": select_best_lookback(results),
            "results": results,
        }

    training_summary.update({
        "trained_at": datetime.now().isoformat(),
        "epochs":     epochs,
        "start_year": start_year,
        "wavelet":    wavelet,
        "universe":   "equity",
        "n_classes":  EQUITY_N_CLASSES,
        "etfs":       config.EQUITY_ETFS,
    })

    summary_path = os.path.join(config.MODELS_DIR, "training_summary_equity.json")
    with open(summary_path, "w") as f:
        json.dump(training_summary, f, indent=2)
    print(f"\n  [EQUITY] Summary → {summary_path}")
    print(f"{'='*60}\n  [EQUITY] Training complete.\n{'='*60}")
    return training_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="all")
    parser.add_argument("--epochs",     type=int, default=config.MAX_EPOCHS)
    parser.add_argument("--start_year", type=int, default=None)
    parser.add_argument("--wavelet",    default=None)
    args = parser.parse_args()
    models = ["a", "b", "c"] if args.model == "all" else \
             [m.strip().lower() for m in args.model.split(",")]
    run_training(models, args.epochs,
                 start_year=args.start_year, wavelet=args.wavelet)
