# model_c.py — Option C: Wavelet-Parallel-Dual-Stream-CNN-LSTM
#
# Architecture:
#   Two parallel input streams:
#     Stream 1 (ETF)  : ETF wavelet features  → Conv1D → LSTM(64)  ─┐
#     Stream 2 (Macro): Macro wavelet features → Conv1D → LSTM(64)  ─┤→ Concat
#                                                                      → Dense(64, ReLU)
#                                                                      → Dropout(0.3)
#                                                                      → Dense(5)
#
# Dual streams keep ETF price dynamics and macro regime signals in
# separate learned representations before fusion, preventing macro
# signals from dominating ETF price structure.

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import config

MODEL_NAME = "model_c"


def build_model(lookback: int,
                n_etf_features: int,
                n_macro_features: int) -> keras.Model:

    # ── Stream 1: ETF price dynamics ──────────────────────────────────────────
    inp_etf = keras.Input(shape=(lookback, n_etf_features), name="input_etf")
    e = layers.Conv1D(64, kernel_size=3, padding="causal",
                      activation="relu", name="etf_conv1")(inp_etf)
    e = layers.Conv1D(32, kernel_size=3, padding="causal",
                      activation="relu", name="etf_conv2")(e)
    e = layers.MaxPooling1D(pool_size=2, name="etf_pool")(e)
    e = layers.LSTM(64, name="etf_lstm")(e)

    # ── Stream 2: Macro regime signals ────────────────────────────────────────
    inp_mac = keras.Input(shape=(lookback, n_macro_features), name="input_macro")
    m = layers.Conv1D(64, kernel_size=3, padding="causal",
                      activation="relu", name="mac_conv1")(inp_mac)
    m = layers.Conv1D(32, kernel_size=3, padding="causal",
                      activation="relu", name="mac_conv2")(m)
    m = layers.MaxPooling1D(pool_size=2, name="mac_pool")(m)
    m = layers.LSTM(64, name="mac_lstm")(m)

    # ── Fusion ────────────────────────────────────────────────────────────────
    fused = layers.Concatenate(name="fusion")([e, m])
    x = layers.Dense(64, activation="relu", name="dense1")(fused)
    x = layers.Dropout(0.3, name="dropout")(x)
    out = layers.Dense(len(config.ETFS), activation="linear",
                       name="output")(x)

    model = keras.Model(inputs=[inp_etf, inp_mac], outputs=out, name=MODEL_NAME)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
    return model


def get_callbacks(lookback: int) -> list:
    ckpt_path = os.path.join(config.MODELS_DIR, MODEL_NAME,
                             f"lb{lookback}", "best.keras")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    return [
        keras.callbacks.EarlyStopping(monitor="val_loss",
                                      patience=config.PATIENCE,
                                      restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss",
                                        save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=5, min_lr=1e-6),
    ]


def save_model(model: keras.Model, lookback: int):
    path = os.path.join(config.MODELS_DIR, MODEL_NAME,
                        f"lb{lookback}", "final.keras")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    print(f"  [{MODEL_NAME}] Saved → {path}")


def load_model(lookback: int) -> keras.Model:
    path = os.path.join(config.MODELS_DIR, MODEL_NAME,
                        f"lb{lookback}", "best.keras")
    return keras.models.load_model(path)


def prepare_dual_inputs(X: np.ndarray, n_etf_features: int) -> list:
    """Split combined X into [X_etf, X_macro] for dual-stream input."""
    X_etf   = X[:, :, :n_etf_features]
    X_macro = X[:, :, n_etf_features:]
    return [X_etf, X_macro]


def train(prep: dict, epochs: int = config.MAX_EPOCHS) -> tuple:
    lookback        = prep["lookback"]
    n_features      = prep["n_features"]
    n_etf_features  = prep["n_etf_features"]
    n_macro_features= n_features - n_etf_features

    print(f"\n[{MODEL_NAME}] Building dual-stream model — lookback={lookback}, "
          f"ETF features={n_etf_features}, macro features={n_macro_features}")

    model = build_model(lookback, n_etf_features, n_macro_features)

    # Prepare dual inputs
    tr_inputs = prepare_dual_inputs(prep["X_tr"], n_etf_features)
    va_inputs = prepare_dual_inputs(prep["X_va"], n_etf_features)

    history = model.fit(
        tr_inputs, prep["y_tr"],
        validation_data=(va_inputs, prep["y_va"]),
        epochs     = epochs,
        batch_size = config.BATCH_SIZE,
        callbacks  = get_callbacks(lookback),
        verbose    = 1,
    )
    save_model(model, lookback)
    return model, history
