# model_c.py — Option C: Wavelet-Parallel-Dual-Stream-CNN-LSTM
# Two streams: ETF wavelet features + Macro features → merged classification

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import config

MODEL_NAME = "model_c"
N_CLASSES  = len(config.ETFS)


def build_model(lookback: int, n_etf_features: int,
                n_macro_features: int) -> keras.Model:

    # ── ETF stream ────────────────────────────────────────────────────────────
    etf_inp = keras.Input(shape=(lookback, n_etf_features), name="etf_input")
    e = layers.Conv1D(64, 3, padding="causal", activation="relu")(etf_inp)
    e = layers.BatchNormalization()(e)
    e = layers.Conv1D(32, 3, padding="causal", activation="relu")(e)
    e = layers.BatchNormalization()(e)
    e = layers.Dropout(0.2)(e)
    e = layers.LSTM(64, return_sequences=True)(e)
    e = layers.Dropout(0.2)(e)
    e = layers.LSTM(32)(e)

    # ── Macro stream ─────────────────────────────────────────────────────────
    mac_inp = keras.Input(shape=(lookback, n_macro_features), name="macro_input")
    m = layers.Conv1D(32, 3, padding="causal", activation="relu")(mac_inp)
    m = layers.BatchNormalization()(m)
    m = layers.Dropout(0.2)(m)
    m = layers.LSTM(32, return_sequences=True)(m)
    m = layers.Dropout(0.2)(m)
    m = layers.LSTM(16)(m)

    # ── Fusion ───────────────────────────────────────────────────────────────
    fused = layers.Concatenate()([e, m])
    x     = layers.Dense(64, activation="relu")(fused)
    x     = layers.Dropout(0.3)(x)
    x     = layers.Dense(32, activation="relu")(x)
    out   = layers.Dense(N_CLASSES, activation="softmax", name="output")(x)

    model = keras.Model(inputs=[etf_inp, mac_inp], outputs=out, name=MODEL_NAME)
    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate=3e-4),
        loss      = "sparse_categorical_crossentropy",
        metrics   = ["accuracy"],
    )
    return model


def get_callbacks(lookback: int) -> list:
    ckpt_path = os.path.join(config.MODELS_DIR, MODEL_NAME,
                             f"lb{lookback}", "best.keras")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=config.PATIENCE,
            restore_best_weights=True, mode="max"),
        keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor="val_accuracy",
            save_best_only=True, mode="max"),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
    ]


def save_model(model, lookback):
    path = os.path.join(config.MODELS_DIR, MODEL_NAME, f"lb{lookback}", "final.keras")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    print(f"  [{MODEL_NAME}] Saved → {path}")


def load_model(lookback):
    path = os.path.join(config.MODELS_DIR, MODEL_NAME, f"lb{lookback}", "best.keras")
    return keras.models.load_model(path)


def train(prep: dict, epochs: int = config.MAX_EPOCHS):
    lookback       = prep["lookback"]
    n_etf          = prep["n_etf_features"]
    n_macro        = prep["n_features"] - n_etf
    y_tr = prep["y_tr"].flatten().astype(np.int32)
    y_va = prep["y_va"].flatten().astype(np.int32)

    print(f"\n[{MODEL_NAME}] lookback={lookback}  "
          f"etf_feats={n_etf}  macro_feats={n_macro}")
    print(f"  Class dist (train): {dict(zip(*np.unique(y_tr, return_counts=True)))}")

    from sklearn.utils.class_weight import compute_class_weight
    cw = compute_class_weight("balanced", classes=np.arange(N_CLASSES), y=y_tr)
    class_weights = {i: w for i, w in enumerate(cw)}

    model = build_model(lookback, n_etf, max(n_macro, 1))
    X_tr_etf  = prep["X_tr"][:, :, :n_etf]
    X_tr_mac  = prep["X_tr"][:, :, n_etf:]
    X_va_etf  = prep["X_va"][:, :, :n_etf]
    X_va_mac  = prep["X_va"][:, :, n_etf:]

    history = model.fit(
        [X_tr_etf, X_tr_mac], y_tr,
        validation_data = ([X_va_etf, X_va_mac], y_va),
        epochs          = epochs,
        batch_size      = config.BATCH_SIZE,
        callbacks       = get_callbacks(lookback),
        class_weight    = class_weights,
        verbose         = 1,
    )
    save_model(model, lookback)
    return model, history
