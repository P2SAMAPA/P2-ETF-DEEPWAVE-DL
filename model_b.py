# model_b.py — Option B: Wavelet-Attention-CNN-LSTM
#
# Architecture:
#   Input (lookback, n_features)
#   → Conv1D(64, k=3, ReLU) × 2
#   → MaxPool1D
#   → LSTM(128, return_sequences=True)
#   → MultiHeadAttention(heads=4, key_dim=32)   ← attention over LSTM outputs
#   → GlobalAveragePooling1D
#   → Dense(64, ReLU) → Dropout(0.3)
#   → Dense(5)

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import config

MODEL_NAME = "model_b"


def build_model(lookback: int, n_features: int) -> keras.Model:
    inp = keras.Input(shape=(lookback, n_features), name="input")

    # CNN block
    x = layers.Conv1D(64, kernel_size=3, padding="causal",
                      activation="relu", name="conv1")(inp)
    x = layers.Conv1D(64, kernel_size=3, padding="causal",
                      activation="relu", name="conv2")(x)
    x = layers.MaxPooling1D(pool_size=2, name="pool")(x)

    # LSTM with full sequence output for attention
    lstm_out = layers.LSTM(128, return_sequences=True, name="lstm")(x)

    # Multi-Head Self-Attention
    # Attention lets model focus on the most regime-relevant timesteps
    attn_out, attn_weights = layers.MultiHeadAttention(
        num_heads=4, key_dim=32, name="mha"
    )(lstm_out, lstm_out, return_attention_scores=True)

    # Residual connection + layer norm
    x = layers.Add(name="residual")([lstm_out, attn_out])
    x = layers.LayerNormalization(name="layer_norm")(x)

    # Pool across time
    x = layers.GlobalAveragePooling1D(name="gap")(x)

    # Head
    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    out = layers.Dense(len(config.ETFS), activation="linear",
                       name="output")(x)

    model = keras.Model(inputs=inp, outputs=out, name=MODEL_NAME)
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


def train(prep: dict, epochs: int = config.MAX_EPOCHS) -> tuple:
    lookback   = prep["lookback"]
    n_features = prep["n_features"]
    print(f"\n[{MODEL_NAME}] Building model — lookback={lookback}, "
          f"features={n_features}")

    model = build_model(lookback, n_features)

    history = model.fit(
        prep["X_tr"], prep["y_tr"],
        validation_data=(prep["X_va"], prep["y_va"]),
        epochs     = epochs,
        batch_size = config.BATCH_SIZE,
        callbacks  = get_callbacks(lookback),
        verbose    = 1,
    )
    save_model(model, lookback)
    return model, history
