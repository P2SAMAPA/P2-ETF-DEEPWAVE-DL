# model_a.py — Option A: Wavelet-CNN-LSTM (CLASSIFICATION)
# Output: softmax over 5 ETF classes. Loss: sparse_categorical_crossentropy.
# This prevents near-zero MSE collapse.

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import config

MODEL_NAME = "model_a"
N_CLASSES  = len(config.ETFS)   # 5


def build_model(lookback: int, n_features: int) -> keras.Model:
    inp = keras.Input(shape=(lookback, n_features), name="input")

    x = layers.Conv1D(64, kernel_size=3, padding="causal",
                      activation="relu", name="conv1")(inp)
    x = layers.Conv1D(64, kernel_size=3, padding="causal",
                      activation="relu", name="conv2")(x)
    x = layers.MaxPooling1D(pool_size=2, name="pool")(x)
    x = layers.LSTM(128, return_sequences=True, name="lstm1")(x)
    x = layers.LSTM(64, name="lstm2")(x)
    x = layers.Dense(32, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    # Softmax output — probability distribution over 5 ETFs
    out = layers.Dense(N_CLASSES, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inp, outputs=out, name=MODEL_NAME)
    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate=1e-3),
        loss      = "sparse_categorical_crossentropy",
        metrics   = ["accuracy"],
    )
    return model


def get_callbacks(lookback: int) -> list:
    ckpt_path = os.path.join(config.MODELS_DIR, MODEL_NAME,
                             f"lb{lookback}", "best.keras")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    return [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=config.PATIENCE,
                                       restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss",
                                         save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                           patience=5, min_lr=1e-6),
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
    lookback   = prep["lookback"]
    n_features = prep["n_features"]
    print(f"\n[{MODEL_NAME}] lookback={lookback}  features={n_features}  "
          f"classes={N_CLASSES}")

    model = build_model(lookback, n_features)
    history = model.fit(
        prep["X_tr"], prep["y_tr"],
        validation_data = (prep["X_va"], prep["y_va"]),
        epochs     = epochs,
        batch_size = config.BATCH_SIZE,
        callbacks  = get_callbacks(lookback),
        verbose    = 1,
    )
    save_model(model, lookback)
    return model, history
