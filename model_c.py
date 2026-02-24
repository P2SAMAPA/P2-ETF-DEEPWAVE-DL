# model_c.py — Option C: Wavelet-Parallel-Dual-Stream (CLASSIFICATION)
# Two parallel streams: ETF wavelet features + Macro wavelet features → fused → softmax
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import config

MODEL_NAME = "model_c"
N_CLASSES  = len(config.ETFS)


def build_model(lookback, n_features, n_etf_features):
    n_macro = n_features - n_etf_features

    # ETF stream
    inp_etf = keras.Input(shape=(lookback, n_etf_features), name="input_etf")
    x_etf   = layers.Conv1D(64, 3, padding="causal", activation="relu")(inp_etf)
    x_etf   = layers.LSTM(64, return_sequences=True)(x_etf)
    x_etf   = layers.LSTM(32)(x_etf)

    # Macro stream
    inp_mac = keras.Input(shape=(lookback, max(n_macro, 1)), name="input_macro")
    x_mac   = layers.Conv1D(32, 3, padding="causal", activation="relu")(inp_mac)
    x_mac   = layers.LSTM(32)(x_mac)

    # Fusion
    x   = layers.Concatenate()([x_etf, x_mac])
    x   = layers.Dense(64, activation="relu")(x)
    x   = layers.Dropout(0.3)(x)
    out = layers.Dense(N_CLASSES, activation="softmax", name="output")(x)

    model = keras.Model(inputs=[inp_etf, inp_mac], outputs=out, name=MODEL_NAME)
    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate=5e-4),
        loss      = "sparse_categorical_crossentropy",
        metrics   = ["accuracy"],
    )
    return model


def get_callbacks(lookback):
    ckpt = os.path.join(config.MODELS_DIR, MODEL_NAME, f"lb{lookback}", "best.keras")
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    return [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=config.PATIENCE,
                                       restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(ckpt, monitor="val_loss", save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                           patience=5, min_lr=1e-6),
    ]


def save_model(model, lookback):
    path = os.path.join(config.MODELS_DIR, MODEL_NAME, f"lb{lookback}", "final.keras")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    print(f"  [{MODEL_NAME}] Saved → {path}")


def load_model(lookback):
    return keras.models.load_model(
        os.path.join(config.MODELS_DIR, MODEL_NAME, f"lb{lookback}", "best.keras"))


def train(prep, epochs=config.MAX_EPOCHS):
    lookback     = prep["lookback"]
    n_features   = prep["n_features"]
    n_etf        = prep["n_etf_features"]
    print(f"\n[{MODEL_NAME}] lookback={lookback}  ETF={n_etf}  "
          f"macro={n_features-n_etf}  classes={N_CLASSES}")

    model = build_model(lookback, n_features, n_etf)

    X_tr_etf = prep["X_tr"][:, :, :n_etf]
    X_tr_mac = prep["X_tr"][:, :, n_etf:]
    X_va_etf = prep["X_va"][:, :, :n_etf]
    X_va_mac = prep["X_va"][:, :, n_etf:]

    if X_tr_mac.shape[2] == 0:
        X_tr_mac = np.zeros((X_tr_etf.shape[0], lookback, 1), dtype=np.float32)
        X_va_mac = np.zeros((X_va_etf.shape[0], lookback, 1), dtype=np.float32)

    history = model.fit(
        [X_tr_etf, X_tr_mac], prep["y_tr"],
        validation_data = ([X_va_etf, X_va_mac], prep["y_va"]),
        epochs=epochs, batch_size=config.BATCH_SIZE,
        callbacks=get_callbacks(lookback), verbose=1,
    )
    save_model(model, lookback)
    return model, history
