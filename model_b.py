# model_b.py — Option B: Wavelet-Attention-CNN-LSTM (CLASSIFICATION)
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import config

MODEL_NAME = "model_b"
N_CLASSES  = len(config.ETFS)


def build_model(lookback, n_features):
    inp = keras.Input(shape=(lookback, n_features), name="input")
    x   = layers.Conv1D(64, kernel_size=3, padding="causal", activation="relu")(inp)
    x   = layers.BatchNormalization()(x)
    attn= layers.MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
    x   = layers.Add()([x, attn])
    x   = layers.LayerNormalization()(x)
    x   = layers.LSTM(128, return_sequences=True)(x)
    x   = layers.LSTM(64)(x)
    x   = layers.Dense(64, activation="relu")(x)
    x   = layers.Dropout(0.3)(x)
    out = layers.Dense(N_CLASSES, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inp, outputs=out, name=MODEL_NAME)
    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate=5e-4),
        loss      = "sparse_categorical_crossentropy",   # integer labels
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
    lookback, n_features = prep["lookback"], prep["n_features"]
    print(f"\n[{MODEL_NAME}] lookback={lookback}  features={n_features}  classes={N_CLASSES}")
    model   = build_model(lookback, n_features)
    # y is integer class labels — no one-hot needed with sparse_categorical
    history = model.fit(
        prep["X_tr"], prep["y_tr"],
        validation_data = (prep["X_va"], prep["y_va"]),
        epochs=epochs, batch_size=config.BATCH_SIZE,
        callbacks=get_callbacks(lookback), verbose=1,
    )
    save_model(model, lookback)
    return model, history
