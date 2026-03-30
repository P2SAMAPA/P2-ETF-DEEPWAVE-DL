# model_b.py — Option B: Wavelet-Attention-CNN-LSTM
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import config

MODEL_NAME = "model_b"
N_CLASSES  = len(config.FI_ETFS)   # 7 — FI only, not the combined ETFS list


def build_model(lookback: int, n_features: int) -> keras.Model:
    inp = keras.Input(shape=(lookback, n_features))
    x = layers.Conv1D(128, 3, padding="causal", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(64, 3, padding="causal", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
    x = layers.LayerNormalization()(x + attn)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(64)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(N_CLASSES, activation="softmax")(x)
    model = keras.Model(inputs=inp, outputs=out, name=MODEL_NAME)
    model.compile(
        optimizer=keras.optimizers.Adam(3e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_callbacks(lookback: int) -> list:
    ckpt = os.path.join(config.MODELS_DIR, MODEL_NAME, f"lb{lookback}", "best.keras")
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    return [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=config.PATIENCE,
                                       restore_best_weights=True, mode="max"),
        keras.callbacks.ModelCheckpoint(ckpt, monitor="val_accuracy",
                                         save_best_only=True, mode="max"),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                           patience=5, min_lr=1e-6),
    ]


def save_model(model, lookback):
    path = os.path.join(config.MODELS_DIR, MODEL_NAME, f"lb{lookback}", "final.keras")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)


def load_model(lookback):
    path = os.path.join(config.MODELS_DIR, MODEL_NAME, f"lb{lookback}", "best.keras")
    return keras.models.load_model(path)


def train(prep: dict, epochs: int = config.MAX_EPOCHS):
    lookback   = prep["lookback"]
    n_features = prep["n_features"]

    _ytr = prep["y_tr"]
    _yva = prep["y_va"]
    if _ytr.ndim == 2 and _ytr.shape[1] > 1:
        print(f"  WARNING: y shape {_ytr.shape} — converting via argmax")
        y_tr = _ytr.argmax(axis=1).astype(np.int32)
        y_va = _yva.argmax(axis=1).astype(np.int32)
    else:
        y_tr = _ytr.flatten().astype(np.int32)
        y_va = _yva.flatten().astype(np.int32)

    print(f"\n[{MODEL_NAME}] lookback={lookback}  features={n_features}  classes={N_CLASSES}")
    print(f"  Class dist: {dict(zip(*np.unique(y_tr, return_counts=True)))}")

    from sklearn.utils.class_weight import compute_class_weight
    present    = np.unique(y_tr)
    cw         = compute_class_weight("balanced", classes=present, y=y_tr)
    weight_map = dict(zip(present.tolist(), cw.tolist()))
    class_weights = {i: float(weight_map.get(i, 1.0)) for i in range(N_CLASSES)}

    model = build_model(lookback, n_features)
    history = model.fit(
        prep["X_tr"], y_tr,
        validation_data=(prep["X_va"], y_va),
        epochs=epochs,
        batch_size=config.BATCH_SIZE,
        callbacks=get_callbacks(lookback),
        class_weight=class_weights,
        verbose=1,
    )
    save_model(model, lookback)
    return model, history
