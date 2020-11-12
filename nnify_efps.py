from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
import h5py
import numpy as np
import pandas as pd
import glob
import os
import pathlib
path = pathlib.Path.cwd()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}
import tensorflow as tf


def nn(dfi):
    X, y = dfi["features"].values, dfi["targets"].values
    X_out = np.hstack(X)
    scaler = preprocessing.StandardScaler()
    X = scaler.fit_transform(X.reshape(-1, 1))
    model = tf.keras.Sequential()
    nodes = 25
    layers = 3
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False,
        name="Adam",
    )
    model.add(tf.keras.layers.Dense(nodes, input_dim=1))
    for _ in range(layers):
        model.add(
            tf.keras.layers.Dense(
                nodes,
                kernel_initializer="normal",
                activation="relu",
                kernel_constraint=tf.keras.constraints.MaxNorm(3),
            )
        )
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizer,
        metrics=[tf.keras.metrics.AUC()],
    )

    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", mode="auto", verbose=0, patience=5
    )

    model.fit(
        X,
        y,
        batch_size=256,
        epochs=100,
        verbose=0,
        validation_split=0.25,
        callbacks=[es],
    )
    predictions = np.hstack(model.predict(X))
    dfi_out = pd.DataFrame({"features": X_out, "nnify": predictions, "targets": y})
    return dfi_out


def nnify_efps():
    # Calculate HL variables from ET, eta, phi
    efp_path = path / "data" / "efp" 
    t = tqdm(list(pathlib.Path(efp_path).rglob("*.feather")))
    for efp_file in t:
        dfi = pd.read_feather(efp_file)
        t.set_description(f"Training: {efp_file.stem}")
        if True: #"nnify" not in dfi.columns:
            dfi_out = nn(dfi)
            auc_val = roc_auc_score(dfi_out["targets"].values, dfi_out["nnify"].values)
            dfi_out.to_feather(efp_file)
            print(efp_file, auc_val)


if __name__ == "__main__":
    nnify_efps()
