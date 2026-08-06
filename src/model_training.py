"""
Stage 4: Model Training
-------------------------
Builds and trains a configurable MLP classifier on the selected
features, using hyperparameters defined in params.yaml.
"""
import os
import json
import yaml
import numpy as np
from tensorflow.keras import layers, models, optimizers


def load_params(params_path="params.yaml"):
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def build_optimizer(name, lr):
    name = name.lower()
    if name == "adam":
        return optimizers.Adam(learning_rate=lr)
    if name == "sgd":
        return optimizers.SGD(learning_rate=lr, momentum=0.9)
    if name == "rmsprop":
        return optimizers.RMSprop(learning_rate=lr)
    raise ValueError(f"Unsupported optimizer: {name}")


def build_model(input_dim, num_classes, cfg):
    model = models.Sequential(name="mlp_classifier")
    model.add(layers.Input(shape=(input_dim,)))
    for units in cfg["hidden_units"]:
        model.add(layers.Dense(units, activation=cfg["activation"]))
        model.add(layers.Dropout(cfg["dropout_rate"]))
    model.add(layers.Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer=build_optimizer(cfg["optimizer"], cfg["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(params):
    fs_cfg = params["feature_selection"]
    mt_cfg = params["model_training"]
    dp_cfg = params["data_processing"]

    feat_dir = fs_cfg["features_dir"]
    proc_dir = dp_cfg["processed_data_dir"]
    model_dir = mt_cfg["model_dir"]
    os.makedirs(model_dir, exist_ok=True)

    x_train = np.load(os.path.join(feat_dir, "x_train.npy"))
    x_val = np.load(os.path.join(feat_dir, "x_val.npy"))
    y_train = np.load(os.path.join(proc_dir, "y_train.npy"))
    y_val = np.load(os.path.join(proc_dir, "y_val.npy"))

    num_classes = int(max(y_train.max(), y_val.max())) + 1
    model = build_model(x_train.shape[1], num_classes, mt_cfg)
    model.summary()

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        batch_size=mt_cfg["batch_size"],
        epochs=mt_cfg["epochs"],
        verbose=2,
    )

    model_path = os.path.join(model_dir, "model.keras")
    model.save(model_path)

    with open(os.path.join(model_dir, "history.json"), "w") as f:
        json.dump(
            {k: [float(v) for v in vals] for k, vals in history.history.items()},
            f, indent=2,
        )

    print(f"[model_training] Saved model to {model_path}")


if __name__ == "__main__":
    params = load_params()
    train_model(params)
