"""
Stage 5: Model Evaluation
----------------------------
Evaluates the trained model on the held-out test set, writes
metrics to a JSON file (tracked by `dvc metrics`), and saves a
confusion matrix plot to the reports directory.
"""
import os
import json
import yaml
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_params(params_path="params.yaml"):
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def evaluate_model(params):
    fs_cfg = params["feature_selection"]
    dp_cfg = params["data_processing"]
    mt_cfg = params["model_training"]
    me_cfg = params["model_evaluation"]

    feat_dir = fs_cfg["features_dir"]
    proc_dir = dp_cfg["processed_data_dir"]
    model_dir = mt_cfg["model_dir"]

    x_test = np.load(os.path.join(feat_dir, "x_test.npy"))
    y_test = np.load(os.path.join(proc_dir, "y_test.npy"))

    model = tf.keras.models.load_model(os.path.join(model_dir, "model.keras"))
    y_prob = model.predict(x_test)
    y_pred = np.argmax(y_prob, axis=1)

    requested = me_cfg.get("metrics", ["accuracy"])
    results = {}
    if "accuracy" in requested:
        results["accuracy"] = float(accuracy_score(y_test, y_pred))
    if "precision" in requested:
        results["precision"] = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    if "recall" in requested:
        results["recall"] = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    if "f1_score" in requested:
        results["f1_score"] = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    metrics_file = me_cfg["metrics_file"]
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    with open(metrics_file, "w") as f:
        json.dump(results, f, indent=2)

    reports_dir = me_cfg["reports_dir"]
    os.makedirs(reports_dir, exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.savefig(os.path.join(reports_dir, "confusion_matrix.png"))
    plt.close()

    print(f"[model_evaluation] Metrics: {results}")
    print(f"[model_evaluation] Saved metrics to {metrics_file}")


if __name__ == "__main__":
    params = load_params()
    evaluate_model(params)
