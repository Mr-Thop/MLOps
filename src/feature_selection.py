"""
Stage 3: Feature Selection
----------------------------
Flattens the processed images and reduces dimensionality using either
PCA or a variance-threshold filter, as configured in params.yaml.
"""
import os
import json
import yaml
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold


def load_params(params_path="params.yaml"):
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def flatten(images):
    return images.reshape(images.shape[0], -1)


def select_features(params):
    dp_cfg = params["data_processing"]
    fs_cfg = params["feature_selection"]

    proc_dir = dp_cfg["processed_data_dir"]
    feat_dir = fs_cfg["features_dir"]
    os.makedirs(feat_dir, exist_ok=True)

    method = fs_cfg["method"]

    x_train = flatten(np.load(os.path.join(proc_dir, "x_train.npy")))
    x_val = flatten(np.load(os.path.join(proc_dir, "x_val.npy")))
    x_test = flatten(np.load(os.path.join(proc_dir, "x_test.npy")))

    print(f"[feature_selection] Method: {method}, input dim: {x_train.shape[1]}")

    if method == "pca":
        n_components = min(fs_cfg["pca_n_components"], x_train.shape[0], x_train.shape[1])
        selector = PCA(n_components=n_components, random_state=42)
        x_train_sel = selector.fit_transform(x_train)
        x_val_sel = selector.transform(x_val)
        x_test_sel = selector.transform(x_test)
        explained = float(np.sum(selector.explained_variance_ratio_))
        info = {"method": "pca", "n_components": n_components, "explained_variance": explained}

    elif method == "variance_threshold":
        threshold = fs_cfg["variance_threshold"]
        selector = VarianceThreshold(threshold=threshold)
        x_train_sel = selector.fit_transform(x_train)
        x_val_sel = selector.transform(x_val)
        x_test_sel = selector.transform(x_test)
        info = {
            "method": "variance_threshold",
            "threshold": threshold,
            "n_selected_features": int(x_train_sel.shape[1]),
        }
    else:
        raise ValueError(f"Unsupported feature_selection method: {method}")

    np.save(os.path.join(feat_dir, "x_train.npy"), x_train_sel)
    np.save(os.path.join(feat_dir, "x_val.npy"), x_val_sel)
    np.save(os.path.join(feat_dir, "x_test.npy"), x_test_sel)

    with open(os.path.join(feat_dir, "feature_selection_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print(f"[feature_selection] Output dim: {x_train_sel.shape[1]}")
    print(f"[feature_selection] Saved features to {feat_dir}")


if __name__ == "__main__":
    params = load_params()
    select_features(params)
