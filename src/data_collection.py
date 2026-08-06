"""
Stage 1: Data Collection
-------------------------
Loads a raw image dataset (from tf.keras.datasets) and saves the images
and labels to disk as numpy arrays for the downstream pipeline stages.
"""
import os
import yaml
import numpy as np
from tensorflow.keras.datasets import cifar10, mnist, fashion_mnist


def load_params(params_path="params.yaml"):
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


DATASET_LOADERS = {
    "cifar10": cifar10.load_data,
    "mnist": mnist.load_data,
    "fashion_mnist": fashion_mnist.load_data,
}


def collect_data(params):
    cfg = params["data_collection"]
    dataset_name = cfg["dataset_name"]
    raw_dir = cfg["raw_data_dir"]
    seed = cfg.get("seed", 42)

    np.random.seed(seed)
    os.makedirs(raw_dir, exist_ok=True)

    if dataset_name not in DATASET_LOADERS:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. "
            f"Choose one of {list(DATASET_LOADERS.keys())}"
        )

    print(f"[data_collection] Loading dataset: {dataset_name}")
    (x_train, y_train), (x_test, y_test) = DATASET_LOADERS[dataset_name]()

    # Combine train/test splits here; the processing stage owns the
    # official train/val/test split so it stays reproducible and tunable.
    x = np.concatenate([x_train, x_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0).reshape(-1)

    np.save(os.path.join(raw_dir, "images.npy"), x)
    np.save(os.path.join(raw_dir, "labels.npy"), y)

    print(f"[data_collection] Saved {x.shape[0]} images to {raw_dir}")
    print(f"[data_collection] Image shape: {x.shape[1:]}, classes: {len(np.unique(y))}")


if __name__ == "__main__":
    params = load_params()
    collect_data(params)
