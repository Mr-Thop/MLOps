"""
Stage 2: Data Processing
-------------------------
Resizes and normalizes the raw images, then splits them into
train / validation / test sets according to params.yaml.
"""
import os
import yaml
import numpy as np
import tensorflow as tf


def load_params(params_path="params.yaml"):
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def resize_images(images, size):
    images = tf.image.resize(images, (size, size)).numpy()
    return images.astype("float32")


def process_data(params):
    dc_cfg = params["data_collection"]
    dp_cfg = params["data_processing"]

    raw_dir = dc_cfg["raw_data_dir"]
    out_dir = dp_cfg["processed_data_dir"]
    os.makedirs(out_dir, exist_ok=True)

    images = np.load(os.path.join(raw_dir, "images.npy"))
    labels = np.load(os.path.join(raw_dir, "labels.npy"))

    if images.ndim == 3:  # grayscale datasets (e.g. MNIST) -> add channel dim
        images = images[..., np.newaxis]

    size = dp_cfg["image_size"]
    print(f"[data_processing] Resizing images to {size}x{size}")
    images = resize_images(images, size)

    if dp_cfg.get("normalize", True):
        images = images / 255.0

    n = images.shape[0]
    seed = dc_cfg.get("seed", 42)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)

    test_split = dp_cfg["test_split"]
    val_split = dp_cfg["validation_split"]

    n_test = int(n * test_split)
    n_val = int(n * val_split)
    n_train = n - n_test - n_val

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    splits = {
        "train": (images[train_idx], labels[train_idx]),
        "val": (images[val_idx], labels[val_idx]),
        "test": (images[test_idx], labels[test_idx]),
    }

    for split_name, (x, y) in splits.items():
        np.save(os.path.join(out_dir, f"x_{split_name}.npy"), x)
        np.save(os.path.join(out_dir, f"y_{split_name}.npy"), y)
        print(f"[data_processing] {split_name}: {x.shape[0]} samples")

    # Persist the augmentation config so it's visible to anyone inspecting
    # the processed data directory (actual augmentation is applied on the
    # fly during training if you extend model_training.py to use it).
    aug_cfg = dp_cfg.get("augmentation", {})
    with open(os.path.join(out_dir, "augmentation_config.yaml"), "w") as f:
        yaml.safe_dump(aug_cfg, f)

    print(f"[data_processing] Saved processed data to {out_dir}")


if __name__ == "__main__":
    params = load_params()
    process_data(params)
