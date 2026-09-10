"""
Unit tests for the ML data pipeline (Stage 1: data_collection.py,
Stage 2: data_processing.py).

All tests run offline: the real Keras dataset loaders (which hit the
network) are mocked out with small synthetic arrays, and all file I/O
happens inside pytest's `tmp_path` fixture so nothing touches the repo.

Run with:
    pytest test_pipeline.py -v
"""
import os
import sys
from unittest.mock import patch

import numpy as np
import pytest
import yaml

# Allow running this file from anywhere in the repo.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import data_collection
from src import data_processing


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_rgb_dataset():
    """Tiny fake dataset shaped like cifar10.load_data()'s output."""
    x_train = np.random.randint(0, 256, size=(20, 32, 32, 3), dtype=np.uint8)
    y_train = np.random.randint(0, 10, size=(20, 1), dtype=np.uint8)
    x_test = np.random.randint(0, 256, size=(5, 32, 32, 3), dtype=np.uint8)
    y_test = np.random.randint(0, 10, size=(5, 1), dtype=np.uint8)
    return (x_train, y_train), (x_test, y_test)


@pytest.fixture
def base_params(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    return {
        "data_collection": {
            "dataset_name": "cifar10",
            "raw_data_dir": str(raw_dir),
            "seed": 42,
        },
        "data_processing": {
            "processed_data_dir": str(processed_dir),
            "image_size": 16,
            "normalize": True,
            "test_split": 0.2,
            "validation_split": 0.2,
            "augmentation": {"horizontal_flip": True},
        },
    }


@pytest.fixture
def params_yaml_file(tmp_path, base_params):
    path = tmp_path / "params.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(base_params, f)
    return path


def _write_raw_data(raw_dir, images, labels):
    os.makedirs(raw_dir, exist_ok=True)
    np.save(os.path.join(raw_dir, "images.npy"), images)
    np.save(os.path.join(raw_dir, "labels.npy"), labels)


# ---------------------------------------------------------------------------
# load_params (shared pattern in both stages)
# ---------------------------------------------------------------------------

class TestLoadParams:
    def test_data_collection_load_params_reads_yaml(self, params_yaml_file, base_params):
        loaded = data_collection.load_params(str(params_yaml_file))
        assert loaded == base_params

    def test_data_processing_load_params_reads_yaml(self, params_yaml_file, base_params):
        loaded = data_processing.load_params(str(params_yaml_file))
        assert loaded == base_params

    def test_load_params_missing_file_raises(self, tmp_path):
        missing = tmp_path / "does_not_exist.yaml"
        with pytest.raises(FileNotFoundError):
            data_collection.load_params(str(missing))


# ---------------------------------------------------------------------------
# Stage 1: data_collection.collect_data
# ---------------------------------------------------------------------------

class TestCollectData:
    def test_unsupported_dataset_raises_value_error(self, base_params):
        base_params["data_collection"]["dataset_name"] = "not_a_real_dataset"
        with pytest.raises(ValueError):
            data_collection.collect_data(base_params)

    def test_collect_data_saves_expected_arrays(self, base_params, fake_rgb_dataset):
        with patch.dict(
            data_collection.DATASET_LOADERS,
            {"cifar10": lambda: fake_rgb_dataset},
        ):
            data_collection.collect_data(base_params)

        raw_dir = base_params["data_collection"]["raw_data_dir"]
        images_path = os.path.join(raw_dir, "images.npy")
        labels_path = os.path.join(raw_dir, "labels.npy")
        assert os.path.exists(images_path)
        assert os.path.exists(labels_path)

        images = np.load(images_path)
        labels = np.load(labels_path)

        (x_train, y_train), (x_test, y_test) = fake_rgb_dataset
        expected_n = x_train.shape[0] + x_test.shape[0]

        assert images.shape[0] == expected_n
        assert labels.shape[0] == expected_n
        # collect_data() does y.reshape(-1) -> labels must come out 1-D
        assert labels.ndim == 1

    def test_collect_data_creates_raw_dir_if_missing(self, base_params, fake_rgb_dataset):
        raw_dir = base_params["data_collection"]["raw_data_dir"]
        assert not os.path.exists(raw_dir)

        with patch.dict(
            data_collection.DATASET_LOADERS,
            {"cifar10": lambda: fake_rgb_dataset},
        ):
            data_collection.collect_data(base_params)

        assert os.path.isdir(raw_dir)


# ---------------------------------------------------------------------------
# Stage 2: data_processing.resize_images
# ---------------------------------------------------------------------------

class TestResizeImages:
    def test_resize_changes_spatial_dimensions(self):
        images = np.random.randint(0, 256, size=(4, 32, 32, 3)).astype("float32")
        resized = data_processing.resize_images(images, 16)
        assert resized.shape == (4, 16, 16, 3)

    def test_resize_output_dtype_is_float32(self):
        images = np.random.randint(0, 256, size=(2, 10, 10, 3)).astype("float32")
        resized = data_processing.resize_images(images, 8)
        assert resized.dtype == np.float32

    def test_resize_grayscale_with_channel_dim(self):
        images = np.random.randint(0, 256, size=(3, 28, 28, 1)).astype("float32")
        resized = data_processing.resize_images(images, 14)
        assert resized.shape == (3, 14, 14, 1)


# ---------------------------------------------------------------------------
# Stage 2: data_processing.process_data
# ---------------------------------------------------------------------------

class TestProcessData:
    def test_process_data_splits_and_shapes(self, base_params):
        raw_dir = base_params["data_collection"]["raw_data_dir"]
        n_samples = 100
        images = np.random.randint(0, 256, size=(n_samples, 32, 32, 3)).astype("uint8")
        labels = np.random.randint(0, 10, size=(n_samples,)).astype("int64")
        _write_raw_data(raw_dir, images, labels)

        data_processing.process_data(base_params)

        out_dir = base_params["data_processing"]["processed_data_dir"]
        x_train = np.load(os.path.join(out_dir, "x_train.npy"))
        x_val = np.load(os.path.join(out_dir, "x_val.npy"))
        x_test = np.load(os.path.join(out_dir, "x_test.npy"))
        y_train = np.load(os.path.join(out_dir, "y_train.npy"))
        y_val = np.load(os.path.join(out_dir, "y_val.npy"))
        y_test = np.load(os.path.join(out_dir, "y_test.npy"))

        # No samples lost or duplicated across splits.
        assert x_train.shape[0] + x_val.shape[0] + x_test.shape[0] == n_samples
        assert y_train.shape[0] == x_train.shape[0]
        assert y_val.shape[0] == x_val.shape[0]
        assert y_test.shape[0] == x_test.shape[0]

        # Resized correctly.
        size = base_params["data_processing"]["image_size"]
        assert x_train.shape[1:3] == (size, size)

    def test_process_data_normalizes_to_unit_range(self, base_params):
        raw_dir = base_params["data_collection"]["raw_data_dir"]
        images = np.full((10, 32, 32, 3), 255, dtype="uint8")
        labels = np.zeros((10,), dtype="int64")
        _write_raw_data(raw_dir, images, labels)

        data_processing.process_data(base_params)

        out_dir = base_params["data_processing"]["processed_data_dir"]
        parts = [
            np.load(os.path.join(out_dir, f"x_{split}.npy"))
            for split in ("train", "val", "test")
        ]
        combined = np.concatenate([p for p in parts if p.size], axis=0)
        assert combined.max() <= 1.0
        assert combined.min() >= 0.0

    def test_process_data_skips_normalization_when_disabled(self, base_params):
        base_params["data_processing"]["normalize"] = False
        raw_dir = base_params["data_collection"]["raw_data_dir"]
        images = np.full((10, 32, 32, 3), 200, dtype="uint8")
        labels = np.zeros((10,), dtype="int64")
        _write_raw_data(raw_dir, images, labels)

        data_processing.process_data(base_params)

        out_dir = base_params["data_processing"]["processed_data_dir"]
        x_train = np.load(os.path.join(out_dir, "x_train.npy"))
        if x_train.size:
            assert x_train.max() > 1.0

    def test_process_data_adds_channel_dim_for_grayscale(self, base_params):
        raw_dir = base_params["data_collection"]["raw_data_dir"]
        images = np.random.randint(0, 256, size=(10, 28, 28)).astype("uint8")
        labels = np.zeros((10,), dtype="int64")
        _write_raw_data(raw_dir, images, labels)

        data_processing.process_data(base_params)

        out_dir = base_params["data_processing"]["processed_data_dir"]
        x_train = np.load(os.path.join(out_dir, "x_train.npy"))
        if x_train.size:
            assert x_train.shape[-1] == 1

    def test_process_data_writes_augmentation_config(self, base_params):
        raw_dir = base_params["data_collection"]["raw_data_dir"]
        images = np.random.randint(0, 256, size=(10, 32, 32, 3)).astype("uint8")
        labels = np.zeros((10,), dtype="int64")
        _write_raw_data(raw_dir, images, labels)

        data_processing.process_data(base_params)

        out_dir = base_params["data_processing"]["processed_data_dir"]
        aug_path = os.path.join(out_dir, "augmentation_config.yaml")
        assert os.path.exists(aug_path)

        with open(aug_path) as f:
            saved_cfg = yaml.safe_load(f)
        assert saved_cfg == base_params["data_processing"]["augmentation"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
