"""
Fast sanity checks for the MNIST pipeline.

These tests avoid any full training run — they only check shapes,
value ranges, and a single forward pass — so the whole suite finishes
in seconds. Run with: pytest tests/test_pipeline.py
"""

import numpy as np
import pytest

from config import get_config
from src.data.loader import load_mnist, subsample
from src.data.preprocessor import DigitPreprocessor
from src.models.cnn import build_baseline_cnn

CONFIG = get_config()


@pytest.fixture(scope="module")
def mnist_splits():
    """Load MNIST once and share it across tests in this module."""
    full = load_mnist(CONFIG)
    # Keep the fixture itself fast: shrink to a tiny slice up front.
    return subsample(full, fraction=0.01, config=CONFIG)


def test_loader_shapes_and_range(mnist_splits):
    height, width = CONFIG.data.image_size
    assert mnist_splits.x_train.shape[1:] == (height, width, CONFIG.data.num_channels)
    assert mnist_splits.x_val.shape[1:] == (height, width, CONFIG.data.num_channels)
    assert mnist_splits.x_test.shape[1:] == (height, width, CONFIG.data.num_channels)

    assert mnist_splits.x_train.dtype == np.float32
    assert 0.0 <= mnist_splits.x_train.min()
    assert mnist_splits.x_train.max() <= 1.0

    assert mnist_splits.y_train.ndim == 1
    assert set(np.unique(mnist_splits.y_test)).issubset(set(range(10)))


def test_loader_train_val_split_sizes(mnist_splits):
    assert len(mnist_splits.x_train) > 0
    assert len(mnist_splits.x_val) > 0
    assert len(mnist_splits.x_train) == len(mnist_splits.y_train)
    assert len(mnist_splits.x_val) == len(mnist_splits.y_val)


def test_subsample_rejects_invalid_fraction(mnist_splits):
    with pytest.raises(ValueError):
        subsample(mnist_splits, fraction=0.0, config=CONFIG)
    with pytest.raises(ValueError):
        subsample(mnist_splits, fraction=1.5, config=CONFIG)


def test_preprocessor_from_array_shape_and_range():
    preprocessor = DigitPreprocessor(CONFIG)
    canvas_like = np.zeros((280, 280, 4), dtype=np.uint8)
    canvas_like[100:180, 100:180, :3] = 255
    canvas_like[100:180, 100:180, 3] = 255

    tensor = preprocessor.from_array(canvas_like)

    height, width = CONFIG.data.image_size
    assert tensor.shape == (1, height, width, CONFIG.data.num_channels)
    assert tensor.dtype == np.float32
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0


def test_preprocessor_grayscale_input():
    preprocessor = DigitPreprocessor(CONFIG)
    grayscale = np.full((280, 280), 200, dtype=np.uint8)

    tensor = preprocessor.from_array(grayscale)

    height, width = CONFIG.data.image_size
    assert tensor.shape == (1, height, width, CONFIG.data.num_channels)


def test_build_baseline_cnn_io_shapes():
    model = build_baseline_cnn(CONFIG)

    height, width = CONFIG.data.image_size
    assert model.input_shape == (None, height, width, CONFIG.data.num_channels)
    assert model.output_shape == (None, CONFIG.data.num_classes)


def test_model_forward_pass_smoke():
    """One forward pass on random data — no training, just wiring sanity."""
    model = build_baseline_cnn(CONFIG)
    height, width = CONFIG.data.image_size
    dummy_batch = np.random.rand(4, height, width, CONFIG.data.num_channels).astype("float32")

    predictions = model.predict(dummy_batch, verbose=0)

    assert predictions.shape == (4, CONFIG.data.num_classes)
    # Softmax outputs must sum to ~1 per row.
    row_sums = predictions.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-4)
