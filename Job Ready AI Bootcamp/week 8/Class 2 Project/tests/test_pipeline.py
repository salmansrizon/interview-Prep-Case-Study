"""
Fast, offline pytest suite for the MNIST optimization pipeline.

Deliberately avoids any real training — every test either uses tiny
synthetic tensors or checks architecture/shape properties of freshly built
(untrained) models, so the whole suite runs in seconds.

HIGHLIGHTS: notice what these tests check and what they deliberately
DON'T. They verify *structure* (shapes, presence/absence of specific layer
types, that a forward pass doesn't crash) rather than *accuracy* — you
cannot assert "this model gets >98% accuracy" in a fast unit test, because
that requires real training on real data, which takes minutes and would
make the test suite painfully slow to run on every code change. Structural
tests like "does the optimized model actually contain BatchNormalization
and Dropout layers" still catch the most common regression in a project
like this: someone edits architectures.py and accidentally removes the
layer that was supposed to be the whole point of that variant.
"""

from __future__ import annotations

import numpy as np
import pytest
from tensorflow.keras.layers import BatchNormalization, Dropout

from config import get_config
from src.data.loader import MNISTData, make_quick_subset
from src.data.preprocessor import preprocess_batch, preprocess_canvas_image
from src.models.architectures import (
    build_baseline_cnn,
    build_dropout_cnn,
    build_optimized_cnn,
    build_variant,
)

CONFIG = get_config()


# ── Fixtures ─────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def synthetic_mnist_data() -> MNISTData:
    """A tiny, fully synthetic MNIST-shaped dataset (no real data download)."""
    rng = np.random.default_rng(0)
    size = CONFIG.data.image_size
    n_train, n_val, n_test = 40, 10, 10

    def _make(n: int):
        x = rng.random((n, size, size, CONFIG.data.channels)).astype("float32")
        y = rng.integers(0, CONFIG.data.num_classes, size=n).astype("int64")
        return x, y

    x_train, y_train = _make(n_train)
    x_val, y_val = _make(n_val)
    x_test, y_test = _make(n_test)
    return MNISTData(x_train, y_train, x_val, y_val, x_test, y_test)


# ── Data loader ──────────────────────────────────────────────────────
class TestDataLoader:
    def test_quick_subset_shapes(self, synthetic_mnist_data: MNISTData) -> None:
        subset = make_quick_subset(synthetic_mnist_data)
        size = CONFIG.data.image_size
        assert subset.x_train.shape[1:] == (size, size, CONFIG.data.channels)
        assert subset.x_train.shape[0] == subset.y_train.shape[0]
        assert subset.x_val.shape[0] == subset.y_val.shape[0]
        assert subset.x_test.shape[0] == subset.y_test.shape[0]

    def test_quick_subset_respects_input_size(self, synthetic_mnist_data: MNISTData) -> None:
        subset = make_quick_subset(synthetic_mnist_data)
        # Synthetic fixture is smaller than the configured quick-subset size,
        # so the subset should never exceed the source split size.
        assert subset.x_train.shape[0] <= synthetic_mnist_data.x_train.shape[0]


# ── Preprocessor ─────────────────────────────────────────────────────
class TestPreprocessor:
    def test_output_shape_from_array(self) -> None:
        size = CONFIG.data.image_size
        raw = (np.random.rand(64, 64, 3) * 255).astype("uint8")
        tensor = preprocess_canvas_image(raw)
        assert tensor.shape == (1, size, size, CONFIG.data.channels)

    def test_output_is_normalized(self) -> None:
        raw = (np.random.rand(40, 40) * 255).astype("uint8")
        tensor = preprocess_canvas_image(raw)
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0
        assert tensor.dtype == np.float32

    def test_dark_background_not_inverted(self) -> None:
        # Mostly-black image (mean < 127) should keep its polarity.
        dark = np.zeros((28, 28), dtype="uint8")
        dark[10:18, 10:18] = 255
        tensor = preprocess_canvas_image(dark)
        assert tensor.max() > 0.5

    def test_light_background_is_inverted(self) -> None:
        # Mostly-white image (mean > 127) should be inverted to white-on-black.
        light = np.full((28, 28), 255, dtype="uint8")
        light[10:18, 10:18] = 0
        tensor = preprocess_canvas_image(light)
        # After inversion, the digit stroke (originally 0) becomes bright.
        assert tensor[0, 12, 12, 0] > 0.5

    def test_preprocess_batch_stacks_correctly(self) -> None:
        images = [(np.random.rand(28, 28) * 255).astype("uint8") for _ in range(3)]
        batch = preprocess_batch(images)
        size = CONFIG.data.image_size
        assert batch.shape == (3, size, size, CONFIG.data.channels)


# ── Model architectures ──────────────────────────────────────────────
class TestArchitectures:
    @pytest.mark.parametrize(
        "builder",
        [build_baseline_cnn, build_dropout_cnn, build_optimized_cnn],
    )
    def test_input_output_shape(self, builder) -> None:
        model = builder()
        assert model.input_shape == (None, *CONFIG.data.input_shape)
        assert model.output_shape == (None, CONFIG.data.num_classes)

    def test_optimized_has_batchnorm_and_dropout(self) -> None:
        model = build_optimized_cnn()
        has_batchnorm = any(isinstance(layer, BatchNormalization) for layer in model.layers)
        has_dropout = any(isinstance(layer, Dropout) for layer in model.layers)
        assert has_batchnorm, "Fully Optimized CNN must contain BatchNormalization layers"
        assert has_dropout, "Fully Optimized CNN must contain Dropout layers"

    def test_dropout_variant_has_dropout_but_no_batchnorm(self) -> None:
        model = build_dropout_cnn()
        has_batchnorm = any(isinstance(layer, BatchNormalization) for layer in model.layers)
        has_dropout = any(isinstance(layer, Dropout) for layer in model.layers)
        assert has_dropout
        assert not has_batchnorm

    def test_baseline_has_no_regularization_layers(self) -> None:
        model = build_baseline_cnn()
        has_batchnorm = any(isinstance(layer, BatchNormalization) for layer in model.layers)
        has_dropout = any(isinstance(layer, Dropout) for layer in model.layers)
        assert not has_batchnorm
        assert not has_dropout

    def test_build_variant_dispatch(self) -> None:
        for name in ("baseline", "dropout", "optimized"):
            model = build_variant(name)
            assert model.output_shape == (None, CONFIG.data.num_classes)

    def test_build_variant_rejects_unknown_name(self) -> None:
        with pytest.raises(ValueError):
            build_variant("not_a_real_variant")

    @pytest.mark.parametrize(
        "builder",
        [build_baseline_cnn, build_dropout_cnn, build_optimized_cnn],
    )
    def test_forward_pass_smoke(self, builder, synthetic_mnist_data: MNISTData) -> None:
        """One forward pass per architecture, no training, must not raise."""
        model = builder()
        batch = synthetic_mnist_data.x_train[:4]
        predictions = model.predict(batch, verbose=0)
        assert predictions.shape == (4, CONFIG.data.num_classes)
        # Softmax outputs should sum to ~1 per row.
        np.testing.assert_allclose(predictions.sum(axis=1), 1.0, atol=1e-4)
