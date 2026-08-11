"""
MNIST data ingestion layer.

Loads the Keras-bundled MNIST dataset, normalizes pixel values, reshapes
images to the `(N, H, W, C)` format Conv2D layers expect, and carves out
a validation split from the training data.

HIGHLIGHTS — why does this live in its own module instead of inline in
the trainer or the app? Every consumer of MNIST data (the trainer, the EDA notebook, the
"Model Insights" tab's confusion matrix, and the test suite) needs the
*exact same* normalization and reshaping applied consistently. Centralizing
it here means there is exactly one place that can get the pixel scaling or
the channel dimension wrong, instead of four.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from tensorflow import keras

from config import Config, get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MnistSplits:
    """Container for the train/validation/test arrays."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def _normalize_and_reshape(images: np.ndarray, config: Config) -> np.ndarray:
    """Scale pixels to ``[0, 1]`` and add the channel dimension.

    Two separate transformations happen here, and both matter:

    1. **Reshape** — `keras.datasets.mnist.load_data()` returns images as
       plain `(N, 28, 28)` arrays. Conv2D layers require an explicit
       channel axis — `(N, 28, 28, 1)` — even when there is only one
       channel (grayscale), because the same layer type is used for RGB
       images where that axis would be 3. Without this reshape, building
       the model in `src/models/cnn.py` would fail with a shape mismatch.

    2. **Normalize** — dividing by `pixel_max_value` (255.0) rescales raw
       `uint8` pixels from `[0, 255]` down to `float32` values in
       `[0, 1]`. As covered in the Class 1 lecture, skipping this step
       risks unstable/exploding gradients during backpropagation and
       slows optimizer convergence, because the loss landscape is much
       harder to navigate when inputs sit on a large, arbitrary scale.
    """
    height, width = config.data.image_size
    reshaped = images.reshape(-1, height, width, config.data.num_channels)
    return reshaped.astype("float32") / config.data.pixel_max_value


def load_mnist(config: Config | None = None) -> MnistSplits:
    """Load MNIST and return normalized train/validation/test splits.

    Args:
        config: Optional injected config (defaults to the global singleton).

    Returns:
        An :class:`MnistSplits` instance with float32 image arrays of
        shape ``(N, 28, 28, 1)`` in the ``[0, 1]`` range and integer
        label arrays of shape ``(N,)``.
    """
    config = config or get_config()
    logger.info("Loading MNIST dataset via keras.datasets.mnist ...")

    # Keras ships MNIST already split into "60k train" / "10k test" by the
    # dataset's original authors. We treat that test set as sacred — it is
    # only ever touched once, at final evaluation time in trainer.py — so
    # accuracy numbers reported to the student are never contaminated by
    # accidentally training or tuning on data the model has already seen.
    (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train_full = _normalize_and_reshape(x_train_full, config)
    x_test = _normalize_and_reshape(x_test, config)

    # The validation split comes out of the 60k *training* images, not the
    # test set. Its job is different from the test set's: it is checked
    # after every epoch (via `validation_data=` in trainer.py) so we can
    # watch for overfitting and let EarlyStopping react to it, while the
    # test set stays reserved for a single, final, unbiased accuracy number.
    rng = np.random.default_rng(config.data.random_state)
    n_samples = x_train_full.shape[0]
    permutation = rng.permutation(n_samples)
    n_val = int(n_samples * config.data.val_split)

    val_idx, train_idx = permutation[:n_val], permutation[n_val:]

    splits = MnistSplits(
        x_train=x_train_full[train_idx],
        y_train=y_train_full[train_idx],
        x_val=x_train_full[val_idx],
        y_val=y_train_full[val_idx],
        x_test=x_test,
        y_test=y_test,
    )

    logger.info(
        "MNIST loaded — train=%d, val=%d, test=%d",
        len(splits.x_train), len(splits.x_val), len(splits.x_test),
    )
    return splits


def subsample(splits: MnistSplits, fraction: float, config: Config | None = None) -> MnistSplits:
    """Return a random subset of the training/validation data.

    Used by the Streamlit "quick demo" toggle so a live training run in
    the browser finishes in seconds instead of minutes. The test set is
    left untouched so evaluation numbers stay meaningful.

    Args:
        splits: Full-size splits returned by :func:`load_mnist`.
        fraction: Fraction of train/val samples to keep, in ``(0, 1]``.
        config: Optional injected config (defaults to the global singleton).
    """
    if not 0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    config = config or get_config()
    rng = np.random.default_rng(config.data.random_state)

    def _take(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Random (not sequential) sampling matters here: MNIST's stored
        # order is not shuffled by digit, and grabbing the "first N" rows
        # could easily produce a subset that is missing some digits
        # entirely, which would silently break both training and the
        # quick-demo's accuracy numbers.
        n_keep = max(1, int(len(x) * fraction))
        idx = rng.choice(len(x), size=n_keep, replace=False)
        return x[idx], y[idx]

    x_train, y_train = _take(splits.x_train, splits.y_train)
    x_val, y_val = _take(splits.x_val, splits.y_val)

    logger.info(
        "Subsampled to fraction=%.2f — train=%d, val=%d",
        fraction, len(x_train), len(x_val),
    )

    return MnistSplits(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=splits.x_test,
        y_test=splits.y_test,
    )
