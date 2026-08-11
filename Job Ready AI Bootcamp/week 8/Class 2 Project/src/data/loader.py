"""
MNIST dataset loading.

Loads the Keras-bundled MNIST dataset, normalizes pixel values to [0, 1],
reshapes images to the (H, W, C) format Conv2D layers expect, and carves a
validation split out of the training set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from tensorflow import keras

from config import DataConfig, get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MNISTData:
    """Container for a fully prepared train/val/test split."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def _normalize_and_reshape(images: np.ndarray, cfg: DataConfig) -> np.ndarray:
    """Scale pixels to [0, 1] float32 and add the channel dimension.

    HIGHLIGHTS: two separate things happen here and both matter for
    training stability:
      1. Dividing by 255.0 rescales raw 0-255 pixel intensities into
         [0, 1]. Neural nets train more reliably on small, zero-centered-ish
         input ranges — large raw pixel values (up to 255) would produce
         correspondingly large initial activations and gradients, making
         the optimizer's job harder and the loss landscape less stable.
      2. Reshaping from (N, 28, 28) to (N, 28, 28, 1) adds an explicit
         channel dimension. Keras' Conv2D layers always expect a channel
         axis (1 for grayscale, 3 for RGB) — raw MNIST arrays don't include
         it, so this step isn't optional bookkeeping, it's required for the
         data to even match the model's Input(shape=...) declaration.
    """
    images = images.astype("float32") / 255.0
    return images.reshape((-1, cfg.image_size, cfg.image_size, cfg.channels))


def load_mnist(cfg: DataConfig | None = None) -> MNISTData:
    """Load, normalize, reshape, and split the MNIST dataset.

    Args:
        cfg: Optional data configuration. Defaults to the global config.

    Returns:
        An ``MNISTData`` bundle with train/val/test splits ready for Keras.
    """
    cfg = cfg or get_config().data
    logger.info("Loading MNIST via keras.datasets.mnist ...")

    # keras.datasets.mnist.load_data() downloads (once, then caches under
    # ~/.keras/datasets/) and returns the classic 60k-train / 10k-test
    # split. Note that split is a TRAIN/TEST split only — MNIST does not
    # ship with a separate validation set, which is why we carve one out
    # of the training data ourselves below.
    (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train_full = _normalize_and_reshape(x_train_full, cfg)
    x_test = _normalize_and_reshape(x_test, cfg)
    y_train_full = y_train_full.astype("int64")
    y_test = y_test.astype("int64")

    # WHY we hold out a validation split at all: EarlyStopping and
    # ReduceLROnPlateau (src/training/trainer.py) both need a signal that
    # is NOT used for gradient updates to detect overfitting. If we
    # monitored performance on the training set itself, it would just keep
    # improving as the model memorizes it — never triggering a stop. The
    # validation set stands in for "unseen data" during training, while the
    # test set stays completely untouched until final evaluation, giving
    # us an unbiased read on true generalization.
    rng = np.random.default_rng(cfg.random_seed)
    n = x_train_full.shape[0]
    indices = rng.permutation(n)
    val_size = int(n * cfg.val_split)
    val_idx, train_idx = indices[:val_size], indices[val_size:]

    data = MNISTData(
        x_train=x_train_full[train_idx],
        y_train=y_train_full[train_idx],
        x_val=x_train_full[val_idx],
        y_val=y_train_full[val_idx],
        x_test=x_test,
        y_test=y_test,
    )

    logger.info(
        "MNIST loaded: train=%s val=%s test=%s",
        data.x_train.shape, data.x_val.shape, data.x_test.shape,
    )
    return data


def make_quick_subset(data: MNISTData, cfg: DataConfig | None = None) -> MNISTData:
    """Down-sample a full split into a small "quick demo" subset for fast training.

    Args:
        data: A full ``MNISTData`` split, typically from ``load_mnist``.
        cfg: Optional data configuration controlling subset sizes.

    Returns:
        A new ``MNISTData`` bundle with fewer samples per split, preserving
        the same relative val/test proportions.
    """
    cfg = cfg or get_config().data
    rng = np.random.default_rng(cfg.random_seed)

    def _sample(x: np.ndarray, y: np.ndarray, size: int) -> Tuple[np.ndarray, np.ndarray]:
        size = min(size, x.shape[0])
        idx = rng.choice(x.shape[0], size=size, replace=False)
        return x[idx], y[idx]

    train_size = cfg.quick_subset_train_size
    val_size = max(1, int(train_size * cfg.val_split))
    test_size = cfg.quick_subset_test_size

    x_train, y_train = _sample(data.x_train, data.y_train, train_size)
    x_val, y_val = _sample(data.x_val, data.y_val, val_size)
    x_test, y_test = _sample(data.x_test, data.y_test, test_size)

    return MNISTData(x_train, y_train, x_val, y_val, x_test, y_test)
