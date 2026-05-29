"""
Data Loader for TensorFlow/Keras.

Converts preprocessed numpy arrays into tf.data.Dataset objects
with batching, prefetching, and optional shuffling.
"""

import os
from typing import Tuple

import numpy as np
import tensorflow as tf

from src.config import get_config
from src.utils import logger


def create_tf_dataset(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True,
    buffer_size: int = 1000,
) -> tf.data.Dataset:
    """
    Create a tf.data.Dataset from numpy arrays.

    Args:
        X: Feature matrix
        y: Target vector
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle the data
        buffer_size: Shuffle buffer size

    Returns:
        Prefetched tf.data.Dataset
    """
    dataset = tf.data.Dataset.from_tensor_slices((X, y))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def load_preprocessed_data(processed_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load all preprocessed numpy arrays from disk.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    logger.info("Loading preprocessed data from {}", processed_dir)

    X_train = np.load(os.path.join(processed_dir, "X_train.npy"))
    X_val = np.load(os.path.join(processed_dir, "X_val.npy"))
    X_test = np.load(os.path.join(processed_dir, "X_test.npy"))
    y_train = np.load(os.path.join(processed_dir, "y_train.npy"))
    y_val = np.load(os.path.join(processed_dir, "y_val.npy"))
    y_test = np.load(os.path.join(processed_dir, "y_test.npy"))

    logger.info(
        "Loaded — Train: {}, Val: {}, Test: {}",
        X_train.shape, X_val.shape, X_test.shape,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def get_datasets(processed_dir: str = None, batch_size: int = None) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, np.ndarray, np.ndarray]:
    """
    Convenience function: load data and create all three tf datasets.

    Returns:
        train_dataset, val_dataset, test_dataset, X_test, y_test
    """
    config = get_config()
    processed_dir = processed_dir or config.paths.data_processed
    batch_size = batch_size or config.model.batch_size

    X_train, X_val, X_test, y_train, y_val, y_test = load_preprocessed_data(processed_dir)

    train_ds = create_tf_dataset(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_ds = create_tf_dataset(X_val, y_val, batch_size=batch_size, shuffle=False)
    test_ds = create_tf_dataset(X_test, y_test, batch_size=batch_size, shuffle=False)

    return train_ds, val_ds, test_ds, X_test, y_test


def get_input_dim(processed_dir: str = None) -> int:
    """Get the number of input features from training data shape."""
    processed_dir = processed_dir or get_config().paths.data_processed
    X_train = np.load(os.path.join(processed_dir, "X_train.npy"))
    return X_train.shape[1]
