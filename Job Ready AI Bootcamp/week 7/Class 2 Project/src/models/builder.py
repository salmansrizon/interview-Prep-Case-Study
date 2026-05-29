"""
Neural Network Model Builder using TensorFlow/Keras.

Constructs a configurable Sequential model with:
    - Dense hidden layers
    - Batch Normalization
    - Dropout regularization
    - He initialization
"""

import os
from typing import List

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.config import get_config
from src.utils import logger


def build_model(
    input_dim: int = None,
    hidden_units: List[int] = None,
    dropout_rate: float = None,
    learning_rate: float = None,
    loss_function: str = None,
    metrics: List[str] = None,
    optimizer_name: str = None,
) -> keras.Model:
    """
    Build a configurable deep neural network for regression.

    Architecture (default):
        Input → Dense(128) → BatchNorm → Dropout(0.3)
              → Dense(64)  → BatchNorm → Dropout(0.3)
              → Dense(32)  → BatchNorm → Dropout(0.3)
              → Dense(1)   (linear output)

    Returns:
        Compiled Keras model ready for training.
    """
    config = get_config()

    input_dim = input_dim or config.model.input_dim
    hidden_units = hidden_units or config.model.hidden_units
    dropout_rate = dropout_rate if dropout_rate is not None else config.model.dropout_rate
    learning_rate = learning_rate if learning_rate is not None else config.model.learning_rate
    loss_function = loss_function or config.training.loss_function
    metrics = metrics or config.training.metrics
    optimizer_name = optimizer_name or config.training.optimizer

    logger.info(
        "Building model — input_dim={}, hidden={}, dropout={}, lr={}",
        input_dim, hidden_units, dropout_rate, learning_rate,
    )

    model = keras.Sequential(name="EquipmentSuccessPredictor")

    # Input layer
    model.add(layers.Input(shape=(input_dim,), name="input"))

    # Hidden layers
    for i, units in enumerate(hidden_units):
        model.add(layers.Dense(
            units,
            activation="relu",
            kernel_initializer="he_normal",
            name=f"dense_{i+1}",
        ))
        model.add(layers.BatchNormalization(name=f"batchnorm_{i+1}"))
        model.add(layers.Dropout(dropout_rate, name=f"dropout_{i+1}"))

    # Output layer (regression: single neuron, linear activation)
    model.add(layers.Dense(1, activation="linear", name="output"))

    # Optimizer
    if optimizer_name.lower() == "adam":
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name.lower() == "sgd":
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    elif optimizer_name.lower() == "rmsprop":
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    # Compile
    model.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=metrics,
    )

    logger.info("Model built successfully. Total parameters: {:,}", model.count_params())
    return model


def get_callbacks(config=None) -> List[keras.callbacks.Callback]:
    """
    Create training callbacks:
        - EarlyStopping
        - ModelCheckpoint
        - ReduceLROnPlateau
        - TensorBoard
    """
    config = config or get_config()
    os.makedirs(config.paths.logs_dir, exist_ok=True)
    os.makedirs(config.paths.models_dir, exist_ok=True)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.model.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(config.paths.models_dir, "best_model.keras"),
            monitor=config.model.checkpoint_monitor,
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(config.paths.logs_dir, "tensorboard"),
            histogram_freq=1,
            update_freq="epoch",
        ),
    ]

    return callbacks


def print_model_summary(model: keras.Model) -> None:
    """Print model architecture summary."""
    model.summary()
