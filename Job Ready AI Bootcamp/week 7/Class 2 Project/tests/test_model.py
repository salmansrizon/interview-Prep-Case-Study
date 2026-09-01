"""
Tests for model builder and architecture.
"""

import pytest
import numpy as np
import tensorflow as tf

from src.models.builder import build_model, get_callbacks
from src.config import get_config


class TestModelBuilder:
    """Test suite for neural network model."""

    def test_model_is_compiled(self):
        """Built model should be compiled with optimizer and loss."""
        model = build_model(input_dim=19)
        assert model.optimizer is not None
        assert model.loss is not None

    def test_model_has_correct_input_shape(self):
        """Model input layer should match specified dimensions."""
        for dim in [5, 19, 50]:
            model = build_model(input_dim=dim)
            assert model.input_shape == (None, dim)

    def test_model_output_is_single_neuron(self):
        """Output layer should have exactly 1 neuron."""
        model = build_model(input_dim=10)
        assert model.output_shape == (None, 1)

    def test_model_has_dropout_layers(self):
        """Model should contain Dropout layers."""
        model = build_model(input_dim=10, dropout_rate=0.3)
        layer_names = [layer.name for layer in model.layers]
        assert any("dropout" in name for name in layer_names)

    def test_model_has_batchnorm_layers(self):
        """Model should contain BatchNormalization layers."""
        model = build_model(input_dim=10)
        layer_names = [layer.name for layer in model.layers]
        assert any("batchnorm" in name for name in layer_names)

    def test_model_parameters_positive(self):
        """Model should have trainable parameters."""
        model = build_model(input_dim=19)
        assert model.count_params() > 0
        assert sum([w.size for w in model.get_weights()]) > 0

    def test_model_predicts_correct_shape(self):
        """Model prediction should output (batch_size, 1)."""
        model = build_model(input_dim=10)
        batch_size = 32
        X = np.random.randn(batch_size, 10).astype(np.float32)
        preds = model.predict(X, verbose=0)
        assert preds.shape == (batch_size, 1)

    def test_callbacks_list_length(self):
        """get_callbacks should return 4 callbacks."""
        callbacks = get_callbacks()
        assert len(callbacks) == 4
        names = [type(c).__name__ for c in callbacks]
        assert "EarlyStopping" in names
        assert "ModelCheckpoint" in names
        assert "ReduceLROnPlateau" in names
        assert "TensorBoard" in names

    def test_model_trainable(self):
        """All layers should be trainable by default."""
        model = build_model(input_dim=10)
        for layer in model.layers:
            assert layer.trainable

    def test_different_hidden_configs(self):
        """Model should build with various hidden layer configs."""
        configs = [
            [64],
            [128, 64, 32],
            [256, 128, 64, 32],
        ]
        for hidden in configs:
            model = build_model(input_dim=10, hidden_units=hidden)
            assert model is not None
            # Should have dense layers matching hidden config
            dense_count = sum(
                1 for layer in model.layers
                if isinstance(layer, tf.keras.layers.Dense)
            )
            assert dense_count == len(hidden) + 1  # +1 for output layer
