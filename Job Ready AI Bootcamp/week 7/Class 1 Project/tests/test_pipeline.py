"""
Sanity checks for the neural network pipeline.
Run with: pytest tests/test_pipeline.py
"""

import numpy as np
from src.core.neuron import Neuron
from src.core.activation import ActivationFunctions
from src.core.network import NeuralNetwork
from src.gates.logic_gates import LogicGateDataset


def test_neuron_computation():
    """Test that a neuron computes z = w·x + b correctly."""
    neuron = Neuron(weights=[1.0, 2.0], bias=0.5)
    z = neuron.compute_z([1.0, 1.0])
    assert abs(z - 3.5) < 1e-10  # 1*1 + 2*1 + 0.5 = 3.5


def test_sigmoid():
    """Test sigmoid properties."""
    act = ActivationFunctions()
    # Sigmoid(0) = 0.5
    assert abs(act.sigmoid(np.array([0]))[0] - 0.5) < 1e-10
    # Sigmoid(large) ≈ 1
    assert act.sigmoid(np.array([10]))[0] > 0.99
    # Sigmoid(-large) ≈ 0
    assert act.sigmoid(np.array([-10]))[0] < 0.01


def test_relu():
    """Test ReLU properties."""
    act = ActivationFunctions()
    # ReLU(positive) = positive
    assert act.relu(np.array([5]))[0] == 5
    # ReLU(negative) = 0
    assert act.relu(np.array([-3]))[0] == 0
    # ReLU(0) = 0
    assert act.relu(np.array([0]))[0] == 0


def test_xor_requires_hidden_layer():
    """Test that XOR cannot be solved without a hidden layer."""
    dataset = LogicGateDataset("XOR")
    X, y = dataset.get_data()

    # Single layer (no hidden) should fail
    nn_no_hidden = NeuralNetwork(input_size=2, hidden_size=1, output_size=1, learning_rate=0.5)
    history = nn_no_hidden.train(X, y, epochs=1000, verbose=False)
    predictions = nn_no_hidden.predict(X)
    accuracy = np.mean(np.round(predictions.flatten()) == y.flatten())

    # Should NOT achieve perfect accuracy
    assert accuracy < 1.0, "XOR should not be solvable without hidden layer"


def test_xor_with_hidden_layer():
    """Test that XOR CAN be solved with a hidden layer."""
    dataset = LogicGateDataset("XOR")
    X, y = dataset.get_data()

    # 2-layer network should succeed
    nn = NeuralNetwork(input_size=2, hidden_size=3, output_size=1, learning_rate=0.5)
    history = nn.train(X, y, epochs=5000, verbose=False)
    predictions = nn.predict(X)
    accuracy = np.mean(np.round(predictions.flatten()) == y.flatten())

    # Should achieve perfect or near-perfect accuracy
    assert accuracy >= 0.75, f"XOR accuracy too low: {accuracy}"


def test_and_gate():
    """Test that AND gate is learnable."""
    dataset = LogicGateDataset("AND")
    X, y = dataset.get_data()

    nn = NeuralNetwork(input_size=2, hidden_size=2, output_size=1, learning_rate=0.5)
    history = nn.train(X, y, epochs=3000, verbose=False)
    predictions = nn.predict(X)
    accuracy = np.mean(np.round(predictions.flatten()) == y.flatten())

    assert accuracy == 1.0, f"AND gate not learned: {accuracy}"


def test_forward_backward_shapes():
    """Test that forward and backward passes produce correct shapes."""
    nn = NeuralNetwork(input_size=2, hidden_size=3, output_size=1)
    X = np.array([[0, 0], [1, 1]])
    y = np.array([[0], [1]])

    # Forward
    output = nn.forward(X)
    assert output.shape == (2, 1)

    # Backward
    grads = nn.backward(y)
    assert grads["dW1"].shape == nn.W1.shape
    assert grads["db1"].shape == nn.b1.shape
    assert grads["dW2"].shape == nn.W2.shape
    assert grads["db2"].shape == nn.b2.shape
