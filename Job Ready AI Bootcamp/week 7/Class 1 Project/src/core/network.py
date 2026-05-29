"""
Neural Network — A 2-layer network built from scratch using only NumPy.

Architecture:
    Input Layer → Hidden Layer (ReLU) → Output Layer (Sigmoid)

This implementation demonstrates:
- Forward propagation (matrix operations)
- Backpropagation (chain rule application)
- Gradient descent (weight updates)
- Training loop with loss tracking
"""

from typing import Dict, List, Tuple, Optional
from pathlib import Path

import numpy as np
import joblib

from src.core.activation import ActivationFunctions
from src.utils.logger import get_logger
import config

logger = get_logger(__name__)


class NeuralNetwork:
    """
    A fully-connected 2-layer neural network implemented from scratch.

    Layer 1 (Hidden): Linear → ReLU
    Layer 2 (Output): Linear → Sigmoid

    Loss: Binary Cross-Entropy
    Optimizer: Stochastic Gradient Descent (SGD)
    """

    def __init__(
        self,
        input_size: int = 2,
        hidden_size: int = 3,
        output_size: int = 1,
        learning_rate: float = 0.5,
        seed: int = config.RANDOM_STATE,
    ) -> None:
        """
        Initialize the network with random weights.

        Args:
            input_size: Number of input features (e.g., 2 for logic gates)
            hidden_size: Number of hidden neurons (more = more capacity)
            output_size: Number of outputs (1 for binary classification)
            learning_rate: Step size for gradient descent
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Xavier/Glorot initialization: weights scaled by sqrt(1/fan_in)
        # This prevents vanishing/exploding gradients in early training
        self.W1 = np.random.randn(hidden_size, input_size) * np.sqrt(1.0 / input_size)
        self.b1 = np.zeros((hidden_size, 1))

        self.W2 = np.random.randn(output_size, hidden_size) * np.sqrt(1.0 / hidden_size)
        self.b2 = np.zeros((output_size, 1))

        # Cache for backpropagation (stores intermediate values)
        self.cache: Dict[str, np.ndarray] = {}

        logger.info(
            "Network initialized: [%d → %d → %d], lr=%.3f",
            input_size, hidden_size, output_size, learning_rate,
        )

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward propagation: compute predictions.

        X shape: (n_samples, n_features)
        Returns: (n_samples, n_outputs)

        Steps:
        1. z1 = X·W1ᵀ + b1ᵀ  (hidden pre-activation)
        2. a1 = ReLU(z1)      (hidden activation)
        3. z2 = a1·W2ᵀ + b2ᵀ (output pre-activation)
        4. a2 = Sigmoid(z2)   (output activation = prediction)
        """
        # Layer 1: Input → Hidden
        self.cache["X"] = X
        self.cache["z1"] = np.dot(X, self.W1.T) + self.b1.T  # (n_samples, hidden_size)
        self.cache["a1"] = ActivationFunctions.relu(self.cache["z1"])

        # Layer 2: Hidden → Output
        self.cache["z2"] = np.dot(self.cache["a1"], self.W2.T) + self.b2.T  # (n_samples, output_size)
        self.cache["a2"] = ActivationFunctions.sigmoid(self.cache["z2"])

        return self.cache["a2"]

    def backward(self, y: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Backpropagation: compute gradients of loss w.r.t. all parameters.

        This is where the "learning" happens. We use the chain rule to
        propagate the error backward through the network.

        Chain rule application:
        dL/dW2 = dL/da2 × da2/dz2 × dz2/dW2
        dL/dW1 = dL/da2 × da2/dz2 × dz2/da1 × da1/dz1 × dz1/dW1
        """
        m = y.shape[0]  # Number of samples (for averaging)

        # Output layer gradients
        # dL/dz2 = (a2 - y) / m  (derivative of BCE loss + sigmoid combined)
        dz2 = (self.cache["a2"] - y) / m

        dW2 = np.dot(dz2.T, self.cache["a1"])  # (output_size, hidden_size)
        db2 = np.sum(dz2, axis=0, keepdims=True).T  # (output_size, 1)

        # Hidden layer gradients (chain rule through ReLU)
        da1 = np.dot(dz2, self.W2)  # (n_samples, hidden_size)
        dz1 = da1 * ActivationFunctions.relu_derivative(self.cache["z1"])  # ReLU derivative

        dW1 = np.dot(dz1.T, self.cache["X"])  # (hidden_size, input_size)
        db1 = np.sum(dz1, axis=0, keepdims=True).T  # (hidden_size, 1)

        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    def update_weights(self, grads: Dict[str, np.ndarray]) -> None:
        """
        Gradient descent: update weights in the opposite direction of gradients.

        W_new = W_old - learning_rate × dL/dW

        This is the "descent" part — we walk downhill on the loss landscape.
        """
        self.W1 -= self.learning_rate * grads["dW1"]
        self.b1 -= self.learning_rate * grads["db1"]
        self.W2 -= self.learning_rate * grads["dW2"]
        self.b2 -= self.learning_rate * grads["db2"]

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Binary Cross-Entropy Loss.

        L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]

        Measures how far predictions are from targets.
        Lower = better. Minimum is 0 (perfect predictions).
        """
        epsilon = 1e-15  # Prevent log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return float(loss)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 2000,
        verbose: bool = True,
    ) -> Dict[str, List]:
        """
        Training loop: forward → compute loss → backward → update weights.

        Returns history dict with loss and weight trajectories for visualization.
        """
        history = {
            "loss": [],
            "W1_history": [],
            "W2_history": [],
        }

        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(X)

            # Compute loss
            loss = self.compute_loss(y, y_pred)
            history["loss"].append(loss)

            # Store weights every 50 epochs for visualization
            if epoch % 50 == 0:
                history["W1_history"].append(self.W1.copy())
                history["W2_history"].append(self.W2.copy())

            # Backward pass
            grads = self.backward(y)

            # Update weights
            self.update_weights(grads)

            if verbose and epoch % 500 == 0:
                logger.info("Epoch %d/%d — Loss: %.6f", epoch, epochs, loss)

        # Store final weights
        history["W1_history"].append(self.W1.copy())
        history["W2_history"].append(self.W2.copy())

        logger.info("Training complete. Final loss: %.6f", history["loss"][-1])
        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        return self.forward(X)

    def save(self, path: Path) -> None:
        """Save model weights to disk."""
        params = {
            "W1": self.W1, "b1": self.b1,
            "W2": self.W2, "b2": self.b2,
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
        }
        joblib.dump(params, path)
        logger.info("Model saved to %s", path)

    def load(self, path: Path) -> None:
        """Load model weights from disk."""
        params = joblib.load(path)
        self.W1 = params["W1"]
        self.b1 = params["b1"]
        self.W2 = params["W2"]
        self.b2 = params["b2"]
        logger.info("Model loaded from %s", path)
