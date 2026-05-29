"""
Activation Functions — The non-linear transformers of neural networks.

Without activation functions, no matter how many layers you stack,
the network remains a linear model. These functions introduce the
non-linearity that allows networks to learn complex patterns.
"""

import numpy as np
from typing import Callable

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ActivationFunctions:
    """
    Collection of standard activation functions and their derivatives.

    Each activation function:
    1. Introduces non-linearity
    2. Has a derivative for backpropagation
    3. Has specific advantages and trade-offs
    """

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        """
        Sigmoid: σ(z) = 1 / (1 + e^(-z))

        Maps any real number to (0, 1). Interpretable as probability.
        Classic choice for binary classification output layers.

        Problem: Vanishing gradient for large |z| (gradient ≈ 0).
        """
        # Clip to prevent overflow in exp
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def sigmoid_derivative(z: np.ndarray) -> np.ndarray:
        """
        Derivative of sigmoid: σ'(z) = σ(z) × (1 - σ(z))

        Beautiful mathematical property: derivative can be computed
        directly from the output, making backpropagation efficient.
        """
        s = ActivationFunctions.sigmoid(z)
        return s * (1 - s)

    @staticmethod
    def relu(z: np.ndarray) -> np.ndarray:
        """
        ReLU: max(0, z)

        Rectified Linear Unit. The most popular activation for hidden layers.

        Why it works:
        - Fast to compute (simple comparison)
        - No vanishing gradient for positive values (gradient = 1)
        - Sparsity: ~50% of neurons are "off" (output 0), making representation efficient

        Problem: "Dying ReLU" — neurons with negative inputs permanently output 0.
        """
        return np.maximum(0, z)

    @staticmethod
    def relu_derivative(z: np.ndarray) -> np.ndarray:
        """
        Derivative of ReLU: 1 if z > 0, else 0

        The gradient is either 0 or 1 — no vanishing gradient for active neurons.
        This is why ReLU trains faster than sigmoid in deep networks.
        """
        return (z > 0).astype(float)

    @staticmethod
    def tanh(z: np.ndarray) -> np.ndarray:
        """
        Tanh: (e^z - e^(-z)) / (e^z + e^(-z))

        Hyperbolic tangent. Zero-centered output (-1 to 1).

        Advantage over sigmoid: zero-centered outputs help gradient descent
        converge faster (no bias toward positive or negative).

        Still suffers from vanishing gradients for large |z|.
        """
        return np.tanh(z)

    @staticmethod
    def tanh_derivative(z: np.ndarray) -> np.ndarray:
        """Derivative of tanh: 1 - tanh²(z)"""
        t = np.tanh(z)
        return 1 - t ** 2

    @staticmethod
    def leaky_relu(z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """
        Leaky ReLU: max(αz, z) where α is a small constant (e.g., 0.01)

        Fixes the "dying ReLU" problem by allowing a small negative slope.
        Neurons can recover even if they receive negative inputs.
        """
        return np.where(z > 0, z, alpha * z)

    @staticmethod
    def leaky_relu_derivative(z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """Derivative of Leaky ReLU: 1 if z > 0, else α"""
        return np.where(z > 0, 1.0, alpha)

    @staticmethod
    def get(name: str) -> Callable:
        """Get activation function by name."""
        mapping = {
            "sigmoid": ActivationFunctions.sigmoid,
            "relu": ActivationFunctions.relu,
            "tanh": ActivationFunctions.tanh,
            "leaky_relu": ActivationFunctions.leaky_relu,
        }
        if name not in mapping:
            raise ValueError(f"Unknown activation: {name}. Choose from {list(mapping.keys())}")
        return mapping[name]
