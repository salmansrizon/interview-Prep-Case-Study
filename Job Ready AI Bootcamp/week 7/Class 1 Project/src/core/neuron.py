"""
The Neuron — Fundamental building block of neural networks.
Implements a single perceptron with weights, bias, and activation.
"""

import numpy as np
from typing import List, Callable, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class Neuron:
    """
    A single neuron (perceptron) that computes:
        z = Σ(wᵢ × xᵢ) + b
        output = activation(z)

    This is the atomic unit that, when stacked in layers,
    forms the power of neural networks.
    """

    def __init__(
        self,
        weights: Optional[List[float]] = None,
        bias: float = 0.0,
        activation: Optional[Callable] = None,
    ) -> None:
        """
        Initialize a neuron.

        Args:
            weights: Input weights. If None, randomly initialized.
            bias: Bias term. Shifts the activation threshold.
            activation: Activation function. If None, identity (linear).
        """
        self.weights = np.array(weights, dtype=float) if weights is not None else None
        self.bias = float(bias)
        self.activation = activation or (lambda x: x)

        if self.weights is not None:
            logger.info("Neuron initialized with %d inputs", len(self.weights))

    def compute_z(self, inputs: List[float]) -> float:
        """
        Compute the weighted sum (pre-activation).

        z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b

        This is the "signal strength" before the activation function
        decides whether the neuron fires.
        """
        x = np.array(inputs, dtype=float)
        if self.weights is None:
            self.weights = np.random.randn(len(x)) * 0.1

        z = np.dot(self.weights, x) + self.bias
        return float(z)

    def forward(self, inputs: List[float]) -> float:
        """
        Complete forward pass: weighted sum + activation.

        Returns the neuron's output after applying the activation function.
        """
        z = self.compute_z(inputs)
        return float(self.activation(z))

    def __repr__(self) -> str:
        return f"Neuron(weights={self.weights.round(3).tolist()}, bias={self.bias:.3f})"
