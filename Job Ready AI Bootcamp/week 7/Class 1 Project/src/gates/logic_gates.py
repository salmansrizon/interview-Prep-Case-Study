"""
Logic Gate Datasets — The classic testbed for neural networks.

Logic gates (AND, OR, XOR) are the "hello world" of neural networks.
- AND and OR are linearly separable (single layer can learn them)
- XOR is NOT linearly separable (requires hidden layer — the proof that deep networks work)
"""

import numpy as np
from typing import Tuple, Dict

from src.utils.logger import get_logger

logger = get_logger(__name__)


class LogicGateDataset:
    """
    Generates truth table data for logic gates.

    Each gate has 4 input combinations (00, 01, 10, 11) and 1 output.
    XOR is the famous case that requires a hidden layer to solve.
    """

    GATE_TRUTH_TABLES: Dict[str, np.ndarray] = {
        "AND": np.array([[0], [0], [0], [1]]),
        "OR":  np.array([[0], [1], [1], [1]]),
        "XOR": np.array([[0], [1], [1], [0]]),
        "NAND": np.array([[1], [1], [1], [0]]),
        "NOR": np.array([[1], [0], [0], [0]]),
    }

    # All possible 2-bit inputs
    INPUTS = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ], dtype=float)

    def __init__(self, gate_type: str = "XOR") -> None:
        """
        Initialize a logic gate dataset.

        Args:
            gate_type: One of "AND", "OR", "XOR", "NAND", "NOR"
        """
        gate_type = gate_type.upper()
        if gate_type not in self.GATE_TRUTH_TABLES:
            raise ValueError(f"Unknown gate: {gate_type}. Choose from {list(self.GATE_TRUTH_TABLES.keys())}")

        self.gate_type = gate_type
        self.outputs = self.GATE_TRUTH_TABLES[gate_type]

        logger.info("LogicGateDataset initialized: %s", gate_type)

    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (inputs, targets) as NumPy arrays."""
        return self.INPUTS.copy(), self.outputs.copy()

    def is_linearly_separable(self) -> bool:
        """
        Check if the gate is linearly separable.

        AND, OR, NAND, NOR: Yes (single perceptron can solve)
        XOR: No (requires hidden layer)
        """
        return self.gate_type in ["AND", "OR", "NAND", "NOR"]

    def __repr__(self) -> str:
        return f"LogicGateDataset(gate={self.gate_type}, linearly_separable={self.is_linearly_separable()})"
