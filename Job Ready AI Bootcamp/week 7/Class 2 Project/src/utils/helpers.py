"""
Shared helper functions for the project.
"""

import os
import json
import pickle
import joblib
from typing import Any, Dict

import numpy as np


def save_json(data: Dict[str, Any], path: str) -> None:
    """Save dictionary to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str) -> Dict[str, Any]:
    """Load dictionary from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_pickle(obj: Any, path: str) -> None:
    """Save object with pickle."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str) -> Any:
    """Load object from pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def save_joblib(obj: Any, path: str) -> None:
    """Save object with joblib (better for large numpy arrays)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)


def load_joblib(path: str) -> Any:
    """Load object from joblib file."""
    return joblib.load(path)


def set_seed(seed: int = 42) -> None:
    """Set available random seeds for reproducibility.

    TensorFlow is optional for data-generation and preprocessing commands,
    so only seed it when the deep-learning dependency is installed.
    """
    import random

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import tensorflow as tf
    except ImportError:
        return
    tf.random.set_seed(seed)


def format_number(n: float, decimals: int = 2) -> str:
    """Format a number with thousand separators and fixed decimals."""
    return f"{n:,.{decimals}f}"


def human_readable_size(size_bytes: int) -> str:
    """Convert bytes to human-readable string."""
    if size_bytes == 0:
        return "0 B"
    size_names = ["B", "KB", "MB", "GB"]
    i = int(np.floor(np.log10(size_bytes) / np.log10(1024)))
    p = np.power(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"
