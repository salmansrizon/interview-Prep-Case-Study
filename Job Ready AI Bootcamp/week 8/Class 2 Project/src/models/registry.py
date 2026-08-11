"""
Save/load helpers for the three trained model variants.

Keeps the Streamlit app decoupled from Keras' serialization details: the
app only asks the registry for "the optimized model" or "is the dropout
model available" and never touches `keras.models.load_model` directly.

HIGHLIGHTS: this is a small but important separation-of-concerns
example. app.py's job is UI (buttons, tabs, charts); it shouldn't need to
know that models are saved in the modern `.keras` format, or care about
file-not-found handling for a model nobody has trained yet. Centralizing
that here means if Keras' recommended save format changes in some future
version, only this one file needs updating.
"""

from __future__ import annotations

from pathlib import Path

from tensorflow import keras

from config import PathsConfig, get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

_VARIANT_PATH_ATTR = {
    "baseline": "baseline_model_path",
    "dropout": "dropout_model_path",
    "optimized": "optimized_model_path",
}


def _path_for(variant: str, paths_cfg: PathsConfig | None = None) -> Path:
    if variant not in _VARIANT_PATH_ATTR:
        raise ValueError(f"Unknown model variant: {variant}. Choose from {list(_VARIANT_PATH_ATTR)}")
    paths_cfg = paths_cfg or get_config().paths
    return getattr(paths_cfg, _VARIANT_PATH_ATTR[variant])


def save_variant(model: keras.Model, variant: str, paths_cfg: PathsConfig | None = None) -> Path:
    """Persist a trained model variant to its configured path.

    Args:
        model: The trained Keras model to save.
        variant: One of "baseline", "dropout", "optimized".
        paths_cfg: Optional paths configuration override.

    Returns:
        The path the model was written to.
    """
    path = _path_for(variant, paths_cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    logger.info("Saved '%s' model to %s", variant, path)
    return path


def load_variant(variant: str, paths_cfg: PathsConfig | None = None) -> keras.Model:
    """Load a previously saved model variant from disk.

    Args:
        variant: One of "baseline", "dropout", "optimized".
        paths_cfg: Optional paths configuration override.

    Returns:
        The loaded Keras model.

    Raises:
        FileNotFoundError: If no saved model exists for that variant yet.
    """
    path = _path_for(variant, paths_cfg)
    if not path.exists():
        raise FileNotFoundError(
            f"No saved model found for variant '{variant}' at {path}. Train it first."
        )
    logger.info("Loading '%s' model from %s", variant, path)
    return keras.models.load_model(path)


def variant_is_available(variant: str, paths_cfg: PathsConfig | None = None) -> bool:
    """Check whether a trained model variant exists on disk."""
    return _path_for(variant, paths_cfg).exists()
