"""
Model registry: save, load, and version trained CNN checkpoints.

Every save is tagged with a UTC timestamp and its test accuracy, and an
`index.json` manifest tracks all versions so the best one can be recovered
without re-training. This mirrors the persistence spirit of
`week 6/Class 1 Project/src/models/registry.py`, adapted for Keras models.

HIGHLIGHTS — why version checkpoints by timestamp+accuracy instead of
just overwriting one `model.keras` file on every training run?
  - The Streamlit "Train" tab is designed to be re-run repeatedly (that's
    the whole point of the "quick demo" toggle) — a student might try a
    quick 3-epoch run, like the result, then try a full 10-epoch run that
    happens to do *worse* due to randomness in initialization/shuffling.
    If every run just overwrote the same file, a worse run would silently
    destroy a better one with no way to go back.
  - Keeping every version's test accuracy alongside its file means
    `load_best()` can always serve the strongest model that's been
    trained so far in this session, regardless of what order the runs
    happened in — the "Predict" and "Model Insights" tabs never have to
    care about run history, they just ask for the best one.
  - It also gives students a tangible artifact of experimentation: they
    can look at `data/models/index.json` and see, e.g., "the quick-subset
    run got 94% but the full run got 99%" — turning an abstract
    hyperparameter discussion into something they can inspect.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from tensorflow import keras

from config import Config, get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

INDEX_FILENAME = "index.json"


@dataclass
class ModelVersion:
    """Metadata describing a single saved model checkpoint."""

    version_id: str
    filename: str
    test_accuracy: float
    test_loss: float
    created_at: str
    epochs_trained: int


class ModelRegistry:
    """Handles versioned persistence of trained CNN models."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or get_config()
        self.model_dir: Path = self.config.paths.model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.model_dir / INDEX_FILENAME

    def save(
        self,
        model: keras.Model,
        test_accuracy: float,
        test_loss: float,
        epochs_trained: int,
    ) -> ModelVersion:
        """Save a trained model as a new version and update the index.

        Args:
            model: The trained Keras model to persist.
            test_accuracy: Accuracy on the held-out test set.
            test_loss: Loss on the held-out test set.
            epochs_trained: Number of epochs the model was trained for.

        Returns:
            The :class:`ModelVersion` metadata record for this save.
        """
        # The version_id embeds *both* a timestamp and the accuracy in the
        # filename itself (not just inside index.json). That's deliberate:
        # even if the JSON index were ever lost or corrupted, a student
        # could still tell which saved `.keras` file was the best one just
        # by reading the filenames in `data/models/` — the accuracy is
        # human-readable metadata baked directly into the artifact name.
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        version_id = f"{timestamp}-acc{test_accuracy:.4f}"
        filename = f"mnist_cnn_{version_id}.keras"
        filepath = self.model_dir / filename

        model.save(filepath)

        version = ModelVersion(
            version_id=version_id,
            filename=filename,
            test_accuracy=float(test_accuracy),
            test_loss=float(test_loss),
            created_at=datetime.now(timezone.utc).isoformat(),
            epochs_trained=int(epochs_trained),
        )

        versions = self._load_index()
        versions.append(version)
        self._save_index(versions)

        # Also keep a stable "latest" copy for convenience, matching the
        # `training.model_filename` path other modules default to. This is
        # purely a convenience alias for tools/scripts that want "whatever
        # was trained most recently" without reading the index — the
        # *authoritative* answer to "which model should the app serve" is
        # always `load_best()` below, driven by accuracy, not recency.
        latest_path = self.model_dir / self.config.training.model_filename
        model.save(latest_path)

        logger.info("Saved model version %s (test_acc=%.4f)", version_id, test_accuracy)
        return version

    def load_best(self) -> Optional[keras.Model]:
        """Load the highest-accuracy model version, or ``None`` if empty.

        This is what `app.py`'s Predict and Model Insights tabs call when
        there's no model in the current Streamlit session — e.g. right
        after the app restarts. Picking by `test_accuracy` (rather than by
        most-recent `created_at`) means a student's earlier good run
        survives being served to the app even if their *next*
        experiment — a different hyperparameter, a shorter quick-demo
        run — happens to train a weaker model.
        """
        versions = self._load_index()
        if not versions:
            logger.warning("No model versions found in registry.")
            return None

        best = max(versions, key=lambda v: v.test_accuracy)
        return self.load_version(best.version_id)

    def load_version(self, version_id: str) -> Optional[keras.Model]:
        """Load a specific model version by its ``version_id``."""
        versions = self._load_index()
        match = next((v for v in versions if v.version_id == version_id), None)
        if match is None:
            logger.warning("Version %s not found in registry.", version_id)
            return None

        filepath = self.model_dir / match.filename
        logger.info("Loading model version %s from %s", version_id, filepath)
        return keras.models.load_model(filepath)

    def list_versions(self) -> List[ModelVersion]:
        """Return all saved versions, most recent first."""
        return sorted(self._load_index(), key=lambda v: v.created_at, reverse=True)

    def _load_index(self) -> List[ModelVersion]:
        if not self.index_path.exists():
            return []
        with open(self.index_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return [ModelVersion(**entry) for entry in raw]

    def _save_index(self, versions: List[ModelVersion]) -> None:
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump([asdict(v) for v in versions], f, indent=2)
