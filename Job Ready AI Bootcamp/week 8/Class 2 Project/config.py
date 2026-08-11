"""
Centralized configuration for the High-Accuracy MNIST Digit Classifier.

All hyperparameters, paths, and constants used across the app live here so
that `src/` and `app.py` never hardcode "magic numbers" — everything is
traceable to a single, documented source of truth.

HIGHLIGHTS: notice this file groups values by *what they configure*
(data shapes, model architecture, training loop, filesystem paths) rather
than by *which variant uses them*. That's deliberate: the three CNN
variants in src/models/architectures.py deliberately share almost all of
these numbers (same filter counts, same kernel size, same learning rate,
same batch size) so that when their results differ, you know it's because
of the specific technique each variant adds (Dropout, BatchNorm,
EarlyStopping) — not because someone quietly also changed the learning
rate for one of them. Keeping every shared value in one place makes that
guarantee easy to verify at a glance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


# ── Paths ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = DATA_DIR / "models"
LOG_DIR = DATA_DIR / "logs"

for _d in (DATA_DIR, MODEL_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class DataConfig:
    """Shape and split configuration for the MNIST dataset."""

    image_size: int = 28  # native MNIST resolution; also what the canvas/upload preprocessor resizes to
    channels: int = 1  # grayscale — MNIST has no color information
    num_classes: int = 10  # digits 0-9
    val_split: float = 0.1  # fraction of the training set held out for validation
    # A fixed seed makes the train/val split (and the quick-demo sampling
    # below) reproducible run-to-run, which matters for a teaching app:
    # students comparing their own run against these docs should see the
    # same split, not a different random shuffle each time.
    random_seed: int = 42
    # "Quick demo subset" sizes: small enough that even the Fully Optimized
    # variant (with EarlyStopping/ReduceLROnPlateau still doing real work)
    # trains in well under a minute, so a student can click through all
    # three variants in one sitting without waiting on a full 60k-image
    # MNIST epoch. Accuracy will be lower than the full-dataset numbers
    # quoted in the README — that trade-off is intentional and explained
    # in the UI.
    quick_subset_train_size: int = 4000
    quick_subset_test_size: int = 1000

    @property
    def input_shape(self) -> tuple:
        """(H, W, C) shape Conv2D layers expect — computed, not duplicated,
        so image_size/channels never drift out of sync with input_shape."""
        return (self.image_size, self.image_size, self.channels)


@dataclass(frozen=True)
class ModelConfig:
    """Architecture hyperparameters shared across the three CNN variants."""

    conv1_filters: int = 32  # standard "start small" choice for a first conv layer on 28x28 inputs
    conv2_filters: int = 64  # doubled filter count as spatial resolution shrinks (standard CNN pattern)
    kernel_size: int = 3  # 3x3 is the de-facto default receptive field for small-image CNNs
    pool_size: int = 2  # halves each spatial dimension per MaxPool
    dense_units: int = 128
    # Two different dropout rates on purpose (see architectures.py for the
    # full reasoning): 0.3 is the gentler rate used in the Fully Optimized
    # variant, where BatchNorm already contributes some regularization.
    # 0.5 is the stronger, "textbook" rate used to demonstrate Dropout in
    # isolation in the dropout-only variant, without BatchNorm alongside it.
    dropout_conv_rate: float = 0.3  # used only in the "optimized" variant
    dropout_dense_rate: float = 0.5  # used only in the "dropout" variant
    learning_rate: float = 1e-3  # Adam's well-tested default; kept identical across all 3 variants


@dataclass(frozen=True)
class TrainingConfig:
    """Training loop, callback, and epoch-budget configuration per variant."""

    batch_size: int = 128
    # Baseline and dropout both train for a FIXED epoch count — no
    # EarlyStopping — because part of the lesson is watching the baseline
    # keep "training" long after it has stopped actually improving on
    # validation data.
    baseline_epochs: int = 15
    dropout_epochs: int = 15
    # The optimized variant gets a generous epoch ceiling because
    # EarlyStopping (not this number) decides when to actually stop; 30 is
    # just a safety cap so a pathological run can't train forever.
    optimized_epochs: int = 30  # capped by EarlyStopping in practice
    early_stopping_patience: int = 5
    # See src/training/trainer.py for the full reasoning on why val_loss
    # (not val_accuracy) is the right signal to watch.
    early_stopping_monitor: str = "val_loss"
    reduce_lr_factor: float = 0.5  # halve the learning rate on each plateau event
    reduce_lr_patience: int = 3  # shorter than early-stopping patience, so LR gets a chance to help first
    reduce_lr_min_lr: float = 1e-6  # floor so the LR never decays to (effectively) zero
    verbose: int = 2  # Keras "one line per epoch" logging — readable in both terminal and Streamlit logs


@dataclass(frozen=True)
class PathsConfig:
    """Per-variant artifact save paths."""

    baseline_model_path: Path = MODEL_DIR / "baseline_cnn.keras"
    dropout_model_path: Path = MODEL_DIR / "dropout_cnn.keras"
    optimized_model_path: Path = MODEL_DIR / "optimized_cnn.keras"
    history_dir: Path = MODEL_DIR / "history"


@dataclass(frozen=True)
class AppConfig:
    """Top-level aggregate configuration consumed by app.py and src/."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)


# ── Variant metadata (used by both training orchestration and the UI) ──
VARIANT_NAMES: List[str] = ["baseline", "dropout", "optimized"]
VARIANT_DISPLAY_NAMES = {
    "baseline": "Baseline CNN",
    "dropout": "With Dropout",
    "optimized": "Fully Optimized",
}

CONFIG = AppConfig()
CONFIG.paths.history_dir.mkdir(parents=True, exist_ok=True)


def get_config() -> AppConfig:
    """Return the singleton application configuration."""
    return CONFIG
