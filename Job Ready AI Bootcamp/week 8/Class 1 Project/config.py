"""
Centralized configuration for the High-Accuracy MNIST Digit Classifier.

Dataclass-based configuration (mirrors the style used in
`week 7/Class 2 Project/src/config.py`) so every hyperparameter, path, and
UI constant lives in one place instead of being scattered across the code.

HIGHLIGHTS — why a dedicated config module at all?
Every "magic number" a student would otherwise find buried inside
`src/models/cnn.py` or `app.py` (32 filters? 0.5 dropout? 128 batch size?)
lives here instead. That means:
  1. You can change the CNN's shape or the training schedule without
     hunting through multiple files.
  2. Tests can inject a different `Config` instance (e.g. a smaller image
     size) without monkeypatching constants scattered across modules.
  3. The engineering standard for this bootcamp explicitly forbids
     hardcoded magic numbers outside `config.py` — this file is where
     they are supposed to live.

HIGHLIGHTS — why `frozen=True` dataclasses?
Configuration should be read, not mutated, once the app starts. Freezing
the dataclasses turns "someone accidentally changed `config.model.dropout_rate`
mid-run" into a `FrozenInstanceError` at the point of the mistake, instead of
a silent bug that only shows up as "why did my model's accuracy change
between runs?"
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


# ── Paths ───────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = DATA_DIR / "models"
HISTORY_DIR = DATA_DIR / "history"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem locations used across the project."""

    base_dir: Path = BASE_DIR
    data_dir: Path = DATA_DIR
    model_dir: Path = MODEL_DIR
    history_dir: Path = HISTORY_DIR

    def ensure_dirs(self) -> None:
        """Create every directory referenced here if it does not exist."""
        for d in (self.data_dir, self.model_dir, self.history_dir):
            d.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class DataConfig:
    """MNIST data-loading and preprocessing constants."""

    # MNIST images are natively 28x28 grayscale. This is small enough that
    # a Dense-only network is *technically* feasible (784 inputs), but the
    # lecture's whole point is that CNNs preserve the 2D spatial structure
    # a flattened 784-vector would destroy — see cnn.py for how this
    # tuple becomes the Conv2D `input_shape`.
    image_size: Tuple[int, int] = (28, 28)
    # 1 channel because MNIST is grayscale, not RGB (which would be 3).
    # This directly sets the last dimension of every image tensor
    # (N, H, W, C) that flows through the model.
    num_channels: int = 1
    num_classes: int = 10  # digits 0-9
    # Held out from the *training* split (not from Keras's separate test
    # set) so we can watch for overfitting during `model.fit(..., validation_data=...)`
    # without ever letting the model see the true test set until final evaluation.
    val_split: float = 0.1
    # Fixed seed so the train/val split and any subsampling are
    # reproducible across runs — important for students comparing their
    # results with a classmate's or with the lecture's numbers.
    random_state: int = 42
    # Raw MNIST pixels are 8-bit integers in [0, 255]. Dividing by this
    # value rescales them to [0, 1] float32 before they ever reach the
    # model. HIGHLIGHTS: why do this instead of feeding raw [0, 255] ints?
    #   - Large, unbounded inputs make gradients during backpropagation
    #     unstable (they can explode), which is exactly what the lecture's
    #     "Image Normalization" section warns about.
    #   - Keeping every input feature on the same small scale means no
    #     single pixel's magnitude can dominate the loss function.
    #   - Optimizers like Adam converge noticeably faster on normalized
    #     inputs than on raw 0-255 ranges.
    # This same constant is reused at *inference* time in preprocessor.py
    # so a user-drawn digit is normalized identically to how the model
    # was trained — mismatched train/inference scaling is a classic silent
    # bug (the model would see values in the "wrong ballpark").
    pixel_max_value: float = 255.0
    # Fraction of the training set to use when the Streamlit "quick demo"
    # toggle is enabled, so a live in-browser training run finishes fast.
    # A 10% subset trains in seconds rather than minutes, which matters a
    # lot when a student is live-demoing the app on a laptop with no GPU.
    quick_subset_fraction: float = 0.1


@dataclass(frozen=True)
class ModelConfig:
    """Baseline CNN architecture hyperparameters (see Class 1 Lecture)."""

    # HIGHLIGHTS: why 32 filters in the first conv layer and 64 in the
    # second (not the other way around)? The first layer sees raw pixels and only needs to
    # learn simple, generic patterns — edges, strokes, curves — of which
    # there are relatively few distinct kinds. 32 filters is plenty to
    # cover that vocabulary. By the second conv layer, the input is no
    # longer raw pixels but a 13x13x32 stack of *feature maps* — the
    # network now needs to learn combinations of those simple patterns
    # (loops, corners, intersections), and there are many more possible
    # combinations than there are primitive edges. Doubling the filter
    # count (64) gives the network enough capacity to represent that
    # richer, more specific vocabulary. This "start narrow, get wider with
    # depth" progression is the standard CNN design pattern.
    conv1_filters: int = 32
    conv2_filters: int = 64
    # 3x3 is the smallest kernel that can still detect a directional
    # pattern (a straight edge needs at least a 3-pixel span to have an
    # orientation). It also keeps the parameter count per filter tiny
    # (9 weights) so the "weight sharing" argument from the lecture holds:
    # the same 9 numbers get reused across every position in the image.
    kernel_size: Tuple[int, int] = (3, 3)
    # 2x2 max pooling halves both spatial dimensions after each conv
    # block. This (a) shrinks the feature maps so later layers are cheaper
    # to compute, and (b) buys "translation invariance" — a digit drawn a
    # couple pixels off-center still produces the same strongest signal
    # inside each pooling window.
    pool_size: Tuple[int, int] = (2, 2)
    dense_units: int = 128
    # HIGHLIGHTS: why does Dropout(0.5) sit right before the final
    # Dense(10) layer and not, say, between the conv layers? The Dense(128) layer is where the
    # network has the most parameters concentrated in one place (1600 ->
    # 128 is over 200K weights), which makes it the layer most prone to
    # memorizing the training set instead of learning general digit
    # shapes. Randomly zeroing 50% of its activations on every training
    # step forces the network to not rely on any single neuron, which is
    # a strong, cheap regularizer exactly where overfitting risk is
    # highest. Dropout is deliberately *not* placed on the conv layers:
    # convolutional filters already share weights across the whole image
    # (far fewer independent parameters per layer), so they are much less
    # prone to overfitting and dropping activations there would mostly
    # just destroy useful spatial signal.
    dropout_rate: float = 0.5
    num_classes: int = 10
    learning_rate: float = 1e-3  # Adam's well-tested default; stable without tuning.
    optimizer: str = "adam"
    # HIGHLIGHTS: why `sparse_categorical_crossentropy` instead of `categorical_crossentropy`?
    # MNIST labels are plain integers (5, not [0,0,0,0,0,1,0,0,0,0]). The
    # "sparse" variant accepts integer labels directly, so we skip the
    # `to_categorical()` one-hot conversion step entirely — one less thing
    # to keep in sync between training and evaluation code.
    loss: str = "sparse_categorical_crossentropy"
    metrics: List[str] = field(default_factory=lambda: ["accuracy"])


@dataclass(frozen=True)
class TrainingConfig:
    """Training-loop defaults."""

    # 10 epochs is enough for the baseline architecture to reach ~99% test
    # accuracy on full MNIST without needing the advanced tricks (LR
    # schedules, augmentation) that Class 2 covers.
    epochs: int = 10
    # A much shorter run for the Streamlit "quick demo" toggle, paired
    # with `DataConfig.quick_subset_fraction` — the goal there isn't peak
    # accuracy, it's letting a student see the whole train -> evaluate ->
    # predict loop complete in the time it takes to read the next slide.
    quick_epochs: int = 3
    # 128 is a common sweet spot: large enough for efficient, vectorized
    # gradient updates, small enough to keep memory use low and give the
    # optimizer many updates per epoch.
    batch_size: int = 128
    # Stops training if `val_accuracy` hasn't improved for this many
    # epochs, restoring the best-seen weights. This guards against
    # wasting time (or overfitting further) once the model has plateaued,
    # without a student having to babysit the run.
    early_stopping_patience: int = 3
    model_filename: str = "mnist_cnn.keras"
    history_filename: str = "training_history.json"
    metadata_filename: str = "training_metadata.json"


@dataclass(frozen=True)
class AppConfig:
    """Streamlit UI constants."""

    page_title: str = "MNIST Digit Classifier"
    page_icon: str = "🔢"
    canvas_size: int = 280
    canvas_stroke_width: int = 18


@dataclass(frozen=True)
class Config:
    """Top-level configuration object aggregating every sub-config."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    app: AppConfig = field(default_factory=AppConfig)


_config_instance: Config | None = None


def get_config() -> Config:
    """Return the process-wide singleton :class:`Config` instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
        _config_instance.paths.ensure_dirs()
    return _config_instance
