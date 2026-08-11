"""
Baseline CNN architecture for MNIST digit classification.

This is the exact architecture taught in `Class 1 Lecture`:

    Conv2D(32, 3x3) -> MaxPool(2x2)
    -> Conv2D(64, 3x3) -> MaxPool(2x2)
    -> Flatten -> Dense(128) -> Dropout(0.5) -> Dense(10, softmax)

No advanced tuning (batch norm, data augmentation, learning-rate
schedules, etc.) is applied here on purpose — that is the subject of
Class 2. This module only builds and compiles the model; it never loads
data or trains.

HIGHLIGHTS — why keep model-building completely separate from
training/data code? This module has exactly one job: describe the architecture and return a
compiled model. It never touches a dataset, a file path, or `st.*`
anything. That separation is what lets `tests/test_pipeline.py` build a
real model and check its input/output shapes in milliseconds, without
needing MNIST downloaded or a training loop to run — and it is what lets
`app.py` build a model without importing anything Streamlit-unfriendly.

Reading this file top to bottom mirrors the "Big Picture Pipeline" from
the lecture almost line for line:

    Input -> [Conv2D] -> [ReLU] -> [MaxPool] -> [Conv2D] -> [ReLU]
          -> [MaxPool] -> [Flatten] -> [Dense] -> [Softmax]
"""

from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers

from config import Config, get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_baseline_cnn(config: Config | None = None) -> keras.Model:
    """Build and compile the baseline MNIST CNN.

    Args:
        config: Optional injected config (defaults to the global singleton).

    Returns:
        A compiled `keras.Sequential` model ready for `model.fit(...)`.
    """
    config = config or get_config()
    height, width = config.data.image_size
    # (28, 28, 1): the trailing 1 is the channel axis Conv2D layers
    # require. It exists purely because Keras's Conv2D API is generic
    # over grayscale/RGB/etc — for MNIST it will always be 1.
    input_shape = (height, width, config.data.num_channels)

    model = keras.Sequential(
        [
            layers.Input(shape=input_shape, name="input"),

            # ── Block 1: learn simple, generic patterns ──────────────
            # A 3x3 filter sliding across the raw 28x28 image. With no
            # padding specified, Keras defaults to `padding="valid"`
            # (no border added), so a 3x3 filter shrinks a 28x28 image
            # to 26x26 (28 - 3 + 1 = 26) — this matches the lecture
            # notebook's worked example exactly, so students can compare
            # `model.summary()` output directly against the lecture.
            # 32 filters is enough capacity to learn the relatively small
            # vocabulary of primitive strokes (edges, curves) that appear
            # at this shallow depth — see config.py's ModelConfig for the
            # full reasoning on the 32 -> 64 progression.
            layers.Conv2D(
                config.model.conv1_filters,
                kernel_size=config.model.kernel_size,
                activation="relu",  # zeroes out negative activations, adding the
                                     # non-linearity a CNN needs to learn anything
                                     # beyond a single linear transform of the input.
                name="conv1",
            ),
            # Halves the spatial size (26x26 -> 13x13) by keeping only the
            # strongest activation in each 2x2 window. This both reduces
            # compute for every layer downstream AND makes the network
            # tolerant to a digit being drawn a pixel or two off-center
            # (the "translation invariance" property from the lecture).
            layers.MaxPooling2D(pool_size=config.model.pool_size, name="pool1"),

            # ── Block 2: combine block 1's patterns into shapes ───────
            # This layer's input is no longer raw pixels but 32 stacked
            # feature maps from block 1, so it is effectively learning
            # "which combinations of edges/curves form a loop, a corner,
            # an intersection" — a larger and more specific vocabulary,
            # which is why the filter count doubles to 64 here.
            layers.Conv2D(
                config.model.conv2_filters,
                kernel_size=config.model.kernel_size,
                activation="relu",
                name="conv2",
            ),
            # 13x13 -> 11x11 (conv) -> 5x5 (pool). By this point the
            # network has compressed a 784-pixel image down to a compact
            # 5x5x64 = 1600-value summary of "what shapes are present and
            # roughly where" — small enough for a Dense layer to reason
            # over without an unreasonable number of weights.
            layers.MaxPooling2D(pool_size=config.model.pool_size, name="pool2"),

            # ── Classification head ───────────────────────────────────
            # Only now — after the convolutional layers have already
            # extracted spatial features — do we flatten to 1D. Doing
            # this at the *start* (as a plain Dense network would) is
            # exactly the "destroys spatial relationships" problem the
            # lecture opens with; doing it here means the spatial
            # reasoning has already happened and we're just flattening a
            # compact feature summary, not raw pixels.
            layers.Flatten(name="flatten"),
            layers.Dense(config.model.dense_units, activation="relu", name="dense1"),
            # Dropout sits *here* — right before the final classification
            # layer — rather than between the conv blocks, because this
            # Dense(128) layer concentrates the most parameters in one
            # place (1600 -> 128 is ~200K weights) and is therefore the
            # part of the network most likely to memorize training
            # examples instead of learning general digit shapes.
            # Randomly zeroing half its activations on every training
            # step prevents any single neuron from being relied on,
            # which is a cheap, effective way to fight overfitting
            # exactly where the risk is concentrated.
            layers.Dropout(config.model.dropout_rate, name="dropout"),
            # Softmax turns 10 raw scores into a probability distribution
            # over the 10 digit classes that sums to 1 — this is what
            # lets the Streamlit "Predict" tab show a per-class confidence
            # bar chart instead of just a single hard prediction.
            layers.Dense(config.model.num_classes, activation="softmax", name="output"),
        ],
        name="MNIST_Baseline_CNN",
    )

    # Adam adapts its per-parameter learning rate automatically, which is
    # why it needs far less manual learning-rate tuning than plain SGD —
    # a good default for a *baseline* model where the point is showing
    # the architecture works, not squeezing out the last 0.1% accuracy.
    optimizer = keras.optimizers.Adam(learning_rate=config.model.learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=config.model.loss,
        metrics=config.model.metrics,
    )

    logger.info(
        "Built baseline CNN — input_shape=%s, params=%d",
        input_shape, model.count_params(),
    )
    return model
