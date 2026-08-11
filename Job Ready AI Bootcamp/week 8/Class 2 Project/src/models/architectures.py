"""
CNN architecture builders for the three MNIST classifier variants.

    - build_baseline_cnn()   -> no regularization, prone to overfitting.
    - build_dropout_cnn()    -> baseline + Dropout(0.5) before the output layer.
    - build_optimized_cnn()  -> BatchNorm + Dropout(0.3) throughout, meant to
                                be paired with EarlyStopping / ReduceLROnPlateau
                                during training (see src/training/trainer.py).

HIGHLIGHTS — why three separate builder functions instead of one
parameterized function with `use_dropout=True/False` flags?
Because this project's whole purpose is a *controlled experiment*: each
variant changes exactly ONE thing relative to the one before it
(baseline -> +Dropout -> +BatchNorm +EarlyStopping), so that when you
compare their train/val curves later, any difference you see is
attributable to that one change. A single mega-function with a dozen
boolean flags would make it easy to accidentally introduce a second
variable (e.g. a different learning rate) and invalidate the comparison.
Explicit, separate functions make the "what changed" story obvious to a
reader who has never seen this file before.

All builders return a compiled ``keras.Model`` so callers never need to
touch Keras/TensorFlow APIs directly (see app.py, which only calls into
src/).
"""

from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers

from config import DataConfig, ModelConfig, get_config


def _optimizer(learning_rate: float) -> keras.optimizers.Optimizer:
    """Adam is used for all three variants (again, to keep the optimizer
    itself from becoming a confounding variable in the comparison)."""
    return keras.optimizers.Adam(learning_rate=learning_rate)


def build_baseline_cnn(
    data_cfg: DataConfig | None = None,
    model_cfg: ModelConfig | None = None,
) -> keras.Model:
    """Build the Baseline CNN: two conv blocks, no regularization.

    Architecture:
        Conv2D(32,3x3,relu) -> MaxPool
        Conv2D(64,3x3,relu) -> MaxPool
        Flatten -> Dense(128,relu) -> Dense(10,softmax)

    HIGHLIGHTS: this is deliberately the "naive" version a student would
    write before learning any optimization techniques. It has enough
    capacity (parameters) to memorize the training set almost perfectly,
    but nothing stops it from doing so — no dropout, no batch norm, and it
    trains for a fixed number of epochs regardless of whether validation
    loss is still improving. Expect train accuracy to climb toward ~100%
    while validation accuracy plateaus below it — that growing gap *is*
    overfitting, and it's the problem the other two variants exist to fix.
    """
    cfg = get_config()
    data_cfg = data_cfg or cfg.data
    model_cfg = model_cfg or cfg.model

    model = keras.Sequential(name="baseline_cnn")
    model.add(layers.Input(shape=data_cfg.input_shape, name="input"))

    # Conv block 1: learn low-level features (edges, strokes, curves).
    # activation="relu" is baked directly into the Conv2D call here because
    # this variant has no BatchNorm — there's no reason to delay the
    # activation to a separate layer (contrast with build_optimized_cnn
    # below, where BatchNorm has to sit *between* the conv and the ReLU).
    model.add(layers.Conv2D(model_cfg.conv1_filters, model_cfg.kernel_size, activation="relu", name="conv1"))
    model.add(layers.MaxPooling2D(model_cfg.pool_size, name="pool1"))

    # Conv block 2: combine block-1 features into higher-level shapes
    # (loops, intersections) that start to resemble whole digits.
    model.add(layers.Conv2D(model_cfg.conv2_filters, model_cfg.kernel_size, activation="relu", name="conv2"))
    model.add(layers.MaxPooling2D(model_cfg.pool_size, name="pool2"))

    # Flatten the 2D feature maps into a 1D vector so Dense layers can
    # consume them, then classify with a small fully-connected head.
    model.add(layers.Flatten(name="flatten"))
    model.add(layers.Dense(model_cfg.dense_units, activation="relu", name="dense1"))
    model.add(layers.Dense(data_cfg.num_classes, activation="softmax", name="output"))

    model.compile(
        optimizer=_optimizer(model_cfg.learning_rate),
        # sparse_categorical_crossentropy expects integer labels (0-9)
        # rather than one-hot vectors, which matches what
        # keras.datasets.mnist.load_data() gives us — no extra encoding step.
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_dropout_cnn(
    data_cfg: DataConfig | None = None,
    model_cfg: ModelConfig | None = None,
) -> keras.Model:
    """Build the "With Dropout" CNN: baseline conv stack + Dropout before output.

    Architecture:
        Conv2D(32,3x3,relu) -> MaxPool
        Conv2D(64,3x3,relu) -> MaxPool
        Flatten -> Dense(128,relu) -> Dropout(0.5) -> Dense(10,softmax)

    HIGHLIGHTS: this variant changes exactly one thing versus the
    baseline — it inserts a single Dropout layer right before the output.
    During training, Dropout randomly zeroes ~50% of the Dense(128)
    activations on every batch, which forces the network to spread useful
    information across many neurons instead of letting a few neurons
    "specialize" in memorizing particular training examples (a failure
    mode called co-adaptation). At inference time Dropout is automatically
    a no-op (Keras handles the train/inference switch for you), so nothing
    else about how you use the trained model changes.

    Why rate=0.5 here specifically? 0.5 is the rate used in the original
    Dropout paper for the dense/fully-connected layers, and it's a
    reasonable "maximum strength" demonstration of the technique in
    isolation. Contrast this with the Fully Optimized variant below, which
    uses a gentler 0.3 — because there BatchNorm is already doing some of
    the regularization work, so less Dropout is needed on top of it.
    """
    cfg = get_config()
    data_cfg = data_cfg or cfg.data
    model_cfg = model_cfg or cfg.model

    model = keras.Sequential(name="dropout_cnn")
    model.add(layers.Input(shape=data_cfg.input_shape, name="input"))

    # Conv stack is byte-for-byte identical to the baseline on purpose —
    # this isolates Dropout as the only variable in this comparison.
    model.add(layers.Conv2D(model_cfg.conv1_filters, model_cfg.kernel_size, activation="relu", name="conv1"))
    model.add(layers.MaxPooling2D(model_cfg.pool_size, name="pool1"))

    model.add(layers.Conv2D(model_cfg.conv2_filters, model_cfg.kernel_size, activation="relu", name="conv2"))
    model.add(layers.MaxPooling2D(model_cfg.pool_size, name="pool2"))

    model.add(layers.Flatten(name="flatten"))
    model.add(layers.Dense(model_cfg.dense_units, activation="relu", name="dense1"))
    # The one change: drop 50% of dense1's activations at random each
    # training step, right before they feed into the final classifier.
    model.add(layers.Dropout(model_cfg.dropout_dense_rate, name="dropout_output"))
    model.add(layers.Dense(data_cfg.num_classes, activation="softmax", name="output"))

    model.compile(
        optimizer=_optimizer(model_cfg.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_optimized_cnn(
    data_cfg: DataConfig | None = None,
    model_cfg: ModelConfig | None = None,
) -> keras.Model:
    """Build the Fully Optimized CNN: BatchNorm + Dropout throughout.

    Architecture:
        Conv2D(32,3x3) -> BatchNorm -> ReLU -> MaxPool
        Conv2D(64,3x3) -> BatchNorm -> ReLU -> MaxPool
        Flatten -> Dropout(0.3) -> Dense(128,relu) -> Dropout(0.3) -> Dense(10,softmax)

    Intended to be trained with EarlyStopping(monitor="val_loss", patience=5,
    restore_best_weights=True) and optionally ReduceLROnPlateau — see
    ``src/training/trainer.py``. Target: >98% test accuracy.

    HIGHLIGHTS: this stacks multiple techniques, each addressing a
    different failure mode:
      - BatchNormalization tackles *training instability/speed*: without
        it, the distribution of activations feeding each layer keeps
        shifting as earlier layers' weights update ("internal covariate
        shift"), which forces you to use a smaller, more conservative
        learning rate. BatchNorm re-centers and re-scales activations
        every mini-batch, which lets the network train faster and more
        reliably.
      - Dropout tackles *overfitting/memorization*, exactly as in the
        dropout variant above, just at a gentler 0.3 rate because
        BatchNorm's per-batch noise already contributes a mild
        regularization effect of its own — stacking two strong
        regularizers (BatchNorm + Dropout(0.5)) tends to *hurt* accuracy
        by underfitting, so we dial Dropout back here.
      - EarlyStopping (configured in the trainer, not here) tackles
        *wasted compute and late-stage overfitting*: it stops training the
        moment validation loss stops improving, rather than trusting a
        fixed epoch count to be exactly right.
    """
    cfg = get_config()
    data_cfg = data_cfg or cfg.data
    model_cfg = model_cfg or cfg.model

    model = keras.Sequential(name="optimized_cnn")
    model.add(layers.Input(shape=data_cfg.input_shape, name="input"))

    # Conv block 1.
    # IMPORTANT ORDERING: Conv2D (no activation) -> BatchNorm -> ReLU.
    # BatchNorm normalizes the *raw linear output* of the convolution
    # (mean 0, unit variance per channel, then its own learned scale/shift)
    # before the nonlinearity clips negative values to zero. If we applied
    # ReLU first, BatchNorm would be normalizing an already-nonlinear,
    # already-asymmetric distribution (all values >= 0), which is a
    # harder statistic to normalize usefully and is not what the original
    # Batch Normalization paper recommends. Conv -> BN -> ReLU is the
    # standard, well-tested ordering.
    model.add(layers.Conv2D(model_cfg.conv1_filters, model_cfg.kernel_size, name="conv1"))
    model.add(layers.BatchNormalization(name="bn1"))
    model.add(layers.Activation("relu", name="relu1"))
    model.add(layers.MaxPooling2D(model_cfg.pool_size, name="pool1"))

    # Conv block 2: same Conv -> BatchNorm -> ReLU -> Pool pattern, now
    # operating on the richer feature maps produced by block 1.
    model.add(layers.Conv2D(model_cfg.conv2_filters, model_cfg.kernel_size, name="conv2"))
    model.add(layers.BatchNormalization(name="bn2"))
    model.add(layers.Activation("relu", name="relu2"))
    model.add(layers.MaxPooling2D(model_cfg.pool_size, name="pool2"))

    model.add(layers.Flatten(name="flatten"))
    # Dropout after Flatten: regularizes the raw flattened feature vector
    # before it's compressed into the Dense(128) classifier head.
    model.add(layers.Dropout(model_cfg.dropout_conv_rate, name="dropout1"))
    model.add(layers.Dense(model_cfg.dense_units, activation="relu", name="dense1"))
    # A second Dropout after the Dense layer, same rate. Two lighter
    # Dropout applications (0.3 + 0.3) tend to generalize better here than
    # one aggressive Dropout(0.5) in a single spot, because each one only
    # has to compensate for a moderate amount of missing information.
    model.add(layers.Dropout(model_cfg.dropout_conv_rate, name="dropout2"))
    model.add(layers.Dense(data_cfg.num_classes, activation="softmax", name="output"))

    model.compile(
        optimizer=_optimizer(model_cfg.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# A simple string -> builder-function lookup table. Using a plain dict
# (rather than if/elif chains scattered through the codebase) means
# app.py and the trainer can loop over "all variants" generically instead
# of hardcoding three near-identical code paths.
BUILDERS = {
    "baseline": build_baseline_cnn,
    "dropout": build_dropout_cnn,
    "optimized": build_optimized_cnn,
}


def build_variant(name: str) -> keras.Model:
    """Build a model variant by its registry key ("baseline"/"dropout"/"optimized").

    Args:
        name: One of the keys in ``BUILDERS``.

    Returns:
        A freshly built, compiled ``keras.Model``.

    Raises:
        ValueError: If ``name`` is not a known variant.
    """
    if name not in BUILDERS:
        raise ValueError(f"Unknown model variant: {name}. Choose from {list(BUILDERS)}")
    return BUILDERS[name]()
