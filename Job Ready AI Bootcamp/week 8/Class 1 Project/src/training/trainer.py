"""
Training orchestration for the baseline MNIST CNN.

Wires together data loading, model building, `model.fit`, test-set
evaluation, and artifact persistence (model + history) so both the CLI
entry point and the Streamlit "Train" tab share one code path.

HIGHLIGHTS — why a dedicated `Trainer` class instead of one big function
(or worse, inline code in `app.py`)? Training has several *sequential dependencies*
(load data -> build model -> fit -> evaluate -> persist) where a subtle
bug in one step (e.g. evaluating on the wrong split) can quietly produce
misleading numbers. Keeping this as one orchestrator with clearly named
steps makes each dependency explicit and testable, and — critically —
means `app.py` never touches `model.fit(...)` directly, keeping every
Keras call inside `src/` as required by the engineering standard for
this bootcamp.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from tensorflow import keras

from config import Config, get_config
from src.data.loader import MnistSplits, load_mnist, subsample
from src.models.cnn import build_baseline_cnn
from src.models.registry import ModelRegistry, ModelVersion
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingResult:
    """Everything a caller needs after a training run completes."""

    model: keras.Model
    history: Dict[str, List[float]]
    test_loss: float
    test_accuracy: float
    version: ModelVersion


class Trainer:
    """Trains, evaluates, and persists the baseline CNN."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or get_config()
        self.registry = ModelRegistry(self.config)

    def train(
        self,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        quick_demo: bool = False,
        splits: Optional[MnistSplits] = None,
        callbacks: Optional[List[keras.callbacks.Callback]] = None,
        on_epoch_end: Optional[Callable[[int, Dict[str, float]], None]] = None,
    ) -> TrainingResult:
        """Run the full train -> evaluate -> persist pipeline.

        Args:
            epochs: Number of epochs (defaults to config, or the shorter
                `quick_epochs` value when `quick_demo` is True).
            batch_size: Mini-batch size (defaults to config).
            quick_demo: If True, trains on a small random subset for a
                fast in-browser demo instead of the full 54k-image set.
            splits: Optionally inject pre-loaded data splits (mainly for
                tests, to avoid re-downloading MNIST).
            callbacks: Optional extra Keras callbacks (e.g. a Streamlit
                progress-bar callback) appended to the default ones.
            on_epoch_end: Optional callback invoked as
                `on_epoch_end(epoch, logs)` after every epoch — a simple
                hook for live UI updates without depending on Keras
                callback internals.

        Returns:
            A :class:`TrainingResult` with the trained model, history,
            test metrics, and the registry version record.
        """
        # `quick_demo` swaps in the shorter epoch count from config rather
        # than just letting a short run use fewer epochs "by accident" —
        # this makes the tradeoff (fast-but-lower-accuracy demo vs.
        # slow-but-close-to-99%-accuracy full run) an explicit, named
        # choice a student can reason about instead of a side effect.
        epochs = epochs or (
            self.config.training.quick_epochs if quick_demo else self.config.training.epochs
        )
        batch_size = batch_size or self.config.training.batch_size

        # `splits` can be injected by callers (the Streamlit app caches
        # the loaded MNIST data in `st.session_state` so re-training
        # doesn't re-download/re-split every time; tests inject a tiny
        # pre-built split to stay fast). Defaulting to `load_mnist(...)`
        # keeps the simple CLI path (`python -m src.training.trainer`)
        # working with zero setup.
        data = splits or load_mnist(self.config)
        if quick_demo:
            # Subsampling happens *after* the full train/val split, and
            # only on train/val — never on `data.x_test`/`data.y_test` —
            # so even a quick demo's reported test accuracy is measured
            # against the same full, untouched test set as a full run,
            # making the two numbers genuinely comparable.
            data = subsample(data, self.config.data.quick_subset_fraction, self.config)

        model = build_baseline_cnn(self.config)

        keras_callbacks: List[keras.callbacks.Callback] = [
            # HIGHLIGHTS: why monitor `val_accuracy` (not `val_loss`) for
            # early stopping here? Accuracy is the metric a student actually
            # cares about and sees in the UI, and for a balanced
            # multi-class problem like MNIST digit classification the two
            # tend to track closely — using the same metric the app
            # reports avoids a confusing situation where training stops
            # "early" by one measure while another is still improving.
            # `restore_best_weights=True` ensures that if accuracy dipped
            # in the final epochs (overfitting), we still hand back the
            # best-performing checkpoint, not just the last one.
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=self.config.training.early_stopping_patience,
                restore_best_weights=True,
                verbose=0,
            )
        ]
        if on_epoch_end is not None:
            # This hook exists specifically so `app.py` can update a
            # Streamlit progress bar live, without the trainer needing to
            # import Streamlit or know anything about the UI — the
            # trainer stays UI-agnostic and testable, and the app supplies
            # a plain callback function instead.
            keras_callbacks.append(_EpochEndCallback(on_epoch_end))
        if callbacks:
            keras_callbacks.extend(callbacks)

        logger.info(
            "Starting training — epochs=%d, batch_size=%d, quick_demo=%s, train_samples=%d",
            epochs, batch_size, quick_demo, len(data.x_train),
        )

        history = model.fit(
            data.x_train,
            data.y_train,
            # Passing `validation_data` explicitly (rather than Keras's
            # `validation_split=`) matters here because our validation set
            # was already carved out by `load_mnist` using a fixed random
            # seed — reusing `validation_split` on top of that would
            # validate on yet another arbitrary slice, and worse, Keras's
            # `validation_split` takes the *last* N% of the array
            # un-shuffled, which could easily be biased toward whichever
            # digits happen to sort last after our own shuffling.
            validation_data=(data.x_val, data.y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=keras_callbacks,
            verbose=0,
        )

        # The held-out Keras test set (never seen during `.fit`, not even
        # for early stopping) is the only number we report as "test
        # accuracy" — this is what gets embedded in the registry's
        # versioned filename and is the figure a student would quote as
        # the model's real-world performance.
        test_loss, test_accuracy = model.evaluate(data.x_test, data.y_test, verbose=0)
        logger.info("Test evaluation — loss=%.4f, accuracy=%.4f", test_loss, test_accuracy)

        history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        self._save_history(history_dict)

        version = self.registry.save(
            model,
            test_accuracy=test_accuracy,
            test_loss=test_loss,
            epochs_trained=len(history_dict.get("loss", [])),
        )

        return TrainingResult(
            model=model,
            history=history_dict,
            test_loss=float(test_loss),
            test_accuracy=float(test_accuracy),
            version=version,
        )

    def _save_history(self, history_dict: Dict[str, List[float]]) -> None:
        """Persist the epoch-by-epoch train/val loss & accuracy curves.

        Saved separately from the model weights (`.keras` file) because
        the two serve different purposes: the model is what you load to
        make predictions, while this history is what you'd load later to
        re-plot the accuracy/loss curves — e.g. comparing a quick-demo run
        against a full run — without needing to re-train either.
        """
        path = self.config.paths.history_dir / self.config.training.history_filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(history_dict, f, indent=2)
        logger.info("Training history saved to %s", path)


class _EpochEndCallback(keras.callbacks.Callback):
    """Thin Keras callback that forwards `on_epoch_end` to a plain function.

    Subclassing `keras.callbacks.Callback` is the only way to hook into
    Keras's per-epoch loop, but we don't want that Keras-specific
    machinery to leak into `app.py`. This adapter is intentionally tiny —
    its only job is translating Keras's callback protocol into a plain
    `(epoch, logs) -> None` function call, so the Streamlit layer can stay
    oblivious to how Keras callbacks work.
    """

    def __init__(self, fn: Callable[[int, Dict[str, float]], None]) -> None:
        super().__init__()
        self._fn = fn

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None) -> None:
        self._fn(epoch, logs or {})
