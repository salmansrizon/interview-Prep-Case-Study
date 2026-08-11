"""
Training orchestration for the three MNIST CNN variants.

``train_variant`` trains a single model and returns its history plus test
metrics. ``compare_variants`` trains all three (baseline, dropout,
optimized) back to back and returns a tidy comparison table — this backs
both the "Train & Compare" and "Model Insights" tabs of the Streamlit app.

HIGHLIGHTS: this module is where the "training loop" side of
optimization lives (as opposed to architectures.py, which handles the
"model structure" side). Concretely, this is where EarlyStopping and
ReduceLROnPlateau get attached — and, just as importantly, where you can
see that they are *only* attached to the "optimized" variant. That's not
an accident: the baseline and dropout variants are meant to show what
training looks like *without* those safeguards, so wiring them in
everywhere would quietly erase the comparison the whole project is built
to demonstrate.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
from tensorflow import keras

from config import TrainingConfig, get_config
from src.data.loader import MNISTData
from src.models.architectures import build_variant
from src.models.registry import save_variant
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class VariantResult:
    """Everything the app needs to know about one trained variant."""

    name: str
    model: keras.Model
    history: Dict[str, List[float]]
    test_loss: float
    test_accuracy: float
    epochs_run: int
    train_time_seconds: float

    @property
    def final_train_accuracy(self) -> float:
        return float(self.history["accuracy"][-1])

    @property
    def final_val_accuracy(self) -> float:
        return float(self.history["val_accuracy"][-1])

    @property
    def overfitting_gap(self) -> float:
        """Train-minus-val accuracy gap at the final (or restored best) epoch.

        HIGHLIGHTS: this single number is the headline diagnostic for
        overfitting used throughout the app. A model that has *memorized*
        its training set scores much higher on data it has seen (train)
        than on data it hasn't (val), so a large positive gap means
        overfitting. A gap near zero means the model generalizes about as
        well as it fits — the "good fit" outcome regularization aims for.
        (A large *negative* gap would suggest underfitting or a fluke, e.g.
        Dropout being active during a train-accuracy evaluation.)
        """
        return self.final_train_accuracy - self.final_val_accuracy


def _callbacks_for(variant: str, cfg: TrainingConfig) -> List[keras.callbacks.Callback]:
    """Build the callback list for a given variant.

    Only the "optimized" variant uses EarlyStopping / ReduceLROnPlateau, by
    design — the whole point of the baseline and dropout variants is to
    show what happens *without* those safeguards.
    """
    if variant != "optimized":
        return []

    return [
        # WHY monitor="val_loss" instead of "val_accuracy"?
        # Loss is a continuous, smooth, more sensitive signal than accuracy.
        # Accuracy on MNIST is often "clumpy" near the top end (99.1% vs
        # 99.2% both round to "basically the same 10,000 predictions
        # correct/incorrect"), so it can plateau for several epochs even
        # while the model is quietly getting *worse* at generalizing —
        # which shows up immediately as a rising val_loss, because loss
        # also captures how *confident* the wrong predictions are, not
        # just whether they crossed the argmax threshold. Monitoring loss
        # catches overfitting earlier than monitoring accuracy would.
        keras.callbacks.EarlyStopping(
            monitor=cfg.early_stopping_monitor,
            # patience=5: don't stop the instant val_loss ticks up once —
            # a single bad epoch can be noise (mini-batch randomness,
            # learning-rate schedule, etc). Wait for 5 consecutive epochs
            # with no improvement before concluding the model has actually
            # stopped getting better.
            patience=cfg.early_stopping_patience,
            # restore_best_weights=True is what makes EarlyStopping safe to
            # use aggressively: even though training keeps running for
            # `patience` extra epochs after the best epoch (to confirm the
            # trend), the final model handed back to us is rolled back to
            # the weights from that best epoch, not whatever epoch we
            # happened to stop on.
            restore_best_weights=True,
            verbose=cfg.verbose,
        ),
        # ReduceLROnPlateau is a complementary, gentler safeguard: instead
        # of stopping outright when val_loss stalls, first try shrinking
        # the learning rate (smaller steps can help the optimizer settle
        # into a sharper minimum). It shares the same val_loss monitor and
        # a shorter patience than EarlyStopping, so in practice the
        # learning rate gets a chance to drop once or twice *before*
        # EarlyStopping gives up entirely.
        keras.callbacks.ReduceLROnPlateau(
            monitor=cfg.early_stopping_monitor,
            factor=cfg.reduce_lr_factor,
            patience=cfg.reduce_lr_patience,
            min_lr=cfg.reduce_lr_min_lr,
            verbose=cfg.verbose,
        ),
    ]


_EPOCHS_ATTR = {
    "baseline": "baseline_epochs",
    "dropout": "dropout_epochs",
    "optimized": "optimized_epochs",
}


def train_variant(
    variant: str,
    data: MNISTData,
    epochs: Optional[int] = None,
    training_cfg: Optional[TrainingConfig] = None,
    save: bool = True,
) -> VariantResult:
    """Train a single CNN variant and evaluate it on the held-out test set.

    Args:
        variant: One of "baseline", "dropout", "optimized".
        data: Prepared MNIST train/val/test split.
        epochs: Optional epoch override (defaults to the variant's
            configured epoch budget).
        training_cfg: Optional training configuration override.
        save: Whether to persist the trained model via the model registry.

    Returns:
        A ``VariantResult`` with the trained model, history, and test metrics.
    """
    cfg = training_cfg or get_config().training
    epochs = epochs or getattr(cfg, _EPOCHS_ATTR[variant])

    # NOTE the phrasing "for up to N epochs": for baseline/dropout this is
    # a hard target because they have no EarlyStopping — they will run
    # every one of those epochs even if val_loss got worse ten epochs ago.
    # For the optimized variant, `epochs` is a *ceiling*: EarlyStopping can
    # (and usually does) stop training well before it's reached, which is
    # exactly why we look at `epochs_run` after training rather than
    # assuming it equals the requested `epochs`.
    logger.info("Training variant='%s' for up to %d epochs", variant, epochs)
    model = build_variant(variant)
    callbacks = _callbacks_for(variant, cfg)

    start = time.time()
    history = model.fit(
        data.x_train,
        data.y_train,
        validation_data=(data.x_val, data.y_val),
        epochs=epochs,
        batch_size=cfg.batch_size,
        callbacks=callbacks,
        verbose=cfg.verbose,
    )
    elapsed = time.time() - start

    test_loss, test_accuracy = model.evaluate(data.x_test, data.y_test, verbose=0)
    epochs_run = len(history.history["loss"])

    if save:
        save_variant(model, variant)

    logger.info(
        "Finished variant='%s': epochs_run=%d test_accuracy=%.4f train_time=%.1fs",
        variant, epochs_run, test_accuracy, elapsed,
    )

    return VariantResult(
        name=variant,
        model=model,
        history=history.history,
        test_loss=float(test_loss),
        test_accuracy=float(test_accuracy),
        epochs_run=epochs_run,
        train_time_seconds=elapsed,
    )


def compare_variants(
    data: MNISTData,
    variants: Optional[List[str]] = None,
    epochs: Optional[int] = None,
) -> Dict[str, VariantResult]:
    """Train multiple variants back-to-back and collect their results.

    Args:
        data: Prepared MNIST train/val/test split (shared across variants).
        variants: Which variants to train, defaults to all three.
        epochs: Optional shared epoch override applied to every variant.

    Returns:
        A dict mapping variant name -> ``VariantResult``.
    """
    variants = variants or ["baseline", "dropout", "optimized"]
    results: Dict[str, VariantResult] = {}
    for variant in variants:
        results[variant] = train_variant(variant, data, epochs=epochs)
    return results


def results_to_dataframe(results: Dict[str, VariantResult]) -> pd.DataFrame:
    """Summarize a set of ``VariantResult`` objects as a comparison DataFrame.

    Columns: variant, test_accuracy, test_loss, train_accuracy, val_accuracy,
    overfitting_gap, epochs_run, train_time_seconds.
    """
    rows = []
    for name, result in results.items():
        rows.append(
            {
                "variant": name,
                "test_accuracy": result.test_accuracy,
                "test_loss": result.test_loss,
                "train_accuracy": result.final_train_accuracy,
                "val_accuracy": result.final_val_accuracy,
                "overfitting_gap": result.overfitting_gap,
                "epochs_run": result.epochs_run,
                "train_time_seconds": result.train_time_seconds,
            }
        )
    return pd.DataFrame(rows)
