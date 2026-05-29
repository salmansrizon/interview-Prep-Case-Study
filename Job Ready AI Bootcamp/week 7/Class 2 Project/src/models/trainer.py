"""
Model Trainer.

Orchestrates the full training loop: load data, build model, train with callbacks,
save history, and persist final artifacts.
"""

import os
import json
from typing import Dict, Any

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.config import get_config
from src.utils import logger
from src.data.loader import get_datasets, get_input_dim
from src.models.builder import build_model, get_callbacks


class ModelTrainer:
    """Handles model training, history tracking, and artifact saving."""

    def __init__(self, config=None):
        self.config = config or get_config()
        self.model = None
        self.history = None
        self.training_metadata: Dict[str, Any] = {}

    def train(self, processed_dir: str = None, epochs: int = None, batch_size: int = None) -> keras.callbacks.History:
        """
        Full training pipeline.

        Steps:
            1. Load preprocessed datasets
            2. Infer input dimension from data
            3. Build model
            4. Train with callbacks
            5. Save training history and metadata

        Returns:
            Keras History object
        """
        processed_dir = processed_dir or self.config.paths.data_processed
        epochs = epochs or self.config.model.epochs
        batch_size = batch_size or self.config.model.batch_size

        # Load data
        logger.info("Loading datasets for training...")
        train_ds, val_ds, test_ds, X_test, y_test = get_datasets(processed_dir, batch_size)

        # Infer input dimension from actual data (overrides config if needed)
        input_dim = get_input_dim(processed_dir)
        logger.info("Inferred input dimension from data: {}", input_dim)

        # Build model
        self.model = build_model(input_dim=input_dim)

        # Log model architecture
        logger.info("Model architecture:")
        self.model.summary(print_fn=logger.info)

        # Train
        logger.info("Starting training — epochs={}, batch_size={}", epochs, batch_size)
        callbacks = get_callbacks(self.config)

        self.history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
        )

        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_metrics = self.model.evaluate(test_ds, verbose=0, return_dict=True)
        logger.info("Test metrics: {}", test_metrics)

        # Save artifacts
        self._save_artifacts(test_metrics)

        return self.history

    def _save_artifacts(self, test_metrics: Dict[str, float]) -> None:
        """Save model, history, and metadata."""
        # Save final model
        final_model_path = os.path.join(self.config.paths.models_dir, "final_model.keras")
        self.model.save(final_model_path)
        logger.info("Final model saved to {}", final_model_path)

        # Save history
        history_path = os.path.join(self.config.paths.models_dir, "training_history.json")
        history_dict = {k: [float(v) for v in vals] for k, vals in self.history.history.items()}
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history_dict, f, indent=2)
        logger.info("Training history saved to {}", history_path)

        # Save metadata
        self.training_metadata = {
            "epochs_trained": len(self.history.history["loss"]),
            "final_train_loss": float(self.history.history["loss"][-1]),
            "final_val_loss": float(self.history.history["val_loss"][-1]),
            "test_metrics": {k: float(v) for k, v in test_metrics.items()},
            "model_params": self.model.count_params(),
            "model_path": final_model_path,
        }
        metadata_path = os.path.join(self.config.paths.models_dir, "training_metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.training_metadata, f, indent=2)
        logger.info("Training metadata saved to {}", metadata_path)

    def load_best_model(self) -> keras.Model:
        """Load the best model saved by ModelCheckpoint."""
        best_path = os.path.join(self.config.paths.models_dir, "best_model.keras")
        if os.path.exists(best_path):
            logger.info("Loading best model from {}", best_path)
            self.model = keras.models.load_model(best_path)
            return self.model
        else:
            logger.warning("Best model not found at {}. Returning current model.", best_path)
            return self.model


def main() -> None:
    """CLI entry point for training."""
    import argparse

    parser = argparse.ArgumentParser(description="Train the equipment success predictor")
    parser.add_argument("--processed-dir", type=str, default=None, help="Path to preprocessed data")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    args = parser.parse_args()

    trainer = ModelTrainer()
    trainer.train(
        processed_dir=args.processed_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
