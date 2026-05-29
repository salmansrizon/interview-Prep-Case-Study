"""
Model Evaluator.

Computes comprehensive regression metrics, generates visualizations,
and produces an evaluation report.
"""

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    explained_variance_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import get_config
from src.utils import logger
from src.data.loader import load_preprocessed_data
from src.data.preprocessor import DataPreprocessor


class ModelEvaluator:
    """Evaluate a trained model and produce metrics + visualizations."""

    def __init__(self, model_path: str = None, preprocessor_path: str = None):
        self.config = get_config()
        self.model_path = model_path or os.path.join(self.config.paths.models_dir, "best_model.keras")
        self.preprocessor_path = preprocessor_path or os.path.join(self.config.paths.data_processed, "preprocessor.joblib")

        self.model = None
        self.preprocessor = None
        self.metrics: Dict[str, float] = {}
        self.y_true = None
        self.y_pred = None

    def load_artifacts(self) -> None:
        """Load model and preprocessor from disk."""
        logger.info("Loading model from {}", self.model_path)
        self.model = keras.models.load_model(self.model_path)

        logger.info("Loading preprocessor from {}", self.preprocessor_path)
        self.preprocessor = DataPreprocessor()
        self.preprocessor.load(self.preprocessor_path)

    def evaluate(self, processed_dir: str = None, dataset: str = "test") -> Dict[str, float]:
        """
        Evaluate model on a dataset split.

        Args:
            processed_dir: Directory with preprocessed .npy files
            dataset: Which split to evaluate — "train", "val", or "test"

        Returns:
            Dictionary of metric names and values
        """
        processed_dir = processed_dir or self.config.paths.data_processed
        self.load_artifacts()

        # Load data
        X_train, X_val, X_test, y_train, y_val, y_test = load_preprocessed_data(processed_dir)

        if dataset == "train":
            X, y = X_train, y_train
        elif dataset == "val":
            X, y = X_val, y_val
        else:
            X, y = X_test, y_test

        # Predict
        logger.info("Running predictions on {} set ({} samples)...", dataset, len(X))
        y_pred_scaled = self.model.predict(X, verbose=0).flatten()

        # Inverse transform to original scale
        self.y_true = self.preprocessor.inverse_transform_target(y)
        self.y_pred = self.preprocessor.inverse_transform_target(y_pred_scaled)

        # Compute metrics
        self.metrics = {
            "mae": mean_absolute_error(self.y_true, self.y_pred),
            "mse": mean_squared_error(self.y_true, self.y_pred),
            "rmse": np.sqrt(mean_squared_error(self.y_true, self.y_pred)),
            "r2": r2_score(self.y_true, self.y_pred),
            "explained_variance": explained_variance_score(self.y_true, self.y_pred),
            "mape": np.mean(np.abs((self.y_true - self.y_pred) / (self.y_true + 1e-8))) * 100,
        }

        logger.info("{} set metrics: {}", dataset, self.metrics)
        return self.metrics

    def plot_predictions(self, output_path: str = None) -> str:
        """
        Generate scatter plot: predicted vs actual.

        Returns:
            Path to saved plot image
        """
        if self.y_true is None or self.y_pred is None:
            raise RuntimeError("Must call evaluate() before plotting.")

        output_path = output_path or os.path.join(self.config.paths.artifacts_dir, "predictions_scatter.png")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        plt.figure(figsize=(8, 8))
        sns.scatterplot(x=self.y_true, y=self.y_pred, alpha=0.5, edgecolor=None)

        # Perfect prediction line
        min_val = min(self.y_true.min(), self.y_pred.min())
        max_val = max(self.y_true.max(), self.y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Prediction")

        plt.xlabel("Actual Success Score", fontsize=12)
        plt.ylabel("Predicted Success Score", fontsize=12)
        plt.title("Predicted vs Actual Success Score", fontsize=14, fontweight="bold")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info("Prediction scatter plot saved to {}", output_path)
        return output_path

    def plot_residuals(self, output_path: str = None) -> str:
        """Generate residual distribution plot."""
        if self.y_true is None or self.y_pred is None:
            raise RuntimeError("Must call evaluate() before plotting.")

        output_path = output_path or os.path.join(self.config.paths.artifacts_dir, "residuals_distribution.png")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        residuals = self.y_true - self.y_pred

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Residuals vs predicted
        sns.scatterplot(x=self.y_pred, y=residuals, alpha=0.5, ax=axes[0], edgecolor=None)
        axes[0].axhline(y=0, color="r", linestyle="--")
        axes[0].set_xlabel("Predicted Success Score")
        axes[0].set_ylabel("Residuals (Actual - Predicted)")
        axes[0].set_title("Residuals vs Predicted")
        axes[0].grid(True, alpha=0.3)

        # Residuals histogram
        sns.histplot(residuals, kde=True, bins=50, ax=axes[1], color="steelblue")
        axes[1].set_xlabel("Residuals")
        axes[1].set_title("Residuals Distribution")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info("Residuals plot saved to {}", output_path)
        return output_path

    def plot_feature_importance(self, output_path: str = None, top_n: int = 15) -> str:
        """
        Approximate feature importance using permutation importance on a sample.

        Returns:
            Path to saved plot image
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_artifacts() first.")

        output_path = output_path or os.path.join(self.config.paths.artifacts_dir, "feature_importance.png")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Load a sample of test data
        processed_dir = self.config.paths.data_processed
        X_train, X_val, X_test, y_train, y_val, y_test = load_preprocessed_data(processed_dir)

        # Use small sample for speed
        sample_size = min(500, len(X_test))
        X_sample = X_test[:sample_size]
        y_sample = y_test[:sample_size]

        baseline_pred = self.model.predict(X_sample, verbose=0).flatten()
        baseline_mse = mean_squared_error(y_sample, baseline_pred)

        importances = []
        feature_names = self.preprocessor.feature_names_after_transform

        for i in range(X_sample.shape[1]):
            X_permuted = X_sample.copy()
            np.random.shuffle(X_permuted[:, i])
            permuted_pred = self.model.predict(X_permuted, verbose=0).flatten()
            permuted_mse = mean_squared_error(y_sample, permuted_pred)
            importances.append(permuted_mse - baseline_mse)

        # Sort and plot top N
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=True).tail(top_n)

        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df, y="feature", x="importance", palette="viridis")
        plt.xlabel("Importance (MSE Increase When Permuted)")
        plt.ylabel("Feature")
        plt.title(f"Top {top_n} Feature Importances (Permutation Method)", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info("Feature importance plot saved to {}", output_path)
        return output_path

    def generate_report(self, output_dir: str = None) -> str:
        """
        Generate a comprehensive evaluation report with all metrics and plots.

        Returns:
            Path to saved report directory
        """
        output_dir = output_dir or self.config.paths.artifacts_dir
        os.makedirs(output_dir, exist_ok=True)

        # Evaluate on all splits
        test_metrics = self.evaluate(dataset="test")
        val_metrics = self.evaluate(dataset="val")

        # Generate plots
        self.plot_predictions(os.path.join(output_dir, "predictions_scatter.png"))
        self.plot_residuals(os.path.join(output_dir, "residuals_distribution.png"))
        self.plot_feature_importance(os.path.join(output_dir, "feature_importance.png"))

        # Save metrics JSON
        report = {
            "test_metrics": {k: float(v) for k, v in test_metrics.items()},
            "val_metrics": {k: float(v) for k, v in val_metrics.items()},
        }
        report_path = os.path.join(output_dir, "evaluation_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            import json
            json.dump(report, f, indent=2)

        logger.info("Evaluation report saved to {}", output_dir)
        return output_dir


def main() -> None:
    """CLI entry point for evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--model", type=str, default=None, help="Path to .keras model")
    parser.add_argument("--output", type=str, default=None, help="Output directory for plots")
    args = parser.parse_args()

    evaluator = ModelEvaluator(model_path=args.model)
    evaluator.generate_report(output_dir=args.output)
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
