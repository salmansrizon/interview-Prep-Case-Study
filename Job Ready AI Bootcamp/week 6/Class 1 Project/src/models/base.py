"""
Abstract base class enforcing a consistent interface across
Naive Bayes, SVM, and KNN. Demonstrates OOP principles.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


class BaseClassifier(ABC):
    """
    All classifiers must implement: train, predict, evaluate, save, load.
    """

    def __init__(self, name: str, model: Any) -> None:
        self.name = name
        self.model = model
        self.is_trained = False

    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError(f"{self.name} has not been trained yet.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probabilities if the underlying model supports it."""
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        # SVM and some KNN configs may need decision_function or manual fallback
        if hasattr(self.model, "decision_function"):
            # Convert decision scores to pseudo-probabilities via softmax
            scores = self.model.decision_function(X)
            exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        raise NotImplementedError(f"{self.name} does not support probability estimates.")

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        y_pred = self.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="weighted", zero_division=0
        )
        cm = confusion_matrix(y_test, y_pred)

        return {
            "accuracy": round(acc, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "confusion_matrix": cm,
            "predictions": y_pred,
        }

    def save(self, path: Path) -> None:
        joblib.dump(self.model, path)

    def load(self, path: Path) -> None:
        self.model = joblib.load(path)
        self.is_trained = True
