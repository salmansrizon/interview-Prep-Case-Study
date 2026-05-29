"""
End-to-end training pipeline.
Orchestrates preprocessing → vectorization → model training → evaluation.
"""

from typing import Dict, Tuple, Any
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

import config
from src.data.preprocessor import TextPreprocessor
from src.data.loader import DataLoader
from src.features.vectorizer import VectorizerEngine
from src.models.base import BaseClassifier
from src.models.registry import get_model
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Trainer:
    """
    Unified training orchestrator.
    """

    def __init__(self, task: str = "spam") -> None:
        """
        task: 'spam' (binary) or 'intent' (multi-class)
        """
        self.task = task
        self.preprocessor = TextPreprocessor()
        self.vectorizer = VectorizerEngine()
        self.model: BaseClassifier = None

    def prepare_data(self, df: pd.DataFrame) -> Tuple[Any, Any, Any, Any]:
        """Clean, split, and vectorize data."""
        logger.info("Preparing data for task: %s", self.task)

        # 1. Preprocess text
        df["cleaned"] = self.preprocessor.transform_series(df["text"].tolist())

        # 2. Split
        X_train, X_test, y_train, y_test = train_test_split(
            df["cleaned"],
            df["label"],
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE,
            stratify=df["label"],  # Maintain class distribution
        )

        # 3. Vectorize
        X_train_vec = self.vectorizer.fit_transform(X_train.tolist())
        X_test_vec = self.vectorizer.transform(X_test.tolist())

        return X_train_vec, X_test_vec, y_train, y_test

    def run(
        self,
        df: pd.DataFrame,
        model_name: str,
        model_params: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Full pipeline: data prep → train → evaluate → persist.
        """
        model_params = model_params or {}

        # Data pipeline
        X_train, X_test, y_train, y_test = self.prepare_data(df)

        # Model
        self.model = get_model(model_name, **model_params)

        # Train
        logger.info("Training %s...", model_name)
        self.model.train(X_train, y_train)

        # Evaluate
        metrics = self.model.evaluate(X_test, y_test)
        metrics["model_name"] = model_name
        metrics["task"] = self.task
        metrics["test_size"] = len(y_test)

        # Persist artifacts
        self._save_artifacts()

        return metrics

    def _save_artifacts(self) -> None:
        """Save vectorizer and model to disk for inference."""
        vec_path = config.MODEL_DIR / f"{self.task}_vectorizer.pkl"
        model_path = config.MODEL_DIR / f"{self.task}_{self.model.name.lower().replace(' ', '_')}.pkl"

        self.vectorizer.save(vec_path)
        self.model.save(model_path)
        logger.info("Artifacts saved to %s", config.MODEL_DIR)

    def predict_text(self, raw_text: str) -> Tuple[str, Dict[str, float]]:
        """
        Run inference on a single raw string.
        Returns: (predicted_label, probability_dict)
        """
        if not self.model or not self.model.is_trained:
            raise RuntimeError("Trainer has no trained model. Run .run() first.")

        cleaned = self.preprocessor.clean(raw_text)
        vec = self.vectorizer.transform([cleaned])
        pred = self.model.predict(vec)[0]

        try:
            proba = self.model.predict_proba(vec)[0]
            classes = self.model.model.classes_
            prob_dict = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}
        except Exception:
            prob_dict = {pred: 1.0}

        return pred, prob_dict
