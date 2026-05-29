"""
Feature extraction: converts cleaned text into numerical vectors.
Uses TF-IDF, the industry standard for text classification.
"""

from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from pathlib import Path

import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class VectorizerEngine:
    """
    Wrapper around scikit-learn's TF-IDF with production defaults.
    """

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            max_features=config.MAX_FEATURES,
            ngram_range=config.NGRAM_RANGE,
            min_df=config.MIN_DF,
            max_df=config.MAX_DF,
            sublinear_tf=True,  # Apply sublinear tf scaling (1 + log(tf))
        )
        self.is_fitted = False

    def fit_transform(self, texts: List[str]) -> Tuple:
        logger.info("Fitting TF-IDF on %d documents...", len(texts))
        X = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
        return X

    def transform(self, texts: List[str]):
        if not self.is_fitted:
            raise RuntimeError("Vectorizer must be fitted before transform.")
        return self.vectorizer.transform(texts)

    def save(self, path: Path) -> None:
        joblib.dump(self.vectorizer, path)
        logger.info("Vectorizer saved to %s", path)

    def load(self, path: Path) -> None:
        self.vectorizer = joblib.load(path)
        self.is_fitted = True
        logger.info("Vectorizer loaded from %s", path)
