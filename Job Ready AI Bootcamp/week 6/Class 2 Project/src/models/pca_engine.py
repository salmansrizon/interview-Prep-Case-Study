"""
Principal Component Analysis (PCA) Engine
─────────────────────────────────────────
Reduces dimensionality while preserving maximum variance.
Uses scikit-learn PCA implementation.
"""

from typing import Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import joblib

from src.utils.logger import get_logger
import config

logger = get_logger(__name__)


class PCAEngine:
    """
    PCA for dimensionality reduction and visualization.
    """

    def __init__(
        self,
        n_components: int = config.PCA_DEFAULT_COMPONENTS,
        random_state: int = config.RANDOM_STATE,
    ) -> None:
        self.n_components = n_components
        self.random_state = random_state
        self.model = PCA(n_components=n_components, random_state=random_state)
        self.is_fitted = False
        self.explained_variance: np.ndarray = None
        self.cumulative_variance: np.ndarray = None
        self.total_variance: float = 0.0
        self.components: np.ndarray = None

    def fit(self, X: pd.DataFrame) -> None:
        """Fit PCA and compute variance statistics."""
        logger.info("Fitting PCA with %d components", self.n_components)
        self.model.fit(X)
        self.is_fitted = True

        self.explained_variance = self.model.explained_variance_ratio_
        self.cumulative_variance = np.cumsum(self.explained_variance)
        self.total_variance = self.cumulative_variance[-1]
        self.components = self.model.components_

        logger.info(
            "PCA fitted: %d components explain %.1f%% variance",
            self.n_components, self.total_variance * 100,
        )

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform data to principal component space."""
        if not self.is_fitted:
            raise RuntimeError("PCA not fitted. Call fit() first.")
        return self.model.transform(X)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """Reconstruct original space from PCA components."""
        if not self.is_fitted:
            raise RuntimeError("PCA not fitted.")
        return self.model.inverse_transform(X_transformed)

    def save(self, path: Path) -> None:
        joblib.dump(self.model, path)
        logger.info("PCA model saved to %s", path)

    def load(self, path: Path) -> None:
        self.model = joblib.load(path)
        self.is_fitted = True
        logger.info("PCA model loaded from %s", path)
