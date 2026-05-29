"""
K-Means Clustering Engine
─────────────────────────
Partitions data into K clusters by minimizing within-cluster
sum of squares (WCSS). Uses scikit-learn implementation.
"""

from typing import Tuple, Any, Dict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import joblib

from src.utils.logger import get_logger
import config

logger = get_logger(__name__)


class KMeansEngine:
    """
    Production-grade K-Means clustering with evaluation metrics.
    """

    def __init__(
        self,
        n_clusters: int = config.KMEANS_DEFAULT_CLUSTERS,
        init: str = "k-means++",
        max_iter: int = config.KMEANS_MAX_ITER,
        n_init: int = config.KMEANS_N_INIT,
        random_state: int = config.RANDOM_STATE,
    ) -> None:
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state

        self.model = KMeans(
            n_clusters=n_clusters,
            init=init,
            max_iter=max_iter,
            n_init=n_init,
            random_state=random_state,
        )
        self.is_fitted = False
        self.inertia: float = 0.0
        self.silhouette: float = 0.0
        self.calinski: float = 0.0
        self.labels: np.ndarray = None
        self.centers: np.ndarray = None

    def fit(self, X: pd.DataFrame) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Fit K-Means and compute evaluation metrics.
        Returns: (labels, inertia, cluster_centers)
        """
        logger.info("Fitting K-Means with K=%d", self.n_clusters)

        self.labels = self.model.fit_predict(X)
        self.centers = self.model.cluster_centers_
        self.inertia = self.model.inertia_
        self.is_fitted = True

        # Evaluation metrics
        self.silhouette = silhouette_score(X, self.labels)
        self.calinski = calinski_harabasz_score(X, self.labels)

        logger.info(
            "K-Means fitted: inertia=%.2f, silhouette=%.3f, calinski=%.1f",
            self.inertia, self.silhouette, self.calinski,
        )
        return self.labels, self.inertia, self.centers

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Assign new data to clusters."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.model.predict(X)

    def get_cluster_profiles(self, X: pd.DataFrame) -> pd.DataFrame:
        """Get mean feature values per cluster."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        df = X.copy()
        df["cluster"] = self.labels
        return df.groupby("cluster").mean()

    def save(self, path: Path) -> None:
        joblib.dump(self.model, path)
        logger.info("K-Means model saved to %s", path)

    def load(self, path: Path) -> None:
        self.model = joblib.load(path)
        self.is_fitted = True
        logger.info("K-Means model loaded from %s", path)
