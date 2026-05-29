"""
K-Nearest Neighbors (KNN)
─────────────────────────
Instance-based learning: classification is determined by the
majority label among the 'k' closest training examples in
vector space.

Best for: Small-to-medium datasets with clear local structure.
Strength: Simple, intuitive, no training phase (lazy learning).
Weakness: Slow at inference time; curse of dimensionality.
"""

from sklearn.neighbors import KNeighborsClassifier
import numpy as np

from src.models.base import BaseClassifier
import config


class KNNClassifier(BaseClassifier):
    def __init__(
        self,
        n_neighbors: int = config.KNN_N_NEIGHBORS,
        weights: str = config.KNN_WEIGHTS,
    ) -> None:
        super().__init__(
            name="K-Nearest Neighbors",
            model=KNeighborsClassifier(
                n_neighbors=n_neighbors,
                weights=weights,
                metric="cosine",  # Cosine similarity is excellent for text
            ),
        )

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        # KNN is lazy, but we still 'fit' to store the training data
        self.model.fit(X_train, y_train)
        self.is_trained = True
