"""
Support Vector Machine (SVM)
────────────────────────────
Finds the optimal hyperplane that maximizes the margin between
classes. The 'kernel trick' allows non-linear separation, but
for text data, a linear kernel is usually sufficient and much faster.

Best for: High-dimensional sparse data (like TF-IDF).
Strength: Robust against overfitting in high-dim spaces.
"""

from sklearn.svm import LinearSVC
import numpy as np

from src.models.base import BaseClassifier
import config


class SVMClassifier(BaseClassifier):
    def __init__(self, C: float = config.SVM_C) -> None:
        # LinearSVC is faster than SVC(kernel='linear') for text
        super().__init__(
            name="Support Vector Machine",
            model=LinearSVC(C=C, random_state=config.RANDOM_STATE, dual=False),
        )

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.model.fit(X_train, y_train)
        self.is_trained = True
