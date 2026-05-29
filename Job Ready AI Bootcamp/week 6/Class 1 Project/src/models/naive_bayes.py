"""
Naive Bayes Classifier
──────────────────────
Assumes feature independence (hence 'naive'). Despite this
simplification, it performs exceptionally well on text due to
the high dimensionality of TF-IDF vectors.

Best for: Spam detection, topic categorization.
Strength: Fast training, works well with small data.
"""

from sklearn.naive_bayes import MultinomialNB
import numpy as np

from src.models.base import BaseClassifier
import config


class NaiveBayesClassifier(BaseClassifier):
    def __init__(self, alpha: float = config.NB_ALPHA) -> None:
        super().__init__(
            name="Naive Bayes",
            model=MultinomialNB(alpha=alpha),
        )

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.model.fit(X_train, y_train)
        self.is_trained = True
