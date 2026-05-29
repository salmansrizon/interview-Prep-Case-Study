"""
Model Factory.
Allows the Streamlit app to instantiate models by string name,
keeping the UI decoupled from implementation details.
"""

from typing import Dict, Type

from src.models.base import BaseClassifier
from src.models.naive_bayes import NaiveBayesClassifier
from src.models.svm import SVMClassifier
from src.models.knn import KNNClassifier


MODEL_REGISTRY: Dict[str, Type[BaseClassifier]] = {
    "Naive Bayes": NaiveBayesClassifier,
    "SVM": SVMClassifier,
    "KNN": KNNClassifier,
}


def get_model(name: str, **kwargs) -> BaseClassifier:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Choose from {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)
