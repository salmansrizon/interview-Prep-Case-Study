"""
Data preprocessing for unsupervised learning.
Handles scaling, encoding, and feature selection.
"""

from typing import List
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataPreprocessor:
    """Preprocesses raw data into ML-ready features."""

    def __init__(self, scaling: str = "standard") -> None:
        self.scaling = scaling
        self.scaler = StandardScaler() if scaling == "standard" else MinMaxScaler()
        self.feature_cols: List[str] = []
        self.is_fitted = False

    def prepare_for_clustering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract and scale numeric features for clustering algorithms.
        Drops ID columns and non-numeric data.
        """
        logger.info("Preprocessing data for clustering")

        # Drop identifier columns
        drop_cols = [c for c in df.columns if "id" in c.lower() or "ground_truth" in c.lower()]
        features = df.drop(columns=drop_cols, errors="ignore")

        # Keep only numeric columns
        numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
        features = features[numeric_cols]

        # Handle missing values
        features = features.fillna(features.mean())

        # Scale features
        scaled = self.scaler.fit_transform(features)
        self.is_fitted = True
        self.feature_cols = numeric_cols

        result = pd.DataFrame(scaled, columns=numeric_cols)
        logger.info("Prepared %d features: %s", len(numeric_cols), numeric_cols)
        return result

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted scaler."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted first.")
        features = df[self.feature_cols].fillna(0)
        scaled = self.scaler.transform(features)
        return pd.DataFrame(scaled, columns=self.feature_cols)
