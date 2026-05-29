"""
Feature engineering utilities for customer segmentation.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    """Creates derived features for better segmentation."""

    @staticmethod
    def create_rfm_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create RFM (Recency, Frequency, Monetary) features.
        Classic customer segmentation metrics.
        """
        logger.info("Creating RFM features")
        result = df.copy()

        # Recency: lower days_since = higher recency score
        result["recency_score"] = 1 / (1 + result["days_since_last_purchase"])

        # Frequency: normalize purchase frequency
        max_freq = result["purchase_frequency"].max()
        result["frequency_score"] = result["purchase_frequency"] / max_freq

        # Monetary: normalize total spend
        max_spend = result["total_spend"].max()
        result["monetary_score"] = result["total_spend"] / max_spend

        # Customer Lifetime Value proxy
        result["clv_proxy"] = result["total_spend"] * result["satisfaction_score"] / 5

        # Engagement score
        result["engagement_score"] = (
            result["purchase_frequency"] * 0.4 +
            result["online_purchase_ratio"] * 0.3 +
            (5 - result["returns_count"]) * 0.3
        )

        logger.info("Created RFM + derived features")
        return result

    @staticmethod
    def select_top_features(df: pd.DataFrame, n_features: int = 8) -> List[str]:
        """Select most variance-rich features for clustering."""
        variances = df.var().sort_values(ascending=False)
        selected = variances.head(n_features).index.tolist()
        logger.info("Selected top %d features by variance: %s", n_features, selected)
        return selected
