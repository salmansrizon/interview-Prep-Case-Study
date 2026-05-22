"""
feature_engineer.py
-------------------
Motive: Transform raw data into model-ready features using Pandas operations.
WHY: Models don't eat raw data. They eat numbers. Feature engineering is the
     art of converting business logic into numerical representations.
WHAT IT DOES: Creates aggregations, encodings, and derived columns.
ANALOGY: This is the "chef's prep station." Raw ingredients (data) are chopped,
         seasoned, and arranged into portions (features) ready for the oven (model).
"""

import pandas as pd
import numpy as np
from typing import List, Dict


class FeatureEngineer:
    """
    Creates features from cleaned data using GroupBy and transformations.
    """

    def __init__(self):
        self.feature_log: List[Dict] = []

    def create_datetime_features(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """
        Extracts features from datetime columns.

        WHY? Models cannot read "2024-01-15 14:30:00". They can read:
        - hour_of_day (14)
        - day_of_week (1 = Monday)
        - is_weekend (0 or 1)
        - month (1)

        These capture temporal patterns (weekend shopping, morning commutes).
        """
        df = df.copy()
        df[col] = pd.to_datetime(df[col], errors="coerce")

        df[f"{col}_hour"] = df[col].dt.hour
        df[f"{col}_dayofweek"] = df[col].dt.dayofweek
        df[f"{col}_month"] = df[col].dt.month
        df[f"{col}_is_weekend"] = (df[col].dt.dayofweek >= 5).astype(int)

        self.feature_log.append({
            "action": "datetime_decomposition",
            "source_column": col,
            "new_columns": [f"{col}_hour", f"{col}_dayofweek", f"{col}_month", f"{col}_is_weekend"],
        })

        return df

    def create_aggregations(
        self,
        df: pd.DataFrame,
        group_col: str,
        agg_col: str,
        agg_funcs: List[str],
    ) -> pd.DataFrame:
        """
        Creates group-level aggregations and merges them back.

        WHY? "Average spend per user" is more predictive than raw spend.
        GroupBy captures behavioral patterns at the entity level.

        ANALOGY: Instead of knowing "John spent $50 today," you want to know
                 "John's average daily spend is $45, so today is above average."
        """
        df = df.copy()

        # Compute aggregations
        agg_df = df.groupby(group_col)[agg_col].agg(agg_funcs).reset_index()

        # Rename columns
        agg_df.columns = [group_col] + [f"{agg_col}_{func}" for func in agg_funcs]

        # Merge back to original DataFrame
        df = df.merge(agg_df, on=group_col, how="left")

        self.feature_log.append({
            "action": "groupby_aggregation",
            "group_column": group_col,
            "aggregated_column": agg_col,
            "functions": agg_funcs,
        })

        return df

    def bin_numeric(self, df: pd.DataFrame, col: str, bins: int = 5, labels: List[str] = None) -> pd.DataFrame:
        """
        Converts continuous numeric data into categorical bins.

        WHY? Sometimes the RANGE matters more than the exact value.
        "Income: high" is more robust than "Income: $87,342.50" (which might be noisy).
        """
        df = df.copy()

        if labels is None:
            labels = [f"bin_{i}" for i in range(bins)]

        df[f"{col}_binned"] = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)

        self.feature_log.append({
            "action": "numeric_binning",
            "source_column": col,
            "new_column": f"{col}_binned",
            "bins": bins,
        })

        return df
