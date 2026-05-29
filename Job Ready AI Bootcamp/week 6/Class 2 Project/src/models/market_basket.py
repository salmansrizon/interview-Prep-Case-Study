"""
Market Basket Analysis Engine
─────────────────────────────
Implements Apriori algorithm for association rule mining.
Uses mlxtend for efficient frequent itemset generation.
"""

from typing import List, Dict, Any
from pathlib import Path

import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

from src.utils.logger import get_logger
import config

logger = get_logger(__name__)


class MarketBasketAnalyzer:
    """
    Association rule mining using the Apriori algorithm.
    Discovers patterns like: {Milk, Bread} → {Butter}
    """

    def __init__(
        self,
        min_support: float = config.MBA_MIN_SUPPORT,
        min_confidence: float = config.MBA_MIN_CONFIDENCE,
        min_lift: float = config.MBA_MIN_LIFT,
        max_len: int = config.MBA_MAX_LEN,
    ) -> None:
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        self.max_len = max_len
        self.is_fitted = False
        self.frequent_itemsets: pd.DataFrame = None
        self.rules: pd.DataFrame = None

    def fit(self, transactions: List[List[str]]) -> pd.DataFrame:
        """
        Mine association rules from transaction data.
        Returns DataFrame of rules with support, confidence, lift, conviction.
        """
        logger.info(
            "Running Apriori: min_support=%.2f, min_confidence=%.2f, min_lift=%.2f",
            self.min_support, self.min_confidence, self.min_lift,
        )

        # Encode transactions to binary matrix
        te = TransactionEncoder()
        te_array = te.fit_transform(transactions)
        df = pd.DataFrame(te_array, columns=te.columns_)

        # Find frequent itemsets
        self.frequent_itemsets = apriori(
            df,
            min_support=self.min_support,
            use_colnames=True,
            max_len=self.max_len,
        )

        logger.info("Found %d frequent itemsets", len(self.frequent_itemsets))

        if self.frequent_itemsets.empty:
            self.rules = pd.DataFrame()
            return self.rules

        # Generate association rules
        self.rules = association_rules(
            self.frequent_itemsets,
            metric="confidence",
            min_threshold=self.min_confidence,
        )

        # Filter by lift
        self.rules = self.rules[self.rules["lift"] >= self.min_lift]

        # Sort by lift (most interesting rules first)
        self.rules = self.rules.sort_values("lift", ascending=False).reset_index(drop=True)

        self.is_fitted = True
        logger.info("Generated %d association rules", len(self.rules))
        return self.rules

    def get_recommendations(self, item: str, top_n: int = 5) -> pd.DataFrame:
        """Get top recommended items to pair with a given item."""
        if not self.is_fitted or self.rules.empty:
            return pd.DataFrame()

        # Find rules where item is in antecedent
        mask = self.rules["antecedents"].apply(lambda x: item in set(x))
        recs = self.rules[mask].nlargest(top_n, "lift")[["consequents", "confidence", "lift"]]
        return recs

    def get_item_stats(self) -> pd.DataFrame:
        """Get frequency statistics for all items."""
        if self.frequent_itemsets is None:
            return pd.DataFrame()

        stats = []
        for _, row in self.frequent_itemsets.iterrows():
            for item in row["itemsets"]:
                stats.append({"item": item, "support": row["support"]})

        df_stats = pd.DataFrame(stats)
        return df_stats.groupby("item")["support"].max().sort_values(ascending=False).reset_index()
