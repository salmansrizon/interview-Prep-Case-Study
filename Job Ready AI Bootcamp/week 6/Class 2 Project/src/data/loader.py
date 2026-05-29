"""
Data ingestion layer.
Supports synthetic customer and transaction data generation.
"""

import random
from typing import List, Dict, Any
from pathlib import Path

import pandas as pd
import numpy as np

from src.utils.logger import get_logger
import config

logger = get_logger(__name__)

# Product catalog for synthetic transactions
PRODUCT_CATEGORIES = {
    "Electronics": ["Laptop", "Phone", "Tablet", "Headphones", "Charger", "Camera", "Smartwatch"],
    "Groceries": ["Milk", "Bread", "Eggs", "Cheese", "Butter", "Yogurt", "Cereal"],
    "Produce": ["Apple", "Banana", "Orange", "Tomato", "Potato", "Onion", "Carrot"],
    "Beverages": ["Coffee", "Tea", "Juice", "Soda", "Water", "Beer", "Wine"],
    "Household": ["Detergent", "Soap", "Toothpaste", "Shampoo", "Towel", "Cleaner"],
    "Clothing": ["T-Shirt", "Jeans", "Jacket", "Shoes", "Socks", "Hat", "Scarf"],
    "Books": ["Novel", "Textbook", "Magazine", "Comic", "Dictionary", "Biography"],
}


class DataLoader:
    """Handles synthetic data generation for customer segmentation."""

    @staticmethod
    def generate_customer_dataset(n_customers: int = 1000) -> pd.DataFrame:
        """Generate realistic synthetic customer purchase behavior data."""
        logger.info("Generating synthetic customer dataset: %d customers", n_customers)
        np.random.seed(config.RANDOM_STATE)

        # Customer segments with distinct behaviors
        segments = {
            "Budget Shopper": {"freq": (1, 3), "spend": (20, 80), "items": (2, 5), "online": 0.3},
            "Regular Customer": {"freq": (4, 8), "spend": (50, 200), "items": (3, 8), "online": 0.5},
            "Premium Buyer": {"freq": (2, 5), "spend": (200, 800), "items": (2, 6), "online": 0.7},
            "Bulk Purchaser": {"freq": (1, 2), "spend": (300, 1000), "items": (15, 40), "online": 0.2},
            "Loyal Member": {"freq": (6, 12), "spend": (100, 400), "items": (5, 12), "online": 0.6},
        }

        records = []
        segment_names = list(segments.keys())
        segment_weights = [0.25, 0.30, 0.15, 0.10, 0.20]

        for i in range(n_customers):
            segment = np.random.choice(segment_names, p=segment_weights)
            seg = segments[segment]

            freq = np.random.randint(*seg["freq"])
            avg_spend = np.random.uniform(*seg["spend"])
            items = np.random.randint(*seg["items"])
            total_spend = freq * avg_spend
            online_ratio = np.random.beta(2, 2) * 0.4 + seg["online"] * 0.6
            days_since = np.random.exponential(30 / freq)
            satisfaction = np.clip(np.random.normal(3.5 + (seg["online"] * 1.5), 0.8), 1, 5)

            records.append({
                "customer_id": f"CUST_{i+1:05d}",
                "segment_ground_truth": segment,
                "purchase_frequency": freq,
                "avg_order_value": round(avg_spend, 2),
                "total_items_purchased": items,
                "total_spend": round(total_spend, 2),
                "online_purchase_ratio": round(online_ratio, 2),
                "days_since_last_purchase": round(days_since, 1),
                "satisfaction_score": round(satisfaction, 1),
                "returns_count": np.random.poisson(max(0, 5 - satisfaction)),
                "discount_usage": round(np.random.beta(2, 3) * 100, 1),
            })

        df = pd.DataFrame(records)
        logger.info("Generated %d customer records", len(df))
        return df

    @staticmethod
    def generate_transaction_data(n_transactions: int = 1000, n_products: int = 50) -> List[List[str]]:
        """Generate synthetic market basket transactions with realistic associations."""
        logger.info("Generating %d transactions with %d products", n_transactions, n_products)
        np.random.seed(config.RANDOM_STATE)
        random.seed(config.RANDOM_STATE)

        # Flatten all products
        all_products = []
        for cat, items in PRODUCT_CATEGORIES.items():
            all_products.extend(items)
        all_products = all_products[:n_products]

        # Association rules (items that frequently appear together)
        associations = [
            (["Laptop", "Charger"], 0.7),
            (["Phone", "Charger", "Headphones"], 0.5),
            (["Milk", "Bread", "Butter"], 0.6),
            (["Coffee", "Milk", "Sugar"], 0.4),
            (["Tomato", "Onion", "Potato"], 0.5),
            (["Shampoo", "Soap", "Towel"], 0.4),
            (["Beer", "Chips"], 0.3),
            (["T-Shirt", "Jeans", "Socks"], 0.35),
        ]

        transactions = []
        for _ in range(n_transactions):
            basket = set()

            # Add associated items with probability
            for assoc, prob in associations:
                if random.random() < prob:
                    basket.update(assoc)

            # Add random filler items
            n_random = np.random.poisson(3)
            filler = random.sample(all_products, min(n_random, len(all_products)))
            basket.update(filler)

            transactions.append(list(basket))

        logger.info("Generated %d transactions", len(transactions))
        return transactions

    @staticmethod
    def save_processed(df: pd.DataFrame, name: str) -> Path:
        path = config.PROCESSED_DIR / f"{name}.csv"
        df.to_csv(path, index=False)
        logger.info("Saved processed data to %s", path)
        return path
