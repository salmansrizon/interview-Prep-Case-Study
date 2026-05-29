"""
Data ingestion layer.
Supports:
  1. Loading user-provided CSVs (100% offline)
  2. Generating realistic synthetic datasets for demo purposes.
"""

import random
from pathlib import Path
from typing import Tuple, Optional

import pandas as pd
import numpy as np

from src.utils.logger import get_logger
import config

logger = get_logger(__name__)


# ── Synthetic Data Templates ────────────────────────
SPAM_TEMPLATES = [
    "Congratulations! You've won a ${amount} gift card. Call now!",
    "URGENT: You have won a free prize. Text MONEY to {number}",
    "Buy cheap viagra pills now!!! Click here {url}",
    "You are selected for a cash reward of ${amount}. Claim now.",
    "Free entry to win a car! Call {number} immediately.",
    "Dear customer, your account has been compromised. Reset password at {url}",
    "Act now! Limited time offer. Buy one get one free.",
    "You have a refund of ${amount} pending. Confirm your bank details.",
]

HAM_TEMPLATES = [
    "Hey, are we still on for lunch tomorrow?",
    "Can you send me the notes from today's class?",
    "The meeting is rescheduled to 3 PM in conference room B.",
    "Happy birthday! Hope you have a great day.",
    "Please find the attached report for your review.",
    "Thanks for the help with the project yesterday.",
    "Don't forget to pick up milk on your way home.",
    "The package has been delivered. Tracking ID: {number}",
]

INTENT_TEMPLATES = {
    "purchase": [
        "I want to buy the blue shirt in size M",
        "Can I place an order for 2 units?",
        "How do I complete my checkout?",
        "Do you have this item in stock? I need to buy it today.",
        "What payment methods do you accept for buying this?",
    ],
    "support": [
        "My app keeps crashing after the update",
        "I can't log into my account, please help",
        "The device stopped working this morning",
        "How do I reset my password?",
        "I need technical assistance with my recent order",
    ],
    "inquiry": [
        "What are your store hours this weekend?",
        "Do you ship internationally to Canada?",
        "Is there a warranty on this product?",
        "Can you tell me the difference between model A and B?",
        "When will the new collection be available?",
    ],
    "complaint": [
        "I received a damaged item and I want a refund",
        "Your delivery was late by 3 days. This is unacceptable.",
        "The product description was misleading",
        "I am very disappointed with the customer service",
        "This is the worst experience I've ever had with your brand",
    ],
    "feedback": [
        "Great service, will definitely recommend to friends",
        "The website is very intuitive and easy to use",
        "Loved the packaging, very eco-friendly",
        "Quick delivery and excellent product quality",
        "Thank you for the handwritten note in my package",
    ],
    "other": [
        "Just saying hello to the team",
        "Random thought: your logo is really cool",
        "Hope everyone is having a good day",
        "Is this the right channel for general chat?",
        "Testing if this form works",
    ],
}


class DataLoader:
    """
    Handles both file-based and synthetic data generation.
    """

    @staticmethod
    def load_csv(path: Path, text_col: str = "text", label_col: str = "label") -> pd.DataFrame:
        """Load a user-provided CSV. Must contain text and label columns."""
        logger.info("Loading dataset from %s", path)
        df = pd.read_csv(path)
        df = df[[text_col, label_col]].copy()
        df.columns = ["text", "label"]
        df.dropna(inplace=True)
        logger.info("Loaded %d records", len(df))
        return df

    @staticmethod
    def generate_spam_dataset(n_samples: int = 1000) -> pd.DataFrame:
        """
        Generate a balanced synthetic spam/ham dataset.
        Useful for offline demos when no CSV is available.
        """
        logger.info("Generating synthetic spam dataset with %d samples", n_samples)
        records = []
        half = n_samples // 2

        for _ in range(half):
            template = random.choice(SPAM_TEMPLATES)
            text = template.format(
                amount=random.randint(100, 9999),
                number=random.randint(100000, 999999),
                url=f"http://spam{random.randint(1,99)}.com",
            )
            records.append({"text": text, "label": "spam"})

        for _ in range(n_samples - half):
            template = random.choice(HAM_TEMPLATES)
            text = template.format(
                number=random.randint(100000, 999999),
            )
            records.append({"text": text, "label": "ham"})

        df = pd.DataFrame(records).sample(frac=1, random_state=config.RANDOM_STATE).reset_index(drop=True)
        return df

    @staticmethod
    def generate_intent_dataset(n_per_class: int = 200) -> pd.DataFrame:
        """
        Generate synthetic multi-class intent data.
        """
        logger.info("Generating synthetic intent dataset")
        records = []

        for intent, templates in INTENT_TEMPLATES.items():
            for _ in range(n_per_class):
                template = random.choice(templates)
                # Add slight noise/variation
                text = template + f" (ref: {random.randint(1000,9999)})"
                records.append({"text": text, "label": intent})

        df = pd.DataFrame(records).sample(frac=1, random_state=config.RANDOM_STATE).reset_index(drop=True)
        return df

    @staticmethod
    def save_processed(df: pd.DataFrame, name: str) -> Path:
        path = config.PROCESSED_DIR / f"{name}.csv"
        df.to_csv(path, index=False)
        logger.info("Saved processed data to %s", path)
        return path
