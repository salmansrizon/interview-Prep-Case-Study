"""
Sanity checks for the customer segmentation pipeline.
Run with: pytest tests/test_pipeline.py
"""

import numpy as np
import pandas as pd
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.models.kmeans_engine import KMeansEngine
from src.models.pca_engine import PCAEngine
from src.models.market_basket import MarketBasketAnalyzer


def test_data_generation():
    df = DataLoader.generate_customer_dataset(n_customers=100)
    assert len(df) == 100
    assert "customer_id" in df.columns
    assert "total_spend" in df.columns


def test_preprocessing():
    df = DataLoader.generate_customer_dataset(n_customers=50)
    prep = DataPreprocessor()
    features = prep.prepare_for_clustering(df)
    assert features.shape[0] == 50
    assert features.shape[1] > 0
    # Check scaling (mean ≈ 0, std ≈ 1)
    assert abs(features.mean().mean()) < 0.1
    assert abs(features.std().mean() - 1.0) < 0.2


def test_kmeans():
    df = DataLoader.generate_customer_dataset(n_customers=100)
    prep = DataPreprocessor()
    features = prep.prepare_for_clustering(df)

    km = KMeansEngine(n_clusters=3)
    labels, inertia, centers = km.fit(features)
    assert len(labels) == 100
    assert len(set(labels)) == 3
    assert inertia > 0
    assert km.silhouette > -1 and km.silhouette <= 1


def test_pca():
    df = DataLoader.generate_customer_dataset(n_customers=100)
    prep = DataPreprocessor()
    features = prep.prepare_for_clustering(df)

    pca = PCAEngine(n_components=2)
    pca.fit(features)
    transformed = pca.transform(features)
    assert transformed.shape == (100, 2)
    assert pca.total_variance > 0


def test_market_basket():
    transactions = [
        ["Milk", "Bread", "Butter"],
        ["Milk", "Bread"],
        ["Milk", "Butter"],
        ["Bread", "Butter"],
        ["Milk", "Bread", "Butter", "Cheese"],
        ["Beer", "Chips"],
        ["Beer", "Chips", "Nuts"],
    ]
    mba = MarketBasketAnalyzer(min_support=0.2, min_confidence=0.5, min_lift=1.0)
    rules = mba.fit(transactions)
    assert isinstance(rules, pd.DataFrame)
    if not rules.empty:
        assert "lift" in rules.columns
        assert "confidence" in rules.columns
        assert "support" in rules.columns
