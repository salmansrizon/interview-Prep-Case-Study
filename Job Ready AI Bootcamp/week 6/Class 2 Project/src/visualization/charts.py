"""
Visualization utilities for clustering and association rules.
"""

from typing import List, Dict, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ClusterVisualizer:
    """Static matplotlib charts for cluster analysis."""

    @staticmethod
    def plot_cluster_heatmap(features: pd.DataFrame, labels: np.ndarray, title: str = "Cluster Profiles"):
        """Plot heatmap of cluster centroids."""
        df = features.copy()
        df["cluster"] = labels
        profiles = df.groupby("cluster").mean()

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(profiles.T, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax)
        ax.set_title(title)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_pairwise(features: pd.DataFrame, labels: np.ndarray, cols: List[str] = None):
        """Pairplot of selected features colored by cluster."""
        df = features.copy()
        df["cluster"] = labels.astype(str)
        plot_cols = cols or features.columns[:4].tolist()
        g = sns.pairplot(df, hue="cluster", vars=plot_cols, palette="Set2", diag_kind="kde")
        g.fig.suptitle("Pairwise Feature Relationships by Cluster", y=1.02)
        return g


class MBAVisualizer:
    """Static matplotlib charts for market basket analysis."""

    @staticmethod
    def plot_rules_scatter(rules: pd.DataFrame, title: str = "Association Rules"):
        """Scatter plot of support vs confidence, sized by lift."""
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(
            rules["support"],
            rules["confidence"],
            s=rules["lift"] * 50,
            c=rules["lift"],
            cmap="viridis",
            alpha=0.7,
            edgecolors="black",
            linewidth=0.5,
        )
        ax.set_xlabel("Support")
        ax.set_ylabel("Confidence")
        ax.set_title(title)
        plt.colorbar(scatter, label="Lift")
        plt.tight_layout()
        return fig
