"""
Visualization utilities.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid')

def plot_feature_importance(coef_df):
    """Plot feature importance (coefficients)."""
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['green' if c > 0 else 'red' for c in coef_df['coefficient']]
    ax.barh(coef_df['feature'], coef_df['coefficient'], color=colors, edgecolor='black')
    ax.set_xlabel('Coefficient Value')
    ax.set_title('Feature Importance: Linear Regression Coefficients', fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=0.8)
    plt.tight_layout()
    return fig

def plot_actual_vs_predicted(y_true, y_pred):
    """Plot actual vs predicted values."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_pred, alpha=0.5, edgecolors='black', linewidth=0.5)

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

    ax.set_xlabel('Actual Price ($)')
    ax.set_ylabel('Predicted Price ($)')
    ax.set_title('Actual vs Predicted House Prices', fontweight='bold')
    ax.legend()
    return fig

def plot_residuals(y_true, y_pred):
    """Plot residual distribution."""
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.5, edgecolors='black', linewidth=0.5)
    axes[0].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Predicted Price ($)')
    axes[0].set_ylabel('Residuals ($)')
    axes[0].set_title('Residuals vs Predicted', fontweight='bold')

    # Residual histogram
    axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Residual ($)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Residual Distribution', fontweight='bold')

    plt.tight_layout()
    return fig

def plot_feature_distribution(df, feature, color='steelblue'):
    """Plot distribution of a single feature."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(df[feature], bins=40, edgecolor='black', alpha=0.7, color=color)
    axes[0].axvline(df[feature].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean={df[feature].mean():.1f}')
    axes[0].axvline(df[feature].median(), color='green', linestyle='--', linewidth=2, label=f'Median={df[feature].median():.1f}')
    axes[0].set_xlabel(feature.replace('_', ' ').title())
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'Distribution of {feature.replace("_", " ").title()}', fontweight='bold')
    axes[0].legend()

    axes[1].boxplot(df[feature], patch_artist=True,
                    boxprops=dict(facecolor=color, alpha=0.7))
    axes[1].set_ylabel(feature.replace('_', ' ').title())
    axes[1].set_title(f'Box Plot: {feature.replace("_", " ").title()}', fontweight='bold')

    plt.tight_layout()
    return fig

def plot_prediction_breakdown(contributions):
    """Plot waterfall chart of prediction breakdown."""
    features = list(contributions.keys())
    values = list(contributions.values())

    fig = go.Figure(go.Waterfall(
        name="Prediction Breakdown",
        orientation="v",
        measure=["relative"] * len(features),
        x=features,
        y=values,
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": "#ff7f0e"}},
        increasing={"marker": {"color": "#2ca02c"}},
    ))

    fig.update_layout(
        title="How Each Feature Contributes to the Prediction",
        showlegend=False,
        yaxis_title="Price Contribution ($)",
        xaxis_title="Feature"
    )
    return fig
