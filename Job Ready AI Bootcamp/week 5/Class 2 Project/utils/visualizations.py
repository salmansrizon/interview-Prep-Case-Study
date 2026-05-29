"""
Visualization utilities for classification models.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import streamlit as st

plt.style.use('seaborn-v0_8-whitegrid')

def plot_confusion_matrix(cm, labels=['Denied', 'Approved'], title="Confusion Matrix"):
    """Plot confusion matrix as heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                square=True, linewidths=1, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_roc_curve(fpr, tpr, auc_score, model_name):
    """Plot ROC curve."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'{model_name} (AUC = {auc_score:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax.fill_between(fpr, tpr, alpha=0.2, color='darkorange')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve — {model_name}', fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    return fig

def plot_feature_importance(importance_df, title="Feature Importance", top_n=10):
    """Plot feature importance horizontal bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    data = importance_df.head(top_n)
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(data)))
    ax.barh(data['feature'], data['importance'], color=colors, edgecolor='black')
    ax.set_xlabel('Importance')
    ax.set_title(title, fontweight='bold')
    ax.invert_yaxis()
    plt.tight_layout()
    return fig

def plot_model_comparison(results_dict):
    """Plot model comparison radar/bar chart."""
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    models = list(results_dict.keys())

    fig = go.Figure()
    for model in models:
        values = [results_dict[model][m] for m in metrics]
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=metrics + [metrics[0]],
            fill='toself',
            name=model.replace('_', ' ').title()
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Model Performance Comparison"
    )
    return fig

def plot_decision_boundary_viz():
    """Placeholder for decision boundary visualization."""
    pass

def plot_explanation_bars(explanations):
    """Plot explanation factors as colored bars."""
    color_map = {
        'strong_positive': '#2ca02c',
        'positive': '#98df8a',
        'neutral': '#ffbb78',
        'negative': '#ff9896',
        'strong_negative': '#d62728'
    }

    labels = [e[1] for e in explanations]
    colors = [color_map.get(e[2], '#999') for e in explanations]
    values = [3 if e[2] == 'strong_positive' else 
              2 if e[2] == 'positive' else
              1 if e[2] == 'neutral' else
              -2 if e[2] == 'negative' else -3 
              for e in explanations]

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(range(len(labels)), values, color=colors, edgecolor='black')
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(x=0, color='black', linewidth=1)
    ax.set_xlabel('Impact on Approval')
    ax.set_title('Why This Decision Was Made', fontweight='bold')
    ax.set_xlim(-4, 4)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ca02c', label='Strongly Favors Approval'),
        Patch(facecolor='#98df8a', label='Favors Approval'),
        Patch(facecolor='#ffbb78', label='Neutral'),
        Patch(facecolor='#ff9896', label='Favors Denial'),
        Patch(facecolor='#d62728', label='Strongly Favors Denial')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8)

    plt.tight_layout()
    return fig
