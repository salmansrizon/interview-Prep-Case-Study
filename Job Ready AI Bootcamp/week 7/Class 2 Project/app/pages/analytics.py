"""
Analytics Dashboard — Model Performance & Insights.
"""

import os
import json

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import get_config


def render():
    """Render the analytics dashboard."""
    config = get_config()

    st.title("📈 Analytics Dashboard")
    st.markdown("Visualize model performance, predictions, and feature importance.")
    st.markdown("---")

    # Check for artifacts
    report_path = os.path.join(config.paths.artifacts_dir, "evaluation_report.json")

    if not os.path.exists(report_path):
        st.warning("⚠️ No evaluation report found. Please run the pipeline first.")
        return

    with open(report_path, "r") as f:
        report = json.load(f)

    # Metrics cards
    st.subheader("📊 Performance Metrics")

    test_m = report.get("test_metrics", {})
    val_m = report.get("val_metrics", {})

    c1, c2, c3, c4, c5 = st.columns(5)
    metrics = [
        ("MAE", "mae", "Mean Absolute Error"),
        ("RMSE", "rmse", "Root Mean Squared Error"),
        ("R²", "r2", "Coefficient of Determination"),
        ("MAPE", "mape", "Mean Absolute % Error"),
        ("Exp. Var", "explained_variance", "Explained Variance"),
    ]

    for col, (label, key, desc) in zip([c1, c2, c3, c4, c5], metrics):
        test_val = test_m.get(key, 0)
        val_val = val_m.get(key, 0)
        delta = test_val - val_val

        with col:
            st.metric(
                label=label,
                value=f"{test_val:.4f}" if key != "mape" else f"{test_val:.2f}%",
                delta=f"Δ val: {delta:+.4f}",
                help=desc,
            )

    st.markdown("---")

    # Plots
    tab1, tab2, tab3 = st.tabs(["🎯 Predictions", "📉 Residuals", "⭐ Feature Importance"])

    with tab1:
        st.subheader("Predicted vs Actual")
        scatter_path = os.path.join(config.paths.artifacts_dir, "predictions_scatter.png")
        if os.path.exists(scatter_path):
            st.image(scatter_path, use_container_width=True)
        else:
            st.info("Scatter plot not generated yet.")

    with tab2:
        st.subheader("Residual Analysis")
        residual_path = os.path.join(config.paths.artifacts_dir, "residuals_distribution.png")
        if os.path.exists(residual_path):
            st.image(residual_path, use_container_width=True)
        else:
            st.info("Residual plot not generated yet.")

    with tab3:
        st.subheader("Feature Importance (Permutation Method)")
        fi_path = os.path.join(config.paths.artifacts_dir, "feature_importance.png")
        if os.path.exists(fi_path):
            st.image(fi_path, use_container_width=True)
        else:
            st.info("Feature importance plot not generated yet.")

    st.markdown("---")

    # Training history
    st.subheader("📉 Training History")
    history_path = os.path.join(config.paths.models_dir, "training_history.json")

    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            history = json.load(f)

        epochs = list(range(1, len(history["loss"]) + 1))

        fig = make_subplots(rows=1, cols=2, subplot_titles=("Loss", "MAE"))

        fig.add_trace(
            go.Scatter(x=epochs, y=history["loss"], mode="lines", name="Train Loss", line=dict(color="steelblue")),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=history["val_loss"], mode="lines", name="Val Loss", line=dict(color="coral")),
            row=1, col=1,
        )

        if "mae" in history:
            fig.add_trace(
                go.Scatter(x=epochs, y=history["mae"], mode="lines", name="Train MAE", line=dict(color="steelblue")),
                row=1, col=2,
            )
        if "val_mae" in history:
            fig.add_trace(
                go.Scatter(x=epochs, y=history["val_mae"], mode="lines", name="Val MAE", line=dict(color="coral")),
                row=1, col=2,
            )

        fig.update_layout(height=400, showlegend=True, title_text="Training Curves")
        st.plotly_chart(fig, use_container_width=True)

        # Early stopping info
        best_epoch = np.argmin(history["val_loss"]) + 1
        st.info(f"🏆 Best validation loss at epoch **{best_epoch}** (val_loss = {min(history['val_loss']):.4f})")
    else:
        st.info("Training history not available.")

    st.markdown("---")

    # Metadata
    st.subheader("📝 Training Metadata")
    metadata_path = os.path.join(config.paths.models_dir, "training_metadata.json")

    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        meta_col1, meta_col2 = st.columns(2)

        with meta_col1:
            st.json({
                "epochs_trained": metadata.get("epochs_trained"),
                "model_params": metadata.get("model_params"),
                "final_train_loss": metadata.get("final_train_loss"),
                "final_val_loss": metadata.get("final_val_loss"),
            })

        with meta_col2:
            st.json({
                "test_metrics": metadata.get("test_metrics"),
                "model_path": metadata.get("model_path"),
            })
    else:
        st.info("Metadata not available.")
