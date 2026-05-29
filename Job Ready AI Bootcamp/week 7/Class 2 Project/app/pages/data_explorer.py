"""
Interactive Data Exploration Page.
"""

import os

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import get_config


def render():
    """Render the data explorer page."""
    config = get_config()

    st.title("📊 Data Explorer")
    st.markdown("Explore the equipment dataset with interactive visualizations.")
    st.markdown("---")

    raw_path = os.path.join(config.paths.data_raw, "equipment_data.csv")

    if not os.path.exists(raw_path):
        st.warning("⚠️ No data found. Please generate data first from the Home page.")
        return

    df = pd.read_csv(raw_path)

    # Filters
    st.subheader("🔍 Filters")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        selected_types = st.multiselect(
            "Equipment Type",
            options=sorted(df["equipment_type"].unique()),
            default=sorted(df["equipment_type"].unique()),
        )
    with col2:
        selected_mfr = st.multiselect(
            "Manufacturer",
            options=sorted(df["manufacturer"].unique()),
            default=sorted(df["manufacturer"].unique()),
        )
    with col3:
        score_range = st.slider(
            "Success Score Range",
            min_value=float(df["success_score"].min()),
            max_value=float(df["success_score"].max()),
            value=(0.0, 100.0),
        )
    with col4:
        n_samples = st.slider("Sample Size", 100, len(df), min(1000, len(df)), 100)

    # Apply filters
    filtered = df[
        (df["equipment_type"].isin(selected_types)) &
        (df["manufacturer"].isin(selected_mfr)) &
        (df["success_score"] >= score_range[0]) &
        (df["success_score"] <= score_range[1])
    ].sample(min(n_samples, len(df)), random_state=42)

    st.caption(f"Showing {len(filtered):,} of {len(df):,} records")
    st.markdown("---")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Distributions", "🔗 Correlations", "📋 Table", "🗺️ Map"])

    with tab1:
        st.subheader("Feature Distributions")

        numeric_cols = config.data.feature_columns
        selected_feature = st.selectbox("Select feature to visualize", numeric_cols)

        fig = make_subplots(rows=1, cols=2, subplot_titles=("Histogram", "Box Plot"))

        fig.add_trace(
            go.Histogram(x=filtered[selected_feature], nbinsx=50, name="Histogram", marker_color="steelblue"),
            row=1, col=1,
        )
        fig.add_trace(
            go.Box(y=filtered[selected_feature], name="Box Plot", marker_color="coral"),
            row=1, col=2,
        )

        fig.update_layout(height=400, showlegend=False, title_text=f"{selected_feature} Distribution")
        st.plotly_chart(fig, use_container_width=True)

        # Success score by category
        st.subheader("Success Score by Category")
        cat_col = st.selectbox("Categorical feature", config.data.categorical_columns)

        fig2 = px.box(
            filtered, x=cat_col, y="success_score",
            color=cat_col, points="all",
            title=f"Success Score Distribution by {cat_col}",
        )
        st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        st.subheader("Correlation Heatmap")

        numeric_df = filtered[numeric_cols + [config.data.target_column]]
        corr = numeric_df.corr()

        fig3 = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            title="Feature Correlation Matrix",
        )
        fig3.update_layout(height=600)
        st.plotly_chart(fig3, use_container_width=True)

        # Scatter matrix
        st.subheader("Scatter Matrix (Top Correlated Features)")
        target_corr = corr["success_score"].drop("success_score").abs().sort_values(ascending=False)
        top_features = target_corr.head(4).index.tolist()

        fig4 = px.scatter_matrix(
            filtered,
            dimensions=top_features + ["success_score"],
            color="equipment_type",
            opacity=0.5,
            title="Feature Relationships",
        )
        fig4.update_traces(diagonal_visible=False)
        fig4.update_layout(height=700)
        st.plotly_chart(fig4, use_container_width=True)

    with tab3:
        st.subheader("Data Table")
        st.dataframe(filtered, use_container_width=True, height=600)

        # Download
        csv = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Filtered CSV",
            data=csv,
            file_name="filtered_equipment_data.csv",
            mime="text/csv",
        )

    with tab4:
        st.subheader("Facility Overview")

        facility_stats = filtered.groupby("facility_location").agg({
            "success_score": ["mean", "std", "count"],
            "error_count_24h": "mean",
            "vibration_level": "mean",
        }).round(2)
        facility_stats.columns = ["_".join(col).strip() for col in facility_stats.columns]
        facility_stats = facility_stats.reset_index()

        st.dataframe(facility_stats, use_container_width=True)

        fig5 = px.bar(
            facility_stats,
            x="facility_location",
            y="success_score_mean",
            error_y="success_score_std",
            color="success_score_mean",
            color_continuous_scale="RdYlGn",
            title="Average Success Score by Facility",
        )
        st.plotly_chart(fig5, use_container_width=True)
