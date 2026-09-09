"""
Home / Dashboard Overview Page.
"""

import os
import json
import sys

import streamlit as st
import pandas as pd
import numpy as np

from src.config import get_config
from src.utils import logger


def render():
    """Render the home dashboard."""
    config = get_config()

    st.title("🔧 Industrial Equipment Success Predictor")
    st.markdown("Predict equipment reliability scores using deep learning.")
    st.markdown("---")

    # Check pipeline status
    col1, col2, col3, col4 = st.columns(4)

    raw_exists = os.path.exists(os.path.join(config.paths.data_raw, "equipment_data.csv"))
    proc_exists = os.path.exists(os.path.join(config.paths.data_processed, "X_train.npy"))
    model_exists = os.path.exists(os.path.join(config.paths.models_dir, "best_model.keras"))
    report_exists = os.path.exists(os.path.join(config.paths.artifacts_dir, "evaluation_report.json"))

    with col1:
        st.metric(
            label="📁 Raw Data",
            value="✅ Ready" if raw_exists else "❌ Missing",
            delta="5000 rows" if raw_exists else None,
        )
    with col2:
        st.metric(
            label="⚙️ Preprocessed",
            value="✅ Ready" if proc_exists else "❌ Missing",
        )
    with col3:
        st.metric(
            label="🤖 Model",
            value="✅ Trained" if model_exists else "❌ Missing",
        )
    with col4:
        st.metric(
            label="📊 Evaluated",
            value="✅ Done" if report_exists else "❌ Missing",
        )

    st.markdown("---")

    # Quick actions
    st.subheader("🚀 Quick Actions")

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("▶️ Run Full Pipeline", use_container_width=True):
            with st.spinner("Running pipeline... This may take a few minutes."):
                import subprocess
                result = subprocess.run(
                    [sys.executable, "run_pipeline.py"],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    st.success("Pipeline completed successfully!")
                    st.rerun()
                else:
                    st.error(f"Pipeline failed:\n```\n{result.stderr}\n```")

    with c2:
        if st.button("🔄 Regenerate Data", use_container_width=True):
            with st.spinner("Generating new synthetic data..."):
                import subprocess
                subprocess.run([sys.executable, "-m", "src.data.generator"], check=True)
                st.success("Data regenerated!")
                st.rerun()

    with c3:
        if st.button("📊 View Latest Report", use_container_width=True, disabled=not report_exists):
            st.switch_page("pages/analytics.py")

    st.markdown("---")

    # Model metrics preview
    if model_exists and report_exists:
        st.subheader("📈 Latest Model Performance")

        report_path = os.path.join(config.paths.artifacts_dir, "evaluation_report.json")
        with open(report_path, "r") as f:
            report = json.load(f)

        m1, m2, m3, m4 = st.columns(4)
        test_m = report.get("test_metrics", {})

        with m1:
            st.metric("MAE", f"{test_m.get('mae', 0):.3f}")
        with m2:
            st.metric("RMSE", f"{test_m.get('rmse', 0):.3f}")
        with m3:
            st.metric("R²", f"{test_m.get('r2', 0):.3f}")
        with m4:
            st.metric("MAPE", f"{test_m.get('mape', 0):.1f}%")

    # Data preview
    if raw_exists:
        st.markdown("---")
        st.subheader("📋 Raw Data Preview")
        df = pd.read_csv(os.path.join(config.paths.data_raw, "equipment_data.csv"))
        st.dataframe(df.head(10), use_container_width=True)

        st.caption(f"Total rows: {len(df):,} | Columns: {len(df.columns)}")
