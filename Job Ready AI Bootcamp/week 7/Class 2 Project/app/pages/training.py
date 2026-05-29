"""
Training Dashboard — Monitor & Retrain Models.
"""

import os
import json
import subprocess

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import get_config
from src.models.builder import build_model, get_callbacks
from src.models.trainer import ModelTrainer
from src.data.loader import get_datasets


def render():
    """Render the training dashboard."""
    config = get_config()

    st.title("🧠 Training Dashboard")
    st.markdown("Monitor training progress, view history, and retrain models.")
    st.markdown("---")

    # Check prerequisites
    proc_dir = config.paths.data_processed
    has_data = os.path.exists(os.path.join(proc_dir, "X_train.npy"))

    if not has_data:
        st.warning("⚠️ Preprocessed data not found. Please run preprocessing first.")
        if st.button("▶️ Run Preprocessing", use_container_width=True):
            with st.spinner("Preprocessing..."):
                raw_path = os.path.join(config.paths.data_raw, "equipment_data.csv")
                if os.path.exists(raw_path):
                    from src.data.preprocessor import preprocess_pipeline
                    preprocess_pipeline(raw_path, proc_dir)
                    st.success("Preprocessing complete!")
                    st.rerun()
                else:
                    st.error("Raw data not found. Generate data first.")
        return

    # ── Section 1: Training History ──
    st.subheader("📉 Training History")

    history_path = os.path.join(config.paths.models_dir, "training_history.json")

    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            history = json.load(f)

        epochs = list(range(1, len(history["loss"]) + 1))

        # Loss + MAE curves
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Training Loss", "Validation Loss", "Training MAE", "Validation MAE"),
            vertical_spacing=0.12,
        )

        fig.add_trace(go.Scatter(x=epochs, y=history["loss"], mode="lines", line=dict(color="steelblue", width=2), name="Train Loss"), row=1, col=1)
        fig.add_trace(go.Scatter(x=epochs, y=history["val_loss"], mode="lines", line=dict(color="coral", width=2), name="Val Loss"), row=1, col=2)

        if "mae" in history:
            fig.add_trace(go.Scatter(x=epochs, y=history["mae"], mode="lines", line=dict(color="steelblue", width=2), name="Train MAE"), row=2, col=1)
        if "val_mae" in history:
            fig.add_trace(go.Scatter(x=epochs, y=history["val_mae"], mode="lines", line=dict(color="coral", width=2), name="Val MAE"), row=2, col=2)

        fig.update_layout(height=600, showlegend=False, title_text="Complete Training History")
        st.plotly_chart(fig, use_container_width=True)

        # Stats table
        st.subheader("📋 Epoch Statistics")

        hist_df = pd.DataFrame(history)
        hist_df.insert(0, "epoch", epochs)
        hist_df = hist_df.round(6)

        st.dataframe(hist_df, use_container_width=True, height=400)

        # Download history
        csv = hist_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download History CSV", data=csv, file_name="training_history.csv", mime="text/csv")
    else:
        st.info("No training history found. Train a model to see results here.")

    st.markdown("---")

    # ── Section 2: Retrain Model ──
    st.subheader("🔄 Retrain Model")

    with st.form("retrain_form"):
        st.markdown("Configure training hyperparameters:")

        col1, col2, col3 = st.columns(3)
        with col1:
            epochs_override = st.number_input("Epochs", min_value=1, max_value=500, value=config.model.epochs, step=10)
        with col2:
            lr_override = st.number_input("Learning Rate", min_value=1e-6, max_value=1e-1, value=config.model.learning_rate, format="%.5f", step=1e-4)
        with col3:
            batch_override = st.number_input("Batch Size", min_value=8, max_value=256, value=config.model.batch_size, step=8)

        col4, col5 = st.columns(2)
        with col4:
            dropout_override = st.slider("Dropout Rate", 0.0, 0.8, config.model.dropout_rate, 0.05)
        with col5:
            hidden_override = st.text_input("Hidden Units (comma-separated)", value=",".join(map(str, config.model.hidden_units)))

        submitted = st.form_submit_button("🚀 Start Training", use_container_width=True)

    if submitted:
        hidden_units = [int(x.strip()) for x in hidden_override.split(",") if x.strip()]

        with st.spinner("Training in progress... This may take several minutes."):
            try:
                # Override config temporarily
                from src.models.builder import build_model
                from src.data.loader import get_input_dim

                input_dim = get_input_dim()

                model = build_model(
                    input_dim=input_dim,
                    hidden_units=hidden_units,
                    dropout_rate=dropout_override,
                    learning_rate=lr_override,
                )

                train_ds, val_ds, test_ds, X_test, y_test = get_datasets(batch_size=batch_override)

                callbacks = get_callbacks()

                progress_bar = st.progress(0)
                status_text = st.empty()

                class StreamlitCallback(go.Figure):
                    pass  # Placeholder for custom callback if needed

                history = model.fit(
                    train_ds,
                    validation_data=val_ds,
                    epochs=epochs_override,
                    callbacks=callbacks,
                    verbose=0,
                )

                # Save
                model.save(os.path.join(config.paths.models_dir, "best_model.keras"))

                # Save history
                history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
                with open(history_path, "w") as f:
                    json.dump(history_dict, f, indent=2)

                st.success(f"✅ Training complete! {len(history.history['loss'])} epochs. Best val_loss: {min(history.history['val_loss']):.4f}")
                st.rerun()

            except Exception as e:
                st.error(f"❌ Training failed: {e}")
                raise

    st.markdown("---")

    # ── Section 3: Model Comparison ──
    st.subheader("📊 Model Comparison")

    # List all models in models_dir
    models_dir = config.paths.models_dir
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.endswith(".keras") or f.endswith(".h5")]

        if model_files:
            model_data = []
            for mf in model_files:
                mpath = os.path.join(models_dir, mf)
                mtime = os.path.getmtime(mpath)
                size = os.path.getsize(mpath)
                model_data.append({
                    "Model": mf,
                    "Modified": pd.to_datetime(mtime, unit="s").strftime("%Y-%m-%d %H:%M"),
                    "Size (MB)": round(size / (1024 * 1024), 2),
                })

            st.dataframe(pd.DataFrame(model_data), use_container_width=True)
        else:
            st.info("No saved models found.")

    # ── Section 4: TensorBoard Link ──
    st.markdown("---")
    st.subheader("📈 TensorBoard Logs")

    tb_dir = os.path.join(config.paths.logs_dir, "tensorboard")
    if os.path.exists(tb_dir):
        st.info(f"TensorBoard logs available at: `{tb_dir}`")
        st.code(f"tensorboard --logdir={tb_dir}", language="bash")
    else:
        st.info("No TensorBoard logs found. They are created during training.")
