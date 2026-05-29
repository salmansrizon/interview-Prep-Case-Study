"""
Prediction Page — Single & Batch Prediction.
"""

import os

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.config import get_config
from src.data.preprocessor import DataPreprocessor
from src.utils import logger


def load_model_and_preprocessor():
    """Load model and preprocessor with caching."""
    config = get_config()

    model_path = os.path.join(config.paths.models_dir, "best_model.keras")
    preprocessor_path = os.path.join(config.paths.data_processed, "preprocessor.joblib")

    if not os.path.exists(model_path):
        st.error(f"❌ Model not found at `{model_path}`. Please train a model first.")
        return None, None

    if not os.path.exists(preprocessor_path):
        st.error(f"❌ Preprocessor not found at `{preprocessor_path}`. Please run preprocessing first.")
        return None, None

    try:
        model = keras.models.load_model(model_path)
        preprocessor = DataPreprocessor()
        preprocessor.load(preprocessor_path)
        return model, preprocessor
    except Exception as e:
        st.error(f"❌ Error loading model/preprocessor: {e}")
        return None, None


def predict_single(model, preprocessor, features: dict) -> float:
    """Run prediction on a single sample."""
    df = pd.DataFrame([features])
    X, _ = preprocessor.transform(df)
    pred_scaled = model.predict(X, verbose=0).flatten()[0]
    return preprocessor.inverse_transform_target(np.array([pred_scaled]))[0]


def predict_batch(model, preprocessor, df: pd.DataFrame) -> pd.DataFrame:
    """Run prediction on a batch of samples."""
    X, _ = preprocessor.transform(df)
    preds_scaled = model.predict(X, verbose=0).flatten()
    preds = preprocessor.inverse_transform_target(preds_scaled)

    result = df.copy()
    result["predicted_score"] = np.round(preds, 2)
    result["prediction_error"] = np.round(result["success_score"] - result["predicted_score"], 2) if "success_score" in result.columns else None
    return result


def render():
    """Render the prediction page."""
    config = get_config()

    st.title("🔮 Predict Success Score")
    st.markdown("Predict equipment reliability scores using the trained deep learning model.")
    st.markdown("---")

    model, preprocessor = load_model_and_preprocessor()

    if model is None:
        st.info("💡 Go to **Home** and run the pipeline to train a model.")
        return

    # Model info
    with st.expander("🤖 Loaded Model Info"):
        st.write(f"**Model path:** `{os.path.join(config.paths.models_dir, 'best_model.keras')}`")
        st.write(f"**Total parameters:** {model.count_params():,}")
        st.write(f"**Input features:** {len(preprocessor.feature_names_after_transform)}")
        st.code("\n".join(preprocessor.feature_names_after_transform))

    st.markdown("---")

    # Tabs
    tab1, tab2 = st.tabs(["📝 Single Prediction", "📁 Batch Prediction"])

    with tab1:
        st.subheader("Enter Equipment Parameters")

        col1, col2, col3 = st.columns(3)

        with col1:
            equipment_type = st.selectbox("Equipment Type", ["Pump", "Compressor", "Turbine", "Motor", "Generator", "Heat_Exchanger"])
            manufacturer = st.selectbox("Manufacturer", ["Siemens", "ABB", "GE", "Schneider", "Mitsubishi", "Honeywell"])
            facility_location = st.selectbox("Facility", ["Plant_A", "Plant_B", "Plant_C", "Plant_D"])
            operating_temperature = st.slider("Operating Temperature (°C)", 20.0, 120.0, 65.0, 0.5)

        with col2:
            vibration_level = st.slider("Vibration Level (mm/s)", 0.1, 15.0, 2.5, 0.1)
            pressure_reading = st.slider("Pressure (bar)", 2.0, 18.0, 8.5, 0.1)
            power_consumption = st.slider("Power Consumption (kW)", 10.0, 200.0, 75.0, 1.0)
            runtime_hours = st.number_input("Runtime Hours", 50, 10000, 2400, 100)

        with col3:
            days_since_maintenance = st.slider("Days Since Maintenance", 1, 180, 45, 1)
            error_count_24h = st.slider("Errors (24h)", 0, 15, 1, 1)
            oil_quality_index = st.slider("Oil Quality Index", 0.0, 100.0, 75.0, 1.0)
            load_factor = st.slider("Load Factor (%)", 30.0, 100.0, 72.0, 1.0)
            ambient_temperature = st.slider("Ambient Temp (°C)", 5.0, 45.0, 28.0, 0.5)

        features = {
            "equipment_type": equipment_type,
            "manufacturer": manufacturer,
            "facility_location": facility_location,
            "operating_temperature": operating_temperature,
            "vibration_level": vibration_level,
            "pressure_reading": pressure_reading,
            "power_consumption": power_consumption,
            "runtime_hours": runtime_hours,
            "days_since_maintenance": days_since_maintenance,
            "error_count_24h": error_count_24h,
            "oil_quality_index": oil_quality_index,
            "load_factor": load_factor,
            "ambient_temperature": ambient_temperature,
            "success_score": 0.0,  # dummy for transform
        }

        st.markdown("---")

        if st.button("🚀 Predict Success Score", type="primary", use_container_width=True):
            with st.spinner("Running prediction..."):
                score = predict_single(model, preprocessor, features)

            # Display result
            st.markdown("---")
            st.subheader("Prediction Result")

            result_col1, result_col2, result_col3 = st.columns([1, 2, 1])

            with result_col2:
                # Color based on score
                if score >= 80:
                    color = "#28a745"
                    status = "🟢 Excellent"
                elif score >= 60:
                    color = "#ffc107"
                    status = "🟡 Good"
                elif score >= 40:
                    color = "#fd7e14"
                    status = "🟠 Fair"
                else:
                    color = "#dc3545"
                    status = "🔴 Poor"

                st.markdown(f"""
                <div style="text-align: center; padding: 30px; border-radius: 15px; background: linear-gradient(135deg, {color}22, {color}11); border: 2px solid {color};">
                    <h1 style="color: {color}; margin: 0; font-size: 4rem;">{score:.1f}</h1>
                    <p style="color: {color}; font-size: 1.5rem; margin: 10px 0 0 0;">{status}</p>
                    <p style="color: #666; margin-top: 10px;">Predicted Success Score</p>
                </div>
                """, unsafe_allow_html=True)

            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Success Score"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": color},
                    "steps": [
                        {"range": [0, 40], "color": "#ffcccc"},
                        {"range": [40, 60], "color": "#ffe6cc"},
                        {"range": [60, 80], "color": "#ffffcc"},
                        {"range": [80, 100], "color": "#ccffcc"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 4},
                        "thickness": 0.75,
                        "value": score,
                    },
                },
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Batch Prediction")
        st.markdown("Upload a CSV file with equipment data. The model will predict success scores for all rows.")

        uploaded = st.file_uploader("📤 Upload CSV", type=["csv"])

        if uploaded is not None:
            try:
                batch_df = pd.read_csv(uploaded)
                st.success(f"✅ Loaded {len(batch_df):,} rows")

                # Validate columns
                required = set(config.data.feature_columns + config.data.categorical_columns)
                missing = required - set(batch_df.columns)
                if missing:
                    st.error(f"❌ Missing columns: {missing}")
                    return

                with st.spinner("Predicting..."):
                    result_df = predict_batch(model, preprocessor, batch_df)

                st.subheader("📊 Results Preview")
                st.dataframe(result_df.head(20), use_container_width=True)

                # Distribution
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=result_df["predicted_score"], name="Predicted", marker_color="steelblue"))
                if "success_score" in result_df.columns:
                    fig.add_trace(go.Histogram(x=result_df["success_score"], name="Actual", marker_color="coral", opacity=0.7))
                fig.update_layout(
                    barmode="overlay",
                    title="Score Distribution",
                    xaxis_title="Success Score",
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)

                # Download
                csv_out = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Predictions CSV",
                    data=csv_out,
                    file_name="predictions.csv",
                    mime="text/csv",
                )

            except Exception as e:
                st.error(f"❌ Error processing file: {e}")
        else:
            # Template download
            template = pd.DataFrame(columns=config.data.feature_columns + config.data.categorical_columns)
            template_csv = template.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📄 Download CSV Template",
                data=template_csv,
                file_name="prediction_template.csv",
                mime="text/csv",
            )
