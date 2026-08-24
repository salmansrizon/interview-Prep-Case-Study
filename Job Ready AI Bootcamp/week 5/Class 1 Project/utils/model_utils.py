"""
Model loading and prediction utilities.
"""
import joblib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

@st.cache_resource
def load_model():
    """Load the trained linear regression model."""
    return joblib.load(MODELS_DIR / "linear_regression.pkl")

@st.cache_resource
def load_scaler():
    """Load the feature scaler."""
    return joblib.load(MODELS_DIR / "scaler.pkl")

def load_feature_names():
    """Load feature names."""
    with open(MODELS_DIR / "feature_names.json") as f:
        return json.load(f)

def load_model_results():
    """Load model evaluation results."""
    with open(MODELS_DIR / "model_results.json") as f:
        return json.load(f)

def predict_price(features_dict):
    """
    Predict house price given feature values.

    Args:
        features_dict: Dictionary with feature names as keys

    Returns:
        Predicted price (float)
    """
    model = load_model()
    feature_names = load_feature_names()

    # Create DataFrame in correct order
    X = pd.DataFrame([features_dict])[feature_names]
    prediction = model.predict(X)[0]
    return max(0, prediction)

def explain_prediction(features_dict):
    """
    Explain how each feature contributes to the prediction.

    Returns dict with feature contributions.
    """
    model = load_model()
    feature_names = load_feature_names()

    X = pd.DataFrame([features_dict])[feature_names]

    contributions = {}
    intercept = model.intercept_
    contributions['Intercept'] = intercept

    for feat, coef in zip(feature_names, model.coef_):
        contributions[feat] = coef * features_dict[feat]

    return contributions
