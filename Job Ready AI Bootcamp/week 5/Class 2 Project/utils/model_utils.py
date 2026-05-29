"""
Model loading, prediction, and explanation utilities.
"""
import joblib
import json
import numpy as np
import pandas as pd
import streamlit as st

@st.cache_resource
def load_model(model_name='random_forest'):
    """Load a trained classification model."""
    return joblib.load(f"models/{model_name}.pkl")

@st.cache_resource
def load_scaler():
    """Load the feature scaler."""
    return joblib.load("models/scaler.pkl")

@st.cache_resource
def load_label_encoders():
    """Load label encoders for categorical features."""
    return joblib.load("models/label_encoders.pkl")

def load_feature_names():
    """Load feature names."""
    with open("models/feature_names.json", "r") as f:
        return json.load(f)

def load_model_results():
    """Load model evaluation results."""
    with open("models/model_results.json", "r") as f:
        return json.load(f)

def preprocess_input(features_dict, label_encoders, scaler=None, for_model='random_forest'):
    """
    Preprocess user input for model prediction.

    Args:
        features_dict: Dictionary with raw feature values
        label_encoders: Dict of fitted LabelEncoders
        scaler: StandardScaler (for Logistic Regression)
        for_model: Which model to prepare for

    Returns:
        Preprocessed feature array
    """
    feature_names = load_feature_names()

    # Create DataFrame in correct order
    X = pd.DataFrame([features_dict])[feature_names]

    # Encode categoricals
    categorical_cols = ['gender', 'married', 'dependents', 'education', 'self_employed', 'property_area']
    for col in categorical_cols:
        if col in X.columns and col in label_encoders:
            # Handle unseen categories
            val = str(X[col].iloc[0])
            if val in label_encoders[col].classes_:
                X[col] = label_encoders[col].transform([val])[0]
            else:
                X[col] = 0  # Default to first class

    X_array = X.values.astype(float)

    # Scale if needed
    if for_model == 'logistic_regression' and scaler is not None:
        X_array = scaler.transform(X_array)

    return X_array

def predict_loan(features_dict, model_name='random_forest'):
    """
    Predict loan approval with full explanation.

    Returns dict with prediction, probability, and explanation.
    """
    model = load_model(model_name)
    scaler = load_scaler()
    encoders = load_label_encoders()

    X = preprocess_input(features_dict, encoders, scaler, model_name)

    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0]

    # Feature importance explanation
    explanation = explain_prediction(features_dict, model_name, encoders, scaler)

    return {
        'approved': bool(prediction),
        'approval_probability': probability[1],
        'denial_probability': probability[0],
        'confidence': max(probability),
        'explanation': explanation,
        'model_used': model_name
    }

def explain_prediction(features_dict, model_name, encoders, scaler):
    """
    Generate human-readable explanation for the prediction.
    """
    explanations = []

    # Credit score analysis
    cs = features_dict.get('credit_score', 680)
    if cs >= 750:
        explanations.append(("credit_score", "Excellent credit score (750+)", "strong_positive"))
    elif cs >= 650:
        explanations.append(("credit_score", f"Good credit score ({cs})", "positive"))
    elif cs >= 580:
        explanations.append(("credit_score", f"Fair credit score ({cs})", "neutral"))
    else:
        explanations.append(("credit_score", f"Poor credit score ({cs})", "negative"))

    # Credit history
    ch = features_dict.get('credit_history', 1)
    if ch == 1:
        explanations.append(("credit_history", "Has established credit history", "strong_positive"))
    else:
        explanations.append(("credit_history", "No credit history on file", "strong_negative"))

    # Income analysis
    income = features_dict.get('applicant_income', 0)
    co_income = features_dict.get('coapplicant_income', 0)
    loan = features_dict.get('loan_amount', 1)
    monthly_payment = loan / features_dict.get('loan_term_months', 360)
    dti = monthly_payment / ((income + co_income) / 12) * 100 if (income + co_income) > 0 else 999

    if dti < 20:
        explanations.append(("debt_to_income", f"Low DTI ratio ({dti:.1f}%) — easily affordable", "strong_positive"))
    elif dti < 36:
        explanations.append(("debt_to_income", f"Moderate DTI ratio ({dti:.1f}%) — manageable", "positive"))
    elif dti < 43:
        explanations.append(("debt_to_income", f"High DTI ratio ({dti:.1f}%) — stretched budget", "negative"))
    else:
        explanations.append(("debt_to_income", f"Very high DTI ratio ({dti:.1f}%) — repayment risk", "strong_negative"))

    # Education
    edu = features_dict.get('education', 'Graduate')
    if edu == 'Graduate':
        explanations.append(("education", "Graduate degree — higher earning potential", "positive"))
    else:
        explanations.append(("education", "Non-graduate — may affect income stability", "neutral"))

    # Employment
    se = features_dict.get('self_employed', 'No')
    if se == 'Yes':
        explanations.append(("self_employed", "Self-employed — income variability risk", "negative"))
    else:
        explanations.append(("self_employed", "Salaried employment — stable income", "positive"))

    # Property area
    area = features_dict.get('property_area', 'Urban')
    if area == 'Urban':
        explanations.append(("property_area", "Urban property — strong resale value", "positive"))
    elif area == 'Semiurban':
        explanations.append(("property_area", "Semi-urban property — moderate value", "neutral"))
    else:
        explanations.append(("property_area", "Rural property — lower liquidity", "neutral"))

    # Age
    age = features_dict.get('age', 35)
    if 30 <= age <= 50:
        explanations.append(("age", f"Age {age} — prime earning years", "positive"))
    elif age < 30:
        explanations.append(("age", f"Age {age} — limited credit history", "neutral"))
    else:
        explanations.append(("age", f"Age {age} — nearing retirement", "neutral"))

    return explanations
