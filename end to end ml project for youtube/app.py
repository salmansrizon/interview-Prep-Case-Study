"""
app.py  —  Customer Spending Category Predictor
================================================
Loads three saved artefacts and applies the EXACT same pipeline as the notebook:
  1.  fitted_params.joblib   → quantile edges, segment means, min_date, column list
  2.  imputer.joblib         → fills NaN with training column means
  3.  final_best_model.joblib→ predicts Low / Medium / High

Run with:
  streamlit run app.py
"""

import os
import traceback

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Load saved artefacts (cached — loaded only once per session)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    base = os.path.dirname(os.path.abspath(__file__))
    paths = {
        "model":  os.path.join(base, "outputs", "final_best_model.joblib"),
        "params": os.path.join(base, "outputs", "fitted_params.joblib"),
        "imputer":os.path.join(base, "outputs", "imputer.joblib"),
    }
    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        st.error("❌ Required files not found. Run all notebook cells (Steps 1–8) first:")
        for f in missing:
            st.code(f)
        st.stop()
    try:
        model         = joblib.load(paths["model"])
        fitted_params = joblib.load(paths["params"])
        imputer       = joblib.load(paths["imputer"])
        return model, fitted_params, imputer
    except Exception as exc:
        st.error(f"❌ Failed to load assets: {exc}")
        st.code(traceback.format_exc())
        st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Streamlit UI
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Spending Predictor", page_icon="🛒", layout="centered")
st.title("🛒 Customer Spending Category Predictor")
st.write("Fill in the customer details below, then click **Predict**.")

model, fitted_params, imputer = load_assets()

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age      = st.number_input("Age",           min_value=18,  max_value=100, value=30)
        segment  = st.selectbox("Customer Segment",   ["Basic", "Silver", "Gold", "Platinum"])
        referral = st.checkbox("Referred by a friend?", value=False)
        txn_date = st.date_input("Transaction Date",  value=pd.Timestamp.now().date())

    with col2:
        amount     = st.number_input("Amount Spent (৳)", min_value=0.0, value=1000.0, step=50.0)
        gender     = st.selectbox("Gender",           ["Male", "Female"])
        marital    = st.selectbox("Marital Status",   ["Single", "Married"])
        emp_status = st.selectbox("Employment Status",["Employees", "self-employed"])
        payment    = st.selectbox("Payment Method",   ["Card", "Cash", "PayPal"])

    submitted = st.form_submit_button("🔍 Predict Spending Category")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Prediction pipeline (mirrors notebook Steps 3 → 4 → predict)
# ─────────────────────────────────────────────────────────────────────────────
if submitted:
    try:
        # ── Step 3 replica: apply the same feature engineering ────────────────
        # All thresholds come from fitted_params — never recomputed from user input

        txn_dt = pd.Timestamp(txn_date)   # convert date widget value to Timestamp

        # Feature: Spending bin using training quantile edges
        q25 = fitted_params['q25']
        q50 = fitted_params['q50']
        q75 = fitted_params['q75']
        bins   = [0, q25, q50, q75, float('inf')]
        labels = ['Low_Spender', 'Mid_Spender', 'Upper-Mid_Spender', 'High_Spender']
        # pd.cut on a single value requires a Series
        amount_series = pd.Series([float(amount)])
        spending_bin = pd.cut(amount_series, bins=bins, labels=labels,
                              include_lowest=True).iloc[0]

        # Feature: spend-per-age ratio (clipped to training max)
        spending_per_age = float(amount) / (float(age) + 1e-6)
        age_min          = fitted_params['age_min']
        amount_spent_max = fitted_params['amount_spent_max']
        max_ratio        = amount_spent_max / age_min if age_min > 0 else float('inf')
        spending_per_age = min(spending_per_age, max_ratio)

        # Feature: segment target encoding (mean spend for this segment)
        segment_means = fitted_params['segment_means']
        segment_encoded = float(segment_means.get(segment, 0.0))

        # Feature: days since earliest training date
        min_date        = pd.Timestamp(fitted_params['min_date'])
        days_since_start = (txn_dt - min_date).days

        # Feature: calendar decomposition
        txn_year      = txn_dt.year
        txn_month     = txn_dt.month
        txn_dayofweek = txn_dt.dayofweek
        is_weekend    = int(txn_dayofweek >= 5)

        # Feature: cyclical month encoding
        month_sin = float(np.sin(2 * np.pi * txn_month / 12))
        month_cos = float(np.cos(2 * np.pi * txn_month / 12))

        # ── Step 4 replica: build feature row in EXACT column order ──────────
        FEATURE_COLS = fitted_params['feature_cols']
        feature_values = {
            'Age':                    float(age),
            'Referral':               1.0 if referral else 0.0,
            'spending_per_age':       spending_per_age,
            'segment_target_encoded': segment_encoded,
            'days_since_start':       float(days_since_start),
            'transaction_year':       float(txn_year),
            'transaction_month':      float(txn_month),
            'transaction_dayofweek':  float(txn_dayofweek),
            'is_weekend':             float(is_weekend),
            'month_sin':              month_sin,
            'month_cos':              month_cos,
        }

        # Build a one-row DataFrame with columns in the exact training order
        X_input = pd.DataFrame([feature_values])[FEATURE_COLS]

        # Apply the loaded imputer to handle any NaN (uses training column means)
        X_imputed = imputer.transform(X_input)

        # ── Predict ───────────────────────────────────────────────────────────
        prediction    = model.predict(X_imputed)[0]
        probabilities = model.predict_proba(X_imputed)[0]
        class_names   = model.classes_

        # ── Display results ───────────────────────────────────────────────────
        colour = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}.get(str(prediction), "⚪")
        st.success(f"## {colour} Predicted Category: **{prediction}**")

        st.write("**Confidence per class:**")
        prob_df = (
            pd.DataFrame({"Category": class_names, "Probability": probabilities})
            .sort_values("Probability", ascending=False)
            .reset_index(drop=True)
        )
        prob_df["Probability"] = prob_df["Probability"].map("{:.1%}".format)
        st.table(prob_df)

        with st.expander("🔍 Feature values sent to the model"):
            st.dataframe(pd.DataFrame([feature_values])[FEATURE_COLS])

    except Exception as exc:
        st.error(f"❌ Prediction failed: {type(exc).__name__}: {exc}")
        with st.expander("Full traceback"):
            st.code(traceback.format_exc())