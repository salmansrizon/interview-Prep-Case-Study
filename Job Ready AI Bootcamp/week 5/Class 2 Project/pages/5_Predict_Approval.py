import streamlit as st
import pandas as pd
import numpy as np
from utils.model_utils import predict_loan
from utils.visualizations import plot_explanation_bars
from utils.data_loader import load_loan_data

st.set_page_config(page_title="Predict Approval", page_icon="🔮", layout="wide")

st.title("🔮 Loan Approval Predictor")
st.markdown("Submit a loan application and get an instant decision with full explanation.")

st.markdown("""
<div style="background-color: #E3F2FD; padding: 20px; border-radius: 10px; border-left: 5px solid #1565C0;">
<h3 style="margin-top: 0; color: #1565C0;">How It Works</h3>
<p>
Our trained models analyze <strong>13 features</strong> from your application and predict approval probability. 
The explanation breaks down exactly which factors helped or hurt your application — no black boxes!
</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# MODEL SELECTION
# ============================================================
st.markdown("---")
model_choice = st.selectbox(
    "Choose Prediction Model",
    ['random_forest', 'logistic_regression', 'decision_tree'],
    format_func=lambda x: x.replace('_', ' ').title()
)

st.info(f"📊 Using: **{model_choice.replace('_', ' ').title()}** — " + 
        ("Ensemble of 100 trees" if model_choice == 'random_forest' else
         "Probability-based classifier" if model_choice == 'logistic_regression' else
         "Rule-based decision tree"))

# ============================================================
# INPUT FORM
# ============================================================
st.markdown("---")
st.subheader("📝 Loan Application Form")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**👤 Personal Information**")
    gender = st.selectbox("Gender", ['Male', 'Female'])
    married = st.selectbox("Marital Status", ['Yes', 'No'])
    dependents = st.selectbox("Dependents", ['0', '1', '2', '3+'])
    age = st.slider("Age", 21, 70, 35, 1)
    education = st.selectbox("Education", ['Graduate', 'Not Graduate'])
    self_employed = st.selectbox("Self Employed", ['No', 'Yes'])

with col2:
    st.markdown("**💰 Financial Details**")
    applicant_income = st.number_input("Annual Income ($)", min_value=10000, max_value=500000, value=50000, step=1000)
    coapplicant_income = st.number_input("Co-applicant Income ($)", min_value=0, max_value=500000, value=0, step=1000)
    loan_amount = st.number_input("Loan Amount Requested ($)", min_value=5000, max_value=500000, value=100000, step=5000)
    loan_term_months = st.selectbox("Loan Term (months)", [12, 24, 36, 60, 84, 120, 180, 240, 300, 360], index=7)
    credit_history = st.selectbox("Credit History", [1, 0], format_func=lambda x: "Yes (has history)" if x == 1 else "No (no history)")

with col3:
    st.markdown("**🏠 Property & Credit**")
    property_area = st.selectbox("Property Area", ['Urban', 'Semiurban', 'Rural'])
    credit_score = st.slider("Credit Score", 300, 850, 680, 1)

    st.markdown("---")
    st.markdown("**📊 Quick Stats**")
    monthly_income = (applicant_income + coapplicant_income) / 12
    monthly_payment = loan_amount / loan_term_months
    dti = (monthly_payment / monthly_income * 100) if monthly_income > 0 else 999

    st.metric("Monthly Income", f"${monthly_income:,.0f}")
    st.metric("Monthly Payment", f"${monthly_payment:,.0f}")
    st.metric("Debt-to-Income", f"{dti:.1f}%")

# Build feature dictionary
features = {
    'gender': gender,
    'married': married,
    'dependents': dependents,
    'education': education,
    'self_employed': self_employed,
    'applicant_income': applicant_income,
    'coapplicant_income': coapplicant_income,
    'loan_amount': loan_amount,
    'loan_term_months': loan_term_months,
    'credit_history': credit_history,
    'property_area': property_area,
    'credit_score': credit_score,
    'age': age
}

# ============================================================
# PREDICTION
# ============================================================
st.markdown("---")

if st.button("🔮 Predict Loan Approval", type="primary", use_container_width=True):
    result = predict_loan(features, model_name=model_choice)

    # Display result
    if result['approved']:
        st.markdown(f"""
        <div class="approved-box">
            <h1 style="margin: 0; font-size: 3.5rem;">✅ APPROVED</h1>
            <p style="font-size: 1.5rem; margin: 10px 0 0 0;">Approval Probability: <strong>{result['approval_probability']:.1%}</strong></p>
            <p style="font-size: 1rem; opacity: 0.9;">Confidence: {result['confidence']:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="denied-box">
            <h1 style="margin: 0; font-size: 3.5rem;">❌ DENIED</h1>
            <p style="font-size: 1.5rem; margin: 10px 0 0 0;">Approval Probability: <strong>{result['approval_probability']:.1%}</strong></p>
            <p style="font-size: 1rem; opacity: 0.9;">Confidence: {result['confidence']:.1%}</p>
        </div>
        """, unsafe_allow_html=True)

    # Explanation
    st.markdown("---")
    st.subheader("📋 Why This Decision Was Made")

    explanations = result['explanation']

    # Categorize
    positive_factors = [e for e in explanations if e[2] in ['strong_positive', 'positive']]
    negative_factors = [e for e in explanations if e[2] in ['strong_negative', 'negative']]
    neutral_factors = [e for e in explanations if e[2] == 'neutral']

    col_pos, col_neg = st.columns(2)

    with col_pos:
        st.markdown("**🟢 Factors Favoring Approval:**")
        for _, desc, strength in positive_factors:
            emoji = "⭐" if strength == 'strong_positive' else "✓"
            st.markdown(f"{emoji} {desc}")

    with col_neg:
        st.markdown("**🔴 Factors Against Approval:**")
        for _, desc, strength in negative_factors:
            emoji = "⚠️" if strength == 'strong_negative' else "•"
            st.markdown(f"{emoji} {desc}")

    if neutral_factors:
        st.markdown("**⚪ Neutral Factors:**")
        for _, desc, _ in neutral_factors:
            st.markdown(f"• {desc}")

    # Visual explanation
    st.markdown("---")
    st.subheader("📊 Visual Explanation")

    fig = plot_explanation_bars(explanations)
    st.pyplot(fig)

    # Recommendations
    st.markdown("---")
    st.subheader("💡 Recommendations")

    if not result['approved']:
        recs = []
        if credit_score < 650:
            recs.append("📈 **Improve Credit Score:** Pay down existing debt. Target 700+ for better rates.")
        if credit_history == 0:
            recs.append("📋 **Build Credit History:** Open a secured credit card and make on-time payments for 12+ months.")
        if dti > 43:
            recs.append("💰 **Reduce Debt-to-Income:** Pay off existing loans or increase income before reapplying.")
        if loan_amount > applicant_income * 3:
            recs.append("🏠 **Reduce Loan Amount:** Request a smaller loan or increase down payment.")
        if self_employed == 'Yes' and applicant_income < 60000:
            recs.append("📊 **Document Income:** Provide 2+ years of tax returns to verify stable self-employment income.")

        if recs:
            for rec in recs:
                st.markdown(rec)
        else:
            st.markdown("📝 Your application is close to approval. Consider adding a co-signer or increasing your down payment.")
    else:
        st.success("🎉 Congratulations! Your application is approved. Review the loan terms and sign the agreement.")
        if result['approval_probability'] < 0.7:
            st.info("💡 Your approval was marginal. Consider improving your credit profile for better interest rates on future loans.")

st.markdown("---")
st.caption("""
⚠️ **Disclaimer:** This prediction is for educational purposes only. Actual loan decisions depend on 
additional factors including employment verification, asset evaluation, and lender-specific policies. 
Always consult with a qualified financial advisor.
""")
