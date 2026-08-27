import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.model_utils import MODELS_DIR
from utils.data_loader import load_loan_data
from utils.visualizations import plot_feature_importance

st.set_page_config(page_title="Feature Importance", page_icon="📊", layout="wide")

st.title("📊 Feature Importance")
st.markdown("Understanding which factors most influence loan approval decisions.")

st.markdown("""
<div style="background-color: #E3F2FD; padding: 20px; border-radius: 10px; border-left: 5px solid #1565C0;">
<h3 style="margin-top: 0; color: #1565C0;">Why Feature Importance Matters</h3>
<p>
In regulated industries like banking, lenders must explain <strong>why</strong> a loan was denied. 
Feature importance tells us which variables the model relies on most, enabling:
</p>
<ul>
<li>✅ Transparent, explainable decisions</li>
<li>✅ Regulatory compliance (Fair Lending laws)</li>
<li>✅ Identifying biased or discriminatory patterns</li>
<li>✅ Focusing data collection on what matters</li>
</ul>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🌲 Random Forest Importance", "📐 Logistic Regression Coefficients", "📋 Comparison"])

with tab1:
    st.subheader("Random Forest: Gini Importance")
    st.markdown("""
    Random Forest calculates importance by measuring how much each feature decreases impurity 
    (Gini index) across all trees. Features that create the "cleanest" splits are most important.
    """)

    rf_imp = pd.read_csv(MODELS_DIR / "feature_importance.csv")

    fig = plot_feature_importance(rf_imp, title="Random Forest Feature Importance", top_n=13)
    st.pyplot(fig)

    st.dataframe(rf_imp, use_container_width=True)

    st.markdown("""
    **Key Insights:**
    - **Credit Score** and **Credit History** dominate — as expected in lending
    - **Applicant Income** and **Loan Amount** are strong secondary factors
    - **Gender** and **Property Area** have minimal impact (good for fairness!)
    """)

with tab2:
    st.subheader("Logistic Regression: Coefficient Magnitudes")
    st.markdown("""
    In Logistic Regression, the **coefficient** of each feature represents the log-odds change 
    in approval probability for a one-unit increase in that feature. Larger absolute values = more influence.
    """)

    lr_coef = pd.read_csv(MODELS_DIR / "logistic_coefficients.csv")

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['green' if c > 0 else 'red' for c in lr_coef['coefficient']]
    ax.barh(lr_coef['feature'], lr_coef['coefficient'], color=colors, edgecolor='black')
    ax.axvline(x=0, color='black', linewidth=1)
    ax.set_xlabel('Coefficient Value (Log-Odds)')
    ax.set_title('Logistic Regression Coefficients', fontweight='bold')
    ax.invert_yaxis()

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='Increases Approval Probability'),
        Patch(facecolor='red', label='Decreases Approval Probability')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    plt.tight_layout()
    st.pyplot(fig)

    st.dataframe(lr_coef, use_container_width=True)

    st.markdown("""
    **Interpretation:**
    - **Positive coefficients** (green) → Higher value = More likely to approve
    - **Negative coefficients** (red) → Higher value = More likely to deny
    - **Credit History (+)** and **Credit Score (+)** strongly favor approval
    - **Loan Amount (-)** and **Self-Employed (-)** work against approval
    """)

with tab3:
    st.subheader("Side-by-Side Comparison")

    # Merge both importance measures
    merged = rf_imp.merge(lr_coef[['feature', 'abs_coefficient']], on='feature')
    merged.columns = ['Feature', 'RF Importance', 'LR |Coefficient|']

    # Normalize to 0-1 for comparison
    merged['RF Normalized'] = merged['RF Importance'] / merged['RF Importance'].max()
    merged['LR Normalized'] = merged['LR |Coefficient|'] / merged['LR |Coefficient|'].max()

    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(merged))
    width = 0.35

    ax.barh(x - width/2, merged['RF Normalized'], width, label='Random Forest', color='steelblue', edgecolor='black')
    ax.barh(x + width/2, merged['LR Normalized'], width, label='Logistic Regression', color='coral', edgecolor='black')

    ax.set_yticks(x)
    ax.set_yticklabels(merged['Feature'])
    ax.set_xlabel('Normalized Importance (0-1)')
    ax.set_title('Feature Importance: Random Forest vs Logistic Regression', fontweight='bold')
    ax.legend()
    ax.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig)

    st.dataframe(merged[['Feature', 'RF Importance', 'LR |Coefficient|']].sort_values('RF Importance', ascending=False), 
                 use_container_width=True)

    st.markdown("""
    **Agreement:** Both models agree that **Credit Score**, **Credit History**, and **Income** 
    are the top predictors. This increases confidence in the model's fairness and reliability.
    """)

st.markdown("---")

st.subheader("⚖️ Fairness Check")
st.markdown("""
Let's verify that protected attributes (gender, marital status) do NOT disproportionately 
affect loan decisions:
""")

df = load_loan_data()

for attr in ['gender', 'married']:
    rates = df.groupby(attr)['loan_approved'].mean() * 100
    st.markdown(f"**Approval Rate by {attr.title()}:**")
    for idx, val in rates.items():
        st.markdown(f"• {idx}: {val:.1f}%")

    diff = rates.max() - rates.min()
    if diff < 5:
        st.success(f"✅ Fair: Difference is only {diff:.1f} percentage points")
    elif diff < 10:
        st.warning(f"⚠️ Moderate gap: {diff:.1f} percentage points — monitor closely")
    else:
        st.error(f"❌ Significant gap: {diff:.1f} percentage points — investigate bias")
