"""
🏦 Loan Approval Predictor — Production-Grade Classification Dashboard
=====================================================================
A Streamlit application for predicting loan approvals using three
supervised learning algorithms: Logistic Regression, Decision Trees,
and Random Forests. Includes full explainability for every decision.

Topics Covered:
- Logistic Regression (probability-based classification)
- Decision Trees (rule-based splits)
- Random Forests (ensemble voting)
- Feature Importance & Model Explainability

Run: streamlit run app.py
"""
import streamlit as st

st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: #1565C0;
        margin-bottom: 0.3rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .approved-box {
        background: linear-gradient(135deg, #2E7D32 0%, #4CAF50 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin: 15px 0;
    }
    .denied-box {
        background: linear-gradient(135deg, #C62828 0%, #EF5350 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin: 15px 0;
    }
    .edu-box {
        background-color: #E3F2FD;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 5px solid #1565C0;
        margin: 1rem 0;
    }
    .formula-box {
        background-color: #FFF3E0;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #FF9800;
        margin: 0.8rem 0;
        font-family: 'Courier New', monospace;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🏦 Loan Approval Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Supervised Learning for Credit Risk Assessment — With Full Explainability</div>', unsafe_allow_html=True)

st.markdown("""
## Welcome to the Loan Approval System

This application demonstrates **three classification algorithms** that predict whether a loan 
application should be approved or denied — and **explains why** in plain English.

### 📚 What You'll Learn

| Topic | Description | Where to Find |
|-------|-------------|---------------|
| **Logistic Regression** | Probability-based binary classification | 📐 Algorithm Theory |
| **Decision Trees** | Rule-based splits with visual flow | 🌳 Decision Trees |
| **Random Forests** | Ensemble of trees for robust predictions | 🌲 Random Forests |
| **Feature Importance** | Which factors matter most for approval | 📊 Feature Importance |
| **Model Comparison** | How algorithms perform against each other | ⚖️ Compare Models |
| **Live Prediction** | Get instant approval with explanation | 🔮 Predict Approval |

### 🏗️ Project Architecture

```
loan_approval_predictor/
├── app.py                        ← You are here
├── data/loan_data.csv            ← 8,000 loan applications
├── models/
│   ├── logistic_regression.pkl   ← Probability model
│   ├── decision_tree.pkl         ← Rule-based model
│   ├── random_forest.pkl         ← Ensemble model
│   ├── scaler.pkl                ← Feature scaler
│   ├── label_encoders.pkl        ← Categorical encoders
│   ├── feature_importance.csv    ← Random Forest importance
│   └── model_results.json        ← Evaluation metrics
├── utils/
│   ├── data_loader.py            ← Data I/O
│   ├── model_utils.py            ← Prediction & explainability
│   └── visualizations.py         ← Chart utilities
└── pages/
    ├── 1_Explore_Data.py         ← EDA & distributions
    ├── 2_Algorithm_Theory.py     ← How each algorithm works
    ├── 3_Model_Comparison.py     ← Side-by-side evaluation
    ├── 4_Feature_Importance.py   ← What drives decisions
    └── 5_Predict_Approval.py     ← Live prediction with explanation
```

**Navigate using the sidebar →**
""")

st.divider()

st.markdown("""
### 🎯 Quick Start

1. **Explore Data** — Understand the loan application dataset
2. **Algorithm Theory** — Learn how Logistic Regression, Decision Trees, and Random Forests work
3. **Model Comparison** — See which algorithm performs best on our data
4. **Feature Importance** — Discover what factors most influence loan decisions
5. **Predict Approval** — Submit a loan application and get an explainable decision
""")

st.info("💡 **Tip:** Every prediction includes a human-readable explanation. No black boxes here!")
