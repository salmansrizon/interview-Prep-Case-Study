"""
🏠 House Price Predictor — Production-Grade ML Dashboard
=============================================================
A Streamlit application for predicting real estate values using
Linear Regression with full educational walkthrough.

Topics Covered:
- Linear Regression fundamentals
- Gradient Descent (simplified)
- Mean Squared Error (MSE)
- Feature Scaling with Scikit-Learn
- Model evaluation & interpretation

Run: streamlit run app.py
"""
import streamlit as st

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: #2E7D32;
        margin-bottom: 0.3rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .edu-box {
        background-color: #E8F5E9;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 5px solid #2E7D32;
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
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🏠 House Price Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Machine Learning for Real Estate Valuation — Powered by Linear Regression</div>', unsafe_allow_html=True)

st.markdown("""
## Welcome to Your Local Real Estate Valuation Tool

This application teaches and demonstrates **Linear Regression** — the foundation of predictive modeling — 
through an interactive house price prediction system.

### 📚 What You'll Learn

| Topic | Description | Where to Find |
|-------|-------------|---------------|
| **Linear Regression** | How the model learns to predict continuous values | 📊 Model Theory page |
| **Gradient Descent** | How the model "learns" by minimizing error | 📉 Gradient Descent page |
| **MSE & Metrics** | How we measure prediction accuracy | 📏 Model Evaluation page |
| **Feature Scaling** | Why and how to normalize features | ⚖️ Feature Scaling page |
| **Live Prediction** | Predict prices for any house configuration | 🔮 Predict Price page |

### 🏗️ Project Architecture

```
houseprice_predictor/
├── app.py                    ← You are here
├── data/housing_data.csv     ← 5,000 real estate records
├── models/                   ← Trained models & artifacts
│   ├── linear_regression.pkl
│   ├── scaler.pkl
│   ├── feature_names.json
│   └── model_results.json
├── utils/
│   ├── data_loader.py        ← Data I/O
│   ├── model_utils.py        ← Prediction engine
│   └── visualizations.py     ← Chart utilities
└── pages/
    ├── 1_Explore_Data.py     ← EDA & distributions
    ├── 2_Model_Theory.py     ← Linear Regression math
    ├── 3_Gradient_Descent.py ← Interactive GD demo
    ├── 4_Model_Evaluation.py ← MSE, R², residuals
    ├── 5_Feature_Scaling.py  ← StandardScaler demo
    └── 6_Predict_Price.py    ← Live prediction tool
```

**Navigate using the sidebar →**
""")

st.divider()

st.markdown("""
### 🎯 Quick Start

1. **Explore Data** — Understand the dataset structure and feature distributions
2. **Model Theory** — Learn the math behind Linear Regression
3. **Gradient Descent** — See how the model learns step-by-step
4. **Model Evaluation** — Understand MSE, RMSE, MAE, and R²
5. **Feature Scaling** — See why scaling matters for ML
6. **Predict Price** — Use the trained model to estimate house values
""")

st.info("💡 **Tip:** All visualizations are interactive. Hover over charts for details.")
