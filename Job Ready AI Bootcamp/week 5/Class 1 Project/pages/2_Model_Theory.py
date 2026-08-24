import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Model Theory", page_icon="📐", layout="wide")

st.title("📐 Linear Regression: The Theory")
st.markdown("Understanding the math behind the most fundamental machine learning algorithm.")

st.markdown("""
<div style="background-color: #E8F5E9; padding: 20px; border-radius: 10px; border-left: 5px solid #2E7D32;">
<h3 style="margin-top: 0; color: #2E7D32;">What is Linear Regression?</h3>
<p>
Linear Regression models the <strong>linear relationship</strong> between input features (X) and 
a continuous target variable (y). It finds the best-fitting straight line (or hyperplane) 
that minimizes the prediction error.
</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

st.subheader("📝 The Mathematical Model")

st.markdown("""
For a single feature (Simple Linear Regression):

<div style="background-color: #FFF3E0; padding: 15px; border-radius: 8px; font-family: 'Courier New', monospace; font-size: 1.3em; text-align: center;">
    <strong>y = β₀ + β₁ · x + ε</strong>
</div>

For multiple features (Multiple Linear Regression):

<div style="background-color: #FFF3E0; padding: 15px; border-radius: 8px; font-family: 'Courier New', monospace; font-size: 1.2em; text-align: center;">
    <strong>y = β₀ + β₁·x₁ + β₂·x₂ + ... + βₙ·xₙ + ε</strong>
</div>

| Symbol | Meaning | In Our Context |
|--------|---------|----------------|
| **y** | Target variable (what we predict) | House price ($) |
| **β₀** | Intercept (bias term) | Base price when all features = 0 |
| **β₁, β₂, ... βₙ** | Coefficients (weights) | How much each feature affects price |
| **x₁, x₂, ... xₙ** | Input features | Income, rooms, age, etc. |
| **ε** | Error term (noise) | What the model cannot explain |
""", unsafe_allow_html=True)

st.markdown("---")

st.subheader("🎯 The Goal: Minimize Prediction Error")

st.markdown("""
The model learns by finding the coefficients (β) that make predictions as close as possible 
to actual values. We measure "closeness" using **Mean Squared Error (MSE)**:

<div style="background-color: #E3F2FD; padding: 15px; border-radius: 8px; font-family: 'Courier New', monospace; font-size: 1.2em; text-align: center;">
    <strong>MSE = (1/n) · Σ(yᵢ - ŷᵢ)²</strong>
</div>

Where:
- **n** = number of houses in the dataset
- **yᵢ** = actual price of house i
- **ŷᵢ** = predicted price of house i = β₀ + β₁·xᵢ₁ + β₂·xᵢ₂ + ...

**The smaller the MSE, the better the model fits the data.**
""", unsafe_allow_html=True)

st.markdown("---")

st.subheader("📊 Interactive Demo: How Coefficients Affect the Line")

st.markdown("Adjust the slope (β₁) and intercept (β₀) to see how they change the prediction line.")

col1, col2 = st.columns(2)
with col1:
    beta0 = st.slider("Intercept (β₀)", -100, 100, 20)
    beta1 = st.slider("Slope (β₁)", -5.0, 5.0, 2.5, 0.1)

# Generate simple data
np.random.seed(42)
x_demo = np.linspace(0, 50, 100)
y_true = 20 + 2.5 * x_demo + np.random.normal(0, 10, 100)
y_pred = beta0 + beta1 * x_demo

# Calculate MSE
mse_demo = np.mean((y_true - y_pred) ** 2)

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(x_demo, y_true, alpha=0.5, label='Actual Data', color='steelblue', edgecolors='black', linewidth=0.5)
ax.plot(x_demo, y_pred, color='red', linewidth=3, label=f'Your Line: y = {beta0} + {beta1}·x')
ax.set_xlabel('Feature Value (e.g., Median Income)')
ax.set_ylabel('Target Value (e.g., House Price in $K)')
ax.set_title(f'Linear Regression Demo — MSE = {mse_demo:.1f}', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig)

st.markdown(f"""
<div style="background-color: {'#E8F5E9' if mse_demo < 200 else '#FFEBEE'}; padding: 15px; border-radius: 8px;">
    <strong>MSE = {mse_demo:.1f}</strong> — {'Great fit! The line closely follows the data.' if mse_demo < 200 else 'High error. Try adjusting β₀ and β₁ to minimize MSE.'}
</div>
""", unsafe_allow_html=True)

st.markdown("---")

st.subheader("🧠 How Scikit-Learn Solves This")

st.markdown("""
Scikit-Learn uses the **Ordinary Least Squares (OLS)** method to find the optimal coefficients analytically:

<div style="background-color: #F3E5F5; padding: 15px; border-radius: 8px; font-family: 'Courier New', monospace; font-size: 1.1em; text-align: center;">
    <strong>β = (XᵀX)⁻¹ Xᵀy</strong>
</div>

This closed-form solution finds the exact coefficients that minimize MSE — no iteration needed!

**Our model's coefficients:**
""", unsafe_allow_html=True)

# Load and display coefficients
import pandas as pd
from pathlib import Path
from utils.data_loader import load_housing_data

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
coef_df = pd.read_csv(MODELS_DIR / "feature_importance.csv")

# A raw coefficient is "dollars per one unit of the feature", and one unit means
# something different for every feature. Multiplying by the feature's standard
# deviation puts them all on the same ruler: dollars per 1 std move.
df = load_housing_data()
coef_df['feature_std'] = coef_df['feature'].map(df.std(numeric_only=True))
coef_df['impact_per_std'] = coef_df['coefficient'] * coef_df['feature_std']
coef_df = coef_df.reindex(coef_df['impact_per_std'].abs().sort_values(ascending=False).index)
st.dataframe(coef_df, width="stretch")

st.markdown("""
**Interpretation:**
- **Positive coefficient** → Higher feature value = Higher predicted price
- **Negative coefficient** → Higher feature value = Lower predicted price
- **Larger absolute value ≠ more important.** The coefficient is measured *per unit of
  the feature*, and the units are not comparable. `population` has a tiny coefficient
  (~$0.01) only because it is counted in thousands of people; `has_pool` has a huge one
  (~$24,000) only because its single unit is the whole jump from "no pool" to "pool".
- **Use `impact_per_std`** (coefficient × feature std) to rank features fairly — that is
  the dollar move you get from a typical-sized change in each feature. This is the same
  idea as feature scaling, applied after training instead of before.
""")

st.warning(
    "⚠️ **Spot the trap:** `house_age` and `year_built` are the same fact written twice "
    "(`year_built = 2024 − house_age`). Their coefficients come out as an equal and "
    "opposite pair (+106.58 / −106.58) — a textbook case of **multicollinearity**. The "
    "predictions stay fine, but neither coefficient can be trusted on its own."
)
