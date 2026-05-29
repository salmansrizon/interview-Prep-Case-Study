import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from utils.data_loader import load_housing_data
from utils.visualizations import plot_actual_vs_predicted, plot_residuals

st.set_page_config(page_title="Model Evaluation", page_icon="📏", layout="wide")

st.title("📏 Model Evaluation: How Good Are Our Predictions?")
st.markdown("Understanding the metrics that tell us if our model is useful.")

st.markdown("""
<div style="background-color: #E8F5E9; padding: 20px; border-radius: 10px; border-left: 5px solid #2E7D32;">
<h3 style="margin-top: 0; color: #2E7D32;">Why Evaluate?</h3>
<p>
A model that fits training data perfectly might fail on new data (<strong>overfitting</strong>). 
We split data into <strong>training</strong> and <strong>testing</strong> sets to measure 
<strong>generalization</strong> — how well the model performs on unseen houses.
</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

st.subheader("📐 The Four Key Metrics")

st.markdown("""
| Metric | Formula | Interpretation | Goal |
|--------|---------|----------------|------|
| **MSE** | (1/n) · Σ(y - ŷ)² | Average squared error | Minimize |
| **RMSE** | √MSE | Error in same units as target ($) | Minimize |
| **MAE** | (1/n) · Σ\|y - ŷ\| | Average absolute error (robust to outliers) | Minimize |
| **R²** | 1 - (SS_res / SS_tot) | % of variance explained by model | Maximize (→ 1.0) |

**R² = 0.85** means the model explains **85%** of price variation. The remaining 15% is due to 
features we don't have (e.g., neighborhood prestige, recent renovations) or random noise.
""")

# Load data and train model
df = load_housing_data()
feature_cols = [c for c in df.columns if c != 'price']
X = df[feature_cols]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

st.markdown("---")

st.subheader("📊 Our Model's Performance")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("MSE", f"${mse:,.0f}")
col2.metric("RMSE", f"${rmse:,.0f}", help="Typical prediction error")
col3.metric("MAE", f"${mae:,.0f}", help="Average absolute error")
col4.metric("R² Score", f"{r2:.4f}", help="Variance explained")
col5.metric("MAPE", f"{mape:.1f}%", help="Mean absolute % error")

st.markdown("""
<div style="background-color: #E3F2FD; padding: 15px; border-radius: 8px;">
    <strong>Interpretation:</strong> On average, our predictions are off by <strong>${rmse:,.0f}</strong> 
    (RMSE) or <strong>${mae:,.0f}</strong> (MAE). The model explains <strong>{r2:.1%}</strong> of price variation.
</div>
""".format(rmse=rmse, mae=mae, r2=r2), unsafe_allow_html=True)

st.markdown("---")

st.subheader("📈 Actual vs Predicted")

fig1 = plot_actual_vs_predicted(y_test.values, y_pred)
st.pyplot(fig1)

st.markdown("""
💡 **How to read this:** Points close to the red dashed line mean accurate predictions. 
Scatter around the line shows prediction uncertainty.
""")

st.markdown("---")

st.subheader("📉 Residual Analysis")

fig2 = plot_residuals(y_test.values, y_pred)
st.pyplot(fig2)

st.markdown("""
**What residuals tell us:**
- **Random scatter around 0** → Good! Model captures the pattern
- **Funnel shape** → Heteroscedasticity (error increases with price)
- **Curved pattern** → Model is missing non-linear relationships
- **Normal distribution** → Validates statistical assumptions
""")

st.markdown("---")

st.subheader("🎯 Prediction Examples")

n_examples = 10
sample_idx = np.random.choice(len(y_test), n_examples, replace=False)
examples_df = pd.DataFrame({
    'Actual Price': y_test.iloc[sample_idx].values,
    'Predicted Price': y_pred[sample_idx],
    'Error': y_test.iloc[sample_idx].values - y_pred[sample_idx],
    'Error %': ((y_test.iloc[sample_idx].values - y_pred[sample_idx]) / y_test.iloc[sample_idx].values * 100).round(1)
})
examples_df['Actual Price'] = examples_df['Actual Price'].apply(lambda x: f"${x:,.0f}")
examples_df['Predicted Price'] = examples_df['Predicted Price'].apply(lambda x: f"${x:,.0f}")
examples_df['Error'] = examples_df['Error'].apply(lambda x: f"${x:,.0f}")
st.dataframe(examples_df, use_container_width=True)
