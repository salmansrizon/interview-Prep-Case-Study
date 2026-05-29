import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils.data_loader import load_housing_data

st.set_page_config(page_title="Feature Scaling", page_icon="⚖️", layout="wide")

st.title("⚖️ Feature Scaling: Why Size Matters")
st.markdown("Understanding why and how to normalize features for machine learning.")

st.markdown("""
<div style="background-color: #E8F5E9; padding: 20px; border-radius: 10px; border-left: 5px solid #2E7D32;">
<h3 style="margin-top: 0; color: #2E7D32;">The Problem</h3>
<p>
Features have <strong>different scales</strong>. House age ranges 1-50 years, while lot size ranges 
3,000-20,000 sqft. Algorithms that use distance or gradients can be dominated by large-scale features, 
ignoring small but important ones.
</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

st.subheader("📊 Before Scaling: Features on Different Scales")

df = load_housing_data()
features_to_scale = ['median_income', 'house_age', 'lot_size_sqft', 'distance_to_city', 'school_rating']

fig, ax = plt.subplots(figsize=(12, 6))
box_data = [df[f].values for f in features_to_scale]
bp = ax.boxplot(box_data, labels=[f.replace('_', ' ').title() for f in features_to_scale], 
                patch_artist=True)
colors = plt.cm.Set3(np.linspace(0, 1, len(features_to_scale)))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
ax.set_ylabel('Value')
ax.set_title('Feature Distributions BEFORE Scaling', fontweight='bold')
ax.set_yscale('log')
st.pyplot(fig)

st.warning("⚠️ See how `lot_size_sqft` (thousands) dwarfs `school_rating` (1-10)? The model would ignore school quality!")

st.markdown("---")

st.subheader("🔧 StandardScaler: Z-Score Normalization")

st.markdown("""
<div style="background-color: #FFF3E0; padding: 15px; border-radius: 8px; font-family: 'Courier New', monospace; font-size: 1.2em; text-align: center;">
    <strong>x_scaled = (x - μ) / σ</strong>
</div>

Where:
- **μ (mu)** = mean of the feature
- **σ (sigma)** = standard deviation of the feature

**Result:** All features have **mean = 0** and **std = 1**. No feature dominates!
""")

# Apply StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[features_to_scale])
scaled_df = pd.DataFrame(scaled_data, columns=features_to_scale)

fig2, ax2 = plt.subplots(figsize=(12, 6))
box_data2 = [scaled_df[f].values for f in features_to_scale]
bp2 = ax2.boxplot(box_data2, labels=[f.replace('_', ' ').title() for f in features_to_scale], 
                  patch_artist=True)
for patch, color in zip(bp2['boxes'], colors):
    patch.set_facecolor(color)
ax2.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Mean = 0')
ax2.set_ylabel('Standardized Value')
ax2.set_title('Feature Distributions AFTER StandardScaler', fontweight='bold')
ax2.legend()
st.pyplot(fig2)

st.success("✅ Now all features are on the same scale! Each contributes fairly to the model.")

st.markdown("---")

st.subheader("🎮 Interactive Scaling Demo")

feature_demo = st.selectbox("Select a Feature to Scale", features_to_scale)

original = df[feature_demo]
scaled = scaled_df[feature_demo]

col1, col2 = st.columns(2)
with col1:
    st.markdown(f"**Original: {feature_demo.replace('_', ' ').title()}**")
    st.markdown(f"Mean: {original.mean():.2f}")
    st.markdown(f"Std:  {original.std():.2f}")
    st.markdown(f"Min:  {original.min():.2f}")
    st.markdown(f"Max:  {original.max():.2f}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(original, bins=40, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(original.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean={original.mean():.1f}')
    ax.set_title('Original Distribution')
    ax.legend()
    st.pyplot(fig)

with col2:
    st.markdown(f"**Scaled: {feature_demo.replace('_', ' ').title()}**")
    st.markdown(f"Mean: {scaled.mean():.4f}")
    st.markdown(f"Std:  {scaled.std():.4f}")
    st.markdown(f"Min:  {scaled.min():.2f}")
    st.markdown(f"Max:  {scaled.max():.2f}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(scaled, bins=40, edgecolor='black', alpha=0.7, color='darkgreen')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Mean=0')
    ax.set_title('Scaled Distribution (StandardScaler)')
    ax.legend()
    st.pyplot(fig)

st.markdown("---")

st.subheader("📚 Scaling Methods Comparison")

st.markdown("""
| Method | Formula | Range | Best For |
|--------|---------|-------|----------|
| **StandardScaler** | (x - μ) / σ | Unbounded (-∞, +∞) | Most ML algorithms (LR, SVM, Neural Nets) |
| **MinMaxScaler** | (x - min) / (max - min) | [0, 1] | Neural networks, image data |
| **RobustScaler** | (x - median) / IQR | Unbounded | Data with outliers |

**Note:** For Linear Regression, scaling is NOT strictly necessary (coefficients adjust automatically). 
But it's critical for regularized models (Ridge, Lasso) and distance-based algorithms (KNN, SVM, Neural Networks).
""")

st.info("💡 **Key Takeaway:** Always scale features when using regularization or algorithms sensitive to feature magnitudes. It ensures fair contribution from all variables!")
