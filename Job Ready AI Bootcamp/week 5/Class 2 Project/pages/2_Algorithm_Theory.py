import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Algorithm Theory", page_icon="📐", layout="wide")

st.title("📐 Classification Algorithms: The Theory")
st.markdown("Understanding how Logistic Regression, Decision Trees, and Random Forests make predictions.")

# ============================================================
# LOGISTIC REGRESSION
# ============================================================
st.markdown("---")
st.header("1️⃣ Logistic Regression")

st.markdown("""
<div style="background-color: #E3F2FD; padding: 20px; border-radius: 10px; border-left: 5px solid #1565C0;">
<h3 style="margin-top: 0; color: #1565C0;">What is Logistic Regression?</h3>
<p>
Despite the name, Logistic Regression is a <strong>classification</strong> algorithm. 
It predicts the <strong>probability</strong> that an input belongs to a class (0 or 1), 
then converts that probability into a decision.
</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
### The Sigmoid Function

Logistic Regression uses the **sigmoid (logistic) function** to squeeze any real number into a probability between 0 and 1:

<div style="background-color: #FFF3E0; padding: 15px; border-radius: 8px; font-family: 'Courier New', monospace; font-size: 1.3em; text-align: center;">
    <strong>σ(z) = 1 / (1 + e⁻ᶻ)</strong>
</div>

Where **z = β₀ + β₁·x₁ + β₂·x₂ + ... + βₙ·xₙ** (same linear combination as Linear Regression!)

### Decision Rule

<div style="background-color: #E8F5E9; padding: 15px; border-radius: 8px; font-family: 'Courier New', monospace; font-size: 1.2em; text-align: center;">
    <strong>If σ(z) ≥ 0.5 → Approve (Class 1)</strong><br>
    <strong>If σ(z) < 0.5 → Deny (Class 0)</strong>
</div>

### Why Not Linear Regression for Classification?

Linear Regression predicts unbounded values (-∞ to +∞). For classification, we need **probabilities** (0 to 1). 
The sigmoid function provides this bounded output.
""")

# Interactive sigmoid demo
st.subheader("🎮 Interactive Sigmoid Demo")

col1, col2 = st.columns(2)
with col1:
    z_input = st.slider("Input Value (z)", -10.0, 10.0, 0.0, 0.1)
    sigmoid = 1 / (1 + np.exp(-z_input))
    st.markdown(f"**σ({z_input:.1f}) = {sigmoid:.4f}**")

    if sigmoid >= 0.5:
        st.success(f"✅ Prediction: APPROVE (probability = {sigmoid:.1%})")
    else:
        st.error(f"❌ Prediction: DENY (probability = {sigmoid:.1%})")

with col2:
    z_range = np.linspace(-10, 10, 200)
    sig_range = 1 / (1 + np.exp(-z_range))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(z_range, sig_range, 'b-', linewidth=2, label='Sigmoid σ(z)')
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, label='Decision Threshold (0.5)')
    ax.axvline(x=0, color='green', linestyle=':', linewidth=1.5, label='z = 0')
    ax.scatter([z_input], [sigmoid], color='red', s=100, zorder=5, label=f'Your Input: z={z_input:.1f}')
    ax.fill_between(z_range, 0, sig_range, where=(sig_range >= 0.5), alpha=0.2, color='green', label='Approve Region')
    ax.fill_between(z_range, 0, sig_range, where=(sig_range < 0.5), alpha=0.2, color='red', label='Deny Region')
    ax.set_xlabel('z (Linear Combination of Features)')
    ax.set_ylabel('Probability σ(z)')
    ax.set_title('The Sigmoid Function', fontweight='bold')
    ax.legend(loc='lower right', fontsize=8)
    ax.set_ylim(0, 1)
    st.pyplot(fig)

# ============================================================
# DECISION TREES
# ============================================================
st.markdown("---")
st.header("2️⃣ Decision Trees")

st.markdown("""
<div style="background-color: #E3F2FD; padding: 20px; border-radius: 10px; border-left: 5px solid #1565C0;">
<h3 style="margin-top: 0; color: #1565C0;">What is a Decision Tree?</h3>
<p>
A Decision Tree asks a series of <strong>yes/no questions</strong> about the input features, 
splitting the data at each step until it reaches a final decision (leaf node).
</p>
</div>
""", unsafe_allow_html=True)

st.markdown(r"""
### How It Works

1. **Select the best feature** to split on (using Gini Impurity or Entropy)
2. **Split the data** into two groups based on a threshold
3. **Repeat** for each subgroup until stopping criteria met
4. **Leaf nodes** contain the final prediction (majority class)

### Example Decision Path for Loan Approval

```
                    [Credit Score ≥ 650?]
                    /         \
                  Yes          No
                  /             \
        [Income ≥ $50K?]    [Has Co-signer?]
         /        \          /        \
       Yes        No        Yes        No
        |          |          |          |
    APPROVE     DENY      APPROVE     DENY
```

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Gini Impurity** | Measures how "mixed" a node is (0 = pure, 0.5 = maximum impurity) |
| **Information Gain** | Reduction in impurity after a split |
| **Max Depth** | How many questions the tree can ask (prevents overfitting) |
| **Min Samples Leaf** | Minimum samples required in a leaf node |
""")

st.info("💡 **Strength:** Highly interpretable — you can literally trace the decision path! **Weakness:** Prone to overfitting on deep trees.")

# ============================================================
# RANDOM FORESTS
# ============================================================
st.markdown("---")
st.header("3️⃣ Random Forests")

st.markdown("""
<div style="background-color: #E3F2FD; padding: 20px; border-radius: 10px; border-left: 5px solid #1565C0;">
<h3 style="margin-top: 0; color: #1565C0;">What is a Random Forest?</h3>
<p>
A <strong>Random Forest</strong> is an <strong>ensemble</strong> of many Decision Trees. 
Each tree votes on the prediction, and the majority vote wins. This reduces overfitting 
and improves accuracy compared to a single tree.
</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
### How It Works (Bagging + Random Subspaces)

1. **Bootstrap Sampling:** Create 100 random subsets of the training data (with replacement)
2. **Random Feature Selection:** At each split, only consider a random subset of features
3. **Train Trees:** Build a Decision Tree on each bootstrap sample
4. **Vote:** Aggregate predictions from all trees (majority vote for classification)

<div style="background-color: #FFF3E0; padding: 15px; border-radius: 8px; font-family: 'Courier New', monospace; font-size: 1.1em; text-align: center;">
    <strong>Final Prediction = mode(Tree₁, Tree₂, ..., Tree₁₀₀)</strong>
</div>

### Why Random Forests Work Better

| Problem | Single Tree | Random Forest |
|---------|------------|---------------|
| Overfitting | High risk (memorizes training data) | Low risk (averaging reduces variance) |
| Accuracy | Good on training, poor on new data | Good generalization |
| Stability | Sensitive to data changes | Robust — small changes don't affect all trees |
| Interpretability | Very interpretable | Less interpretable (but feature importance helps) |

### Hyperparameters

| Parameter | Effect | Typical Value |
|-----------|--------|---------------|
| **n_estimators** | Number of trees | 100 |
| **max_depth** | Max tree depth | 8-12 |
| **min_samples_leaf** | Min samples per leaf | 20-50 |
| **max_features** | Features considered per split | sqrt(n_features) |
""")

st.success("✅ **Key Insight:** Random Forests combine the simplicity of Decision Trees with the power of ensemble learning. They're often the best 'first algorithm' to try on tabular data!")
