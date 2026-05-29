import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Gradient Descent", page_icon="📉", layout="wide")

st.title("📉 Gradient Descent: How Models Learn")
st.markdown("A step-by-step visualization of how optimization finds the best model parameters.")

st.markdown("""
<div style="background-color: #E8F5E9; padding: 20px; border-radius: 10px; border-left: 5px solid #2E7D32;">
<h3 style="margin-top: 0; color: #2E7D32;">What is Gradient Descent?</h3>
<p>
Gradient Descent is an <strong>optimization algorithm</strong> that iteratively adjusts model parameters 
(coefficients) to minimize the cost function (MSE). It works like a hiker finding the bottom of a valley 
by always walking downhill.
</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

st.subheader("🧗 The Algorithm")

st.markdown("""
**Step 1:** Start with random coefficients (β₀, β₁)

**Step 2:** Calculate the gradient (slope) of the cost function at current position

**Step 3:** Update coefficients by moving in the OPPOSITE direction of the gradient:

<div style="background-color: #FFF3E0; padding: 15px; border-radius: 8px; font-family: 'Courier New', monospace; font-size: 1.2em; text-align: center;">
    <strong>β_new = β_old - α · ∇J(β)</strong>
</div>

Where:
- **α (alpha)** = Learning Rate — how big each step is
- **∇J(β)** = Gradient of the cost function — direction of steepest ascent
- **-∇J(β)** = Direction of steepest descent

**Step 4:** Repeat until convergence (MSE stops decreasing significantly)
""")

st.markdown("---")

st.subheader("🎮 Interactive Gradient Descent Visualization")

st.markdown("Watch how the learning rate affects convergence speed and stability.")

col1, col2, col3 = st.columns(3)
with col1:
    learning_rate = st.select_slider("Learning Rate (α)", 
                                      options=[0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0],
                                      value=0.1)
with col2:
    n_iterations = st.slider("Iterations", 10, 200, 50, 10)
with col3:
    start_beta = st.slider("Starting β₁", -4.0, 8.0, 0.0, 0.5)

# Simulate gradient descent
np.random.seed(42)
x_gd = np.linspace(0, 50, 100)
y_gd = 20 + 2.5 * x_gd + np.random.normal(0, 8, 100)

# Cost function landscape for β1 (fixing β0 = 20)
beta1_range = np.linspace(-1, 6, 200)
mse_landscape = []
for b1 in beta1_range:
    pred = 20 + b1 * x_gd
    mse_landscape.append(np.mean((y_gd - pred) ** 2))

# Run gradient descent
beta1_history = [start_beta]
mse_history = []
for i in range(n_iterations):
    current_b1 = beta1_history[-1]
    pred = 20 + current_b1 * x_gd
    # Gradient of MSE w.r.t β1: (2/n) * Σ(pred - y) * x
    gradient = (2 / len(x_gd)) * np.sum((pred - y_gd) * x_gd)
    new_b1 = current_b1 - learning_rate * gradient
    beta1_history.append(new_b1)
    mse_history.append(np.mean((y_gd - pred) ** 2))

# Plot 1: Cost function landscape with GD path
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].plot(beta1_range, mse_landscape, 'b-', linewidth=2, label='Cost Function (MSE)')
axes[0].scatter(beta1_history, [np.mean((y_gd - (20 + b * x_gd)) ** 2) for b in beta1_history], 
                c=range(len(beta1_history)), cmap='viridis', s=50, zorder=5, edgecolors='black')
axes[0].axvline(x=2.5, color='red', linestyle='--', linewidth=2, label='Optimal β₁ = 2.5')
axes[0].set_xlabel('Coefficient β₁')
axes[0].set_ylabel('MSE (Cost)')
axes[0].set_title(f'Gradient Descent Path (α = {learning_rate})', fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: MSE over iterations
axes[1].plot(range(len(mse_history)), mse_history, 'g-', linewidth=2, marker='o', markersize=4)
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('MSE')
axes[1].set_title('MSE Decrease Over Iterations', fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig)

# Final stats
final_mse = mse_history[-1] if mse_history else 0
final_beta = beta1_history[-1]

col1, col2, col3 = st.columns(3)
col1.metric("Final β₁", f"{final_beta:.3f}", f"{final_beta - 2.5:.3f} from optimal")
col2.metric("Final MSE", f"{final_mse:.1f}")
col3.metric("Converged?", "✅ Yes" if abs(final_beta - 2.5) < 0.1 else "❌ No")

st.markdown("---")

st.subheader("⚠️ Learning Rate Matters!")

st.markdown("""
| Learning Rate | Behavior | Result |
|--------------|----------|--------|
| **Too Small (α < 0.01)** | Tiny steps, very slow | Takes forever to converge |
| **Just Right (α ≈ 0.1)** | Smooth descent | Fast, stable convergence |
| **Too Large (α > 0.5)** | Giant leaps, overshoots | Diverges — MSE increases! |

**In practice:** Scikit-Learn's `LinearRegression` uses the **Normal Equation** (closed-form) 
instead of Gradient Descent for linear regression, which finds the exact solution instantly. 
However, Gradient Descent is essential for more complex models (neural networks, logistic regression).
""")

st.info("💡 **Key Insight:** Gradient Descent is the engine that powers almost all modern ML. Understanding it unlocks neural networks, deep learning, and beyond!")
