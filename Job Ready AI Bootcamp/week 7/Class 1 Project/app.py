"""
Neural Network Lab
──────────────────
A production-grade Streamlit application for learning Neural Networks from scratch.
Covers: The Neuron, Weights & Biases, Activation Functions, Backpropagation.

Runs 100% offline. No API keys required.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config
from src.core.neuron import Neuron
from src.core.activation import ActivationFunctions
from src.core.network import NeuralNetwork
from src.gates.logic_gates import LogicGateDataset
from src.visualization.nn_viz import NetworkVisualizer
from src.utils.logger import get_logger

logger = get_logger("app")

st.set_page_config(
    page_title="Neural Network Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #6C5CE7; }
    .sub-header { font-size: 1.2rem; color: #555; margin-bottom: 1rem; }
    .metric-card { background: linear-gradient(135deg, #6C5CE7 0%, #A29BFE 100%); 
                   color: white; border-radius: 12px; padding: 1.5rem; text-align: center; }
    .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("## ⚙️ Configuration")

module = st.sidebar.radio(
    "Select Module",
    ["🔬 The Neuron", "⚡ Activation Functions", "🕸️ Build a Neural Network", "📊 Training & Backpropagation"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Class — Neural Networks**
    - The Neuron
    - Weights & Biases
    - Activation Functions
    - Backpropagation Intuition
    """
)

st.markdown('<div class="main-header">🧠 Neural Network Lab</div>', unsafe_allow_html=True)

if "nn_model" not in st.session_state:
    st.session_state.nn_model = None
if "training_history" not in st.session_state:
    st.session_state.training_history = []

# ═══════════════════════════════════════════════════════
# MODULE 1: THE NEURON
# ═══════════════════════════════════════════════════════
if module == "🔬 The Neuron":
    st.subheader("1. The Neuron — Building Block of Neural Networks")

    col1, col2 = st.columns([3, 2])

    with col2:
        st.markdown("#### 📚 About the Neuron")
        st.markdown("""
        A **neuron** (or perceptron) is the fundamental unit of a neural network.

        **Mathematical Model:**
        ```
        output = activation(w1*x1 + w2*x2 + ... + wn*xn + b)
        ```

        Where:
        - **xᵢ**: Input features
        - **wᵢ**: Weights (importance of each input)
        - **b**: Bias (shifts the activation threshold)
        - **activation**: Non-linear transformation
        """)

    with col1:
        st.markdown("#### 🎛️ Interactive Neuron Simulator")

        n_inputs = st.slider("Number of Inputs", 1, 5, 2, 1)

        inputs = []
        weights = []
        cols = st.columns(n_inputs)
        for i, col in enumerate(cols):
            with col:
                x = st.slider(f"x{i+1}", -5.0, 5.0, 1.0, 0.1, key=f"x_{i}")
                w = st.slider(f"w{i+1}", -3.0, 3.0, 0.5, 0.1, key=f"w_{i}")
                inputs.append(x)
                weights.append(w)

        bias = st.slider("Bias (b)", -5.0, 5.0, 0.0, 0.1)

        neuron = Neuron(weights=weights, bias=bias)
        z = neuron.compute_z(inputs)

        st.markdown("---")
        st.markdown("**Neuron Computation:**")

        terms = [f"{w:.2f}×{x:.2f}" for w, x in zip(weights, inputs)]
        equation = " + ".join(terms) + f" + {bias:.2f}"

        st.code(f"""
z (weighted sum) = {equation}
                 = {z:.4f}
        """)

        # Visualize the neuron
        viz = NetworkVisualizer()
        fig = viz.draw_single_neuron(inputs, weights, bias, z)
        st.pyplot(fig)

        st.markdown("---")
        st.markdown("**What does z represent?**")
        st.info(f"""
        **z = {z:.4f}** is the weighted sum before activation.

        - If z is **large positive**: The neuron is "strongly activated"
        - If z is **large negative**: The neuron is "strongly inhibited"
        - If z is **near zero**: The neuron is at the "decision boundary"

        The **bias** shifts this boundary. A positive bias makes the neuron more likely to fire.
        """)

# ═══════════════════════════════════════════════════════
# MODULE 2: ACTIVATION FUNCTIONS
# ═══════════════════════════════════════════════════════
elif module == "⚡ Activation Functions":
    st.subheader("2. Activation Functions — Adding Non-Linearity")

    col1, col2 = st.columns([3, 2])

    with col2:
        st.markdown("#### 📚 Why Activation Functions?")
        st.markdown("""
        Without activation functions, a neural network is just a **linear model** — no matter how many layers, it could only learn straight-line relationships.

        **Activation functions introduce non-linearity**, allowing networks to learn:
        - Curved decision boundaries
        - Complex patterns in data
        - XOR and other non-linear logic

        **Common Functions:**
        - **Sigmoid**: S-shaped, outputs 0–1 (good for probabilities)
        - **ReLU**: max(0, x) — fast, prevents vanishing gradients
        - **Tanh**: S-shaped, outputs -1–1 (zero-centered)
        """)

    with col1:
        st.markdown("#### 📈 Interactive Activation Function Explorer")

        act_choice = st.selectbox(
            "Select Activation Function",
            ["Sigmoid", "ReLU", "Tanh", "Leaky ReLU", "Compare All"],
            index=0,
        )

        x_range = np.linspace(-10, 10, 500)
        act = ActivationFunctions()

        if act_choice == "Sigmoid":
            y = act.sigmoid(x_range)
            y_deriv = act.sigmoid_derivative(x_range)
            formula = r"$\sigma(x) = \frac{1}{1 + e^{-x}}$"
            description = "S-shaped curve. Outputs between 0 and 1. Used for binary classification output layer."
            pros_cons = "✅ Smooth gradient | ❌ Vanishing gradient problem for large |x|"

        elif act_choice == "ReLU":
            y = act.relu(x_range)
            y_deriv = act.relu_derivative(x_range)
            formula = r"$ReLU(x) = max(0, x)$"
            description = "Rectified Linear Unit. Zero for negative inputs, linear for positive. Most popular hidden layer activation."
            pros_cons = "✅ Fast, no vanishing gradient for x>0 | ❌ 'Dying ReLU' — neurons can permanently turn off"

        elif act_choice == "Tanh":
            y = act.tanh(x_range)
            y_deriv = act.tanh_derivative(x_range)
            formula = r"$tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$"
            description = "Hyperbolic tangent. Zero-centered, outputs -1 to 1. Better than sigmoid for hidden layers."
            pros_cons = "✅ Zero-centered (helps convergence) | ❌ Still has vanishing gradient problem"

        elif act_choice == "Leaky ReLU":
            alpha = st.slider("Leak Parameter (α)", 0.01, 0.5, 0.1, 0.01)
            y = act.leaky_relu(x_range, alpha)
            y_deriv = act.leaky_relu_derivative(x_range, alpha)
            formula = r"$LeakyReLU(x) = max(αx, x)$"
            description = "Variant of ReLU that allows small negative values. Prevents dying neurons."
            pros_cons = "✅ No dying neurons | ❌ Adds another hyperparameter (α)"

        else:  # Compare All
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_range, y=act.sigmoid(x_range), name="Sigmoid", line=dict(color="#6C5CE7")))
            fig.add_trace(go.Scatter(x=x_range, y=act.relu(x_range), name="ReLU", line=dict(color="#00B894")))
            fig.add_trace(go.Scatter(x=x_range, y=act.tanh(x_range), name="Tanh", line=dict(color="#0984E3")))
            fig.update_layout(
                title="Activation Functions Comparison",
                xaxis_title="Input (z)",
                yaxis_title="Output",
                yaxis=dict(range=[-1.5, 1.5]),
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.stop()

        # Plot function and derivative
        fig = make_subplots(rows=1, cols=2, subplot_titles=(f"{act_choice} Function", f"{act_choice} Derivative"))

        fig.add_trace(go.Scatter(x=x_range, y=y, name=act_choice, line=dict(color="#6C5CE7", width=3)), row=1, col=1)
        fig.add_trace(go.Scatter(x=x_range, y=y_deriv, name="Derivative", line=dict(color="#E17055", width=3)), row=1, col=2)

        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"**Formula:** {formula}")
        st.markdown(f"**Description:** {description}")
        st.markdown(f"**Pros & Cons:** {pros_cons}")

        # Interactive test
        st.markdown("---")
        st.markdown("#### 🎯 Test with Your Own Value")
        test_z = st.slider("Input z", -10.0, 10.0, 2.0, 0.1)

        if act_choice == "Sigmoid":
            test_out = act.sigmoid(np.array([test_z]))[0]
            test_deriv = act.sigmoid_derivative(np.array([test_z]))[0]
        elif act_choice == "ReLU":
            test_out = act.relu(np.array([test_z]))[0]
            test_deriv = act.relu_derivative(np.array([test_z]))[0]
        elif act_choice == "Tanh":
            test_out = act.tanh(np.array([test_z]))[0]
            test_deriv = act.tanh_derivative(np.array([test_z]))[0]
        else:
            test_out = act.leaky_relu(np.array([test_z]), alpha)[0]
            test_deriv = act.leaky_relu_derivative(np.array([test_z]), alpha)[0]

        c1, c2 = st.columns(2)
        c1.metric(f"{act_choice}(z)", f"{test_out:.4f}")
        c2.metric(f"{act_choice}' (z)", f"{test_deriv:.4f}")

# ═══════════════════════════════════════════════════════
# MODULE 3: BUILD A NEURAL NETWORK
# ═══════════════════════════════════════════════════════
elif module == "🕸️ Build a Neural Network":
    st.subheader("3. Build a 2-Layer Neural Network from Scratch")

    col1, col2 = st.columns([3, 2])

    with col2:
        st.markdown("#### 📚 Architecture")
        st.markdown("""
        **2-Layer Network:**
        ```
        Input Layer    Hidden Layer    Output Layer
           x1  ──→    h1 (ReLU)   ──→    y (Sigmoid)
           x2  ──→    h2 (ReLU)
                      h3 (ReLU)
        ```

        **Forward Pass:**
        ```
        z1 = W1·x + b1      (hidden pre-activation)
        h = ReLU(z1)        (hidden activation)
        z2 = W2·h + b2      (output pre-activation)
        y = Sigmoid(z2)     (output activation)
        ```

        **Why 2 layers?** A single layer can only learn linear functions. Two layers with non-linear activation can learn XOR — the classic proof that neural networks work.
        """)

    with col1:
        st.markdown("#### 🏗️ Network Architecture Builder")

        gate_type = st.selectbox(
            "Logic Gate to Learn",
            ["AND", "OR", "XOR", "NAND", "NOR"],
            index=2,
        )

        hidden_size = st.slider("Hidden Layer Size", 2, 10, 3, 1)
        learning_rate = st.slider("Learning Rate", 0.01, 1.0, 0.5, 0.01)

        if st.button("🚀 Initialize Network", type="primary"):
            dataset = LogicGateDataset(gate_type)
            X, y = dataset.get_data()

            nn = NeuralNetwork(
                input_size=2,
                hidden_size=hidden_size,
                output_size=1,
                learning_rate=learning_rate,
            )

            st.session_state.nn_model = nn
            st.session_state.dataset = dataset
            st.session_state.gate_type = gate_type

            st.success(f"Network initialized for **{gate_type}** gate!")
            st.markdown(f"**Training Data:**")

            df_data = pd.DataFrame({
                "x1": X[:, 0],
                "x2": X[:, 1],
                f"{gate_type}": y.flatten(),
            })
            st.dataframe(df_data, use_container_width=True)

            # Visualize network architecture
            viz = NetworkVisualizer()
            fig_arch = viz.draw_network_architecture(2, hidden_size, 1)
            st.pyplot(fig_arch)

            st.markdown("**Initial Weights (Random):**")
            st.code(f"""
W1 (Input → Hidden): {nn.W1.shape}
{nn.W1.round(4)}

b1 (Hidden Bias): {nn.b1.flatten().round(4)}

W2 (Hidden → Output): {nn.W2.shape}
{nn.W2.round(4)}

b2 (Output Bias): {nn.b2.flatten().round(4)}
            """)

# ═══════════════════════════════════════════════════════
# MODULE 4: TRAINING & BACKPROPAGATION
# ═══════════════════════════════════════════════════════
else:
    st.subheader("4. Training & Backpropagation — The Learning Process")

    if st.session_state.nn_model is None:
        st.warning("Please build a network in the **Build a Neural Network** tab first!")
    else:
        nn = st.session_state.nn_model
        dataset = st.session_state.dataset
        gate_type = st.session_state.gate_type
        X, y = dataset.get_data()

        col1, col2 = st.columns([2, 1])

        with col2:
            st.markdown("#### 📚 Backpropagation Intuition")
            st.markdown("""
            **Backpropagation = "Blame Assignment"**

            1. **Forward Pass**: Make a prediction
            2. **Calculate Error**: How wrong was the prediction?
            3. **Backward Pass**: Distribute the blame to each weight
            4. **Update Weights**: Adjust weights to reduce error

            **The Chain Rule:**
            ```
            dL/dW = dL/dy × dy/dz × dz/dW
            ```

            Error flows backward from output to input, updating each weight proportionally to its contribution to the error.
            """)

        with col1:
            st.markdown("#### 🏋️ Train Your Network")

            epochs = st.slider("Training Epochs", 100, 10000, 2000, 100)

            if st.button("🏋️ Start Training", type="primary"):
                with st.spinner("Training in progress..."):
                    history = nn.train(X, y, epochs=epochs, verbose=False)
                    st.session_state.training_history = history

                # Final predictions
                predictions = nn.predict(X)

                st.markdown("---")
                st.markdown("#### 📊 Training Results")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Final Loss", f"{history['loss'][-1]:.6f}")
                m2.metric("Initial Loss", f"{history['loss'][0]:.4f}")
                m3.metric("Loss Reduction", f"{history['loss'][0] / history['loss'][-1]:.1f}x")
                m4.metric("Epochs", epochs)

                # Loss curve
                st.markdown("---")
                st.markdown("**Loss Curve — Learning Progress**")

                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(
                    x=list(range(len(history["loss"]))),
                    y=history["loss"],
                    mode="lines",
                    name="Loss",
                    line=dict(color="#6C5CE7", width=2),
                ))
                fig_loss.update_layout(
                    title="Training Loss Over Time",
                    xaxis_title="Epoch",
                    yaxis_title="Binary Cross-Entropy Loss",
                    height=400,
                )
                st.plotly_chart(fig_loss, use_container_width=True)

                # Predictions table
                st.markdown("---")
                st.markdown("**Predictions vs. Ground Truth**")

                results_df = pd.DataFrame({
                    "x1": X[:, 0],
                    "x2": X[:, 1],
                    "Target": y.flatten(),
                    "Prediction": predictions.flatten().round(4),
                    "Rounded": np.round(predictions.flatten()),
                    "Correct?": (np.round(predictions.flatten()) == y.flatten()),
                })
                st.dataframe(results_df, use_container_width=True)

                accuracy = np.mean(np.round(predictions.flatten()) == y.flatten())
                st.metric("Accuracy", f"{accuracy*100:.1f}%")

                # Decision boundary visualization
                st.markdown("---")
                st.markdown("**Decision Boundary Visualization**")

                viz = NetworkVisualizer()
                fig_boundary = viz.plot_decision_boundary(nn, X, y, gate_type)
                st.pyplot(fig_boundary)

                # Weight evolution
                st.markdown("---")
                st.markdown("**Weight Evolution During Training**")

                fig_weights = go.Figure()
                for i in range(nn.W1.shape[0]):
                    for j in range(nn.W1.shape[1]):
                        fig_weights.add_trace(go.Scatter(
                            x=list(range(len(history["W1_history"]))),
                            y=[w[i, j] for w in history["W1_history"]],
                            mode="lines",
                            name=f"W1[{i},{j}]",
                            line=dict(width=1.5),
                        ))
                fig_weights.update_layout(
                    title="Hidden Layer Weights Over Time",
                    xaxis_title="Epoch",
                    yaxis_title="Weight Value",
                    height=400,
                    showlegend=True,
                )
                st.plotly_chart(fig_weights, use_container_width=True)

                # Final weights
                st.markdown("---")
                st.markdown("**Final Trained Weights:**")
                st.code(f"""
W1 (Input → Hidden):
{nn.W1.round(4)}

b1 (Hidden Bias):
{nn.b1.flatten().round(4)}

W2 (Hidden → Output):
{nn.W2.round(4)}

b2 (Output Bias):
{nn.b2.flatten().round(4)}
                """)

                if accuracy == 1.0:
                    st.balloons()
                    st.success("🎉 Perfect accuracy! The network has learned the logic gate completely!")
                elif accuracy >= 0.75:
                    st.info("✅ Good accuracy! The network has mostly learned the pattern.")
                else:
                    st.warning("⚠️ Low accuracy. Try increasing epochs or hidden layer size.")

st.markdown("---")
st.caption("Built for Neural Networks Module | Runs 100% Offline | NumPy + Streamlit")
