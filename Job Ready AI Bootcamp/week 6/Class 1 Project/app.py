"""
Spam & Intent Classifier
────────────────────────
A production-grade Streamlit application demonstrating
Naive Bayes, SVM, and KNN on text classification tasks.

Runs 100% offline. No API keys required.
"""

import sys
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import joblib

import config
from src.data.loader import DataLoader
from src.training.trainer import Trainer
from src.utils.logger import get_logger

logger = get_logger("app")

# ── Page Config ─────────────────────────────────────
st.set_page_config(
    page_title="Spam & Intent Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1f77b4; }
    .sub-header { font-size: 1.2rem; color: #555; margin-bottom: 1rem; }
    .metric-card { background: #f0f2f6; border-radius: 10px; padding: 1rem; }
    .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────
st.sidebar.markdown("## ⚙️ Configuration")

task = st.sidebar.radio(
    "Select Task",
    ["Spam Detection (Binary)", "Intent Classification (Multi-class)"],
    index=0,
)

task_key = "spam" if "Spam" in task else "intent"

# Model selection
model_choice = st.sidebar.selectbox(
    "Choose Algorithm",
    ["Naive Bayes", "SVM", "KNN"],
    index=0,
)

# Hyperparameters (educational exposure)
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔧 Hyperparameters")

if model_choice == "Naive Bayes":
    nb_alpha = st.sidebar.slider("Alpha (Laplace smoothing)", 0.1, 2.0, 1.0, 0.1)
    model_params = {"alpha": nb_alpha}

elif model_choice == "SVM":
    svm_c = st.sidebar.slider("C (Regularization)", 0.01, 10.0, 1.0, 0.01)
    model_params = {"C": svm_c}

else:  # KNN
    knn_k = st.sidebar.slider("K (Neighbors)", 1, 15, 5, 1)
    knn_weights = st.sidebar.selectbox("Weighting", ["uniform", "distance"], index=1)
    model_params = {"n_neighbors": knn_k, "weights": knn_weights}

st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Class 11 — Module 3**
    - Naive Bayes
    - Support Vector Machines
    - K-Nearest Neighbors
    """
)

# ── Main Content ──────────────────────────────────────
st.markdown('<div class="main-header">🛡️ Spam & Intent Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Advanced Classification Models — 100% Offline</div>', unsafe_allow_html=True)

tabs = st.tabs(["📊 Data", "🚀 Train & Evaluate", "🔮 Predict", "📈 Compare Models"])

# ── Shared State ──────────────────────────────────────
if "trainer" not in st.session_state:
    st.session_state.trainer = None
if "dataset" not in st.session_state:
    st.session_state.dataset = None
if "history" not in st.session_state:
    st.session_state.history = []

# ── Tab 1: Data ───────────────────────────────────────
with tabs[0]:
    st.subheader("1. Data Management")

    data_source = st.radio(
        "Data Source",
        ["Generate Synthetic (Offline Demo)", "Upload CSV"],
        horizontal=True,
    )

    if data_source == "Generate Synthetic (Offline Demo)":
        col1, col2 = st.columns(2)
        with col1:
            n_samples = st.number_input(
                "Samples per class",
                min_value=50,
                max_value=2000,
                value=500 if task_key == "spam" else 200,
                step=50,
            )

        if st.button("🎲 Generate Dataset", type="primary"):
            loader = DataLoader()
            if task_key == "spam":
                df = loader.generate_spam_dataset(n_samples=n_samples * 2)
            else:
                df = loader.generate_intent_dataset(n_per_class=n_samples)

            st.session_state.dataset = df
            path = loader.save_processed(df, f"synthetic_{task_key}")
            st.success(f"Generated {len(df)} records. Saved to `{path}`")

    else:
        uploaded = st.file_uploader("Upload CSV (columns: `text`, `label`)", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
            if "text" not in df.columns or "label" not in df.columns:
                st.error("CSV must contain `text` and `label` columns.")
            else:
                st.session_state.dataset = df
                st.success(f"Loaded {len(df)} records from upload.")

    if st.session_state.dataset is not None:
        df = st.session_state.dataset
        st.markdown("---")
        st.markdown(f"**Preview:** `{len(df)}` rows, `{df['label'].nunique()}` classes")
        st.dataframe(df.head(10), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Class Distribution**")
            st.bar_chart(df["label"].value_counts())

        with col2:
            st.markdown("**Sample Text Lengths**")
            df["length"] = df["text"].str.len()
            fig, ax = plt.subplots()
            sns.boxplot(data=df, x="label", y="length", ax=ax, palette="Set2")
            plt.xticks(rotation=45)
            st.pyplot(fig)

# ── Tab 2: Train & Evaluate ───────────────────────────
with tabs[1]:
    st.subheader("2. Training Pipeline")

    if st.session_state.dataset is None:
        st.warning("Please generate or upload data in the **Data** tab first.")
    else:
        df = st.session_state.dataset

        if st.button("🏋️ Train Model", type="primary"):
            with st.spinner(f"Training {model_choice} for {task_key} classification..."):
                trainer = Trainer(task=task_key)
                metrics = trainer.run(
                    df,
                    model_name=model_choice,
                    model_params=model_params,
                )
                st.session_state.trainer = trainer
                st.session_state.history.append(metrics)

            # Metrics
            st.success("Training complete!")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
            c2.metric("Precision", f"{metrics['precision']:.2%}")
            c3.metric("Recall", f"{metrics['recall']:.2%}")
            c4.metric("F1 Score", f"{metrics['f1_score']:.2%}")

            # Confusion Matrix
            st.markdown("---")
            st.markdown("**Confusion Matrix**")
            cm = metrics["confusion_matrix"]
            labels = sorted(df["label"].unique())

            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=labels,
                yticklabels=labels,
                ax=ax,
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

            # Interpretation
            st.markdown("---")
            st.markdown("#### 🧠 Why these numbers matter")
            st.markdown("""
            - **Accuracy**: Overall correctness. Misleading if classes are imbalanced.
            - **Precision**: Of all predicted positives, how many were correct? (Minimize false alarms)
            - **Recall**: Of all actual positives, how many did we catch? (Minimize misses)
            - **F1 Score**: Harmonic mean of Precision & Recall. Best single metric for balance.
            """)

# ── Tab 3: Predict ────────────────────────────────────
with tabs[2]:
    st.subheader("3. Live Prediction")

    if st.session_state.trainer is None:
        st.warning("Train a model in the **Train & Evaluate** tab first.")
    else:
        user_input = st.text_area(
            "Enter text to classify",
            height=120,
            placeholder="e.g., 'Congratulations! You won a $1000 gift card. Call now!'",
        )

        if st.button("🔍 Classify", type="primary") and user_input.strip():
            trainer = st.session_state.trainer
            pred, probs = trainer.predict_text(user_input)

            # Result banner
            if task_key == "spam":
                if pred == "spam":
                    st.error(f"🚨 Prediction: **{pred.upper()}**")
                else:
                    st.success(f"✅ Prediction: **{pred.upper()}**")
            else:
                st.info(f"📌 Prediction: **{pred.upper()}**")

            # Probability bars
            st.markdown("**Confidence Scores:**")
            prob_df = pd.DataFrame.from_dict(probs, orient="index", columns=["Probability"])
            prob_df = prob_df.sort_values("Probability", ascending=True)

            st.bar_chart(prob_df)

            # Show preprocessing for transparency
            with st.expander("🔬 View Preprocessed Text"):
                from src.data.preprocessor import TextPreprocessor
                cleaned = TextPreprocessor().clean(user_input)
                st.code(cleaned, language="text")

# ── Tab 4: Compare Models ─────────────────────────────
with tabs[3]:
    st.subheader("4. Model Comparison")

    if not st.session_state.history:
        st.info("Train multiple models to see comparison. Try switching algorithms in the sidebar and training again!")
    else:
        hist_df = pd.DataFrame(st.session_state.history)
        hist_df = hist_df[["model_name", "accuracy", "precision", "recall", "f1_score"]]

        st.markdown("**Comparison Table**")
        st.dataframe(hist_df.style.highlight_max(subset=["accuracy", "f1_score"], color="green"), use_container_width=True)

        # Radar / Bar comparison
        st.markdown("**Metric Comparison Chart**")
        melted = hist_df.melt(id_vars=["model_name"], var_name="Metric", value_name="Score")

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=melted, x="Metric", y="Score", hue="model_name", ax=ax, palette="muted")
        ax.set_ylim(0, 1.05)
        ax.set_title("Algorithm Performance Comparison")
        st.pyplot(fig)

        st.markdown("""
        **Student Notes:**
        - **Naive Bayes** is usually fastest and surprisingly accurate for text.
        - **SVM** often wins on high-dimensional sparse data (like TF-IDF).
        - **KNN** is intuitive but can struggle with very high dimensions (curse of dimensionality).
        """)

# ── Footer ────────────────────────────────────────────
st.markdown("---")
st.caption("Built for Class 11 Module 3 | Runs 100% Offline | scikit-learn + Streamlit")
