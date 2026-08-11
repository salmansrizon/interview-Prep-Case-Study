"""
Streamlit entry point for the High-Accuracy MNIST Digit Classifier.

Run with:
    streamlit run app.py

This file is UI-only: it renders widgets and wires user actions to
functions in ``src/`` (data loading/preprocessing, model building,
training, registry). No Keras/TensorFlow calls happen directly here.

HIGHLIGHTS: keeping app.py free of direct Keras/TensorFlow calls isn't
just tidiness — it's what makes src/ independently testable. tests/
test_pipeline.py never has to spin up Streamlit to exercise the model
builders or the preprocessor; it imports directly from src/. If model
logic were tangled into button-click handlers here, "does the Fully
Optimized model contain BatchNorm?" would only be answerable by clicking
through the running app.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from config import VARIANT_DISPLAY_NAMES, VARIANT_NAMES, get_config
from src.data.loader import MNISTData, load_mnist, make_quick_subset
from src.data.preprocessor import preprocess_canvas_image
from src.models.registry import load_variant, variant_is_available
from src.training.trainer import VariantResult, results_to_dataframe, train_variant

CONFIG = get_config()

st.set_page_config(
    page_title="MNIST Optimization Lab",
    page_icon="\U0001F522",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Cached resources ─────────────────────────────────────────────────
# st.cache_resource (not st.cache_data) is the right decorator here because
# the return value is a container of numpy arrays we want to *share by
# reference* across reruns and across users of the same server process —
# MNIST is ~50k+10k images, and Streamlit reruns this script top-to-bottom
# on every widget interaction, so without caching we'd re-download and
# re-normalize the full dataset on every single click.
@st.cache_resource(show_spinner="Loading MNIST dataset ...")
def _cached_full_data() -> MNISTData:
    return load_mnist()


def _get_training_data(quick_demo: bool) -> MNISTData:
    """Return the full dataset, or a fast quick-demo subset of it."""
    full = _cached_full_data()
    if quick_demo:
        return make_quick_subset(full)
    return full


def _init_session_state() -> None:
    # st.session_state (NOT a module-level global) is what keeps "models
    # trained this session" scoped per browser tab/user. Streamlit reruns
    # this whole script on every interaction, so any plain Python variable
    # declared at module scope would be wiped out on the very next rerun —
    # session_state is the one place that survives across reruns.
    if "results" not in st.session_state:
        st.session_state.results: Dict[str, VariantResult] = {}


_init_session_state()


# ── Sidebar ──────────────────────────────────────────────────────────
def render_sidebar() -> None:
    st.sidebar.title("\U0001F522 MNIST Optimization Lab")
    st.sidebar.markdown(
        "Baseline vs. Dropout vs. **Fully Optimized** CNNs for handwritten "
        "digit recognition, built to demonstrate regularization and "
        "training-loop optimization techniques."
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Session Status")
    for variant in VARIANT_NAMES:
        trained = variant in st.session_state.results
        saved = variant_is_available(variant)
        if trained:
            acc = st.session_state.results[variant].test_accuracy
            st.sidebar.success(f"{VARIANT_DISPLAY_NAMES[variant]}: {acc:.2%} (this session)")
        elif saved:
            st.sidebar.info(f"{VARIANT_DISPLAY_NAMES[variant]}: saved on disk")
        else:
            st.sidebar.warning(f"{VARIANT_DISPLAY_NAMES[variant]}: not trained yet")
    st.sidebar.markdown("---")
    st.sidebar.caption("Target for the Fully Optimized variant: **>98% test accuracy**.")


# ── Tab: Overview ────────────────────────────────────────────────────
def render_overview_tab() -> None:
    st.header("Overview: Why Optimize a CNN?")
    st.markdown(
        """
A convolutional network can reach very high *training* accuracy simply by
memorizing the training set. The real goal is **generalization** — high
accuracy on data the model has never seen. This app builds and compares
three MNIST CNN variants that sit on that spectrum:
"""
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("1. Baseline")
        st.markdown(
            "- Conv(32) -> Pool -> Conv(64) -> Pool -> Dense(128) -> Dense(10)\n"
            "- No dropout, no batch norm, no early stopping\n"
            "- Fixed epoch budget\n"
            "- **Tends to overfit**: train accuracy keeps climbing while "
            "validation accuracy plateaus or degrades."
        )
    with col2:
        st.subheader("2. With Dropout")
        st.markdown(
            "- Same conv stack as baseline\n"
            "- `Dropout(0.5)` before the output layer\n"
            "- Randomly zeroes neurons during training to prevent "
            "co-adaptation\n"
            "- Narrows the train/val gap, but doesn't fix everything on its own."
        )
    with col3:
        st.subheader("3. Fully Optimized")
        st.markdown(
            "- `Conv -> BatchNorm -> ReLU -> Pool` twice\n"
            "- `Dropout(0.3)` after Flatten and after Dense(128)\n"
            "- `EarlyStopping(monitor='val_loss', patience=5, "
            "restore_best_weights=True)`\n"
            "- Optional `ReduceLROnPlateau`\n"
            "- **Target: >98% test accuracy** with a small train/val gap."
        )

    st.markdown("---")
    st.subheader("Technique Cheat Sheet")
    st.markdown(
        """
| Technique | What it does | Where it's used here |
|---|---|---|
| **Dropout** | Randomly disables a fraction of neurons each training step, forcing redundant, robust representations | Dropout variant (0.5), Optimized variant (0.3) |
| **Batch Normalization** | Normalizes layer activations per mini-batch, stabilizing and speeding up training | Optimized variant only, after each Conv layer |
| **Early Stopping** | Monitors validation loss and halts training once it stops improving, restoring the best epoch's weights | Optimized variant only, patience=5 |
| **ReduceLROnPlateau** | Shrinks the learning rate when validation loss stalls, helping fine-tune convergence | Optimized variant only |
"""
    )

    st.subheader("Typical Train-vs-Validation Accuracy Gap")
    st.markdown(
        """
| Variant | Typical Train Acc. | Typical Val Acc. | Typical Gap | Diagnosis |
|---|---|---|---|---|
| Baseline | ~99.5% | ~98.5% | ~1.0 pt, growing with more epochs | Mild-to-moderate overfitting |
| With Dropout | ~99.0% | ~98.7% | ~0.3 pt | Reduced overfitting |
| Fully Optimized | ~99.2% | ~99.0% | ~0.2 pt, stable | Good fit, generalizes well |

Exact numbers vary by run/seed/epoch budget — use the **Train & Compare**
tab to reproduce them yourself.
"""
    )


# ── Tab: Train & Compare ────────────────────────────────────────────
def _plot_history_comparison(results: Dict[str, VariantResult]) -> None:
    """Plot train (solid) vs. validation (dashed) accuracy and loss curves
    for every variant trained so far, one color per variant.

    HIGHLIGHTS: this chart is the single most important diagnostic in
    the app. A widening solid/dashed gap for a color IS overfitting, drawn
    directly from that variant's own history — this is exactly the
    "training loss keeps dropping, validation loss turns back up" picture
    described in the lecture. Plotting all trained variants on the same
    axes (rather than one chart per variant) is what makes the *relative*
    improvement from Dropout and then BatchNorm+EarlyStopping visible at a
    glance, instead of requiring the reader to mentally compare separate
    charts.
    """
    if not results:
        st.info("Train at least one variant to see its curves here.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    palette = sns.color_palette("tab10", n_colors=len(results))

    for color, (name, result) in zip(palette, results.items()):
        # x-axis uses each variant's OWN epochs_run, not a shared range —
        # the optimized variant may have stopped early (see trainer.py),
        # so its curve is simply shorter; plotting it against the same
        # x-range as a full 15/30-epoch baseline would be misleading.
        epochs_range = range(1, result.epochs_run + 1)
        label = VARIANT_DISPLAY_NAMES[name]
        # Solid line = train metric, dashed line = validation metric, same
        # color per variant — the gap between a solid and its matching
        # dashed line at the same x-position is the overfitting_gap made visual.
        axes[0].plot(epochs_range, result.history["accuracy"], color=color, linestyle="-", label=f"{label} (train)")
        axes[0].plot(epochs_range, result.history["val_accuracy"], color=color, linestyle="--", label=f"{label} (val)")
        axes[1].plot(epochs_range, result.history["loss"], color=color, linestyle="-", label=f"{label} (train)")
        axes[1].plot(epochs_range, result.history["val_loss"], color=color, linestyle="--", label=f"{label} (val)")

    axes[0].set_title("Accuracy: Train (solid) vs. Val (dashed)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(fontsize=8)

    axes[1].set_title("Loss: Train (solid) vs. Val (dashed)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_train_tab() -> None:
    st.header("Train & Compare Variants")
    st.markdown(
        "Train any of the three variants in this session. Trained models "
        "are saved to `data/models/` and immediately available in the "
        "**Predict** and **Model Insights** tabs."
    )

    quick_demo = st.checkbox(
        "Quick demo subset (few thousand samples, trains in seconds instead of minutes)",
        value=True,
    )

    cols = st.columns(3)
    for col, variant in zip(cols, VARIANT_NAMES):
        with col:
            st.subheader(VARIANT_DISPLAY_NAMES[variant])
            if st.button(f"Train {VARIANT_DISPLAY_NAMES[variant]}", key=f"train_{variant}", use_container_width=True):
                data = _get_training_data(quick_demo)
                with st.spinner(f"Training {VARIANT_DISPLAY_NAMES[variant]} ..."):
                    result = train_variant(variant, data)
                st.session_state.results[variant] = result
                st.success(
                    f"Done: test accuracy {result.test_accuracy:.2%} "
                    f"in {result.epochs_run} epochs ({result.train_time_seconds:.1f}s)"
                )

    st.markdown("---")
    st.subheader("Training Curves (this session)")
    _plot_history_comparison(st.session_state.results)

    if st.session_state.results:
        st.subheader("Summary Table")
        df = results_to_dataframe(st.session_state.results)
        df["variant"] = df["variant"].map(VARIANT_DISPLAY_NAMES)
        st.dataframe(
            df.style.format(
                {
                    "test_accuracy": "{:.2%}",
                    "test_loss": "{:.4f}",
                    "train_accuracy": "{:.2%}",
                    "val_accuracy": "{:.2%}",
                    "overfitting_gap": "{:.2%}",
                    "train_time_seconds": "{:.1f}s",
                }
            ),
            use_container_width=True,
        )


# ── Tab: Predict ─────────────────────────────────────────────────────
def _load_predict_model() -> Optional[object]:
    """Prefer the in-memory model from this session (no disk I/O, and it's
    guaranteed to match what the Train & Compare tab just showed); fall
    back to a model saved on disk from a previous session/run so the
    Predict tab still works without forcing a retrain every time the app
    restarts."""
    if "optimized" in st.session_state.results:
        return st.session_state.results["optimized"].model
    if variant_is_available("optimized"):
        return load_variant("optimized")
    return None


def render_predict_tab() -> None:
    st.header("Predict a Digit")
    model = _load_predict_model()

    if model is None:
        st.warning(
            "No trained **Fully Optimized** model available yet. Go to "
            "**Train & Compare** and train it first (the quick demo subset "
            "works fine for this)."
        )
        return

    st.markdown("Draw a digit (0-9) below, or upload an image, then click **Predict**.")

    input_image = None
    draw_col, upload_col = st.columns(2)

    with draw_col:
        st.markdown("**Draw**")
        try:
            from streamlit_drawable_canvas import st_canvas

            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 1)",
                stroke_width=18,
                stroke_color="#FFFFFF",
                background_color="#000000",
                height=CONFIG.data.image_size * 8,
                width=CONFIG.data.image_size * 8,
                drawing_mode="freedraw",
                key="digit_canvas",
            )
            if canvas_result is not None and canvas_result.image_data is not None:
                if np.any(canvas_result.image_data[..., :3] > 0):
                    input_image = canvas_result.image_data
        except ImportError:
            st.info("Install `streamlit-drawable-canvas` to enable the drawing pad.")

    with upload_col:
        st.markdown("**Or upload**")
        uploaded = st.file_uploader("Upload a PNG/JPG of a digit", type=["png", "jpg", "jpeg"])
        if uploaded is not None:
            from PIL import Image

            input_image = Image.open(uploaded)
            st.image(input_image, caption="Uploaded image", width=150)

    if st.button("Predict", type="primary", disabled=input_image is None):
        tensor = preprocess_canvas_image(input_image)
        probabilities = model.predict(tensor, verbose=0)[0]
        predicted_digit = int(np.argmax(probabilities))

        st.markdown("---")
        result_col, chart_col = st.columns([1, 2])
        with result_col:
            st.metric("Predicted Digit", predicted_digit)
            st.metric("Confidence", f"{probabilities[predicted_digit]:.2%}")

        with chart_col:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.bar(range(10), probabilities, color="#4C72B0")
            ax.set_xticks(range(10))
            ax.set_xlabel("Digit")
            ax.set_ylabel("Confidence")
            ax.set_title("Per-Class Confidence")
            st.pyplot(fig)
            plt.close(fig)


# ── Tab: Model Insights ──────────────────────────────────────────────
def render_insights_tab() -> None:
    st.header("Model Insights")

    if "optimized" not in st.session_state.results:
        st.warning(
            "Train the **Fully Optimized** model in the **Train & Compare** "
            "tab to see its confusion matrix here."
        )
    else:
        result = st.session_state.results["optimized"]
        st.subheader("Confusion Matrix (Fully Optimized, Test Set)")

        quick_demo_note = st.checkbox("Use quick demo subset for the confusion matrix", value=True)
        data = _get_training_data(quick_demo_note)
        predictions = result.model.predict(data.x_test, verbose=0)
        predicted_labels = np.argmax(predictions, axis=1)

        matrix = pd.crosstab(
            pd.Series(data.y_test, name="Actual"),
            pd.Series(predicted_labels, name="Predicted"),
        ).reindex(index=range(10), columns=range(10), fill_value=0)

        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title("Confusion Matrix: Fully Optimized CNN")
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")
    st.subheader("Cross-Variant Metrics")
    if not st.session_state.results:
        st.info("No variants trained yet this session. Visit **Train & Compare** first.")
        return

    df = results_to_dataframe(st.session_state.results)
    df["variant"] = df["variant"].map(VARIANT_DISPLAY_NAMES)
    st.dataframe(
        df[["variant", "test_accuracy", "overfitting_gap", "epochs_run", "train_time_seconds"]].style.format(
            {
                "test_accuracy": "{:.2%}",
                "overfitting_gap": "{:.2%}",
                "train_time_seconds": "{:.1f}s",
            }
        ),
        use_container_width=True,
    )


# ── Main ─────────────────────────────────────────────────────────────
def main() -> None:
    render_sidebar()
    st.title("High-Accuracy MNIST Digit Classifier")
    st.caption("Baseline vs. Dropout vs. Fully Optimized CNNs, side by side.")

    tab_overview, tab_train, tab_predict, tab_insights = st.tabs(
        ["Overview", "Train & Compare", "Predict", "Model Insights"]
    )

    with tab_overview:
        render_overview_tab()
    with tab_train:
        render_train_tab()
    with tab_predict:
        render_predict_tab()
    with tab_insights:
        render_insights_tab()


if __name__ == "__main__":
    main()
