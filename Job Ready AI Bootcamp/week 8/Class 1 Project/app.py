"""
High-Accuracy MNIST Digit Classifier
─────────────────────────────────────
A Streamlit application demonstrating a baseline CNN (Conv-Pool-Conv-Pool
-Dense) trained on `keras.datasets.mnist`, with a draw-a-digit predictor
and model-introspection tools.

Runs 100% offline (MNIST ships with Keras). No API keys required.

All TensorFlow/Keras calls live in `src/`; this file only wires the UI
together and calls into that package.

HIGHLIGHTS — why does this file never call `model.fit`, `model.predict`,
or `keras.Sequential` directly, even though the tabs below clearly train
and run a CNN? Streamlit re-executes this entire script from top to
bottom on every single UI interaction. If TensorFlow calls were sprinkled
throughout this file, every widget tweak would risk accidentally
re-triggering expensive work or subtly different behavior than the
`src/` pipeline the tests exercise. Keeping every Keras call inside
`src/` means the *exact same*, already-tested code path runs whether it's
invoked from `app.py`, a notebook, or `pytest`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from PIL import Image
from sklearn.metrics import confusion_matrix
from streamlit_drawable_canvas import st_canvas

import config
from src.data.loader import load_mnist
from src.data.preprocessor import DigitPreprocessor
from src.models.registry import ModelRegistry
from src.training.trainer import Trainer
from src.utils.logger import get_logger

logger = get_logger("app")
CFG = config.get_config()

st.set_page_config(
    page_title=CFG.app.page_title,
    page_icon=CFG.app.page_icon,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .main-header { font-size: 2.4rem; font-weight: 700; color: #1f77b4; }
        .sub-header { font-size: 1.1rem; color: #666; margin-bottom: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Shared session state ──────────────────────────────────────────
# HIGHLIGHTS: because Streamlit reruns this whole script on every click,
# any plain Python variable (e.g. `trained_model = None` at module level)
# would be reset to `None` on the very next interaction — clicking
# "Predict" right after "Train" would "forget" the model that was just
# trained. `st.session_state` is Streamlit's mechanism for values that
# must survive across reruns for the same browser session, so we stash
# the trained model, its history, and the loaded MNIST splits there
# instead of as local variables.
if "trained_model" not in st.session_state:
    st.session_state.trained_model = None
if "training_history" not in st.session_state:
    st.session_state.training_history = None
if "test_metrics" not in st.session_state:
    st.session_state.test_metrics = None
if "mnist_splits" not in st.session_state:
    # Caching the loaded/split MNIST arrays here (rather than re-calling
    # `load_mnist()` on every rerun) avoids re-downloading and
    # re-shuffling ~70k images every time a student clicks a slider in
    # the Predict or Model Insights tabs.
    st.session_state.mnist_splits = None


def _get_active_model():
    """Return a model to run inference/insights with: session > registry > None.

    HIGHLIGHTS: this fallback chain matters because a student's session
    can end (browser refresh, app restart) without losing their trained
    model. `session_state.trained_model` covers "I just trained this a
    moment ago" (fastest, no disk I/O). Falling back to
    `ModelRegistry(CFG).load_best()` covers "I trained a model earlier,
    restarted the app, and want to keep using it" — reading the
    highest-accuracy checkpoint back from disk. Only if neither exists do
    the Predict/Insights tabs show their "train a model first" warning.
    """
    if st.session_state.trained_model is not None:
        return st.session_state.trained_model
    return ModelRegistry(CFG).load_best()


# ── Sidebar ────────────────────────────────────────────────────────
st.sidebar.markdown("## Navigation")
st.sidebar.info(
    """
    **Week 8, Class 1 — Baseline CNN**

    Architecture:
    - Conv2D(32, 3x3) + ReLU
    - MaxPool(2x2)
    - Conv2D(64, 3x3) + ReLU
    - MaxPool(2x2)
    - Flatten
    - Dense(128) + ReLU
    - Dropout(0.5)
    - Dense(10, Softmax)

    Advanced tuning (batch norm, augmentation,
    LR schedules) is covered in Class 2.
    """
)

registry_versions = ModelRegistry(CFG).list_versions()
if registry_versions:
    best = max(registry_versions, key=lambda v: v.test_accuracy)
    st.sidebar.success(f"Best saved model: {best.test_accuracy:.2%} test accuracy")
else:
    st.sidebar.warning("No trained model yet — visit the **Train** tab.")

# ── Header ─────────────────────────────────────────────────────────
st.markdown(f'<div class="main-header">{CFG.app.page_icon} High-Accuracy MNIST Digit Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Baseline CNN · TensorFlow/Keras · Streamlit</div>', unsafe_allow_html=True)

tab_overview, tab_train, tab_predict, tab_insights = st.tabs(
    ["Overview", "Train", "Predict", "Model Insights"]
)

# ── Tab: Overview ─────────────────────────────────────────────────
with tab_overview:
    st.subheader("Project Overview")
    st.markdown(
        """
        This app trains and serves a **baseline Convolutional Neural Network** for
        recognizing handwritten digits (0-9) from the classic MNIST dataset.
        It is intentionally the *simple* architecture — no augmentation, no batch
        normalization, no learning-rate schedules — so the fundamentals of CNNs
        (convolution, pooling, dropout) are easy to see end to end.
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Pipeline")
        st.code(
            "Draw / Upload Digit\n"
            "        │\n"
            "        ▼\n"
            "  Preprocessor          (src/data/preprocessor.py)\n"
            "  grayscale -> resize 28x28 -> normalize [0,1]\n"
            "        │\n"
            "        ▼\n"
            "  Baseline CNN          (src/models/cnn.py)\n"
            "  Conv(32) -> Pool -> Conv(64) -> Pool -> Dense(128) -> Dropout -> Dense(10)\n"
            "        │\n"
            "        ▼\n"
            "  Softmax Probabilities (10 classes)\n"
            "        │\n"
            "        ▼\n"
            "  Predicted Digit + Confidence Bar Chart",
            language="text",
        )

    with col2:
        st.markdown("#### Configuration")
        cfg_table = pd.DataFrame(
            {
                "Setting": [
                    "Image size", "Channels", "Conv1 filters", "Conv2 filters",
                    "Dense units", "Dropout rate", "Batch size", "Epochs (full)",
                    "Epochs (quick demo)",
                ],
                "Value": [
                    f"{CFG.data.image_size[0]}x{CFG.data.image_size[1]}",
                    CFG.data.num_channels,
                    CFG.model.conv1_filters,
                    CFG.model.conv2_filters,
                    CFG.model.dense_units,
                    CFG.model.dropout_rate,
                    CFG.training.batch_size,
                    CFG.training.epochs,
                    CFG.training.quick_epochs,
                ],
            }
        )
        st.dataframe(cfg_table, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown(
        """
        **Tabs:**
        - **Train** — Train the baseline CNN live (full run or a quick subset for a fast demo).
        - **Predict** — Draw a digit or upload an image and get per-class confidence scores.
        - **Model Insights** — Confusion matrix and learned first-layer filters.
        """
    )

# ── Tab: Train ──────────────────────────────────────────────────────
with tab_train:
    st.subheader("Train the Baseline CNN")

    quick_demo = st.checkbox(
        "Use a quick subset for demo speed",
        value=True,
        help=(
            f"Trains on a {CFG.data.quick_subset_fraction:.0%} random subset for "
            f"{CFG.training.quick_epochs} epochs instead of the full dataset for "
            f"{CFG.training.epochs} epochs."
        ),
    )

    if st.button("Train Model", type="primary"):
        progress_bar = st.progress(0.0, text="Loading MNIST...")
        total_epochs = CFG.training.quick_epochs if quick_demo else CFG.training.epochs

        # HIGHLIGHTS: this closure is the entire bridge between Keras's
        # training loop and the Streamlit progress bar. `Trainer.train()`
        # (in src/training/trainer.py) has no idea Streamlit exists — it
        # just calls whatever plain function is passed as `on_epoch_end`
        # after each epoch. That keeps the trainer UI-agnostic and
        # testable while still letting this specific call site render a
        # live-updating progress bar with per-epoch accuracy.
        def _on_epoch_end(epoch, logs):
            fraction = min(1.0, (epoch + 1) / total_epochs)
            progress_bar.progress(
                fraction,
                text=f"Epoch {epoch + 1}/{total_epochs} — "
                     f"acc={logs.get('accuracy', 0):.3f}, val_acc={logs.get('val_accuracy', 0):.3f}",
            )

        with st.spinner("Training in progress..."):
            # Reuse the cached split from session_state if we already
            # loaded it this session (see the session-state block above),
            # otherwise load it once and cache it for every tab that
            # needs MNIST data afterward (Predict's canvas doesn't, but
            # Model Insights' confusion matrix does).
            splits = st.session_state.mnist_splits or load_mnist(CFG)
            st.session_state.mnist_splits = splits

            trainer = Trainer(CFG)
            result = trainer.train(
                quick_demo=quick_demo,
                splits=splits,
                on_epoch_end=_on_epoch_end,
            )

        st.session_state.trained_model = result.model
        st.session_state.training_history = result.history
        st.session_state.test_metrics = {
            "test_accuracy": result.test_accuracy,
            "test_loss": result.test_loss,
        }

        st.success(
            f"Training complete — test accuracy: {result.test_accuracy:.2%}, "
            f"test loss: {result.test_loss:.4f}"
        )

    if st.session_state.training_history is not None:
        history = st.session_state.training_history
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Accuracy Curve**")
            acc_df = pd.DataFrame(
                {"train": history.get("accuracy", []), "val": history.get("val_accuracy", [])}
            )
            st.line_chart(acc_df)

        with col2:
            st.markdown("**Loss Curve**")
            loss_df = pd.DataFrame(
                {"train": history.get("loss", []), "val": history.get("val_loss", [])}
            )
            st.line_chart(loss_df)

        metrics = st.session_state.test_metrics
        c1, c2 = st.columns(2)
        c1.metric("Test Accuracy", f"{metrics['test_accuracy']:.2%}")
        c2.metric("Test Loss", f"{metrics['test_loss']:.4f}")

# ── Tab: Predict ──────────────────────────────────────────────────
with tab_predict:
    st.subheader("Draw or Upload a Digit")

    active_model = _get_active_model()
    if active_model is None:
        st.warning("No trained model available yet. Train one in the **Train** tab first.")
    else:
        preprocessor = DigitPreprocessor(CFG)
        input_mode = st.radio("Input Method", ["Draw", "Upload Image"], horizontal=True)

        tensor = None

        if input_mode == "Draw":
            st.caption("Draw a single digit (0-9), centered, using your mouse.")
            # HIGHLIGHTS: stroke_color white on background_color black is
            # a deliberate choice, not a cosmetic one — it matches MNIST's
            # own convention (bright digit, dark background) at the
            # source. Because of this, `DigitPreprocessor.from_array()`
            # can skip the invert-detection heuristic entirely and treat
            # canvas input as already correctly oriented (see
            # `already_digit_bright=True` in preprocessor.py).
            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 1)",
                stroke_width=CFG.app.canvas_stroke_width,
                stroke_color="rgba(255, 255, 255, 1)",
                background_color="rgba(0, 0, 0, 1)",
                height=CFG.app.canvas_size,
                width=CFG.app.canvas_size,
                drawing_mode="freedraw",
                key="digit_canvas",
            )

            if canvas_result.image_data is not None and canvas_result.image_data[..., :3].sum() > 0:
                tensor = preprocessor.from_array(canvas_result.image_data)

        else:
            uploaded = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])
            if uploaded is not None:
                st.image(Image.open(uploaded), caption="Uploaded image", width=150)
                # HIGHLIGHTS: `Image.open(uploaded)` above already
                # consumed (read to the end of) the uploaded file's
                # stream to render the preview. `seek(0)` rewinds it so
                # `preprocessor.from_upload()` can open and read the same
                # bytes again — without this, the second `Image.open`
                # call inside the preprocessor would see an empty stream
                # and fail.
                uploaded.seek(0)
                tensor = preprocessor.from_upload(uploaded)

        if tensor is not None:
            st.markdown("---")
            probabilities = active_model.predict(tensor, verbose=0)[0]
            predicted_digit = int(np.argmax(probabilities))
            confidence = float(probabilities[predicted_digit])

            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("**Model Input (28x28)**")
                st.image(tensor.reshape(28, 28), width=150, clamp=True)
                st.metric("Predicted Digit", predicted_digit)
                st.metric("Confidence", f"{confidence:.2%}")

            with col2:
                st.markdown("**Per-Class Confidence Scores**")
                prob_df = pd.DataFrame(
                    {"digit": [str(i) for i in range(10)], "probability": probabilities}
                ).set_index("digit")
                st.bar_chart(prob_df)

# ── Tab: Model Insights ─────────────────────────────────────────────
with tab_insights:
    st.subheader("Model Insights")

    active_model = _get_active_model()
    if active_model is None:
        st.warning("No trained model available yet. Train one in the **Train** tab first.")
    else:
        with st.spinner("Loading test data for evaluation..."):
            splits = st.session_state.mnist_splits or load_mnist(CFG)
            st.session_state.mnist_splits = splits

        st.markdown("### Confusion Matrix (Test Set)")
        # HIGHLIGHTS: computing a confusion matrix over all 10,000 test
        # images is cheap for the model itself (a single batched
        # `predict` call), but rendering + reasoning about a 10x10 heatmap
        # is the same either way, so we let the student trade off
        # "faster" against "closer to the true test accuracy" rather than
        # hardcoding one or the other.
        n_eval = st.slider(
            "Number of test samples to evaluate (larger = slower but more accurate)",
            min_value=200, max_value=len(splits.x_test), value=min(2000, len(splits.x_test)), step=200,
        )

        if st.button("Compute Confusion Matrix"):
            x_eval, y_eval = splits.x_test[:n_eval], splits.y_test[:n_eval]
            with st.spinner("Running predictions..."):
                preds = np.argmax(active_model.predict(x_eval, verbose=0), axis=1)

            cm = confusion_matrix(y_eval, preds)
            fig, ax = plt.subplots(figsize=(7, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10), ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

            accuracy = float((preds == y_eval).mean())
            st.metric("Sampled Accuracy", f"{accuracy:.2%}")

        st.markdown("---")
        st.markdown("### Learned First-Layer Filters")
        st.caption("Each tile is one of the 32 learned 3x3 filters in `conv1` (visualized via `viridis`).")

        try:
            # `get_weights()[0]` is the kernel/filter weights (index [1]
            # would be the bias vector) — its shape is
            # (kernel_h, kernel_w, in_channels, out_channels), i.e.
            # (3, 3, 1, 32) for conv1. Indexing `[:, :, 0, i]` pulls out
            # the i-th 3x3 filter as a plain 2D grid we can imshow.
            # HIGHLIGHTS: visualizing conv1 specifically (not conv2) is
            # deliberate — conv1's filters operate directly on raw pixels,
            # so they're interpretable as simple edge/stroke detectors a
            # student can eyeball. conv2's filters operate on conv1's
            # *output* feature maps, not on pixels, so they wouldn't be
            # meaningfully visualizable the same way.
            conv1_weights = active_model.get_layer("conv1").get_weights()[0]
            n_filters = conv1_weights.shape[-1]

            fig, axes = plt.subplots(4, 8, figsize=(12, 6))
            for i, ax in enumerate(axes.flat):
                if i < n_filters:
                    ax.imshow(conv1_weights[:, :, 0, i], cmap="viridis")
                    ax.set_title(f"F{i + 1}", fontsize=8)
                ax.axis("off")
            plt.suptitle("First Conv Layer: Learned 3x3 Filters", fontsize=13, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig)
        except ValueError:
            # Guards against a future/alternate architecture (e.g. a
            # student experimenting with a differently-named or
            # differently-shaped first layer) not having a "conv1" layer
            # at all — fail soft with a message instead of crashing the
            # whole Model Insights tab.
            st.info("This model does not expose a `conv1` layer to visualize.")

# ── Footer ────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Week 8, Class 1 Project | Baseline CNN | TensorFlow + Streamlit | Runs 100% Offline")
