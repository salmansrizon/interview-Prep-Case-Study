# 🔢 High-Accuracy MNIST Digit Classifier

## সহজ ভাষায় Project Overview

**🔢 High-Accuracy MNIST Digit Classifier** project-এ lecture-এর theory-কে working software বা executable notebook-এ convert করা হয়েছে। লক্ষ্য শুধু final output দেখা নয়; input থেকে preprocessing, core logic/model, evaluation এবং output—পুরো pipeline বোঝা।

### কোন Problem Solve করে?

Manual বা disconnected workflow-কে repeatable code pipeline-এ আনে। এর ফলে একই process নতুন data-তে আবার চালানো, result compare করা, error trace করা এবং future feature add করা সহজ হয়।

### কীভাবে কাজ করে?

Input/Data → Validation ও Preprocessing → Core Algorithm/Model → Evaluation → UI, Report বা Saved Output। নিচের detailed section-গুলোতে project-specific command, feature এবং architecture দেওয়া আছে।

### কেন এই Approach ভালো?

- **Repeatable:** একই input দিলে একই workflow follow করে।
- **Testable:** প্রতিটি stage আলাদাভাবে verify করা যায়।
- **Explainable:** কোন step কী কাজ করছে তা code এবং output দিয়ে দেখা যায়।
- **Portfolio-ready:** শুধু notebook result নয়, setup, structure এবং usage-সহ complete project হিসেবে দেখানো যায়।

> **Run করার নিয়ম:** আগে virtual environment তৈরি করে dependency install করুন। তারপর README-এর Quick Start follow করুন, sample input দিয়ে smoke test করুন এবং expected metric/output-এর সাথে result compare করুন।

---

A production-grade Streamlit application that builds, trains, and compares
three convolutional neural network variants — **Baseline**, **With
Dropout**, and **Fully Optimized** — on the classic MNIST handwritten
digit dataset. It's built to demonstrate, hands-on, *why* regularization
and training-loop optimization techniques (Dropout, Batch Normalization,
Early Stopping, ReduceLROnPlateau) matter: not as abstract theory, but as
directly observable differences in train-vs-validation curves and final
test accuracy.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.15+-orange.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![Tests](https://img.shields.io/badge/tests-pytest-green.svg)

## ✨ Features

- 🏗️ **Three CNN variants, one codebase** — Baseline, With Dropout, and
  Fully Optimized architectures, built from a shared, isolated-variable
  design so their comparison is fair and interpretable.
- 🎛️ **Train & Compare tab** — train any variant in-browser (with a fast
  "quick demo subset" option), see live train-vs-validation accuracy/loss
  curves for every variant trained this session, and a summary metrics
  table.
- ✍️ **Interactive Predict tab** — draw a digit on a canvas or upload an
  image; the app preprocesses it to MNIST's format and shows a per-class
  confidence bar chart from the Fully Optimized model.
- 📊 **Model Insights tab** — confusion matrix for the optimized model and
  a cross-variant comparison table (accuracy, overfitting gap, epochs run,
  training time).
- 🧪 **Fast, offline test suite** — structural tests (shapes, layer
  presence, forward-pass smoke tests) that run in seconds, no real
  training required.
- 🧱 **Clean separation of concerns** — `app.py` is UI-only; all
  Keras/TensorFlow logic lives in `src/`, fully testable and reusable
  without Streamlit.
- 📚 **Teaching-oriented code** — every non-obvious design choice (why
  BatchNorm sits between Conv and ReLU, why EarlyStopping monitors
  `val_loss` instead of `val_accuracy`, why Dropout rates differ between
  variants) is explained inline in the source, not just in this README.

## 🚀 Quick Start

```bash
# 1. Navigate to the project folder
cd "Job Ready AI Bootcamp/week 8/Class 2 Project"

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app
streamlit run app.py

# 5. (Optional) Run the test suite
pytest tests/ -v
```

The app opens at `http://localhost:8501`. In the **Train & Compare** tab,
enable "Quick demo subset" and train all three variants — each takes only
seconds on a laptop CPU — to see the overfitting-gap story play out
end-to-end.

## 📁 Project Structure

```
Class 2 Project/
├── app.py                       # Streamlit entry point (UI only, tabs a-d)
├── config.py                    # Dataclass-based configuration (single source of truth)
├── requirements.txt
├── README.md
├── data/
│   └── models/                  # Saved .keras artifacts per variant (created at runtime)
├── notebooks/
│   └── 01_eda.ipynb             # MNIST EDA + comparison-preview notebook
├── src/
│   ├── data/
│   │   ├── loader.py            # Load, normalize, reshape, split MNIST
│   │   └── preprocessor.py      # Canvas/upload image -> model-ready tensor
│   ├── models/
│   │   ├── architectures.py     # build_baseline_cnn / build_dropout_cnn / build_optimized_cnn
│   │   └── registry.py          # Save/load trained variants
│   ├── training/
│   │   └── trainer.py           # train_variant / compare_variants orchestration
│   └── utils/
│       └── logger.py            # Structured logging
└── tests/
    └── test_pipeline.py         # Fast, offline pytest suite
```

## 🔧 Architecture

```
                     ┌───────────────────────────┐
                     │   keras.datasets.mnist     │
                     └─────────────┬─────────────┘
                                   │
                          src/data/loader.py
                  normalize [0,1] · reshape (28,28,1) · train/val/test split
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
   ┌─────────────────┐  ┌───────────────────┐  ┌──────────────────────┐
   │  Baseline CNN    │  │  Dropout CNN      │  │  Fully Optimized CNN │
   │  Conv->Pool x2   │  │  Conv->Pool x2     │  │  Conv->BN->ReLU->Pool │
   │  Dense->Dense     │  │  + Dropout(0.5)    │  │  x2 + Dropout(0.3)x2  │
   │  fixed epochs     │  │  fixed epochs      │  │  EarlyStopping +      │
   │                   │  │                    │  │  ReduceLROnPlateau    │
   └────────┬─────────┘  └─────────┬──────────┘  └───────────┬───────────┘
            │                       │                         │
            └───────────────────────┼─────────────────────────┘
                                    ▼
                        src/training/trainer.py
                   train_variant() · compare_variants()
                                    │
                                    ▼
                         src/models/registry.py
                    save/load per-variant .keras artifacts
                                    │
                                    ▼
                               app.py (Streamlit)
       ┌───────────┬─────────────────┬──────────┬────────────────────┐
       │ Overview  │ Train & Compare │ Predict  │  Model Insights     │
       └───────────┴─────────────────┴──────────┴────────────────────┘
```

## ⚙️ Configuration

All hyperparameters live in `config.py` as frozen dataclasses — nothing is
hardcoded in `src/` or `app.py`.

| Setting | Default | Where | Why |
|---|---|---|---|
| `image_size` | 28 | `DataConfig` | Native MNIST resolution; also the preprocessor's resize target |
| `val_split` | 0.1 | `DataConfig` | Held-out fraction of the training set for EarlyStopping/ReduceLROnPlateau to monitor |
| `quick_subset_train_size` | 4000 | `DataConfig` | Sample size for the app's "quick demo" training mode |
| `conv1_filters` / `conv2_filters` | 32 / 64 | `ModelConfig` | Standard "start small, double up" CNN filter progression |
| `dropout_dense_rate` | 0.5 | `ModelConfig` | Dropout strength for the isolated Dropout-only variant |
| `dropout_conv_rate` | 0.3 | `ModelConfig` | Gentler Dropout strength for the Fully Optimized variant (BatchNorm already regularizes) |
| `learning_rate` | 1e-3 | `ModelConfig` | Adam's well-tested default, identical across all 3 variants |
| `baseline_epochs` / `dropout_epochs` | 15 | `TrainingConfig` | Fixed epoch budget — no EarlyStopping, by design |
| `optimized_epochs` | 30 | `TrainingConfig` | Ceiling only; EarlyStopping decides the actual stop point |
| `early_stopping_patience` | 5 | `TrainingConfig` | Epochs to wait for `val_loss` improvement before stopping |
| `early_stopping_monitor` | `val_loss` | `TrainingConfig` | Smoother, earlier overfitting signal than `val_accuracy` |
| `reduce_lr_factor` / `reduce_lr_patience` | 0.5 / 3 | `TrainingConfig` | Learning-rate decay on plateau, triggers before EarlyStopping |

## 🧠 Model Architecture & Optimization

### The three variants

**1. Baseline** — `Conv2D(32,3x3,relu) → MaxPool → Conv2D(64,3x3,relu) → MaxPool → Flatten → Dense(128,relu) → Dense(10,softmax)`.
No dropout, no batch normalization, no early stopping — trained for a
fixed epoch count regardless of what validation loss is doing. This is
the "naive" version a student would write before learning optimization
techniques: it has plenty of capacity to memorize the training set, and
nothing stops it from doing so.

**2. With Dropout** — identical convolutional backbone, plus
`Dropout(0.5)` inserted right before the output layer. During training,
Dropout randomly zeroes half of the Dense(128) activations on every
batch, forcing the network to spread useful information across many
neurons instead of over-relying on a few that memorize specific examples
(*co-adaptation*). At inference time, Dropout automatically becomes a
no-op.

**3. Fully Optimized** —
`Conv2D(32,3x3) → BatchNorm → ReLU → MaxPool → Conv2D(64,3x3) → BatchNorm → ReLU → MaxPool → Flatten → Dropout(0.3) → Dense(128,relu) → Dropout(0.3) → Dense(10,softmax)`,
trained with `EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)`
and `ReduceLROnPlateau`. **Target: >98% test accuracy.** BatchNorm sits
between each convolution and its ReLU activation — normalizing the raw
linear output before the nonlinearity clips negative values, per the
original Batch Normalization paper's recommended ordering — and
stabilizes/accelerates training. Dropout rates are gentler here (0.3 vs.
0.5) because BatchNorm already contributes a mild regularizing effect on
its own; stacking two strong regularizers tends to underfit.

### Why each technique is there

| Technique | Problem it solves | How |
|---|---|---|
| **Dropout** | Overfitting / memorization | Randomly disables neurons during training, preventing co-adaptation |
| **Batch Normalization** | Training instability & slow convergence | Normalizes each mini-batch's activations, reducing internal covariate shift |
| **Early Stopping** | Wasted compute & late-stage overfitting | Monitors `val_loss`; stops once it stops improving for `patience` epochs, restoring the best weights |
| **ReduceLROnPlateau** | Getting "stuck" near a good-but-not-great minimum | Shrinks the learning rate when `val_loss` plateaus, before EarlyStopping gives up entirely |

### Typical train-vs-validation accuracy gap

| Variant | Typical Train Acc. | Typical Val Acc. | Typical Gap | Diagnosis |
|---|---|---|---|---|
| Baseline | ~99.5% | ~98.5% | ~1.0 pt, growing with more epochs | Mild-to-moderate overfitting |
| With Dropout | ~99.0% | ~98.7% | ~0.3 pt | Reduced overfitting |
| Fully Optimized | ~99.2% | ~99.0% | ~0.2 pt, stable | Good fit — generalizes well |

Exact numbers depend on the epoch budget, random seed, and whether you
use the full dataset or the quick-demo subset — use the **Train &
Compare** tab to reproduce and inspect these curves yourself.

## 🧪 Testing

```bash
pytest tests/ -v
```

The suite is intentionally fast (seconds, not minutes): it never trains a
real model. Instead it verifies data-loader shapes, preprocessor
output/normalization, that each architecture builder produces the correct
input/output shapes, that the Fully Optimized model actually contains
`BatchNormalization` and `Dropout` layers (and the others don't have
layers they shouldn't), and a one-batch forward-pass smoke test per
architecture.

## 📜 License

MIT License — feel free to use in your own projects and portfolios.
