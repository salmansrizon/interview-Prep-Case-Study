# 🔢 High-Accuracy MNIST Digit Classifier

A production-grade Convolutional Neural Network (CNN) for handwritten digit
recognition, built with TensorFlow/Keras and served through an interactive
Streamlit app — for **Week 8, Class 1: Convolutional Neural Networks**.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.15+-orange.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![Tests](https://img.shields.io/badge/tests-pytest-green.svg)

## ✨ Features

- 🧠 **Baseline CNN** — Conv(32) → Pool → Conv(64) → Pool → Dense(128) → Dropout(0.5) → Dense(10), ~99% test accuracy on MNIST
- ✏️ **Draw-to-Predict** — Draw a digit with your mouse on an in-browser canvas and get an instant prediction
- 📤 **Upload Fallback** — Predict from any uploaded digit photo, with automatic grayscale/invert handling
- 📊 **Per-Class Confidence** — Bar chart of softmax probabilities for all 10 digits
- 🏋️ **Live Training** — Train the baseline CNN from the browser, with a "quick demo" subset toggle for fast iteration
- 📈 **Live Curves** — Accuracy/loss curves streamed epoch-by-epoch during training
- 🔍 **Model Insights** — Confusion matrix on the test set and a visualization of the learned first-layer filters
- 🗂️ **Model Registry** — Every trained model is versioned by timestamp + test accuracy, with the best one auto-selected for inference
- ✅ **Tested** — Fast pytest suite covering data shapes, preprocessing, and a model forward-pass smoke test (no full training in CI)

## 🚀 Quick Start

```bash
# 1. Navigate to the project folder
cd "Job Ready AI Bootcamp/week 8/Class 1 Project"

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app
streamlit run app.py
```

The app opens at `http://localhost:8501`. Head to the **Train** tab first (use
the "quick subset" toggle for a fast demo run), then try **Predict**.

### Run the tests

```bash
pytest tests/ -v
```

## 📁 Project Structure

```
Class 1 Project/
├── app.py                     # Streamlit entry point (UI only, no TF calls)
├── config.py                  # Dataclass-based configuration (paths, hyperparameters)
├── requirements.txt
├── README.md
├── notebooks/
│   └── 01_eda.ipynb           # Class distribution, sample grid, pixel histogram
├── src/
│   ├── data/
│   │   ├── loader.py          # Loads keras MNIST, normalizes, splits train/val/test
│   │   └── preprocessor.py    # Canvas/upload image -> model-ready (1,28,28,1) tensor
│   ├── models/
│   │   ├── cnn.py             # build_baseline_cnn() — the CNN architecture
│   │   └── registry.py        # Save/load/version trained checkpoints
│   ├── training/
│   │   └── trainer.py         # Orchestrates fit -> evaluate -> persist
│   └── utils/
│       └── logger.py          # Shared structured logging
├── tests/
│   └── test_pipeline.py       # Shape/range/forward-pass sanity tests
└── data/
    ├── models/                # Saved .keras checkpoints + index.json manifest
    └── history/                # training_history.json per run
```

## 🔧 Architecture

```
┌───────────────────┐
│  Draw on Canvas /  │
│  Upload Image      │
└─────────┬──────────┘
          │
          ▼
┌─────────────────────────┐
│  DigitPreprocessor       │   grayscale → resize 28x28 → normalize [0,1]
│  (src/data/preprocessor) │   → reshape (1, 28, 28, 1)
└─────────┬────────────────┘
          │
          ▼
┌───────────────────────────────────────────────────────────┐
│                     Baseline CNN                            │
│  Conv2D(32,3x3)+ReLU → MaxPool(2x2)                          │
│  → Conv2D(64,3x3)+ReLU → MaxPool(2x2)                        │
│  → Flatten → Dense(128)+ReLU → Dropout(0.5) → Dense(10)     │
│  (src/models/cnn.py)                                         │
└─────────┬─────────────────────────────────────────────────┘
          │
          ▼
┌───────────────────────────┐
│  Softmax Probabilities      │  10 values, one per digit
└─────────┬───────────────────┘
          │
          ▼
┌───────────────────────────┐
│  Predicted Digit +          │
│  Confidence Bar Chart       │
└───────────────────────────┘
```

## ⚙️ Configuration

All hyperparameters and paths live in `config.py` as frozen dataclasses. Key values:

| Setting | Default | Description |
|---|---|---|
| `data.image_size` | `(28, 28)` | Input image dimensions |
| `data.num_channels` | `1` | Grayscale |
| `data.val_split` | `0.1` | Fraction of training data held out for validation |
| `data.quick_subset_fraction` | `0.1` | Subset size used by the "quick demo" training toggle |
| `model.conv1_filters` / `conv2_filters` | `32` / `64` | Conv layer filter counts |
| `model.dense_units` | `128` | Hidden dense layer width |
| `model.dropout_rate` | `0.5` | Dropout before the output layer |
| `model.learning_rate` | `1e-3` | Adam optimizer learning rate |
| `training.epochs` | `10` | Full training run length |
| `training.quick_epochs` | `3` | Quick-demo training run length |
| `training.batch_size` | `128` | Mini-batch size |
| `training.early_stopping_patience` | `3` | Epochs without `val_accuracy` improvement before stopping |

## 🧠 Model Architecture

This is the **baseline** CNN taught in the Class 1 lecture — deliberately
simple so the fundamentals (convolution, pooling, dropout) stay visible.
Advanced optimization (batch normalization, data augmentation, learning-rate
schedules) is the subject of Class 2 and is intentionally not applied here.

```
Input (28, 28, 1)
  → Conv2D(32, 3x3, ReLU)      # 26x26x32
  → MaxPooling2D(2x2)          # 13x13x32
  → Conv2D(64, 3x3, ReLU)      # 11x11x64
  → MaxPooling2D(2x2)          #  5x5x64
  → Flatten                    #  1600
  → Dense(128, ReLU)
  → Dropout(0.5)
  → Dense(10, Softmax)
```

- **Optimizer:** Adam (`lr=1e-3`)
- **Loss:** `sparse_categorical_crossentropy` (integer labels, no one-hot needed)
- **Expected performance:** ~99% test accuracy after a full training run

## 🛠️ Development

### Adding a new architecture variant

Add a new builder function alongside `build_baseline_cnn()` in
`src/models/cnn.py`, and register it wherever the app or trainer selects a
model — the rest of the pipeline (data, training, registry, app) is
architecture-agnostic.

### Model versioning

`src/models/registry.py` saves every trained model as
`data/models/mnist_cnn_<timestamp>-acc<value>.keras` and tracks all versions
in `data/models/index.json`. `ModelRegistry.load_best()` always returns the
highest test-accuracy version, so the Streamlit app keeps working across
restarts.

## 📜 License

MIT License — feel free to use in your own projects!
