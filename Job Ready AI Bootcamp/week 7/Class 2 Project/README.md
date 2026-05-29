# Industrial Equipment Success Score Predictor

A production-grade deep learning application built with **TensorFlow/Keras** and **Streamlit** to predict a "Success Score" (0-100) for industrial equipment based on operational metrics.

## Architecture

```
industrial_equipment_predictor/
├── config.yaml              # Centralized hyperparameters & paths
├── src/
│   ├── data/                # Data generation, loading, preprocessing
│   ├── models/              # Model builder, trainer, evaluator
│   └── utils/               # Logging, helpers, constants
├── app/                     # Streamlit application
│   ├── main.py              # Entry point
│   └── pages/               # Multi-page app views
├── data/
│   ├── raw/                 # Generated synthetic data
│   ├── processed/           # Scaled, split datasets
│   └── models/              # Saved .keras / .h5 models
├── notebooks/               # EDA & experimentation
└── tests/                   # Unit tests
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Data
```bash
python -m src.data.generator
```

### 3. Train Model
```bash
python -m src.models.trainer
```

### 4. Launch Streamlit App
```bash
streamlit run app/main.py
```

## Features

- **Synthetic Data Generation** — Realistic industrial equipment features
- **Preprocessing Pipeline** — Scaling, encoding, train/val/test splits
- **Deep Learning Model** — TensorFlow/Keras sequential model with dropout & batch normalization
- **Training Dashboard** — TensorBoard + early stopping + model checkpointing
- **Streamlit UI** — Data exploration, model training, live prediction, analytics dashboard
- **Docker Support** — Containerized deployment ready

## Target Variable

**Success Score** (0–100): A composite metric representing equipment reliability, efficiency, and maintenance health.

## License

MIT
