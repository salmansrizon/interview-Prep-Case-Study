# Industrial Equipment Success Score Predictor

## সহজ ভাষায় Project Overview

**Industrial Equipment Success Score Predictor** project-এ lecture-এর theory-কে working software বা executable notebook-এ convert করা হয়েছে। লক্ষ্য শুধু final output দেখা নয়; input থেকে preprocessing, core logic/model, evaluation এবং output—পুরো pipeline বোঝা।

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
