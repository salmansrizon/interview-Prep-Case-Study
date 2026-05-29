# 🛡️ Spam & Intent Classifier

A production-grade, 100% offline text classification system built with
Streamlit and scikit-learn for **Class 11 — Module 3: Advanced Classification Models**.

## Algorithms Covered
1. **Naive Bayes** — Probabilistic classifier based on Bayes' theorem
2. **Support Vector Machine (SVM)** — Maximal margin hyperplane classifier
3. **K-Nearest Neighbors (KNN)** — Instance-based lazy learner

## Quick Start

```bash
# 1. Clone / create project folder
cd spam-intent-classifier

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app
streamlit run app.py
```

## Usage
1. **Data Tab**: Generate synthetic data or upload your own CSV (`text`, `label` columns).
2. **Train Tab**: Select an algorithm, tune hyperparameters via the sidebar, and train.
3. **Predict Tab**: Type any message to see real-time classification & confidence scores.
4. **Compare Tab**: Train all three models and compare their metrics side-by-side.

## Project Highlights
- **Modular Architecture**: Separate modules for data, features, models, and training.
- **Type Hints & Docstrings**: Production-readable code.
- **Abstract Base Class**: All models share a unified interface (`train`, `predict`, `evaluate`, `save`, `load`).
- **Persistent Artifacts**: Trained models and vectorizers are saved to `data/models/` via `joblib`.
- **Educational UI**: Inline explanations of Accuracy, Precision, Recall, and F1.
