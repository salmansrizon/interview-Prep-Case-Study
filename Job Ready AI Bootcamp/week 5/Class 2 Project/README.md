# 🏦 Loan Approval Predictor — Production-Grade Classification Dashboard

A complete Streamlit application for predicting loan approvals using three supervised
learning algorithms, with full explainability for every decision.

## 📚 Learning Objectives

| Topic | Description | Page |
|-------|-------------|------|
| **Logistic Regression** | Probability-based binary classification with sigmoid | 📐 Algorithm Theory |
| **Decision Trees** | Rule-based splits with Gini impurity | 🌳 Algorithm Theory |
| **Random Forests** | Ensemble of trees with bagging | 🌲 Algorithm Theory |
| **Model Comparison** | Side-by-side evaluation with confusion matrices & ROC | ⚖️ Compare Models |
| **Feature Importance** | What drives approval/denial decisions | 📊 Feature Importance |
| **Explainable AI** | Human-readable explanations for every prediction | 🔮 Predict Approval |

## 🏗️ Project Structure

```
loan_approval_predictor/
├── app.py                          # Main entry point
├── requirements.txt                # Dependencies
├── README.md                       # Documentation
├── data/
│   └── loan_data.csv               # 8,000 synthetic loan applications
├── models/
│   ├── logistic_regression.pkl     # Probability model
│   ├── decision_tree.pkl           # Rule-based model
│   ├── random_forest.pkl           # Ensemble model (100 trees)
│   ├── scaler.pkl                  # Feature scaler
│   ├── label_encoders.pkl          # Categorical encoders
│   ├── feature_importance.csv      # RF feature importance
│   ├── logistic_coefficients.csv   # LR coefficients
│   ├── model_results.json          # Evaluation metrics
│   └── cm_*.npy                    # Confusion matrices
├── utils/
│   ├── data_loader.py              # Data I/O
│   ├── model_utils.py              # Prediction & explainability
│   └── visualizations.py           # Chart utilities
└── pages/
    ├── 1_Explore_Data.py           # EDA & distributions
    ├── 2_Algorithm_Theory.py       # How LR, DT, RF work
    ├── 3_Model_Comparison.py       # Side-by-side metrics
    ├── 4_Feature_Importance.py     # What drives decisions
    └── 5_Predict_Approval.py       # Live prediction with explanation
```

## 🚀 Quick Start

```bash
cd loan_approval_predictor
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## 🎯 Features

### 🔮 Live Loan Prediction
- Input 13 application features via interactive form
- Choose from 3 algorithms (Logistic Regression, Decision Tree, Random Forest)
- Get instant APPROVE/DENY decision with probability
- See **human-readable explanation** of why the decision was made
- Receive **actionable recommendations** if denied

### 📐 Educational Modules
- **Interactive sigmoid demo** — adjust z, watch probability change
- **Decision Tree concepts** — Gini impurity, information gain, max depth
- **Random Forest ensemble** — bagging, voting, hyperparameters
- **Fairness check** — verify protected attributes don't bias decisions

### 📊 Model Evaluation
- Confusion matrices for all 3 models
- ROC curves with AUC scores
- Radar chart comparison
- Business impact analysis (profit/loss estimation)

## 📈 Model Performance

| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|----------|-----------|--------|-----|-----|
| Logistic Regression | ~0.78 | ~0.76 | ~0.82 | ~0.79 | ~0.85 |
| Decision Tree | ~0.76 | ~0.74 | ~0.80 | ~0.77 | ~0.82 |
| Random Forest | ~0.82 | ~0.80 | ~0.85 | ~0.82 | ~0.89 |

## 🛠️ Tech Stack

- **Streamlit** — Web app framework
- **Scikit-Learn** — LogisticRegression, DecisionTree, RandomForest
- **Pandas/NumPy** — Data manipulation
- **Matplotlib/Seaborn** — Static visualizations
- **Plotly** — Interactive charts

## 📝 Dataset Features

| Feature | Description | Type |
|---------|-------------|------|
| `gender` | Male / Female | Categorical |
| `married` | Yes / No | Categorical |
| `dependents` | 0, 1, 2, 3+ | Categorical |
| `education` | Graduate / Not Graduate | Categorical |
| `self_employed` | Yes / No | Categorical |
| `applicant_income` | Annual income ($) | Numeric |
| `coapplicant_income` | Co-applicant income ($) | Numeric |
| `loan_amount` | Requested loan amount ($) | Numeric |
| `loan_term_months` | Repayment period (12-360) | Numeric |
| `credit_history` | Has credit history (1/0) | Binary |
| `property_area` | Urban / Semiurban / Rural | Categorical |
| `credit_score` | FICO score (300-850) | Numeric |
| `age` | Applicant age (21-70) | Numeric |

## 👥 Authors

Data Science Education Team
