"""
Generate the loan dataset and train every model the app reads.

Run this when you change the generator, or when Streamlit warns that the saved
models were pickled by a different scikit-learn version:

    python train_model.py

Writes data/loan_data.csv and, into models/:
    logistic_regression.pkl / decision_tree.pkl / random_forest.pkl
    scaler.pkl               StandardScaler fitted on the training features
    label_encoders.pkl       one LabelEncoder per categorical column
    feature_names.json       column order every model expects
    feature_importance.csv   Random Forest Gini importance
    logistic_coefficients.csv  log-odds weights, sorted by absolute value
    cm_*.npy                 test-set confusion matrices
    roc_curves.npz           real fpr/tpr points for the ROC page
    model_results.json       test-set metrics for all three models
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "loan_data.csv"
MODELS_DIR = ROOT / "models"
RANDOM_STATE = 42
N_APPLICATIONS = 8000

CATEGORICAL = ["gender", "married", "dependents", "education",
               "self_employed", "property_area"]


def generate_applications(n=N_APPLICATIONS, seed=RANDOM_STATE):
    """Synthetic loan book where approval genuinely depends on the features.

    Every applicant gets a true approval probability from a logistic rule that a
    credit officer would recognise — score, credit history and debt-to-income do
    the work; gender and marital status are deliberately given zero weight so
    the Fairness Check page has something honest to report.
    """
    rng = np.random.default_rng(seed)

    credit_score = np.clip(rng.normal(680, 85, n), 300, 850).round()
    # Thin files cluster at the low-score end, exactly as they do in a real book.
    credit_history = rng.binomial(1, np.clip((credit_score - 380) / 380, 0.05, 0.95))

    applicant_income = np.clip(rng.lognormal(10.6, 0.55, n), 12_000, 500_000)
    has_coapplicant = rng.binomial(1, 0.42, n)
    coapplicant_income = has_coapplicant * np.clip(rng.lognormal(10.1, 0.6, n), 0, 400_000)
    household_income = applicant_income + coapplicant_income

    loan_term_months = rng.choice([120, 180, 240, 300, 360], n, p=[.08, .17, .25, .2, .3])
    # Requested amount tracks income, with enough spread to create real DTI risk.
    loan_amount = np.clip(household_income * rng.lognormal(1.45, 0.45, n), 5_000, 500_000)

    monthly_payment = loan_amount / loan_term_months
    dti = monthly_payment / (household_income / 12)

    education = rng.choice(["Graduate", "Not Graduate"], n, p=[0.72, 0.28])
    self_employed = rng.choice(["No", "Yes"], n, p=[0.83, 0.17])
    dependents = rng.choice(["0", "1", "2", "3+"], n, p=[0.4, 0.28, 0.2, 0.12])
    property_area = rng.choice(["Urban", "Semiurban", "Rural"], n, p=[0.4, 0.35, 0.25])
    gender = rng.choice(["Male", "Female"], n, p=[0.52, 0.48])
    married = rng.choice(["Yes", "No"], n, p=[0.62, 0.38])
    age = rng.integers(21, 70, n)

    logit = (
        -2.15
        + 3.0 * (credit_score - 665) / 85          # the dominant factor
        + 1.6 * credit_history                      # a thin file is expensive
        - 3.2 * np.clip(dti - 0.30, -0.30, 0.60) / 0.15  # affordability
        + 0.45 * (education == "Graduate")
        - 0.60 * (self_employed == "Yes")
        - 0.18 * np.array([{"0": 0, "1": 1, "2": 2, "3+": 3}[d] for d in dependents])
        + 0.30 * has_coapplicant
        + 0.20 * (property_area == "Urban")
        - 0.15 * (property_area == "Rural")
        + 0.012 * (age - 45) - 0.0009 * (age - 45) ** 2   # prime earning years
    )
    # gender and married are absent on purpose — see the Fairness Check page.

    approval_probability = 1 / (1 + np.exp(-logit))
    loan_approved = rng.binomial(1, approval_probability)

    return pd.DataFrame({
        "applicant_id": [f"APP-{i:06d}" for i in range(1, n + 1)],
        "gender": gender,
        "married": married,
        "dependents": dependents,
        "education": education,
        "self_employed": self_employed,
        "applicant_income": applicant_income.round(2),
        "coapplicant_income": coapplicant_income.round(2),
        "loan_amount": loan_amount.round(2),
        "loan_term_months": loan_term_months,
        "credit_history": credit_history,
        "property_area": property_area,
        "credit_score": credit_score.astype(int),
        "age": age,
        "loan_approved": loan_approved,
        "approval_probability": approval_probability.round(4),
    })


def evaluate(y_test, y_pred, y_prob):
    """Test-set scores for one fitted classifier."""
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }


def main():
    df = generate_applications()
    DATA_FILE.parent.mkdir(exist_ok=True)
    df.to_csv(DATA_FILE, index=False)

    feature_names = [c for c in df.columns
                     if c not in ("applicant_id", "loan_approved", "approval_probability")]

    # One encoder per categorical column, saved so the app encodes user input the
    # same way the model was trained. Encoding on the full frame is safe here:
    # it only learns the category vocabulary, not the labels.
    X = df[feature_names].copy()
    label_encoders = {}
    for col in CATEGORICAL:
        enc = LabelEncoder().fit(X[col].astype(str))
        X[col] = enc.transform(X[col].astype(str))
        label_encoders[col] = enc

    y = df["loan_approved"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Fitted on training data only — the test set stays unseen.
    scaler = StandardScaler().fit(X_train)

    def scaled(frame):
        """Scaled copy that keeps its column names, so predictions never warn."""
        return pd.DataFrame(scaler.transform(frame), columns=frame.columns, index=frame.index)

    # Logistic Regression is the only one that needs scaling; trees split on raw
    # thresholds, so scaling them would change nothing but the readability.
    log_reg = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    log_reg.fit(scaled(X_train), y_train)

    tree = DecisionTreeClassifier(max_depth=5, min_samples_split=40,
                                  min_samples_leaf=20, random_state=RANDOM_STATE)
    tree.fit(X_train, y_train)

    forest = RandomForestClassifier(n_estimators=200, max_depth=10,
                                    min_samples_split=20, min_samples_leaf=10,
                                    random_state=RANDOM_STATE, n_jobs=-1)
    forest.fit(X_train, y_train)

    fitted = {
        "logistic_regression": (log_reg, scaled(X_test)),
        "decision_tree": (tree, X_test),
        "random_forest": (forest, X_test),
    }

    results, curves = {}, {}
    MODELS_DIR.mkdir(exist_ok=True)
    for name, (model, X_eval) in fitted.items():
        y_pred = model.predict(X_eval)
        y_prob = model.predict_proba(X_eval)[:, 1]
        results[name] = evaluate(y_test, y_pred, y_prob)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        curves[f"{name}_fpr"], curves[f"{name}_tpr"] = fpr, tpr
        joblib.dump(model, MODELS_DIR / f"{name}.pkl")

    # The app's confusion-matrix page reads these three file names.
    np.save(MODELS_DIR / "cm_logistic.npy", confusion_matrix(y_test, log_reg.predict(scaled(X_test))))
    np.save(MODELS_DIR / "cm_decision_tree.npy", confusion_matrix(y_test, tree.predict(X_test)))
    np.save(MODELS_DIR / "cm_random_forest.npy", confusion_matrix(y_test, forest.predict(X_test)))
    np.savez(MODELS_DIR / "roc_curves.npz", **curves)

    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(label_encoders, MODELS_DIR / "label_encoders.pkl")
    (MODELS_DIR / "feature_names.json").write_text(json.dumps(feature_names))
    (MODELS_DIR / "model_results.json").write_text(json.dumps(results, indent=2))

    pd.DataFrame({
        "feature": feature_names,
        "importance": forest.feature_importances_,
    }).sort_values("importance", ascending=False).to_csv(
        MODELS_DIR / "feature_importance.csv", index=False)

    pd.DataFrame({
        "feature": feature_names,
        "coefficient": log_reg.coef_[0],
        "abs_coefficient": np.abs(log_reg.coef_[0]),
    }).sort_values("abs_coefficient", ascending=False).to_csv(
        MODELS_DIR / "logistic_coefficients.csv", index=False)

    print(f"{len(df):,} applications written to {DATA_FILE.name} "
          f"({y.mean() * 100:.1f}% approved)")
    # The ceiling any model could reach on this data — labels are sampled from
    # these probabilities, so the noise above it is irreducible.
    print(f"{'bayes ceiling':<20} AUC={roc_auc_score(df.loan_approved, df.approval_probability):.4f}")
    for name, scores in results.items():
        print(f"{name:<20} acc={scores['accuracy']:.4f}  prec={scores['precision']:.4f}  "
              f"rec={scores['recall']:.4f}  f1={scores['f1']:.4f}  AUC={scores['roc_auc']:.4f}")
    print(f"\nArtifacts written to {MODELS_DIR}")


if __name__ == "__main__":
    main()
