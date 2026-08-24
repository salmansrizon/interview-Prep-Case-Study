"""
Train the house price model and write every artifact the app reads.

Run this when you change the data, or when Streamlit warns that the saved model
was pickled by a different scikit-learn version:

    python train_model.py

Writes into models/:
    linear_regression.pkl   trained LinearRegression (raw, unscaled features)
    scaler.pkl              StandardScaler fitted on the training features
    feature_names.json      column order the model expects
    feature_importance.csv  coefficients, sorted by absolute value
    model_results.json      test-set metrics for Linear, Ridge and Lasso
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "housing_data.csv"
MODELS_DIR = ROOT / "models"
RANDOM_STATE = 42


def evaluate(model, X_test, y_test):
    """Test-set scores for one fitted model."""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
        "mape": float(np.mean(np.abs((y_test - y_pred) / y_test)) * 100),
    }


def main():
    df = pd.read_csv(DATA_FILE)
    feature_names = [c for c in df.columns if c != "price"]
    X, y = df[feature_names], df["price"]

    # The same split the Model Evaluation page reproduces, so the numbers match.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # The scaler is fitted on training data only — never on the test set, or the
    # model gets a peek at data it is supposed to be judged on.
    scaler = StandardScaler().fit(X_train)

    # Ridge and Lasso are penalised per coefficient, so they need scaled inputs to
    # be judged fairly. Plain LinearRegression does not care, and keeping it on raw
    # features is what lets the Predict page feed it human numbers directly.
    models = {
        "linear": LinearRegression().fit(X_train, y_train),
        "ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE).fit(
            scaler.transform(X_train), y_train
        ),
        "lasso": Lasso(alpha=1.0, random_state=RANDOM_STATE, max_iter=10000).fit(
            scaler.transform(X_train), y_train
        ),
    }

    results = {
        "linear": evaluate(models["linear"], X_test, y_test),
        "ridge": evaluate(models["ridge"], scaler.transform(X_test), y_test),
        "lasso": evaluate(models["lasso"], scaler.transform(X_test), y_test),
    }

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(models["linear"], MODELS_DIR / "linear_regression.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    (MODELS_DIR / "feature_names.json").write_text(json.dumps(feature_names))
    (MODELS_DIR / "model_results.json").write_text(json.dumps(results, indent=2))

    coef = pd.DataFrame({
        "feature": feature_names,
        "coefficient": models["linear"].coef_,
        "abs_coefficient": np.abs(models["linear"].coef_),
    }).sort_values("abs_coefficient", ascending=False)
    coef.to_csv(MODELS_DIR / "feature_importance.csv", index=False)

    for name, scores in results.items():
        print(f"{name:8} R²={scores['r2']:.4f}  RMSE=${scores['rmse']:,.0f}  "
              f"MAE=${scores['mae']:,.0f}  MAPE={scores['mape']:.2f}%")
    print(f"\nArtifacts written to {MODELS_DIR}")


if __name__ == "__main__":
    main()
