"""
Data Preprocessing Pipeline.

Handles scaling, encoding, missing value imputation, and artifact persistence.
Produces train/validation/test splits ready for TensorFlow/Keras.
"""

import os
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.config import get_config
from src.utils import logger
from src.utils.helpers import save_joblib, load_joblib


class DataPreprocessor:
    """
    End-to-end preprocessing pipeline for industrial equipment data.

    Steps:
        1. Separate features and target
        2. Impute missing values
        3. Scale numeric features
        4. One-hot encode categorical features
        5. Persist transformers for inference
    """

    def __init__(self):
        self.config = get_config()
        self.numeric_features = self.config.data.feature_columns
        self.categorical_features = self.config.data.categorical_columns
        self.target_column = self.config.data.target_column

        self.numeric_transformer = None
        self.categorical_transformer = None
        self.preprocessor = None
        self.target_scaler = None

        self.feature_names_after_transform: List[str] = []
        self._is_fitted = False

    def _build_preprocessor(self) -> ColumnTransformer:
        """Build sklearn preprocessing pipeline."""
        scaler_cls = StandardScaler if self.config.preprocessing.numeric_scaler == "standard" else MinMaxScaler

        self.numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy=self.config.preprocessing.handle_missing)),
            ("scaler", scaler_cls()),
        ])

        self.categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", self.numeric_transformer, self.numeric_features),
                ("cat", self.categorical_transformer, self.categorical_features),
            ],
            remainder="drop",
        )

        return preprocessor

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit preprocessing pipeline and transform data.

        Returns:
            X: Preprocessed feature matrix (numpy array)
            y: Target vector (numpy array)
        """
        logger.info("Fitting preprocessing pipeline on {} rows...", len(df))

        # Separate target
        y = df[self.target_column].values.reshape(-1, 1)
        X_df = df[self.numeric_features + self.categorical_features].copy()

        # Build and fit preprocessor
        self.preprocessor = self._build_preprocessor()
        X = self.preprocessor.fit_transform(X_df)

        # Extract feature names
        num_names = self.numeric_features
        cat_names = list(
            self.preprocessor.named_transformers_["cat"]
            .named_steps["onehot"]
            .get_feature_names_out(self.categorical_features)
        )
        self.feature_names_after_transform = num_names + cat_names

        # Scale target (optional, helps neural network convergence)
        self.target_scaler = StandardScaler()
        y_scaled = self.target_scaler.fit_transform(y).flatten()

        self._is_fitted = True
        logger.info(
            "Preprocessing complete. Features: {} ({} numeric + {} categorical)",
            len(self.feature_names_after_transform),
            len(num_names),
            len(cat_names),
        )

        return X, y_scaled

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Transform new data using fitted pipeline."""
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform(). Call fit_transform() first.")

        y = df[self.target_column].values.reshape(-1, 1) if self.target_column in df.columns else None
        X_df = df[self.numeric_features + self.categorical_features].copy()
        X = self.preprocessor.transform(X_df)

        if y is not None and self.target_scaler is not None:
            y = self.target_scaler.transform(y).flatten()

        return X, y

    def inverse_transform_target(self, y_scaled: np.ndarray) -> np.ndarray:
        """Convert scaled target back to original scale (0-100)."""
        if self.target_scaler is None:
            return y_scaled
        return self.target_scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()

    def save(self, path: str) -> None:
        """Save fitted preprocessor and target scaler."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        artifact = {
            "preprocessor": self.preprocessor,
            "target_scaler": self.target_scaler,
            "feature_names": self.feature_names_after_transform,
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "target_column": self.target_column,
        }
        save_joblib(artifact, path)
        logger.info("Preprocessor artifacts saved to {}", path)

    def load(self, path: str) -> None:
        """Load fitted preprocessor and target scaler."""
        artifact = load_joblib(path)
        self.preprocessor = artifact["preprocessor"]
        self.target_scaler = artifact["target_scaler"]
        self.feature_names_after_transform = artifact["feature_names"]
        self.numeric_features = artifact["numeric_features"]
        self.categorical_features = artifact["categorical_features"]
        self.target_column = artifact["target_column"]
        self._is_fitted = True
        logger.info("Preprocessor artifacts loaded from {}", path)


def preprocess_pipeline(raw_csv_path: str, output_dir: str) -> Dict[str, Any]:
    """
    Full preprocessing pipeline: load raw data, preprocess, split, save artifacts.

    Returns dict with paths to saved files and dataset shapes.
    """
    from sklearn.model_selection import train_test_split

    config = get_config()
    os.makedirs(output_dir, exist_ok=True)

    # Load raw data
    logger.info("Loading raw data from {}", raw_csv_path)
    df = pd.read_csv(raw_csv_path)
    logger.info("Loaded {} rows, {} columns", len(df), len(df.columns))

    # Preprocess
    preprocessor = DataPreprocessor()
    X, y = preprocessor.fit_transform(df)

    # Split: train / val / test
    test_size = config.data.test_size
    val_size = config.data.val_size

    # First split: train+val vs test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=config.project.random_seed
    )

    # Second split: train vs val
    val_relative_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_relative_size,
        random_state=config.project.random_seed
    )

    logger.info(
        "Split complete — Train: {}, Val: {}, Test: {}",
        len(X_train), len(X_val), len(X_test),
    )

    # Save arrays
    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "X_val.npy"), X_val)
    np.save(os.path.join(output_dir, "X_test.npy"), X_test)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "y_val.npy"), y_val)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test)

    # Save preprocessor
    preprocessor.save(os.path.join(output_dir, "preprocessor.joblib"))

    # Save feature names
    pd.DataFrame({"feature": preprocessor.feature_names_after_transform}).to_csv(
        os.path.join(output_dir, "feature_names.csv"), index=False
    )

    result = {
        "X_train_shape": X_train.shape,
        "X_val_shape": X_val.shape,
        "X_test_shape": X_test.shape,
        "output_dir": output_dir,
        "preprocessor_path": os.path.join(output_dir, "preprocessor.joblib"),
        "feature_names": preprocessor.feature_names_after_transform,
    }

    logger.info("Preprocessing pipeline complete. Artifacts saved to {}", output_dir)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run preprocessing pipeline")
    parser.add_argument("--input", type=str, required=True, help="Path to raw CSV")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    config = get_config()
    output = args.output or config.paths.data_processed

    result = preprocess_pipeline(args.input, output)
    print("Preprocessing result:", result)
