"""
Tests for data preprocessing pipeline.
"""

import os
import pytest
import numpy as np
import pandas as pd

from src.data.generator import generate_equipment_data
from src.data.preprocessor import DataPreprocessor, preprocess_pipeline
from src.config import get_config


class TestPreprocessor:
    """Test suite for preprocessing."""

    @pytest.fixture
    def sample_df(self):
        """Generate a small sample dataset."""
        return generate_equipment_data(n_samples=200, seed=42)

    @pytest.fixture
    def preprocessor(self):
        """Create a fresh preprocessor instance."""
        return DataPreprocessor()

    def test_fit_transform_returns_arrays(self, sample_df, preprocessor):
        """fit_transform should return numpy arrays."""
        X, y = preprocessor.fit_transform(sample_df)
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.ndim == 2
        assert y.ndim == 1

    def test_feature_count_expanded(self, sample_df, preprocessor):
        """One-hot encoding should expand feature count."""
        X, _ = preprocessor.fit_transform(sample_df)
        # 10 numeric + (6 equip types + 6 manufacturers + 4 facilities) = 26
        assert X.shape[1] >= 10 + 3  # at least original features

    def test_target_is_scaled(self, sample_df, preprocessor):
        """Target should be approximately standard-normal after scaling."""
        _, y = preprocessor.fit_transform(sample_df)
        assert abs(y.mean()) < 0.5  # roughly centered
        assert 0.5 < y.std() < 2.0  # roughly unit variance

    def test_transform_without_fit_raises(self, sample_df, preprocessor):
        """Calling transform before fit_transform should raise."""
        with pytest.raises(RuntimeError, match="fitted"):
            preprocessor.transform(sample_df)

    def test_inverse_transform_target(self, sample_df, preprocessor):
        """Inverse transform should restore approximate original scale."""
        _, y_scaled = preprocessor.fit_transform(sample_df)
        y_orig = preprocessor.inverse_transform_target(y_scaled)
        original_scores = sample_df["success_score"].values
        np.testing.assert_allclose(y_orig, original_scores, rtol=0.01)

    def test_save_and_load(self, sample_df, preprocessor, tmp_path):
        """Preprocessor should be saveable and loadable."""
        X1, y1 = preprocessor.fit_transform(sample_df)

        save_path = os.path.join(tmp_path, "preprocessor.joblib")
        preprocessor.save(save_path)
        assert os.path.exists(save_path)

        loaded = DataPreprocessor()
        loaded.load(save_path)

        X2, y2 = loaded.transform(sample_df)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_preprocess_pipeline_creates_files(self, sample_df, tmp_path):
        """Full pipeline should create all expected output files."""
        raw_path = os.path.join(tmp_path, "raw.csv")
        sample_df.to_csv(raw_path, index=False)

        output_dir = os.path.join(tmp_path, "processed")
        result = preprocess_pipeline(raw_path, output_dir)

        expected_files = [
            "X_train.npy", "X_val.npy", "X_test.npy",
            "y_train.npy", "y_val.npy", "y_test.npy",
            "preprocessor.joblib", "feature_names.csv",
        ]
        for fname in expected_files:
            assert os.path.exists(os.path.join(output_dir, fname)), f"Missing: {fname}"

        assert result["X_train_shape"][0] > 0
        assert result["X_test_shape"][0] > 0

    def test_splits_sum_correctly(self, sample_df, tmp_path):
        """Train + val + test should equal total rows."""
        raw_path = os.path.join(tmp_path, "raw.csv")
        sample_df.to_csv(raw_path, index=False)

        output_dir = os.path.join(tmp_path, "processed")
        result = preprocess_pipeline(raw_path, output_dir)

        total = (result["X_train_shape"][0] + 
                 result["X_val_shape"][0] + 
                 result["X_test_shape"][0])
        assert total == len(sample_df)
