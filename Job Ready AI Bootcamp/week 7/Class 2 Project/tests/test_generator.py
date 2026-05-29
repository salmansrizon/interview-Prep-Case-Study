"""
Tests for synthetic data generator.
"""

import pytest
import pandas as pd
import numpy as np

from src.data.generator import generate_equipment_data


class TestDataGenerator:
    """Test suite for data generation."""

    def test_generate_returns_dataframe(self):
        """Generator should return a pandas DataFrame."""
        df = generate_equipment_data(n_samples=100, seed=42)
        assert isinstance(df, pd.DataFrame)

    def test_correct_number_of_rows(self):
        """Generated DataFrame should have requested number of rows."""
        n = 250
        df = generate_equipment_data(n_samples=n, seed=42)
        assert len(df) == n

    def test_all_required_columns_present(self):
        """All expected columns should exist."""
        df = generate_equipment_data(n_samples=50, seed=42)
        required = [
            "equipment_id", "equipment_type", "manufacturer", "facility_location",
            "operating_temperature", "vibration_level", "pressure_reading",
            "power_consumption", "runtime_hours", "days_since_maintenance",
            "error_count_24h", "oil_quality_index", "load_factor",
            "ambient_temperature", "success_score",
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_success_score_range(self):
        """Success score must be within [0, 100]."""
        df = generate_equipment_data(n_samples=1000, seed=42)
        assert df["success_score"].min() >= 0
        assert df["success_score"].max() <= 100

    def test_equipment_types_valid(self):
        """Equipment types should be from expected set."""
        valid_types = {"Pump", "Compressor", "Turbine", "Motor", "Generator", "Heat_Exchanger"}
        df = generate_equipment_data(n_samples=100, seed=42)
        assert set(df["equipment_type"].unique()).issubset(valid_types)

    def test_numeric_features_positive(self):
        """Numeric features that should be positive are indeed positive."""
        df = generate_equipment_data(n_samples=100, seed=42)
        positive_cols = ["vibration_level", "runtime_hours", "power_consumption"]
        for col in positive_cols:
            assert (df[col] >= 0).all(), f"Column {col} has negative values"

    def test_reproducibility(self):
        """Same seed should produce identical data."""
        df1 = generate_equipment_data(n_samples=100, seed=123)
        df2 = generate_equipment_data(n_samples=100, seed=123)
        pd.testing.assert_frame_equal(df1, df2)

    def test_equipment_id_format(self):
        """Equipment IDs should follow EQ_XXXXX format."""
        df = generate_equipment_data(n_samples=10, seed=42)
        for eq_id in df["equipment_id"]:
            assert eq_id.startswith("EQ_")
            assert len(eq_id) == 8  # EQ_ + 5 digits
            assert eq_id[3:].isdigit()
