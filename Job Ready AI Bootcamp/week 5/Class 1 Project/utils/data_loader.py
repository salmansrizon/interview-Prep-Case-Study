"""
Data loading and preprocessing utilities.
"""
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st

# Paths are resolved from this file, not the working directory, so the app
# runs the same whether you launch it from here or from the repo root.
ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "housing_data.csv"

@st.cache_data
def load_housing_data():
    """Load and preprocess housing dataset."""
    df = pd.read_csv(DATA_FILE)
    return df

@st.cache_data
def get_feature_stats(df):
    """Get descriptive statistics for all features."""
    return df.describe().T

def get_feature_ranges(df):
    """Get min/max ranges for input features."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    ranges = {}
    for col in numeric_cols:
        if col != 'price':
            ranges[col] = {
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'mean': float(df[col].mean()),
                'median': float(df[col].median()),
                'std': float(df[col].std())
            }
    return ranges
