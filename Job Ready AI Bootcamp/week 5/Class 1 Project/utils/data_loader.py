"""
Data loading and preprocessing utilities.
"""
import pandas as pd
import numpy as np
import streamlit as st

@st.cache_data
def load_housing_data():
    """Load and preprocess housing dataset."""
    df = pd.read_csv("data/housing_data.csv")
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
