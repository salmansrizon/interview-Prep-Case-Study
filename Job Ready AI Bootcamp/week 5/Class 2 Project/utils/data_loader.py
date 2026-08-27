"""
Data loading and preprocessing utilities for Loan Approval Predictor.
"""
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st

# Anchored to this file, not the working directory, so the app runs from anywhere.
ROOT = Path(__file__).resolve().parents[1]

@st.cache_data
def load_loan_data():
    """Load and preprocess loan dataset."""
    df = pd.read_csv(ROOT / "data" / "loan_data.csv")
    return df

@st.cache_data
def get_feature_stats(df):
    """Get descriptive statistics for all features."""
    return df.describe(include='all').T

def get_categorical_options(df):
    """Get unique values for categorical features."""
    cat_cols = df.select_dtypes(include=['object']).columns
    options = {}
    for col in cat_cols:
        if col != 'applicant_id':
            options[col] = df[col].unique().tolist()
    return options
