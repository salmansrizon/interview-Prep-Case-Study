"""
Data loading and preprocessing utilities for Loan Approval Predictor.
"""
import pandas as pd
import numpy as np
import streamlit as st

@st.cache_data
def load_loan_data():
    """Load and preprocess loan dataset."""
    df = pd.read_csv("data/loan_data.csv")
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
