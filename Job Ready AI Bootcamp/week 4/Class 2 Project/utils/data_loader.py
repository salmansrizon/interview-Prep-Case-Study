"""
Data loading and preprocessing utilities for TechNova Analytics.
This module handles loading CSV files and performing initial data transformations.
"""
# Pandas library for data manipulation, analysis, and DataFrame operations
import pandas as pd
# NumPy library for numerical computations and array operations
import numpy as np
# DateTime module for parsing and manipulating date columns
from datetime import datetime
# Streamlit caching decorator to avoid reloading data on every rerun
import streamlit as st

# STREAMLIT CACHE: Avoid reloading data on every script rerun (improves performance)
@st.cache_data
def load_customers():
    """Load and preprocess customer dataset."""
    # Read customer data from CSV file
    df = pd.read_csv("data/customers.csv")
    # Convert signup_date from string to datetime format for time-based analysis
    df['signup_date'] = pd.to_datetime(df['signup_date'])
    # Convert churned column to boolean (True/False) for binary classification
    df['churned'] = df['churned'].astype(bool)
    # Calculate Customer Lifetime Value = Monthly Recurring Revenue × Tenure
    df['customer_lifetime_value'] = df['mrr'] * df['tenure_months']
    # Categorize NPS scores into meaningful groups using pd.cut()
    # Bins: Detractor (-100 to 0), Passive Low (0-30), Passive High (30-50), Promoter (50-70), Champion (70-100)
    df['nps_category'] = pd.cut(
        df['nps_score'],
        bins=[-101, 0, 30, 50, 70, 101],
        labels=['Detractor', 'Passive Low', 'Passive High', 'Promoter', 'Champion']
    )
    return df

# STREAMLIT CACHE: Avoid reloading data on every script rerun
@st.cache_data
def load_transactions():
    """Load and preprocess transaction dataset."""
    # Read transaction data from CSV file
    df = pd.read_csv("data/transactions.csv")
    # Convert transaction_date from string to datetime format
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    # Extract year-month for monthly aggregation (e.g., "2023-01")
    df['year_month'] = df['transaction_date'].dt.to_period('M').astype(str)
    # Extract quarter for quarterly analysis (e.g., "2023Q1")
    df['quarter'] = df['transaction_date'].dt.to_period('Q').astype(str)
    return df

# STREAMLIT CACHE: Avoid reloading data on every script rerun
@st.cache_data
def load_support_tickets():
    """Load and preprocess support tickets dataset."""
    # Read support ticket data from CSV file
    df = pd.read_csv("data/support_tickets.csv")
    # Convert created_date from string to datetime format
    df['created_date'] = pd.to_datetime(df['created_date'])
    # Extract year-month for monthly aggregation
    df['year_month'] = df['created_date'].dt.to_period('M').astype(str)
    # Map numeric satisfaction ratings (1-5) to descriptive categories
    df['satisfaction_category'] = df['satisfaction_rating'].map({
        1: 'Very Dissatisfied', 2: 'Dissatisfied', 3: 'Neutral',
        4: 'Satisfied', 5: 'Very Satisfied'
    })
    return df

# STREAMLIT CACHE: Avoid reloading data on every script rerun
@st.cache_data
def load_ab_test():
    """Load and preprocess A/B test dataset."""
    # Read A/B test data from CSV file
    df = pd.read_csv("data/ab_test.csv")
    # Convert converted column to boolean (True/False) for binary classification
    df['converted'] = df['converted'].astype(bool)
    return df

def get_kpis(customers_df, transactions_df, support_df, ab_df):
    """Calculate key business KPIs (Key Performance Indicators)."""
    # Total number of customers (unique customer count)
    total_customers = len(customers_df)
    # Total revenue from all transactions (sum of all transaction amounts)
    total_revenue = transactions_df['amount'].sum()
    # Average Monthly Recurring Revenue per customer
    avg_mrr = customers_df['mrr'].mean()
    # Churn rate as percentage (proportion of churned customers × 100)
    churn_rate = customers_df['churned'].mean() * 100
    # Average time to resolve support tickets (in hours)
    avg_resolution = support_df['resolution_hours'].mean()
    # Average Net Promoter Score (range: -100 to +100)
    nps_avg = customers_df['nps_score'].mean()
    # A/B test conversion rates by group (Control vs Treatment) as percentage
    ab_conversion = ab_df.groupby('test_group')['converted'].mean() * 100

    # Return dictionary with all KPIs for dashboard display
    return {
        'total_customers': total_customers,
        'total_revenue': total_revenue,
        'avg_mrr': avg_mrr,
        'churn_rate': churn_rate,
        'avg_resolution': avg_resolution,
        'nps_avg': nps_avg,
        'ab_conversion': ab_conversion
    }
