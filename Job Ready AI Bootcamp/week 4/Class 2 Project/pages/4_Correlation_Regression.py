# Streamlit framework for building interactive web applications
import streamlit as st
# Pandas library for data manipulation, analysis, and DataFrame operations
import pandas as pd
# NumPy library for numerical computations and array operations
import numpy as np
# Matplotlib for creating static plots (used for scatter plots with regression lines)
import matplotlib.pyplot as plt
# Seaborn for enhanced statistical data visualization
import seaborn as sns
# SciPy statistical functions for correlation analysis
from scipy.stats import pearsonr
# Scikit-learn for machine learning (linear regression modeling)
from sklearn.linear_model import LinearRegression
# Scikit-learn metrics for evaluating regression model performance
from sklearn.metrics import r2_score
# Custom utility functions to load datasets from CSV files
from utils.data_loader import load_customers, load_transactions, load_support_tickets
# Custom visualization functions for creating Plotly charts
from utils.visualizations import plotly_heatmap

# Configure Streamlit page settings: set title in browser tab, icon, and use wide layout for better data display
st.set_page_config(page_title="Correlation & Regression", page_icon="🔗", layout="wide")

# Display main page title with icon
st.title("🔗 Correlation & Regression Analysis")
# Brief description of the page purpose - correlation analysis and predictive modeling
st.markdown("Discover relationships between variables and build predictive models.")

# Load data from CSV files using custom data loader functions
# Each function reads and preprocesses the respective dataset
customers = load_customers()        # Customer demographics, plans, churn status
transactions = load_transactions()  # Transaction history with amounts and dates
support = load_support_tickets()    # Support ticket data with resolution times

# Create three tabs for correlation analysis, scatter plots, and regression modeling
tab1, tab2, tab3 = st.tabs(["📊 Correlation Matrix", "📈 Scatter Analysis", "🤖 Regression Model"])

# TAB 1: Correlation Matrix (Pearson correlation between all numeric variables)
with tab1:
    st.subheader("Correlation Matrix — Numeric Variables")

    # MERGE DATASETS: Combine customer, transaction, and support data for comprehensive analysis
    # Select key customer variables for correlation analysis
    customer_summary = customers[['customer_id', 'mrr', 'tenure_months', 'nps_score', 
                                   'support_tickets', 'feature_adoption_score', 'churned']].copy()
    # Convert boolean churned to numeric (0/1) for correlation calculation
    customer_summary['churned_num'] = customer_summary['churned'].astype(int)

    # Aggregate transaction data by customer (total revenue, count, average)
    trans_agg = transactions.groupby('customer_id').agg(
        total_revenue=('amount', 'sum'),           # Total money spent by customer
        transaction_count=('transaction_id', 'count'), # Number of transactions
        avg_transaction=('amount', 'mean')          # Average transaction amount
    ).reset_index()

    # Aggregate support data by customer (ticket count, resolution time, satisfaction)
    support_agg = support.groupby('customer_id').agg(
        total_tickets=('ticket_id', 'count'),       # Total tickets submitted
        avg_resolution=('resolution_hours', 'mean'), # Average resolution time
        avg_satisfaction=('satisfaction_rating', 'mean') # Average satisfaction rating
    ).reset_index()

    # Merge all aggregated data into a single dataframe
    merged = customer_summary.merge(trans_agg, on='customer_id', how='left')
    merged = merged.merge(support_agg, on='customer_id', how='left')
    # Fill missing values with 0 (customers with no transactions or support tickets)
    merged = merged.fillna(0)

    # Define numeric columns for correlation analysis
    numeric_cols = ['mrr', 'tenure_months', 'nps_score', 'support_tickets', 
                    'feature_adoption_score', 'churned_num', 'total_revenue',
                    'transaction_count', 'avg_transaction', 'total_tickets',
                    'avg_resolution', 'avg_satisfaction']

    # Calculate Pearson correlation matrix (measures linear relationships between all pairs)
    corr_matrix = merged[numeric_cols].corr()

    # Display interactive heatmap of correlation matrix
    fig = plotly_heatmap(corr_matrix, "Correlation Matrix — All Numeric Variables")
    st.plotly_chart(fig, use_container_width=True)

    # List all correlation pairs with strength classification
    st.markdown("**Key Correlations:**")
    corr_pairs = []
    # Iterate through all unique pairs of variables (upper triangle of matrix)
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            r = corr_matrix.iloc[i, j]
            corr_pairs.append({
                'Variable 1': corr_matrix.columns[i],
                'Variable 2': corr_matrix.columns[j],
                'Pearson r': r,
                # Classify correlation strength: |r| > 0.7 = Strong, > 0.3 = Moderate, else Weak
                'Strength': 'Strong' if abs(r) > 0.7 else ('Moderate' if abs(r) > 0.3 else 'Weak')
            })

    # Create DataFrame and sort by absolute correlation value (strongest first)
    corr_df = pd.DataFrame(corr_pairs).sort_values('Pearson r', key=abs, ascending=False)
    st.dataframe(corr_df, use_container_width=True)

# TAB 2: Scatter Plot Analysis (Visualize relationships between two variables)
with tab2:
    st.subheader("Scatter Plot Analysis")

    # Dropdown to select X and Y variables for scatter plot
    x_var = st.selectbox("X Variable", numeric_cols, index=0)  # Predictor variable
    y_var = st.selectbox("Y Variable", numeric_cols, index=1)  # Response variable
    # Optional: Color points by categorical variable to see group differences
    color_var = st.selectbox("Color By", ['None'] + ['plan', 'region', 'company_size', 'industry'], index=0)

    # Prepare data for scatter plot
    plot_df = merged.copy()
    if color_var != 'None':
        # Merge categorical variable for coloring points
        plot_df = plot_df.merge(customers[['customer_id', color_var]], on='customer_id', how='left')

    # Create scatter plot with Matplotlib/Seaborn
    fig, ax = plt.subplots(figsize=(10, 7))
    if color_var != 'None':
        # Colored scatter plot by group
        sns.scatterplot(data=plot_df, x=x_var, y=y_var, hue=color_var, alpha=0.6, ax=ax)
    else:
        # Simple scatter plot without grouping
        sns.scatterplot(data=plot_df, x=x_var, y=y_var, alpha=0.6, ax=ax)

    # Add regression line (linear trend) to visualize relationship
    valid = plot_df[[x_var, y_var]].dropna()
    if len(valid) > 1:
        # Fit linear regression line: y = mx + b
        z = np.polyfit(valid[x_var], valid[y_var], 1)
        p = np.poly1d(z)
        # Plot the regression line in red dashed style
        ax.plot(valid[x_var].sort_values(), p(valid[x_var].sort_values()), 
                "r--", alpha=0.8, linewidth=2, label='Linear fit')

    ax.set_title(f"{y_var} vs {x_var}")
    ax.legend()
    st.pyplot(fig)

    # Calculate and display Pearson correlation statistics
    valid_corr = plot_df[[x_var, y_var]].dropna()
    if len(valid_corr) > 2:
        # Pearson correlation: measures linear relationship (-1 to +1)
        r, p_val = pearsonr(valid_corr[x_var], valid_corr[y_var])
        st.markdown(f"**Pearson Correlation:** r = {r:.4f}, p = {p_val:.6f}")

        # Interpret statistical significance
        if p_val < 0.05:
            st.success(f"✅ Significant {'positive' if r > 0 else 'negative'} correlation")
        else:
            st.info("ℹ️ No significant linear correlation detected")

        # R²: Proportion of variance in Y explained by X
        st.markdown(f"**R² (Explained Variance):** {r**2:.2%} of {y_var} variance is explained by {x_var}")

# TAB 3: Predictive Regression Model (Linear Regression)
with tab3:
    st.subheader("Predictive Regression Model")
    st.markdown("Predict customer MRR based on other features.")

    # Multi-select to choose predictor variables (features) for the regression model
    feature_cols = st.multiselect(
        "Select Predictor Variables",
        ['tenure_months', 'nps_score', 'support_tickets', 'feature_adoption_score',
         'transaction_count', 'avg_satisfaction'],
        default=['tenure_months', 'feature_adoption_score', 'transaction_count']
    )

    # Build regression model if at least one feature is selected
    if len(feature_cols) >= 1:
        # Prepare data: features (X) and target variable (y = MRR)
        model_df = merged[feature_cols + ['mrr']].dropna()
        X = model_df[feature_cols].values    # Feature matrix (n_samples × n_features)
        y = model_df['mrr'].values        # Target vector (Monthly Recurring Revenue)

        # Create and train linear regression model: y = β₀ + β₁x₁ + β₂x₂ + ... + ε
        model = LinearRegression()
        model.fit(X, y)                    # Fit model to training data
        y_pred = model.predict(X)           # Predictions on same data (for evaluation)
        r2 = r2_score(y, y_pred)          # R²: proportion of variance explained

        # Display model performance metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("R² Score", f"{r2:.4f}")                                    # Higher = better fit
        col2.metric("RMSE", f"{np.sqrt(np.mean((y - y_pred)**2)):.2f}")       # Root Mean Squared Error
        col3.metric("MAE", f"{np.mean(np.abs(y - y_pred)):.2f}")              # Mean Absolute Error

        # Display feature coefficients (β values): how much MRR changes per unit change
        st.markdown("**Feature Coefficients:**")
        coef_df = pd.DataFrame({
            'Feature': feature_cols,
            'Coefficient': model.coef_,                                    # β₁, β₂, etc.
            'Abs Coefficient': np.abs(model.coef_)                       # Magnitude of effect
        }).sort_values('Abs Coefficient', ascending=False)               # Strongest predictors first
        st.dataframe(coef_df, use_container_width=True)

        # Intercept (β₀): predicted MRR when all features are zero
        st.markdown(f"**Intercept:** {model.intercept_:.2f}")

        # Actual vs Predicted
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(y, y_pred, alpha=0.5)
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
        ax.set_xlabel('Actual MRR')
        ax.set_ylabel('Predicted MRR')
        ax.set_title(f'Actual vs Predicted MRR (R² = {r2:.3f})')
        st.pyplot(fig)

        st.markdown("**Model Interpretation:**")
        if r2 > 0.5:
            st.success(f"The model explains {r2:.1%} of MRR variance — strong predictive power!")
        elif r2 > 0.2:
            st.info(f"The model explains {r2:.1%} of MRR variance — moderate predictive power.")
        else:
            st.warning(f"The model only explains {r2:.1%} of MRR variance — weak predictive power. Consider other features.")
    else:
        st.warning("Please select at least one predictor variable.")
