# Streamlit framework for building interactive web applications
import streamlit as st
# NumPy library for numerical computations and array operations
import numpy as np
# Custom utility functions to load datasets from CSV files
from utils.data_loader import load_customers, load_transactions, load_support_tickets, load_ab_test
# Custom visualization functions for creating Plotly charts
from utils.visualizations import plotly_histogram, plotly_bar

# Configure Streamlit page settings: set title in browser tab, icon, and use wide layout for better data display
st.set_page_config(page_title="Data Overview", page_icon="📋", layout="wide")

# Display main page title with icon
st.title("📋 Data Overview")
# Brief description of the page purpose
st.markdown("Explore the structure, quality, and distributions of all datasets.")

# Load data from CSV files using custom data loader functions
# Each function reads and preprocesses the respective dataset
customers = load_customers()        # Customer demographics, plans, churn status
transactions = load_transactions()  # Transaction history with amounts and dates
support = load_support_tickets()    # Support ticket data with resolution times
ab_test = load_ab_test()            # A/B test results for dashboard feature

# Create a dictionary to store all datasets with descriptive names for easy access
datasets = {
    "Customers": customers,           # Customer demographics and subscription info
    "Transactions": transactions,      # Financial transaction records
    "Support Tickets": support,        # Customer support interactions
    "A/B Test": ab_test              # A/B test results for feature testing
}

# Create four tabs for different data exploration views
# Each tab provides a different perspective on the data
tab1, tab2, tab3, tab4 = st.tabs(["📊 Schema & Summary", "🔍 Data Quality", "📈 Distributions", "🔗 Relationships"])

# TAB 1: Schema & Summary Statistics
# Display dataset structure, column info, and summary statistics
with tab1:
    st.subheader("Dataset Schema & Summary Statistics")
    # Dropdown to select which dataset to analyze
    selected_ds = st.selectbox("Select Dataset", list(datasets.keys()))
    # Get the selected dataframe from the datasets dictionary
    df = datasets[selected_ds]

    # Create two columns for side-by-side layout
    col1, col2 = st.columns(2)
    with col1:
        # Display dataset dimensions (rows × columns) with comma formatting for thousands
        st.markdown(f"**Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")
        st.markdown("**Columns:**")
        # Iterate through all columns and display their names and data types
        for col in df.columns:
            dtype = str(df[col].dtype)  # Get the data type of each column
            st.markdown(f"- `{col}` ({dtype})")  # Display column name with its type
    with col2:
        st.markdown("**First 5 Rows:**")
        # Show first 5 rows of the dataframe for quick preview
        st.dataframe(df.head(), use_container_width=True)

    # Display comprehensive descriptive statistics (count, mean, std, min, quartiles, max, etc.)
    st.markdown("**Descriptive Statistics:**")
    # include='all' ensures both numeric and categorical columns are included
    st.dataframe(df.describe(include='all').T, use_container_width=True)

# TAB 2: Data Quality Assessment
# Check for missing values, duplicates, and column types across all datasets
with tab2:
    st.subheader("Data Quality Assessment")
    # Iterate through each dataset to assess its quality metrics
    for name, df in datasets.items():
        # Create an expandable section for each dataset
        with st.expander(f"{name} — Quality Metrics"):
            # Create four columns for quality metrics display
            col1, col2, col3, col4 = st.columns(4)
            # Count total missing values across all columns in the dataset
            col1.metric("Missing Values", df.isnull().sum().sum())
            # Count duplicate rows (identical across all columns)
            col2.metric("Duplicate Rows", df.duplicated().sum())
            # Count numeric columns (int, float) using numpy number dtype
            col3.metric("Numeric Cols", len(df.select_dtypes(include=[np.number]).columns))
            # Count categorical columns (string, category) for analysis
            col4.metric("Categorical Cols", len(df.select_dtypes(include=['object', 'category']).columns))

            # Check if there are any missing values in the dataset
            if df.isnull().sum().sum() > 0:
                # Display warning and list columns with missing values
                st.warning("⚠️ Missing values detected:")
                # Show only columns that have missing values with their counts
                st.write(df.isnull().sum()[df.isnull().sum() > 0])
            else:
                # All clean, no missing values
                st.success("✅ No missing values")

# TAB 3: Variable Distributions
# Visualize distributions of numeric and categorical variables
with tab3:
    st.subheader("Variable Distributions")
    # Dropdown to select which dataset to visualize
    selected_ds_viz = st.selectbox("Select Dataset for Visualization", list(datasets.keys()), key="viz_ds")
    # Get the selected dataset
    df_viz = datasets[selected_ds_viz]
    # Extract numeric column names (int, float) for histogram analysis
    numeric_cols = df_viz.select_dtypes(include=[np.number]).columns.tolist()
    # Extract categorical column names (string, category) for frequency analysis
    categorical_cols = df_viz.select_dtypes(include=['object', 'category']).columns.tolist()

    # Display histogram for numeric variables
    if numeric_cols:
        # Dropdown to select which numeric variable to visualize
        selected_num = st.selectbox("Numeric Variable", numeric_cols)
        # Create interactive histogram with box plot overlay using Plotly
        fig = plotly_histogram(df_viz, selected_num, f"Distribution of {selected_num}")
        # Display the histogram in Streamlit with full width
        st.plotly_chart(fig, use_container_width=True)

    # Display bar chart for categorical variables
    if categorical_cols:
        # Dropdown to select which categorical variable to visualize
        selected_cat = st.selectbox("Categorical Variable", categorical_cols)
        # Calculate frequency counts for each category
        counts = df_viz[selected_cat].value_counts().reset_index()
        # Rename columns for clarity
        counts.columns = [selected_cat, 'Count']
        # Create interactive bar chart showing frequency of each category
        fig2 = plotly_bar(counts, selected_cat, 'Count', f"Frequency of {selected_cat}")
        # Display the bar chart in Streamlit with full width
        st.plotly_chart(fig2, use_container_width=True)

# TAB 4: Cross-Dataset Relationships
# Demonstrate how datasets can be joined and analyzed together
with tab4:
    st.subheader("Cross-Dataset Relationships")
    # Explain the relationship being demonstrated
    st.markdown("Sample join between Customers and Transactions:")
    # Merge transactions with customer details (plan, region, industry) on customer_id
    # Using left join to keep all transactions even if customer details are missing
    merged = transactions.merge(customers[['customer_id', 'plan', 'region', 'industry']], 
                                on='customer_id', how='left')
    # Display first 10 rows of the merged dataset to show the relationship
    st.dataframe(merged.head(10), use_container_width=True)

    # Revenue analysis by subscription plan
    st.markdown("**Revenue by Plan:**")
    # Group by plan and calculate total revenue, average transaction, and transaction count
    revenue_by_plan = merged.groupby('plan')['amount'].agg(['sum', 'mean', 'count']).reset_index()
    # Rename columns for better readability in the chart
    revenue_by_plan.columns = ['Plan', 'Total Revenue', 'Avg Transaction', 'Count']
    # Create interactive bar chart showing revenue by subscription plan
    fig3 = plotly_bar(revenue_by_plan, 'Plan', 'Total Revenue', 'Revenue by Subscription Plan')
    # Display the chart with full width
    st.plotly_chart(fig3, use_container_width=True)
