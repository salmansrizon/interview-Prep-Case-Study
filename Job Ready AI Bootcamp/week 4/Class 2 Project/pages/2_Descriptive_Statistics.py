# Streamlit framework for building interactive web applications
import streamlit as st

# Pandas library for data manipulation, analysis, and DataFrame operations
import pandas as pd

# NumPy library for numerical computations and array operations
import numpy as np

# Matplotlib for creating static plots (used for Q-Q plots)
import matplotlib.pyplot as plt
from scipy.stats import probplot

# Custom utility functions to load datasets from CSV files
from utils.data_loader import load_customers, load_transactions, load_support_tickets

# Custom visualization functions for creating Plotly charts
from utils.visualizations import plotly_histogram, plotly_box

# Configure Streamlit page settings: set title in browser tab, icon, and use wide layout for better data display
st.set_page_config(page_title="Descriptive Statistics", page_icon="📈", layout="wide")

# Display main page title with icon
st.title("📈 Descriptive Statistics")
# Brief description of the page purpose - covering key statistical concepts
st.markdown("Central tendency, spread, distribution shape, and outliers.")

# Load data from CSV files using custom data loader functions
# Each function reads and preprocesses the respective dataset
customers = load_customers()  # Customer demographics, plans, churn status
transactions = load_transactions()  # Transaction history with amounts and dates
support = load_support_tickets()  # Support ticket data with resolution times

# Create four tabs for different statistical analysis views
tab1, tab2, tab3, tab4 = st.tabs(
    ["📐 Central Tendency", "📏 Spread & Shape", "📊 Distributions", "🎯 Outliers"]
)

# TAB 1: Central Tendency Measures (Mean, Median, Mode)
with tab1:
    st.subheader("Measures of Central Tendency")

    # Dictionary mapping metric names to their corresponding data series
    # Each series contains numeric data for statistical analysis
    metrics = {
        "MRR ($)": customers["mrr"],  # Monthly recurring revenue per customer
        "Tenure (months)": customers[
            "tenure_months"
        ],  # How long customer has been subscribed
        "NPS Score": customers["nps_score"],  # Net Promoter Score (-100 to 100)
        "Feature Adoption (%)": customers[
            "feature_adoption_score"
        ],  # Percentage of features used
        "Transaction Amount ($)": transactions[
            "amount"
        ],  # Dollar amount of each transaction
        "Resolution Hours": support[
            "resolution_hours"
        ],  # Time to resolve support tickets
    }

    # Iterate through each metric and display its central tendency measures
    for name, series in metrics.items():
        # Create expandable section for each metric to avoid cluttering the page
        with st.expander(f"📊 {name}"):
            # Create four columns for side-by-side metric display
            col1, col2, col3, col4 = st.columns(4)
            # Mean: arithmetic average (sum of all values ÷ number of values)
            col1.metric("Mean", f"{series.mean():.2f}")
            # Median: middle value when sorted (50th percentile, robust to outliers)
            col2.metric("Median", f"{series.median():.2f}")
            # Mode: most frequently occurring value in the dataset
            col3.metric("Mode", f"{series.mode().iloc[0]:.2f}")
            # Count: total number of non-null observations
            col4.metric("Count", f"{len(series):,}")

            # Calculate skewness to understand distribution asymmetry
            # Skewness > 0: right-skewed (tail on right), < 0: left-skewed (tail on left)
            skew = float(series.skew())  # Convert to float to avoid type issues
            skew_abs = abs(skew)
            skew_text = (
                "Symmetric"
                if skew_abs < 0.5
                else ("Right-skewed" if skew > 0 else "Left-skewed")
            )
            st.markdown(f"**Skewness:** {skew:.3f} → {skew_text}")

            # Interpret relationship between mean and median for skew detection
            # In right-skewed: mean > median (outliers pull mean right)
            # In left-skewed: mean < median (outliers pull mean left)
            if series.mean() > series.median() * 1.1:
                st.info(
                    "💡 Mean > Median suggests right skew (high values pulling mean up)"
                )
            elif series.mean() < series.median() * 0.9:
                st.info(
                    "💡 Mean < Median suggests left skew (low values pulling mean down)"
                )

# TAB 2: Measures of Spread (Variability) and Distribution Shape
with tab2:
    st.subheader("Measures of Spread & Distribution Shape")

    # Dropdown to select which metric to analyze for spread
    selected_metric = st.selectbox("Select Metric", list(metrics.keys()))
    # Get the selected series for analysis
    series = metrics[selected_metric]

    # Display four key spread metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    # Standard Deviation: average distance from mean (sqrt of variance)
    col1.metric("Std Dev", f"{series.std():.2f}")
    # Variance: average squared deviation from mean (std dev squared)
    col2.metric("Variance", f"{series.var():.2f}")
    # Range: difference between maximum and minimum values
    col3.metric("Range", f"{series.max() - series.min():.2f}")
    # IQR (Interquartile Range): middle 50% of data (Q3 - Q1), robust to outliers
    col4.metric("IQR", f"{series.quantile(0.75) - series.quantile(0.25):.2f}")

    # Display five-number summary in columns
    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Min", f"{series.min():.2f}")  # Smallest value
    col6.metric("Q1 (25%)", f"{series.quantile(0.25):.2f}")  # 25th percentile
    col7.metric("Q3 (75%)", f"{series.quantile(0.75):.2f}")  # 75th percentile
    col8.metric("Max", f"{series.max():.2f}")  # Largest value

    st.markdown("---")
    # Percentile analysis: show key percentiles for understanding data distribution
    st.markdown("**Percentile Analysis:**")
    # Define percentiles to analyze (5th, 10th, 25th, 50th, 75th, 90th, 95th, 99th)
    percentiles = [5, 10, 25, 50, 75, 90, 95, 99]
    # Create DataFrame with percentile labels and their corresponding values
    perc_df = pd.DataFrame(
        {
            "Percentile": [f"{p}th" for p in percentiles],
            "Value": [series.quantile(p / 100) for p in percentiles],
        }
    )
    st.dataframe(perc_df, use_container_width=True)

    # Empirical Rule Check: For normal distributions, ~68% within 1σ, ~95% within 2σ, ~99.7% within 3σ
    mean, std = series.mean(), series.std()
    st.markdown("**Empirical Rule (68-95-99.7) Check:**")
    for k in [1, 2, 3]:
        # Calculate percentage of data within k standard deviations
        within = ((series >= mean - k * std) & (series <= mean + k * std)).mean() * 100
        # Compare actual percentage with theoretical normal distribution percentages
        st.markdown(
            f"Within {k}σ: **{within:.1f}%** of data (theoretical: {68 if k==1 else (95 if k==2 else 99.7)}%)"
        )

# TAB 3: Distribution Visualizations (Histogram, Box Plot, Q-Q Plot)
with tab3:
    st.subheader("Distribution Visualizations")

    # Dropdown to select which variable to visualize
    viz_metric = st.selectbox(
        "Select Variable to Visualize", list(metrics.keys()), key="viz"
    )
    # Get the selected series for visualization
    series_viz = metrics[viz_metric]

    # Create two columns for side-by-side histogram and box plot
    col1, col2 = st.columns(2)
    with col1:
        # Create interactive histogram with 40 bins to show frequency distribution
        fig = plotly_histogram(
            pd.DataFrame({viz_metric: series_viz}),
            viz_metric,
            f"Histogram: {viz_metric}",
            nbins=40,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Create box plot to show quartiles, median, and outliers
        fig2 = plotly_box(
            pd.DataFrame({viz_metric: series_viz}),
            None,
            viz_metric,
            f"Box Plot: {viz_metric}",
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Q-Q Plot: Compare data distribution against normal distribution
    # Points on diagonal line = normal distribution
    st.markdown("**Q-Q Plot (Normality Check):**")
    fig_qq, ax = plt.subplots(figsize=(8, 6))
    # Import probplot for quantile-quantile plot against normal distribution
    from scipy.stats import probplot

    # Create Q-Q plot: if points follow diagonal line, data is normally distributed
    probplot(series_viz.dropna(), dist="norm", plot=ax)
    ax.set_title(f"Q-Q Plot: {viz_metric}")
    st.pyplot(fig_qq)

    # Check for heavy skewness and warn about potential need for transformation
    # Convert skew to float to avoid type-checker issues with numpy scalars
    skew_viz = float(series_viz.skew())
    if abs(skew_viz) > 1:
        st.warning(
            "⚠️ Distribution is heavily skewed. Consider log transformation for modeling."
        )

# TAB 4: Outlier Detection using IQR and Z-Score Methods
with tab4:
    st.subheader("Outlier Detection")

    # Dropdown to select which variable to check for outliers
    outlier_metric = st.selectbox(
        "Select Variable", list(metrics.keys()), key="outlier"
    )
    # Get the selected series for outlier analysis
    series_out = metrics[outlier_metric]

    # IQR METHOD: Identify outliers using Interquartile Range (robust to extreme values)
    # Outliers are values below Q1 - 1.5×IQR or above Q3 + 1.5×IQR
    Q1 = series_out.quantile(0.25)  # 25th percentile (first quartile)
    Q3 = series_out.quantile(0.75)  # 75th percentile (third quartile)
    IQR = Q3 - Q1  # Interquartile Range (middle 50% of data)
    # Lower fence: values below this are considered outliers
    lower_fence = Q1 - 1.5 * IQR
    # Upper fence: values above this are considered outliers
    upper_fence = Q3 + 1.5 * IQR

    # Identify outliers using IQR method
    outliers_iqr = series_out[(series_out < lower_fence) | (series_out > upper_fence)]

    # Z-SCORE METHOD: Identify outliers based on standard deviations from mean
    # Values with |z-score| > 3 are considered outliers (0.27% of normal data)
    z_scores = np.abs((series_out - series_out.mean()) / series_out.std())
    outliers_z = series_out[z_scores > 3]

    # Display results from both methods side by side
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**IQR Method (1.5×IQR)**")
        st.markdown(f"Lower Fence: {lower_fence:.2f}")
        st.markdown(f"Upper Fence: {upper_fence:.2f}")
        st.markdown(
            f"Outliers Detected: **{len(outliers_iqr)}** ({len(outliers_iqr)/len(series_out)*100:.2f}%)"
        )
        # Show descriptive statistics of the outliers if any exist
        if len(outliers_iqr) > 0:
            st.dataframe(outliers_iqr.describe(), use_container_width=True)

    with col2:
        st.markdown("**Z-Score Method (|z| > 3)**")
        st.markdown(
            f"Outliers Detected: **{len(outliers_z)}** ({len(outliers_z)/len(series_out)*100:.2f}%)"
        )
        # Show descriptive statistics of the outliers if any exist
        if len(outliers_z) > 0:
            st.dataframe(outliers_z.describe(), use_container_width=True)

    st.markdown("---")
    # Show how removing outliers affects statistical measures
    st.markdown("**Outlier Impact on Statistics:**")
    # Create cleaned series by removing IQR outliers
    clean_series = series_out[(series_out >= lower_fence) & (series_out <= upper_fence)]

    impact_df = pd.DataFrame(
        {
            "Metric": ["Mean", "Median", "Std Dev"],
            "With Outliers": [series_out.mean(), series_out.median(), series_out.std()],
            "Without Outliers": [
                clean_series.mean(),
                clean_series.median(),
                clean_series.std(),
            ],
        }
    )
    impact_df["Difference"] = impact_df["With Outliers"] - impact_df["Without Outliers"]
    impact_df["Pct Change"] = (
        impact_df["Difference"] / impact_df["Without Outliers"] * 100
    ).round(2)
    st.dataframe(impact_df, use_container_width=True)
