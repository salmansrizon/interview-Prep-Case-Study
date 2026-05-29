"""
Visualization utilities for consistent chart styling.
Provides functions for creating Plotly charts and Streamlit KPI cards.
"""

# Matplotlib for static plot styling (though mainly using Plotly)
import matplotlib.pyplot as plt

# Seaborn for statistical data visualization styling
import seaborn as sns

# Streamlit for displaying KPI cards
import streamlit as st

# Plotly Express for creating interactive charts quickly
import plotly.express as px

# Plotly Graph Objects for more customized chart creation
import plotly.graph_objects as go

# Plotly Subplots for creating multiple plots in one figure
from plotly.subplots import make_subplots

# Set default Matplotlib style for any static plots
plt.style.use("seaborn-v0_8-whitegrid")
# Set Seaborn color palette for consistent coloring
sns.set_palette("husl")


def plot_kpi_cards(kpis):
    """Display KPI cards in Streamlit (Key Performance Indicators)."""
    # First row of KPI cards (4 columns)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        # Total number of customers (formatted with comma for thousands)
        st.metric("Total Customers", f"{kpis['total_customers']:,}")
    with col2:
        # Total revenue from all transactions (formatted as currency)
        st.metric("Total Revenue", f"${kpis['total_revenue']:,.0f}")
    with col3:
        # Average Monthly Recurring Revenue per customer
        st.metric("Avg MRR", f"${kpis['avg_mrr']:.0f}")
    with col4:
        # Churn rate as percentage (proportion of churned customers)
        st.metric("Churn Rate", f"{kpis['churn_rate']:.1f}%")

    # Second row of KPI cards (4 columns)
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        # Average Net Promoter Score (-100 to +100)
        st.metric("Avg NPS", f"{kpis['nps_avg']:.1f}")
    with col6:
        # Average time to resolve support tickets (in hours)
        st.metric("Avg Resolution", f"{kpis['avg_resolution']:.1f}h")
    with col7:
        # A/B test conversion rate for Control group
        ctrl_conv = kpis["ab_conversion"].get("Control", 0)
        st.metric("Control Conv.", f"{ctrl_conv:.1f}%")
    with col8:
        # A/B test conversion rate for Treatment group
        trt_conv = kpis["ab_conversion"].get("Treatment", 0)
        st.metric("Treatment Conv.", f"{trt_conv:.1f}%")


def plotly_histogram(df, x_col, title, color_col=None, nbins=30):
    """
    Create an interactive histogram using Plotly Express.
    Includes marginal box plot for additional distribution insight.
    """
    if color_col:
        # Histogram colored by group (shows distribution per category)
        fig = px.histogram(
            df,
            x=x_col,
            color=color_col,
            nbins=nbins,
            marginal="box",
            opacity=0.7,
            title=title,
        )
    else:
        # Simple histogram with box plot overlay
        fig = px.histogram(
            df, x=x_col, nbins=nbins, marginal="box", opacity=0.7, title=title
        )
    # Reduce gap between bars for better visual appearance
    fig.update_layout(bargap=0.1)
    return fig


def plotly_bar(data, x_col, y_col, title, color_col=None):
    """
    Create an interactive bar chart using Plotly Express.
    Useful for comparing categorical data (e.g., frequency counts, group means).
    """
    if color_col:
        # Bar chart with color grouping (different color per category)
        fig = px.bar(data, x=x_col, y=y_col, color=color_col, title=title)
    else:
        # Simple bar chart without grouping
        fig = px.bar(data, x=x_col, y=y_col, title=title)
    return fig


def plotly_line(data, x_col, y_col, title, color_col=None):
    """
    Create an interactive line chart using Plotly Express.
    Useful for time series data (e.g., monthly revenue trends).
    """
    if color_col:
        # Line chart with multiple lines (colored by group)
        fig = px.line(
            data, x=x_col, y=y_col, color=color_col, title=title, markers=True
        )  # markers=True shows data points
    else:
        # Simple line chart with single line
        fig = px.line(data, x=x_col, y=y_col, title=title, markers=True)
    return fig


def plotly_heatmap(corr_matrix, title="Correlation Matrix"):
    """
    Create an interactive correlation heatmap using Plotly Express.
    Shows Pearson correlation coefficients between all numeric variables.
    Red = positive correlation, Blue = negative correlation.
    """
    # imshow displays matrix as heatmap with color scale
    # text_auto=True shows correlation values on each cell
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title=title,
    )
    # Format text to show 2 decimal places
    fig.update_traces(texttemplate="%{text:.2f}", textfont_size=10)
    return fig


def plotly_box(df, x_col, y_col, title, color_col=None):
    """
    Create an interactive box plot using Plotly Express.
    Box plot shows: median (line), Q1-Q3 (box), whiskers (1.5×IQR), outliers (dots).
    """
    if color_col:
        # Box plot with color grouping (different color per category)
        fig = px.box(df, x=x_col, y=y_col, color=color_col, title=title)
    else:
        # Simple box plot without grouping
        fig = px.box(df, x=x_col, y=y_col, title=title)
    return fig
