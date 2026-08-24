import streamlit as st
import pandas as pd
import numpy as np
from utils.data_loader import load_housing_data, get_feature_stats
from utils.visualizations import plot_feature_distribution

st.set_page_config(page_title="Explore Data", page_icon="📊", layout="wide")

st.title("📊 Explore the Housing Dataset")
st.markdown("Understand the data before building models. EDA is the first step in any ML project.")

df = load_housing_data()

tab1, tab2, tab3, tab4 = st.tabs(["📋 Overview", "📈 Distributions", "🔗 Correlations", "🎯 Target Analysis"])

with tab1:
    st.subheader("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Features", f"{len(df.columns)-1}")
    col3.metric("Target Range", f"${df['price'].min():,.0f} - ${df['price'].max():,.0f}")

    st.markdown("**First 10 Rows:**")
    st.dataframe(df.head(10), width="stretch")

    st.markdown("**Descriptive Statistics:**")
    st.dataframe(df.describe().T, width="stretch")

    st.markdown("**Data Types & Missing Values:**")
    dtype_df = pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes.astype(str),
        'Missing': df.isnull().sum(),
        'Unique': [df[col].nunique() for col in df.columns]
    })
    st.dataframe(dtype_df, width="stretch")

with tab2:
    st.subheader("Feature Distributions")
    feature = st.selectbox("Select Feature", [c for c in df.columns if c != 'price'])
    fig = plot_feature_distribution(df, feature)
    st.pyplot(fig)

    stats = df[feature].describe()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean", f"{stats['mean']:.2f}")
    col2.metric("Std Dev", f"{stats['std']:.2f}")
    col3.metric("Min", f"{stats['min']:.2f}")
    col4.metric("Max", f"{stats['max']:.2f}")

with tab3:
    st.subheader("Correlation Matrix")
    corr = df.corr()

    import plotly.express as px
    fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                    color_continuous_scale="RdBu_r", title="Feature Correlation Heatmap")
    st.plotly_chart(fig, width="stretch")

    st.markdown("**Top Correlations with Price:**")
    price_corr = corr['price'].drop('price').sort_values(key=abs, ascending=False)
    corr_df = pd.DataFrame({
        'Feature': price_corr.index,
        'Correlation': price_corr.values,
        'Strength': ['Strong' if abs(v) > 0.5 else 'Moderate' if abs(v) > 0.3 else 'Weak' for v in price_corr.values]
    })
    st.dataframe(corr_df, width="stretch")

with tab4:
    st.subheader("Target Variable: House Price")

    fig = plot_feature_distribution(df, 'price', color='darkgreen')
    st.pyplot(fig)

    st.markdown("**Price Statistics:**")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean", f"${df['price'].mean():,.0f}")
    col2.metric("Median", f"${df['price'].median():,.0f}")
    col3.metric("Std Dev", f"${df['price'].std():,.0f}")
    col4.metric("Skewness", f"{df['price'].skew():.2f}")

    if df['price'].skew() > 0.5:
        st.info("💡 The price distribution is right-skewed. Consider log transformation for some models.")
