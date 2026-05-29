import streamlit as st
import pandas as pd
import numpy as np
from utils.data_loader import load_loan_data

st.set_page_config(page_title="Explore Data", page_icon="📊", layout="wide")

st.title("📊 Explore the Loan Dataset")
st.markdown("Understand the data before building classification models.")

df = load_loan_data()

tab1, tab2, tab3, tab4 = st.tabs(["📋 Overview", "📈 Distributions", "🔗 Correlations", "🎯 Approval Analysis"])

with tab1:
    st.subheader("Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Applications", f"{len(df):,}")
    col2.metric("Approved", f"{(df['loan_approved']==1).sum():,}")
    col3.metric("Denied", f"{(df['loan_approved']==0).sum():,}")
    col4.metric("Approval Rate", f"{df['loan_approved'].mean()*100:.1f}%")

    st.markdown("**First 10 Rows:**")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("**Descriptive Statistics:**")
    st.dataframe(df.describe().T, use_container_width=True)

with tab2:
    st.subheader("Feature Distributions")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ['loan_approved', 'approval_probability']]
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    categorical_cols = [c for c in categorical_cols if c != 'applicant_id']

    viz_type = st.radio("Select Visualization", ["Numeric Features", "Categorical Features"])

    if viz_type == "Numeric Features" and numeric_cols:
        feature = st.selectbox("Select Feature", numeric_cols)
        import plotly.express as px
        fig = px.histogram(df, x=feature, color='loan_approved', 
                           color_discrete_map={0: '#EF5350', 1: '#4CAF50'},
                           barmode='overlay', opacity=0.7,
                           title=f"Distribution of {feature.replace('_', ' ').title()}")
        st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "Categorical Features" and categorical_cols:
        feature = st.selectbox("Select Feature", categorical_cols)
        counts = df.groupby([feature, 'loan_approved']).size().reset_index(name='count')
        counts['status'] = counts['loan_approved'].map({0: 'Denied', 1: 'Approved'})
        fig = px.bar(counts, x=feature, y='count', color='status',
                     color_discrete_map={'Denied': '#EF5350', 'Approved': '#4CAF50'},
                     title=f"{feature.replace('_', ' ').title()} vs Approval Status",
                     barmode='group')
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Correlation Matrix")
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    import plotly.express as px
    fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                    color_continuous_scale="RdBu_r", title="Feature Correlation Heatmap")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Top Correlations with Approval:**")
    approval_corr = corr['loan_approved'].drop('loan_approved').sort_values(key=abs, ascending=False)
    corr_df = pd.DataFrame({
        'Feature': approval_corr.index,
        'Correlation': approval_corr.values,
        'Direction': ['Positive' if v > 0 else 'Negative' for v in approval_corr.values]
    })
    st.dataframe(corr_df, use_container_width=True)

with tab4:
    st.subheader("Approval Rate Analysis")

    analysis_feature = st.selectbox("Analyze Approval Rate By", 
                                     ['credit_history', 'education', 'property_area', 
                                      'self_employed', 'married', 'gender'])

    approval_rates = df.groupby(analysis_feature)['loan_approved'].agg(['mean', 'count']).reset_index()
    approval_rates.columns = [analysis_feature, 'Approval Rate', 'Count']
    approval_rates['Approval Rate'] = approval_rates['Approval Rate'] * 100

    fig = px.bar(approval_rates, x=analysis_feature, y='Approval Rate',
                 text=approval_rates['Approval Rate'].round(1).astype(str) + '%',
                 title=f"Approval Rate by {analysis_feature.replace('_', ' ').title()}")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(approval_rates, use_container_width=True)
