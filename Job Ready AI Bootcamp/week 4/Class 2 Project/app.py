"""
TechNova Analytics — Statistical Insight Report Dashboard
============================================================
A production-grade Streamlit application for end-to-end 
statistical analysis of company data.

Run with: streamlit run app.py
"""
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="TechNova Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
    }
    .insight-box {
        background-color: #e8f4f8;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #2ca02c;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #ff7f0e;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📊 TechNova Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Statistical Insight Report — End-to-End Business Intelligence</div>', unsafe_allow_html=True)

st.markdown("""
## Welcome to the TechNova Analytics Dashboard

This application provides a **production-grade statistical analysis** of TechNova Solutions'
SaaS business data. Navigate through the sidebar to explore:

| Page | Description | Key Analyses |
|------|-------------|--------------|
| **🏠 Home** | Executive overview & KPIs | Business health snapshot |
| **📋 Data Overview** | Dataset exploration | Schema, distributions, quality checks |
| **📈 Descriptive Stats** | Central tendency & spread | Mean, median, std, percentiles |
| **🧪 Hypothesis Testing** | A/B tests & ANOVA | T-tests, p-values, effect sizes |
| **🔗 Correlation & Regression** | Relationships & predictions | Correlation matrix, scatter plots |
| **📑 Insight Report** | Executive summary | Actionable recommendations |

### About the Dataset
- **5,000 customers** across 4 regions and 6 industries
- **25,000 transactions** spanning 2023–2024
- **8,000 support tickets** with resolution metrics
- **4,000 A/B test participants** evaluating a new dashboard feature

**Use the sidebar to navigate →**
""")

st.divider()

st.markdown("""
### Quick Start Guide

1. **Data Overview** — Understand the structure and quality of each dataset
2. **Descriptive Statistics** — Explore distributions, outliers, and central tendencies
3. **Hypothesis Testing** — Validate business assumptions with statistical rigor
4. **Correlation & Regression** — Discover relationships between variables
5. **Insight Report** — Read the executive summary with actionable recommendations
""")

st.info("💡 **Tip:** All visualizations are interactive. Hover, zoom, and click to explore details.")
