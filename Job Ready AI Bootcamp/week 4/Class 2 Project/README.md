# 📊 TechNova Analytics — Statistical Insight Report Dashboard

A production-grade Streamlit application for end-to-end statistical analysis of SaaS business data.

## 🏗️ Project Structure

```
technova_analytics/
├── app.py                          # Main entry point
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── data/                           # Mock datasets
│   ├── customers.csv               # 5,000 customer records
│   ├── transactions.csv            # 25,000 transaction records
│   ├── support_tickets.csv         # 8,000 support ticket records
│   └── ab_test.csv                 # 4,000 A/B test participants
├── utils/                          # Reusable utilities
│   ├── __init__.py
│   ├── data_loader.py              # Data loading & preprocessing
│   ├── statistics.py               # Statistical test functions
│   └── visualizations.py           # Chart utilities
└── pages/                          # Streamlit multi-page app
    ├── 1_Data_Overview.py          # Schema, quality, distributions
    ├── 2_Descriptive_Statistics.py # Central tendency, spread, outliers
    ├── 3_Hypothesis_Testing.py     # A/B tests, t-tests, ANOVA, chi-square
    ├── 4_Correlation_Regression.py # Correlation matrix, scatter, regression
    └── 5_Insight_Report.py         # Executive summary & recommendations
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd technova_analytics
pip install -r requirements.txt
```

### 2. Run the Application

```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`.

## 📊 Features

| Module | Description | Statistical Methods |
|--------|-------------|---------------------|
| **Data Overview** | Schema exploration, quality checks, cross-dataset joins | Data profiling |
| **Descriptive Stats** | Central tendency, spread, distribution shape, outliers | Mean, median, std, IQR, percentiles, empirical rule |
| **Hypothesis Testing** | A/B test analysis, group comparisons, categorical associations | T-tests, ANOVA, chi-square, Cohen's d, bootstrap CI |
| **Correlation & Regression** | Relationship discovery, predictive modeling | Pearson/Spearman correlation, linear regression, R² |
| **Insight Report** | Executive summary with actionable recommendations | All methods combined |

## 🎯 Business Context

**Company:** TechNova Solutions (B2B SaaS)
**Datasets:**
- **Customers:** Demographics, plans, MRR, tenure, churn, NPS, feature adoption
- **Transactions:** Revenue events by customer and type
- **Support Tickets:** Ticket categories, priorities, resolution times, satisfaction
- **A/B Test:** New dashboard feature evaluation (Control vs Treatment)

## 📈 Statistical Techniques Covered

- **Central Limit Theorem** demonstration via sampling distributions
- **Descriptive Statistics:** Mean, median, mode, variance, standard deviation, IQR
- **Distribution Analysis:** Histograms, box plots, Q-Q plots, normality checks
- **Outlier Detection:** IQR method, Z-score method, impact analysis
- **Hypothesis Testing:** Null/alternative hypotheses, p-values, significance levels
- **T-Tests:** One-sample, two-sample (Welch's), paired
- **ANOVA:** One-way analysis of variance with post-hoc considerations
- **Chi-Square:** Tests of independence for categorical variables
- **Effect Sizes:** Cohen's d interpretation
- **Confidence Intervals:** 95% CI for all estimates
- **Correlation:** Pearson and Spearman coefficients
- **Linear Regression:** Predictive modeling with feature importance
- **Bootstrap:** Non-parametric confidence intervals

## 📝 Key Insights (Sample)

1. **Churn is predictable** — Chi-square test shows significant association with plan type
2. **A/B test is significant** — New dashboard improves engagement (p < 0.05)
3. **Regional gaps exist** — ANOVA confirms MRR differences across regions
4. **Support quality matters** — Critical tickets drive lower satisfaction

## 🔧 Customization

To use your own data:
1. Replace CSV files in `data/` with your datasets
2. Update `utils/data_loader.py` column names and preprocessing logic
3. Adjust analysis parameters in page scripts as needed

## 📚 Dependencies

- streamlit >= 1.28.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- scipy >= 1.11.0
- plotly >= 5.15.0
- scikit-learn >= 1.3.0

