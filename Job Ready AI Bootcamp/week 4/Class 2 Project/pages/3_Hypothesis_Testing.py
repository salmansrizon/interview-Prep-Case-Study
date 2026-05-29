import pandas as pd
# Streamlit framework for building interactive web applications
import streamlit as st
# NumPy library for numerical computations and array operations
import numpy as np
# SciPy library for statistical functions (t-tests, chi-square, ANOVA)
from scipy import stats
# Custom utility functions to load datasets from CSV files
from utils.data_loader import load_customers, load_ab_test, load_support_tickets
# Custom statistical functions for hypothesis testing
from utils.statistics import ab_test_summary, chi_square_test, anova_test
# Custom visualization functions for creating Plotly charts
from utils.visualizations import plotly_histogram, plotly_box, plotly_bar

# Configure Streamlit page settings: set title in browser tab, icon, and use wide layout for better data display
st.set_page_config(page_title="Hypothesis Testing", page_icon="🧪", layout="wide")

# Display main page title with icon
st.title("🧪 Hypothesis Testing")
# Brief description of the page purpose - covers key statistical tests
st.markdown("Statistical validation of business assumptions with A/B tests, t-tests, ANOVA, and chi-square tests.")

# Load data from CSV files using custom data loader functions
# Each function reads and preprocesses the respective dataset
customers = load_customers()        # Customer demographics, plans, churn status
ab_test = load_ab_test()            # A/B test results for dashboard feature
support = load_support_tickets()    # Support ticket data with resolution times

# Create four tabs for different hypothesis testing methods
tab1, tab2, tab3, tab4 = st.tabs(["🅰️🅱️ A/B Test Analysis", "📊 T-Test Comparisons", "🔬 ANOVA", "📋 Chi-Square Tests"])

# TAB 1: A/B Test Analysis (Two-sample t-test)
with tab1:
    st.subheader("A/B Test: New Dashboard Feature")
    # Explain the business context and hypotheses being tested
    st.markdown("""
    **Business Question:** Does the new dashboard design improve user engagement and conversion?

    **Hypotheses:**
    - **H₀:** The new dashboard has NO effect on engagement/conversion (μ_treatment = μ_control)
    - **H₁:** The new dashboard IMPROVES engagement/conversion (μ_treatment > μ_control)
    """)

    # Split data into control and treatment groups based on test_group column
    control = ab_test[ab_test['test_group'] == 'Control']
    treatment = ab_test[ab_test['test_group'] == 'Treatment']

    # Display group sizes in columns
    col1, col2, col3 = st.columns(3)
    col1.metric("Control Group", f"{len(control):,} users")      # Users seeing old dashboard
    col2.metric("Treatment Group", f"{len(treatment):,} users")  # Users seeing new dashboard
    col3.metric("Total Participants", f"{len(ab_test):,} users") # Total sample size

    # ENGAGEMENT SCORE ANALYSIS: Compare control vs treatment groups
    st.markdown("---")
    st.markdown("### 📊 Engagement Score Analysis")

    # Perform comprehensive A/B test analysis using custom function
    # Compares engagement scores between control and treatment groups
    eng_result = ab_test_summary(control['engagement_score'], treatment['engagement_score'], "Engagement Score")

    # Display key test statistics in columns
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Control Mean", f"{eng_result['control_mean']:.2f}")      # Average engagement for control
    col2.metric("Treatment Mean", f"{eng_result['treatment_mean']:.2f}")  # Average engagement for treatment
    col3.metric("Difference", f"{eng_result['mean_diff']:.2f}")          # Treatment - Control
    col4.metric("P-Value", f"{eng_result['p_value']:.4f}")               # Statistical significance

    # Display effect size metrics
    col5, col6, col7 = st.columns(3)
    col5.metric("Cohen's d", f"{eng_result['cohens_d']:.3f}")           # Standardized effect size
    col6.metric("Effect Size", eng_result['effect_size'])                 # Interpretation (Small/Medium/Large)
    col7.metric("Significant?", "✅ YES" if eng_result['significant'] else "❌ NO")

    # Display 95% confidence interval for the difference in means
    st.markdown(f"**95% Confidence Interval:** [{eng_result['ci_lower']:.3f}, {eng_result['ci_upper']:.3f}]")

    # Interpret results and provide business recommendation
    if eng_result['significant']:
        # Statistically significant: evidence suggests treatment is better
        st.success(f"""
        ✅ **STATISTICALLY SIGNIFICANT RESULT**

        The treatment group shows a **{eng_result['mean_diff']:.2f} point** higher engagement score 
        (p = {eng_result['p_value']:.4f}). The effect size is **{eng_result['effect_size']}** (Cohen's d = {eng_result['cohens_d']:.3f}).

        **Recommendation:** Roll out the new dashboard to all users.
        """)
    else:
        # Not significant: difference could be due to chance
        st.warning("""
        ⚠️ **NOT STATISTICALLY SIGNIFICANT**

        The observed difference could be due to random variation. 
        Consider increasing sample size or extending the test duration.
        """)

    # Visual comparison using box plot
    fig = plotly_box(ab_test, 'test_group', 'engagement_score', 'Engagement Score by Group')
    st.plotly_chart(fig, use_container_width=True)

    # CONVERSION RATE ANALYSIS: Compare conversion proportions between groups
    st.markdown("---")
    st.markdown("### 💰 Conversion Rate Analysis")

    # Calculate conversion rates (proportion of users who converted)
    conv_control = control['converted'].mean()    # Conversion rate for control group
    conv_treatment = treatment['converted'].mean() # Conversion rate for treatment group

    # TWO-PROPORTION Z-TEST: Compare conversion rates between two groups
    # Used when comparing proportions (e.g., conversion rates, success rates)
    n1, n2 = len(control), len(treatment)           # Sample sizes
    x1, x2 = control['converted'].sum(), treatment['converted'].sum()  # Number of successes
    p1, p2 = x1/n1, x2/n2                          # Sample proportions
    p_pool = (x1 + x2) / (n1 + n2)                # Pooled proportion under H₀
    # Standard error for difference in proportions
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    z = (p2 - p1) / se                              # Z-statistic
    p_val_conv = 1 - stats.norm.cdf(z)              # One-tailed p-value (treatment > control)

    # Display conversion rate metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Control Conv.", f"{p1*100:.2f}%")           # Control conversion %
    col2.metric("Treatment Conv.", f"{p2*100:.2f}%")         # Treatment conversion %
    col3.metric("Lift", f"{(p2-p1)*100:.2f} pp")           # Percentage point difference
    col4.metric("P-Value", f"{p_val_conv:.4f}")             # Statistical significance

    # Interpret results
    if p_val_conv < 0.05:
        st.success(f"✅ Conversion rate improvement is significant (p = {p_val_conv:.4f})")
    else:
        st.warning(f"⚠️ Conversion rate difference not significant (p = {p_val_conv:.4f})")

    # Visual comparison of conversion rates
    conv_df = pd.DataFrame({
        'Group': ['Control', 'Treatment'],
        'Conversion Rate (%)': [p1*100, p2*100]
    })
    fig2 = plotly_bar(conv_df, 'Group', 'Conversion Rate (%)', 'Conversion Rate by Group')
    st.plotly_chart(fig2, use_container_width=True)

# TAB 2: T-Test Comparisons (Two-sample t-tests)
with tab2:
    st.subheader("T-Test: Comparing Groups")

    # Dropdown to select which comparison to perform
    test_type = st.selectbox("Select Comparison", [
        "MRR: Churned vs Active Customers",
        "Resolution Time: Critical vs Non-Critical",
        "NPS: Enterprise vs Basic Plan"
    ])

    # Define groups based on selected comparison
    if test_type == "MRR: Churned vs Active Customers":
        # Compare Monthly Recurring Revenue between churned and active customers
        group_a = customers[customers['churned'] == True]['mrr']
        group_b = customers[customers['churned'] == False]['mrr']
        group_a_name, group_b_name = "Churned", "Active"
    elif test_type == "Resolution Time: Critical vs Non-Critical":
        # Compare support ticket resolution times by priority
        group_a = support[support['priority'] == 'Critical']['resolution_hours']
        group_b = support[support['priority'] != 'Critical']['resolution_hours']
        group_a_name, group_b_name = "Critical", "Non-Critical"
    else:
        # Compare NPS scores between Enterprise and Basic plan customers
        group_a = customers[customers['plan'] == 'Enterprise']['nps_score']
        group_b = customers[customers['plan'] == 'Basic']['nps_score']
        group_a_name, group_b_name = "Enterprise", "Basic"

    # Perform t-test analysis using custom function
    result = ab_test_summary(group_a, group_b, test_type)

    # Display group comparison header with sample sizes
    st.markdown(f"**{group_a_name}** (n={len(group_a):,}) vs **{group_b_name}** (n={len(group_b):,})")

    # Display key test statistics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(f"{group_a_name} Mean", f"{result['control_mean']:.2f}")
    col2.metric(f"{group_b_name} Mean", f"{result['treatment_mean']:.2f}")
    col3.metric("Difference", f"{result['mean_diff']:.2f}")
    col4.metric("P-Value", f"{result['p_value']:.4f}")

    # Display additional statistics
    st.markdown(f"**T-Statistic:** {result['t_statistic']:.4f} | **Cohen's d:** {result['cohens_d']:.3f} ({result['effect_size']})")
    st.markdown(f"**95% CI:** [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}]")

    # Interpret results
    if result['significant']:
        st.success("✅ Statistically significant difference detected!")
    else:
        st.info("ℹ️ No statistically significant difference detected.")

    # Distribution comparison
    compare_df = pd.DataFrame({
        'Value': list(group_a) + list(group_b),
        'Group': [group_a_name]*len(group_a) + [group_b_name]*len(group_b)
    })
    fig = plotly_histogram(compare_df, 'Value', f"Distribution: {test_type}", color_col='Group')
    st.plotly_chart(fig, use_container_width=True)

# TAB 3: ANOVA (Analysis of Variance) - Compare 3+ Groups
with tab3:
    st.subheader("ANOVA: Comparing Multiple Groups")

    # Dropdown to select which ANOVA comparison to perform
    anova_type = st.selectbox("Select ANOVA", [
        "MRR across Plans (Basic, Pro, Enterprise)",
        "Resolution Time across Priorities",
        "NPS across Regions"
    ])

    # Define groups based on selected comparison
    if anova_type == "MRR across Plans (Basic, Pro, Enterprise)":
        # Compare Monthly Recurring Revenue across subscription plans
        groups = [customers[customers['plan'] == p]['mrr'].values for p in ['Basic', 'Pro', 'Enterprise']]
        group_names = ['Basic', 'Pro', 'Enterprise']
    elif anova_type == "Resolution Time across Priorities":
        # Compare resolution times across ticket priorities
        groups = [support[support['priority'] == p]['resolution_hours'].values for p in ['Low', 'Medium', 'High', 'Critical']]
        group_names = ['Low', 'Medium', 'High', 'Critical']
    else:
        # Compare NPS scores across different regions
        groups = [customers[customers['region'] == r]['nps_score'].values for r in customers['region'].unique()]
        group_names = list(customers['region'].unique())

    # Perform one-way ANOVA test (compares means across 3+ groups)
    result = anova_test(*groups)

    # Display group names and test statistics
    st.markdown(f"**Groups:** {', '.join(group_names)}")
    st.markdown(f"**F-Statistic:** {result['f_statistic']:.4f}")  # Ratio of between-group to within-group variance
    st.markdown(f"**P-Value:** {result['p_value']:.6f}")          # Probability of observing F-stat if all means equal

    # Interpret ANOVA results
    if result['significant']:
        # Significant: At least one group mean differs from others
        st.success("✅ ANOVA is significant — at least one group differs from the others!")
        st.markdown("**Post-hoc analysis needed** to identify which specific groups differ.")
    else:
        # Not significant: No evidence that group means differ
        st.info("ℹ️ ANOVA is not significant — no evidence of group differences.")

    # Create summary table with group statistics
    means_data = []
    for name, group in zip(group_names, groups):
        means_data.append({'Group': name, 'Mean': np.mean(group), 'Std': np.std(group), 'N': len(group)})
    means_df = pd.DataFrame(means_data)
    st.dataframe(means_df, use_container_width=True)

# TAB 4: Chi-Square Test of Independence (Categorical Variables)
with tab4:
    st.subheader("Chi-Square Test: Categorical Associations")

    # Dropdown to select which categorical association to test
    chi_type = st.selectbox("Select Test", [
        "Plan vs Churn Status",
        "Priority vs Satisfaction",
        "Region vs Plan Type"
    ])

    # Perform chi-square test based on selected comparison
    # Chi-square tests whether two categorical variables are independent (no association)
    if chi_type == "Plan vs Churn Status":
        # Test if subscription plan type is associated with churn behavior
        result = chi_square_test(customers, 'plan', 'churned')
        title = "Plan Type vs Churn Status"
    elif chi_type == "Priority vs Satisfaction":
        # Test if ticket priority is associated with customer satisfaction
        result = chi_square_test(support, 'priority', 'satisfaction_category')
        title = "Ticket Priority vs Satisfaction"
    else:
        # Test if customer region is associated with subscription plan type
        result = chi_square_test(customers, 'region', 'plan')
        title = "Region vs Plan Type"

    # Display test results
    st.markdown(f"**{title}**")
    st.markdown(f"**Chi-Square:** {result['chi2']:.4f}")        # Test statistic (measures deviation from independence)
    st.markdown(f"**P-Value:** {result['p_value']:.6f}")        # Probability of observing this association by chance
    st.markdown(f"**Degrees of Freedom:** {result['dof']}")     # (rows-1) × (cols-1) in contingency table

    # Interpret chi-square results
    if result['significant']:
        # Significant: Variables are associated (not independent)
        st.success("✅ Significant association detected — variables are NOT independent!")
    else:
        # Not significant: No evidence of association (variables are independent)
        st.info("ℹ️ No significant association — variables appear independent.")

    # Display contingency table (observed frequencies for each combination)
    st.markdown("**Contingency Table:**")
    st.dataframe(result['contingency'], use_container_width=True)
