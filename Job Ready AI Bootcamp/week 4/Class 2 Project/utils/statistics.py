"""
Statistical analysis utilities for hypothesis testing and inference.
Contains functions for t-tests, ANOVA, chi-square tests, and effect size calculations.
"""
# NumPy library for numerical computations and array operations
import numpy as np
# Pandas library for data manipulation and DataFrame operations
import pandas as pd
# SciPy library for statistical functions
from scipy import stats
# Specific statistical tests from SciPy
from scipy.stats import ttest_ind, chi2_contingency, f_oneway, pearsonr

def cohens_d(x, y):
    """
    Calculate Cohen's d effect size.
    Measures standardized difference between two group means.
    Formula: d = (mean₁ - mean₂) / pooled_standard_deviation
    Interpretation: 0.2=Small, 0.5=Medium, 0.8=Large effect
    """
    nx = len(x)  # Sample size of group x (treatment)
    ny = len(y)  # Sample size of group y (control)
    dof = nx + ny - 2  # Degrees of freedom for two-sample t-test
    # Pooled standard deviation (weighted average of both group variances)
    pooled_std = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / dof)
    # Cohen's d: standardized mean difference
    return (x.mean() - y.mean()) / pooled_std

def interpret_cohens_d(d):
    """
    Interpret Cohen's d magnitude using standard benchmarks.
    Cohen's guidelines: 0.2=Small, 0.5=Medium, 0.8=Large
    """
    d = abs(d)  # Use absolute value (direction doesn't matter for magnitude)
    if d < 0.2:
        return "Negligible"  # Practically no effect
    elif d < 0.5:
        return "Small"       # Small but potentially meaningful
    elif d < 0.8:
        return "Medium"       # Medium/moderate effect
    else:
        return "Large"        # Large/practically significant effect

def ab_test_summary(control, treatment, metric_name="Metric"):
    """
    Perform comprehensive A/B test analysis using Welch's t-test (unequal variances).
    Returns dictionary with means, p-value, confidence interval, and effect size.
    """
    # Welch's t-test: does not assume equal variances between groups
    t_stat, p_val = ttest_ind(treatment, control, equal_var=False)

    # Calculate difference in means (treatment - control)
    mean_diff = treatment.mean() - control.mean()
    # Standard error of the difference (for CI calculation)
    se_diff = np.sqrt(treatment.var(ddof=1)/len(treatment) + control.var(ddof=1)/len(control))
    # 95% Confidence Interval for the difference in means
    ci_lower = mean_diff - 1.96 * se_diff
    ci_upper = mean_diff + 1.96 * se_diff

    # Calculate Cohen's d (standardized effect size)
    d = cohens_d(treatment, control)

    # Return comprehensive results dictionary
    return {
        'metric': metric_name,
        'control_mean': control.mean(),      # Average for control group
        'treatment_mean': treatment.mean(),   # Average for treatment group
        'mean_diff': mean_diff,              # Treatment - Control
        't_statistic': t_stat,               # t-statistic from Welch's test
        'p_value': p_val,                    # Probability of observing this difference by chance
        'ci_lower': ci_lower,                # Lower bound of 95% CI
        'ci_upper': ci_upper,                # Upper bound of 95% CI
        'cohens_d': d,                       # Standardized effect size
        'effect_size': interpret_cohens_d(d), # Interpretation (Small/Medium/Large)
        'significant': p_val < 0.05,        # Is result statistically significant?
        'control_n': len(control),           # Sample size of control group
        'treatment_n': len(treatment)         # Sample size of treatment group
    }

def chi_square_test(df, col1, col2):
    """
    Perform chi-square test of independence between two categorical variables.
    H₀: Variables are independent (no association)
    H₁: Variables are associated (not independent)
    """
    # Create contingency table (observed frequencies for each combination)
    contingency = pd.crosstab(df[col1], df[col2])
    # Chi-square test: compares observed vs expected frequencies under H₀
    chi2, p, dof, expected = chi2_contingency(contingency)
    return {
        'chi2': chi2,           # Chi-square test statistic (measures deviation from independence)
        'p_value': p,           # Probability of observing this association by chance
        'dof': dof,             # Degrees of freedom = (rows-1) × (cols-1)
        'contingency': contingency, # Contingency table (observed frequencies)
        'significant': p < 0.05   # Is the association statistically significant?
    }

def anova_test(*groups):
    """
    Perform one-way ANOVA (Analysis of Variance).
    Compares means across 3+ groups.
    H₀: All group means are equal
    H₁: At least one group mean differs
    """
    # f_oneway performs F-test comparing between-group to within-group variance
    f_stat, p_val = f_oneway(*groups)
    return {
        'f_statistic': f_stat,    # F-statistic (ratio of variances)
        'p_value': p_val,        # Probability of observing F-stat if all means equal
        'significant': p_val < 0.05  # Is there a significant difference?
    }

def bootstrap_ci(data, n_bootstrap=10000, ci=95):
    """
    Calculate bootstrap confidence interval (non-parametric, no normality assumption).
    Resamples data with replacement to estimate sampling distribution of the mean.
    Returns: (lower_bound, upper_bound, bootstrap_samples)
    """
    boot_means = []        # Store bootstrap sample means
    n = len(data)         # Original sample size
    # Generate n_bootstrap bootstrap samples (default: 10,000)
    for _ in range(n_bootstrap):
        # Resample with replacement (same size as original)
        sample = np.random.choice(data, size=n, replace=True)
        boot_means.append(sample.mean())  # Calculate mean of bootstrap sample
    boot_means = np.array(boot_means)
    # Calculate percentile-based confidence interval
    # For 95% CI: 2.5th percentile (lower) and 97.5th percentile (upper)
    lower = np.percentile(boot_means, (100 - ci) / 2)
    upper = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return lower, upper, boot_means
