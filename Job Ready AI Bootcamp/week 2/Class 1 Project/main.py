"""
main.py
-------
Motive: Single entry point for the Data Audit Pipeline.
WHAT IT DOES: Creates intentionally dirty data, runs full audit, cleans it,
              generates reports, and demonstrates GroupBy + Merge operations.
ANALOGY: The quality control manager. Inspects raw materials, rejects defects,
         approves clean batches, and files compliance reports.
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import yaml

# WHY THIS BLOCK? So `python main.py` works no matter which directory you
# launch it from. PROJECT_ROOT is the folder holding this file; every path
# below is built from it instead of from your shell's current directory.
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_auditor import DataAuditor
from src.utils.file_handler import read_csv, write_csv, write_json_report


def create_dirty_dataset() -> pd.DataFrame:
    """
    Generates an intentionally messy dataset for the audit demo.

    WHY? Students must see REAL problems to understand cleaning.
    This dataset contains every common data quality issue:
      - Missing values (NaN)
      - Outliers (impossible ages, negative incomes)
      - Inconsistent categories ("USA", "US", "U.S.")
      - Wrong data types (dates as strings, booleans as strings)
      - Duplicates
      - Whitespace and case issues
    """
    np.random.seed(42)  # Reproducibility

    n = 200
    data = {
        "customer_id": [f"CUST_{i:04d}" for i in range(n)],
        "age": np.concatenate([
            np.random.randint(18, 80, size=180),
            [150, 200, -5, 999],  # Outliers
            [np.nan] * 16         # Missing
        ]),
        "income": np.concatenate([
            np.random.randint(20000, 150000, size=170),
            [5000000, -10000, 9999999],  # Outliers
            [np.nan] * 27        # Missing
        ]),
        "signup_date": np.concatenate([
            pd.date_range("2020-01-01", periods=160, freq="W").tolist(),
            ["not_a_date", "2021-13-45", ""] * 10,  # Invalid dates
            [np.nan] * 10
        ]),
        "country": np.concatenate([
            np.random.choice(["USA", "US", "U.S.", "UK", "U.K.", "GB", "Canada", "Germany", "France"], size=170),
            ["  usa  ", "UNITED STATES", "u.k.", ""] * 5,  # Inconsistent
            [np.nan] * 10
        ]),
        "is_active": np.concatenate([
            np.random.choice([True, False], size=170),
            ["true", "false", "1", "0", "yes", "no"] * 4,  # Wrong type
            [np.nan] * 6
        ]),
        "category": np.concatenate([
            np.random.choice(["premium", "standard", "basic"], size=185),
            [np.nan] * 15
        ]),
        "purchase_amount": np.concatenate([
            np.random.exponential(100, size=175),
            [50000, 99999, -500] * 3,  # Outliers
            [np.nan] * 16
        ]),
    }

    df = pd.DataFrame(data)

    # Add some duplicate rows
    duplicates = df.sample(10, random_state=42)
    df = pd.concat([df, duplicates], ignore_index=True)

    return df


def create_secondary_dataset() -> pd.DataFrame:
    """
    Creates a second dataset to demonstrate merging.

    WHY merge demo? Real AI uses data from multiple sources (CRM + ERP + web analytics).
    Students must learn to combine datasets safely.
    """
    np.random.seed(43)

    # 150 customers overlap with main dataset, 50 are new
    customer_ids = [f"CUST_{i:04d}" for i in range(150)] + [f"NEW_{i:04d}" for i in range(50)]

    data = {
        "customer_id": customer_ids,
        "last_login": pd.date_range("2024-01-01", periods=200, freq="D"),
        "session_count": np.random.poisson(10, 200),
        "total_clicks": np.random.poisson(100, 200),
        "device": np.random.choice(["mobile", "desktop", "tablet"], size=200),
    }

    return pd.DataFrame(data)


def main():
    print("=" * 70)
    print("DATA AUDIT PIPELINE")
    print("Week 2, Class 3 — Data Orchestration with Pandas")
    print("=" * 70)

    # Load config
    # WHY ["audit"]? Every rule lives under the top-level `audit:` key,
    # and DataAuditor expects `cleaning` at the root of what it receives.
    with open(PROJECT_ROOT / "config" / "settings.yaml", "r") as f:
        config = yaml.safe_load(f)["audit"]

    # ============================================================
    # PART 1: CREATE AND INSPECT DIRTY DATA
    # ============================================================
    print("\n[PART 1] Creating intentionally dirty dataset...")
    df = create_dirty_dataset()
    print(f"         Shape: {df.shape}")
    print(f"         Memory: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")

    print("\n         First 3 rows (raw):")
    print(df.head(3).to_string())

    # ============================================================
    # PART 2: RUN THE AUDIT
    # ============================================================
    print("\n[PART 2] Running comprehensive data audit...")
    auditor = DataAuditor(config)
    audit_report = auditor.audit(df)

    print(f"\n         Overall Health:")
    print(f"         - Total cells: {audit_report['overall']['total_cells']:,}")
    print(f"         - Missing cells: {audit_report['overall']['missing_cells']:,} ({audit_report['overall']['missing_pct']}%)")
    print(f"         - Duplicate rows: {audit_report['overall']['duplicate_rows']}")
    print(f"         - Memory usage: {audit_report['overall']['memory_usage_mb']} MB")

    print(f"\n         Column-level issues:")
    for col, metrics in audit_report['columns'].items():
        issues = []
        if metrics['missing_pct'] > 0:
            issues.append(f"{metrics['missing_pct']}% missing")
        if 'outliers' in metrics and metrics['outliers']['count'] > 0:
            issues.append(f"{metrics['outliers']['count']} outliers")
        if issues:
            print(f"         - {col}: {', '.join(issues)}")

    # Save audit report
    write_json_report(audit_report, PROJECT_ROOT / "data" / "audit_reports" / "audit_raw.json")
    print("\n         ✓ Audit report saved to data/audit_reports/audit_raw.json")

    # ============================================================
    # PART 3: CLEAN THE DATA
    # ============================================================
    print("\n[PART 3] Applying automated cleaning...")
    df_clean = auditor.clean(df)

    print(f"\n         Cleaning actions performed:")
    cleaning_report = auditor.get_cleaning_report()
    for action in cleaning_report['actions']:
        print(f"         - {action['action']}: {action['description']}")

    print(f"\n         Cleaned shape: {df_clean.shape}")
    print(f"         Missing values after cleaning: {df_clean.isnull().sum().sum()}")

    # Save cleaned data
    write_csv(df_clean, PROJECT_ROOT / "data" / "clean" / "customers_clean.csv")
    print("\n         ✓ Cleaned data saved to data/clean/customers_clean.csv")

    # ============================================================
    # PART 4: GROUPBY ANALYSIS
    # ============================================================
    print("\n[PART 4] GroupBy analysis — understanding segments...")

    # Group by category and compute statistics
    category_stats = df_clean.groupby("category").agg({
        "age": ["mean", "median", "std"],
        "income": ["mean", "median"],
        "purchase_amount": ["mean", "sum", "count"],
    }).round(2)

    print("\n         Statistics by customer category:")
    print(category_stats.to_string())

    # Group by country
    country_stats = df_clean.groupby("country").agg({
        "income": "mean",
        "purchase_amount": "mean",
        "customer_id": "count",
    }).round(2)
    country_stats = country_stats.rename(columns={"customer_id": "customer_count"})
    country_stats = country_stats.sort_values("customer_count", ascending=False)

    print("\n         Top countries by customer count:")
    print(country_stats.head(5).to_string())

    # ============================================================
    # PART 5: MERGE DEMONSTRATION
    # ============================================================
    print("\n[PART 5] Merging with secondary dataset...")
    df_secondary = create_secondary_dataset()
    print(f"         Secondary dataset shape: {df_secondary.shape}")

    # Inner join: only customers present in BOTH datasets
    merged_inner = pd.merge(df_clean, df_secondary, on="customer_id", how="inner")
    print(f"\n         Inner merge result: {merged_inner.shape} (customers in both datasets)")

    # Left join: all customers from main dataset + matching secondary data
    merged_left = pd.merge(df_clean, df_secondary, on="customer_id", how="left")
    print(f"         Left merge result: {merged_left.shape} (all customers + matching sessions)")

    # Show merged sample
    print("\n         Sample of merged data:")
    sample_cols = ["customer_id", "age", "income", "country", "session_count", "device"]
    print(merged_left[sample_cols].head(5).to_string())

    # Save merged data
    write_csv(merged_left, PROJECT_ROOT / "data" / "clean" / "customers_merged.csv")
    print("\n         ✓ Merged data saved to data/clean/customers_merged.csv")

    # ============================================================
    # PART 6: ADVANCED ANALYSIS
    # ============================================================
    print("\n[PART 6] Advanced analysis on merged data...")

    # Pivot: average income by device and category
    pivot = merged_left.pivot_table(
        values="income",
        index="category",
        columns="device",
        aggfunc="mean"
    ).round(0)

    print("\n         Average income by customer category and device:")
    print(pivot.to_string())

    # Correlation analysis
    numeric_cols = ["age", "income", "purchase_amount", "session_count", "total_clicks"]
    corr = merged_left[numeric_cols].corr().round(2)

    print("\n         Correlation matrix (what predicts what?):")
    print(corr.to_string())

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 70)
    print("AUDIT PIPELINE COMPLETE")
    print("=" * 70)
    print(f"✓ Raw data: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"✓ Clean data: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")
    print(f"✓ Cleaning actions: {len(cleaning_report['actions'])}")
    print(f"✓ Merged data: {merged_left.shape[0]} rows, {merged_left.shape[1]} columns")
    print(f"✓ Reports saved to: data/audit_reports/")
    print("=" * 70)


if __name__ == "__main__":
    main()
