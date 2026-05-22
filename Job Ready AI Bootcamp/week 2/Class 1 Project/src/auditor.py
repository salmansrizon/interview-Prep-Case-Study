"""
auditor.py
----------
Motive: Generate comprehensive data quality reports.
WHY: Before cleaning, you must understand WHAT is dirty. Blind cleaning
     destroys valid data. A data audit is the "diagnosis" before the "surgery."
WHAT IT DOES: Computes statistics, identifies issues, and generates a
              human-readable audit report — like a health checkup for data.
ANALOGY: This is a *medical diagnostic lab*. Before a doctor prescribes
         treatment (cleaning), they run blood tests (audits) to find the
         problem. You don't take antibiotics for a viral infection.
"""

import pandas as pd
import numpy as np
from typing import Dict, List


class DataAuditor:
    """
    Comprehensive data quality auditor.

    WHY separate from DataCleaner? Separation of concerns.
    Auditing (diagnosis) should not modify data. Cleaning (treatment) does.
    You audit first, then clean based on audit findings.
    """

    def __init__(self):
        self.issues: List[Dict] = []

    def audit(self, df: pd.DataFrame) -> Dict:
        """
        Performs a full data quality audit.

        RETURNS: Dictionary with shape, types, missing values, duplicates,
                 outliers, and categorical inconsistencies.
        """
        print("=" * 60)
        print("DATA QUALITY AUDIT REPORT")
        print("=" * 60)

        report = {
            "shape": df.shape,
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
            "columns": {}
        }

        print(f"
Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"Memory Usage: {report['memory_usage_mb']:.2f} MB")

        # Per-column analysis
        for col in df.columns:
            col_report = {
                "dtype": str(df[col].dtype),
                "missing_count": int(df[col].isnull().sum()),
                "missing_pct": float(df[col].isnull().mean() * 100),
                "unique_count": int(df[col].nunique(dropna=False)),
            }

            # Numeric columns: statistics
            if pd.api.types.is_numeric_dtype(df[col]):
                col_report.update({
                    "min": float(df[col].min()) if not df[col].isnull().all() else None,
                    "max": float(df[col].max()) if not df[col].isnull().all() else None,
                    "mean": float(df[col].mean()) if not df[col].isnull().all() else None,
                    "median": float(df[col].median()) if not df[col].isnull().all() else None,
                    "std": float(df[col].std()) if not df[col].isnull().all() else None,
                })

                # Outlier detection (IQR method)
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)][col]
                col_report["outlier_count"] = len(outliers)

                # Flag suspicious values
                if col_report["outlier_count"] > 0:
                    self.issues.append({
                        "column": col,
                        "issue": "outliers",
                        "details": f"{col_report['outlier_count']} outliers detected"
                    })

            # Categorical columns: value distribution
            else:
                top_values = df[col].value_counts().head(5).to_dict()
                col_report["top_values"] = {str(k): int(v) for k, v in top_values.items()}

                # Flag inconsistencies (mixed case, typos)
                unique_vals = df[col].dropna().astype(str).unique()
                lower_vals = [v.lower() for v in unique_vals]
                if len(set(lower_vals)) < len(unique_vals):
                    self.issues.append({
                        "column": col,
                        "issue": "inconsistent_case",
                        "details": f"Mixed case variants detected"
                    })

            # Flag high missing rates
            if col_report["missing_pct"] > 50:
                self.issues.append({
                    "column": col,
                    "issue": "high_missing",
                    "details": f"{col_report['missing_pct']:.1f}% missing"
                })

            report["columns"][col] = col_report

        # Dataset-level checks
        report["duplicate_rows"] = int(df.duplicated().sum())
        if report["duplicate_rows"] > 0:
            self.issues.append({
                "column": "ALL",
                "issue": "duplicates",
                "details": f"{report['duplicate_rows']} duplicate rows"
            })

        # Print summary
        print(f"
Issues Found: {len(self.issues)}")
        for issue in self.issues:
            print(f"  ⚠ {issue['column']}: {issue['issue']} — {issue['details']}")

        print("
" + "=" * 60)
        return report

    def generate_health_score(self, df: pd.DataFrame) -> float:
        """
        Computes an overall data health score (0-100).

        WHY? Stakeholders understand scores, not technical reports.
        "Your data health is 73/100" is more actionable than a 10-page PDF.

        SCORING:
          - Completeness (40%): 1 - (missing cells / total cells)
          - Uniqueness (20%): 1 - (duplicates / total rows)
          - Validity (20%): 1 - (outliers / total numeric values)
          - Consistency (20%): 1 - (inconsistent categories / total categories)
        """
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        completeness = 1 - (missing_cells / total_cells)

        uniqueness = 1 - (df.duplicated().sum() / len(df))

        # Validity: check numeric columns for outliers
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_ratio = 0
        if len(numeric_cols) > 0:
            total_numeric = len(df) * len(numeric_cols)
            total_outliers = 0
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
                total_outliers += len(outliers)
            outlier_ratio = total_outliers / total_numeric if total_numeric > 0 else 0
        validity = 1 - outlier_ratio

        # Consistency: check categorical columns for case variants
        cat_cols = df.select_dtypes(include=["object"]).columns
        inconsistency_ratio = 0
        if len(cat_cols) > 0:
            total_cats = sum(df[col].nunique() for col in cat_cols)
            inconsistent = 0
            for col in cat_cols:
                unique_vals = df[col].dropna().astype(str).unique()
                lower_vals = [v.lower() for v in unique_vals]
                inconsistent += len(unique_vals) - len(set(lower_vals))
            inconsistency_ratio = inconsistent / total_cats if total_cats > 0 else 0
        consistency = 1 - inconsistency_ratio

        score = (completeness * 0.4 + uniqueness * 0.2 +
                validity * 0.2 + consistency * 0.2) * 100

        print(f"
📊 DATA HEALTH SCORE: {score:.1f}/100")
        print(f"   Completeness:  {completeness*100:.1f}%")
        print(f"   Uniqueness:    {uniqueness*100:.1f}%")
        print(f"   Validity:      {validity*100:.1f}%")
        print(f"   Consistency:   {consistency*100:.1f}%")

        return score
