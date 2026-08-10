"""
audit_reporter.py
-----------------
Motive: Generate human-readable audit reports from data cleaning operations.
WHY: In production, data transformations must be auditable. You need to prove
     to stakeholders (and yourself) what was changed and why.
WHAT IT DOES: Creates markdown and summary reports from cleaning logs.
ANALOGY: This is the "inspection certificate" for a used car. It documents
         every repair, replacement, and adjustment so the buyer knows exactly
         what they're getting.
"""

import pandas as pd
from pathlib import Path
from typing import Dict


class AuditReporter:
    """Generates audit reports from data cleaning operations."""

    def generate_report(
        self,
        original_df: pd.DataFrame,
        cleaned_df: pd.DataFrame,
        cleaning_log: pd.DataFrame,
        output_path: str = "reports/audit_report.md",
    ) -> str:
        """
        Generates a comprehensive markdown audit report.

        WHAT IT DOES: Compares before/after statistics and documents all changes.
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        report = f"""# Data Audit Report

## Summary
| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Rows | {len(original_df)} | {len(cleaned_df)} | {len(cleaned_df) - len(original_df)} |
| Columns | {len(original_df.columns)} | {len(cleaned_df.columns)} | {len(cleaned_df.columns) - len(original_df.columns)} |
| Missing Values | {original_df.isnull().sum().sum()} | {cleaned_df.isnull().sum().sum()} | {cleaned_df.isnull().sum().sum() - original_df.isnull().sum().sum()} |
| Duplicates | {original_df.duplicated().sum()} | {cleaned_df.duplicated().sum()} | {cleaned_df.duplicated().sum() - original_df.duplicated().sum()} |

## Cleaning Operations
{cleaning_log.to_markdown(index=False) if not cleaning_log.empty else "No cleaning operations performed."}

## Column Statistics (After Cleaning)
"""

        for col in cleaned_df.columns:
            if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                report += f"\n### {col}\n"
                report += f"- Type: {cleaned_df[col].dtype}\n"
                report += f"- Min: {cleaned_df[col].min():.2f}\n"
                report += f"- Max: {cleaned_df[col].max():.2f}\n"
                report += f"- Mean: {cleaned_df[col].mean():.2f}\n"
                report += f"- Missing: {cleaned_df[col].isnull().sum()}\n"

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)

        return report
