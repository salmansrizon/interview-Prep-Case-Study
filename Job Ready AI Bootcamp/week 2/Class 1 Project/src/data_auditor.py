"""
data_auditor.py
---------------
Motive: Build a production-grade data auditing system using Pandas.
WHY: In AI, data quality is model quality. This module automates the
     "janitorial work" that consumes 80% of an engineer's time.
WHAT IT DOES: Detects missing values, outliers, inconsistencies, and
              generates a comprehensive audit report.
ANALOGY: This is a "health inspector" for your data. It checks every
         column for diseases (NaN, outliers, wrong types) and writes
         a diagnosis report with recommended treatments.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple


class DataAuditor:
    """
    Automated data quality auditing and cleaning pipeline.

    WHY a class? Encapsulates configuration (thresholds, strategies) and
    state (original stats, cleaning log) in one object.
    WHAT IF functions? Every audit would need 5 parameters passed around.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cleaning_log: List[Dict[str, Any]] = []
        self.original_shape: Optional[Tuple[int, int]] = None

    def audit(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Performs a comprehensive data audit.

        WHAT IT DOES:
          1. Records original shape
          2. Computes missing value statistics
          3. Detects outliers per numeric column
          4. Checks data types vs. expected schema
          5. Identifies duplicate rows
          6. Summarizes categorical distributions

        RETURNS: A dictionary with all audit metrics.

        ANALOGY: A full medical checkup. Blood pressure, cholesterol,
                 heart rate — every vital sign is measured and recorded.
        """
        self.original_shape = df.shape
        report = {
            "audit_timestamp": pd.Timestamp.now().isoformat(),
            "original_shape": {"rows": df.shape[0], "columns": df.shape[1]},
            "columns": {},
            "overall": {},
        }

        # Per-column analysis
        for col in df.columns:
            report["columns"][col] = self._audit_column(df, col)

        # Overall statistics
        report["overall"] = {
            "total_cells": df.shape[0] * df.shape[1],
            "missing_cells": df.isnull().sum().sum(),
            "missing_pct": round(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100, 2),
            "duplicate_rows": df.duplicated().sum(),
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
        }

        return report

    def _audit_column(self, df: pd.DataFrame, col: str) -> Dict[str, Any]:
        """Audit a single column and return metrics."""
        series = df[col]
        audit = {
            "dtype": str(series.dtype),
            "missing_count": int(series.isnull().sum()),
            "missing_pct": round(series.isnull().mean() * 100, 2),
            "unique_count": int(series.nunique()),
            "memory_bytes": int(series.memory_usage(deep=True)),
        }

        # Numeric columns: compute statistics
        if pd.api.types.is_numeric_dtype(series):
            audit["stats"] = {
                "mean": float(series.mean()) if not series.isnull().all() else None,
                "median": float(series.median()) if not series.isnull().all() else None,
                "std": float(series.std()) if not series.isnull().all() else None,
                "min": float(series.min()) if not series.isnull().all() else None,
                "max": float(series.max()) if not series.isnull().all() else None,
            }

            # Outlier detection (IQR method)
            if not series.isnull().all():
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outliers = series[(series < lower) | (series > upper)]
                audit["outliers"] = {
                    "count": int(len(outliers)),
                    "pct": round(len(outliers) / len(series) * 100, 2),
                    "lower_bound": float(lower),
                    "upper_bound": float(upper),
                }

        # Categorical columns: show top values
        elif pd.api.types.is_string_dtype(series) or pd.api.types.is_categorical_dtype(series):
            audit["top_values"] = series.value_counts().head(5).to_dict()

        return audit

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies automated cleaning based on audit results and config.

        WHAT IT DOES:
          1. Drops columns with excessive missing values
          2. Drops duplicate rows
          3. Standardizes text and categorical labels
          4. Enforces data types
          5. Handles outliers (cap or remove)
          6. Imputes missing values (numeric: median, categorical: mode)

        RETURNS: Cleaned DataFrame.

        WHY THIS ORDER? Standardization and dtype enforcement both CREATE
        missing values ("not_a_date" -> NaT, "maybe" -> NA). Imputing first
        would leave those holes unfilled, so imputation runs last.

        ANALOGY: The treatment plan after diagnosis. The doctor prescribes
                 medicine (imputation), surgery (drop columns), and lifestyle
                 changes (standardization) based on the audit report.
        """
        df = df.copy()  # Never modify original
        cleaning_cfg = self.config.get("cleaning", {})

        # Step 1: Drop columns with too many missing values
        threshold = cleaning_cfg.get("drop_column_if_missing_pct", 0.50)
        missing_pcts = df.isnull().mean()
        cols_to_drop = missing_pcts[missing_pcts > threshold].index.tolist()
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            self._log_action("drop_columns", f"Dropped {len(cols_to_drop)} columns with >{threshold*100:.0f}% missing", cols_to_drop)

        # Step 2: Drop duplicate rows
        df = self._drop_duplicates(df, cleaning_cfg)

        # Step 3: Standardize text and categories
        df = self._standardize_data(df, cleaning_cfg)

        # Step 4: Enforce data types
        df = self._enforce_dtypes(df, cleaning_cfg)

        # Step 5: Handle outliers
        df = self._handle_outliers(df, cleaning_cfg)

        # Step 6: Impute missing values (last — earlier steps create NaN)
        df = self._impute_missing(df, cleaning_cfg)

        return df

    def _drop_duplicates(self, df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
        """
        Removes exact duplicate rows.

        WHY? Duplicates inflate group counts and bias every aggregate —
        a customer counted twice looks twice as valuable as they are.
        """
        if not cfg.get("drop_duplicates", True):
            return df

        n_dupes = df.duplicated().sum()
        if n_dupes:
            df = df.drop_duplicates().reset_index(drop=True)
            self._log_action("drop_duplicates", f"Dropped {n_dupes} duplicate rows", [])

        return df

    def _impute_missing(self, df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
        """Fill missing values based on column type."""
        numeric_strategy = cfg.get("impute_numeric", "median")
        cat_strategy = cfg.get("impute_categorical", "mode")
        constant = cfg.get("impute_constant", "UNKNOWN")

        for col in df.columns:
            if df[col].isnull().any():
                # Booleans report as numeric, but a median of 0.0 is not a
                # valid boolean — mode is the only sane fill for them.
                is_numeric = (
                    pd.api.types.is_numeric_dtype(df[col])
                    and not pd.api.types.is_bool_dtype(df[col])
                )
                if is_numeric:
                    if numeric_strategy == "median":
                        fill_val = df[col].median()
                    elif numeric_strategy == "mean":
                        fill_val = df[col].mean()
                    else:
                        fill_val = df[col].mode()[0] if not df[col].mode().empty else 0

                    df[col] = df[col].fillna(fill_val)
                    self._log_action("impute", f"Imputed {col} with {numeric_strategy}={fill_val}", [col])
                else:
                    if cat_strategy == "mode":
                        fill_val = df[col].mode()[0] if not df[col].mode().empty else constant
                    else:
                        fill_val = constant

                    df[col] = df[col].fillna(fill_val)
                    self._log_action("impute", f"Imputed {col} with '{fill_val}'", [col])

        return df

    def _handle_outliers(self, df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
        """Detect and handle outliers in numeric columns."""
        method = cfg.get("outlier_method", "none")
        if method == "none":
            return df

        action = cfg.get("outlier_action", "cap")

        for col in df.select_dtypes(include=[np.number]).columns:
            if method == "iqr":
                q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                iqr = q3 - q1
                multiplier = cfg.get("outlier_iqr_multiplier", 1.5)
                lower, upper = q1 - multiplier * iqr, q3 + multiplier * iqr
            elif method == "zscore":
                mean, std = df[col].mean(), df[col].std()
                threshold = cfg.get("outlier_zscore_threshold", 3.0)
                lower, upper = mean - threshold * std, mean + threshold * std
            else:
                continue

            if action == "cap":
                # Integer columns reject float bounds ("Invalid value '114.5'
                # for dtype 'Int64'"), so round the fence to whole numbers.
                if pd.api.types.is_integer_dtype(df[col]):
                    lower, upper = np.floor(lower), np.ceil(upper)
                df[col] = df[col].clip(lower, upper)
                self._log_action("cap_outliers", f"Capped {col} to [{lower:.2f}, {upper:.2f}]", [col])
            elif action == "remove":
                mask = (df[col] >= lower) & (df[col] <= upper)
                removed = (~mask).sum()
                df = df[mask]
                self._log_action("remove_outliers", f"Removed {removed} rows with outliers in {col}", [col])

        return df

    def _standardize_data(self, df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
        """
        Standardize text and categorical labels.

        WHY lowercase the map keys? Text standardization runs first, so by the
        time we map labels every value is already lowercase — a map keyed on
        "USA" would never match "usa" and silently do nothing.
        """
        # Case-insensitive lookup: "  USA " and "usa" both resolve to one key.
        cat_map = {
            str(k).strip().lower(): v
            for k, v in cfg.get("standardize_categories", {}).items()
        }

        # Identifiers must keep their original casing — lowercasing them
        # silently breaks every join key downstream.
        protected = set(cfg.get("preserve_case_columns", []))

        if cfg.get("standardize_text", False):
            for col in df.select_dtypes(include=["object", "string"]):
                if col in protected:
                    continue
                # .where() preserves real NaN — astype(str) alone turns them
                # into the literal string "nan", which imputation can't see.
                cleaned = df[col].astype(str).str.strip().str.lower()
                cleaned = cleaned.replace({"": None, "nan": None, "none": None})
                df[col] = cleaned.where(df[col].notna())

        for col in df.columns:
            if col in protected:
                continue
            if df[col].dtype == "object" or isinstance(df[col].dtype, pd.CategoricalDtype):
                normalized = df[col].map(
                    lambda v: cat_map.get(str(v).strip().lower(), v) if pd.notna(v) else v
                )
                remapped = (normalized != df[col]) & df[col].notna()
                if remapped.any():
                    self._log_action(
                        "standardize",
                        f"Standardized {remapped.sum()} labels in {col}",
                        [col],
                    )
                df[col] = normalized

        return df

    def _enforce_dtypes(self, df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
        """Enforce data types from config."""
        dtype_map = cfg.get("dtype_mapping", {})
        for col, dtype in dtype_map.items():
            if col in df.columns:
                try:
                    if dtype == "datetime64[ns]":
                        df[col] = pd.to_datetime(df[col], errors="coerce", format="mixed")
                    elif dtype == "category":
                        df[col] = df[col].astype("category")
                    elif dtype == "boolean":
                        df[col] = df[col].map(self._to_boolean).astype("boolean")
                    elif dtype in ("Int64", "int64"):
                        # WHY to_numeric first? The column arrives as object
                        # (floats + NaN), and a direct .astype("Int64") raises
                        # "cannot safely cast non-equivalent object to int64".
                        df[col] = pd.to_numeric(df[col], errors="coerce").round().astype("Int64")
                    elif dtype in ("Float64", "float64"):
                        df[col] = pd.to_numeric(df[col], errors="coerce").astype(dtype)
                    else:
                        df[col] = df[col].astype(dtype)
                    self._log_action("dtype", f"Converted {col} to {dtype}", [col])
                except Exception as e:
                    print(f"[WARN] Could not convert {col} to {dtype}: {e}")

        return df

    @staticmethod
    def _to_boolean(value: Any) -> Optional[bool]:
        """
        Maps the many spellings of true/false onto real booleans.

        WHY? Source systems export booleans as "yes", "1", "TRUE", or True.
        Anything unrecognized becomes NA so imputation can fill it later.
        """
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if pd.isna(value):
            return pd.NA

        token = str(value).strip().lower()
        if token in {"true", "1", "yes", "y", "t"}:
            return True
        if token in {"false", "0", "no", "n", "f"}:
            return False
        return pd.NA

    def _log_action(self, action: str, description: str, columns: List[str]) -> None:
        """Record every cleaning action for the audit trail."""
        self.cleaning_log.append({
            "action": action,
            "description": description,
            "columns": columns,
            "timestamp": pd.Timestamp.now().isoformat(),
        })

    def get_cleaning_report(self) -> Dict[str, Any]:
        """Generate a summary of all cleaning actions performed."""
        return {
            "total_actions": len(self.cleaning_log),
            "actions": self.cleaning_log,
            "actions_by_type": pd.DataFrame(self.cleaning_log)["action"].value_counts().to_dict() if self.cleaning_log else {},
        }
