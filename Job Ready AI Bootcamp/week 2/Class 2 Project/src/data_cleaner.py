"""
data_cleaner.py
---------------
Motive: Implement production-grade data cleaning operations using Pandas.
WHY: 80% of a data scientist's time is spent cleaning data. In AI, dirty data
     produces dirty models. "Garbage in, garbage out" is not a cliché — it is
     the #1 cause of failed AI projects in industry.
WHAT IT DOES: Handles missing values, outliers, duplicates, type conversions,
              and categorical standardization — all with configurable rules.
ANALOGY: This is the *quality control department* of a factory. Raw materials
         (data) arrive with defects. QC inspects, fixes, or rejects them before
         they enter the production line (model training).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


class DataCleaner:
    """
    Configurable data cleaning pipeline.

    WHY a class? Encapsulates cleaning rules and state. You can create
    different cleaners for different datasets (customers vs products)
    without rewriting logic.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize with cleaning rules.

        WHY config-driven? Changing "max acceptable age" from 100 to 120
        should not require editing Python code. Config files let non-coders
        adjust rules safely.
        """
        self.config = config or {}
        self.report: Dict[str, any] = {
            "rows_before": 0,
            "rows_after": 0,
            "missing_fixed": 0,
            "outliers_removed": 0,
            "duplicates_removed": 0,
            "categories_standardized": 0,
        }

    def handle_missing(self, df: pd.DataFrame, strategy: str = "auto") -> pd.DataFrame:
        """
        Handles missing values with multiple strategies.

        STRATEGIES:
          - "drop": Remove rows with any missing values (nuclear option)
          - "mean": Fill numeric with column mean
          - "median": Fill numeric with column median (robust to outliers)
          - "mode": Fill categorical with most common value
          - "auto": Choose best strategy per column type

        WHY "auto"? Different columns need different treatment:
          - Age (numeric, roughly normal): use mean
          - Salary (numeric, skewed by executives): use median
          - Department (categorical): use mode

        VALID POINT: Dropping all rows with ANY missing value can delete 90%
        of your data. Imputation (filling) preserves sample size but introduces
        bias. There is no free lunch — you must choose based on your data.

        ANALOGY: Missing data is like a hole in a wall.
          - "drop" = demolish the whole wall
          - "mean" = patch with average-colored plaster (invisible but generic)
          - "median" = patch with middle-colored plaster (ignores extreme colors)
          - "mode" = patch with most common color (blends in best)
        """
        self.report["rows_before"] = len(df)
        df_clean = df.copy()

        missing_before = df_clean.isnull().sum().sum()

        if strategy == "drop":
            df_clean = df_clean.dropna()
        elif strategy == "auto":
            for col in df_clean.columns:
                if df_clean[col].isnull().any():
                    if pd.api.types.is_numeric_dtype(df_clean[col]):
                        # WHY median for skewed data? Mean is pulled by outliers.
                        # Median represents the "typical" value.
                        skewness = df_clean[col].skew()
                        if abs(skewness) > 1:  # Highly skewed
                            fill_value = df_clean[col].median()
                            print(f"  [MISSING] {col}: filled with median ({fill_value:.2f}) [skew={skewness:.2f}]")
                        else:
                            fill_value = df_clean[col].mean()
                            print(f"  [MISSING] {col}: filled with mean ({fill_value:.2f})")
                        df_clean[col] = df_clean[col].fillna(fill_value)
                    else:
                        # Categorical: use mode
                        mode_val = df_clean[col].mode()
                        if not mode_val.empty:
                            fill_value = mode_val[0]
                            print(f"  [MISSING] {col}: filled with mode ('{fill_value}')")
                            df_clean[col] = df_clean[col].fillna(fill_value)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        missing_after = df_clean.isnull().sum().sum()
        self.report["missing_fixed"] = missing_before - missing_after
        return df_clean

    def remove_outliers(self, df: pd.DataFrame, columns: List[str],
                       method: str = "iqr", threshold: float = 1.5) -> pd.DataFrame:
        """
        Removes statistical outliers from specified columns.

        METHODS:
          - "iqr": Interquartile Range. Values below Q1 - 1.5*IQR or above
                   Q3 + 1.5*IQR are outliers. Robust, non-parametric.
          - "zscore": Values with |z| > threshold are outliers. Assumes normality.

        WHY IQR? It does not assume normal distribution. Z-score fails on
        heavily skewed data (like salaries with a few millionaire executives).

        VALID POINT: Outlier removal is dangerous. A 150-year-old person is
        clearly an error. But a $500K salary might be a real CEO. Always
        inspect outliers before removing them. Domain knowledge > statistics.

        ANALOGY: IQR is like a nightclub bouncer who only lets in people
        between the 25th and 75th percentile of "coolness," plus a little wiggle
        room. Anyone way outside that range is rejected — but a genuine celebrity
        (valid extreme) might get wrongly bounced.
        """
        df_clean = df.copy()
        outliers_count = 0

        for col in columns:
            if not pd.api.types.is_numeric_dtype(df_clean[col]):
                continue

            if method == "iqr":
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR

                mask = (df_clean[col] >= lower) & (df_clean[col] <= upper)
                removed = (~mask).sum()
                outliers_count += removed

                if removed > 0:
                    print(f"  [OUTLIER] {col}: removed {removed} rows outside [{lower:.2f}, {upper:.2f}]")
                    df_clean = df_clean[mask]

            elif method == "zscore":
                mean = df_clean[col].mean()
                std = df_clean[col].std()
                z_scores = np.abs((df_clean[col] - mean) / std)
                mask = z_scores <= threshold
                removed = (~mask).sum()
                outliers_count += removed

                if removed > 0:
                    print(f"  [OUTLIER] {col}: removed {removed} rows with |z| > {threshold}")
                    df_clean = df_clean[mask]

        self.report["outliers_removed"] = outliers_count
        return df_clean

    def remove_duplicates(self, df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Removes duplicate rows.

        WHY subset? Sometimes only certain columns define uniqueness.
        A customer might have multiple orders (different order_id) but the
        same customer_id should not appear twice in a customer table.
        """
        before = len(df)
        df_clean = df.drop_duplicates(subset=subset, keep="first")
        removed = before - len(df_clean)
        self.report["duplicates_removed"] = removed
        if removed > 0:
            print(f"  [DUP] Removed {removed} duplicate rows")
        return df_clean

    def standardize_categories(self, df: pd.DataFrame, column: str,
                               mapping: Dict[str, str]) -> pd.DataFrame:
        """
        Standardizes categorical values using a mapping dictionary.

        WHY? Real data has typos and inconsistencies:
          "Sales", "sales", "SALES", "Salez" → all mean "Sales"
          "ENG", "Engineering", "engineer" → all mean "Engineering"

        Standardization ensures groupby operations and ML encoders work correctly.

        ANALOGY: A spell-checker for categories. It does not change meaning,
        just ensures consistent spelling so the computer understands.
        """
        df_clean = df.copy()
        before = df_clean[column].nunique()
        df_clean[column] = df_clean[column].replace(mapping)
        after = df_clean[column].nunique()
        standardized = before - after
        self.report["categories_standardized"] = standardized
        if standardized > 0:
            print(f"  [CAT] {column}: standardized {standardized} variants")
        return df_clean

    def convert_types(self, df: pd.DataFrame, type_map: Dict[str, str]) -> pd.DataFrame:
        """
        Converts columns to specified data types.

        WHY? Pandas often infers wrong types:
          - Dates as strings → cannot calculate durations
          - Booleans as strings → cannot filter
          - IDs as integers → meaningless math operations

        VALID POINT: Explicit type conversion prevents silent bugs. A column
        of "yes"/"no" strings will crash a logistic regression expecting 0/1.
        """
        df_clean = df.copy()
        for col, dtype in type_map.items():
            if col not in df_clean.columns:
                continue
            try:
                if dtype == "datetime":
                    # WHY errors='coerce'? Invalid dates become NaT (Not a Time)
                    # instead of crashing the entire conversion.
                    df_clean[col] = pd.to_datetime(df_clean[col], errors="coerce", infer_datetime_format=True)
                elif dtype == "bool":
                    # Handle multiple boolean representations
                    bool_map = {"yes": True, "no": False, "true": True, "false": False,
                               "1": True, "0": False, 1: True, 0: False}
                    df_clean[col] = df_clean[col].replace(bool_map).astype("boolean")
                else:
                    df_clean[col] = df_clean[col].astype(dtype)
                print(f"  [TYPE] {col} → {dtype}")
            except Exception as e:
                print(f"  [WARN] Could not convert {col} to {dtype}: {e}")
        return df_clean

    def get_report(self) -> Dict:
        """Returns the cleaning report."""
        self.report["rows_after"] = self.report.get("rows_before", 0) - (
            self.report.get("duplicates_removed", 0) +
            self.report.get("outliers_removed", 0)
        )
        return self.report
