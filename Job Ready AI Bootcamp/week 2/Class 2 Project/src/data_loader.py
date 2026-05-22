"""
data_loader.py
--------------
Motive: Centralize all data ingestion operations.
WHY: Real-world data comes in many formats: CSV, Excel, JSON, databases.
     A single loader module handles format detection, encoding issues,
     and initial validation — preventing messy code scattered across notebooks.
WHAT IT DOES: Loads data from multiple sources, handles errors gracefully,
              and returns clean DataFrames ready for analysis.
ANALOGY: This is the *receiving dock* of a warehouse. Trucks (files) arrive
         in different sizes and conditions. The dock checks them in, logs
         issues, and passes them to the sorting area (cleaning pipeline).
"""

import pandas as pd
from pathlib import Path
from typing import Optional, List


class DataLoader:
    """
    Production-grade data loader supporting multiple formats.

    WHY a class? Holds configuration (file paths, encoding, dtypes) and
    state (loaded data cache). Multiple loaders can coexist for different sources.
    """

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self._cache: dict = {}

    def load_csv(self, filename: str, encoding: str = "utf-8") -> pd.DataFrame:
        """
        Loads a CSV file with robust error handling.

        WHY encoding parameter? CSVs from Windows often use latin-1 or cp1252.
        Default utf-8 fails on these. Specifying encoding prevents crashes.

        WHY low_memory=False? Pandas guesses dtypes per chunk. If a column has
        mixed types (strings in row 1000, ints in row 1), it warns. low_memory=False
        reads the full file first, then infers types — slower but more accurate.
        """
        path = self.data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        try:
            df = pd.read_csv(path, encoding=encoding, low_memory=False)
            print(f"[INFO] Loaded CSV: {filename} ({len(df)} rows, {len(df.columns)} columns)")
            return df
        except UnicodeDecodeError:
            # WHY fallback? If utf-8 fails, try latin-1 which accepts any byte.
            print(f"[WARN] UTF-8 failed for {filename}, trying latin-1...")
            return pd.read_csv(path, encoding="latin-1", low_memory=False)

    def load_excel(self, filename: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
        """
        Loads an Excel file.

        WHY sheet_name? Excel files often have multiple sheets (raw, clean, summary).
        Specifying the sheet prevents loading the wrong data.
        """
        path = self.data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        df = pd.read_excel(path, sheet_name=sheet_name)
        print(f"[INFO] Loaded Excel: {filename} ({len(df)} rows, {len(df.columns)} columns)")
        return df

    def generate_dirty_data(self, filename: str = "dirty_dataset.csv", n_rows: int = 1000) -> str:
        """
        Generates intentionally dirty data for teaching purposes.

        WHY dirty data? Students must learn to handle real-world mess:
        - Missing values (NaN)
        - Outliers (impossible ages, negative salaries)
        - Duplicates
        - Inconsistent formats (dates as strings, mixed currencies)
        - Typos in categorical data

        ANALOGY: A flight simulator creates turbulence and engine failures so
                 pilots learn to handle emergencies. This data creates "turbulence"
                 so data engineers learn to clean it.
        """
        import numpy as np

        np.random.seed(42)  # WHY? Reproducible randomness. Same seed = same data.

        data = {
            "customer_id": range(1, n_rows + 1),
            "name": [f"Customer_{i}" for i in range(n_rows)],
            "age": np.random.choice(
                list(range(18, 80)) + [np.nan, 150, -5, 999],  # Outliers and NaN
                n_rows
            ),
            "salary": np.random.choice(
                list(range(30000, 150000, 5000)) + [np.nan, 500000, -10000, 0],
                n_rows
            ),
            "department": np.random.choice(
                ["Sales", "Engineering", "Marketing", "sales", "ENG", "marketing", np.nan, "Unknown"],
                n_rows
            ),
            "join_date": np.random.choice(
                ["2023-01-15", "15/03/2023", "2023-06-30", "July 2023", np.nan, "2025-01-01"],
                n_rows
            ),
            "satisfaction_score": np.random.choice(
                list(range(1, 11)) + [np.nan, 99, -1, 0],
                n_rows
            ),
            "is_active": np.random.choice(
                [True, False, "yes", "no", 1, 0, np.nan],
                n_rows
            ),
        }

        df = pd.DataFrame(data)

        # Add duplicates
        duplicate_rows = df.sample(50, random_state=42)
        df = pd.concat([df, duplicate_rows], ignore_index=True)

        # Add some completely empty rows
        empty_rows = pd.DataFrame(np.nan, index=range(20), columns=df.columns)
        df = pd.concat([df, empty_rows], ignore_index=True)

        # Shuffle
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        output_path = self.data_dir / filename
        self.data_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        print(f"[INFO] Generated dirty dataset: {output_path}")
        print(f"[INFO] Issues injected: NaN, outliers, duplicates, inconsistent formats, typos")
        return str(output_path)
