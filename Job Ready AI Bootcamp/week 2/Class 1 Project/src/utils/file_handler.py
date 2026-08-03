"""
file_handler.py
---------------
Motive: Centralize all data I/O operations.
WHY: One point of change for paths, formats, and encoding.
WHAT IT DOES: Reads CSV/Excel/JSON, writes cleaned data, saves audit reports.
ANALOGY: The warehouse manager. Handles loading/unloading so workers focus on the job.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional


def _json_default(obj: Any) -> Any:
    """
    Converts NumPy/pandas scalars into plain Python types.

    WHY? pandas aggregations return np.int64/np.float64, which json.dump
    rejects. Without this, every audit report write crashes.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp, pd.Period)):
        return str(obj)
    if obj is pd.NaT:
        return None
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def read_csv(file_path: str, **kwargs) -> pd.DataFrame:
    """
    Reads a CSV file with robust error handling.

    WHY pd.read_csv? It handles encoding, delimiters, and type inference automatically.
    WHY **kwargs? Allows callers to pass custom options (delimiter, encoding) without
    modifying this function.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    return pd.read_csv(file_path, **kwargs)


def read_excel(file_path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """Reads an Excel file (common in enterprise environments)."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    return pd.read_excel(file_path, sheet_name=sheet_name)


def write_csv(df: pd.DataFrame, file_path: str, index: bool = False) -> None:
    """
    Writes DataFrame to CSV.

    WHY index=False? The index is often just row numbers (0, 1, 2...).
    Including it creates an unnamed column that confuses downstream tools.
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=index, encoding="utf-8")


def write_json_report(data: Dict[str, Any], file_path: str) -> None:
    """Writes an audit report as formatted JSON."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_json_default)
