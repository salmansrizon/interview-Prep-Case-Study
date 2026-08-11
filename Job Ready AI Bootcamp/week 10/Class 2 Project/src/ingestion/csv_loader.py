"""
CSV ingestion for batch review analysis.

HIGHLIGHTS: this module never imports `transformers` and never calls the
classifier. Its only job is turning "a CSV file" into "a clean list of
review strings" — a strict separation of concerns from
`src/sentiment/classifier.py`. That split means the CSV-handling logic
(finding the right column, skipping blank rows, guarding against a
missing column) can be unit-tested with plain pandas DataFrames in
`tests/test_pipeline.py`, with zero risk of accidentally triggering a
model download in CI or on a machine with no internet access.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# HIGHLIGHTS: candidate column names are checked in priority order and
# matched case-insensitively (`"Review"`, `"REVIEW"`, `"review"` all
# match). Real-world CSVs exported from spreadsheets, support ticket
# systems, or review-platform exports are inconsistent about casing and
# exact wording, so a strict `df["review"]` lookup would fail on a file
# that's clearly the right shape just because someone named the column
# "Feedback" or "Review Text". This list covers the common cases directly
# requested by the project spec; anything not on this list falls through
# to letting the caller (the Streamlit UI) pick the column explicitly
# rather than guessing wrong silently.
_CANDIDATE_COLUMN_NAMES = [
    "review",
    "text",
    "feedback",
    "review_text",
    "reviewtext",
    "comment",
    "comments",
]


def find_text_column(df: pd.DataFrame) -> Optional[str]:
    """Guess which column in `df` holds the review/feedback text.

    Matches column names case-insensitively against a small list of
    common names (`review`, `text`, `feedback`, etc. — see
    `_CANDIDATE_COLUMN_NAMES`). Returns the FIRST matching column's actual
    (original-case) name, or `None` if nothing matches, so the caller can
    fall back to asking the user to pick a column explicitly instead of
    guessing wrong.

    Args:
        df: The loaded CSV as a DataFrame.

    Returns:
        The original column name (preserving its case) if a match is
        found, otherwise `None`.
    """
    lowered_to_original = {str(c).strip().lower(): c for c in df.columns}
    for candidate in _CANDIDATE_COLUMN_NAMES:
        if candidate in lowered_to_original:
            return lowered_to_original[candidate]
    return None


def extract_reviews(df: pd.DataFrame, column: str) -> List[str]:
    """Pull a clean list of non-empty review strings out of `df[column]`.

    HIGHLIGHTS: missing values (`NaN`) and empty/whitespace-only strings
    are dropped here rather than passed downstream. This keeps
    `src/sentiment/classifier.py` simple (it can assume every string it
    receives from this function is real, non-blank text) and avoids
    wasting a model call on rows that have nothing to classify. If you
    need to preserve the original row count/alignment (e.g. to write
    results back next to a `date` column), do that reconciliation in the
    caller using the DataFrame's index — this function intentionally
    keeps a narrow, single responsibility: "text in, clean list of
    strings out".

    Args:
        df: The loaded CSV as a DataFrame.
        column: Name of the column containing review text.

    Returns:
        A list of stripped, non-empty review strings.

    Raises:
        KeyError: If `column` is not a column of `df`.
    """
    if column not in df.columns:
        raise KeyError(f"Column {column!r} not found in CSV columns: {list(df.columns)}")

    series = df[column].dropna().astype(str).str.strip()
    return [s for s in series.tolist() if s]


def load_reviews_from_csv(
    path_or_buffer,
    column: Optional[str] = None,
) -> List[str]:
    """Load a CSV and return a clean list of review strings.

    Args:
        path_or_buffer: A file path, or a file-like object (e.g. what
            Streamlit's `st.file_uploader` returns) — anything
            `pandas.read_csv` accepts.
        column: Explicit column name to use. If `None`, `find_text_column`
            is used to auto-detect one of the common names (review/text/
            feedback/...).

    Returns:
        A list of non-empty review strings, in file order.

    Raises:
        ValueError: If `column` is `None` and no candidate column name is
            found — the caller should catch this and prompt the user to
            pick a column explicitly (see app.py's Analyze tab).
    """
    df = pd.read_csv(path_or_buffer)

    resolved_column = column or find_text_column(df)
    if resolved_column is None:
        raise ValueError(
            "Could not auto-detect a review-text column. Expected one of "
            f"{_CANDIDATE_COLUMN_NAMES} (case-insensitive). Available "
            f"columns: {list(df.columns)}. Pass `column=` explicitly to "
            "choose one."
        )

    reviews = extract_reviews(df, resolved_column)
    logger.info(
        "Loaded %d non-empty reviews from column %r (%d total rows).",
        len(reviews),
        resolved_column,
        len(df),
    )
    return reviews
