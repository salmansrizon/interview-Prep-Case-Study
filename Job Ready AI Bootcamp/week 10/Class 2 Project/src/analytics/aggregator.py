"""
Aggregation of a batch of `SentimentResult`s into dashboard-ready summaries.

HIGHLIGHTS: like `src/ingestion/csv_loader.py`, this module never imports
`transformers` and never calls the classifier — it only consumes
`SentimentResult` objects that some other layer already produced. That
means the aggregation math (counts, percentages, the flagged-for-review
count) can be exercised in `tests/test_pipeline.py` with hand-crafted
`SentimentResult` instances, with no model, no network, and no
dependency on what the real model would have predicted for any given
sentence. Keeping "what did the model say" and "what do we do with what
it said" in separate modules is what makes that possible.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from src.sentiment.classifier import SentimentResult

# Fixed label order so the dashboard's bar/pie charts and the summary dict
# always present labels in the same, predictable order (Positive, then
# Negative, then Neutral) regardless of which order they happened to
# appear in the input batch. A dict/DataFrame that reorders itself batch
# to batch would make two dashboard screenshots hard to compare at a
# glance.
LABEL_ORDER = ["Positive", "Negative", "Neutral"]


def results_to_dataframe(results: List[SentimentResult]) -> pd.DataFrame:
    """Convert a list of SentimentResult into a flat DataFrame.

    One row per review, columns: `text`, `label`, `score`, `needs_review`.
    This is the shape the Streamlit results table renders directly, and
    the shape `aggregate()` below groups over.
    """
    if not results:
        return pd.DataFrame(columns=["text", "label", "score", "needs_review"])

    return pd.DataFrame(
        {
            "text": [r.text for r in results],
            "label": [r.label for r in results],
            "score": [r.score for r in results],
            "needs_review": [r.needs_review for r in results],
        }
    )


def summarize_dict(results: List[SentimentResult]) -> Dict:
    """Build a plain-dict summary: per-label counts/percentages, flagged
    count, and totals. This is the shape used to drive the dashboard's
    headline metrics (`st.metric`) and bar/pie charts.

    HIGHLIGHTS: percentages are computed against `total`, and `total=0`
    is special-cased to avoid a ZeroDivisionError when the dashboard is
    rendered before any batch has been analyzed yet (Streamlit renders
    the Dashboard tab's layout even if no analysis has run in the
    session) — every percentage is simply 0.0 in that case rather than
    the app crashing.

    Returns:
        {
            "total": int,
            "counts": {"Positive": int, "Negative": int, "Neutral": int},
            "percentages": {"Positive": float, ...},  # 0-100, rounded to 1dp
            "flagged_count": int,
            "flagged_percentage": float,
            "average_confidence": float,
        }
    """
    total = len(results)
    counts = {label: 0 for label in LABEL_ORDER}
    flagged_count = 0
    score_sum = 0.0

    for r in results:
        # Any label outside the expected three still gets counted (under
        # its own key) rather than silently dropped — if a future model
        # swap introduces a label we didn't anticipate, we'd rather see
        # it show up oddly in the dashboard than vanish without a trace.
        counts[r.label] = counts.get(r.label, 0) + 1
        if r.needs_review:
            flagged_count += 1
        score_sum += r.score

    percentages = {
        label: round((count / total) * 100, 1) if total else 0.0
        for label, count in counts.items()
    }

    return {
        "total": total,
        "counts": counts,
        "percentages": percentages,
        "flagged_count": flagged_count,
        "flagged_percentage": round((flagged_count / total) * 100, 1) if total else 0.0,
        "average_confidence": round(score_sum / total, 3) if total else 0.0,
    }


def aggregate(
    results: List[SentimentResult],
    dates: List[str] | None = None,
) -> Dict:
    """Full aggregation: the `summarize_dict()` summary, plus an optional
    simple daily trend if per-review dates were supplied.

    Args:
        results: The classified reviews.
        dates: Optional list of date-like strings/values, same length and
            order as `results` (e.g. a `date` column pulled from the
            uploaded CSV). If given, an additional `"trend"` DataFrame is
            included: one row per calendar day, with columns
            `Positive`/`Negative`/`Neutral` counts for that day.

    Returns:
        The `summarize_dict()` output, plus:
            "dataframe": the full per-review DataFrame (see
                `results_to_dataframe`).
            "trend": a day-by-label count DataFrame, or `None` if `dates`
                was not supplied or couldn't be parsed as dates.

    HIGHLIGHTS: the trend feature is deliberately simple — a day x label
    pivot table, nothing more (no forecasting, no smoothing, no rolling
    averages). The project spec explicitly calls this piece "optional,
    keep it simple" — a student's time and this project's scope are
    better spent on the core classify -> confidence-check -> aggregate
    pipeline than on a bespoke time-series feature that most reviewers
    will only glance at once.
    """
    summary = summarize_dict(results)
    summary["dataframe"] = results_to_dataframe(results)
    summary["trend"] = None

    if dates and results and len(dates) == len(results):
        try:
            df = summary["dataframe"].copy()
            df["date"] = pd.to_datetime(pd.Series(dates), errors="coerce").dt.date
            df = df.dropna(subset=["date"])
            if not df.empty:
                trend = (
                    df.groupby(["date", "label"]).size().unstack(fill_value=0)
                )
                # Ensure all three label columns exist even if a given
                # batch happened to contain zero Neutral (etc.) reviews,
                # so the dashboard's line/bar chart doesn't error on a
                # missing column.
                for label in LABEL_ORDER:
                    if label not in trend.columns:
                        trend[label] = 0
                summary["trend"] = trend[LABEL_ORDER].sort_index()
        except Exception:
            # Malformed/unparseable date column: degrade gracefully to
            # "no trend available" rather than crashing the whole
            # dashboard over an optional feature.
            summary["trend"] = None

    return summary
