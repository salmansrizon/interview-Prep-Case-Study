"""
Tests for the Local Customer Feedback Analyzer's surrounding logic:
CSV ingestion, aggregation, and confidence-threshold flagging.

HIGHLIGHTS: none of these tests import `transformers` or call
`get_classifier()`/`classify()` against the real model. Downloading and
running `cardiffnlp/twitter-roberta-base-sentiment-latest` takes real
time, real disk space, and real internet access — none of which a test
suite (or a CI runner) should depend on. Instead, everywhere a real model
prediction would normally appear, we hand-craft `SentimentResult` objects
directly (label/score/needs_review already decided by us) and test that
the surrounding code — column detection, missing-value handling,
percentage math, the flagging boundary rule — behaves correctly given
those results. `src/sentiment/classifier.py`'s `_needs_review()` boundary
logic is simple enough to unit test directly since it's pure arithmetic
with no model involved.

Run with: pytest tests/test_pipeline.py
"""

import pandas as pd

from src.analytics.aggregator import aggregate, results_to_dataframe, summarize_dict
from src.ingestion.csv_loader import extract_reviews, find_text_column, load_reviews_from_csv
from src.sentiment.classifier import SentimentResult, _needs_review


# ── CSV ingestion ────────────────────────────────────────────────────────


def test_find_text_column_matches_common_names_case_insensitively():
    for name in ["review", "Review", "REVIEW", "text", "Feedback", "review_text"]:
        df = pd.DataFrame({name: ["a", "b"], "id": [1, 2]})
        assert find_text_column(df) == name


def test_find_text_column_returns_none_when_no_match():
    df = pd.DataFrame({"customer_id": [1, 2], "rating": [5, 3]})
    assert find_text_column(df) is None


def test_find_text_column_prefers_first_candidate_in_priority_order():
    # "review" is earlier in the candidate list than "comment", so it
    # should win when both are present.
    df = pd.DataFrame({"comment": ["x"], "review": ["y"]})
    assert find_text_column(df) == "review"


def test_extract_reviews_drops_missing_and_empty_values():
    df = pd.DataFrame({"review": ["Great product!", None, "   ", "", "Not bad."]})
    reviews = extract_reviews(df, "review")
    assert reviews == ["Great product!", "Not bad."]


def test_extract_reviews_strips_whitespace():
    df = pd.DataFrame({"review": ["  padded text  "]})
    assert extract_reviews(df, "review") == ["padded text"]


def test_load_reviews_from_csv_autodetects_column(tmp_path):
    csv_path = tmp_path / "reviews.csv"
    pd.DataFrame({"id": [1, 2, 3], "feedback": ["Loved it", "", "Hated it"]}).to_csv(
        csv_path, index=False
    )
    reviews = load_reviews_from_csv(csv_path)
    assert reviews == ["Loved it", "Hated it"]


def test_load_reviews_from_csv_raises_when_column_undetectable(tmp_path):
    csv_path = tmp_path / "mystery.csv"
    pd.DataFrame({"id": [1], "notes": ["something"]}).to_csv(csv_path, index=False)
    try:
        load_reviews_from_csv(csv_path)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "Could not auto-detect" in str(exc)


# ── Confidence-threshold flagging boundary ──────────────────────────────


def test_needs_review_just_below_threshold_is_flagged():
    assert _needs_review(score=0.59, threshold=0.6) is True


def test_needs_review_exactly_at_threshold_is_not_flagged():
    # score == threshold counts as "confident enough" — see the HIGHLIGHTS
    # comment on _needs_review in classifier.py for why `<` (not `<=`)
    # was chosen, and why that boundary needs to be pinned by a test.
    assert _needs_review(score=0.6, threshold=0.6) is False


def test_needs_review_just_above_threshold_is_not_flagged():
    assert _needs_review(score=0.61, threshold=0.6) is False


def test_needs_review_respects_custom_threshold():
    # A stricter threshold (Brain Teaser #2: what if it's 0.9?) flags far
    # more predictions, including ones that would pass at the default.
    assert _needs_review(score=0.75, threshold=0.9) is True
    assert _needs_review(score=0.75, threshold=0.6) is False


# ── Aggregation ──────────────────────────────────────────────────────────


def _hand_crafted_results():
    return [
        SentimentResult(text="Loved it!", label="Positive", score=0.95, needs_review=False),
        SentimentResult(text="Broke immediately.", label="Negative", score=0.92, needs_review=False),
        SentimentResult(text="It's okay.", label="Neutral", score=0.55, needs_review=True),
        SentimentResult(text="Pretty good overall.", label="Positive", score=0.58, needs_review=True),
    ]


def test_results_to_dataframe_shape_and_columns():
    df = results_to_dataframe(_hand_crafted_results())
    assert list(df.columns) == ["text", "label", "score", "needs_review"]
    assert len(df) == 4


def test_results_to_dataframe_empty_input():
    df = results_to_dataframe([])
    assert len(df) == 0
    assert list(df.columns) == ["text", "label", "score", "needs_review"]


def test_summarize_dict_counts_and_percentages():
    summary = summarize_dict(_hand_crafted_results())
    assert summary["total"] == 4
    assert summary["counts"] == {"Positive": 2, "Negative": 1, "Neutral": 1}
    assert summary["percentages"]["Positive"] == 50.0
    assert summary["percentages"]["Negative"] == 25.0
    assert summary["percentages"]["Neutral"] == 25.0


def test_summarize_dict_flagged_count():
    summary = summarize_dict(_hand_crafted_results())
    # Two of the four hand-crafted results have needs_review=True.
    assert summary["flagged_count"] == 2
    assert summary["flagged_percentage"] == 50.0


def test_summarize_dict_handles_empty_list_without_crashing():
    summary = summarize_dict([])
    assert summary["total"] == 0
    assert summary["flagged_percentage"] == 0.0
    assert summary["average_confidence"] == 0.0


def test_aggregate_includes_dataframe_and_defaults_trend_to_none():
    result = aggregate(_hand_crafted_results())
    assert result["total"] == 4
    assert len(result["dataframe"]) == 4
    assert result["trend"] is None


def test_aggregate_builds_trend_when_dates_supplied():
    results = _hand_crafted_results()
    dates = ["2026-07-01", "2026-07-01", "2026-07-02", "2026-07-02"]
    result = aggregate(results, dates=dates)
    assert result["trend"] is not None
    assert set(result["trend"].columns) == {"Positive", "Negative", "Neutral"}
    assert len(result["trend"]) == 2  # two distinct days


def test_aggregate_degrades_gracefully_on_bad_dates():
    results = _hand_crafted_results()
    dates = ["not-a-date", "also-not-a-date", "nope", "nope"]
    result = aggregate(results, dates=dates)
    # All dates fail to parse -> no trend rows survive -> trend stays None
    # rather than the app crashing on an optional feature.
    assert result["trend"] is None
