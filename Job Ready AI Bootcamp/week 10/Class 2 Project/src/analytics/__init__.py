"""Aggregation of per-review SentimentResults into dashboard-ready summaries."""

from src.analytics.aggregator import aggregate, results_to_dataframe, summarize_dict

__all__ = ["aggregate", "results_to_dataframe", "summarize_dict"]
