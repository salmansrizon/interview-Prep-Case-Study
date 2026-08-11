"""Sentiment classification package: lazy-loaded HuggingFace pipeline wrapper.

Public API:
    get_classifier()      — returns the singleton pipeline, loading it on
                             first call only.
    classify(texts)       — classify a list of review strings, batched.
    SentimentResult        — dataclass holding label/score/needs_review per review.
"""

from src.sentiment.classifier import SentimentResult, classify, get_classifier

__all__ = ["SentimentResult", "classify", "get_classifier"]
