"""Matching package: cosine-similarity ranking of CVs against a job description."""

from src.matching.ranker import MatchResult, cosine_similarity, rank_cvs

__all__ = ["MatchResult", "cosine_similarity", "rank_cvs"]
