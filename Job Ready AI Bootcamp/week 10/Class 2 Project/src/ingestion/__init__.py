"""CSV ingestion: locating and extracting the review-text column."""

from src.ingestion.csv_loader import find_text_column, load_reviews_from_csv

__all__ = ["find_text_column", "load_reviews_from_csv"]
