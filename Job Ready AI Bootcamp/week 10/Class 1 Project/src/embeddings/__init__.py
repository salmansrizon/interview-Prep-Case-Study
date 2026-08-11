"""Embeddings package: lazy-loaded sentence-transformers wrapper."""

from src.embeddings.service import embed, get_model

__all__ = ["embed", "get_model"]
