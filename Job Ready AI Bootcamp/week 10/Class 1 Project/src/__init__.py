"""Semantic CV-to-Job Matcher — src package.

HIGHLIGHTS: this package deliberately contains NO import of
sentence-transformers at the top level (not here, not transitively through
any __init__.py). Importing this package — or any of its sub-packages —
must never trigger the ~61MB model download. The download only happens the
first time src.embeddings.service.get_model() is actually called, which in
this app is the first time a user clicks "Match" in the Streamlit UI. See
src/embeddings/service.py for the full lazy-loading rationale.
"""
