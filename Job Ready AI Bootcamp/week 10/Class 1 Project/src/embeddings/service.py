"""
Sentence-embedding service: wraps ``sentence-transformers`` behind a lazy,
singleton-loaded ``get_model()`` and a small ``embed()`` convenience
function.

HIGHLIGHTS: WHY LAZY LOADING, SPECIFICALLY?
`SentenceTransformer(...)` is NOT called at import time anywhere in this
module (or transitively via src/__init__.py, src/embeddings/__init__.py).
It is only constructed the first time ``get_model()`` actually runs, which
in this app happens the first time a user clicks "Match" in the Streamlit
UI. Three concrete reasons this matters, all stemming from the lecture's
section 4.2 "Local Model" framing:

  1. Cold browse / cold install cost. `sentence-transformers/
     paraphrase-MiniLM-L3-v2` is ~61MB and has to be downloaded from the
     HuggingFace Hub on first use. If we loaded it at import time, simply
     running `pytest`, `python -c "import app"`, or opening this file in
     an editor with a linter that imports the module would silently
     trigger a 61MB download — surprising and slow, especially offline or
     in CI. Import should be free; only *using* the feature should cost
     anything.
  2. "One download, then fully offline" is a feature, not an accident.
     Per the lecture: after the first successful load, sentence-transformers
     caches the model on disk (~/.cache/torch/sentence_transformers/) and
     every subsequent call is 100% local — no network, no per-call cost,
     unlike a cloud embedding API (e.g. OpenAI's text-embedding-3-small).
     Lazy loading is what makes "first run downloads, every run after that
     is offline" actually observable and true in this codebase, rather
     than just a claim in the README.
  3. Testability. tests/test_pipeline.py tests extraction and the ranking
     math (src/matching/ranker.py) using hand-crafted vectors — it never
     needs to load the real model, so tests stay fast and don't require
     internet access in CI. If the model loaded at import time, importing
     ANY module that (even indirectly) imports this one would force every
     test run to hit the network.

The mechanism used here — a module-level ``_model`` cache guarded by a
function — is the same singleton pattern week 7's ``config.get_config()``
uses for its Config object, just applied to a (much heavier) ML model
instead of a config dataclass.
"""

from __future__ import annotations

import numpy as np

from config import get_config
from src.utils.logger import get_logger

logger = get_logger("embeddings")

# Module-level cache. Starts as None so import time stays cheap (see
# module docstring); becomes a real SentenceTransformer instance after the
# first get_model() call, and is reused for the lifetime of the process.
_model = None


def get_model():
    """Return the shared SentenceTransformer instance, loading it on first use.

    HIGHLIGHTS — MODEL REFERENCE TABLE (adapted from Week 10 lecture, section 4.1):

    | Model                                  | Use Case                              | Advantage                              | Limitation                                            |
    |-----------------------------------------|----------------------------------------|------------------------------------------|--------------------------------------------------------|
    | `paraphrase-MiniLM-L3-v2` (used here)    | Fast, local semantic similarity        | Small (~61MB), fast on CPU               | Slightly less accurate than larger models              |
    | `all-MiniLM-L6-v2`                       | General-purpose sentence embeddings    | Good accuracy, still small               | ~50% larger than the L3 variant, a bit slower          |
    | `all-mpnet-base-v2`                      | High-accuracy semantic search          | Most accurate in this model family       | Much heavier (~420MB), slow on CPU                     |
    | OpenAI `text-embedding-3-small`          | Cloud-based embeddings                 | No local compute required                | Not local — per-call cost, data leaves the machine      |

    We use `paraphrase-MiniLM-L3-v2` because this is a portfolio-scale CV
    matcher: its accuracy is more than sufficient for ranking a handful of
    CVs against one job description, and it returns real-time results on a
    CPU with no GPU and no API key. A production system serving many
    concurrent users where every extra point of accuracy matters would be
    a reasonable place to upgrade to `all-mpnet-base-v2` instead — that
    tradeoff is a config change (see config.py's EmbeddingConfig), not a
    code change.

    HIGHLIGHTS — WHY IMPORT sentence_transformers INSIDE THIS FUNCTION,
    NOT AT THE TOP OF THE FILE? The `import` statement itself is cheap
    (it doesn't download anything), but keeping it local to this function
    is a visual/structural signal reinforcing the lazy-loading contract:
    anyone skimming this file's top-level imports sees no
    sentence_transformers import at all, so it's immediately obvious that
    importing THIS module cannot possibly trigger a download — only
    calling get_model() can.
    """
    global _model

    if _model is None:
        from sentence_transformers import SentenceTransformer

        cfg = get_config().embedding
        logger.info("Loading embedding model %r (first use — may download)...", cfg.model_name)
        _model = SentenceTransformer(cfg.model_name)
        logger.info("Embedding model loaded.")

    return _model


def embed(texts: list[str]) -> np.ndarray:
    """Embed a list of texts into an (N, embedding_dim) array of vectors.

    Args:
        texts: List of plain-text strings (already extracted/cleaned by
            src/extraction/text_extractor.py).

    Returns:
        A numpy array of shape (len(texts), embedding_dim), dtype float32.

    HIGHLIGHTS: this is the ONLY function in the whole project that calls
    `model.encode(...)`. app.py never touches SentenceTransformer directly
    — it calls `src.embeddings.service.embed()`, which is what keeps the
    "no direct sentence-transformers calls in app.py" boundary from the
    project spec intact, and what makes it possible to swap the embedding
    backend later (a different model, a different library entirely)
    without touching app.py at all.

    `convert_to_numpy=True` and `normalize_embeddings=False` are explicit
    (rather than relying on encode()'s defaults) because ranker.py performs
    its OWN normalization as part of computing cosine similarity — see
    src/matching/ranker.py's HIGHLIGHTS for why that manual dot-product
    path exists rather than trusting a pre-normalized embedding.
    """
    if not texts:
        return np.empty((0, get_config().embedding.embedding_dim), dtype=np.float32)

    model = get_model()
    cfg = get_config().embedding
    vectors = model.encode(
        texts,
        batch_size=cfg.encode_batch_size,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype=np.float32)
