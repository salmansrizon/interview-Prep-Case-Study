"""
Cosine-similarity ranking: given a job-description embedding and a set of
CV embeddings, rank the CVs from most to least semantically similar.

HIGHLIGHTS: WHY COSINE SIMILARITY, NOT EUCLIDEAN DISTANCE OR RAW DOT PRODUCT?
This directly reuses the reasoning from the Week 10 lecture's section 4.3:
cosine similarity measures the ANGLE between two vectors — how much they
point in the same "semantic direction" — while ignoring their MAGNITUDE
(length). That distinction matters a lot for this project specifically:

  - A long, detailed CV and a short, terse job description will generally
    produce embeddings with different magnitudes purely because of text
    length/verbosity, even when they describe the exact same role. Raw dot
    product (`a @ b`, unnormalized) is sensitive to those magnitudes: a
    longer CV could score "more similar" than a shorter, better-matching
    one simply because its vector is longer, not because its content is
    more relevant. That would silently bias the ranking toward verbose
    CVs regardless of actual fit.
  - Euclidean distance has a related problem: it measures straight-line
    distance in embedding space, which is also magnitude-sensitive, and
    it produces a "lower is better" score that has to be inverted/rescaled
    to compare against an intuitive [-1, 1] similarity scale.
  - Cosine similarity divides the dot product by the product of both
    vectors' norms (`(a @ b) / (|a| * |b|)`), which cancels out length
    differences and leaves purely "how aligned are these two directions",
    on a bounded, easy-to-interpret [-1, 1] scale (in practice, sentence
    embeddings for related text land in roughly [0, 1]).

WHY COMPUTE THIS MANUALLY WITH NUMPY INSTEAD OF CALLING
`sentence_transformers.util.cos_sim` OR `sklearn.metrics.pairwise.
cosine_similarity`? Both would work fine here. We implement it by hand
with explicit `np.dot` / `np.linalg.norm` calls specifically because this
project's whole narrative arc — tokenization -> embeddings -> attention,
all the way back to the Week 2 NumPy dot-product lecture (see the Week 10
lecture's section 2 and 3.3) — is "the same dot-product math from Week 2,
just applied to sentence vectors instead of raw numbers/attention scores".
Hiding that behind a library call would obscure the exact connection this
project exists to teach.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.utils.logger import get_logger

logger = get_logger("matching")


@dataclass(frozen=True)
class MatchResult:
    """One CV's similarity result against the job description.

    HIGHLIGHTS: a small, explicit dataclass (rather than returning bare
    tuples or a dict) gives callers (app.py, tests) named, typed fields —
    `result.score` instead of `result[1]` — and makes it obvious at a
    glance what `rank_cvs` promises to return, which matters for a
    portfolio piece a student walks an interviewer through.
    """

    name: str
    score: float
    rank: int  # 1-based: 1 == best match


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1D vectors, computed by hand.

    similarity = (a . b) / (||a|| * ||b||)

    Returns a float in [-1, 1] (in practice, close to [0, 1] for related
    sentence embeddings). Returns 0.0 if either vector has zero magnitude
    (a degenerate edge case — e.g. an all-zero embedding — that would
    otherwise divide by zero).

    HIGHLIGHTS: see the module docstring for the full "why cosine, not
    Euclidean/raw dot product" reasoning. The zero-norm guard below isn't
    just defensive boilerplate — it's the one case where cosine similarity
    is mathematically undefined (0/0), and silently returning NaN into a
    ranking table would be a confusing bug for a student to debug later.
    """
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    # The Week 2 dot product, dividing out both vectors' magnitudes so
    # only their shared *direction* contributes to the score.
    dot_product = float(np.dot(a, b))
    return dot_product / (norm_a * norm_b)


def rank_cvs(
    job_desc_embedding: np.ndarray,
    cv_embeddings: np.ndarray,
    cv_names: list[str],
) -> list[MatchResult]:
    """Rank CVs by cosine similarity to a job-description embedding.

    Args:
        job_desc_embedding: 1D array, shape (embedding_dim,).
        cv_embeddings: 2D array, shape (n_cvs, embedding_dim).
        cv_names: Human-readable label for each row of ``cv_embeddings``,
            same length and order as ``cv_embeddings``.

    Returns:
        A list of ``MatchResult``, sorted by descending similarity score
        (best match first), with ``rank`` set to each result's 1-based
        position in that sorted order.

    Raises:
        ValueError: if ``cv_embeddings`` and ``cv_names`` have mismatched
            lengths.

    HIGHLIGHTS: this function is intentionally "dumb" — pure numpy math,
    no embedding-model calls, no Streamlit calls. That's what lets
    tests/test_pipeline.py exercise the ranking logic with tiny,
    hand-crafted vectors (e.g. orthogonal vectors -> score 0, identical
    vectors -> score 1) instead of needing the real ~61MB model, keeping
    tests fast and offline-safe (see src/embeddings/service.py's
    HIGHLIGHTS for why that separation matters).
    """
    cv_embeddings = np.asarray(cv_embeddings)
    if cv_embeddings.ndim == 1:
        # A single CV was passed as a 1D vector — treat it as one row.
        cv_embeddings = cv_embeddings.reshape(1, -1)

    if len(cv_names) != cv_embeddings.shape[0]:
        raise ValueError(
            f"cv_names has {len(cv_names)} entries but cv_embeddings has "
            f"{cv_embeddings.shape[0]} rows — they must match 1:1."
        )

    scores = [cosine_similarity(job_desc_embedding, cv_embeddings[i]) for i in range(cv_embeddings.shape[0])]

    # Sort indices by descending score. Using argsort on the negated
    # scores (rather than sort(reverse=True) on tuples) keeps ties broken
    # by original input order, which is a stable, predictable tie-break
    # for a UI table.
    order = np.argsort([-s for s in scores], kind="stable")

    results = [
        MatchResult(name=cv_names[i], score=scores[i], rank=rank)
        for rank, i in enumerate(order, start=1)
    ]

    if results:
        logger.info(
            "Ranked %d CVs; top match=%r score=%.4f",
            len(results), results[0].name, results[0].score,
        )
    return results
