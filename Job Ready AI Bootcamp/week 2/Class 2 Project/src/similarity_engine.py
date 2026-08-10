"""
similarity_engine.py
--------------------
Motive: Search a corpus by meaning, using one matrix multiplication.
WHY: This is Topic 3 of the lecture. Dot product measures alignment; matmul
     does every pairwise dot product at once. That is the whole of semantic
     search, and the same operation drives every neural network layer.
WHAT IT DOES: Normalizes embeddings, builds a pairwise similarity matrix,
              and answers top-k queries.
ANALOGY: A voting system. Each dimension is an issue; the dot product counts
         how strongly two voters agree across all issues at once.
"""

from typing import Any, Dict, List, Tuple

import numpy as np

from .vector_ops import EPSILON, l2_normalize


class SimilarityEngine:
    """
    Cosine-similarity search over a document embedding matrix.

    WHY normalize once at index time instead of per query? Normalizing is
    O(n*d) work. Do it once when the index is built and every later query is
    a single matmul with no division at all. Doing it per query repeats the
    same work on every search forever.
    """

    def __init__(self, config: Dict[str, Any]):
        sim_cfg = config.get("similarity", {})
        self.metric: str = sim_cfg.get("metric", "cosine")
        self.normalize: bool = bool(sim_cfg.get("normalize", True))
        self.top_k: int = int(sim_cfg.get("top_k", 3))

        self.matrix_: np.ndarray | None = None   # the indexed vectors
        self.documents_: List[str] = []

    # ------------------------------------------------------------------
    def index(self, matrix: np.ndarray, documents: List[str]) -> "SimilarityEngine":
        """
        Stores the embedding matrix, normalizing it if cosine is requested.

        WHY the length check? A mismatch between vectors and documents means
        every result maps to the wrong text — and nothing crashes to tell you.
        Silent misalignment is the worst kind of bug, so fail loudly here.
        """
        if matrix.ndim != 2:
            raise ValueError(f"Expected a 2D (n_docs, dim) matrix, got {matrix.shape}")
        if matrix.shape[0] != len(documents):
            raise ValueError(
                f"Row count {matrix.shape[0]} != document count {len(documents)}. "
                f"Every row must correspond to exactly one document."
            )

        self.documents_ = list(documents)
        # cosine == dot product on unit vectors, so normalizing converts one into the other
        use_cosine = self.normalize and self.metric == "cosine"
        self.matrix_ = l2_normalize(matrix) if use_cosine else matrix
        return self

    # ------------------------------------------------------------------
    def similarity_matrix(self) -> np.ndarray:
        """
        Every pairwise similarity in one matmul: (n, d) @ (d, n) -> (n, n).

        THE SHAPE RULE: (m, n) @ (n, p) = (m, p). The inner dimensions must
        match and they cancel. Here the inner dimension is the embedding size.

        COST WARNING: the output is n x n. At 100k documents that is 10^10
        floats — 40 GB. Full matrices are for inspection on small corpora;
        real search never materialises one, it does a single (n, d) @ (d,).
        """
        m = self._require_index()
        return m @ m.T

    def search(self, query_vector: np.ndarray,
               top_k: int | None = None) -> List[Tuple[str, float, int]]:
        """
        Returns the top_k most similar documents as (document, score, index).

        ONE MATMUL, ZERO LOOPS: (n_docs, dim) @ (dim,) -> (n_docs,).
        Every document is scored in a single compiled call.

        WHY argsort on the negated scores? np.argsort only sorts ascending
        and returns INDICES, not values. Negating flips it to descending;
        the indices then let us recover both the score and the source text.
        """
        m = self._require_index()
        k = top_k if top_k is not None else self.top_k

        if query_vector.ndim != 1:
            raise ValueError(f"Query must be a 1D vector, got shape {query_vector.shape}")
        if query_vector.shape[0] != m.shape[1]:
            raise ValueError(
                f"Query dim {query_vector.shape[0]} != index dim {m.shape[1]}. "
                f"The query must be embedded with the SAME EmbeddingStore as the corpus."
            )

        q = query_vector
        if self.normalize and self.metric == "cosine":
            q = q / max(float(np.linalg.norm(q)), EPSILON)

        scores = m @ q                                 # the entire search
        k = min(k, scores.shape[0])
        order = np.argsort(-scores)[:k]

        return [(self.documents_[i], float(scores[i]), int(i)) for i in order]

    def most_similar_pair(self) -> Dict[str, Any]:
        """
        Finds the two most alike documents in the corpus.

        WHY mask the diagonal? Every document scores a perfect 1.0 against
        itself, so an unmasked argmax always returns a document paired with
        itself — technically correct, completely useless.

        EDGE CASE: with a single document there IS no pair. Masking the only
        cell leaves an all -inf matrix, argmax returns 0, and the score comes
        back as -inf — which json.dump writes as `-Infinity`, something no
        strict JSON parser will read back. Return None instead.
        """
        m = self._require_index()
        if m.shape[0] < 2:
            return {"doc_a": None, "doc_b": None, "score": None}

        sim = self.similarity_matrix().copy()
        np.fill_diagonal(sim, -np.inf)               # exclude self-matches

        flat = int(np.argmax(sim))                    # index into the flattened array
        row_col = np.unravel_index(flat, sim.shape)   # back to (row, col)
        i, j = int(row_col[0]), int(row_col[1])

        return {
            "doc_a": self.documents_[i],
            "doc_b": self.documents_[j],
            "score": round(float(sim[i, j]), 4),
        }

    def report(self) -> Dict[str, Any]:
        """
        Summary statistics of the similarity matrix, for the JSON report.

        EDGE CASE: a one-document corpus has no off-diagonal cells at all.
        `.max()` on an empty array raises ValueError and `.mean()` returns
        NaN with a warning, so both stats are reported as None instead.
        """
        m = self._require_index()
        sim = self.similarity_matrix()
        off_diagonal = sim[~np.eye(sim.shape[0], dtype=bool)]
        has_pairs = off_diagonal.size > 0

        return {
            "n_documents": int(m.shape[0]),
            "dimensions": int(m.shape[1]),
            "metric": self.metric,
            "normalized": self.normalize,
            "diagonal_is_one": bool(np.allclose(np.diag(sim), 1.0, atol=1e-5)),
            "matrix_is_symmetric": bool(np.allclose(sim, sim.T, atol=1e-5)),
            "mean_off_diagonal_similarity":
                round(float(off_diagonal.mean()), 4) if has_pairs else None,
            "max_off_diagonal_similarity":
                round(float(off_diagonal.max()), 4) if has_pairs else None,
            "most_similar_pair": self.most_similar_pair(),
        }

    # ------------------------------------------------------------------
    def _require_index(self) -> np.ndarray:
        """
        Returns the indexed matrix, or raises if index() was never called.

        WHY RETURN IT instead of just checking? `self.matrix_` is typed
        `ndarray | None`, so every use of it after a bare check still looks
        possibly-None to a type checker. Handing back the narrowed value
        lets callers work with a plain ndarray.
        """
        if self.matrix_ is None:
            raise RuntimeError("Engine has no index — call index(matrix, documents) first.")
        return self.matrix_


# ----------------------------------------------------------------------
# Plain functions — the lecture's geometry, usable without the class
# ----------------------------------------------------------------------
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    cos(theta) = (a . b) / (|a| |b|), always between -1 and 1.

    WHY divide out the magnitudes? A raw dot product rewards long vectors.
    In search that means a long document beats a relevant one purely on
    length. Cosine keeps only the direction, which is the meaning.
    """
    denom = max(float(np.linalg.norm(a) * np.linalg.norm(b)), EPSILON)
    return float(np.dot(a, b) / denom)


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """
    The angle in degrees between two vectors.

    0 degrees   -> identical direction, cosine +1
    90 degrees  -> orthogonal, cosine 0, no relationship
    180 degrees -> opposite, cosine -1

    The clip is essential: floating-point rounding can hand arccos a value
    like 1.0000000002, and arccos of anything past 1 returns NaN.
    """
    cos = np.clip(cosine_similarity(a, b), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def dense_layer(x: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """
    One neural network layer: relu(x @ W + b).

    THE POINT OF THIS FUNCTION: a "layer" sounds mysterious until you see it
    is a matmul, a broadcast add, and an elementwise max. Stack these and you
    have a neural network. There is nothing else in there.

    Shapes: (batch, in_dim) @ (in_dim, out_dim) + (out_dim,) -> (batch, out_dim)
    """
    if x.shape[1] != weights.shape[0]:
        raise ValueError(
            f"Inner dimensions must match: x{x.shape} @ W{weights.shape}. "
            f"{x.shape[1]} != {weights.shape[0]}"
        )
    return np.maximum(x @ weights + bias, 0)
