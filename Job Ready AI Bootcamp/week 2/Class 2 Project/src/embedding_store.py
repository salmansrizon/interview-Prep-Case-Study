"""
embedding_store.py
------------------
Motive: Turn text into an ndarray and expose every property of that array.
WHY: This is Topic 1 of the lecture made concrete. Shape, dtype, strides,
     memory layout, views vs copies — all of it on real data you can print.
WHAT IT DOES: Embeds documents into a (n_docs, dim) matrix and reports the
              memory characteristics of that matrix.
ANALOGY: A warehouse manager. Decides the shelf size (shape), the box type
         (dtype), how far apart the shelves sit (strides), and whether a
         request gets a window into the existing shelf (view) or a whole
         second warehouse (copy).
"""

from typing import Any, Dict, List

import numpy as np


class EmbeddingStore:
    """
    Builds and owns the embedding matrix.

    WHY a class? The matrix, the vocabulary, and the config that produced
    them must travel together. Split them apart and you eventually embed a
    query with a different vocabulary than the corpus — silently wrong results.
    """

    def __init__(self, config: Dict[str, Any]):
        emb_cfg = config.get("embedding", {})
        self.method: str = emb_cfg.get("method", "hashing")
        self.dim: int = int(emb_cfg.get("dimensions", 128))
        self.dtype: np.dtype = np.dtype(emb_cfg.get("dtype", "float32"))
        self.lowercase: bool = bool(emb_cfg.get("lowercase", True))
        self.seed: int = int(emb_cfg.get("seed", 42))

        self.vocabulary_: List[str] = []
        self.vocab_index_: Dict[str, int] = {}
        self.matrix_: np.ndarray | None = None
        self.documents_: List[str] = []

    # ------------------------------------------------------------------
    # Tokenisation
    # ------------------------------------------------------------------
    def _tokenize(self, text: str) -> List[str]:
        """Splits on whitespace. Deliberately simple — the lesson is arrays, not NLP."""
        if self.lowercase:
            text = text.lower()
        return [t for t in text.split() if t]

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------
    def fit(self, documents: List[str]) -> "EmbeddingStore":
        """
        Learns the vocabulary from the corpus.

        WHY a separate fit step? The query must be embedded into the SAME
        space as the corpus. Fit once on the corpus, then transform anything.
        """
        self.documents_ = list(documents)
        vocab = sorted({tok for doc in documents for tok in self._tokenize(doc)})
        self.vocabulary_ = vocab
        # WHY cache the lookup here? Rebuilding this dict inside embed_one()
        # would make embedding the corpus O(n_docs * vocab_size) instead of
        # O(total_tokens) — quadratic work hidden inside a "vectorized" project.
        self.vocab_index_ = {w: i for i, w in enumerate(vocab)}
        return self

    def embed_one(self, text: str) -> np.ndarray:
        """
        Embeds a single string into a 1D vector of shape (dim,).

        TWO METHODS:
          bag_of_words — one slot per vocabulary word. Exact, but the vector
                         grows with the vocabulary and is mostly zeros.
          hashing      — hash each token into a fixed number of slots. The
                         vector size never changes no matter how big the
                         corpus grows. This is what production systems use.
        ANALOGY: bag_of_words = a labelled pigeonhole for every word in the
                 dictionary. hashing = a fixed wall of 128 pigeonholes and a
                 rule for which hole each word goes in. Collisions happen;
                 that is the price of a fixed size.
        """
        tokens = self._tokenize(text)

        if self.method == "bag_of_words":
            if not self.vocabulary_:
                raise RuntimeError(
                    "bag_of_words needs a vocabulary — call fit() before embed_one()."
                )
            vec = np.zeros(len(self.vocabulary_), dtype=self.dtype)
            for tok in tokens:
                slot = self.vocab_index_.get(tok)
                if slot is not None:
                    vec[slot] += 1
            return vec

        # hashing (default)
        vec = np.zeros(self.dim, dtype=self.dtype)
        for tok in tokens:
            # WHY the seed in the hash? Python's built-in hash() is randomised
            # per process, so the same word would land in a different slot on
            # every run. A fixed hash keeps results reproducible.
            slot = self._stable_hash(tok) % self.dim
            vec[slot] += 1
        return vec

    def _stable_hash(self, token: str) -> int:
        """A tiny deterministic string hash (FNV-1a style)."""
        h = 2166136261 ^ self.seed
        for ch in token.encode("utf-8"):
            h = ((h ^ ch) * 16777619) & 0xFFFFFFFF
        return h

    def transform(self, documents: List[str]) -> np.ndarray:
        """
        Embeds many documents into a 2D matrix of shape (n_docs, dim).

        WHY np.stack and not a Python list? np.stack lays every vector into
        ONE contiguous block of memory. A list of arrays is scattered across
        the heap, and no matmul can run over it at full speed.
        """
        vectors = [self.embed_one(doc) for doc in documents]
        matrix = np.stack(vectors).astype(self.dtype, copy=False)
        self.matrix_ = matrix
        return matrix

    def fit_transform(self, documents: List[str]) -> np.ndarray:
        """Convenience: fit the vocabulary, then embed the corpus."""
        return self.fit(documents).transform(documents)

    def _require_matrix(self) -> np.ndarray:
        """
        Returns the embedding matrix, or raises if nothing was embedded yet.

        WHY RETURN IT? `self.matrix_` is typed `ndarray | None`. A bare None
        check leaves every later use looking possibly-None to a type checker;
        handing back the narrowed value gives callers a plain ndarray.
        """
        if self.matrix_ is None:
            raise RuntimeError("Nothing embedded yet — call fit_transform() first.")
        return self.matrix_

    # ------------------------------------------------------------------
    # Reporting — this is Topic 1 of the lecture, printed
    # ------------------------------------------------------------------
    def describe(self) -> Dict[str, Any]:
        """
        Reports every structural property of the embedding matrix.

        These are the four properties from the lecture:
          shape   — the dimensions
          dtype   — bytes per element
          strides — bytes to jump to reach the next element on each axis
          layout  — C-order (row-major) or F-order (column-major)
        """
        m = self._require_matrix()
        return {
            "shape": list(m.shape),
            "ndim": int(m.ndim),
            "dtype": str(m.dtype),
            "itemsize_bytes": int(m.itemsize),
            "strides": list(m.strides),
            "nbytes": int(m.nbytes),
            "memory_kb": round(m.nbytes / 1024, 2),
            "c_contiguous": bool(m.flags["C_CONTIGUOUS"]),
            "f_contiguous": bool(m.flags["F_CONTIGUOUS"]),
            "vocabulary_size": len(self.vocabulary_),
            "n_documents": len(self.documents_),
            "sparsity_pct": round(float((m == 0).mean() * 100), 2),
        }

    def memory_comparison(self) -> Dict[str, Any]:
        """
        Compares this matrix against the alternatives.

        WHY show this? A float64 copy is exactly twice the bytes, and a
        Python list of lists is roughly an order of magnitude worse. At GPT
        scale that difference is "fits on the GPU" vs "OOM crash".
        """
        m = self._require_matrix()
        n_values = m.size
        return {
            "as_float32_kb": round(n_values * 4 / 1024, 2),
            "as_float64_kb": round(n_values * 8 / 1024, 2),
            "as_python_list_kb": round(n_values * 32 / 1024, 2),  # ~8 ptr + 24 obj
            "current_dtype": str(m.dtype),
            "current_kb": round(m.nbytes / 1024, 2),
        }

    def view_vs_copy_demo(self) -> Dict[str, Any]:
        """
        Proves the difference between a view and a copy on the real matrix.

        WHY this matters: slicing a batch out of your data does NOT duplicate
        it. That is what lets you iterate over a dataset far larger than RAM.
        But it also means writing into that batch edits your source data.
        ANALOGY: a view is a window cut into the warehouse wall. A copy is
                 renting a second warehouse and hauling every box across.
        """
        probe = self._require_matrix().copy()  # never mutate the real matrix in a demo

        # Size the demo batch to the corpus. Hard-coding rows 0 and 1 would
        # raise IndexError on a single-document corpus, and a student trimming
        # corpus.txt down to one line is exactly the kind of thing that happens.
        n_rows = min(2, probe.shape[0])

        batch_view = probe[0:n_rows]                    # basic slicing -> VIEW
        batch_copy = probe[0:n_rows].copy()             # explicit      -> COPY
        fancy = probe[list(range(n_rows))]              # fancy index   -> COPY

        before = float(probe[0, 0])
        batch_view[0, 0] = 999.0                # writes straight through
        after_view_write = float(probe[0, 0])

        probe[0, 0] = before                    # restore
        batch_copy[0, 0] = -999.0               # writes nowhere near the source
        after_copy_write = float(probe[0, 0])

        return {
            "slice_is_view": bool(np.shares_memory(batch_view, probe)),
            "copy_is_view": bool(np.shares_memory(batch_copy, probe)),
            "fancy_index_is_view": bool(np.shares_memory(fancy, probe)),
            "source_before": before,
            "source_after_view_write": after_view_write,
            "source_after_copy_write": after_copy_write,
        }
