"""
exercises.py
------------
🎓 THIS IS YOUR EXAM. Everything else in this project already works.

HOW IT WORKS:
  1. Each function below raises NotImplementedError and has a TODO.
  2. Replace the `raise` line with your own NumPy code.
  3. Run `python main.py` — the SKILL CHECK at the end grades you
     automatically against hidden random test cases.
  4. Keep going until the scorecard reads 8/8.

THE ONE RULE: no Python `for` loops and no list comprehensions over elements.
Every answer fits in one to three vectorized lines. The grader checks your
source code for loops and marks the answer as not-vectorized if it finds one —
a correct-but-looping answer earns a partial pass, not a full one.

WHY grade it this way? Getting the right numbers with a loop means you
understood the maths. Getting them without a loop means you understood NumPy.
The job needs both.
"""

# The TODO markers below are the assignment itself, not leftover debt.
# pylint: disable=fixme
import numpy as np


# ======================================================================
# TOPIC 1 — N-Dimensional Arrays
# ======================================================================

def exercise_1_make_batch(n_samples: int, n_features: int) -> np.ndarray:
    """
    Return a float32 array of shape (n_samples, n_features) filled with ZEROS.

    WHY float32? Half the memory of float64, plenty of precision for a model.
    HINT: np.zeros takes a `dtype=` argument.
    """
    # TODO: replace this line with your answer
    raise NotImplementedError("exercise_1_make_batch")


def exercise_2_add_bias(batch: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """
    Add a bias vector of shape (n_features,) to EVERY row of `batch`,
    which has shape (n_samples, n_features). Return the result.

    Do NOT tile, repeat, or stack the bias. Let broadcasting do it.
    HINT: this is shorter than you think.
    """
    # TODO: replace this line with your answer
    raise NotImplementedError("exercise_2_add_bias")


def exercise_3_scale_rows(batch: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """
    Multiply each ROW of `batch` (n_samples, n_features) by its own scalar
    from `scale` (n_samples,). Row 0 times scale[0], row 1 times scale[1], ...

    THE TRAP: `batch * scale` raises a ValueError, or worse, silently scales
    the wrong axis when the two dimensions happen to be equal. Broadcasting
    aligns shapes from the RIGHT, so you must give `scale` a second axis.
    HINT: scale[:, None] has shape (n_samples, 1).
    """
    # TODO: replace this line with your answer
    raise NotImplementedError("exercise_3_scale_rows")


# ======================================================================
# TOPIC 2 — Vectorization
# ======================================================================

def exercise_4_feature_means(batch: np.ndarray) -> np.ndarray:
    """
    Given `batch` of shape (n_samples, n_features), return the mean of each
    FEATURE. The result must have shape (n_features,).

    THINK: which axis collapses? You are averaging ACROSS samples, so the
    sample axis is the one that disappears.
    """
    # TODO: replace this line with your answer
    raise NotImplementedError("exercise_4_feature_means")


def exercise_5_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """
    Scale every row of `matrix` to unit L2 length (length exactly 1.0).

    HINT: np.linalg.norm(matrix, axis=1, keepdims=True)
    WHY keepdims? Without it the norms come back as shape (n,), and dividing
    an (n, d) matrix by an (n,) vector fails — or silently divides along the
    wrong axis when n happens to equal d. That bug is nearly invisible.
    """
    # TODO: replace this line with your answer
    raise NotImplementedError("exercise_5_normalize_rows")


def exercise_6_count_above(matrix: np.ndarray, threshold: float) -> int:
    """
    Return how many elements of `matrix` are strictly greater than `threshold`.

    No loop, no if-statement. A boolean mask summed is a count, because
    True counts as 1 and False as 0.
    """
    # TODO: replace this line with your answer
    raise NotImplementedError("exercise_6_count_above")


# ======================================================================
# TOPIC 3 — Dot Products & Matrix Multiplication
# ======================================================================

def exercise_7_pairwise_cosine(matrix: np.ndarray) -> np.ndarray:
    """
    Given `matrix` of shape (n_docs, dim), return the (n_docs, n_docs) matrix
    of cosine similarities between every pair of rows.

    The diagonal must come out as 1.0 — every row matches itself perfectly.

    TWO STEPS: normalize the rows, then one matrix multiply against the
    transpose. On unit vectors the dot product IS the cosine.
    HINT: you may call exercise_5_normalize_rows.
    """
    # TODO: replace this line with your answer
    raise NotImplementedError("exercise_7_pairwise_cosine")


def exercise_8_top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """
    Given a 1D array of `scores`, return the INDICES of the k largest values,
    ordered best first. Return them as an integer array of length k.

    HINT: np.argsort sorts ascending and returns indices. Negate the scores
    to flip the order, then slice the first k.
    """
    # TODO: replace this line with your answer
    raise NotImplementedError("exercise_8_top_k")


# ======================================================================
# The grader discovers exercises through this list. Do not rename them.
# ======================================================================
ALL_EXERCISES = [
    exercise_1_make_batch,
    exercise_2_add_bias,
    exercise_3_scale_rows,
    exercise_4_feature_means,
    exercise_5_normalize_rows,
    exercise_6_count_above,
    exercise_7_pairwise_cosine,
    exercise_8_top_k,
]
