"""
vector_ops.py
-------------
Motive: Every operation here is vectorized. Not one Python loop over elements.
WHY: This is Topic 2 of the lecture. A transformer forward pass is ~10^12
     multiply-adds; done in Python loops a single model would take centuries.
WHAT IT DOES: Normalization, standardization, softmax, and a loop-vs-vectorized
              benchmark that measures the gap on your own machine.
ANALOGY: A paint sprayer versus a one-inch brush. Same paint, same fence —
         one pass instead of a thousand trips back to the bucket.
"""

import math
import time
from typing import Any, Callable, Dict, Tuple

import numpy as np

EPSILON = 1e-9  # guards every division; a zero vector must not produce NaN


def l2_normalize(matrix: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Scales every row to unit length.

    WHY? Once vectors are unit length, their dot product IS the cosine of the
    angle between them. One matmul then computes cosine similarity for every
    pair at once — no per-pair division needed.

    THE KEEPDIMS DETAIL: norm(axis=-1) on a (n, d) matrix returns shape (n,).
    Dividing (n, d) by (n,) fails, because broadcasting aligns from the RIGHT
    and n != d. keepdims=True returns (n, 1), which broadcasts correctly.
    This one flag is the most common shape bug in the whole topic.

    ANALOGY: resizing every arrow to length 1 so you compare only the
             direction they point, never how long they are.
    """
    norms = np.linalg.norm(matrix, axis=axis, keepdims=True)
    return matrix / np.maximum(norms, EPSILON)


def standardize(matrix: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Z-score standardization: subtract the mean, divide by the std.

    WHY axis=0 by default? On a (samples, features) matrix, axis=0 collapses
    the SAMPLES, giving one mean per feature. That is per-feature scaling,
    which is what models want. axis=1 would scale each sample by its own
    statistics — almost never what you mean.

    ANALOGY: axis is the direction you squash the sponge. axis=0 squashes
             top to bottom, leaving one value per column.
    """
    mu = matrix.mean(axis=axis, keepdims=True)
    sd = matrix.std(axis=axis, keepdims=True)
    return (matrix - mu) / np.maximum(sd, EPSILON)


def softmax(scores: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Turns raw scores into a probability distribution along `axis`.

    THE MAX SUBTRACTION IS NOT OPTIONAL: exp(1000) overflows to inf, and
    inf/inf is NaN. Subtracting the row max makes the largest exponent
    exp(0) = 1, so nothing can overflow. The result is mathematically
    identical because exp(a-c)/sum(exp(a-c)) == exp(a)/sum(exp(a)).

    ANALOGY: converting raw vote counts into percentages of the total.
    """
    shifted = scores - np.max(scores, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.maximum(exp.sum(axis=axis, keepdims=True), EPSILON)


def add_bias(batch: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """
    Adds one bias vector to every row of a batch — the broadcasting demo.

    (batch, dim) + (dim,) works because broadcasting right-aligns the shapes
    and treats the missing axis as 1. No copy of the bias is ever made:
    NumPy walks that axis with a stride of 0.

    ANALOGY: a rubber stamp. One stamp, pressed onto every page, no photocopies.
    """
    if batch.ndim != 2:
        raise ValueError(f"batch must be 2D, got shape {batch.shape}")
    if bias.shape != (batch.shape[1],):
        raise ValueError(
            f"bias shape {bias.shape} cannot broadcast onto batch {batch.shape}. "
            f"Expected ({batch.shape[1]},). "
            f"If you meant per-ROW scaling, reshape with bias[:, None]."
        )
    return batch + bias


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU as a single vectorized call. No if-statement, no loop."""
    return np.maximum(x, 0)


# ----------------------------------------------------------------------
# Benchmark — measures the loop-vs-vectorized gap on YOUR machine
# ----------------------------------------------------------------------
def _time_it(fn: Callable[[], Any], repeat: int = 1) -> Tuple[float, Any]:
    """
    Runs fn and returns (average milliseconds, result).

    The repeat guard matters: with repeat=0 the loop body never runs, `out`
    would be unbound at the return, and the division would be by zero.
    """
    repeat = max(1, int(repeat))
    out = fn()  # warm up: first call pays for cache and lazy imports
    t0 = time.perf_counter()
    for _ in range(repeat):
        out = fn()
    ms = (time.perf_counter() - t0) * 1000 / repeat
    return ms, out # type: ignore


def benchmark_loop_vs_vectorized(n: int = 1_000_000) -> Dict[str, Any]:
    """
    Races a Python loop against NumPy on sqrt(a^2 + b^2).

    WHY a compound expression and not a bare add? A single add is so cheap
    that the timing is dominated by memory bandwidth. A sqrt of a sum of
    squares does real arithmetic, so the interpreter overhead shows honestly.

    WHY the loop is slow: per element it pays bytecode dispatch, a type check,
    unboxing two PyObjects, re-boxing the result, and refcount updates.
    NumPy pays that once for the entire array, then runs a tight C loop that
    the CPU can execute with SIMD instructions.
    """
    a_list = list(range(n))
    b_list = list(range(n))
    a = np.arange(n, dtype=np.float64)
    b = np.arange(n, dtype=np.float64)

    loop_ms, loop_out = _time_it(
        lambda: [math.sqrt(x * x + y * y) for x, y in zip(a_list, b_list)]
    )
    vec_ms, vec_out = _time_it(lambda: np.sqrt(a**2 + b**2))

    return {
        "n_elements": n,
        "python_loop_ms": round(loop_ms, 2),
        "numpy_vectorized_ms": round(vec_ms, 2),
        "speedup_x": round(loop_ms / max(vec_ms, EPSILON), 1),
        "results_match": bool(np.allclose(np.asarray(loop_out), vec_out)),
    }


def benchmark_memory_layout(size: int = 3000) -> Dict[str, Any]:
    """
    Measures the cost of memory layout, holding the reduction constant.

    THE FAIR TEST: identical numbers, identical operation — only C-order vs
    F-order differs. Comparing sum(axis=0) against sum(axis=1) on one array
    is NOT a layout test; those are different amounts of work, and on C-order
    data axis=0 often wins because NumPy accumulates whole contiguous rows
    with SIMD. What actually costs you is scanning across the strided axis.
    """
    c_order = np.random.rand(size, size)
    f_order = np.asfortranarray(c_order)

    c_ms, _ = _time_it(lambda: c_order.sum(axis=1), repeat=3)
    f_ms, _ = _time_it(lambda: f_order.sum(axis=1), repeat=3)

    return {
        "matrix": [size, size],
        "sum_axis1_c_order_ms": round(c_ms, 2),
        "sum_axis1_f_order_ms": round(f_ms, 2),
        "c_order_strides": list(c_order.strides),
        "f_order_strides": list(f_order.strides),
        "transpose_copies_data": bool(not np.shares_memory(c_order, c_order.T)),
    }
