"""
attention.py
------------
Motive: Scaled dot-product attention in plain NumPy, ~6 lines of real work.
WHY: This is the engine inside every LLM. Once you see that Q @ K.T is just
     "every token scores every other token", the transformer stops being magic.
WHAT IT DOES: Builds Q/K/V, computes attention weights, applies a causal mask.
ANALOGY: A meeting. Every person asks everyone else "how relevant are you to
         me?" (Q @ K.T). The answers are normalised into a 100% budget
         (softmax). Then everyone's opinion (V) is blended by that budget.
"""

from typing import Any, Dict, Tuple

import numpy as np

from .vector_ops import softmax


def causal_mask(seq_len: int) -> np.ndarray:
    """
    A lower-triangular boolean mask: token i may attend to tokens 0..i only.

    WHY? This is what makes GPT autoregressive. Without it, predicting token 5
    could peek at token 6 — the model would score perfectly in training and
    produce nonsense at generation time, because the future is not there yet.
    ANALOGY: an exam where covering the answer key is enforced by the desk.
    """
    return np.tril(np.ones((seq_len, seq_len), dtype=bool))


def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: np.ndarray | None = None,
    scale: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (output, attention_weights).

    THE THREE STEPS:
      1. scores  = Q @ K.T / sqrt(d_k)   -> (seq, seq) token-to-token affinity
      2. weights = softmax(scores)       -> each row sums to 1
      3. output  = weights @ V           -> a weighted blend of the values

    WHY DIVIDE BY sqrt(d_k)? The dot product of two random d-dimensional
    vectors grows like sqrt(d). Feed those large scores to softmax and it
    saturates: one weight goes to ~1.0, the rest to ~0, and the gradient
    through them vanishes. Dividing keeps the score variance near 1 so the
    distribution stays soft and trainable. Set scale=False in the config to
    watch the max weight jump toward 1.0.

    THE .swapaxes(-1, -2) DETAIL: this transposes only the last two axes, so
    the same code works for both (seq, d_k) and batched (batch, seq, d_k).
    Plain .T would reverse ALL axes and silently scramble a batch.
    """
    d_k = Q.shape[-1]
    scores = Q @ K.swapaxes(-1, -2)
    if scale:
        scores = scores / np.sqrt(d_k)

    if mask is not None:
        # -inf becomes exactly 0 after exp() — a hard block, not a soft penalty
        scores = np.where(mask, scores, -np.inf)

    weights = softmax(scores, axis=-1)
    return weights @ V, weights


def run_attention_demo(config: Dict[str, Any], seed: int = 42) -> Dict[str, Any]:
    """
    Runs attention once with the configured settings and reports what happened.

    The saturation comparison is the teaching moment: identical Q and K, the
    only difference is whether the scores were scaled.
    """
    att_cfg = config.get("attention", {})
    seq_len = int(att_cfg.get("seq_len", 6))
    d_k = int(att_cfg.get("d_k", 16))
    use_mask = bool(att_cfg.get("causal_mask", True))
    use_scale = bool(att_cfg.get("scale_by_sqrt_dk", True))

    rng = np.random.default_rng(seed)
    Q = rng.standard_normal((seq_len, d_k))
    K = rng.standard_normal((seq_len, d_k))
    V = rng.standard_normal((seq_len, d_k))

    mask = causal_mask(seq_len) if use_mask else None
    output, weights = scaled_dot_product_attention(Q, K, V, mask=mask, scale=use_scale)

    # Same inputs, scaling toggled — proves why the sqrt(d_k) divide exists
    _, w_scaled = scaled_dot_product_attention(Q, K, V, mask=None, scale=True)
    _, w_unscaled = scaled_dot_product_attention(Q, K, V, mask=None, scale=False)

    # With a causal mask, row i has exactly i+1 non-zero weights
    expected_nonzero = [i + 1 for i in range(seq_len)] if use_mask else [seq_len] * seq_len
    actual_nonzero = [int((weights[i] > 1e-12).sum()) for i in range(seq_len)]

    return {
        "seq_len": seq_len,
        "d_k": d_k,
        "causal_mask": use_mask,
        "scaled": use_scale,
        "scores_shape": [seq_len, seq_len],
        "output_shape": list(output.shape),
        "rows_sum_to_one": bool(np.allclose(weights.sum(axis=-1), 1.0)),
        "max_weight_scaled": round(float(w_scaled.max()), 4),
        "max_weight_unscaled": round(float(w_unscaled.max()), 4),
        "nonzero_per_row": actual_nonzero,
        "causal_structure_correct": actual_nonzero == expected_nonzero,
        "weights": np.round(weights, 3),
    }
