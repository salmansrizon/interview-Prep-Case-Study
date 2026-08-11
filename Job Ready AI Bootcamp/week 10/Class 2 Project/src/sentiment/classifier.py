"""
Lazy-loading singleton wrapper around a HuggingFace `pipeline()` sentiment
classifier.

HIGHLIGHTS: `transformers` (and the ~500MB model weights it downloads on
first use) is NOT imported or touched at module import time. It is only
imported and instantiated inside `get_classifier()`, the first time
someone actually calls `classify()`. This mirrors the same lazy-loading
pattern used for the embedding model in this course's other Week 10
project: a student — or an interviewer skimming this repo before an
interview — should be able to `import` this module, read every line of
this file, and run the test suite, WITHOUT accidentally kicking off a
large network download just because Python imported the file. The
download (and the one-time cost of loading the model into memory) only
happens the moment the app is actually used for its real purpose.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SentimentResult:
    """The outcome of classifying a single review.

    Attributes:
        text: The original review text (kept alongside the result so
            callers — the results table, the aggregator — don't need to
            zip it back together with a separate list and risk misaligning
            indices).
        label: One of "Positive", "Negative", "Neutral".
        score: The model's confidence in `label`, in [0.0, 1.0].
        needs_review: True if `score` fell below the configured confidence
            threshold — see `_needs_review()` below for the boundary rule.
    """

    text: str
    label: str
    score: float
    needs_review: bool


# HIGHLIGHTS: the raw model output uses lowercase labels ("positive",
# "negative", "neutral") because that's simply how this checkpoint's
# config.json defines them. We normalize to Title Case here, in exactly
# ONE place, so every downstream consumer (the results table, the
# dashboard charts, the aggregator's group-by keys, the tests) can rely on
# a single consistent spelling ("Positive"/"Negative"/"Neutral") without
# each one having to defensively `.capitalize()` or `.lower()` the label
# itself. If this project ever swaps in a different checkpoint whose
# labels are spelled differently (e.g. "LABEL_0"/"LABEL_1"/"LABEL_2" for
# some other model), this is the one map that needs to change.
_LABEL_MAP = {
    "positive": "Positive",
    "negative": "Negative",
    "neutral": "Neutral",
}

# Module-level singleton slot. `None` until `get_classifier()` is called
# for the first time; after that, the same pipeline object is reused for
# the lifetime of the process (or the Streamlit session) instead of being
# rebuilt — and re-downloaded/re-loaded into memory — on every call.
_pipeline: Optional[Any] = None


def get_classifier():
    """Return the singleton HuggingFace sentiment-analysis pipeline,
    constructing (and downloading, on first run) it lazily.

    HIGHLIGHTS — Model Reference Table (adapted from the Class 2 Lecture,
    Section 4.1 "কোন সেন্টিমেন্ট মডেল কখন ব্যবহার করবেন"). Read this before
    changing `config.ModelConfig.model_name`:

        | Model                                                      | Use Case                                   | Advantage                                                    | Limitation                                                                 |
        |--------------------------------------------------------------|---------------------------------------------|----------------------------------------------------------------|-------------------------------------------------------------------------------|
        | cardiffnlp/twitter-roberta-base-sentiment-latest (USED HERE) | General 3-class sentiment (Pos/Neg/Neutral) | Genuine Neutral class — no hacky thresholding needed           | Trained on tweet-style text; some domain shift vs. formal product reviews     |
        | distilbert-base-uncased-finetuned-sst-2-english               | Fast binary (Pos/Neg only) sentiment        | Small, fast, extremely popular default                        | No Neutral class — forcing one in requires an unreliable score threshold      |
        | nlptown/bert-base-multilingual-uncased-sentiment               | 1-5 star rating prediction                  | Multilingual, rating-style output                             | Outputs a star scale, not Pos/Neg/Neutral — needs a mapping layer             |
        | OpenAI GPT (zero-shot prompting)                              | Flexible classification w/ custom categories | No fine-tuning needed, just change the prompt                 | Not local, per-call cost, data leaves your machine (third-party API)          |

    We chose `cardiffnlp/twitter-roberta-base-sentiment-latest` because
    this project's spec explicitly wants three real classes, and the
    "most popular" model (DistilBERT SST-2) simply isn't trained to
    produce a Neutral class at all — see `config.ModelConfig` for the
    fuller version of this reasoning. The tradeoff we accept in return is
    Section 4.2's "Domain Shift" caveat: this checkpoint was fine-tuned on
    tweets (short, informal, emoji/hashtag-heavy), while we're feeding it
    product reviews (longer, more formal). The two domains share enough
    general sentiment vocabulary ("broke", "disappointed", "fantastic")
    that the model still works well in practice, but don't expect
    tweet-benchmark-level accuracy — and treat every prediction's
    confidence score, not just its label, as part of the answer. In a
    real job, the standard mitigation is exactly what the lecture
    describes as out-of-scope-but-important: collect a sample of your
    own domain's labeled reviews and run a second, lighter fine-tuning
    pass on top of this checkpoint to close the gap.

    Returns:
        A `transformers.Pipeline` configured for `config.ModelConfig.task`
        ("sentiment-analysis") with `config.ModelConfig.model_name`.
    """
    global _pipeline

    if _pipeline is None:
        # Import transformers here, not at module scope — see the module
        # docstring. This is the ONLY place in this file (and one of the
        # only places in the whole project) that touches the transformers
        # library or the network.
        from transformers import pipeline

        cfg = get_config().model
        logger.info(
            "Loading sentiment model %r for the first time. This downloads "
            "~500MB on first run and is cached by HuggingFace afterwards; "
            "subsequent runs load from local cache instead of the network.",
            cfg.model_name,
        )
        _pipeline = pipeline(
            cfg.task,
            model=cfg.model_name,
            top_k=None if cfg.return_all_scores else 1,
        )
        logger.info("Sentiment model loaded.")

    return _pipeline


def _needs_review(score: float, threshold: float) -> bool:
    """Decide whether a prediction should be flagged for human review.

    HIGHLIGHTS: uses `score < threshold`, i.e. a prediction that lands
    EXACTLY ON the threshold is treated as confident enough and is NOT
    flagged. This boundary choice is arbitrary in the abstract (you could
    just as validly flag `<=`), but it has to be chosen and documented
    somewhere so behavior at the boundary is deterministic and testable —
    see `tests/test_pipeline.py` for a test that pins this exact behavior
    at, just above, and just below the configured threshold. Directly
    implements the Class 2 Lecture's Brain Teaser #2.
    """
    return score < threshold


def classify(
    texts: List[str],
    threshold: Optional[float] = None,
) -> List[SentimentResult]:
    """Classify a batch of review strings and return one SentimentResult each.

    Args:
        texts: Review strings to classify. Order is preserved in the
            returned list — `results[i]` corresponds to `texts[i]`.
        threshold: Confidence threshold below which a result is flagged
            `needs_review=True`. Defaults to
            `config.AnalysisConfig.confidence_threshold` if not given.

    Returns:
        A list of `SentimentResult`, same length and order as `texts`.
        Empty/whitespace-only entries in `texts` are skipped (not sent to
        the model) — an empty string has no sentiment to classify, and
        HuggingFace pipelines error on empty input.

    HIGHLIGHTS: notice this function calls `classifier(texts)` — passing
    the ENTIRE list of texts to the pipeline in one call — rather than
    looping `for t in texts: classifier(t)`. This directly applies Week
    2's Vectorization lecture and Section 4.3 of the Class 2 Lecture: a
    HuggingFace pipeline batches its tokenization and forward pass
    internally when given a list, which is dramatically more efficient on
    both CPU and GPU than issuing N separate single-example calls (each of
    which pays fixed per-call overhead — tokenizer setup, padding to a
    batch of one, a full forward pass — that a real batch amortizes across
    many examples at once). For very large inputs (e.g. a CSV with
    thousands of rows) callers should still chunk into pieces of
    `config.AnalysisConfig.batch_size` — see `classify_in_batches()` below
    — but each chunk itself is still passed as a list, never as a Python
    loop of one-at-a-time calls.
    """
    cfg = get_config().analysis
    threshold = cfg.confidence_threshold if threshold is None else threshold

    # Keep track of which original indices had real (non-empty) text, so
    # we can splice results back into a list that's the same length/order
    # as the caller's input even though we only send non-empty text to the
    # model.
    cleaned: List[str] = []
    index_map: List[int] = []
    for i, t in enumerate(texts):
        if t and t.strip():
            cleaned.append(t.strip()[: cfg.max_review_chars])
            index_map.append(i)

    results: List[Optional[SentimentResult]] = [None] * len(texts)

    if cleaned:
        classifier = get_classifier()
        # Single batched call — see HIGHLIGHTS above. `top_k=1` (set in
        # get_classifier) makes each element of `raw` a one-item list like
        # [{'label': 'positive', 'score': 0.94}], mirroring the simpler
        # `[{'label': ..., 'score': ...}]` shape from the lecture when
        # top_k is left at its default.
        raw = classifier(cleaned)
        for text, original_idx, prediction in zip(cleaned, index_map, raw):
            # top_k=1 still wraps the single prediction in a list; unwrap it.
            pred = prediction[0] if isinstance(prediction, list) else prediction
            label = _LABEL_MAP.get(str(pred["label"]).lower(), str(pred["label"]))
            score = float(pred["score"])
            results[original_idx] = SentimentResult(
                text=text,
                label=label,
                score=score,
                needs_review=_needs_review(score, threshold),
            )

    # Any remaining `None` slots correspond to empty/whitespace-only input
    # rows. We still return a same-length list (so a CSV row count always
    # matches the results table row count) but mark them clearly rather
    # than silently dropping them or crashing.
    for i, r in enumerate(results):
        if r is None:
            results[i] = SentimentResult(
                text="", label="Neutral", score=0.0, needs_review=True
            )

    return results  # type: ignore[return-value]


def classify_in_batches(
    texts: List[str],
    threshold: Optional[float] = None,
    batch_size: Optional[int] = None,
) -> List[SentimentResult]:
    """Classify a (potentially large) list of reviews in fixed-size batches.

    HIGHLIGHTS: this exists as a thin wrapper around `classify()` for
    large CSV uploads. Each chunk is still a genuine batch call into the
    pipeline (list in, list out) — chunking is purely about bounding peak
    memory/compute and giving a caller (the Streamlit progress bar) a
    place to report incremental progress, not about avoiding batching
    itself. If you don't need progress reporting or memory bounds, calling
    `classify()` directly with the full list is equally correct.
    """
    cfg = get_config().analysis
    batch_size = cfg.batch_size if batch_size is None else batch_size

    results: List[SentimentResult] = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        results.extend(classify(chunk, threshold=threshold))
    return results
