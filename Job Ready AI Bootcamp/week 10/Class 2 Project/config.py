"""
Centralized configuration for the Local Customer Feedback Analyzer.

All model names, thresholds, and paths used across `src/` and `app.py` live
here so nothing is hardcoded as a "magic string" three directories deep.

HIGHLIGHTS: this file does NOT import `transformers` or touch the network.
It only holds plain strings, numbers, and paths. That separation matters
for a project whose whole selling point is a ~500MB model download the
first time it actually runs — a student (or an interviewer skimming the
repo) must be able to open every file, including this one, without
accidentally triggering that download. See src/sentiment/classifier.py
for where the actual `pipeline()` construction (and therefore the
download) happens, and why it is deferred until the first real call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


# ── Paths ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_DIR = DATA_DIR / "sample"
LOG_DIR = DATA_DIR / "logs"

for _d in (DATA_DIR, SAMPLE_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ModelConfig:
    """Which HuggingFace model backs the sentiment pipeline, and how it's called.

    HIGHLIGHTS: the model name is a single, well-documented choice, not a
    dropdown of five options the student has to reason about at runtime.
    Per the Class 2 Lecture's Section 4.1 "Model Reference Table", we chose
    `cardiffnlp/twitter-roberta-base-sentiment-latest` specifically because
    it is a GENUINE 3-class model (Positive/Negative/Neutral) — the more
    commonly reached-for default, `distilbert-base-uncased-finetuned-sst-2-
    english`, is binary (Positive/Negative only) and has no real Neutral
    class. Bolting a "Neutral" bucket onto a binary model means picking an
    arbitrary confidence-band threshold (e.g. "anything between 40-60% is
    Neutral") — that's a hack layered on top of a model that was never
    trained to make that distinction, and it behaves unpredictably. This
    project's spec explicitly calls for 3-class output, so "which model is
    actually trained for this task" wins over "which model is the most
    popular default" (see the lecture's Valid Point in 4.1).
    """

    model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    # `pipeline("sentiment-analysis", ...)` is the correct task string even
    # for this 3-class model — "sentiment-analysis" is a HuggingFace task
    # alias for text classification with sentiment-style labels, it does
    # not mean "must be binary".
    task: str = "sentiment-analysis"
    # top_k=None (previously return_all_scores=True) would return every
    # class's probability instead of just the argmax label+score. We keep
    # it off by default because the dashboard only needs the winning label
    # and its confidence — see classifier.py for where this could be
    # flipped on if a future feature wanted the full 3-way distribution.
    return_all_scores: bool = False


@dataclass(frozen=True)
class AnalysisConfig:
    """Runtime knobs for how reviews are scored and flagged."""

    # HIGHLIGHTS: 0.6 is a deliberate middle-ground default, not an
    # arbitrary round number. The lecture's Brain Teaser #2 asks exactly
    # this: what breaks if the threshold is too high (e.g. 0.9) or too low
    # (e.g. 0.3)? Too high and almost every review gets flagged "needs
    # human review" — including ones the model is genuinely confident
    # about — which defeats the point of automating triage and buries
    # reviewers in false positives. Too low and genuinely ambiguous
    # predictions (a 52%-confidence coin flip between Negative and
    # Neutral) sail through unflagged, so a human never catches the
    # model's actual uncertainty. 0.6 is chosen as a reasonable starting
    # point above "barely better than random for 3 classes" (~33%) while
    # still leaving real headroom below "the model is basically certain"
    # (~90%+). It is exposed as a config value (and, in the UI, a slider)
    # precisely so a student can experiment with Brain Teaser #2 instead
    # of having it hardcoded and invisible.
    confidence_threshold: float = 0.6
    # HIGHLIGHTS: batch_size caps how many review strings we hand to the
    # HuggingFace pipeline in a single call. Passing the whole list in one
    # shot (see src/sentiment/classifier.py's `classify()`) is what makes
    # batch processing fast in the first place (Section 4.3 of the
    # lecture, tied to Week 2's Vectorization lecture) — but an unbounded
    # batch on a CPU-only laptop with a 10,000-row CSV can spike memory
    # and make the UI look frozen with no feedback. Chunking into fixed-
    # size batches keeps memory bounded and lets us show incremental
    # progress, while each chunk is still processed as a true batch
    # (a list passed to the pipeline), not a Python for-loop calling the
    # model one review at a time.
    batch_size: int = 16
    max_review_chars: int = 2000  # guard against pathological/huge pasted text


@dataclass(frozen=True)
class UIConfig:
    """Display-only constants consumed by app.py."""

    label_colors: dict = field(
        default_factory=lambda: {
            "Positive": "#2ecc71",
            "Negative": "#e74c3c",
            "Neutral": "#95a5a6",
        }
    )
    needs_review_color: str = "#f39c12"
    example_reviews: List[str] = field(
        default_factory=lambda: [
            "This product exceeded my expectations, the build quality is fantastic!",
            "Terrible experience, it broke after two days and support never replied.",
            "It's fine. Does what it says, nothing more, nothing less.",
            "Oh great, ANOTHER broken product. Just what I needed.",
        ]
    )


@dataclass(frozen=True)
class AppConfig:
    """Top-level aggregate configuration consumed by app.py and src/."""

    model: ModelConfig = field(default_factory=ModelConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    sample_csv_path: Path = SAMPLE_DIR / "sample_reviews.csv"
    log_dir: Path = LOG_DIR


CONFIG = AppConfig()


def get_config() -> AppConfig:
    """Return the singleton application configuration.

    HIGHLIGHTS: mirrors the `get_config()` pattern used in the Week 7 and
    Week 8 projects — a module-level singleton instance handed out by a
    tiny accessor function, rather than every caller instantiating its own
    `AppConfig()`. This keeps config values consistent across `app.py`,
    `src/sentiment/classifier.py`, and the notebook within a single run,
    and gives us one obvious place to swap in env-var overrides later
    without touching every call site.
    """
    return CONFIG
