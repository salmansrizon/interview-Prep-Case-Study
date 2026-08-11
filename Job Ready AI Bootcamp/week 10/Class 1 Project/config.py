"""
Centralized configuration for the Semantic CV-to-Job Matcher.

Follows the dataclass-based config style used in week 7/Class 2 Project's
``src/config.py`` — a single place to change model choice, size limits, and
display formatting without hunting through the app/src code.

HIGHLIGHTS: the embedding model name lives here as *data*, not as a literal
string scattered through src/embeddings/service.py, app.py, and tests. If a
future student wants to try ``all-mpnet-base-v2`` for higher accuracy (see
the Model Reference Table below), they change ONE line here instead of
grepping the whole repo. This is the same reasoning as week 7/8's config.py
pattern: config is the single source of truth for "what model, what
thresholds" so the rest of the code stays generic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# ── Paths ─────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class EmbeddingConfig:
    """Everything about *which* sentence-embedding model we use and how.

    HIGHLIGHTS: ``model_name`` is deliberately the ONLY thing that couples
    this project to a specific model. src/embeddings/service.py reads this
    value lazily (inside a function, on first use) rather than importing
    and loading the model at module import time — see that file for the
    full "why lazy loading" explanation. Keeping the model name here (not
    hardcoded in service.py) also means the Model Reference Table in the
    README and the HIGHLIGHTS comment at the SentenceTransformer(...) call
    site both point back to this single field when explaining the choice.
    """

    # sentence-transformers/paraphrase-MiniLM-L3-v2:
    #   ~61MB, 384-dim output, CPU-fast. Chosen per the Week 10 lecture's
    #   Model Reference Table (section 4.1) for a portfolio-scale demo
    #   where CPU-only real-time response matters more than the last
    #   couple points of accuracy that a heavier model (e.g.
    #   all-mpnet-base-v2, ~420MB) would buy us.
    model_name: str = "sentence-transformers/paraphrase-MiniLM-L3-v2"

    # Output embedding dimensionality of paraphrase-MiniLM-L3-v2. Recorded
    # here (rather than discovered at runtime) so tests and UI code can
    # reference an expected shape without loading the real model.
    embedding_dim: int = 384

    # Batch size for sentence-transformers' .encode(). CPU inference on a
    # handful of CVs doesn't need tuning, but exposing it in config (rather
    # than hardcoding it in service.py) keeps the door open for a student
    # to bump it if they scale this up to hundreds of CVs.
    encode_batch_size: int = 16


@dataclass(frozen=True)
class UploadConfig:
    """Limits and accepted formats for file uploads.

    HIGHLIGHTS: max_upload_size_mb exists mainly as a UI guardrail —
    Streamlit's file_uploader already enforces a global limit via
    .streamlit/config.toml, but surfacing the *intended* limit here lets
    app.py show a clear, project-specific message instead of a generic
    Streamlit error.
    """

    max_upload_size_mb: int = 10
    allowed_extensions: tuple[str, ...] = ("pdf", "docx", "txt")


@dataclass(frozen=True)
class MatchingConfig:
    """Display/formatting knobs for the ranking output.

    HIGHLIGHTS: ``similarity_precision`` controls how many decimal places
    the UI shows. Cosine similarity for two independently-authored texts
    (a CV vs. a JD) rarely needs more than 3-4 significant digits of
    precision to be meaningful to a human reader — showing 15 decimal
    places would just be float noise, not signal.
    """

    similarity_precision: int = 4          # decimal places shown in the UI
    min_text_length_chars: int = 20        # below this, extraction is probably junk


@dataclass(frozen=True)
class AppConfig:
    """Streamlit page-level settings."""

    page_title: str = "Semantic CV-to-Job Matcher"
    page_icon: str = "🧭"
    layout: str = "wide"


@dataclass(frozen=True)
class Config:
    """Top-level config aggregating all sub-configs.

    HIGHLIGHTS: composing several small frozen dataclasses (rather than
    one giant flat dataclass) mirrors week 7's Config/ProjectConfig/
    PathsConfig/... pattern — each concern (embeddings, uploads, matching,
    app) can be read, tested, and reasoned about independently. ``frozen``
    is used throughout because config values are meant to be constants for
    the lifetime of the process, not mutable state — accidentally
    reassigning ``config.embedding.model_name`` mid-run would be a bug we'd
    rather catch immediately (a FrozenInstanceError) than debug later.
    """

    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    upload: UploadConfig = field(default_factory=UploadConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    app: AppConfig = field(default_factory=AppConfig)


_config_instance: Config | None = None


def get_config() -> Config:
    """Get or create the process-wide singleton Config instance.

    HIGHLIGHTS: same singleton pattern as week 7's ``get_config()`` — a
    module-level cache means every caller (app.py, src/embeddings/service.py,
    tests) sees the exact same config object, and building it is cheap
    (just dataclass construction, no I/O), so there's no meaningful cost to
    reusing this pattern here even though we don't load from YAML.
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance
