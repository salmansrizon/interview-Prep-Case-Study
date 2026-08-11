"""
Streamlit entry point for the Semantic CV-to-Job Matcher.

HIGHLIGHTS: this file NEVER imports ``sentence_transformers``, ``pypdf``, or
``docx`` directly, and never calls ``SentenceTransformer(...)`` or a PDF/DOCX
parser itself. Every piece of "real work" is delegated to ``src/``:

  - Text extraction (plain/PDF/DOCX -> str)   -> src.extraction.text_extractor
  - Embedding (str -> vector)                  -> src.embeddings.service
  - Ranking (vectors -> sorted MatchResults)   -> src.matching.ranker

That boundary is deliberate, not incidental. It's what lets those three
modules stay independently testable (tests/test_pipeline.py exercises them
without ever starting Streamlit or loading the real ~61MB model), and it's
what keeps this UI file focused on ONE job — layout, input widgets, and
displaying results — instead of being a 400-line file that mixes ML calls,
file parsing, and UI code together. If the embedding backend or the parsing
library ever changes, this file shouldn't need to change at all.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from config import get_config
from src.embeddings.service import embed
from src.extraction.text_extractor import ExtractionError, extract_text
from src.matching.ranker import MatchResult, cosine_similarity, rank_cvs

cfg = get_config()

st.set_page_config(
    page_title=cfg.app.page_title,
    page_icon=cfg.app.page_icon,
    layout=cfg.app.layout,
)


# ─────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────

def _read_text_input(label: str, key_prefix: str) -> str | None:
    """Render a "paste text OR upload a file" widget pair and return
    extracted plain text, or None if nothing usable was provided.

    HIGHLIGHTS: pasted text and uploaded files are two different input
    shapes (a str vs. raw bytes with a filename/extension), but both must
    end up going through the SAME ``extract_text()`` dispatcher in
    src/extraction/text_extractor.py — that's what guarantees a pasted job
    description and an uploaded PDF job description are cleaned up
    identically before they ever reach the embedding model. Centralizing
    that logic here (rather than duplicating this widget pair once for the
    job description and once per CV) is what makes it trivial to support
    "however many CVs the user wants" in the Match tab below.
    """
    tab_paste, tab_upload = st.tabs(["📋 Paste text", "📁 Upload file"])

    with tab_paste:
        pasted = st.text_area(
            f"Paste {label} text",
            key=f"{key_prefix}_paste",
            height=180,
            label_visibility="collapsed",
            placeholder=f"Paste the {label} here...",
        )

    with tab_upload:
        uploaded = st.file_uploader(
            f"Upload {label} ({', '.join(cfg.upload.allowed_extensions)})",
            type=list(cfg.upload.allowed_extensions),
            key=f"{key_prefix}_upload",
            label_visibility="collapsed",
        )

    # Uploaded file takes priority if both are provided — a user who
    # uploads a file almost certainly didn't mean to also submit whatever
    # leftover text happens to be sitting in the paste box from a previous
    # attempt.
    if uploaded is not None:
        filetype = uploaded.name.rsplit(".", 1)[-1] if "." in uploaded.name else "txt"
        try:
            return extract_text(uploaded.read(), filetype=filetype)
        except ExtractionError as exc:
            st.error(f"Could not read **{uploaded.name}**: {exc}")
            return None

    if pasted and pasted.strip():
        try:
            return extract_text(pasted, filetype="plain")
        except ExtractionError as exc:
            st.error(f"Could not use pasted {label}: {exc}")
            return None

    return None


def _results_to_dataframe(results: list[MatchResult]) -> pd.DataFrame:
    """Convert ranker output into a display-ready DataFrame.

    HIGHLIGHTS: this is presentation-only formatting (rounding, column
    naming) — it deliberately does NOT recompute or re-sort anything.
    src/matching/ranker.py already decided the order and the precise
    score; this function's only job is to make that decision readable in
    a Streamlit table.
    """
    precision = cfg.matching.similarity_precision
    return pd.DataFrame(
        {
            "Rank": [r.rank for r in results],
            "CV": [r.name for r in results],
            "Similarity": [round(r.score, precision) for r in results],
        }
    ).set_index("Rank")


# ─────────────────────────────────────────────────────────────────
# Page layout
# ─────────────────────────────────────────────────────────────────

st.title(f"{cfg.app.page_icon} {cfg.app.page_title}")
st.caption(
    "Rank CVs against a job description by **meaning**, not keyword overlap — "
    "powered by sentence embeddings + cosine similarity."
)

tab_overview, tab_match, tab_how = st.tabs(["🏠 Overview", "🎯 Match", "🔬 How It Works"])


# ── Overview tab ────────────────────────────────────────────────
with tab_overview:
    st.header("What this app does")
    st.markdown(
        """
Traditional keyword search checks whether a CV literally contains the words
in a job description — it misses a candidate who wrote **"ML"** when the job
description says **"machine learning"**, even though they mean the same
thing. This app instead converts both the job description and every CV into
**sentence embeddings** — vectors that capture *meaning*, not spelling — and
ranks CVs by how closely their vector points in the same direction as the
job description's vector (**cosine similarity**).

### The pipeline

1. **Extract** — pull plain text out of pasted text, a PDF, or a DOCX file
   (`src/extraction/text_extractor.py`)
2. **Embed** — convert each piece of text into a 384-dimensional vector
   using a local sentence-transformers model (`src/embeddings/service.py`)
3. **Rank** — score every CV against the job description with cosine
   similarity, sorted best match first (`src/matching/ranker.py`)

### Why this matters

The same idea — text to meaning-preserving vectors, then compare vectors —
powers modern search engines, recommendation systems, and retrieval-
augmented generation (RAG). This project is a small, fully local, from-
scratch demonstration of that exact mechanism, built on the same dot-product
math taught in the Week 2 NumPy lecture.

Head to the **🎯 Match** tab to try it, or **🔬 How It Works** for the model
details.
        """
    )

    with st.expander("Quick start"):
        st.markdown(
            """
1. Go to the **Match** tab.
2. Paste or upload a job description.
3. Paste or upload one or more CVs.
4. Click **Match** and review the ranked results.

The first click loads the embedding model (~61MB download, needs internet
once) — every click after that runs fully offline.
            """
        )


# ── Match tab ────────────────────────────────────────────────────
with tab_match:
    st.header("Job Description")
    job_text = _read_text_input("job description", key_prefix="job")

    st.divider()

    st.header("Candidate CVs")
    num_cvs = st.number_input(
        "How many CVs do you want to compare?",
        min_value=1,
        max_value=20,
        value=2,
        step=1,
        help="Add as many CVs as you want to rank against the job description.",
    )

    cv_texts: dict[str, str] = {}
    for i in range(int(num_cvs)):
        with st.expander(f"CV #{i + 1}", expanded=(i < 2)):
            cv_name = st.text_input(
                "Label (optional)",
                value=f"CV #{i + 1}",
                key=f"cv_label_{i}",
            )
            text = _read_text_input(f"CV #{i + 1}", key_prefix=f"cv_{i}")
            if text:
                # Guard against near-empty extractions slipping into the
                # embedding step — see MatchingConfig.min_text_length_chars
                # in config.py for why this threshold exists.
                if len(text) < cfg.matching.min_text_length_chars:
                    st.warning(
                        f"{cv_name}: extracted text is very short "
                        f"({len(text)} chars) — this may rank inaccurately."
                    )
                cv_texts[cv_name or f"CV #{i + 1}"] = text

    st.divider()

    match_clicked = st.button("🎯 Match", type="primary", use_container_width=True)

    if match_clicked:
        if not job_text:
            st.error("Please provide a job description (paste text or upload a file).")
        elif not cv_texts:
            st.error("Please provide at least one CV (paste text or upload a file).")
        else:
            with st.spinner(
                "Embedding text and computing similarity scores "
                "(first run downloads the model, ~61MB)..."
            ):
                # HIGHLIGHTS: this is the ONLY place in app.py that calls
                # into src.embeddings.service / src.matching.ranker. We
                # embed the job description and all CVs together in one
                # embed() call rather than one call per CV — a single
                # batched call lets sentence-transformers use its
                # `encode_batch_size` (config.py) efficiently instead of
                # paying model-call overhead once per CV.
                cv_names = list(cv_texts.keys())
                all_texts = [job_text] + list(cv_texts.values())
                vectors = embed(all_texts)
                job_vector, cv_vectors = vectors[0], vectors[1:]

                results = rank_cvs(job_vector, cv_vectors, cv_names)

            st.success(f"Ranked {len(results)} CV(s) against the job description.")

            top = results[0]
            st.metric(
                label="🏆 Top Match",
                value=top.name,
                delta=f"similarity {top.score:.{cfg.matching.similarity_precision}f}",
            )

            df = _results_to_dataframe(results)

            col_table, col_chart = st.columns([1, 1])
            with col_table:
                st.subheader("Ranked Results")

                def _highlight_top(row: pd.Series) -> list[str]:
                    return [
                        "background-color: #fff3cd; font-weight: 600" if row.name == 1 else ""
                        for _ in row
                    ]

                st.dataframe(
                    df.style.apply(_highlight_top, axis=1).format({"Similarity": "{:.4f}"}),
                    use_container_width=True,
                )

            with col_chart:
                st.subheader("Similarity Scores")
                chart_df = df.set_index("CV")[["Similarity"]]
                st.bar_chart(chart_df)


# ── How It Works tab ─────────────────────────────────────────────
with tab_how:
    st.header("The Embedding Model")
    st.markdown(
        f"""
This app embeds text with **`{cfg.embedding.model_name}`**
(`{cfg.embedding.embedding_dim}`-dimensional vectors), loaded lazily on the
first **Match** click — see `src/embeddings/service.py`. It's a portfolio-
scale CV matcher, so CPU-only real-time response matters more than the last
couple points of accuracy a heavier model would buy.
        """
    )

    st.subheader("Model Reference Table")
    model_table = pd.DataFrame(
        [
            {
                "Model": "paraphrase-MiniLM-L3-v2 (used here)",
                "Use Case": "Fast, local semantic similarity",
                "Advantage": "Small (~61MB), fast on CPU",
                "Limitation": "Slightly less accurate than larger models",
            },
            {
                "Model": "all-MiniLM-L6-v2",
                "Use Case": "General-purpose sentence embeddings",
                "Advantage": "Good accuracy, still small",
                "Limitation": "~50% larger than the L3 variant, a bit slower",
            },
            {
                "Model": "all-mpnet-base-v2",
                "Use Case": "High-accuracy semantic search",
                "Advantage": "Most accurate in this model family",
                "Limitation": "Much heavier (~420MB), slow on CPU",
            },
            {
                "Model": "OpenAI text-embedding-3-small",
                "Use Case": "Cloud-based embeddings",
                "Advantage": "No local compute required",
                "Limitation": "Not local — per-call cost, data leaves the machine",
            },
        ]
    )
    st.table(model_table.set_index("Model"))

    st.subheader("Local Model: download once, then fully offline")
    st.markdown(
        """
`paraphrase-MiniLM-L3-v2` is downloaded from the HuggingFace Hub **the first
time it's used** (needs internet), then cached on disk. Every run after that
is **100% local** — no network call, no per-request cost, unlike a cloud
embedding API where every request leaves the machine. This app never loads
the model at import time (see `src/embeddings/service.py`'s lazy `get_model()`)
— only the first time you click **Match** — so simply opening this app or
running its tests never silently triggers a download.
        """
    )

    st.subheader("Why Cosine Similarity?")
    st.markdown(
        r"""
$$\text{similarity}(a, b) = \frac{a \cdot b}{\lVert a \rVert \, \lVert b \rVert}$$

Cosine similarity measures the **angle** between two vectors — how much they
point in the same semantic direction — while ignoring their **magnitude**
(length). A long, detailed CV and a short job description will naturally
produce embeddings of different magnitude purely from text length, even when
they describe the same role. Dividing out both vectors' norms cancels that
out, leaving a score that reflects meaning, not verbosity. See
`src/matching/ranker.py` for the hand-rolled NumPy implementation — the same
dot-product math from the Week 2 lecture, applied to sentence vectors.
        """
    )

    with st.expander("🔎 Try it: quick similarity sandbox"):
        st.caption(
            "A tiny, self-contained demo — type two short phrases and see "
            "their cosine similarity, without needing the Match tab's full "
            "job description / CV inputs."
        )
        col_a, col_b = st.columns(2)
        with col_a:
            phrase_a = st.text_input("Phrase A", value="machine learning engineer")
        with col_b:
            phrase_b = st.text_input("Phrase B", value="deep learning developer")

        if st.button("Compare phrases"):
            if phrase_a.strip() and phrase_b.strip():
                vecs = embed([phrase_a, phrase_b])
                score = cosine_similarity(vecs[0], vecs[1])
                st.metric("Cosine Similarity", f"{score:.{cfg.matching.similarity_precision}f}")
            else:
                st.warning("Enter both phrases to compare.")
