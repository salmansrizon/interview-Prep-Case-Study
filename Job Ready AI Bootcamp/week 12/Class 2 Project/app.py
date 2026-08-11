"""
Streamlit entry point for the Local Document Q&A Bot (RAG Engine).

HIGHLIGHTS: এই file কখনো ``chromadb``, ``sentence_transformers``, ``ollama``,
বা ``pypdf`` সরাসরি import করে না, আর কখনো তাদের কোনো object সরাসরি ছোঁয় না।
প্রতিটা "real work" ``src/``-এ delegate করা হয়:

  - PDF extraction (bytes -> per-page text)   -> src.extraction.pdf_extractor
  - Chunking (page text -> Chunk objects)      -> src.chunking.chunker
  - Indexing + retrieval (Chunk -> ChromaDB)   -> src.vectorstore.store
  - Orchestration (question -> answer+cites)   -> src.rag.pipeline

এই boundary-টা incidental না, ইচ্ছাকৃত। এটাই src/-এর প্রতিটা module-কে
Streamlit ছাড়াই স্বাধীনভাবে টেস্ট করা সম্ভব করে (tests/test_pipeline.py
কখনো streamlit চালু করে না), আর এই UI file-কে ONE কাজে ফোকাসড রাখে —
layout, input widget, আর ফলাফল দেখানো — 500-লাইনের একটা ফাইলে ML call,
PDF parsing, আর UI code মিশিয়ে না ফেলে।
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from config import get_config
from src.chunking.chunker import chunk_page_texts
from src.extraction.pdf_extractor import ExtractionError, extract_pages
from src.llm.client import OllamaConnectionError
from src.rag.pipeline import answer_question, format_citation
from src.vectorstore.store import index_chunks, list_indexed_sources

cfg = get_config()

st.set_page_config(
    page_title=cfg.app.page_title,
    page_icon=cfg.app.page_icon,
    layout=cfg.app.layout,
)

# HIGHLIGHTS: ``st.session_state`` এখানে শুধু "এই session-এ এখন পর্যন্ত কী
# ইনডেক্স হয়েছে" track করতে ব্যবহার হচ্ছে (UI feedback-এর জন্য) — এটা
# vectorstore-এর SOURCE OF TRUTH না। আসল source of truth সবসময়
# ``list_indexed_sources()`` (ChromaDB নিজেই), যেটা Ask ট্যাবের
# document-picker populate করে — তাই একটা আগের session-এ ইনডেক্স করা
# ডকুমেন্টও (ডিস্কে persist করা ChromaDB collection থেকে) সঠিকভাবে picker-এ
# দেখা যায়।
if "indexed_this_session" not in st.session_state:
    st.session_state["indexed_this_session"] = []


# ─────────────────────────────────────────────────────────────────
# Page layout
# ─────────────────────────────────────────────────────────────────

st.title(f"{cfg.app.page_icon} {cfg.app.page_title}")
st.caption(
    "Ask questions about your own PDFs — answered **strictly from the "
    "documents**, with source + page citations, 100% local via ChromaDB + Ollama."
)

tab_overview, tab_upload, tab_ask, tab_how = st.tabs(
    ["🏠 Overview", "📤 Upload & Index", "💬 Ask", "🔬 How It Works"]
)


# ── Overview tab ────────────────────────────────────────────────
with tab_overview:
    st.header("What this app does")
    st.markdown(
        """
A plain LLM answers questions from its **training data** — which may be
outdated, generic, or entirely wrong for *your* documents (it has never seen
your company's handbook). This app instead uses **Retrieval-Augmented
Generation (RAG)**: it retrieves the most relevant passages from PDFs *you*
upload, hands them to a local LLM as the only source of truth, and instructs
the model to answer **strictly from that context** — like a student taking
an **open-book exam**.

### The pipeline

1. **Extract** — pull page-labeled text out of each uploaded PDF
   (`src/extraction/pdf_extractor.py`)
2. **Chunk** — split into overlap-preserving, paragraph-aware chunks, each
   tagged with `{source, page}` (`src/chunking/chunker.py`)
3. **Embed + Index** — convert chunks to vectors and store them in ChromaDB
   (`src/embeddings/service.py`, `src/vectorstore/store.py`)
4. **Retrieve** — on a question, find the top-K most similar chunks,
   optionally restricted to one document (`src/vectorstore/store.py`)
5. **Guardrail** — if nothing retrieved is similar enough, skip generation
   entirely and say so honestly (`src/rag/pipeline.py`)
6. **Generate** — a local LLM (Ollama) answers using only the retrieved
   context, and cites its sources (`src/llm/client.py`, `src/rag/pipeline.py`)

### Why this matters

RAG is the standard architecture behind production "chat with your docs"
tools — internal knowledge bases, support bots, legal/compliance search. This
project is a small, fully local, from-scratch build of that exact pattern.

Head to **📤 Upload & Index** to add PDFs, then **💬 Ask** to query them.
        """
    )

    with st.expander("Quick start"):
        st.markdown(
            """
1. Go to **📤 Upload & Index** and upload one or more PDFs.
2. Wait for chunking + indexing to finish (progress shown per file).
3. Go to **💬 Ask**, pick "Search all documents" or one specific document,
   and type a question.
4. Read the answer and its citations — or the honest "not found in the
   documents" message if nothing relevant was indexed.

The first indexing click loads the embedding model (~61MB download, needs
internet once); every click after that runs fully offline. Answering
questions needs a running local Ollama server with the model pulled — see
the README's Quick Start.
            """
        )


# ── Upload & Index tab ──────────────────────────────────────────
with tab_upload:
    st.header("Upload PDFs")
    st.caption(
        "Text-layer extraction only — scanned/image-only PDFs with no "
        "embedded text will be rejected with a clear message (no OCR fallback)."
    )

    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    index_clicked = st.button("📥 Chunk & Index", type="primary", disabled=not uploaded_files)

    if index_clicked and uploaded_files:
        progress = st.progress(0.0, text="Starting...")
        total_chunks = 0
        indexed_names: list[str] = []
        errors: list[str] = []

        for i, uploaded in enumerate(uploaded_files):
            fraction = i / len(uploaded_files)
            progress.progress(fraction, text=f"Extracting {uploaded.name}...")
            try:
                pages = extract_pages(uploaded.read(), filename=uploaded.name)
            except ExtractionError as exc:
                errors.append(f"**{uploaded.name}**: {exc}")
                continue

            progress.progress(fraction, text=f"Chunking {uploaded.name}...")
            chunks = chunk_page_texts(pages, source=uploaded.name)

            progress.progress(fraction, text=f"Embedding + indexing {uploaded.name}...")
            n_indexed = index_chunks(chunks)
            total_chunks += n_indexed
            indexed_names.append(uploaded.name)

        progress.progress(1.0, text="Done.")

        if indexed_names:
            st.session_state["indexed_this_session"].extend(indexed_names)
            st.success(
                f"Indexed {total_chunks} chunk(s) from {len(indexed_names)} "
                f"document(s): {', '.join(indexed_names)}"
            )
        for err in errors:
            st.error(err)

    st.divider()

    st.subheader("Currently indexed documents")
    # HIGHLIGHTS: এই তালিকা সরাসরি ChromaDB থেকে আসে (list_indexed_sources()),
    # session_state থেকে না — যাতে আগের session-এ ইনডেক্স করা ডকুমেন্টও
    # (একই ./data/chroma_db-তে persist করা) এখানে সঠিকভাবে দেখা যায়, শুধু
    # এই session-এর আপলোডই না।
    sources = list_indexed_sources()
    if sources:
        st.table(pd.DataFrame({"Document": sources}))
    else:
        st.info("No documents indexed yet. Upload a PDF above to get started.")


# ── Ask tab ──────────────────────────────────────────────────────
with tab_ask:
    st.header("Ask a question")

    available_sources = list_indexed_sources()
    if not available_sources:
        st.warning("No documents indexed yet — go to **📤 Upload & Index** first.")
    else:
        ALL_DOCS_LABEL = "🔎 Search all documents"
        picker_options = [ALL_DOCS_LABEL] + available_sources
        # HIGHLIGHTS: এই picker-ই metadata filtering-এর REAL, ইউজার-facing
        # entry point (README/spec-এর দাবি অনুযায়ী "cosmetic না")। picker-এর
        # সিলেকশন সরাসরি src.rag.pipeline.answer_question()-এর
        # ``source_filter`` argument হয়ে যায়, যেটা src.vectorstore.store.query()-এর
        # ChromaDB ``where`` clause পর্যন্ত পৌঁছায় — UI থেকে ডেটাবেস কুয়েরি
        # পর্যন্ত একটা সরাসরি, unbroken chain।
        selected = st.selectbox("Search scope", options=picker_options)
        source_filter = None if selected == ALL_DOCS_LABEL else selected

        question = st.text_input(
            "Your question",
            placeholder="e.g. How many sick leave days do I get?",
        )

        ask_clicked = st.button("💬 Ask", type="primary", disabled=not question.strip())

        if ask_clicked and question.strip():
            with st.spinner("Retrieving relevant passages and generating an answer..."):
                try:
                    result = answer_question(question, source_filter=source_filter)
                except OllamaConnectionError as exc:
                    st.error(
                        "Could not reach the local Ollama server. Is it running "
                        f"(`ollama serve`) and has the model been pulled "
                        f"(`ollama pull {cfg.llm.model}`)?\n\nDetails: {exc}"
                    )
                    result = None

            if result is not None:
                if not result.found:
                    st.warning(f"🚫 {result.answer}")
                else:
                    st.markdown("### Answer")
                    st.write(result.answer)

                    st.markdown("### Citations")
                    for citation in result.citations:
                        st.markdown(f"- 📄 {format_citation(citation)}")

                with st.expander("🔍 Retrieved chunks (debug view)"):
                    for chunk in result.retrieved_chunks:
                        st.markdown(
                            f"**{chunk.source}, page {chunk.page}** "
                            f"(distance={chunk.distance:.3f})"
                        )
                        st.text(chunk.text)


# ── How It Works tab ─────────────────────────────────────────────
with tab_how:
    st.header("The RAG Pipeline")
    st.markdown(
        """
```
PDFs -> extraction -> chunking -> embedding -> ChromaDB index
                                                     |
                                              [query] |
                                                     v
                              retrieval (+ optional metadata filter)
                                                     |
                                                     v
                                       augmented prompt (Open-Book Exam)
                                                     |
                                                     v
                                        Ollama generation (llama3.1:8b)
                                                     |
                                                     v
                                        answer + citations (or "not found")
```
        """
    )

    st.subheader("Vector Database Reference Table")
    vector_db_table = pd.DataFrame(
        [
            {
                "Database": "ChromaDB (used here)",
                "Use Case": "Local/small-to-medium scale projects",
                "Advantage": "No server setup, Python-native",
                "Limitation": "Limited performance at very large (10M+ vector) scale",
            },
            {
                "Database": "FAISS",
                "Use Case": "Research, very large scale, in-memory",
                "Advantage": "Extremely fast, GPU support",
                "Limitation": "Doesn't manage metadata/persistence itself — extra code needed",
            },
            {
                "Database": "Pinecone",
                "Use Case": "Production, managed cloud service",
                "Advantage": "Scalable, managed infrastructure",
                "Limitation": "Cloud-based — not local, subscription cost, data leaves the machine",
            },
            {
                "Database": "Weaviate",
                "Use Case": "Production, built-in hybrid search",
                "Advantage": "Self-host or cloud, powerful filtering",
                "Limitation": "More complex setup than ChromaDB",
            },
        ]
    )
    st.table(vector_db_table.set_index("Database"))

    st.subheader("LLM Model Reference Table")
    llm_table = pd.DataFrame(
        [
            {
                "Model": "llama3.1:8b (used here, default)",
                "Use Case": "General-purpose text generation, instruction-following",
                "Advantage": "Strong output quality, runs comfortably on 16GB RAM",
                "Limitation": "Tight on 8GB RAM machines, can be slow there",
            },
            {
                "Model": "llama3.2:3b (fallback)",
                "Use Case": "Low-resource machines",
                "Advantage": "Small, runs fine even on 8GB RAM",
                "Limitation": "Less accurate than the 8B model, weaker on complex instructions",
            },
            {
                "Model": "mistral:7b",
                "Use Case": "Fast inference, code/structured tasks",
                "Advantage": "Similar size to llama3.1, faster on some benchmarks",
                "Limitation": "Smaller community/fine-tune ecosystem than Llama",
            },
            {
                "Model": "OpenAI GPT-4 / Claude (Cloud API)",
                "Use Case": "When you need best-in-class output quality",
                "Advantage": "State-of-the-art benchmark performance",
                "Limitation": "Not local — per-call cost, data leaves the machine",
            },
        ]
    )
    st.table(llm_table.set_index("Model"))

    st.subheader('"Answers strictly from documents" — how it is enforced')
    st.markdown(
        f"""
This isn't just a prompt-wording promise — it's two enforced layers:

1. **Similarity-threshold guardrail** (`src/rag/pipeline.py`): if the best
   retrieved chunk's distance is above
   `config.retrieval.max_distance_threshold` (currently
   **{cfg.retrieval.max_distance_threshold}**), the LLM is **never called** —
   the app returns the honest "not found in the documents" message directly.
2. **System-prompt instruction**: even when relevant chunks ARE retrieved,
   the model is explicitly told to answer only from the given context and to
   say so plainly if the context doesn't actually contain the answer.

Try it yourself: ask a question completely unrelated to your uploaded PDFs
on the **💬 Ask** tab (Brain Teaser #1 from the Class 2 Lecture) and watch
the guardrail trigger.
        """
    )
