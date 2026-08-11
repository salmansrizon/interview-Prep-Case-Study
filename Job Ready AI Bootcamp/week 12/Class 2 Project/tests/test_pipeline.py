"""
Tests for the Local Document Q&A Bot (RAG Engine) pipeline.

Run with: pytest tests/ -q

HIGHLIGHTS: এই test file কখনো real Ollama server-এর সাথে যোগাযোগ করে না —
যে একমাত্র জায়গায় network touch হতে পারত (src/llm/client.py-এর
``generate()``, যেটা নিজের ভেতরে ``import ollama`` করে), সেখানে আমরা
``sys.modules['ollama']``-কে monkeypatch দিয়ে একটা fake module দিয়ে বদলে
দিই — ঠিক Week 11-এর tests/test_pipeline.py-এর ``fake_ollama_module``
fixture-এর একই প্যাটার্ন (ADR 0006-এর standing testing constraint)।

Chunking আর vector-store tests real ChromaDB (একটা tmp_path
PersistentClient-এ) ব্যবহার করে hand-crafted embedding vector দিয়ে — কোনো
sentence-transformers model লোড/ডাউনলোড ছাড়াই, যাতে এই test suite CI-তে
পুরোপুরি অফলাইন আর দ্রুত চলে। ChromaDB নিজেই সম্পূর্ণ local (কোনো সার্ভার
লাগে না), তাই এখানে mocking-ও লাগে না।
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass

import pytest

from src.chunking.chunker import Chunk, chunk_page_texts
from src.rag.pipeline import (
    NOT_FOUND_MESSAGE,
    Citation,
    format_citation,
)
from src.vectorstore.store import RetrievedChunk


# ─────────────────────────────────────────────────────────────────
# Chunking: boundaries, overlap, and page-metadata preservation
# ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _FakeChunkingConfig:
    chunk_size_chars: int = 80
    chunk_overlap_chars: int = 20
    min_chunk_size_chars: int = 5


class _FakeConfig:
    chunking = _FakeChunkingConfig()


@pytest.fixture
def small_chunking_config(monkeypatch):
    """Swap in a small chunk_size/overlap so tests can exercise multiple
    chunks and overlap behavior without huge input text.

    HIGHLIGHTS: config.py-এর ``get_config()`` module-level singleton cache
    করে, তাই সরাসরি ``ChunkingConfig`` field বদলানো যায় না (frozen
    dataclass)। এর বদলে আমরা src.chunking.chunker-এর ভেতরে imported
    ``get_config`` name-টাই monkeypatch করি — এটাই সেই seam যা প্রতিটা
    টেস্টকে নিজের chunk_size ঠিক করতে দেয়, config.py-এর real singleton
    না ছুঁয়ে।
    """
    monkeypatch.setattr("src.chunking.chunker.get_config", lambda: _FakeConfig())


def test_chunk_page_texts_preserves_page_metadata(small_chunking_config):
    pages = [
        (1, "Employees get twelve sick days per year. Sick leave must be "
            "reported early. Managers approve all requests promptly."),
        (2, "Annual leave accrues monthly for staff. Unused leave can carry "
            "over once. Carryover is capped at five days."),
    ]
    chunks = chunk_page_texts(pages, source="policy.pdf")

    assert len(chunks) >= 2
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.source == "policy.pdf" for c in chunks)
    # First chunk must attribute to page 1, last chunk to page 2 — the
    # document starts on page 1 and ends on page 2.
    assert chunks[0].page == 1
    assert chunks[-1].page == 2
    # Every page number present in the input must appear on at least one
    # chunk — no page silently dropped.
    assert {c.page for c in chunks} == {1, 2}


def test_chunk_page_texts_respects_target_chunk_size(small_chunking_config):
    pages = [
        (1, "Employees get twelve sick days per year. Sick leave must be "
            "reported early. Managers approve all requests promptly. "
            "Annual leave accrues monthly for staff. Unused leave can carry "
            "over once. Carryover is capped at five days."),
    ]
    chunks = chunk_page_texts(pages, source="policy.pdf")

    cfg = _FakeChunkingConfig()
    # Every chunk should be close to (not wildly over) the target size —
    # a small allowance for the "\n\n" join between merged sentences.
    for c in chunks:
        assert len(c.text) <= cfg.chunk_size_chars + 5


def test_chunk_page_texts_overlap_carries_context_across_boundary(small_chunking_config):
    """Overlap should mean the tail of one chunk reappears at the head of
    the next — this is what stops a sentence from being lost entirely at a
    chunk border (Class 1 Lecture, section 3.4's overlap explanation)."""
    pages = [
        (1, "Employees get twelve sick days per year. Sick leave must be "
            "reported early. Managers approve all requests promptly."),
        (2, "Annual leave accrues monthly for staff. Unused leave can carry "
            "over once. Carryover is capped at five days."),
    ]
    chunks = chunk_page_texts(pages, source="policy.pdf")

    assert len(chunks) >= 2
    cfg = _FakeChunkingConfig()
    tail_of_first = chunks[0].text[-cfg.chunk_overlap_chars :]
    assert tail_of_first in chunks[1].text


def test_chunk_page_texts_no_paragraphs_returns_empty(small_chunking_config):
    assert chunk_page_texts([], source="empty.pdf") == []
    assert chunk_page_texts([(1, "   ")], source="blank.pdf") == []


# ─────────────────────────────────────────────────────────────────
# Vector store: metadata filtering via ChromaDB's `where` clause
# ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _FakeVectorStoreConfig:
    persist_directory: str
    collection_name: str = "test_collection"


@dataclass(frozen=True)
class _FakeRetrievalConfig:
    top_k: int = 5
    max_distance_threshold: float = 22.0


class _FakeStoreConfig:
    def __init__(self, persist_directory: str):
        self.vectorstore = _FakeVectorStoreConfig(persist_directory=persist_directory)
        self.retrieval = _FakeRetrievalConfig()


@pytest.fixture
def isolated_vectorstore(tmp_path, monkeypatch):
    """Point src.vectorstore.store at a fresh temp ChromaDB directory and
    reset its module-level client/collection singletons so tests don't
    leak state into each other or into the real ./data/chroma_db.

    HIGHLIGHTS: hand-crafted 3-dim vectors ব্যবহার করা হয়েছে এখানে —
    real sentence-transformers embedding না — যাতে এই test কোনো model
    download/load ছাড়াই, দ্রুত আর deterministic ভাবে ঠিক কোন vector কোন
    metadata-র সাথে কতটা "কাছাকাছি" তা নিয়ন্ত্রণ করতে পারে।
    """
    import src.vectorstore.store as store_module

    fake_config = _FakeStoreConfig(persist_directory=str(tmp_path / "chroma_db"))
    monkeypatch.setattr(store_module, "get_config", lambda: fake_config)
    store_module._client = None
    store_module._collection = None
    yield store_module
    store_module._client = None
    store_module._collection = None


def test_query_applies_source_metadata_filter(isolated_vectorstore):
    store_module = isolated_vectorstore
    collection = store_module.get_collection()

    # Two documents. The "it_guidelines.pdf" chunk is embedded CLOSER to
    # the query vector than the "hr_policy.pdf" chunk, so an unfiltered
    # query would return it first — the filter must override that and
    # return ONLY the hr_policy.pdf chunk when asked to.
    collection.add(
        ids=["hr-1", "it-1"],
        embeddings=[[1.0, 0.0, 0.0], [0.9, 0.1, 0.0]],
        documents=["Sick leave policy text.", "Laptop encryption policy text."],
        metadatas=[
            {"source": "hr_policy.pdf", "page": 1},
            {"source": "it_guidelines.pdf", "page": 1},
        ],
    )

    query_vector = [0.9, 0.1, 0.0]  # closest to the it_guidelines chunk

    unfiltered = store_module.query(query_vector, top_k=2)
    assert unfiltered[0].source == "it_guidelines.pdf"

    filtered = store_module.query(query_vector, top_k=2, source_filter="hr_policy.pdf")
    assert len(filtered) == 1
    assert filtered[0].source == "hr_policy.pdf"
    assert filtered[0].page == 1


def test_list_indexed_sources_returns_sorted_unique_sources(isolated_vectorstore):
    store_module = isolated_vectorstore
    collection = store_module.get_collection()
    collection.add(
        ids=["a", "b", "c"],
        embeddings=[[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
        documents=["x", "y", "z"],
        metadatas=[
            {"source": "zeta.pdf", "page": 1},
            {"source": "alpha.pdf", "page": 1},
            {"source": "alpha.pdf", "page": 2},
        ],
    )
    assert store_module.list_indexed_sources() == ["alpha.pdf", "zeta.pdf"]


def test_query_on_empty_collection_returns_empty_list(isolated_vectorstore):
    store_module = isolated_vectorstore
    assert store_module.query([1.0, 0.0], top_k=3) == []


# ─────────────────────────────────────────────────────────────────
# RAG pipeline: the similarity-threshold "not found" guardrail
# ─────────────────────────────────────────────────────────────────

@pytest.fixture
def fake_ollama_module(monkeypatch):
    """Install a fake ``ollama`` module into sys.modules — see this file's
    module docstring for why this works (local import inside generate())."""
    calls = []

    def fake_chat(model, messages, options):
        calls.append({"model": model, "messages": messages, "options": options})
        return {"message": {"content": "  Employees get 12 sick days per year.  "}}

    fake_module = types.SimpleNamespace(chat=fake_chat)
    monkeypatch.setitem(sys.modules, "ollama", fake_module)
    return calls


def test_answer_question_triggers_not_found_on_high_distance(monkeypatch, fake_ollama_module):
    """A synthetic LOW-similarity (high-distance) retrieval result must
    trigger the guardrail and skip calling the LLM entirely."""
    import src.rag.pipeline as pipeline_module

    high_distance_chunk = RetrievedChunk(
        text="Unrelated content about gardening.",
        source="misc.pdf",
        page=1,
        distance=40.0,  # well above config default max_distance_threshold (22.0)
    )
    monkeypatch.setattr(pipeline_module, "embed", lambda texts: [[0.0, 0.0]])
    monkeypatch.setattr(pipeline_module, "query", lambda *a, **k: [high_distance_chunk])

    result = pipeline_module.answer_question("What is the capital of France?")

    assert result.found is False
    assert result.answer == NOT_FOUND_MESSAGE
    assert result.citations == []
    # The guardrail must skip the LLM entirely — no ollama.chat() call.
    assert len(fake_ollama_module) == 0


def test_answer_question_generates_answer_on_low_distance(monkeypatch, fake_ollama_module):
    """A synthetic HIGH-similarity (low-distance) retrieval result must
    skip the guardrail and produce a real answer with citations."""
    import src.rag.pipeline as pipeline_module

    low_distance_chunk = RetrievedChunk(
        text="Employees are entitled to 12 days of paid sick leave per year.",
        source="hr_policy.pdf",
        page=3,
        distance=10.0,  # well below the threshold
    )
    monkeypatch.setattr(pipeline_module, "embed", lambda texts: [[0.0, 0.0]])
    monkeypatch.setattr(pipeline_module, "query", lambda *a, **k: [low_distance_chunk])

    result = pipeline_module.answer_question("How many sick days do I get?")

    assert result.found is True
    assert result.answer == "Employees get 12 sick days per year."
    assert result.citations == [Citation(source="hr_policy.pdf", page=3, distance=10.0)]
    assert len(fake_ollama_module) == 1
    # The system prompt sent to the model must contain the guardrail instruction.
    system_message = fake_ollama_module[0]["messages"][0]["content"]
    assert "not found in the documents" in system_message.lower()


def test_answer_question_empty_question_short_circuits(monkeypatch, fake_ollama_module):
    import src.rag.pipeline as pipeline_module

    result = pipeline_module.answer_question("   ")
    assert result.found is False
    assert len(fake_ollama_module) == 0


def test_answer_question_dedupes_repeated_citations(monkeypatch, fake_ollama_module):
    """Overlap can cause the same (source, page) to be retrieved twice —
    the citation list must collapse that to one entry."""
    import src.rag.pipeline as pipeline_module

    same_page_chunk_a = RetrievedChunk(text="First half.", source="doc.pdf", page=2, distance=8.0)
    same_page_chunk_b = RetrievedChunk(text="Second half.", source="doc.pdf", page=2, distance=9.0)
    monkeypatch.setattr(pipeline_module, "embed", lambda texts: [[0.0, 0.0]])
    monkeypatch.setattr(
        pipeline_module, "query", lambda *a, **k: [same_page_chunk_a, same_page_chunk_b]
    )

    result = pipeline_module.answer_question("A question.")
    assert len(result.citations) == 1
    assert result.citations[0].page == 2


# ─────────────────────────────────────────────────────────────────
# Citation formatting
# ─────────────────────────────────────────────────────────────────

def test_format_citation_produces_readable_string():
    citation = Citation(source="hr_policy.pdf", page=4, distance=9.5)
    assert format_citation(citation) == "hr_policy.pdf, page 4"


# ─────────────────────────────────────────────────────────────────
# LLM client: Ollama mocked, connection failure wrapped
# ─────────────────────────────────────────────────────────────────

def test_generate_calls_mocked_ollama_and_strips_response(fake_ollama_module):
    from src.llm.client import generate

    result = generate(prompt="Answer this.", system_prompt="Be honest.")

    assert result == "Employees get 12 sick days per year."
    assert len(fake_ollama_module) == 1
    call = fake_ollama_module[0]
    assert call["messages"][0]["role"] == "system"
    assert call["messages"][1]["role"] == "user"


def test_generate_wraps_ollama_failures_in_ollama_connection_error(monkeypatch):
    from src.llm.client import OllamaConnectionError, generate

    def broken_chat(model, messages, options):
        raise ConnectionRefusedError("no server listening")

    fake_module = types.SimpleNamespace(chat=broken_chat)
    monkeypatch.setitem(sys.modules, "ollama", fake_module)

    with pytest.raises(OllamaConnectionError):
        generate(prompt="X", system_prompt="Y")
