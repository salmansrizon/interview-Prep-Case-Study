"""
RAG core tests: real PDF bytes (fpdf2), real ChromaDB in a temp dir, fake embedder + fake LLM.

HIGHLIGHTS: Ollama বা embedding model লাগে না (ADR 0006)। Embedder-কে একটা
deterministic bag-of-words vector দিয়ে বদলানো হয়েছে — retrieval-এর logic (guardrail,
filter, idempotent upsert) সত্যিকারের Chroma-তে test হয়, শুধু model নকল।
"""
from __future__ import annotations

import hashlib

import pytest
from fpdf import FPDF

from app.rag.components import ChromaStore, LLMUnavailable, Usage
from app.rag.engine import NOT_FOUND, RagEngine
from app.rag.ingest import BadDocument, chunk_pages, extract_pages

VOCAB = ["sick", "leave", "days", "annual", "password", "laptop", "encryption", "characters", "manager", "carried"]


class FakeEmbedder:
    """Counts vocabulary words: same topic -> close vectors, unrelated text -> far (all zeros)."""

    def embed(self, texts):
        return [[float(t.lower().count(w)) * 10 for w in VOCAB] for t in texts]


class FakeLLM:
    def __init__(self, answer="Employees get 12 sick days [1].", fail=False):
        self.answer, self.fail, self.calls = answer, fail, 0

    def generate(self, system, prompt):
        self.calls += 1
        if self.fail:
            raise LLMUnavailable("down")
        return self.answer, Usage(prompt_tokens=100, completion_tokens=8, decode_ms=400)

    def stream(self, system, prompt):
        self.calls += 1
        if self.fail:
            raise LLMUnavailable("down")
        for w in self.answer.split(" "):
            yield "token", w + " "
        yield "usage", Usage(prompt_tokens=100, completion_tokens=8, decode_ms=400, ttft_ms=50)

    def ping(self):
        return True, "ok"


def make_pdf(*pages: str) -> bytes:
    pdf = FPDF()
    pdf.set_font("Helvetica", size=11)
    for text in pages:
        pdf.add_page()
        pdf.multi_cell(0, 6, text)
    return bytes(pdf.output())


HANDBOOK = make_pdf(
    "Sick leave: employees get 12 sick leave days per year. Report sick leave to the manager before 10am.\n\n"
    "Annual leave accrues monthly; up to 5 unused days can be carried over.",
    "Every laptop needs full-disk encryption. A password must have at least 14 characters.")


@pytest.fixture
def engine(tmp_path):
    return RagEngine(FakeEmbedder(), ChromaStore(str(tmp_path / "chroma"), "test"), FakeLLM(),
                     top_k=3, max_distance=5000.0, chunk_size=120, chunk_overlap=20)


# ── ingestion ─────────────────────────────────────────────────────────
def test_extract_pages_and_errors():
    pages = extract_pages(HANDBOOK, "h.pdf")
    assert [p for p, _ in pages] == [1, 2] and "sick" in pages[0][1].lower()
    with pytest.raises(BadDocument, match="not a PDF"):
        extract_pages(b"hello", "x.pdf")
    with pytest.raises(BadDocument, match="Could not open"):
        extract_pages(b"%PDF-1.4 garbage", "x.pdf")
    with pytest.raises(BadDocument, match="limit is 1"):
        extract_pages(HANDBOOK, "h.pdf", max_pages=1)


def test_chunking_overlap_page_and_stable_ids():
    pages = [(1, "A" * 50 + ". " + "B" * 50 + ".\n\n" + "C" * 60 + "."), (2, "D" * 80 + ".")]
    chunks = chunk_pages(pages, "doc.pdf", size=110, overlap=10, min_size=5)
    assert len(chunks) >= 3 and chunks[0].page == 1 and chunks[-1].page == 2
    assert chunks[1].text.startswith(chunks[0].text[-10:])                # overlap carried forward
    assert chunks[-1].text.startswith("D")                                 # ...but never across a page
    assert [c.id for c in chunks] == [c.id for c in chunk_pages(pages, "doc.pdf", 110, 10, 5)]   # deterministic
    assert chunks[0].id == hashlib.sha256(f"doc.pdf\x00{chunks[0].text}".encode()).hexdigest()[:16]


def test_ingest_is_idempotent(engine):
    first = engine.ingest("handbook.pdf", HANDBOOK)
    again = engine.ingest("handbook.pdf", HANDBOOK)
    assert first.pages == 2 and first.chunks == first.new_chunks > 0
    assert again.new_chunks == 0 and engine.store.count() == first.chunks
    assert engine.sources() == ["handbook.pdf"]


# ── retrieval, guardrail, degradation ─────────────────────────────────
def test_answer_with_citations(engine):
    engine.ingest("handbook.pdf", HANDBOOK)
    r = engine.answer("How many sick leave days?")
    assert r.found and not r.degraded and any("12 sick" in h.text for h in r.hits)
    assert set(r.timings_ms) == {"embed_ms", "search_ms", "llm_ms"} and r.usage.completion_tokens == 8


def test_guardrail_skips_llm_when_nothing_relevant(engine):
    engine.ingest("handbook.pdf", HANDBOOK)
    engine.max_distance = 1.0
    r = engine.answer("What is the capital of France?")
    assert not r.found and r.answer == NOT_FOUND and engine.llm.calls == 0


def test_empty_store_is_not_found(engine):
    assert not engine.answer("anything").found


def test_source_filter(engine):
    engine.ingest("handbook.pdf", HANDBOOK)
    engine.ingest("other.pdf", make_pdf("Password rules: every password needs 14 characters minimum, no reuse."))
    hits, _ = engine.retrieve("password characters", source="other.pdf")
    assert {h.source for h in hits} == {"other.pdf"}


def test_llm_down_degrades_to_passages(engine):
    engine.ingest("handbook.pdf", HANDBOOK)
    engine.llm = FakeLLM(fail=True)
    r = engine.answer("How many sick leave days?")
    assert r.degraded and r.found and r.hits and "unavailable" in r.answer


def test_model_saying_not_found_sets_found_false(engine):
    engine.ingest("handbook.pdf", HANDBOOK)
    engine.llm = FakeLLM(answer=NOT_FOUND)
    assert not engine.answer("sick leave days?").found


def test_stream_event_order(engine):
    engine.ingest("handbook.pdf", HANDBOOK)
    events = list(engine.stream("How many sick leave days?"))
    kinds = [k for k, _ in events]
    assert kinds[0] == "hits" and kinds[-1] == "done" and set(kinds[1:-1]) == {"token"}
    done = events[-1][1]
    assert done.answer.startswith("Employees get 12") and done.usage.ttft_ms == 50


def test_stream_not_found_sends_no_citations(engine):
    engine.ingest("handbook.pdf", HANDBOOK)
    engine.max_distance = 1.0
    events = list(engine.stream("capital of France?"))
    assert events[0] == ("hits", []) and [k for k, _ in events] == ["hits", "done"]


def test_prompt_numbers_passages(engine):
    engine.ingest("handbook.pdf", HANDBOOK)
    hits, _ = engine.retrieve("sick leave")
    p = RagEngine.prompt("q?", hits)
    assert p.startswith("CONTEXT:\n[1] (handbook.pdf, page") and p.endswith("QUESTION: q?")
