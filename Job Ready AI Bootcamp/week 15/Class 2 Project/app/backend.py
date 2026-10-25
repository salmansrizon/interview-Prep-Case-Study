"""Backends behind one small blocking interface: ask, stream, ingest, sources, ready.

HIGHLIGHTS — কেন plain ``def``, ``async def`` না? Ollama, Chroma, sentence-transformers
সবই blocking। ``async def``-এর ভেতরে সরাসরি ডাকলে পুরো event loop আটকে যেত —
এমনকি /health-ও। Service layer এগুলো threadpool-এ চালায় (service.py)।

* ``RealBackend`` — এই project-এর নিজস্ব RAG engine (app/rag/), Week 12-এর code import না (ADR 0002)।
* ``FakeBackend`` — কোনো dependency ছাড়া, ``time.sleep`` দিয়ে LLM latency নকল করে; tests আর benchmark।
"""
from __future__ import annotations

import hashlib
import time
from collections.abc import Iterator

from .errors import BadDocument
from .rag import ingest as rag_ingest
from .rag.components import Hit, Usage
from .rag.engine import NOT_FOUND, AskResult, IngestResult, build_engine
from .settings import Settings


class RealBackend:
    name = "real"

    def __init__(self, settings: Settings):
        self.engine = build_engine(settings)

    def ask(self, question, source=None, top_k=None) -> AskResult:
        return self.engine.answer(question, source, top_k)

    def stream(self, question, source=None, top_k=None) -> Iterator[tuple[str, object]]:
        return self.engine.stream(question, source, top_k)

    def ingest(self, filename, data) -> IngestResult:
        try:
            return self.engine.ingest(filename, data)
        except rag_ingest.BadDocument as exc:
            raise BadDocument(str(exc)) from exc

    def sources(self) -> list[str]:
        return self.engine.sources()

    def ready(self) -> tuple[bool, str]:
        return self.engine.llm.ping()

    def warmup(self) -> None:
        """Load the embedding model and ask Ollama to load llama3.1 into memory (empty prompt = load only)."""
        self.engine.embedder.embed(["warmup"])
        self.engine.store.count()
        try:
            self.engine.llm.load()
        except Exception:  # noqa: BLE001 - Ollama down at startup is allowed; /ready reports it
            pass


class FakeBackend:
    name = "fake"

    def __init__(self, delay: float = 0.0):
        self.delay = delay
        self.docs: dict[str, str] = {}

    def _hits(self, source):
        src = source or sorted(self.docs)[0]
        return [Hit(hashlib.sha256(src.encode()).hexdigest()[:16], f"Text from {src}", src, 1, 10.0)]

    def ask(self, question, source=None, top_k=None) -> AskResult:
        time.sleep(self.delay)
        if not self.docs or (source and source not in self.docs):
            return AskResult(NOT_FOUND, found=False, timings_ms={"embed_ms": 1.0, "search_ms": 1.0})
        return AskResult(f"Fake answer to: {question} [1]", found=True, hits=self._hits(source),
                         timings_ms={"embed_ms": 1.0, "search_ms": 1.0, "llm_ms": self.delay * 1000},
                         usage=Usage(prompt_tokens=120, completion_tokens=12, decode_ms=max(self.delay * 1000, 1)))

    def stream(self, question, source=None, top_k=None):
        res = self.ask(question, source, top_k)
        yield "hits", res.hits
        if res.found:
            for word in res.answer.split(" "):
                yield "token", word + " "
        yield "done", res

    def ingest(self, filename, data) -> IngestResult:
        if not data.startswith(b"%PDF-"):
            raise BadDocument(f"{filename!r} is not a PDF (missing %PDF- header)")
        new = 0 if self.docs.get(filename) == hashlib.sha256(data).hexdigest() else 3
        self.docs[filename] = hashlib.sha256(data).hexdigest()
        return IngestResult(filename, 1, 3, new)

    def sources(self) -> list[str]:
        return sorted(self.docs)

    def ready(self) -> tuple[bool, str]:
        return True, "fake"


def make_backend(settings: Settings):
    return FakeBackend(settings.fake_delay_s) if settings.backend == "fake" else RealBackend(settings)
