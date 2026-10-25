"""Embedder, vector store and LLM client — each a small class, injected into the engine.

HIGHLIGHTS:
* Heavy import (sentence_transformers, chromadb, ollama) function-এর ভেতরে, প্রথম
  ব্যবহারে — app import বা test-এ model load/network লাগে না।
* কোনো global singleton নেই: engine এগুলো constructor-এ পায়, তাই test-এ fake বসানো
  সহজ, আর একই process-এ দুটো আলাদা store চালানো যায়।
* LLM client দুটো পথ দেয়: ``generate`` (পুরো উত্তর) আর ``stream`` (token-by-token)।
  দুটোই Ollama-র নিজের timing আর token count ফেরত দেয় — monitoring-এর কাঁচামাল।
"""
from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field


class LLMUnavailable(RuntimeError):
    """Ollama is down, the model is missing, or the call timed out."""


class Embedder:
    def __init__(self, model_name: str):
        self.model_name, self._model = model_name, None

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # cached locally after first download
            self._model = SentenceTransformer(self.model_name)
        return self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False).tolist()


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    text: str
    source: str
    page: int
    distance: float


class ChromaStore:
    def __init__(self, path: str, collection: str = "rag_api"):
        self.path, self.collection_name, self._col = path, collection, None

    @property
    def col(self):
        if self._col is None:
            import chromadb
            self._col = chromadb.PersistentClient(path=self.path).get_or_create_collection(self.collection_name)
        return self._col

    def upsert(self, chunks, vectors) -> int:
        """Upsert by content-hash id; returns how many ids were NEW (idempotent ingestion)."""
        ids = [c.id for c in chunks]
        existing = set(self.col.get(ids=ids, include=[])["ids"]) if ids else set()
        self.col.upsert(ids=ids, embeddings=vectors, documents=[c.text for c in chunks],
                        metadatas=[{"source": c.source, "page": c.page} for c in chunks])
        return len(set(ids) - existing)

    def query(self, vector, k: int, source: str | None = None) -> list[Hit]:
        n = self.col.count()
        if n == 0:
            return []
        r = self.col.query(query_embeddings=[vector], n_results=min(k, n),
                           where={"source": source} if source else None)
        return [Hit(i, d, m.get("source", "?"), int(m.get("page", 0)), float(dist))
                for i, d, m, dist in zip(r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0])]

    def sources(self) -> list[str]:
        if self.col.count() == 0:
            return []
        return sorted({m["source"] for m in self.col.get(include=["metadatas"])["metadatas"] if m})

    def count(self) -> int:
        return self.col.count()

    def texts(self, ids: list[str]) -> dict[str, str]:
        """chunk id -> text (the offline judge re-reads exactly the passages an answer saw)."""
        r = self.col.get(ids=ids, include=["documents"]) if ids else {"ids": [], "documents": []}
        return dict(zip(r["ids"], r["documents"]))


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    load_ms: float = 0.0          # model load time (cold start if > 0)
    prefill_ms: float = 0.0       # Ollama prompt_eval_duration
    decode_ms: float = 0.0        # Ollama eval_duration
    ttft_ms: float | None = None  # measured by us: request sent -> first token
    extra: dict = field(default_factory=dict)

    @property
    def tokens_per_s(self) -> float:
        return round(self.completion_tokens / (self.decode_ms / 1000), 1) if self.decode_ms else 0.0


def _usage(resp) -> Usage:
    ns = 1e6
    return Usage(prompt_tokens=getattr(resp, "prompt_eval_count", 0) or 0,
                 completion_tokens=getattr(resp, "eval_count", 0) or 0,
                 load_ms=(getattr(resp, "load_duration", 0) or 0) / ns,
                 prefill_ms=(getattr(resp, "prompt_eval_duration", 0) or 0) / ns,
                 decode_ms=(getattr(resp, "eval_duration", 0) or 0) / ns)


class OllamaLLM:
    def __init__(self, host: str, model: str, temperature: float = 0.2, num_ctx: int = 8192,
                 num_predict: int = 512, timeout_s: float = 120.0, keep_alive: str = "30m"):
        self.host, self.model, self.timeout_s, self.keep_alive = host, model, timeout_s, keep_alive
        self.options = {"temperature": temperature, "seed": 42, "num_ctx": num_ctx, "num_predict": num_predict}

    def _client(self):
        import ollama
        return ollama.Client(host=self.host, timeout=self.timeout_s)

    def _messages(self, system: str, prompt: str) -> list[dict]:
        return [{"role": "system", "content": system}, {"role": "user", "content": prompt}]

    def generate(self, system: str, prompt: str) -> tuple[str, Usage]:
        t0 = time.perf_counter()
        try:
            r = self._client().chat(model=self.model, messages=self._messages(system, prompt),
                                    options=self.options, keep_alive=self.keep_alive)
        except Exception as exc:  # noqa: BLE001 - connection refused, timeout, model missing ...
            raise LLMUnavailable(f"Ollama call failed: {exc}") from exc
        u = _usage(r)
        u.extra["wall_ms"] = (time.perf_counter() - t0) * 1000
        return (r.message.content or "").strip(), u

    def stream(self, system: str, prompt: str) -> Iterator[tuple[str, object]]:
        """Yields ("token", str) ... then ("usage", Usage). Raises LLMUnavailable before the first token."""
        t0 = time.perf_counter()
        try:
            it = self._client().chat(model=self.model, messages=self._messages(system, prompt),
                                     options=self.options, keep_alive=self.keep_alive, stream=True)
            first = None
            for part in it:
                text = part.message.content or ""
                if text and first is None:
                    first = (time.perf_counter() - t0) * 1000
                if text:
                    yield "token", text
                if getattr(part, "done", False):
                    u = _usage(part)
                    u.ttft_ms = first
                    yield "usage", u
        except Exception as exc:  # noqa: BLE001
            raise LLMUnavailable(f"Ollama stream failed: {exc}") from exc

    def load(self) -> None:
        """Empty prompt = Ollama only loads the model. Same num_ctx as real calls, or Ollama reloads it."""
        self._client().generate(model=self.model, prompt="", keep_alive=self.keep_alive,
                                options={"num_ctx": self.options["num_ctx"]})

    def ping(self) -> tuple[bool, str]:
        """Readiness: server reachable AND model pulled."""
        try:
            names = [m.model for m in self._client().list().models]
        except Exception as exc:  # noqa: BLE001
            return False, f"Ollama not reachable at {self.host}: {exc}"
        ok = any(n == self.model or n.startswith(self.model + ":") for n in names)
        return ok, "ok" if ok else f"model {self.model} not pulled"
