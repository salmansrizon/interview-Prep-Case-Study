"""Embedder, Chroma vector store and Ollama client — small classes injected into the assistant.

HIGHLIGHTS:
* Heavy import (sentence_transformers, chromadb, ollama) function-এর ভেতরে, প্রথম ব্যবহারে।
  App import বা test-এ model load, download বা network লাগে না।
* Chroma দুই রূপে, একই code: ``HC_CHROMA_HOST`` খালি হলে local ``PersistentClient(HC_CHROMA_DIR)``
  (এক মেশিনে dev), set করলে ``HttpClient(host, port)`` (Week 18-এ আলাদা Chroma container)।
  দুটোতেই ``anonymized_telemetry=False`` — privacy-first মানে vector DB-ও বাইরে কিছু পাঠায় না।
* Collection ``cosine`` distance-এ: ০ = একই দিক, ১ = সম্পর্কহীন। L2-এর চেয়ে threshold বোঝা সহজ, আর
  vector-এর দৈর্ঘ্যে নির্ভর করে না।
* Embedding model বদলালে vector-এর মাপ (dimension) বদলায়: ``ingest --reset`` দিয়ে collection নতুন করুন।

── Embedding model (loaded here) ──
| Model | Use Case | Advantage | Limitation |
|---|---|---|---|
| **sentence-transformers/paraphrase-MiniLM-L3-v2** (chosen) | ছোট English handbook-এ semantic search | ~61 MB, CPU-তে দ্রুত, course standard | General-domain, medical synonym কম বোঝে |
| sentence-transformers/all-MiniLM-L6-v2 | General English retrieval | ভালো quality, এখনো হালকা | L3-এর চেয়ে ধীর |
| pritamdeka/S-PubMedBert-MS-MARCO | Biomedical text retrieval | PubMed vocabulary, retrieval-tuned | ~440 MB, ধীর, সাধারণ ভাষার প্রশ্নে লাভ অনিশ্চিত |
| paraphrase-multilingual-MiniLM-L12-v2 | বাংলা প্রশ্ন, English KB | Cross-lingual retrieval | বড়; threshold আবার calibrate করতে হয় |
"""
from __future__ import annotations

from dataclasses import dataclass


class LLMUnavailable(RuntimeError):
    """Ollama is down, the model is missing, or the call timed out."""


class Embedder:
    def __init__(self, model_name: str):
        self.model_name, self._model = model_name, None

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # downloaded once, then cached locally
            self._model = SentenceTransformer(self.model_name)
        return self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False).tolist()


@dataclass(frozen=True)
class Hit:
    id: str            # kb:fever:2
    page: str
    title: str
    text: str
    distance: float    # cosine distance, 0 = same direction


class ChromaStore:
    def __init__(self, path: str = "data/chroma_db", collection: str = "clinic_kb", host: str = "", port: int = 8000):
        self.path, self.collection_name, self.host, self.port = path, collection, host, port
        self._client = self._col = None

    def client(self):
        if self._client is None:
            import chromadb
            from chromadb.config import Settings
            cfg = Settings(anonymized_telemetry=False)
            self._client = (chromadb.HttpClient(host=self.host, port=self.port, settings=cfg) if self.host
                            else chromadb.PersistentClient(path=self.path, settings=cfg))
        return self._client

    @property
    def col(self):
        if self._col is None:
            self._col = self.client().get_or_create_collection(
                self.collection_name, configuration={"hnsw": {"space": "cosine"}}, embedding_function=None)
        return self._col

    def sync(self, sections, vectors) -> int:
        """Upsert every section by its stable id and delete ids no longer in the KB. Returns how many were removed.
        Removing stale sections matters here: outdated medical guidance must not stay retrievable."""
        ids = [s.id for s in sections]
        self.col.upsert(ids=ids, embeddings=vectors, documents=[s.text for s in sections],
                        metadatas=[{"page": s.page, "title": s.title, "file": s.file} for s in sections])
        stale = sorted(set(self.col.get(include=[])["ids"]) - set(ids))
        if stale:
            self.col.delete(ids=stale)
        return len(stale)

    def query(self, vector, k: int) -> list[Hit]:
        n = self.col.count()
        if n == 0:
            return []
        r = self.col.query(query_embeddings=[vector], n_results=min(k, n))
        return [Hit(i, m.get("page", "?"), m.get("title", "?"), d, float(dist))
                for i, d, m, dist in zip(r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0])]

    def pages(self) -> list[dict]:
        """What is actually indexed: one row per handbook page with its section count."""
        if self.col.count() == 0:
            return []
        rows: dict[str, dict] = {}
        for m in self.col.get(include=["metadatas"])["metadatas"]:
            row = rows.setdefault(m["file"], {"file": m["file"], "page": m["page"], "sections": 0})
            row["sections"] += 1
        return sorted(rows.values(), key=lambda r: r["file"])

    def count(self) -> int:
        return self.col.count()

    def reset(self) -> None:
        try:
            self.client().delete_collection(self.collection_name)
        except Exception:  # noqa: BLE001 - collection did not exist yet
            pass
        self._col = None


class OllamaLLM:
    def __init__(self, host: str, model: str, temperature: float = 0.1, num_ctx: int = 4096,
                 num_predict: int = 400, timeout_s: float = 120.0, keep_alive: str = "30m"):
        self.host, self.model, self.timeout_s, self.keep_alive = host, model, timeout_s, keep_alive
        self.options = {"temperature": temperature, "seed": 42, "num_ctx": num_ctx, "num_predict": num_predict}

    def _client(self):
        import ollama
        return ollama.Client(host=self.host, timeout=self.timeout_s)

    def generate(self, system: str, prompt: str) -> str:
        try:
            r = self._client().chat(model=self.model, options=self.options, keep_alive=self.keep_alive,
                                    messages=[{"role": "system", "content": system},
                                              {"role": "user", "content": prompt}])
        except Exception as exc:  # noqa: BLE001 - connection refused, timeout, model missing ...
            raise LLMUnavailable(f"Ollama call failed: {type(exc).__name__}") from exc
        return (r.message.content or "").strip()

    def ping(self) -> tuple[bool, str]:
        """Readiness: server reachable AND model pulled."""
        try:
            names = [m.model for m in self._client().list().models]
        except Exception as exc:  # noqa: BLE001
            return False, f"Ollama not reachable at {self.host}: {type(exc).__name__}"
        ok = any(n == self.model or n.startswith(self.model + ":") for n in names)
        return ok, "ok" if ok else f"model {self.model} not pulled (ollama pull {self.model})"
