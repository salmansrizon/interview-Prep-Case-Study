"""Build the assistant for a backend: ``real`` (MiniLM + Chroma + Ollama) or ``fake`` (no models, no network).

HIGHLIGHTS: Week 15-এ FakeBackend পুরো RAG বাদ দিত। এখানে fake শুধু দুটো model বদলায় —
embedding (keyword hashing) আর LLM (passage [1]-এর প্রথম বাক্য তুলে দেয়)। Guardrail, Chroma, route,
citation check — সব আসল code। তাই ``HC_BACKEND=fake``-এ API, UI আর eval-এর পুরো আচরণ দেখা যায়,
Ollama বা model download ছাড়াই।
"""
from __future__ import annotations

import re
import tempfile
import zlib

from .assistant import HealthcareAssistant
from .rag.components import ChromaStore, Embedder, OllamaLLM
from .settings import Settings

# Keyword vectors live on a different distance scale than MiniLM. On data/eval they keep every answerable
# question but let some in-domain off-topic ones through. ponytail: demo-only; real abstention needs real embeddings.
FAKE_MAX_DISTANCE = 0.75

_WORD = re.compile(r"[a-z0-9]+")
_STOP = set("a an the and or of to in on for with is are was were be by as at it its this that from what how "
            "do does can i my me you your we our should when which who why there their they them if not no".split())


class HashEmbedder:
    """Deterministic bag-of-words vector (crc32 buckets). Shared words -> small cosine distance."""

    def __init__(self, dims: int = 512):
        self.dims = dims

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = [0.0] * self.dims
            for w in _WORD.findall(t.lower()):
                if w not in _STOP:
                    v[zlib.crc32(w.rstrip("s").encode()) % self.dims] += 1.0
            v[0] += 1e-6            # never an all-zero vector (cosine undefined)
            out.append(v)
        return out


class ExtractiveLLM:
    """Stands in for Ollama: answers with the first sentence of passage [1] and cites it."""

    def __init__(self):
        self.calls = 0

    def generate(self, system: str, prompt: str) -> str:
        self.calls += 1
        m = re.search(r"\[1\] \(.*?\)\n(.+?[.!?])(?:\s|$)", prompt, re.S)
        return f"{m.group(1)} [1]" if m else "NOT_IN_KB"

    def ping(self) -> tuple[bool, str]:
        return True, "fake"


def build_assistant(s: Settings) -> HealthcareAssistant:
    if s.backend == "fake":
        # ponytail: a fresh temp dir per process, never cleaned up; fine for tests and demos only
        store = ChromaStore(tempfile.mkdtemp(prefix="hc_fake_"), s.collection)
        return HealthcareAssistant(HashEmbedder(), store, ExtractiveLLM(), top_k=s.top_k, max_distance=FAKE_MAX_DISTANCE)
    return HealthcareAssistant(
        Embedder(s.embedding_model),
        ChromaStore(str(s.chroma_dir), s.collection, s.chroma_host, s.chroma_port),
        OllamaLLM(s.ollama_host, s.model, s.temperature, s.num_ctx, s.num_predict, s.llm_timeout_s),
        top_k=s.top_k, max_distance=s.max_distance)
