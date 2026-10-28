"""Shared fixtures. Fully offline: keyword HashEmbedder, scripted LLM, real Chroma in a temp dir.

HIGHLIGHTS: ``HC_BACKEND=fake`` — Ollama, embedding model download, network কিছুই লাগে না (ADR 0006/0008)।
Chroma আসল (temp folder-এ), তাই ingest, cosine distance আর stale-section delete সত্যিই test হয়।
"""
from __future__ import annotations

import os

os.environ["HC_BACKEND"] = "fake"           # set before anything imports the settings

import pytest  # noqa: E402

from src.assistant import HealthcareAssistant  # noqa: E402
from src.backend import FAKE_MAX_DISTANCE, HashEmbedder  # noqa: E402
from src.rag.components import ChromaStore, LLMUnavailable  # noqa: E402
from src.settings import PROJECT_DIR  # noqa: E402

KB_DIR = PROJECT_DIR / "data" / "kb"
EVAL_FILE = PROJECT_DIR / "data" / "eval" / "questions.jsonl"


class ScriptedLLM:
    """Returns a fixed reply (or fails) and counts calls, so tests can prove when the LLM was NOT called."""

    def __init__(self, reply: str = "Mix the packet with the amount of clean water written on it [1].",
                 fail: bool = False):
        self.reply, self.fail, self.calls, self.prompts = reply, fail, 0, []

    def generate(self, system, prompt):
        self.calls += 1
        self.prompts.append(prompt)
        if self.fail:
            raise LLMUnavailable("Ollama call failed: ConnectError")
        return self.reply

    def ping(self):
        return (not self.fail), ("ok" if not self.fail else "down")


class CountingEmbedder(HashEmbedder):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        return super().embed(texts)


@pytest.fixture(scope="session")
def kb_store(tmp_path_factory):
    store = ChromaStore(str(tmp_path_factory.mktemp("chroma")), "test_kb")
    HealthcareAssistant(HashEmbedder(), store, None).ingest(KB_DIR)
    return store


@pytest.fixture
def assistant(kb_store):
    """Fresh assistant (own LLM + embedder counters) over the shared, already-indexed store."""
    return HealthcareAssistant(CountingEmbedder(), kb_store, ScriptedLLM(), top_k=4, max_distance=FAKE_MAX_DISTANCE)
