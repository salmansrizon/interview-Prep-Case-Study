"""
Central config — Collaborative Multi-Agent Research Team (Week 14, Class 2: CrewAI).

HIGHLIGHTS:
* Telemetry বন্ধ — CrewAI default-এ telemetry চালু রাখে। তিনটা flag লাগে, আর
  crewai import-এর **আগে** set করতে হয়, তাই এগুলো এই file-এর একদম শুরুতে।
  Sovereign/offline project-এ data বাইরে যাওয়ার কোনো পথ রাখা যাবে না।
* Memory/knowledge বন্ধ — চালু করলে embedder না দিলে CrewAI OpenAI-তে যায়।
  Retrieval আমাদের নিজের corpus tool-এ, তাই এগুলো লাগে না।

── Model Reference Table (তুলনার জন্য; course চালায় শুধু llama3.1:8b) ──

| Model | Use Case | Advantage | Limitation |
|---|---|---|---|
| **`llama3.1:8b`** (course standard, ADR 0006) | Sequential crew, ছোট report | Native tool calling, laptop-এ চলে | ReAct format মাঝে মাঝে ভাঙে; hierarchical manager হিসেবে দুর্বল |
| বড় local model (70B) | Hierarchical manager | Delegation-এ ভালো | ~40GB+ memory |
| Cloud (GPT/Claude) | Production research team | সবচেয়ে নির্ভরযোগ্য | Local না — data বাইরে যায় |
"""
from __future__ import annotations

import os

for _k, _v in {"CREWAI_DISABLE_TELEMETRY": "true", "OTEL_SDK_DISABLED": "true",
               "CREWAI_TRACING_ENABLED": "false"}.items():
    os.environ.setdefault(_k, _v)

from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class LLMConfig:
    base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model: str = "llama3.1:8b"
    temperature: float = 0.1     # কম randomness — format ভাঙা কমে
    seed: int = 42
    max_tokens: int = 1200


@dataclass(frozen=True)
class CrewConfig:
    max_iter: int = 6            # agent-প্রতি সর্বোচ্চ think/act step (default 25 — ছোট model-এ অনেক বেশি)
    guardrail_retries: int = 2   # guardrail fail করলে task কতবার আবার চেষ্টা করবে
    flow_max_rounds: int = 2     # verification fail করলে পুরো crew সর্বোচ্চ কতবার
    search_top_k: int = 4        # corpus search-এ কয়টা passage (context ছোট রাখা)
    retrieve_k: int = 6          # grounded mode-এ code কয়টা passage researcher-কে দেয়
    passage_chars: int = 450


@dataclass(frozen=True)
class VerifyConfig:
    min_citations: int = 4
    min_support: float = 0.45    # sentence-এর content word-এর কত অংশ cited passage-এ থাকা লাগবে
    max_weak_ratio: float = 0.2      # cited-কিন্তু-unsupported বাক্য (আসল বিপদ) — cited-এর ২০% পর্যন্ত
    max_uncited_ratio: float = 0.5   # citation-ছাড়া বাক্য (প্রায়ই transition/উপসংহার) — মোটের ৫০% পর্যন্ত


@dataclass(frozen=True)
class DataConfig:
    corpus_dir: Path = BASE_DIR / "data" / "corpus"
    output_dir: Path = BASE_DIR / "output"


@dataclass(frozen=True)
class AppConfig:
    page_title: str = "Multi-Agent Research Team"
    page_icon: str = "🧑‍🔬"
    topics: tuple[str, ...] = (
        "Why multi-agent systems fail and how to make them reliable",
        "How retrieval-augmented generation grounds answers",
        "Safe tool calling with local models",
        "Self-correcting code generation",
    )


@dataclass(frozen=True)
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    crew: CrewConfig = field(default_factory=CrewConfig)
    verify: VerifyConfig = field(default_factory=VerifyConfig)
    data: DataConfig = field(default_factory=DataConfig)
    app: AppConfig = field(default_factory=AppConfig)


_cfg: Config | None = None


def get_config() -> Config:
    global _cfg
    if _cfg is None:
        _cfg = Config()
    return _cfg
