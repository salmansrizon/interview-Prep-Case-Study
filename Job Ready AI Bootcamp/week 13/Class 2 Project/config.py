"""
Central config — Deterministic Function-Calling Travel Agent (Week 13, Class 2).

HIGHLIGHTS: সব tunable value এখানে data হিসেবে থাকে, agent code-এর ভেতরে
magic number হিসেবে না। frozen dataclass + singleton — week 7/10/11/12-এর
একই Config pattern।

── LLM Model Reference Table (ReAct / tool-calling-এর দৃষ্টিকোণ থেকে) ──
(BFCL v4 leaderboard, 2026-04-12 snapshot)

| Model | Use Case | Advantage | Limitation |
|---|---|---|---|
| **`llama3.1:8b`** (default, ADR 0006) | Course standard agent | Native tool calling, parallel call পারে | BFCL multi-turn ~11% — loop guard ছাড়া আটকে যায় |
| `qwen3:8b` | ভালো local agent | BFCL multi-turn ~42%, irrelevance ~79% | Course standard না; thinking mode ধীর |
| `llama3.2:3b` | Low-RAM মেশিন | ছোট ও দ্রুত, 8GB-এও চলে | BFCL multi-turn ~4%, ভুল argument বানায় |
| Cloud (GPT / Claude) | সবচেয়ে নির্ভরযোগ্য agent | Strict schema, forced tool choice | Local না — খরচ, data বাইরে যায় (Sovereign AI-র বিপরীত) |
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class LLMConfig:
    host: str = "http://localhost:11434"
    model: str = "llama3.1:8b"
    # HIGHLIGHTS: Ollama-র default temperature 0.8, context ছোট, num_predict
    # অসীম। Agent-এ তিনটাই বিপজ্জনক, তাই explicitly set করা।
    temperature: float = 0.0
    seed: int = 42
    num_ctx: int = 8192
    num_predict: int = 512
    timeout_seconds: float = 120.0


@dataclass(frozen=True)
class AgentConfig:
    # HIGHLIGHTS: loop-এর পাঁচটা "লাগাম"। প্রতিটা আলাদা typed stop reason দেয়।
    mode: str = "native"              # "native" (Ollama tool_calls) | "text" (classic Thought/Action)
    nudge_after_tools: bool = True    # llama3.1 template workaround, see src/agent/react_agent.py
    max_steps: int = 8                # LLM call-এর সর্বোচ্চ সংখ্যা
    max_repeat_calls: int = 2         # একই (tool, args) এর বেশি হলে loop ধরা হবে
    max_consecutive_errors: int = 3   # পরপর এতগুলো invalid call হলে error budget শেষ
    deadline_seconds: float = 180.0   # পুরো run-এর wall-clock সীমা
    max_llm_retries: int = 2          # transport failure-এ কতবার আবার চেষ্টা
    max_results: int = 5              # search observation compact রাখা (context বাঁচায়)


@dataclass(frozen=True)
class DataConfig:
    flights_path: Path = BASE_DIR / "data" / "flights.json"


@dataclass(frozen=True)
class AppConfig:
    page_title: str = "Travel Agent"
    page_icon: str = "✈️"
    examples: tuple[str, ...] = (
        "Find me flights from DAC to CXB on 2026-11-05",
        "Book the cheapest DAC to CXB flight on 2026-11-05 for Rahim",
        "Book BG102 for Karim",
        "Fly me from DAC to Paris tomorrow",
    )


@dataclass(frozen=True)
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    app: AppConfig = field(default_factory=AppConfig)


_cfg: Config | None = None


def get_config() -> Config:
    global _cfg
    if _cfg is None:
        _cfg = Config()
    return _cfg
