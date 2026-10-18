"""
Central config — Travel Agent (Week 13, Class 2: The ReAct Loop).

HIGHLIGHTS: সব tunable value এখানে data হিসেবে থাকে, agent code-এর ভেতরে
magic number হিসেবে না। frozen dataclass + singleton — week 7/10/11/12-এর
একই Config pattern।

── LLM Model Reference Table (ReAct / tool-calling এর দৃষ্টিকোণ থেকে) ──

| Model | Use Case | Advantage | Limitation |
|---|---|---|---|
| **`llama3.1:8b`** (default) | Multi-step reasoning + tool use | Instruction/JSON format ভালো মানে, tool-calling-এর জন্য train করা | ~16GB RAM লাগে, 8GB মেশিনে ধীর |
| `mistral:7b` | দ্রুত general-purpose agent | llama3.1-এর কাছাকাছি size, latency প্রায়ই কম | Format মাঝে মাঝে ভাঙে, extra prose মিশিয়ে ফেলে |
| `llama3.2:3b` | Low-RAM মেশিন, সহজ task | ছোট ও দ্রুত, 8GB-এও চলে | বেশি step-এ হারিয়ে যায়, ভুল argument বানায়, loop-এ আটকায় |
| Cloud (GPT-4 / Claude) | সবচেয়ে নির্ভরযোগ্য agent | Planning ও tool use সবচেয়ে ভালো | Local না — খরচ আছে, data বাইরে যায় (Sovereign AI-র বিপরীত) |

ছোট model-এ ReAct loop-এর failure handling (retry, max_steps, loop
detection) বেশি দরকার হয় — এটাই Class 2-এর মূল শিক্ষা।
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class LLMConfig:
    host: str = "http://localhost:11434"
    model: str = "llama3.1:8b"
    compare_models: tuple = ("llama3.1:8b", "mistral:7b", "llama3.2:3b")
    temperature: float = 0.0      # agent-এ determinism দরকার, creativity না
    timeout_seconds: int = 120


@dataclass(frozen=True)
class AgentConfig:
    # HIGHLIGHTS: এই তিনটা সংখ্যাই failure-handling-এর "লাগাম"।
    max_steps: int = 8             # Thought->Action->Observation চক্রের সর্বোচ্চ সংখ্যা
    max_llm_retries: int = 2       # LLM call fail হলে কতবার আবার চেষ্টা
    max_repeat_actions: int = 2    # একই action+input এর বেশি হলে loop ধরা হবে


@dataclass(frozen=True)
class DataConfig:
    flights_path: str = str(BASE_DIR / "data" / "flights.json")
    bookings_path: str = str(BASE_DIR / "data" / "bookings.json")


@dataclass(frozen=True)
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    data: DataConfig = field(default_factory=DataConfig)


_cfg: Config | None = None


def get_config() -> Config:
    global _cfg
    if _cfg is None:
        _cfg = Config()
    return _cfg
