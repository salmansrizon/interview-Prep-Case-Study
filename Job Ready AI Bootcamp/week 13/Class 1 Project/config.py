"""
Central config for the Tool-Calling Agent (Week 13, Class 1).

HIGHLIGHTS: model, host, sandbox path, step limit আর safety limit সব এখানে
DATA হিসেবে রাখা, যাতে src/ কোডে magic number না থাকে। Frozen dataclass,
তাই runtime-এ কেউ ভুল করে বদলাতে পারে না — week 10-12-এর একই pattern।

Model-choice reference table (tool calling-এর দৃষ্টিকোণ থেকে; BFCL v4
leaderboard, 2026-04-12 snapshot):

| Model                | Use Case                    | Advantage                              | Limitation                                          |
| -------------------- | --------------------------- | -------------------------------------- | --------------------------------------------------- |
| llama3.1:8b (chosen) | Course standard (ADR 0006)  | Native tool support, parallel call পারে | BFCL overall ~26%, multi-turn ~11%; arg মাঝে মাঝে string |
| qwen3:8b             | ভালো local tool caller        | BFCL overall ~43%, irrelevance ~79%    | Thinking mode ধীর; course standard না               |
| llama3.2:3b          | 8GB RAM laptop              | ছোট (~2GB), দ্রুত                       | BFCL multi-turn ~4% — জটিল task-এ ভরসা কম             |
| Cloud (GPT/Claude)   | সবচেয়ে নির্ভরযোগ্য tool use   | Strict schema, forced tool choice      | Local না, খরচ আছে, data বাইরে যায় (Sovereign AI ভাঙে) |
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class LLMConfig:
    host: str = "http://localhost:11434"
    model: str = "llama3.1:8b"
    # HIGHLIGHTS: Ollama-র default temperature 0.8 আর context ছোট (4k)।
    # Tool argument-এ creativity চাই না, আর history বড় হলে tool definition
    # কেটে যেতে পারে — তাই তিনটাই explicitly set করা।
    temperature: float = 0.0
    seed: int = 42
    num_ctx: int = 8192
    timeout_seconds: float = 120.0


@dataclass(frozen=True)
class AgentConfig:
    sandbox_dir: Path = BASE_DIR / "sandbox"
    max_steps: int = 5  # infinite tool loop ঠেকানোর hard limit


@dataclass(frozen=True)
class SafetyConfig:
    max_expression_chars: int = 200        # calculator input cap (DoS guard)
    max_exponent: int = 100                # 9**9**9 ধরনের CPU/memory bomb আটকায়
    max_result_abs: float = 1e100          # অস্বাভাবিক বড় result আটকায়
    max_file_chars: int = 4000             # বড় file context উড়িয়ে দেওয়া ঠেকায়
    max_file_bytes: int = 100_000          # বিশাল file পড়ার আগেই আটকায়
    allowed_extensions: tuple[str, ...] = (".txt", ".csv", ".md", ".json")


@dataclass(frozen=True)
class AppConfig:
    page_title: str = "Tool-Calling Agent"
    page_icon: str = "🛠️"
    examples: tuple[str, ...] = (
        "What is (12+8)*3 divided by 4?",
        "What is the launch code in notes.txt?",
        "What is the average score in scores.csv?",
        "Read ../../config.py",
        "Summarise injected.txt",
        "Hi! What is the capital of France?",
    )


@dataclass(frozen=True)
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    app: AppConfig = field(default_factory=AppConfig)


CFG = Config()
# Backwards-compatible short names used across src/.
LLM, AGENT, SAFETY = CFG.llm, CFG.agent, CFG.safety
