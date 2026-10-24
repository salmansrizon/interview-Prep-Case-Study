"""
Central config — Self-Correcting LLM Coder (Week 14, Class 1: LangGraph).

HIGHLIGHTS: model, retry budget, sandbox limit, checkpoint path — সব DATA
হিসেবে এখানে, graph-এর logic-এ পোঁতা না। frozen dataclass, week 10-13-এর
একই pattern।

── Code-generation Model Reference Table (তুলনার জন্য; course চালায় শুধু llama3.1:8b) ──

| Model | Use Case | Advantage | Limitation |
|---|---|---|---|
| **`llama3.1:8b`** (course standard, ADR 0006) | ছোট Python function, self-repair loop | Meta model card: HumanEval 72.6, MBPP++ 72.8; instruction-following ভালো | Edge case প্রায়ই মিস — তাই test + retry loop |
| `qwen2.5-coder:7b` (শুধু তুলনা) | Code-specialised | Qwen report: HumanEval 88.4, HumanEval+ 84.1 | General প্রশ্নে দুর্বল; course-এ ব্যবহার হয় না |
| GPT/Claude (cloud) | কঠিন algorithm, বড় codebase | সবচেয়ে শক্তিশালী | Local না, খরচ, code বাইরে যায় |
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class LLMConfig:
    host: str = "http://localhost:11434"
    model: str = "llama3.1:8b"
    # HIGHLIGHTS: Self-Debugging paper-এ সব repair step greedy (temperature 0)।
    # Ollama-র default 0.8 — code-এ randomness চাই না।
    temperature: float = 0.0
    seed: int = 42
    num_ctx: int = 8192
    num_predict: int = 1024
    timeout_seconds: float = 180.0


@dataclass(frozen=True)
class AgentConfig:
    max_attempts: int = 4          # retry budget — research: বেশিরভাগ fix প্রথম ~৩ turn-এ
    # LangGraph 1.x-এ default recursion_limit আর 25 না (১০ হাজারের ঘরে) — তাই নিজে সীমা দিই।
    recursion_limit: int = 40
    require_approval: bool = True  # human-in-the-loop interrupt


@dataclass(frozen=True)
class SandboxConfig:
    timeout_seconds: int = 10      # LLM-এর লেখা `while True` app আটকাবে না
    cpu_seconds: int = 10          # RLIMIT_CPU (POSIX)
    memory_mb: int = 512           # RLIMIT_AS — Linux-এ কাজ করে, macOS-এ প্রায়ই enforce হয় না
    max_output_chars: int = 2000
    banned_imports: tuple[str, ...] = ("os", "subprocess", "socket", "shutil", "sys", "ctypes",
                                       "multiprocessing", "threading", "pathlib", "urllib", "requests")
    banned_calls: tuple[str, ...] = ("eval", "exec", "open", "__import__", "compile", "input",
                                     "globals", "vars", "getattr", "setattr", "delattr")


@dataclass(frozen=True)
class DataConfig:
    checkpoint_db: Path = BASE_DIR / "data" / "checkpoints.sqlite"


@dataclass(frozen=True)
class AppConfig:
    page_title: str = "Self-Correcting LLM Coder"
    page_icon: str = "🔁"


@dataclass(frozen=True)
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    data: DataConfig = field(default_factory=DataConfig)
    app: AppConfig = field(default_factory=AppConfig)


_cfg: Config | None = None


def get_config() -> Config:
    global _cfg
    if _cfg is None:
        _cfg = Config()
    return _cfg
