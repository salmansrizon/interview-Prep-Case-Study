"""Self-Correcting Writer-এর central config (week 11/12-এর dataclass style)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """HIGHLIGHTS: model আর retry limit এখানে DATA, graph-এর logic-এ পোঁতা না।

    ── Code-generation Model Reference Table (Class 27 lecture, section 4.1) ──

    | Model | Use Case | Advantage | Limitation |
    |---|---|---|---|
    | **`llama3.1:8b`** (default) | General-purpose, ছোট Python function লেখা | Instruction-following ভালো, 16GB RAM-এ comfortable | কঠিন algorithm-এ কখনো ভুল করে, তাই retry loop |
    | `mistral:7b` | দ্রুত inference, structured output | 8B-এর চেয়ে হালকা, দ্রুত | Code quality llama3.1-এর কাছাকাছি, বেশি ভালো না |
    | `qwen2.5-coder:7b` (code-specialised) | শুধু code লেখা/ঠিক করা | এই size-এ code benchmark-এ সাধারণত সেরা | General প্রশ্নে দুর্বল, আলাদা pull লাগে |
    | `llama3.2:3b` | কম RAM (8GB) machine | খুব ছোট ও দ্রুত | বেশি ভুল করে, retry বেশি লাগে |
    """

    model: str = "llama3.1:8b"
    fallback_model: str = "llama3.2:3b"
    temperature: float = 0.2       # code-এর জন্য কম temperature: বেশি deterministic
    max_attempts: int = 3          # retry loop-এর hard limit
    test_timeout_sec: int = 10     # generated code-এর infinite loop ঠেকায়
    require_approval: bool = True  # human-in-the-loop on/off


def get_config() -> Config:
    return Config()
