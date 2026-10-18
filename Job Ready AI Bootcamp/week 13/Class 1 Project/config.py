"""
Central config for the Tool-Calling Agent (Week 13, Class 1).

HIGHLIGHTS: model, host, sandbox path আর step limit এখানে DATA হিসেবে রাখা
হয়েছে, যাতে src/ কোডে magic string না থাকে। Model-choice reference table:

| Model                | Use Case                    | Advantage                          | Limitation                                  |
| -------------------- | --------------------------- | ---------------------------------- | ------------------------------------------- |
| llama3.1:8b (chosen) | Local tool calling, agents  | Native tool support, ভালো arguments | ~5GB RAM, CPU-তে ধীর                        |
| mistral:7b           | General chat + tool calling | দ্রুত, ভালো instruction following   | Multi-step-এ মাঝে মাঝে tool call ভুলে যায     |
| llama3.2:3b          | Low-RAM laptop, দ্রুত demo   | ছোট (~2GB), দ্রুত                   | জটিল/একাধিক tool-এ নির্ভরযোগ্যতা কম            |
| OpenAI gpt-4o-mini   | Cloud tool calling          | সবচেয়ে নির্ভরযোগ্য                 | Local না, প্রতি call-এ খরচ (Sovereign AI ভাঙে)  |
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class LLMConfig:
    host: str = "http://localhost:11434"
    model: str = "llama3.1:8b"
    fallback_model: str = "llama3.2:3b"
    temperature: float = 0.0  # tool arguments deterministic হওয়া দরকার


@dataclass(frozen=True)
class AgentConfig:
    sandbox_dir: Path = BASE_DIR / "sandbox"
    max_steps: int = 5  # infinite tool loop ঠেকানোর hard limit
    max_file_chars: int = 4000  # বড় file context উড়িয়ে দেওয়া ঠেকায়


LLM = LLMConfig()
AGENT = AgentConfig()
