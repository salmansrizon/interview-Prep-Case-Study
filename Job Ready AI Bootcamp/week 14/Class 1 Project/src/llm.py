"""Ollama-র সাথে একমাত্র যোগাযোগের জায়গা।

HIGHLIGHTS: `ollama` function-এর ভেতরে import হয়, তাই Ollama না থাকলেও
module import ও test চলে। Graph শুধু `llm(prompt) -> str` callable চেনে,
test-এ সেখানে fake বসানো হয়।
"""

from __future__ import annotations

from config import get_config


class OllamaUnavailable(RuntimeError):
    """Ollama server বন্ধ বা model pull করা নেই।"""


def ollama_llm(prompt: str) -> str:
    cfg = get_config()
    try:
        import ollama

        r = ollama.chat(
            model=cfg.model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": cfg.temperature},
        )
        return r["message"]["content"]
    except Exception as exc:  # noqa: BLE001
        raise OllamaUnavailable(
            f"Ollama চালু আছে? `ollama pull {cfg.model}` করেছ? ({exc})"
        ) from exc
