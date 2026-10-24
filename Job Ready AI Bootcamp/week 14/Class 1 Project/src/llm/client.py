"""The only module that imports ``ollama``.

HIGHLIGHTS: graph শুধু ``generate(prompt) -> str`` callable চেনে — test-এ
fake বসানো যায়, Ollama লাগে না (ADR 0006)। temperature 0 — Self-Debugging
paper-এর repair step-গুলোও greedy decoding-এ।
"""
from __future__ import annotations

from config import get_config


class OllamaUnavailable(RuntimeError):
    """Ollama server not running or the model is not pulled."""


SYSTEM = ("You are a careful Python programmer. Write one self-contained Python function that solves the "
          "task. Use only the standard library and no I/O. Reply with exactly one ```python code block "
          "and nothing after it.")


def generate(prompt: str, model: str | None = None) -> str:
    cfg = get_config().llm
    name = model or cfg.model
    try:
        import ollama

        resp = ollama.Client(host=cfg.host, timeout=cfg.timeout_seconds).chat(
            model=name,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            options={"temperature": cfg.temperature, "seed": cfg.seed,
                     "num_ctx": cfg.num_ctx, "num_predict": cfg.num_predict},
        )
        return resp.message.content or ""
    except Exception as exc:  # noqa: BLE001 - connection/model errors come in many types
        raise OllamaUnavailable(
            f"Ollama call failed. Is `ollama serve` running and `ollama pull {name}` done? ({exc})") from exc
