"""The only module that imports ``ollama``."""

from __future__ import annotations

from config import LLM


class OllamaConnectionError(RuntimeError):
    """Ollama not running, model not pulled, or the request failed."""


def chat(messages: list[dict], tools: list[dict], model: str | None = None) -> dict:
    """Send one chat turn; return {"content": str, "tool_calls": [{"name", "arguments"}]}.

    HIGHLIGHTS:
    * ``import ollama`` function-এর ভেতরে, যাতে module import করতে Ollama
      package বা server লাগে না — test এখানেই fake বসায় (ADR 0006)।
    * ollama-python ≥0.4 typed object ফেরত দেয় (``resp.message.tool_calls``)।
      Ollama-তে ``arguments`` আগে থেকেই dict — OpenAI-এর মতো JSON string না।
    * temperature/seed/num_ctx explicitly পাঠানো হয়, কারণ Ollama-র default
      (0.8, ছোট context) tool calling-এর জন্য খারাপ।
    * সব failure একটাই ``OllamaConnectionError``-এ আসে; UI একটা friendly
      message দেখায়।
    """
    name = model or LLM.model
    try:
        import ollama

        resp = ollama.Client(host=LLM.host, timeout=LLM.timeout_seconds).chat(
            model=name, messages=messages, tools=tools,
            options={"temperature": LLM.temperature, "seed": LLM.seed, "num_ctx": LLM.num_ctx},
        )
        msg = resp.message
        return {
            "content": msg.content or "",
            "tool_calls": [
                {"name": c.function.name, "arguments": dict(c.function.arguments or {})}
                for c in (msg.tool_calls or [])
            ],
        }
    except Exception as exc:  # noqa: BLE001 - connection/model errors come in many types
        raise OllamaConnectionError(
            f"Ollama call failed. Is `ollama serve` running and `ollama pull {name}` done? ({exc})"
        ) from exc
