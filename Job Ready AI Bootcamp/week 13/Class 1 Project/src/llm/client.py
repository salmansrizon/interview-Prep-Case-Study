"""Only module that imports ``ollama``."""

from __future__ import annotations

from config import LLM


class OllamaConnectionError(RuntimeError):
    """Ollama not running / model not pulled."""


def chat(messages: list[dict], tools: list[dict], model: str | None = None) -> dict:
    """Send one chat turn; return {"content": str, "tool_calls": [{name, arguments}]}.

    HIGHLIGHTS: ``import ollama`` function-এর ভেতরে, যাতে module import
    করতে Ollama server বা package লাগে না — test এখানেই mock বসায়। সব
    failure একটাই OllamaConnectionError-এ আসে, app.py একটাই friendly message
    দেখায়।
    """
    name = model or LLM.model
    try:
        import ollama

        resp = ollama.Client(host=LLM.host).chat(
            model=name, messages=messages, tools=tools,
            options={"temperature": LLM.temperature},
        )
        m = resp["message"]
        return {
            "content": m.get("content") or "",
            "tool_calls": [
                {"name": c["function"]["name"],
                 "arguments": dict(c["function"].get("arguments") or {})}
                for c in (m.get("tool_calls") or [])
            ],
        }
    except Exception as exc:  # noqa: BLE001 - connection/model errors vary
        raise OllamaConnectionError(
            f"Ollama চালু আছে তো? `ollama serve` আর `ollama pull {name}` করুন. ({exc})"
        ) from exc
