"""
Ollama chat client (stdlib only).

HIGHLIGHTS: Ollama বন্ধ থাকলে বা model pull না করা থাকলে সবকিছু একটাই
LLMUnavailable exception-এ আসে। Agent শুধু `chat(messages) -> str` callable
চায়, তাই tests-এ real Ollama-র বদলে scripted fake LLM বসানো যায়।
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

from config import get_config


class LLMUnavailable(RuntimeError):
    pass


def ollama_chat(messages: list[dict], model: str | None = None) -> str:
    cfg = get_config().llm
    name = model or cfg.model
    body = json.dumps({"model": name, "messages": messages, "stream": False,
                       "format": "json", "options": {"temperature": cfg.temperature}}).encode()
    req = urllib.request.Request(f"{cfg.host}/api/chat", body, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=cfg.timeout_seconds) as r:
            return json.loads(r.read())["message"]["content"]
    except (urllib.error.URLError, TimeoutError, KeyError, ValueError, OSError) as e:
        raise LLMUnavailable(
            f"Ollama call failed ({e}). Is `ollama serve` running and `ollama pull {name}` done?"
        ) from e
