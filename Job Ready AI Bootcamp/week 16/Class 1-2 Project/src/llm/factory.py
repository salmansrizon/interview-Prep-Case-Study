"""LLM factory + a network-light Ollama health check."""
from __future__ import annotations

import json
import urllib.request

import config  # noqa: F401 - sets the telemetry env vars before crewai is imported
from crewai import LLM

from config import get_config


def build_llm(model: str | None = None, base_url: str | None = None) -> LLM:
    """HIGHLIGHTS: CrewAI 1.12+ Ollama-কে native ভাবে চালায় (OpenAI-compatible ``/v1`` endpoint দিয়ে) —
    LiteLLM লাগে না। ``"ollama/"`` prefix provider বেছে দেয়। Object বানাতে কোনো network call হয় না,
    তাই import আর test নিরাপদ।"""
    c = get_config().llm
    return LLM(model=f"ollama/{model or c.model}", base_url=base_url or c.base_url, temperature=c.temperature,
               seed=c.seed, max_tokens=c.max_tokens)


def ollama_status(model: str | None = None, base_url: str | None = None) -> tuple[bool, str]:
    """(ok, message). Never raises, so the app works with or without Ollama."""
    c = get_config().llm
    model, base_url = model or c.model, base_url or c.base_url
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=2) as r:
            names = [m["name"] for m in json.load(r).get("models", [])]
    except Exception as e:  # noqa: BLE001
        return False, f"Ollama not reachable at {base_url} ({e}). Run `ollama serve`."
    if not any(n == model or n.startswith(model + ":") for n in names):
        return False, f"Model {model} is not pulled. Run: ollama pull {model}"
    return True, "Ollama ready"
